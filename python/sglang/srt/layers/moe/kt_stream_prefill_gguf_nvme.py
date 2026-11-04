"""GPU-streamed prefill for GGUF CPU experts fed from the GGUF files (NVMe), for hosts whose RAM cannot hold
the experts (the 64 GB V4.1-1017 setup: 119 GB of IQ1_M / IQ2_XXS experts, 40 GiB of them in kt-kernel's own
RAM cache).

kt_stream_prefill_gguf.py (the Flash-Next edition) cuts every layer's experts out of the GGUF mmap into a pinned
bank once (about the whole expert bytes in RAM) and DMAs groups of G experts from it into a device slot. This
edition keeps its banks out: a reader pool fills a ring of pinned group buffers straight from the GGUF files
with O_DIRECT reads (no page cache: under a memory limit the page cache evicted read-ahead before use, 10-24),
in the order the layers are called, one layer ahead; _stage DMAs from the ring buffer instead of the bank. The
device side (two slots, dequant to bf16, FlashInfer's bf16 CUTLASS MoE, one CUDA graph per group shape) is the
parent's; IQ1_S / IQ1_M / IQ2_XXS / IQ2_XS, which kt_gguf_dequant.py's Triton kernels do not cover, are
dequantized with sgl_kernel.ggml_dequantize (matches gguf-py's reference to bf16 rounding, 10-24).

Rates measured with one SN5100-class NVMe and one 16 GB SM120 card: NVMe 5.2-6.3 GB/s with 4-8
parallel sequential streams, pinned H2D 27.9 GB/s, sgl_kernel IQ1_M dequant 65 GB/s of raw bytes, bf16 GEMM 9.2
TFLOPS: a 6144-token chunk is bound by the NVMe read of the layer's experts (~2.97 GB a layer, ~20 s a chunk),
a 12288-token chunk by the GEMMs.

Env: KT_GGUF_STREAM_SOURCE=nvme selects this edition (kt_stream_prefill.py's LLAMAFILE dispatch);
KT_GGUF_STREAM_READERS (8) reader threads, on KT_IO_CPUS when set; KT_GGUF_STREAM_RING (6) pinned group buffers; the parent's
KT_GPU_STREAM_PREFILL / KT_GPU_STREAM_GROUP / KT_GPU_STREAM_TIMING apply.
"""
import collections
import ctypes
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.distributed.parallel_state import get_tensor_model_parallel_rank
from sglang.srt.layers.moe import kt_gguf_dequant as DQ
from sglang.srt.layers.moe import kt_stream_prefill_gguf as P

# ggml types kt_gguf_dequant.py has no Triton kernel for: (block elements, block bytes); dequantized by sgl_kernel
GGML_IQ2_XXS, GGML_IQ2_XS, GGML_IQ1_S, GGML_IQ1_M = 16, 17, 19, 29
SGL_TYPES = {GGML_IQ2_XXS: (256, 66), GGML_IQ2_XS: (256, 74), GGML_IQ1_S: (256, 50), GGML_IQ1_M: (256, 56)}
for _t, _info in SGL_TYPES.items():
    DQ.BLOCK_INFO.setdefault(_t, _info)  # _Geometry accepts them; _group_body below routes them to sgl_kernel

READERS = int(os.environ.get("KT_GGUF_STREAM_READERS", "8"))
# KT_GGUF_STREAM_MMQ=1: the experts stay in their GGUF blocks on the GPU and llama.cpp's MMQ (kt_gguf_moe_mmq,
# int8 tiles in shared memory) multiplies them; nothing is dequantized to bf16 (no 566 MB of group buffers)
SYNC_DMA = os.environ.get("KT_GGUF_STREAM_SYNC_DMA", "0") == "1"  # diagnostic: wait for every group's copies
MMQ_PROFILE = int(os.environ.get("KT_GGUF_STREAM_MMQ_PROFILE", "0") or 0)  # diagnostic: time the body's pieces on this many layers
# KT_GGUF_STREAM_DIRECT=0: buffered reads (the page cache serves repeated prefills on a host whose RAM holds the
# experts; under a memory limit the cache evicted read-ahead before use, hence O_DIRECT by default)
DIRECT = os.environ.get("KT_GGUF_STREAM_DIRECT", "1") == "1"
# KT_GGUF_STREAM_TIMELINE=<layers>: events around every group's H2D and body on the first layers, read after
# the layer (no sync inside): H2D and body totals, the wall from the first copy to the last body, the gaps
TIMELINE = int(os.environ.get("KT_GGUF_STREAM_TIMELINE", "0") or 0)
SWITCH_MS = float(os.environ.get("KT_GGUF_STREAM_SWITCH_MS", "1") or 0)  # Python's thread switch interval while streaming
# KT_GGUF_STREAM_FROM_CACHE=1: experts resident in kt-kernel's GGUF expert cache (KT_GGUF_EXPERT_CACHE_GIB) are
# copied from its RAM instead of read from the NVMe; under a 64 GB
# limit the cache holds about a third of the experts and the NVMe (6.6 GB/s, 3.6 GB a layer) is the prefill's wall
FROM_CACHE = os.environ.get("KT_GGUF_STREAM_FROM_CACHE", "1") == "1"
MMQ_ON = os.environ.get("KT_GGUF_STREAM_MMQ", "0") == "1"
RING = int(os.environ.get("KT_GGUF_STREAM_RING", "6"))
ALIGN = 4096


def _ep_weights(world: int) -> List[float]:
    """Each rank's share of the streamed experts: KT_GGUF_STREAM_EP_WEIGHTS ("3,2"), else its GPU's SM count
    (rank r on CUDA device r). Cards that differ in architecture as well may want the variable."""
    spec = os.environ.get("KT_GGUF_STREAM_EP_WEIGHTS", "")
    if spec:
        w = [float(x) for x in spec.split(",")]
        if len(w) == world and all(x > 0 for x in w):
            return w
        _log(f"KT_GGUF_STREAM_EP_WEIGHTS={spec!r} is not {world} positive numbers: by SM count")
    return [float(torch.cuda.get_device_properties(r).multi_processor_count) for r in range(world)]


def _pin_io_thread() -> None:
    """Run this reader on KT_IO_CPUS ("a,b,c-d", serve.sh sets it): the CPUs every thread that reads the drive
    shares, apart from the ones computing the experts. Unset: not pinned."""
    spec = os.environ.get("KT_IO_CPUS", "")
    cpus = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    if cpus:
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as ex:
            _log(f"reader affinity {spec!r} not set: {ex}")


def _log(msg: str):
    P._log("[nvme] " + msg)


class _TensorMap:
    """Expert tensors of every layer: name -> (shard path, byte offset of the data, bytes)."""

    def __init__(self, path: str):
        import glob

        import gguf

        self.where: Dict[str, Tuple[str, int, int]] = {}
        files = sorted(glob.glob(os.path.join(path, "*.gguf"))) if os.path.isdir(path) else [path]
        for f in files:
            r = gguf.GGUFReader(f)
            for t in r.tensors:
                if ".ffn_" in t.name and "_exps." in t.name:
                    self.where[t.name] = (f, int(t.data_offset), int(t.n_bytes))
        if not self.where:
            raise RuntimeError(f"KT_GGUF_STREAM_SOURCE=nvme: no ffn_*_exps tensors under {path}")
        self._fds: Dict[str, int] = {}
        self._lock = threading.Lock()

    def fd(self, f: str) -> int:
        with self._lock:
            d = self._fds.get(f)
            if d is None:
                d = self._fds[f] = os.open(f, os.O_RDONLY | (os.O_DIRECT if DIRECT else 0))
            return d


class _RingBuf:
    """One group's raw bytes of the three projections, pinned, with room for the O_DIRECT alignment."""

    def __init__(self, strides, G: int):
        self.buf = [torch.empty(G * st + 2 * ALIGN, dtype=torch.uint8, pin_memory=True) for st in strides]
        self.head = [0, 0, 0]
        self.futs = []
        self.key = None
        self.peeked = None  # (layer_idx, base, [pointers or None per expert]) while its cache pins are held


def _read_into(tm: _TensorMap, f: str, off: int, nbytes: int, buf: torch.Tensor) -> int:
    """Read [off, off + nbytes) of file f into buf with O_DIRECT (aligned superset); returns the byte in buf
    where off lands."""
    lo = off // ALIGN * ALIGN
    hi = -(-(off + nbytes) // ALIGN) * ALIGN
    n = hi - lo
    assert n <= buf.numel(), (n, buf.numel())
    mv = memoryview(buf.numpy())[:n]
    fd = tm.fd(f)
    need = off + nbytes - lo  # the aligned tail past the file's end (a shard's last tensor) is not there to read
    got = 0
    while got < need:
        k = os.preadv(fd, [mv[got:]], lo + got)
        if k <= 0:
            raise IOError(f"short O_DIRECT read of {f} at {lo + got}: {k} (need {need}, got {got})")
        got += k
    return off - lo


class _SglDequant:
    """sgl_kernel.ggml_dequantize for the types kt_gguf_dequant.py has no Triton kernel for."""

    def _dequant(self, raw: torch.Tensor, pb: int, qt: int, n: int, rows: int, cols: int, out: torch.Tensor):
        if qt in SGL_TYPES:
            from sgl_kernel import ggml_dequantize

            blck, ts = SGL_TYPES[qt]
            rowbytes = cols // blck * ts
            w = raw[:n * pb].view(n * rows, rowbytes)
            out.copy_(ggml_dequantize(w, qt, n * rows, cols, torch.bfloat16).view(n, rows, cols))
        else:
            DQ.dequant_experts(raw, pb, qt, n, rows, cols, out)

    def _group_body(self, layer, layer_idx: int, gi: int, base: int, n: int, g):
        slot = self.slots[gi % 2]
        types, pb = self._layer_types[layer_idx], self._part_bytes[layer_idx]
        N, K = self.N, self.K
        x, topk_weights, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        # CUTLASS fc1 rows are [up; gate]
        self._dequant(slot.raw[1], pb[1], types[1], n, N, K, self.w13_bf16[:n, :N])
        self._dequant(slot.raw[0], pb[0], types[0], n, N, K, self.w13_bf16[:n, N:])
        self._dequant(slot.raw[2], pb[2], types[2], n, K, N, self.w2_bf16[:n])
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        out.zero_()
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=self.w13_bf16[:n],
            fc2_expert_weights=self.w2_bf16[:n],
            output_dtype=torch.bfloat16,
            quant_scales=None,
            input_sf=None,
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            ep_size=self._ep_size,
            ep_rank=self._ep_rank,
            activation_type=self._activation_type,
            tune_max_num_tokens=self._tune_tokens,
            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )
        acc.add_(out)


# the routes of a group go through the MMQ kernels this many at a time: the gathered rows, gate, up and down
# (about 60 KB a route) stay the same size however long the chunk (0: a group's routes at once)
MMQ_ROUTES = int(os.environ.get("KT_GGUF_STREAM_MMQ_ROUTES", "8192") or 0)
# the eager MMQ body adds into the wrapper's output instead of a chunk-sized accumulator of its own
INTO = os.environ.get("KT_GGUF_STREAM_INTO", "1") == "1"


class _MmqGroup:
    """The group body on raw GGUF blocks, run eagerly (no CUDA graph: the routes to a group are gathered
    exactly, which needs their count on the host; a dozen launches a group against ~15 s a chunk is nothing):
    gate and up as one-route MoE matmuls on the gathered rows, V4.1's SwiGLU clamp, down the same, the weighted
    rows added back to the accumulator by token."""

    def _mmq_init(self, layer):
        from sglang.srt.layers.moe import kt_gguf_moe_mmq as MMQ

        self._mmq = MMQ
        # the slots come sorted by expert (the per-layer argsort): the extension's sorted entry skips the ids helper
        self._mmq_sorted = hasattr(MMQ, "moe_mmq_sorted") and os.environ.get("KT_GGUF_STREAM_MMQ_SORTED", "1") == "1"
        lim = getattr(getattr(layer, "moe_runner_config", None), "swiglu_limit", None)
        self._mmq_limit = float(lim) if lim else None
        # slots with room for the bytes the kernel reads past the last row (zeroed at staging)
        self.slots = [P._Slot(self.G, [st + 512 for st in self._strides], self.device) for _ in range(2)]
        # the dequant body's bf16 expansion (566 MB) and its CUTLASS workspace (2 GB at 16K rows) are never used
        # here: drop the one, satisfy the parent's guard for the other
        self.w13_bf16 = self.w2_bf16 = None
        self._workspace, self._workspace_tokens = torch.empty(0, dtype=torch.uint8, device=self.device), 1 << 40
        _log(f"rank {self.rank}: MMQ group body (raw GGUF blocks, eager, swiglu limit {self._mmq_limit})")

    def _mmq_graph_inputs(self, T: int, x, physical_ids, topk_weights):
        """The per-chunk-size inputs in the lend scratch: no graph holds them, so they go with the window (the
        default pool kept 3 x 117 MB a chunk size across windows, 10-25); the scratch keeps their addresses and
        every call rewrites their contents."""
        t_in = time.perf_counter()
        try:
            return self._mmq_graph_inputs_(T, x, physical_ids, topk_weights)
        finally:
            self._t_inputs = time.perf_counter() - t_in

    def _mmq_graph_inputs_(self, T: int, x, physical_ids, topk_weights):
        n = x.shape[0]
        # the body is eager and gathers its routes exactly: no padding rows, and the caller's tensors as they are
        into = getattr(self, "_into", None)
        acc = into
        if acc is None:
            acc = self._gin.get(n)
            if acc is None:
                # one size at a time: the scratch is mapped again at every window's start, and every size ever
                # seen kept in it made that mapping fail under a 16 GB card with hot experts -- torch_memory_saver's
                # resume ran out of memory and the first touch of the unmapped slot was an illegal address (10-25)
                self._gin.clear()
                acc = P._scratch_empty((n, self.K), torch.bfloat16, self.device)
                self._gin[n] = acc
            acc.zero_()
        g = {"x": x, "ids": physical_ids if physical_ids.dtype == torch.int32 else physical_ids.to(torch.int32),
             "w": topk_weights, "acc": acc}
        return self._mmq_sort(g)

    def _mmq_sort(self, g):
        # the routes sorted by expert, once a layer: every group takes a slice whose bounds the host knows
        # from one transfer here, so the group bodies enqueue without a sync each (torch.nonzero a group
        # stalled the host until that group's DMA had landed, which serialized DMA and compute: a layer took
        # ~900 ms for ~125 ms of kernels and ~330 ms of copies, 10-25)
        flat = g["ids"].reshape(-1)
        E = self.E
        key = torch.where(flat < 0, torch.full_like(flat, E), flat)  # routes to no expert last
        order = torch.argsort(key, stable=True)
        counts = torch.bincount(key, minlength=E + 1)[:E]
        bounds = torch.zeros(E + 1, dtype=torch.int64, device=flat.device)
        torch.cumsum(counts, 0, out=bounds[1:])
        g["mmq_order"] = order
        g["mmq_e"] = key.index_select(0, order).to(torch.int32)
        g["mmq_bounds32"] = bounds.to(torch.int32)
        t_sync = time.perf_counter()
        g["mmq_bounds"] = bounds.tolist()  # the one host sync of the layer: it also drains the layer's attention and
        self._presync = time.perf_counter() - t_sync  # dense work enqueued before (the timeline reports the wait)
        return g

    def _mmq_group_body(self, layer, layer_idx: int, gi: int, base: int, n: int, g):
        slot = self.slots[gi % 2]
        bounds = g["mmq_bounds"]
        lo, hi = bounds[base], bounds[base + n]
        if hi == lo:
            return
        step = MMQ_ROUTES if MMQ_ROUTES > 0 else hi - lo
        for blo in range(lo, hi, step):
            self._mmq_routes(layer_idx, gi, base, n, g, slot, blo, min(hi, blo + step))

    def _mmq_routes(self, layer_idx: int, gi: int, base: int, n: int, g, slot, lo: int, hi: int):
        """Routes lo..hi of the sorted list (all of them to experts base..base+n) through gate, up and down."""
        types, pb = self._layer_types[layer_idx], self._part_bytes[layer_idx]
        N, K = self.N, self.K
        x, topk_weights, ids32, acc = g["x"], g["w"], g["ids"], g["acc"]
        T, topk = ids32.shape
        r = g["mmq_order"][lo:hi]
        t = torch.div(r, topk, rounding_mode="floor")
        sidx = r - t * topk
        e = (g["mmq_e"][lo:hi] - base).view(-1, 1)
        prof = MMQ_PROFILE and layer_idx < MMQ_PROFILE
        if prof:
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
            ev[0].record()
        xc = x.index_select(0, t)
        if prof:
            ev[1].record()
        blck = lambda qt: (DQ.BLOCK_INFO.get(qt) or SGL_TYPES[qt])
        rb = [cols // blck(types[pi])[0] * blck(types[pi])[1] for pi, cols in ((0, K), (1, K), (2, N))]
        wg = slot.raw[0][:n * pb[0]].view(n, N, rb[0])
        wu = slot.raw[1][:n * pb[1]].view(n, N, rb[1])
        wd = slot.raw[2][:n * pb[2]].view(n, K, rb[2])
        if self._mmq_sorted:
            # the block's own bounds: each expert's range clipped to lo..hi
            b32 = (g["mmq_bounds32"][base:base + n + 1] - lo).clamp_(0, hi - lo)
            e1 = e.view(-1)
            gate = self._mmq.moe_mmq_sorted(xc, wg, e1, b32, types[0], N).view(-1, N)
            if prof:
                ev[2].record()
            up = self._mmq.moe_mmq_sorted(xc, wu, e1, b32, types[1], N).view(-1, N)
        else:
            gate = self._mmq.moe_mmq(xc, wg, e, types[0], N, False).view(-1, N)
            if prof:
                ev[2].record()
            up = self._mmq.moe_mmq(xc, wu, e, types[1], N, False).view(-1, N)
        if prof:
            ev[3].record()
        if self._mmq_limit is not None:
            gate.clamp_(max=self._mmq_limit)
            up.clamp_(-self._mmq_limit, self._mmq_limit)
        h = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
        if prof:
            ev[4].record()
        if self._mmq_sorted:
            down = self._mmq.moe_mmq_sorted(h, wd, e1, b32, types[2], K).view(-1, K)
        else:
            down = self._mmq.moe_mmq(h, wd, e, types[2], K, False).view(-1, K)
        if prof:
            ev[5].record()
        down.mul_(topk_weights[t, sidx].to(torch.float32).unsqueeze(1))
        acc.index_add_(0, t, down.to(acc.dtype))
        if prof:
            ev[6].record()
            ev[6].synchronize()
            ms = [ev[i].elapsed_time(ev[i + 1]) for i in range(6)]
            _log(f"rank {self.rank}: mmq body layer {layer_idx} group {gi}: {r.numel()} routes of {n} experts "
                 f"(types {types}): gather {ms[0]:.2f} gate {ms[1]:.2f} up {ms[2]:.2f} silu {ms[3]:.2f} "
                 f"down {ms[4]:.2f} scale+add {ms[5]:.2f} ms")


class _Eager:
    def __init__(self, fn):
        self.fn = fn

    def replay(self):
        self.fn()


class KTStreamPrefillGGUFBank(_SglDequant, P.KTStreamPrefillGGUF):
    """The pinned-bank edition (experts in RAM) with the sgl_kernel dequant: KT_GGUF_STREAM_SOURCE=bank."""


class KTStreamPrefillGGUFNVMe(_MmqGroup, _SglDequant, P.KTStreamPrefillGGUF):
    def __init__(self, method, layer):
        super().__init__(method, layer)
        if MMQ_ON:
            _MmqGroup._mmq_init(self, layer)
        self._tl = {}
        self._tm = _TensorMap(self.path)
        self._pool = ThreadPoolExecutor(
            max_workers=READERS, thread_name_prefix="kt-gguf-nvme", initializer=_pin_io_thread
        )
        self._peek = self._unpin = None
        self._cache_bytes = 0
        if FROM_CACHE:
            try:
                import kt_kernel.kt_kernel_ext as _ext

                self._peek, self._unpin = getattr(_ext, "kt_gguf_cache_peek", None), getattr(_ext, "kt_gguf_cache_unpin", None)
            except ImportError:
                pass
            if self._peek is None:
                _log(f"rank {self.rank}: kt_kernel_ext has no kt_gguf_cache_peek: every expert read from the files")
        # a reader's preadv ends on a Python turn: the forward's thread holds the GIL between its enqueues,
        # and the default 5 ms turn held every read's completion that long
        if SWITCH_MS > 0:
            sys.setswitchinterval(SWITCH_MS / 1e3)
        self._ring: List[_RingBuf] = [_RingBuf(self._strides, self.G) for _ in range(RING)]
        self._free: collections.deque = collections.deque(self._ring)
        self._ready: Dict[Tuple[int, int], _RingBuf] = {}
        self._cv = threading.Condition()
        self._plan: collections.deque = collections.deque()  # (layer_idx, gi, base, n) still to read
        self._queued: set = set()  # (layer_idx, gi) planned or ready, not yet taken
        # the layer the model called after each one last time: what to read ahead (the call order of a
        # V4.1 prefill is not 0..39 in one pass; it is learned, not assumed)
        self._next_after: Dict[int, int] = {}
        self._prev_layer: Optional[int] = None
        self._calls_logged = 0
        self._stop = False
        self._feeder = threading.Thread(target=self._feed, name="kt-gguf-nvme-feeder", daemon=True)
        self._feeder.start()
        self._bytes = 0
        self._t_read = 0.0
        mb = sum(st * self.G + 2 * ALIGN for st in self._strides) * RING / 1e6
        _log(f"rank {self.rank}: experts read from the GGUF files on demand, {READERS} readers, "
             f"ring of {RING} groups ({mb:.0f} MB pinned); no banks")

    # -- the plan: groups in the order the layers are called --------------------------------------

    def _groups(self, method) -> List[Tuple[int, int]]:
        first = 0 if getattr(method.gpu_method, "streams_hot_experts", False) else method.num_gpu_experts
        lo, hi = first, self.E
        if self.ep:
            # this rank's contiguous share of the experts, in whole groups, in proportion to its GPU's weight
            n_groups = -(-(self.E - first) // self.G)
            w = _ep_weights(self.world)
            cut = [round(n_groups * sum(w[:r]) / sum(w)) for r in range(self.world + 1)]
            lo = first + cut[self.rank] * self.G
            hi = min(self.E, first + cut[self.rank + 1] * self.G)
        return [(b, min(self.G, hi - b)) for b in range(lo, hi, self.G)]

    def _plan_layer(self, layer_idx: int, groups, front: bool = False) -> None:
        """Queue the layer's groups not already queued or ready; front: before everything else planned."""
        with self._cv:
            new = [(layer_idx, gi, base, n) for gi, (base, n) in enumerate(groups) if (layer_idx, gi) not in self._queued]
            if not new:
                return
            self._queued.update((e[0], e[1]) for e in new)
            if front:
                self._plan.extendleft(reversed(new))
            else:
                self._plan.extend(new)
            self._dispatch_locked()
            self._cv.notify_all()

    def _drop_others(self, layer_idx: int) -> int:
        """Under the lock: unqueue planned groups and give back ready buffers of other layers (a read-ahead that
        guessed the next layer wrong would otherwise hold the ring). Returns how many buffers came back."""
        keep = collections.deque(e for e in self._plan if e[0] == layer_idx)
        for e in self._plan:
            if e[0] != layer_idx:
                self._queued.discard((e[0], e[1]))
        self._plan = keep
        back = 0
        for key in [k for k in self._ready if k[0] != layer_idx]:
            rb = self._ready.pop(key)
            self._queued.discard(key)
            for fut in rb.futs:
                fut.cancel()
            rb.key, rb.futs = None, []
            self._free.append(rb)
            back += 1
        if back:
            self._cv.notify_all()
        return back

    def _dispatch_locked(self) -> int:
        """Under the lock: every planned group a free ring buffer can take gets its three reads submitted.
        Called by whoever holds the lock (the forward's thread at a give-back or a plan, the feeder otherwise):
        a feeder thread alone dispatched one group per GIL turn while the forward's thread enqueued flat out,
        ~5 ms a group, and the readers sat idle 90% of a layer (10-25)."""
        n_disp = 0
        while not self._stop and self._plan and self._free:
            layer_idx, gi, base, n = self._plan.popleft()
            rb = self._free.popleft()
            rb.key = (layer_idx, gi)
            rb.futs = []
            rb.peeked = None
            if self._peek is not None:
                ptrs = [self._peek(layer_idx, base + j) for j in range(n)]
                if any(p_ is not None for p_ in ptrs):
                    rb.peeked = (layer_idx, base, ptrs)
            for pi, proj in enumerate(("gate", "up", "down")):
                f, off, _ = self._tm.where[f"blk.{layer_idx}.ffn_{proj}_exps.weight"]
                pb = self._part_bytes[layer_idx][pi]
                if rb.peeked is None:
                    rb.futs.append(self._pool.submit(self._read_one, rb, pi, f, off + base * pb, n * pb))
                else:
                    rb.futs.append(self._pool.submit(self._read_mixed, rb, pi, f, off + base * pb, pb, n, rb.peeked[2]))
            old = self._ready.get(rb.key)
            if old is not None:
                # the same group planned twice: the earlier buffer would be lost with its reads
                _log(f"group {rb.key} read twice; the first copy goes back")
                self._free.append(old)
            self._ready[rb.key] = rb
            n_disp += 1
        if n_disp:
            self._cv.notify_all()
        return n_disp

    def _feed(self) -> None:
        """Dispatches whenever a plan and a free buffer meet and no one else did it."""
        while not self._stop:
            with self._cv:
                while not self._stop and (not self._plan or not self._free):
                    self._cv.wait()
                if self._stop:
                    return
                self._dispatch_locked()

    def _read_mixed(self, rb: _RingBuf, pi: int, f: str, off: int, pb: int, n: int, ptrs) -> None:
        """The group's projection pi: runs of experts absent from the cache read from the file (aligned, into
        their place), the resident ones copied from the cache's RAM."""
        t0 = time.perf_counter()
        buf = rb.buf[pi]
        lo = off // ALIGN * ALIGN
        rb.head[pi] = off - lo
        j = 0
        while j < n:
            if ptrs[j] is not None:
                j += 1
                continue
            j0 = j
            while j < n and ptrs[j] is None:
                j += 1
            r_off = off + j0 * pb
            r_lo = r_off // ALIGN * ALIGN
            _read_into(self._tm, f, r_off, (j - j0) * pb, buf[r_lo - lo:])
            self._bytes += (j - j0) * pb
        base_ptr = buf.data_ptr() + (off - lo)
        for j, p_ in enumerate(ptrs):
            if p_ is not None:
                ctypes.memmove(base_ptr + j * pb, p_[pi], pb)
                self._cache_bytes += pb
        self._t_read += time.perf_counter() - t0

    def _read_one(self, rb: _RingBuf, pi: int, f: str, off: int, nbytes: int) -> None:
        t0 = time.perf_counter()
        rb.head[pi] = _read_into(self._tm, f, off, nbytes, rb.buf[pi])
        self._t_read += time.perf_counter() - t0
        self._bytes += nbytes

    def _take(self, layer_idx: int, gi: int) -> _RingBuf:
        with self._cv:
            waited = 0.0
            while (layer_idx, gi) not in self._ready:
                if not self._free and all(k[0] != layer_idx for k in self._ready):
                    # the ring is full of another layer's groups (a wrong read-ahead): they go back
                    back = self._drop_others(layer_idx)
                    _log(f"group ({layer_idx}, {gi}) wanted, the ring held other layers' groups: {back} given back")
                if not self._cv.wait(timeout=30.0):
                    waited += 30.0
                    _log(f"waiting {waited:.0f} s for group ({layer_idx}, {gi}): ready {sorted(self._ready)}, "
                         f"free {len(self._free)}, plan head {list(self._plan)[:3]} ({len(self._plan)} queued)")
            rb = self._ready.pop((layer_idx, gi))
            self._queued.discard((layer_idx, gi))
        for fut in rb.futs:
            fut.result()
        if rb.peeked is not None:
            layer_idx, base, ptrs = rb.peeked
            for j, p_ in enumerate(ptrs):
                if p_ is not None:
                    self._unpin(layer_idx, base + j)
            rb.peeked = None
        return rb

    def _graph_inputs(self, T: int, x, physical_ids, topk_weights):
        if MMQ_ON:
            return self._mmq_graph_inputs(T, x, physical_ids, topk_weights)
        return super()._graph_inputs(T, x, physical_ids, topk_weights)

    def _graph_for(self, layer, layer_idx: int, T: int, gi: int, base: int, n: int, g):
        if MMQ_ON:
            if TIMELINE and layer_idx < TIMELINE:
                def body():
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    self._mmq_group_body(layer, layer_idx, gi, base, n, g)
                    b.record()
                    self._tl.setdefault(gi, {})["body"] = (a, b)
                return _Eager(body)
            return _Eager(lambda: self._mmq_group_body(layer, layer_idx, gi, base, n, g))
        return super()._graph_for(layer, layer_idx, T, gi, base, n, g)

    def _reference_check(self, method, layer_idx, x, ids, w, acc):
        """KT_GPU_STREAM_SELFTEST=2 without banks: the used experts' bytes read from the GGUF, dequantized on the
        device, V4.1's SwiGLU clamp applied (the CPU path's semantics), fp32 math, against acc."""
        import numpy as np

        torch.cuda.current_stream(self.device).synchronize()
        types, part_bytes = self._layer_types[layer_idx], self._part_bytes[layer_idx]
        N, K = self.N, self.K
        first = 0 if getattr(method.gpu_method, "streams_hot_experts", False) else method.num_gpu_experts
        lim = getattr(getattr(self, "_mmq_limit", None), "real", None)
        if lim is None:
            l = getattr(getattr(method, "moe_runner_config", None), "swiglu_limit", None) if hasattr(method, "moe_runner_config") else None
            lim = float(l) if l else 10.0
        ref = torch.zeros_like(acc, dtype=torch.float32)
        ids64 = ids.to(torch.int64)
        xf = x.float()
        used = 0
        bufs = [torch.empty((1, N, K), dtype=torch.bfloat16, device=self.device),
                torch.empty((1, N, K), dtype=torch.bfloat16, device=self.device),
                torch.empty((1, K, N), dtype=torch.bfloat16, device=self.device)]
        for e in range(first, self.E):
            rows, cols = torch.where(ids64 == e)
            if rows.numel() == 0:
                continue
            used += 1
            for pi, proj in enumerate(("gate", "up", "down")):
                f, off, _ = self._tm.where[f"blk.{layer_idx}.ffn_{proj}_exps.weight"]
                pb = part_bytes[pi]
                buf = np.empty(pb, dtype=np.uint8)
                fd = os.open(f, os.O_RDONLY)
                got = 0
                while got < pb:
                    k = os.preadv(fd, [memoryview(buf)[got:]], off + e * pb + got)
                    got += k
                os.close(fd)
                raw = torch.from_numpy(buf).to(self.device)
                r_, c_ = (N, K) if pi < 2 else (K, N)
                self._dequant(raw, pb, types[pi], 1, r_, c_, bufs[pi])
            h = xf[rows]
            gate = (h @ bufs[0][0].float().t()).clamp(max=lim)
            up = (h @ bufs[1][0].float().t()).clamp(-lim, lim)
            a = torch.nn.functional.silu(gate) * up
            o = (a @ bufs[2][0].float().t()) * w[rows, cols].float().unsqueeze(1)
            ref.index_add_(0, rows, o)
        d = (acc.float() - ref).abs()
        cos = torch.nn.functional.cosine_similarity(acc.float().flatten(), ref.flatten(), dim=0).item()
        _log(f"rank{self.rank} refcheck layer{layer_idx} T={x.shape[0]} experts_used={used} (swiglu limit {lim}): cos={cos:.5f} "
             f"max|acc-ref|={d.max().item():.4f} mean={d.mean().item():.5f} ref_mean|.|={ref.abs().mean().item():.5f}")

    def _give_back(self, rb: _RingBuf) -> None:
        rb.key, rb.futs = None, []
        with self._cv:
            self._free.append(rb)
            self._dispatch_locked()
            self._cv.notify_all()

    # -- the parent's bank API, served from the ring ----------------------------------------------

    def _bank(self, layer_idx: int):
        return None

    def build_all_banks(self):
        _log(f"rank {self.rank}: nothing to build (experts stream from the files)")

    def _stage(self, layer_idx: int, gi: int, base: int, n: int, stats):
        part_bytes = self._part_bytes[layer_idx]
        slot = self.slots[gi % 2]
        t0 = time.perf_counter()
        rb = self._take(layer_idx, gi)
        stats["read_wait"] = stats.get("read_wait", 0.0) + time.perf_counter() - t0
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_used:
                self.transfer_stream.wait_event(slot.compute_done)
            if TIMELINE and layer_idx < TIMELINE:
                a = torch.cuda.Event(enable_timing=True)
                a.record(self.transfer_stream)
                b = torch.cuda.Event(enable_timing=True)
                self._tl.setdefault(gi, {})["h2d"] = (a, b)
                self._tl[gi]["read_wait"] = time.perf_counter() - t0
            for pi in range(3):
                pb, h = part_bytes[pi], rb.head[pi]
                slot.raw[pi][:n * pb].view(n, pb).copy_(rb.buf[pi][h:h + n * pb].view(n, pb), non_blocking=True)
                if MMQ_ON:  # the bytes the MMQ kernel reads past the last row: zero (a stale float scale is a NaN)
                    slot.raw[pi][n * pb:n * pb + 512].zero_()
            slot.dma_done.record(self.transfer_stream)
            if TIMELINE and layer_idx < TIMELINE:
                self._tl[gi]["h2d"][1].record(self.transfer_stream)
        if SYNC_DMA:
            slot.dma_done.synchronize()  # KT_GGUF_STREAM_SYNC_DMA=1: a faulting copy is reported here
        # the buffer goes back once its DMA is done: the previous group's DMA on the other slot is older
        prev = getattr(self, "_staged", None)
        self._staged = (rb, slot)
        if prev is not None:
            prev[1].dma_done.synchronize()
            self._give_back(prev[0])
        stats["dma_enq"] += time.perf_counter() - t0

    def _flush_staged(self) -> None:
        prev = getattr(self, "_staged", None)
        if prev is not None:
            prev[1].dma_done.synchronize()
            self._give_back(prev[0])
            self._staged = None
        tl = getattr(self, "_tl", None)
        if tl:
            torch.cuda.synchronize(self.device)
            gis = sorted(tl)
            first = tl[gis[0]]["h2d"][0]
            h2d = sum(tl[g]["h2d"][0].elapsed_time(tl[g]["h2d"][1]) for g in gis if "h2d" in tl[g])
            body = sum(tl[g]["body"][0].elapsed_time(tl[g]["body"][1]) for g in gis if "body" in tl[g])
            last = max((tl[g]["body"][1] for g in gis if "body" in tl[g]), key=lambda e: first.elapsed_time(e))
            wall = first.elapsed_time(last)
            rw = sum(tl[g].get("read_wait", 0.0) for g in gis) * 1e3
            # the gap before each body: from its H2D's end to its start
            gaps = [tl[g]["h2d"][1].elapsed_time(tl[g]["body"][0]) for g in gis if "h2d" in tl[g] and "body" in tl[g]]
            _log(f"rank {self.rank}: timeline {len(gis)} groups: inputs {getattr(self, '_t_inputs', 0.0) * 1e3:.0f} ms host "
                 f"(of it the sync {getattr(self, '_presync', 0.0) * 1e3:.0f}), since the layer's call {(time.perf_counter() - self._t_call) * 1e3:.0f} ms, "
                 f"H2D {h2d:.0f} ms, bodies {body:.0f} ms, wall first copy -> last body "
                 f"{wall:.0f} ms, host read waits {rw:.0f} ms, H2D-end -> body-start gaps mean {sum(gaps) / max(1, len(gaps)):.1f} max {max(gaps) if gaps else 0:.1f} ms")
            self._tl = {}

    # -- one layer: plan this layer and the next, then the parent's loop -------------------------------

    @property
    def accumulates_into(self) -> bool:
        """run(..., into=output) adds this rank's streamed experts into output and returns it."""
        return MMQ_ON and INTO

    def run(self, method, layer, dispatch_output, physical_ids: torch.Tensor, into=None) -> torch.Tensor:
        self._into = into if self.accumulates_into else None
        layer_idx = method.kt_config.layer_idx
        groups = self._groups(method)
        if self._prev_layer is not None:
            self._next_after[self._prev_layer] = layer_idx
        self._prev_layer = layer_idx
        self._plan_layer(layer_idx, groups, front=True)
        # read ahead the layer that followed this one last time (none known on the first pass)
        nxt = self._next_after.get(layer_idx)
        t0 = time.perf_counter()
        self._t_call = t0
        b0, r0, c0 = self._bytes, self._t_read, self._cache_bytes
        try:
            acc = super().run(method, layer, dispatch_output, physical_ids)
        finally:
            self._into = None
            self._flush_staged()
        if nxt is not None:
            self._plan_layer(nxt, groups)
        if P.TIMING:
            torch.cuda.current_stream(self.device).synchronize()
            dt = time.perf_counter() - t0
            _log(f"rank{self.rank} layer{layer_idx} {(self._bytes - b0) / 1e9:.2f} GB read "
                 f"({(self._bytes - b0) / max(1e-6, dt) / 1e9:.1f} GB/s of wall), {(self._cache_bytes - c0) / 1e9:.2f} GB from the cache, "
                 f"reader time {self._t_read - r0:.1f} s over {READERS} threads, wall {dt:.2f} s")
        return acc


EDITION = os.environ.get("KT_GGUF_STREAM_SOURCE", "nvme")
_CLASS = KTStreamPrefillGGUFBank if EDITION == "bank" else KTStreamPrefillGGUFNVMe
_instance = None
_failed = False


def stream_prefill_init_early(method, layer) -> None:
    global _instance, _failed
    if not P.EARLY_INIT or P.STREAM_THRESHOLD <= 0 or _failed or _instance is not None:
        return
    try:
        _instance = _CLASS(method, layer)
        if EDITION == "bank":
            _instance.build_all_banks()
    except Exception:
        _failed = True
        _log("early init failed:\n" + traceback.format_exc())
        raise


def stream_prefill_for(method, layer, num_tokens: int):
    global _instance, _failed
    if P.STREAM_THRESHOLD <= 0 or _failed:
        return None
    if _instance is not None and getattr(_instance, "_calls_logged", 120) < 120:
        # the first calls' order and widths, to see how a prefill walks the layers (KT_GPU_STREAM_TIMING)
        _instance._calls_logged += 1
        if P.TIMING:
            _log(f"call layer {method.kt_config.layer_idx} rows {num_tokens}"
                 f"{'' if num_tokens >= P.STREAM_THRESHOLD else ' (below the threshold, CPU)'}")
    if num_tokens < P.STREAM_THRESHOLD or torch.cuda.is_current_stream_capturing():
        return None
    if _instance is None:
        _log(f"rank {get_tensor_model_parallel_rank()}: first qualifying call, layer {method.kt_config.layer_idx}, "
             f"tokens={num_tokens}")
        try:
            _instance = _CLASS(method, layer)
        except Exception:
            _failed = True
            _log("init failed:\n" + traceback.format_exc())
            raise
    return _instance
