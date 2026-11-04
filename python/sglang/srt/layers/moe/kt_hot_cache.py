"""Decode-adaptive hot experts for the KT hybrid with GGUF experts
(KT_HOT_DYNAMIC=1): the GPU slots of every layer follow the decode's own
routing instead of a static map.

Measured on a 38K brief + 512 generated tokens: a corpus map catches 0.29 /
0.42 / 0.53 of the decode routes with 32 / 64 / 96 slots, the decode's own
top-n 0.50 / 0.69 / 0.80. Every route a
slot catches is one expert the CPU does not compute that layer.

How it works
- tables on the GPU: log2slot [L, E] (slot or -1), log2cpu [L, E] (-1 or the
  logical id itself), both int32 like the router's ids, and counts [L, E]; the
  captured decode graph only does table lookups and an index_add_ on them,
  so the policy runs outside the graph, between steps (model_runner.forward
  -> before_forward), and changes the tables in place.
- the CPU pools are in logical order (launch with KT_EXPERT_PLACEMENT=none);
  the CPU gets the logical ids with the resident ones set to -1, the GPU
  method the slot ids with the non-resident ones -1.
- a swap copies the expert's TP slice from the GGUF streamer's pinned banks
  (every expert of every layer, this rank's rows) into the slot; the lend's
  host mirror of the slot gets the same bytes, so the restore after a prefill
  keeps the swapped experts.
- policy every KT_HOT_INTERVAL decode (or MTP verify) steps: counts *= KT_HOT_DECAY; per
  layer, the coldest residents give way to the hottest non-residents when
  candidate > KT_HOT_MARGIN * resident, at most KT_HOT_SWAPS per layer.
- KT_HOT_SEED=<placement json> fills the slots at start from a map
  (physical_to_logical_map, first N per layer); without it they start as
  logical 0..N-1.
- KT_HOT_KV_LEND=1 (kt_hot_kv_lend.py): more slots per layer, ids N.., in free
  KV pages. A slot entry is the resident logical id, -1 for an empty slot the
  policy fills first, -2 for a slot that does not exist (not lent now). The
  kernels read every slot through per-layer pointer tables; base tables
  (log2slot_base / log2cpu_base) leave the lent residents on the CPU for the
  kernel path that cannot read the tables (kt_gguf_moe_multi.reads_table).
"""
import collections
import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional

import torch

ENABLED = os.environ.get("KT_HOT_DYNAMIC") == "1"
INTERVAL = int(os.environ.get("KT_HOT_INTERVAL", "32"))
MAX_SWAPS = int(os.environ.get("KT_HOT_SWAPS", "8"))
DECAY = float(os.environ.get("KT_HOT_DECAY", "0.9"))
MARGIN = float(os.environ.get("KT_HOT_MARGIN", "1.5"))
SEED = os.environ.get("KT_HOT_SEED")
LOG_EVERY = int(os.environ.get("KT_HOT_LOG", "0"))  # rounds between log lines (0: only the first and every 64th)
KV_LEND = os.environ.get("KT_HOT_KV_LEND", "0") == "1"
KV_LEND_MAX = int(os.environ.get("KT_HOT_KV_LEND_MAX", "64"))  # lent slots per layer at most (table size)
# KT_HOT_FUSED_ROUTE=1: route() as one Triton kernel (the two table lookups and the count update) instead of
# index_add_ + two gathers (+ casts): 4-5 launches, ~10 us a layer on the critical path between the router and the
# CPU submit; integer tables, the counts stay exact integers, so the result is identical. =0 keeps the torch ops.
FUSED_ROUTE = os.environ.get("KT_HOT_FUSED_ROUTE", "1") == "1"
# KT_HOT_BANKS=own: build the cache's own pinned expert banks even when the GGUF streamer is loaded (otherwise
# the streamer's banks are shared; without the streamer the own banks are the only source)
OWN_BANKS = os.environ.get("KT_HOT_BANKS", "auto")
# KT_HOT_SOURCE=file: a swap reads the expert's rows from the GGUF file mapping itself (the hot method's reader),
# no pinned bank of every expert: for hosts whose RAM cannot hold the experts (the NVMe-fed streamer, 64 GB)
SOURCE = os.environ.get("KT_HOT_SOURCE", "banks")
# with the file source the swap's read and copy run on a worker thread and a side stream (KT_HOT_ASYNC=0: on the
# forward's thread, 0.5-1 s a round of 80 swaps from the NVMe, 10-25): the expert leaves the table when the swap is
# decided (its routes go to the CPU), the new one enters on a later round once its copy has landed
ASYNC = os.environ.get("KT_HOT_ASYNC", "1") == "1"
STAGES = int(os.environ.get("KT_HOT_STAGES", "16"))  # pinned staging buffers (swaps in flight at most)
_ROUTE_BLOCK = 64  # routes per call at most for the fused kernel (decode / verify windows: tokens x top_k)


def _route_kernel_fn():
    import triton
    import triton.language as tl

    @triton.jit
    def _route_kernel(ids_ptr, slot_tab_ptr, cpu_tab_ptr, counts_ptr, slot_out_ptr, cpu_out_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        ids = tl.load(ids_ptr + offs, mask=m, other=0)
        tl.store(slot_out_ptr + offs, tl.load(slot_tab_ptr + ids, mask=m, other=-1), mask=m)
        tl.store(cpu_out_ptr + offs, tl.load(cpu_tab_ptr + ids, mask=m, other=-1), mask=m)
        tl.atomic_add(counts_ptr + ids, 1.0, mask=m)

    return _route_kernel


_route_kernel = None
EMPTY, ABSENT, PENDING = -1, -2, -3  # PENDING: a swap decided, its copy not landed (never chosen, not computed)


def _log(msg):
    print(f"[kt-hot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class HotCache:
    def __init__(self, num_layers: int, num_experts: int, slots: int, device: torch.device, rank: int):
        self.L, self.E, self.N, self.device, self.rank = num_layers, num_experts, slots, device, rank
        # int32 like the router's topk ids: route() is two lookups, no casts
        self.log2slot_cpu = torch.full((num_layers, num_experts), -1, dtype=torch.int32)
        self.log2slot = torch.full((num_layers, num_experts), -1, dtype=torch.int32, device=device)
        # the CPU's view: -1 for a resident expert, the logical id otherwise
        self.log2cpu_cpu = torch.arange(num_experts, dtype=torch.int32).repeat(num_layers, 1)
        self.log2cpu = self.log2cpu_cpu.to(device)
        self.counts = torch.zeros((num_layers, num_experts), dtype=torch.float32, device=device)
        self.ones = torch.ones((1,), dtype=torch.float32, device=device)
        self.S = slots + (KV_LEND_MAX if KV_LEND else 0)  # slot ids per layer (resident + lent)
        self.slot2log: List[List[int]] = [[EMPTY] * slots + [ABSENT] * (self.S - slots) for _ in range(num_layers)]
        # lent slot (layer, slot) -> (w13 view, w2 view) in borrowed KV pages
        self.lent_views: Dict[tuple, tuple] = {}
        if KV_LEND:
            self.log2slot_base_cpu = self.log2slot_cpu.clone()
            self.log2cpu_base_cpu = self.log2cpu_cpu.clone()
            self.log2slot_base = self.log2slot.clone()
            self.log2cpu_base = self.log2cpu.clone()
        self.layers: Dict[int, tuple] = {}  # layer_idx -> (hot method, FusedMoE layer)
        self._mirrors: Dict[tuple, Optional[torch.Tensor]] = {}
        self._counts_host = torch.zeros((num_layers, num_experts), dtype=torch.float32).pin_memory()
        self._counts_event: Optional[torch.cuda.Event] = None
        self._q8_banks: Dict[int, torch.Tensor] = {}
        self._own_banks: Dict[int, list] = {}
        self.ready = False
        # the asynchronous swaps (SOURCE=file, ASYNC): worker thread, side stream, pinned staging, landed copies
        self._q: "queue.Queue" = queue.Queue()
        self._done: "collections.deque" = collections.deque()
        self._worker: Optional[threading.Thread] = None
        self._side: Optional[torch.cuda.Stream] = None
        self._stages: "queue.Queue" = queue.Queue()
        self._go = threading.Event()  # copies may be issued (cleared across a prefill window: the slots are lent)
        self._go.set()
        self._lock = threading.Lock()
        self._after = None  # event: the restore's copies, which a swap's copy follows
        self._t_read = self._t_wait = self._t_issue = 0.0
        self._n_swapped = 0
        self._inflight: "collections.deque" = collections.deque()  # (event, stage): the worker recycles the stages
        self.step = 0
        self.rounds = 0
        self.swaps = 0
        self.bytes = 0
        self.swap_s = 0.0

    # -- graph side ----------------------------------------------------------

    def route(self, layer_idx: int, ids: torch.Tensor):
        """ids: logical [T, k] (>= 0). Returns (slot ids with -1 for CPU experts,
        logical ids with -1 for the resident ones). Captured in the graph."""
        flat = ids.reshape(-1)
        slot_tab, cpu_tab = self.log2slot, self.log2cpu
        if KV_LEND:
            from sglang.srt.layers.moe import kt_gguf_moe_multi as _multi

            if not _multi.reads_table(ids.shape[0]):
                # a kernel that addresses base + slot * stride: lent residents
                # are computed by the CPU in this call
                slot_tab, cpu_tab = self.log2slot_base, self.log2cpu_base
        if FUSED_ROUTE and flat.numel() <= _ROUTE_BLOCK and flat.is_contiguous():
            global _route_kernel
            if _route_kernel is None:
                _route_kernel = _route_kernel_fn()
            slot_ids = torch.empty_like(flat, dtype=torch.int32)
            cpu_ids = torch.empty_like(flat, dtype=torch.int32)
            _route_kernel[(1,)](flat, slot_tab[layer_idx], cpu_tab[layer_idx], self.counts[layer_idx], slot_ids, cpu_ids,
                                flat.numel(), BLOCK=_ROUTE_BLOCK, num_warps=1)
            return slot_ids.view_as(ids), cpu_ids.view_as(ids)
        self.counts[layer_idx].index_add_(0, flat, self.ones.expand(flat.numel()))
        return slot_tab[layer_idx][ids], cpu_tab[layer_idx][ids]

    def push_tables(self) -> None:
        """The host tables to the GPU on the current stream."""
        self.log2slot.copy_(self.log2slot_cpu, non_blocking=True)
        self.log2cpu.copy_(self.log2cpu_cpu, non_blocking=True)
        if KV_LEND:
            self.log2slot_base.copy_(self.log2slot_base_cpu, non_blocking=True)
            self.log2cpu_base.copy_(self.log2cpu_base_cpu, non_blocking=True)

    # -- filling -------------------------------------------------------------

    def _bank(self, layer_idx: int):
        """This rank's slice of every expert of the layer (gate, up, down), pinned, physical order: the GGUF
        streamer's banks when it is loaded (one pinned copy serves both), otherwise the cache's own, built from the
        hot method's GGUF reader (KT_HOT_BANKS=own forces these; the cache does not need the streamer)."""
        if OWN_BANKS != "own":
            try:
                from sglang.srt.layers.moe import kt_stream_prefill_gguf as G
            except ImportError:
                G = None
            inst = getattr(G, "_instance", None)
            if inst is not None:
                return inst._bank(layer_idx)
        b = self._own_banks.get(layer_idx)
        if b is not None:
            return b
        t0 = time.time()
        method, _layer = self.layers[layer_idx]
        rd, E, N, K, r, world = method.reader, method.reader.E, method.N, method.K, method.rank, method.world
        from sglang.srt.layers.moe.kt_gguf_hot import BLOCK_INFO

        p2l = None
        try:
            from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata

            meta = get_global_expert_location_metadata()
            if meta is not None:
                m = meta.physical_to_logical_map_cpu[layer_idx].to(torch.int64)
                if not torch.equal(m, torch.arange(m.numel(), dtype=torch.int64)):
                    p2l = m
        except ImportError:
            pass
        banks = []
        for proj in ("gate", "up", "down"):
            data, qt = rd._raw(f"blk.{layer_idx}.ffn_{proj}_exps.weight")
            blck, ts = BLOCK_INFO[qt]
            if proj != "down":
                v = data.view(E, N * world, (K // blck) * ts)[:, N * r:N * (r + 1), :]
            else:
                v = data.view(E, K, (N * world) // blck, ts)[:, :, (N * r) // blck:(N * (r + 1)) // blck, :]
            v = v.reshape(E, -1)
            if p2l is not None:
                v = v[p2l]
            h = torch.empty((E, v.shape[1]), dtype=torch.uint8, pin_memory=True)
            h.copy_(v)
            banks.append(h)
            del data
        self._own_banks[layer_idx] = banks
        if layer_idx == min(self.layers):
            _log(f"rank {self.rank}: layer {layer_idx} own bank {sum(b.numel() for b in banks) / 1e9:.2f} GB pinned in "
                 f"{time.time() - t0:.1f} s (no streamer{', placement map' if p2l is not None else ''})")
        return banks

    def _mirror(self, layer_idx: int, param: torch.Tensor) -> Optional[torch.Tensor]:
        """The lend's pinned host copy of this parameter (same shape), or None.
        Resolved once per parameter after the lend has mirrored."""
        from sglang.srt.layers.moe import kt_lend

        key = (layer_idx, param.data_ptr())
        if key in self._mirrors:
            return self._mirrors[key]
        st = kt_lend.STATE
        if st.get("host") is None or "dev_spans" not in st:
            # not mirrored yet (seeding at load), lend off, or a mirror without a pinned host copy
            # (KT_PREFILL_LEND_MIRROR=gguf: the restore reads the files, after_restore refills the swaps)
            return None
        found = None
        for i, dspan in enumerate(st["dev_spans"]):
            lo = dspan.data_ptr()
            if lo <= param.data_ptr() < lo + dspan.numel():
                off = param.data_ptr() - lo + st["bases"][i]
                found = st["host"][off:off + param.numel() * param.element_size()].view(param.dtype).view(param.shape)
                break
        self._mirrors[key] = found
        return found

    def _down_source(self, layer_idx: int, method, down_bank: torch.Tensor) -> torch.Tensor:
        """The bank to copy a slot's down matrix from: the raw bank, or for the
        Q2_0 layers a Q8_0 repack of the whole layer built on first use
        (every expert, 0.45 GB pinned per such layer)."""
        if method.types[2] == method.type2:
            return down_bank
        b = self._q8_banks.get(layer_idx)
        if b is None:
            from sglang.srt.layers.moe.kt_gguf_hot import q2_0_to_q8_0

            t0 = time.time()
            b = q2_0_to_q8_0(down_bank).pin_memory()
            self._q8_banks[layer_idx] = b
            _log(f"rank {self.rank}: layer {layer_idx} down bank repacked Q2_0 -> Q8_0 for the slots "
                 f"({b.numel() / 1e9:.2f} GB pinned, {time.time() - t0:.1f} s)")
        return b

    def _file_rows(self, layer_idx: int, logical: int, method, layer):
        """A logical expert's rows of this rank (gate, up, down) straight from the GGUF file mapping (page cache
        or the NVMe): pageable host tensors, ~8 MB, a few ms."""
        N, w2 = method.N, layer.w2_qweight.data
        e = torch.tensor([logical], dtype=torch.int64)
        g = method.reader.slice(layer_idx, "gate", N, method.rank, e)[0].view(N, -1)
        u = method.reader.slice(layer_idx, "up", N, method.rank, e)[0].view(N, -1)
        d = method.reader.slice(layer_idx, "down", N, method.rank, e)
        if method.types[2] != method.type2:
            from sglang.srt.layers.moe.kt_gguf_hot import q2_0_to_q8_0

            d = q2_0_to_q8_0(d)
        return g, u, d[0].view(w2.shape[1], -1)

    def _pread_rows(self, layer_idx: int, logical: int, method, layer, st13, st2):
        """The expert's rows read with pread straight into the pinned staging buffers (one read a projection;
        this rank's rows are contiguous in the file when it holds every row, world 1). Returns (n13, n2) bytes,
        or None when the rows are not contiguous (TP > 1: the mmap path)."""
        if method.world != 1:
            return None
        rd = method.reader
        fds = self.__dict__.setdefault("_fds", {})
        n13 = 0
        for proj, dst in (("gate", 0), ("up", 1)):
            name = f"blk.{layer_idx}.ffn_{proj}_exps.weight"
            info = rd.loader.tensor_info[name]
            path = rd.loader.tensor_file_map[name]
            per = layer.w13_qweight.data[0, :method.N].numel()
            fd = fds.get(path)
            if fd is None:
                fd = fds[path] = os.open(path, os.O_RDONLY)
            view = st13[dst * per:(dst + 1) * per].numpy()
            got = os.preadv(fd, [memoryview(view)], int(info["offset"]) + logical * per)
            if got != per:
                raise RuntimeError(f"pread {name} expert {logical}: {got} of {per} bytes")
            n13 += per
        name = f"blk.{layer_idx}.ffn_down_exps.weight"
        info = rd.loader.tensor_info[name]
        path = rd.loader.tensor_file_map[name]
        if method.types[2] != method.type2:
            return None  # a Q2_0 down is repacked on the mmap path
        per = layer.w2_qweight.data[0].numel()
        fd = fds.get(path)
        if fd is None:
            fd = fds[path] = os.open(path, os.O_RDONLY)
        got = os.preadv(fd, [memoryview(st2[:per].numpy())], int(info["offset"]) + logical * per)
        if got != per:
            raise RuntimeError(f"pread {name} expert {logical}: {got} of {per} bytes")
        return n13, per

    def fill(self, layer_idx: int, slot: int, logical: int) -> None:
        method, layer = self.layers[layer_idx]
        N = method.N
        w13, w2 = layer.w13_qweight.data, layer.w2_qweight.data
        if SOURCE == "file":
            g, u, d = self._file_rows(layer_idx, logical, method, layer)
        else:
            gate, up, down = self._bank(layer_idx)
            g = gate[logical].view(N, -1)
            u = up[logical].view(N, -1)
            d = self._down_source(layer_idx, method, down)[logical].view(w2.shape[1], -1)
        if slot >= self.N:
            # a lent slot: KV pages, no lend mirror (lent slots are given back
            # before every streamed prefill, kt_hot_kv_lend.py)
            v13, v2 = self.lent_views[(layer_idx, slot)]
            v13[:N].copy_(g, non_blocking=True)
            v13[N:].copy_(u, non_blocking=True)
            v2.copy_(d, non_blocking=True)
            self.bytes += g.numel() + u.numel() + d.numel()
            return
        w13[slot, :N].copy_(g, non_blocking=True)
        w13[slot, N:].copy_(u, non_blocking=True)
        w2[slot].copy_(d, non_blocking=True)
        m13, m2 = self._mirror(layer_idx, layer.w13_qweight), self._mirror(layer_idx, layer.w2_qweight)
        if m13 is not None:
            m13[slot, :N].copy_(g)
            m13[slot, N:].copy_(u)
        if m2 is not None:
            m2[slot].copy_(d)
        self.bytes += g.numel() + u.numel() + d.numel()

    def _set(self, layer_idx: int, slot: int, logical: int) -> None:
        """Slot <- logical (>= 0), or EMPTY / ABSENT. Residency is carried by
        the ids alone: an expert is in log2slot or in log2cpu, never both,
        never neither (check_tables)."""
        old = self.slot2log[layer_idx][slot]
        if old >= 0:
            self.log2slot_cpu[layer_idx, old] = -1
            self.log2cpu_cpu[layer_idx, old] = old
            if KV_LEND and slot < self.N:
                self.log2slot_base_cpu[layer_idx, old] = -1
                self.log2cpu_base_cpu[layer_idx, old] = old
        self.slot2log[layer_idx][slot] = logical
        if logical < 0:
            return
        self.log2slot_cpu[layer_idx, logical] = slot
        self.log2cpu_cpu[layer_idx, logical] = -1
        if KV_LEND and slot < self.N:
            self.log2slot_base_cpu[layer_idx, logical] = slot
            self.log2cpu_base_cpu[layer_idx, logical] = -1

    def check_tables(self) -> str:
        """Every expert of every layer exactly once: on a slot that holds it
        (full and base tables, host and device) or on the CPU. Returns '' or
        the first violation."""
        pairs = [("full", self.log2slot_cpu, self.log2cpu_cpu, self.log2slot, self.log2cpu, False)]
        if KV_LEND:
            pairs.append(("base", self.log2slot_base_cpu, self.log2cpu_base_cpu, self.log2slot_base,
                          self.log2cpu_base, True))
        for name, s_cpu, c_cpu, s_dev, c_dev, base in pairs:
            if not torch.equal(s_dev.cpu(), s_cpu) or not torch.equal(c_dev.cpu(), c_cpu):
                return f"{name}: device tables differ from the host's"
            if not bool(((s_cpu >= 0) ^ (c_cpu >= 0)).all()):
                return f"{name}: an expert both or neither on a slot and on the CPU"
            if not bool(((c_cpu < 0) | (c_cpu == torch.arange(self.E, dtype=torch.int32))).all()):
                return f"{name}: CPU id is not the logical id"
            for li in range(self.L):
                for e in (s_cpu[li] >= 0).nonzero().flatten().tolist():
                    sl = int(s_cpu[li, e])
                    if self.slot2log[li][sl] != e or (base and sl >= self.N):
                        return f"{name}: layer {li} expert {e} -> slot {sl} holding {self.slot2log[li][sl]}"
                for sl, e in enumerate(self.slot2log[li]):
                    if e >= 0 and (int(self.log2slot_cpu[li, e]) != sl):
                        return f"{name}: layer {li} slot {sl} holds {e} but the table says {int(self.log2slot_cpu[li, e])}"
        return ""

    # -- lent slots (kt_hot_kv_lend.py) -------------------------------------------

    def lend_slot(self, layer_idx: int, slot: int, v13: torch.Tensor, v2: torch.Tensor) -> None:
        assert self.slot2log[layer_idx][slot] == ABSENT, (layer_idx, slot)
        self.lent_views[(layer_idx, slot)] = (v13, v2)
        self.slot2log[layer_idx][slot] = EMPTY

    def take_back_slot(self, layer_idx: int, slot: int) -> int:
        """The slot's expert goes back to the CPU (host tables; push_tables
        before the next forward). Returns the expert it held or -1."""
        old = self.slot2log[layer_idx][slot]
        assert old != ABSENT, (layer_idx, slot)
        self._set(layer_idx, slot, ABSENT)
        del self.lent_views[(layer_idx, slot)]
        return old

    def fill_empty(self, counts: torch.Tensor, seed_rank: Optional[torch.Tensor]) -> int:
        """Fill every EMPTY slot with the hottest non-residents of its layer
        (counts [L, E] on the host; ties by the seed map's order). Host tables
        only; the caller pushes them."""
        n = 0
        for li in sorted(self.layers):
            empty = [s_ for s_, e in enumerate(self.slot2log[li]) if e == EMPTY]
            if not empty:
                continue
            score = counts[li].double()
            if seed_rank is not None:
                score = score + (self.E - seed_rank[li].double()) / (self.E + 1) * 1e-3
            res = [e for e in self.slot2log[li] if e >= 0]
            if res:
                score[torch.tensor(res)] = float("-inf")
            for s_, c in zip(empty, torch.topk(score, len(empty)).indices.tolist()):
                self._set(li, s_, c)
                self.fill(li, s_, c)
                n += 1
        return n

    def seed(self, placement: Optional[str]) -> None:
        """Initial residents: the map's first N per layer (filled from the
        banks), or logical 0..N-1 (what the loader put there)."""
        t0 = time.time()
        p2l = json.load(open(placement))["physical_to_logical_map"] if placement else None
        for li in sorted(self.layers):
            for s in range(self.N):
                e = int(p2l[li][s]) if p2l else s
                self._set(li, s, e)
                if p2l and e != s:
                    self.fill(li, s, e)
        self.push_tables()
        torch.cuda.current_stream(self.device).synchronize()
        self.ready = True
        if self._async():
            self._start_worker()
        _log(f"rank {self.rank}: {len(self.layers)} layers x {self.N} slots seeded from "
             f"{placement or 'logical order'} in {time.time() - t0:.1f} s ({self.bytes / 1e9:.2f} GB copied); "
             f"interval {INTERVAL}, max {MAX_SWAPS} swaps/layer, decay {DECAY}, margin {MARGIN}")
        self.bytes = 0

    # -- policy, between steps --------------------------------------------------

    def before_forward(self, forward_batch) -> None:
        # decode steps, and under MTP the target's verify steps (the decode of a
        # speculative serve; is_decode() alone never ran the policy there)
        mode = forward_batch.forward_mode
        if not self.ready or not (mode.is_decode() or mode.is_target_verify()):
            return
        self.step += 1
        if self.step % INTERVAL:
            return
        t0 = time.time()
        # the counts come through a pinned buffer one round late, so the policy
        # never waits for the forward stream (a .cpu() here would block on the
        # step in flight)
        if self._counts_event is not None and not self._counts_event.query():
            return
        counts = self._counts_host.clone() if self._counts_event is not None else None
        self._counts_host.copy_(self.counts, non_blocking=True)
        self.counts.mul_(DECAY)
        if self._counts_event is None:
            self._counts_event = torch.cuda.Event()
        self._counts_event.record(torch.cuda.current_stream(self.device))
        if counts is None:
            return
        # all layers at once: coldest residents vs hottest non-residents. An
        # EMPTY slot scores -1 (filled by any candidate seen at all), an ABSENT
        # one +inf (never chosen); without lent slots res has neither
        res = torch.tensor(self.slot2log)  # [L, S]
        score_res = counts.gather(1, res.clamp(min=0))
        score_res = torch.where(res == EMPTY, torch.full_like(score_res, -1.0), score_res)
        score_res = torch.where(res <= ABSENT, torch.full_like(score_res, float("inf")), score_res)
        mask = torch.ones((counts.shape[0], counts.shape[1] + 1), dtype=torch.bool)
        mask.scatter_(1, torch.where(res >= 0, res, torch.full_like(res, counts.shape[1])), False)
        mask = mask[:, :-1]
        cand_score, cand_idx = torch.topk(torch.where(mask, counts, torch.full_like(counts, -1.0)), MAX_SWAPS, dim=1)
        order = torch.argsort(score_res, dim=1)[:, :MAX_SWAPS]  # coldest slots first
        rs = score_res.gather(1, order)
        do = ((cand_score > 0) & (cand_score > MARGIN * rs)).to(torch.int64).cumprod(1).bool()
        swaps = 0
        landed = self._apply_landed()
        queued = []
        for li, j in do.nonzero().tolist():
            if li not in self.layers:
                continue
            s_, c = int(order[li, j]), int(cand_idx[li, j])
            if self._async():
                # the old expert leaves now (its routes go to the CPU); the new one lands later
                self._set(li, s_, PENDING)
                queued.append((li, s_, c))
            else:
                self._set(li, s_, c)
                self.fill(li, s_, c)
            swaps += 1
        if swaps or landed:
            self.push_tables()
            if self._async():
                # a swap's copy follows this table change on the GPU (and the forward in flight before it):
                # the event is recorded before the worker can see the swaps
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream(self.device))
                self._after = ev
        for item in queued:
            self._q.put(item)
        self.rounds += 1
        self.swaps += swaps
        self.swap_s += time.time() - t0
        if self.rounds == 1 or (LOG_EVERY and self.rounds % LOG_EVERY == 0) or self.rounds % 64 == 0:
            resident = sum(float(counts[li][torch.tensor([e for e in self.slot2log[li] if e >= 0], dtype=torch.long)].sum())
                           for li in self.layers)
            total = float(counts.sum())
            _log(f"rank {self.rank}: round {self.rounds} (step {self.step}): {swaps} swaps this round, "
                 f"{self.swaps} total, {self.bytes / 1e9:.2f} GB, {self.swap_s * 1e3 / self.rounds:.1f} ms/round; "
                 f"GPU share of recent routes {resident / max(total, 1):.3f}"
                 + (f"; go {self._go.is_set()}, queued {self._q.qsize()}, landed-waiting {len(self._done)}, "
                    f"stages free {self._stages.qsize()}, worker {self._n_swapped} swaps: read "
                    f"{self._t_read * 1e3 / max(1, self._n_swapped):.0f} wait {self._t_wait * 1e3 / max(1, self._n_swapped):.0f} "
                    f"issue {self._t_issue * 1e3 / max(1, self._n_swapped):.0f} ms each" if self._async() else ""))


    def after_restore(self) -> None:
        """After the prefill lend put the weights back: a mirror that sources a hot slot from its load-time expert
        (the GGUF mirror, kt_lend KT_PREFILL_LEND_MIRROR=gguf) has undone the swaps, so every slot holding another
        expert is filled again; on the current stream, behind the restore's copies."""
        if not self.ready or SOURCE != "file":
            return
        t0, n = time.time(), 0
        for li in self.layers:
            for s_, e in enumerate(self.slot2log[li]):
                if e >= 0 and e != s_ and s_ < self.N:
                    self.fill(li, s_, e)
                    n += 1
        if self._async():
            # pending copies follow the restore and these refills
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream(self.device))
            self._after = ev
            self._go.set()
        if n and (self.rounds <= 1 or LOG_EVERY):
            _log(f"rank {self.rank}: {n} swapped slots filled again after a prefill window in {time.time() - t0:.2f} s")

    # -- asynchronous swaps -----------------------------------------------------

    def _async(self) -> bool:
        return ASYNC and SOURCE == "file"

    def _start_worker(self) -> None:
        if self._worker is not None:
            return
        n13 = max(l.w13_qweight.data[0].numel() for _, l in self.layers.values())
        n2 = max(l.w2_qweight.data[0].numel() for _, l in self.layers.values())
        for _ in range(STAGES):
            self._stages.put((torch.empty(n13, dtype=torch.uint8, pin_memory=True),
                              torch.empty(n2, dtype=torch.uint8, pin_memory=True)))
        self._side = torch.cuda.Stream(device=self.device)
        self._worker = threading.Thread(target=self._work, name="kt-hot-swap", daemon=True)
        self._worker.start()
        _log(f"rank {self.rank}: swaps on a worker thread and a side stream, {STAGES} staging buffers "
             f"({STAGES * (n13 + n2) / 1e6:.0f} MB pinned)")

    def _work(self) -> None:
        try:
            # off kt-kernel's spinning compute threads: the cores kept for the expert cache's readers
            # (serve.sh sets KT_GGUF_PREFETCH_CPUS); starved among them, a swap took ~0.7 s (10-25)
            cpus = os.environ.get("KT_HOT_SWAP_CPUS") or os.environ.get("KT_GGUF_PREFETCH_CPUS")
            if cpus:
                try:
                    os.sched_setaffinity(0, {int(c) for c in cpus.split(",") if c.strip()})
                except (OSError, ValueError) as ex:
                    _log(f"rank {self.rank}: swap worker affinity {cpus!r} not set: {ex}")
            torch.cuda.set_device(self.device)
            self._work_loop()
        except Exception:
            import traceback

            _log(f"rank {self.rank}: swap worker died:\n{traceback.format_exc()}")

    def _work_loop(self) -> None:
        first = True
        while True:
            item = self._q.get()
            if item is None:
                return
            li, s_, e = item
            t0 = time.time()
            method, layer = self.layers[li]
            # stages come back as their copies land (recycling them only at the policy's rounds held the
            # worker to 8 swaps a round: ~0.6 s a swap, 10-25)
            while self._inflight and self._inflight[0][0].query():
                self._stages.put(self._inflight.popleft()[1])
            if self._stages.empty() and self._inflight:
                ev0, stage0 = self._inflight.popleft()
                ev0.synchronize()
                self._stages.put(stage0)
            st13, st2 = self._stages.get()
            t1 = time.time()
            try:
                got = self._pread_rows(li, e, method, layer, st13, st2)
                if got is None:
                    g, u, d = self._file_rows(li, e, method, layer)
                    n13, n2 = g.numel() + u.numel(), d.numel()
                    st13[:g.numel()].copy_(g.reshape(-1))
                    st13[g.numel():n13].copy_(u.reshape(-1))
                    st2[:n2].copy_(d.reshape(-1))
                else:
                    n13, n2 = got
            except Exception as ex:  # a read that failed leaves the slot pending: logged, never computed
                _log(f"rank {self.rank}: swap read failed for layer {li} expert {e}: {ex}")
                self._stages.put((st13, st2))
                continue
            t2 = time.time()
            self._t_read += t2 - t1
            self._go.wait()
            t3 = time.time()
            with self._lock:
                if not self._go.is_set():
                    # a window began between the wait and the lock: back to the queue's head
                    self._stages.put((st13, st2))
                    self._q.put(item)
                    continue
                side = self._side
                if self._after is not None:
                    side.wait_event(self._after)
                with torch.cuda.stream(side):
                    w13, w2 = layer.w13_qweight.data, layer.w2_qweight.data
                    w13[s_].view(-1)[:n13].copy_(st13[:n13], non_blocking=True)
                    w2[s_].view(-1)[:n2].copy_(st2[:n2], non_blocking=True)
                    ev = torch.cuda.Event()
                    ev.record(side)
                self._done.append((li, s_, e, ev))
                self._inflight.append((ev, (st13, st2)))
                self.bytes += n13 + n2
            self._t_wait += t3 - t2
            self._t_issue += time.time() - t3
            self._n_swapped += 1
            if first:
                first = False
                _log(f"rank {self.rank}: first swap (layer {li} slot {s_} <- expert {e}): stage {(t1 - t0) * 1e3:.0f} ms, "
                     f"read {(t2 - t1) * 1e3:.0f} ms, waited {(t3 - t2) * 1e3:.0f} ms, copy issued")

    def _apply_landed(self) -> int:
        """Copies that have landed enter the tables (host side; the caller pushes). Returns how many."""
        n = 0
        while self._done:
            li, s_, e, ev = self._done[0]
            if not ev.query():
                break
            self._done.popleft()
            if self.slot2log[li][s_] == PENDING:
                self._set(li, s_, e)
                n += 1
            else:
                _log(f"rank {self.rank}: layer {li} slot {s_} was {self.slot2log[li][s_]} when expert {e} landed")
        return n

    def _pause(self) -> None:
        """No copy may land while the slots are lent to a prefill window (kt_lend calls window_start before it
        unmaps the weights; after_restore releases): issued ones finish first. Pausing on every extend batch
        instead left the cache paused for good after a short extend that opened no window (10-25)."""
        if not self._async():
            return
        with self._lock:
            self._go.clear()
            if self._side is not None:
                self._side.synchronize()


_cache: Optional[HotCache] = None


def after_restore() -> None:
    if _cache is not None:
        _cache.after_restore()


def window_start() -> None:
    """kt_lend, before the weights (the slots among them) are unmapped for a prefill window."""
    if _cache is not None and _cache.ready:
        _cache._pause()


def register(method, layer) -> Optional[HotCache]:
    """Called by the KT wrapper on its first decode apply of each layer; the
    cache is built on the first call and seeded once every layer is known."""
    global _cache
    if not ENABLED:
        return None
    kt = method.kt_config
    if _cache is None:
        if kt.num_gpu_experts <= 0:
            raise RuntimeError("KT_HOT_DYNAMIC=1 needs --kt-num-gpu-experts > 0")
        _cache = HotCache(kt.num_layers, method.global_num_experts, kt.num_gpu_experts,
                          layer.w13_qweight.device, method.tp_rank)
    c = _cache
    if kt.layer_idx not in c.layers:
        c.layers[kt.layer_idx] = (method.gpu_method, layer)
        if len(c.layers) == kt.num_layers:
            c.seed(SEED)
    return c if c.ready else None


def before_forward(forward_batch) -> None:
    if _cache is not None:
        _cache.before_forward(forward_batch)
