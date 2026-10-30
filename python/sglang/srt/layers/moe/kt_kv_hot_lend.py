"""More hot experts in the KV pool's free pages, outside prefill, for any GPU MoE
method of the KT wrapper (KT_KV_HOT_LEND=1, KT_KV_HOT_EXTRA=<X>; with
KT_KV_VMM=1).

The KT wrapper runs physical experts 0..N-1 on the GPU (N =
--kt-num-gpu-experts) and the rest on the CPU. With a hot-expert placement the
physical order is the hotness order, so the next X experts, N..N+X-1, are the
ones most worth adding. This module:

- at load, makes a shadow of each MoE layer outside the model, with a copy of
  the GPU method's weights for X experts on the CPU (make_shadow); the weight
  loader puts experts N..N+X-1 there (shadow_param); after the layer's own
  processing the shadow's are processed on the GPU into the method's format,
  copied to pinned host memory, and its per-expert tensors (dim 0 of X) become
  slots on kv_vmm shadow ranges, every byte of which reads as zero until mapped
  (split). The layer itself is exactly as without this; the CPU loads all of
  its experts as before, those X included;
- outside prefill, takes the KV pool's free pages beyond KT_KV_HOT_RESERVE
  tokens out of the free list and unmaps their whole chunks
  (kt_kv_prefill_lend.decode_lend), maps that much memory under the slots,
  layer by layer one slot at a time, copies the experts in from the host, and
  sets each layer's lent count k on the device;
- in the wrapper's apply, a route to a physical id in [N, N + k) goes to the
  CPU with weight 0 at id 0 (a GPU id, which the CPU skips) and to the shadow's
  GPU method as slot id - N; every other route is as before;
- when the pool needs the pages (an allocation the free list cannot serve, a
  prefill window, a cache flush), sets every k to 0, syncs, maps the zero chunk
  back under the slots, then the pages' chunks are mapped back, zeroed
  (kt_kv_prefill_lend.restore, which calls release first).

The pool never changes size and the lent pages count as available, as with the
prefill loan. k lives on the device and the slots never move, so the captured
decode graphs stay valid; a slot that holds nothing reads zeros. Every rank
lends at the same forward (the allocator's free page count is the same on each),
and every rank holds the same slots: the CPU runs on rank 0 with its routing, so
a slot another rank lacked would leave its share of that route uncomputed.

KT_KV_HOT_DYNAMIC_SLOTS=1: the slots follow what is routed, so whoever serves
whatever gets their own hot experts without recording a placement (a writer
placement's 21 a layer catch 3-6% of the routes of code, JSON or math, 10-21):
- a device table, physical id -> slot or -1, replaces the [N, N + k) range in
  route, and route counts every routed physical id on the device;
- every KT_KV_HOT_DYN_EVERY decode forwards (8), outside the graph, the counts
  come to the host as decayed scores (half-life KT_KV_HOT_DYN_HALF_LIFE forwards,
  256), and each layer swaps up to KT_KV_HOT_DYN_SWAPS a forward (2) of its held
  experts for better-scored ones; a swapped-out expert leaves the table at once
  (its routes go to the CPU, which holds every non-resident expert), the new one
  is copied into the slot on a side stream from kt-kernel's expert arenas (the
  zero-copy mapping the streamed prefill registered: no host copy) and enters
  the table on the next round, behind the copy;
- the decisions use only the counts, the same on every rank, and the slot counts
  are the smallest over the ranks: the CPU (rank 0's routing) skips exactly the
  routes every rank's slots take;
- a new lend loads what was held before the window, from the arenas.
"""
import copy
import os
import time
from typing import List, Optional

import numpy as np
import torch

from sglang.srt.layers.moe import kt_kv_vmm

ENABLED = os.environ.get("KT_KV_HOT_LEND", "0") == "1" and kt_kv_vmm.ENABLED
# KT_KV_HOT_EXTRA=<X> slots a layer, or auto: with the dynamic slots every expert past the
# resident ones (virtual address space only: how many hold memory follows the free memory
# at each lend), else 16
_EXTRA_RAW = (os.environ.get("KT_KV_HOT_EXTRA", "0") or "0").strip().lower()
EXTRA_AUTO = _EXTRA_RAW == "auto"
EXTRA = 0 if EXTRA_AUTO else int(_EXTRA_RAW)
RESERVE = int(os.environ.get("KT_KV_HOT_RESERVE", "4096"))  # tokens kept free outside prefill
MIN_MB = int(os.environ.get("KT_KV_HOT_MIN_MB", "256"))  # smallest lend worth a sync and a copy
RETRY = int(os.environ.get("KT_KV_HOT_RETRY", "256"))  # forwards between tries when nothing was lent
MARGIN_MB = int(os.environ.get("KT_KV_HOT_MARGIN_MB", "64"))  # of the unmapped bytes left unused
# KT_KV_HOT_SPARE_FLOOR_MB=<n> (sglang-kt-kv-hot-spare.py): the slots also take what the
# driver has free in decode beyond n MiB (the prefill's room, idle between windows)
SPARE_FLOOR_MB = int(os.environ.get("KT_KV_HOT_SPARE_FLOOR_MB", "-1") or -1)
# KT_KV_HOT_SLOTS_MAX=<n>: at most n slots a rank (0: no cap). Measurements set it below
# what every lend gets, so each lend, launch and rank holds the same slots and the same
# routes run on the GPU (the count otherwise follows the free memory)
SLOTS_MAX = int(os.environ.get("KT_KV_HOT_SLOTS_MAX", "0") or 0)
# KT_KV_HOT_CHECK=<n>: every n-th eager MoE call, the layer's output is compared
# with the same call computed without the lend (the resident experts again, the
# CPU with every route), summed over the TP ranks (check)
CHECK = int(os.environ.get("KT_KV_HOT_CHECK", "0") or 0)
_CHECK_DUMP = os.environ.get("KT_KV_HOT_CHECK_DUMP", "")  # a directory: see check
_CHECK_DUMP_FIRST = int(os.environ.get("KT_KV_HOT_CHECK_DUMP_FIRST", "2"))  # checks dumped whatever they say
_MASK_ZERO = os.environ.get("KT_GPU_MASK_ZERO", "0") == "1"
_SLOT_REL = float(os.environ.get("KT_KV_HOT_CHECK_SLOT_REL", "0.05") or 0.05)
_DUMP_REL = float(os.environ.get("KT_KV_HOT_CHECK_DUMP_REL", "0.2") or 0.2)  # a call dumped past this
_CHECK_REPEAT = os.environ.get("KT_KV_HOT_CHECK_REPEAT", "0") == "1"
DYNAMIC = os.environ.get("KT_KV_HOT_DYNAMIC_SLOTS", "0") == "1"
DYN_EVERY = max(1, int(os.environ.get("KT_KV_HOT_DYN_EVERY", "8") or 8))
DYN_SWAPS = int(os.environ.get("KT_KV_HOT_DYN_SWAPS", "2") or 2)
DYN_HALF_LIFE = float(os.environ.get("KT_KV_HOT_DYN_HALF_LIFE", "256") or 256)
DYN_HYST = float(os.environ.get("KT_KV_HOT_DYN_HYST", "1.0") or 1.0)  # replace when new > victim x this
DYN_LOG = int(os.environ.get("KT_KV_HOT_DYN_LOG", "512") or 512)  # rounds between log lines
# KT_KV_HOT_DYNAMIC_RESIDENT=1 (with the dynamic slots): the N resident experts swap too.
# kt-kernel then loads every expert (no GPU mask), so an expert that leaves a resident row
# runs on the CPU; route sends the CPU -1 for every route the GPU takes. In a prefill
# window the rows hold experts 0..N-1 again, as the streamer expects.
DYN_RES = DYNAMIC and os.environ.get("KT_KV_HOT_DYNAMIC_RESIDENT", "0") == "1"

# KT_KV_HOT_ROUTE_TORCH=1: the dynamic route as separate torch ops instead of one kernel
ROUTE_TORCH = os.environ.get("KT_KV_HOT_ROUTE_TORCH", "0") == "1"

try:
    import triton
    import triton.language as tl
except ImportError:  # the torch ops serve
    triton = None

if triton is not None:

    @triton.jit
    def _route_kernel(ids_ptr, w_ptr, tbl_ptr, rtbl_ptr, cnt_ptr, cid_ptr, cw_ptr, sid_ptr, sw_ptr, row_ptr, n,
                      HAS_RES: tl.constexpr, MASK_ZERO: tl.constexpr, CPU_SKIP: tl.constexpr,
                      BLOCK: tl.constexpr):
        """One layer's routes in one launch: count each routed expert, look it up in the
        slot table (and the resident rows' table), and write the CPU's ids and weights
        (CPU_SKIP at weight 0 for what the GPU takes), the slots' and the rows' ids."""
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        e = tl.load(ids_ptr + offs, mask=m, other=-1).to(tl.int64)
        w = tl.load(w_ptr + offs, mask=m, other=0.0)
        valid = m & (e >= 0)
        ec = tl.where(valid, e, 0)
        tl.atomic_add(cnt_ptr + ec, tl.where(valid, 1.0, 0.0), mask=valid)
        s = tl.load(tbl_ptr + ec, mask=valid, other=-1)
        if HAS_RES:
            r = tl.load(rtbl_ptr + ec, mask=valid, other=-1)
            res = valid & (r >= 0)
            lent = valid & (s >= 0) & (r < 0)
            tl.store(row_ptr + offs, tl.where(res, r, -1).to(row_ptr.dtype.element_ty), mask=m)
        else:
            res = valid & (s < -1)  # never
            lent = valid & (s >= 0)
        gpu = lent | res
        tl.store(cid_ptr + offs, tl.where(gpu, CPU_SKIP, e).to(cid_ptr.dtype.element_ty), mask=m)
        tl.store(cw_ptr + offs, tl.where(gpu, 0.0, w).to(cw_ptr.dtype.element_ty), mask=m)
        if MASK_ZERO:
            tl.store(sid_ptr + offs, tl.where(lent, s, 0).to(sid_ptr.dtype.element_ty), mask=m)
            tl.store(sw_ptr + offs, tl.where(lent, w, 0.0).to(sw_ptr.dtype.element_ty), mask=m)
        else:
            tl.store(sid_ptr + offs, tl.where(lent, s, -1).to(sid_ptr.dtype.element_ty), mask=m)
            tl.store(sw_ptr + offs, w.to(sw_ptr.dtype.element_ty), mask=m)
else:
    _route_kernel = None


def _route_fused(L, ids, w):
    """The dynamic route in one kernel launch (route lists what each output is)."""
    ids, w = ids.contiguous(), w.contiguous()
    cid, cw, sw = torch.empty_like(ids), torch.empty_like(w), torch.empty_like(w)
    sid = torch.empty(ids.shape, dtype=torch.int64, device=ids.device) if _MASK_ZERO else torch.empty_like(ids)
    rows = torch.empty_like(ids) if DYN_RES else None
    n = ids.numel()
    if n:
        tbl, cnt = STATE["tbl"][L.idx], STATE["cnt"][L.idx]
        _route_kernel[(triton.cdiv(n, 128),)](
            ids, w, tbl, STATE["rtbl"][L.idx] if DYN_RES else tbl, cnt, cid, cw, sid, sw,
            rows if DYN_RES else cid, n, HAS_RES=DYN_RES, MASK_ZERO=_MASK_ZERO,
            CPU_SKIP=-1 if DYN_RES else 0, BLOCK=128)
    return cid, cw, sid, sw, rows


def _on() -> bool:
    return ENABLED and (EXTRA > 0 or EXTRA_AUTO)


def _banks() -> int:
    """Experts the shadow loads from the checkpoint and keeps in pinned memory: with the
    dynamic slots two (the self test's reference; the slots load from kt-kernel's arenas),
    else one a slot."""
    x = EXTRA if EXTRA > 0 else 16
    return min(2, x) if DYNAMIC else x


def _cap(wrapper) -> int:
    """Slots a layer can have (KT_KV_HOT_EXTRA)."""
    if EXTRA_AUTO:
        return int(wrapper.global_num_experts) - int(wrapper.num_gpu_experts) if DYNAMIC else 16
    return EXTRA


LAYERS: List["_Layer"] = []
STATE = {"k": None, "want": True, "since": 0, "lent_slots": 0, "lends": 0, "releases": 0, "hooked": False,
         "calls": 0, "checks": 0}


def _log(msg: str) -> None:
    print(f"[kt-kv-hot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_count(n: int) -> int:
    """GPU experts a loader that skips CPU experts' tensors must still read: the
    resident N and, with the hot lend, the X the shadows hold."""
    return n + _banks() if _on() and n > 0 else n


def cpu_holds_all(wrapper) -> bool:
    """Whether kt-kernel is to load every expert, the resident ones too (no GPU mask):
    the resident rows swap (KT_KV_HOT_DYNAMIC_RESIDENT), so route decides alone what
    the CPU computes. Only where make_shadow makes a shadow, so route runs."""
    return (DYN_RES and _on() and int(wrapper.num_gpu_experts) > 0
            and os.environ.get("KT_HOT_DYNAMIC", "0") != "1")


def make_shadow(wrapper, layer, hidden_size: int, intermediate_size_per_partition: int,
                params_dtype, extra_weight_attrs) -> None:
    """The wrapper's create_weights, after the GPU method made the layer's N
    experts: a shadow of the layer, outside the model (so no span of the prefill
    lend holds it), with the method's weights for X experts on the CPU; the
    weight loader fills them (shadow_param), split processes them."""
    if not _on() or int(wrapper.num_gpu_experts) <= 0:
        return
    if os.environ.get("KT_HOT_DYNAMIC", "0") == "1":
        # the GGUF stack's dynamic cache routes the CPU and the GPU itself, and
        # lends the KV pool's pages its own way (KT_HOT_KV_LEND)
        raise RuntimeError("KT_KV_HOT_LEND=1 with KT_HOT_DYNAMIC=1: use KT_HOT_KV_LEND for the dynamic cache")
    shadow = copy.copy(layer)
    shadow.__dict__["_parameters"] = {}
    shadow.__dict__["_buffers"] = {}
    shadow.__dict__["_modules"] = {}
    method = copy.copy(wrapper.gpu_method)
    with torch.device("cpu"):
        method.create_weights(layer=shadow, num_experts=_banks(), hidden_size=hidden_size,
                              intermediate_size_per_partition=intermediate_size_per_partition,
                              params_dtype=params_dtype, **extra_weight_attrs)
    for p in shadow._parameters.values():
        if isinstance(p, torch.Tensor) and p.is_cuda:
            p.data = p.data.cpu()
    wrapper.__dict__["_kt_kv_hot_shadow"] = (shadow, method, set(vars(shadow)))
    # a loader that skips the checkpoint tensors of experts with no GPU slot
    # (DeepSeek V4's KT_LOAD_SKIP_CPU_EXPERTS) must still read these X
    wrapper.__dict__["load_gpu_experts"] = int(wrapper.num_gpu_experts) + _banks()


def shadow_param(wrapper, layer, param, expert_id: int):
    """The layer's weight loader, for a physical expert the GPU does not hold:
    (the shadow's tensor of the same name, its slot) when it is one of the X,
    else None."""
    sh = wrapper.__dict__.get("_kt_kv_hot_shadow")
    if sh is None:
        return None
    n = int(wrapper.num_gpu_experts)
    if not n <= expert_id < n + _banks():
        return None
    for name, p in layer._parameters.items():
        if p is param:
            sp = sh[0]._parameters.get(name)
            if sp is None:
                return None
            wrapper.__dict__["_kt_kv_hot_loads"] = wrapper.__dict__.get("_kt_kv_hot_loads", 0) + 1
            return sp, expert_id - n
    return None


def _per_expert(module, rows: int):
    """(kind, name, tensor) of a module's own CUDA tensors with rows rows: its
    parameters, buffers and plain tensor attributes (some methods keep the
    repacked weights as attributes)."""
    out, seen = [], set()
    for kind, items in (("param", module._parameters.items()), ("buffer", module._buffers.items()),
                        ("attr", [(k, v) for k, v in vars(module).items() if not k.startswith("_")])):
        for name, t in items:
            if not isinstance(t, torch.Tensor) or not t.is_cuda or t.dim() < 1 or t.shape[0] != rows:
                continue
            if 0 in t.stride():  # one value seen at a per-expert shape: nothing per expert to move
                continue
            if id(t) in seen:
                continue
            seen.add(id(t))
            out.append((kind, name, t))
    return out


class _Layer:
    def __init__(self, wrapper, idx: int):
        self.wrapper, self.idx = wrapper, idx
        # x slots, the first b of which have a bank (the checkpoint's processed experts)
        self.n, self.x, self.b = int(wrapper.num_gpu_experts), max(_cap(wrapper), _banks()), _banks()
        self.k = 0
        self.names, self.slots, self.banks, self.sz = [], {}, {}, {}
        self.shadow = self.method = None
        self.layer_idx = int(wrapper.kt_config.layer_idx)
        self.e = int(wrapper.global_num_experts)
        # KT_KV_HOT_DYNAMIC_SLOTS: the physical expert each slot is to hold (-1: none);
        # the placement's N..N+X-1 to start with, as the banks hold
        self.held = [self.n + j for j in range(self.x)]
        # KT_KV_HOT_DYNAMIC_RESIDENT: the expert each resident row is to hold, the one it
        # holds now, and the layer's per-expert tensors by role
        self.res_held = list(range(self.n))
        self.res_now = list(range(self.n))
        self.res_t = None


def split(wrapper, layer) -> None:
    """The wrapper's process_weights_after_loading, after the layer's own: the
    shadow's X experts are processed on the GPU by its copy of the method, moved
    to pinned host memory, and its per-expert tensors become slots."""
    sh = wrapper.__dict__.pop("_kt_kv_hot_shadow", None)
    if sh is None:
        return
    shadow, method, made = sh
    x = _banks()
    loads = wrapper.__dict__.pop("_kt_kv_hot_loads", 0)
    if loads == 0:
        # a loader that never hands these experts to the layer's weight loader
        # (one that skips the CPU experts' tensors) leaves the shadow empty: its
        # slots would compute nothing for the routes they take
        raise RuntimeError(f"KT_KV_HOT_LEND: no checkpoint tensor of experts {wrapper.num_gpu_experts}.."
                           f"{wrapper.num_gpu_experts + x - 1} reached layer {wrapper.kt_config.layer_idx}'s shadow")
    dev = next(p.device for p in layer.parameters() if p.is_cuda)
    torch.cuda.synchronize(dev)
    before = torch.cuda.memory_allocated(dev)
    # what the layer got after create_weights (the rest of its __init__): the
    # shadow takes its plain attributes, never its tensors
    for k, v in vars(layer).items():
        if k not in made and k not in shadow.__dict__ and not isinstance(v, torch.Tensor):
            shadow.__dict__[k] = v
    for p in shadow._parameters.values():
        if isinstance(p, torch.Tensor):
            p.data = p.data.to(dev)
    for k, b in list(shadow._buffers.items()):
        if isinstance(b, torch.Tensor):
            shadow._buffers[k] = b.to(dev)
    # the shadow's runner first: a method may read it while processing (Marlin
    # MXFP4 reads runner.config.gemm1_alpha)
    cfg = copy.copy(getattr(wrapper, "moe_runner_config", None))
    if cfg is not None and hasattr(method, "create_moe_runner"):
        cfg.num_local_experts = max(_cap(wrapper), x)
        method.create_moe_runner(shadow, cfg)
    # the method's global-experts parameters (ModelOpt NVFP4's input scales) load no row
    # into the shadow and kept their 1.0 fill, which the method's max took: the layer's
    # loaded rows instead (patches/sglang-kt-global-expert-params.py)
    try:
        from sglang.srt.layers.moe.kt_ep_wrapper import kt_fill_global_expert_params
    except ImportError:  # a tree without that patch
        kt_fill_global_expert_params = None
    if kt_fill_global_expert_params is not None:
        kt_fill_global_expert_params(shadow, src=layer)
    with torch.cuda.device(dev):
        if hasattr(method, "process_weights_after_loading"):
            method.process_weights_after_loading(shadow)
    L = _Layer(wrapper, len(LAYERS))
    found = _per_expert(shadow, x)
    # only the method's per-expert tensors (the layer has them with N rows): a bank of two
    # rows could match some other tensor's first dimension
    res_names = {nm for _, nm, _ in _per_expert(layer, L.n)}
    if res_names:
        found = [f for f in found if f[1] in res_names]
    if not found:
        raise RuntimeError(f"KT_KV_HOT_LEND: the shadow of layer {wrapper.kt_config.layer_idx} has no tensor "
                           f"of {x} experts")
    for kind, name, t in found:
        bank = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        bank.copy_(t)
        slot = kt_kv_vmm.shadow_tensor((L.x,) + tuple(t.shape[1:]), t.dtype, dev.index)
        new = torch.nn.Parameter(slot, requires_grad=False) if kind == "param" else slot
        # every name the tensor goes by: a method may register its processed form
        # under a second name as an alias of the same Parameter (ModelOpt NVFP4's
        # w13_blockscale_swizzled is w13_weight_scale), and the kernel reads that one
        for store in (shadow._parameters, shadow._buffers, shadow.__dict__):
            for k in [k for k, v in store.items() if v is t]:
                store[k] = new
        L.names.append((kind, name))
        L.slots[name], L.banks[name] = slot, bank
        L.sz[name] = t[0].nbytes
    del found, t
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    # what the shadow keeps on the GPU besides its slots (a method's per-layer
    # tensors that are not per-expert, a tensor still held under another name)
    left = torch.cuda.memory_allocated(dev) - before
    STATE["left"] = STATE.get("left", 0) + left
    if left > 1 << 20 and STATE.get("left_logged", 0) < 3:
        STATE["left_logged"] = STATE.get("left_logged", 0) + 1
        held = [(k, tuple(v.shape), v.nbytes) for store in (shadow._parameters, shadow._buffers, shadow.__dict__)
                for k, v in store.items() if isinstance(v, torch.Tensor) and v.is_cuda
                and kt_kv_vmm.chunk_of(v) == 0 and v.nbytes >= 1 << 16]
        _log(f"layer {L.idx}: the shadow keeps {left / 2**20:.1f} MiB on the GPU "
             f"outside its slots: {held}")
    L.shadow, L.method = shadow, method
    LAYERS.append(L)
    # the lent counts, made before any graph is captured (route reads them)
    STATE["k"] = torch.zeros(len(LAYERS), dtype=torch.int64, device=dev)
    if DYNAMIC:
        # the slot table and the route counts, also made before any capture
        STATE["tbl"] = torch.full((len(LAYERS), L.e), -1, dtype=torch.int64, device=dev)
        STATE["cnt"] = torch.zeros((len(LAYERS), L.e), dtype=torch.float32, device=dev)
    if DYN_RES:
        # the resident rows by role (None: not the packed layout the arenas give, so they
        # stay put) and their table, physical id -> row, the identity to start with
        rows = {}
        for _, name, t in _per_expert(layer, L.n):
            key = ("w13" if name.startswith("w13") else "w2" if name.startswith("w2") else None, "scale" in name)
            if key[0] is None or key in rows or t.element_size() != 1:
                rows = None
                break
            rows[key] = t
        L.res_t = rows if rows and len(rows) == 4 else None
        rtbl = torch.full((len(LAYERS), L.e), -1, dtype=torch.int64, device=dev)
        if "rtbl" in STATE:
            rtbl[: STATE["rtbl"].shape[0]] = STATE["rtbl"]
        rtbl[L.idx, : L.n] = torch.arange(L.n, device=dev)
        STATE["rtbl"] = rtbl
    wrapper.__dict__["_kt_kv_hot"] = L
    if L.idx == 0:
        _log(f"N {L.n} + X {L.x} slots ({L.b} with a bank): per-expert tensors "
             f"{[(nm, tuple(L.slots[nm].shape), L.sz[nm]) for _, nm in L.names]}, "
             f"{sum(L.sz.values()) * L.b / 2**20:.1f} MiB a layer in pinned memory, up to "
             f"{sum(L.sz.values()) * L.x / 2**20:.1f} MiB in slots")


def _k_dev(dev):
    return STATE["k"]


def _in_window() -> bool:
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return kt_lend.ENABLED and kt_lend.phase() == "prefill"


def route(wrapper, dispatch_output):
    """The wrapper's apply, after the logical -> physical remap, before the CPU
    submit: (dispatch for the CPU, dispatch for the slots or None, the check's
    copy, the resident rows' ids or None: then the wrapper's own mask) or None."""
    L = wrapper.__dict__.get("_kt_kv_hot")
    if L is None:
        return None
    topk = dispatch_output.topk_output
    ids, w = topk.topk_ids, topk.topk_weights
    if _in_window():
        # a streamed prefill: nothing is lent, the streamer runs them, and the resident
        # rows hold 0..N-1 (res_guard); a CPU that holds every expert must skip those
        if not DYN_RES:
            return None
        res = (ids >= 0) & (ids < L.n)
        return dispatch_output._replace(topk_output=topk._replace(
            topk_ids=torch.where(res, torch.full_like(ids, -1), ids),
            topk_weights=torch.where(res, torch.zeros_like(w), w))), None, None, None
    if DYNAMIC and _route_kernel is not None and not ROUTE_TORCH and not CHECK:
        # ten-odd small kernels a layer cost 4% of decode with the resident rows' table
        # (10-21): one launch instead
        cid, cw, sid, sw, rows = _route_fused(L, ids, w)
        return (dispatch_output._replace(topk_output=topk._replace(topk_ids=cid, topk_weights=cw)),
                dispatch_output._replace(topk_output=topk._replace(topk_ids=sid, topk_weights=sw)),
                None, rows)
    rows = None
    if DYNAMIC:
        flat = ids.reshape(-1).long()
        STATE["cnt"][L.idx].index_add_(0, flat.clamp(min=0), (flat >= 0).float())
        ic = ids.long().clamp(min=0)
        slot = STATE["tbl"][L.idx][ic].to(ids.dtype)
        lent = (slot >= 0) & (ids >= 0)
        if DYN_RES:
            row = STATE["rtbl"][L.idx][ic].to(ids.dtype)
            res = (row >= 0) & (ids >= 0)
            lent = lent & ~res
            rows = torch.where(res, row, torch.full_like(ids, -1))
    else:
        k = _k_dev(ids.device)[L.idx]
        lent = (ids >= L.n) & (ids < L.n + k)
        slot = ids - L.n
    if rows is not None:
        # kt-kernel holds every expert: -1 (skipped) for each route the GPU takes
        gpu = lent | (rows >= 0)
        cpu = dispatch_output._replace(topk_output=topk._replace(
            topk_ids=torch.where(gpu, torch.full_like(ids, -1), ids),
            topk_weights=torch.where(gpu, torch.zeros_like(w), w)))
    else:
        cpu = dispatch_output._replace(topk_output=topk._replace(
            topk_ids=torch.where(lent, torch.zeros_like(ids), ids),
            topk_weights=torch.where(lent, torch.zeros_like(w), w)))
    if _MASK_ZERO:
        # a runner that cannot take -1 (Marlin): slot 0 at weight 0, which reads
        # zeros when nothing is lent
        sid = torch.where(lent, slot, torch.zeros_like(ids)).to(torch.int64)
        sw = torch.where(lent, w, torch.zeros_like(w))
    else:
        sid = torch.where(lent, slot, torch.full_like(ids, -1))
        sw = w
    slots = dispatch_output._replace(topk_output=topk._replace(topk_ids=sid, topk_weights=sw))
    # (the check's copy; with the resident rows' table it checks the whole layer: lent None)
    saved = (ids.clone(), w.clone(), lent if rows is None else None) if CHECK else None
    return cpu, slots, saved, rows


def check(wrapper, layer, dispatch_output, hot, output) -> None:
    """KT_KV_HOT_CHECK: the slots' share of the output against the same routes on
    the CPU (the CPU with every route minus the CPU without the lent ones), and the
    whole output against the call without the lend; summed over the TP ranks
    (every rank joins: the call count is the same on each)."""
    if not CHECK or hot is None or hot[2] is None or torch.cuda.is_current_stream_capturing():
        return
    STATE["calls"] += 1
    if STATE["calls"] % CHECK:
        return
    from sglang.srt.distributed import tensor_model_parallel_all_reduce

    ids, w, lent = hot[2]
    L = wrapper.__dict__["_kt_kv_hot"]
    if lent is None:
        # KT_KV_HOT_DYNAMIC_RESIDENT: kt-kernel holds every expert, so the reference is the
        # CPU with every route; the layer's output, summed over the ranks, against it
        from sglang.srt.distributed import tensor_model_parallel_all_reduce as ar

        x = dispatch_output.hidden_states
        topk = dispatch_output.topk_output
        torch.cuda.synchronize(x.device)
        ref = torch.zeros_like(output, dtype=torch.float32)
        if wrapper.tp_rank == 0 and wrapper.wrapper is not None:
            wrapper.submit(layer, dispatch_output._replace(topk_output=topk._replace(topk_ids=ids, topk_weights=w)))
            ref = wrapper.sync(x).float().clone()
            torch.cuda.synchronize(x.device)
        ref, out = ar(ref.contiguous()), ar(output.float().contiguous())
        rel = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
        on = int(((hot[3] >= 0) | (hot[1].topk_output.topk_ids >= 0)).sum()) if hot[1] is not None else 0
        STATE["checks"] += 1
        if wrapper.tp_rank == 0:
            _log(f"check {STATE['checks']}: layer {L.idx}, {x.shape[0]} rows, {on} of {ids.numel()} routes on the "
                 f"GPU; |out - all on the CPU| / |all on the CPU| = {rel:.3e}")
        return
    x = dispatch_output.hidden_states
    topk = dispatch_output.topk_output
    cpu_id = ids >= L.n
    if _MASK_ZERO:
        rid = torch.where(cpu_id, torch.zeros_like(ids), ids).to(torch.int64)
        rw = torch.where(cpu_id, torch.zeros_like(w), w)
    else:
        rid, rw = torch.where(cpu_id, torch.full_like(ids, -1), ids), w
    # every call finished and copied out before the next one starts: the CPU's result
    # comes through a host buffer the next submit writes again (10-18: rows that came
    # back wrong on whichever side was read second)
    torch.cuda.synchronize(x.device)
    res = wrapper.gpu_method.apply(layer, dispatch_output._replace(
        topk_output=topk._replace(topk_ids=rid, topk_weights=rw))).hidden_states.float().clone()
    slots = L.method.apply(L.shadow, hot[1]).hidden_states.float().clone()
    torch.cuda.synchronize(x.device)
    cpu_full = cpu_rest = torch.zeros_like(res)
    if wrapper.tp_rank == 0 and wrapper.wrapper is not None:
        wrapper.submit(layer, dispatch_output._replace(topk_output=topk._replace(topk_ids=ids, topk_weights=w)))
        cpu_full = wrapper.sync(x).float().clone()
        torch.cuda.synchronize(x.device)
        wrapper.submit(layer, hot[0])
        cpu_rest = wrapper.sync(x).float().clone()
        torch.cuda.synchronize(x.device)
        if _CHECK_REPEAT:
            # KT_KV_HOT_CHECK_REPEAT=1: both CPU calls once more, in the same order, so a
            # dump shows whether a call's result depends on what ran before it
            wrapper.submit(layer, dispatch_output._replace(topk_output=topk._replace(topk_ids=ids, topk_weights=w)))
            STATE["cpu_full2"] = wrapper.sync(x).float().clone()
            torch.cuda.synchronize(x.device)
            wrapper.submit(layer, hot[0])
            STATE["cpu_rest2"] = wrapper.sync(x).float().clone()
            torch.cuda.synchronize(x.device)
    ar = tensor_model_parallel_all_reduce
    slots, lent_cpu = ar(slots.contiguous()), ar((cpu_full - cpu_rest).contiguous())
    ref = ar((res + cpu_full).contiguous())
    diff = ar((output.float() - (res + cpu_full)).contiguous())
    n_lent = int(lent.sum())
    nr = ref.norm().clamp(min=1e-30)
    share = float(lent_cpu.norm() / nr)
    rel_slots = float((slots - lent_cpu).norm() / lent_cpu.norm().clamp(min=1e-30))
    rel = float(diff.norm() / nr)
    # the CPU returns each row's routes summed in bf16, so the lent routes' CPU side
    # (a difference of two such sums) is good to about 2^-8 of the row's CPU output:
    # where a lent route is a small part of its row, that floor is above the slots'
    # own error (10-17 Vision: 0.49 against the CPU, 5e-3 against a float32 reference)
    cpu_all = ar(cpu_full.contiguous())
    full_rows = cpu_all[lent.any(dim=1)]
    floor = float(2.0 ** -8 * full_rows.norm() / lent_cpu.norm().clamp(min=1e-30))
    STATE["checks"] += 1
    if _CHECK_DUMP and wrapper.tp_rank == 0 and n_lent and (
            STATE["checks"] <= _CHECK_DUMP_FIRST or (rel_slots > _DUMP_REL and STATE.get("dumps", 0) < 6)):
        # KT_KV_HOT_CHECK_DUMP=<dir>: the call's inputs and both sides of the lent
        # routes, for a reference computed from the checkpoint offline
        STATE["dumps"] = STATE.get("dumps", 0) + (STATE["checks"] > _CHECK_DUMP_FIRST)
        os.makedirs(_CHECK_DUMP, exist_ok=True)
        # res: the resident experts' routes on the GPU method, the same method as the
        # slots (a reference can tell a property of the method from one of the lend)
        # cpu_full / cpu_rest: the CPU's two calls (every route; the lent ones out), so a
        # reference can tell a wrong CPU row from a wrong slot
        torch.save({"layer": L.idx, "n": L.n, "x": x.cpu(), "ids": ids.cpu(), "w": w.cpu(), "lent": lent.cpu(),
                    "slots": slots.cpu(), "cpu": lent_cpu.cpu(), "res": (ref - cpu_all).cpu(),
                    "cpu_full": cpu_all.cpu(), "cpu_rest": ar(cpu_rest.contiguous()).cpu(),
                    "cpu_full_raw": cpu_full.cpu(), "cpu_rest_raw": cpu_rest.cpu(),
                    "cpu_full2": STATE.pop("cpu_full2", torch.zeros(0)).cpu(),
                    "cpu_rest2": STATE.pop("cpu_rest2", torch.zeros(0)).cpu(),
                    "cpu_ids": hot[0].topk_output.topk_ids.cpu(), "cpu_w": hot[0].topk_output.topk_weights.cpu(),
                    "rel_slots": rel_slots, "share": share},
                   os.path.join(_CHECK_DUMP, f"check{STATE['checks']}-layer{L.idx}.pt"))
    if n_lent and rel_slots > _SLOT_REL and STATE.get("slotdiag", 0) < 4:
        # KT_KV_HOT_CHECK_SLOT_REL: past this, which slot holds what its bank does not
        # (a slot read through an unmapped or stale chunk differs from its bank)
        STATE["slotdiag"] = STATE.get("slotdiag", 0) + 1
        k = int(_k_dev(x.device)[L.idx])
        # a dynamic slot holds its bank's expert only while it still holds N + j
        bad = [f"{nm}[{j}]" for _, nm in L.names for j in range(min(k, L.b)) if L.held[j] == L.n + j
               and not torch.equal(L.slots[nm][j].cpu(), L.banks[nm][j].cpu())]
        _log(f"slot diag rank {wrapper.tp_rank} layer {L.idx} (k {k}, slots vs CPU {rel_slots:.3e}): "
             f"{len(bad)} of {k * len(L.names)} slot tensors differ from their bank: {bad[:16]}")
    if n_lent and rel_slots > 0.5 and STATE.get("diag", 0) < 3:
        # the slots gave (almost) nothing for routes they hold: what do they hold?
        STATE["diag"] = STATE.get("diag", 0) + 1
        k = int(_k_dev(x.device)[L.idx])
        for kind, nm in L.names:
            cur = (L.shadow._parameters.get(nm) if kind == "param" else L.shadow._buffers.get(nm)
                   if kind == "buffer" else L.shadow.__dict__.get(nm))
            same = cur is not None and cur.data_ptr() == L.slots[nm].data_ptr()
            # on the host: in decode the driver's memory may be the lend's (a 226 MiB
            # float copy here ran 1017 out of memory, 10-17)
            sv = float(L.slots[nm][:k].cpu().view(torch.uint8).float().abs().sum()) if k else 0.0
            bv = float(L.banks[nm][:k].cpu().view(torch.uint8).float().abs().sum()) if k else 0.0
            _log(f"diag rank {wrapper.tp_rank} layer {L.idx} {kind} {nm}: shadow reads the slot {same}, "
                 f"getattr is the slot {getattr(L.shadow, nm).data_ptr() == L.slots[nm].data_ptr()}, "
                 f"slot bytes sum {sv:.4g}, bank {bv:.4g}")
        _log(f"diag rank {wrapper.tp_rank}: this rank's slot output norm "
             f"{float(L.method.apply(L.shadow, hot[1]).hidden_states.float().norm()):.4g}, "
             f"method {type(L.method).__name__}, runner {type(getattr(L.method, 'runner', None)).__name__}")
    if wrapper.tp_rank == 0:
        _log(f"check {STATE['checks']}: layer {L.idx}, {x.shape[0]} rows, {n_lent} of {lent.numel()} routes lent "
             f"(k {int(_k_dev(x.device)[L.idx])}); lent routes are {share:.3e} of |out|, slots vs CPU on them "
             f"{rel_slots:.3e} (the CPU's bf16 floor there {floor:.1e}); |out - ref| / |ref| = {rel:.3e}")


def apply(wrapper, slots_dispatch) -> torch.Tensor:
    """The slots' share of the layer's output (zero where nothing is lent)."""
    L = wrapper.__dict__["_kt_kv_hot"]
    return L.method.apply(L.shadow, slots_dispatch).hidden_states


# -- lending -------------------------------------------------------------------------

def _span(L: _Layer, name: str, k: int) -> int:
    """Bytes from a slot tensor's start that cover k slots, in whole chunks (the
    range is reserved from a chunk boundary and in whole chunks)."""
    c = kt_kv_vmm.chunk_of(L.slots[name]) or (2 << 20)
    return -(-(k * L.sz[name]) // c) * c


def _cost(L: _Layer, k: int) -> int:
    """Bytes mapped under a layer's slots for k experts."""
    return sum(_span(L, nm, k) for _, nm in L.names) if k > 0 else 0


def _plan(budget: int) -> List[int]:
    """Slots per layer within budget bytes: one slot at a time to the layer with
    the fewest, while the next one's chunks fit."""
    ks = [0] * len(LAYERS)
    spent = 0
    while True:
        best = None
        for i, L in enumerate(LAYERS):
            if ks[i] >= L.x:
                continue
            if best is None or ks[i] < ks[best]:
                best = i
        if best is None or (SLOTS_MAX > 0 and sum(ks) >= SLOTS_MAX):
            break
        L = LAYERS[best]
        extra = _cost(L, ks[best] + 1) - _cost(L, ks[best])
        if spent + extra > budget:
            # another layer's next slot may still fit in a chunk already mapped
            cheaper = [i for i, M in enumerate(LAYERS) if ks[i] < M.x and ks[i] == ks[best]
                       and spent + _cost(M, ks[i] + 1) - _cost(M, ks[i]) <= budget]
            if not cheaper:
                break
            best = cheaper[0]
            L = LAYERS[best]
            extra = _cost(L, ks[best] + 1) - _cost(L, ks[best])
        ks[best] += 1
        spent += extra
    return ks


# -- KT_KV_HOT_DYNAMIC_SLOTS ---------------------------------------------------------

DYN = {"on": None, "order": -1, "roles": {}, "stage": None, "stream": None, "pending": None,
       "ev_unmap": None, "ev_copy": None, "ev_cnt": None, "buf": None, "scores": None, "seen": None,
       "calls": 0, "rounds": 0, "swaps": 0, "swaps_logged": 0}


def _over_ranks(vals: List[int], op=torch.distributed.ReduceOp.MIN) -> List[int]:
    """vals reduced with op over the TP ranks (the CPU group: every rank lends at the
    same forward, the free page count being the same on each)."""
    t = torch.tensor(vals, dtype=torch.int64)
    from sglang.srt.distributed.parallel_state import get_tp_group

    g = get_tp_group()
    if g.world_size > 1:
        torch.distributed.all_reduce(t, op=op, group=g.cpu_group)
    return t.tolist()


def _roles(L: _Layer):
    """{(w13|w2, is scale): slot name} when the shadow holds exactly the packed
    weights and the e8m0 scales of the two projections (bytes: uint8 or int8
    weights, uint8 or float8_e8m0fnu scales), else None."""
    r = {}
    for _, nm in L.names:
        key = ("w13" if nm.startswith("w13") else "w2" if nm.startswith("w2") else None, "scale" in nm)
        if key[0] is None or key in r or L.slots[nm].element_size() != 1:
            return None
        r[key] = nm
    return r if len(r) == 4 else None


def _to_dev(rows, dev) -> torch.Tensor:
    """A small int64 table update through pinned memory: no wait for the stream (a pageable copy waits)."""
    return torch.tensor(rows, dtype=torch.int64).pin_memory().to(dev, non_blocking=True)


def _pending(swaps, dev):
    """The table entries a round's swaps make, split by table on the host: applying them needs no boolean
    mask of a device tensor, whose element count waits for the stream."""
    out = []
    for which in (0, 1):
        rows = [(L.idx, e, i) for L, w, i, _, e in swaps if w == which]
        if rows:
            out.append((which, _to_dev(rows, dev)))
    return out


def _arena_load(L: _Layer, e: int, dst, order: int) -> None:
    """Physical expert e of the layer from this rank's part of kt-kernel's arenas
    (gate, up, down packed e2m1; their fp32 group scales) into dst, the slot
    layout: w13 [2N, K/2] with gate rows first (order 0) or up rows first (1), w2
    [K, N/2], e8m0 scale bytes. On the current stream."""
    from sglang.srt.layers.moe import kt_stream_prefill

    st = kt_stream_prefill._instance
    w13, s13, w2, s2 = (dst[k].view(torch.uint8) for k in (("w13", False), ("w13", True), ("w2", False), ("w2", True)))
    if getattr(st, "_mxc", None) is not None:
        _mxc_load(st, L, e, w13, s13, w2, s2, order)
        return
    st._register_arenas(L.layer_idx)
    if not getattr(st, "_seg_whole", True):
        _arena_load_parts(st, L, e, w13, s13, w2, s2, order)
        return
    ar = st._arenas[L.layer_idx]
    n = w13.shape[0] // 2
    ws, ss = ar[0][1], ar[3][1]
    a, b = (0, 1) if order == 0 else (1, 0)
    nb = w13[:n].numel()
    w13[:n].view(-1).copy_(ar[a][0][e * ws:e * ws + nb], non_blocking=True)
    w13[n:].view(-1).copy_(ar[b][0][e * ws:e * ws + nb], non_blocking=True)
    w2.view(-1).copy_(ar[2][0][e * ws:e * ws + w2.numel()], non_blocking=True)
    stage = DYN["stage"]
    for src, out in ((ar[3 + a], s13[:n]), (ar[3 + b], s13[n:]), (ar[5], s2)):
        c = out.numel()
        stage[:c].view(torch.uint8).copy_(src[0][e * ss:e * ss + 4 * c], non_blocking=True)
        out.view(-1).copy_(((stage[:c].view(torch.int32) >> 23) & 0xFF).to(torch.uint8))


def _mxc_unpin_done() -> None:
    """Gives the cache back the slots whose copies to the GPU are done."""
    left = []
    for moe, lid, ev in DYN.get("mxc_pins", []):
        if ev.query():
            moe.kt_cache_unpin(lid)
        else:
            left.append((moe, lid, ev))
    DYN["mxc_pins"] = left


def _mxc_load(st, L: _Layer, e: int, w13, s13, w2, s2, order: int) -> None:
    """_arena_load from kt-kernel's expert cache (KT_MXFP4_EXPERT_CACHE_GIB): the slot's gate, up, down weights and
    ue8m0 scale bytes as they are; the slot stays pinned in the cache until the copy is done. An expert the cache
    does not hold is read from the checkpoint through the streamer's bounce (and waited for)."""
    _mxc_unpin_done()
    X = st._mxc
    w = X["layers"][L.layer_idx]
    moe, lid = w.moe, w._mxc_p2l[int(e)]
    n = w13.shape[0] // 2
    a, b = (0, 1) if order == 0 else (1, 0)
    s = moe.kt_cache_pin(lid)
    if s >= 0:
        base, cache = s * X["sb"], X["cache"]
        src = [cache[base + o:base + o + c] for o, c in zip(X["off"], X["nb"])]
    else:
        src = st._mxc_read(0, [(0, lid)], w._mxc_where)[0]
    w13[:n].view(-1).copy_(src[a], non_blocking=True)
    w13[n:].view(-1).copy_(src[b], non_blocking=True)
    w2.view(-1).copy_(src[2], non_blocking=True)
    s13[:n].view(-1).copy_(src[3 + a], non_blocking=True)
    s13[n:].view(-1).copy_(src[3 + b], non_blocking=True)
    s2.view(-1).copy_(src[5], non_blocking=True)
    if s >= 0:
        ev = torch.cuda.Event()
        ev.record()
        DYN.setdefault("mxc_pins", []).append((moe, lid, ev))
    else:
        torch.cuda.current_stream().synchronize()  # the bounce is reused by the next read


def _arena_load_parts(st, L: _Layer, e: int, w13, s13, w2, s2, order: int) -> None:
    """_arena_load when this rank's rows sit in other than one whole kt-kernel part
    (kt_stream_prefill._rank_segments): gate/up rows lo..lo+c of each part are contiguous;
    the down projection's columns of a part come through a staging buffer on the device."""
    n, k = w13.shape[0] // 2, w2.shape[0]
    a, b = (0, 1) if order == 0 else (1, 0)
    stage = DYN["stage"]
    for (_, lo, c, d, m), ar in zip(st._segs, st._arena_parts[L.layer_idx]):
        ws, ss = ar[0][1], ar[3][1]
        kb, kg = w13.shape[1], s13.shape[1]  # bytes and scale groups a gate/up row
        for src, row0 in ((a, 0), (b, n)):
            w13[row0 + d:row0 + d + c].view(-1).copy_(
                ar[src][0][e * ws + lo * kb:e * ws + (lo + c) * kb], non_blocking=True)
            cnt = c * kg
            stage[:cnt].view(torch.uint8).copy_(
                ar[3 + src][0][e * ss + 4 * lo * kg:e * ss + 4 * (lo + c) * kg], non_blocking=True)
            s13[row0 + d:row0 + d + c].view(-1).copy_(((stage[:cnt].view(torch.int32) >> 23) & 0xFF).to(torch.uint8))
        # down: k rows of m/2 bytes and m/32 scales in the part; columns lo..lo+c
        if DYN.get("stage_b") is None or DYN["stage_b"].numel() < k * m // 2:
            DYN["stage_b"] = torch.empty(k * m // 2, dtype=torch.uint8, device=w2.device)
        sb = DYN["stage_b"][: k * m // 2]
        sb.copy_(ar[2][0][e * ws:e * ws + k * m // 2], non_blocking=True)
        w2[:, d // 2:(d + c) // 2].copy_(sb.view(k, m // 2)[:, lo // 2:(lo + c) // 2])
        cnt = k * (m // 32)
        if stage.numel() < cnt:
            DYN["stage"] = stage = torch.empty(cnt, dtype=torch.float32, device=w2.device)
        stage[:cnt].view(torch.uint8).copy_(ar[5][0][e * ss:e * ss + 4 * cnt], non_blocking=True)
        s2[:, d // 32:(d + c) // 32].copy_(
            ((stage[:cnt].view(torch.int32).view(k, m // 32)[:, lo // 32:(lo + c) // 32] >> 23) & 0xFF).to(torch.uint8))


def _dyn_setup(dev) -> None:
    """First lend: whether the arenas give the bytes the slots hold. Experts N, N+1
    of the first layer loaded from the arenas against their banks (the method's
    processed form), in both row orders; on only when every rank found the same."""
    from sglang.srt.layers.moe import kt_stream_prefill

    order, why = -1, ""
    L = LAYERS[0]
    roles = _roles(L)
    st = getattr(kt_stream_prefill, "_instance", None)
    if st is None or (getattr(st, "_mxc", None) is None and L.layer_idx not in getattr(st, "_arenas", {})):
        why = "no zero-copy arenas (KT_GPU_STREAM_ZEROCOPY=1 with KT_EXPERT_SHM=1)"
    elif roles is None:
        why = f"slot tensors {[(nm, str(L.slots[nm].dtype)) for _, nm in L.names]} are not packed MXFP4 bytes"
    else:
        DYN["stage"] = torch.empty(max(L.slots[roles[("w13", True)]][0].numel() // 2 + 1,
                                       L.slots[roles[("w2", True)]][0].numel()), dtype=torch.float32, device=dev)
        diff = {}
        for o in (0, 1):
            good = True
            for j in range(min(2, L.b)):
                dst = {key: torch.empty_like(L.slots[nm][j]) for key, nm in roles.items()}
                _arena_load(L, L.n + j, dst, o)
                torch.cuda.synchronize(dev)
                for key, nm in roles.items():
                    bad = int((dst[key].view(torch.uint8).cpu() != L.banks[nm][j].view(torch.uint8)).sum())
                    if bad:
                        good = False
                        diff[(o, nm)] = diff.get((o, nm), 0) + bad
            if good:
                order = o
                break
        if order < 0:
            why = f"arena bytes differ from the banks (row order, slot name: bytes): {diff}"
    lo, hi = _over_ranks([order, order], torch.distributed.ReduceOp.MIN)[0], \
        _over_ranks([order], torch.distributed.ReduceOp.MAX)[0]
    DYN["on"] = lo >= 0 and lo == hi
    if DYN["on"]:
        DYN["order"] = order
        DYN["roles"] = {M.idx: _roles(M) for M in LAYERS}
        DYN["stream"] = torch.cuda.Stream(device=dev)
        DYN["ev_unmap"], DYN["ev_copy"], DYN["ev_cnt"] = torch.cuda.Event(), torch.cuda.Event(), None
        DYN["buf"] = torch.zeros(tuple(STATE["cnt"].shape), dtype=torch.float32, pin_memory=True)
        DYN["scores"] = np.zeros(tuple(STATE["cnt"].shape))
        _log(f"dynamic slots on: arena experts match the banks ({'gate' if order == 0 else 'up'} rows first), "
             f"a round every {DYN_EVERY} forwards, up to {DYN_SWAPS * DYN_EVERY} swaps a layer, "
             f"half-life {DYN_HALF_LIFE:g}")
    if DYN_RES:
        # the resident rows swap when experts 0, 1 from the arenas match what the rows hold
        ok, why = 1, ""
        L = LAYERS[0]
        if not DYN["on"]:
            ok, why = 0, "the dynamic slots are off"
        elif L.res_t is None:
            ok, why = 0, "the resident tensors are not the packed layout"
        else:
            for r in range(min(2, L.n)):
                dst = {key: torch.empty_like(t[r]) for key, t in L.res_t.items()}
                _arena_load(L, r, dst, order)
                torch.cuda.synchronize(dev)
                bad = {str(key): int((dst[key].view(torch.uint8) != t[r].view(torch.uint8)).sum())
                       for key, t in L.res_t.items()}
                if any(bad.values()):
                    ok, why = 0, f"arena expert {r} differs from resident row {r} (bytes by role): {bad}"
                    break
        if os.environ.get("KT_KV_HOT_DYN_RES_SWAP", "1") == "0":
            # (diagnosis: the CPU holds every expert and route sends it -1, but the rows stay)
            ok, why = 0, "KT_KV_HOT_DYN_RES_SWAP=0"
        DYN["res"] = bool(_over_ranks([ok], torch.distributed.ReduceOp.MIN)[0])
        _log("dynamic resident rows " + ("on: experts 0, 1 from the arenas match their rows" if DYN["res"]
                                          else f"off: {why or 'another rank could not'}; they hold 0..N-1"))
    # the slots shrink when the dynamic slots are off (this `else` was
    # the resident rows', which cut the slots under KT_KV_HOT_DYNAMIC_RESIDENT=0, 10-25)
    if not DYN["on"]:
        for M in LAYERS:
            # the placement's experts, as many as there are banks
            M.x = M.b
            M.held = [M.n + j for j in range(M.x)]
        _log(f"dynamic slots off: {why or 'another rank could not load from its arenas'}; "
             f"the slots hold the placement's experts")


def _dyn_fill(L: _Layer, k: int) -> None:
    """A lend's copy into a layer's k slots: what each held before the window, from
    its bank when it is still the placement's expert, else from the arenas."""
    # an expert held twice (a slot past an earlier lend's k kept its expert while a
    # round put it in a lower slot, 10-21) takes one slot; the other is left empty
    seen = set(L.res_held) if DYN.get("res") and L.res_t is not None else set()
    for j in range(k):
        if L.held[j] in seen:
            L.held[j] = -1
        elif L.held[j] >= 0:
            seen.add(L.held[j])
    for j in range(k):
        e = L.held[j]
        if (e == L.n + j or not DYN["on"]) and j < L.b:
            L.held[j] = L.n + j
            for _, nm in L.names:
                L.slots[nm][j].copy_(L.banks[nm][j], non_blocking=True)
        elif not DYN["on"]:
            L.held[j] = -1
        elif e >= 0:
            _arena_load(L, e, {key: L.slots[nm][j] for key, nm in DYN["roles"][L.idx].items()}, DYN["order"])
    for j in range(k, L.x):
        L.held[j] = -1  # unmapped: what it held is gone
    if DYN.get("res") and L.res_t is not None:
        # the resident rows back to what they held before the window (res_guard put
        # 0..N-1 in them for the streamer)
        for r in range(L.n):
            if L.res_now[r] != L.res_held[r]:
                _arena_load(L, L.res_held[r], {key: t[r] for key, t in L.res_t.items()}, DYN["order"])
                L.res_now[r] = L.res_held[r]


def _dyn_table(dev) -> None:
    """The slot table from what each lent slot holds, and the resident rows' table."""
    tbl = STATE["tbl"]
    tbl.fill_(-1)
    rows = [(L.idx, L.held[j], j) for L in LAYERS for j in range(L.k) if L.held[j] >= 0]
    if rows:
        t = torch.tensor(rows, dtype=torch.int64).to(dev)
        tbl[t[:, 0], t[:, 1]] = t[:, 2]
    if DYN_RES:
        rtbl = STATE["rtbl"]
        rtbl.fill_(-1)
        t = torch.tensor([(L.idx, L.res_now[r], r) for L in LAYERS for r in range(L.n)], dtype=torch.int64).to(dev)
        rtbl[t[:, 0], t[:, 1]] = t[:, 2]


def res_guard() -> None:
    """The KT wrapper's apply, first thing: in a prefill window the resident rows hold
    experts 0..N-1 again. The prefill lend reads every layer's weights from its pinned
    mirror, made at load, and copies the mirror back when decode resumes, so no copy is
    made here (writing the rows inside a window hits the lent memory: a segfault,
    10-21); the table says so, and the next lend fills the rows again (_dyn_fill). A
    round's copies finished before the window paused the weights: its release syncs."""
    if not DYN_RES or not DYN.get("res") or not _in_window() or torch.cuda.is_current_stream_capturing():
        return
    if all(L.res_now == list(range(L.n)) for L in LAYERS):
        return
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    DYN["pending"] = None
    rtbl = STATE["rtbl"]
    rtbl.fill_(-1)
    for L in LAYERS:
        L.res_now = list(range(L.n))
        rtbl[L.idx, : L.n] = torch.arange(L.n, device=dev)


def _dyn_round(dev) -> None:
    """Every DYN_EVERY decode forwards, outside the graph: last round's copies enter
    the table, last round's counts become scores, and each layer swaps held experts
    for better-scored ones (out of the table now, copied in on the side stream)."""
    DYN["calls"] += 1
    if DYN["calls"] % DYN_EVERY:
        return
    cur = torch.cuda.current_stream(dev)
    tbl = STATE["tbl"]
    rtbl = STATE.get("rtbl")
    if DYN["pending"] is not None:
        cur.wait_event(DYN["ev_copy"])
        for which, t in DYN["pending"]:  # split on the host
            table = tbl if which == 0 else rtbl
            if table is not None:  # (no resident rows' table without KT_KV_HOT_DYNAMIC_RESIDENT)
                table[t[:, 0], t[:, 1]] = t[:, 2]
        DYN["pending"] = None
    if DYN["ev_cnt"] is not None:
        DYN["ev_cnt"].synchronize()
        seen = DYN["buf"].numpy().astype(np.float64)
        DYN["scores"] *= 0.5 ** (DYN_EVERY / DYN_HALF_LIFE)
        DYN["scores"] += seen
        DYN["seen"] = seen if DYN["seen"] is None else DYN["seen"] + seen
        DYN["last"] = seen
    else:
        DYN["ev_cnt"] = torch.cuda.Event()
    DYN["buf"].copy_(STATE["cnt"], non_blocking=True)
    STATE["cnt"].zero_()
    DYN["ev_cnt"].record(cur)
    DYN["rounds"] += 1
    if DYN["ev_cnt"] is not None and DYN.get("last") is not None:
        # this round's routes against what the tables held through it
        seen, res = DYN["last"], bool(DYN.get("res"))
        for L in LAYERS:
            on = [e for e in L.held[: L.k] if e >= 0] + (L.res_now if res else [])
            DYN["hit"] = DYN.get("hit", 0.0) + float(seen[L.idx, on].sum())
            DYN["all"] = DYN.get("all", 0.0) + float(seen[L.idx, 0 if res else L.n:].sum())
    budget = DYN_SWAPS * DYN_EVERY
    swaps = []  # (layer, 0: slot / 1: resident row, its index, expert out, expert in)
    for L in LAYERS:
        k = L.k
        if not k:
            continue
        sc, held = DYN["scores"][L.idx], L.held
        # the places a layer's GPU experts sit in: its lent slots, and with the dynamic
        # resident rows those rows too (one pool: the lowest score anywhere goes)
        res = bool(DYN.get("res")) and L.res_t is not None
        places = [(0, j) for j in range(k)] + ([(1, r) for r in range(L.n)] if res else [])

        def has(p):
            return held[p[1]] if p[0] == 0 else L.res_held[p[1]]

        hs, changed, n = {has(p) for p in places}, set(), 0
        for e in np.argsort(-sc, kind="stable").tolist():
            if n >= budget or sc[e] <= 0:
                break
            if (e < L.n and not res) or e in hs:
                continue
            opts = [p for p in places if p not in changed]
            if not opts:
                break
            empty = [p for p in opts if has(p) < 0]
            if empty:
                p = empty[0]
            else:
                p = min(opts, key=lambda q: (sc[has(q)], q))
                if not sc[e] > sc[has(p)] * DYN_HYST:
                    break
            old = has(p)
            swaps.append((L, p[0], p[1], old, e))
            hs.discard(old)
            hs.add(e)
            if p[0] == 0:
                held[p[1]] = e
            else:
                L.res_held[p[1]] = e
            changed.add(p)
            n += 1
    if swaps:
        for which, table in ((0, tbl), (1, rtbl)):
            out = [(L.idx, old) for L, w, _, old, _ in swaps if w == which and old >= 0]
            if out and table is not None:
                # flat indices and index_fill_: -1 as a kernel argument (an index put of the
                # Python -1 waits for the stream)
                t = _to_dev([i * table.shape[1] + e for i, e in out], dev)
                table.view(-1).index_fill_(0, t, -1)
        DYN["ev_unmap"].record(cur)
        side = DYN["stream"]
        with torch.cuda.stream(side):
            side.wait_event(DYN["ev_unmap"])
            for L, w, i, _, e in swaps:
                if w == 0:
                    dst = {key: L.slots[nm][i] for key, nm in DYN["roles"][L.idx].items()}
                else:
                    dst = {key: t[i] for key, t in L.res_t.items()}
                    L.res_now[i] = e
                _arena_load(L, e, dst, DYN["order"])
            DYN["ev_copy"].record(side)
        DYN["pending"] = _pending(swaps, dev)
        DYN["swaps"] += len(swaps)
    if DYN["rounds"] % DYN_LOG == 0 and DYN["seen"] is not None:
        res = bool(DYN.get("res"))
        hit, cpu = DYN.pop("hit", 0.0), DYN.pop("all", 0.0)
        _log(f"dynamic round {DYN['rounds']}: the GPU took {hit / max(cpu, 1):.1%} of the routes"
             + (" (resident rows and slots)" if res else " past the resident experts")
             + f" over the last {DYN_LOG} rounds, {DYN['swaps'] - DYN['swaps_logged']} swaps "
             f"({(DYN['swaps'] - DYN['swaps_logged']) / (DYN_LOG * DYN_EVERY * max(len(LAYERS), 1)):.3f} "
             f"a forward and layer)")
        DYN["seen"], DYN["swaps_logged"] = None, DYN["swaps"]


def _hook() -> None:
    if STATE["hooked"]:
        return
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    m.PRE_RESTORE.append(release)
    if hasattr(m, "WINDOW_START"):
        m.WINDOW_START.append(rows_home)
    STATE["hooked"] = True


def rows_home(why: str) -> None:
    """A prefill window opens, the weights not yet paused: the resident rows hold experts
    0..N-1 again, loaded from the arenas whole (the lend's mirror brings back only the
    tensors it lends: the scales stayed the swapped-in expert's), and the table says so."""
    if not DYN_RES or not DYN.get("res") or not LAYERS:
        return
    if all(L.res_now == list(range(L.n)) for L in LAYERS):
        return
    t0 = time.perf_counter()
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    torch.cuda.synchronize(dev)  # a round's copies into the rows land first
    DYN["pending"] = None
    moved = 0
    for L in LAYERS:
        if L.res_t is None:
            continue
        for r in range(L.n):
            if L.res_now[r] != r:
                _arena_load(L, r, {key: t[r] for key, t in L.res_t.items()}, DYN["order"])
                L.res_now[r] = r
                moved += 1
    rtbl = STATE["rtbl"]
    rtbl.fill_(-1)
    for L in LAYERS:
        rtbl[L.idx, : L.n] = torch.arange(L.n, device=dev)
    torch.cuda.synchronize(dev)  # before the pause unmaps them
    if moved and DYN.get("home_logged", 0) < 3:
        DYN["home_logged"] = DYN.get("home_logged", 0) + 1
        _log(f"{why}: {moved} resident rows back to experts 0..N-1 in {(time.perf_counter() - t0) * 1e3:.1f} ms")


def tick(runner) -> None:
    """kt_lend.before_forward outside a prefill window (decode, and extends below
    the streaming threshold), outside any graph: lend when there is room."""
    if not ENABLED or not LAYERS or runner is None or getattr(runner, "is_draft_worker", False):
        return
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    if m.STATE["lent"] is not None:
        if DYNAMIC and DYN["on"] and LAYERS[0].k:
            _dyn_round(LAYERS[0].slots[LAYERS[0].names[0][1]].device)
        return
    if not STATE["want"]:
        STATE["since"] += 1
        if STATE["since"] < RETRY:
            return
    STATE["want"], STATE["since"] = False, 0
    _hook()
    t0 = time.perf_counter()
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    # MIN_MB is the whole lend's floor, free pages and the driver's spare together: a
    # smaller KV layout (V4.1's FP4 compressed caches halve the full pool's pages)
    # leaves fewer free-page bytes and puts the rest in the driver's free memory
    min_bytes = MIN_MB << 20
    if SPARE_FLOOR_MB >= 0:
        torch.cuda.empty_cache()
        spare_now = torch.cuda.mem_get_info(dev)[0] - (SPARE_FLOOR_MB << 20)
        min_bytes = max(1, min_bytes - max(0, spare_now))
    got = m.decode_lend(runner, RESERVE, min_bytes)
    if got <= 0:
        return
    budget = got - (MARGIN_MB << 20)
    spare = 0
    if SPARE_FLOOR_MB >= 0:
        # what the driver has free now counts the KV chunks just unmapped
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(dev)[0] - (SPARE_FLOOR_MB << 20)
        if free > budget:
            spare, budget = free - budget, free
    ks = _plan(budget)
    # rank 0's routing decides what the CPU skips, so every rank's slots must take
    # the same routes: the slot counts are the smallest over the ranks (the spare
    # memory differs between the cards; the free pages do not)
    ks = _over_ranks(ks)
    if DYNAMIC and DYN["on"] is None:
        _dyn_setup(dev)
        # the setup may have cut the slots to the banks
        ks = [min(k, L.x) for L, k in zip(LAYERS, ks)]
    for i, (L, k) in enumerate(zip(LAYERS, ks)):
        if k and any(kt_kv_vmm.restore(L.slots[nm], 0, _span(L, nm, k)) < 0 for _, nm in L.names):
            # no memory after all: this layer and the rest get nothing
            _log(f"mapping layer {i}'s slots failed ({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free): "
                 f"stopping at {sum(ks[:i])} slots")
            for M in LAYERS[i:]:
                for _, nm in M.names:
                    kt_kv_vmm.release(M.slots[nm], 0, _span(M, nm, M.x))
            ks = ks[:i] + [0] * (len(LAYERS) - i)
            break
    # a rank that could not map them all: the others give back what it lacks
    agreed = _over_ranks(ks)
    for L, k, a in zip(LAYERS, ks, agreed):
        if a < k:
            for _, nm in L.names:
                kt_kv_vmm.release(L.slots[nm], _span(L, nm, a), _span(L, nm, k) - _span(L, nm, a))
    ks = agreed
    for L, k in zip(LAYERS, ks):
        if k and DYNAMIC:
            _dyn_fill(L, k)
        elif k:
            for _, nm in L.names:
                L.slots[nm][:k].copy_(L.banks[nm][:k], non_blocking=True)
        L.k = k
    if DYNAMIC:
        _dyn_table(dev)
    kd = _k_dev(dev)
    kd.copy_(torch.tensor(ks, dtype=torch.int64), non_blocking=True)
    torch.cuda.synchronize(dev)
    STATE["lent_slots"] = sum(ks)
    # what the slots hold beyond the KV pages: back at the window's start, so the
    # prefill plan counts it as free (sglang-kt-kv-hot-spare-plan.py)
    STATE["spare_held"] = max(0, sum(_cost(L, k) for L, k in zip(LAYERS, ks)) - got)
    STATE["lends"] += 1
    if STATE["lends"] in (1, 2, 10, 100, 1000) or CHECK:
        _log(f"lend {STATE['lends']}: {got / 2**30:.2f} GiB of free pages unmapped"
             + (f" + {spare / 2**30:.2f} GiB the driver had free" if spare else "") + f", {sum(ks)} slots "
             f"({min(ks)}-{max(ks)} a layer of {LAYERS[0].x}) in {(time.perf_counter() - t0) * 1e3:.1f} ms"
             + (f"; the shadows keep {STATE.get('left', 0) / 2**20:.0f} MiB outside their slots"
                if STATE["lends"] == 1 else ""))


def release(why: str) -> None:
    """Before the lent pages are mapped back (kt_kv_prefill_lend.restore): the
    routing leaves the slots, then the slots give their memory back."""
    if not LAYERS:
        return
    t0 = time.perf_counter()
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    kd = _k_dev(dev)
    kd.zero_()
    if DYNAMIC:
        # the slots leave the table; a round's copies in flight finish in the sync,
        # and what they were for is in each layer's held list for the next lend
        STATE["tbl"].fill_(-1)
        DYN["pending"] = None
    torch.cuda.synchronize(dev)
    for L in LAYERS:
        if L.k:
            for _, name in L.names:
                kt_kv_vmm.release(L.slots[name], 0, _span(L, name, L.x))
        L.k = 0
    torch.cuda.synchronize(dev)
    STATE["releases"] += 1
    STATE["want"] = True
    if STATE["releases"] in (1, 2, 10, 100, 1000) or CHECK or why not in ("window start",):
        _log(f"release {STATE['releases']} ({why}): {STATE['lent_slots']} slots out in "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms")
    STATE["lent_slots"] = 0
    STATE["spare_held"] = 0
