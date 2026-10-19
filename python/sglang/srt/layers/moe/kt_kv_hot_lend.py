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
lends the same: the decision depends only on the allocator's free page count.
"""
import copy
import os
import time
from typing import List, Optional

import torch

from sglang.srt.layers.moe import kt_kv_vmm

ENABLED = os.environ.get("KT_KV_HOT_LEND", "0") == "1" and kt_kv_vmm.ENABLED
EXTRA = int(os.environ.get("KT_KV_HOT_EXTRA", "0") or 0)
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

LAYERS: List["_Layer"] = []
STATE = {"k": None, "want": True, "since": 0, "lent_slots": 0, "lends": 0, "releases": 0, "hooked": False,
         "calls": 0, "checks": 0}


def _log(msg: str) -> None:
    print(f"[kt-kv-hot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_count(n: int) -> int:
    """GPU experts a loader that skips CPU experts' tensors must still read: the
    resident N and, with the hot lend, the X the shadows hold."""
    return n + EXTRA if ENABLED and EXTRA > 0 and n > 0 else n


def make_shadow(wrapper, layer, hidden_size: int, intermediate_size_per_partition: int,
                params_dtype, extra_weight_attrs) -> None:
    """The wrapper's create_weights, after the GPU method made the layer's N
    experts: a shadow of the layer, outside the model (so no span of the prefill
    lend holds it), with the method's weights for X experts on the CPU; the
    weight loader fills them (shadow_param), split processes them."""
    if not ENABLED or EXTRA <= 0 or int(wrapper.num_gpu_experts) <= 0:
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
        method.create_weights(layer=shadow, num_experts=EXTRA, hidden_size=hidden_size,
                              intermediate_size_per_partition=intermediate_size_per_partition,
                              params_dtype=params_dtype, **extra_weight_attrs)
    for p in shadow._parameters.values():
        if isinstance(p, torch.Tensor) and p.is_cuda:
            p.data = p.data.cpu()
    wrapper.__dict__["_kt_kv_hot_shadow"] = (shadow, method, set(vars(shadow)))
    # a loader that skips the checkpoint tensors of experts with no GPU slot
    # (DeepSeek V4's KT_LOAD_SKIP_CPU_EXPERTS) must still read these X
    wrapper.__dict__["load_gpu_experts"] = int(wrapper.num_gpu_experts) + EXTRA


def shadow_param(wrapper, layer, param, expert_id: int):
    """The layer's weight loader, for a physical expert the GPU does not hold:
    (the shadow's tensor of the same name, its slot) when it is one of the X,
    else None."""
    sh = wrapper.__dict__.get("_kt_kv_hot_shadow")
    if sh is None:
        return None
    n = int(wrapper.num_gpu_experts)
    if not n <= expert_id < n + EXTRA:
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
        self.n, self.x = int(wrapper.num_gpu_experts), EXTRA
        self.k = 0
        self.names, self.slots, self.banks, self.sz = [], {}, {}, {}
        self.shadow = self.method = None


def split(wrapper, layer) -> None:
    """The wrapper's process_weights_after_loading, after the layer's own: the
    shadow's X experts are processed on the GPU by its copy of the method, moved
    to pinned host memory, and its per-expert tensors become slots."""
    sh = wrapper.__dict__.pop("_kt_kv_hot_shadow", None)
    if sh is None:
        return
    shadow, method, made = sh
    x = EXTRA
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
        cfg.num_local_experts = x
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
    if not found:
        raise RuntimeError(f"KT_KV_HOT_LEND: the shadow of layer {wrapper.kt_config.layer_idx} has no tensor "
                           f"of {x} experts")
    for kind, name, t in found:
        bank = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        bank.copy_(t)
        slot = kt_kv_vmm.shadow_tensor(t.shape, t.dtype, dev.index)
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
    wrapper.__dict__["_kt_kv_hot"] = L
    if L.idx == 0:
        _log(f"N {L.n} + X {x}: per-expert tensors {[(nm, tuple(L.slots[nm].shape), L.sz[nm]) for _, nm in L.names]}, "
             f"{sum(L.sz.values()) * x / 2**20:.1f} MiB a layer in pinned memory and in slots")


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
    submit: (dispatch for the CPU, dispatch for the slots) or None."""
    L = wrapper.__dict__.get("_kt_kv_hot")
    if L is None:
        return None
    if _in_window():
        return None  # a streamed prefill: nothing is lent, the streamer runs them
    topk = dispatch_output.topk_output
    ids, w = topk.topk_ids, topk.topk_weights
    k = _k_dev(ids.device)[L.idx]
    lent = (ids >= L.n) & (ids < L.n + k)
    cpu = dispatch_output._replace(topk_output=topk._replace(
        topk_ids=torch.where(lent, torch.zeros_like(ids), ids),
        topk_weights=torch.where(lent, torch.zeros_like(w), w)))
    if _MASK_ZERO:
        # a runner that cannot take -1 (Marlin): slot 0 at weight 0, which reads
        # zeros when nothing is lent
        sid = torch.where(lent, ids - L.n, torch.zeros_like(ids)).to(torch.int64)
        sw = torch.where(lent, w, torch.zeros_like(w))
    else:
        sid = torch.where(lent, ids - L.n, torch.full_like(ids, -1))
        sw = w
    slots = dispatch_output._replace(topk_output=topk._replace(topk_ids=sid, topk_weights=sw))
    saved = (ids.clone(), w.clone(), lent) if CHECK else None
    return cpu, slots, saved


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
        bad = [f"{nm}[{j}]" for _, nm in L.names for j in range(k)
               if not torch.equal(L.slots[nm][j].cpu(), L.banks[nm][j].cpu())]
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


def _hook() -> None:
    if STATE["hooked"]:
        return
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    m.PRE_RESTORE.append(release)
    STATE["hooked"] = True


def tick(runner) -> None:
    """kt_lend.before_forward outside a prefill window (decode, and extends below
    the streaming threshold), outside any graph: lend when there is room."""
    if not ENABLED or not LAYERS or runner is None or getattr(runner, "is_draft_worker", False):
        return
    from sglang.srt.layers.moe import kt_kv_prefill_lend as m

    if m.STATE["lent"] is not None:
        return
    if not STATE["want"]:
        STATE["since"] += 1
        if STATE["since"] < RETRY:
            return
    STATE["want"], STATE["since"] = False, 0
    _hook()
    t0 = time.perf_counter()
    got = m.decode_lend(runner, RESERVE, MIN_MB << 20)
    if got <= 0:
        return
    dev = LAYERS[0].slots[LAYERS[0].names[0][1]].device
    budget = got - (MARGIN_MB << 20)
    spare = 0
    if SPARE_FLOOR_MB >= 0:
        # what the driver has free now counts the KV chunks just unmapped
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info(dev)[0] - (SPARE_FLOOR_MB << 20)
        if free > budget:
            spare, budget = free - budget, free
    ks = _plan(budget)
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
    for L, k in zip(LAYERS, ks):
        if k:
            for _, nm in L.names:
                L.slots[nm][:k].copy_(L.banks[nm][:k], non_blocking=True)
        L.k = k
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
