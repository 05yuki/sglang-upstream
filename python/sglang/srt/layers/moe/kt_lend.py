# SPDX-License-Identifier: Apache-2.0
"""KT prefill lend (KT_PREFILL_LEND=1): during a streamed prefill the
GPU-resident weights hand their physical memory to the prefill and come back,
at the same addresses, before decode.

Model-agnostic; a tree wires it in at five points (see
patches/sglang-kt-prefill-lend.py for SGLang):

  bind_model(model)        right after the model is built, before any weight
                           loads (model_loader._initialize_model)
  after_load()             after every module's weight processing, before the
                           KV pool is sized (load_weights_and_postprocess)
  before_forward(batch)    at the model runner's forward entry, outside any graph
  scratch_empty(...)       for buffers only a streamed prefill uses (the
                           streamer's slots and inputs, the prefill workspace)
  scratch_region()         the same for allocations made lazily inside a
                           streamed prefill (contents are not kept from one
                           prefill window to the next)
  phase()                  "load" / "decode" / "prefill", for code that picks
                           a buffer by phase

What is lent: the first model bound (a draft built later stays resident),
its decoder layers (the largest ModuleList named "layers")
and, per layer, every CUDA parameter of at least KT_PREFILL_LEND_MIN_BYTES
(default 1 MiB) whose name contains none of KT_PREFILL_LEND_SKIP
(comma-separated, default "engram": V4.1's engram is read ahead of its layer
on another stream). Each layer gets one span, its own allocation in
torch_memory_saver tag "kt_lend_weights", made layer by layer at the bind so
the originals are dropped as it goes; the span has room for the layer's
small parameters too, which the loading may stack into a fused owner.

Fused owners: after loading, a model may have stacked several parameters into
one tensor attribute and turned the parameters into views of it (Flash-Next's
gated delta net: _fused_in_proj_weight = [in_proj_qkvz; in_proj_ba]). Such a
tensor (a plain CUDA tensor attribute of a module in the layer, holding at
least one lent parameter) is taken whole into the span; every parameter of
the layer that points into it becomes a view of it there, and it is pointed
at the slot too while lent. Tensors the model derives as copies (no
parameter inside) are left alone: they stay resident.

After load, every layer is laid out again from what the loading left:
fused owners first, then the lent parameters as they are now (a repack may
have changed their shape, or made new ones of at least the minimum size),
through pinned host memory back into the same span when it fits.

Phase swap: to prefill = pause the weights tag (physical memory released,
addresses kept), resume the scratch tag; to decode = empty_cache, pause
scratch, resume weights, copy the pinned mirror back. While paused, the
weights' addresses map nothing: a read that misses the per-layer entry
faults at once. Entering layer i (a forward pre-hook on the layer and on
each child holding lent parameters, and every forward* method of the layer,
for models that call the layer by another method, like V4's
forward_hc_pre_from_prev) points the previous layer back at its span, points
layer i's lent tensors at slot i % 2 (prefetched on a side stream while
layer i-1 ran) and starts layer i+1's copy.

Chunk hint (KT_PREFILL_LEND_HINT=<prefix>): each prefill window measures the
free memory at its start, torch's peak reserved growth and the largest
forward; the rank writes <prefix>-rank<r>.json with the chunk that would
still have fitted (margin KT_PREFILL_LEND_MARGIN_MB, 256), the smallest since
the server started.

Requires torch_memory_saver (hook mode "torch").
"""

import contextlib
import math
import os
from typing import Optional
import time

import torch

ENABLED = os.environ.get("KT_PREFILL_LEND", "0") == "1"
MIN_BYTES = int(os.environ.get("KT_PREFILL_LEND_MIN_BYTES", str(1 << 20)))
TAG_WEIGHTS = "kt_lend_weights"
TAG_SCRATCH = "kt_lend_scratch"
HINT = os.environ.get("KT_PREFILL_LEND_HINT", "")
# 256 MiB (was 512): what a window adds outside torch's allocator measured 0 to
# 62 MiB over V4.1 and Qwen3.5 windows of 7K to 989K tokens (10-06); growth with
# the sequence is in the measured peak of a window, not in this margin
MARGIN = int(os.environ.get("KT_PREFILL_LEND_MARGIN_MB", "256")) << 20
DEBUG = os.environ.get("KT_PREFILL_LEND_DEBUG", "0") == "1"
SKIP = [w for w in os.environ.get("KT_PREFILL_LEND_SKIP", "engram").split(",") if w]
# KT_PREFILL_LEND_IDLE: lent but not brought back layer by layer during a
# prefill, for weights the prefill path never reads (GLM-5.3's hot experts:
# its full-GPU prefill writes every expert into its own slot). Laid out at the
# end of the span; only the part before them is copied into the slot, and they
# keep pointing at the paused span, so a read faults instead of going wrong.
IDLE = [w for w in os.environ.get("KT_PREFILL_LEND_IDLE", "").split(",") if w]
# KT_PREFILL_LEND_PFONLY: lent weights only the prefill reads (Flash-Next's bf16
# hyper-connection mix; decode runs int8 copies): in the mirror and the slot
# copy, laid out at the end of the layer, but not in the decode span; outside a
# layer's prefill turn they point at an empty tensor (kt-lend-prefill-only.py)
PFONLY = [w for w in os.environ.get("KT_PREFILL_LEND_PFONLY", "").split(",") if w]
if PFONLY and IDLE:
    raise RuntimeError("KT_PREFILL_LEND_PFONLY and KT_PREFILL_LEND_IDLE do not combine")
# KT_PREFILL_LEND_LONG_SEQ: an extend below the streaming threshold still runs
# lent when its sequence is this long (65536). A long prompt's last chunk, or a
# short turn on a long conversation, makes prefix-sized buffers (V4.1's indexer
# gathers every compressed K: 484 MiB at a 989K-token prefix) that a card with
# its weights and KV pool back does not have room for (10-06). Not with IDLE:
# its weights are not brought back for a forward the streamer does not take.
LONG_SEQ = int(os.environ.get("KT_PREFILL_LEND_LONG_SEQ", "65536"))
# KT_PREFILL_LEND_SNAPSHOT=<prefix>: torch's allocation history of the first
# window of at least KT_PREFILL_LEND_SNAPSHOT_ROWS rows (16384), dumped to
# <prefix>-<device>.pickle at its end (tools/mem-peak-breakdown.py reads it)
SNAPSHOT = os.environ.get("KT_PREFILL_LEND_SNAPSHOT", "")
SNAPSHOT_ROWS = int(os.environ.get("KT_PREFILL_LEND_SNAPSHOT_ROWS", "16384"))

STATE = {"phase": "load", "scratch_bytes": 0, "scratch_chunk_bytes": 0, "to_prefill": 0, "to_decode": 0}
_saver = None


def _log(msg: str) -> None:
    print(f"[kt-lend] {msg}", flush=True)


def _tms():
    global _saver
    if _saver is None:
        import torch_memory_saver as m

        _saver = m.torch_memory_saver
        _saver.hook_mode = "torch"
    return _saver


def _align(n: int) -> int:
    return (n + 255) & ~255


def _ready(t) -> bool:
    return not isinstance(t, torch.nn.parameter.UninitializedParameter)


def _nbytes(t) -> int:
    if isinstance(t, torch.nn.parameter.UninitializedParameter):
        # a GGUF qweight before the load materializes it: room for the
        # largest type the GGUF uses for dense weights (Q8_0, 8.5 bits)
        shape = getattr(t, "tensor_shape", None)
        return int(math.prod(shape)) * 9 // 8 if shape else 0
    return t.numel() * t.element_size()


def _pinned_bounce(n: int):
    """(a pinned host buffer of n bytes, whether it is reused). KT_FAST_PINNED=1
    with kt_pinned: one reused bounce instead of a cudaHostAlloc per layer; the
    copies through it are synchronous, so the next layer may reuse it."""
    try:
        from sglang.srt.layers.moe import kt_pinned
    except ImportError:
        return torch.empty(n, dtype=torch.uint8, pin_memory=True), False
    return kt_pinned.bounce(n), bool(kt_pinned.ENABLED)


def _pinned_empty(n: int) -> torch.Tensor:
    """The host mirror: THP + cudaHostRegister with kt_pinned (KT_FAST_PINNED=1)."""
    try:
        from sglang.srt.layers.moe import kt_pinned
    except ImportError:
        return torch.empty(n, dtype=torch.uint8, pin_memory=True)
    return kt_pinned.empty((n,), torch.uint8)


def phase() -> str:
    return STATE["phase"]


# -- scratch ---------------------------------------------------------------------

def scratch_empty(shape, dtype, device, per_chunk: bool = False) -> torch.Tensor:
    """A tensor resident only during a streamed prefill (a plain one when
    lending is off). Call before the weights load. per_chunk: its size grows
    with the prefill chunk (counted for the chunk hint)."""
    if not ENABLED:
        return torch.empty(shape, dtype=dtype, device=device)
    with _tms().region(tag=TAG_SCRATCH):
        t = torch.empty(shape, dtype=dtype, device=device)
    STATE["scratch_bytes"] += _nbytes(t)
    if per_chunk:
        STATE["scratch_chunk_bytes"] += _nbytes(t)
    return t


def scratch_region():
    """Allocations inside go to the scratch tag while lending is on. Meant for
    buffers a streamed prefill makes on its first use (the tag is resident
    then); outside a prefill they are made as plain allocations."""
    if not ENABLED or STATE["phase"] == "decode":
        return contextlib.nullcontext()
    return _tms().region(tag=TAG_SCRATCH)


# -- layout ----------------------------------------------------------------------
# A layer's layout is a list of items, each placed at a byte offset of its span:
#   ("param", name, shape, dtype, off, nbytes)
#   ("owner", module, attr, shape, dtype, off, nbytes, [(name, shape, dtype, delta, nbytes), ...])
# where an owner's inner parameters are views at off + delta.

def _decoder_layers(model):
    best, best_bytes = None, 0
    for name, m in model.named_modules():
        if not isinstance(m, torch.nn.ModuleList) or not (name == "layers" or name.endswith(".layers")):
            continue
        nb = sum(_nbytes(p) for p in m.parameters() if p.device.type == "cuda")
        if nb > best_bytes:
            best, best_bytes = m, nb
    if best is None:
        return []
    return [l for l in best if isinstance(l, torch.nn.Module) and type(l).__name__ != "PPMissingLayer"]


def _place(dspan, item):
    if item[0] == "param":
        _, name, shape, dtype, off, nb = item
        return dspan[off:off + nb].view(dtype).view(shape)
    _, mod, attr, shape, dtype, off, nb, inner = item
    return dspan[off:off + nb].view(dtype).view(shape)


def _point(layer, dspan, layout, limit=None) -> None:
    """Point the layer's lent tensors at dspan (its device span or a slot);
    with a limit, only those placed before it."""
    for item in layout:
        if limit is not None and (item[4] if item[0] == "param" else item[5]) >= limit:
            continue
        if item[0] == "param":
            layer.get_parameter(item[1]).data = _place(dspan, item)
        else:
            _, mod, attr, shape, dtype, off, nb, inner = item
            setattr(mod, attr, _place(dspan, item))
            for name, pshape, pdtype, delta, pnb in inner:
                layer.get_parameter(name).data = dspan[off + delta:off + delta + pnb].view(pdtype).view(pshape)


def _pfonly_empty(layer, layout, dec) -> None:
    """The layer's prefill-only weights (placed at or after dec) point at empty tensors."""
    for item in layout:
        if item[0] == "param" and item[4] >= dec:
            layer.get_parameter(item[1]).data = torch.empty(0, dtype=item[3], device=STATE["device"])


def _point_back(i: int) -> None:
    """Layer i back at its decode span (prefill-only weights at empty tensors)."""
    dec = STATE.get("dec_end", {}).get(i)
    _point(STATE["layers"][i], STATE["dev_spans"][i], STATE["layouts"][i], limit=dec)
    if dec is not None and dec < STATE["spans"][i]:
        _pfonly_empty(STATE["layers"][i], STATE["layouts"][i], dec)


def bind_model(model) -> None:
    """Right after the model is built: each decoder layer's large CUDA
    parameters move into one span of their own, layer by layer."""
    if not ENABLED or "layers" in STATE or STATE["phase"] != "load":
        return
    layers = _decoder_layers(model)
    entries = []
    for layer in layers:
        lay, off, slack = [], 0, 0
        for name, p in layer.named_parameters():
            if isinstance(p, torch.nn.parameter.UninitializedParameter):
                slack += _align(_nbytes(p))  # GGUF qweight: made by the load, taken in after it
                continue
            if p.device.type != "cuda" or not p.is_contiguous() or any(w in name for w in SKIP + PFONLY):
                continue
            if _nbytes(p) < MIN_BYTES:
                # room for it in case the loading stacks it into a fused
                # owner with lent parameters (Flash-Next's in_proj_ba)
                slack += _align(_nbytes(p))
                continue
            lay.append(("param", name, tuple(p.shape), p.dtype, off, _nbytes(p)))
            off = _align(off + _nbytes(p))
        if lay:
            entries.append((layer, lay, _align(off) + slack))
    if not entries:
        _log("no decoder layer with lendable weights: off")
        return
    device = entries[0][0].get_parameter(entries[0][1][0][1]).device
    dev_spans = []
    for idx, (layer, lay, span) in enumerate(entries):
        with _tms().region(tag=TAG_WEIGHTS):
            dspan = torch.empty(span, dtype=torch.uint8, device=device)
        for item in lay:
            _place(dspan, item).copy_(layer.get_parameter(item[1]))
        _point(layer, dspan, lay)
        dev_spans.append(dspan)
        # the dropped originals go back to torch's cache, not to the driver,
        # and torch_memory_saver allocates from the driver: hand them back
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        _hook_entries(layer, idx, {item[1] for item in lay})
    STATE.update(layers=[e[0] for e in entries], layouts=[e[1] for e in entries],
                 spans=[e[2] for e in entries], dev_spans=dev_spans, dev_full=list(dev_spans),
                 device=device, counts=dict(owners=0, grown=0, moved=0, dropped=0, aliases=0, early=0))
    for idx, (layer, _, _) in enumerate(entries):
        for m in layer.modules():
            m._kt_lend_layer = idx
    # the one-layer slots now, while the card is empty (the spans' room
    # covers what moving the owners in adds)
    STATE["slots"] = [scratch_empty((max(STATE["spans"]),), torch.uint8, device) for _ in range(2)]
    _log(f"{len(entries)} decoder layers, {sum(STATE['spans']) / 1e9:.3f} GB of weights in per-layer "
         f"spans on {device} (min parameter {MIN_BYTES / 1e6:.1f} MB)")


def _hook_entries(layer, idx: int, lent) -> None:
    """Every way into layer idx enters it: its own forward, the children
    holding lent parameters, and its other forward* methods."""
    layer._kt_lend_idx = idx
    layer.register_forward_pre_hook(_enter_hook)
    for cname, child in layer.named_children():
        if any(n.startswith(cname + ".") for n in lent):
            child._kt_lend_idx = idx
            child.register_forward_pre_hook(_enter_hook)
    for name in dir(type(layer)):
        if name.startswith("forward") and name != "forward" and callable(getattr(type(layer), name)):
            setattr(layer, name, _entering(getattr(layer, name), idx))


def _entering(fn, idx: int):
    def wrapper(*args, **kwargs):
        _enter(idx)
        return fn(*args, **kwargs)

    return wrapper


def _lent_names(lay):
    """The parameter names a layout lends, an owner's inner ones included."""
    names = []
    for item in lay:
        if item[0] == "param":
            names.append(item[1])
        else:
            names += [rec[0] for rec in item[7]]
    return names


def _owners_after_load(layer, lay):
    """Plain CUDA tensor attributes in the layer that hold at least one lent
    parameter: [(module, attr, tensor, [(name, delta)]...)]."""
    params = dict(layer.named_parameters())
    lent = set(_lent_names(lay))
    found = []
    for mod in layer.modules():
        for attr, val in list(vars(mod).items()):
            if attr.startswith("_parameters") or attr.startswith("_buffers"):
                continue
            if not torch.is_tensor(val) or isinstance(val, torch.nn.Parameter) or not val.is_cuda:
                continue
            if not val.is_contiguous() or _nbytes(val) < MIN_BYTES:
                continue
            lo, hi = val.data_ptr(), val.data_ptr() + _nbytes(val)
            inner = [(n, p.data_ptr() - lo) for n, p in params.items()
                     if p.is_cuda and lo <= p.data_ptr() < hi and p.is_contiguous()]
            if any(n in lent for n, _ in inner):
                found.append((mod, attr, val, inner))
    return found


def _relayout(i: int, counts) -> None:
    """Lay layer i out again from what the loading left: fused owners
    first, then the lent parameters as they are now, through pinned host
    memory back into the same span when it fits."""
    device = STATE["device"]
    layer = STATE["layers"][i]
    lay = STATE["layouts"][i]
    full = STATE["dev_full"][i]
    lo, hi = full.data_ptr(), full.data_ptr() + full.numel()
    found = _owners_after_load(layer, lay)
    inner_names = {n for _, _, _, inner in found for n, _ in inner}
    new, off = [], 0
    for mod, attr, val, inner in found:
        recs = [(n, tuple(layer.get_parameter(n).shape), layer.get_parameter(n).dtype, d,
                 _nbytes(layer.get_parameter(n))) for n, d in inner]
        new.append(("owner", mod, attr, tuple(val.shape), val.dtype, off, _nbytes(val), recs))
        off = _align(off + _nbytes(val))
        counts["owners"] += 1
    mod = val = inner = None  # the owners' originals go when the layer is pointed away
    params = dict(layer.named_parameters())
    # a layer laid out again during the load has owner items: their inner
    # parameters are what it lends (taking item[1], the owner's module, as a
    # name dropped the owner and overwrote it in place, 10-05)
    names = _lent_names(lay)
    # what the loading made anew (a repack under another name) and is big
    # enough is lent too
    names += [n for n, q in params.items() if n not in names and _ready(q) and q.is_cuda and q.is_contiguous()
              and _nbytes(q) >= MIN_BYTES and not any(w in n for w in SKIP)
              and (STATE.get("final") or not any(w in n for w in PFONLY))]
    # prefill-only weights only in after_load's final pass: a layer laid out
    # again during the load (after_module) keeps them in its span, or the
    # final pass would find them emptied
    pfonly = PFONLY if STATE.get("final") else []
    names = ([n for n in names if not any(w in n for w in IDLE + pfonly)]
             + [n for n in names if any(w in n for w in IDLE)]
             + [n for n in names if any(w in n for w in pfonly)])
    active = pfo = None
    for name in names:
        q = params.get(name)
        if active is None and any(w in name for w in IDLE):
            active = off
        if pfo is None and any(w in name for w in pfonly):
            pfo = off
        if name in inner_names or q is None or not _ready(q) or not q.is_cuda or not q.is_contiguous():
            if q is None:
                # a second name for a parameter registered under another one
                # (alias_or_bind_derived_param: Flash-Next's *_blockscale_swizzled
                # is w*_weight_scale), which named_parameters() lists once
                try:
                    alias = any(layer.get_parameter(name) is v for v in params.values())
                except AttributeError:
                    alias = False
                counts["aliases" if alias else "dropped"] += 1
                if not alias:
                    counts.setdefault("gone_names", set()).add(name)
            continue
        if not lo <= q.data_ptr() < hi:
            counts["moved"] += 1
        new.append(("param", name, tuple(q.shape), q.dtype, off, _nbytes(q)))
        off = _align(off + _nbytes(q))
    params = q = None
    span = _align(off)
    STATE.setdefault("active", {})[i] = span if active is None else active
    dec = span if pfo is None else pfo
    STATE.setdefault("dec_end", {})[i] = dec
    # prefill-only bytes the current span holds: only those come back by making
    # a smaller one (bind_model leaves them out, so a layer the load did not lay
    # out again holds none, and its span differs from dec by the bind's slack)
    pf_held = sum(item[5] for item in new if item[0] == "param" and item[4] >= dec
                  and lo <= layer.get_parameter(item[1]).data_ptr() < hi)
    # through pinned host memory, back into the same span (the bind left
    # room for the small parameters an owner may take in): freeing a span
    # and making a slightly larger one cut a new segment per layer on one
    # Flash-Next rank, and the card ran out after ~20 layers
    (host, reused), src = _pinned_bounce(span), None
    for item in new:
        src = getattr(item[1], item[2]) if item[0] == "owner" else layer.get_parameter(item[1])
        _place(host, item).copy_(src)
    _point(layer, host, new)
    STATE["dev_spans"][i] = None
    del found, src
    # the decode span ends before the prefill-only weights; one that holds 1 MiB
    # or more of them is made anew so the memory goes back. Not for the bind's
    # slack: a span freed and made again does not go back to the driver (the
    # pool keeps the segment), and 48 layers of that ran a Flash-Next rank out
    # of memory with KT_PREFILL_LEND_PFONLY (10-15)
    held = full.numel()
    shrink = dec < span and pf_held >= (1 << 20)
    in_place = dec <= held and not shrink
    if in_place:
        dspan = full[:dec]
    else:
        STATE["dev_full"][i] = None
        del full
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        with _tms().region(tag=TAG_WEIGHTS):
            dspan = torch.empty(dec, dtype=torch.uint8, device=device)
        STATE["dev_full"][i] = dspan
        counts["grown"] += 1
    if DEBUG:
        _log(f"relayout layer {i} on {device}: span {STATE['spans'][i]} -> {span} bytes (decode {dec}, "
             f"held {held}, prefill-only in it {pf_held}) "
             f"({'in place' if in_place else 'new span'}), free {torch.cuda.mem_get_info(device)[0] >> 20} MiB, "
             f"torch reserved {torch.cuda.memory_reserved(device) >> 20} MiB")
    dspan.copy_(host[:dec])
    _point(layer, dspan, new, limit=dec)
    if dec < span:
        _pfonly_empty(layer, new, dec)
        # the mirror is built from this copy (the span no longer holds it all)
        STATE.setdefault("relayout_host", {})[i] = host.clone() if reused else host
    STATE["dev_spans"][i] = dspan
    STATE["layouts"][i] = new
    STATE["spans"][i] = span
    del host


def after_module(module) -> None:
    """The loader, after each module's weight processing: when that left a
    lent layer's parameters outside its span (a repack makes new tensors),
    take them back now, so the originals in the span and the repacked
    copies do not pile up over the whole load (Vision-Exp with 12 Marlin hot
    experts per layer ran out of memory that way)."""
    if not ENABLED or STATE["phase"] != "load" or "dev_full" not in STATE:
        return
    i = getattr(module, "_kt_lend_layer", None)
    if i is None:
        return
    full = STATE["dev_full"][i]
    lo, hi = full.data_ptr(), full.data_ptr() + full.numel()
    if all(lo <= q.data_ptr() < hi for n, q in STATE["layers"][i].named_parameters()
           if _ready(q) and q.is_cuda and _nbytes(q) >= MIN_BYTES and not any(w in n for w in SKIP + PFONLY)):
        return
    _relayout(i, dict(owners=0, grown=0, moved=0, dropped=0, aliases=0))
    STATE["counts"]["early"] += 1


def after_load() -> None:
    """After every module's weight processing, before the KV pool: take back
    what the loading rebound, move fused owners into the spans, mirror,
    release scratch."""
    if not ENABLED or "dev_spans" not in STATE or STATE["phase"] != "load":
        return
    counts = STATE["counts"]
    device = STATE["device"]
    if DEBUG:
        _log(f"after_load on {device} (current {torch.cuda.current_device()}): free "
             f"{torch.cuda.mem_get_info(device)[0] >> 20} MiB, torch reserved {torch.cuda.memory_reserved(device) >> 20} MiB")
    STATE["final"] = True
    for i in range(len(STATE["layers"])):
        _relayout(i, counts)
    bases, total = [], 0
    for span in STATE["spans"]:
        bases.append(total)
        total += span
    host = _pinned_empty(total)
    kept = STATE.pop("relayout_host", {})
    for i, (base, span, dspan) in enumerate(zip(bases, STATE["spans"], STATE["dev_spans"])):
        host[base:base + span].copy_(kept[i] if i in kept else dspan)
    del kept
    if PFONLY:
        dec_total = sum(d.numel() for d in STATE["dev_spans"])
        _log(f"prefill-only weights ({','.join(PFONLY)}): {(total - dec_total) / 1e9:.3f} GB kept out of the "
             f"decode spans ({dec_total / 1e9:.3f} GB resident)")
    STATE.update(host=host, bases=bases, nbytes=total,
                 dec_nbytes=sum(d.numel() for d in STATE["dev_spans"]))
    maxspan = max(STATE["spans"])
    if STATE["slots"][0].numel() < maxspan:
        STATE["slots"] = None
        STATE["slots"] = [scratch_empty((maxspan,), torch.uint8, device) for _ in range(2)]
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    _tms().pause(TAG_SCRATCH)
    STATE["phase"] = "decode"
    _log(f"weights {total / 1e9:.3f} GB mirrored ({counts['owners']} fused owners moved in, {counts['grown']} spans "
         f"outgrown, {counts['moved']} parameters rebound or made by the loading taken in, {counts['aliases']} aliases, "
         f"{counts['dropped']} gone, "
         f"{counts['early']} layers taken back during the load"
         + (f"; gone: {', '.join(sorted(counts['gone_names'])[:6])}" if counts.get("gone_names") else "")
         + f"); scratch {STATE['scratch_bytes'] / 1e9:.3f} GB released "
         f"until the first long prefill")


# -- phase swap ------------------------------------------------------------------

def _threshold() -> int:
    """Rows from which a prefill does not need the resident MoE weights: the
    KT streamer's KT_GPU_STREAM_PREFILL, or in a tree without the streamer
    (GLM-5.3's) the full-GPU prefill's --kt-gpu-prefill-token-threshold."""
    if "threshold" not in STATE:
        try:
            from sglang.srt.layers.moe.kt_exl3_stream_prefill import THRESHOLD as _exl3_t
        except ImportError:
            _exl3_t = 0
        if _exl3_t > 0:  # the EXL3 streamed prefill (kt_exl3_stream_prefill.py)
            STATE["threshold"] = _exl3_t
            return _exl3_t
        try:
            from sglang.srt.layers.moe.kt_stream_prefill import STREAM_THRESHOLD as t
        except ImportError:
            try:
                from sglang.srt.server_args import get_global_server_args

                t = int(getattr(get_global_server_args(), "kt_gpu_prefill_token_threshold", 0) or 0)
            except Exception:
                t = 0
        STATE["threshold"] = t
    return STATE["threshold"]


def _streamed(forward_batch) -> bool:
    """A forward that runs in the lend window: a streamed prefill, or an extend
    over a long sequence (KT_PREFILL_LEND_LONG_SEQ)."""
    t = _threshold()
    if t <= 0 or not forward_batch.forward_mode.is_extend():
        return False
    if forward_batch.input_ids.shape[0] >= t:
        return True
    return LONG_SEQ > 0 and not IDLE and _seq_len(forward_batch) >= LONG_SEQ


def _chunk() -> int:
    try:
        from sglang.srt.server_args import get_global_server_args

        return int(get_global_server_args().chunked_prefill_size or 0)
    except Exception:
        return 0


def _seq_len(forward_batch) -> int:
    """The longest sequence in a forward, prompt so far included (CPU copy only)."""
    lens = getattr(forward_batch, "seq_lens_cpu", None)
    try:
        return int(lens.max()) if lens is not None and len(lens) else 0
    except Exception:
        return 0


# -- per-forward records (kt_prefill_plan's cost model) ------------------------------
# Every forward of a prefill window: rows, sequence end, the window's free memory
# at its start and the KV pages lent to it, and the peaks above the window's
# start (allocated, and reserved: what has to fit in free + loan). Appended to
# <KT_PREFILL_LEND_HINT>-rank<r>.windows.jsonl; tools/kt-prefill-plan-constants.py
# fits reserved = fixed + a * rows + b * sequence over them.


def _rank() -> int:
    if "rank" not in STATE:
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            STATE["rank"] = get_tensor_model_parallel_rank()
        except Exception:
            STATE["rank"] = STATE["device"].index
    return STATE["rank"]


def _fw_open(rows: int, seq: int) -> None:
    dev = STATE["device"]
    torch.cuda.reset_peak_memory_stats(dev)
    STATE["fw"] = {"rows": int(rows), "seq": int(seq)}


def _fw_close() -> None:
    fw = STATE.pop("fw", None)
    if fw is None:
        return
    dev = STATE["device"]
    resv = torch.cuda.max_memory_reserved(dev)
    alloc = torch.cuda.max_memory_allocated(dev)
    STATE["win_resv_peak"] = max(STATE.get("win_resv_peak", 0), resv)
    STATE["win_alloc_peak"] = max(STATE.get("win_alloc_peak", 0), alloc)
    if not HINT:
        return
    loan = 0
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend

        loan = int(kt_kv_prefill_lend.STATE.get("bytes", 0) or 0)
    except Exception:
        pass
    rec = dict(fw, free=int(STATE["pf_free"]), loan=loan, resv=int(resv - STATE["pf_base"]),
               alloc=int(alloc - STATE.get("pf_abase", 0)), chunk=_chunk(), window=STATE.get("to_prefill", 0))
    try:
        import json

        with open(f"{HINT}-rank{_rank()}.windows.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _write_hint(rank: int) -> None:
    import json

    rows = STATE.get("pf_rows", 0)
    chunk = launch_chunk = _chunk()
    # under the prefill plan the launch chunk is its cap; the lend's own chunk is
    # the one this hint is about (with the cap here, a 12,288-row window kept the
    # cap on record and GLM's next launch took 32,768 as its own chunk, 10-16)
    planned = os.environ.get("KT_PREFILL_PLAN", "0") == "1"
    if planned:
        chunk = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0) or chunk
    if not HINT or rows < 2048 or chunk <= 0:
        return
    growth = STATE["pf_peak"] - STATE["pf_base"]
    # the chunk-sized scratch is sized for the launch chunk (the plan's cap): spread
    # over the lend's own chunk it made a row 1.9x as costly on Flash-Next (558
    # against 295 KB at 18,432 rows, 10-17), and the next launch's plan began with
    # a first-window room below zero
    per_tok = growth / rows + STATE["scratch_chunk_bytes"] / launch_chunk
    room = STATE["pf_free"] - growth - MARGIN
    hint = chunk + int(room / per_tok) if per_tok > 0 else chunk
    # the cost per token grows with the rows (Flash-Next 371 -> 417 KB from
    # 14336 to 16384 rows), so a far extrapolation is too hopeful: at most
    # twice the running chunk per launch, the next launch measures again
    hint = min(hint, 2 * chunk)
    # a window that never filled a chunk says nothing about a larger one (a
    # 6K-token window on Vision-Exp hinted 22528 from chunk 16384; the full
    # 16384 rows then left no room): it may only lower the chunk
    keep_min = True
    if planned and rows > chunk:
        # a window wider than the lend's own chunk ran on the plan's reading of
        # free memory and the KV pages lent to it, which the room here leaves
        # out: it says nothing about the lend's chunk (10-17 Flash-Next: one
        # 35,000-row window peaked 9.22 GB with 6.87 GB free and took the chunk
        # from 6,144 to 2,048)
        hint = min(chunk, STATE.get("hint_min", chunk))
        keep_min = False
    elif rows < 4096:
        # too few rows to say anything about the chunk; still a point below
        hint = min(chunk, STATE.get("hint_min", chunk))
        keep_min = False
    elif rows < chunk and hint >= chunk:
        # nothing learned about a larger chunk: keep the running one on record
        # (a launcher needs a value: MiMo's SWA pool held every window of the
        # calibration at 12.4K rows of 16384, and without a file it fell back
        # to 2048), but out of hint_min, which would pin it there for good
        # (10-06: Qwen3.5 TP=1 stayed at 23552 behind one 5.5K-token window)
        hint = min(chunk, STATE.get("hint_min", chunk))
        keep_min = False
    # 1024 steps: whole pages for every model here (pages of 1 to 256), and 2048
    # steps threw away up to 2047 rows (V4.1: 7,462 measured, 6,144 hinted)
    hint = max(2048, min(32768, hint // 1024 * 1024))
    # the free memory at a window's start wobbles: keep the smallest
    hint = min(hint, STATE.get("hint_min", hint))
    if keep_min:
        STATE["hint_min"] = hint
    # allocated bytes a row, which unlike the reserved peak does not grow with
    # the free memory (kt_prefill_plan's per-row cost; the largest of the launch)
    alloc_row = int(((STATE.get("win_alloc_peak") or torch.cuda.max_memory_allocated(STATE["device"]))
                     - STATE.get("pf_abase", 0)) / rows)
    STATE["alloc_row_max"] = max(STATE.get("alloc_row_max", 0), alloc_row)
    rec = {"chunk": hint, "current_chunk": chunk, "rows_seen": rows, "alloc_per_row": STATE["alloc_row_max"],
           # free at a window's start minus free in decode before it: the plan's step
           # from the live decode free memory, known to the next launch's first request
           "pf_delta_mb": STATE.get("pf_delta", 0) >> 20,
           "free_at_prefill_mb": STATE["pf_free"] >> 20, "peak_growth_mb": growth >> 20,
           "margin_mb": MARGIN >> 20, "bytes_per_token": round(per_tok),
           "scratch_chunk_mb": STATE["scratch_chunk_bytes"] >> 20,
           # the longest sequence of the window, and the full KV pool the server
           # got (kt_lend_swa_ratio sizes an SWA pool from it, not from the cap)
           "seq_seen": STATE.get("pf_seq", 0), "full_tokens": STATE.get("full_tokens", 0)}
    # the window as a point of the cost per row against the sequence length,
    # merged with the points earlier launches of this setup left (one per
    # power-of-two sequence bucket, the costlier kept): kt_lend_table builds
    # the prefix -> chunk table of KT_PREFILL_LEND_DYNAMIC from them
    path = f"{HINT}-rank{rank}.json"
    points = STATE.get("points")
    if points is None:
        try:
            with open(path) as f:
                points = {int(k): v for k, v in json.load(f).get("points", {}).items()}
        except Exception:
            points = {}
    seq = STATE.get("pf_seq", 0)
    # a window of 2048 rows or more says what a row costs at this length (with
    # the dynamic chunk a long prompt's windows are narrower than the launch
    # chunk, and they are exactly the points the table lacks)
    if seq > 0 and rows >= 2048:
        bucket = seq.bit_length()
        old = points.get(bucket)
        if old is None or per_tok > old["per_row"]:
            points[bucket] = {"seq": seq, "per_row": round(per_tok), "free": STATE["pf_free"],
                              "rows": rows}
    STATE["points"] = points
    rec["points"] = points
    with open(path, "w") as f:
        json.dump(rec, f)
    if STATE.get("hint_logged") != hint:
        STATE["hint_logged"] = hint
        _log(f"chunk hint {hint} (now {chunk}): {rows} rows peaked {growth / 1e9:.2f} GB above the "
             f"prefill baseline with {STATE['pf_free'] / 1e9:.2f} GB free, {per_tok / 1e3:.0f} KB a token "
             f"with the chunk-sized scratch, margin {MARGIN / 1e9:.2f} GB")


def table_chunk(prefix: int, table: Optional[dict] = None) -> Optional[int]:
    """The chunk for a request whose prompt has reached prefix tokens, from
    KT_PREFILL_LEND_TABLE (built at launch by tools/kt-lend-auto.sh from the
    points of every rank, so all ranks answer the same): the largest c with
    c * cost(prefix + c) <= free - margin, cost per row interpolated between
    the measured sequence lengths and, past the last one, taken to grow with
    the sequence. None when there is no table."""
    if table is None:
        table = STATE.get("table")
        if table is None:
            import json

            raw = os.environ.get("KT_PREFILL_LEND_TABLE", "")
            try:
                table = json.loads(raw) if raw else {}
            except Exception:
                table = {}
            STATE["table"] = table
    pts = sorted((p["seq"], p["per_row"]) for p in table.get("points", []))
    if not pts:
        return None
    room = table["free"] - table.get("margin", MARGIN)
    top, step = int(table["max"]), int(table.get("step", 1024))

    def cost(s):
        if s <= pts[0][0]:
            return pts[0][1]
        for (s0, c0), (s1, c1) in zip(pts, pts[1:]):
            if s <= s1:
                return c0 + (c1 - c0) * (s - s0) / max(1, s1 - s0)
        # past the longest measured sequence: the cost per row is assumed to
        # grow with the sequence (V4.1 doubled from 510K to 989K while it was
        # flat from 38K to 510K), or along the last slope if that is steeper;
        # a longer prompt adds its own point and loosens this
        s1, c1 = pts[-1]
        grown = c1 * s / s1
        if len(pts) == 1:
            return grown
        s0, c0 = pts[-2]
        return max(grown, c1 + max(0.0, (c1 - c0) / max(1, s1 - s0)) * (s - s1))

    c = top
    while c > 2048 and c * cost(prefix + c) > room:
        c -= step
    return max(2048, c)  # no narrower than the chunk without the lend


def before_forward(forward_batch) -> None:
    """Model runner, before every forward, outside any graph."""
    if not ENABLED or STATE["phase"] == "load":
        return
    streamed = _streamed(forward_batch)
    rows = forward_batch.input_ids.shape[0]
    # SWA pages handed back since the last forward are mapped before it runs; a
    # window that opens maps them after the weights have left (KT_PREFILL_SWA_LEND)
    m = _kv_mod()
    if m is not None and not (streamed and STATE["phase"] == "decode") \
            and any(getattr(m, "SWA", {}).get("pending") or ()):
        m.swa_flush("forward")
    if streamed and STATE["phase"] == "prefill":
        _fw_close()
        STATE["pf_rows"] = max(STATE.get("pf_rows", 0), rows)
        seq = _seq_len(forward_batch)
        STATE["pf_seq"] = max(STATE.get("pf_seq", 0), seq)
        if LONG_SEQ > 0 and seq >= LONG_SEQ:
            # Past a long prefix every chunk's prefix-sized buffers (page tables,
            # the indexer's gathered K) are a little larger than the last one's,
            # so the freed ones do not fit them and torch's cache fills with
            # pieces: at a 999K-token prefix 5,624 MiB reserved for 2,927 MiB
            # allocated (10-07). Hand the unused segments back before each chunk.
            torch.cuda.empty_cache()
        _fw_open(rows, seq)
    elif streamed and STATE["phase"] == "decode":
        t = time.perf_counter()
        dev = STATE["device"]
        # the hot experts in the KV pool's free pages (KT_KV_HOT_LEND) give them
        # back before anything is measured
        if m is not None and hasattr(m, "hot_release"):
            m.hot_release("window start")
        # free memory in decode, the KV pool and the graphs made: what is gone
        # from it when the weights come back was left behind by this prefill
        STATE["dec_free"] = torch.cuda.mem_get_info(dev)[0]
        STATE.setdefault("dec_free0", STATE["dec_free"])
        if DEBUG and not STATE.get("history"):
            torch.cuda.memory._record_memory_history(max_entries=500000)
            STATE["history"] = True
        _tms().pause(TAG_WEIGHTS)
        free_a = torch.cuda.mem_get_info(dev)[0]
        _tms().resume(TAG_SCRATCH)
        # what the scratch maps at a window's start (end_window compares)
        STATE["scratch_at_start"] = max(0, free_a - torch.cuda.mem_get_info(dev)[0])
        torch.cuda.synchronize(dev)
        # what decode freed (the last prefill's attention metadata, released at
        # the first graph replay) sits in torch's cache, which mem_get_info
        # counts as used: hand it back so the free memory measured is real
        torch.cuda.empty_cache()
        STATE["pf_free"] = torch.cuda.mem_get_info(dev)[0]
        STATE["pf_base"] = torch.cuda.memory_reserved(dev)
        if DEBUG:
            # what the device holds outside torch (graphs, modules, NCCL) is
            # total - free - reserved; a rise across windows is not torch's
            total = torch.cuda.mem_get_info(dev)[1]
            _log(f"prefill window {STATE['to_prefill'] + 1} start on {dev}: free {STATE['pf_free'] >> 20} MiB, "
                 f"torch reserved {STATE['pf_base'] >> 20} MiB, allocated "
                 f"{torch.cuda.memory_allocated(dev) >> 20} MiB, outside torch "
                 f"{(total - STATE['pf_free'] - STATE['pf_base']) >> 20} MiB")
        torch.cuda.reset_peak_memory_stats(dev)
        STATE["pf_outside"] = torch.cuda.mem_get_info(dev)[1] - STATE["pf_free"] - STATE["pf_base"]
        STATE["pf_abase"] = torch.cuda.memory_allocated(dev)
        if "full_tokens" not in STATE:
            # the full KV pool the server got, from the ModelRunner that calls
            # this (no tree here puts the pool on the batch, and the attention
            # backends differ): max_total_num_tokens after memory profiling
            try:
                import inspect

                runner = inspect.currentframe().f_back.f_locals.get("self")
                STATE["full_tokens"] = int(getattr(runner, "max_total_num_tokens", 0) or 0)
            except Exception:
                STATE["full_tokens"] = 0
        if SNAPSHOT and not STATE.get("snap_done") and rows >= SNAPSHOT_ROWS:
            torch.cuda.memory._record_memory_history(enabled="all", context="all", stacks="python",
                                                     max_entries=4_000_000)
            STATE["snap_on"] = True
        STATE.update(phase="prefill", pf_issued=set(), pf_rows=rows, pf_seq=_seq_len(forward_batch))
        # free at a window's start minus free in decode just before it (weights out,
        # scratch in): kt_prefill_plan adds it to the live decode free memory
        STATE["pf_delta"] = STATE["pf_free"] - STATE["dec_free"]
        STATE["win_resv_peak"] = STATE["win_alloc_peak"] = 0
        _fw_open(rows, STATE["pf_seq"])
        STATE["to_prefill"] += 1
        _kv_loan(forward_batch)
        if m is not None and hasattr(m, "swa_flush"):
            m.swa_flush("window start")
        if STATE["to_prefill"] in (1, 10, 100, 1000):
            _log(f"to prefill in {(time.perf_counter() - t) * 1e3:.1f} ms ({STATE['to_prefill']} so far)")
    elif not streamed and STATE["phase"] == "prefill":
        end_window()
    elif STATE["phase"] == "decode":
        m = _kv_mod()
        if m is not None and getattr(m, "SWA_ENABLED", False):
            m.swa_tick()
        if m is not None and getattr(m, "HOT", False):
            # the KV pool's free pages to more hot experts (kt_kv_hot_lend)
            import inspect

            from sglang.srt.layers.moe import kt_kv_hot_lend

            kt_kv_hot_lend.tick(inspect.currentframe().f_back.f_locals.get("self"))


def _kv_mod():
    """kt_kv_prefill_lend when kv-vmm is installed in this tree, else None."""
    if "kv_mod" not in STATE:
        try:
            from sglang.srt.layers.moe import kt_kv_prefill_lend
        except ImportError:
            kt_kv_prefill_lend = None
        STATE["kv_mod"] = kt_kv_prefill_lend
    return STATE["kv_mod"]


def _kv_loan(forward_batch) -> None:
    """A window just opened: the KV pool's free pages lend their memory too, when
    kv-vmm is installed (kt_kv_prefill_lend; it decides whether the window needs them)."""
    try:
        from sglang.srt.layers.moe import kt_kv_prefill_lend
    except ImportError:
        return
    if not kt_kv_prefill_lend.ENABLED:
        return
    import inspect

    # the ModelRunner that calls before_forward (as for full_tokens above)
    frame = inspect.currentframe().f_back.f_back
    runner = frame.f_locals.get("self") if frame is not None else None
    kt_kv_prefill_lend.lend(runner, forward_batch)


def end_window() -> None:
    """Close a prefill window: record it, bring the weights back. The model
    runner does this before the first forward after a prefill; the probe
    (kt_lend_probe.py) calls it after each of its forwards."""
    if not ENABLED or STATE["phase"] != "prefill":
        return
    if True:
        t = time.perf_counter()
        dev = STATE["device"]
        _fw_close()
        STATE["pf_peak"] = STATE.get("win_resv_peak") or torch.cuda.max_memory_reserved(dev)
        if STATE.get("snap_on"):
            path = f"{SNAPSHOT}-{dev.index}.pickle"
            torch.cuda.memory._dump_snapshot(path)
            torch.cuda.memory._record_memory_history(enabled=None)
            STATE.update(snap_on=False, snap_done=True)
            _log(f"allocation history of window {STATE['to_prefill']} ({STATE.get('pf_rows', 0)} rows) in {path}")
        if DEBUG:
            # one line a window for fitting the growth to rows and sequence
            # length; "outside torch" is how much more the device held outside
            # torch's allocator at the end than at the start (NCCL, cuBLAS,
            # graphs; a transient that is gone by the end does not show)
            free_now, total = torch.cuda.mem_get_info(dev)
            outside = total - free_now - torch.cuda.memory_reserved(dev)
            _log(f"prefill window {STATE['to_prefill']} on {dev}: rows {STATE.get('pf_rows', 0)}, "
                 f"seq {STATE.get('pf_seq', 0)}, peak growth "
                 f"{(STATE['pf_peak'] - STATE['pf_base']) >> 20} MiB of {STATE['pf_free'] >> 20} MiB free "
                 f"(allocated {(torch.cuda.max_memory_allocated(dev) - STATE.get('pf_abase', 0)) >> 20} MiB), "
                 f"outside torch {(outside - STATE['pf_outside']) >> 20:+d} MiB, full pool "
                 f"{STATE.get('full_tokens', 0)}")
        try:
            from sglang.srt.distributed import get_tensor_model_parallel_rank

            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = dev.index
        _leave()
        torch.cuda.empty_cache()
        free_c = torch.cuda.mem_get_info(dev)[0]
        _tms().pause(TAG_SCRATCH)
        grew = max(0, torch.cuda.mem_get_info(dev)[0] - free_c) - STATE.get("scratch_at_start", 0)
        if grew > (16 << 20) and STATE.get("pf_delta") is not None:
            # the scratch keeps what this window first made in it (GLM-5.3's 2.11 GiB
            # prefill slot, a lent CUTLASS workspace): the next window starts with that
            # much less free, and it is no per-row cost of later windows. The step
            # measured at this window's start did not know it (a planned first 35K ran
            # out of memory on it, 10-16), and the hint must not count it as growth
            STATE["pf_delta"] -= grew
            STATE["pf_free"] -= grew
            STATE["pf_peak"] -= grew
            if STATE.get("win_alloc_peak"):
                STATE["win_alloc_peak"] -= grew
            _log(f"the scratch kept {grew >> 20} MiB made in window {STATE['to_prefill']}: the decode -> "
                 f"window step is {STATE['pf_delta'] >> 20} MiB from now on")
        _write_hint(rank)
        # the free KV pages lent to this window first (kv-vmm); the sliding-window
        # pool's free pages leave again (KT_PREFILL_SWA_LEND)
        kt_kv_prefill_lend = _kv_mod()
        if kt_kv_prefill_lend is not None:
            kt_kv_prefill_lend.restore()
            if getattr(kt_kv_prefill_lend, "SWA_ENABLED", False):
                kt_kv_prefill_lend.swa_lend("end of window")
        _check_room(dev)
        _tms().resume(TAG_WEIGHTS)
        for base, span, dspan in zip(STATE["bases"], STATE["spans"], STATE["dev_spans"]):
            dspan.copy_(STATE["host"][base:base + dspan.numel()], non_blocking=True)
        STATE["phase"] = "decode"
        STATE["to_decode"] += 1
        if STATE["to_decode"] in (1, 10, 100, 1000):
            _log(f"to decode in {(time.perf_counter() - t) * 1e3:.1f} ms, copy queued "
                 f"({STATE['to_decode']} so far)")


def _check_room(dev) -> None:
    """Before the weights come back: what the prefill left outside the lend
    scratch is room they no longer have. Say how much, and stop with a reason
    instead of torch_memory_saver failing to map them (a process abort)."""
    free = torch.cuda.mem_get_info(dev)[0]
    back = STATE.get("dec_nbytes", STATE["nbytes"])  # the decode spans (prefill-only weights stay out)
    need = back + len(STATE["spans"]) * (2 << 20)  # mapped in 2 MiB granules
    left = STATE["dec_free"] + back - free  # kept from this prefill window
    total = STATE["dec_free0"] + back - free  # since the first window
    if DEBUG and (left > 256 << 20 or total > 384 << 20) and not STATE.get("dumped"):
        path = f"/tmp/kt-lend-kept-{dev.index}-w{STATE['to_decode'] + 1}.pickle"
        torch.cuda.memory._dump_snapshot(path)
        STATE["dumped"] = True
        _log(f"memory snapshot {path}")
    # DEBUG: every window, so a slow pile-up under 64 MB a window shows too
    if left > 64 << 20 or DEBUG:
        _log(f"prefill window {STATE['to_decode'] + 1} on {dev} kept {left / 1e9:.2f} GB outside the lend "
             f"scratch ({total / 1e9:.2f} GB since the first; the weights need {need / 1e9:.2f} GB, "
             f"{free / 1e9:.2f} GB free)")
    if free < need:
        raise RuntimeError(
            f"KT_PREFILL_LEND: the weights ({need / 1e9:.2f} GB) cannot come back on {dev}: "
            f"{free / 1e9:.2f} GB free, {left / 1e9:.2f} GB was allocated during prefill outside the "
            f"lend scratch and kept. Fewer GPU experts or a smaller KV pool leaves the room.")


# -- per-layer streaming ---------------------------------------------------------

def _prefetch(i: int) -> None:
    if i >= len(STATE["spans"]) or i in STATE["pf_issued"]:
        return
    k = i % 2
    base, span = STATE["bases"][i], STATE["spans"][i]
    cs = STATE["pf_stream"]
    with torch.cuda.stream(cs):
        if STATE["pf_done"][k] is not None:
            cs.wait_event(STATE["pf_done"][k])
        n = STATE.get("active", {}).get(i, span)  # the idle tail stays out
        STATE["slots"][k][:n].copy_(STATE["host"][base:base + n], non_blocking=True)
        STATE["pf_ready"][k].record(cs)
    STATE["pf_issued"].add(i)


def _enter_hook(module, args):
    _enter(module._kt_lend_idx)
    return None


def _enter(i: int) -> None:
    """Layer i is about to run in a prefill: the previous layer goes back to
    its span (its slot is free once the compute stream gets here), layer i
    to its slot, layer i+1's copy starts."""
    if STATE["phase"] != "prefill" or STATE.get("pf_cur") == i:
        return
    dev = STATE["device"]
    if "pf_stream" not in STATE:
        STATE["pf_stream"] = torch.cuda.Stream(device=dev)
        STATE["pf_ready"] = [torch.cuda.Event(), torch.cuda.Event()]
        STATE["pf_done"] = [None, None]
    cur = torch.cuda.current_stream(dev)
    prev = STATE.get("pf_cur")
    if prev is not None:
        _point_back(prev)
        ev = torch.cuda.Event()
        ev.record(cur)
        STATE["pf_done"][prev % 2] = ev
    if prev is None or i <= prev:
        STATE["pf_issued"] = set()  # a new forward
    if i not in STATE["pf_issued"]:
        # nothing prefetched: the side stream must see the compute stream's past
        STATE["pf_stream"].wait_stream(cur)
        _prefetch(i)
    cur.wait_event(STATE["pf_ready"][i % 2])
    _point(STATE["layers"][i], STATE["slots"][i % 2], STATE["layouts"][i],
           limit=STATE.get("active", {}).get(i))
    STATE["pf_cur"] = i
    _prefetch(i + 1)


def _leave() -> None:
    i = STATE.pop("pf_cur", None)
    if i is not None:
        _point_back(i)
