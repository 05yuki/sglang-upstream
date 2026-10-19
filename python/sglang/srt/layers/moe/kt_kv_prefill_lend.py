"""The KV pool's free pages lend their physical memory to a streamed prefill
(KT_PREFILL_KV_LEND=1, with KT_KV_VMM=1 and the prefill lend KT_PREFILL_LEND=1).

The KV always wins:
- at the start of a prefill window (kt_lend.before_forward, after the weights
  leave) the free pages are split: the lowest-numbered ones that the request
  still needs for the rest of its prompt, plus KT_PREFILL_KV_LEND_RESERVE
  tokens, stay in the free list; the others leave it for the window, and every
  2 MiB chunk of a K/V tensor that holds only such pages is unmapped
  (kt_kv_vmm.release), so the driver, and torch, have that memory for the
  prefill's activations;
- at the end of the window (kt_lend.end_window, before the weights come back)
  the chunks are mapped again, zeroed, and the pages rejoin the free list;
- the pool's size never changes: available_size counts the lent pages, and an
  alloc that needs more pages than the free list holds maps everything back
  first (a device sync, then kt_kv_vmm.restore), so nothing the scheduler
  admitted can fail; that path is logged, it means the request's remaining
  prompt was not known.

The request's prompt length comes from the scheduler (note_request, by rid);
the window's forward batch carries the rids and the sequence lengths.

Outside prefill the same free pages can lend their memory to more hot experts
(KT_KV_HOT_LEND=1, kt_kv_hot_lend.py): decode_lend keeps KT_KV_HOT_RESERVE
tokens free and unmaps the rest the same way; restore runs the hot lend's
release (PRE_RESTORE) before it maps anything back, and a window that opens
takes everything back first (hot_release).

The sliding-window pool of a hybrid SWA model lends the other way round
(KT_PREFILL_SWA_LEND=1, with KT_KV_VMM=1): it must hold a whole prefill chunk,
but decode needs only the window. So it is sized for the largest chunk the plan
may take (KT_PREFILL_SWA_TOKENS, set in the KV cache configurator) and, outside
prefill, every free page beyond KT_PREFILL_SWA_LEND_KEEP tokens is unmapped: at
the end of pool setup, at the end of each prefill window, and in decode once
the window's frees have piled up. An allocation that needs more pages than are
mapped maps back the lowest-numbered lent ones first (swa_map), and the plan
counts what a chunk would map (swa_terms).
"""
import contextlib
import os
import time
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.moe import kt_kv_vmm

ENABLED = os.environ.get("KT_PREFILL_KV_LEND", "0") == "1" and kt_kv_vmm.ENABLED
HOT = os.environ.get("KT_KV_HOT_LEND", "0") == "1" and kt_kv_vmm.ENABLED
PRE_RESTORE = []  # fn(why), run before lent pages are mapped back (the hot lend's slots out)
# fn(why), run when a window opens before the weights are paused (the hot lend's resident
# rows back to experts 0..N-1: kv-hot-lend-rows-home.py)
WINDOW_START = []
RESERVE = int(os.environ.get("KT_PREFILL_KV_LEND_RESERVE", "4096"))  # tokens kept free beyond the prompt
CHECK = os.environ.get("KT_PREFILL_KV_LEND_CHECK", "0") == "1"
# KT_PREFILL_KV_LEND_VERIFY=1: every page in use when a window lends (but the
# window's own batch's) must read the same after it, checked by a per-page hash
VERIFY = os.environ.get("KT_PREFILL_KV_LEND_VERIFY", "0") == "1"

PROMPT: Dict[str, int] = {}  # rid -> prompt tokens
STATE = {"alloc": None, "tensors": None, "lent": None, "ranges": [], "bytes": 0, "owner": None,
         "windows": 0, "take_backs": 0, "unknown": 0, "hot_lends": 0}


def _log(msg: str) -> None:
    print(f"[kt-kv-pf-lend {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def note_request(rid: str, prompt_tokens: int) -> None:
    """The scheduler, when a request is queued."""
    if not ENABLED:
        return
    if len(PROMPT) > 4096:
        PROMPT.clear()
    PROMPT[rid] = int(prompt_tokens)


def _pool_tensors(kvp) -> list:
    """A pool's per-layer buffers, its sub-pools' included: K and V (MHA), one KV
    buffer (MLA, DeepSeek V4), and DeepSeek V4's pool per compress ratio and its
    indexer's (slot = token loc // ratio, page = page_size // ratio rows: their
    pages are the full pool's)."""
    out = []
    for name in ("k_buffer", "v_buffer", "kv_buffer", "index_k_with_scale_buffer"):
        v = getattr(kvp, name, None)
        if isinstance(v, (list, tuple)):
            out += [t for t in v if isinstance(t, torch.Tensor)]
    for name in ("kv_pools", "index_pools"):
        v = getattr(kvp, name, None)
        if isinstance(v, dict):
            for sub in v.values():
                out += _pool_tensors(sub)
    return out


def _kv_tensors(kvp, n: int, ps: int = 1, what: str = "K/V", extra=()):
    """[(flat uint8 view, bytes a page, chunk, first chunk boundary)] of a pool's
    buffers laid out by the allocator's pages: k rows a page from row 0 for the n + 1
    pages (1 for a row a page, page_size for a row a token, or as many as a compress
    ratio's or an indexer's smaller pages make one), and up to k + 1 rows of padding
    after them, on kv_vmm memory; [] when there are none. A buffer of any other
    length (another page space) is left alone. extra: (tensor, bytes a page) pairs
    known to follow the same pages from byte 0."""
    tensors, other = [(t.view(-1).view(torch.uint8), pb) for t, pb in extra], 0
    for t in _pool_tensors(kvp):
        rows = t.shape[0] if t.dim() >= 1 else 0
        k = rows // (n + 1)
        if t.is_cuda and t.is_contiguous() and k >= 1 and rows - k * (n + 1) <= k + 1:
            tensors.append((t.view(-1).view(torch.uint8), t.nbytes // rows * k))
        else:
            other += 1
    if other:
        _log(f"{other} buffers of {kvp.__class__.__name__} not in the allocator's pages: left out")
    if not tensors:
        _log(f"no {what} tensors with whole pages in {kvp.__class__.__name__}: off")
        return []
    # each tensor's own chunk (large K/V buffers use KV_VMM_CHUNK_MB, a small
    # pool's the device granularity). Chunks sit at multiples of their size (each
    # range is reserved aligned to it), but torch packs small buffers into one
    # segment, so a tensor may start inside one: its first whole chunk begins c0
    # bytes in, and only whole chunks inside the tensor are ever unmapped
    tensors = [(t, pb, kt_kv_vmm.chunk_of(t)) for t, pb in tensors]
    off = [t for t, _, c in tensors if not c]
    if off:
        # made outside the KV region (GLM-5.3's DSA indexer buffer): their pages stay
        # mapped, unused while lent; the others still lend
        _log(f"{len(off)} {what} tensors not on kv_vmm memory ({sum(t.nbytes for t in off) >> 20} MiB): "
             "left mapped")
    tensors = [(t, pb, c, (c - t.data_ptr() % c) % c) for t, pb, c in tensors if c]
    if not tensors:
        _log(f"no {what} tensors on kv_vmm memory: off")
    return tensors


def _whole_chunks(tensors, mark, n: int, dev):
    """Per tensor, the runs of chunks every page of which is marked: [(ti, off, len)]."""
    csum = torch.cat((torch.zeros(1, dtype=torch.int64, device=dev), mark.cumsum(0)))
    runs = []
    for ti, (flat, pb, C, c0) in enumerate(tensors):
        nch = max(0, flat.numel() - c0) // C
        j = torch.arange(nch, device=dev, dtype=torch.int64)
        p0 = (c0 + j * C) // pb
        p1 = (c0 + (j + 1) * C - 1) // pb  # inclusive
        # a buffer may run past page n + 1 (a compress state's padding rows):
        # those chunks are never whole
        inside = p1 <= n + 1
        p0, p1 = torch.clamp(p0, max=n + 1), torch.clamp(p1, max=n + 1)
        whole = inside & ((csum[p1 + 1] - csum[p0]) == (p1 - p0 + 1))
        idx = torch.nonzero(whole).flatten().cpu().tolist()
        i = 0
        while i < len(idx):
            k = i
            while k + 1 < len(idx) and idx[k + 1] == idx[k] + 1:
                k += 1
            runs.append((ti, c0 + idx[i] * C, (idx[k] - idx[i] + 1) * C))
            i = k + 1
    return runs


def _setup(runner) -> bool:
    if STATE["alloc"] is not None:
        return True
    if STATE.get("off"):
        return False
    STATE["off"] = True  # until set up
    a = runner.token_to_kv_pool_allocator
    # an SWA allocator (V4.1, Vision, MiMo) hands full-attention allocations to its
    # full_attn_allocator: that one's free pages are the full pool's, and wrapping
    # it covers every path the SWA allocator takes into it
    a = getattr(a, "full_attn_allocator", None) or a
    # one free list of pages (or tokens, page size 1) handed out from its front:
    # the paged allocator and the token one (Qwen3.5's, and SWA's full side); the
    # unified and sparse ones keep more than one list and are not handled yet
    if type(a).__name__ not in ("PagedTokenToKVPoolAllocator", "TokenToKVPoolAllocator") \
            or not isinstance(getattr(a, "free_pages", None), torch.Tensor):
        _log(f"allocator {type(a).__name__} has no single free list: off")
        return False
    pool = runner.token_to_kv_pool
    kvp = getattr(pool, "full_kv_pool", None) or pool
    n = int(getattr(a, "num_pages", 0) or a.size // a.page_size)
    tensors = _kv_tensors(kvp, n, a.page_size)
    if not tensors:
        return False
    chunk = max(t[2] for t in tensors)
    STATE.update(alloc=a, tensors=tensors, chunk=chunk, ps=a.page_size, n=n, off=False)
    _wrap_allocator(a)
    _log(f"{len(tensors)} K/V tensors of {kvp.__class__.__name__}, {n} pages of {a.page_size} tokens "
         f"({min(t[1] for t in tensors)}-{max(t[1] for t in tensors)} B a page per tensor), "
         f"chunks {min(t[2] for t in tensors) >> 20}-{chunk >> 20} MiB, "
         f"{sum(1 for t in tensors if t[3])} not chunk-aligned, reserve {RESERVE} tokens")
    return True


def _free_pages(a) -> torch.Tensor:
    """Every free page of the allocator: the free list, and the pages freed but not
    merged into it yet (release_pages, or the paged allocator's staged_pages)."""
    parts = [a.free_pages]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        parts.append(a.release_pages)
    parts += [p for p in getattr(a, "staged_pages", None) or [] if len(p)]
    return torch.cat(parts) if len(parts) > 1 else parts[0]


def _free_count(a) -> int:
    """len(_free_pages(a)) without the concatenation: what the allocator can hand
    out, merging what it freed when its free list runs short."""
    return (len(a.free_pages) + (len(a.release_pages) if getattr(a, "release_pages", None) is not None else 0)
            + int(getattr(a, "num_staged_pages", 0) or 0))


def _merge_free(a) -> None:
    """The freed pages into the free list, as the allocator does when it runs short:
    pages are taken out of the free list only, so none is left behind elsewhere."""
    merge = getattr(a, "merge_and_sort_free", None)
    if merge is not None:
        merge()


def _remaining(fb) -> Optional[int]:
    """Prompt tokens the window's requests have not reached yet (None: unknown)."""
    rids = getattr(fb, "rids", None)
    seqs = getattr(fb, "seq_lens_cpu", None)
    if not rids or seqs is None:
        return None
    left = 0
    for rid, s in zip(rids, seqs.tolist()):
        p = PROMPT.get(rid)
        if p is None:
            return None
        left += max(0, p - int(s))
    return left


def _runner_of(scheduler):
    w = getattr(scheduler, "tp_worker", None)
    for path in (("model_runner",), ("_model_runner",), ("worker", "model_runner"), ("worker", "_model_runner")):
        o = w
        for p in path:
            o = getattr(o, p, None)
        if o is not None:
            return o
    return None


def loan_estimate(scheduler, rem: int) -> int:
    """Bytes a window for a request with rem prompt tokens to go could borrow
    (kt_prefill_plan). Only the allocator's free page count enters it, which is
    the same on every TP rank."""
    if not ENABLED:
        return 0
    if STATE["alloc"] is None:
        runner = _runner_of(scheduler)
        if runner is None or not _setup(runner):
            return 0
    a, ps = STATE["alloc"], STATE["ps"]
    free = _free_count(a)
    if STATE["lent"] is not None:
        free += int(STATE["lent"].numel())
    keep = -(-(rem + RESERVE) // ps) + 1
    pages = max(0, free - keep)
    # only whole chunks are unmapped: a tensor loses up to one chunk at the
    # boundary of the pages it keeps
    return sum(max(0, pages * pb - 2 * c) for _, pb, c, _ in STATE["tensors"])


def _release_free(keep_pages: int):
    """Take every free page but the keep_pages lowest-numbered out of the free list
    and unmap each chunk of a K/V tensor that holds only such pages (a device sync
    first). (lent pages, mark [n + 2] of them, [(ti, off, len)] unmapped, bytes)."""
    a, n = STATE["alloc"], STATE["n"]
    _merge_free(a)
    dev = a.free_pages.device
    free = torch.sort(_free_pages(a)).values
    if free.numel() <= keep_pages:
        return None
    cand = free[keep_pages:]
    mark = torch.zeros(n + 2, dtype=torch.int32, device=dev)
    mark[cand] = 1
    torch.cuda.synchronize(dev)
    ranges, total = [], 0
    # runs of chunks, one release call each
    for ti, off, ln in _whole_chunks(STATE["tensors"], mark, n, dev):
        got = kt_kv_vmm.release(STATE["tensors"][ti][0], off, ln)
        if got > 0:
            ranges.append((ti, off, ln))
            total += got
    keep_mask = mark[a.free_pages] == 0
    a.free_pages = a.free_pages[keep_mask]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        a.release_pages = a.release_pages[mark[a.release_pages] == 0]
    return cand, mark, ranges, total


def decode_lend(runner, reserve_tokens: int, min_bytes: int) -> int:
    """The hot lend, outside prefill: every free page beyond reserve_tokens (and a
    page a running request) leaves the free list and its whole chunks are unmapped,
    when that is at least min_bytes. Returns the bytes unmapped (0: nothing lent)."""
    if not HOT or STATE["lent"] is not None or runner is None or not _setup(runner):
        return 0
    a, ps = STATE["alloc"], STATE["ps"]
    keep = -(-reserve_tokens // ps) + 1
    free = _free_count(a)
    if free <= keep:
        return 0
    est = sum(max(0, (free - keep) * pb - 2 * c) for _, pb, c, _ in STATE["tensors"])
    if est < min_bytes:
        return 0
    got = _release_free(keep)
    if got is None:
        return 0
    cand, _, ranges, total = got
    STATE.update(lent=cand, ranges=ranges, bytes=total, owner="hot")
    STATE["hot_lends"] += 1
    return total


def hot_release(why: str) -> None:
    """Give back what the hot lend holds (a window opens, the pool is cleared)."""
    if why == "window start":
        for fn in WINDOW_START:
            fn(why)
    if STATE["lent"] is not None and STATE.get("owner") == "hot":
        restore(why)


def lend(runner, fb) -> None:
    """kt_lend.before_forward, opening a prefill window, after the weights left."""
    hot_release("window start")
    if not ENABLED or STATE["lent"] is not None or runner is None or not _setup(runner):
        return
    from sglang.srt.layers.moe import kt_prefill_plan

    if not kt_prefill_plan.needs_loan(fb):
        STATE["skipped"] = STATE.get("skipped", 0) + 1
        return
    t0 = time.perf_counter()
    a, ps, n = STATE["alloc"], STATE["ps"], STATE["n"]
    dev = a.free_pages.device
    rem = _remaining(fb)
    if rem is None:
        STATE["unknown"] += 1
        if STATE["unknown"] in (1, 10, 100):
            _log(f"window without a known prompt length ({STATE['unknown']} so far): nothing lent")
        return
    keep_pages = -(-(rem + RESERVE) // ps) + len(fb.rids)
    got = _release_free(keep_pages)
    if got is None:
        return
    cand, mark, ranges, total = got
    if VERIFY:
        # the release above synchronised; the pages hashed are in use, so never
        # in a released chunk
        used = torch.ones(n + 2, dtype=torch.bool, device=dev)
        used[_free_pages(a).long()] = False
        used[cand.long()] = False  # lent (out of the free list already): unmapped
        used[0] = False  # the padding page takes padded tokens' writes
        used[n + 1:] = False
        cur = getattr(fb, "out_cache_loc", None)
        if isinstance(cur, torch.Tensor) and cur.numel():
            used[(cur.long() // ps).clamp(0, n + 1)] = False  # this batch writes them
        STATE["verify"] = (used, _page_hashes(used, n))
    STATE.update(lent=cand, ranges=ranges, bytes=total, owner="window")
    STATE["windows"] += 1
    if STATE["windows"] in (1, 2, 10, 100) or CHECK:
        _log(f"window {STATE['windows']}: {rem} prompt tokens to go, {keep_pages} pages kept, {cand.numel()} lent, "
             f"{total / 1e9:.2f} GB unmapped in {len(ranges)} ranges in {(time.perf_counter() - t0) * 1e3:.1f} ms")


def _page_hashes(mask, n: int) -> torch.Tensor:
    """A 64-bit hash of every page of every K/V tensor where mask is set (else 0):
    the int32 words of a page times fixed odd weights, summed, over the tensors."""
    dev = mask.device
    out = torch.zeros(n + 1, dtype=torch.int64, device=dev)
    idx = torch.nonzero(mask[: n + 1]).flatten()
    if not idx.numel():
        return out
    for ti, (flat, pb, C, c0) in enumerate(STATE["tensors"]):
        if pb % 4:
            continue
        pages = flat[: (n + 1) * pb].view(n + 1, pb).view(torch.int32)
        g = torch.Generator(device="cpu").manual_seed(1000 + ti)
        w = (torch.randint(1, 1 << 30, (pages.shape[1],), generator=g, dtype=torch.int64) * 2 + 1).to(dev)
        for i in range(0, idx.numel(), 256):
            sel = idx[i:i + 256]
            out[sel] += (pages[sel].to(torch.int64) * w).sum(dim=1)
    return out


def restore(why: str = "end of window") -> None:
    """kt_lend.end_window before the weights come back (or an alloc that needs the pages)."""
    if STATE["lent"] is None:
        return
    t0 = time.perf_counter()
    a = STATE["alloc"]
    dev = a.free_pages.device
    owner = STATE.get("owner")
    if owner == "hot":
        # the slots give their memory back first (and leave the routing)
        for fn in PRE_RESTORE:
            fn(why)
    torch.cuda.synchronize(dev)
    torch.cuda.empty_cache()
    for ti, off, ln in STATE["ranges"]:
        flat = STATE["tensors"][ti][0]
        if kt_kv_vmm.restore(flat, off, ln) < 0:
            raise RuntimeError(
                f"KT_PREFILL_KV_LEND: {ln >> 20} MiB of KV tensor {ti} cannot be mapped back on {dev} "
                f"({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free): the prefill kept memory it was lent")
    torch.cuda.synchronize(dev)
    if VERIFY and STATE.get("verify") is not None:
        used, before = STATE.pop("verify")
        still = used.clone()
        still[_free_pages(a).long()] = False  # freed in the window: no longer ours to judge
        still[STATE["lent"].long()] = False
        after = _page_hashes(still, STATE["n"])
        bad = torch.nonzero(still[: STATE["n"] + 1] & (after != before)).flatten()
        STATE["verified"] = STATE.get("verified", 0) + 1
        STATE["verify_bad"] = STATE.get("verify_bad", 0) + int(bad.numel())
        _log(f"verify window {STATE['windows']}: {int(still.sum())} pages in use checked, "
             f"{int(bad.numel())} changed{f' (pages {bad[:8].tolist()})' if bad.numel() else ''}; "
             f"{STATE['verified']} windows so far, {STATE['verify_bad']} pages changed in all")
    a.free_pages = torch.cat((a.free_pages, STATE["lent"].to(a.free_pages.dtype)))
    if owner == "hot":
        if STATE["hot_lends"] in (1, 2, 10, 100) or CHECK or why not in ("window start",):
            _log(f"hot lend {STATE['hot_lends']}, {why}: {STATE['bytes'] / 1e9:.2f} GB mapped back, "
                 f"{STATE['lent'].numel()} pages returned in {(time.perf_counter() - t0) * 1e3:.1f} ms")
    elif STATE["windows"] in (1, 2, 10, 100) or CHECK or why != "end of window":
        _log(f"{why}: {STATE['bytes'] / 1e9:.2f} GB mapped back, {STATE['lent'].numel()} pages returned in "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms")
    STATE.update(lent=None, ranges=[], bytes=0, owner=None)


def _extend_pages(a, args, kw) -> int:
    """Pages an alloc_extend(prefix_lens, prefix_lens_cpu, seq_lens, seq_lens_cpu,
    last_loc, extend_num_tokens, ...) may take: the new tokens' pages and two
    partial ones a request."""
    prefix_lens = kw.get("prefix_lens", args[0] if args else ())
    tokens = kw.get("extend_num_tokens", args[5] if len(args) > 5 else 0)
    return int(tokens) // a.page_size + 2 * len(prefix_lens) + 1


def _wrap_allocator(a) -> None:
    if getattr(a, "_kt_kv_pf_lend", False):
        return
    a._kt_kv_pf_lend = True
    # the token allocator has alloc only (its callers extend and decode through it)
    orig = {k: getattr(a, k) for k in ("alloc", "alloc_extend", "alloc_decode", "available_size", "clear")
            if hasattr(a, k)}

    def _lent() -> int:
        return 0 if STATE["lent"] is None else int(STATE["lent"].numel())

    def ensure(pages: int, why: str) -> None:
        # the allocator merges what it freed when its free list runs short: those
        # pages count (the paged allocator stages every freed page until then)
        if STATE["lent"] is not None and pages > _free_count(a):
            STATE["take_backs"] += 1
            restore(f"take back for {why} ({pages} pages needed, {_free_count(a)} kept)")

    def alloc(need_size: int):
        ensure(-(-need_size // a.page_size), "alloc")
        return orig["alloc"](need_size)

    # the arguments pass through as given: trees differ in them (num_new_pages)
    def alloc_extend(*args, **kw):
        if STATE["lent"] is not None:
            ensure(_extend_pages(a, args, kw), "alloc_extend")
        return orig["alloc_extend"](*args, **kw)

    def alloc_decode(*args, **kw):
        ensure(len(kw.get("seq_lens", args[0] if args else ())) + 1, "alloc_decode")
        return orig["alloc_decode"](*args, **kw)

    def available_size():
        return orig["available_size"]() + _lent() * a.page_size

    def clear():
        restore("clear")
        return orig["clear"]()

    for k, fn in (("alloc", alloc), ("alloc_extend", alloc_extend), ("alloc_decode", alloc_decode),
                  ("available_size", available_size), ("clear", clear)):
        if k in orig:
            setattr(a, k, fn)


# -- the sliding-window pool: lent outside prefill ---------------------------------

SWA_ENABLED = os.environ.get("KT_PREFILL_SWA_LEND", "0") == "1" and kt_kv_vmm.ENABLED
SWA_TOKENS = int(os.environ.get("KT_PREFILL_SWA_TOKENS", "0") or 0)
SWA_KEEP = int(os.environ.get("KT_PREFILL_SWA_LEND_KEEP", "4096"))  # tokens kept mapped outside prefill
SWA_MIN = int(os.environ.get("KT_PREFILL_SWA_LEND_MIN_MB", "64")) << 20  # smallest lend worth a sync
SWA_SLACK = 1024  # tokens mapped beyond an allocation's shortfall
SWA = {"alloc": None, "tensors": None, "lent": None, "bytes": 0, "lends": 0, "maps": 0, "ticks": 0}


SAFE = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
# the chunk the pools are budgeted at under the SWA lend: small, so the SWA pool
# takes no more of the KV budget than a launcher's own ratio does (at the lend's
# chunk MiMo's full pool fell 262,144 -> 225,340); the rest of it is lent
SIZING = int(os.environ.get("KT_PREFILL_SWA_SIZING_CHUNK", "2048") or 0)


def _in_flight() -> int:
    """Chunks of one request the SWA pool holds at once: sglang's own rule (its SWA
    request cap), two with the overlap scheduler, whose next chunk is allocated
    before the last one's slots are freed, else one. V4.1 mapped 0.17, 0.34 and
    0.25 GB of SWA over three chunks of one window."""
    env = os.environ.get("KT_PREFILL_SWA_CHUNKS_IN_FLIGHT")
    if env:
        return max(1, int(env))
    try:
        from sglang.srt.runtime_context import get_server_args

        return 1 if get_server_args().disable_overlap_schedule else 2
    except Exception:
        return 2


@contextlib.contextmanager
def swa_sizing():
    """KV cache configurator, around the memory budget: with the SWA lend the pools
    are budgeted at a small chunk (KT_PREFILL_SWA_SIZING_CHUNK), not at the plan's
    cap, so a pool that counts chunks in flight (DeepSeek V4's SWA cap) takes no
    more of the budget than before; swa_size then widens the SWA pool alone."""
    try:
        from sglang.srt.runtime_context import get_schedule

        bag = get_schedule()
        chunk = int(bag.chunked_prefill_size or 0)
    except Exception:
        bag, chunk = None, 0
    at = min(x for x in (SIZING, SAFE) if x > 0) if (SIZING > 0 or SAFE > 0) else 0
    if not SWA_ENABLED or bag is None or at <= 0 or chunk <= at:
        yield
        return
    SWA["sized_at"] = (at, chunk)
    _log(f"pools budgeted at chunk {at}, the plan's cap {chunk} widens the SWA pool only")
    with bag.override(chunked_prefill_size=at):
        yield


def swa_size(size: int, page: int, follow):
    """KV cache configurator: (SWA pool tokens, the pool that follows its pages).
    At least KT_PREFILL_SWA_TOKENS, and as many more chunks of the cap over the
    budgeted chunk as the pool holds at once (_in_flight); whole pages. A pool
    sized per SWA page (DeepSeek V4's c4 compress state: pages x ring rows) grows
    with it; when it does not divide into the pages the SWA pool is not widened."""
    if not SWA_ENABLED or not size:
        return size, follow
    want = SWA_TOKENS
    # what stays mapped outside prefill at most (swa_trim): the pool a launch
    # without the plan would have had at the lend's own chunk
    SWA["budget"] = size
    if "sized_at" in SWA:
        at, cap = SWA["sized_at"]
        want = max(want, size + _in_flight() * (cap - at))
        SWA["budget"] = size + max(0, SAFE - at)
    page = max(1, int(page))
    want = -(-want // page) * page
    if want <= size:
        return size, follow
    if follow:
        pages = size // page
        if size % page or not pages or follow % pages:
            _log(f"SWA pool {size} not widened: {follow} rows that follow it do not divide into its {pages} pages")
            return size, follow
        follow = want // page * (follow // pages)
    _log(f"SWA pool {size} -> {want} tokens{f' ({follow} compress-state rows)' if follow else ''}: "
         "sized for the largest chunk, its free pages lent outside prefill")
    _lazy_claim(want, page, follow)
    return want, follow


def _lazy_claim(tokens: int, page: int, follow):
    """The widened SWA pool's buffers are made with only their kept pages (and last
    chunk) mapped: the widening costs no memory at setup (MiMo at 74,874 tokens
    asked 3.5 GB there). Claimed by their first dimension: tokens + page (a token
    layout), the pages + 1 (a page layout), or the follower's rows plus its ring
    padding; swa_init checks every claimed buffer is one it lends."""
    pages = tokens // page
    dims = {tokens + page, pages + 1}
    ring = (follow // pages) if follow and pages else 0
    lo, hi = (follow, follow + ring + 8) if ring else (0, -1)
    keep = min(1.0, (SWA_KEEP + SWA_SLACK + 2 * page) / max(1, tokens))

    def claim(shape, nbytes):
        if not shape or nbytes < (2 << 20):
            return None
        d = int(shape[0])
        if d in dims or lo <= d <= hi:
            return int(nbytes * keep) + 1
        return None

    kt_kv_vmm.LAZY = claim


def swa_tokens(size: int) -> int:
    """The SWA pool's size alone (a configurator patched before swa_size)."""
    return swa_size(size, 1, None)[0]


def _swa_state_tensors(pool, n: int):
    """DeepSeek V4's c4 compress-state buffers: ring rows per SWA page
    (state loc = SWA page x ring + loc % ring), so they lend with the SWA pages.
    Only the ratio-4 pools: their size follows the SWA pool (c4_state_pool_size);
    the ratio-128 ones are request-scoped. The pool may hold more rows than its
    ring uses a page (the configurator's ring is the larger): those stay mapped."""
    out, why = [], {}
    for name in ("compress_state_pools", "indexer_compress_state_pools"):
        for sp in getattr(pool, name, None) or []:
            if sp is None or getattr(sp, "ratio", 0) != 4:
                continue
            buf = getattr(getattr(sp, "kv_score_buffer", None), "kv_score", None)
            k = int(getattr(sp, "ring_size", 0) or 0)
            if getattr(sp, "online", False) or not getattr(sp, "swa_page_size", 0):
                r = f"online {getattr(sp, 'online', None)} swa_page_size {getattr(sp, 'swa_page_size', None)}"
            elif not isinstance(buf, torch.Tensor):
                r = f"no kv_score tensor ({type(getattr(sp, 'kv_score_buffer', None)).__name__})"
            elif not (buf.is_cuda and buf.is_contiguous()):
                r = "not a contiguous CUDA tensor"
            elif k <= 0 or buf.shape[0] < n * k:
                r = f"{buf.shape[0]} rows for {n} pages x ring {k}"
            else:
                out.append((buf, k * buf[0].nbytes))
                continue
            why[r] = why.get(r, 0) + 1
    if out or why:
        _log(f"{len(out)} compress-state buffers lend with the SWA pages"
             + "".join(f"; {c} left: {r}" for r, c in why.items()))
    return out


def swa_init(allocator, pool) -> None:
    """KV cache configurator, after the pools are made: set up and lend."""
    kt_kv_vmm.LAZY = None
    if not SWA_ENABLED or SWA["alloc"] is not None or SWA.get("off"):
        _map_lazy("SWA lend not set up")
        return
    SWA["off"] = True  # until set up
    try:
        _swa_init(allocator, pool)
    finally:
        # a lazily made buffer the lend does not hold is mapped whole now
        _map_lazy("not lent", keep={t[0].data_ptr() for t in (SWA["tensors"] or [])} if not SWA.get("off") else set())


def _map_lazy(why: str, keep=frozenset()) -> None:
    made, kt_kv_vmm.LAZY_MADE[:] = list(kt_kv_vmm.LAZY_MADE), []
    rest = [t for t in made if t.data_ptr() not in keep]
    for t in rest:
        if kt_kv_vmm.restore(t.view(-1).view(torch.uint8), 0, t.nbytes) < 0:
            raise RuntimeError(f"KT_PREFILL_SWA_LEND: a lazily made pool buffer cannot be mapped ({why})")
    if rest:
        torch.cuda.synchronize()
        _log(f"{len(rest)} lazily made buffers mapped whole ({why})")
    if made:
        _log(f"{len(made) - len(rest)} SWA buffers made with only their kept pages mapped")


def _swa_init(allocator, pool) -> None:
    a = getattr(allocator, "swa_attn_allocator", None)
    kvp = getattr(pool, "swa_kv_pool", None)
    if a is None or kvp is None:
        _log("no SWA allocator or pool: the SWA lend is off")
        return
    if type(a).__name__ not in ("PagedTokenToKVPoolAllocator", "TokenToKVPoolAllocator") \
            or not isinstance(getattr(a, "free_pages", None), torch.Tensor):
        _log(f"SWA allocator {type(a).__name__} has no single free list: the SWA lend is off")
        return
    n = int(getattr(a, "num_pages", 0) or a.size // a.page_size)
    tensors = _kv_tensors(kvp, n, a.page_size, "SWA K/V", _swa_state_tensors(pool, n))
    if not tensors:
        return
    SWA.update(alloc=a, tensors=tensors, ps=a.page_size, n=n, off=False,
               row=sum(t[1] for t in tensors) / a.page_size)
    _wrap_swa(a)
    _log(f"SWA lend: {len(tensors)} K/V tensors of {kvp.__class__.__name__}, {n} pages of {a.page_size} tokens, "
         f"{SWA['row'] / 1024:.1f} KiB a token, chunks {min(t[2] for t in tensors) >> 10}-"
         f"{max(t[2] for t in tensors) >> 10} KiB, {SWA_KEEP} tokens kept mapped")
    swa_lend("pool setup", force=True)
    # what is unmapped now (lent, and never mapped in a lazily made buffer)
    SWA["bytes"] = kt_kv_vmm.released_bytes(a.free_pages.device.index)


def _swa_free(a) -> int:
    return _free_count(a)


def swa_lend(why: str, force: bool = False) -> None:
    """Unmap the free SWA pages beyond the kept ones (outside prefill only). force:
    whatever the surplus (at setup, where the pages past the kept ones of a lazily
    made pool have never been mapped and must leave the free list)."""
    a = SWA["alloc"]
    if a is None:
        return
    swa_flush(why)  # pages handed back but not mapped yet are not lent twice
    if not force:
        swa_trim(why)
    keep = -(-SWA_KEEP // SWA["ps"]) + 1
    surplus = _swa_free(a) - keep
    if surplus <= 0 or (not force and surplus * SWA["row"] * SWA["ps"] < SWA_MIN):
        return
    t0 = time.perf_counter()
    _merge_free(a)
    n, dev = SWA["n"], a.free_pages.device
    free = torch.sort(_free_pages(a)).values
    cand = free[keep:]
    lent = cand if SWA["lent"] is None else torch.cat((SWA["lent"], cand))
    mark = torch.zeros(n + 2, dtype=torch.int32, device=dev)
    mark[lent] = 1
    torch.cuda.synchronize(dev)
    got = 0
    for ti, off, ln in _whole_chunks(SWA["tensors"], mark, n, dev):
        got += max(0, kt_kv_vmm.release(SWA["tensors"][ti][0], off, ln))
    a.free_pages = a.free_pages[mark[a.free_pages] == 0]
    if getattr(a, "release_pages", None) is not None and len(a.release_pages):
        a.release_pages = a.release_pages[mark[a.release_pages] == 0]
    SWA["lent"] = torch.sort(lent).values
    SWA["bytes"] += got
    SWA["lends"] += 1
    if SWA["lends"] <= 3 or SWA["lends"] in (10, 100, 1000) or CHECK:
        _log(f"SWA lend ({why}): {cand.numel()} pages more, {SWA['lent'].numel()} lent, "
             f"{SWA['bytes'] / 1e9:.2f} GB unmapped (+{got / 1e9:.2f}) in {(time.perf_counter() - t0) * 1e3:.1f} ms")


def _defer_ok() -> bool:
    """A lent page handed out now may be mapped at the next forward's start: the
    lend sees every forward (kt_lend.before_forward flushes), and a window maps
    after the weights have left, when there is room."""
    try:
        from sglang.srt.layers.moe import kt_lend
    except ImportError:
        return False
    return bool(kt_lend.ENABLED) and kt_lend.STATE.get("phase") in ("decode", "prefill")


def bind_tree(scheduler) -> None:
    """kt_prefill_plan, from the scheduler: its prefix cache, for swa_trim."""
    if SWA_ENABLED and "tree" not in SWA:
        SWA["tree"] = getattr(scheduler, "tree_cache", None)


def _swa_in_use(a) -> int:
    lent = 0 if SWA["lent"] is None else int(SWA["lent"].numel())
    return (SWA["n"] - _swa_free(a) - lent) * SWA["ps"]


def swa_trim(why: str) -> None:
    """Outside prefill the SWA pages in use stay within the budget: a prefix
    cache keeps a cached prompt's SWA tokens until its pool runs short, which with
    the lent pages counted it never does (MiMo kept ~1 GB a cached 24K prompt and
    the weights could not come back after the fifth). The cache gives back the
    excess, least recently used first, as it would under pressure (sglang's
    tree_cache.evict with swa_num_tokens)."""
    a, tree, budget = SWA["alloc"], SWA.get("tree"), SWA.get("budget")
    if a is None or tree is None or not budget:
        return
    over = _swa_in_use(a) - budget
    if over <= 4 * SWA["ps"]:
        return
    try:
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        res = tree.evict(EvictParams(swa_num_tokens=int(over)))
    except Exception as e:  # a cache without SWA eviction: nothing to trim
        if not SWA.get("trim_off"):
            SWA["trim_off"] = True
            _log(f"SWA trim unavailable ({type(e).__name__}: {e})")
        return
    SWA["trims"] = SWA.get("trims", 0) + 1
    if SWA["trims"] <= 3 or SWA["trims"] in (10, 100) or CHECK:
        _log(f"SWA trim ({why}): {over} tokens over the {budget}-token budget, the cache gave back "
             f"{getattr(res, 'swa_num_tokens_evicted', '?')}")


def swa_map(short: int, why: str, defer: bool = False) -> None:
    """Hand back the lowest-numbered lent pages, short pages and some slack more.
    Their chunks are mapped now, or with defer at the next forward's start
    (swa_flush): an allocation for a prefill comes before its window, while
    decode still holds the weights (V4.1's second 35K could not map 279 MiB there)."""
    a, lent = SWA["alloc"], SWA["lent"]
    if lent is None or not lent.numel():
        return
    t0 = time.perf_counter()
    dev = a.free_pages.device
    k = min(int(lent.numel()), short + -(-SWA_SLACK // SWA["ps"]))
    top = int(lent[k - 1])
    # each tensor from its first byte up to a chunk boundary past page top (the
    # chunks still mapped are skipped); the pages every tensor then has rejoin
    pend = SWA.setdefault("pending", [0] * len(SWA["tensors"]))
    upto = None
    for i, (flat, pb, C, c0) in enumerate(SWA["tensors"]):
        # the first chunk boundary at or past page top's end, at most the last whole chunk's end
        last = c0 + max(0, flat.numel() - c0) // C * C
        end = min(c0 + -(-max(0, (top + 1) * pb - c0) // C) * C, last)
        pend[i] = max(pend[i], end)
        # a tensor's last, partial chunk is never lent: mapped through to its end
        has = SWA["n"] + 1 if end >= last else end // pb - 1
        upto = has if upto is None else min(upto, has)
    back = lent[lent <= upto]
    SWA["lent"] = lent[lent > upto] if back.numel() < lent.numel() else None
    a.free_pages = torch.cat((a.free_pages, back.to(a.free_pages.dtype)))
    SWA["maps"] += 1
    note = SWA["maps"] <= 3 or SWA["maps"] in (10, 100, 1000) or CHECK
    if note:
        _log(f"SWA pages back for {why}: {back.numel()} ({short} short), "
             f"{'mapped at the next forward' if defer else 'mapping now'}")
    if not defer:
        swa_flush(why, quiet=not note)


def swa_flush(why: str, quiet: bool = False) -> None:
    """Map the chunks of the pages handed back since the last flush (zeroed)."""
    pend = SWA.get("pending")
    if not pend or not any(pend):
        return
    t0 = time.perf_counter()
    dev = SWA["alloc"].free_pages.device
    got = 0
    for (flat, _, _, _), end in zip(SWA["tensors"], pend):
        if end <= 0:
            continue
        r = kt_kv_vmm.restore(flat, 0, end)
        if r < 0:
            # inside a window torch's cache holds the earlier chunks' activations,
            # free to torch but not to the driver (V4.1 failed here with 2 MiB
            # free and a plan that fit): hand it back and map again
            torch.cuda.synchronize(dev)
            torch.cuda.empty_cache()
            r = kt_kv_vmm.restore(flat, 0, end)
            SWA["map_retries"] = SWA.get("map_retries", 0) + 1
            if SWA["map_retries"] in (1, 10, 100):
                _log(f"SWA map at {why} needed torch's cache handed back ({SWA['map_retries']} so far)")
        if r < 0:
            raise RuntimeError(
                f"KT_PREFILL_SWA_LEND: SWA pages for {why} cannot be mapped back on {dev} "
                f"({torch.cuda.mem_get_info(dev)[0] >> 20} MiB free)")
        got += r
    torch.cuda.synchronize(dev)  # the zeroing is done before any kernel reads them
    SWA["pending"] = [0] * len(SWA["tensors"])
    SWA["bytes"] -= got
    if not quiet:
        _log(f"SWA map at {why}: {got / 1e9:.2f} GB mapped, {SWA['bytes'] / 1e9:.2f} GB still lent, "
             f"{(time.perf_counter() - t0) * 1e3:.1f} ms")


def swa_restore_all(why: str) -> None:
    a, lent = SWA["alloc"], SWA["lent"]
    if lent is None:
        return
    swa_map(int(lent.numel()), why)


def swa_terms():
    """kt_prefill_plan: (SWA tokens a request takes without mapping, bytes a token
    beyond them, bytes of rounding and slack, chunks held at once); None without
    the SWA lend."""
    a = SWA["alloc"]
    if a is None or SWA["lent"] is None:
        return None
    ps = SWA["ps"]
    have = _swa_free(a) * ps - ps  # the request's first page may be partly used
    extra = sum(t[2] for t in SWA["tensors"]) + SWA_SLACK * SWA["row"]
    # a later chunk is allocated before the last one's slots are freed
    return have, SWA["row"], int(extra), _in_flight()


SWA_EAGER = int(os.environ.get("KT_PREFILL_SWA_LEND_EAGER_MB", "256")) << 20


def swa_tick() -> None:
    """kt_lend, before every forward outside a window: lend what the frees gave
    back, at once when it is SWA_EAGER or more (a long prompt left 2.5 GB of SWA
    mapped on MiMo, and extend-only requests after it ran out of memory waiting
    for the 32-forward check), else every 32 forwards."""
    a = SWA["alloc"]
    if a is None:
        return
    SWA["ticks"] += 1
    surplus = (_swa_free(a) - -(-SWA_KEEP // SWA["ps"]) - 1) * SWA["row"] * SWA["ps"]
    over = SWA.get("budget") and _swa_in_use(a) > SWA["budget"] + 4 * SWA["ps"]
    if over or surplus >= SWA_EAGER or (SWA["ticks"] % 32 == 0 and surplus >= SWA_MIN):
        swa_lend("decode")


def _wrap_swa(a) -> None:
    if getattr(a, "_kt_swa_lend", False):
        return
    a._kt_swa_lend = True
    orig = {k: getattr(a, k) for k in ("alloc", "alloc_extend", "alloc_decode", "available_size", "clear")
            if hasattr(a, k)}

    def _lent() -> int:
        return 0 if SWA["lent"] is None else int(SWA["lent"].numel())

    def ensure(pages: int, why: str) -> None:
        if SWA["lent"] is not None and pages > _swa_free(a):
            swa_map(pages - _swa_free(a), why, defer=_defer_ok())

    def alloc(need_size: int):
        ensure(-(-need_size // a.page_size), "alloc")
        return orig["alloc"](need_size)

    def alloc_extend(*args, **kw):
        if SWA["lent"] is not None:
            ensure(_extend_pages(a, args, kw), "alloc_extend")
        return orig["alloc_extend"](*args, **kw)

    def alloc_decode(*args, **kw):
        ensure(len(kw.get("seq_lens", args[0] if args else ())) + 1, "alloc_decode")
        return orig["alloc_decode"](*args, **kw)

    def available_size():
        return orig["available_size"]() + _lent() * a.page_size

    def clear():
        swa_restore_all("clear")
        return orig["clear"]()

    for k, fn in (("alloc", alloc), ("alloc_extend", alloc_extend), ("alloc_decode", alloc_decode),
                  ("available_size", available_size), ("clear", clear)):
        if k in orig:
            setattr(a, k, fn)
