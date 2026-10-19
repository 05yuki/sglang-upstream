"""The prefill chunk planned per request (KT_PREFILL_PLAN=1, with the prefill lend
and, for the room of the free KV pages, KT_PREFILL_KV_LEND=1).

The launch chunk (--chunked-prefill-size) becomes the largest chunk the
streamer's buffers are sized for; each request then gets the fewest chunks its
remaining prompt needs and, for that count, the smallest equal chunk.

A chunk of c rows whose forward ends at sequence s needs, above a window's start,

  cost(c, s) = FIXED + ROW * c + SEQ * s          (KT_PREFILL_PLAN_COST="FIXED_MiB,ROW_B,SEQ_B,CARRY_MiB",
                                                   fitted by tools/kt-prefill-plan-constants.py
                                                   over the lend's per-forward records)

(+ CARRY for a chunk after the first of a window) and has
  room = free at the window's start + the free KV pages it borrows - kt_lend.MARGIN.
The free memory is live: each TP rank's decode free memory now plus the step
kt_lend measured from decode to a window's start (weights out, scratch in),
the smallest over the ranks (an all-reduce on the scheduler's TP CPU group, so
every rank plans the same chunk). Before the first window has measured that
step, KT_PREFILL_PLAN_FREE_MB stands in. The last chunk of a request is the
one with the longest sequence, so it is the one checked.

Without KT_PREFILL_PLAN_COST the older model applies: cost = KT_PREFILL_PLAN_ROW_KB * rows.
"""
import math
import os

ENABLED = os.environ.get("KT_PREFILL_PLAN", "0") == "1"
FREE = int(os.environ.get("KT_PREFILL_PLAN_FREE_MB", "0") or 0) << 20
ROW_KB = int(os.environ.get("KT_PREFILL_PLAN_ROW_KB", "0") or 0) << 10
_cost = [x for x in os.environ.get("KT_PREFILL_PLAN_COST", "").split(",") if x]
FIXED, ROW, SEQ = ((int(float(_cost[0]) * (1 << 20)), float(_cost[1]), float(_cost[2]))
                   if len(_cost) >= 3 else (0, float(ROW_KB), 0.0))
# what a later chunk of a window holds from the earlier ones (4th field, MiB)
CARRY = int(float(_cost[3]) * (1 << 20)) if len(_cost) >= 4 else 0
LIVE = os.environ.get("KT_PREFILL_PLAN_LIVE", "1") == "1"
DELTA = int(os.environ.get("KT_PREFILL_PLAN_DELTA_MB", "0") or 0) << 20
ALIGN = int(os.environ.get("KT_PREFILL_PLAN_ALIGN", "1024"))
# the chunk this setup is known to take without a plan (the lend's hint): what a
# request gets while there is no free-memory step to plan with (the first window
# of a launch without KT_PREFILL_PLAN_DELTA_MB); unset, the launch chunk
SAFE = int(os.environ.get("KT_PREFILL_PLAN_SAFE", "0") or 0)
STATS = {"plans": 0, "last": None}
# what a request's own forward puts back on the GPU before its rows run, by name (bytes):
# a model that keeps its vision tower on the host between images registers it
# (sglang-dsv4-vision-on-demand.py), and a request with images is planned with that
# much less room
ON_DEMAND = {}


def on_demand(name: str, nbytes: int) -> None:
    ON_DEMAND[name] = int(nbytes)


def _on_demand_for(req) -> int:
    if not ON_DEMAND:
        return 0
    mm = getattr(req, "multimodal_inputs", None)
    if mm is None or not getattr(mm, "mm_items", None):
        return 0
    return sum(ON_DEMAND.values())


# room kept free in a planned window (MiB); unset: the lend's own margin. The
# fitted cost already lies above every recorded forward (its envelope), so this
# is the second allowance, not the only one
_PLAN_MARGIN = os.environ.get("KT_PREFILL_PLAN_MARGIN_MB")


def _margin() -> int:
    if _PLAN_MARGIN not in (None, ""):
        return int(float(_PLAN_MARGIN) * (1 << 20))
    from sglang.srt.layers.moe import kt_lend

    return kt_lend.MARGIN


def _ready() -> bool:
    return ENABLED and ROW > 0 and (FREE > 0 or LIVE)


# KT_PREFILL_KV_LEND_FORCE=1: every window lends whatever the plan says (for checks
# such as KT_PREFILL_KV_LEND_VERIFY, which see only windows that lend)
FORCE_LOAN = os.environ.get("KT_PREFILL_KV_LEND_FORCE", "0") == "1"
NEED = {}  # rid -> the plan needs the KV loan (decided on the all-reduced free memory)
PLAN = {}  # rid -> the chunk planned when the request was first taken


def chunk_in_flight(req, default: int) -> int:
    """sglang's chunk-cache SWA eviction under the overlap scheduler: the chunk still
    in flight when the next is scheduled is the one planned for this request, not
    the launch chunk (the cap); with the cap, pre_len - 36,864 left a 35K request
    holding every chunk's SWA until it ended."""
    if not ENABLED:
        return default
    return int(PLAN.get(getattr(req, "rid", None), default) or default)


def needs_loan(fb) -> bool:
    """kt_kv_prefill_lend.lend: whether the window's requests were planned with the
    loan. The plan decided it from values every rank shares, so every rank lends
    or none does; a request the plan never saw is lent for."""
    if not _ready() or FORCE_LOAN:
        return True
    rids = getattr(fb, "rids", None) or []
    return not rids or any(NEED.get(r, True) for r in rids)


def _remaining(req):
    """(tokens to prefill, tokens already in the cache) for the request."""
    total = len(getattr(req, "origin_input_ids", None) or []) + len(getattr(req, "output_ids", None) or [])
    done = int(getattr(req, "prefix_len", 0) or 0)
    return max(0, total - done), done


def _free_now(scheduler) -> int:
    """Free memory a window would start with, the smallest over the TP ranks."""
    if not LIVE:
        return FREE
    import torch

    from sglang.srt.layers.moe import kt_lend

    delta = kt_lend.STATE.get("pf_delta")
    if delta is None and DELTA:
        # before this launch's first window: the step an earlier launch measured
        delta = DELTA
    if kt_lend.STATE.get("phase") == "prefill":
        # a request taken while the window is still open (long prompts back to back,
        # no decode between): the device is already in the window's state, so decode
        # free + step reads low by torch's cached blocks of the last chunk (35K then
        # 131K: 5,575 MiB planned against 7,390 at the window's start). What the next
        # chunks have is the device's free memory now plus those blocks; not the
        # window's start, which misses what the window has kept since (the stream's
        # buffers grown to a larger chunk: 131K planned at 13,312 rows ran out)
        mine = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    elif delta is not None:
        mine = torch.cuda.mem_get_info()[0] + delta
    else:
        mine = -1
    if mine >= 0:
        # what earlier windows left in the lend's scratch (the stream's buffers and
        # graph pool) is reused by the next chunks, and the fitted cost includes it
        # (kt_lend.end_window)
        mine += int(kt_lend.STATE.get("scratch_kept", 0) or 0)
    if mine >= 0 and kt_lend.STATE.get("phase") != "prefill":
        # the hot lend's share of the driver's free memory comes back at the window's
        # start (sglang-kt-kv-hot-spare-plan.py)
        from sglang.srt.layers.moe import kt_kv_hot_lend

        mine += int(kt_kv_hot_lend.STATE.get("spare_held", 0) or 0)
    t = torch.tensor([mine], dtype=torch.int64)
    group = getattr(scheduler, "tp_cpu_group", None)
    if group is not None and torch.distributed.is_initialized():
        # every rank reaches this call under the same conditions (the queue and
        # the allocator are the same on every rank); a rank without a measured
        # step sends -1 and all of them fall back to the constant
        torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MIN, group=group)
    v = int(t.item())
    return v if v >= 0 else FREE


def chunk_for(scheduler, chunk: int) -> int:
    """Scheduler.get_new_batch_prefill: the chunk for the batch it is about to make."""
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    getattr(kt_kv_prefill_lend, "bind_tree", lambda s: None)(scheduler)
    if not _ready() or not chunk or chunk <= 0:
        return chunk
    req = scheduler.chunked_req
    if req is None:
        q = scheduler.waiting_queue
        req = q[0] if q else None
    if req is None:
        return chunk
    if req is not scheduler.chunked_req:
        # a queued request: the prefix the radix cache already holds is not work
        # (the same read-only match the scheduling policy makes; the adder
        # matches again when it takes the request)
        try:
            from sglang.srt.mem_cache.common import match_kv_cache

            match_kv_cache(req, scheduler.tree_cache)
        except Exception:
            pass
    rem, done = _remaining(req)
    if rem <= 0:
        return chunk
    if req is scheduler.chunked_req and req.rid in PLAN:
        # a later chunk keeps the request's plan: with the overlap scheduler this
        # runs while the previous chunk's forward still holds its activations, so
        # the live free memory would be low by them
        return min(chunk, PLAN[req.rid])
    from sglang.srt.layers.moe import kt_kv_prefill_lend

    free = _free_now(scheduler)
    if free <= 0:
        return min(chunk, SAFE) if SAFE > 0 else chunk
    loan = kt_kv_prefill_lend.loan_estimate(scheduler, rem)
    back = _on_demand_for(req)
    room = free + loan - _margin() - back
    # the sliding-window pool's lent pages a request's chunks map back
    # (KT_PREFILL_SWA_LEND): nothing up to the mapped free ones, then a token's SWA
    # bytes a row, for every row the request may hold at once
    swa = getattr(kt_kv_prefill_lend, "swa_terms", lambda: None)()
    if swa is not None and len(swa) < 4:
        swa = tuple(swa) + (1,)

    def swa_cost(c: int) -> int:
        if swa is None:
            return 0
        # one chunk holds its own rows; with more, the chunks held at once (swa[3]:
        # two under the overlap scheduler, whose next chunk is allocated before the
        # last one's slots are freed). Not the whole remaining prompt: a 512K prompt
        # in 2,048-row chunks held 6% of a 77K-token pool throughout (2.25 chunks),
        # 7,168-row chunks 19% (2.04), 18,432-row 46% (1.92) (10-18), while charging
        # the remaining prompt planned 11.5 GB and kept a 512K prefill at 2,048 rows.
        need = c if rem <= c else c * max(1, int(swa[3]))
        return 0 if need <= swa[0] else int((need - swa[0]) * swa[1] + swa[2])

    def fits(c: int, n: int) -> bool:
        """The last chunk ends at the longest sequence and, when there is more than
        one, carries what the earlier ones left: it is the one that must fit."""
        carry = CARRY if n > 1 else 0
        return FIXED + carry + ROW * c + SEQ * (done + rem) + swa_cost(c) <= room

    # the fewest chunks whose equal size, in whole ALIGN steps, fits (the step
    # rounds a chunk up: 35,000 in three is 11,667 rows, 12,288 once aligned)
    for n in range(1, 257):
        c = math.ceil(rem / n)
        c = min(chunk, max(ALIGN, -(-c // ALIGN) * ALIGN))
        if fits(c, n) or c <= ALIGN:
            break
    n = max(1, math.ceil(rem / c))  # the launch chunk may cap it (V4.1's SWA pool)
    if len(NEED) > 4096:
        NEED.clear()
        PLAN.clear()
    PLAN[req.rid] = c
    # the loan when the estimate needs it, and always past the lend's own chunk:
    # there the estimate is extrapolated, and lending costs a few tens of ms
    NEED[req.rid] = (FIXED + (CARRY if n > 1 else 0) + ROW * c + SEQ * (done + rem) + swa_cost(c)
                     + _margin() + back > free) or (SAFE > 0 and c > SAFE)
    STATS["plans"] += 1
    plan = (rem, n, c, back)
    if plan != STATS["last"]:
        STATS["last"] = plan
        print(f"[kt-prefill-plan] {rem} tokens to prefill after {done} cached: {n} chunk(s) of {c} "
              f"(free {free >> 20} MiB + loan {loan >> 20} MiB, fixed {FIXED >> 20} MiB, "
              f"{ROW / 1024:.0f} KiB a row, {SEQ:.0f} B a sequence token, carry {CARRY >> 20} MiB, "
              f"SWA mapped back {swa_cost(c) >> 20} MiB"
              + (f", {back >> 20} MiB back on the GPU for the images ({', '.join(ON_DEMAND)})" if back else "")
              + f", cap {chunk})", flush=True)
    return c
