"""Prefill-lend probe: what a chunk costs at a long prefix, without the prefill.

KT_PREFILL_LEND_PROBE="<chunks>@<prefixes>", e.g. "2048,6144@0,131072,524288,991232":
before the scheduler's event loop starts, for every prefix P and chunk C, one
extend forward of C rows runs over a P-token prefix whose KV was allocated but
never computed (its contents are whatever the pool held). The memory such a
forward needs is shaped by P and C alone (page tables [C, P/256], the
indexer's compressed K and its logits slices), so the lend window it opens
records the same point a real chunked prefill reaching P would, in seconds
instead of the 40 minutes a 1M-token prefill takes. Outputs are discarded.

The prefix is laid out as a disaggregated decode receives one: full-pool
slots for all P tokens, SWA slots only for the last window
(alloc_extend_swa_tail), the state a long chunked prefill leaves after it
evicts the SWA slots behind the window. Every TP rank's scheduler runs the
same steps in the same order, so the forward's collectives line up.

Runs only where the allocator has alloc_extend_swa_tail (DeepSeek V4's SWA
pool), or with P = 0 anywhere. Called from dispatch_event_loop
(patches/sglang-kt-lend-probe.py); the lend must be on.
"""
import logging
import os
import time
from array import array

import torch

logger = logging.getLogger(__name__)
# chunks per probe point, the last one at the point's prefix
STEPS = int(os.environ.get("KT_PREFILL_LEND_PROBE_STEPS", "4"))


def _spec():
    raw = os.environ.get("KT_PREFILL_LEND_PROBE", "").strip()
    if not raw:
        return [], []
    chunks, _, prefixes = raw.partition("@")
    chunks = [int(x) for x in chunks.split(",") if x.strip()]
    prefixes = [int(x) for x in (prefixes or "0").split(",") if x.strip()]
    return chunks, prefixes


@torch.no_grad()
def run(scheduler) -> None:
    chunks, prefixes = _spec()
    if not chunks:
        return
    from sglang.srt.layers.moe import kt_lend
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.mem_cache.common import release_kv_cache
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if not kt_lend.ENABLED:
        logger.warning("kt-lend probe: the lend is off; nothing to measure")
        return
    mr = scheduler.tp_worker.model_runner
    rtp = scheduler.req_to_token_pool
    alloc = scheduler.token_to_kv_pool_allocator
    tree = scheduler.tree_cache
    page = int(getattr(alloc, "page_size", 1) or 1)
    window = int(getattr(tree, "sliding_window_size", None) or 0)
    dev = mr.device
    can_tail = hasattr(alloc, "alloc_extend_swa_tail")
    t_all = time.perf_counter()
    for P_point in prefixes:
        P_point = P_point // page * page
        if P_point > 0 and not can_tail:
            logger.warning("kt-lend probe: no alloc_extend_swa_tail here; prefix %d skipped", P_point)
            continue
        for C in chunks:
            P = P_point
            t0 = time.perf_counter()
            # the window ends at P after STEPS chunks, as a real chunked prefill's
            # last chunks do: one forward alone misses what the growing
            # prefix-sized buffers leave in torch's cache from chunk to chunk
            # (1.3-1.5 GB at 989K, 10-07)
            P0 = max(0, (P - (STEPS - 1) * C) // page * page)
            req = Req(
                rid=f"kt-lend-probe-{P}-{C}",
                origin_input_text="",
                origin_input_ids=array("q", [1000] * (P + C)),
                sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
            )
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.logprob_start_len = -1
            P_end, P = P, P0
            if P > 0:
                idx = rtp.alloc_rows(1)[0]
                req.kv.req_pool_idx = idx
                tail = P - max(0, (P - window) // page * page) if window else P
                i64 = torch.int64
                loc = alloc.alloc_extend_swa_tail(
                    prefix_lens=torch.zeros(1, dtype=i64, device=dev),
                    prefix_lens_cpu=torch.zeros(1, dtype=i64),
                    seq_lens=torch.tensor([P], dtype=i64, device=dev),
                    seq_lens_cpu=torch.tensor([P], dtype=i64),
                    last_loc=torch.tensor([-1], dtype=i64, device=dev),
                    extend_num_tokens=P,
                    swa_tail_len=tail,
                )
                if loc is None:
                    logger.warning("kt-lend probe: no room for a %d-token prefix", P)
                    rtp.free_rows([idx])
                    req.kv.req_pool_idx = None
                    continue
                rtp.write((idx, slice(0, P)), loc)
                req.kv.kv_allocated_len = req.kv.kv_committed_len = P
                req.kv.swa_evicted_seqlen = P - tail
                req.prefix_indices = loc.to(torch.int64)
            while True:
                n = min(C, P_end + C - P)
                req.set_extend_range(P, P + n)
                batch = ScheduleBatch.init_new(
                    reqs=[req],
                    req_to_token_pool=rtp,
                    token_to_kv_pool_allocator=alloc,
                    tree_cache=tree,
                    model_config=mr.model_config,
                    enable_overlap=False,
                    spec_algorithm=SpeculativeAlgorithm.NONE,
                )
                batch.prepare_for_extend()
                if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
                    batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
                    batch.prefill_input_ids_cpu = None
                fb = ForwardBatch.init_new(batch, mr, return_hidden_states_before_norm=False)
                mr.forward(fb)
                torch.cuda.synchronize(dev)
                P += n
                if P >= P_end + C:
                    break
                # the next chunk continues the same request over what this one wrote
                req.prefix_indices = rtp.req_to_token[req.kv.req_pool_idx, :P].to(torch.int64)
            kt_lend.end_window()
            release_kv_cache(req, tree, is_insert=False)
            logger.info("kt-lend probe: prefix %d, chunk %d (%d chunks from %d) in %.1f s",
                        P_end, C, -(-(P_end + C - P0) // C), P0, time.perf_counter() - t0)
    logger.info("kt-lend probe: done in %.1f s", time.perf_counter() - t_all)
