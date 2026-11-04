"""Hot GPU experts straight from a GGUF, for the KT hybrid (--kt-method
LLAMAFILE with KT_GGUF_DENSE): the GPU slots 0..num_gpu_experts-1 hold this
rank's TP slice of the expert's raw GGUF blocks, and decode runs SGLang's
MMVQ MoE kernel (sgl_kernel.ggml_moe_a8_vec: int8 activations, every K-quant
and i-quant type) on them; windows of up to 4 tokens (decode, MTP verify) run
kt_gguf_moe_multi instead (same sums, one read per expert, CPU routes skipped);
longer windows (an extend below the streamer threshold) run that kernel's
per-route build (moe_a8_vec_routes: ggml's IQ3_S vec_dot, which sgl-kernel's
lacks; KT_GGUF_MOE_ROUTES=0 goes back to ggml_moe_a8_vec and the Q2_0 window loop).
Prefill above the streamer threshold never gets here.
With KV-lent hot experts (KT_HOT_KV_LEND=1, kt_hot_kv_lend.py) the layer carries
per-slot pointer tables (layer.kt_kv_ptr13 / kt_kv_ptr2, slots 0..E_gpu-1 point
into w13_qweight / w2_qweight, the lent ones into free KV pages) and the window
kernel and its per-route build read the weights through them; the hot cache
gives sgl-kernel's ggml_moe_a8_vec (KT_GGUF_MOE_ROUTES=0) resident-only ids.

Layout per layer (E_gpu = num_gpu_experts, N = intermediate per rank, K = hidden):
  w13_qweight  uint8 [E_gpu, 2N, row_bytes(K, gate/up type)]   rows [gate; up]
  w2_qweight   uint8 [E_gpu, K,  row_bytes(N, down type)]
Gate and up must share a type (they do in every llama.cpp quantization).
The down of the Q2_0 layers (type 42; sgl-kernel has no kernel for it) stays
2-bit: kt_gguf_moe_multi has a Q2_0 dot product, and those layers run the window
kernel at every length. KT_GGUF_Q2_NATIVE=0 repacks it losslessly into Q8_0
instead (same fp16 d, codes -1..2 as int8, two 32-blocks per 64-block, 4x the
bytes) for the sgl-kernel path.
"""
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch.nn import Parameter

from sglang.srt.layers.moe import kt_gguf_moe_multi as _multi

# KT_HOT_FUSED_COMBINE=1 (10-13): the route weighting and the sum over top_k as one Triton kernel instead of
# out.mul_(w) + moe_sum (an elementwise and a 4 us reduce per layer); bit-identical by construction.
FUSED_COMBINE = os.environ.get("KT_HOT_FUSED_COMBINE", "1") == "1"
_combine_kernel = None


def _combine_kernel_fn():
    import triton
    import triton.language as tl

    @triton.jit
    def _kt_combine_kernel(out_ptr, w_ptr, res_ptr, K, TOPK: tl.constexpr, BLOCK: tl.constexpr):
        t = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < K
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(TOPK):
            o = tl.load(out_ptr + (t * TOPK + k) * K + offs, mask=m, other=0.0).to(tl.float32)
            wk = tl.load(w_ptr + t * TOPK + k).to(tl.bfloat16).to(tl.float32)  # mul_ cast w to bf16 first
            acc += (o * wk).to(tl.bfloat16).to(tl.float32)
        tl.store(res_ptr + t * K + offs, acc.to(tl.bfloat16), mask=m)

    return _kt_combine_kernel


def _combine(out, w):
    """out [T, top_k, K] bf16 (the routes' rows), w [T, top_k]: sum_k bf16(out * w) in fp32, rounded once."""
    global _combine_kernel
    if _combine_kernel is None:
        _combine_kernel = _combine_kernel_fn()
    T, top_k, K = out.shape
    out = out.contiguous()
    res = torch.empty((T, K), dtype=out.dtype, device=out.device)
    _combine_kernel[(T, (K + 1023) // 1024)](out, w.contiguous(), res, K, TOPK=top_k, BLOCK=1024)
    return res
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase

GGML_Q8_0 = 8
GGML_Q2_0 = 42
BLOCK_INFO = {  # ggml type -> (block elements, block bytes)
    8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210),
    16: (256, 66), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    18: (256, 98), 42: (64, 18), 17: (256, 74), 29: (256, 56),
}
MMVQ_TYPES = {2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 29, 30}


def _log(msg):
    print(f"[kt-gguf-hot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _row_bytes(cols: int, t: int) -> int:
    blck, ts = BLOCK_INFO[t]
    assert cols % blck == 0, (cols, t)
    return cols // blck * ts


def q2_0_to_q8_0(raw: torch.Tensor) -> torch.Tensor:
    """[..., n_blocks*18] Q2_0 bytes -> [..., n_blocks*2*34] Q8_0 bytes, exact."""
    shape = raw.shape[:-1]
    b = raw.reshape(-1, 18).numpy()
    n = b.shape[0]
    d = b[:, :2]
    qs = b[:, 2:]  # 16 bytes, 4 codes each
    codes = np.stack([(qs >> (2 * s)) & 3 for s in range(4)], axis=2).reshape(n, 64).astype(np.int8) - 1
    out = np.empty((n, 2, 34), dtype=np.uint8)
    out[:, :, :2] = d[:, None, :]
    out[:, 0, 2:] = codes[:, :32].view(np.uint8)
    out[:, 1, 2:] = codes[:, 32:].view(np.uint8)
    return torch.from_numpy(out.reshape(*shape, -1))


class _GGUFExperts:
    """One reader shared by all layers of a rank."""

    def __init__(self, path: str):
        from kt_kernel.utils.loader import GGUFLoader

        self.loader = GGUFLoader(path)
        cfg = self.loader.get_model_config()
        self.E = int(cfg["num_experts"])
        self.inter = int(cfg["moe_intermediate_size"])
        self.K = int(cfg["hidden_size"])

    def _raw(self, name: str) -> Tuple[torch.Tensor, int]:
        """The tensor's bytes as a view over the file mapping (the loader's
        accessor copies; an expert tensor is 1-2 GB)."""
        import warnings

        info = self.loader.tensor_info[name]
        t = int(info["dtype"])
        blck, ts = BLOCK_INFO[t]
        n_bytes = int(info["n_elements"]) // blck * ts
        mm = self.loader.file_data_map[self.loader.tensor_file_map[name]]
        raw = np.frombuffer(mm, dtype=np.uint8, count=n_bytes, offset=int(info["offset"]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # read-only buffer, only copied out
            return torch.from_numpy(raw), t

    def types(self, layer_idx: int) -> Tuple[int, int, int]:
        return tuple(int(self.loader.get_ggml_type(f"blk.{layer_idx}.ffn_{p}_exps.weight")) for p in ("gate", "up", "down"))

    def slice(self, layer_idx: int, proj: str, N: int, rank: int, experts: torch.Tensor) -> torch.Tensor:
        """Raw bytes of this rank's TP part of the given logical experts: rows
        [N*rank, N*(rank+1)) of gate/up, the matching K-columns (block-aligned)
        of down. Returns uint8 [len(experts), rows, part_row_bytes]."""
        data, t = self._raw(f"blk.{layer_idx}.ffn_{proj}_exps.weight")
        blck, ts = BLOCK_INFO[t]
        if proj != "down":
            v = data.view(self.E, self.inter, (self.K // blck) * ts)[experts, N * rank:N * (rank + 1), :]
        else:
            v = data.view(self.E, self.K, self.inter // blck, ts)[experts, :, (N * rank) // blck:(N * (rank + 1)) // blck, :]
            v = v.reshape(len(experts), self.K, -1)
        return v.contiguous()


_readers: Dict[str, _GGUFExperts] = {}


def _reader(path: str) -> _GGUFExperts:
    r = _readers.get(path)
    if r is None:
        r = _readers[path] = _GGUFExperts(path)
    return r


class KTGGUFHotExpertsMethod(FusedMoEMethodBase):
    """GPU-expert method the KT wrapper wraps when the experts are a GGUF."""

    # at streamed prefill sizes the wrapper skips this method and the GGUF
    # streamer covers the hot slots too (its banks hold every expert); the
    # MMVQ kernel is a decode kernel
    streams_hot_experts = True

    def __init__(self, kt_config, layer):
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.reader = _reader(kt_config.weight_path)
        self.rank, self.world = layer.moe_tp_rank, layer.moe_tp_size
        # KT_GGUF_HOT_EP (on with more than one rank): whole experts on every rank, not a TP slice of each (a GGUF
        # block of 256 values cannot be cut); kt_kv_hot_lend gives each rank its own experts' routes
        self.ep = self.world > 1 and os.environ.get("KT_GGUF_HOT_EP", "1") == "1"
        self.slice_rank = 0 if self.ep else self.rank
        self.types = self.reader.types(kt_config.layer_idx)
        if self.types[0] != self.types[1]:
            raise RuntimeError(f"layer {kt_config.layer_idx}: gate {self.types[0]} and up {self.types[1]} differ")
        self.type13 = self.types[0]
        # Q2_0 downs stay 2-bit on the GPU: kt_gguf_moe_multi has a Q2_0 kernel
        # (KT_GGUF_Q2_NATIVE=0 keeps the 4x Q8_0 repack for sgl-kernel's)
        self.q2_native = os.environ.get("KT_GGUF_Q2_NATIVE", "1") == "1"
        if self.types[2] == GGML_Q2_0:
            self.type2 = GGML_Q2_0 if self.q2_native else GGML_Q8_0
        else:
            self.type2 = self.types[2]
        for t in (self.type13, self.type2):
            if t not in MMVQ_TYPES and not (t == GGML_Q2_0 and self.q2_native):
                raise RuntimeError(f"layer {kt_config.layer_idx}: ggml type {t} has no MMVQ MoE kernel")
        self.moe_runner_config = None

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        E, N, K = num_experts, intermediate_size_per_partition, hidden_size
        if self.ep:
            N = N * self.world
        self.N, self.K = N, K
        if (N if self.ep else N * self.world) != self.reader.inter or K != self.reader.K:
            raise RuntimeError(f"GGUF experts {self.reader.inter}x{self.reader.K} vs layer {N}x{self.world} / {K}")
        if E == 0:
            return
        dev = torch.device("cuda", torch.cuda.current_device())
        # the loader never touches these (no checkpoint tensor maps here);
        # process_weights_after_loading fills them from the GGUF
        w13 = Parameter(torch.empty((E, 2 * N, _row_bytes(K, self.type13)), dtype=torch.uint8, device=dev), requires_grad=False)
        w2 = Parameter(torch.empty((E, K, _row_bytes(N, self.type2)), dtype=torch.uint8, device=dev), requires_grad=False)
        w13.weight_loader = w2.weight_loader = self._refuse_loader
        layer.register_parameter("w13_qweight", w13)
        layer.register_parameter("w2_qweight", w2)

    @staticmethod
    def _refuse_loader(param, loaded_weight, *a, **k):
        raise RuntimeError("GGUF hot experts take no checkpoint expert tensors")

    def process_weights_after_loading(self, layer) -> None:
        E = layer.w13_qweight.shape[0] if hasattr(layer, "w13_qweight") else self.num_gpu_experts
        if E == 0:
            return
        from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata

        meta = get_global_expert_location_metadata()
        p2l = meta.physical_to_logical_map_cpu[self.kt_config.layer_idx][:E].to(torch.int64) if meta is not None else torch.arange(E)
        self.fill_experts(layer, p2l)

    def fill_experts(self, layer, p2l) -> None:
        """The rows of the logical experts p2l (one a slot of layer.w13_qweight / w2_qweight) from the GGUF."""
        E = int(p2l.numel())
        t0 = time.time()
        li, N = self.kt_config.layer_idx, self.N
        gate = self.reader.slice(li, "gate", N, self.slice_rank, p2l)
        up = self.reader.slice(li, "up", N, self.slice_rank, p2l)
        layer.w13_qweight.data[:, :N].copy_(gate, non_blocking=False)
        layer.w13_qweight.data[:, N:].copy_(up, non_blocking=False)
        down = self.reader.slice(li, "down", N, self.slice_rank, p2l)
        if self.types[2] == GGML_Q2_0 and self.type2 == GGML_Q8_0:
            down = q2_0_to_q8_0(down)
        assert tuple(down.shape) == tuple(layer.w2_qweight.shape), (down.shape, layer.w2_qweight.shape)
        layer.w2_qweight.data.copy_(down)
        if (li == 0 and E == self.num_gpu_experts) or os.environ.get("KT_GGUF_HOT_LOG") == "1":
            mb = (layer.w13_qweight.numel() + layer.w2_qweight.numel()) / 1e6
            _log(f"rank {self.rank}: layer {li} {E} hot experts from the GGUF, types {self.types} -> ({self.type13}, {self.type2}), "
                 f"{mb:.0f} MB on the GPU, {time.time() - t0:.1f} s")

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config

    def apply(self, layer, dispatch_output):
        from sgl_kernel import ggml_moe_a8_vec, moe_sum, silu_and_mul

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        T, top_k = topk_ids.shape
        # KV-lent slots: the weights through the per-slot pointer tables
        p13 = getattr(layer, "kt_kv_ptr13", None)
        if self.ep and self.rank != 0 and p13 is None and not layer.__dict__.get("kt_hot_slots", False):
            # the resident experts are whole on every rank: rank 0 adds them (the slots come in a shadow layer)
            return StandardCombineInput(hidden_states=torch.zeros_like(x))
        tbl = p13 is not None
        W13 = p13 if tbl else layer.w13_qweight
        W2 = layer.kt_kv_ptr2 if tbl else layer.w2_qweight
        if T <= _multi.MAX_TOKENS and _multi.enabled():
            # decode / MTP verify window: CPU routes (-1) give zero rows, so their
            # weights need no masking; the routes of one expert share one read
            # of it (kt_gguf_moe_multi.py)
            w = topk_weights
            ids = topk_ids.to(torch.int32)
            h = _multi.moe_a8_vec_multi(x, W13, ids, top_k, self.type13, 2 * self.N, T, tbl)
            h = silu_and_mul(h)
            out = _multi.moe_a8_vec_multi(h, W2, ids, 1, self.type2, self.K, T * top_k, tbl)
        elif _multi.ROUTES and _multi.enabled():
            # an extend below the streamer threshold: the per-route build of the
            # same kernel (identical sums per route to the window above; ggml's
            # IQ3_S, which sgl-kernel's ggml_moe_a8_vec decodes with the
            # pre-release grid and scale), CPU routes give zero rows
            w = topk_weights
            ids = topk_ids.to(torch.int32)
            h = _multi.moe_a8_vec_routes(x, W13, ids, top_k, self.type13, 2 * self.N, T, tbl)
            h = silu_and_mul(h)
            out = _multi.moe_a8_vec_routes(h, W2, ids, 1, self.type2, self.K, T * top_k, tbl)
        elif GGML_Q2_0 in (self.type13, self.type2):
            # sgl-kernel has no Q2_0 kernel: the window kernel, 4 tokens at a time
            # (an expert holds at most 4 routes of a window because the router's
            # top-k ids are distinct within a token)
            w = topk_weights
            M, outs = _multi.MAX_TOKENS, []
            for c0 in range(0, T, M):
                c1 = min(T, c0 + M)
                ids = topk_ids[c0:c1].to(torch.int32)
                h = _multi.moe_a8_vec_multi(x[c0:c1], W13, ids, top_k, self.type13, 2 * self.N, c1 - c0, tbl)
                h = silu_and_mul(h)
                outs.append(_multi.moe_a8_vec_multi(h, W2, ids, 1, self.type2, self.K, (c1 - c0) * top_k, tbl))
            out = torch.cat(outs)
        else:
            # the wrapper masks CPU experts to -1 and this kernel indexes by id:
            # route those to slot 0 with weight 0. It reads the ids as int32
            # without checking (int64 ids give every other route the wrong expert).
            # It addresses base + id * stride: with KV-lent slots the hot cache
            # hands this path resident-only ids (kt_gguf_moe_multi.reads_table)
            on_gpu = topk_ids >= 0
            w = torch.where(on_gpu, topk_weights, torch.zeros_like(topk_weights))
            ids = torch.where(on_gpu, topk_ids, torch.zeros_like(topk_ids)).to(torch.int32)
            h = ggml_moe_a8_vec(x, layer.w13_qweight, ids, top_k, self.type13, 2 * self.N, T)
            h = silu_and_mul(h)
            out = ggml_moe_a8_vec(h, layer.w2_qweight, ids, 1, self.type2, self.K, T * top_k)
        if os.environ.get("KT_GGUF_HOT_DEBUG") == "1" and T <= 4 and not torch.cuda.is_current_stream_capturing():
            _dbg = globals().setdefault("_DBG", [0])
            if _dbg[0] < 12:
                _dbg[0] += 1
                torch.cuda.synchronize()
                _log(f"debug layer {getattr(layer, 'layer_id', '?')} T {T} tbl {tbl} ids {topk_ids.tolist()} "
                     f"w {[round(v, 3) for v in topk_weights.float().flatten().tolist()]} "
                     f"x nan {int(x.float().isnan().sum())} |x| {float(x.float().abs().max()):.3g} "
                     f"W13 {tuple(W13.shape)} {W13.dtype} ptr {W13.data_ptr():#x} "
                     f"w13[0] bytes sum {int(layer.w13_qweight[0].view(torch.uint8).sum())} "
                     f"w2[0] bytes sum {int(layer.w2_qweight[0].view(torch.uint8).sum())} "
                     f"out nan {int(out.float().isnan().sum())} |out| {float(out.float().abs().nan_to_num().max()):.3g}")
        if FUSED_COMBINE and T <= 64:
            # one kernel for the weighting and the sum over the routes (same roundings and order as
            # out.mul_(w) in bf16 followed by moe_sum's fp32 accumulation: identical results)
            return StandardCombineInput(hidden_states=_combine(out.view(T, top_k, self.K), w.view(T, top_k)))
        out = out.reshape(T, top_k, self.K).mul_(w.view(T, top_k, 1).to(out.dtype))
        result = torch.empty_like(x)
        moe_sum(out, result)
        return StandardCombineInput(hidden_states=result)
