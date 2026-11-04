"""Dense (non-expert) weights of a DeepSeek-V4.1 GGUF (llama.cpp arch "deepseek41") as the checkpoint's own
tensor names, so DeepseekV4ForCausalLM.load_weights loads a GGUF the way it loads the safetensors checkpoint.

The converter changed no values (every F32 / BF16 tensor is bit-identical to the
checkpoint) and one shape: wo_a comes as [8, 1024, 4096] for [8192, 4096]. The pack's llama.cpp layout (2026-10-27
and later) names engram_wkv engram_kv, keeps wo_a 2D, and folds engram_q and engram_k into the F32 engram_gate
(q * k, exact), which is split back into a bf16 pair with exactly that product (_split_gate) and loaded as
engram.q_weight / engram.k_weight; its vision router bias (exp_probs_b_vl) is skipped. Quantized tensors (Q5_K, Q8_0, ...)
are dequantized to bf16 with gguf-py; F32 / BF16 are viewed. Not yielded: the routed experts
(blk.N.ffn_*_exps, kt-kernel reads them from the same GGUF) and the engram tables (the engram adapter reads its
rows itself). KT_GGUF_DENSE_THREADS=<n> dequantizes the big tensors in row blocks on n threads.

With a model whose layers carry SGLang's GGUF linear method (quantization_config {"quant_method": "gguf",
"modules_to_not_convert": [...]}), a quantized tensor whose layer has a .qweight parameter is yielded as its
GGUF bytes (<name>.qweight, [rows, row bytes] uint8) and type (<name>.qweight_type), so it stays quantized on the
GPU; everything else is dequantized as above.
"""
import glob
import os
import re
from typing import Iterator, Tuple

import numpy as np
import torch

import gguf.quants
from gguf import GGUFReader

_LAYER = {
    "attn_norm": "attn_norm.weight",
    "ffn_norm": "ffn_norm.weight",
    "attn_q_a": "attn.wq_a.weight",
    "attn_q_a_norm": "attn.q_norm.weight",
    "attn_q_b": "attn.wq_b.weight",
    "attn_kv": "attn.wkv.weight",
    "attn_kv_a_norm": "attn.kv_norm.weight",
    "attn_output_a": "attn.wo_a.weight",
    "attn_output_b": "attn.wo_b.weight",
    "attn_sinks": "attn.attn_sink",
    "attn_compressor_kv": "attn.compressor.wkv.weight",
    "attn_compressor_gate": "attn.compressor.wgate.weight",
    "attn_compressor_norm": "attn.compressor.norm.weight",
    "indexer.attn_q_b": "attn.indexer.wq_b.weight",
    "indexer.attn_k": "attn.indexer.wk.weight",
    "indexer.k_norm": "attn.indexer.k_norm.weight",
    "indexer.proj": "attn.indexer.weights_proj.weight",
    "ffn_gate_inp": "ffn.gate.weight",
    "exp_probs_b": "ffn.gate.bias",
    "ffn_gate_shexp": "ffn.shared_experts.w1.weight",
    "ffn_down_shexp": "ffn.shared_experts.w2.weight",
    "ffn_up_shexp": "ffn.shared_experts.w3.weight",
    "hc_attn_base": "hc_attn_base",
    "hc_attn_fn": "hc_attn_fn",
    "hc_attn_scale": "hc_attn_scale",
    "hc_ffn_base": "hc_ffn_base",
    "hc_ffn_fn": "hc_ffn_fn",
    "hc_ffn_scale": "hc_ffn_scale",
    "engram_q": "engram.q_weight",
    "engram_k": "engram.k_weight",
    "engram_wkv": "engram.wkv.weight",
    # the llama.cpp layout
    "engram_kv": "engram.wkv.weight",
    "engram_gate": "engram.gate",  # split into q_weight and k_weight (_split_gate)
}
_GLOBAL = {"token_embd": "embed.weight", "output": "head.weight", "output_norm": "norm.weight"}
_SKIP = ("_exps", "engram_embd", "exp_probs_b_vl")


def checkpoint_name(gguf_name: str):
    """The checkpoint name of a GGUF tensor, or None for what this source does not serve."""
    stem = re.sub(r"\.(weight|bias)$", "", gguf_name)
    if stem in _GLOBAL:
        return _GLOBAL[stem]
    m = re.match(r"blk\.(\d+)\.(.+)$", stem)
    if not m:
        return None  # engram hash tables (engram_token_map / primes / offsets / multipliers): config, not weights
    layer, key = int(m.group(1)), m.group(2)
    if key.endswith(_SKIP):
        return None
    if key not in _LAYER:
        raise KeyError(f"no checkpoint name for GGUF tensor {gguf_name}")
    return f"layers.{layer}.{_LAYER[key]}"


def _to_torch(t) -> torch.Tensor:
    shape = tuple(int(x) for x in reversed(t.shape.tolist()))  # ggml order -> torch order
    kind = t.tensor_type.name
    raw = np.asarray(t.data)
    if kind == "F32":
        return torch.from_numpy(raw.view(np.float32).reshape(shape).copy())
    if kind == "F16":
        return torch.from_numpy(raw.view(np.float16).reshape(shape).copy()).to(torch.bfloat16)
    if kind == "BF16":
        return torch.from_numpy(raw.view(np.int16).reshape(shape).copy()).view(torch.bfloat16)
    threads = int(os.environ.get("KT_GGUF_DENSE_THREADS", "8") or 1)
    rows = raw.reshape(-1, raw.shape[-1])  # [..., row bytes]: blocks never cross a row
    if threads > 1 and rows.shape[0] >= 4 * threads and raw.nbytes >= (32 << 20):
        from concurrent.futures import ThreadPoolExecutor

        parts = np.array_split(rows, 4 * threads, axis=0)
        with ThreadPoolExecutor(max_workers=threads) as ex:
            f = np.concatenate(list(ex.map(lambda r: gguf.quants.dequantize(r, t.tensor_type), parts)), axis=0)
    else:
        f = gguf.quants.dequantize(rows, t.tensor_type)
    return torch.from_numpy(f.reshape(shape)).to(torch.bfloat16)


_STACKED = ((".gate_proj.", ".gate_up_proj."), (".up_proj.", ".gate_up_proj."))


def _quantized_params(model):
    """Checkpoint name -> True for the tensors whose layer keeps GGUF bytes (has a .qweight)."""
    if model is None:
        return lambda name: False
    params = {n for n, _ in model.named_parameters()}
    remap = getattr(type(model), "remap_weight_name_to_dpsk_hf_format", None)
    if remap is None:
        return lambda name: False

    def quantized(name):
        if not name.endswith(".weight"):
            return False
        pname = remap(name)[: -len(".weight")] + ".qweight"
        if pname in params:
            return True
        return any(a in pname and pname.replace(a, b) in params for a, b in _STACKED)

    return quantized


def _split_gate(g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """bf16 q, k with q * k == g exactly in fp32. g is a product of two bf16 values, so |g| = n * 2^E with n a product
    of two integers below 256; any such pair of factors serves (the Engram gate multiplies q and k in fp32 first)."""
    g = g.float().contiguous()
    nz = g != 0
    a = g.abs().masked_fill(~nz, 1.0)
    _, e = torch.frexp(a)
    n = torch.ldexp(a, 16 - e).round().long()  # 16-bit integer significand, exact: g has at most 16 significant bits
    E = e - 16
    tz = (n & -n).float().log2().long()  # strip the powers of two into the exponent
    n, E = n >> tz, E + tz
    d = torch.arange(1, 256, dtype=torch.long)
    ok = (n.reshape(-1, 1) % d == 0) & (n.reshape(-1, 1) // d <= 255)
    assert bool(ok.any(dim=1).all()), "engram_gate is not a product of two bf16 values"
    qa = d[ok.float().argmax(dim=1)].reshape(n.shape)
    kb = n // qa
    q = (torch.ldexp(qa.float(), E) * torch.sign(g)).masked_fill(~nz, 0.0).to(torch.bfloat16)
    k = kb.float().masked_fill(~nz, 1.0).to(torch.bfloat16)
    assert torch.equal(q.float() * k.float(), g), "engram_gate split is not exact"
    return q, k


def gguf_v41_dense_weights(gguf_dir: str, model=None) -> Iterator[Tuple[str, torch.Tensor]]:
    shards = sorted(glob.glob(os.path.join(gguf_dir, "*-000??-of-000??.gguf")))
    if not shards:
        raise FileNotFoundError(f"no GGUF shards (*-0000N-of-0000M.gguf) in {gguf_dir}")
    quantized = _quantized_params(model)
    for path in shards:
        for t in GGUFReader(path).tensors:
            name = checkpoint_name(t.name)
            if name is None:
                continue
            if quantized(name):
                base = name[: -len(".weight")]
                if t.tensor_type.name in ("F32", "F16", "BF16"):
                    # a GGUF linear whose weight the GGUF keeps unquantized: the bf16 matrix itself, typed BF16
                    # (SGLang's GGUF method multiplies those directly)
                    yield base + ".qweight_type", torch.tensor(30, dtype=torch.uint8)
                    yield base + ".qweight", _to_torch(t).to(torch.bfloat16)
                    continue
                raw = torch.from_numpy(np.asarray(t.data).reshape(-1, np.asarray(t.data).shape[-1]).copy())
                yield base + ".qweight_type", torch.tensor(int(t.tensor_type), dtype=torch.uint8)
                yield base + ".qweight", raw
                continue
            w = _to_torch(t)
            if name.endswith(".engram.gate"):
                q, k = _split_gate(w)
                yield name[: -len("gate")] + "q_weight", q
                yield name[: -len("gate")] + "k_weight", k
                continue
            if name.endswith("attn.wo_a.weight") and w.dim() == 3:
                w = w.reshape(-1, w.shape[-1])  # [8, 1024, 4096] -> [8192, 4096]
            yield name, w
    if model is not None:
        # (runs once the loader has taken the last tensor) parameters nothing filled: a name map gap
        left = [n for n, prm in model.named_parameters() if type(prm).__name__.endswith("UninitializedParameter")]
        if left:
            print(f"[kt-gguf-dense] {len(left)} parameters left uninitialized, e.g. {left[:8]}", flush=True)
