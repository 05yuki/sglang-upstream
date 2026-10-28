"""llama.cpp's int8-mma matmul on raw GGUF blocks (MMQ) as a small torch extension next to sgl-kernel (which it
does not replace): MUL_MAT_ID for the GGUF experts (Q2_0, Q8_0, IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M) and the MXFP4 checkpoint's streamed
experts (MXFP4, regrouped into ggml blocks by kt_stream_prefill.py), and the plain MUL_MAT for the dense layers'
prefill (also Q5_K).

`mul_mat(X, W, type, out)` writes out[M, rows] = X @ W^T in out's dtype (fp32 accumulation, one rounding);
X and out may be row views (unit column stride), W [rows, row_bytes] must be followed in its storage by
`tail_bytes(type, K)` readable bytes (zero them at load).

`moe_mmq(X, W, topk_ids, type, rows, x_per_slot)` is ggml_moe_mmq of our SGLang
branch perf/gguf-mmq-q2_0 (5ac69dd): out[t, s, :] = W[topk_ids[t, s]] @ x(t, s)
in fp32, x(t, s) = X[t] (gate/up, each token quantized once) or
X[t * top_k + s] (x_per_slot, down). The routes are sorted by expert on the GPU,
ids < 0 or >= n_experts give zero rows, any expert id below W.shape[0] works
(sgl-kernel's dp4a ggml_moe_a8 skips ids > 255).

W is [n_experts, rows, row_bytes] of raw GGUF blocks whose storage extends
`tail_bytes(type, K)` zero bytes past the last row of the last expert (ggml's
row padding to 512 values: 108 bytes for Q2_0 at K = 640, 54 at K = 320);
without them the op copies the whole weight on every call.

Sources: kt_gguf_moe_mmq/ (llama.cpp 23b0202, MIT; its
ggml/ is fetched and patched by the recipe's lib/llama-mmq.sh, see SOURCES). Built once for the GPU's arch with torch's JIT loader into the
torch extensions directory (the TP ranks share it under torch's file lock);
`python kt_gguf_moe_mmq.py` builds it ahead of a launch.
"""
import os
from pathlib import Path

import torch


def _arch():
    # the GPU's own arch, with the "a" target where it has one (as setup.sh builds FlashInfer)
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}" + ("a" if (major, minor) in ((9, 0), (10, 0), (10, 3), (12, 0), (12, 1)) else "")


ARCH = os.environ.get("KT_GGUF_MOE_MMQ_ARCH") or _arch()
_ext = None

FLAGS = [
    "-O3", "-DNDEBUG", "-use_fast_math", "--expt-relaxed-constexpr", "--expt-extended-lambda",
    "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    # ggml's symbols stay inside this library (kt-kernel carries its own ggml)
    "-Xcompiler", "-fvisibility=hidden", "-Xcompiler", "-Wno-unused-function",
]


def _load():
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load

        d = Path(__file__).with_name("kt_gguf_moe_mmq")
        lm = d / "llama_mmq"
        cu = lm / "ggml/src/ggml-cuda"
        _ext = load(
            name=f"kt_gguf_moe_mmq_{ARCH}",
            sources=[str(d / "kt_gguf_moe_mmq_bind.cpp"), str(d / "kt_gguf_moe_mmq.cu"), str(lm / "ggml_shim.cu"),
                     str(lm / "mmq_types.cu"), str(lm / "mmq_moe.cu"), str(lm / "quantize_mmq_q8_1.cu"),
                     str(cu / "mmid.cu"), str(cu / "template-instances/mmq-instance-q2_0.cu"),
                     str(cu / "template-instances/mmq-instance-q8_0.cu"),
                     str(cu / "template-instances/mmq-instance-q5_k.cu"),
                     str(cu / "template-instances/mmq-instance-iq2_xxs.cu"),
                     str(cu / "template-instances/mmq-instance-iq2_xs.cu"),
                     str(cu / "template-instances/mmq-instance-iq1_s.cu"),
                     str(cu / "template-instances/mmq-instance-iq1_m.cu"),
                     str(cu / "template-instances/mmq-instance-mxfp4.cu")],
            extra_include_paths=[str(d), str(lm), str(lm / "ggml/include"), str(lm / "ggml/src"), str(cu)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=FLAGS + [f"-gencode=arch=compute_{ARCH},code=sm_{ARCH}"],
            verbose=os.environ.get("KT_GGUF_MOE_MMQ_VERBOSE") == "1",
        )
    return _ext


def moe_mmq(X, W, topk_ids, qtype: int, rows: int, x_per_slot: bool) -> torch.Tensor:
    return _load().ggml_moe_mmq(X, W, topk_ids, qtype, rows, x_per_slot)


def moe_mmq_sorted(X, W, slot_expert, bounds, qtype: int, rows: int) -> torch.Tensor:
    """X [n, K] rows already sorted by expert; slot_expert [n] int32 (their experts, non-decreasing); bounds
    [n_experts + 1] int32 on the device. No ids helper (it cost as much as the GEMMs, 10-25)."""
    return _load().ggml_moe_mmq_sorted(X, W, slot_expert, bounds, qtype, rows)


def mul_mat(X, W, qtype: int, out, qbuf=None) -> None:
    _load().ggml_mul_mat_mmq(X, W, qtype, out, qbuf)


def q8_bytes(M: int, K: int) -> int:
    """Bytes of the q8_1 buffer mul_mat needs for M rows of K values."""
    return int(_load().ggml_mul_mat_mmq_q8_bytes(M, K))


def tail_bytes(qtype: int, K: int) -> int:
    return int(_load().ggml_moe_mmq_tail_bytes(qtype, K))


if __name__ == "__main__":
    m = _load()
    print("built", m.__file__, "tail Q2_0 K=640:", m.ggml_moe_mmq_tail_bytes(42, 640),
          "K=320:", m.ggml_moe_mmq_tail_bytes(42, 320), "K=2560:", m.ggml_moe_mmq_tail_bytes(42, 2560))
