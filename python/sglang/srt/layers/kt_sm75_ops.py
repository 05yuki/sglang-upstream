"""sgl-kernel's CUDA ops that DeepSeek V4 GGUF serving calls, built for GPUs older than the sgl-kernel wheel's arches.

The sgl-kernel wheel carries sm_80 and newer code only (no PTX), so on Turing (SM75) its ops fail with "no kernel
image". The same sources (python/sglang/kernels/aot/csrc) are built here with torch's JIT loader for the GPU's own
arch and registered as the sgl_kernel ops' CUDA kernels in their place; the Python call sites stay as they are.
Nothing is built or registered on SM80 and newer.
"""

import logging
import os
import re
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_AOT = Path(__file__).resolve().parents[2] / "kernels" / "aot"
_SOURCES = (
    "csrc/quantization/gguf/gguf_kernel.cu",
    "csrc/elementwise/activation.cu",
    "csrc/elementwise/pos_enc.cu",
    "csrc/moe/moe_sum.cu",
    "csrc/moe/moe_topk_sigmoid_kernels.cu",
    "csrc/moe/moe_topk_softmax_kernels.cu",
)
_OPS = (
    "silu_and_mul",
    "gelu_and_mul",
    "rotary_embedding",
    "topk_softmax",
    "topk_sigmoid",
    "moe_sum",
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
)
_BIND = """#include <torch/all.h>
#include <torch/library.h>
#include "sgl_kernel_ops.h"

TORCH_LIBRARY_IMPL(sgl_kernel, CUDA, m) {
%s
}
""" % "\n".join(f'  m.impl("{op}", &{op});' for op in _OPS)

_loaded = False


def needed() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() < (8, 0)
        and os.environ.get("KT_SM75_OPS", "1") == "1"
    )


def _load():
    from torch.utils.cpp_extension import _get_build_directory, load

    import sgl_kernel  # noqa: F401  (the op schemas this registers kernels for)

    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}{minor}"
    name = f"kt_sm75_ops_sm{arch}"
    build = Path(_get_build_directory(name, verbose=False))
    bind = build / "bind.cc"
    if not bind.exists() or bind.read_text() != _BIND:
        bind.write_text(_BIND)
    try:
        import flashinfer

        fi_include = [str(Path(flashinfer.__file__).parent / "data" / "include")]
    except Exception:
        fi_include = []
    sources = [str(bind)]
    for s in _SOURCES:
        if s.endswith("activation.cu"):
            # the installed FlashInfer's act_and_mul_kernel takes its vector width as a third template argument and
            # is bounded to 256 threads a block (sgl-kernel's wheel was built against one without either): a copy
            # with the width the old kernel used and blocks of at most 256 (the kernel strides over the row)
            text = re.sub(
                r"flashinfer::activation::act_and_mul_kernel<c_type, (\w+)>",
                r"flashinfer::activation::act_and_mul_kernel<c_type, \1, 16 / sizeof(c_type)>",
                (_AOT / s).read_text(),
            ).replace("std::min(d / vec_size, 1024U)", "std::min(d / vec_size, 256U)")
            copy = build / "activation.cu"
            if not copy.exists() or copy.read_text() != text:
                copy.write_text(text)
            sources.append(str(copy))
        else:
            sources.append(str(_AOT / s))
    return load(
        name=name,
        sources=sources,
        extra_include_paths=[
            str(_AOT / "include"),
            str(_AOT / "csrc"),
            str(_AOT / "csrc" / "quantization" / "gguf"),
            str(_AOT / "csrc" / "moe"),
        ]
        + fi_include,
        extra_cflags=["-O3", "-std=c++20"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            # as sgl-kernel's own build (its CMakeLists): the fp16 / bf16 dispatch, and the half / bf16 conversions
            # torch's JIT turns off
            "-DNDEBUG",
            "-DFLASHINFER_ENABLE_F16",
            "-DFLASHINFER_ENABLE_BF16",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        ],
        is_python_module=False,  # the library only registers kernels for the sgl_kernel ops
        verbose=os.environ.get("KT_SM75_OPS_VERBOSE") == "1",
    )


def load_if_needed() -> None:
    """Build (once, cached in torch's extensions directory) and register the kernels when the GPU needs them."""
    global _loaded
    if _loaded or not needed():
        return
    _load()
    _loaded = True
    logger.info("sgl-kernel ops built for sm_%d%d: %s", *torch.cuda.get_device_capability(), ", ".join(_OPS))
