"""Multi-token GGUF MoE GEMV for the KT hot experts (kt_gguf_hot.py).

`moe_a8_vec_multi(X, W, ids, top_k, type, rows, tokens)` has the signature of
sgl_kernel.ggml_moe_a8_vec, but ids < 0 (experts on the CPU) produce zero rows
instead of being computed, and the routes of one expert in the window share one
read of its weights (kt_gguf_moe_multi/kt_gguf_moe_multi.cu).
Windows of up to KT_GGUF_MOE_MAXM tokens (4); larger batches keep the
original kernel. KT_GGUF_MOE_MMV_Y warps per block (2; sgl-kernel uses 1), KT_GGUF_MOE_RPW
rows per warp (4; sgl-kernel 1).

`moe_a8_vec_routes(...)` (same signature) is the per-route build of the same
kernel (KT_PER_ROUTE=1): any number of tokens, one route per row block, zero rows
for CPU routes; for an extend above the window the hot method uses it instead of
sgl_kernel.ggml_moe_a8_vec, whose IQ3_S vec_dot still decodes the pre-release
grid and scale (2x the error on the IQ3_S layers, tools/hot-numerics/).

Built with torch's load_inline on first use (the TP ranks share the build
directory; torch serializes the build with a file lock).

With table=True, W is not the slot tensor but an int64 table on the device with
one weight address per slot (the build's KT_PTR_TABLE=1): the KV-lent hot
experts (kt_hot_kv_lend.py) live in borrowed KV pages, not at base + slot *
stride. The table has a fixed address, so captured graphs stay valid while its
entries change between steps; the sums are the same as the plain build's.

dense_a8_vec(X, W, type) runs a dense GGUF linear ([rows, row_bytes]) as a
one-expert MoE for windows of 2-4 tokens, so each weight row is read once for
all of them (sgl-kernel's MMVQ runs one block per row and token). It uses its
own build with KT_GGUF_DENSE_MMV_Y=4 warps per block and KT_GGUF_DENSE_RPW=1
row per warp: on the Flash-Next GGUF's dense shards at 2 tokens, Q6_K 5120x2560
22.8 -> 16.1 us, IQ4_XS 6144x2560 23.2 -> 16.0 us, Q6_K 2560x3072 14.3 -> 10.4 us,
IQ4_NL 2560x320 5.1 -> 3.5 us, bit-identical (tools/bench_gguf_dense_multi.py);
matrices of 512 rows or fewer stay on MMVQ (320-row ones were faster there).
"""
import os
from pathlib import Path

import torch

MAX_TOKENS = int(os.environ.get("KT_GGUF_MOE_MAXM", "4"))
MMV_Y = int(os.environ.get("KT_GGUF_MOE_MMV_Y", "2"))  # warps per block
RPW = int(os.environ.get("KT_GGUF_MOE_RPW", "4"))  # output rows per warp
DENSE = os.environ.get("KT_GGUF_DENSE_MULTI", "1") == "1"
DENSE_MMV_Y = int(os.environ.get("KT_GGUF_DENSE_MMV_Y", "4"))
DENSE_RPW = int(os.environ.get("KT_GGUF_DENSE_RPW", "1"))
DENSE_MIN_ROWS = 513
# KT_GGUF_MOE_ROUTES=0: extends above the window go back to sgl-kernel's ggml_moe_a8_vec
ROUTES = os.environ.get("KT_GGUF_MOE_ROUTES", "1") == "1"
# ggml types the kernel has a vec_dot for (dense layers use K-quants and IQ4)
DENSE_TYPES = {2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 22, 23}
_exts = {}
_zero_ids = {}


def _load(mmv_y=None, rpw=None, per_route=False, table=False):
    mmv_y = MMV_Y if mmv_y is None else mmv_y
    rpw = RPW if rpw is None else rpw
    key = (MAX_TOKENS, mmv_y, rpw, per_route, table)
    if key not in _exts:
        from torch.utils.cpp_extension import load_inline

        d = Path(__file__).with_name("kt_gguf_moe_multi")
        _exts[key] = load_inline(
            f"kt_gguf_moe_multi_m{MAX_TOKENS}_y{mmv_y}_r{rpw}" + ("_routes" if per_route else "") + ("_ptr" if table else ""),
            cpp_sources="torch::Tensor kt_moe_a8_vec_multi(torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids,"
                        " int64_t top_k, int64_t type, int64_t row, int64_t tokens);",
            cuda_sources=(d / "kt_gguf_moe_multi.cu").read_text(),
            functions=["kt_moe_a8_vec_multi"],
            extra_cuda_cflags=["-O3", f"-I{d}", f"-DKT_MAXM={MAX_TOKENS}", f"-DKT_MMV_Y={mmv_y}", f"-DKT_RPW={rpw}",
                               f"-DKT_PER_ROUTE={int(per_route)}", f"-DKT_PTR_TABLE={int(table)}"],
            extra_include_paths=[str(d)],
            verbose=False,
        )
    return _exts[key]


def enabled() -> bool:
    return os.environ.get("KT_GGUF_MOE_MULTI", "1") == "1"


def moe_a8_vec_multi(X, W, ids, top_k, qtype, rows, tokens, table=False):
    return _load(table=table).kt_moe_a8_vec_multi(X, W, ids, top_k, qtype, rows, tokens)


def moe_a8_vec_routes(X, W, ids, top_k, qtype, rows, tokens, table=False):
    """ggml_moe_a8_vec's contract at any token count: route p's rows from expert ids[p], zeros for ids < 0."""
    return _load(per_route=True, table=table).kt_moe_a8_vec_multi(X, W, ids, top_k, qtype, rows, tokens)


def reads_table(tokens: int) -> bool:
    """Whether the hot method's kernel for a window of this many tokens reads
    the slot weights through the pointer table (the window kernel and its
    per-route build do; sgl-kernel's ggml_moe_a8_vec, the fallback with
    KT_GGUF_MOE_ROUTES=0, does not and must never see a KV-lent slot id)."""
    return enabled() and (tokens <= MAX_TOKENS or ROUTES)


def dense_ok(x, qweight, qtype) -> bool:
    return (DENSE and 2 <= x.shape[0] <= MAX_TOKENS and qtype in DENSE_TYPES and qweight.dim() == 2
            and qweight.shape[0] >= DENSE_MIN_ROWS and qweight.dtype == torch.uint8 and qweight.is_contiguous())


def dense_a8_vec(x, qweight, qtype):
    """y = x @ W^T for a GGUF weight [rows, row_bytes], each row read once for all tokens."""
    T = x.shape[0]
    rows = qweight.shape[0]
    ids = _zero_ids.get((x.device, T))
    if ids is None:
        ids = _zero_ids[(x.device, T)] = torch.zeros(T, dtype=torch.int32, device=x.device)
    y = _load(DENSE_MMV_Y, DENSE_RPW).kt_moe_a8_vec_multi(x.contiguous(), qweight.view(1, rows, qweight.shape[1]), ids,
                                                          1, qtype, rows, T)
    return y.view(T, rows)
