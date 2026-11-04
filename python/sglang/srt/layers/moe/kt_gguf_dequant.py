"""GGUF block dequantization on the GPU (Triton): IQ2_S, IQ3_XXS, IQ3_S,
IQ4_XS, IQ4_NL and Q2_0 blocks to bf16 rows, for streaming GGUF experts
through the W16 prefill path.

Block layouts follow ggml-common.h / dequantize_row_* in ggml-quants.c
(vendored a94e6ff87) and the upstream Q2_0 (type 42, 64-wide, codes -1..2).
One program handles one block of one expert; raw bytes are expert-major with
a byte stride, the output is [n, rows, cols] bf16 with contiguous rows, and a
block covers BLCK consecutive elements of the flattened expert matrix.
"""
import torch
import triton
import triton.language as tl

from sglang.srt.layers.moe import kt_gguf_tables as T

QK_K = 256

GGML_IQ2_S = 22
GGML_IQ3_XXS = 18
GGML_IQ3_S = 21
GGML_IQ4_XS = 23
GGML_IQ4_NL = 20
GGML_Q2_0 = 42

# type id -> (block elements, block bytes)
BLOCK_INFO = {
    GGML_IQ2_S: (256, 82),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ3_S: (256, 110),
    GGML_IQ4_XS: (256, 136),
    GGML_IQ4_NL: (32, 18),
    GGML_Q2_0: (64, 18),
}

_tables = {}


def tables(device):
    t = _tables.get(device)
    if t is None:
        u8 = torch.uint8
        t = {
            "iq2s": torch.tensor(T.IQ2S_GRID_BYTES, dtype=u8, device=device),
            "iq3xxs": torch.tensor(T.IQ3XXS_GRID_BYTES, dtype=u8, device=device),
            "iq3s": torch.tensor(T.IQ3S_GRID_BYTES, dtype=u8, device=device),
            "ksigns": torch.tensor(T.KSIGNS_IQ2XS, dtype=u8, device=device),
            "kvalues": torch.tensor(T.KVALUES_IQ4NL, dtype=torch.int8, device=device),
        }
        _tables[device] = t
    return t


@triton.jit
def _d16(ptr):
    # the block's fp16 scale at its first two bytes
    return tl.load(ptr.to(tl.pointer_type(tl.float16))).to(tl.float32)


@triton.jit
def _iq2_s_kernel(raw_ptr, out_ptr, grid_ptr, raw_stride, out_stride, TYPE_SIZE: tl.constexpr):
    e = tl.program_id(1)
    b = tl.program_id(0)
    base = raw_ptr + e * raw_stride + b * TYPE_SIZE
    d = _d16(base)
    j = tl.arange(0, 256)
    ib32 = j // 32
    l = (j % 32) // 8
    jj = j % 8
    qs = tl.load(base + 2 + ib32 * 4 + l).to(tl.int32)
    sign = tl.load(base + 2 + 32 + ib32 * 4 + l).to(tl.int32)
    qh = tl.load(base + 2 + 64 + ib32).to(tl.int32)
    sc = tl.load(base + 2 + 72 + ib32).to(tl.int32)
    idx = qs | ((qh << (8 - 2 * l)) & 0x300)
    g = tl.load(grid_ptr + idx * 8 + jj).to(tl.float32)
    nib = tl.where(l < 2, sc & 0xF, sc >> 4).to(tl.float32)
    db = d * (0.5 + nib) * 0.25
    neg = (sign & (tl.full([256], 1, tl.int32) << jj)) != 0
    y = tl.where(neg, -db * g, db * g)
    tl.store(out_ptr + e * out_stride + b * 256 + j, y.to(tl.bfloat16))


@triton.jit
def _iq3_xxs_kernel(raw_ptr, out_ptr, grid_ptr, ksigns_ptr, raw_stride, out_stride, TYPE_SIZE: tl.constexpr):
    e = tl.program_id(1)
    b = tl.program_id(0)
    base = raw_ptr + e * raw_stride + b * TYPE_SIZE
    d = _d16(base)
    j = tl.arange(0, 256)
    ib32 = j // 32
    l = (j % 32) // 8
    jj = j % 8
    a0 = tl.load(base + 66 + ib32 * 4 + 0).to(tl.int32)
    a1 = tl.load(base + 66 + ib32 * 4 + 1).to(tl.int32)
    a2 = tl.load(base + 66 + ib32 * 4 + 2).to(tl.int32)
    a3 = tl.load(base + 66 + ib32 * 4 + 3).to(tl.int32)
    aux = a0 | (a1 << 8) | (a2 << 16) | (a3 << 24)
    db = d * (0.5 + (a3 >> 4).to(tl.float32)) * 0.5
    q3 = tl.load(base + 2 + ib32 * 8 + 2 * l + (jj // 4)).to(tl.int32)
    g = tl.load(grid_ptr + q3 * 4 + (jj % 4)).to(tl.float32)
    s = tl.load(ksigns_ptr + ((aux >> (7 * l)) & 127)).to(tl.int32)
    neg = (s & (tl.full([256], 1, tl.int32) << jj)) != 0
    y = tl.where(neg, -db * g, db * g)
    tl.store(out_ptr + e * out_stride + b * 256 + j, y.to(tl.bfloat16))


@triton.jit
def _iq3_s_kernel(raw_ptr, out_ptr, grid_ptr, raw_stride, out_stride, TYPE_SIZE: tl.constexpr):
    e = tl.program_id(1)
    b = tl.program_id(0)
    base = raw_ptr + e * raw_stride + b * TYPE_SIZE
    d = _d16(base)
    j = tl.arange(0, 256)
    ib32 = j // 32
    l = (j % 32) // 8
    jj = j % 8
    sub = jj // 4
    qs = tl.load(base + 2 + ib32 * 8 + 2 * l + sub).to(tl.int32)
    qh = tl.load(base + 66 + ib32).to(tl.int32)
    idx = qs | ((qh << (8 - 2 * l - sub)) & 256)
    g = tl.load(grid_ptr + idx * 4 + (jj % 4)).to(tl.float32)
    sign = tl.load(base + 74 + ib32 * 4 + l).to(tl.int32)
    sb = tl.load(base + 106 + ib32 // 2).to(tl.int32)
    nib = tl.where((ib32 % 2) == 0, sb & 0xF, sb >> 4).to(tl.float32)
    db = d * (1.0 + 2.0 * nib)
    neg = (sign & (tl.full([256], 1, tl.int32) << jj)) != 0
    y = tl.where(neg, -db * g, db * g)
    tl.store(out_ptr + e * out_stride + b * 256 + j, y.to(tl.bfloat16))


@triton.jit
def _iq4_xs_kernel(raw_ptr, out_ptr, kvalues_ptr, raw_stride, out_stride, TYPE_SIZE: tl.constexpr):
    e = tl.program_id(1)
    b = tl.program_id(0)
    base = raw_ptr + e * raw_stride + b * TYPE_SIZE
    d = _d16(base)
    j = tl.arange(0, 256)
    ib = j // 32
    jj = j % 32
    q = tl.load(base + 8 + ib * 16 + (jj % 16)).to(tl.int32)
    nib = tl.where(jj < 16, q & 0xF, q >> 4)
    sl = tl.load(base + 4 + ib // 2).to(tl.int32)
    sl = tl.where((ib % 2) == 0, sl & 0xF, sl >> 4)
    sh = tl.load(base + 2).to(tl.int32) | (tl.load(base + 3).to(tl.int32) << 8)
    ls = sl | (((sh >> (2 * ib)) & 3) << 4)
    dl = d * (ls - 32).to(tl.float32)
    v = tl.load(kvalues_ptr + nib).to(tl.float32)
    tl.store(out_ptr + e * out_stride + b * 256 + j, (dl * v).to(tl.bfloat16))


@triton.jit
def _iq4_nl_kernel(raw_ptr, out_ptr, kvalues_ptr, raw_stride, out_stride, TYPE_SIZE: tl.constexpr):
    # eight 32-wide blocks per program
    e = tl.program_id(1)
    b = tl.program_id(0)
    j = tl.arange(0, 256)
    bl = b * 8 + j // 32
    jl = j % 32
    base = raw_ptr + e * raw_stride + bl * TYPE_SIZE
    d = tl.load(base.to(tl.pointer_type(tl.float16))).to(tl.float32)
    q = tl.load(base + 2 + (jl % 16)).to(tl.int32)
    nib = tl.where(jl < 16, q & 0xF, q >> 4)
    v = tl.load(kvalues_ptr + nib).to(tl.float32)
    tl.store(out_ptr + e * out_stride + b * 256 + j, (d * v).to(tl.bfloat16))


@triton.jit
def _q2_0_kernel(raw_ptr, out_ptr, raw_stride, out_stride, TYPE_SIZE: tl.constexpr):
    # four 64-wide blocks per program
    e = tl.program_id(1)
    b = tl.program_id(0)
    j = tl.arange(0, 256)
    bl = b * 4 + j // 64
    jl = j % 64
    base = raw_ptr + e * raw_stride + bl * TYPE_SIZE
    d = tl.load(base.to(tl.pointer_type(tl.float16))).to(tl.float32)
    q = tl.load(base + 2 + jl // 4).to(tl.int32)
    code = ((q >> (2 * (jl % 4))) & 3).to(tl.float32) - 1.0
    tl.store(out_ptr + e * out_stride + b * 256 + j, (d * code).to(tl.bfloat16))


def dequant_experts(raw: torch.Tensor, raw_stride: int, type_id: int, n: int, rows: int, cols: int,
                    out: torch.Tensor) -> None:
    """raw: uint8 device tensor, expert-major, raw_stride bytes per expert
    (expert bytes = rows * cols / blck * type_size, padding allowed);
    out: [n, rows, cols] bf16 with contiguous rows (out.stride(0) elements
    between experts). Runs on the current stream."""
    blck, tsize = BLOCK_INFO[type_id]
    assert cols % blck == 0, (type_id, cols, blck)
    assert out.stride(1) == cols and out.stride(2) == 1
    assert raw_stride % 2 == 0
    assert (rows * cols) % 256 == 0, (rows, cols)
    # every kernel writes 256 elements per program
    grid = (rows * cols // 256, n)
    t = tables(out.device)
    if type_id == GGML_IQ2_S:
        _iq2_s_kernel[grid](raw, out, t["iq2s"], raw_stride, out.stride(0), TYPE_SIZE=tsize)
    elif type_id == GGML_IQ3_XXS:
        _iq3_xxs_kernel[grid](raw, out, t["iq3xxs"], t["ksigns"], raw_stride, out.stride(0), TYPE_SIZE=tsize)
    elif type_id == GGML_IQ3_S:
        _iq3_s_kernel[grid](raw, out, t["iq3s"], raw_stride, out.stride(0), TYPE_SIZE=tsize)
    elif type_id == GGML_IQ4_XS:
        _iq4_xs_kernel[grid](raw, out, t["kvalues"], raw_stride, out.stride(0), TYPE_SIZE=tsize)
    elif type_id == GGML_IQ4_NL:
        _iq4_nl_kernel[grid](raw, out, t["kvalues"], raw_stride, out.stride(0), TYPE_SIZE=tsize)
    elif type_id == GGML_Q2_0:
        _q2_0_kernel[grid](raw, out, raw_stride, out.stride(0), TYPE_SIZE=tsize)
    else:
        raise ValueError(f"no GPU dequantizer for ggml type {type_id}")
