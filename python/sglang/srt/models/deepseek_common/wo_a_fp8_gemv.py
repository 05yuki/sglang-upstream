"""V4 wo_a from FP8 weights (patches/sglang-dsv4-wo-a-fp8-gemv.py,
patches/sglang-dsv4-wo-a-fp8-block.py)."""

import torch
import triton
import triton.language as tl


def fp8_blocks_from_bf16(w: torch.Tensor, blocks=(128, 32)):
    """[G, R, D] bf16 -> ([G, R, D] e4m3, [G, R/b, D/b] fp32) by the converter's rule
    (scale 2^ceil(log2(amax / 448)) per b x b block), at the first block size b that
    reproduces w exactly; None if none does."""
    G, R, D = w.shape
    for b in blocks:
        if R % b or D % b:
            continue
        tiles = w.float().view(G, R // b, b, D // b, b)
        amax = tiles.abs().amax(dim=(2, 4))
        exp = torch.ceil(torch.log2(torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))))
        exp = torch.where(amax > 0, exp, torch.full_like(exp, -127.0))
        scale = torch.exp2(exp)
        q = (tiles / scale[:, :, None, :, None]).to(torch.float8_e4m3fn)
        back = (q.float() * scale[:, :, None, :, None]).view(G, R, D).to(torch.bfloat16)
        del tiles
        if torch.equal(back, w):
            return q.view(G, R, D).contiguous(), scale.contiguous()
    return None


@triton.jit
def _wo_a_fp8_gemv_kernel(x_ptr, w_ptr, s_ptr, o_ptr, R, D, SB: tl.constexpr,
                          BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    g = tl.program_id(0)
    rb = tl.program_id(1)
    rows = rb * BLOCK_R + tl.arange(0, BLOCK_R)
    acc = tl.zeros([BLOCK_R, BLOCK_D], dtype=tl.float32)
    for d0 in range(0, D, BLOCK_D):
        cols = d0 + tl.arange(0, BLOCK_D)
        x = tl.load(x_ptr + g * D + cols).to(tl.float32)
        w = tl.load(w_ptr + (g * R + rows[:, None]) * D + cols[None, :]).to(tl.float32)
        s = tl.load(s_ptr + (g * (R // SB) + rows[:, None] // SB) * (D // SB) + cols[None, :] // SB)
        acc += w * s * x[None, :]
    tl.store(o_ptr + g * R + rows, tl.sum(acc, axis=1).to(o_ptr.dtype.element_ty))


def wo_a_fp8_gemv(o: torch.Tensor, q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """o [1, G, D] bf16, q [G, R, D] e4m3, s [G, R/b, D/b] -> [1, G, R] bf16."""
    G, R, D = q.shape
    x = o.reshape(G, D).contiguous()
    out = torch.empty((1, G, R), device=o.device, dtype=o.dtype)
    _wo_a_fp8_gemv_kernel[(G, R // 16)](x, q, s, out, R, D, SB=R // s.shape[1],
                                        BLOCK_R=16, BLOCK_D=256, num_warps=4)
    return out


@triton.jit
def _wo_a_fp8_dequant_kernel(w_ptr, s_ptr, o_ptr, R, D, SB: tl.constexpr,
                             BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    g = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.program_id(2) * BLOCK_D + tl.arange(0, BLOCK_D)
    offs = (g * R + rows[:, None]).to(tl.int64) * D + cols[None, :]
    w = tl.load(w_ptr + offs).to(tl.float32)
    s = tl.load(s_ptr + (g * (R // SB) + rows[:, None] // SB) * (D // SB) + cols[None, :] // SB)
    tl.store(o_ptr + offs, (w * s).to(o_ptr.dtype.element_ty))


def wo_a_fp8_dequant(q: torch.Tensor, s: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """q [G, R, D] e4m3, s [G, R/b, D/b] -> out [G, R, D] bf16 (exact: s is a power of two)."""
    G, R, D = q.shape
    _wo_a_fp8_dequant_kernel[(G, R // 32, D // 128)](q, s, out, R, D, SB=R // s.shape[1],
                                                     BLOCK_R=32, BLOCK_D=128, num_warps=4)
    return out
