"""Block-FP8 W8A8 GEMV for decode (M <= 4): the same arithmetic as _w8a8_block_fp8_matmul_hopper (the activation
quantized per token group, the weight per block; each group's fp8 products summed in fp32, times the two scales,
summed over the groups), with the K loop in steps of several groups instead of one.

The block kernel steps K by the scale group (32 on DeepSeek-V4.1-Flash-1017), so a decode call loads a 32 x 128
weight tile per iteration and ran at 2.5-3x the bandwidth bound (wq_b 59.7 us for 21 MB on an RTX 5070 Ti). Here
a program holds BLOCK_N weight rows over BLOCK_K = several groups, sums each group with tl.sum, and splits K across
programs when N alone gives too few of them; the split partials are summed by a second kernel in a fixed order.
Only the fp32 summation order differs from the block kernel.
(patches/sglang-fp8-block-gemv.py installs it; SGLANG_FP8_GEMV=0 turns it off.)"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_block_gemv_kernel(
    A, B, C, As, Bs,
    M, N, K,
    stride_am, stride_bn, stride_cs, stride_cm,
    stride_As_m, stride_As_k, stride_Bs_n, stride_Bs_k,
    k_per_split,
    GROUP_N: tl.constexpr, GROUP_K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, MB: tl.constexpr,
):
    pid_n = tl.program_id(0)
    split = tl.program_id(1)
    G: tl.constexpr = BLOCK_K // GROUP_K
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = offs_n < N
    k_start = split * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)
    n_groups_k = tl.cdiv(K, GROUP_K)
    acc0 = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_N], dtype=tl.float32)
    b_s_row = (offs_n // GROUP_N) * stride_Bs_n
    for k0 in range(k_start, k_end, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        kmask = offs_k < k_end
        b = tl.load(B + offs_n[:, None] * stride_bn + offs_k[None, :],
                    mask=nmask[:, None] & kmask[None, :], other=0.0).to(tl.float32)
        kg = k0 // GROUP_K + tl.arange(0, G)
        gmask = (kg < n_groups_k) & (kg * GROUP_K < k_end)
        b_s = tl.load(Bs + b_s_row[:, None] + kg[None, :] * stride_Bs_k,
                      mask=nmask[:, None] & gmask[None, :], other=0.0)
        for mi in tl.static_range(MB):
            if mi < M:
                a = tl.load(A + mi * stride_am + offs_k, mask=kmask, other=0.0).to(tl.float32)
                a_s = tl.load(As + mi * stride_As_m + kg * stride_As_k, mask=gmask, other=0.0)
                part = tl.sum(tl.reshape(b * a[None, :], (BLOCK_N, G, GROUP_K)), axis=2)
                contrib = tl.sum(part * a_s[None, :] * b_s, axis=1)
                if mi == 0:
                    acc0 += contrib
                elif mi == 1:
                    acc1 += contrib
                elif mi == 2:
                    acc2 += contrib
                else:
                    acc3 += contrib
    out = C + split * stride_cs + offs_n
    tl.store(out, acc0.to(C.dtype.element_ty), mask=nmask)
    if MB > 1:
        if M > 1:
            tl.store(out + stride_cm, acc1.to(C.dtype.element_ty), mask=nmask)
    if MB > 2:
        if M > 2:
            tl.store(out + 2 * stride_cm, acc2.to(C.dtype.element_ty), mask=nmask)
    if MB > 3:
        if M > 3:
            tl.store(out + 3 * stride_cm, acc3.to(C.dtype.element_ty), mask=nmask)


@triton.jit
def _fp8_block_gemv_reduce(P, C, MN, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        acc += tl.load(P + s * MN + offs, mask=mask, other=0.0)
    tl.store(C + offs, acc.to(C.dtype.element_ty), mask=mask)


MAX_M = 4
_SMS = {}


def _sms(dev) -> int:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _SMS:
        _SMS[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _SMS[i]


def fp8_block_gemv(A, B, As, Bs, block_size, C, block_n=16, block_k=512):
    """A [M, K] e4m3 (M <= 4), B [N, K] e4m3, As [M, K/gk] fp32, Bs [N/gn, K/gk] fp32 -> C [M, N]."""
    M, K = A.shape
    N = B.shape[0]
    gn, gk = block_size
    block_k = max(gk, min(block_k, triton.next_power_of_2(K)))
    assert block_k % gk == 0 and M <= MAX_M
    nb = triton.cdiv(N, block_n)
    # enough programs for the SMs: split K when N alone gives fewer than 8 a SM
    split = 1
    while nb * split < 8 * _sms(A.device) and K // (split * 2) >= block_k:
        split *= 2
    k_per_split = triton.cdiv(triton.cdiv(K, split), block_k) * block_k
    split = triton.cdiv(K, k_per_split)
    mb = 1 if M == 1 else (2 if M == 2 else 4)
    if split == 1:
        P, stride_cs = C, 0
    else:
        P = torch.empty((split, M, N), device=A.device, dtype=torch.float32)
        stride_cs = M * N
    _fp8_block_gemv_kernel[(nb, split)](
        A, B, P, As, Bs, M, N, K,
        A.stride(0), B.stride(0), stride_cs, N,
        As.stride(0), As.stride(1), Bs.stride(0), Bs.stride(1),
        k_per_split,
        GROUP_N=gn, GROUP_K=gk, BLOCK_N=block_n, BLOCK_K=block_k, MB=mb,
        num_warps=4, num_stages=3,
    )
    if split > 1:
        _fp8_block_gemv_reduce[(triton.cdiv(M * N, 512),)](P, C, M * N, SPLITS=split, BLOCK=512)
    return C
