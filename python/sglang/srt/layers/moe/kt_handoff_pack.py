"""Packed GPU->CPU request handoff for the KT expert path, and the shared-expert switch.

Developed for this SGLang/KT recipe in the local GPU-glue investigation
(2026-10-22). The stream-memop handoff it shortens is the KT_STREAM_MEMOPS
recipe in kt-kernel's cpuinfer.h; no Strata implementation is included.

KT_HANDOFF_PACK=1: submit_forward stages the MoE input for the CPU with one kernel
that writes the hidden rows (bf16), the routing weights (fp32) and the expert ids
(widened to int64) straight into kt-kernel's pinned host buffers through their
device-mapped addresses. It replaces the ids cast kernel and the three DtoH copy
nodes (10240 + 24 + 48 bytes at bs=1) that the graph serialises before the
``req`` stream write. Ordering: kt-kernel issues that cuStreamWriteValue32 with
flags 0, which places a memory barrier (like __threadfence_system, scoped to the
stream) before the write, so the host sees the stores before the flag.
Bytes only (the int32->int64 widening of the ids is exact): bit-identical.

Pinned memory from torch's host allocator is page-locked and, with unified
addressing, mapped into the device address space. The pinned tensors are passed to
the kernel as ordinary pointer arguments; Triton's launcher resolves each with
cuPointerGetAttribute(DEVICE_POINTER). Each buffer is checked once (pinned,
contiguous, cuMemHostGetDevicePointer succeeds) and the verdict is cached by
address, which is stable because kt-kernel never frees these buffers; the first
failing buffer turns the pack off for the process (copies, as without the switch).

KT_SHARED_AFTER_SUBMIT=1: DeepseekV2MoE.forward_normal_dual_stream forks the shared
expert from an event KTEPWrapperMethod.submit records right after the CPU request,
instead of before the router. Scheduling only: bit-identical.
"""

from __future__ import annotations

import ctypes
import logging
import os

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

PACK = os.environ.get("KT_HANDOFF_PACK", "0") == "1"
SHARED_AFTER_SUBMIT = os.environ.get("KT_SHARED_AFTER_SUBMIT", "0") == "1"

# Decode and verify graphs only: one CTA writes all ids, and large prefill chunks
# move their rows faster with the copy engine than with SM stores over PCIe.
_PACK_MAX_ROWS = 256

_usable_cache: dict = {}
_libcuda = None
_disabled = False


def _host_dev_ptr(t: torch.Tensor) -> int:
    """Device address of pinned host tensor ``t`` (0 if it is not mapped)."""
    global _libcuda
    hp = t.data_ptr()
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return hp  # CPU interpreter: loads and stores go through host addresses
    if _libcuda is None:
        _libcuda = ctypes.CDLL("libcuda.so.1")
    out = ctypes.c_uint64(0)
    rc = _libcuda.cuMemHostGetDevicePointer_v2(ctypes.byref(out), ctypes.c_void_p(hp), 0)
    return int(out.value) if rc == 0 else 0


def _usable(*host: torch.Tensor) -> bool:
    global _disabled
    if _disabled:
        return False
    for t in host:
        key = (t.data_ptr(), t.numel(), t.dtype)
        ok = _usable_cache.get(key)
        if ok is None:
            ok = t.is_pinned() and t.is_contiguous() and _host_dev_ptr(t) != 0
            _usable_cache[key] = ok
        if not ok:
            _disabled = True
            logger.warning("kt_handoff_pack: pinned buffer not device-mapped; using copies")
            return False
    return True


@triton.jit
def _pack_kernel(
    x_ptr,
    w_ptr,
    ids_ptr,
    hx_ptr,  # pinned host buffers (device-mapped)
    hw_ptr,
    hid_ptr,
    NX,
    NK,
    BLOCK: tl.constexpr,
    KBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < NX
    tl.store(hx_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)
    if pid == 0:
        k = tl.arange(0, KBLOCK)
        mk = k < NK
        tl.store(hw_ptr + k, tl.load(w_ptr + k, mask=mk), mask=mk)
        tl.store(hid_ptr + k, tl.load(ids_ptr + k, mask=mk).to(tl.int64), mask=mk)


def pack_to_host(x, topk_weights, topk_ids, host_x, host_w, host_ids) -> bool:
    """One kernel instead of ``host_x.copy_(x); host_w.copy_(w); host_ids.copy_(ids.long())``.
    False (and nothing issued) when the inputs do not fit the fast path."""
    if not (
        x.is_cuda
        and x.dim() == 2
        and x.shape[0] <= _PACK_MAX_ROWS
        and x.dtype == torch.bfloat16
        and host_x.dtype == torch.bfloat16
        and topk_weights.dtype == torch.float32
        and host_w.dtype == torch.float32
        and topk_ids.dtype in (torch.int32, torch.int64)
        and host_ids.dtype == torch.int64
        and x.is_contiguous()
        and topk_weights.is_contiguous()
        and topk_ids.is_contiguous()
        and host_x.numel() == x.numel()
        and host_w.numel() == topk_weights.numel()
        and host_ids.numel() == topk_ids.numel()
        and _usable(host_x, host_w, host_ids)
    ):
        return False
    nx, nk = x.numel(), topk_ids.numel()
    BLOCK = 2048
    _pack_kernel[(triton.cdiv(nx, BLOCK),)](
        x,
        topk_weights,
        topk_ids,
        host_x,
        host_w,
        host_ids,
        nx,
        nk,
        BLOCK=BLOCK,
        KBLOCK=max(16, triton.next_power_of_2(nk)),
        num_warps=4,
    )
    return True
