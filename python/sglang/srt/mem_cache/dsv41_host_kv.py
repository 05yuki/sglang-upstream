"""DeepSeek-V4.1's ratio-1/2 compressed KV in host memory (SGLANG_DSV41_HOST_KV=1).

The compressed KV pools (on 1017 KVLayout.V41_FP4, 288 bytes an entry; any KVLayout, the
geometry read from the pool; kv_source layers only) live in
pinned host memory mapped into the GPU's address space and presented as CUDA tensors, so every
writer stores into them unchanged. Decode reads only the indexer's top-k entries: each index
source gathers its rows into a small device buffer laid out as the pool's pages, and attention
reads that buffer with the rows renumbered row * k + j. Attention reading host memory directly
would cost ~100 us a layer on PCIe; the gather is ~10 us an index source.

Prefill copies each kv_source pool layer's used pages to the GPU once a chunk instead.

The index-K pools stay on the GPU: decode scores the whole context of them at every index
source, before any top-k exists.
"""

import logging
import mmap
import os
from typing import Dict, Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)



def host_kv_enabled() -> bool:
    """On by default: the pool covers the context, and what the GPU cannot hold of the
    compressed KV goes to host memory. SGLANG_DSV41_HOST_KV=0: the pool is what fits."""
    return os.environ.get("SGLANG_DSV41_HOST_KV", "1") != "0"


# Set by the pool configurator: how many full tokens of the ratio-1/2 compressed KV the GPU
# holds (None: all of it, the pools are plain device tensors).
PLAN = {"device_tokens": None}


def _gpu_numa_node(index: int) -> int:
    try:
        p = torch.cuda.get_device_properties(index)
        path = f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0/numa_node"
        return max(0, int(open(path).read().strip()))
    except Exception:
        return 0


def make_pool_buffer(shape, device, device_pages: int) -> torch.Tensor:
    """A ratio-1/2 pool layer: its first device_pages pages on the GPU, the rest in host
    memory on the GPU's NUMA node, one range (kv_vmm). Without kv_vmm the whole layer goes
    to host memory when any of it must."""
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    page_bytes = shape[1]
    from sglang.srt.layers.moe import kt_kv_vmm

    if kt_kv_vmm.ENABLED:
        return kt_kv_vmm.split_tensor(shape, torch.uint8, index, device_pages * page_bytes, _gpu_numa_node(index))
    if device_pages >= shape[0]:
        return torch.zeros(shape, dtype=torch.uint8, device=device)
    return host_mapped_uint8(shape, device)


class _CudaArray:
    def __init__(self, ptr: int, shape):
        self.__cuda_array_interface__ = {
            "shape": tuple(shape),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def host_mapped_uint8(shape, device) -> torch.Tensor:
    """A zeroed uint8 tensor in pinned host memory (2 MiB pages where THP allows), registered
    with CUDA and returned as a CUDA tensor on `device` (UVA: the host address)."""
    nbytes = 1
    for s in shape:
        nbytes *= s
    mm = mmap.mmap(
        -1,
        max(nbytes, mmap.PAGESIZE),
        flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
        prot=mmap.PROT_READ | mmap.PROT_WRITE,
    )
    mm.madvise(mmap.MADV_HUGEPAGE)
    host = torch.frombuffer(mm, dtype=torch.uint8)[:nbytes]
    host.zero_()
    err = torch.cuda.cudart().cudaHostRegister(host.data_ptr(), nbytes, 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister({nbytes} bytes) failed: {err}")
    t = torch.as_tensor(_CudaArray(host.data_ptr(), shape), device=device)
    t._host_kv_keep = (mm, host)  # the mapping lives as long as the tensor
    return t


def keep_free_pages_sorted(allocator) -> None:
    """Freed pages go back in order, so the lowest free pages (the device part of a split
    pool) are handed out first; the paged allocator otherwise puts them in front, and a
    long request's host pages would come before device pages nobody used."""
    a = getattr(allocator, "full_attn_allocator", None) or allocator
    if getattr(a, "_dsv41_sorted", False) or not hasattr(a, "_release_page_ids"):
        return
    orig = a._release_page_ids

    def release(*page_ids):
        if getattr(a, "need_sort", False):
            return orig(*page_ids)
        a.free_pages = torch.sort(torch.cat((*page_ids, a.free_pages))).values

    a._release_page_ids = release
    a._dsv41_sorted = True
    logger.info(f"DSV4.1 host KV: {type(a).__name__} hands out the lowest free pages first")


@triton.jit
def _gather_rows_kernel(
    src,
    idx,
    stage,
    remap,
    idx_stride,
    K: tl.constexpr,
    P: tl.constexpr,
    DATA: tl.constexpr,
    SCALE: tl.constexpr,
    PAGE_BYTES: tl.constexpr,
    DATA_WORDS_P2: tl.constexpr,
    SCALE_WORDS_P2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # A page holds P entries' data rows (DATA bytes each), then their scale rows
    # (SCALE bytes each), padded to PAGE_BYTES: every KVLayout (kv_layout.py).
    row = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    s = tl.load(idx + row * idx_stride + j)
    valid = s >= 0
    s = tl.where(valid, s, 0).to(tl.int64)
    d = row.to(tl.int64) * K + j
    w = tl.arange(0, DATA_WORDS_P2)
    wm = valid[:, None] & (w[None, :] < DATA // 4)
    src_data = (src + (s // P * PAGE_BYTES + s % P * DATA)[:, None] + w[None, :] * 4).to(tl.pointer_type(tl.int32))
    dst_data = (stage + (d // P * PAGE_BYTES + d % P * DATA)[:, None] + w[None, :] * 4).to(tl.pointer_type(tl.int32))
    tl.store(dst_data, tl.load(src_data, mask=wm), mask=wm)
    if SCALE > 0:
        v = tl.arange(0, SCALE_WORDS_P2)
        vm = valid[:, None] & (v[None, :] < SCALE // 4)
        src_sc = (src + (s // P * PAGE_BYTES + P * DATA + s % P * SCALE)[:, None] + v[None, :] * 4).to(
            tl.pointer_type(tl.int32)
        )
        dst_sc = (stage + (d // P * PAGE_BYTES + P * DATA + d % P * SCALE)[:, None] + v[None, :] * 4).to(
            tl.pointer_type(tl.int32)
        )
        tl.store(dst_sc, tl.load(src_sc, mask=vm), mask=vm)
    tl.store(remap + row * K + j, tl.where(valid, d, -1).to(tl.int32))


def _p2(n: int) -> int:
    return 1 << max(0, (max(1, n) - 1).bit_length())


class HostLowRatioKV:
    """Device staging for the host-resident ratio-1/2 pools: per ratio, room for `max_rows`
    query rows of `topk` entries in the pool's page layout, and the renumbered indices."""

    def __init__(self, layouts, topk: int, max_rows: int, device):
        """layouts: {ratio: (data bytes, scale bytes, entries a page, page bytes)}, read
        from each pool's own KVLayout when it is made, so any layout works."""
        self.topk = topk
        self.max_rows = max_rows
        self.geometry: Dict[int, tuple] = {}
        self.entries_per_page: Dict[int, int] = {}
        self.stage: Dict[int, torch.Tensor] = {}
        self.remap: Dict[int, torch.Tensor] = {}
        for r, (data, scale, p, page_bytes) in layouts.items():
            assert data % 4 == 0 and scale % 4 == 0 and (topk % p == 0 or p % topk == 0)
            pages = max(1, max_rows * topk // p)
            self.geometry[r] = (data, scale, p, page_bytes)
            self.entries_per_page[r] = p
            self.stage[r] = torch.zeros(pages, page_bytes, dtype=torch.uint8, device=device)
            self.remap[r] = torch.full((max_rows, topk), -1, dtype=torch.int32, device=device)
        self.check = os.environ.get("SGLANG_DSV41_HOST_KV_CHECK", "0") == "1"
        self._checks = self._direct = 0

    def prefill_copy(self, fb, ratio: int, pool_layer: int, src: torch.Tensor, max_slot, layer_id: int):
        """The GPU copy of a pool layer's pages up to the batch's highest token slot (a
        compressed page holds one full page's entries), made once a forward at the first
        attention that reads the layer: after every writer of the forward (the kv_source's
        own pass and the bounded-replay tail rewrite), which a copy at the kv_source missed.
        Rows x top-k gathers would move ~1.2 GB for an 8K-token chunk."""
        copies = getattr(fb, "dsv41_host_kv_copies", None)
        if copies is None:
            copies = fb.dsv41_host_kv_copies = {}
        m = copies.get((ratio, pool_layer))
        if m is None:
            # the layers run in order and a pool layer's readers are consecutive (2-7,
            # 8-13, 14-19, 20-39): the previous copy has no reader left
            copies.clear()
            pages = max_slot() // (self.entries_per_page[ratio] * ratio) + 1
            m = torch.empty((pages, src.shape[1]), dtype=src.dtype, device=self.stage[ratio].device)
            # an SM kernel, not cudaMemcpy: a kv_vmm range whose tail is host-NUMA memory
            # is not a valid copy-engine source ("invalid argument")
            torch.add(src[:pages].view(torch.int32), 0, out=m.view(torch.int32))
            copies[(ratio, pool_layer)] = m
        elif self.check:
            self.check_mirror(ratio, m, src, layer_id)
        return m

    def check_mirror(self, ratio: int, m: torch.Tensor, src: torch.Tensor, layer_id: int) -> None:
        """SGLANG_DSV41_HOST_KV_CHECK=1: a later layer's view of the copy against the host pool,
        byte for byte (the pool is typed float8: a value comparison calls NaN bytes unequal)."""
        m, src = m.view(torch.uint8), src.view(torch.uint8)
        now = src[: m.shape[0]].clone()
        data, scale, p, _ = self.geometry[ratio]
        data_m, data_n = m[:, : p * data].reshape(-1, data), now[:, : p * data].reshape(-1, data)
        sc_m = m[:, p * data : p * (data + scale)].reshape(-1, max(1, scale))
        sc_n = now[:, p * data : p * (data + scale)].reshape(-1, max(1, scale))
        bad_d = (data_m != data_n).any(1).nonzero().flatten()
        if bad_d.numel():
            # the CPU's view of the same host bytes: who is stale?
            import ctypes

            nb = m.shape[0] * src.stride(0)
            host = torch.frombuffer((ctypes.c_uint8 * nb).from_address(src.data_ptr()), dtype=torch.uint8)
            host = host.view(m.shape[0], src.stride(0))[:, : src.shape[1]]
            torch.cuda.current_stream().synchronize()
            data_h = host[:, : p * data].reshape(-1, data)
            i = bad_d[:64].cpu()
            same_copy = int((data_h[i] == data_m[bad_d[:64]].cpu()).all(1).sum())
            same_now = int((data_h[i] == data_n[bad_d[:64]].cpu()).all(1).sum())
            logger.info(f"host KV check: layer {layer_id}: of {i.numel()} differing entries the CPU sees the copy's bytes in "
                        f"{same_copy}, the GPU's later read in {same_now}")
        bad_s = (sc_m != sc_n).any(1).nonzero().flatten()
        self._checks += 1
        if bad_d.numel() or bad_s.numel() or self._checks <= 4:
            nz = (data_n != 0).any(1).nonzero().flatten()
            logger.info(
                f"host KV check: layer {layer_id} ratio {ratio}: {m.shape[0]} pages copied, "
                f"written entries {nz.numel()} (last {int(nz[-1]) if nz.numel() else -1}); "
                f"differ: data {bad_d.numel()} {bad_d[:6].tolist()}, scales {bad_s.numel()} {bad_s[:6].tolist()}; "
                f"zero in the copy among them {int((data_m[bad_d] == 0).all(1).sum()) if bad_d.numel() else 0}; "
                f"bytes {((data_m[bad_d] != data_n[bad_d]).any(0).nonzero().flatten()[[0, -1]].tolist()) if bad_d.numel() else []}"
            )

    def covers(self, ratio: int, indices: Optional[torch.Tensor]) -> bool:
        return (
            ratio in self.stage
            and indices is not None
            and indices.shape[0] <= self.max_rows
            and indices.shape[-1] == self.topk
        )

    def gather(self, ratio: int, src: torch.Tensor, indices: torch.Tensor) -> None:
        rows = indices.shape[0]
        if rows == 0:
            return
        assert indices.dtype == torch.int32 and indices.stride(-1) == 1
        block = 32
        data, scale, p, page_bytes = self.geometry[ratio]
        _gather_rows_kernel[(rows, self.topk // block)](
            src,
            indices,
            self.stage[ratio],
            self.remap[ratio],
            indices.stride(0),
            K=self.topk,
            P=p,
            DATA=data,
            SCALE=scale,
            PAGE_BYTES=page_bytes,
            DATA_WORDS_P2=_p2(data // 4),
            SCALE_WORDS_P2=_p2(max(1, scale // 4)),
            BLOCK=block,
            num_warps=4,
        )
