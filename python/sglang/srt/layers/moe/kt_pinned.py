"""Fast pinned host memory for launch-time buffers (KT_FAST_PINNED=1).

torch.empty(..., pin_memory=True) goes through cudaHostAlloc, which pins 4K
pages at ~1.3 GB/s on this host (tools/bench_pinned_alloc.py, 10-11: 8.6 GB in
6.6 s). An anonymous mapping advised for transparent huge pages, touched by
a few threads and then cudaHostRegister'ed is pinned at ~17 GB/s (8.6 GB:
touch 0.16 s, register 0.33 s), with the same H2D rate (27.9 GB/s). The PLE
table (14.4 GB per rank), the GGUF streamer's banks (25 GB per rank) and the
lend's mirror (6.7 GB) were allocated the slow way.

empty(shape, dtype) returns a CPU tensor over such memory. The memory stays
mapped and registered until the process exits (torch.frombuffer does not keep
the exporting object alive, 10-11: a buffer freed under its tensor crashed),
so this is for buffers that live as long as the server: the PLE table, the
streamer banks, the lend mirror.
First touch happens on threads of the calling process, so the pages land
where its memory policy puts them (the scheduler is NUMA-bound), as the
cudaHostAlloc pages did. Contents are zero, like a fresh cudaHostAlloc.
"""
import ctypes
import mmap
import os
import threading
import time

import torch

ENABLED = os.environ.get("KT_FAST_PINNED") == "1"
MIN_BYTES = int(os.environ.get("KT_FAST_PINNED_MIN", str(64 << 20)))
TOUCH_THREADS = int(os.environ.get("KT_FAST_PINNED_THREADS", "16"))
_HUGE = 2 << 20
_stats = {"n": 0, "bytes": 0, "sec": 0.0}


def _touch(addr: int, size: int, threads: int) -> None:
    """Write one byte per 4K page from several threads (fault the huge pages in parallel)."""
    if threads <= 1 or size < (256 << 20):
        ctypes.memset(addr, 0, size)
        return
    step = -(-size // threads)
    step = (step + _HUGE - 1) & ~(_HUGE - 1)

    def w(lo):
        hi = min(size, lo + step)
        if hi > lo:
            ctypes.memset(addr + lo, 0, hi - lo)

    # ctypes calls release the GIL, so the memsets run in parallel
    ts = [threading.Thread(target=w, args=(i * step,)) for i in range(threads)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


_live = []  # every buffer handed out; never freed before exit


class _PinnedBuffer:
    """An anonymous THP mapping registered with CUDA."""

    def __init__(self, nbytes: int):
        size = (max(nbytes, 1) + _HUGE - 1) & ~(_HUGE - 1)
        # over-allocate by one huge page and use the 2 MB-aligned part
        self._mm = mmap.mmap(-1, size + _HUGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self._mm))
        off = ((base + _HUGE - 1) & ~(_HUGE - 1)) - base
        self._mm.madvise(mmap.MADV_HUGEPAGE, off, size)
        self.addr, self.nbytes, self._off = base + off, nbytes, off
        _touch(self.addr, size, TOUCH_THREADS)
        rc = torch.cuda.cudart().cudaHostRegister(self.addr, size, 0)
        if int(rc) != 0:
            raise RuntimeError(f"cudaHostRegister({size} bytes) failed: {rc}")
        self._size = size
        self._registered = True


def empty(shape, dtype=torch.uint8) -> torch.Tensor:
    """A pinned CPU tensor (zero-filled on the fast path); cudaHostAlloc below
    KT_FAST_PINNED_MIN bytes or when KT_FAST_PINNED is off."""
    if isinstance(shape, int):
        shape = (shape,)
    numel = 1
    for s in shape:
        numel *= int(s)
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    if not ENABLED or nbytes < MIN_BYTES:
        return torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    t0 = time.perf_counter()
    buf = _PinnedBuffer(nbytes)
    _live.append(buf)
    t = torch.frombuffer(memoryview(buf._mm)[buf._off:buf._off + nbytes], dtype=torch.uint8, count=nbytes)
    assert t.device.type == "cpu" and t.data_ptr() == buf.addr, (t.device, t.data_ptr(), buf.addr)
    _stats["n"] += 1
    _stats["bytes"] += nbytes
    _stats["sec"] += time.perf_counter() - t0
    return t.view(dtype).view(shape)


def stats() -> str:
    s = _stats
    return f"{s['n']} buffers, {s['bytes'] / 1e9:.2f} GB in {s['sec']:.2f} s"


_bounce = {"buf": None, "n": 0}


def bounce(nbytes: int) -> torch.Tensor:
    """A [nbytes] uint8 view of one process-wide pinned staging buffer, reused
    by every caller (each must finish its synchronous copies before the next
    call); grows when a larger one is asked for. Off: a fresh cudaHostAlloc
    buffer, as before."""
    if not ENABLED:
        return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    if _bounce["n"] < nbytes:
        _bounce["buf"] = empty((nbytes,), torch.uint8)
        _bounce["n"] = nbytes
    return _bounce["buf"][:nbytes]
