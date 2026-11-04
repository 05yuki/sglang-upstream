"""GPU-streamed prefill for KT CPU experts, GGUF edition (--kt-method LLAMAFILE).

Above KT_GPU_STREAM_PREFILL tokens a MoE layer's CPU experts are not computed
on the CPU: each rank DMAs its TP part of the experts' raw GGUF blocks, G at a
time, from a pinned host bank into a device slot, dequantizes them to bf16 on
the device (kt_gguf_dequant.py: IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL /
Q2_0) and runs FlashInfer's bf16 CUTLASS MoE on bf16 activations (W16), as
the V4.1 tree's KT_GPU_STREAM_W16 does for MXFP4. Two slots double-buffer
DMA against dequant + compute; each (token count, group, type triple) is one
CUDA graph.

The host bank of a layer is this rank's slice of the intermediate dimension
(rows [N*r, N*(r+1)) of gate/up, the same columns of down) for every expert,
in physical expert order (the router's ids after the placement remap), cut
out of the GGUF mmap once and kept pinned: about half the expert bytes of the
checkpoint per rank. Built at load when KT_GPU_STREAM_EARLY_INIT=1, else on a
layer's first streamed call.

Env (shared with the NVFP4 edition): KT_GPU_STREAM_PREFILL, KT_GPU_STREAM_GROUP,
KT_GPU_STREAM_TIMING, KT_GPU_STREAM_EARLY_INIT, KT_STREAM_GRAPH_KEEP,
KT_GPU_STREAM_SELFTEST=2 (reference check of the first streamed layer).
"""
import os
import time
import traceback
from typing import Dict, Optional

import torch

from sglang.srt.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.moe import kt_gguf_dequant as DQ

STREAM_THRESHOLD = int(os.environ.get("KT_GPU_STREAM_PREFILL", "0") or 0)
GROUP_SIZE = int(os.environ.get("KT_GPU_STREAM_GROUP", "8") or 8)
TIMING = os.environ.get("KT_GPU_STREAM_TIMING") == "1"
GRAPH_KEEP = int(os.environ.get("KT_STREAM_GRAPH_KEEP", "8"))
EARLY_INIT = os.environ.get("KT_GPU_STREAM_EARLY_INIT", "0") == "1"
# Chunks shorter than the prefill chunk run padded to the next multiple of
# ROW_STEP rows (ids -1, weights 0), so graphs and inputs exist for a few row
# counts only and the lend's accounting of the scratch holds.
ROW_STEP = int(os.environ.get("KT_GPU_STREAM_ROW_STEP", "2048") or 0)


def _chunk_rows() -> int:
    try:
        from sglang.srt.server_args import get_global_server_args

        return int(get_global_server_args().chunked_prefill_size or 0)
    except Exception:
        return 0


def _scratch_empty(shape, dtype, device, per_chunk=False):
    try:
        from sglang.srt.layers.moe import kt_lend

        return kt_lend.scratch_empty(shape, dtype, device, per_chunk=per_chunk)
    except ImportError:
        return torch.empty(shape, dtype=dtype, device=device)


def _scratch_region():
    try:
        from sglang.srt.layers.moe import kt_lend

        return kt_lend.scratch_region()
    except ImportError:
        import contextlib

        return contextlib.nullcontext()


def _log(msg: str):
    print(f"[kt-stream-gguf {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _align64(n: int) -> int:
    return (n + 63) & ~63


class _Slot:
    def __init__(self, G: int, strides, device):
        # flat per projection; a layer views its n x part_bytes in the head
        self.raw = [_scratch_empty((G * st,), torch.uint8, device) for st in strides]
        self.dma_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.compute_used = False


class _Geometry:
    """Expert types and this rank's part bytes per layer, from the GGUF headers."""

    def __init__(self, path: str, N: int, K: int, world: int):
        from kt_kernel.utils.loader import GGUFLoader

        self.loader = GGUFLoader(path)
        cfg = self.loader.get_model_config()
        self.E = int(cfg["num_experts"])
        self.N, self.K, self.world = N, K, world
        if N * world != int(cfg["moe_intermediate_size"]):
            raise RuntimeError(f"KT_GPU_STREAM_PREFILL (GGUF): N={N} x {world} ranks != intermediate {cfg['moe_intermediate_size']}")
        if int(cfg["hidden_size"]) != K:
            raise RuntimeError(f"KT_GPU_STREAM_PREFILL (GGUF): hidden {cfg['hidden_size']} != {K}")
        self.layers = []
        i = 0
        while self.loader.has_tensor(f"blk.{i}.ffn_gate_exps.weight"):
            self.layers.append(i)
            i += 1
        if not self.layers:
            raise RuntimeError("KT_GPU_STREAM_PREFILL (GGUF): no blk.N.ffn_gate_exps.weight in the GGUF")
        self.types: Dict[int, tuple] = {}
        self.part_bytes: Dict[int, tuple] = {}
        for layer_idx in self.layers:
            types = tuple(int(self.loader.get_ggml_type(f"blk.{layer_idx}.ffn_{p}_exps.weight"))
                          for p in ("gate", "up", "down"))
            for t in types:
                if t not in DQ.BLOCK_INFO:
                    raise RuntimeError(f"KT_GPU_STREAM_PREFILL (GGUF): layer {layer_idx} uses ggml type {t}, no GPU dequantizer")
            gb, gs = DQ.BLOCK_INFO[types[0]]
            ub, us = DQ.BLOCK_INFO[types[1]]
            db, ds = DQ.BLOCK_INFO[types[2]]
            if K % gb or K % ub or N % db:
                raise RuntimeError(f"KT_GPU_STREAM_PREFILL (GGUF): layer {layer_idx} blocks {gb}/{ub}/{db} vs K={K} N={N}")
            self.types[layer_idx] = types
            self.part_bytes[layer_idx] = (N * (K // gb) * gs, N * (K // ub) * us, K * (N // db) * ds)
        self.strides = [_align64(max(pb[pi] for pb in self.part_bytes.values())) for pi in range(3)]


_geometry: Optional[_Geometry] = None


def _geometry_for(path: str, N: int, K: int, world: int) -> _Geometry:
    global _geometry
    if _geometry is None or _geometry.loader is None:
        _geometry = _Geometry(path, N, K, world)
    return _geometry


# device index -> dict(slots, w13, w2, T, gin): made before SGLang sizes its KV
# pool (preallocate() from the resident method's create_moe_runner), in the
# lend's scratch so the chunk hint counts them.
_prealloc = {}


def preallocate(layer):
    if STREAM_THRESHOLD <= 0:
        return
    from sglang.srt.server_args import get_global_server_args

    args = get_global_server_args()
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    key = device.index
    if key in _prealloc:
        return
    N, K = layer.intermediate_size_per_partition, layer.hidden_size
    geo = _geometry_for(args.kt_weight_path, N, K, get_tensor_model_parallel_world_size())
    G = GROUP_SIZE
    slots = [_Slot(G, geo.strides, device) for _ in range(2)]
    w13 = _scratch_empty((G, 2 * N, K), torch.bfloat16, device)
    w2 = _scratch_empty((G, K, N), torch.bfloat16, device)
    T = _chunk_rows()
    gin = None
    if T > 0:
        topk, f32 = layer.top_k, torch.float32
        gin = {
            "x": _scratch_empty((T, K), torch.bfloat16, device, per_chunk=True),
            "ids": _scratch_empty((T, topk), torch.int32, device, per_chunk=True),
            "w": _scratch_empty((T, topk), f32, device, per_chunk=True),
            "acc": _scratch_empty((T, K), torch.bfloat16, device, per_chunk=True),
            "out": _scratch_empty((T, K), torch.bfloat16, device, per_chunk=True),
        }
    _prealloc[key] = {"slots": slots, "w13": w13, "w2": w2, "T": T, "gin": gin}
    mb = (sum(geo.strides) * G * 2 + (w13.numel() + w2.numel()) * 2
          + (sum(t.numel() * t.element_size() for t in gin.values()) if gin else 0)) / 1e6
    _log(f"preallocated 2 slots, bf16 group buffers and inputs for {T} rows ({mb:.0f} MB) on {device} before the KV pool")


class KTStreamPrefillGGUF:
    """Process-wide (per rank) streaming state, shared by every MoE layer."""

    def __init__(self, method, layer):
        from flashinfer.fused_moe import cutlass_fused_moe
        from flashinfer.fused_moe.core import ActivationType
        from sglang.srt.environ import envs
        from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import _activation_type

        self.rank = get_tensor_model_parallel_rank()
        self.world = get_tensor_model_parallel_world_size()
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self.N = layer.intermediate_size_per_partition
        self.K = layer.hidden_size
        self.G = GROUP_SIZE
        self.top_k = layer.top_k
        self.path = method.kt_config.weight_path
        # KT_GGUF_STREAM_EP (on with more than one rank): every rank streams whole experts, a contiguous share of
        # them, instead of a slice of every expert. A GGUF block of 256 values cannot be cut, and V4.1's
        # intermediate 2304 split in two is 1152; the MoE output's all-reduce adds the shares as it adds slices.
        self.ep = self.world > 1 and os.environ.get("KT_GGUF_STREAM_EP", "1") == "1"
        if self.ep:
            self.N = self.N * self.world
        geo = _geometry_for(self.path, self.N, self.K, 1 if self.ep else self.world)
        self.loader = geo.loader
        self.E = geo.E
        self._kt_layers = geo.layers
        self._layer_types = geo.types
        self._part_bytes = geo.part_bytes
        self._strides = geo.strides
        self._tp_size, self._tp_rank = layer.moe_tp_size, layer.moe_tp_rank
        self._ep_size, self._ep_rank = layer.moe_ep_size, layer.moe_ep_rank
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self._cutlass_fused_moe = cutlass_fused_moe
        self._activation_type = _activation_type(layer.moe_runner_config)
        assert self._activation_type == ActivationType.Swiglu, self._activation_type
        self._use_fused_finalize = envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE.get()
        self._workspace = None
        self._workspace_tokens = 0
        self._graphs = {}
        self._graph_use = {}
        self._gin = {}
        self._capture_stream = torch.cuda.Stream(device=self.device)
        self._graph_pool = torch.cuda.graph_pool_handle()
        # layer_idx -> pinned banks [3][E, part bytes]
        self._banks: Dict[int, list] = {}
        pre = _prealloc.get(self.device.index)
        if pre is not None:
            self.slots, self.w13_bf16, self.w2_bf16 = pre["slots"], pre["w13"], pre["w2"]
            self._pre_T, self._pre_gin = pre["T"], pre["gin"]
        else:
            self.slots = [_Slot(self.G, self._strides, self.device) for _ in range(2)]
            self.w13_bf16 = torch.empty((self.G, 2 * self.N, self.K), dtype=torch.bfloat16, device=self.device)
            self.w2_bf16 = torch.empty((self.G, self.K, self.N), dtype=torch.bfloat16, device=self.device)
            self._pre_T, self._pre_gin = 0, None
        strides = self._strides
        self.calls = 0
        mb = (sum(strides) * self.G * 2 + (self.w13_bf16.numel() + self.w2_bf16.numel()) * 2) / 1e6
        _log(f"rank {self.rank}: streamed GGUF prefill above {STREAM_THRESHOLD} tokens, G={self.G}, "
             f"N={self.N} K={self.K} E={self.E}, {len(self._kt_layers)} layers, device buffers {mb:.0f} MB")

    def _physical_to_logical(self, layer_idx: int) -> Optional[torch.Tensor]:
        from sglang.srt.eplb.expert_location_dispatch import get_global_expert_location_metadata

        meta = get_global_expert_location_metadata()
        if meta is None:
            return None
        m = meta.physical_to_logical_map_cpu[layer_idx].to(torch.int64)
        if torch.equal(m, torch.arange(m.numel(), dtype=torch.int64)):
            return None
        return m

    def _bank(self, layer_idx: int):
        """This rank's slice of every expert of the layer, pinned, physical order."""
        b = self._banks.get(layer_idx)
        if b is not None:
            return b
        t0 = time.time()
        types, part_bytes = self._layer_types[layer_idx], self._part_bytes[layer_idx]
        p2l = self._physical_to_logical(layer_idx)
        E, N, K, r = self.E, self.N, self.K, self.rank
        banks = []
        for pi, proj in enumerate(("gate", "up", "down")):
            data, qt = self.loader.get_undequanted_tensor_and_ggml_type(f"blk.{layer_idx}.ffn_{proj}_exps.weight")
            blck, ts = DQ.BLOCK_INFO[types[pi]]
            if proj != "down":
                rowbytes = (K // blck) * ts
                v = data.view(E, N * self.world, rowbytes)[:, N * r:N * (r + 1), :]
            else:
                v = data.view(E, K, (N * self.world) // blck, ts)[:, :, (N * r) // blck:(N * (r + 1)) // blck, :]
            v = v.reshape(E, -1)
            assert v.shape[1] == part_bytes[pi], (proj, v.shape, part_bytes[pi])
            if p2l is not None:
                v = v[p2l]
            # pinned from the start (empty().pin_memory() made a pageable buffer and copied it)
            h = torch.empty((E, part_bytes[pi]), dtype=torch.uint8, pin_memory=True)
            h.copy_(v)
            banks.append(h)
            del data
        b = self._banks[layer_idx] = banks
        if TIMING or layer_idx == self._kt_layers[0]:
            _log(f"rank {self.rank}: layer {layer_idx} bank {sum(part_bytes) * E / 1e9:.2f} GB pinned in {time.time() - t0:.1f} s"
                 f" (types {types}{', placement map' if p2l is not None else ''})")
        return b

    def build_all_banks(self):
        """Every layer's bank, KT_STREAM_BANK_THREADS (8) layers at a time: the
        copies out of the GGUF mmap release the GIL, and one at a time ran at
        ~0.5 GB/s (10-10)."""
        t0 = time.time()
        threads = int(os.environ.get("KT_STREAM_BANK_THREADS", "8"))
        if threads > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=threads) as ex:
                list(ex.map(self._bank, self._kt_layers))
        else:
            for layer_idx in self._kt_layers:
                self._bank(layer_idx)
        gb = sum(sum(self._part_bytes[l]) * self.E for l in self._banks) / 1e9
        _log(f"rank {self.rank}: {len(self._banks)} layer banks, {gb:.1f} GB pinned, built in {time.time() - t0:.0f} s")

    # -- staging ---------------------------------------------------------------

    def _stage(self, layer_idx: int, gi: int, base: int, n: int, stats):
        banks = self._bank(layer_idx)
        part_bytes = self._part_bytes[layer_idx]
        slot = self.slots[gi % 2]
        t0 = time.perf_counter()
        with torch.cuda.stream(self.transfer_stream):
            if slot.compute_used:
                self.transfer_stream.wait_event(slot.compute_done)
            for pi in range(3):
                pb = part_bytes[pi]
                slot.raw[pi][:n * pb].view(n, pb).copy_(banks[pi][base:base + n], non_blocking=True)
            slot.dma_done.record(self.transfer_stream)
        stats["dma_enq"] += time.perf_counter() - t0

    # -- one group ---------------------------------------------------------------

    def _group_body(self, layer, layer_idx: int, gi: int, base: int, n: int, g):
        slot = self.slots[gi % 2]
        types, pb = self._layer_types[layer_idx], self._part_bytes[layer_idx]
        N, K = self.N, self.K
        x, topk_weights, ids32, acc, out = g["x"], g["w"], g["ids"], g["acc"], g["out"]
        # CUTLASS fc1 rows are [up; gate]
        DQ.dequant_experts(slot.raw[1], pb[1], types[1], n, N, K, self.w13_bf16[:n, :N])
        DQ.dequant_experts(slot.raw[0], pb[0], types[0], n, N, K, self.w13_bf16[:n, N:])
        DQ.dequant_experts(slot.raw[2], pb[2], types[2], n, K, N, self.w2_bf16[:n])
        local = torch.where((ids32 >= base) & (ids32 < base + n), ids32 - base, torch.full_like(ids32, -1))
        out.zero_()
        self._cutlass_fused_moe(
            input=x,
            token_selected_experts=local,
            token_final_scales=topk_weights,
            fc1_expert_weights=self.w13_bf16[:n],
            fc2_expert_weights=self.w2_bf16[:n],
            output_dtype=torch.bfloat16,
            quant_scales=None,
            input_sf=None,
            tp_size=self._tp_size,
            tp_rank=self._tp_rank,
            ep_size=self._ep_size,
            ep_rank=self._ep_rank,
            activation_type=self._activation_type,
            tune_max_num_tokens=self._tune_tokens,
            output=out,
            use_fused_finalize=self._use_fused_finalize,
            workspace_buffer=self._workspace,
        )
        acc.add_(out)

    def _reference_check(self, method, layer_idx, x, ids, w, acc):
        """This rank's partial output for the streamed experts from a torch
        dequantization of the bank (bf16 weights, fp32 math) against acc."""
        torch.cuda.current_stream(self.device).synchronize()
        banks = self._bank(layer_idx)
        types, part_bytes = self._layer_types[layer_idx], self._part_bytes[layer_idx]
        N, K = self.N, self.K
        first = 0 if getattr(method.gpu_method, "streams_hot_experts", False) else method.num_gpu_experts
        ref = torch.zeros_like(acc, dtype=torch.float32)
        ids64 = ids.to(torch.int64)
        xf = x.float()
        used = 0
        w_g = torch.empty((1, N, K), dtype=torch.bfloat16, device=self.device)
        w_u = torch.empty((1, N, K), dtype=torch.bfloat16, device=self.device)
        w_d = torch.empty((1, K, N), dtype=torch.bfloat16, device=self.device)
        for e in range(first, self.E):
            rows, cols = torch.where(ids64 == e)
            if rows.numel() == 0:
                continue
            used += 1
            for pi, (dst, r_, c_) in enumerate(((w_g, N, K), (w_u, N, K), (w_d, K, N))):
                raw = banks[pi][e:e + 1].to(self.device)
                DQ.dequant_experts(raw, part_bytes[pi], types[pi], 1, r_, c_, dst)
            h = xf[rows]
            a = torch.nn.functional.silu(h @ w_g[0].float().t()) * (h @ w_u[0].float().t())
            o = (a @ w_d[0].float().t()) * w[rows, cols].float().unsqueeze(1)
            ref.index_add_(0, rows, o)
        d = (acc.float() - ref).abs()
        cos = torch.nn.functional.cosine_similarity(acc.float().flatten(), ref.flatten(), dim=0).item()
        _log(f"rank{self.rank} refcheck layer{layer_idx} T={x.shape[0]} experts_used={used}: cos={cos:.5f} "
             f"max|acc-ref|={d.max().item():.4f} mean={d.mean().item():.5f} ref_mean|.|={ref.abs().mean().item():.5f}")

    # -- graphs ---------------------------------------------------------------

    def _graph_inputs(self, T: int, x, physical_ids, topk_weights):
        g = self._gin.get(T)
        n = x.shape[0]
        if g is None and self._pre_gin is not None and self._pre_T >= T:
            g = {k: v[:T] for k, v in self._pre_gin.items()}
            self._gin[T] = g
        if g is None:
            f32 = torch.float32
            g = {
                "x": torch.zeros((T, self.K), dtype=torch.bfloat16, device=self.device),
                "ids": torch.empty((T, physical_ids.shape[1]), dtype=torch.int32, device=self.device),
                "w": torch.empty((T, physical_ids.shape[1]), dtype=f32, device=self.device),
                "acc": torch.zeros((T, self.K), dtype=torch.bfloat16, device=self.device),
                "out": torch.empty((T, self.K), dtype=torch.bfloat16, device=self.device),
            }
            self._gin[T] = g
        g["x"][:n].copy_(x)
        g["ids"][:n].copy_(physical_ids)
        g["w"][:n].copy_(topk_weights)
        if n < T:
            # padding rows: no expert in any group, nothing added to acc
            g["x"][n:].zero_()
            g["ids"][n:].fill_(-1)
            g["w"][n:].zero_()
        g["acc"].zero_()
        return g

    def _trim_graphs(self):
        top = max(self._graph_use)
        old = [t for t in self._graph_use if t != top]
        if len(old) <= GRAPH_KEEP:
            return
        drop = set(old[: len(old) - GRAPH_KEEP])
        torch.cuda.synchronize(self.device)
        for k in [k for k in self._graphs if k[0] in drop]:
            del self._graphs[k]
        for t in drop:
            del self._graph_use[t]
            self._gin.pop(t, None)

    def _graph_for(self, layer, layer_idx: int, T: int, gi: int, base: int, n: int, g):
        key = (T, gi, n, self._layer_types[layer_idx])
        graph = self._graphs.get(key)
        self._graph_use.pop(T, None)
        self._graph_use[T] = None
        if graph is None:
            t0 = time.perf_counter()
            cs = self._capture_stream
            cs.wait_stream(torch.cuda.current_stream(self.device))
            gw = dict(g)
            gw["acc"] = torch.zeros_like(g["acc"])
            with torch.cuda.stream(cs):
                self._group_body(layer, layer_idx, gi, base, n, gw)
            cs.synchronize()
            del gw
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._graph_pool, stream=cs):
                self._group_body(layer, layer_idx, gi, base, n, g)
            torch.cuda.current_stream(self.device).wait_stream(cs)
            self._graphs[key] = graph
            self._trim_graphs()
            if TIMING and gi == 0:
                _log(f"rank{self.rank} captured group graph T={T} gi={gi} n={n} types={key[3]} in {(time.perf_counter()-t0)*1e3:.0f} ms")
        return graph

    # -- one layer ----------------------------------------------------------

    def run(self, method, layer, dispatch_output, physical_ids: torch.Tensor) -> torch.Tensor:
        """Sum of the streamed experts' outputs for this layer, [tokens, hidden]."""
        from sglang.srt.utils.common import next_power_of_2

        x = dispatch_output.hidden_states
        rows = x.shape[0]
        T = rows
        if ROW_STEP > 0 and self._pre_T > 0 and rows < self._pre_T and not getattr(self, "accumulates_into", False):
            T = min(self._pre_T, -(-rows // ROW_STEP) * ROW_STEP)
        layer_idx = method.kt_config.layer_idx
        self._tune_tokens = next_power_of_2(T)
        if self._workspace is None or self._workspace_tokens < T:
            from flashinfer.fused_moe import cutlass_fused_moe_workspace_size
            from sglang.srt.layers.moe.moe_runner.flashinfer_cutlass import (
                _workspace_max_tokens,
                get_shared_cutlass_workspace,
                preallocate_shared_cutlass_workspace,
            )

            T_ws = _workspace_max_tokens(T)
            nbytes = cutlass_fused_moe_workspace_size(
                T_ws, self.K, self.N, self.G, physical_ids.shape[1],
                x_dtype=torch.bfloat16, weight_dtype=torch.bfloat16, output_dtype=torch.bfloat16,
                activation_type=self._activation_type,
                tp_size=self._tp_size, tp_rank=self._tp_rank, ep_size=self._ep_size, ep_rank=self._ep_rank,
                use_fused_finalize=self._use_fused_finalize, device=x.device)
            self._workspace = get_shared_cutlass_workspace(nbytes, x.device)
            if self._workspace is None:
                with _scratch_region():
                    self._workspace = preallocate_shared_cutlass_workspace(nbytes, x.device)
            self._workspace_tokens = T_ws
            _log(f"rank{self.rank} cutlass workspace {nbytes/1e6:.0f} MB for {T_ws} tokens x {self.G} experts")
        topk_weights = dispatch_output.topk_output.topk_weights
        first = 0 if getattr(method.gpu_method, "streams_hot_experts", False) else method.num_gpu_experts
        if hasattr(self, "_groups"):
            groups = self._groups(method)  # the NVMe edition: under KT_GGUF_STREAM_EP, this rank's share
        else:
            groups = [(b, min(self.G, self.E - b)) for b in range(first, self.E, self.G)]
        stats = {"dma_enq": 0.0, "graph_enq": 0.0}
        t0 = time.perf_counter()
        self._bank(layer_idx)
        t_a = time.perf_counter()
        g = self._graph_inputs(T, x, physical_ids, topk_weights)
        t_b = time.perf_counter()
        cur = torch.cuda.current_stream(self.device)
        self._stage(layer_idx, 0, groups[0][0], groups[0][1], stats)
        t_c = time.perf_counter()
        for gi, (base, n) in enumerate(groups):
            slot = self.slots[gi % 2]
            graph = self._graph_for(layer, layer_idx, T, gi, base, n, g)
            t1 = time.perf_counter()
            cur.wait_event(slot.dma_done)
            graph.replay()
            slot.compute_done.record(cur)
            slot.compute_used = True
            stats["graph_enq"] += time.perf_counter() - t1
            if gi + 1 < len(groups):
                self._stage(layer_idx, gi + 1, groups[gi + 1][0], groups[gi + 1][1], stats)
        self.calls += 1
        if TIMING:
            t_d = time.perf_counter()
            cur.synchronize()
            t_e = time.perf_counter()
            _log(f"rank{self.rank} layer{layer_idx} tokens={rows}/{T} groups={len(groups)} "
                 f"dma_enq={stats['dma_enq']*1e3:.0f}ms graph_enq={stats['graph_enq']*1e3:.0f}ms "
                 f"total={(t_e-t0)*1e3:.0f}ms (before inputs {(t_a-t0)*1e3:.0f}, inputs {(t_b-t_a)*1e3:.0f}, "
                 f"stage0 {(t_c-t_b)*1e3:.0f}, loop {(t_d-t_c)*1e3:.0f}, final sync {(t_e-t_d)*1e3:.0f} ms)")
        if os.environ.get("KT_GPU_STREAM_SELFTEST") == "2" and layer_idx == self._kt_layers[0]:
            self._reference_check(method, layer_idx, x, physical_ids, topk_weights, g["acc"][:rows])
        return g["acc"] if rows == T else g["acc"][:rows]


_instance: Optional[KTStreamPrefillGGUF] = None
_failed = False


def stream_prefill_init_early(method, layer) -> None:
    global _instance, _failed
    if not EARLY_INIT or STREAM_THRESHOLD <= 0 or _failed or _instance is not None:
        return
    t0 = time.time()
    try:
        _instance = KTStreamPrefillGGUF(method, layer)
        _instance.build_all_banks()
    except Exception:
        _failed = True
        _log("early init failed:\n" + traceback.format_exc())
        raise
    _log(f"rank {_instance.rank}: GGUF streamer built at load time in {time.time() - t0:.0f} s")


def stream_prefill_for(method, layer, num_tokens: int) -> Optional[KTStreamPrefillGGUF]:
    global _instance, _failed
    if STREAM_THRESHOLD <= 0 or _failed:
        return None
    if num_tokens < STREAM_THRESHOLD or torch.cuda.is_current_stream_capturing():
        return None
    if _instance is None:
        _log(f"rank {get_tensor_model_parallel_rank()}: first qualifying call, layer {method.kt_config.layer_idx}, tokens={num_tokens}")
        try:
            _instance = KTStreamPrefillGGUF(method, layer)
        except Exception:
            _failed = True
            _log("init failed:\n" + traceback.format_exc())
            raise
    return _instance
