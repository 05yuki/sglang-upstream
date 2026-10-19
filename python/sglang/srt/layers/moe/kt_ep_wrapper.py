# SPDX-License-Identifier: Apache-2.0
"""
KT Expert Parallelism Wrapper for MoE layers.

This module provides a generic wrapper that enables CPU-GPU expert parallelism
for any MoE quantization method. It coordinates parallel execution of GPU experts
(using any quantization method) and CPU experts (using AMX/AVX instructions).
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.arg_groups.overrides import model_config_of
from sglang.srt.layers.moe.kt_stream_prefill import stream_prefill_for
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase
from sglang.srt.runtime_context import (
    get_exec,
    get_parallel,
    get_schedule,
)
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.server_args import ServerArgs

try:
    from kt_kernel import KTMoEWrapper

    KTRANSFORMERS_AVAILABLE = True
except ImportError:
    KTRANSFORMERS_AVAILABLE = False



def _drop_checkpoint_cache(layer_idx: int, num_layers: int = 0) -> None:
    """KT_GPU_STREAM_DROP_CACHE=<dir>: posix_fadvise(DONTNEED) every file
    under it, so the arenas kt-kernel is about to create find free 2 MB
    blocks (patches/sglang-kt-drop-cache-per-layer.py)."""
    import os
    import time

    root_dir = os.environ.get("KT_GPU_STREAM_DROP_CACHE", "")
    if not root_dir:
        return
    # the loader's prefetch evicts each finished layer itself; a whole-tree
    # drop here would throw away the next layer's background read
    # (the loader reads the last KT_LOAD_PREFETCH_TAIL layers after their
    # arena exists; this drop runs again for them so the arena finds 2 MB
    # blocks, as it did before the prefetch)
    if layer_idx > 0 and int(os.environ.get("KT_LOAD_PREFETCH", "0") or 0) > 0:
        tail = int(os.environ.get("KT_LOAD_PREFETCH_TAIL", "4"))
        if not num_layers or layer_idx < num_layers - tail - 1:
            return
    t0, n = time.time(), 0
    for root, _, files in os.walk(root_dir, followlinks=True):
        for name in files:
            try:
                fd = os.open(os.path.join(root, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n += 1
            finally:
                os.close(fd)
    if layer_idx == 0 or os.environ.get("KT_GPU_STREAM_TIMING") == "1":
        print(f"[kt-stream] layer {layer_idx}: dropped the page cache of {n} files in "
              f"{time.time() - t0:.1f} s", flush=True)

@dataclass
class KTConfig:
    """Configuration for KTransformers heterogeneous computing CPU part.

    Args:
        layer_idx: Layer index in the model
        num_gpu_experts: Number of experts to run on GPU
        cpuinfer_threads: Number of CPU inference threads
        threadpool_count: Number of thread pools for CPU computation
        weight_path: Path to CPU quantized weights
        chunked_prefill_size: Chunk size for prefill computation
        method: CPU computation method (e.g., "int4")
        num_layers: Total number of layers in the model (optional)
    """

    layer_idx: int
    num_gpu_experts: int
    cpuinfer_threads: int
    threadpool_count: int
    weight_path: str
    chunked_prefill_size: int
    max_deferred_experts_per_token: int
    method: str
    num_layers: Optional[int] = None


def create_kt_config_from_server_args(
    server_args: "ServerArgs", layer_idx: int
) -> Optional[KTConfig]:
    """Create KTConfig from ServerArgs if KT is configured.

    Args:
        server_args: Global server arguments
        layer_idx: Layer index in the model

    Returns:
        KTConfig if KT is configured, None otherwise
    """
    if get_exec().moe.kt_weight_path is None:
        return None

    num_layers = getattr(
        model_config_of(server_args).hf_config, "num_hidden_layers", None
    )

    return KTConfig(
        layer_idx=layer_idx,
        num_gpu_experts=get_exec().moe.kt_num_gpu_experts,
        cpuinfer_threads=get_exec().moe.kt_cpuinfer,
        threadpool_count=get_exec().moe.kt_threadpool_count,
        weight_path=get_exec().moe.kt_weight_path,
        chunked_prefill_size=get_schedule().chunked_prefill_size,
        method=get_exec().moe.kt_method,
        max_deferred_experts_per_token=get_exec().moe.kt_max_deferred_experts_per_token,
        num_layers=num_layers,
    )


@torch.compile(dynamic=True, backend=get_compiler_backend())
def mask_cpu_expert_ids(topk_ids: torch.Tensor, num_gpu_experts: int) -> torch.Tensor:
    """Mask CPU expert IDs by setting them to -1.

    This function masks expert IDs that should be computed on CPU (IDs >= num_gpu_experts)
    so they won't be computed on GPU. The masked IDs are set to -1, which causes the
    GPU MoE kernel to skip those experts.

    Args:
        topk_ids: Tensor of shape [num_tokens, top_k] containing expert IDs
        num_gpu_experts: Number of experts that should run on GPU (experts 0 to num_gpu_experts-1)

    Returns:
        Modified topk_ids tensor with CPU expert IDs masked as -1
    """
    topk_ids[topk_ids >= num_gpu_experts] = -1
    return topk_ids


_routing_dump_prefix = os.environ.get("KT_ROUTING_DUMP")
_gpu_mask_zero = os.environ.get("KT_GPU_MASK_ZERO") == "1"
_hot_check = os.environ.get("KT_HOT_CHECK") == "1"
_routing_state = {}
# layer_idx -> KTEPWrapperMethod on the rank that owns the kt-kernel wrapper
# (rank 0); the streamed-prefill zero-copy path hands every layer's expert
# arenas to the other ranks in one go.
KT_LAYERS = {}


def _routing_count(device, layer_idx, topk_ids, rank):
    import threading
    import time

    st = _routing_state.get(device.index)
    if st is None:
        counts = torch.zeros((64, 512), dtype=torch.int32, device=device)
        ones = torch.ones((1,), dtype=torch.int32, device=device)
        st = _routing_state[device.index] = {"counts": counts, "ones": ones}

        def dump():
            while True:
                time.sleep(30)
                try:
                    c = st["counts"].cpu()
                    torch.save(
                        {"rank": rank, "logical_count": c},
                        f"{_routing_dump_prefix}-rank{rank}.pt",
                    )
                except Exception as e:  # never take the scheduler down for a dump
                    print(f"[kt-routing] dump failed: {e}", flush=True)

        threading.Thread(target=dump, daemon=True, name="kt-routing-dump").start()
    ids = topk_ids.reshape(-1).to(torch.int64)
    st["counts"][layer_idx].index_add_(0, ids, st["ones"].expand(ids.numel()))


def _kt_swiglu_kwargs(layer: torch.nn.Module) -> dict:
    """{"swiglu_limit": limit} for KTMoEWrapper when the layer clamps its
    SwiGLU, else {} (older kt-kernel builds keep working for models without
    a clamp). The value is what the resident GPU runner applies:
    moe_runner_config.swiglu_limit (DeepSeek V4: gate <= limit, |up| <= limit
    before silu(gate) * up). KT_SWIGLU_CLAMP=0 leaves it out, the behaviour
    before 10-12."""
    if os.environ.get("KT_SWIGLU_CLAMP", "1") == "0":
        return {}
    cfg = getattr(layer, "moe_runner_config", None)
    limit = getattr(cfg, "swiglu_limit", None)
    if not limit:
        return {}
    if getattr(cfg, "gemm1_alpha", None) is not None or getattr(cfg, "gemm1_clamp_limit", None) is not None:
        raise RuntimeError(
            "KT_SWIGLU_CLAMP: this layer uses gemm1_alpha/gemm1_clamp_limit, which "
            "the KT CPU wrapper in this tree does not map; set KT_SWIGLU_CLAMP=0"
        )
    return {"swiglu_limit": float(limit)}


class KTEPWrapperMethod(FusedMoEMethodBase):
    """Wrapper for any MoE quantization method to enable CPU-GPU expert parallelism.

    This wrapper coordinates parallel execution of:
    - GPU experts (0 to num_gpu_experts-1) using any quantization method
    - CPU experts (num_gpu_experts to total_experts-1) using AMX/AVX instructions

    The wrapper implements the submit-compute-sync pattern:
    1. Submit CPU expert computation (non-blocking)
    2. Execute GPU expert computation in parallel
    3. Synchronize and merge CPU+GPU results

    Example:
        # Wrap any GPU method with AMX/AVX CPU expert support
        gpu_method = CompressedTensorsWNA16MoE(quant_config, prefix)
        kt_config = KTConfig(layer_idx=0, num_gpu_experts=4, ...)
        method = KTEPWrapperMethod(gpu_method, kt_config)
    """

    def __init__(
        self,
        gpu_method: FusedMoEMethodBase,
        kt_config: KTConfig,
    ):
        """Initialize the KT EP wrapper.

        Args:
            gpu_method: The quantization method to use for GPU experts
            kt_config: Configuration for KT CPU expert computation
        """
        if not KTRANSFORMERS_AVAILABLE:
            raise ImportError(
                "kt_kernel is not installed. To use KTransformers EP wrapper, please install kt_kernel."
            )

        self.gpu_method = gpu_method
        self.kt_config = kt_config
        self.num_gpu_experts = kt_config.num_gpu_experts
        self.override_num_local_experts = True
        self.gpu_method.num_gpu_experts = self.num_gpu_experts
        self.tp_rank = get_parallel().tp_rank

        # KT wrapper will be initialized in create_weights
        self.wrapper: Optional[KTMoEWrapper] = None

        # Store parameters needed for KT initialization
        self._layer_params = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weights for both GPU and CPU experts.

        Args:
            layer: The MoE layer module
            num_experts: Total number of experts (GPU + CPU)
            hidden_size: Hidden dimension size
            intermediate_size_per_partition: Intermediate size per TP partition
            params_dtype: Data type for parameters
            **extra_weight_attrs: Additional weight attributes
        """
        self.global_num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size_per_partition

        # Get required parameters from layer object
        # top_k: number of experts selected per token
        num_experts_per_tok = layer.top_k

        # intermediate_size_full: full intermediate size before TP partitioning
        intermediate_size_full = (
            layer.intermediate_size_per_partition * layer.moe_tp_size
        )

        layer_max_deferred = self.kt_config.max_deferred_experts_per_token or 0
        if (
            self.kt_config.max_deferred_experts_per_token is not None
            and self.kt_config.num_layers is not None
            and self.kt_config.layer_idx == self.kt_config.num_layers - 1
        ):
            layer_max_deferred = 0

        # 1. Create weights for GPU experts using the wrapped method
        # GPU experts: 0 to num_gpu_experts-1
        self.gpu_method.create_weights(
            layer=layer,
            num_experts=self.num_gpu_experts,
            hidden_size=hidden_size,
            intermediate_size_per_partition=intermediate_size_per_partition,
            params_dtype=params_dtype,
            **extra_weight_attrs,
        )
        # kt_kv_hot_lend: the next X experts' weights on a shadow layer, for the
        # KV pool's free pages
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.make_shadow(self, layer, hidden_size, intermediate_size_per_partition,
                                   params_dtype, extra_weight_attrs)

        # 2. Initialize KT wrapper for CPU experts
        # CPU experts: num_gpu_experts to num_experts-1
        if self.tp_rank == 0:
            # kt-kernel 0.7.0 inference API takes an explicit boolean mask of
            # GPU-resident experts instead of num_gpu_experts (SFT-only).
            # GPU experts occupy ordinals 0..num_gpu_experts-1 (see above).
            gpu_mask = torch.zeros(num_experts, dtype=torch.bool)
            # kt_kv_hot_lend's dynamic resident rows: kt-kernel loads every expert, the
            # routes say what it computes (-1 for what the GPU takes)
            if not kt_kv_hot_lend.cpu_holds_all(self):
                gpu_mask[: self.num_gpu_experts] = True
            KT_LAYERS[self.kt_config.layer_idx] = self
            self.wrapper = KTMoEWrapper(
                layer_idx=self.kt_config.layer_idx,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                hidden_size=hidden_size,
                moe_intermediate_size=intermediate_size_full,
                gpu_experts_mask=gpu_mask,
                cpuinfer_threads=self.kt_config.cpuinfer_threads,
                threadpool_count=self.kt_config.threadpool_count,
                weight_path=self.kt_config.weight_path,
                chunked_prefill_size=self.kt_config.chunked_prefill_size,
                method=self.kt_config.method,
                max_deferred_experts_per_token=layer_max_deferred,
                # patches/sglang-kt-swiglu-clamp.py (KT_SWIGLU_CLAMP)
                **_kt_swiglu_kwargs(layer),
                # KT_NUMA_NODES="1" pins a single pool to node 1, the node
                # that does not host the TP0 scheduler, for the prefill-decay
                # isolation; unset keeps kt-kernel's sequential default.
                numa_nodes=(
                    [int(x) for x in os.environ["KT_NUMA_NODES"].split(",")]
                    if os.environ.get("KT_NUMA_NODES")
                    else None
                ),
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading from checkpoint.

        Args:
            layer: The MoE layer module
        """
        # 1. Process GPU weights
        # Empty resident weights need no GPU repacking; still load CPU experts below.
        # (dsv41-flash-dual-epyc#2)
        if self.num_gpu_experts > 0 and hasattr(
            self.gpu_method, "process_weights_after_loading"
        ):
            self.gpu_method.process_weights_after_loading(layer)

        # 2. Load CPU weights using KT wrapper
        if self.tp_rank == 0 and self.wrapper is not None:
            torch.cuda.synchronize()

            # Get expert location metadata for CPU expert mapping
            from sglang.srt.eplb.expert_location_dispatch import (
                get_global_expert_location_metadata,
            )

            physical_to_logical_map_cpu = (
                get_global_expert_location_metadata()
                .physical_to_logical_map_cpu[self.kt_config.layer_idx]
                .contiguous()
            )
            _drop_checkpoint_cache(self.kt_config.layer_idx, getattr(self.kt_config, "num_layers", 0) or 0)
            self.wrapper.load_weights(physical_to_logical_map_cpu)
        # kt_kv_hot_lend: experts N..N+X-1 to pinned memory and the shadow slots
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.split(self, layer)
        # Last layer loaded on this rank: optionally build the streamer and
        # pin every layer's arenas now (KT_GPU_STREAM_EARLY_INIT).
        num_layers = getattr(self.kt_config, "num_layers", None)
        if num_layers and self.kt_config.layer_idx == num_layers - 1:
            from sglang.srt.layers.moe.kt_stream_prefill import stream_prefill_init_early

            stream_prefill_init_early(self, layer)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: "MoeRunnerConfig"
    ):
        """Create MoE runner for computation.

        Args:
            layer: The MoE layer module
            moe_runner_config: Configuration for MoE runner
        """
        self.moe_runner_config = moe_runner_config
        if self.override_num_local_experts:
            moe_runner_config.num_local_experts = self.num_gpu_experts
        # Delegate to GPU method to create its runner
        self.gpu_method.create_moe_runner(layer, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.gpu_method.get_moe_runner_backend(layer)

    def submit(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> None:
        """Submit CPU expert computation asynchronously (non-blocking).

        This method submits the CPU expert computation to AMX/AVX without waiting
        for completion, allowing GPU computation to proceed in parallel.

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information
        """
        assert self.moe_runner_config.activation == "silu", (
            "Only SiLU activation is supported."
        )

        if self.tp_rank != 0 or self.wrapper is None:
            return

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        # attributes, not unpacking: the packed variant carries a fourth field
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        # KT_ROUTING_DUMP=<prefix>: accumulate a [layers, experts] histogram of
        # the routed expert ids on the GPU (a captured index_add_, so it keeps
        # counting under CUDA graph replay) and save it as
        # <prefix>-rank<r>.pt every 30 s. SGLang's expert distribution recorder
        # never sees the V4 router's ids (09-13); this does.
        # Submit forward task to CPU (non-blocking)
        self.wrapper.submit_forward(
            x, topk_ids, topk_weights, torch.cuda.current_stream(x.device).cuda_stream
        )

    def sync(self, x: torch.Tensor) -> torch.Tensor:
        """Synchronize and retrieve CPU expert computation results.

        This method waits for the CPU computation to complete and returns the results.

        Args:
            x: Reference tensor for shape and device information

        Returns:
            CPU expert computation results
        """
        if self.tp_rank != 0 or self.wrapper is None:
            return torch.zeros_like(x)

        # Wait for CPU computation and retrieve results
        return self.wrapper.sync_forward(
            x, torch.cuda.current_stream(x.device).cuda_stream
        )

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "CombineInput":
        """Execute hybrid CPU+GPU MoE forward pass with parallelism.

        This is the main computation method that coordinates:
        1. Submit CPU expert computation (non-blocking)
        2. Execute GPU expert computation in parallel
        3. Synchronize CPU results and merge with GPU results

        Args:
            layer: The MoE layer module
            dispatch_output: Dispatched tokens and routing information

        Returns:
            Combined computation results from CPU and GPU experts
        """
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        # The ids are physical: --init-expert-location moves experts between the GPU
        # slots (physical 0..num_gpu_experts-1) and the CPU, and TopK resolves the
        # placement for every router (resolve_routing_inputs).
        # KT_ROUTING_DUMP=<prefix>: a [layers, experts] histogram of the routed physical ids
        # (a captured index_add_, so it keeps counting under CUDA graph replay), saved as
        # <prefix>-rank<r>.pt every 30 s. Counted here, before the hot lend and the
        # streamer split the routes (sglang-kt-routing-count-before-split.py); SGLang's
        # expert distribution recorder never sees the V4 router's ids (09-13).
        if _routing_dump_prefix and self.tp_rank == 0:
            _routing_count(x.device, self.kt_config.layer_idx, topk_output.topk_ids, self.tp_rank)

        # KT_GPU_STREAM_PREFILL=<tokens>: above that many tokens the CPU
        # experts are streamed to the GPU in groups instead of computed here
        # (kt_stream_prefill.py). The ids are taken before the mask below
        # mutates them in place.
        # kt_kv_hot_lend: a prefill window finds the resident rows holding 0..N-1
        from sglang.srt.layers.moe import kt_kv_hot_lend

        kt_kv_hot_lend.res_guard()
        streamer = stream_prefill_for(self, layer, x.shape[0])
        cpu_share = False
        if streamer is not None:
            physical_ids = topk_output.topk_ids.clone()
            from sglang.srt.layers.moe.kt_stream_prefill import stream_cpu_cutoff

            cutoff = stream_cpu_cutoff(self, streamer)
            if cutoff < self.global_num_experts and self.tp_rank == 0:
                if self.num_gpu_experts <= 0:
                    raise RuntimeError("KT_STREAM_CPU_GROUPS needs --kt-num-gpu-experts > 0")
                # Everything below the cutoff is streamed or resident: send it
                # to GPU expert 0 at weight 0, which kt-kernel skips.
                theirs = physical_ids < cutoff
                # (-1 when kt-kernel holds every expert: it would compute expert 0)
                skip = -1 if kt_kv_hot_lend.cpu_holds_all(self) else 0
                cpu_do = dispatch_output._replace(
                    topk_output=topk_output._replace(
                        topk_ids=torch.where(theirs, torch.full_like(physical_ids, skip), physical_ids),
                        topk_weights=torch.where(
                            theirs, torch.zeros_like(topk_output.topk_weights), topk_output.topk_weights
                        ),
                    )
                )
                self.submit(layer, cpu_do)
                cpu_share = True

        # kt_kv_hot_lend: routes to the experts lent the KV pool's free pages go to
        # their slots, and to the CPU at weight 0
        hot = kt_kv_hot_lend.route(self, dispatch_output) if streamer is None else None

        # Step 1: Submit CPU expert computation (non-blocking)
        if streamer is None and self.tp_rank == 0:
            self.submit(layer, hot[0] if hot is not None else dispatch_output)

        # Step 2: Prepare GPU computation by masking CPU expert IDs
        # CPU expert IDs (>= num_gpu_experts) are set to -1 so GPU kernel skips them
        topk_ids = topk_output.topk_ids
        if hot is not None and hot[3] is not None:
            # kt_kv_hot_lend's dynamic resident rows: the row each route's expert is in
            masked_topk_ids = hot[3]
        else:
            masked_topk_ids = mask_cpu_expert_ids(topk_ids, self.num_gpu_experts)

        # Create modified dispatch output for GPU computation
        if _gpu_mask_zero:
            # KT_GPU_MASK_ZERO=1: the Marlin MoE path cannot take -1 ids (its
            # alignment kernels index weights with them: illegal address at
            # graph capture, 09-13). Send the CPU-bound rows to GPU expert 0
            # with a zero routing weight instead — the second GEMM multiplies
            # by the weight, so they add nothing — and hand the ids over as
            # int64 so fused_marlin_moe takes its generic alignment path,
            # which is fine with duplicate ids within a token.
            m = masked_topk_ids < 0
            masked_topk_output = topk_output._replace(
                topk_ids=torch.where(m, torch.zeros_like(masked_topk_ids), masked_topk_ids).to(torch.int64),
                topk_weights=torch.where(m, torch.zeros_like(topk_output.topk_weights), topk_output.topk_weights),
            )
        else:
            masked_topk_output = topk_output._replace(topk_ids=masked_topk_ids)
        masked_dispatch_output = dispatch_output._replace(
            topk_output=masked_topk_output
        )

        # Step 3: Execute GPU expert computation (any quantization method)
        # This runs in parallel with CPU computation. With no resident GPU
        # experts (--kt-num-gpu-experts 0, every slot given to the streamer
        # and the KV pool) there is nothing to run and the FlashInfer MXFP4
        # runner refuses a zero-expert workspace (09-16).
        if self.num_gpu_experts > 0:
            gpu_combine_input = self.gpu_method.apply(layer, masked_dispatch_output)
            output = gpu_combine_input.hidden_states
            if streamer is not None and _hot_check:
                from sglang.srt.layers.moe.kt_stream_prefill import hot_reference_check

                hot_reference_check(self, layer, x, physical_ids, topk_output.topk_weights, output, streamer)
        else:
            output = torch.zeros_like(x)
        if hot is not None and hot[1] is not None:
            output = output + kt_kv_hot_lend.apply(self, hot[1])

        # Step 4: Synchronize CPU results and merge with GPU results
        if streamer is not None:
            output = output + streamer.run(self, layer, dispatch_output, physical_ids)
            if cpu_share:
                output = output + self.sync(x)
        elif self.tp_rank == 0:
            cpu_output = self.sync(x)
            output = output + cpu_output

        kt_kv_hot_lend.check(self, layer, dispatch_output, hot, output)
        return StandardCombineInput(hidden_states=output)

    def __getattr__(self, name: str):
        """Delegate attribute access to the wrapped GPU method.

        This allows the wrapper to transparently expose attributes and methods
        from the wrapped GPU quantization method.

        Args:
            name: Attribute name

        Returns:
            Attribute value from gpu_method
        """
        # Avoid infinite recursion for internal attributes
        if name in ("gpu_method", "wrapper", "kt_config"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )

        return getattr(self.gpu_method, name)
