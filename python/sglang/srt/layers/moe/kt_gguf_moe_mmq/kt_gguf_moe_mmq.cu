// Generated from gguf_kernel.cu of SGLang 5ac69dd
// (branch perf/gguf-mmq-q2_0): ggml_moe_mmq, llama.cpp's MUL_MAT_ID MMQ on raw GGUF expert blocks
// (llama_mmq/, llama.cpp 23b0202, MIT, see llama_mmq/ggml/LICENSE).
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <mutex>

#include "llama_mmq/sgl_llama_mmq.h"

namespace {
void* llama_mmq_alloc(size_t nbytes, cudaStream_t stream) {
  return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(nbytes, stream);
}
void llama_mmq_free(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
}
}  // namespace

// MoE with llama.cpp's MMQ (MUL_MAT_ID path, llama_mmq/mmq_moe.cu): out[t, s, :] = W[topk_ids[t, s]] @ x(t, s)
// for every token t and slot s, with x(t, s) = X[t] (gate / up: each token is quantized once) or, with
// x_per_slot, X[t * top_k + s] (down projection). The routes are sorted by expert on the GPU (no host sync).
// Slots with an id < 0 or >= n_experts are skipped and their output rows are zero; the experts of a token must be
// distinct (as top-k gives them), a repeated one is treated the same way. fp32 output.
torch::Tensor ggml_moe_mmq(
    torch::Tensor X,         // [T, K] or [T * top_k, K]
    torch::Tensor W,         // [n_experts, >= row, row_bytes] raw GGUF blocks
    torch::Tensor topk_ids,  // [T, top_k] int32 / int64
    int64_t type,
    int64_t row,
    bool x_per_slot) {
  static std::once_flag allocator_installed;
  std::call_once(allocator_installed, [] { sgl_llama_mmq::set_allocator(llama_mmq_alloc, llama_mmq_free); });

  int64_t qk = 0, block_bytes = 0;
  TORCH_CHECK(
      sgl_llama_mmq::type_supported(type) && sgl_llama_mmq::type_info(type, &qk, &block_bytes),
      "ggml_moe_mmq: GGML type ",
      type,
      " is not supported by MMQ");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kUInt8 && W.dim() == 3, "ggml_moe_mmq: W must be 3D uint8");
  TORCH_CHECK(X.is_cuda() && X.dim() == 2, "ggml_moe_mmq: X must be a 2D CUDA tensor");
  TORCH_CHECK(
      X.scalar_type() == torch::kFloat || X.scalar_type() == torch::kHalf || X.scalar_type() == torch::kBFloat16,
      "ggml_moe_mmq: X must be float32, float16 or bfloat16");
  TORCH_CHECK(
      topk_ids.is_cuda() && topk_ids.dim() == 2 &&
          (topk_ids.scalar_type() == torch::kInt32 || topk_ids.scalar_type() == torch::kInt64),
      "ggml_moe_mmq: topk_ids must be a 2D int32 / int64 CUDA tensor");
  TORCH_CHECK(W.device() == X.device() && topk_ids.device() == X.device(), "ggml_moe_mmq: tensors on different devices");
  const int64_t n_experts = W.size(0);
  const int64_t row_bytes = W.size(2);
  TORCH_CHECK(row >= 0 && row <= W.size(1), "ggml_moe_mmq: row ", row, " out of range for W rows ", W.size(1));
  TORCH_CHECK(row_bytes % block_bytes == 0, "ggml_moe_mmq: W row bytes ", row_bytes, " not a multiple of ", block_bytes);
  const int64_t K = row_bytes / block_bytes * qk;
  TORCH_CHECK(X.size(1) == K, "ggml_moe_mmq: X has ", X.size(1), " columns, W rows hold ", K, " values");
  const int64_t T = topk_ids.size(0);
  const int64_t top_k = topk_ids.size(1);
  TORCH_CHECK(
      X.size(0) == (x_per_slot ? T * top_k : T),
      "ggml_moe_mmq: X has ",
      X.size(0),
      " rows, expected ",
      x_per_slot ? T * top_k : T);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  torch::Tensor out = torch::empty({T, top_k, row}, X.options().dtype(torch::kFloat));
  if (T == 0 || top_k == 0 || row == 0) {
    return out;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  // The kernel reads up to weight_tail_bytes() past the end of each row; for every expert but the last one that
  // lands in the next expert's rows, so only the storage after the last row of the last expert needs it.
  const int64_t tail_bytes = sgl_llama_mmq::weight_tail_bytes(type, K);
  bool weight_ok = W.stride(2) == 1 && W.stride(1) >= row_bytes && W.stride(1) % block_bytes == 0 &&
                   W.stride(0) >= (row - 1) * W.stride(1) + row_bytes && W.stride(0) % block_bytes == 0;
  if (weight_ok) {
    const int64_t needed = (n_experts - 1) * W.stride(0) + (row - 1) * W.stride(1) + row_bytes + tail_bytes;
    const int64_t available = static_cast<int64_t>(W.storage().nbytes()) - W.storage_offset();
    weight_ok = available >= needed;
  }
  torch::Tensor Wk = W;
  if (!weight_ok) {
    TORCH_WARN_ONCE(
        "ggml_moe_mmq: the weight storage does not extend ",
        tail_bytes,
        " bytes past the last row (ggml's row padding); a zero-padded copy is made on every call. "
        "Pad the weight storage once at load time.");
    const int64_t n = n_experts * row * row_bytes;
    torch::Tensor buf = torch::empty({n + tail_bytes}, W.options());
    buf.narrow(0, n, tail_bytes).zero_();
    Wk = buf.narrow(0, 0, n).view({n_experts, row, row_bytes});
    Wk.copy_(W.narrow(1, 0, row));
  }

  const torch::Tensor Xk = X.stride(1) == 1 ? X : X.contiguous();
  const torch::Tensor ids = topk_ids.is_contiguous() ? topk_ids : topk_ids.contiguous();
  const sgl_llama_mmq::XType xtype = Xk.scalar_type() == torch::kFloat ? sgl_llama_mmq::XType::F32
                                     : Xk.scalar_type() == torch::kHalf ? sgl_llama_mmq::XType::F16
                                                                        : sgl_llama_mmq::XType::BF16;
  const auto i32 = X.options().dtype(torch::kInt32);
  torch::Tensor ids32 = torch::empty({T * top_k}, i32);
  sgl_llama_mmq::moe_prepare_ids(
      ids.data_ptr(),
      ids.scalar_type() == torch::kInt64,
      T * top_k,
      top_k,
      n_experts,
      ids32.data_ptr<int32_t>(),
      out.data_ptr<float>(),
      row,
      stream);

  // Token chunks of at most moe_max_tokens_per_call() (mm_ids_helper's shared memory); scratch reused per chunk.
  const int64_t chunk = std::min<int64_t>(T, sgl_llama_mmq::moe_max_tokens_per_call(top_k, row, K));
  torch::Tensor ids_src1 = torch::empty({chunk * top_k}, i32);
  torch::Tensor ids_dst = torch::empty({chunk * top_k + 128}, i32);
  torch::Tensor expert_bounds = torch::empty({n_experts + 1}, i32);
  torch::Tensor quant_X =
      torch::empty({sgl_llama_mmq::q8_buffer_bytes(chunk * top_k, K)}, X.options().dtype(torch::kUInt8));
  const int64_t x_rows_per_token = x_per_slot ? top_k : 1;
  for (int64_t t0 = 0; t0 < T; t0 += chunk) {
    const int64_t n = std::min<int64_t>(chunk, T - t0);
    sgl_llama_mmq::moe_mul_mat(
        Wk.data_ptr(),
        n_experts,
        row,
        K,
        Wk.stride(1),
        Wk.stride(0),
        type,
        static_cast<const char*>(Xk.data_ptr()) + t0 * x_rows_per_token * Xk.stride(0) * Xk.element_size(),
        xtype,
        Xk.stride(0),
        x_per_slot,
        ids32.data_ptr<int32_t>() + t0 * top_k,
        n,
        top_k,
        ids_src1.data_ptr<int32_t>(),
        ids_dst.data_ptr<int32_t>(),
        expert_bounds.data_ptr<int32_t>(),
        quant_X.data_ptr(),
        out.data_ptr<float>() + t0 * top_k * row,
        stream);
  }
  return out;
}

// Zero bytes the kernel may read past the last row of the last expert for rows of K values.
// Dense MMQ (llama_mmq/mmq_types.cu mul_mat): out[M, rows] = X[M, K] @ W[rows, K]^T on raw GGUF blocks, the
// activations quantized to q8_1 here, accumulated in fp32 and written once in out's dtype. X and out may be row
// views of larger tensors (unit column stride); W's storage must extend ggml_moe_mmq_tail_bytes(type, K) past its
// last row (pad it once at load time). qbuf (uint8, ggml_mul_mat_mmq_q8_bytes(M, K) bytes or more) holds the
// quantized activations when given: one buffer reused by every call instead of one allocation of each size.
void ggml_mul_mat_mmq(torch::Tensor X, torch::Tensor W, int64_t type, torch::Tensor out, std::optional<torch::Tensor> qbuf) {
  static std::once_flag allocator_installed;
  std::call_once(allocator_installed, [] { sgl_llama_mmq::set_allocator(llama_mmq_alloc, llama_mmq_free); });

  int64_t qk = 0, block_bytes = 0;
  TORCH_CHECK(
      sgl_llama_mmq::type_supported(type) && sgl_llama_mmq::type_info(type, &qk, &block_bytes),
      "ggml_mul_mat_mmq: GGML type ", type, " is not supported by MMQ");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kUInt8 && W.dim() == 2 && W.stride(1) == 1,
              "ggml_mul_mat_mmq: W must be a 2D uint8 tensor with contiguous rows");
  TORCH_CHECK(X.is_cuda() && X.dim() == 2 && X.stride(1) == 1, "ggml_mul_mat_mmq: X must be 2D with unit column stride");
  TORCH_CHECK(out.is_cuda() && out.dim() == 2 && out.stride(1) == 1, "ggml_mul_mat_mmq: out must be 2D with unit column stride");
  auto xtype_of = [](const torch::Tensor& t) {
    TORCH_CHECK(t.scalar_type() == torch::kFloat || t.scalar_type() == torch::kHalf || t.scalar_type() == torch::kBFloat16,
                "ggml_mul_mat_mmq: float32, float16 or bfloat16 expected");
    return t.scalar_type() == torch::kFloat ? sgl_llama_mmq::XType::F32
           : t.scalar_type() == torch::kHalf ? sgl_llama_mmq::XType::F16
                                             : sgl_llama_mmq::XType::BF16;
  };
  const sgl_llama_mmq::XType xtype = xtype_of(X), otype = xtype_of(out);
  const int64_t rows = W.size(0), row_bytes = W.size(1), M = X.size(0);
  TORCH_CHECK(row_bytes % block_bytes == 0 && W.stride(0) % block_bytes == 0, "ggml_mul_mat_mmq: W rows are not whole blocks");
  const int64_t K = row_bytes / block_bytes * qk;
  TORCH_CHECK(X.size(1) == K, "ggml_mul_mat_mmq: X has ", X.size(1), " columns, W rows hold ", K, " values");
  TORCH_CHECK(out.size(0) == M && out.size(1) == rows, "ggml_mul_mat_mmq: out must be [M, rows]");
  if (M == 0 || rows == 0) return;
  const int64_t needed = (rows - 1) * W.stride(0) + row_bytes + sgl_llama_mmq::weight_tail_bytes(type, K);
  TORCH_CHECK(static_cast<int64_t>(W.storage().nbytes()) - W.storage_offset() >= needed,
              "ggml_mul_mat_mmq: the weight storage must extend ", sgl_llama_mmq::weight_tail_bytes(type, K),
              " bytes past the last row (pad it at load time)");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t q8_bytes = sgl_llama_mmq::q8_buffer_bytes(M, K);
  torch::Tensor quant_X;
  if (qbuf.has_value()) {
    TORCH_CHECK(qbuf->is_cuda() && qbuf->scalar_type() == torch::kUInt8 && qbuf->is_contiguous() &&
                    qbuf->numel() >= q8_bytes && qbuf->device() == X.device(),
                "ggml_mul_mat_mmq: qbuf must be a contiguous uint8 tensor of ", q8_bytes, " bytes or more on X's device");
    quant_X = *qbuf;
  } else {
    quant_X = torch::empty({q8_bytes}, X.options().dtype(torch::kUInt8));
  }
  sgl_llama_mmq::quantize_q8_1_mmq(X.data_ptr(), xtype, M, K, X.stride(0), type, quant_X.data_ptr(), stream);
  sgl_llama_mmq::mul_mat(W.data_ptr(), rows, K, W.stride(0), type, quant_X.data_ptr(), M, out.data_ptr(), otype,
                         out.stride(0), stream);
}

int64_t ggml_mul_mat_mmq_q8_bytes(int64_t M, int64_t K) {
  return sgl_llama_mmq::q8_buffer_bytes(M, K);
}

int64_t ggml_moe_mmq_tail_bytes(int64_t type, int64_t K) {
  int64_t qk = 0, block_bytes = 0;
  TORCH_CHECK(sgl_llama_mmq::type_info(type, &qk, &block_bytes), "ggml_moe_mmq: GGML type ", type,
              " is not in this build");
  return sgl_llama_mmq::weight_tail_bytes(type, K);
}


// The slots already sorted by expert (slot_expert [n] int32 non-decreasing, bounds [n_experts + 1] int32 on the
// device): no ids helper, one call. X: [n, K]; dst [n, 1, row] fp32.
torch::Tensor ggml_moe_mmq_sorted(
    torch::Tensor X, torch::Tensor W, torch::Tensor slot_expert, torch::Tensor bounds, int64_t type, int64_t row) {
  static std::once_flag allocator_installed;
  std::call_once(allocator_installed, [] { sgl_llama_mmq::set_allocator(llama_mmq_alloc, llama_mmq_free); });
  int64_t qk = 0, block_bytes = 0;
  TORCH_CHECK(sgl_llama_mmq::type_supported(type) && sgl_llama_mmq::type_info(type, &qk, &block_bytes),
              "ggml_moe_mmq_sorted: GGML type ", type, " is not supported by MMQ");
  TORCH_CHECK(W.is_cuda() && W.scalar_type() == torch::kUInt8 && W.dim() == 3, "ggml_moe_mmq_sorted: W must be 3D uint8");
  TORCH_CHECK(X.is_cuda() && X.dim() == 2, "ggml_moe_mmq_sorted: X must be a 2D CUDA tensor");
  TORCH_CHECK(slot_expert.is_cuda() && slot_expert.scalar_type() == torch::kInt32 && slot_expert.dim() == 1 &&
                  slot_expert.is_contiguous(), "ggml_moe_mmq_sorted: slot_expert must be a contiguous 1D int32 CUDA tensor");
  TORCH_CHECK(bounds.is_cuda() && bounds.scalar_type() == torch::kInt32 && bounds.is_contiguous() &&
                  bounds.numel() == W.size(0) + 1, "ggml_moe_mmq_sorted: bounds must be int32 [n_experts + 1]");
  const int64_t n_experts = W.size(0);
  const int64_t row_bytes = W.size(2);
  TORCH_CHECK(row >= 0 && row <= W.size(1), "ggml_moe_mmq_sorted: row out of range");
  TORCH_CHECK(row_bytes % block_bytes == 0, "ggml_moe_mmq_sorted: W row bytes not a multiple of the block");
  const int64_t K = row_bytes / block_bytes * qk;
  TORCH_CHECK(X.size(1) == K, "ggml_moe_mmq_sorted: X columns vs W row values");
  const int64_t T = slot_expert.size(0);
  TORCH_CHECK(X.size(0) == T, "ggml_moe_mmq_sorted: X rows vs slots");
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  torch::Tensor out = torch::empty({T, 1, row}, X.options().dtype(torch::kFloat));
  if (T == 0 || row == 0) return out;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const int64_t tail_bytes = sgl_llama_mmq::weight_tail_bytes(type, K);
  bool weight_ok = W.stride(2) == 1 && W.stride(1) >= row_bytes && W.stride(1) % block_bytes == 0 &&
                   W.stride(0) >= (row - 1) * W.stride(1) + row_bytes && W.stride(0) % block_bytes == 0;
  if (weight_ok) {
    const int64_t needed = (n_experts - 1) * W.stride(0) + (row - 1) * W.stride(1) + row_bytes + tail_bytes;
    const int64_t available = static_cast<int64_t>(W.storage().nbytes()) - W.storage_offset();
    weight_ok = available >= needed;
  }
  torch::Tensor Wk = W;
  if (!weight_ok) {
    const int64_t n = n_experts * row * row_bytes;
    torch::Tensor buf = torch::empty({n + tail_bytes}, W.options());
    buf.narrow(0, n, tail_bytes).zero_();
    Wk = buf.narrow(0, 0, n).view({n_experts, row, row_bytes});
    Wk.copy_(W.narrow(1, 0, row));
  }
  const torch::Tensor Xk = X.stride(1) == 1 ? X : X.contiguous();
  const sgl_llama_mmq::XType xtype = Xk.scalar_type() == torch::kFloat ? sgl_llama_mmq::XType::F32
                                     : Xk.scalar_type() == torch::kHalf ? sgl_llama_mmq::XType::F16
                                                                        : sgl_llama_mmq::XType::BF16;
  // slot chunks of at most moe_max_tokens_per_call() (mm_ids_helper's shared memory; a calibration prompt routes
  // 16K rows x 6 to a handful of experts): the chunk's bounds are the layer's, shifted and clamped to it
  const int64_t chunk = std::min<int64_t>(T, sgl_llama_mmq::moe_max_tokens_per_call(1, row, K));
  const auto i32 = X.options().dtype(torch::kInt32);
  torch::Tensor ids_src1 = torch::empty({chunk}, i32);
  torch::Tensor ids_dst = torch::empty({chunk + 128}, i32);
  torch::Tensor expert_bounds = torch::empty({n_experts + 1}, i32);
  torch::Tensor quant_X = torch::empty({sgl_llama_mmq::q8_buffer_bytes(chunk, K)}, X.options().dtype(torch::kUInt8));
  for (int64_t t0 = 0; t0 < T; t0 += chunk) {
    const int64_t n = std::min<int64_t>(chunk, T - t0);
    torch::Tensor cb = (T <= chunk) ? bounds : (bounds - (int32_t)t0).clamp_(0, (int64_t)n).contiguous();
    sgl_llama_mmq::moe_mul_mat(Wk.data_ptr(), n_experts, row, K, Wk.stride(1), Wk.stride(0), type,
                               static_cast<const char*>(Xk.data_ptr()) + t0 * Xk.stride(0) * Xk.element_size(), xtype,
                               Xk.stride(0), /*x_per_slot=*/false,
                               slot_expert.data_ptr<int32_t>() + t0, n, /*n_used=*/1,
                               ids_src1.data_ptr<int32_t>(), ids_dst.data_ptr<int32_t>(), expert_bounds.data_ptr<int32_t>(),
                               quant_X.data_ptr(), out.data_ptr<float>() + t0 * row, stream, cb.data_ptr<int32_t>());
  }
  return out;
}
