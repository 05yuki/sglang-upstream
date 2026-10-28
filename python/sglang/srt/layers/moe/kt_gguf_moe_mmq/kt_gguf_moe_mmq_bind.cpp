#include <torch/extension.h>
torch::Tensor ggml_moe_mmq(torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids, int64_t type, int64_t row,
                           bool x_per_slot);
int64_t ggml_moe_mmq_tail_bytes(int64_t type, int64_t K);
void ggml_mul_mat_mmq(torch::Tensor X, torch::Tensor W, int64_t type, torch::Tensor out, std::optional<torch::Tensor> qbuf);
int64_t ggml_mul_mat_mmq_q8_bytes(int64_t M, int64_t K);
torch::Tensor ggml_moe_mmq_sorted(torch::Tensor X, torch::Tensor W, torch::Tensor slot_expert, torch::Tensor bounds,
                                  int64_t type, int64_t row);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ggml_moe_mmq", &ggml_moe_mmq);
  m.def("ggml_moe_mmq_sorted", &ggml_moe_mmq_sorted);
  m.def("ggml_moe_mmq_tail_bytes", &ggml_moe_mmq_tail_bytes);
  m.def("ggml_mul_mat_mmq", &ggml_mul_mat_mmq, pybind11::arg("X"), pybind11::arg("W"), pybind11::arg("type"),
        pybind11::arg("out"), pybind11::arg("qbuf") = pybind11::none());
  m.def("ggml_mul_mat_mmq_q8_bytes", &ggml_mul_mat_mmq_q8_bytes);
}
