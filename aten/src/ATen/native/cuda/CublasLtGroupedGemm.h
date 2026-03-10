#pragma once
#include <ATen/core/TensorBase.h>
#include <optional>

namespace at::cuda::detail {

TORCH_API bool cublaslt_grouped_gemm_supported();

TORCH_API void cublaslt_bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    std::optional<at::Tensor> offs,
    at::Tensor& out);

TORCH_API void cublaslt_f8f8bf16_grouped_mm(
    at::Tensor mat_a,
    at::Tensor mat_b,
    at::Tensor scale_a,
    at::Tensor scale_b,
    std::optional<at::Tensor> offs,
    bool use_fast_accum,
    at::Tensor& out);

} // namespace at::cuda::detail
