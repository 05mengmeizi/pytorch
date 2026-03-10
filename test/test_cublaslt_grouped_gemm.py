"""Tests for cuBLASLt grouped GEMM integration.

Validates cuBLASLt grouped GEMM backend produces results consistent
with the default CUTLASS backend.
"""

import os
import unittest

import torch
import torch.nn.functional as F
from torch.testing._internal.common_cuda import SM90OrLater
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
@unittest.skipIf(not SM90OrLater, "cuBLASLt grouped gemm requires SM90+")
class TestCublasLtGroupedGemm(TestCase):
    def _run_with_backend(self, backend, fn, *args, **kwargs):
        old = os.environ.get("TORCH_GROUPED_GEMM_BACKEND")
        try:
            if backend:
                os.environ["TORCH_GROUPED_GEMM_BACKEND"] = backend
            elif "TORCH_GROUPED_GEMM_BACKEND" in os.environ:
                del os.environ["TORCH_GROUPED_GEMM_BACKEND"]
            return fn(*args, **kwargs)
        finally:
            if old is not None:
                os.environ["TORCH_GROUPED_GEMM_BACKEND"] = old
            elif "TORCH_GROUPED_GEMM_BACKEND" in os.environ:
                del os.environ["TORCH_GROUPED_GEMM_BACKEND"]

    @parametrize("n_groups", [2, 4, 8])
    def test_bf16_grouped_gemm_2d_3d(self, n_groups):
        device = "cuda"
        m, n, k = 64, 128, 64
        dtype = torch.bfloat16

        a = torch.randn(m * n_groups, k, device=device, dtype=dtype)
        b = torch.randn(n_groups, n, k, device=device, dtype=dtype)
        offs = torch.arange(m, n_groups * m + 1, m, device=device, dtype=torch.int32)

        out_cutlass = self._run_with_backend(
            None, F.grouped_mm, a, b.transpose(-2, -1), offs=offs, out_dtype=dtype
        )
        out_cublaslt = self._run_with_backend(
            "cublaslt", F.grouped_mm, a, b.transpose(-2, -1), offs=offs, out_dtype=dtype
        )

        self.assertEqual(out_cutlass, out_cublaslt, atol=1e-2, rtol=1e-2)

    @parametrize("n_groups", [2, 4])
    def test_bf16_grouped_gemm_3d_3d(self, n_groups):
        device = "cuda"
        m, n, k = 64, 128, 64
        dtype = torch.bfloat16

        a = torch.randn(n_groups, m, k, device=device, dtype=dtype)
        b = torch.randn(n_groups, n, k, device=device, dtype=dtype)

        out_cutlass = self._run_with_backend(
            None, F.grouped_mm, a, b.transpose(-2, -1), out_dtype=dtype
        )
        out_cublaslt = self._run_with_backend(
            "cublaslt", F.grouped_mm, a, b.transpose(-2, -1), out_dtype=dtype
        )

        self.assertEqual(out_cutlass, out_cublaslt, atol=1e-2, rtol=1e-2)

    @parametrize("n_groups", [2, 4])
    def test_bf16_grouped_gemm_2d_2d(self, n_groups):
        device = "cuda"
        m, n, k = 64, 128, 64
        dtype = torch.bfloat16

        a = torch.randn(m, k * n_groups, device=device, dtype=dtype)
        b = torch.randn(n, k * n_groups, device=device, dtype=dtype)
        offs = torch.arange(k, n_groups * k + 1, k, device=device, dtype=torch.int32)

        out_cutlass = self._run_with_backend(
            None, F.grouped_mm, a, b.t(), offs=offs, out_dtype=dtype
        )
        out_cublaslt = self._run_with_backend(
            "cublaslt", F.grouped_mm, a, b.t(), offs=offs, out_dtype=dtype
        )

        self.assertEqual(out_cutlass, out_cublaslt, atol=1e-2, rtol=1e-2)


instantiate_parametrized_tests(TestCublasLtGroupedGemm)


if __name__ == "__main__":
    run_tests()
