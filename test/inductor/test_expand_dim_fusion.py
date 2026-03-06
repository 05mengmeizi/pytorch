# Owner(s): ["module: inductor"]
"""
Tests for expand_dimension_for_pointwise_nodes fusion optimization.

This feature expands a smaller node's iteration domain to match a larger node's,
using out mask (masked tl.load/tl.store) to skip out-of-bounds accesses, enabling
kernel fusion between nodes with mismatched iteration domains.

References:
  - pytorch#125075: cat + to_fp16 fusion
  - vllm#24917 / vllm#25477: mul + pad + addmm fusion
"""

import re
import unittest

import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


def _count_triton_kernels(code: str) -> int:
    """Count the number of triton kernel definitions in generated code."""
    return len(re.findall(r"def triton_\w+\(", code))


class TestExpandDimFusion(TestCase):
    """Tests for padding fusion via expand_dimension_for_pointwise_nodes."""

    # ─── Case 1: cat + to_fp16 (pytorch#125075) ─────────────────────────

    @requires_gpu()
    @inductor_config.patch(expand_dimension_for_pointwise_nodes=True)
    def test_cat_to_fp16(self):
        """cat [30528,768] + to_fp16 [30522,768] should fuse into 1 kernel."""

        def fn(x):
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(30522, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)
        ref = fn(x)

        self.assertTrue(torch.allclose(result[0], ref[0]))
        self.assertTrue(torch.allclose(result[1], ref[1]))
        self.assertEqual(
            _count_triton_kernels(code),
            1,
            "cat and to_fp16 should be fused via dimension expansion.",
        )

    # ─── Case 2: mul + pad + addmm (vllm#24917) ────────────────────────

    @requires_gpu()
    @inductor_config.patch(expand_dimension_for_pointwise_nodes=True)
    def test_mul_pad_addmm(self):
        """mul [4096,2880] + pad [4096,3072] should fuse into 1 triton kernel."""

        def fn(x, scale, bias, weight):
            mul_result = x * scale
            padded = torch.nn.functional.pad(mul_result, [0, 192])
            mm_result = torch.addmm(bias, mul_result, weight)
            return padded, mm_result

        x = torch.randn(4096, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(4096, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(2880, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x, scale, bias, weight)
        ref = fn(x, scale, bias, weight)

        self.assertTrue(torch.allclose(result[0], ref[0]))
        self.assertTrue(torch.allclose(result[1], ref[1], atol=1e-2, rtol=1e-2))
        self.assertEqual(
            _count_triton_kernels(code),
            1,
            "mul and pad should be fused; addmm is an extern kernel.",
        )

    # ─── Large expansion ratio ──────────────────────────────────────────

    @requires_gpu()
    @inductor_config.patch(expand_dimension_for_pointwise_nodes=True)
    def test_large_expansion_ratio(self):
        """50% expansion ratio should still fuse — no ratio guard."""

        def fn(x):
            z = torch.cat([x, torch.zeros([512, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(512, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)
        ref = fn(x)

        self.assertTrue(torch.allclose(result[0], ref[0]))
        self.assertTrue(torch.allclose(result[1], ref[1]))
        self.assertEqual(
            _count_triton_kernels(code),
            1,
            "Large expansion ratio should not prevent fusion.",
        )

    # ─── Config ─────────────────────────────────────────────────────────

    def test_expand_disabled_by_default(self):
        """Feature is disabled by default."""
        self.assertFalse(inductor_config.expand_dimension_for_pointwise_nodes)


if __name__ == "__main__":
    unittest.main()
