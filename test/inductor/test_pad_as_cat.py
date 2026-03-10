# Owner(s): ["module: inductor"]
"""Tests for cat multi-consumer and pad-as-cat optimizations."""

import re

import torch
from torch._dynamo.utils import counters
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


def _count_triton_kernels(code: str) -> int:
    return len(re.findall(r"def triton_\w+\(", code))


class TestCatMultiConsumer(TestCase):

    @requires_gpu()
    def test_cat_to_fp16(self):
        """Multi-consumer cat avoids duplicate computation."""

        def fn(x):
            z = torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)
            y = x.to(torch.float16)
            return z, y

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x)
        ref = fn(x)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1])
        kernel_count = _count_triton_kernels(code)
        self.assertLessEqual(
            kernel_count,
            2,
            f"Expected at most 2 kernels (fused + fill), got {kernel_count}.",
        )

    @requires_gpu()
    def test_single_consumer_cat_unchanged(self):
        """Single-consumer cat unchanged."""

        def fn(x):
            return torch.cat([x, torch.zeros([6, 768], device=GPU_TYPE)], dim=0)

        x = torch.randn(1024, 768, device=GPU_TYPE)
        compiled = torch.compile(fn)
        result = compiled(x)
        ref = fn(x)

        self.assertEqual(result, ref)


class TestPadAsCat(TestCase):

    @requires_gpu()
    def test_mul_pad_addmm(self):
        """Multi-consumer F.pad uses ConcatKernel zero-copy."""
        counters.clear()

        def fn(x, scale, bias, weight):
            mul_result = x * scale
            padded = torch.nn.functional.pad(mul_result, [0, 192])
            mm_result = torch.addmm(bias, mul_result, weight)
            return padded, mm_result

        x = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        scale = torch.randn(128, 2880, device=GPU_TYPE, dtype=torch.bfloat16)
        bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.bfloat16)
        weight = torch.randn(2880, 1024, device=GPU_TYPE, dtype=torch.bfloat16)

        compiled = torch.compile(fn)
        result, (code,) = run_and_get_code(compiled, x, scale, bias, weight)
        ref = fn(x, scale, bias, weight)

        self.assertEqual(result[0], ref[0])
        self.assertEqual(result[1], ref[1], atol=1e-2, rtol=1e-2)
        self.assertIn("reinterpret_tensor", code)
        self.assertGreater(counters["inductor"]["pad_as_cat"], 0)

    @requires_gpu()
    def test_single_consumer_pad_unchanged(self):
        """Single-consumer F.pad skips _pad_as_cat."""
        counters.clear()

        def fn(x, scale):
            return torch.nn.functional.pad(x * scale, [0, 192])

        x = torch.randn(128, 2880, device=GPU_TYPE)
        scale = torch.randn(128, 2880, device=GPU_TYPE)

        compiled = torch.compile(fn)
        result = compiled(x, scale)
        ref = fn(x, scale)

        self.assertEqual(result, ref)
        self.assertEqual(counters["inductor"]["pad_as_cat"], 0)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
