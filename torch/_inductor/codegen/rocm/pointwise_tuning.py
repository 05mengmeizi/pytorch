# mypy: allow-untyped-defs
"""Empirically tuned Triton configs for ROCm pointwise kernels."""

from __future__ import annotations

from typing import List, NamedTuple, Tuple


class RocmPointwiseParams1d(NamedTuple):
    dominant: Tuple[int, int, int]          # (XBLOCK, num_warps, num_stages)
    candidates: List[Tuple[int, int, int]]  # dominant first; full list for autotuning


class RocmPointwiseParams2d(NamedTuple):
    dominant: Tuple[int, int, int, int]          # (XBLOCK, YBLOCK, num_warps, num_stages)
    candidates: List[Tuple[int, int, int, int]]  # dominant first; full list for autotuning


# 1-D configs: (XBLOCK, num_warps, num_stages)
_P1: dict = {
     1: (  256, 1, 1),
     2: (    1, 2, 1),
     3: (  512, 1, 1),
     4: (  128, 2, 1),
     5: ( 1024, 2, 1),
     6: ( 1024, 4, 1),
     7: (   16, 2, 1),
     8: ( 4096, 4, 1),
     9: ( 1024, 8, 1),
    10: (    2, 4, 1),
    12: (  128, 1, 1),
    17: (    1, 8, 1),
    18: (   16, 8, 1),
    20: (    1, 1, 1),
    24: (    2, 1, 1),
    25: (   64, 1, 1),
    28: (   32, 4, 1),
    30: ( 4096, 1, 1),
    31: ( 8192, 4, 1),
    33: (    4, 2, 1),
    43: (   16, 1, 1),
}

# (upper_bound, [config_ids])  — first id is the no-autotune default
_DISPATCH_1D: List[Tuple] = [
    (          64, [ 2,  7, 10, 17, 25, 33]),   # xnumel ≤ 64
    (       32768, [ 1,  2,  3,  4, 10]),        # 64 < x ≤ 32 K
    (       65536, [ 1,  3,  4,  6, 12]),        # 32 K < x ≤ 64 K
    (      262144, [ 1,  3,  4, 30, 31, 43]),    # 64 K < x ≤ 256 K
    (     1048576, [ 1,  3,  4,  5,  6,  7]),    # 256 K < x ≤ 1 M
    (     2097152, [ 1,  3,  4, 12, 20, 24]),    # 1 M < x ≤ 2 M
    (     4194304, [ 1,  3,  4,  5,  6,  8]),    # 2 M < x ≤ 4 M
    (     8388608, [ 1,  3,  4,  5,  6,  7]),    # 4 M < x ≤ 8 M
    (    16777216, [ 1,  2,  3,  4,  5,  6]),    # 8 M < x ≤ 16 M
    (    33554432, [ 1,  3,  4,  5,  6,  8]),    # 16 M < x ≤ 32 M
    (    67108864, [ 1,  3,  4,  5,  6,  8]),    # 32 M < x ≤ 64 M
    (   134217728, [ 1,  3,  5,  6,  9, 12]),    # 64 M < x ≤ 128 M
    (  1073741824, [ 1,  3,  5,  6, 18, 28]),    # 128 M < x ≤ 1 G
]


def rocm_pointwise_params_1d(xnumel: int) -> RocmPointwiseParams1d:
    """Config params for a 1-D ROCm pointwise kernel, dispatched by xnumel."""
    for upper, indices in _DISPATCH_1D:
        if xnumel <= upper:
            params = [_P1[i] for i in indices]
            return RocmPointwiseParams1d(dominant=params[0], candidates=params)
    # xnumel > 1 G: fall back to the largest-range set
    params = [_P1[i] for i in _DISPATCH_1D[-1][1]]
    return RocmPointwiseParams1d(dominant=params[0], candidates=params)


# 2-D configs: (XBLOCK, YBLOCK, num_warps, num_stages)
_P2: dict = {
     1: (  64,   32, 4, 1),
     2: (   8,   64, 2, 1),
     3: (  64,    2, 2, 1),
     4: (   2,    4, 2, 1),
     5: (   1,  512, 4, 1),
     6: ( 256,   32, 8, 1),
     7: (  32,   64, 4, 1),
     8: (  16,   32, 4, 1),
     9: (  64,   64, 8, 1),
    10: (   4,  128, 4, 1),
    12: (1024,    2, 4, 1),
    13: (  64,   16, 4, 1),
    14: (  16,   64, 1, 1),
    15: (  64,  128, 2, 1),
    16: ( 256,    8, 2, 1),
    17: ( 128,   32, 8, 1),
    19: (  32,  256, 4, 1),
    20: (   4,  128, 8, 1),
    21: (  16,   16, 2, 1),
    22: ( 256,   32, 4, 1),
    23: (   4,  256, 4, 1),
    24: (  64,   32, 2, 1),
    26: (   8, 1024, 2, 1),
    28: ( 256,    4, 1, 1),
    30: (1024,   16, 8, 1),
    32: ( 256,    4, 2, 1),
    34: (  32,   32, 1, 1),
    42: (   2,  128, 1, 1),
    44: (   4, 2048, 4, 1),
    45: (  64,    2, 1, 1),
}

# (upper_bound, [config_ids])  — first id is the no-autotune default
_DISPATCH_2D: List[Tuple] = [
    (     1048576, [ 1,  2,  3,  4,  6,  8, 20, 42, 45]),              # ≤ 1 M
    (     2097152, [ 1,  2,  3,  7,  8, 10, 13, 21]),                   # 1 M – 2 M
    (     4194304, [ 1,  2,  3,  5,  6,  7,  8, 10, 12, 13, 14, 20]),  # 2 M – 4 M
    (     8388608, [ 1,  2,  3,  6,  7,  9, 15, 17, 22, 34, 44]),       # 4 M – 8 M
    (    16777216, [ 1,  6,  7,  8, 12, 13, 15, 20, 22, 24, 26, 28]),   # 8 M – 16 M
    (    33554432, [ 1,  5,  9, 12, 14, 16, 17, 19, 22, 23, 30, 32]),   # 16 M – 32 M
    (   268435456, [ 1,  2, 5,  9, 12, 14, 16, 17, 19, 22, 23, 30, 32]), # 32 M – 256 M
]


def rocm_pointwise_params_2d(xnumel: int, ynumel: int) -> RocmPointwiseParams2d:
    """Config params for a 2-D ROCm pointwise kernel, dispatched by xnumel * ynumel."""
    total = xnumel * ynumel
    for upper, indices in _DISPATCH_2D:
        if total <= upper:
            params = [_P2[i] for i in indices]
            return RocmPointwiseParams2d(dominant=params[0], candidates=params)
    # For total > 256 M: fall back to the largest-range set
    params = [_P2[i] for i in _DISPATCH_2D[-1][1]]
    return RocmPointwiseParams2d(dominant=params[0], candidates=params)
