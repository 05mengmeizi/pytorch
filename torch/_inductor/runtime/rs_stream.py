"""Dedicated CUDA stream for reduce-scatter wait_tensor calls.

When enabled, RS wait_tensor blocks only the RS stream instead of the
compute stream, allowing AG/compute to proceed independently.
"""

from threading import local

import torch


_tls = local()


def get_rs_stream(device_idx: int) -> torch.cuda.Stream:
    """Switch to a cached high-priority RS CUDA stream.

    The compute stream waits on the RS stream before switching, so the
    RS wait_tensor sees all prior compute work.  After switching, the
    current CUDA stream is the RS stream.
    """
    key = f"rs_stream_{device_idx}"
    if not hasattr(_tls, key):
        setattr(_tls, key, torch.cuda.Stream(device=device_idx, priority=-1))
    prev_key = f"prev_stream_{device_idx}"
    prev_stream = torch.cuda.current_stream(device_idx)
    setattr(_tls, prev_key, prev_stream)
    rs_stream = getattr(_tls, key)
    # RS stream must wait for all prior work on compute stream
    rs_stream.wait_stream(prev_stream)
    torch.cuda.set_stream(rs_stream)
    return rs_stream


def restore_stream(device_idx: int) -> None:
    """Restore the previous CUDA stream."""
    prev_key = f"prev_stream_{device_idx}"
    prev = getattr(_tls, prev_key)
    torch.cuda.set_stream(prev)
