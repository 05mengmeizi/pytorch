"""Dedicated CUDA streams for collective operations.

Provides separate AG and RS streams so neither blocks the compute stream,
matching the FSDP2 stream separation approach.
"""

from threading import local

import torch


_tls = local()


def _get_stream(name: str, device_idx: int) -> torch.cuda.Stream:
    key = f"{name}_{device_idx}"
    if not hasattr(_tls, key):
        setattr(_tls, key, torch.cuda.Stream(device=device_idx, priority=-1))
    return getattr(_tls, key)


def switch_to_comm_stream(name: str, device_idx: int) -> None:
    """Switch to a dedicated comm stream (ag or rs). Saves previous stream."""
    prev_key = f"prev_stream_{device_idx}"
    setattr(_tls, prev_key, torch.cuda.current_stream(device_idx))
    comm_stream = _get_stream(name, device_idx)
    comm_stream.wait_stream(torch.cuda.current_stream(device_idx))
    torch.cuda.set_stream(comm_stream)


def restore_stream(device_idx: int) -> None:
    """Restore the previous stream without sync (for start nodes)."""
    prev_key = f"prev_stream_{device_idx}"
    prev = getattr(_tls, prev_key)
    torch.cuda.set_stream(prev)


def restore_stream_with_sync(device_idx: int) -> None:
    """Restore the previous stream and make it wait on the comm stream.

    Use after wait nodes where the output must be visible on the compute stream.
    """
    prev_key = f"prev_stream_{device_idx}"
    prev = getattr(_tls, prev_key)
    comm_stream = torch.cuda.current_stream(device_idx)
    prev.wait_stream(comm_stream)
    torch.cuda.set_stream(prev)
