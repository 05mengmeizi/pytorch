import functools
import logging

from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .registry import _RegisterFn, register_op_registerer


log = logging.getLogger(__name__)


_CUTEDSL_BLESSED_VERSIONS: set[tuple[int, int, int]] = {
    # Current version
    (4, 4, 1),
}


@functools.cache
def _check_runtime_available() -> tuple[bool, tuple[int, int, int] | None]:
    """
    Check if cutedsl (and deps) are available.

    NOTE: Doesn't import at this point
    """
    deps = [
        ("nvidia_cutlass_dsl", "cutlass"),
        ("apache_tvm_ffi", "tvm_ffi"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("nvidia_cutlass_dsl")
    else:
        log.info(
            "CuTeDSL operators require optional Python packages "
            "`nvidia-cutlass-dsl` and `apache-tvm-ffi`; "
            "%s",
            reason,
        )
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | tuple[int, int, int]:
    _, version = _check_runtime_available()
    return version


def _version_is_blessed() -> bool:
    _, version = _check_runtime_available()
    if version is None:
        return False
    if check_native_version_skip():
        return True
    return version in _CUTEDSL_BLESSED_VERSIONS


def register_op(fn: _RegisterFn) -> None:
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_blessed():
        log.warning(
            "cutedsl version %s is not blessed (blessed: %s); "
            "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
            version,
            _CUTEDSL_BLESSED_VERSIONS,
        )
        return

    register_op_registerer(fn)
