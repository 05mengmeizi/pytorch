import functools
import logging

from .common_utils import (
    _available_version,
    _unavailable_reason,
    check_native_jit_disabled,
    check_native_version_skip,
)
from .registry import _OpOverrideFn, _register_op_override


log = logging.getLogger(__name__)


_TRITON_REQUIRED_VERSION_MAJOR = 3
_TRITON_MINIMUM_VERSION_MINOR = 6


@functools.cache
def _check_runtime_available() -> tuple[bool, tuple[int, int, int] | None]:
    """
    Check if triton is available

    NOTE: must not import at this point
    """

    deps = [
        ("triton", "triton"),
    ]
    reason = _unavailable_reason(deps)
    if reason is None:
        available = True
        version = _available_version("triton")
    else:
        log.info("triton native DSL ops require: `triton` %s", reason)
        available = False
        version = None
    return available, version


def runtime_available() -> bool:
    available, _ = _check_runtime_available()
    return available


def runtime_version() -> None | tuple[int, int, int]:
    _, version = _check_runtime_available()
    return version


def _version_is_sufficient() -> bool:
    _, version = _check_runtime_available()
    if version is None:
        return False
    if check_native_version_skip():
        return True
    # Either exact version, or same major
    major_ok = version[0] == _TRITON_REQUIRED_VERSION_MAJOR
    minor_ok = version[1] >= _TRITON_MINIMUM_VERSION_MINOR
    return major_ok and minor_ok


def register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    impl: _OpOverrideFn,
    *,
    allow_override=False,
) -> None:
    available, version = _check_runtime_available()
    if (not available) or check_native_jit_disabled():
        return

    if not _version_is_sufficient():
        log.warning(
            "triton version %s is not sufficient (>= (%s.%s.*)); "
            "set TORCH_NATIVE_SKIP_VERSION_CHECK=1 to override",
            version,
            _TRITON_REQUIRED_VERSION_MAJOR,
            _TRITON_MINIMUM_VERSION_MINOR,
        )
        return

    _register_op_override(
        lib_symbol,
        op_symbol,
        dispatch_key,
        impl,
        allow_override=allow_override,
    )
