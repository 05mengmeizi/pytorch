from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeVar

import torch.library


P = ParamSpec("P")
R = TypeVar("R")

_OpOverrideFn = Callable[Concatenate[torch.DispatchKeySet, P], R]


libs = {}


def _get_library(lib_symbol: str, dispatch_key: str) -> torch.library.Library:
    """
    Return a `torch.library.Library` instance unique to the passed
    (lib_symbol, dispatch_key) pair. Create a new instance if necessary.
    """
    global libs

    if (lib_symbol, dispatch_key) not in libs:
        print(f"CREATING LIB: {lib_symbol=} : {dispatch_key=}")
        libs[(lib_symbol, dispatch_key)] = torch.library.Library(
            lib_symbol, "IMPL", dispatch_key
        )

    return libs[(lib_symbol, dispatch_key)]


def _register_op_override(
    lib_symbol: str,
    op_symbol: str,
    dispatch_key: str,
    impl: _OpOverrideFn,
    *,
    allow_override=False,
) -> None:
    """
    Register a passed override function to the dispatcher, based on the
    passed lib and op symbols, and the dispatch key.
    """
    lib = _get_library(lib_symbol, dispatch_key)

    lib.impl(
        op_symbol,
        impl,
        dispatch_key,
        with_keyset=True,
        allow_override=allow_override,
    )
