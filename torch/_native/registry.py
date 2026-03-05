from collections.abc import Callable


_RegisterFn = Callable[[], None]

_RegisteredFns: list[_RegisterFn] = []


def register_op_registerer(fn: _RegisterFn) -> None:
    _RegisteredFns.append(fn)


def register_all_operators() -> None:
    for fn in _RegisteredFns:
        fn()
