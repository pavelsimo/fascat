from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

from fascat.errors import FascatError, FascatIOError

P = ParamSpec("P")
R = TypeVar("R")


def wrap_io_errors(operation: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap public I/O failures in FascatIOError without hiding programmer bugs."""

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except FascatError:
                raise
            except (RuntimeError, ValueError, OSError) as exc:
                raise FascatIOError(f"{operation} failed: {exc}") from exc

        return wrapper

    return decorate
