from __future__ import annotations

__all__ = ["Error", "FascatError", "FascatIOError"]


class FascatError(Exception):
    """Common base class for fascat exceptions."""


class FascatIOError(FascatError):
    """Raised when reading or writing an asset file fails."""


Error = FascatError
