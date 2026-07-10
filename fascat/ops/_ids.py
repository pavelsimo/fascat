from __future__ import annotations

from collections.abc import Container


def unique_id(existing: Container[str], base: str) -> str:
    """Return an action resource id, preserving the historical ``_2`` suffix."""
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
