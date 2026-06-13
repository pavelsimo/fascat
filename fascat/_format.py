"""Human-readable count formatting for reprs and summaries."""

from __future__ import annotations


def human_count(value: int) -> str:
    """412 -> '412', 1_234 -> '1.2K', 1_200_000 -> '1.2M', 2_100_000_000 -> '2.1G'."""
    magnitude = abs(value)
    for threshold, suffix in ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "K")):
        if magnitude >= threshold:
            scaled = value / threshold
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(value)


_IRREGULAR_PLURALS = {"vertex": "vertices"}


def count_phrase(value: int, noun: str) -> str:
    """38 + 'material' -> '38 materials'; 1 -> '1 material'; 1_200_000 + 'triangle' -> '1.2M triangles'."""
    plural = noun if value == 1 else _IRREGULAR_PLURALS.get(noun, f"{noun}s")
    return f"{human_count(value)} {plural}"
