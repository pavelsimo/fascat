"""Human-readable count formatting and byte-unit conversion."""

from __future__ import annotations

BYTES_PER_MIB = 1024 * 1024


def mib_to_bytes(value: float) -> int:
    """Convert a user-facing "MB" size limit to bytes.

    Every size flag in fascat (``--max-file-size-mb``, ``--file-size-budget-mb``)
    is binary megabytes, so 10 always means 10,485,760 bytes.
    """
    return int(value * BYTES_PER_MIB)


_MAGNITUDES = ((1_000_000_000, "G"), (1_000_000, "M"), (1_000, "K"))


def human_count(value: int) -> str:
    """412 -> '412', 1_234 -> '1.2K', 1_200_000 -> '1.2M', 2_100_000_000 -> '2.1G'."""
    magnitude = abs(value)
    for index, (threshold, suffix) in enumerate(_MAGNITUDES):
        if magnitude < threshold:
            continue
        scaled = round(value / threshold, 1)
        if abs(scaled) >= 1000 and index > 0:
            # Rounding pushed the value onto the next magnitude: 999_999_999 is
            # "1G", not "1000M".
            threshold, suffix = _MAGNITUDES[index - 1]
            scaled = round(value / threshold, 1)
        text = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{text}{suffix}"
    return str(value)


_IRREGULAR_PLURALS = {"vertex": "vertices"}


def count_phrase(value: int, noun: str) -> str:
    """38 + 'material' -> '38 materials'; 1 -> '1 material'; 1_200_000 + 'triangle' -> '1.2M triangles'."""
    plural = noun if value == 1 else _IRREGULAR_PLURALS.get(noun, f"{noun}s")
    return f"{human_count(value)} {plural}"
