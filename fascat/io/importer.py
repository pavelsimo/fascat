from __future__ import annotations

from pathlib import Path

from fascat.asset import Asset
from fascat.io.jt import JT_SUFFIXES, read_jt, read_jt_bytes
from fascat.io.step import read_step, read_step_bytes

STEP_SUFFIXES = {".step", ".stp"}
CAD_INPUT_SUFFIXES = STEP_SUFFIXES | JT_SUFFIXES


def read_cad(path: str | Path) -> Asset:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in STEP_SUFFIXES:
        return read_step(source)
    if suffix in JT_SUFFIXES:
        return read_jt(source)
    raise ValueError(f"unsupported CAD input extension: {source.suffix or '<none>'}. Use .step, .stp, or .jt.")


def read_cad_bytes(data: bytes, *, name: str = "stdin.step") -> Asset:
    suffix = Path(name).suffix.lower()
    if suffix in STEP_SUFFIXES:
        return read_step_bytes(data, name=name)
    if suffix in JT_SUFFIXES:
        return read_jt_bytes(data, name=name)
    raise ValueError(f"unsupported CAD input extension: {suffix or '<none>'}. Use .step, .stp, or .jt.")
