import struct
from pathlib import Path

import pytest

from fascat.io.stl import validate_stl

_ASCII = """solid triangle
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid triangle
"""
_BINARY = b"solid binary header".ljust(80, b"\0") + struct.pack("<I12fH", 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0x8001)


@pytest.mark.parametrize("payload", [_ASCII.encode(), _BINARY, (_ASCII + _ASCII).encode()])
def test_stl_accepts_complete_facets(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "triangle.stl"
    path.write_bytes(payload)
    triangles = 2 if payload == (_ASCII + _ASCII).encode() else 1
    assert validate_stl(path) == {"meshes": 1, "points": triangles * 3, "triangles": triangles}


@pytest.mark.parametrize(
    "payload",
    [
        b"facet normal 0 0 1\n",
        b"\0" * 84,
        b"solid empty\nendsolid empty\n",
        _ASCII.replace("      vertex 0 1 0\n", "").encode(),
        _ASCII.replace("    endloop", "      vertex 0 0 0\n    endloop").encode(),
        _ASCII.replace("  endfacet\n", "").encode(),
        _ASCII.replace("endsolid triangle\n", "").encode(),
        _ASCII.replace("vertex 1 0 0", "vertex nan 0 0").encode(),
        _ASCII.replace("normal 0 0 1", "normal 0 inf 1").encode(),
        _ASCII.replace("vertex 1 0 0", "vertex bad 0 0").encode(),
        _BINARY[:-1],
        _BINARY + b"extra",
        _ASCII.encode() + b"\xff",
    ],
)
def test_stl_rejects_malformed_or_empty_meshes(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "invalid.stl"
    path.write_bytes(payload)
    with pytest.raises(RuntimeError):
        validate_stl(path)


@pytest.mark.parametrize("component", [0, 3, 11])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_binary_stl_rejects_non_finite_vectors(tmp_path: Path, component: int, value: float) -> None:
    payload = bytearray(_BINARY)
    struct.pack_into("<f", payload, 84 + 4 * component, value)
    path = tmp_path / "invalid.stl"
    path.write_bytes(payload)
    with pytest.raises(RuntimeError, match="non-finite"):
        validate_stl(path)


def test_ascii_stl_accepts_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "bom.stl"
    path.write_text(_ASCII, encoding="utf-8-sig")
    assert validate_stl(path) == {"meshes": 1, "points": 3, "triangles": 1}
