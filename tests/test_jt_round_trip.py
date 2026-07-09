"""Round-trip regression: JT → GLB/USD output must preserve geometry, materials, and transforms.

JT import is pure stdlib+numpy and pre-tessellated, so unlike the STEP round
trips these tests need no OCP; the OCP-blocking test below proves it.
"""

from __future__ import annotations

import importlib.abc
import sys
from pathlib import Path

import numpy as np
import pytest

import fascat as fc
from fascat.io.gltf import _read_document, validate_gltf
from tests._jt_builder import SyntheticPart, build_jt

_TETRA_POINTS = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
_TETRA_FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
_JT_TRANSLATION = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [25.0, -5.0, 12.5, 1.0],
]


def _write_assembly_jt(path: Path) -> None:
    parts = [
        SyntheticPart(
            name="body.part",
            points=_TETRA_POINTS,
            triangles=_TETRA_FACES,
            diffuse=(0.8, 0.2, 0.1, 1.0),
            jt_transform=_JT_TRANSLATION,
            instances=1,
        ),
        SyntheticPart(
            name="lid.part",
            points=[[float(x) + 20.0, float(y), float(z)] for x, y, z in _TETRA_POINTS],
            triangles=_TETRA_FACES,
            diffuse=(0.1, 0.3, 0.9, 1.0),
        ),
    ]
    path.write_bytes(build_jt(parts))


def _exact_count_convert(source: Path, output: Path) -> fc.Asset:
    return fc.convert(
        source,
        output,
        optimize=fc.OptimizeOptions(simplify=False, optimize_buffers=False),
        lods=None,
    )


def test_jt_to_glb_round_trip_preserves_triangles_and_materials(tmp_path: Path) -> None:
    source = tmp_path / "assembly.jt"
    _write_assembly_jt(source)
    output = tmp_path / "round-trip.glb"

    asset = _exact_count_convert(source, output)
    stats = validate_gltf(output)
    document, _buffers = _read_document(output)

    assert asset.part_count == 2
    assert asset.occurrence_count == 3  # body + its instance + lid
    assert asset.triangle_count == 8  # per-part: two tetrahedra
    assert stats["triangles"] == 12  # occurrence-weighted: instance counted twice
    assert stats["meshes"] == asset.occurrence_count
    assert len(document.get("materials", [])) == len(asset.materials) == 2
    assert asset.report.steps[0].options["format"] == "JT"


def test_jt_transform_survives_to_gltf_nodes(tmp_path: Path) -> None:
    source = tmp_path / "assembly.jt"
    _write_assembly_jt(source)
    output = tmp_path / "round-trip.glb"

    asset = _exact_count_convert(source, output)
    document, _buffers = _read_document(output)
    # JT keeps translation in the last matrix row; after the transpose to column
    # vectors and glTF space normalization (mm -> m, Z-up -> Y-up) the JT
    # translation (25, -5, 12.5) mm lands as (0.025, 0.0125, 0.005) m.
    matrices = [node["matrix"] for node in document.get("nodes", []) if "matrix" in node]
    assert any(np.allclose(matrix[12:15], [0.025, 0.0125, 0.005]) for matrix in matrices)
    assert asset.units == "millimetre"


@pytest.mark.requires_usd
def test_jt_to_usdc_round_trip(tmp_path: Path) -> None:
    from fascat.io.usd import validate_usd

    source = tmp_path / "assembly.jt"
    _write_assembly_jt(source)
    output = tmp_path / "round-trip.usdc"

    asset = _exact_count_convert(source, output)
    stats = validate_usd(output)
    assert asset.triangle_count == 8
    assert stats["triangles"] == 12


class _BlockOcp(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
        if fullname == "OCP" or fullname.startswith("OCP."):
            raise ImportError("OCP import blocked: JT conversion must not require cadquery-ocp")
        return None


def test_jt_conversion_never_imports_ocp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mesh-reuse pipeline path must keep JT -> glTF conversion OCP-free."""
    source = tmp_path / "assembly.jt"
    _write_assembly_jt(source)
    output = tmp_path / "no-ocp.glb"

    for name in [name for name in sys.modules if name == "OCP" or name.startswith("OCP.")]:
        monkeypatch.delitem(sys.modules, name)
    blocker = _BlockOcp()
    sys.meta_path.insert(0, blocker)
    try:
        asset = _exact_count_convert(source, output)
    finally:
        sys.meta_path.remove(blocker)
    assert output.exists()
    assert asset.triangle_count == 8
