from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from fascat.analysis import _asset_from_usd, analyze_output
from fascat.asset import Asset, Node, Part
from fascat.io.usd import validate_usd, write_usd
from fascat.mesh import Mesh
from fascat.options import AnalyzeOptions, UsdExportOptions
from fascat.visual import VisualPreviewOptions, write_output_preview

pytestmark = pytest.mark.requires_usd
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402


def _assert_occurrence_analysis_and_preview(output: Path, count: int, tmp_path: Path) -> None:
    stats = validate_usd(output)
    assert stats == {"meshes": count, "points": 3 * count, "triangles": count}

    report = analyze_output(output, AnalyzeOptions(open_boundaries=True, draw_call_estimate=True))
    assert report.stats["occurrences"] == count
    assert report.stats["validated_meshes"] == count
    assert report.summary["parts"] == count
    assert report.summary["vertices"] == 3 * count
    assert report.summary["triangles"] == count
    assert report.summary["boundary_edges"] == 3 * count
    assert report.summary["draw_call_mesh_instances"] == count
    assert len(report.parts) == count
    assert not any("validation stats only" in warning for warning in report.warnings)

    preview_path = tmp_path / "preview.png"
    preview = write_output_preview(
        output, preview_path, VisualPreviewOptions(width=64, height=64, padding=8, supersample=1)
    )
    assert preview.occurrences == count
    assert preview.meshes == count
    assert preview.triangles == count
    assert not preview.warnings
    with Image.open(preview_path) as image:
        assert image.size == (64, 64)
        assert len(image.getcolors(maxcolors=4096) or []) > 1


@pytest.mark.parametrize("suffix", [".usda", ".usdc"])
def test_exported_usd_instances_support_geometry_analysis_and_preview(tmp_path: Path, suffix: str) -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[Node(id="a", name="A", part_id="triangle"), Node(id="b", name="B", part_id="triangle")],
        ),
        parts={"triangle": Part(id="triangle", name="Triangle", mesh=mesh)},
    )
    output = tmp_path / f"instances{suffix}"
    write_usd(asset, output, options=UsdExportOptions(layout="instanced"))

    stage = Usd.Stage.Open(str(output))
    assert stage.GetPrimAtPath("/Scene/A").IsInstance()
    assert stage.GetPrimAtPath("/Scene/B").IsInstance()
    assert stage.GetPrimAtPath("/__Prototypes/triangle_lod0/Mesh")
    _assert_occurrence_analysis_and_preview(output, 2, tmp_path)


def _define_triangle(stage: Usd.Stage, path: str) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr("none")


def test_nested_usd_instances_and_real_meshes_are_analyzed_once_per_occurrence(tmp_path: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Scene").GetPrim())
    UsdGeom.Xform.Define(stage, "/Library/Shape")
    _define_triangle(stage, "/Library/Shape/Mesh")
    _define_triangle(stage, "/Library/UnusedMesh")
    UsdGeom.Xform.Define(stage, "/Library/Assembly")
    for name in ("A", "B"):
        inner = UsdGeom.Xform.Define(stage, f"/Library/Assembly/{name}").GetPrim()
        inner.GetReferences().AddInternalReference("/Library/Shape")
        inner.SetInstanceable(True)
    for name in ("First", "Second"):
        outer = UsdGeom.Xform.Define(stage, f"/Scene/{name}").GetPrim()
        outer.GetReferences().AddInternalReference("/Library/Assembly")
        outer.SetInstanceable(True)
    _define_triangle(stage, "/Scene/RealMesh")
    _define_triangle(stage, "/Scene/InactiveMesh")
    stage.GetPrimAtPath("/Scene/InactiveMesh").SetActive(False)
    assert stage.GetPrimAtPath("/Scene/First").IsInstance()
    assert stage.GetPrimAtPath("/Scene/First/A").IsInstance()
    assert stage.GetPrimAtPath("/Scene/First/A/Mesh").IsInstanceProxy()
    output = tmp_path / "nested.usda"
    stage.GetRootLayer().Export(str(output))

    _assert_occurrence_analysis_and_preview(output, 5, tmp_path)
    decoded_paths: list[str] = []
    get_points_attr = UsdGeom.Mesh.GetPointsAttr

    def record_points_read(mesh: UsdGeom.Mesh) -> Usd.Attribute:
        decoded_paths.append(str(mesh.GetPath()))
        return get_points_attr(mesh)

    with patch.object(UsdGeom.Mesh, "GetPointsAttr", new=record_points_read):
        loaded = _asset_from_usd(output)
    assert len(decoded_paths) == len(set(decoded_paths)) == 2
    assert len({node.id for node in loaded.root.children}) == 5
    assert len({node.part_id for node in loaded.root.children}) == 5
