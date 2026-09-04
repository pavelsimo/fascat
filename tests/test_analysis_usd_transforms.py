from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fascat.analysis import _asset_from_usd, analyze_output
from fascat.asset import Asset, Node, Part
from fascat.io.usd import write_usd
from fascat.mesh import Mesh
from fascat.options import AnalyzeOptions, UsdExportOptions

Usd = pytest.importorskip("pxr.Usd")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
pytestmark = pytest.mark.requires_usd


def _triangle(stage: Any, path: str) -> Any:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr("none")
    return mesh


@pytest.mark.parametrize("reset_stack", [False, True])
def test_usd_analysis_composes_world_transforms_and_reset_stack(tmp_path: Path, reset_stack: bool) -> None:
    path = tmp_path / "nested.usda"
    stage = Usd.Stage.CreateNew(str(path))
    ancestor = UsdGeom.Xform.Define(stage, "/World")
    ancestor.AddTranslateOp().Set((10, 20, 30))
    scene = UsdGeom.Xform.Define(stage, "/World/Scene")
    stage.SetDefaultPrim(scene.GetPrim())
    scene.AddRotateZOp().Set(90)
    parent = UsdGeom.Xform.Define(stage, "/World/Scene/Parent")
    parent.AddScaleOp().Set((2, 3, 4))
    mesh = _triangle(stage, "/World/Scene/Parent/Triangle")
    mesh.AddTranslateOp().Set((1, 2, 3))
    mesh.SetResetXformStack(reset_stack)
    stage.GetRootLayer().Save()

    report = analyze_output(path)

    expected = [[1, 2, 3], [2, 2, 3], [1, 3, 3]] if reset_stack else [[4, 22, 42], [4, 24, 42], [1, 22, 42]]
    loaded_mesh = next(iter(_asset_from_usd(path).parts.values())).mesh
    assert loaded_mesh is not None
    np.testing.assert_allclose(loaded_mesh.points, expected, atol=1e-12)
    assert len(report.parts) == 1
    bounds = report.parts[0]["bounds"]
    assert isinstance(bounds, dict)
    np.testing.assert_allclose(bounds["min"], np.min(expected, axis=0))
    np.testing.assert_allclose(bounds["max"], np.max(expected, axis=0))


@pytest.mark.parametrize("meters_per_unit", [0.001, 0.01, 1.0])
@pytest.mark.parametrize("scale", [(0.01, 0.01, 0.01), (-20.0, 0.01, 2.0)])
def test_usd_analysis_measures_scaled_geometry_in_stage_units(
    tmp_path: Path, meters_per_unit: float, scale: tuple[float, float, float]
) -> None:
    path = tmp_path / "scaled.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    scene = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(scene.GetPrim())
    mesh = _triangle(stage, "/Scene/Triangle")
    mesh.AddScaleOp().Set(scale)
    stage.GetRootLayer().Save()
    options = AnalyzeOptions(tiny_parts=True, tiny_part_diagonal=0.1, sliver_triangles=True, sliver_aspect_ratio=10)

    report = analyze_output(path, options)

    expected_points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float) * scale
    expected_mesh = Mesh(points=expected_points, faces=np.asarray([[0, 1, 2]]))
    expected_asset = Asset(
        root=Node(id="root", name="root"), parts={"expected": Part(id="expected", name="expected", mesh=expected_mesh)}
    )
    expected = expected_asset.analyze(options).parts[0]
    actual = report.parts[0]
    assert actual["tiny"] is expected["tiny"]
    assert actual["sliver_triangles"] == expected["sliver_triangles"]
    assert actual["max_aspect_ratio"] == pytest.approx(expected["max_aspect_ratio"], rel=1e-6)
    assert actual["bounds"]["diagonal"] == pytest.approx(expected["bounds"]["diagonal"], rel=1e-6)
    np.testing.assert_allclose(actual["bounds"]["min"], np.min(expected_points, axis=0), rtol=1e-6)
    np.testing.assert_allclose(actual["bounds"]["max"], np.max(expected_points, axis=0), rtol=1e-6)


def test_usd_analysis_applies_exported_root_normalization_once(tmp_path: Path) -> None:
    path = tmp_path / "normalized.usda"
    root_transform = np.diag([0.001, 0.001, 0.001, 1.0])
    child_transform = np.eye(4)
    child_transform[:3, 3] = [1000, 2000, 3000]
    mesh = Mesh(points=np.asarray([[0, 0, 0], [1000, 0, 0], [0, 1000, 0]], dtype=float), faces=np.asarray([[0, 1, 2]]))
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            transform=root_transform,
            children=[Node(id="child", name="child", part_id="part", transform=child_transform)],
        ),
        parts={"part": Part(id="part", name="part", mesh=mesh)},
        meters_per_unit=1.0,
        up_axis="Y",
    )
    write_usd(asset, path, options=UsdExportOptions(layout="flat"))

    report = analyze_output(path, AnalyzeOptions(tiny_parts=True, tiny_part_diagonal=2))

    assert len(report.parts) == 1
    assert report.parts[0]["tiny"] is True
    np.testing.assert_allclose(report.parts[0]["bounds"]["min"], [1, 2, 3])
    np.testing.assert_allclose(report.parts[0]["bounds"]["max"], [2, 3, 3])


def test_usd_analysis_transforms_each_instance_without_mutating_shared_geometry(tmp_path: Path) -> None:
    path = tmp_path / "transformed_instances.usda"
    stage = Usd.Stage.CreateNew(str(path))
    scene = UsdGeom.Xform.Define(stage, "/Scene")
    scene.AddTranslateOp().Set((0, 10, 0))
    stage.SetDefaultPrim(scene.GetPrim())
    UsdGeom.Xform.Define(stage, "/Library/Shape")
    source = _triangle(stage, "/Library/Shape/Triangle")
    source.AddTranslateOp().Set((0, 0, 4))
    for name, translation, scale in (
        ("First", (10, 0, 0), (2, 3, 1)),
        ("Second", (-10, 0, 0), (-0.5, 0.25, 2)),
    ):
        occurrence = UsdGeom.Xform.Define(stage, f"/Scene/{name}")
        occurrence.AddTranslateOp().Set(translation)
        occurrence.AddScaleOp().Set(scale)
        occurrence.GetPrim().GetReferences().AddInternalReference("/Library/Shape")
        occurrence.GetPrim().SetInstanceable(True)
    assert stage.GetPrimAtPath("/Scene/First").GetPrototype() == stage.GetPrimAtPath("/Scene/Second").GetPrototype()
    stage.GetRootLayer().Save()

    asset = _asset_from_usd(path)

    first, second = [part.mesh for part in asset.parts.values()]
    assert first is not None and second is not None
    np.testing.assert_allclose(first.points, [[10, 10, 4], [12, 10, 4], [10, 13, 4]])
    np.testing.assert_allclose(second.points, [[-10, 10, 8], [-10.5, 10, 8], [-10, 10.25, 8]])
    assert not np.shares_memory(first.points, second.points)
    report = analyze_output(path)
    assert len(report.parts) == 2
    np.testing.assert_allclose(report.parts[0]["bounds"]["max"], [12, 13, 4])
    np.testing.assert_allclose(report.parts[1]["bounds"]["max"], [-10, 10.25, 8])
