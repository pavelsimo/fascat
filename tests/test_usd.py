from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.io.usd import validate_usd, write_usd
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import OptimizeOptions

pytestmark = pytest.mark.requires_usd
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom  # noqa: E402


def cube_mesh() -> Mesh:
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return Mesh(points=points, faces=faces).compute_normals().box_uv()


def test_usd_export_authors_mesh_material_units_and_lods(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(id="root", name="root", children=[Node(id="n1", name="Cube Occurrence", part_id="cube")])
    material = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))
    asset = Asset(
        root=root,
        parts={
            "cube": Part(id="cube", name="Cube", mesh=mesh, material_ids=["red"], lod_meshes=[mesh.simplify(ratio=0.5)])
        },
        materials={"red": material},
    )
    output = tmp_path / "cube.usda"

    write_usd(asset, output)
    stats = validate_usd(output)

    assert stats["meshes"] == 1
    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.001
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    prim = stage.GetPrimAtPath("/Scene/Cube_Occurrence")
    assert prim.GetVariantSets().GetVariantSet("lod").GetVariantSelection() == "lod0"
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    assert UsdGeom.Mesh(mesh_prim).GetSubdivisionSchemeAttr().Get() == "none"
    assert UsdGeom.Mesh(mesh_prim).GetDisplayColorAttr().Get()[0] == (1.0, 0.0, 0.0)


def test_usd_export_uses_instanceable_references_for_repeated_parts(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="n1", name="Cube A", part_id="cube"),
            Node(id="n2", name="Cube B", part_id="cube"),
        ],
    )
    asset = Asset(root=root, parts={"cube": Part(id="cube", name="Cube", mesh=mesh)}, materials={})
    output = tmp_path / "instances.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetPrimAtPath("/Scene/Cube_A").IsInstanceable()
    assert stage.GetPrimAtPath("/Scene/Cube_B").IsInstanceable()
    assert stage.GetPrimAtPath("/__Prototypes/cube_lod0/Mesh")


def test_usd_export_marks_repeated_parts_instanceable_across_assemblies(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="g1", name="Group A", children=[Node(id="n1", name="Cube", part_id="cube")]),
            Node(id="g2", name="Group B", children=[Node(id="n2", name="Cube", part_id="cube")]),
        ],
    )
    asset = Asset(root=root, parts={"cube": Part(id="cube", name="Cube", mesh=mesh)}, materials={})
    output = tmp_path / "nested-instances.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetPrimAtPath("/Scene/Group_A/Cube").IsInstanceable()
    assert stage.GetPrimAtPath("/Scene/Group_B/Cube").IsInstanceable()


def test_usd_export_authors_face_material_subsets(tmp_path: Path) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )
    root = Node(id="root", name="root", children=[Node(id="node", name="Panel", part_id="panel")])
    red = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))
    blue = Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0))
    asset = Asset(
        root=root,
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["red", "blue"])},
        materials={"red": red, "blue": blue},
    )
    output = tmp_path / "subsets.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    subsets = [prim for prim in Usd.PrimRange(mesh_prim) if prim.GetTypeName() == "GeomSubset"]
    subset_indices = sorted(UsdGeom.Subset(prim).GetIndicesAttr().Get()[0] for prim in subsets)

    assert len(subsets) == 2
    assert subset_indices == [0, 1]


def test_usd_export_does_not_instance_when_instances_are_not_preserved(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="n1", name="Cube A", part_id="cube"),
            Node(id="n2", name="Cube B", part_id="cube"),
        ],
    )
    asset = Asset(root=root, parts={"cube": Part(id="cube", name="Cube", mesh=mesh)}, materials={})
    optimized = asset.optimize(OptimizeOptions(simplify=False, optimize_buffers=False, preserve_instances=False))
    output = tmp_path / "duplicates.usda"

    write_usd(optimized, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert not stage.GetPrimAtPath("/Scene/Cube_A").IsInstanceable()
    assert not stage.GetPrimAtPath("/Scene/Cube_B").IsInstanceable()


def test_usd_export_authors_debug_metadata_and_display_color_only_materials(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(id="root", name="root", children=[Node(id="n1", name="Cube", part_id="cube")])
    part = Part(
        id="cube",
        name="Cube",
        mesh=mesh,
        metadata={"display_color": "0.100000,0.200000,0.300000,0.400000"},
    )
    asset = Asset(root=root, parts={"cube": part}, materials={})
    output = tmp_path / "debug.usda"

    write_usd(asset, output, debug=True)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetDefaultPrim().GetCustomDataByKey("fascat:debug") is True
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    usd_mesh = UsdGeom.Mesh(mesh_prim)

    assert usd_mesh.GetDisplayColorAttr().Get()[0] == (0.1, 0.2, 0.3)
    assert usd_mesh.GetDisplayOpacityAttr().Get()[0] == pytest.approx(0.4)
