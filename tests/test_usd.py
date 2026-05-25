from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from fascat.asset import Asset, Node, Part
from fascat.io.usd import validate_usd, write_usd
from fascat.material import Material
from fascat.mesh import Mesh


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
