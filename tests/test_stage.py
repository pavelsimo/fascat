from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import StageOptions


def test_stage_material_modes_control_bindings_and_display_color() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        material_indices=np.array([0], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh, material_ids=["red"])},
        materials={"red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))},
    )

    cad = asset.stage(StageOptions(materials="cad", uv0="none", uv1=None))
    cad_part = cad.parts["part"]
    cad_mesh = cad_part.mesh

    assert set(cad.materials) == {"red"}
    assert cad_part.material_ids == ["red"]
    assert "display_color" not in cad_part.metadata
    assert cad_mesh is not None
    assert cad_mesh.material_indices is not None
    assert cad_mesh.material_indices.tolist() == [0]

    display = asset.stage(StageOptions(materials="display", uv0="none", uv1=None))
    display_part = display.parts["part"]
    display_mesh = display_part.mesh

    assert display.materials == {}
    assert display_part.material_ids == []
    assert display_part.metadata["display_color"] == "1.000000,0.000000,0.000000,1.000000"
    assert display_mesh is not None
    assert display_mesh.material_indices is None

    none = asset.stage(StageOptions(materials="none", uv0="none", uv1=None))
    none_part = none.parts["part"]

    assert none.materials == {}
    assert none_part.material_ids == []
    assert "display_color" not in none_part.metadata


def test_stage_respects_normals_false_and_uv0_none() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(normals=False, uv0=None, uv1=None))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.normals is None
    assert staged_mesh.uvs == {}
    assert staged.report.steps[-1].options["uv0"] == "none"


def test_stage_preserves_existing_normals_when_generation_is_disabled() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    ).compute_normals()
    original_normals = mesh.normals.copy()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(normals=False, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.normals is not None
    assert np.array_equal(staged_mesh.normals, original_normals)


@pytest.mark.requires_xatlas
def test_stage_unwrap_uv_uses_xatlas_backend() -> None:
    pytest.importorskip("xatlas")
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="unwrap", uv1="unwrap"))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0"] == "xatlas"
    assert staged_mesh.metadata["uv1"] == "xatlas"
    assert sorted(staged_mesh.uvs) == [0, 1]
