from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import AtlasOptions, StageOptions, UnwrapOptions


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


def test_stage_merges_equivalent_materials_and_normalizes_pbr_metadata() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh, material_ids=["red_a", "red_b"])},
        materials={
            "red_a": Material(id="red_a", name="Red A", base_color=(1.0, 0.0, 0.0, 0.8), roughness=0.0),
            "red_b": Material(id="red_b", name="Red B", base_color=(1.0, 0.0, 0.0, 0.8), roughness=0.0),
        },
    )

    staged = asset.stage(
        StageOptions(
            merge_equivalent_materials=True,
            material_mode="pbr",
            uv0="none",
            uv1=None,
        )
    )
    part = staged.parts["part"]
    staged_mesh = part.mesh

    assert list(staged.materials) == ["red_a"]
    assert part.material_ids == ["red_a"]
    assert staged_mesh is not None
    assert staged_mesh.material_indices is not None
    assert staged_mesh.material_indices.tolist() == [0, 0]
    material = staged.materials["red_a"]
    assert material.opacity == 0.8
    assert material.roughness == 0.04
    assert material.metadata["pbr_normalized"] == "true"


def test_stage_records_uv_and_atlas_workflow_metadata() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh, material_ids=["mat"])},
        materials={"mat": Material(id="mat", name="Mat", base_color=(0.2, 0.3, 0.4, 1.0))},
    )

    staged = asset.stage(
        StageOptions(
            uv0="box",
            uv1="box",
            unwrap=UnwrapOptions(texel_density=256.0, padding=4, max_stretch=0.15),
            atlas=AtlasOptions(enabled=True, max_size=2048),
        )
    )
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0_texel_density"] == "256.0"
    assert staged_mesh.metadata["uv1_padding"] == "4"
    assert staged_mesh.metadata["uv0_atlas_size"] == "2048"
    assert staged.materials["mat"].metadata["atlas"] == "atlas_0"
    assert staged.materials["mat"].metadata["texture_bake_hooks"] == "base_color,opacity"


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


def test_stage_generates_hard_edge_normals_and_tangents() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(
        StageOptions(
            normal_mode="hard_edges",
            hard_edge_angle=30.0,
            tangents=True,
            validate_normals=True,
            uv0="box",
            uv1=None,
        )
    )
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.vertex_count > mesh.vertex_count
    assert staged_mesh.normals is not None
    assert staged_mesh.tangents is not None
    assert staged_mesh.tangents.shape == (staged_mesh.vertex_count, 4)


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
