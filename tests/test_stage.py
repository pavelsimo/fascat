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


def test_stage_hard_edge_normals_follow_remapped_material_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["red_a", "red_b"])},
        materials={
            "red_a": Material(id="red_a", name="Red A", base_color=(1.0, 0.0, 0.0, 1.0)),
            "red_b": Material(id="red_b", name="Red B", base_color=(1.0, 0.0, 0.0, 1.0)),
        },
    )

    split = asset.stage(StageOptions(normal_mode="hard_edges", merge_equivalent_materials=False, uv0="none", uv1=None))
    merged = asset.stage(StageOptions(normal_mode="hard_edges", merge_equivalent_materials=True, uv0="none", uv1=None))
    split_mesh = split.parts["panel"].mesh
    merged_mesh = merged.parts["panel"].mesh

    assert split_mesh is not None
    assert merged_mesh is not None
    assert split_mesh.vertex_count == 6
    assert merged_mesh.vertex_count == 4
    assert merged_mesh.material_indices is not None
    assert merged_mesh.material_indices.tolist() == [0, 0]


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


def test_stage_can_copy_uv0_to_uv1() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="box", uv1="copy_uv0"))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert sorted(staged_mesh.uvs) == [0, 1]
    assert np.array_equal(staged_mesh.uvs[1], staged_mesh.uvs[0])
    assert staged_mesh.metadata["uv1_mode"] == "copy_uv0"
    assert staged_mesh.metadata["uv1_source_channel"] == "0"
    assert staged_mesh.metadata["uv1_copy_status"] == "copied"
    assert staged.report.steps[-1].warnings == []


def test_stage_warns_when_uv1_copy_source_is_missing() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="none", uv1="copy_uv0"))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.uvs == {}
    assert staged_mesh.metadata["uv1_copy_status"] == "missing_source"
    assert len(warnings) == 1
    assert "requested UV1 copy from UV0, but UV0 is missing" in warnings[0]


def test_stage_normalizes_requested_uv_channels() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={1: np.array([[2, -1], [4, -1], [2, 3]], dtype=float)},
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="none", uv1=None, normalize_uvs=(1,)))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert np.allclose(staged_mesh.uvs[1], np.array([[0, 0], [1, 0], [0, 1]], dtype=float))
    assert staged_mesh.metadata["uv1_normalize_status"] == "normalized"
    assert staged_mesh.metadata["uv1_normalize_bounds_before"] == "2,-1,4,3"
    assert staged_mesh.metadata["uv1_domain"] == "bake"
    assert staged_mesh.metadata["uv1_bounds"] == "0,0,1,1"
    assert staged_mesh.metadata["uv1_unit_domain_status"] == "ok"
    assert staged_mesh.metadata["uv1_validation_status"] == "ok"
    assert staged_mesh.metadata["uv1_out_of_unit_vertices"] == "0"
    assert staged.report.steps[-1].warnings == []


def test_stage_warns_when_normalize_uv_channel_is_missing() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="none", uv1=None, normalize_uvs=(1,)))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv1_normalize_status"] == "missing_channel"
    assert len(warnings) == 1
    assert "requested UV1 normalization, but UV1 is missing" in warnings[0]


def test_stage_records_uv_layout_quality_and_warns_for_uv1_overlap() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0.2],
                [1, 0, 0.2],
                [0, 1, 0.2],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="box", uv1="box"))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0_domain"] == "tileable"
    assert staged_mesh.metadata["uv0_bounds"] == "0,0,1,1"
    assert staged_mesh.metadata["uv0_unit_domain_status"] == "ok"
    assert staged_mesh.metadata["uv0_validation_status"] == "ok"
    assert staged_mesh.metadata["uv1_domain"] == "bake"
    assert staged_mesh.metadata["uv1_bounds"] == "0,0,1,1"
    assert staged_mesh.metadata["uv1_unit_domain_status"] == "ok"
    assert staged_mesh.metadata["uv1_validation_status"] == "overlap_pairs"
    assert staged_mesh.metadata["uv0_overlap_pairs"] == "1"
    assert staged_mesh.metadata["uv1_overlap_pairs"] == "1"
    assert staged_mesh.metadata["uv0_island_count"] == "2"
    assert staged_mesh.metadata["uv1_island_count"] == "2"
    assert staged_mesh.metadata["uv0_pack_efficiency"] == "1"
    assert staged_mesh.metadata["uv1_normalized_pack_efficiency"] == "1"
    assert staged_mesh.metadata["uv0_max_angle_distortion_degrees"] == "0"
    assert staged_mesh.metadata["uv1_max_edge_length_distortion"] == "0"
    assert len(warnings) == 1
    assert "part part uv1 violates lightmap/baking constraints" in warnings[0]
    assert "uv0" not in warnings[0]


def test_stage_classifies_existing_uv_channels_by_runtime_domain() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={
            0: np.array([[-0.25, 0], [1.5, 0], [0, 1]], dtype=float),
            1: np.array([[-0.25, 0], [1.5, 0], [0, 1]], dtype=float),
        },
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0_domain"] == "tileable"
    assert staged_mesh.metadata["uv0_bounds"] == "-0.25,0,1.5,1"
    assert staged_mesh.metadata["uv0_unit_domain_status"] == "outside_0_1"
    assert staged_mesh.metadata["uv0_validation_status"] == "ok"
    assert staged_mesh.metadata["uv1_domain"] == "bake"
    assert staged_mesh.metadata["uv1_bounds"] == "-0.25,0,1.5,1"
    assert staged_mesh.metadata["uv1_unit_domain_status"] == "outside_0_1"
    assert staged_mesh.metadata["uv1_validation_status"] == "outside_0_1"
    assert len(warnings) == 1
    assert "part part uv1 violates lightmap/baking constraints" in warnings[0]
    assert "2 UV vertices outside 0..1" in warnings[0]


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


def test_stage_can_preserve_existing_normals_when_generation_is_enabled() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        normals=np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=float),
    )
    original_normals = mesh.normals.copy()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(override_normals=False, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.normals is not None
    assert np.array_equal(staged_mesh.normals, original_normals)
    assert staged_mesh.metadata["normal_generation_status"] == "preserved"
    assert staged_mesh.metadata["normal_source"] == "existing"
    assert staged_mesh.metadata["normal_override"] == "false"
    assert staged.metadata["stage_normals_preserved_parts"] == "1"
    assert staged.report.steps[-1].after["stage_normals_preserved_parts"] == 1


def test_stage_supports_area_weighted_smooth_normals() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [4, 0, 0],
                [0, 1, 0],
                [0, 0, 2],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    angle = asset.stage(StageOptions(normal_weighting="angle", uv0="none", uv1=None))
    area = asset.stage(StageOptions(normal_weighting="area", uv0="none", uv1=None))
    angle_mesh = angle.parts["part"].mesh
    area_mesh = area.parts["part"].mesh

    assert angle_mesh is not None
    assert area_mesh is not None
    assert angle_mesh.normals is not None
    assert area_mesh.normals is not None
    assert not np.allclose(angle_mesh.normals[0], area_mesh.normals[0])
    assert area_mesh.metadata["normal_generation_status"] == "generated"
    assert area_mesh.metadata["normal_mode"] == "smooth"
    assert area_mesh.metadata["normal_weighting"] == "area"
    assert area.metadata["stage_normals_generated_parts"] == "1"
    assert area.report.steps[-1].after["stage_normals_generated_parts"] == 1


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
    assert staged_mesh.metadata["tangents_status"] == "generated"
    assert staged_mesh.metadata["tangents_uv_channel"] == "0"
    assert staged.metadata["stage_tangents_generated_parts"] == "1"


def test_stage_warns_when_tangents_are_requested_without_uv0() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(tangents=True, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.tangents is None
    assert staged_mesh.metadata["tangents_status"] == "missing_uv0"
    assert staged_mesh.metadata["tangents_uv_channel"] == "0"
    assert staged.metadata["stage_tangents_missing_uv0_parts"] == "1"
    assert len(warnings) == 1
    assert "requested tangents from UV0, but UV0 is missing" in warnings[0]


def test_stage_generates_tangents_from_requested_uv_channel() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={1: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(tangents=True, tangent_uv_channel=1, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.tangents is not None
    assert staged_mesh.metadata["tangents_status"] == "generated"
    assert staged_mesh.metadata["tangents_uv_channel"] == "1"
    assert staged.metadata["stage_tangents_generated_parts"] == "1"
    assert staged.report.steps[-1].warnings == []


def test_stage_preserves_existing_tangents_by_default() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
    ).compute_normals()
    mesh = mesh.compute_tangents()
    assert mesh.tangents is not None
    original_tangents = mesh.tangents.copy()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(tangents=True, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.tangents is not None
    assert np.array_equal(staged_mesh.tangents, original_tangents)
    assert staged_mesh.metadata["tangents_status"] == "preserved"
    assert staged_mesh.metadata["tangents_source"] == "existing"
    assert staged_mesh.metadata["tangents_requested_uv_channel"] == "0"
    assert staged.metadata["stage_tangents_preserved_parts"] == "1"
    assert "stage_tangents_generated_parts" not in staged.metadata
    assert staged.report.steps[-1].warnings == []


def test_stage_can_override_existing_tangents() -> None:
    custom_tangents = np.array([[0, 1, 0, 1], [0, 1, 0, 1], [0, 1, 0, 1]], dtype=float)
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
        tangents=custom_tangents,
    ).compute_normals()
    mesh.tangents = custom_tangents
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(tangents=True, override_tangents=True, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.tangents is not None
    assert not np.array_equal(staged_mesh.tangents, custom_tangents)
    assert staged_mesh.metadata["tangents_status"] == "regenerated"
    assert staged_mesh.metadata["tangents_override"] == "true"
    assert staged_mesh.metadata["tangents_uv_channel"] == "0"
    assert staged.metadata["stage_tangents_generated_parts"] == "1"
    assert staged.metadata["stage_tangents_regenerated_parts"] == "1"
    assert staged.report.steps[-1].warnings == []


def test_stage_warns_when_requested_tangent_uv_channel_is_missing() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(tangents=True, tangent_uv_channel=1, uv0="none", uv1=None))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.tangents is None
    assert staged_mesh.metadata["tangents_status"] == "missing_uv_channel"
    assert staged_mesh.metadata["tangents_uv_channel"] == "1"
    assert staged.metadata["stage_tangents_missing_uv_channel_parts"] == "1"
    assert len(warnings) == 1
    assert "requested tangents from UV1, but UV1 is missing" in warnings[0]


def test_stage_reports_tangent_invalidation_after_uv_edits() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
    ).compute_normals()
    mesh = mesh.compute_tangents()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    dropped = asset.stage(StageOptions(tangents=False, uv0="box", uv1=None))
    dropped_mesh = dropped.parts["part"].mesh
    dropped_warnings = dropped.report.steps[-1].warnings

    assert dropped_mesh is not None
    assert dropped_mesh.tangents is None
    assert dropped_mesh.metadata["tangents_status"] == "dropped"
    assert dropped_mesh.metadata["tangents_drop_reason"] == "uv_edit"
    assert dropped_mesh.metadata["tangents_invalidated_by_uv_edit"] == "0"
    assert dropped.metadata["stage_tangents_invalidated_parts"] == "1"
    assert dropped.metadata["stage_tangents_dropped_parts"] == "1"
    assert len(dropped_warnings) == 1
    assert "existing tangents were dropped because UVs were regenerated" in dropped_warnings[0]

    regenerated = asset.stage(StageOptions(tangents=True, uv0="box", uv1=None))
    regenerated_mesh = regenerated.parts["part"].mesh

    assert regenerated_mesh is not None
    assert regenerated_mesh.tangents is not None
    assert regenerated_mesh.metadata["tangents_status"] == "regenerated"
    assert regenerated_mesh.metadata["tangents_invalidated_by_uv_edit"] == "0"
    assert regenerated.metadata["stage_tangents_generated_parts"] == "1"
    assert regenerated.metadata["stage_tangents_regenerated_parts"] == "1"
    assert regenerated.report.steps[-1].warnings == []


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

    staged = asset.stage(
        StageOptions(
            uv0="unwrap",
            uv1="unwrap",
            unwrap=UnwrapOptions(
                method="conformal",
                iterations=8,
                tolerance=0.01,
                sharp_to_seam=True,
                forbid_overlapping=True,
            ),
        )
    )
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0"] == "xatlas"
    assert staged_mesh.metadata["uv1"] == "xatlas"
    assert staged_mesh.metadata["uv0_unwrap_backend"] == "xatlas"
    assert staged_mesh.metadata["uv0_unwrap_method"] == "conformal"
    assert staged_mesh.metadata["uv0_unwrap_method_status"] == "intent"
    assert staged_mesh.metadata["uv0_unwrap_iterations"] == "8"
    assert staged_mesh.metadata["uv0_unwrap_iterations_status"] == "intent"
    assert staged_mesh.metadata["uv0_unwrap_tolerance"] == "0.01"
    assert staged_mesh.metadata["uv0_unwrap_tolerance_status"] == "intent"
    assert staged_mesh.metadata["uv0_sharp_to_seam_requested"] == "true"
    assert staged_mesh.metadata["uv0_sharp_to_seam_enforced"] == "false"
    assert staged_mesh.metadata["uv0_sharp_to_seam_status"] == "intent"
    assert staged_mesh.metadata["uv0_forbid_overlapping_requested"] == "true"
    assert staged_mesh.metadata["uv0_forbid_overlapping_effective"] == "true"
    assert staged_mesh.metadata["uv0_forbid_overlapping_enforced"] == "false"
    assert staged_mesh.metadata["uv0_forbid_overlapping_status"] == "validated_not_enforced"
    assert staged.metadata["stage_uv_policy_intent_channels"] == "4"
    assert staged.report.steps[-1].after["stage_uv_policy_intent_channels"] == 4
    assert sorted(staged_mesh.uvs) == [0, 1]
    assert any(
        "records method, iteration, and tolerance controls as intent" in warning for warning in staged.report.warnings
    )
    assert any(
        "records sharp-to-seam, forbid-overlapping UV policy controls as intent" in warning
        for warning in staged.report.warnings
    )


def test_stage_records_forbid_overlapping_policy_for_tileable_uv0() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [1, 3, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [0.8, 0.1]], dtype=float)},
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="none", uv1=None, unwrap=UnwrapOptions(forbid_overlapping=True)))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0_domain"] == "tileable"
    assert staged_mesh.metadata["uv0_forbid_overlapping_requested"] == "true"
    assert staged_mesh.metadata["uv0_forbid_overlapping_effective"] == "true"
    assert staged_mesh.metadata["uv0_forbid_overlapping_status"] == "violation"
    assert staged_mesh.metadata["uv0_validation_status"] == "overlap_pairs"
    assert staged.metadata["stage_uv_forbid_overlapping_violations"] == "1"
    assert staged.report.steps[-1].after["stage_uv_forbid_overlapping_violations"] == 1
    assert any("violates requested forbid-overlapping UV policy" in warning for warning in staged.report.warnings)


@pytest.mark.requires_xatlas
def test_stage_warns_when_bake_uv_is_unwrapped_without_repack() -> None:
    pytest.importorskip("xatlas")
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="none", uv1="unwrap", unwrap=UnwrapOptions(padding=6)))
    staged_mesh = staged.parts["part"].mesh
    warnings = staged.report.steps[-1].warnings

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv1_domain"] == "bake"
    assert staged_mesh.metadata["uv1_workflow_steps"] == "unwrap,validate"
    assert staged_mesh.metadata["uv1_pack_status"] == "missing_repack"
    assert staged_mesh.metadata["uv1_padding_status"] == "metadata_only"
    assert staged.metadata["stage_bake_uv_channels_missing_repack"] == "1"
    assert staged.report.steps[-1].after["stage_bake_uv_channels_missing_repack"] == 1
    assert any("no UV repack/padding backend ran" in warning for warning in warnings)
