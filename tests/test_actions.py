from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import (
    BakeMaterialOptions,
    DecimateOptions,
    LODGeneratorOptions,
    LODLevel,
    RemoveHolesOptions,
    RemoveOccludedOptions,
)

runner = CliRunner()


def _triangle_strip(count: int) -> Mesh:
    points = []
    faces = []
    for index in range(count):
        offset = len(points)
        base = float(index * 2)
        points.extend([[base, 0, 0], [base + 1, 0, 0], [base, 1, 0]])
        faces.append([offset, offset + 1, offset + 2])
    return Mesh(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=int))


def _cube_mesh(scale: float = 1.0) -> Mesh:
    points = np.asarray(
        [
            [-scale, -scale, -scale],
            [scale, -scale, -scale],
            [scale, scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
            [scale, -scale, scale],
            [scale, scale, scale],
            [-scale, scale, scale],
        ],
        dtype=float,
    )
    faces = np.asarray(
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
    return Mesh(points=points, faces=faces)


def _translated_mesh(mesh: Mesh, x: float, y: float = 0.0, z: float = 0.0) -> Mesh:
    translated = mesh.copy()
    translated.points = translated.points + np.asarray([x, y, z], dtype=float)
    return translated


def _merge_meshes(meshes: list[tuple[Mesh, int]]) -> Mesh:
    points: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    material_indices: list[int] = []
    offset = 0
    for mesh, material_index in meshes:
        points.append(mesh.points)
        faces.append(mesh.faces + offset)
        material_indices.extend([material_index] * mesh.triangle_count)
        offset += mesh.vertex_count
    return Mesh(
        points=np.vstack(points),
        faces=np.vstack(faces),
        material_indices=np.asarray(material_indices, dtype=int),
    )


def _open_box_mesh() -> Mesh:
    points = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
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
    return Mesh(points=points, faces=faces)


def _square_tube_mesh() -> Mesh:
    mesh = _open_box_mesh()
    mesh.faces = mesh.faces[2:].copy()
    return mesh


def test_bake_materials_merges_selected_material_slots() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.asarray([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["red", "blue"])},
        materials={
            "red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0)),
            "blue": Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )

    baked = asset.bake_materials(BakeMaterialOptions(force_uv_generation=True, bake=("base_color", "opacity")))
    part = baked.parts["panel"]

    assert baked.material_count == 1
    assert part.material_ids == ["baked_material"]
    assert part.mesh is not None
    assert part.mesh.material_indices is None
    assert 0 in part.mesh.uvs
    assert (
        baked.materials["baked_material"].metadata["baked_texture_base_color_uri"].startswith("data:image/png;base64,")
    )
    assert baked.materials["baked_material"].metadata["baked_texture_resolution"] == "2048"
    assert baked.report.steps[-1].before["draw_calls"] == 2
    assert baked.report.steps[-1].after["draw_calls"] == 1
    assert "constant embedded texture maps" in baked.report.steps[-1].warnings[0]


def test_decimate_uses_selection_budget() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(6))},
    )

    decimated = asset.decimate(DecimateOptions(target_triangles=3, target_ratio=None))

    assert decimated.triangle_count <= 3
    assert decimated.report.steps[-1].name == "decimate"
    assert decimated.report.steps[-1].options["target_triangles"] == 3
    assert decimated.metadata["decimate_source_triangles"] == "6"
    assert decimated.metadata["decimate_output_triangles"] == "3"
    assert decimated.parts["body"].metadata["decimate_error_metric"] == "symmetric_vertex_nearest_distance"


def test_quality_decimate_records_measured_error_metrics() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(8))},
    )

    decimated = asset.decimate(
        DecimateOptions(
            criterion="quality",
            target_ratio=None,
            surface_tolerance=0.25,
            line_tolerance=0.1,
            uv_tolerance=0.05,
            budget_scope="part",
        )
    )

    part = decimated.parts["body"]
    assert part.metadata["decimate_criterion"] == "quality"
    assert part.metadata["decimate_source_triangles"] == "8"
    assert int(part.metadata["decimate_output_triangles"]) < 8
    assert float(part.metadata["decimate_triangle_reduction"]) > 0.0
    assert "measured vertex error" in decimated.report.steps[-1].warnings[0]


def test_remove_holes_fills_small_boundary_loop() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.asarray([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="shell", name="Shell", part_id="shell")]),
        parts={"shell": Part(id="shell", name="Shell", mesh=mesh)},
    )

    filled = asset.remove_holes(RemoveHolesOptions(max_diameter=2.0))

    assert filled.parts["shell"].mesh is not None
    assert filled.parts["shell"].mesh.triangle_count == 4
    assert filled.parts["shell"].metadata["removed_holes"] == "1"
    assert filled.metadata["removed_holes"] == "1"
    assert "BREP feature-level hole removal is not implemented" in filled.report.steps[-1].warnings[0]


def test_remove_holes_respects_surface_hole_filter() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="shell", name="Shell", part_id="shell")]),
        parts={
            "shell": Part(
                id="shell",
                name="Shell",
                mesh=Mesh(
                    points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                    faces=np.asarray([[0, 1, 2]], dtype=int),
                ),
            )
        },
    )

    skipped = asset.remove_holes(RemoveHolesOptions(through=True, blind=False, surface=False, prefer_brep=False))
    filled = asset.remove_holes(RemoveHolesOptions(through=False, blind=False, surface=True, prefer_brep=False))

    assert skipped.parts["shell"].mesh is not None
    assert skipped.parts["shell"].mesh.triangle_count == 1
    assert skipped.metadata["removed_holes"] == "0"
    assert filled.parts["shell"].mesh is not None
    assert filled.parts["shell"].mesh.triangle_count == 2
    assert filled.metadata["removed_surface_holes"] == "1"


def test_remove_holes_respects_blind_hole_filter_and_planar_span_diameter() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="box", name="Box", part_id="box")]),
        parts={"box": Part(id="box", name="Box", mesh=_open_box_mesh())},
    )

    skipped = asset.remove_holes(
        RemoveHolesOptions(through=False, blind=False, surface=True, max_diameter=1.1, prefer_brep=False)
    )
    filled = asset.remove_holes(
        RemoveHolesOptions(through=False, blind=True, surface=False, max_diameter=1.1, prefer_brep=False)
    )

    assert skipped.parts["box"].mesh is not None
    assert skipped.parts["box"].mesh.triangle_count == 10
    assert filled.parts["box"].mesh is not None
    assert filled.parts["box"].mesh.triangle_count == 12
    assert filled.parts["box"].metadata["removed_hole_types"] == "blind"
    assert filled.metadata["removed_blind_holes"] == "1"
    assert filled.metadata["removed_hole_max_diameter"] == "1"
    assert filled.metadata["removed_hole_diameter_method"] == "planar_span"


def test_remove_holes_respects_through_hole_filter() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="tube", name="Tube", part_id="tube")]),
        parts={"tube": Part(id="tube", name="Tube", mesh=_square_tube_mesh())},
    )

    skipped = asset.remove_holes(RemoveHolesOptions(through=False, blind=True, surface=True, prefer_brep=False))
    filled = asset.remove_holes(RemoveHolesOptions(through=True, blind=False, surface=False, prefer_brep=False))

    assert skipped.parts["tube"].mesh is not None
    assert skipped.parts["tube"].mesh.triangle_count == 8
    assert filled.parts["tube"].mesh is not None
    assert filled.parts["tube"].mesh.triangle_count == 12
    assert filled.parts["tube"].metadata["removed_hole_types"] == "through"
    assert filled.metadata["removed_through_holes"] == "2"


def test_remove_occluded_removes_contained_part_nodes() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
        },
    )

    visible = asset.remove_occluded(RemoveOccludedOptions(level="parts", preserve_cavities=False))

    assert visible.occurrence_count == 1
    assert "inner" not in visible.parts
    assert visible.metadata["removed_occluded_nodes"] == "1"
    assert visible.metadata["removed_occluded_triangles"] == "12"
    assert any("sampled visibility" in item for item in visible.report.steps[-1].warnings)


def test_remove_occluded_records_visibility_sampling_metadata() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="part", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=_cube_mesh(1.0))},
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="triangles", hemi_evaluation=True))

    warnings = result.report.steps[-1].warnings
    assert any("sampled visibility" in item for item in warnings)
    assert result.metadata["occlusion_level"] == "triangles"
    assert result.metadata["occlusion_hemi_evaluation"] == "true"
    assert int(result.metadata["occlusion_direction_count"]) < 26
    assert result.metadata["occlusion_candidate_count"] == "1"
    assert result.metadata["occlusion_face_count"] == "12"
    assert result.metadata["occlusion_sample_count"] == "12"
    assert result.metadata["occlusion_visible_sample_count"] == "12"
    assert result.metadata["occlusion_hidden_sample_count"] == "0"
    assert result.metadata["occlusion_sample_coverage"] == "1"
    assert result.metadata["occlusion_direction_coverage"] == "1"
    assert result.metadata["occlusion_confidence"] == "1"


def test_remove_occluded_records_lower_confidence_for_sparse_sampling() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="strip", name="Strip", part_id="strip")]),
        parts={"strip": Part(id="strip", name="Strip", mesh=_triangle_strip(100))},
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="parts", precision=1, strategy="conservative"))

    assert result.metadata["occlusion_face_count"] == "100"
    assert result.metadata["occlusion_sample_count"] == "15"
    assert result.metadata["occlusion_sample_coverage"] == "0.15"
    assert result.metadata["occlusion_direction_coverage"] == "0.230769"
    assert result.metadata["occlusion_confidence"] == "0.15"


def test_remove_occluded_respects_transparent_occluders() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0), material_ids=["glass"]),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
        },
        materials={"glass": Material(id="glass", name="Glass", base_color=(0.8, 0.9, 1.0, 0.35), opacity=0.35)},
    )

    visible = asset.remove_occluded(
        RemoveOccludedOptions(level="parts", preserve_cavities=False, consider_transparency_opaque=False)
    )
    hidden = asset.remove_occluded(
        RemoveOccludedOptions(level="parts", preserve_cavities=False, consider_transparency_opaque=True)
    )

    assert visible.occurrence_count == 2
    assert hidden.occurrence_count == 1
    assert "inner" not in hidden.parts


def test_remove_occluded_keeps_side_by_side_parts() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="left", name="Left", part_id="left"),
                Node(id="right", name="Right", part_id="right"),
            ],
        ),
        parts={
            "left": Part(id="left", name="Left", mesh=_translated_mesh(_cube_mesh(0.5), -1.0)),
            "right": Part(id="right", name="Right", mesh=_translated_mesh(_cube_mesh(0.5), 1.0)),
        },
    )

    visible = asset.remove_occluded(RemoveOccludedOptions(level="parts", preserve_cavities=False))

    assert visible.occurrence_count == 2
    assert visible.metadata["removed_occluded_nodes"] == "0"


def test_remove_occluded_triangle_level_removes_hidden_occurrence() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
        },
    )

    visible = asset.remove_occluded(
        RemoveOccludedOptions(level="triangles", preserve_cavities=False, neighbors_preservation=0)
    )

    assert visible.occurrence_count == 1
    assert "inner" not in visible.parts
    assert visible.metadata["removed_occluded_triangles"] == "12"


def test_remove_occluded_submesh_level_removes_hidden_material_group() -> None:
    candidate_mesh = _merge_meshes(
        [
            (_cube_mesh(0.5), 0),
            (_translated_mesh(_cube_mesh(0.5), 4.0), 1),
        ]
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="candidate", name="Candidate", part_id="candidate"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "candidate": Part(
                id="candidate",
                name="Candidate",
                mesh=candidate_mesh,
                material_ids=["hidden", "visible"],
            ),
        },
        materials={
            "hidden": Material(id="hidden", name="Hidden", base_color=(1.0, 0.0, 0.0, 1.0)),
            "visible": Material(id="visible", name="Visible", base_color=(0.0, 1.0, 0.0, 1.0)),
        },
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="submeshes", preserve_cavities=False))
    part = result.parts["candidate"]

    assert part.mesh is not None
    assert part.mesh.triangle_count == 12
    assert part.material_ids == ["visible"]
    assert result.metadata["removed_occluded_triangles"] == "12"


def test_run_lod_generators_records_screen_coverage_metadata() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="cube", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=_cube_mesh())},
    )

    with_lods = asset.run_lod_generators(
        LODGeneratorOptions(
            preset="vr",
            levels=(LODLevel(screen_coverage=0.5, target_ratio=0.5), LODLevel(0.2, 0.25)),
            validate=True,
        )
    )

    assert len(with_lods.parts["cube"].lod_meshes) == 2
    assert with_lods.parts["cube"].metadata["lod_screen_coverage"] == "0.5,0.2"
    assert with_lods.report.steps[-1].name == "run_lod_generators"


def test_cli_convert_accepts_optimization_action_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--bake-materials",
            "--maps-resolution",
            "1024",
            "--force-uv-generation",
            "--bake",
            "base-color,opacity",
            "--decimate",
            "--decimate-criterion",
            "target",
            "--target-triangles",
            "1000",
            "--surface-tolerance",
            "0.1",
            "--line-tolerance",
            "0.02",
            "--normal-tolerance",
            "15",
            "--uv-tolerance",
            "0.01",
            "--budget-scope",
            "selection",
            "--remove-holes",
            "--hole-types",
            "through,blind,surface",
            "--max-hole-diameter",
            "3.0",
            "--remove-occluded",
            "--occlusion-strategy",
            "advanced",
            "--occlusion-level",
            "triangles",
            "--occlusion-precision",
            "2048",
            "--neighbors-preservation",
            "1",
            "--run-lod-generators",
            "--lod-mode",
            "variants",
            "--lod-per-part-budget",
            "--lod-drop-tiny-parts",
            "--lod-tiny-part-screen-size",
            "2",
            "--lod-preset",
            "vr",
            "--lod-screen-coverage",
            "0.5,0.2,0.05",
            "--lods",
            "0.5,0.25,0.1",
            "--validate-lods",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["bake_materials"] is True
    assert payload["bake"] == ["base_color", "opacity"]
    assert payload["decimate"] is True
    assert payload["remove_holes"] is True
    assert payload["remove_occluded"] is True
    assert payload["run_lod_generators"] is True
    assert payload["lod_per_part_budget"] is True
    assert payload["lod_drop_tiny_parts"] is True
    diagnostics = {item["operation"]: item for item in payload["operation_diagnostics"]}
    assert diagnostics["bake_materials"]["level"] == "approximate"
    assert diagnostics["remove_holes"]["level"] == "approximate"
    assert diagnostics["remove_occluded"]["level"] == "approximate"
    assert diagnostics["decimate"]["level"] == "exact"
    assert diagnostics["run_lod_generators"]["level"] == "exact"
