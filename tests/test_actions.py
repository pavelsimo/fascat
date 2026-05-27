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
    assert baked.report.steps[-1].before["draw_calls"] == 2
    assert baked.report.steps[-1].after["draw_calls"] == 1
    assert "texture image baking is not implemented" in baked.report.steps[-1].warnings[0]


def test_decimate_uses_selection_budget() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(6))},
    )

    decimated = asset.decimate(DecimateOptions(target_triangles=3, target_ratio=None))

    assert decimated.triangle_count <= 3
    assert decimated.report.steps[-1].name == "decimate"
    assert decimated.report.steps[-1].options["target_triangles"] == 3


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
    assert "BREP hole removal is not implemented" in filled.report.steps[-1].warnings[0]


def test_remove_holes_warns_when_hole_type_filtering_is_metadata_only() -> None:
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

    result = asset.remove_holes(RemoveHolesOptions(through=True, blind=False, surface=False, prefer_brep=False))

    assert any("mesh boundary-fill fallback" in item for item in result.report.steps[-1].warnings)
    assert any("cannot classify through, blind, or surface holes" in item for item in result.report.steps[-1].warnings)


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
    assert any("AABB containment fallback" in item for item in visible.report.steps[-1].warnings)


def test_remove_occluded_warns_when_using_part_level_aabb_fallback() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="part", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=_cube_mesh(1.0))},
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="triangles", hemi_evaluation=True))

    warnings = result.report.steps[-1].warnings
    assert any("part-level AABB containment fallback" in item for item in warnings)
    assert any("hemi_evaluation" in item for item in warnings)


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
    assert diagnostics["bake_materials"]["level"] == "metadata_only"
    assert diagnostics["remove_holes"]["level"] == "approximate"
    assert diagnostics["remove_occluded"]["level"] == "approximate"
    assert diagnostics["decimate"]["level"] == "exact"
    assert diagnostics["run_lod_generators"]["level"] == "exact"
