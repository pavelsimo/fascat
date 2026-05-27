from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import (
    BakeMaterialOptions,
    ConversionProfile,
    DecimateOptions,
    ExplodeOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    LODLevel,
    LODOptions,
    MergeVerticesOptions,
    OptimizeOptions,
    PlatformBudget,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    RepairOptions,
    ReplaceOptions,
    StageOptions,
    StepReadOptions,
    Tessellation,
)
from fascat.pipeline import convert
from fascat.pipeline_file import PipelineSpec
from fascat.report import Report


def _triangle_asset() -> Asset:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    report = Report(source_path="input.step")
    report.input_stats = {"parts": 1, "occurrences": 1, "materials": 0, "vertices": 3, "triangles": 1}
    report.add_step("import", options={"format": "STEP"}, before={}, after=report.input_stats)
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
        report=report,
    )


def _mobius_strip_mesh(segments: int = 6) -> Mesh:
    points: list[np.ndarray] = []
    for index in range(segments):
        theta = 2.0 * np.pi * index / segments
        radial = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
        twist = (np.cos(theta * 0.5) * radial) + (np.sin(theta * 0.5) * np.array([0.0, 0.0, 1.0], dtype=float))
        center = 2.0 * radial
        points.append(center - (0.25 * twist))
        points.append(center + (0.25 * twist))

    faces: list[list[int]] = []
    for index in range(segments):
        left = index * 2
        right = left + 1
        next_left = 1 if index == segments - 1 else (index + 1) * 2
        next_right = 0 if index == segments - 1 else next_left + 1
        faces.append([left, next_left, right])
        faces.append([right, next_left, next_right])
    return Mesh(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=int))


def _t_junction_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, -1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 4]], dtype=int),
    )


def _boundary_gap_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1.005, 0, 0],
                [2, 0, 0],
                [1.005, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )


def _flipped_tetrahedron_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2],
            ],
            dtype=int,
        ),
    )


def _test_profile() -> ConversionProfile:
    return ConversionProfile(
        name="test",
        tessellation=None,
        repair=RepairOptions(),
        stage=StageOptions(uv0="none", uv1=None),
        optimize=None,
        lods=None,
    )


def test_asset_operations_return_new_assets_without_mutating_originals() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="box", uv1=None))
    optimized = asset.optimize(OptimizeOptions(target_triangles=1, optimize_buffers=False))
    with_lods = asset.lods(LODOptions((0.5,)))
    tessellated = asset.tessellate()

    original_mesh = asset.parts["part"].mesh
    staged_mesh = staged.parts["part"].mesh
    optimized_mesh = optimized.parts["part"].mesh

    assert staged is not asset
    assert optimized is not asset
    assert with_lods is not asset
    assert tessellated is not asset
    assert asset.report.steps == []
    assert asset.report.warnings == []
    assert original_mesh is not None
    assert original_mesh.normals is None
    assert original_mesh.uvs == {}
    assert original_mesh.triangle_count == 2
    assert asset.parts["part"].lod_meshes == []
    assert staged_mesh is not None
    assert staged_mesh.normals is not None
    assert 0 in staged_mesh.uvs
    assert optimized_mesh is not None
    assert optimized_mesh.triangle_count <= original_mesh.triangle_count
    assert len(with_lods.parts["part"].lod_meshes) == 1
    assert tessellated.report.warnings == []

    dirty = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={
            "part": Part(
                id="part",
                name="Part",
                mesh=Mesh(
                    points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], dtype=float),
                    faces=np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]], dtype=int),
                ),
            )
        },
    )

    repaired = dirty.repair(RepairOptions())

    assert repaired is not dirty
    assert dirty.report.steps == []
    assert dirty.parts["part"].mesh is not None
    assert dirty.parts["part"].mesh.triangle_count == 3
    assert repaired.parts["part"].mesh is not None
    assert repaired.parts["part"].mesh.triangle_count == 1


def test_repair_report_includes_unit_aware_tolerance_policy() -> None:
    asset = _triangle_asset()
    asset.units = "metre"
    asset.meters_per_unit = 1.0
    asset.metadata = {"source_units": "millimetre", "source_meters_per_unit": 0.001}

    repaired = asset.repair(RepairOptions(tolerance=2.0, area_epsilon=3.0))

    step = repaired.report.steps[-1]
    policy = step.options["tolerance_policy"]
    mesh = repaired.parts["part"].mesh
    assert isinstance(policy, dict)
    assert mesh is not None
    assert policy["coordinate_space"] == "source_local"
    assert policy["effective_units"] == "millimetre"
    assert policy["target_units"] == "metre"
    assert policy["vertex_merge_tolerance_meters"] == pytest.approx(0.002)
    assert policy["degenerate_area_epsilon_square_meters"] == pytest.approx(0.000003)
    assert policy["operations"] == {
        "vertex_merge": "enabled",
        "degenerate_polygon_cleanup": "enabled",
        "t_junction_sewing": "not_implemented",
        "boundary_gap_stitching": "not_implemented",
        "non_manifold_edge_cracking": "not_implemented",
    }
    assert mesh.metadata["repair_effective_units"] == "millimetre"
    assert mesh.metadata["repair_target_units"] == "metre"
    assert mesh.metadata["repair_vertex_merge_tolerance_meters"] == "0.002"
    assert mesh.metadata["repair_degenerate_area_epsilon_square_meters"] == "3e-06"
    assert mesh.metadata["repair_t_junction_sewing"] == "not_implemented"


def test_repair_reports_non_orientable_topology_before_winding_fix() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="strip")]),
        parts={"strip": Part(id="strip", name="Strip", mesh=_mobius_strip_mesh())},
    )

    repaired = asset.repair(RepairOptions())

    mesh = repaired.parts["strip"].mesh
    step = repaired.report.steps[-1]
    assert mesh is not None
    assert mesh.metadata["repair_non_orientable_edges_before_orientation"] == "1"
    assert step.after["repair_non_orientable_edges_before_orientation"] == 1
    assert any("non-orientable shared edge" in warning for warning in step.warnings)


def test_repair_report_includes_t_junction_metrics() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=_t_junction_mesh())},
    )

    repaired = asset.repair(RepairOptions())

    mesh = repaired.parts["panel"].mesh
    step = repaired.report.steps[-1]
    assert mesh is not None
    assert mesh.metadata["repair_t_junctions_before"] == "1"
    assert mesh.metadata["repair_t_junctions_after"] == "1"
    assert step.after["repair_t_junctions_before"] == 1
    assert step.after["repair_t_junctions_after"] == 1
    assert any("T-junction sewing is not implemented" in warning for warning in step.warnings)


def test_repair_report_includes_boundary_gap_metrics() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=_boundary_gap_mesh())},
    )

    repaired = asset.repair(RepairOptions(tolerance=0.01, merge_vertices=False))

    mesh = repaired.parts["panel"].mesh
    step = repaired.report.steps[-1]
    assert mesh is not None
    assert mesh.metadata["repair_boundary_gaps_before"] == "1"
    assert mesh.metadata["repair_boundary_gaps_after"] == "1"
    assert mesh.metadata["repair_boundary_gap_stitching"] == "not_implemented"
    assert step.after["repair_boundary_gaps_before"] == 1
    assert step.after["repair_boundary_gaps_after"] == 1
    assert any("boundary gap stitching is not implemented" in warning for warning in step.warnings)


def test_repair_report_includes_flipped_component_metrics() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="solid")]),
        parts={"solid": Part(id="solid", name="Solid", mesh=_flipped_tetrahedron_mesh())},
    )

    repaired = asset.repair(RepairOptions(fix_winding=False))

    mesh = repaired.parts["solid"].mesh
    step = repaired.report.steps[-1]
    assert mesh is not None
    assert mesh.metadata["repair_flipped_components_before_orientation"] == "1"
    assert mesh.metadata["repair_flipped_components_after_orientation"] == "1"
    assert step.after["repair_flipped_components_before_orientation"] == 1
    assert step.after["repair_flipped_components_after_orientation"] == 1
    assert any("flipped closed orientation component" in warning for warning in step.warnings)


def test_merge_vertices_asset_operation_reports_counts() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    merged = asset.merge_vertices(MergeVerticesOptions())
    part = merged.parts["part"]
    step = merged.report.steps[-1]

    assert part.mesh is not None
    assert part.mesh.vertex_count == 3
    assert part.mesh.triangle_count == 1
    assert part.mesh.metadata["merge_vertices_removed"] == "1"
    assert step.name == "merge_vertices"
    assert step.options["tolerance"] == 0.0
    assert step.options["preserve_uvs"] is True
    assert step.after["vertices"] == 3
    assert step.after["triangles"] == 1
    assert step.after["merge_vertices_removed"] == 1
    assert step.after["merge_vertices_degenerate_triangles_removed"] == 1


def test_asset_operation_reports_include_options_and_before_after_counts() -> None:
    required_counts = {"nodes", "parts", "occurrences", "materials", "vertices", "triangles"}
    required_options = {
        "tessellate": {
            "sag",
            "sag_ratio",
            "angle",
            "relative",
            "min_edge_length",
            "max_edge_length",
            "max_polygon_length",
            "preserve_boundaries",
            "curvature_adaptive",
            "avoid_skinny_triangles",
            "quality_report",
            "free_edge_report",
            "create_normals",
            "keep_brep",
            "reuse_existing_meshes",
            "part_settings",
        },
        "repair": {
            "tolerance",
            "merge_vertices",
            "delete_degenerate",
            "fix_winding",
            "fill_small_holes",
            "area_epsilon",
        },
        "merge_vertices": {
            "tolerance",
            "preserve_normals",
            "preserve_tangents",
            "preserve_uvs",
            "preserve_material_boundaries",
            "delete_degenerate",
            "area_epsilon",
        },
        "stage": {
            "materials",
            "material_mode",
            "merge_equivalent_materials",
            "normals",
            "normal_mode",
            "hard_edge_angle",
            "preserve_face_boundaries",
            "tangents",
            "tangent_uv_channel",
            "override_tangents",
            "validate_normals",
            "unwrap",
            "atlas",
            "uv0",
            "uv1",
            "normalize_uvs",
        },
        "optimize": {
            "target_triangles",
            "ratio",
            "preserve_instances",
            "simplify",
            "optimize_buffers",
            "preserve_hard_edges",
            "hard_edge_angle",
            "preserve_holes",
            "preserve_material_boundaries",
            "preserve_uv_seams",
            "preserve_small_parts",
            "small_part_triangle_threshold",
            "preserve_silhouette",
        },
        "lods": {
            "ratios",
            "mode",
            "screen_coverage",
            "per_part_budget",
            "drop_tiny_parts",
            "tiny_part_screen_size",
            "validate",
        },
        "explode": {"mode", "metadata", "remove_empty_nodes"},
        "replace": {"mode", "preserve_transform", "metadata", "proxy_mesh", "external_path"},
        "bake_materials": {"maps_resolution", "force_uv_generation", "uv_channel", "padding", "bake", "merge_output"},
        "decimate": {
            "criterion",
            "target_triangles",
            "target_ratio",
            "surface_tolerance",
            "line_tolerance",
            "normal_tolerance",
            "uv_tolerance",
            "protect_topology",
            "preserve_painted_areas",
            "budget_scope",
            "uv_importance",
        },
        "remove_holes": {"through", "blind", "surface", "max_diameter", "prefer_brep"},
        "remove_occluded": {
            "strategy",
            "level",
            "precision",
            "hemi_evaluation",
            "neighbors_preservation",
            "consider_transparency_opaque",
            "preserve_cavities",
            "minimum_cavity_volume_m3",
        },
        "run_lod_generators": {"preset", "levels", "validate", "output", "allow_non_monotonic"},
    }
    operations = [
        ("tessellate", lambda asset: asset.tessellate()),
        ("repair", lambda asset: asset.repair(RepairOptions())),
        ("merge_vertices", lambda asset: asset.merge_vertices(MergeVerticesOptions())),
        ("stage", lambda asset: asset.stage(StageOptions(uv0="none", uv1=None))),
        ("optimize", lambda asset: asset.optimize()),
        ("lods", lambda asset: asset.lods(LODOptions((0.5,)))),
        ("explode", lambda asset: asset.explode(ExplodeOptions())),
        ("replace", lambda asset: asset.replace(ReplaceOptions())),
        ("bake_materials", lambda asset: asset.bake_materials(BakeMaterialOptions(force_uv_generation=True))),
        ("decimate", lambda asset: asset.decimate(DecimateOptions(target_ratio=0.5))),
        ("remove_holes", lambda asset: asset.remove_holes(RemoveHolesOptions())),
        ("remove_occluded", lambda asset: asset.remove_occluded(RemoveOccludedOptions(level="parts"))),
        (
            "run_lod_generators",
            lambda asset: asset.run_lod_generators(
                LODGeneratorOptions(levels=(LODLevel(screen_coverage=0.5, target_ratio=0.5),))
            ),
        ),
    ]

    for name, operation in operations:
        result = operation(_triangle_asset())
        step = result.report.steps[-1]

        assert step.name == name
        assert required_options[name] <= set(step.options)
        assert required_counts <= set(step.before)
        assert required_counts <= set(step.after)
        assert step.duration >= 0.0


def test_convert_report_includes_timed_write_and_validate_steps(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    asset = _triangle_asset()
    written: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "read_step", lambda _path: asset)

    def fake_write_usd(asset: Asset, path: str | Path, *, debug: bool = False, options: object = None) -> None:
        written["path"] = str(path)
        written["debug"] = debug
        written["triangles"] = asset.triangle_count

    monkeypatch.setattr(pipeline, "_write_usd", fake_write_usd)
    monkeypatch.setattr(pipeline, "validate_usd", lambda _path: {"meshes": 1, "points": 3, "triangles": 1})

    converted = convert(
        "input.step",
        tmp_path / "output.usdc",
        profile=_test_profile(),
    )
    steps = {step.name: step for step in converted.report.steps}

    assert written["path"] == str(tmp_path / "output.usdc")
    assert written["triangles"] == 1
    assert {"import", "repair", "stage", "write", "validate"} <= set(steps)
    assert steps["write"].before == converted.stats()
    assert steps["write"].after == converted.stats()
    assert steps["validate"].after == {
        **converted.stats(),
        "validated_meshes": 1,
        "validated_points": 3,
        "validated_triangles": 1,
    }
    assert steps["write"].duration >= 0.0
    assert steps["validate"].duration >= 0.0
    assert converted.report.finished_at is not None
    assert converted.report.output_stats == converted.stats()


def test_convert_can_run_toml_pipeline_steps(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    asset = _triangle_asset()
    pipeline_file = tmp_path / "pipeline.toml"
    pipeline_file.write_text(
        """
[[filters]]
name = "simple"
part = "part"
max_triangles = 12

[[steps]]
op = "repair"

[[steps]]
op = "replace"
where = "simple"
mode = "bounding_box"
""",
        encoding="utf-8",
    )
    captured: dict[str, Asset] = {}

    monkeypatch.setattr(pipeline, "read_step", lambda _path: asset)
    monkeypatch.setattr(
        pipeline,
        "_write_usd",
        lambda written_asset, _path, *, debug=False, options=None: captured.setdefault("asset", written_asset),
    )
    monkeypatch.setattr(pipeline, "validate_usd", lambda _path: {"meshes": 1, "points": 8, "triangles": 12})

    converted = convert("input.step", tmp_path / "output.usdc", pipeline=PipelineSpec.from_file(pipeline_file))

    written = captured["asset"]
    part = next(iter(written.parts.values()))
    assert part.mesh is not None
    assert part.mesh.triangle_count == 12
    steps = {step.name: step for step in converted.report.steps}
    assert [step.name for step in converted.report.steps[1:]] == [
        "preflight",
        "repair",
        "replace",
        "write",
        "validate",
        "conversion_manifest",
        "workflow_summary",
        "profile_budget",
    ]
    assert steps["replace"].options["where"]["criteria"] == {"part_id": ["part"], "max_triangles": 12}
    assert steps["replace"].options["matched"]["parts"] == 1


def test_pipeline_rejects_unknown_operation_during_parse() -> None:
    with pytest.raises(ValueError, match="unsupported pipeline step op: tesselate"):
        PipelineSpec.from_dict({"steps": [{"op": "tesselate"}]})


def test_pipeline_rejects_unknown_step_keys_with_line(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "bad-step-key.toml"
    pipeline_file.write_text('[[steps]]\nop = "repair"\ntolerence = 0.1\n', encoding="utf-8")

    with pytest.raises(ValueError, match="line 3: unsupported key for repair pipeline step: tolerence"):
        PipelineSpec.from_file(pipeline_file)


def test_pipeline_applies_merge_vertices_step() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {
                    "op": "merge_vertices",
                    "preserve_normals": False,
                    "preserve_tangents": False,
                    "preserve_uvs": False,
                }
            ]
        }
    )
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    merged = spec.apply(asset)

    assert merged.parts["part"].mesh is not None
    assert merged.parts["part"].mesh.vertex_count == 3
    assert merged.report.steps[-1].name == "merge_vertices"
    assert merged.report.steps[-1].options["preserve_uvs"] is False


def test_pipeline_validates_step_options_during_parse() -> None:
    with pytest.raises(
        ValueError,
        match=r"pipeline step 1 \(tessellate\): tessellation sag must be greater than 0",
    ):
        PipelineSpec.from_dict({"steps": [{"op": "tessellate", "sag": 0.0}]})
    with pytest.raises(
        ValueError,
        match=r"pipeline step 1 \(decimate\): uv_importance",
    ):
        PipelineSpec.from_dict({"steps": [{"op": "decimate", "uv_importance": "bad"}]})


def test_pipeline_advises_unity_style_ordering() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {"op": "decimate"},
                {"op": "stage", "tangents": True, "uv0": "none"},
                {"op": "bake_materials", "bake": ["ao"]},
                {"op": "run_lod_generators"},
            ],
        }
    )

    advisories = spec.advisories()

    assert [item["code"] for item in advisories] == [
        "decimate_before_repair",
        "tangents_without_uv0",
        "ao_bake_without_uv1",
        "lods_before_optimize",
    ]
    assert [item["step"] for item in advisories] == [1, 2, 3, 4]
    assert all(item["level"] == "warning" for item in advisories)


def test_pipeline_advises_missing_tangent_uv_channel() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {"op": "stage", "tangents": True, "tangent_uv_channel": 1, "uv0": "box", "uv1": "none"},
            ],
        }
    )

    advisories = spec.advisories()

    assert [item["code"] for item in advisories] == ["tangents_without_uv1"]
    assert advisories[0]["message"] == "tangents are requested before UV1 is available"


def test_pipeline_treats_uv1_copy_as_uv1_when_uv0_exists() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {"op": "stage", "uv0": "box", "uv1": "copy-uv0"},
                {"op": "bake_materials", "bake": ["ao"]},
            ],
        }
    )

    assert [item["code"] for item in spec.advisories()] == []


def test_pipeline_stage_unwrap_solver_controls_are_parsed() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {
                    "op": "stage",
                    "uv0": "box",
                    "unwrap_method": "isometric",
                    "unwrap_iterations": 16,
                    "unwrap_tolerance": 0.001,
                }
            ]
        }
    )

    staged = spec.apply(_triangle_asset())
    step = staged.report.steps[-1]

    assert step.name == "stage"
    assert step.options["unwrap"] == {
        "texel_density": None,
        "padding": 2,
        "max_stretch": None,
        "method": "isometric",
        "iterations": 16,
        "tolerance": 0.001,
    }


def test_pipeline_stage_tangent_override_is_parsed() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {
                    "op": "stage",
                    "tangents": True,
                    "tangent_uv_channel": 1,
                    "override_tangents": True,
                    "uv0": "none",
                    "uv1": "none",
                }
            ]
        }
    )

    staged = spec.apply(_triangle_asset())
    step = staged.report.steps[-1]

    assert step.name == "stage"
    assert step.options["tangents"] is True
    assert step.options["tangent_uv_channel"] == 1
    assert step.options["override_tangents"] is True


def test_pipeline_advisories_are_added_to_convert_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    captured: dict[str, Asset] = {}
    spec = PipelineSpec.from_dict({"steps": [{"op": "decimate", "target_ratio": 0.9}, {"op": "repair"}]})

    monkeypatch.setattr(pipeline, "read_step", lambda _path, *, options=None: _triangle_asset())
    monkeypatch.setattr(
        pipeline,
        "_write_usd",
        lambda written_asset, _path, *, debug=False, options=None: captured.setdefault("asset", written_asset),
    )
    monkeypatch.setattr(pipeline, "validate_usd", lambda _path: {"meshes": 1, "points": 3, "triangles": 1})

    converted = convert("input.step", tmp_path / "output.usdc", pipeline=spec)

    assert "decimation runs before mesh repair" in converted.report.warnings[0]
    assert captured["asset"].report.warnings == converted.report.warnings


def test_convert_report_includes_preflight_checklist(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(pipeline, "_write_gltf", lambda _asset, _path, *, options=None: None)

    converted = convert(
        "input.step",
        tmp_path / "output.glb",
        profile=_test_profile(),
        tessellation=Tessellation(keep_brep=True),
        stage=StageOptions(normal_mode="none", tangents=True, uv0="none", uv1=None),
        bake_materials=BakeMaterialOptions(bake=("ao",)),
        gltf_options=GltfExportOptions(),
        validate_output=False,
    )
    preflight = converted.report.steps[1]
    checks = {item["code"]: item for item in preflight.options["checks"]}  # type: ignore[index]

    assert preflight.name == "preflight"
    assert preflight.after["preflight_checks_warning"] == 5
    assert checks["brep_patches_retained"]["status"] == "warning"
    assert checks["orientation_not_planned"]["status"] == "warning"
    assert checks["tangents_without_uv0"]["status"] == "warning"
    assert checks["ao_bake_without_uv1"]["status"] == "warning"
    assert checks["texture_compression_backend_missing"]["status"] == "warning"
    assert preflight.warnings == converted.report.warnings[:5]


def test_convert_report_includes_workflow_summary(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(pipeline, "_write_usd", lambda _asset, _path, *, debug=False, options=None: None)

    converted = convert(
        "input.step",
        tmp_path / "output.usdc",
        profile=_test_profile(),
        stage=StageOptions(uv0="box", uv1=None),
        bake_materials=BakeMaterialOptions(force_uv_generation=True),
        lods=LODOptions((0.5,)),
        validate_output=False,
    )
    manifest_step = converted.report.steps[-2]
    summary_step = converted.report.steps[-1]
    manifest_steps = manifest_step.options["steps"]  # type: ignore[index]
    stages = {item["stage"]: item for item in summary_step.options["stages"]}  # type: ignore[index]

    assert manifest_step.name == "conversion_manifest"
    assert manifest_step.options["style"] == "resolved_conversion_manifest"
    assert manifest_step.options["mode"] == "direct"
    assert manifest_step.options["profile"]["name"] == "test"  # type: ignore[index]
    assert manifest_step.options["import"]["metadata"] is True  # type: ignore[index]
    assert manifest_steps["stage"]["uv0"] == "box"  # type: ignore[index]
    assert manifest_steps["bake_materials"]["force_uv_generation"] is True  # type: ignore[index]
    assert manifest_steps["lods"]["ratios"] == [0.5]  # type: ignore[index]
    assert manifest_step.options["export"]["output_format"] == "usd"  # type: ignore[index]
    assert manifest_step.options["export"]["options"]["format"] == "OpenUSD"  # type: ignore[index]
    assert manifest_step.after["conversion_manifest_direct_steps"] == 4
    assert summary_step.name == "workflow_summary"
    assert summary_step.options["style"] == "unity_asset_transformer"
    assert summary_step.after["workflow_stages_total"] == 10
    assert summary_step.after["workflow_stages_run"] == 7
    assert summary_step.after["workflow_stages_skipped"] == 3
    assert summary_step.after["workflow_stages_approximate"] == 1
    assert stages["uv_preparation"]["status"] == "run"
    assert stages["material_baking"]["level"] == "approximate"
    assert stages["lod_generation"]["status"] == "run"
    assert stages["export_compression"]["status"] == "skipped"


def test_pipeline_rejects_incompatible_step_options_with_line(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "bad-merge.toml"
    pipeline_file.write_text('[[steps]]\nop = "merge"\nmode = "regions"\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"line 3: pipeline step 1 \(merge\): region_size must be greater than 0 for regions merge mode",
    ):
        PipelineSpec.from_file(pipeline_file)


def test_pipeline_rejects_unknown_filter_keys() -> None:
    with pytest.raises(ValueError, match="unsupported pipeline filter key: namess"):
        PipelineSpec.from_dict({"filters": [{"name": "fasteners", "namess": "Bolt*"}], "steps": [{"op": "repair"}]})


def test_pipeline_rejects_where_and_where_not_during_parse() -> None:
    with pytest.raises(ValueError, match="pipeline step cannot set both where and where_not"):
        PipelineSpec.from_dict(
            {
                "filters": [{"name": "fasteners", "path": "*/Fasteners/*"}],
                "steps": [{"op": "repair", "where": "fasteners", "where_not": "fasteners"}],
            }
        )


def test_pipeline_rejects_invalid_import_modes_with_line(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "bad-import.toml"
    pipeline_file.write_text('[import]\nmetadata = "verbose"\n\n[[steps]]\nop = "repair"\n', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"line 2: pipeline import metadata must be a bool or one of: none, summary, full",
    ):
        PipelineSpec.from_file(pipeline_file)


def test_convert_pipeline_file_can_set_import_and_export_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fascat.pipeline as pipeline

    pipeline_file = tmp_path / "metadata-pipeline.toml"
    pipeline_file.write_text(
        """
[import]
metadata = "none"
pmi = false
design_variants = true
existing_meshes = false
multi_file = true
delete_free_vertices = true
delete_lines = true
source_units = "millimetre"
source_up_axis = "Z"
source_handedness = "right"
target_units = "metre"
target_up_axis = "Y"
target_handedness = "right"

[export]
metadata = "summary"
pmi = "none"

[[steps]]
op = "repair"
""",
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_read_step(_path: str, *, options: object = None) -> Asset:
        captured["import_options"] = options
        return _triangle_asset()

    def fake_write_gltf(_asset: Asset, _path: str | Path, *, options: object = None) -> None:
        captured["export_options"] = options

    monkeypatch.setattr(pipeline, "read_step", fake_read_step)
    monkeypatch.setattr(pipeline, "_write_gltf", fake_write_gltf)
    monkeypatch.setattr(pipeline, "validate_gltf", lambda _path: {"meshes": 1, "points": 3, "triangles": 1})

    converted = convert("input.step", tmp_path / "output.glb", pipeline=PipelineSpec.from_file(pipeline_file))

    import_options = captured["import_options"]
    export_options = captured["export_options"]
    assert isinstance(import_options, StepReadOptions)
    assert isinstance(export_options, GltfExportOptions)
    assert import_options.metadata is False
    assert import_options.pmi is False
    assert import_options.design_variants is True
    assert import_options.existing_meshes is False
    assert import_options.multi_file is True
    assert import_options.delete_free_vertices is True
    assert import_options.delete_lines is True
    assert import_options.source_units == "millimetre"
    assert import_options.source_up_axis == "Z"
    assert import_options.source_handedness == "right"
    assert import_options.target_units == "metre"
    assert import_options.target_up_axis == "Y"
    assert import_options.target_handedness == "right"
    assert export_options.metadata.mode == "summary"
    assert export_options.metadata.pmi == "none"
    write_step = next(step for step in converted.report.steps if step.name == "write")
    assert write_step.options["metadata"] == {"mode": "summary", "pmi": "none"}


def test_convert_dispatches_gltf_writer_and_validator(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    asset = _triangle_asset()
    written: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "read_step", lambda _path: asset)

    def fake_write_gltf(asset: Asset, path: str | Path, *, options: object = None) -> None:
        written["path"] = str(path)
        written["triangles"] = asset.triangle_count

    monkeypatch.setattr(pipeline, "_write_gltf", fake_write_gltf)
    monkeypatch.setattr(pipeline, "validate_gltf", lambda _path: {"meshes": 1, "points": 3, "triangles": 1})

    converted = convert(
        "input.step",
        tmp_path / "output.glb",
        profile=_test_profile(),
    )
    steps = {step.name: step for step in converted.report.steps}

    assert written["path"] == str(tmp_path / "output.glb")
    assert written["triangles"] == 1
    write_options = dict(steps["write"].options)
    runtime_dependencies = write_options.pop("runtime_dependencies")
    assert write_options == {
        "format": "glTF",
        "quantize": False,
        "meshopt": False,
        "draco": False,
        "texture_compression": None,
        "file_size_budget_mb": None,
        "metadata": {"mode": "full", "pmi": "metadata"},
    }
    assert runtime_dependencies["extensions_used"] == []
    assert runtime_dependencies["extras"] == {"fascat": True, "metadata": "full", "pmi": "metadata"}
    assert steps["validate"].options == {"backend": "fascat-gltf"}
    assert steps["validate"].after["validated_triangles"] == 1


def test_convert_report_output_stats_include_lod_totals(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(pipeline, "_write_usd", lambda _asset, _path, *, debug=False, options=None: None)
    monkeypatch.setattr(pipeline, "validate_usd", lambda _path: {"meshes": 1, "points": 3, "triangles": 1})

    converted = convert(
        "input.step",
        tmp_path / "output.usdc",
        profile=_test_profile(),
        lods=LODOptions((0.5,)),
    )
    write_step = next(step for step in converted.report.steps if step.name == "write")

    assert write_step.before["lod_meshes"] == 1
    assert write_step.before["lod_triangles"] == 1
    assert converted.report.output_stats["lod_meshes"] == 1
    assert converted.report.output_stats["lod_triangles"] == 1


def test_convert_report_checks_profile_budget(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    profile = ConversionProfile(
        name="strict",
        tessellation=None,
        repair=RepairOptions(),
        stage=StageOptions(uv0="none", uv1=None),
        optimize=None,
        lods=None,
        budget=PlatformBudget(
            target_fps=60,
            max_triangles=1,
            max_vertices=2,
            max_vertices_per_mesh=2,
            max_texture_resolution=1_024,
            max_texture_memory_mb=1,
            max_load_time_ms=1,
            max_draw_calls=1,
            unity_reference_profile="test-platform",
            unity_reference_triangles=(100, 200),
            unity_reference_draw_calls=3,
        ),
    )
    source = _triangle_asset()
    source.parts["part"].material_ids = ["mat"]
    source.materials["mat"] = Material(
        id="mat",
        name="Material",
        base_color=(1.0, 1.0, 1.0, 1.0),
        metadata={"baked_texture_resolution": "2048", "baked_maps": "base_color,opacity,normal"},
    )
    monkeypatch.setattr(pipeline, "read_step", lambda _path: source)
    monkeypatch.setattr(pipeline, "_write_usd", lambda _asset, _path, *, debug=False, options=None: None)

    converted = convert("input.step", tmp_path / "output.usdc", profile=profile, validate_output=False)
    budget_step = converted.report.steps[-1]

    assert budget_step.name == "profile_budget"
    assert budget_step.options == {
        "profile": "strict",
        "target_fps": 60,
        "max_triangles": 1,
        "max_vertices": 2,
        "max_vertices_per_mesh": 2,
        "max_texture_resolution": 1024,
        "max_texture_memory_mb": 1,
        "max_load_time_ms": 1,
        "max_draw_calls": 1,
        "supported_compression": [],
        "supported_runtime_extensions": [],
        "unity_reference_profile": "test-platform",
        "unity_reference_triangles": [100, 200],
        "unity_reference_draw_calls": 3,
    }
    assert budget_step.before["triangles"] == 1
    assert budget_step.before["vertices"] == 3
    assert budget_step.before["draw_calls"] == 1
    assert budget_step.before["draw_call_meshes"] == 1
    assert budget_step.before["draw_call_materials"] == 1
    assert budget_step.before["draw_call_submesh_slots"] == 1
    assert budget_step.before["draw_call_mesh_instances"] == 1
    assert budget_step.before["draw_call_merged_batches"] == 0
    assert budget_step.after["profile_target_fps"] == 60
    assert budget_step.after["profile_triangles_over_budget"] == 0
    assert budget_step.after["profile_unity_reference_triangle_min"] == 100
    assert budget_step.after["profile_unity_reference_triangle_max"] == 200
    assert budget_step.after["profile_triangle_budget_below_unity_reference_min"] == 99
    assert budget_step.after["profile_triangle_budget_over_unity_reference_max"] == 0
    assert budget_step.after["profile_vertices_over_budget"] == 1
    assert budget_step.after["profile_max_vertices_per_mesh_budget"] == 2
    assert budget_step.after["profile_largest_mesh_vertices"] == 3
    assert budget_step.after["profile_meshes_over_vertex_budget"] == 1
    assert budget_step.after["profile_texture_resolution_budget"] == 1024
    assert budget_step.after["profile_largest_texture_resolution"] == 2048
    assert budget_step.after["profile_texture_sets_with_resolution"] == 1
    assert budget_step.after["profile_textures_over_resolution_budget"] == 1
    assert budget_step.after["profile_texture_memory_budget_bytes"] == 1_000_000
    assert budget_step.after["profile_texture_memory_texture_count"] == 2
    assert budget_step.after["profile_estimated_texture_memory_bytes"] == 33_554_432
    assert budget_step.after["profile_texture_memory_over_budget_bytes"] == 32_554_432
    assert budget_step.after["profile_load_time_budget_ms"] == 1
    assert budget_step.after["profile_estimated_load_time_ms"] == 673
    assert budget_step.after["profile_load_time_file_bytes"] == 0
    assert budget_step.after["profile_load_time_geometry_bytes"] == 108
    assert budget_step.after["profile_load_time_texture_bytes"] == 33_554_432
    assert budget_step.after["profile_load_time_over_budget_ms"] == 672
    assert budget_step.after["profile_draw_calls_over_budget"] == 0
    assert budget_step.after["profile_unity_reference_draw_call_budget"] == 3
    assert budget_step.after["profile_draw_call_budget_over_unity_reference"] == 0
    assert budget_step.after["profile_budget_violations"] == 5
    assert budget_step.warnings == [
        "profile budget exceeded for strict: vertices 3 > 2",
        "profile budget exceeded for strict: 1 mesh(es) exceed 2 vertices (largest 3)",
        "profile budget exceeded for strict: 1 texture set(s) exceed 1024px (largest 2048px)",
        "profile budget exceeded for strict: estimated texture memory 33554432 bytes > 1000000 bytes",
        "profile budget exceeded for strict: estimated load time 673ms > 1ms",
    ]
    assert converted.report.warnings[-5:] == budget_step.warnings


def test_convert_report_checks_profile_runtime_caps(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    profile = ConversionProfile(
        name="web-cap",
        tessellation=None,
        repair=RepairOptions(),
        stage=StageOptions(uv0="none", uv1=None),
        optimize=None,
        lods=None,
        budget=PlatformBudget(
            supported_compression=("quantization",),
            supported_runtime_extensions=("KHR_mesh_quantization",),
        ),
    )
    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(pipeline, "_write_gltf", lambda _asset, _path, *, options=None: None)

    converted = convert(
        "input.step",
        tmp_path / "output.glb",
        profile=profile,
        validate_output=False,
        gltf_options=GltfExportOptions(quantize=True, meshopt=True),
    )
    budget_step = converted.report.steps[-1]

    assert budget_step.name == "profile_budget"
    assert budget_step.options["supported_compression"] == ["quantization"]
    assert budget_step.options["supported_runtime_extensions"] == ["KHR_mesh_quantization"]
    assert budget_step.after["profile_supported_compression_count"] == 1
    assert budget_step.after["profile_emitted_compression_count"] == 2
    assert budget_step.after["profile_unsupported_compression_count"] == 1
    assert budget_step.after["profile_supported_runtime_extension_count"] == 1
    assert budget_step.after["profile_emitted_runtime_extension_count"] == 2
    assert budget_step.after["profile_unsupported_runtime_extension_count"] == 1
    assert budget_step.after["profile_budget_violations"] == 2
    assert budget_step.warnings == [
        "profile budget exceeded for web-cap: runtime compression uses unsupported method(s): meshopt",
        "profile budget exceeded for web-cap: runtime extensions exceed target cap: EXT_meshopt_compression",
    ]


def test_convert_report_finishes_when_validation_is_disabled(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(pipeline, "_write_usd", lambda _asset, _path, *, debug=False, options=None: None)

    converted = convert(
        "input.step",
        tmp_path / "output.usdc",
        profile=_test_profile(),
        validate_output=False,
    )

    assert converted.report.steps[-1].name == "workflow_summary"
    assert converted.report.steps[-2].name == "conversion_manifest"
    assert converted.report.steps[-3].name == "write"
    assert converted.report.finished_at is not None
    assert converted.report.output_stats == converted.stats()


def test_convert_report_records_write_failure(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    captured: dict[str, Asset] = {}
    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())

    def fail_write_usd(asset: Asset, _path: str | Path, *, debug: bool = False, options: object = None) -> None:
        captured["asset"] = asset
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "_write_usd", fail_write_usd)

    with pytest.raises(RuntimeError, match="disk full") as error:
        convert("input.step", tmp_path / "output.usdc", profile=_test_profile())

    failed = captured["asset"]
    assert error.value.report is failed.report
    assert failed.report.errors == ["disk full"]
    assert failed.report.steps[-1].name == "write"
    assert failed.report.steps[-1].before == failed.stats()
    assert failed.report.steps[-1].after == failed.stats()
    assert failed.report.finished_at is not None
    assert failed.report.output_stats == failed.stats()


def test_convert_report_records_validation_failure(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    captured: dict[str, Asset] = {}
    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(
        pipeline, "_write_usd", lambda asset, _path, *, debug=False, options=None: captured.setdefault("asset", asset)
    )
    monkeypatch.setattr(pipeline, "validate_usd", lambda _path: (_ for _ in ()).throw(RuntimeError("invalid usd")))

    with pytest.raises(RuntimeError, match="invalid usd") as error:
        convert("input.step", tmp_path / "output.usdc", profile=_test_profile())

    failed = captured["asset"]
    steps = [step.name for step in failed.report.steps]
    assert error.value.report is failed.report
    assert failed.report.errors == ["invalid usd"]
    assert steps[-2:] == ["write", "validate"]
    assert failed.report.steps[-1].before == failed.stats()
    assert failed.report.steps[-1].after == failed.stats()
    assert failed.report.finished_at is not None


def test_operation_report_step_captures_warnings() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Missing", part_id="missing")]),
        parts={"missing": Part(id="missing", name="Missing")},
    )

    tessellated = asset.tessellate()
    step = tessellated.report.steps[-1]

    assert step.name == "tessellate"
    assert step.warnings == ["part has no source shape and cannot be tessellated: Missing"]
    assert tessellated.report.warnings == step.warnings
