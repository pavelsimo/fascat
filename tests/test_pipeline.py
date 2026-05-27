from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
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
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    RepairOptions,
    ReplaceOptions,
    StageOptions,
    StepReadOptions,
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
        "stage": {
            "materials",
            "material_mode",
            "merge_equivalent_materials",
            "normals",
            "normal_mode",
            "hard_edge_angle",
            "preserve_face_boundaries",
            "tangents",
            "validate_normals",
            "unwrap",
            "atlas",
            "uv0",
            "uv1",
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
    assert [step.name for step in converted.report.steps[1:]] == ["repair", "replace", "write", "validate"]
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
    assert export_options.metadata.mode == "summary"
    assert export_options.metadata.pmi == "none"
    assert converted.report.steps[-2].options["metadata"] == {"mode": "summary", "pmi": "none"}


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
    assert steps["write"].options == {
        "format": "glTF",
        "quantize": False,
        "meshopt": False,
        "draco": False,
        "texture_compression": None,
        "file_size_budget_mb": None,
        "metadata": {"mode": "full", "pmi": "metadata"},
    }
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

    assert converted.report.steps[-1].name == "write"
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
