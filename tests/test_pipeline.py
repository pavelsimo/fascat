from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import ConversionProfile, LODOptions, OptimizeOptions, RepairOptions, StageOptions
from fascat.pipeline import convert
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
            "angle",
            "relative",
            "min_edge_length",
            "max_edge_length",
            "preserve_boundaries",
            "curvature_adaptive",
            "avoid_skinny_triangles",
            "quality_report",
            "create_normals",
            "keep_brep",
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
        "lods": {"ratios", "mode"},
    }
    operations = [
        ("tessellate", lambda asset: asset.tessellate()),
        ("repair", lambda asset: asset.repair(RepairOptions())),
        ("stage", lambda asset: asset.stage(StageOptions(uv0="none", uv1=None))),
        ("optimize", lambda asset: asset.optimize()),
        ("lods", lambda asset: asset.lods(LODOptions((0.5,)))),
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

    def fake_write_usd(asset: Asset, path: str | Path, *, debug: bool = False) -> None:
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


def test_convert_dispatches_gltf_writer_and_validator(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    asset = _triangle_asset()
    written: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "read_step", lambda _path: asset)

    def fake_write_gltf(asset: Asset, path: str | Path) -> None:
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
    assert steps["write"].options == {"format": "glTF"}
    assert steps["validate"].options == {"backend": "fascat-gltf"}
    assert steps["validate"].after["validated_triangles"] == 1


def test_convert_report_output_stats_include_lod_totals(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())
    monkeypatch.setattr(pipeline, "_write_usd", lambda _asset, _path, *, debug=False: None)
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
    monkeypatch.setattr(pipeline, "_write_usd", lambda _asset, _path, *, debug=False: None)

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

    def fail_write_usd(asset: Asset, _path: str | Path, *, debug: bool = False) -> None:
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
        pipeline, "_write_usd", lambda asset, _path, *, debug=False: captured.setdefault("asset", asset)
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
