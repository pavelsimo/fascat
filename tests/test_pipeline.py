from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import ConversionProfile, RepairOptions, StageOptions
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


def test_convert_report_includes_timed_write_and_validate_steps(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    asset = _triangle_asset()
    written: dict[str, object] = {}

    monkeypatch.setattr(pipeline, "read_step", lambda _path: asset)

    def fake_write_usd(asset: Asset, path: str | Path, *, debug: bool = False) -> None:
        written["path"] = str(path)
        written["debug"] = debug
        written["triangles"] = asset.triangle_count

    monkeypatch.setattr(pipeline, "write_usd", fake_write_usd)
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


def test_convert_report_records_write_failure(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.pipeline as pipeline

    captured: dict[str, Asset] = {}
    monkeypatch.setattr(pipeline, "read_step", lambda _path: _triangle_asset())

    def fail_write_usd(asset: Asset, _path: str | Path, *, debug: bool = False) -> None:
        captured["asset"] = asset
        raise RuntimeError("disk full")

    monkeypatch.setattr(pipeline, "write_usd", fail_write_usd)

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
    monkeypatch.setattr(pipeline, "write_usd", lambda asset, _path, *, debug=False: captured.setdefault("asset", asset))
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
