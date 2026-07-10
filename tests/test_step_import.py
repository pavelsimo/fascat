from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc
import fascat.io.step as step_io
from fascat.errors import FascatIOError
from fascat.io._import_base import (
    _ImportCleanupStats,
    _ShapeTopologyCounts,
    _space_normalization,
    _StepHeaderInfo,
)
from fascat.io.step import single as step_single
from fascat.io.step.single import (
    _import_decisions,
    _import_warnings,
)
from fascat.options import StepReadOptions


def test_read_step_bytes_closes_and_removes_temporary_file(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_paths: list[Path] = []

    def fake_read_step_path(path: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        assert source_identity == "stdin.step"
        assert path.exists()
        seen_paths.append(path)
        return fc.Asset(root=fc.Node(id="root", name="Root"))

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    asset = step_io.read_step_bytes(b"ISO-10303-21;")

    assert asset.root.metadata["source"] == "stdin.step"
    assert seen_paths and not seen_paths[0].exists()


@pytest.mark.parametrize(("name", "suffix"), [("input.stp", ".stp"), ("input", ".step")])
def test_read_step_bytes_preserves_temporary_suffix(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    suffix: str,
) -> None:
    seen_paths: list[Path] = []

    def fake_read_step_path(path: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        assert source_identity == name
        assert path.suffix == suffix
        seen_paths.append(path)
        return fc.Asset(root=fc.Node(id="root", name="Root"))

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    step_io.read_step_bytes(b"ISO-10303-21;", name=name)

    assert seen_paths and not seen_paths[0].exists()


def test_read_step_bytes_removes_temporary_file_when_reader_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_paths: list[Path] = []

    def fake_read_step_path(path: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        seen_paths.append(path)
        raise RuntimeError("broken STEP input")

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    with pytest.raises(FascatIOError, match="broken STEP input"):
        step_io.read_step_bytes(b"broken")

    assert seen_paths and not seen_paths[0].exists()


def test_step_package_exports_only_public_readers() -> None:
    assert step_io.__all__ == ["read_step", "read_step_bytes", "read_step_many"]


def test_step_import_warnings_report_unsupported_import_intent() -> None:
    warnings = _import_warnings(
        StepReadOptions(design_variants=True, multi_file=True),
        _StepHeaderInfo(schema="AP242", pmi_present=True),
        unsupported_pmi_count=1,
    )

    assert warnings == [
        "STEP file advertises AP242 PMI, but no supported typed PMI entities were extracted; annotations are omitted",
        "STEP design variant import was requested, but no supported design variant records were detected",
    ]
    assert not any("multi-file" in warning and "not implemented" in warning for warning in warnings)


def test_step_read_options_normalize_construction_curve_policy() -> None:
    options = StepReadOptions(construction_curve_policy="tessellate-tubes")

    assert options.construction_curve_policy == "tessellate_tubes"

    with pytest.raises(ValueError, match="construction_curve_tube_radius"):
        StepReadOptions(construction_curve_policy="tessellate_tubes", construction_curve_tube_radius=0.0)


def test_step_import_decisions_report_requested_effective_states() -> None:
    cleanup = _ImportCleanupStats()
    cleanup.record_deleted("delete_lines", _ShapeTopologyCounts(vertices=4, edges=2))
    space = _space_normalization(
        "millimetre",
        0.001,
        StepReadOptions(target_units="metre", target_up_axis="Y", target_handedness="right"),
    )

    decisions = _import_decisions(
        StepReadOptions(
            design_variants=True,
            multi_file=True,
            delete_free_vertices=True,
            delete_lines=True,
        ),
        _StepHeaderInfo(schema="AP242", pmi_present=True),
        pmi_count=0,
        unsupported_pmi_count=1,
        cleanup=cleanup,
        space=space,
    )

    assert decisions["pmi"]["state"] == "unsupported"
    assert decisions["design_variants"]["state"] == "not_present"
    assert decisions["design_variants"]["counts"]["records"] == 0
    assert decisions["multi_file"]["state"] == "unsupported"
    assert "single-file STEP import" in decisions["multi_file"]["detail"]
    assert "not implemented" not in decisions["multi_file"]["detail"]
    assert decisions["source_textures"]["state"] == "honored"
    assert decisions["material_library_mapping"]["state"] == "honored"
    assert decisions["delete_free_vertices"]["state"] == "honored"
    assert decisions["delete_free_vertices"]["counts"] == {"deleted_parts": 0, "deleted_vertices": 0}
    assert decisions["delete_lines"]["counts"] == {
        "deleted_parts": 1,
        "deleted_edges": 2,
        "deleted_vertices": 4,
    }
    assert decisions["construction_curves"]["requested"] == "delete"
    assert decisions["construction_curves"]["state"] == "honored"
    assert decisions["construction_curves"]["counts"]["deleted_parts"] == 1
    assert decisions["space_normalization"]["state"] == "honored"
