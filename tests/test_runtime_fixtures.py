from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any, cast

from PIL import Image

from fascat.io.gltf import validate_gltf
from fascat.runtime import RuntimeBrowserRenderReport, RuntimeEngineReport
from fascat.runtime_fixtures import (
    audit_runtime_parity_goldens,
    capture_runtime_parity_suite,
    write_runtime_parity_suite,
)
from fascat.validation import (
    RuntimeParityCaptureReport,
    RuntimeParityGolden,
    RuntimeParityGoldenCoverageReport,
    RuntimeParitySuiteReport,
    VisualDiffOptions,
)


def test_runtime_parity_suite_writes_assets_baselines_and_manifest(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    report = write_runtime_parity_suite(suite_dir)

    assert isinstance(report, RuntimeParitySuiteReport)
    assert report.targets == ("browser", "unity", "unreal")
    assert len(report.fixtures) == 6
    assert Path(report.manifest_path).is_file()
    assert Path(report.directory) == suite_dir
    assert report.to_dict()["recommended_diff"] == {
        "pixel_tolerance": 2,
        "max_mean_absolute_error": 4.0,
        "max_changed_pixel_ratio": 0.02,
    }

    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema"] == "fascat.runtime-parity-suite.v1"
    assert manifest["targets"] == ["browser", "unity", "unreal"]
    assert manifest["layout"]["target_goldens"] == "goldens/{target}/{fixture}.png"
    assert [fixture["name"] for fixture in manifest["fixtures"]] == [
        "pbr-material-grid",
        "texture-map-grid",
        "ktx2-basis-fallback",
        "normal-lighting-wedges",
        "lod-profile-unity",
        "lod-profile-unreal",
    ]

    for fixture in report.fixtures:
        asset_path = Path(fixture.asset_path)
        baseline_path = Path(fixture.software_baseline_path)
        assert asset_path.is_file()
        assert baseline_path.is_file()
        assert validate_gltf(asset_path)["triangles"] == fixture.triangles
        with Image.open(baseline_path) as baseline:
            assert baseline.size == (512, 512)

    texture_manifest = manifest["fixtures"][1]
    assert texture_manifest["asset"] == "assets/texture-map-grid.glb"
    assert texture_manifest["software_baseline"] == "baselines/texture-map-grid.png"
    assert texture_manifest["target_goldens"] == {
        "browser": "goldens/browser/texture-map-grid.png",
        "unity": "goldens/unity/texture-map-grid.png",
        "unreal": "goldens/unreal/texture-map-grid.png",
    }
    assert "runtime-browser-preview" in texture_manifest["commands"]["browser"]
    assert "--runtime-engine unity" in texture_manifest["commands"]["unity"]
    assert "--runtime-engine-baseline baselines/texture-map-grid.png" in texture_manifest["commands"]["unity"]
    assert "--runtime-engine unreal" in texture_manifest["commands"]["unreal"]
    assert "--visual-diff-pixel-tolerance 2" in texture_manifest["commands"]["unity"]
    assert "--visual-diff-mean-threshold 4" in texture_manifest["commands"]["unity"]
    assert "--visual-diff-changed-pixel-ratio 0.02" in texture_manifest["commands"]["unreal"]


def test_runtime_parity_suite_preserves_explicit_diff_options(tmp_path: Path) -> None:
    diff_options = VisualDiffOptions(
        pixel_tolerance=6,
        max_mean_absolute_error=12.5,
        max_changed_pixel_ratio=0.1,
    )

    report = write_runtime_parity_suite(tmp_path / "runtime-parity", diff_options=diff_options)
    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))

    assert report.recommended_diff == diff_options
    assert manifest["recommended_diff"] == diff_options.to_dict()
    assert "--visual-diff-pixel-tolerance 6" in manifest["fixtures"][0]["commands"]["unity"]
    assert "--visual-diff-mean-threshold 12.5" in manifest["fixtures"][0]["commands"]["unity"]
    assert "--visual-diff-changed-pixel-ratio 0.1" in manifest["fixtures"][0]["commands"]["unreal"]


def test_runtime_parity_lod_profile_fixtures_exercise_engine_exports(tmp_path: Path) -> None:
    report = write_runtime_parity_suite(tmp_path / "runtime-parity")
    unity_fixture = next(fixture for fixture in report.fixtures if fixture.name == "lod-profile-unity")
    unreal_fixture = next(fixture for fixture in report.fixtures if fixture.name == "lod-profile-unreal")

    unity = _read_glb_document(Path(unity_fixture.asset_path))
    unity_root = unity["nodes"][unity["scenes"][0]["nodes"][0]]
    unity_occurrence = unity["nodes"][unity_root["children"][0]]
    unity_lod_index = unity_occurrence["extensions"]["MSFT_lod"]["ids"][0]
    unity_lod = unity["nodes"][unity_lod_index]
    assert "MSFT_lod" in unity["extensionsUsed"]
    assert unity_occurrence["extras"]["fascat"]["lodEngineProfile"] == "unity"
    assert unity_occurrence["extras"]["fascat"]["lodExportMode"] == "variants"
    assert unity_lod["name"].endswith("_lod1")
    assert "unity_msft_lod_export" in unity_fixture.checks

    unreal = _read_glb_document(Path(unreal_fixture.asset_path))
    unreal_root = unreal["nodes"][unreal["scenes"][0]["nodes"][0]]
    unreal_occurrence = unreal["nodes"][unreal_root["children"][0]]
    unreal_lod = unreal["nodes"][unreal_root["children"][1]]
    assert "MSFT_lod" not in unreal.get("extensionsUsed", [])
    assert unreal_occurrence["extras"]["fascat"]["lodEngineProfile"] == "unreal"
    assert unreal_occurrence["extras"]["fascat"]["lodExportMode"] == "separate"
    assert unreal_lod["name"].endswith("_LOD1")
    assert unreal_lod["extras"]["fascat"]["lodEngineProfile"] == "unreal"
    assert "unreal_separate_lod_nodes" in unreal_fixture.checks


def test_runtime_parity_texture_fixture_exercises_material_texture_slots(tmp_path: Path) -> None:
    report = write_runtime_parity_suite(tmp_path / "runtime-parity")
    texture_fixture = next(fixture for fixture in report.fixtures if fixture.name == "texture-map-grid")

    document = _read_glb_document(Path(texture_fixture.asset_path))
    material = document["materials"][0]

    assert len(document["images"]) == 5
    assert len(document["textures"]) == 5
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"] == 1
    assert material["normalTexture"]["index"] == 2
    assert material["occlusionTexture"]["index"] == 3
    assert material["emissiveTexture"]["index"] == 4
    assert document["extras"]["fascat"]["metadata"]["runtime_parity_fixture"] == "texture-map-grid"


def test_runtime_parity_ktx2_fixture_exercises_basisu_fallback(tmp_path: Path) -> None:
    report = write_runtime_parity_suite(tmp_path / "runtime-parity")
    ktx2_fixture = next(fixture for fixture in report.fixtures if fixture.name == "ktx2-basis-fallback")

    document = _read_glb_document(Path(ktx2_fixture.asset_path))
    texture = document["textures"][0]
    ktx2_image = document["images"][texture["extensions"]["KHR_texture_basisu"]["source"]]

    assert "KHR_texture_basisu" in document["extensionsUsed"]
    assert "KHR_texture_basisu" not in document.get("extensionsRequired", [])
    assert texture["source"] == 0
    assert document["images"][0]["uri"].startswith("data:image/png;base64,")
    assert ktx2_image["mimeType"] == "image/ktx2"
    assert ktx2_image["uri"].startswith("data:image/ktx2;base64,q0tUWCAyMLsNChoK")
    assert document["extras"]["fascat"]["metadata"]["runtime_parity_fixture"] == "ktx2-basis-fallback"
    assert document["extras"]["fascat"]["metadata"]["runtime_parity_ktx2_fallback"] == "png_source"


def test_runtime_parity_pbr_fixture_marks_alpha_material(tmp_path: Path) -> None:
    report = write_runtime_parity_suite(tmp_path / "runtime-parity")
    pbr_fixture = next(fixture for fixture in report.fixtures if fixture.name == "pbr-material-grid")

    document = _read_glb_document(Path(pbr_fixture.asset_path))
    materials_by_name = {material["name"]: material for material in document["materials"]}

    assert materials_by_name["Transparent Green"]["alphaMode"] == "BLEND"
    assert materials_by_name["Polished Blue Metal"]["pbrMetallicRoughness"]["metallicFactor"] == 1.0
    assert pbr_fixture.materials == 4
    assert pbr_fixture.triangles == 8


def test_runtime_parity_capture_records_previews_diffs_and_goldens(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "runtime-parity"
    write_runtime_parity_suite(suite_dir)

    def copy_baseline(asset_path: Path, preview_path: Path) -> None:
        baseline = asset_path.parent.parent / "baselines" / f"{asset_path.stem}.png"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline, preview_path)

    def fake_browser(path: str | Path, preview_path: str | Path, _options: object = None) -> RuntimeBrowserRenderReport:
        copy_baseline(Path(path), Path(preview_path))
        return RuntimeBrowserRenderReport(
            path=str(path),
            status="rendered",
            browser="fake-browser",
            preview_path=str(preview_path),
            width=512,
            height=512,
            meshes=1,
            triangles=1,
        )

    def fake_engine(path: str | Path, options: object) -> RuntimeEngineReport:
        engine_options = cast(Any, options)
        preview_path = Path(engine_options.preview_path)
        copy_baseline(Path(path), preview_path)
        return RuntimeEngineReport(
            path=str(path),
            status="measured",
            engine=engine_options.engine,
            executable="fake-engine",
            project=str(tmp_path / "Harness"),
            engine_version="test-engine",
            load_time_ms=10,
            measured_fps=60.0,
            frame_count=30,
            measurement_duration_ms=500,
            memory_bytes=4096,
            meshes=1,
            triangles=1,
            preview_path=str(preview_path),
            render_status="rendered",
            render_time_ms=5,
            rendered_frames=30,
        )

    monkeypatch.setattr("fascat.runtime_fixtures.write_browser_render_preview", fake_browser)
    monkeypatch.setattr("fascat.runtime_fixtures.measure_engine_runtime", fake_engine)

    report = capture_runtime_parity_suite(
        suite_dir,
        targets=("browser", "unity"),
        browser_command="fake-browser",
        unity_command="Unity",
        promote_goldens=True,
    )

    assert isinstance(report, RuntimeParityCaptureReport)
    assert report.passed is True
    assert len(report.captures) == 12
    assert Path(report.results_path).is_file()
    assert all(capture.passed is True for capture in report.captures)
    assert all(capture.diff is not None for capture in report.captures)
    assert (suite_dir / "previews" / "texture-map-grid-unity.png").is_file()
    assert (suite_dir / "previews" / "ktx2-basis-fallback-browser.png").is_file()
    assert (suite_dir / "goldens" / "browser" / "texture-map-grid.png").is_file()
    assert (suite_dir / "goldens" / "unity" / "texture-map-grid.png").is_file()

    payload = json.loads(Path(report.results_path).read_text(encoding="utf-8"))
    assert payload["schema"] == "fascat.runtime-parity-captures.v1"
    assert payload["passed"] is True
    assert payload["targets"] == ["browser", "unity"]
    assert payload["required_goldens"] is False


def test_runtime_parity_capture_uses_existing_target_goldens(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "runtime-parity"
    write_runtime_parity_suite(suite_dir)
    fixture_name = "pbr-material-grid"
    target_golden = suite_dir / "goldens" / "browser" / f"{fixture_name}.png"
    target_golden.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(suite_dir / "baselines" / f"{fixture_name}.png", target_golden)

    def fake_browser(path: str | Path, preview_path: str | Path, _options: object = None) -> RuntimeBrowserRenderReport:
        baseline = Path(path).parent.parent / "baselines" / f"{Path(path).stem}.png"
        Path(preview_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(baseline, preview_path)
        return RuntimeBrowserRenderReport(
            path=str(path),
            status="rendered",
            browser="fake-browser",
            preview_path=str(preview_path),
            width=512,
            height=512,
            meshes=1,
            triangles=1,
        )

    monkeypatch.setattr("fascat.runtime_fixtures.write_browser_render_preview", fake_browser)

    report = capture_runtime_parity_suite(suite_dir, targets=("browser",))

    target_capture = next(capture for capture in report.captures if capture.fixture == fixture_name)
    other_capture = next(capture for capture in report.captures if capture.fixture == "texture-map-grid")
    assert target_capture.baseline_kind == "target_golden"
    assert target_capture.baseline_path == str(target_golden)
    assert target_capture.golden_path == str(target_golden)
    assert target_capture.diff is not None
    assert target_capture.diff["baseline_path"] == str(target_golden)
    assert other_capture.baseline_kind == "software_baseline"


def test_runtime_parity_capture_can_require_target_goldens(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "runtime-parity"
    write_runtime_parity_suite(suite_dir)

    def fail_browser(*_args: object, **_kwargs: object) -> RuntimeBrowserRenderReport:
        raise AssertionError("browser capture should not run when required goldens are missing")

    monkeypatch.setattr("fascat.runtime_fixtures.write_browser_render_preview", fail_browser)

    report = capture_runtime_parity_suite(suite_dir, targets=("browser",), require_goldens=True)

    assert report.passed is False
    assert report.required_goldens is True
    assert len(report.captures) == 6
    assert all(capture.status == "missing_golden" for capture in report.captures)
    assert all(capture.render_status == "not_rendered" for capture in report.captures)
    assert all(capture.passed is False for capture in report.captures)
    assert all(capture.baseline_kind == "missing_target_golden" for capture in report.captures)
    assert all(capture.golden_path is not None for capture in report.captures)


def test_runtime_parity_golden_audit_reports_missing_and_present_goldens(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"
    write_runtime_parity_suite(suite_dir)
    target_golden = suite_dir / "goldens" / "unity" / "pbr-material-grid.png"
    target_golden.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(suite_dir / "baselines" / "pbr-material-grid.png", target_golden)

    report = audit_runtime_parity_goldens(suite_dir, targets=("unity",))

    assert isinstance(report, RuntimeParityGoldenCoverageReport)
    assert report.passed is False
    assert report.present_count == 1
    assert report.missing_count == 5
    assert report.invalid_count == 0
    assert Path(report.results_path).is_file()
    present = next(golden for golden in report.goldens if golden.status == "present")
    assert isinstance(present, RuntimeParityGolden)
    assert present.fixture == "pbr-material-grid"
    assert present.target == "unity"
    assert present.width == 512
    assert present.expected_width == 512

    payload = json.loads(Path(report.results_path).read_text(encoding="utf-8"))
    assert payload["schema"] == "fascat.runtime-parity-golden-coverage.v1"
    assert payload["present_count"] == 1
    assert payload["missing_count"] == 5
    assert payload["goldens"][0]["target"] == "unity"


def test_runtime_parity_golden_audit_rejects_invalid_or_wrong_size_png(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"
    write_runtime_parity_suite(suite_dir)
    target_golden = suite_dir / "goldens" / "browser" / "pbr-material-grid.png"
    target_golden.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(target_golden)

    report = audit_runtime_parity_goldens(suite_dir, targets=("browser",))

    mismatch = next(golden for golden in report.goldens if golden.fixture == "pbr-material-grid")
    assert mismatch.status == "dimension_mismatch"
    assert mismatch.passed is False
    assert mismatch.width == 64
    assert mismatch.expected_width == 512
    assert report.invalid_count == 1


def _read_glb_document(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))
