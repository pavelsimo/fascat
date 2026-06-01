from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path
from typing import Any, cast

from PIL import Image

import fascat as fc
from fascat.io.gltf import validate_gltf
from fascat.runtime import RuntimeBrowserRenderReport, RuntimeEngineReport
from fascat.runtime_fixtures import capture_runtime_parity_suite, write_runtime_parity_suite


def test_runtime_parity_suite_writes_assets_baselines_and_manifest(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    report = write_runtime_parity_suite(suite_dir)

    assert isinstance(report, fc.RuntimeParitySuiteReport)
    assert report.targets == ("browser", "unity", "unreal")
    assert len(report.fixtures) == 4
    assert Path(report.manifest_path).is_file()
    assert Path(report.directory) == suite_dir
    assert report.to_dict()["recommended_diff"]["pixel_tolerance"] == 8

    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema"] == "fascat.runtime-parity-suite.v1"
    assert manifest["targets"] == ["browser", "unity", "unreal"]
    assert [fixture["name"] for fixture in manifest["fixtures"]] == [
        "pbr-material-grid",
        "texture-map-grid",
        "ktx2-basis-fallback",
        "normal-lighting-wedges",
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
    assert "runtime-browser-preview" in texture_manifest["commands"]["browser"]
    assert "--runtime-engine unity" in texture_manifest["commands"]["unity"]
    assert "--runtime-engine-baseline baselines/texture-map-grid.png" in texture_manifest["commands"]["unity"]
    assert "--runtime-engine unreal" in texture_manifest["commands"]["unreal"]


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

    assert isinstance(report, fc.RuntimeParityCaptureReport)
    assert report.passed is True
    assert len(report.captures) == 8
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


def _read_glb_document(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))
