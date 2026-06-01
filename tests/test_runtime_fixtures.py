from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from PIL import Image

import fascat as fc
from fascat.io.gltf import validate_gltf
from fascat.runtime_fixtures import write_runtime_parity_suite


def test_runtime_parity_suite_writes_assets_baselines_and_manifest(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    report = write_runtime_parity_suite(suite_dir)

    assert isinstance(report, fc.RuntimeParitySuiteReport)
    assert report.targets == ("browser", "unity", "unreal")
    assert len(report.fixtures) == 3
    assert Path(report.manifest_path).is_file()
    assert Path(report.directory) == suite_dir
    assert report.to_dict()["recommended_diff"]["pixel_tolerance"] == 8

    manifest = json.loads(Path(report.manifest_path).read_text(encoding="utf-8"))
    assert manifest["schema"] == "fascat.runtime-parity-suite.v1"
    assert manifest["targets"] == ["browser", "unity", "unreal"]
    assert [fixture["name"] for fixture in manifest["fixtures"]] == [
        "pbr-material-grid",
        "texture-map-grid",
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


def test_runtime_parity_pbr_fixture_marks_alpha_material(tmp_path: Path) -> None:
    report = write_runtime_parity_suite(tmp_path / "runtime-parity")
    pbr_fixture = next(fixture for fixture in report.fixtures if fixture.name == "pbr-material-grid")

    document = _read_glb_document(Path(pbr_fixture.asset_path))
    materials_by_name = {material["name"]: material for material in document["materials"]}

    assert materials_by_name["Transparent Green"]["alphaMode"] == "BLEND"
    assert materials_by_name["Polished Blue Metal"]["pbrMetallicRoughness"]["metallicFactor"] == 1.0
    assert pbr_fixture.materials == 4
    assert pbr_fixture.triangles == 8


def _read_glb_document(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))
