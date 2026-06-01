from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import pytest
import tomli
from PIL import Image

from fascat.asset import Asset, Node, Part
from fascat.io.gltf import validate_gltf
from fascat.mesh import Mesh
from fascat.options import GltfExportOptions
from fascat.runtime import (
    RuntimeBrowserOptions,
    RuntimeBrowserRenderOptions,
    RuntimeEngineOptions,
    _runtime_browser_render_html,
    copy_engine_runtime_harness,
    measure_browser_runtime,
    measure_engine_runtime,
    write_browser_render_preview,
)
from fascat.runtime_fixtures import write_runtime_parity_suite


def test_browser_runtime_reports_unavailable_when_browser_is_missing(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    _asset().write_gltf(output)
    monkeypatch.delenv("FASCAT_BROWSER", raising=False)
    monkeypatch.setattr("fascat.runtime.shutil.which", lambda _name: None)

    report = measure_browser_runtime(output)

    assert report.status == "unavailable"
    assert report.browser is None
    assert report.meshes == 1
    assert report.triangles == 1
    assert report.measured_fps is None
    assert "no chromium-compatible browser" in str(report.error)


def test_browser_runtime_parses_headless_browser_measurements(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    _asset().write_gltf(output)

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = """
        <html><body><pre id="result">{
          &quot;status&quot;:&quot;measured&quot;,
          &quot;load_time_ms&quot;:12,
          &quot;measured_fps&quot;:58.5,
          &quot;frame_count&quot;:117,
          &quot;measurement_duration_ms&quot;:2000,
          &quot;memory_bytes&quot;:4096,
          &quot;meshes&quot;:1,
          &quot;triangles&quot;:1,
          &quot;workload_triangles&quot;:1
        }</pre></body></html>
        """
        return subprocess.CompletedProcess(["fake-browser"], 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_browser_runtime(output, RuntimeBrowserOptions(browser="fake-browser"))

    assert report.status == "measured"
    assert report.browser == "fake-browser"
    assert report.load_time_ms == 12
    assert report.measured_fps == 58.5
    assert report.frame_count == 117
    assert report.memory_bytes == 4096
    assert report.workload_scale == 1.0


def test_browser_render_preview_reports_unavailable_when_browser_is_missing(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "preview.png"
    _asset().write_gltf(output)
    monkeypatch.delenv("FASCAT_BROWSER", raising=False)
    monkeypatch.setattr("fascat.runtime.shutil.which", lambda _name: None)

    report = write_browser_render_preview(output, preview)

    assert report.status == "unavailable"
    assert report.browser is None
    assert report.preview_path == str(preview)
    assert report.meshes == 1
    assert report.triangles == 1
    assert "no chromium-compatible browser" in str(report.error)
    assert not preview.exists()


def test_browser_render_preview_writes_screenshot_and_report(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        screenshot_args = [item for item in command if item.startswith("--screenshot=")]
        if screenshot_args:
            screenshot_path = Path(screenshot_args[0].split("=", 1)[1])
            Image.new("RGBA", (800, 600), (10, 20, 30, 255)).save(screenshot_path)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        assert "--dump-dom" in command
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            '"textured_primitives":1,"sampled_textures":1,"quantized_primitives":1}'
            "</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered"
    assert report.browser == "fake-browser"
    assert report.preview_path == str(preview)
    assert report.width == 800
    assert report.height == 600
    assert report.meshes == 1
    assert report.triangles == 1
    assert report.textured_primitives == 1
    assert report.sampled_textures == 1
    assert report.quantized_primitives == 1
    assert report.to_dict()["textured_primitives"] == 1
    assert report.to_dict()["sampled_textures"] == 1
    assert report.to_dict()["quantized_primitives"] == 1
    assert report.error is None
    assert preview.is_file()


def test_browser_render_preview_writes_screenshot_data_from_payload(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)
    image_bytes = BytesIO()
    Image.new("RGBA", (2, 2), (230, 20, 30, 255)).save(image_bytes, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")
    calls: list[list[str]] = []

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert "--dump-dom" in command
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            '"textured_primitives":1,"sampled_textures":1,'
            '"quantized_primitives":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert len(calls) == 1
    assert report.status == "rendered"
    assert report.textured_primitives == 1
    assert report.sampled_textures == 1
    assert report.quantized_primitives == 1
    assert Image.open(preview).getpixel((0, 0)) == (230, 20, 30, 255)


def test_browser_render_preview_reports_unsupported_draco_decode_failure_without_running_browser(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["extensionsUsed"] = ["KHR_draco_mesh_compression"]
    document["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    document["meshes"][0]["primitives"][0]["extensions"] = {"KHR_draco_mesh_compression": {"bufferView": 0}}
    output.write_text(json.dumps(document), encoding="utf-8")

    def fake_copy(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("glTF Transform copy failed")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("browser should not run for unsupported Draco preview")

    monkeypatch.setattr("fascat.runtime._run_gltf_transform_copy", fake_copy)
    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "unsupported"
    assert report.browser is None
    assert report.required_extensions == ("KHR_draco_mesh_compression",)
    assert report.unsupported_extensions == ("KHR_draco_mesh_compression",)
    assert "could not decode KHR_draco_mesh_compression" in str(report.error)
    assert not preview.exists()


def test_browser_render_preview_decodes_draco_geometry(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["extensionsUsed"] = ["KHR_draco_mesh_compression"]
    document["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    document["meshes"][0]["primitives"][0]["extensions"] = {"KHR_draco_mesh_compression": {"bufferView": 0}}
    output.write_text(json.dumps(document), encoding="utf-8")
    image_bytes = BytesIO()
    Image.new("RGBA", (2, 2), (60, 70, 80, 255)).save(image_bytes, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")

    def fake_copy(input_path: Path, output_path: Path) -> None:
        assert input_path == output
        assert output_path.name == "draco-decoded.glb"
        _asset().write_gltf(output_path)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(urlparse(command[-1]).path)
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(unquote(urlparse(decoded_url).path))
        assert decoded_path.name == "draco-decoded.glb"
        assert validate_gltf(decoded_path)["triangles"] == 1
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime._run_gltf_transform_copy", fake_copy)
    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered"
    assert report.decoded_extensions == ("KHR_draco_mesh_compression",)
    assert report.unsupported_extensions == ()
    assert report.preview_limitations == ()
    assert Image.open(preview).getpixel((0, 0)) == (60, 70, 80, 255)


def test_browser_render_preview_decodes_meshopt_only_buffer_views(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output, options=GltfExportOptions(meshopt=True))
    document = json.loads(output.read_text(encoding="utf-8"))
    document.setdefault("extensionsRequired", []).append("EXT_meshopt_compression")
    document["buffers"].append(
        {
            "byteLength": document["buffers"][0]["byteLength"],
            "extensions": {"EXT_meshopt_compression": {"fallback": True}},
        }
    )
    for view in document["bufferViews"]:
        extension = view.get("extensions", {}).get("EXT_meshopt_compression")
        if extension is None:
            continue
        view["buffer"] = 1
        view["byteOffset"] = 0
        view["byteLength"] = extension["count"] * extension["byteStride"]
    output.write_text(json.dumps(document), encoding="utf-8")
    image_bytes = BytesIO()
    Image.new("RGBA", (2, 2), (70, 80, 90, 255)).save(image_bytes, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(urlparse(command[-1]).path)
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(unquote(urlparse(decoded_url).path))
        decoded_document = json.loads(decoded_path.read_text(encoding="utf-8"))
        assert "EXT_meshopt_compression" not in decoded_document.get("extensionsRequired", [])
        assert all(
            "EXT_meshopt_compression" not in view.get("extensions", {}) for view in decoded_document["bufferViews"]
        )
        assert validate_gltf(decoded_path)["triangles"] == 1
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered"
    assert report.decoded_extensions == ("EXT_meshopt_compression",)
    assert report.unsupported_extensions == ()
    assert report.preview_limitations == ()
    assert Image.open(preview).getpixel((0, 0)) == (70, 80, 90, 255)


def test_browser_render_preview_marks_ktx2_texture_preview_partial(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["extensionsUsed"] = ["KHR_texture_basisu"]
    document["textures"] = [{"extensions": {"KHR_texture_basisu": {"source": 0}}}]
    document["images"] = [{"mimeType": "image/ktx2", "uri": "data:image/ktx2;base64,"}]
    output.write_text(json.dumps(document), encoding="utf-8")
    image_bytes = BytesIO()
    Image.new("RGBA", (2, 2), (40, 50, 60, 255)).save(image_bytes, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")

    def fake_ktxdecompress(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("KTX-Software ktx command not found")

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            '"textured_primitives":0,"sampled_textures":0,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime._run_gltf_transform_ktxdecompress", fake_ktxdecompress)
    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered_partial"
    assert report.unsupported_extensions == ("KHR_texture_basisu",)
    assert "could not decode KHR_texture_basisu" in str(report.error)
    assert "KTX2/Basis texture sampling" in str(report.error)
    assert report.to_dict()["unsupported_extensions"] == ["KHR_texture_basisu"]
    assert Image.open(preview).getpixel((0, 0)) == (40, 50, 60, 255)


def test_browser_render_preview_decodes_ktx2_textures(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["extensionsUsed"] = ["KHR_texture_basisu"]
    document["extensionsRequired"] = ["KHR_texture_basisu"]
    document["textures"] = [{"extensions": {"KHR_texture_basisu": {"source": 0}}}]
    document["images"] = [{"mimeType": "image/ktx2", "uri": "data:image/ktx2;base64,"}]
    output.write_text(json.dumps(document), encoding="utf-8")
    image_bytes = BytesIO()
    Image.new("RGBA", (2, 2), (90, 100, 110, 255)).save(image_bytes, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(image_bytes.getvalue()).decode("ascii")

    def fake_ktxdecompress(input_path: Path, output_path: Path) -> None:
        assert input_path == output
        assert output_path.name == "ktx2-decoded.glb"
        _asset().write_gltf(output_path)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(urlparse(command[-1]).path)
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(unquote(urlparse(decoded_url).path))
        assert decoded_path.name == "ktx2-decoded.glb"
        assert validate_gltf(decoded_path)["triangles"] == 1
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            '"textured_primitives":1,"sampled_textures":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime._run_gltf_transform_ktxdecompress", fake_ktxdecompress)
    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered"
    assert report.decoded_extensions == ("KHR_texture_basisu",)
    assert report.unsupported_extensions == ()
    assert report.preview_limitations == ()
    assert report.textured_primitives == 1
    assert report.sampled_textures == 1
    assert Image.open(preview).getpixel((0, 0)) == (90, 100, 110, 255)


def test_browser_render_preview_decodes_ktx2_textures_with_optional_python_decoder(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    _asset().write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    ktx2_payload = b"fake-ktx2"
    document["extensionsUsed"] = ["KHR_texture_basisu"]
    document["extensionsRequired"] = ["KHR_texture_basisu"]
    document["textures"] = [{"extensions": {"KHR_texture_basisu": {"source": 0}}}]
    document["images"] = [
        {"mimeType": "image/ktx2", "uri": "data:image/ktx2;base64," + base64.b64encode(ktx2_payload).decode("ascii")}
    ]
    output.write_text(json.dumps(document), encoding="utf-8")
    decoded_image = BytesIO()
    Image.new("RGBA", (2, 2), (95, 105, 115, 255)).save(decoded_image, format="PNG")
    screenshot = BytesIO()
    Image.new("RGBA", (2, 2), (100, 110, 120, 255)).save(screenshot, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(screenshot.getvalue()).decode("ascii")

    class FakeAlktx2:
        @staticmethod
        def decode_ktx2_to_bytes(data: bytes, *, format: str) -> tuple[bytes, str]:
            assert data == ktx2_payload
            assert format == "png"
            return decoded_image.getvalue(), "image/png"

    def fail_ktxdecompress(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external KTX-Software fallback should not run when alktx2 is available")

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(urlparse(command[-1]).path)
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(unquote(urlparse(decoded_url).path))
        assert decoded_path.name == "ktx2-decoded.gltf"
        decoded_document = json.loads(decoded_path.read_text(encoding="utf-8"))
        assert "KHR_texture_basisu" not in decoded_document.get("extensionsUsed", [])
        assert "KHR_texture_basisu" not in decoded_document.get("extensionsRequired", [])
        assert decoded_document["textures"][0]["source"] == 0
        assert decoded_document["images"][0]["mimeType"] == "image/png"
        assert decoded_document["images"][0]["uri"].startswith("data:image/png;base64,")
        assert validate_gltf(decoded_path)["triangles"] == 1
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            '"textured_primitives":1,"sampled_textures":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setitem(sys.modules, "alktx2", FakeAlktx2)
    monkeypatch.setattr("fascat.runtime._run_gltf_transform_ktxdecompress", fail_ktxdecompress)
    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered"
    assert report.decoded_extensions == ("KHR_texture_basisu",)
    assert report.unsupported_extensions == ()
    assert report.preview_limitations == ()
    assert report.textured_primitives == 1
    assert report.sampled_textures == 1
    assert Image.open(preview).getpixel((0, 0)) == (100, 110, 120, 255)


def test_browser_render_preview_decodes_bundled_ktx2_with_default_python_decoder(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    pytest.importorskip("alktx2")
    suite = write_runtime_parity_suite(tmp_path / "runtime-parity")
    fixture = next(item for item in suite.fixtures if item.name == "ktx2-basis-fallback")
    preview = tmp_path / "browser-preview.png"
    screenshot = BytesIO()
    Image.new("RGBA", (2, 2), (115, 125, 135, 255)).save(screenshot, format="PNG")
    screenshot_data = "data:image/png;base64," + base64.b64encode(screenshot.getvalue()).decode("ascii")

    def fail_ktxdecompress(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external KTX-Software fallback should not run when the default Python decoder is present")

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(urlparse(command[-1]).path)
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(unquote(urlparse(decoded_url).path))
        assert decoded_path.name == "ktx2-decoded.gltf"
        decoded_document = json.loads(decoded_path.read_text(encoding="utf-8"))
        assert "KHR_texture_basisu" not in decoded_document.get("extensionsUsed", [])
        assert "KHR_texture_basisu" not in decoded_document.get("extensionsRequired", [])
        texture_source = decoded_document["textures"][0]["source"]
        assert decoded_document["images"][texture_source]["mimeType"] == "image/png"
        assert decoded_document["images"][texture_source]["uri"].startswith("data:image/png;base64,")
        assert validate_gltf(decoded_path)["triangles"] == fixture.triangles
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":8,'
            '"textured_primitives":1,"sampled_textures":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime._run_gltf_transform_ktxdecompress", fail_ktxdecompress)
    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = write_browser_render_preview(
        fixture.asset_path,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert report.status == "rendered"
    assert report.decoded_extensions == ("KHR_texture_basisu",)
    assert report.unsupported_extensions == ()
    assert report.preview_limitations == ()
    assert report.textured_primitives == 1
    assert report.sampled_textures == 1
    assert Image.open(preview).getpixel((0, 0)) == (115, 125, 135, 255)


def test_supported_platforms_install_ktx2_python_decoder_by_default() -> None:
    metadata = tomli.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    default_ktx2 = next(item for item in dependencies if item.startswith("alktx2"))

    assert "python_version >= '3.11'" in default_ktx2
    assert "sys_platform == 'linux'" in default_ktx2
    assert "sys_platform == 'win32'" in default_ktx2
    assert "platform_machine" in default_ktx2


def test_browser_render_preview_harness_samples_base_color_textures(tmp_path: Path) -> None:
    html = _runtime_browser_render_html(
        tmp_path / "asset.glb",
        RuntimeBrowserRenderOptions(browser="fake-browser"),
    )

    assert "baseColorTexture" in html
    assert "TEXCOORD_0" in html
    assert "texture2D(baseColorTexture" in html
    assert "textured_primitives" in html
    assert "sampled_textures" in html
    assert "screenshot_data" in html
    assert "preserveDrawingBuffer" in html


def test_browser_render_preview_harness_accepts_quantized_accessors(tmp_path: Path) -> None:
    html = _runtime_browser_render_html(
        tmp_path / "asset.glb",
        RuntimeBrowserRenderOptions(browser="fake-browser"),
    )

    assert "5120: Int8Array" in html
    assert "5122: Int16Array" in html
    assert "VERTEX_ATTRIBUTE_COMPONENT_TYPES" in html
    assert "accessorComponentValue(draw.position" in html
    assert "draw.position.componentType, draw.position.normalized" in html
    assert "draw.texcoord.componentType, draw.texcoord.normalized" in html
    assert "quantized_primitives" in html
    assert "FLOAT or quantized VEC3 positions" in html


def test_engine_runtime_reports_unavailable_when_engine_is_missing(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    project = tmp_path / "UnityProject"
    project.mkdir()
    _asset().write_gltf(output)
    monkeypatch.delenv("FASCAT_UNITY", raising=False)
    monkeypatch.delenv("UNITY_EDITOR", raising=False)
    monkeypatch.setattr("fascat.runtime.shutil.which", lambda _name: None)

    report = measure_engine_runtime(output, RuntimeEngineOptions(engine="unity", project=project))

    assert report.status == "unavailable"
    assert report.engine == "unity"
    assert report.executable is None
    assert report.project == str(project)
    assert report.meshes == 1
    assert report.triangles == 1
    assert "no configured engine executable" in str(report.error)


def test_engine_runtime_parses_unity_harness_measurements(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "engine-preview.png"
    project = tmp_path / "UnityProject"
    project.mkdir()
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        report_path = Path(command[command.index("-fascatReport") + 1])
        assert command[command.index("-fascatPreview") + 1] == str(preview.resolve())
        assert "-nographics" not in command
        Image.new("RGBA", (4, 4), (20, 40, 60, 255)).save(preview)
        report_path.write_text(
            """
            {
              "status": "measured",
              "engine_version": "Unity 6000.0",
              "load_time_ms": 45,
              "measured_fps": 72.5,
              "frame_count": 145,
              "measurement_duration_ms": 2000,
              "memory_bytes": 8192,
              "meshes": 1,
              "triangles": 1,
              "render_status": "rendered",
              "render_time_ms": 16,
              "rendered_frames": 1,
              "render_backend": "unity_gltfast_camera",
              "render_limitations": []
            }
            """,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(
        output,
        RuntimeEngineOptions(
            engine="unity",
            executable="Unity",
            project=project,
            preview_path=preview,
            timeout_seconds=3.0,
        ),
    )

    assert report.status == "measured"
    assert report.engine == "unity"
    assert report.executable == "Unity"
    assert report.project == str(project)
    assert report.engine_version == "Unity 6000.0"
    assert report.load_time_ms == 45
    assert report.measured_fps == 72.5
    assert report.frame_count == 145
    assert report.memory_bytes == 8192
    assert report.preview_path == str(preview)
    assert report.render_status == "rendered"
    assert report.render_time_ms == 16
    assert report.rendered_frames == 1
    assert report.render_backend == "unity_gltfast_camera"
    assert report.render_limitations == ()
    assert report.to_dict()["render_status"] == "rendered"
    assert report.to_dict()["render_backend"] == "unity_gltfast_camera"


def test_copy_engine_runtime_harness_writes_unity_template(tmp_path: Path) -> None:
    project = copy_engine_runtime_harness("unity", tmp_path / "UnityHarness")
    harness = (project / "Assets" / "Editor" / "FascatRuntimeHarness.cs").read_text(encoding="utf-8")
    manifest = json.loads((project / "Packages" / "manifest.json").read_text(encoding="utf-8"))

    assert project == tmp_path / "UnityHarness"
    assert (project / "Assets" / "Editor" / "FascatRuntimeHarness.cs").is_file()
    assert (project / "Packages" / "manifest.json").is_file()
    assert (project / "ProjectSettings" / "ProjectVersion.txt").is_file()
    assert manifest["dependencies"]["com.unity.cloud.gltfast"] == "6.19.0"
    assert "-fascatPreview" in harness
    assert "using GLTFast;" in harness
    assert "new GltfImport()" in harness
    assert "InstantiateMainSceneAsync" in harness
    assert "RenderTexture" in harness
    assert "EncodeToPNG" in harness
    assert "PreviewBenchmarkFrames = 30" in harness
    assert "measured_fps" in harness
    assert "MeasuredFps" in harness
    assert "render_backend" in harness
    assert "unity_gltfast_camera" in harness
    assert "render_limitations" in harness


def test_copy_engine_runtime_harness_writes_unreal_template(tmp_path: Path) -> None:
    project = copy_engine_runtime_harness("unreal", tmp_path / "UnrealHarness")

    assert project == tmp_path / "UnrealHarness" / "FascatUnrealHarness.uproject"
    assert project.is_file()
    assert (project.parent / "Plugins" / "FascatRuntimeHarness" / "FascatRuntimeHarness.uplugin").is_file()
    assert (
        project.parent
        / "Plugins"
        / "FascatRuntimeHarness"
        / "Source"
        / "FascatRuntimeHarness"
        / "Public"
        / "FascatRuntimeHarnessCommandlet.h"
    ).is_file()
    commandlet = (
        project.parent
        / "Plugins"
        / "FascatRuntimeHarness"
        / "Source"
        / "FascatRuntimeHarness"
        / "Private"
        / "FascatRuntimeHarnessCommandlet.cpp"
    )
    build_file = (
        project.parent
        / "Plugins"
        / "FascatRuntimeHarness"
        / "Source"
        / "FascatRuntimeHarness"
        / "FascatRuntimeHarness.Build.cs"
    )
    commandlet_text = commandlet.read_text(encoding="utf-8")
    build_text = build_file.read_text(encoding="utf-8")

    assert "FascatPreview=" in commandlet_text
    assert "PreviewBenchmarkFrames = 30" in commandlet_text
    assert "RenderPreview" in commandlet_text
    assert "ReadPreviewGeometry" in commandlet_text
    assert "DrawGeometryPreviewFrame" in commandlet_text
    assert "unreal_commandlet_geometry_rasterizer" in commandlet_text
    assert "unreal_commandlet_count_preview" in commandlet_text
    assert "rendered_partial" in commandlet_text
    assert "FImageUtils::CompressImageArray" in commandlet_text
    assert "measured_fps" in commandlet_text
    assert "render_status" in commandlet_text
    assert "render_backend" in commandlet_text
    assert "render_limitations" in commandlet_text
    assert '"ImageWrapper"' in build_text


def test_engine_runtime_uses_packaged_unity_harness_when_project_is_omitted(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        project = Path(command[command.index("-projectPath") + 1])
        assert "-nographics" in command
        assert (project / "Assets" / "Editor" / "FascatRuntimeHarness.cs").is_file()
        assert (project / "Packages" / "manifest.json").is_file()
        report_path = Path(command[command.index("-fascatReport") + 1])
        report_path.write_text(
            '{"status":"measured","engine_version":"Unity 6000.0","load_time_ms":33,"meshes":1,"triangles":1}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(output, RuntimeEngineOptions(engine="unity", executable="Unity"))

    assert report.status == "measured"
    assert report.project is not None
    assert report.project.endswith("/harness")
    assert report.load_time_ms == 33
    assert report.meshes == 1
    assert report.triangles == 1


def test_engine_runtime_parses_unreal_stdout_measurements(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    project = tmp_path / "Harness.uproject"
    project.write_text("{}", encoding="utf-8")
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "-run=FascatRuntimeHarness" in command
        assert f"-FascatPreview={tmp_path / 'preview.png'}" not in command
        stdout = '{"status":"measured","engine_version":"Unreal 5.4","load_time_ms":55,"measured_fps":90}'
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(
        output,
        RuntimeEngineOptions(engine="unreal", executable="UnrealEditor-Cmd", project=project),
    )

    assert report.status == "measured"
    assert report.engine == "unreal"
    assert report.engine_version == "Unreal 5.4"
    assert report.load_time_ms == 55
    assert report.measured_fps == 90.0
    assert report.meshes == 1
    assert report.triangles == 1
    assert report.render_status == "not_requested"


def test_engine_runtime_reports_requested_preview_when_custom_harness_writes_file(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "nested" / "preview.png"
    project = tmp_path / "Harness.uproject"
    project.write_text("{}", encoding="utf-8")
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert f"-FascatPreview={preview.resolve()}" in command
        Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(preview)
        stdout = '{"status":"measured","engine_version":"Unreal 5.4","load_time_ms":55}'
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(
        output,
        RuntimeEngineOptions(
            engine="unreal",
            executable="UnrealEditor-Cmd",
            project=project,
            preview_path=preview,
        ),
    )

    assert report.status == "measured"
    assert report.preview_path == str(preview)
    assert report.render_status == "rendered"
    assert report.rendered_frames == 0
    assert Image.open(preview).getpixel((0, 0)) == (1, 2, 3, 255)


def test_engine_runtime_uses_packaged_unreal_harness_when_project_is_omitted(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        project = Path(command[1])
        assert project.name == "FascatUnrealHarness.uproject"
        assert (project.parent / "Plugins" / "FascatRuntimeHarness" / "FascatRuntimeHarness.uplugin").is_file()
        report_arg = next(item for item in command if item.startswith("-FascatReport="))
        report_path = Path(report_arg.split("=", 1)[1])
        report_path.write_text(
            '{"status":"measured","engine_version":"Unreal 5.4","load_time_ms":44,"meshes":1,"triangles":1}',
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(output, RuntimeEngineOptions(engine="unreal", executable="UnrealEditor-Cmd"))

    assert report.status == "measured"
    assert report.project is not None
    assert report.project.endswith("/harness/FascatUnrealHarness.uproject")
    assert report.load_time_ms == 44
    assert report.meshes == 1
    assert report.triangles == 1


def test_engine_runtime_uses_packaged_unreal_harness_preview_when_project_is_omitted(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "unreal-preview.png"
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        project = Path(command[1])
        assert project.name == "FascatUnrealHarness.uproject"
        assert f"-FascatPreview={preview.resolve()}" in command
        commandlet = (
            project.parent
            / "Plugins"
            / "FascatRuntimeHarness"
            / "Source"
            / "FascatRuntimeHarness"
            / "Private"
            / "FascatRuntimeHarnessCommandlet.cpp"
        )
        assert "RenderPreview" in commandlet.read_text(encoding="utf-8")
        report_arg = next(item for item in command if item.startswith("-FascatReport="))
        report_path = Path(report_arg.split("=", 1)[1])
        Image.new("RGBA", (4, 4), (30, 60, 90, 255)).save(preview)
        report_path.write_text(
            json.dumps(
                {
                    "status": "measured",
                    "engine_version": "Unreal 5.4",
                    "load_time_ms": 44,
                    "measured_fps": 61.5,
                    "frame_count": 30,
                    "measurement_duration_ms": 488,
                    "meshes": 1,
                    "triangles": 1,
                    "render_status": "rendered",
                    "render_time_ms": 20,
                    "rendered_frames": 30,
                    "render_backend": "unreal_commandlet_geometry_rasterizer",
                    "render_limitations": [
                        "packaged Unreal commandlet rasterizes GLB triangle geometry and baseColorFactor materials; it is not a full Unreal scene renderer"
                    ],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(
        output,
        RuntimeEngineOptions(engine="unreal", executable="UnrealEditor-Cmd", preview_path=preview),
    )

    assert report.status == "measured"
    assert report.project is not None
    assert report.project.endswith("/harness/FascatUnrealHarness.uproject")
    assert report.preview_path == str(preview)
    assert report.render_status == "rendered"
    assert report.render_time_ms == 20
    assert report.rendered_frames == 30
    assert report.measured_fps == 61.5
    assert report.frame_count == 30
    assert report.measurement_duration_ms == 488
    assert report.render_backend == "unreal_commandlet_geometry_rasterizer"
    assert report.render_limitations == (
        "packaged Unreal commandlet rasterizes GLB triangle geometry and baseColorFactor materials; it is not a full Unreal scene renderer",
    )
    assert report.render_error is None
    assert Image.open(preview).getpixel((0, 0)) == (30, 60, 90, 255)


def _asset() -> Asset:
    mesh = Mesh(
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    return Asset(
        root=Node(id="root", name="Root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    )
