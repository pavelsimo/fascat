from __future__ import annotations

import base64
import json
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

import pytest
import tomli
from PIL import Image

from fascat.io.gltf import validate_gltf
from fascat.options import GltfExportOptions
from fascat.runtime import (
    RuntimeBrowserRenderOptions,
    write_browser_render_preview,
)
from fascat.runtime.html import _runtime_browser_render_html
from fascat.runtime.preview import _preview_document_copy
from fascat.runtime_fixtures import write_runtime_parity_suite

from ._runtime_helpers import runtime_asset


def test_browser_render_preview_reports_unavailable_when_browser_is_missing(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "preview.png"
    runtime_asset().write_gltf(output)
    monkeypatch.delenv("FASCAT_BROWSER", raising=False)
    monkeypatch.setattr("fascat.runtime.browser.shutil.which", lambda _name: None)

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
    runtime_asset().write_gltf(output)

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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)
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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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


@pytest.mark.parametrize(
    ("helper_name", "operation"),
    [
        ("_run_gltf_transform_copy", "copy"),
        ("_run_gltf_transform_ktxdecompress", "ktxdecompress"),
    ],
)
def test_gltf_transform_runtime_helpers_report_missing_cli_before_subprocess(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    helper_name: str,
    operation: str,
) -> None:
    import fascat.runtime.preview as runtime

    monkeypatch.delenv("FASCAT_GLTF_TRANSFORM", raising=False)
    monkeypatch.setattr("fascat.io.gltf.shutil.which", lambda _name: None)

    def fail_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("subprocess should not run when glTF Transform is missing")

    monkeypatch.setattr("fascat._subprocess.run_guarded", fail_run)
    helper = getattr(runtime, helper_name)

    with pytest.raises(RuntimeError, match=f"glTF Transform {operation} requires the glTF Transform CLI"):
        helper(tmp_path / "input.gltf", tmp_path / "output.gltf")


def test_browser_render_preview_falls_back_when_screenshot_data_exceeds_cap(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "browser-preview.png"
    runtime_asset().write_gltf(output)
    oversized_screenshot_data = "data:image/png;base64," + ("A" * 64)
    calls: list[list[str]] = []

    monkeypatch.setattr("fascat.runtime.preview._MAX_BROWSER_RENDER_SCREENSHOT_DATA_URI_LENGTH", 40)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        screenshot_args = [item for item in command if item.startswith("--screenshot=")]
        if screenshot_args:
            screenshot_path = Path(screenshot_args[0].split("=", 1)[1])
            Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(screenshot_path)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            '"textured_primitives":0,"sampled_textures":0,'
            '"quantized_primitives":0,'
            f'"screenshot_data":"{oversized_screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

    report = write_browser_render_preview(
        output,
        preview,
        RuntimeBrowserRenderOptions(browser="fake-browser", timeout_seconds=3.0),
    )

    assert len(calls) == 2
    assert any(item.startswith("--screenshot=") for item in calls[1])
    assert report.status == "rendered"
    assert Image.open(preview).getpixel((0, 0)) == (10, 20, 30, 255)


def test_browser_render_preview_reports_unsupported_draco_decode_failure_without_running_browser(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.gltf"
    preview = tmp_path / "browser-preview.png"
    runtime_asset().write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    document["extensionsUsed"] = ["KHR_draco_mesh_compression"]
    document["extensionsRequired"] = ["KHR_draco_mesh_compression"]
    document["meshes"][0]["primitives"][0]["extensions"] = {"KHR_draco_mesh_compression": {"bufferView": 0}}
    output.write_text(json.dumps(document), encoding="utf-8")

    def fake_copy(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("glTF Transform copy failed")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise AssertionError("browser should not run for unsupported Draco preview")

    monkeypatch.setattr("fascat.runtime.preview._run_gltf_transform_copy", fake_copy)
    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)
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
        runtime_asset().write_gltf(output_path)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(url2pathname(urlparse(command[-1]).path))
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(url2pathname(urlparse(decoded_url).path))
        assert decoded_path.name == "draco-decoded.glb"
        assert validate_gltf(decoded_path)["triangles"] == 1
        stdout = (
            '<html><body><pre id="result">'
            '{"status":"rendered","meshes":1,"triangles":1,'
            f'"screenshot_data":"{screenshot_data}"'
            "}</pre></body></html>"
        )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat.runtime.preview._run_gltf_transform_copy", fake_copy)
    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output, options=GltfExportOptions(meshopt=True))
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
        harness_path = Path(url2pathname(urlparse(command[-1]).path))
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(url2pathname(urlparse(decoded_url).path))
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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)
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

    monkeypatch.setattr("fascat.runtime.preview._run_gltf_transform_ktxdecompress", fake_ktxdecompress)
    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)
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
        runtime_asset().write_gltf(output_path)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--dump-dom" in command
        harness_path = Path(url2pathname(urlparse(command[-1]).path))
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(url2pathname(urlparse(decoded_url).path))
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

    monkeypatch.setattr("fascat.runtime.preview._run_gltf_transform_ktxdecompress", fake_ktxdecompress)
    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)
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
        harness_path = Path(url2pathname(urlparse(command[-1]).path))
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(url2pathname(urlparse(decoded_url).path))
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
    monkeypatch.setattr("fascat.runtime.preview._run_gltf_transform_ktxdecompress", fail_ktxdecompress)
    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
        harness_path = Path(url2pathname(urlparse(command[-1]).path))
        harness = harness_path.read_text(encoding="utf-8")
        match = re.search(r"const ASSET_URL = (?P<value>.*?);", harness)
        assert match is not None
        decoded_url = json.loads(match.group("value"))
        decoded_path = Path(url2pathname(urlparse(decoded_url).path))
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

    monkeypatch.setattr("fascat.runtime.preview._run_gltf_transform_ktxdecompress", fail_ktxdecompress)
    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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


def test_preview_document_copy_only_copies_mutated_containers() -> None:
    document = {
        "extensionsUsed": ["KHR_texture_basisu"],
        "images": [{"uri": "texture.ktx2"}],
        "textures": [{"extensions": {"KHR_texture_basisu": {"source": 0}}}],
        "bufferViews": [{"extensions": {"EXT_meshopt_compression": {"buffer": 0}}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }

    copied = _preview_document_copy(
        document,
        images=True,
        textures=True,
        buffer_views=True,
        extension_lists=True,
    )

    assert copied is not document
    assert copied["images"] is not document["images"]
    assert copied["images"][0] is not document["images"][0]
    assert copied["textures"] is not document["textures"]
    assert copied["textures"][0] is not document["textures"][0]
    assert copied["textures"][0]["extensions"] is not document["textures"][0]["extensions"]
    assert copied["bufferViews"] is not document["bufferViews"]
    assert copied["bufferViews"][0] is not document["bufferViews"][0]
    assert copied["bufferViews"][0]["extensions"] is not document["bufferViews"][0]["extensions"]
    assert copied["extensionsUsed"] is not document["extensionsUsed"]
    assert copied["meshes"] is document["meshes"]


def test_ktx2_python_decoder_is_extra_only() -> None:
    metadata = tomli.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = metadata["project"]["dependencies"]
    extras = metadata["project"]["optional-dependencies"]["ktx2"]

    assert not any(item.startswith("alktx2") for item in dependencies)
    extra_ktx2 = next(item for item in extras if item.startswith("alktx2"))
    assert "python_version >= '3.11'" in extra_ktx2
    assert "sys_platform == 'linux'" in extra_ktx2
    assert "sys_platform == 'win32'" in extra_ktx2
    assert "platform_machine" in extra_ktx2


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
