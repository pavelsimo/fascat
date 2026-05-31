from __future__ import annotations

import base64
import json
import subprocess
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
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
              "rendered_frames": 1
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
    assert report.to_dict()["render_status"] == "rendered"


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
    assert "FascatPreview=" in commandlet.read_text(encoding="utf-8")


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
