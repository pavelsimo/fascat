from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image

from fascat.runtime import (
    RuntimeEngineOptions,
    copy_engine_runtime_harness,
    measure_engine_runtime,
)

from ._runtime_helpers import runtime_asset


def test_engine_runtime_reports_unavailable_when_engine_is_missing(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    project = tmp_path / "UnityProject"
    project.mkdir()
    runtime_asset().write_gltf(output)
    monkeypatch.delenv("FASCAT_UNITY", raising=False)
    monkeypatch.delenv("UNITY_EDITOR", raising=False)
    monkeypatch.setattr("fascat.runtime.engine.shutil.which", lambda _name: None)

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
    runtime_asset().write_gltf(output)

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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)

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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

    report = measure_engine_runtime(output, RuntimeEngineOptions(engine="unity", executable="Unity"))

    assert report.status == "measured"
    assert report.project is not None
    assert Path(report.project).name == "harness"
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
    runtime_asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "-run=FascatRuntimeHarness" in command
        assert f"-FascatPreview={tmp_path / 'preview.png'}" not in command
        stdout = '{"status":"measured","engine_version":"Unreal 5.4","load_time_ms":55,"measured_fps":90}'
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert f"-FascatPreview={preview.resolve()}" in command
        Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(preview)
        stdout = '{"status":"measured","engine_version":"Unreal 5.4","load_time_ms":55}'
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

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
    runtime_asset().write_gltf(output)

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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

    report = measure_engine_runtime(output, RuntimeEngineOptions(engine="unreal", executable="UnrealEditor-Cmd"))

    assert report.status == "measured"
    assert report.project is not None
    assert Path(report.project).name == "FascatUnrealHarness.uproject"
    assert Path(report.project).parent.name == "harness"
    assert report.load_time_ms == 44
    assert report.meshes == 1
    assert report.triangles == 1


def test_engine_runtime_uses_packaged_unreal_harness_preview_when_project_is_omitted(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    preview = tmp_path / "unreal-preview.png"
    runtime_asset().write_gltf(output)

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

    monkeypatch.setattr("fascat._subprocess.run_guarded", fake_run)

    report = measure_engine_runtime(
        output,
        RuntimeEngineOptions(engine="unreal", executable="UnrealEditor-Cmd", preview_path=preview),
    )

    assert report.status == "measured"
    assert report.project is not None
    assert Path(report.project).name == "FascatUnrealHarness.uproject"
    assert Path(report.project).parent.name == "harness"
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
