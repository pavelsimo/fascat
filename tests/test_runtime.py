from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.runtime import (
    RuntimeBrowserOptions,
    RuntimeEngineOptions,
    copy_engine_runtime_harness,
    measure_browser_runtime,
    measure_engine_runtime,
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
    project = tmp_path / "UnityProject"
    project.mkdir()
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        report_path = Path(command[command.index("-fascatReport") + 1])
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
              "triangles": 1
            }
            """,
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("fascat.runtime.subprocess.run", fake_run)

    report = measure_engine_runtime(
        output,
        RuntimeEngineOptions(engine="unity", executable="Unity", project=project, timeout_seconds=3.0),
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


def test_copy_engine_runtime_harness_writes_unity_template(tmp_path: Path) -> None:
    project = copy_engine_runtime_harness("unity", tmp_path / "UnityHarness")

    assert project == tmp_path / "UnityHarness"
    assert (project / "Assets" / "Editor" / "FascatRuntimeHarness.cs").is_file()
    assert (project / "Packages" / "manifest.json").is_file()
    assert (project / "ProjectSettings" / "ProjectVersion.txt").is_file()


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


def test_engine_runtime_uses_packaged_unity_harness_when_project_is_omitted(
    monkeypatch,  # type: ignore[no-untyped-def]
    tmp_path: Path,
) -> None:
    output = tmp_path / "asset.glb"
    _asset().write_gltf(output)

    def fake_run(command: list[str], *_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        project = Path(command[command.index("-projectPath") + 1])
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
