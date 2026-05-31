from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.runtime import RuntimeBrowserOptions, measure_browser_runtime


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
