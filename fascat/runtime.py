from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Literal, Protocol, cast

from fascat.io.gltf import validate_gltf

RuntimeEngineName = Literal["unity", "unreal"]
_RESULT_RE = re.compile(r'<pre id="result">(?P<payload>.*?)</pre>', re.DOTALL)
_BROWSER_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "msedge",
)
_ENGINE_ENV_VARS: dict[RuntimeEngineName, tuple[str, ...]] = {
    "unity": ("FASCAT_UNITY", "UNITY_EDITOR"),
    "unreal": ("FASCAT_UNREAL", "UNREAL_EDITOR"),
}
_ENGINE_CANDIDATES: dict[RuntimeEngineName, tuple[str, ...]] = {
    "unity": ("Unity", "unity", "unity-editor"),
    "unreal": ("UnrealEditor-Cmd", "UnrealEditor", "UE4Editor-Cmd", "UE4Editor"),
}


class _ResourceNode(Protocol):
    @property
    def name(self) -> str: ...

    def iterdir(self) -> Iterable[_ResourceNode]: ...

    def is_dir(self) -> bool: ...

    def read_bytes(self) -> bytes: ...


@dataclass(frozen=True)
class RuntimeBrowserOptions:
    browser: str | None = None
    width: int = 800
    height: int = 600
    warmup_seconds: float = 0.5
    duration_seconds: float = 2.0
    timeout_seconds: float = 15.0
    max_workload_triangles: int = 200_000

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("runtime browser viewport dimensions must be greater than 0")
        if self.warmup_seconds < 0.0:
            raise ValueError("runtime browser warmup_seconds must be greater than or equal to 0")
        if self.duration_seconds <= 0.0:
            raise ValueError("runtime browser duration_seconds must be greater than 0")
        if self.timeout_seconds <= 0.0:
            raise ValueError("runtime browser timeout_seconds must be greater than 0")
        if self.max_workload_triangles <= 0:
            raise ValueError("runtime browser max_workload_triangles must be greater than 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeBrowserReport:
    path: str
    status: str
    browser: str | None
    load_time_ms: int | None
    measured_fps: float | None
    frame_count: int
    measurement_duration_ms: int | None
    memory_bytes: int | None
    meshes: int
    triangles: int
    workload_triangles: int
    workload_scale: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "browser": self.browser,
            "load_time_ms": self.load_time_ms,
            "measured_fps": self.measured_fps,
            "frame_count": self.frame_count,
            "measurement_duration_ms": self.measurement_duration_ms,
            "memory_bytes": self.memory_bytes,
            "meshes": self.meshes,
            "triangles": self.triangles,
            "workload_triangles": self.workload_triangles,
            "workload_scale": self.workload_scale,
            "error": self.error,
        }


@dataclass(frozen=True)
class RuntimeEngineOptions:
    engine: RuntimeEngineName
    executable: str | None = None
    project: str | Path | None = None
    timeout_seconds: float = 120.0
    extra_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.engine not in {"unity", "unreal"}:
            raise ValueError("runtime engine must be one of: unity, unreal")
        if self.timeout_seconds <= 0.0:
            raise ValueError("runtime engine timeout_seconds must be greater than 0")
        if isinstance(self.extra_args, str) or not isinstance(self.extra_args, tuple):
            raise ValueError("runtime engine extra_args must be a tuple of strings")
        if any(not isinstance(item, str) or not item for item in self.extra_args):
            raise ValueError("runtime engine extra_args values must be non-empty strings")

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if self.project is not None:
            data["project"] = str(self.project)
        return data


@dataclass(frozen=True)
class RuntimeEngineReport:
    path: str
    status: str
    engine: RuntimeEngineName
    executable: str | None
    project: str | None
    engine_version: str | None
    load_time_ms: int | None
    measured_fps: float | None
    frame_count: int
    measurement_duration_ms: int | None
    memory_bytes: int | None
    meshes: int
    triangles: int
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "engine": self.engine,
            "executable": self.executable,
            "project": self.project,
            "engine_version": self.engine_version,
            "load_time_ms": self.load_time_ms,
            "measured_fps": self.measured_fps,
            "frame_count": self.frame_count,
            "measurement_duration_ms": self.measurement_duration_ms,
            "memory_bytes": self.memory_bytes,
            "meshes": self.meshes,
            "triangles": self.triangles,
            "error": self.error,
        }


def measure_browser_runtime(path: str | Path, options: RuntimeBrowserOptions | None = None) -> RuntimeBrowserReport:
    opts = options or RuntimeBrowserOptions()
    asset_path = Path(path)
    if asset_path.suffix.lower() not in {".gltf", ".glb"}:
        raise ValueError("browser runtime validation only supports glTF/GLB outputs")
    if not asset_path.exists():
        raise FileNotFoundError(f"missing runtime asset: {asset_path}")

    validation_stats = validate_gltf(asset_path)
    browser = _browser_command(opts)
    if browser is None:
        return _unavailable_report(asset_path, validation_stats, "no chromium-compatible browser executable found")

    with tempfile.TemporaryDirectory(prefix="fascat-runtime-") as directory:
        harness_path = Path(directory) / "runtime.html"
        harness_path.write_text(_runtime_harness_html(asset_path.resolve(), opts), encoding="utf-8")
        command = _browser_invocation(browser, harness_path, opts)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=opts.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return _failed_report(asset_path, validation_stats, browser, "browser runtime validation timed out")
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        message = detail[-1] if detail else f"browser exited with status {completed.returncode}"
        return _failed_report(asset_path, validation_stats, browser, message)
    payload = _parse_browser_payload(completed.stdout)
    if payload is None:
        return _failed_report(asset_path, validation_stats, browser, "browser did not return runtime measurements")
    return _report_from_payload(asset_path, validation_stats, browser, payload)


def measure_engine_runtime(path: str | Path, options: RuntimeEngineOptions) -> RuntimeEngineReport:
    asset_path = Path(path)
    if asset_path.suffix.lower() not in {".gltf", ".glb"}:
        raise ValueError("engine runtime validation currently supports glTF/GLB outputs")
    if not asset_path.exists():
        raise FileNotFoundError(f"missing runtime asset: {asset_path}")

    validation_stats = validate_gltf(asset_path)
    executable = _engine_command(options)
    project = None if options.project is None else Path(options.project)
    if executable is None:
        return _engine_unavailable_report(
            asset_path,
            validation_stats,
            options,
            "no configured engine executable found",
            executable=None,
            project=project,
        )
    if project is not None and not project.exists():
        return _engine_unavailable_report(
            asset_path,
            validation_stats,
            options,
            f"runtime harness project not found: {project}",
            executable=executable,
            project=project,
        )

    with tempfile.TemporaryDirectory(prefix=f"fascat-{options.engine}-runtime-") as directory:
        if project is None:
            try:
                project = _copy_packaged_engine_harness(options.engine, Path(directory) / "harness")
            except OSError as exc:
                return _engine_unavailable_report(
                    asset_path,
                    validation_stats,
                    options,
                    f"packaged {options.engine} runtime harness could not be prepared: {exc}",
                    executable=executable,
                    project=None,
                )
        report_path = Path(directory) / "runtime-report.json"
        command = _engine_invocation(executable, project, asset_path, report_path, options)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=options.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return _engine_failed_report(
                asset_path,
                validation_stats,
                options,
                executable=executable,
                project=project,
                error=f"{options.engine} runtime validation timed out",
            )
        payload = _load_engine_payload(report_path, completed.stdout)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        message = detail[-1] if detail else f"{options.engine} exited with status {completed.returncode}"
        return _engine_failed_report(
            asset_path,
            validation_stats,
            options,
            executable=executable,
            project=project,
            error=message,
        )
    if payload is None:
        return _engine_failed_report(
            asset_path,
            validation_stats,
            options,
            executable=executable,
            project=project,
            error=f"{options.engine} harness did not return runtime measurements",
        )
    return _engine_report_from_payload(
        asset_path,
        validation_stats,
        options,
        executable=executable,
        project=project,
        payload=payload,
    )


def copy_engine_runtime_harness(engine: RuntimeEngineName, destination: str | Path) -> Path:
    """Copy the packaged Unity or Unreal runtime harness template to a project path."""

    if engine not in {"unity", "unreal"}:
        raise ValueError("runtime engine must be one of: unity, unreal")
    return _copy_packaged_engine_harness(engine, Path(destination))


def _copy_packaged_engine_harness(engine: RuntimeEngineName, destination: Path) -> Path:
    template = resources.files("fascat").joinpath("runtime_harnesses").joinpath(engine)
    if not template.is_dir():
        raise FileNotFoundError(f"packaged {engine} runtime harness template is missing")
    _copy_resource_tree(template, destination)
    if engine == "unreal":
        return destination / "FascatUnrealHarness.uproject"
    return destination


def _copy_resource_tree(source: _ResourceNode, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            _copy_resource_tree(child, target)
        else:
            target.write_bytes(child.read_bytes())


def _browser_command(options: RuntimeBrowserOptions) -> str | None:
    if options.browser:
        return options.browser
    env_browser = os.environ.get("FASCAT_BROWSER")
    if env_browser:
        return env_browser
    for candidate in _BROWSER_CANDIDATES:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _engine_command(options: RuntimeEngineOptions) -> str | None:
    if options.executable:
        return options.executable
    for env_name in _ENGINE_ENV_VARS[options.engine]:
        value = os.environ.get(env_name)
        if value:
            return value
    for candidate in _ENGINE_CANDIDATES[options.engine]:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _engine_invocation(
    executable: str,
    project: Path,
    asset_path: Path,
    report_path: Path,
    options: RuntimeEngineOptions,
) -> list[str]:
    asset = str(asset_path.resolve())
    report = str(report_path.resolve())
    if options.engine == "unity":
        return [
            executable,
            "-batchmode",
            "-nographics",
            "-quit",
            "-projectPath",
            str(project),
            "-executeMethod",
            "FascatRuntimeHarness.Run",
            "-fascatAsset",
            asset,
            "-fascatReport",
            report,
            *options.extra_args,
        ]
    return [
        executable,
        str(project),
        "-run=FascatRuntimeHarness",
        f"-FascatAsset={asset}",
        f"-FascatReport={report}",
        "-unattended",
        "-nosplash",
        *options.extra_args,
    ]


def _load_engine_payload(report_path: Path, output: str) -> dict[str, object] | None:
    if report_path.exists():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return cast(dict[str, object], loaded) if isinstance(loaded, dict) else None
    for line in reversed(output.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return cast(dict[str, object], loaded)
    return None


def _browser_invocation(browser: str, harness_path: Path, options: RuntimeBrowserOptions) -> list[str]:
    budget_ms = int((options.warmup_seconds + options.duration_seconds + 1.0) * 1000.0)
    return [
        browser,
        "--headless=new",
        "--allow-file-access-from-files",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
        "--use-gl=swiftshader",
        "--enable-unsafe-swiftshader",
        f"--window-size={options.width},{options.height}",
        f"--virtual-time-budget={budget_ms}",
        "--dump-dom",
        harness_path.resolve().as_uri(),
    ]


def _parse_browser_payload(output: str) -> dict[str, object] | None:
    match = _RESULT_RE.search(output)
    if match is None:
        return None
    try:
        payload = json.loads(html.unescape(match.group("payload")))
    except json.JSONDecodeError:
        return None
    return cast(dict[str, object], payload) if isinstance(payload, dict) else None


def _report_from_payload(
    asset_path: Path,
    validation_stats: dict[str, int],
    browser: str,
    payload: dict[str, object],
) -> RuntimeBrowserReport:
    triangles = _int(payload.get("triangles"), validation_stats["triangles"])
    workload_triangles = _int(payload.get("workload_triangles"), 0)
    status = str(payload.get("status", "failed"))
    error = payload.get("error")
    return RuntimeBrowserReport(
        path=str(asset_path),
        status=status,
        browser=browser,
        load_time_ms=_optional_int(payload.get("load_time_ms")),
        measured_fps=_optional_float(payload.get("measured_fps")),
        frame_count=_int(payload.get("frame_count"), 0),
        measurement_duration_ms=_optional_int(payload.get("measurement_duration_ms")),
        memory_bytes=_optional_int(payload.get("memory_bytes")),
        meshes=_int(payload.get("meshes"), validation_stats["meshes"]),
        triangles=triangles,
        workload_triangles=workload_triangles,
        workload_scale=(workload_triangles / triangles) if triangles > 0 and workload_triangles > 0 else 0.0,
        error=str(error) if error is not None else None,
    )


def _unavailable_report(asset_path: Path, validation_stats: dict[str, int], error: str) -> RuntimeBrowserReport:
    return RuntimeBrowserReport(
        path=str(asset_path),
        status="unavailable",
        browser=None,
        load_time_ms=None,
        measured_fps=None,
        frame_count=0,
        measurement_duration_ms=None,
        memory_bytes=None,
        meshes=validation_stats["meshes"],
        triangles=validation_stats["triangles"],
        workload_triangles=0,
        workload_scale=0.0,
        error=error,
    )


def _failed_report(
    asset_path: Path,
    validation_stats: dict[str, int],
    browser: str,
    error: str,
) -> RuntimeBrowserReport:
    return RuntimeBrowserReport(
        path=str(asset_path),
        status="failed",
        browser=browser,
        load_time_ms=None,
        measured_fps=None,
        frame_count=0,
        measurement_duration_ms=None,
        memory_bytes=None,
        meshes=validation_stats["meshes"],
        triangles=validation_stats["triangles"],
        workload_triangles=0,
        workload_scale=0.0,
        error=error,
    )


def _engine_report_from_payload(
    asset_path: Path,
    validation_stats: dict[str, int],
    options: RuntimeEngineOptions,
    *,
    executable: str,
    project: Path,
    payload: dict[str, object],
) -> RuntimeEngineReport:
    status = str(payload.get("status", "measured"))
    error = payload.get("error")
    engine_version = payload.get("engine_version")
    return RuntimeEngineReport(
        path=str(asset_path),
        status=status,
        engine=options.engine,
        executable=executable,
        project=str(project),
        engine_version=str(engine_version) if engine_version is not None else None,
        load_time_ms=_optional_int(payload.get("load_time_ms")),
        measured_fps=_optional_float(payload.get("measured_fps")),
        frame_count=_int(payload.get("frame_count"), 0),
        measurement_duration_ms=_optional_int(payload.get("measurement_duration_ms")),
        memory_bytes=_optional_int(payload.get("memory_bytes")),
        meshes=_int(payload.get("meshes"), validation_stats["meshes"]),
        triangles=_int(payload.get("triangles"), validation_stats["triangles"]),
        error=str(error) if error is not None else None,
    )


def _engine_unavailable_report(
    asset_path: Path,
    validation_stats: dict[str, int],
    options: RuntimeEngineOptions,
    error: str,
    *,
    executable: str | None,
    project: Path | None,
) -> RuntimeEngineReport:
    return RuntimeEngineReport(
        path=str(asset_path),
        status="unavailable",
        engine=options.engine,
        executable=executable,
        project=None if project is None else str(project),
        engine_version=None,
        load_time_ms=None,
        measured_fps=None,
        frame_count=0,
        measurement_duration_ms=None,
        memory_bytes=None,
        meshes=validation_stats["meshes"],
        triangles=validation_stats["triangles"],
        error=error,
    )


def _engine_failed_report(
    asset_path: Path,
    validation_stats: dict[str, int],
    options: RuntimeEngineOptions,
    *,
    executable: str,
    project: Path,
    error: str,
) -> RuntimeEngineReport:
    return RuntimeEngineReport(
        path=str(asset_path),
        status="failed",
        engine=options.engine,
        executable=executable,
        project=str(project),
        engine_version=None,
        load_time_ms=None,
        measured_fps=None,
        frame_count=0,
        measurement_duration_ms=None,
        memory_bytes=None,
        meshes=validation_stats["meshes"],
        triangles=validation_stats["triangles"],
        error=error,
    )


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, str | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: object, default: int) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _runtime_harness_html(asset_path: Path, options: RuntimeBrowserOptions) -> str:
    asset_url = json.dumps(asset_path.as_uri())
    warmup_ms = int(options.warmup_seconds * 1000.0)
    duration_ms = int(options.duration_seconds * 1000.0)
    max_workload_triangles = int(options.max_workload_triangles)
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>fascat runtime browser harness</title></head>
<body>
<canvas id="canvas" width="{options.width}" height="{options.height}"></canvas>
<pre id="result">{{"status":"running"}}</pre>
<script>
const ASSET_URL = {asset_url};
const WARMUP_MS = {warmup_ms};
const DURATION_MS = {duration_ms};
const MAX_WORKLOAD_TRIANGLES = {max_workload_triangles};
const result = document.getElementById("result");
const canvas = document.getElementById("canvas");

function finish(payload) {{
  result.textContent = JSON.stringify(payload);
  document.title = "fascat-runtime-done";
}}

function countTriangles(document) {{
  let triangles = 0;
  const meshes = Array.isArray(document.meshes) ? document.meshes : [];
  const accessors = Array.isArray(document.accessors) ? document.accessors : [];
  for (const mesh of meshes) {{
    const primitives = Array.isArray(mesh.primitives) ? mesh.primitives : [];
    for (const primitive of primitives) {{
      const mode = primitive.mode === undefined ? 4 : primitive.mode;
      if (mode !== 4) continue;
      if (primitive.indices !== undefined && accessors[primitive.indices]) {{
        triangles += Math.floor((accessors[primitive.indices].count || 0) / 3);
      }} else if (primitive.attributes && primitive.attributes.POSITION !== undefined) {{
        const accessor = accessors[primitive.attributes.POSITION];
        if (accessor) triangles += Math.floor((accessor.count || 0) / 3);
      }}
    }}
  }}
  return {{ meshes: meshes.length, triangles }};
}}

function parseGltf(buffer, url) {{
  const bytes = new Uint8Array(buffer);
  if (bytes.length >= 20 && bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46) {{
    const view = new DataView(buffer);
    const jsonLength = view.getUint32(12, true);
    const jsonType = view.getUint32(16, true);
    if (jsonType !== 0x4e4f534a) throw new Error("first GLB chunk is not JSON");
    const jsonBytes = bytes.slice(20, 20 + jsonLength);
    return JSON.parse(new TextDecoder("utf-8").decode(jsonBytes));
  }}
  return JSON.parse(new TextDecoder("utf-8").decode(bytes));
}}

function shader(gl, type, source) {{
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}}

function program(gl) {{
  const item = gl.createProgram();
  gl.attachShader(item, shader(gl, gl.VERTEX_SHADER, "attribute vec2 p; void main() {{ gl_Position = vec4(p, 0.0, 1.0); }}"));
  gl.attachShader(item, shader(gl, gl.FRAGMENT_SHADER, "precision mediump float; void main() {{ gl_FragColor = vec4(0.25, 0.55, 0.9, 1.0); }}"));
  gl.linkProgram(item);
  if (!gl.getProgramParameter(item, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(item));
  return item;
}}

function workloadVertices(triangles) {{
  const vertexCount = Math.max(3, triangles * 3);
  const data = new Float32Array(vertexCount * 2);
  for (let i = 0; i < triangles; i++) {{
    const x = ((i % 256) / 128.0) - 1.0;
    const y = ((Math.floor(i / 256) % 256) / 128.0) - 1.0;
    const base = i * 6;
    data[base] = x; data[base + 1] = y;
    data[base + 2] = x + 0.006; data[base + 3] = y;
    data[base + 4] = x; data[base + 5] = y + 0.006;
  }}
  return data;
}}

(async () => {{
  try {{
    const loadStart = performance.now();
    const response = await fetch(ASSET_URL);
    const buffer = await response.arrayBuffer();
    const loadTimeMs = performance.now() - loadStart;
    const gltf = parseGltf(buffer, ASSET_URL);
    const counts = countTriangles(gltf);
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL context unavailable");
    const drawTriangles = Math.max(1, Math.min(counts.triangles || 1, MAX_WORKLOAD_TRIANGLES));
    const vertices = workloadVertices(drawTriangles);
    const bufferObject = gl.createBuffer();
    const drawProgram = program(gl);
    const location = gl.getAttribLocation(drawProgram, "p");
    gl.bindBuffer(gl.ARRAY_BUFFER, bufferObject);
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
    gl.useProgram(drawProgram);
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0);
    let frames = 0;
    let measureStart = null;
    function frame(now) {{
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0.0, 0.0, 0.0, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, drawTriangles * 3);
      if (measureStart === null && now >= WARMUP_MS) {{
        measureStart = now;
        frames = 0;
      }}
      if (measureStart !== null) {{
        frames += 1;
        const elapsed = now - measureStart;
        if (elapsed >= DURATION_MS) {{
          finish({{
            status: "measured",
            load_time_ms: Math.round(loadTimeMs),
            measured_fps: frames * 1000.0 / Math.max(1.0, elapsed),
            frame_count: frames,
            measurement_duration_ms: Math.round(elapsed),
            memory_bytes: performance.memory ? performance.memory.usedJSHeapSize : buffer.byteLength + vertices.byteLength,
            meshes: counts.meshes,
            triangles: counts.triangles,
            workload_triangles: drawTriangles
          }});
          return;
        }}
      }}
      requestAnimationFrame(frame);
    }}
    requestAnimationFrame(frame);
  }} catch (error) {{
    finish({{ status: "failed", error: String(error), meshes: 0, triangles: 0, workload_triangles: 0 }});
  }}
}})();
</script>
</body>
</html>
"""
