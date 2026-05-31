from __future__ import annotations

import base64
import binascii
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
_RESULT_RE = re.compile(r'<pre id="result"[^>]*>(?P<payload>.*?)</pre>', re.DOTALL)
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
class RuntimeBrowserRenderOptions:
    browser: str | None = None
    width: int = 800
    height: int = 600
    timeout_seconds: float = 15.0
    virtual_time_budget_ms: int = 3000
    background_color: tuple[int, int, int, int] = (248, 249, 250, 255)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("browser render preview dimensions must be greater than 0")
        if self.timeout_seconds <= 0.0:
            raise ValueError("browser render preview timeout_seconds must be greater than 0")
        if self.virtual_time_budget_ms <= 0:
            raise ValueError("browser render preview virtual_time_budget_ms must be greater than 0")
        if len(self.background_color) != 4:
            raise ValueError("browser render preview background_color must contain RGBA byte values")
        if any(value < 0 or value > 255 for value in self.background_color):
            raise ValueError("browser render preview background_color values must be between 0 and 255")

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
class RuntimeBrowserRenderReport:
    path: str
    status: str
    browser: str | None
    preview_path: str
    width: int
    height: int
    meshes: int
    triangles: int
    textured_primitives: int = 0
    sampled_textures: int = 0
    quantized_primitives: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "status": self.status,
            "browser": self.browser,
            "preview_path": self.preview_path,
            "width": self.width,
            "height": self.height,
            "meshes": self.meshes,
            "triangles": self.triangles,
            "textured_primitives": self.textured_primitives,
            "sampled_textures": self.sampled_textures,
            "quantized_primitives": self.quantized_primitives,
            "error": self.error,
        }


@dataclass(frozen=True)
class RuntimeEngineOptions:
    engine: RuntimeEngineName
    executable: str | None = None
    project: str | Path | None = None
    preview_path: str | Path | None = None
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
        if self.preview_path is not None:
            data["preview_path"] = str(self.preview_path)
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
    preview_path: str | None = None
    render_status: str = "not_requested"
    render_time_ms: int | None = None
    rendered_frames: int = 0
    render_error: str | None = None
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
            "preview_path": self.preview_path,
            "render_status": self.render_status,
            "render_time_ms": self.render_time_ms,
            "rendered_frames": self.rendered_frames,
            "render_error": self.render_error,
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


def write_browser_render_preview(
    path: str | Path,
    preview_path: str | Path,
    options: RuntimeBrowserRenderOptions | None = None,
) -> RuntimeBrowserRenderReport:
    opts = options or RuntimeBrowserRenderOptions()
    asset_path = Path(path)
    if asset_path.suffix.lower() not in {".gltf", ".glb"}:
        raise ValueError("browser render preview only supports glTF/GLB outputs")
    if not asset_path.exists():
        raise FileNotFoundError(f"missing runtime asset: {asset_path}")

    validation_stats = validate_gltf(asset_path)
    output_path = Path(preview_path)
    browser = _browser_command(opts)
    if browser is None:
        return _browser_render_report(
            asset_path,
            output_path,
            opts,
            validation_stats,
            status="unavailable",
            browser=None,
            error="no chromium-compatible browser executable found",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fascat-browser-render-") as directory:
        harness_path = Path(directory) / "browser-render.html"
        harness_path.write_text(_runtime_browser_render_html(asset_path.resolve(), opts), encoding="utf-8")
        command = _browser_render_report_invocation(browser, harness_path, opts)
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=opts.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error="browser render preview timed out",
            )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip().splitlines()
            message = detail[-1] if detail else f"browser exited with status {completed.returncode}"
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error=message,
            )
        payload = _parse_browser_payload(completed.stdout)
        if payload is None:
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error="browser did not return render preview measurements",
            )
        status = str(payload.get("status", "failed"))
        error = payload.get("error")
        if status != "rendered":
            return RuntimeBrowserRenderReport(
                path=str(asset_path),
                status=status,
                browser=browser,
                preview_path=str(output_path),
                width=opts.width,
                height=opts.height,
                meshes=_int(payload.get("meshes"), validation_stats["meshes"]),
                triangles=_int(payload.get("triangles"), validation_stats["triangles"]),
                textured_primitives=_int(payload.get("textured_primitives"), 0),
                sampled_textures=_int(payload.get("sampled_textures"), 0),
                quantized_primitives=_int(payload.get("quantized_primitives"), 0),
                error=str(error) if error is not None else None,
            )
        screenshot_data = payload.get("screenshot_data")
        if not isinstance(screenshot_data, str) or not _write_png_data_uri(screenshot_data, output_path):
            screenshot_command = _browser_render_screenshot_invocation(browser, harness_path, output_path, opts)
            try:
                screenshot_completed = subprocess.run(
                    screenshot_command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=opts.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return _browser_render_report(
                    asset_path,
                    output_path,
                    opts,
                    validation_stats,
                    status="failed",
                    browser=browser,
                    error="browser render preview screenshot timed out",
                )
            if screenshot_completed.returncode != 0:
                detail = (screenshot_completed.stderr or screenshot_completed.stdout).strip().splitlines()
                message = detail[-1] if detail else f"browser exited with status {screenshot_completed.returncode}"
                return _browser_render_report(
                    asset_path,
                    output_path,
                    opts,
                    validation_stats,
                    status="failed",
                    browser=browser,
                    error=message,
                )
    if not output_path.exists():
        status = "failed"
        error = "browser did not write render preview screenshot"
    return RuntimeBrowserRenderReport(
        path=str(asset_path),
        status=status,
        browser=browser,
        preview_path=str(output_path),
        width=opts.width,
        height=opts.height,
        meshes=_int(payload.get("meshes"), validation_stats["meshes"]),
        triangles=_int(payload.get("triangles"), validation_stats["triangles"]),
        textured_primitives=_int(payload.get("textured_primitives"), 0),
        sampled_textures=_int(payload.get("sampled_textures"), 0),
        quantized_primitives=_int(payload.get("quantized_primitives"), 0),
        error=str(error) if error is not None else None,
    )


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
        preview_path = None if options.preview_path is None else Path(options.preview_path)
        if preview_path is not None:
            preview_path.parent.mkdir(parents=True, exist_ok=True)
        report_path = Path(directory) / "runtime-report.json"
        command = _engine_invocation(executable, project, asset_path, report_path, preview_path, options)
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


def _browser_command(options: RuntimeBrowserOptions | RuntimeBrowserRenderOptions) -> str | None:
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
    preview_path: Path | None,
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
            *(["-fascatPreview", str(preview_path.resolve())] if preview_path is not None else []),
            *options.extra_args,
        ]
    return [
        executable,
        str(project),
        "-run=FascatRuntimeHarness",
        f"-FascatAsset={asset}",
        f"-FascatReport={report}",
        *([f"-FascatPreview={preview_path.resolve()}"] if preview_path is not None else []),
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


def _browser_render_report_invocation(
    browser: str,
    harness_path: Path,
    options: RuntimeBrowserRenderOptions,
) -> list[str]:
    return [
        browser,
        "--headless=new",
        "--allow-file-access-from-files",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
        "--use-gl=swiftshader",
        "--enable-unsafe-swiftshader",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={options.width},{options.height}",
        f"--virtual-time-budget={options.virtual_time_budget_ms}",
        "--dump-dom",
        harness_path.resolve().as_uri(),
    ]


def _browser_render_screenshot_invocation(
    browser: str,
    harness_path: Path,
    preview_path: Path,
    options: RuntimeBrowserRenderOptions,
) -> list[str]:
    return [
        browser,
        "--headless=new",
        "--allow-file-access-from-files",
        "--disable-dev-shm-usage",
        "--no-sandbox",
        "--ignore-gpu-blocklist",
        "--use-gl=swiftshader",
        "--enable-unsafe-swiftshader",
        "--run-all-compositor-stages-before-draw",
        f"--window-size={options.width},{options.height}",
        f"--virtual-time-budget={options.virtual_time_budget_ms}",
        f"--screenshot={preview_path.resolve()}",
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


def _write_png_data_uri(value: str, path: Path) -> bool:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        return False
    try:
        data = base64.b64decode(value[len(prefix) :], validate=True)
    except (binascii.Error, ValueError):
        return False
    path.write_bytes(data)
    return True


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


def _browser_render_report(
    asset_path: Path,
    preview_path: Path,
    options: RuntimeBrowserRenderOptions,
    validation_stats: dict[str, int],
    *,
    status: str,
    browser: str | None,
    error: str | None,
) -> RuntimeBrowserRenderReport:
    return RuntimeBrowserRenderReport(
        path=str(asset_path),
        status=status,
        browser=browser,
        preview_path=str(preview_path),
        width=options.width,
        height=options.height,
        meshes=validation_stats["meshes"],
        triangles=validation_stats["triangles"],
        textured_primitives=0,
        sampled_textures=0,
        quantized_primitives=0,
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
    preview_path = None if options.preview_path is None else Path(options.preview_path)
    render_status = str(payload.get("render_status", "not_requested" if preview_path is None else "unavailable"))
    render_error = payload.get("render_error")
    if preview_path is not None:
        if preview_path.exists() and render_status in {"not_requested", "unavailable"}:
            render_status = "rendered"
        elif not preview_path.exists() and render_status == "rendered":
            render_status = "failed"
            render_error = "engine harness reported a rendered preview but did not write the requested PNG"
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
        preview_path=str(preview_path) if preview_path is not None else None,
        render_status=render_status,
        render_time_ms=_optional_int(payload.get("render_time_ms")),
        rendered_frames=_int(payload.get("rendered_frames"), 0),
        render_error=str(render_error) if render_error is not None else None,
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
        preview_path=None if options.preview_path is None else str(options.preview_path),
        render_status="not_requested" if options.preview_path is None else "unavailable",
        render_time_ms=None,
        rendered_frames=0,
        render_error=error if options.preview_path is not None else None,
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
        preview_path=None if options.preview_path is None else str(options.preview_path),
        render_status="not_requested" if options.preview_path is None else "failed",
        render_time_ms=None,
        rendered_frames=0,
        render_error=error if options.preview_path is not None else None,
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


def _runtime_browser_render_html(asset_path: Path, options: RuntimeBrowserRenderOptions) -> str:
    asset_url = json.dumps(asset_path.as_uri())
    background = json.dumps([round(channel / 255.0, 6) for channel in options.background_color])
    return f"""<!doctype html>
<html>
<head><meta charset="utf-8"><title>fascat browser render preview</title></head>
<body style="margin:0;overflow:hidden;background:transparent">
<canvas id="canvas" width="{options.width}" height="{options.height}"></canvas>
<pre id="result" style="display:none">{{"status":"running"}}</pre>
<script>
const ASSET_URL = {asset_url};
const BACKGROUND = {background};
const canvas = document.getElementById("canvas");
const result = document.getElementById("result");

function finish(payload) {{
  result.textContent = JSON.stringify(payload);
  document.title = "fascat-browser-render-done";
}}

function bytesToString(bytes) {{
  return new TextDecoder("utf-8").decode(bytes);
}}

function readUint32(bytes, offset) {{
  return new DataView(bytes.buffer, bytes.byteOffset + offset, 4).getUint32(0, true);
}}

function dataUriToBuffer(uri) {{
  const comma = uri.indexOf(",");
  if (comma < 0) throw new Error("invalid data URI buffer");
  const header = uri.slice(0, comma);
  const payload = uri.slice(comma + 1);
  if (header.includes(";base64")) {{
    const binary = atob(payload);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }}
  return new TextEncoder().encode(decodeURIComponent(payload)).buffer;
}}

function loadBufferUri(uri) {{
  const xhr = new XMLHttpRequest();
  xhr.open("GET", uri, false);
  xhr.overrideMimeType("text/plain; charset=x-user-defined");
  xhr.send(null);
  if (xhr.status !== 0 && (xhr.status < 200 || xhr.status >= 300)) throw new Error("failed to load " + uri);
  if (xhr.responseText === undefined) throw new Error("empty response for " + uri);
  const text = xhr.responseText;
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i) & 0xff;
  return bytes.buffer;
}}

function loadAsset() {{
  const buffer = loadBufferUri(ASSET_URL);
  const bytes = new Uint8Array(buffer);
  if (bytes.length >= 20 && bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46) {{
    const jsonLength = readUint32(bytes, 12);
    const jsonType = readUint32(bytes, 16);
    if (jsonType !== 0x4e4f534a) throw new Error("first GLB chunk is not JSON");
    const jsonStart = 20;
    const jsonEnd = jsonStart + jsonLength;
    const document = JSON.parse(bytesToString(bytes.slice(jsonStart, jsonEnd)));
    const buffers = [];
    let chunkOffset = jsonEnd + (jsonLength % 4 === 0 ? 0 : 4 - (jsonLength % 4));
    if (bytes.length >= chunkOffset + 8) {{
      const binaryLength = readUint32(bytes, chunkOffset);
      const binaryType = readUint32(bytes, chunkOffset + 4);
      if (binaryType === 0x004e4942) buffers[0] = bytes.slice(chunkOffset + 8, chunkOffset + 8 + binaryLength).buffer;
    }}
    return {{ document, buffers }};
  }}
  const document = JSON.parse(bytesToString(bytes));
  const buffers = [];
  const bufferDefs = Array.isArray(document.buffers) ? document.buffers : [];
  for (let i = 0; i < bufferDefs.length; i++) {{
    const uri = bufferDefs[i].uri;
    if (!uri) throw new Error("external glTF buffer is missing a URI");
    if (uri.startsWith("data:")) {{
      buffers[i] = dataUriToBuffer(uri);
    }} else {{
      buffers[i] = loadBufferUri(new URL(uri, ASSET_URL).href);
    }}
  }}
  return {{ document, buffers }};
}}

const COMPONENTS = {{ SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4 }};
const ARRAY_TYPES = {{ 5120: Int8Array, 5121: Uint8Array, 5122: Int16Array, 5123: Uint16Array, 5125: Uint32Array, 5126: Float32Array }};
const COMPONENT_BYTES = {{ 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 }};
const VERTEX_ATTRIBUTE_COMPONENT_TYPES = new Set([5120, 5121, 5122, 5123, 5126]);

function readAccessorComponent(view, byteOffset, componentType) {{
  if (componentType === 5120) return view.getInt8(byteOffset);
  if (componentType === 5121) return view.getUint8(byteOffset);
  if (componentType === 5122) return view.getInt16(byteOffset, true);
  if (componentType === 5123) return view.getUint16(byteOffset, true);
  if (componentType === 5125) return view.getUint32(byteOffset, true);
  if (componentType === 5126) return view.getFloat32(byteOffset, true);
  throw new Error("unsupported accessor component type " + componentType);
}}

function normalizedComponentValue(value, componentType) {{
  if (componentType === 5120) return Math.max(value / 127.0, -1.0);
  if (componentType === 5121) return value / 255.0;
  if (componentType === 5122) return Math.max(value / 32767.0, -1.0);
  if (componentType === 5123) return value / 65535.0;
  return value;
}}

function accessorComponentValue(accessor, row, column) {{
  const value = accessor.array[row * accessor.itemSize + column];
  return accessor.normalized ? normalizedComponentValue(value, accessor.componentType) : value;
}}

function readAccessor(document, buffers, accessorIndex) {{
  const accessor = document.accessors && document.accessors[accessorIndex];
  if (!accessor) throw new Error("missing accessor " + accessorIndex);
  if (accessor.sparse) throw new Error("sparse accessors are not supported by browser render preview");
  const view = document.bufferViews && document.bufferViews[accessor.bufferView];
  if (!view) throw new Error("missing bufferView for accessor " + accessorIndex);
  const buffer = buffers[view.buffer || 0];
  if (!buffer) throw new Error("missing buffer data for accessor " + accessorIndex);
  const itemSize = COMPONENTS[accessor.type] || 1;
  const ArrayType = ARRAY_TYPES[accessor.componentType];
  const componentBytes = COMPONENT_BYTES[accessor.componentType];
  if (!ArrayType || !componentBytes) throw new Error("unsupported accessor component type " + accessor.componentType);
  const byteOffset = (view.byteOffset || 0) + (accessor.byteOffset || 0);
  const stride = view.byteStride || (componentBytes * itemSize);
  const length = accessor.count * itemSize;
  if (stride === componentBytes * itemSize) {{
    return {{ array: new ArrayType(buffer, byteOffset, length), itemSize, count: accessor.count, componentType: accessor.componentType, normalized: !!accessor.normalized }};
  }}
  const sourceLength = accessor.count === 0 ? 0 : (accessor.count - 1) * stride + componentBytes * itemSize;
  const source = new DataView(buffer, byteOffset, sourceLength);
  const values = new ArrayType(length);
  for (let row = 0; row < accessor.count; row++) {{
    for (let column = 0; column < itemSize; column++) {{
      values[row * itemSize + column] = readAccessorComponent(source, row * stride + column * componentBytes, accessor.componentType);
    }}
  }}
  return {{ array: values, itemSize, count: accessor.count, componentType: accessor.componentType, normalized: !!accessor.normalized }};
}}

function identity() {{
  return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
}}

function multiply(a, b) {{
  const out = new Array(16).fill(0);
  for (let column = 0; column < 4; column++) {{
    for (let row = 0; row < 4; row++) {{
      out[column * 4 + row] =
        a[0 * 4 + row] * b[column * 4 + 0] +
        a[1 * 4 + row] * b[column * 4 + 1] +
        a[2 * 4 + row] * b[column * 4 + 2] +
        a[3 * 4 + row] * b[column * 4 + 3];
    }}
  }}
  return out;
}}

function nodeMatrix(node) {{
  if (Array.isArray(node.matrix) && node.matrix.length === 16) return node.matrix.slice();
  const t = node.translation || [0, 0, 0];
  const r = node.rotation || [0, 0, 0, 1];
  const s = node.scale || [1, 1, 1];
  const x = r[0], y = r[1], z = r[2], w = r[3];
  const x2 = x + x, y2 = y + y, z2 = z + z;
  const xx = x * x2, xy = x * y2, xz = x * z2;
  const yy = y * y2, yz = y * z2, zz = z * z2;
  const wx = w * x2, wy = w * y2, wz = w * z2;
  return [
    (1 - (yy + zz)) * s[0], (xy + wz) * s[0], (xz - wy) * s[0], 0,
    (xy - wz) * s[1], (1 - (xx + zz)) * s[1], (yz + wx) * s[1], 0,
    (xz + wy) * s[2], (yz - wx) * s[2], (1 - (xx + yy)) * s[2], 0,
    t[0], t[1], t[2], 1
  ];
}}

function transformPoint(m, p) {{
  return [
    m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12],
    m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13],
    m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]
  ];
}}

function subtract(a, b) {{ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }}
function cross(a, b) {{ return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }}
function dot(a, b) {{ return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }}
function normalize(v) {{
  const length = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / length, v[1] / length, v[2] / length];
}}

function lookAt(eye, center, up) {{
  const z = normalize(subtract(eye, center));
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1
  ];
}}

function perspective(fovy, aspect, near, far) {{
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return [f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0];
}}

function shader(gl, type, source) {{
  const item = gl.createShader(type);
  gl.shaderSource(item, source);
  gl.compileShader(item);
  if (!gl.getShaderParameter(item, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(item));
  return item;
}}

function program(gl) {{
  const vertexSource = [
    "attribute vec3 p;",
    "attribute vec2 uv;",
    "uniform mat4 mvp;",
    "varying vec2 vUv;",
    "void main() {{ vUv = uv; gl_Position = mvp * vec4(p, 1.0); }}"
  ].join("\\n");
  const fragmentSource = [
    "precision mediump float;",
    "uniform vec4 color;",
    "uniform sampler2D baseColorTexture;",
    "uniform bool useTexture;",
    "varying vec2 vUv;",
    "void main() {{ vec4 texel = useTexture ? texture2D(baseColorTexture, vUv) : vec4(1.0); gl_FragColor = color * texel; }}"
  ].join("\\n");
  const item = gl.createProgram();
  gl.attachShader(item, shader(gl, gl.VERTEX_SHADER, vertexSource));
  gl.attachShader(item, shader(gl, gl.FRAGMENT_SHADER, fragmentSource));
  gl.linkProgram(item);
  if (!gl.getProgramParameter(item, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(item));
  return item;
}}

function materialColor(document, materialIndex) {{
  const material = document.materials && document.materials[materialIndex];
  const factor = material && material.pbrMetallicRoughness && material.pbrMetallicRoughness.baseColorFactor;
  return Array.isArray(factor) ? factor : [0.45, 0.58, 0.72, 1.0];
}}

function materialBaseColorImageIndex(document, materialIndex) {{
  const material = document.materials && document.materials[materialIndex];
  const textureInfo = material && material.pbrMetallicRoughness && material.pbrMetallicRoughness.baseColorTexture;
  if (!textureInfo || textureInfo.index === undefined) return null;
  if (textureInfo.texCoord !== undefined && textureInfo.texCoord !== 0) return null;
  const texture = document.textures && document.textures[textureInfo.index];
  if (!texture || texture.source === undefined) return null;
  return texture.source;
}}

function imageUri(document, imageIndex) {{
  const image = document.images && document.images[imageIndex];
  if (!image || !image.uri) return null;
  if (image.uri.startsWith("data:")) return image.uri;
  return new URL(image.uri, ASSET_URL).href;
}}

async function loadImage(uri) {{
  if (typeof fetch === "function" && typeof createImageBitmap === "function") {{
    try {{
      const response = await fetch(uri);
      if (response.ok || uri.startsWith("data:") || uri.startsWith("file:")) {{
        return await createImageBitmap(await response.blob());
      }}
    }} catch (_error) {{
      // Fall back to Image for file URLs or older browser builds.
    }}
  }}
  return await new Promise((resolve, reject) => {{
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("failed to load texture image " + uri));
    image.src = uri;
  }});
}}

function createGlTexture(gl, image) {{
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
  const error = gl.getError();
  if (error !== gl.NO_ERROR) throw new Error("failed to upload texture image: WebGL error " + error);
  return texture;
}}

async function textureForMaterial(gl, document, materialIndex, cache) {{
  const imageIndex = materialBaseColorImageIndex(document, materialIndex);
  if (imageIndex === null) return null;
  if (cache.has(imageIndex)) return cache.get(imageIndex);
  const uri = imageUri(document, imageIndex);
  if (!uri) return null;
  const image = await loadImage(uri);
  const texture = createGlTexture(gl, image);
  cache.set(imageIndex, texture);
  return texture;
}}

function collectDraws(document, buffers) {{
  const draws = [];
  const nodes = Array.isArray(document.nodes) ? document.nodes : [];
  const scenes = Array.isArray(document.scenes) ? document.scenes : [];
  const scene = scenes[document.scene || 0] || scenes[0] || {{ nodes: nodes.map((_, index) => index) }};

  function addMesh(meshIndex, world) {{
    const mesh = document.meshes && document.meshes[meshIndex];
    if (!mesh || !Array.isArray(mesh.primitives)) return;
    for (const primitive of mesh.primitives) {{
      if ((primitive.mode === undefined ? 4 : primitive.mode) !== 4) continue;
      if (!primitive.attributes || primitive.attributes.POSITION === undefined) continue;
      const position = readAccessor(document, buffers, primitive.attributes.POSITION);
      if (!VERTEX_ATTRIBUTE_COMPONENT_TYPES.has(position.componentType) || position.itemSize !== 3) {{
        throw new Error("browser render preview currently supports FLOAT or quantized VEC3 positions");
      }}
      const texcoord = primitive.attributes.TEXCOORD_0 === undefined
        ? null
        : readAccessor(document, buffers, primitive.attributes.TEXCOORD_0);
      if (texcoord && (!VERTEX_ATTRIBUTE_COMPONENT_TYPES.has(texcoord.componentType) || texcoord.itemSize !== 2)) {{
        throw new Error("browser render preview currently supports FLOAT or quantized VEC2 TEXCOORD_0");
      }}
      const indices = primitive.indices === undefined ? null : readAccessor(document, buffers, primitive.indices);
      const triangles = indices ? Math.floor(indices.count / 3) : Math.floor(position.count / 3);
      draws.push({{
        position,
        indices,
        texcoord,
        material: primitive.material,
        texture: null,
        matrix: world,
        color: materialColor(document, primitive.material),
        quantized: position.componentType !== 5126 || position.normalized || (texcoord && (texcoord.componentType !== 5126 || texcoord.normalized)),
        triangles
      }});
    }}
  }}

  function walk(nodeIndex, parent) {{
    const node = nodes[nodeIndex];
    if (!node) return;
    const world = multiply(parent, nodeMatrix(node));
    if (node.mesh !== undefined) addMesh(node.mesh, world);
    if (Array.isArray(node.children)) for (const child of node.children) walk(child, world);
  }}

  const roots = Array.isArray(scene.nodes) ? scene.nodes : [];
  for (const root of roots) walk(root, identity());
  if (!roots.length && Array.isArray(document.meshes)) {{
    for (let index = 0; index < document.meshes.length; index++) addMesh(index, identity());
  }}
  return draws;
}}

function bounds(draws) {{
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const draw of draws) {{
    const values = draw.position.array;
    for (let i = 0; i < draw.position.count; i++) {{
      const p = transformPoint(draw.matrix, [
        accessorComponentValue(draw.position, i, 0),
        accessorComponentValue(draw.position, i, 1),
        accessorComponentValue(draw.position, i, 2)
      ]);
      for (let axis = 0; axis < 3; axis++) {{
        min[axis] = Math.min(min[axis], p[axis]);
        max[axis] = Math.max(max[axis], p[axis]);
      }}
    }}
  }}
  return {{ min, max }};
}}

(async () => {{
  try {{
    const loaded = loadAsset();
    const document = loaded.document;
    const draws = collectDraws(document, loaded.buffers);
    if (!draws.length) throw new Error("asset contains no renderable mesh primitives");
    const contextOptions = {{ preserveDrawingBuffer: true }};
    const gl = canvas.getContext("webgl2", contextOptions) || canvas.getContext("webgl", contextOptions);
    if (!gl) throw new Error("WebGL context unavailable");
    if (draws.some(draw => draw.indices && draw.indices.componentType === 5125) && !gl.getExtension("OES_element_index_uint")) {{
      throw new Error("browser does not support unsigned-int index buffers");
    }}

    const box = bounds(draws);
    const center = [
      (box.min[0] + box.max[0]) * 0.5,
      (box.min[1] + box.max[1]) * 0.5,
      (box.min[2] + box.max[2]) * 0.5
    ];
    const extent = Math.max(box.max[0] - box.min[0], box.max[1] - box.min[1], box.max[2] - box.min[2], 1e-6);
    const eye = [center[0] + extent * 1.3, center[1] + extent * 0.9, center[2] + extent * 1.3];
    const view = lookAt(eye, center, [0, 1, 0]);
    const projection = perspective(Math.PI / 4, canvas.width / canvas.height, extent * 0.01, extent * 10.0);
    const drawProgram = program(gl);
    const positionLocation = gl.getAttribLocation(drawProgram, "p");
    const texcoordLocation = gl.getAttribLocation(drawProgram, "uv");
    const mvpLocation = gl.getUniformLocation(drawProgram, "mvp");
    const colorLocation = gl.getUniformLocation(drawProgram, "color");
    const textureLocation = gl.getUniformLocation(drawProgram, "baseColorTexture");
    const useTextureLocation = gl.getUniformLocation(drawProgram, "useTexture");
    const textureCache = new Map();
    let texturedPrimitives = 0;
    let quantizedPrimitives = 0;
    for (const draw of draws) {{
      draw.texture = draw.texcoord ? await textureForMaterial(gl, document, draw.material, textureCache) : null;
      if (draw.texture) texturedPrimitives += 1;
      if (draw.quantized) quantizedPrimitives += 1;
    }}

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], BACKGROUND[3]);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(drawProgram);

    let triangles = 0;
    for (const draw of draws) {{
      const positionBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, draw.position.array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(positionLocation);
      gl.vertexAttribPointer(positionLocation, 3, draw.position.componentType, draw.position.normalized, 0, 0);
      if (texcoordLocation >= 0) {{
        if (draw.texcoord && draw.texture) {{
          const texcoordBuffer = gl.createBuffer();
          gl.bindBuffer(gl.ARRAY_BUFFER, texcoordBuffer);
          gl.bufferData(gl.ARRAY_BUFFER, draw.texcoord.array, gl.STATIC_DRAW);
          gl.enableVertexAttribArray(texcoordLocation);
          gl.vertexAttribPointer(texcoordLocation, 2, draw.texcoord.componentType, draw.texcoord.normalized, 0, 0);
        }} else {{
          gl.disableVertexAttribArray(texcoordLocation);
          gl.vertexAttrib2f(texcoordLocation, 0.0, 0.0);
        }}
      }}
      gl.uniformMatrix4fv(mvpLocation, false, new Float32Array(multiply(projection, multiply(view, draw.matrix))));
      gl.uniform4fv(colorLocation, new Float32Array(draw.color));
      gl.uniform1i(useTextureLocation, draw.texture ? 1 : 0);
      if (draw.texture) {{
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, draw.texture);
        gl.uniform1i(textureLocation, 0);
      }}
      if (draw.indices) {{
        const indexBuffer = gl.createBuffer();
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
        gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, draw.indices.array, gl.STATIC_DRAW);
        gl.drawElements(gl.TRIANGLES, draw.indices.count, draw.indices.componentType, 0);
      }} else {{
        gl.drawArrays(gl.TRIANGLES, 0, draw.position.count);
      }}
      const drawError = gl.getError();
      if (drawError !== gl.NO_ERROR) throw new Error("failed to draw primitive: WebGL error " + drawError);
      triangles += draw.triangles;
    }}
    gl.finish();
    finish({{ status: "rendered", meshes: Array.isArray(document.meshes) ? document.meshes.length : 0, triangles, textured_primitives: texturedPrimitives, sampled_textures: textureCache.size, quantized_primitives: quantizedPrimitives, screenshot_data: canvas.toDataURL("image/png") }});
  }} catch (error) {{
    finish({{ status: "failed", error: String(error), meshes: 0, triangles: 0, textured_primitives: 0, sampled_textures: 0, quantized_primitives: 0 }});
  }}
}})();
</script>
</body>
</html>
"""


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
