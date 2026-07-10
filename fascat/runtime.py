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
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, unquote_to_bytes, urlparse
from urllib.request import url2pathname

import numpy as np
from numpy.typing import NDArray

from fascat import _subprocess
from fascat.io.gltf import validate_gltf

_RESULT_RE = re.compile(r'<pre id="result"[^>]*>(?P<payload>.*?)</pre>', re.DOTALL)
_BROWSER_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "msedge",
)
_MAX_BROWSER_RENDER_SCREENSHOT_DATA_URI_LENGTH = 16 * 1024 * 1024


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
    required_extensions: tuple[str, ...] = ()
    unsupported_extensions: tuple[str, ...] = ()
    decoded_extensions: tuple[str, ...] = ()
    preview_limitations: tuple[str, ...] = ()
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
            "required_extensions": list(self.required_extensions),
            "unsupported_extensions": list(self.unsupported_extensions),
            "decoded_extensions": list(self.decoded_extensions),
            "preview_limitations": list(self.preview_limitations),
            "error": self.error,
        }


@dataclass(frozen=True)
class _BrowserRenderPreflight:
    required_extensions: tuple[str, ...] = ()
    unsupported_extensions: tuple[str, ...] = ()
    decoded_extensions: tuple[str, ...] = ()
    preview_limitations: tuple[str, ...] = ()
    draco_decode_required: bool = False
    meshopt_decode_required: bool = False
    texture_decode_required: bool = False
    fatal: bool = False


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

    with tempfile.TemporaryDirectory(prefix="fascat-runtime-", ignore_cleanup_errors=True) as directory:
        harness_path = Path(directory) / "runtime.html"
        harness_path.write_text(_runtime_browser_measurement_html(asset_path.resolve(), opts), encoding="utf-8")
        command = _browser_invocation(browser, harness_path, opts)
        try:
            completed = _subprocess.run_guarded(command, timeout=opts.timeout_seconds)
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

    preflight = _browser_render_preflight(asset_path)
    output_path = Path(preview_path)
    with tempfile.TemporaryDirectory(prefix="fascat-browser-render-", ignore_cleanup_errors=True) as directory:
        workdir = Path(directory)
        render_asset_path = asset_path
        if preflight.draco_decode_required:
            try:
                render_asset_path = _write_draco_decoded_preview_asset(render_asset_path, workdir / "draco-decoded.glb")
            except Exception as exc:
                preflight = _preflight_draco_decode_failed(preflight, exc)
            else:
                preflight = _preflight_draco_decoded(preflight)
        if not preflight.fatal and preflight.meshopt_decode_required:
            try:
                render_asset_path = _write_meshopt_decoded_preview_asset(
                    render_asset_path, workdir / "meshopt-decoded.gltf"
                )
            except Exception as exc:
                preflight = _preflight_meshopt_decode_failed(preflight, exc)
            else:
                preflight = _preflight_meshopt_decoded(preflight)
        if not preflight.fatal and preflight.texture_decode_required:
            try:
                render_asset_path = _write_ktx2_decoded_preview_asset(render_asset_path, workdir / "ktx2-decoded.glb")
            except Exception as exc:
                preflight = _preflight_texture_decode_failed(preflight, exc)
            else:
                preflight = _preflight_texture_decoded(preflight)
        if preflight.fatal:
            try:
                validation_stats = validate_gltf(render_asset_path)
            except Exception:
                validation_stats = _gltf_document_stats(_read_gltf_json_document(asset_path))
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="unsupported",
                browser=None,
                error="; ".join(preflight.preview_limitations),
                preflight=preflight,
            )
        validation_stats = validate_gltf(render_asset_path)

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
                preflight=preflight,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        harness_path = Path(directory) / "browser-render.html"
        harness_path.write_text(_runtime_browser_render_html(render_asset_path.resolve(), opts), encoding="utf-8")
        command = _browser_render_report_invocation(browser, harness_path, opts)
        try:
            completed = _subprocess.run_guarded(command, timeout=opts.timeout_seconds)
        except subprocess.TimeoutExpired:
            return _browser_render_report(
                asset_path,
                output_path,
                opts,
                validation_stats,
                status="failed",
                browser=browser,
                error="browser render preview timed out",
                preflight=preflight,
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
                preflight=preflight,
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
                preflight=preflight,
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
                required_extensions=preflight.required_extensions,
                unsupported_extensions=preflight.unsupported_extensions,
                decoded_extensions=preflight.decoded_extensions,
                preview_limitations=preflight.preview_limitations,
                error=str(error) if error is not None else None,
            )
        screenshot_data = payload.get("screenshot_data")
        if not isinstance(screenshot_data, str) or not _write_png_data_uri(screenshot_data, output_path):
            screenshot_command = _browser_render_screenshot_invocation(browser, harness_path, output_path, opts)
            try:
                screenshot_completed = _subprocess.run_guarded(screenshot_command, timeout=opts.timeout_seconds)
            except subprocess.TimeoutExpired:
                return _browser_render_report(
                    asset_path,
                    output_path,
                    opts,
                    validation_stats,
                    status="failed",
                    browser=browser,
                    error="browser render preview screenshot timed out",
                    preflight=preflight,
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
                    preflight=preflight,
                )
    if not output_path.exists():
        status = "failed"
        error = "browser did not write render preview screenshot"
    elif status == "rendered" and preflight.preview_limitations:
        status = "rendered_partial"
        error = "; ".join(preflight.preview_limitations)
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
        required_extensions=preflight.required_extensions,
        unsupported_extensions=preflight.unsupported_extensions,
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=preflight.preview_limitations,
        error=str(error) if error is not None else None,
    )


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
    if len(value) > _MAX_BROWSER_RENDER_SCREENSHOT_DATA_URI_LENGTH:
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
    preflight: _BrowserRenderPreflight | None = None,
) -> RuntimeBrowserRenderReport:
    checks = preflight or _BrowserRenderPreflight()
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
        required_extensions=checks.required_extensions,
        unsupported_extensions=checks.unsupported_extensions,
        decoded_extensions=checks.decoded_extensions,
        preview_limitations=checks.preview_limitations,
        error=error,
    )


def _browser_render_preflight(asset_path: Path) -> _BrowserRenderPreflight:
    document = _read_gltf_json_document(asset_path)
    required = tuple(sorted(_string_list(document.get("extensionsRequired"))))
    unsupported_extensions: set[str] = set()
    limitations: list[str] = []

    draco_decode_required = _document_uses_draco(document)
    if draco_decode_required:
        limitations.append("browser preview requires Draco decode for KHR_draco_mesh_compression geometry")
    meshopt_decode_required = _document_has_meshopt_only_buffer_views(document, asset_path)
    if meshopt_decode_required:
        limitations.append("browser preview requires meshopt decode for bufferViews without fallback buffer data")
    texture_decode_required = _document_uses_basis_textures(document)
    if texture_decode_required:
        unsupported_extensions.add("KHR_texture_basisu")
        limitations.append("browser preview requires KTX2/Basis decode for texture sampling")

    return _BrowserRenderPreflight(
        required_extensions=required,
        unsupported_extensions=tuple(sorted(unsupported_extensions)),
        preview_limitations=tuple(limitations),
        draco_decode_required=draco_decode_required,
        meshopt_decode_required=meshopt_decode_required,
        texture_decode_required=texture_decode_required,
        fatal=False,
    )


def _preflight_draco_decoded(preflight: _BrowserRenderPreflight) -> _BrowserRenderPreflight:
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=preflight.unsupported_extensions,
        decoded_extensions=tuple(sorted((*preflight.decoded_extensions, "KHR_draco_mesh_compression"))),
        preview_limitations=_without_preview_limitations(preflight, "Draco", "KHR_draco"),
        draco_decode_required=False,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=preflight.texture_decode_required,
        fatal=preflight.fatal,
    )


def _preflight_draco_decode_failed(preflight: _BrowserRenderPreflight, exc: Exception) -> _BrowserRenderPreflight:
    unsupported = tuple(sorted((*preflight.unsupported_extensions, "KHR_draco_mesh_compression")))
    limitations = _without_preview_limitations(preflight, "Draco", "KHR_draco") + (
        f"browser preview could not decode KHR_draco_mesh_compression: {exc}",
    )
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=unsupported,
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=limitations,
        draco_decode_required=False,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=preflight.texture_decode_required,
        fatal=True,
    )


def _preflight_meshopt_decoded(preflight: _BrowserRenderPreflight) -> _BrowserRenderPreflight:
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=preflight.unsupported_extensions,
        decoded_extensions=tuple(sorted((*preflight.decoded_extensions, "EXT_meshopt_compression"))),
        preview_limitations=_without_preview_limitations(preflight, "meshopt", "EXT_meshopt"),
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=False,
        texture_decode_required=preflight.texture_decode_required,
        fatal=preflight.fatal,
    )


def _preflight_meshopt_decode_failed(preflight: _BrowserRenderPreflight, exc: Exception) -> _BrowserRenderPreflight:
    unsupported = tuple(sorted((*preflight.unsupported_extensions, "EXT_meshopt_compression")))
    limitations = tuple(
        item for item in preflight.preview_limitations if "meshopt" not in item and "EXT_meshopt" not in item
    ) + (f"browser preview could not decode EXT_meshopt_compression: {exc}",)
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=unsupported,
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=limitations,
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=False,
        texture_decode_required=preflight.texture_decode_required,
        fatal=True,
    )


def _preflight_texture_decoded(preflight: _BrowserRenderPreflight) -> _BrowserRenderPreflight:
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=_without_extensions(preflight.unsupported_extensions, "KHR_texture_basisu"),
        decoded_extensions=tuple(sorted((*preflight.decoded_extensions, "KHR_texture_basisu"))),
        preview_limitations=_without_preview_limitations(preflight, "KTX2", "Basis", "KHR_texture_basisu"),
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=False,
        fatal=preflight.fatal,
    )


def _preflight_texture_decode_failed(preflight: _BrowserRenderPreflight, exc: Exception) -> _BrowserRenderPreflight:
    limitations = _without_preview_limitations(preflight, "KTX2", "Basis", "KHR_texture_basisu") + (
        f"browser preview could not decode KHR_texture_basisu: {exc}",
        "browser preview renders fallback texture sources when present; otherwise geometry without KTX2/Basis texture sampling",
    )
    return _BrowserRenderPreflight(
        required_extensions=preflight.required_extensions,
        unsupported_extensions=tuple(sorted({*preflight.unsupported_extensions, "KHR_texture_basisu"})),
        decoded_extensions=preflight.decoded_extensions,
        preview_limitations=limitations,
        draco_decode_required=preflight.draco_decode_required,
        meshopt_decode_required=preflight.meshopt_decode_required,
        texture_decode_required=False,
        fatal=preflight.fatal,
    )


def _without_preview_limitations(preflight: _BrowserRenderPreflight, *patterns: str) -> tuple[str, ...]:
    return tuple(item for item in preflight.preview_limitations if not any(pattern in item for pattern in patterns))


def _without_extensions(extensions: tuple[str, ...], *names: str) -> tuple[str, ...]:
    return tuple(item for item in extensions if item not in names)


def _read_gltf_json_document(asset_path: Path) -> dict[str, Any]:
    if asset_path.suffix.lower() == ".glb":
        data = asset_path.read_bytes()
        if len(data) < 20 or data[0:4] != b"glTF":
            raise RuntimeError("GLB header is invalid")
        json_length = int.from_bytes(data[12:16], "little")
        json_type = int.from_bytes(data[16:20], "little")
        if json_type != 0x4E4F534A:
            raise RuntimeError("first GLB chunk is not JSON")
        end = 20 + json_length
        if end > len(data):
            raise RuntimeError("GLB JSON chunk length is invalid")
        loaded = json.loads(data[20:end].decode("utf-8"))
    else:
        loaded = json.loads(asset_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise RuntimeError("glTF JSON document must be an object")
    return cast(dict[str, Any], loaded)


def _write_draco_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    _run_gltf_transform_copy(asset_path, output_path)
    decoded = _read_gltf_json_document(output_path)
    if _document_uses_draco(decoded):
        raise RuntimeError("glTF Transform copy preserved KHR_draco_mesh_compression")
    return output_path


def _run_gltf_transform_copy(input_path: Path, output_path: Path) -> None:
    command = [*_gltf_transform_runtime_command("copy"), "copy", str(input_path), str(output_path)]
    try:
        completed = _subprocess.run_guarded(command, timeout=_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"glTF Transform copy timed out after {_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"glTF Transform copy failed: {details}")


def _gltf_transform_runtime_command(operation: str) -> list[str]:
    from fascat.io.gltf import _gltf_transform_command

    try:
        return _gltf_transform_command()
    except RuntimeError as exc:
        raise RuntimeError(
            f"glTF Transform {operation} requires the glTF Transform CLI on PATH or FASCAT_GLTF_TRANSFORM"
        ) from exc


def _write_ktx2_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    python_error: Exception | None = None
    try:
        return _write_python_ktx2_decoded_preview_asset(asset_path, output_path.with_suffix(".gltf"))
    except Exception as exc:
        python_error = exc

    try:
        _run_gltf_transform_ktxdecompress(asset_path, output_path)
    except Exception as exc:
        if python_error is not None:
            raise RuntimeError(f"{python_error}; {exc}") from exc
        raise
    decoded = _read_gltf_json_document(output_path)
    if _document_uses_basis_textures(decoded):
        raise RuntimeError("glTF Transform ktxdecompress preserved KHR_texture_basisu")
    return output_path


def _write_python_ktx2_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    try:
        alktx2 = cast(Any, import_module("alktx2"))
    except ImportError as exc:
        raise RuntimeError("default alktx2 KTX2 decoder is not installed for this environment") from exc

    source_document, buffers = _read_gltf_json_and_buffers_for_preview(asset_path)
    document = _preview_document_copy(source_document, images=True, textures=True, extension_lists=True)
    images = document.get("images")
    if not isinstance(images, list):
        raise RuntimeError("glTF KTX2 decode requires an images array")
    basis_sources = _basis_texture_image_indices(document)
    decoded_images = 0
    for image_index, image in enumerate(images):
        if not isinstance(image, dict):
            raise RuntimeError("glTF image must be an object")
        if image_index not in basis_sources and not _image_is_ktx2(image):
            continue
        ktx2_bytes = _load_gltf_image_bytes(image, document, buffers, asset_path.parent)
        decoded = alktx2.decode_ktx2_to_bytes(ktx2_bytes, format="png")
        if not isinstance(decoded, tuple) or len(decoded) != 2:
            raise RuntimeError("alktx2 decode_ktx2_to_bytes returned an invalid result")
        png_bytes, mime_type = decoded
        if not isinstance(png_bytes, bytes) or mime_type != "image/png":
            raise RuntimeError("alktx2 KTX2 decoder did not return PNG bytes")
        image["uri"] = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
        image["mimeType"] = "image/png"
        image.pop("bufferView", None)
        decoded_images += 1

    if decoded_images == 0:
        raise RuntimeError("no KTX2/Basis images were decoded")

    _promote_basis_texture_sources(document)
    _remove_gltf_extension(document, "KHR_texture_basisu")
    _embed_preview_buffers(document, buffers)
    _rewrite_external_image_uris(document, asset_path.parent)
    if _document_uses_basis_textures(document):
        raise RuntimeError("alktx2 KTX2 decode preserved KHR_texture_basisu")
    output_path.write_text(json.dumps(document), encoding="utf-8")
    return output_path


def _run_gltf_transform_ktxdecompress(input_path: Path, output_path: Path) -> None:
    command = [*_gltf_transform_runtime_command("ktxdecompress"), "ktxdecompress", str(input_path), str(output_path)]
    try:
        completed = _subprocess.run_guarded(command, timeout=_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"glTF Transform ktxdecompress timed out after {_subprocess.GLTF_TRANSFORM_TIMEOUT_SECONDS:g}s"
        ) from exc
    if completed.returncode != 0:
        details = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"glTF Transform ktxdecompress failed: {details}")


def _write_meshopt_decoded_preview_asset(asset_path: Path, output_path: Path) -> Path:
    try:
        import meshoptimizer
    except ImportError as exc:
        raise RuntimeError("meshoptimizer is not installed") from exc

    source_document, buffers = _read_gltf_json_and_buffers_for_preview(asset_path)
    document = _preview_document_copy(source_document, images=True, buffer_views=True, extension_lists=True)
    buffer_views = document.get("bufferViews")
    if not isinstance(buffer_views, list):
        raise RuntimeError("glTF bufferViews must be an array")

    payload = bytearray()
    decoded_views = 0
    for view in buffer_views:
        if not isinstance(view, dict):
            raise RuntimeError("glTF bufferView must be an object")
        extensions = view.get("extensions")
        meshopt_extension = extensions.get("EXT_meshopt_compression") if isinstance(extensions, dict) else None
        if isinstance(meshopt_extension, dict):
            data = _decode_meshopt_buffer_view(meshopt_extension, buffers, meshoptimizer)
            decoded_views += 1
        else:
            data = _copy_buffer_view_bytes(view, buffers)
        offset = _append_preview_buffer(payload, data)
        view["buffer"] = 0
        view["byteOffset"] = offset
        view["byteLength"] = len(data)
        if isinstance(meshopt_extension, dict) and isinstance(meshopt_extension.get("byteStride"), int):
            if meshopt_extension.get("mode") == "ATTRIBUTES":
                view["byteStride"] = meshopt_extension["byteStride"]
            else:
                view.pop("byteStride", None)
        if isinstance(extensions, dict):
            extensions.pop("EXT_meshopt_compression", None)
            if not extensions:
                view.pop("extensions", None)

    if decoded_views == 0:
        raise RuntimeError("no meshopt-compressed bufferViews were found")
    document["buffers"] = [
        {
            "byteLength": len(payload),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(payload)).decode("ascii"),
        }
    ]
    _remove_gltf_extension(document, "EXT_meshopt_compression")
    _rewrite_external_image_uris(document, asset_path.parent)
    if _document_has_meshopt_only_buffer_views(document, output_path):
        raise RuntimeError("meshoptimizer decode preserved EXT_meshopt_compression")
    output_path.write_text(json.dumps(document), encoding="utf-8")
    return output_path


def _read_gltf_json_and_buffers_for_preview(asset_path: Path) -> tuple[dict[str, Any], list[bytes]]:
    binary_chunk = b""
    if asset_path.suffix.lower() == ".glb":
        document, binary_chunk = _read_glb_json_and_binary(asset_path)
    else:
        document = _read_gltf_json_document(asset_path)

    buffers: list[bytes] = []
    for index, buffer_value in enumerate(document.get("buffers", [])):
        if not isinstance(buffer_value, dict):
            raise RuntimeError(f"glTF buffer {index} must be an object")
        uri = buffer_value.get("uri")
        if isinstance(uri, str):
            buffers.append(_load_gltf_uri_bytes(uri, asset_path.parent))
        elif asset_path.suffix.lower() == ".glb" and index == 0:
            buffers.append(binary_chunk)
        else:
            buffers.append(b"")
    return document, buffers


def _preview_document_copy(
    source_document: dict[str, Any],
    *,
    images: bool = False,
    textures: bool = False,
    buffer_views: bool = False,
    extension_lists: bool = False,
) -> dict[str, Any]:
    document = dict(source_document)
    if images:
        _copy_preview_object_list(document, "images")
    if textures:
        _copy_preview_object_list(document, "textures", copy_extensions=True)
    if buffer_views:
        _copy_preview_object_list(document, "bufferViews", copy_extensions=True)
    if extension_lists:
        for field in ("extensionsUsed", "extensionsRequired"):
            value = source_document.get(field)
            if isinstance(value, list):
                document[field] = list(value)
    return document


def _copy_preview_object_list(document: dict[str, Any], field: str, *, copy_extensions: bool = False) -> None:
    value = document.get(field)
    if not isinstance(value, list):
        return
    copied: list[object] = []
    for item in value:
        if isinstance(item, dict):
            item_copy = dict(item)
            extensions = item_copy.get("extensions")
            if copy_extensions and isinstance(extensions, dict):
                item_copy["extensions"] = dict(extensions)
            copied.append(item_copy)
        else:
            copied.append(item)
    document[field] = copied


def _read_glb_json_and_binary(asset_path: Path) -> tuple[dict[str, Any], bytes]:
    data = asset_path.read_bytes()
    if len(data) < 20 or data[0:4] != b"glTF":
        raise RuntimeError("GLB header is invalid")
    json_document: dict[str, Any] | None = None
    binary_chunk = b""
    offset = 12
    while offset < len(data):
        if offset + 8 > len(data):
            raise RuntimeError("invalid GLB chunk header")
        chunk_length = int.from_bytes(data[offset : offset + 4], "little")
        chunk_type = int.from_bytes(data[offset + 4 : offset + 8], "little")
        offset += 8
        chunk = data[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise RuntimeError("invalid GLB chunk length")
        offset += chunk_length
        if chunk_type == 0x4E4F534A:
            loaded = json.loads(chunk.decode("utf-8").rstrip(" \x00"))
            if not isinstance(loaded, dict):
                raise RuntimeError("glTF JSON document must be an object")
            json_document = cast(dict[str, Any], loaded)
        elif chunk_type == 0x004E4942:
            binary_chunk = chunk
    if json_document is None:
        raise RuntimeError("GLB contains no JSON chunk")
    return json_document, binary_chunk


def _load_gltf_uri_bytes(uri: str, base_dir: Path) -> bytes:
    if uri.startswith("data:"):
        comma = uri.find(",")
        if comma < 0:
            raise RuntimeError("invalid data URI buffer")
        header = uri[:comma]
        payload = uri[comma + 1 :]
        if ";base64" in header:
            return base64.b64decode(payload)
        return unquote_to_bytes(payload)
    parsed = urlparse(uri)
    if parsed.scheme and parsed.scheme != "file":
        raise RuntimeError(f"unsupported external glTF URI scheme: {parsed.scheme}")
    # url2pathname handles the /C:/... form a file URI produces on Windows.
    path = Path(url2pathname(parsed.path)) if parsed.scheme == "file" else base_dir / unquote(parsed.path)
    return path.read_bytes()


def _load_gltf_image_bytes(
    image: dict[str, Any],
    document: dict[str, Any],
    buffers: list[bytes],
    base_dir: Path,
) -> bytes:
    uri = image.get("uri")
    if isinstance(uri, str):
        return _load_gltf_uri_bytes(uri, base_dir)
    view_index = image.get("bufferView")
    if not isinstance(view_index, int):
        raise RuntimeError("glTF KTX2 image must use uri or bufferView")
    buffer_views = document.get("bufferViews")
    if not isinstance(buffer_views, list) or view_index < 0 or view_index >= len(buffer_views):
        raise RuntimeError("glTF KTX2 image references an invalid bufferView")
    view = buffer_views[view_index]
    if not isinstance(view, dict):
        raise RuntimeError("glTF KTX2 image bufferView must be an object")
    return _copy_buffer_view_bytes(view, buffers)


def _decode_meshopt_buffer_view(
    extension: dict[str, Any],
    buffers: list[bytes],
    meshoptimizer: Any,
) -> bytes:
    buffer_index = _required_int(extension, "buffer")
    if buffer_index < 0 or buffer_index >= len(buffers):
        raise RuntimeError("meshopt extension references an invalid buffer")
    byte_offset = _optional_json_int(extension.get("byteOffset"), 0)
    byte_length = _required_int(extension, "byteLength")
    byte_stride = _required_int(extension, "byteStride")
    count = _required_int(extension, "count")
    mode = extension.get("mode")
    filter_name = str(extension.get("filter", "NONE"))
    if not isinstance(mode, str):
        raise RuntimeError("meshopt extension mode must be a string")
    source_buffer = buffers[buffer_index]
    source = source_buffer[byte_offset : byte_offset + byte_length]
    if len(source) != byte_length:
        raise RuntimeError("meshopt compressed bufferView is out of range")

    if mode == "ATTRIBUTES":
        decoded = meshoptimizer.decode_vertex_buffer(count, byte_stride, source)
        raw = decoded.view(np.uint8).reshape(-1)[: count * byte_stride].copy()
        raw = _apply_meshopt_filter(raw, count, byte_stride, filter_name, meshoptimizer)
        return raw.tobytes()
    if mode == "TRIANGLES":
        if filter_name != "NONE":
            raise RuntimeError("meshopt TRIANGLES mode does not support filters")
        decoded = meshoptimizer.decode_index_buffer(count, byte_stride, source)
        return cast(bytes, decoded.view(np.uint8).reshape(-1)[: count * byte_stride].tobytes())
    if mode == "INDICES":
        if filter_name != "NONE":
            raise RuntimeError("meshopt INDICES mode does not support filters")
        decoded = meshoptimizer.decode_index_sequence(count, byte_stride, source)
        return cast(bytes, decoded.view(np.uint8).reshape(-1)[: count * byte_stride].tobytes())
    raise RuntimeError(f"unsupported meshopt mode: {mode}")


def _apply_meshopt_filter(
    raw: NDArray[np.uint8],
    count: int,
    byte_stride: int,
    filter_name: str,
    meshoptimizer: Any,
) -> NDArray[np.uint8]:
    if filter_name == "NONE":
        return raw
    if filter_name == "OCTAHEDRAL":
        return cast(NDArray[np.uint8], meshoptimizer.decode_filter_oct(raw, count, byte_stride))
    if filter_name == "QUATERNION":
        return cast(NDArray[np.uint8], meshoptimizer.decode_filter_quat(raw, count, byte_stride))
    if filter_name == "EXPONENTIAL":
        return cast(NDArray[np.uint8], meshoptimizer.decode_filter_exp(raw, count, byte_stride))
    raise RuntimeError(f"unsupported meshopt filter: {filter_name}")


def _copy_buffer_view_bytes(view: dict[str, Any], buffers: list[bytes]) -> bytes:
    buffer_index = _optional_json_int(view.get("buffer"), 0)
    if buffer_index < 0 or buffer_index >= len(buffers):
        raise RuntimeError("glTF bufferView references an invalid buffer")
    byte_offset = _optional_json_int(view.get("byteOffset"), 0)
    byte_length = _required_int(view, "byteLength")
    data = buffers[buffer_index][byte_offset : byte_offset + byte_length]
    if len(data) != byte_length:
        raise RuntimeError("glTF bufferView is out of range")
    return data


def _append_preview_buffer(payload: bytearray, data: bytes) -> int:
    padding = (-len(payload)) % 4
    if padding:
        payload.extend(b"\x00" * padding)
    offset = len(payload)
    payload.extend(data)
    return offset


def _embed_preview_buffers(document: dict[str, Any], buffers: list[bytes]) -> None:
    document["buffers"] = [
        {
            "byteLength": len(data),
            "uri": "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii"),
        }
        for data in buffers
    ]


def _remove_gltf_extension(document: dict[str, Any], extension_name: str) -> None:
    for field in ("extensionsUsed", "extensionsRequired"):
        value = document.get(field)
        if not isinstance(value, list):
            continue
        document[field] = [item for item in value if item != extension_name]
        if not document[field]:
            document.pop(field, None)


def _rewrite_external_image_uris(document: dict[str, Any], base_dir: Path) -> None:
    images = document.get("images")
    if not isinstance(images, list):
        return
    for image in images:
        if not isinstance(image, dict):
            continue
        uri = image.get("uri")
        if not isinstance(uri, str) or uri.startswith("data:"):
            continue
        parsed = urlparse(uri)
        if parsed.scheme:
            continue
        image["uri"] = (base_dir / unquote(parsed.path)).resolve().as_uri()


def _gltf_document_stats(document: dict[str, Any]) -> dict[str, int]:
    meshes = document.get("meshes")
    accessors = document.get("accessors")
    if not isinstance(meshes, list) or not isinstance(accessors, list):
        return {"meshes": 0, "points": 0, "triangles": 0}
    points = 0
    triangles = 0
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            continue
        for primitive in primitives:
            if not isinstance(primitive, dict):
                continue
            attributes = primitive.get("attributes")
            if isinstance(attributes, dict) and isinstance(attributes.get("POSITION"), int):
                position_count = _accessor_count_from_json(accessors, attributes["POSITION"])
                points += position_count
            else:
                position_count = 0
            if isinstance(primitive.get("indices"), int):
                triangles += _accessor_count_from_json(accessors, primitive["indices"]) // 3
            elif position_count:
                triangles += position_count // 3
    return {"meshes": len(meshes), "points": points, "triangles": triangles}


def _accessor_count_from_json(accessors: list[object], index: int) -> int:
    if index < 0 or index >= len(accessors):
        return 0
    accessor = accessors[index]
    if not isinstance(accessor, dict) or not isinstance(accessor.get("count"), int):
        return 0
    return int(accessor["count"])


def _required_int(source: dict[str, Any], key: str) -> int:
    value = source.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError(f"meshopt extension {key} must be an integer")
    return value


def _optional_json_int(value: object, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError("glTF integer value must be an integer")
    return value


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _document_uses_draco(document: dict[str, Any]) -> bool:
    if "KHR_draco_mesh_compression" in _string_list(document.get("extensionsRequired")):
        return True
    for primitive in _iter_gltf_primitives(document):
        extensions = primitive.get("extensions")
        if isinstance(extensions, dict) and "KHR_draco_mesh_compression" in extensions:
            return True
    return False


def _document_has_meshopt_only_buffer_views(document: dict[str, Any], asset_path: Path) -> bool:
    buffer_views = document.get("bufferViews")
    accessors = document.get("accessors")
    if not isinstance(buffer_views, list) or not isinstance(accessors, list):
        return False
    referenced_buffer_views = _referenced_primitive_buffer_views(document, accessors)
    for index, view in enumerate(buffer_views):
        if index not in referenced_buffer_views:
            continue
        if not isinstance(view, dict):
            continue
        extensions = view.get("extensions")
        if not isinstance(extensions, dict) or "EXT_meshopt_compression" not in extensions:
            continue
        buffer_index = view.get("buffer")
        if not isinstance(buffer_index, int):
            return True
        if _buffer_is_meshopt_placeholder(document, buffer_index, asset_path):
            return True
    return False


def _buffer_is_meshopt_placeholder(document: dict[str, Any], buffer_index: int, asset_path: Path) -> bool:
    buffers = document.get("buffers")
    if not isinstance(buffers, list) or buffer_index < 0 or buffer_index >= len(buffers):
        return True
    buffer = buffers[buffer_index]
    if not isinstance(buffer, dict):
        return True
    extensions = buffer.get("extensions")
    meshopt = extensions.get("EXT_meshopt_compression") if isinstance(extensions, dict) else None
    if isinstance(meshopt, dict) and meshopt.get("fallback") is True:
        return True
    uri = buffer.get("uri")
    if isinstance(uri, str):
        return False
    return not (asset_path.suffix.lower() == ".glb" and buffer_index == 0)


def _document_uses_basis_textures(document: dict[str, Any]) -> bool:
    if "KHR_texture_basisu" in _string_list(document.get("extensionsRequired")):
        return True
    textures = document.get("textures")
    if isinstance(textures, list):
        for texture in textures:
            if not isinstance(texture, dict):
                continue
            extensions = texture.get("extensions")
            if isinstance(extensions, dict) and "KHR_texture_basisu" in extensions:
                return True
    images = document.get("images")
    if isinstance(images, list):
        for image in images:
            if not isinstance(image, dict):
                continue
            mime_type = image.get("mimeType")
            uri = image.get("uri")
            if mime_type == "image/ktx2" or (isinstance(uri, str) and uri.lower().endswith(".ktx2")):
                return True
    return False


def _basis_texture_image_indices(document: dict[str, Any]) -> set[int]:
    indices: set[int] = set()
    textures = document.get("textures")
    if not isinstance(textures, list):
        return indices
    for texture in textures:
        if not isinstance(texture, dict):
            continue
        extensions = texture.get("extensions")
        basis = extensions.get("KHR_texture_basisu") if isinstance(extensions, dict) else None
        if isinstance(basis, dict) and isinstance(basis.get("source"), int):
            indices.add(cast(int, basis["source"]))
    return indices


def _promote_basis_texture_sources(document: dict[str, Any]) -> None:
    textures = document.get("textures")
    if not isinstance(textures, list):
        return
    for texture in textures:
        if not isinstance(texture, dict):
            continue
        extensions = texture.get("extensions")
        basis = extensions.get("KHR_texture_basisu") if isinstance(extensions, dict) else None
        if isinstance(basis, dict) and isinstance(basis.get("source"), int):
            texture["source"] = basis["source"]
        if isinstance(extensions, dict):
            extensions.pop("KHR_texture_basisu", None)
            if not extensions:
                texture.pop("extensions", None)


def _image_is_ktx2(image: dict[str, Any]) -> bool:
    mime_type = image.get("mimeType")
    uri = image.get("uri")
    return mime_type == "image/ktx2" or (isinstance(uri, str) and uri.lower().endswith(".ktx2"))


def _referenced_primitive_buffer_views(document: dict[str, Any], accessors: list[object]) -> set[int]:
    indices: set[int] = set()
    for accessor_index in _iter_primitive_accessor_indices(document):
        if accessor_index < 0 or accessor_index >= len(accessors):
            continue
        accessor = accessors[accessor_index]
        if not isinstance(accessor, dict):
            continue
        buffer_view = accessor.get("bufferView")
        if isinstance(buffer_view, int):
            indices.add(buffer_view)
    return indices


def _iter_primitive_accessor_indices(document: dict[str, Any]) -> Iterable[int]:
    for primitive in _iter_gltf_primitives(document):
        index = primitive.get("indices")
        if isinstance(index, int):
            yield index
        attributes = primitive.get("attributes")
        if not isinstance(attributes, dict):
            continue
        for attribute_index in attributes.values():
            if isinstance(attribute_index, int):
                yield attribute_index


def _iter_gltf_primitives(document: dict[str, Any]) -> Iterable[dict[str, Any]]:
    meshes = document.get("meshes")
    if not isinstance(meshes, list):
        return
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        primitives = mesh.get("primitives")
        if not isinstance(primitives, list):
            continue
        for primitive in primitives:
            if isinstance(primitive, dict):
                yield primitive


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))


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


def _runtime_browser_measurement_html(asset_path: Path, options: RuntimeBrowserOptions) -> str:
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
