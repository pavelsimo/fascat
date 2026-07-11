from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from fascat import _subprocess
from fascat.io.gltf import validate_gltf

from .html import _runtime_harness_html
from .options import RuntimeBrowserOptions, RuntimeBrowserRenderOptions, RuntimeBrowserReport

_RESULT_RE = re.compile(r'<pre id="result"[^>]*>(?P<payload>.*?)</pre>', re.DOTALL)
_BROWSER_CANDIDATES = (
    "chromium",
    "chromium-browser",
    "google-chrome",
    "google-chrome-stable",
    "chrome",
    "msedge",
)


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
        harness_path.write_text(_runtime_harness_html(asset_path.resolve(), opts), encoding="utf-8")
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
