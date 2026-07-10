from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from importlib import resources
from pathlib import Path
from typing import Protocol, cast

from fascat import _subprocess
from fascat.io.gltf import validate_gltf

from .browser import _int, _optional_float, _optional_int
from .options import RuntimeEngineName, RuntimeEngineOptions, RuntimeEngineReport

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

    with tempfile.TemporaryDirectory(
        prefix=f"fascat-{options.engine}-runtime-", ignore_cleanup_errors=True
    ) as directory:
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
            completed = _subprocess.run_guarded(command, timeout=options.timeout_seconds)
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
            *([] if preview_path is not None else ["-nographics"]),
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
        render_backend=str(payload.get("render_backend", "unspecified")),
        render_limitations=_string_tuple(payload.get("render_limitations")),
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
        render_backend="none",
        render_limitations=(),
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
        render_backend="none",
        render_limitations=(),
        render_error=error if options.preview_path is not None else None,
        error=error,
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str))
