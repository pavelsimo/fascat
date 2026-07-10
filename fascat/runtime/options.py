from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

RuntimeEngineName = Literal["unity", "unreal"]


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
    render_backend: str = "none"
    render_limitations: tuple[str, ...] = ()
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
            "render_backend": self.render_backend,
            "render_limitations": list(self.render_limitations),
            "render_error": self.render_error,
            "error": self.error,
        }
