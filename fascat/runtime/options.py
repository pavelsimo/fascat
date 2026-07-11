from __future__ import annotations

from dataclasses import asdict, dataclass


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
