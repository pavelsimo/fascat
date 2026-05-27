from __future__ import annotations

from fascat.options import (
    ConversionProfile,
    LODOptions,
    OptimizeOptions,
    PlatformBudget,
    RepairOptions,
    StageOptions,
    Tessellation,
)


def inspect_only() -> ConversionProfile:
    return ConversionProfile(
        name="inspect-only",
        tessellation=None,
        repair=RepairOptions(),
        stage=StageOptions(uv0="none", uv1=None),
        optimize=None,
        lods=None,
        budget=None,
    )


def realtime_desktop(
    *,
    tessellation_sag: float = 0.1,
    angle: float = 15.0,
    max_triangles: int = 1_000_000,
    lod_ratios: list[float] | tuple[float, ...] = (0.5, 0.25, 0.1),
) -> ConversionProfile:
    return ConversionProfile(
        name="realtime-desktop",
        tessellation=Tessellation(sag=tessellation_sag, angle=angle),
        repair=RepairOptions(tolerance=1e-7),
        stage=StageOptions(uv0="box", uv1=None),
        optimize=OptimizeOptions(target_triangles=max_triangles, simplify=True, optimize_buffers=True),
        lods=LODOptions(ratios=tuple(lod_ratios)) if lod_ratios else None,
        budget=PlatformBudget(
            target_fps=60,
            max_triangles=max_triangles,
            max_vertices=max_triangles * 3,
            max_vertices_per_mesh=65_535,
            max_texture_resolution=4_096,
            max_texture_memory_mb=512,
            max_load_time_ms=2_000,
            max_draw_calls=2_000,
        ),
    )


def realtime_web(
    *,
    tessellation_sag: float = 0.2,
    angle: float = 20.0,
    max_triangles: int = 250_000,
    lod_ratios: list[float] | tuple[float, ...] = (0.5, 0.25),
) -> ConversionProfile:
    return ConversionProfile(
        name="realtime-web",
        tessellation=Tessellation(sag=tessellation_sag, angle=angle),
        repair=RepairOptions(tolerance=1e-7),
        stage=StageOptions(uv0="box", uv1=None),
        optimize=OptimizeOptions(target_triangles=max_triangles, simplify=True, optimize_buffers=True),
        lods=LODOptions(ratios=tuple(lod_ratios)) if lod_ratios else None,
        budget=PlatformBudget(
            target_fps=60,
            max_triangles=max_triangles,
            max_vertices=max_triangles * 3,
            max_vertices_per_mesh=65_535,
            max_texture_resolution=2_048,
            max_texture_memory_mb=128,
            max_load_time_ms=3_000,
            max_draw_calls=500,
        ),
    )


def realtime_mobile(
    *,
    tessellation_sag: float = 0.25,
    angle: float = 20.0,
    max_triangles: int = 150_000,
    lod_ratios: list[float] | tuple[float, ...] = (0.5, 0.25),
) -> ConversionProfile:
    return ConversionProfile(
        name="realtime-mobile",
        tessellation=Tessellation(sag=tessellation_sag, angle=angle),
        repair=RepairOptions(tolerance=1e-7),
        stage=StageOptions(uv0="box", uv1=None),
        optimize=OptimizeOptions(target_triangles=max_triangles, simplify=True, optimize_buffers=True),
        lods=LODOptions(ratios=tuple(lod_ratios)) if lod_ratios else None,
        budget=PlatformBudget(
            target_fps=60,
            max_triangles=max_triangles,
            max_vertices=max_triangles * 3,
            max_vertices_per_mesh=65_535,
            max_texture_resolution=2_048,
            max_texture_memory_mb=128,
            max_load_time_ms=2_500,
            max_draw_calls=250,
        ),
    )


def virtual_reality(
    *,
    tessellation_sag: float = 0.15,
    angle: float = 15.0,
    max_triangles: int = 500_000,
    lod_ratios: list[float] | tuple[float, ...] = (0.5, 0.25, 0.125),
) -> ConversionProfile:
    return ConversionProfile(
        name="virtual-reality",
        tessellation=Tessellation(sag=tessellation_sag, angle=angle),
        repair=RepairOptions(tolerance=1e-7),
        stage=StageOptions(uv0="box", uv1=None),
        optimize=OptimizeOptions(target_triangles=max_triangles, simplify=True, optimize_buffers=True),
        lods=LODOptions(ratios=tuple(lod_ratios)) if lod_ratios else None,
        budget=PlatformBudget(
            target_fps=90,
            max_triangles=max_triangles,
            max_vertices=max_triangles * 3,
            max_vertices_per_mesh=65_535,
            max_texture_resolution=2_048,
            max_texture_memory_mb=256,
            max_load_time_ms=1_500,
            max_draw_calls=250,
        ),
    )


def by_name(name: str) -> ConversionProfile:
    if name == "inspect-only":
        return inspect_only()
    if name == "realtime-desktop":
        return realtime_desktop()
    if name == "realtime-web":
        return realtime_web()
    if name == "realtime-mobile":
        return realtime_mobile()
    if name == "virtual-reality":
        return virtual_reality()
    raise ValueError(f"unknown profile: {name}")
