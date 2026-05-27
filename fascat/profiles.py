from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import tomli

from fascat.options import (
    ConversionProfile,
    LODOptions,
    OptimizeOptions,
    PlatformBudget,
    RepairOptions,
    StageOptions,
    Tessellation,
)

_PROFILE_FILE_KEYS = frozenset({"name", "budget"})
_BUDGET_FILE_KEYS = frozenset(
    {
        "target_fps",
        "max_triangles",
        "max_vertices",
        "max_vertices_per_mesh",
        "max_texture_resolution",
        "max_texture_memory_mb",
        "max_load_time_ms",
        "max_draw_calls",
        "unity_reference_profile",
        "unity_reference_triangles",
        "unity_reference_draw_calls",
    }
)


@dataclass(frozen=True)
class TessellationSizeBand:
    max_diagonal: float | None
    sag: float | None = None
    sag_ratio: float | None = None
    angle: float | None = None
    max_polygon_length: float | None = None

    def __post_init__(self) -> None:
        if self.max_diagonal is not None and self.max_diagonal <= 0.0:
            raise ValueError("max_diagonal must be greater than 0 when set")
        if self.sag is not None and self.sag <= 0.0:
            raise ValueError("sag must be greater than 0 when set")
        if self.sag_ratio is not None and self.sag_ratio <= 0.0:
            raise ValueError("sag_ratio must be greater than 0 when set")
        if self.angle is not None and (self.angle <= 0.0 or self.angle > 180.0):
            raise ValueError("angle must be greater than 0 and no more than 180 when set")
        if self.max_polygon_length is not None and self.max_polygon_length <= 0.0:
            raise ValueError("max_polygon_length must be greater than 0 when set")

    def to_part_settings(self) -> dict[str, object]:
        settings: dict[str, object] = {}
        if self.sag is not None:
            settings["sag"] = self.sag
        if self.sag_ratio is not None:
            settings["sag_ratio"] = self.sag_ratio
        if self.angle is not None:
            settings["angle"] = self.angle
        if self.max_polygon_length is not None:
            settings["max_polygon_length"] = self.max_polygon_length
        return settings


def size_adaptive_tessellation(
    asset: Any,
    *,
    base: Tessellation | None = None,
    bands: Sequence[TessellationSizeBand],
) -> Tessellation:
    if not bands:
        raise ValueError("at least one tessellation size band is required")

    options = base or Tessellation()
    part_settings = {part_id: dict(settings) for part_id, settings in options.part_settings.items()}
    for part in getattr(asset, "parts", {}).values():
        if part.id in part_settings or part.name in part_settings:
            continue
        diagonal = _part_diagonal(part)
        if diagonal is None:
            continue
        band = _band_for_diagonal(diagonal, bands)
        if band is None:
            continue
        settings = band.to_part_settings()
        if settings:
            part_settings[part.id] = settings

    values = options.to_dict()
    values["part_settings"] = part_settings
    return Tessellation(**cast(Any, values))


def from_mapping(
    values: dict[str, object],
    *,
    base: str | ConversionProfile = "realtime-desktop",
) -> ConversionProfile:
    """Build a custom target-device profile by overlaying a budget on a base profile."""

    _validate_mapping_keys(values, _PROFILE_FILE_KEYS, "target-device profile")
    base_profile = by_name(base) if isinstance(base, str) else base
    name = values.get("name", base_profile.name)
    if not isinstance(name, str) or not name:
        raise ValueError("target-device profile name must be a non-empty string")
    budget_values = values.get("budget")
    if not isinstance(budget_values, dict):
        raise ValueError("target-device profile must include a budget table")
    _validate_mapping_keys(budget_values, _BUDGET_FILE_KEYS, "target-device profile budget")
    merged_budget = {} if base_profile.budget is None else base_profile.budget.to_dict()
    merged_budget.update(budget_values)
    if "max_triangles" in budget_values and "max_vertices" not in budget_values:
        merged_budget["max_vertices"] = int(cast(Any, budget_values["max_triangles"])) * 3
    return ConversionProfile(
        name=name,
        tessellation=base_profile.tessellation,
        repair=base_profile.repair,
        stage=base_profile.stage,
        optimize=base_profile.optimize,
        lods=base_profile.lods,
        budget=PlatformBudget(**cast(Any, merged_budget)),
    )


def from_file(
    path: str | Path,
    *,
    base: str | ConversionProfile = "realtime-desktop",
) -> ConversionProfile:
    """Load a custom target-device budget profile from TOML or JSON."""

    return from_mapping(_load_profile_file(path), base=base)


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
            unity_reference_profile="desktop",
            unity_reference_triangles=(10_000_000, 100_000_000),
            unity_reference_draw_calls=10_000,
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
            unity_reference_profile="webgl",
            unity_reference_triangles=(100_000, 1_000_000),
            unity_reference_draw_calls=200,
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
            unity_reference_profile="mobile",
            unity_reference_triangles=(100_000, 500_000),
            unity_reference_draw_calls=1_000,
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
            unity_reference_profile="vr",
            unity_reference_triangles=(500_000, 2_000_000),
            unity_reference_draw_calls=1_000,
        ),
    )


def augmented_reality(
    *,
    tessellation_sag: float = 0.3,
    angle: float = 22.5,
    max_triangles: int = 100_000,
    lod_ratios: list[float] | tuple[float, ...] = (0.5, 0.25),
) -> ConversionProfile:
    return ConversionProfile(
        name="augmented-reality",
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
            max_texture_resolution=1_024,
            max_texture_memory_mb=64,
            max_load_time_ms=1_500,
            max_draw_calls=150,
            unity_reference_profile="ar",
            unity_reference_triangles=(50_000, 250_000),
            unity_reference_draw_calls=500,
        ),
    )


def mixed_reality(
    *,
    tessellation_sag: float = 0.35,
    angle: float = 25.0,
    max_triangles: int = 75_000,
    lod_ratios: list[float] | tuple[float, ...] = (0.5, 0.25),
) -> ConversionProfile:
    return ConversionProfile(
        name="mixed-reality",
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
            max_texture_resolution=1_024,
            max_texture_memory_mb=64,
            max_load_time_ms=1_200,
            max_draw_calls=100,
            unity_reference_profile="mixed-reality",
            unity_reference_triangles=(50_000, 200_000),
            unity_reference_draw_calls=500,
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
    if name == "augmented-reality":
        return augmented_reality()
    if name == "mixed-reality":
        return mixed_reality()
    raise ValueError(f"unknown profile: {name}")


def _load_profile_file(path: str | Path) -> dict[str, object]:
    profile_path = Path(path)
    text = profile_path.read_text(encoding="utf-8")
    suffix = profile_path.suffix.lower()
    if suffix == ".json":
        values = json.loads(text)
    elif suffix in {".toml", ".tml"}:
        values = tomli.loads(text)
    else:
        raise ValueError("target-device profile files must use .toml or .json")
    if not isinstance(values, dict):
        raise ValueError("target-device profile file must contain a top-level table/object")
    return cast(dict[str, object], values)


def _validate_mapping_keys(values: dict[str, object], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"{context} contains unsupported key(s): {', '.join(unknown)}")


def _band_for_diagonal(diagonal: float, bands: Sequence[TessellationSizeBand]) -> TessellationSizeBand | None:
    for band in bands:
        if band.max_diagonal is None or diagonal <= band.max_diagonal:
            return band
    return None


def _part_diagonal(part: Any) -> float | None:
    mesh = getattr(part, "mesh", None)
    if mesh is not None:
        mins, maxs = mesh.bounds()
        return float(np.linalg.norm(maxs - mins))
    return _source_shape_diagonal(getattr(part, "source_shape", None))


def _source_shape_diagonal(shape: object | None) -> float | None:
    if shape is None:
        return None
    try:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib

        bounds = Bnd_Box()
        BRepBndLib.Add_s(shape, bounds)
        if bounds.IsVoid():
            return None
        xmin, ymin, zmin, xmax, ymax, zmax = bounds.Get()
    except Exception:
        return None
    mins = np.asarray([xmin, ymin, zmin], dtype=np.float64)
    maxs = np.asarray([xmax, ymax, zmax], dtype=np.float64)
    return float(np.linalg.norm(maxs - mins))
