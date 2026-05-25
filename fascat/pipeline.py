from __future__ import annotations

from pathlib import Path

from fascat import profiles
from fascat.asset import Asset
from fascat.io.step import read_step
from fascat.io.usd import write_usd
from fascat.options import ConversionProfile, LODOptions, OptimizeOptions, StageOptions, Tessellation


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    profile: str | ConversionProfile = "realtime-desktop",
    tessellation: Tessellation | None = None,
    stage: StageOptions | None = None,
    optimize: OptimizeOptions | None = None,
    lods: LODOptions | None = None,
) -> Asset:
    selected = profiles.by_name(profile) if isinstance(profile, str) else profile
    asset = read_step(input_path)
    tessellation_options = tessellation or selected.tessellation
    if tessellation_options is not None:
        asset = asset.tessellate(tessellation_options)
    asset = asset.repair(selected.repair)
    asset = asset.stage(stage or selected.stage)
    optimize_options = optimize if optimize is not None else selected.optimize
    if optimize_options is not None:
        asset = asset.optimize(optimize_options)
    lod_options = lods if lods is not None else selected.lods
    if lod_options is not None:
        asset = asset.lods(lod_options)
    asset.report.finish(asset.stats())
    write_usd(asset, output_path)
    return asset


def tessellate(asset: Asset, *, sag: float = 0.1, angle: float = 15.0) -> Asset:
    return asset.tessellate(Tessellation(sag=sag, angle=angle))


def repair(asset: Asset, *, tolerance: float = 0.0) -> Asset:
    from fascat.options import RepairOptions

    return asset.repair(RepairOptions(tolerance=tolerance))


def optimize(asset: Asset, *, target_triangles: int | None = None, ratio: float | None = None) -> Asset:
    return asset.optimize(OptimizeOptions(target_triangles=target_triangles, ratio=ratio))
