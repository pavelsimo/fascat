from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from fascat import profiles
from fascat.asset import Asset
from fascat.io.step import read_step
from fascat.io.usd import validate_usd, write_usd
from fascat.options import ConversionProfile, LODOptions, OptimizeOptions, StageOptions, Tessellation, UVMode
from fascat.report import timed_step


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    profile: str | ConversionProfile = "realtime-desktop",
    tessellation: Tessellation | None = None,
    stage: StageOptions | None = None,
    optimize: OptimizeOptions | None = None,
    lods: LODOptions | None = None,
    progress: Callable[[str, dict[str, int]], None] | None = None,
    validate_output: bool = True,
    debug: bool = False,
) -> Asset:
    selected = profiles.by_name(profile) if isinstance(profile, str) else profile
    asset = read_step(input_path)
    if progress is not None:
        progress("source", asset.stats())
    tessellation_options = tessellation or selected.tessellation
    if tessellation_options is not None:
        asset = asset.tessellate(tessellation_options)
        if progress is not None:
            progress("tessellate", asset.stats())
    asset = asset.repair(selected.repair)
    if progress is not None:
        progress("repair", asset.stats())
    asset = asset.stage(stage or selected.stage)
    if progress is not None:
        progress("stage", asset.stats())
    optimize_options = optimize if optimize is not None else selected.optimize
    if optimize_options is not None:
        asset = asset.optimize(optimize_options)
        if progress is not None:
            progress("optimize", asset.stats())
    lod_options = lods if lods is not None else selected.lods
    if lod_options is not None:
        asset = asset.lods(lod_options)
        if progress is not None:
            progress("lods", asset.stats())
    write_before = asset.stats()
    write_options: dict[str, object] = {"format": "OpenUSD", "debug": debug}
    write_timer = timed_step()
    try:
        with write_timer:
            write_usd(asset, output_path, debug=debug)
    except Exception as exc:
        _record_failed_step(
            asset,
            "write",
            options=write_options,
            before=write_before,
            duration=write_timer.duration,
            exc=exc,
        )
        raise
    asset.report.add_step(
        "write",
        options=write_options,
        before=write_before,
        after=asset.stats(),
        duration=write_timer.duration,
    )
    if progress is not None:
        progress("write", asset.stats())
    if validate_output:
        validate_before = asset.stats()
        validate_options: dict[str, object] = {"backend": "usd-core"}
        validate_timer = timed_step()
        try:
            with validate_timer:
                validation_stats = validate_usd(output_path)
        except Exception as exc:
            _record_failed_step(
                asset,
                "validate",
                options=validate_options,
                before=validate_before,
                duration=validate_timer.duration,
                exc=exc,
            )
            raise
        after = {
            **validate_before,
            "validated_meshes": validation_stats["meshes"],
            "validated_points": validation_stats["points"],
            "validated_triangles": validation_stats["triangles"],
        }
        asset.report.add_step(
            "validate",
            options=validate_options,
            before=validate_before,
            after=after,
            duration=validate_timer.duration,
        )
        if progress is not None:
            progress("validate", asset.stats())
    asset.report.finish(asset.stats())
    return asset


def _record_failed_step(
    asset: Asset,
    name: str,
    *,
    options: dict[str, object],
    before: dict[str, int],
    duration: float,
    exc: Exception,
) -> None:
    message = str(exc) or exc.__class__.__name__
    asset.report.add_error(message)
    asset.report.add_step(
        name,
        options=options,
        before=before,
        after=asset.stats(),
        duration=duration,
    )
    asset.report.finish(asset.stats())
    cast(Any, exc).report = asset.report


def tessellate(
    asset: Asset,
    *,
    sag: float = 0.1,
    angle: float = 15.0,
    relative: bool = True,
    max_edge_length: float | None = None,
    create_normals: bool = True,
    keep_brep: bool = False,
) -> Asset:
    return asset.tessellate(
        Tessellation(
            sag=sag,
            angle=angle,
            relative=relative,
            max_edge_length=max_edge_length,
            create_normals=create_normals,
            keep_brep=keep_brep,
        )
    )


def repair(asset: Asset, *, tolerance: float = 0.0) -> Asset:
    from fascat.options import RepairOptions

    return asset.repair(RepairOptions(tolerance=tolerance))


def stage(
    asset: Asset,
    *,
    materials: Literal["cad", "display", "none"] = "cad",
    normals: bool = True,
    uv0: UVMode = "box",
    uv1: UVMode | None = None,
) -> Asset:
    return asset.stage(StageOptions(materials=materials, normals=normals, uv0=uv0, uv1=uv1))


def optimize(
    asset: Asset,
    *,
    target_triangles: int | None = None,
    ratio: float | None = None,
    preserve_instances: bool = True,
    simplify: bool = True,
    optimize_buffers: bool = True,
) -> Asset:
    return asset.optimize(
        OptimizeOptions(
            target_triangles=target_triangles,
            ratio=ratio,
            preserve_instances=preserve_instances,
            simplify=simplify,
            optimize_buffers=optimize_buffers,
        )
    )


def lods(asset: Asset, *, ratios: list[float] | tuple[float, ...] = (0.5, 0.25, 0.1)) -> Asset:
    return asset.lods(LODOptions(tuple(ratios)))
