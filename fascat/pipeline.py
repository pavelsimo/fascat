from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, cast

from fascat import profiles
from fascat.asset import Asset
from fascat.filter import Filter
from fascat.io.gltf import GLTF_SUFFIXES, validate_gltf
from fascat.io.gltf import write_gltf as _write_gltf
from fascat.io.step import read_step
from fascat.io.usd import validate_usd
from fascat.io.usd import write_usd as _write_usd
from fascat.options import (
    BrepHealOptions,
    ConversionProfile,
    LODOptions,
    MergeOptions,
    OptimizeOptions,
    StageOptions,
    StepReadOptions,
    Tessellation,
    UVMode,
)
from fascat.report import timed_step

USD_SUFFIXES = {".usd", ".usda", ".usdc"}
ExportFormat = Literal["usd", "gltf"]


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    profile: str | ConversionProfile = "realtime-desktop",
    import_options: StepReadOptions | None = None,
    tessellation: Tessellation | None = None,
    heal_brep: BrepHealOptions | None = None,
    stage: StageOptions | None = None,
    merge: MergeOptions | None = None,
    optimize: OptimizeOptions | None = None,
    lods: LODOptions | None = None,
    progress: Callable[[str, dict[str, int]], None] | None = None,
    validate_output: bool = True,
    debug: bool = False,
    where: Filter | None = None,
) -> Asset:
    output_format = _export_format(output_path)
    if debug and output_format != "usd":
        raise ValueError("--debug is only supported for .usd or .usda exports")
    selected = profiles.by_name(profile) if isinstance(profile, str) else profile
    asset = read_step(input_path, options=import_options) if import_options is not None else read_step(input_path)
    if progress is not None:
        progress("source", asset.stats())
    tessellation_options = tessellation or selected.tessellation
    if heal_brep is not None:
        asset = asset.heal_brep(heal_brep, where=where)
        if progress is not None:
            progress("heal_brep", asset.stats())
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
    if merge is not None:
        asset = asset.merge(merge, where=where)
        if progress is not None:
            progress("merge", asset.stats())
    optimize_options = optimize if optimize is not None else selected.optimize
    if optimize_options is not None:
        asset = asset.optimize(optimize_options, where=where)
        if progress is not None:
            progress("optimize", asset.stats())
    lod_options = lods if lods is not None else selected.lods
    if lod_options is not None:
        asset = asset.lods(lod_options, where=where)
        if progress is not None:
            progress("lods", asset.stats())
    write_before = _report_stats(asset)
    write_options: dict[str, object] = _write_options(output_format, debug=debug)
    write_timer = timed_step()
    try:
        with write_timer:
            _write_output(asset, output_path, output_format, debug=debug)
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
        after=_report_stats(asset),
        duration=write_timer.duration,
    )
    if progress is not None:
        progress("write", asset.stats())
    if validate_output:
        validate_before = _report_stats(asset)
        validate_options: dict[str, object] = _validate_options(output_format)
        validate_timer = timed_step()
        try:
            with validate_timer:
                validation_stats = _validate_output(output_path, output_format)
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
    asset.report.finish(_report_stats(asset))
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
        after=_report_stats(asset),
        duration=duration,
    )
    asset.report.finish(_report_stats(asset))
    cast(Any, exc).report = asset.report


def _report_stats(asset: Asset) -> dict[str, int]:
    return asset.stats(include_lods=any(part.lod_meshes for part in asset.parts.values()))


def write_usd(asset: Asset, path: str | Path, *, debug: bool = False) -> None:
    asset.write_usd(path, debug=debug)


def write_gltf(asset: Asset, path: str | Path) -> None:
    asset.write_gltf(path)


def validate_output(path: str | Path) -> dict[str, int]:
    output_format = _export_format(path)
    return _validate_output(path, output_format)


def _export_format(path: str | Path) -> ExportFormat:
    suffix = Path(path).suffix.lower()
    if suffix in USD_SUFFIXES or str(path) == "-":
        return "usd"
    if suffix in GLTF_SUFFIXES:
        return "gltf"
    raise ValueError(f"unsupported export extension: {suffix or '<none>'}")


def _write_output(asset: Asset, path: str | Path, output_format: ExportFormat, *, debug: bool) -> None:
    if output_format == "usd":
        _write_usd(asset, path, debug=debug)
        return
    _write_gltf(asset, path)


def _validate_output(path: str | Path, output_format: ExportFormat) -> dict[str, int]:
    if output_format == "usd":
        return validate_usd(path)
    return validate_gltf(path)


def _write_options(output_format: ExportFormat, *, debug: bool) -> dict[str, object]:
    if output_format == "usd":
        return {"format": "OpenUSD", "debug": debug}
    return {"format": "glTF"}


def _validate_options(output_format: ExportFormat) -> dict[str, object]:
    if output_format == "usd":
        return {"backend": "usd-core"}
    return {"backend": "fascat-gltf"}


def tessellate(
    asset: Asset,
    *,
    sag: float = 0.1,
    angle: float = 15.0,
    relative: bool = True,
    min_edge_length: float | None = None,
    max_edge_length: float | None = None,
    preserve_boundaries: bool = True,
    curvature_adaptive: bool = False,
    avoid_skinny_triangles: bool = False,
    quality_report: bool = False,
    create_normals: bool = True,
    keep_brep: bool = False,
    part_settings: dict[str, dict[str, object]] | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.tessellate(
        Tessellation(
            sag=sag,
            angle=angle,
            relative=relative,
            min_edge_length=min_edge_length,
            max_edge_length=max_edge_length,
            preserve_boundaries=preserve_boundaries,
            curvature_adaptive=curvature_adaptive,
            avoid_skinny_triangles=avoid_skinny_triangles,
            quality_report=quality_report,
            create_normals=create_normals,
            keep_brep=keep_brep,
            part_settings=part_settings or {},
        ),
        where=where,
    )


def repair(asset: Asset, *, tolerance: float = 0.0, where: Filter | None = None) -> Asset:
    from fascat.options import RepairOptions

    return asset.repair(RepairOptions(tolerance=tolerance), where=where)


def heal_brep(
    asset: Asset,
    *,
    options: BrepHealOptions | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.heal_brep(options or BrepHealOptions(), where=where)


def stage(
    asset: Asset,
    *,
    materials: Literal["cad", "display", "none"] = "cad",
    normals: bool = True,
    uv0: UVMode = "box",
    uv1: UVMode | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.stage(StageOptions(materials=materials, normals=normals, uv0=uv0, uv1=uv1), where=where)


def optimize(
    asset: Asset,
    *,
    target_triangles: int | None = None,
    ratio: float | None = None,
    preserve_instances: bool = True,
    simplify: bool = True,
    optimize_buffers: bool = True,
    where: Filter | None = None,
) -> Asset:
    return asset.optimize(
        OptimizeOptions(
            target_triangles=target_triangles,
            ratio=ratio,
            preserve_instances=preserve_instances,
            simplify=simplify,
            optimize_buffers=optimize_buffers,
        ),
        where=where,
    )


def merge(asset: Asset, *, options: MergeOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.merge(options or MergeOptions(), where=where)


def lods(
    asset: Asset,
    *,
    ratios: list[float] | tuple[float, ...] = (0.5, 0.25, 0.1),
    where: Filter | None = None,
) -> Asset:
    return asset.lods(LODOptions(tuple(ratios)), where=where)
