from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from fascat import profiles
from fascat.asset import Asset
from fascat.filter import Filter
from fascat.io.gltf import GLTF_SUFFIXES, validate_gltf
from fascat.io.gltf import write_gltf as _write_gltf
from fascat.io.obj import OBJ_SUFFIXES, validate_obj
from fascat.io.obj import write_obj as _write_obj
from fascat.io.step import read_step
from fascat.io.stl import STL_SUFFIXES, validate_stl
from fascat.io.stl import write_stl as _write_stl
from fascat.io.usd import validate_usd
from fascat.io.usd import write_usd as _write_usd
from fascat.options import (
    AnalyzeOptions,
    AtlasOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    ConversionProfile,
    DecimateOptions,
    ExplodeOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    LODOptions,
    MergeOptions,
    ObjExportOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    ReplaceOptions,
    SceneOptimizeOptions,
    StageOptions,
    StepReadOptions,
    StlExportOptions,
    Tessellation,
    UnwrapOptions,
    UsdExportOptions,
    UVMode,
)
from fascat.pipeline_file import PipelineSpec
from fascat.report import timed_step

if TYPE_CHECKING:
    from fascat.analysis import AnalysisReport

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
ExportFormat = Literal["usd", "gltf", "obj", "stl"]


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
    explode: ExplodeOptions | None = None,
    replace: ReplaceOptions | None = None,
    scene: SceneOptimizeOptions | None = None,
    bake_materials: BakeMaterialOptions | None = None,
    remove_holes: RemoveHolesOptions | None = None,
    remove_occluded: RemoveOccludedOptions | None = None,
    decimate: DecimateOptions | None = None,
    lod_generator: LODGeneratorOptions | None = None,
    optimize: OptimizeOptions | None = None,
    lods: LODOptions | None = None,
    progress: Callable[[str, dict[str, int]], None] | None = None,
    validate_output: bool = True,
    debug: bool = False,
    gltf_options: GltfExportOptions | None = None,
    usd_options: UsdExportOptions | None = None,
    obj_options: ObjExportOptions | None = None,
    stl_options: StlExportOptions | None = None,
    pipeline: PipelineSpec | None = None,
    where: Filter | None = None,
) -> Asset:
    output_format = _export_format(output_path)
    output_suffix = Path(output_path).suffix.lower()
    if debug and (output_format != "usd" or (str(output_path) != "-" and output_suffix not in {".usd", ".usda"})):
        raise ValueError("--debug is only supported for .usd or .usda exports")
    selected = profiles.by_name(profile) if isinstance(profile, str) else profile
    asset = read_step(input_path, options=import_options) if import_options is not None else read_step(input_path)
    if progress is not None:
        progress("source", asset.stats())
    if pipeline is not None:
        asset = pipeline.apply(asset, progress=progress)
    else:
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
        if explode is not None:
            asset = asset.explode(explode, where=where)
            if progress is not None:
                progress("explode", asset.stats())
        if replace is not None:
            asset = asset.replace(replace, where=where)
            if progress is not None:
                progress("replace", asset.stats())
        if scene is not None:
            asset = asset.optimize_scene(scene, where=where)
            if progress is not None:
                progress("optimize_scene", asset.stats())
        if bake_materials is not None:
            asset = asset.bake_materials(bake_materials, where=where)
            if progress is not None:
                progress("bake_materials", asset.stats())
        if remove_holes is not None:
            asset = asset.remove_holes(remove_holes, where=where)
            if progress is not None:
                progress("remove_holes", asset.stats())
        if remove_occluded is not None:
            asset = asset.remove_occluded(remove_occluded, where=where)
            if progress is not None:
                progress("remove_occluded", asset.stats())
        if decimate is not None:
            asset = asset.decimate(decimate, where=where)
            if progress is not None:
                progress("decimate", asset.stats())
        optimize_options = optimize if optimize is not None else selected.optimize
        if optimize_options is not None:
            asset = asset.optimize(optimize_options, where=where)
            if progress is not None:
                progress("optimize", asset.stats())
        lod_options = lods if lods is not None else selected.lods
        if lod_generator is not None:
            asset = asset.run_lod_generators(lod_generator, where=where)
            if progress is not None:
                progress("run_lod_generators", asset.stats())
        elif lod_options is not None:
            asset = asset.lods(lod_options, where=where)
            if progress is not None:
                progress("lods", asset.stats())
    write_before = _report_stats(asset)
    write_options: dict[str, object] = _write_options(
        output_format,
        debug=debug,
        gltf_options=gltf_options,
        usd_options=usd_options,
        obj_options=obj_options,
        stl_options=stl_options,
    )
    file_size_budget = _file_size_budget(output_format, gltf_options, usd_options, obj_options, stl_options)
    write_timer = timed_step()
    try:
        with write_timer:
            _write_output(
                asset,
                output_path,
                output_format,
                debug=debug,
                gltf_options=gltf_options,
                usd_options=usd_options,
                obj_options=obj_options,
                stl_options=stl_options,
            )
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
        after=_stats_with_file_size(_report_stats(asset), output_path, file_size_budget, asset),
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


def _file_size_budget(
    output_format: ExportFormat,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
) -> float | None:
    if output_format == "gltf":
        return None if gltf_options is None else gltf_options.file_size_budget_mb
    if output_format == "usd":
        return None if usd_options is None else usd_options.file_size_budget_mb
    if output_format == "obj":
        return None if obj_options is None else obj_options.file_size_budget_mb
    return None if stl_options is None else stl_options.file_size_budget_mb


def _stats_with_file_size(
    stats: dict[str, int],
    path: str | Path,
    budget_mb: float | None,
    asset: Asset,
) -> dict[str, int]:
    output_path = Path(path)
    if str(path) == "-" or not output_path.exists():
        return stats
    size = output_path.stat().st_size
    result = {**stats, "file_size_bytes": size}
    if budget_mb is not None:
        budget_bytes = int(budget_mb * 1_000_000)
        result["file_size_budget_bytes"] = budget_bytes
        if size > budget_bytes:
            asset.report.add_warning(f"file size budget exceeded: {size} bytes > {budget_bytes} bytes")
    return result


def write_usd(
    asset: Asset,
    path: str | Path,
    *,
    debug: bool = False,
    options: UsdExportOptions | None = None,
) -> None:
    asset.write_usd(path, debug=debug, options=options)


def write_gltf(asset: Asset, path: str | Path, *, options: GltfExportOptions | None = None) -> None:
    asset.write_gltf(path, options=options)


def write_obj(asset: Asset, path: str | Path, *, options: ObjExportOptions | None = None) -> None:
    asset.write_obj(path, options=options)


def write_stl(asset: Asset, path: str | Path, *, options: StlExportOptions | None = None) -> None:
    asset.write_stl(path, options=options)


def validate_output(path: str | Path) -> dict[str, int]:
    output_format = _export_format(path)
    return _validate_output(path, output_format)


def analyze(asset: Asset, *, options: AnalyzeOptions | None = None, where: Filter | None = None) -> AnalysisReport:
    return asset.analyze(options or AnalyzeOptions(), where=where)


def _export_format(path: str | Path) -> ExportFormat:
    suffix = Path(path).suffix.lower()
    if suffix in USD_SUFFIXES or str(path) == "-":
        return "usd"
    if suffix in GLTF_SUFFIXES:
        return "gltf"
    if suffix in OBJ_SUFFIXES:
        return "obj"
    if suffix in STL_SUFFIXES:
        return "stl"
    raise ValueError(f"unsupported export extension: {suffix or '<none>'}")


def _write_output(
    asset: Asset,
    path: str | Path,
    output_format: ExportFormat,
    *,
    debug: bool,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
) -> None:
    if output_format == "usd":
        _write_usd(asset, path, debug=debug, options=_usd_options_for_path(path, usd_options))
        return
    if output_format == "gltf":
        _write_gltf(asset, path, options=gltf_options)
        return
    if output_format == "obj":
        _write_obj(asset, path, options=obj_options)
        return
    _write_stl(asset, path, options=stl_options)


def _validate_output(path: str | Path, output_format: ExportFormat) -> dict[str, int]:
    if output_format == "usd":
        return validate_usd(path)
    if output_format == "gltf":
        return validate_gltf(path)
    if output_format == "obj":
        return validate_obj(path)
    return validate_stl(path)


def _usd_options_for_path(path: str | Path, options: UsdExportOptions | None) -> UsdExportOptions | None:
    if Path(path).suffix.lower() != ".usdz":
        return options
    if options is None:
        return UsdExportOptions(package="usdz")
    if options.package == "usdz":
        return options
    return UsdExportOptions(package="usdz", file_size_budget_mb=options.file_size_budget_mb)


def _write_options(
    output_format: ExportFormat,
    *,
    debug: bool,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
) -> dict[str, object]:
    if output_format == "usd":
        return {"format": "OpenUSD", "debug": debug, **(usd_options or UsdExportOptions()).to_dict()}
    if output_format == "gltf":
        return {"format": "glTF", **(gltf_options or GltfExportOptions()).to_dict()}
    if output_format == "obj":
        return {"format": "OBJ", **(obj_options or ObjExportOptions()).to_dict()}
    return {"format": "STL", **(stl_options or StlExportOptions()).to_dict()}


def _validate_options(output_format: ExportFormat) -> dict[str, object]:
    if output_format == "usd":
        return {"backend": "usd-core"}
    if output_format == "gltf":
        return {"backend": "fascat-gltf"}
    return {"backend": f"fascat-{output_format}"}


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
    material_mode: Literal["cad", "pbr"] = "cad",
    merge_equivalent_materials: bool = False,
    normals: bool = True,
    normal_mode: Literal["none", "smooth", "hard_edges", "flat"] = "smooth",
    hard_edge_angle: float = 30.0,
    preserve_face_boundaries: bool = False,
    tangents: bool = False,
    validate_normals: bool = False,
    unwrap: UnwrapOptions | None = None,
    atlas: AtlasOptions | None = None,
    uv0: UVMode = "box",
    uv1: UVMode | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.stage(
        StageOptions(
            materials=materials,
            material_mode=material_mode,
            merge_equivalent_materials=merge_equivalent_materials,
            normals=normals,
            normal_mode=normal_mode,
            hard_edge_angle=hard_edge_angle,
            preserve_face_boundaries=preserve_face_boundaries,
            tangents=tangents,
            validate_normals=validate_normals,
            unwrap=unwrap or UnwrapOptions(),
            atlas=atlas or AtlasOptions(),
            uv0=uv0,
            uv1=uv1,
        ),
        where=where,
    )


def optimize(
    asset: Asset,
    *,
    target_triangles: int | None = None,
    ratio: float | None = None,
    preserve_instances: bool = True,
    simplify: bool = True,
    optimize_buffers: bool = True,
    preserve_hard_edges: bool = False,
    hard_edge_angle: float = 30.0,
    preserve_holes: bool = False,
    preserve_material_boundaries: bool = False,
    preserve_uv_seams: bool = False,
    preserve_small_parts: bool = False,
    small_part_triangle_threshold: int = 64,
    preserve_silhouette: bool = False,
    where: Filter | None = None,
) -> Asset:
    return asset.optimize(
        OptimizeOptions(
            target_triangles=target_triangles,
            ratio=ratio,
            preserve_instances=preserve_instances,
            simplify=simplify,
            optimize_buffers=optimize_buffers,
            preserve_hard_edges=preserve_hard_edges,
            hard_edge_angle=hard_edge_angle,
            preserve_holes=preserve_holes,
            preserve_material_boundaries=preserve_material_boundaries,
            preserve_uv_seams=preserve_uv_seams,
            preserve_small_parts=preserve_small_parts,
            small_part_triangle_threshold=small_part_triangle_threshold,
            preserve_silhouette=preserve_silhouette,
        ),
        where=where,
    )


def merge(asset: Asset, *, options: MergeOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.merge(options or MergeOptions(), where=where)


def explode(asset: Asset, *, options: ExplodeOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.explode(options or ExplodeOptions(), where=where)


def replace(asset: Asset, *, options: ReplaceOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.replace(options or ReplaceOptions(), where=where)


def optimize_scene(asset: Asset, *, options: SceneOptimizeOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.optimize_scene(options or SceneOptimizeOptions(), where=where)


def bake_materials(
    asset: Asset,
    *,
    options: BakeMaterialOptions | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.bake_materials(options or BakeMaterialOptions(), where=where)


def decimate(asset: Asset, *, options: DecimateOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.decimate(options or DecimateOptions(), where=where)


def remove_holes(asset: Asset, *, options: RemoveHolesOptions | None = None, where: Filter | None = None) -> Asset:
    return asset.remove_holes(options or RemoveHolesOptions(), where=where)


def remove_occluded(
    asset: Asset,
    *,
    options: RemoveOccludedOptions | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.remove_occluded(options or RemoveOccludedOptions(), where=where)


def run_lod_generators(
    asset: Asset,
    *,
    options: LODGeneratorOptions | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.run_lod_generators(options or LODGeneratorOptions(), where=where)


def lods(
    asset: Asset,
    *,
    ratios: list[float] | tuple[float, ...] = (0.5, 0.25, 0.1),
    mode: Literal["variants", "extras", "separate"] = "variants",
    screen_coverage: list[float] | tuple[float, ...] | None = None,
    per_part_budget: bool = False,
    drop_tiny_parts: bool = False,
    tiny_part_screen_size: float = 2.0,
    validate: bool = False,
    where: Filter | None = None,
) -> Asset:
    return asset.lods(
        LODOptions(
            tuple(ratios),
            mode=mode,
            screen_coverage=screen_coverage,
            per_part_budget=per_part_budget,
            drop_tiny_parts=drop_tiny_parts,
            tiny_part_screen_size=tiny_part_screen_size,
            validate=validate,
        ),
        where=where,
    )
