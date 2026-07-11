from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import typer

from fascat.io._suffixes import (
    CAD_SUFFIXES,
    EXPORT_SUFFIXES,
)
from fascat.options import (
    AnalyzeOptions,
    BrepHealOptions,
    ConversionProfile,
    LODGeneratorOptions,
    LODLevel,
    LODOptions,
    MetadataExportOptions,
    StepReadOptions,
)

if TYPE_CHECKING:
    from fascat.filter import Filter
    from fascat.pipeline_file import PipelineSpec

from ._enums import MetadataMode, PmiMode, Profile
from ._io_helpers import by_name, profile_from_file
from ._output import _fail, _is_stdio, _require_existing_file


def _resolve_convert_output(
    input_path: Path,
    output_path: Path | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> Path:
    if output_path is not None:
        return output_path
    if _is_stdio(input_path):
        _fail(ctx, payload, "Output path is required when reading CAD data from stdin.", code=2)
    return input_path.with_suffix(".usdc")


def _parse_lods(value: str | None, ctx: typer.Context, payload: dict[str, Any]) -> list[float] | None:
    if value is None:
        return None
    try:
        ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        _fail(ctx, payload, "--lods must be a comma-separated list of numbers.", code=2)
        raise AssertionError("unreachable") from exc
    if not ratios:
        _fail(ctx, payload, "--lods must include at least one ratio.", code=2)
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        _fail(ctx, payload, "--lods ratios must be greater than 0 and less than 1.", code=2)
    if ratios != sorted(ratios, reverse=True):
        _fail(ctx, payload, "--lods ratios must be sorted from highest to lowest detail.", code=2)
    return ratios


def _convert_operation_diagnostics(payload: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []

    def add(operation: str, level: str, message: str) -> None:
        diagnostics.append({"operation": operation, "level": level, "message": message})

    add("import", "exact", "CAD import reads hierarchy, metadata, materials, and BREP handles when available")
    if payload["heal_brep"]:
        if payload["remove_sliver_faces"]:
            add(
                "heal_brep",
                "approximate",
                "BREP healing runs, but sliver-face removal is reported only because backend removal is unavailable",
            )
        else:
            add(
                "heal_brep",
                "exact",
                "BREP sewing, edge fixing, tolerance unification, same-domain cleanup, and overlap cleanup are requested",
            )
    add("tessellate", "exact", "BREP tessellation uses the selected sag, angle, and edge cleanup settings")
    add("repair", "exact", "mesh repair applies selected cleanup operations after tessellation")
    if payload["atlas"]:
        add("atlas", "metadata_only", "atlas settings are recorded as metadata; atlas images are not written")
    add("stage", "exact", "material, normal, tangent, and UV staging options are applied before optimization")
    if payload["merge_vertices"]:
        add(
            "merge_vertices",
            "exact",
            "exact or tolerance-close vertices are merged with selected attribute and material-boundary protections",
        )
    if payload["delete_degenerate_polygons"]:
        add(
            "delete_degenerate_polygons",
            "exact",
            "degenerate and optional duplicate polygons are removed with before/after counts and reason metadata",
        )
    if payload["merge"]:
        add("merge", "exact", "selected hierarchy is merged according to the requested merge mode")
    if payload["explode"] is not None:
        add("explode", "exact", "selected meshes are split by material or connected components")
    if payload["replace"] is not None:
        add("replace", "exact", "selected geometry is replaced with the requested proxy mode")
    if (
        payload["batch_by_material"]
        or payload["merge_compatible_meshes"]
        or payload["split_large_meshes"]
        or payload["flatten"] != "safe"
        or payload["index_buffer"] != "auto"
        or payload["instance_policy"] != "auto"
        or payload["instance_similarity_tolerance"] > 0.0
    ):
        add("optimize_scene", "exact", "scene batching, splitting, flattening, and instance policy options are applied")
    if payload["bake_materials"]:
        add(
            "bake_materials",
            "exact",
            "material baking rasterizes selected material maps into first-class atlas images",
        )
    if payload["remove_holes"]:
        add(
            "remove_holes",
            "approximate",
            "hole removal uses mesh boundary classification and filling when BREP feature removal is unavailable",
        )
    if payload["remove_occluded"]:
        add(
            "remove_occluded",
            "approximate",
            "occlusion removal uses deterministic sampled visibility; precision controls the sample budget",
        )
    if payload["decimate"]:
        if payload["decimate_criterion"] == "quality":
            add(
                "decimate",
                "exact",
                "quality decimation passes tolerance-derived target error bounds to the simplification backend",
            )
        else:
            add("decimate", "exact", "target decimation applies the requested triangle budget or ratio")
    add("optimize", "exact", "profile optimization applies triangle reduction and buffer optimization settings")
    if payload["run_lod_generators"]:
        add("run_lod_generators", "exact", "preset or explicit LOD levels are generated from optimized meshes")
    elif payload["lods"] is not None:
        add("lods", "exact", "ratio-based LOD meshes are generated from optimized meshes")
    add(
        "export",
        "exact",
        "the selected writer produces the requested output format and records file-size budget warnings",
    )
    return diagnostics


def _parse_bake_maps(value: str, ctx: typer.Context, payload: dict[str, Any]) -> tuple[str, ...]:
    maps = tuple(item.strip().replace("-", "_") for item in value.split(",") if item.strip())
    allowed = {"base_color", "opacity", "normal", "roughness", "metallic", "ao", "emissive"}
    if not maps:
        _fail(ctx, payload, "--bake must include at least one map.", code=2)
    unknown = set(maps) - allowed
    if unknown:
        _fail(ctx, payload, f"Unsupported --bake maps: {', '.join(sorted(unknown))}.", code=2)
    return maps


def _parse_decimate_cleanup_attributes(value: str, ctx: typer.Context, payload: dict[str, Any]) -> tuple[str, ...]:
    attributes = tuple(item.strip().replace("-", "_") for item in value.split(",") if item.strip())
    allowed = {"unused_uvs", "tangents"}
    unknown = set(attributes) - allowed
    if unknown:
        _fail(ctx, payload, f"Unsupported --decimate-cleanup-attributes values: {', '.join(sorted(unknown))}.", code=2)
    return tuple(dict.fromkeys(attributes))


def _parse_hole_types(value: str, ctx: typer.Context, payload: dict[str, Any]) -> tuple[str, ...]:
    hole_types = tuple(item.strip().replace("-", "_") for item in value.split(",") if item.strip())
    allowed = {"through", "blind", "surface"}
    if not hole_types:
        _fail(ctx, payload, "--hole-types must include at least one type.", code=2)
    unknown = set(hole_types) - allowed
    if unknown:
        _fail(ctx, payload, f"Unsupported --hole-types values: {', '.join(sorted(unknown))}.", code=2)
    return hole_types


def _parse_uv_channels(value: str | None, ctx: typer.Context, payload: dict[str, Any]) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        channels = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        _fail(ctx, payload, "--normalize-uvs must be a comma-separated list of UV channel indices.", code=2)
        raise AssertionError("unreachable") from exc
    if not channels:
        _fail(ctx, payload, "--normalize-uvs must include at least one UV channel.", code=2)
    if any(channel < 0 for channel in channels):
        _fail(ctx, payload, "--normalize-uvs values must be greater than or equal to 0.", code=2)
    return channels


def _parse_lod_screen_coverage(
    value: str | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> list[float] | None:
    if value is None:
        return None
    try:
        coverages = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        _fail(ctx, payload, "--lod-screen-coverage must be a comma-separated list of numbers.", code=2)
        raise AssertionError("unreachable") from exc
    if not coverages:
        _fail(ctx, payload, "--lod-screen-coverage must include at least one value.", code=2)
    if any(coverage <= 0.0 or coverage > 1.0 for coverage in coverages):
        _fail(ctx, payload, "--lod-screen-coverage values must be greater than 0 and no more than 1.", code=2)
    if coverages != sorted(coverages, reverse=True):
        _fail(ctx, payload, "--lod-screen-coverage values must be sorted from highest to lowest.", code=2)
    return coverages


def _lod_generator_options(
    preset: str,
    lod_values: list[float] | None,
    lod_coverages: list[float] | None,
    validate_lods: bool,
    jobs: int,
) -> LODGeneratorOptions:
    if lod_values is None and lod_coverages is None:
        return LODGeneratorOptions(preset=cast(Any, preset), validate=validate_lods, jobs=jobs)
    default_levels = LODGeneratorOptions(preset=cast(Any, preset), validate=validate_lods, jobs=jobs).levels
    ratios = lod_values if lod_values is not None else [level.target_ratio for level in default_levels]
    if lod_coverages is None:
        if len(ratios) == len(default_levels):
            coverages = [level.screen_coverage for level in default_levels]
        else:
            coverages = [max(0.01, 0.5 / (index + 1)) for index in range(len(ratios))]
    else:
        coverages = lod_coverages
    levels = tuple(
        LODLevel(screen_coverage=coverage, target_ratio=ratio)
        for coverage, ratio in zip(coverages, ratios, strict=True)
    )
    return LODGeneratorOptions(preset=cast(Any, preset), levels=levels, validate=validate_lods, jobs=jobs)


def _lod_options_for_cli(
    profile_lods: LODOptions | None,
    lod_values: list[float] | None,
    lod_coverages: list[float] | None,
    lod_mode: str,
    lod_engine_profile: str,
    lod_per_part_budget: bool,
    lod_drop_tiny_parts: bool,
    lod_tiny_part_screen_size: float,
    validate_lods: bool,
    jobs: int,
    lod_source: str,
) -> LODOptions | None:
    ratios = tuple(lod_values) if lod_values is not None else None
    if ratios is None and profile_lods is not None:
        ratios = tuple(profile_lods.ratios)
    if ratios is None:
        return None
    if lod_coverages is not None and len(lod_coverages) != len(ratios):
        raise ValueError("screen_coverage must contain one value per LOD ratio")
    return LODOptions(
        ratios=ratios,
        mode=cast(Any, lod_mode),
        engine_profile=cast(Any, lod_engine_profile),
        screen_coverage=None if lod_coverages is None else tuple(lod_coverages),
        per_part_budget=lod_per_part_budget,
        drop_tiny_parts=lod_drop_tiny_parts,
        tiny_part_screen_size=lod_tiny_part_screen_size,
        validate=validate_lods,
        jobs=jobs,
        source=cast(Any, lod_source),
    )


def _parse_filter_options(
    filters: list[str] | None,
    exclude_filters: list[str] | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> Filter | None:
    from fascat.filter import Filter, FilterExpressionError

    try:
        return Filter.from_cli(filters or [], exclude=exclude_filters or [])
    except FilterExpressionError as exc:
        _fail(ctx, payload, str(exc), code=2)
    except ValueError as exc:
        _fail(ctx, payload, str(exc), code=2)
    raise AssertionError("unreachable")


def _step_read_options(
    metadata: MetadataMode,
    pmi: PmiMode,
    *,
    design_variants: bool = False,
    design_variant_selection: tuple[str, ...] = (),
    existing_meshes: bool = True,
    multi_file: bool = False,
    material_library_paths: list[Path] | tuple[str, ...] | None = None,
    material_library_color_space: str = "auto",
    delete_free_vertices: bool = False,
    delete_lines: bool = False,
    construction_curve_policy: str = "preserve_metadata",
    construction_curve_tube_radius: float = 0.01,
    source_units: str | None = None,
    source_meters_per_unit: float | None = None,
    source_up_axis: str = "Z",
    source_handedness: str = "right",
    target_units: str | None = None,
    target_meters_per_unit: float | None = None,
    target_up_axis: str | None = None,
    target_handedness: str | None = None,
) -> StepReadOptions:
    metadata_enabled = metadata != MetadataMode.NONE
    pmi_enabled = pmi != PmiMode.NONE
    return StepReadOptions(
        metadata=metadata_enabled,
        product_metadata=metadata_enabled,
        properties=metadata_enabled,
        layers=metadata_enabled,
        validation_properties=metadata_enabled,
        pmi=pmi_enabled,
        design_variants=design_variants,
        design_variant_selection=design_variant_selection,
        existing_meshes=existing_meshes,
        multi_file=multi_file,
        material_library_paths=tuple(str(path) for path in material_library_paths or ()),
        material_library_color_space=cast(Any, material_library_color_space),
        delete_free_vertices=delete_free_vertices,
        delete_lines=delete_lines,
        construction_curve_policy=cast(Any, construction_curve_policy.replace("-", "_")),
        construction_curve_tube_radius=construction_curve_tube_radius,
        source_units=source_units,
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=cast(Any, source_up_axis),
        source_handedness=cast(Any, source_handedness),
        target_units=target_units,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=cast(Any, target_up_axis),
        target_handedness=cast(Any, target_handedness),
    )


def _metadata_export_options(metadata: MetadataMode, pmi: PmiMode) -> MetadataExportOptions:
    return MetadataExportOptions(
        mode=cast(Any, metadata.value),
        pmi=cast(Any, pmi.value.replace("-", "_")),
    )


def _profile_for_cli(
    profile: Profile,
    target_device_profile: Path | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> ConversionProfile:
    if target_device_profile is None:
        return by_name(profile.value)
    _require_existing_file(target_device_profile, "target device profile", ctx, payload)
    try:
        return profile_from_file(target_device_profile, base=profile.value)
    except Exception as exc:
        _fail(ctx, payload, f"Invalid target device profile: {exc}", code=2)
    raise AssertionError("unreachable")


def _decimate_target_for_cli(
    *,
    decimate: bool,
    criterion: str,
    requested_target: int | None,
    requested_ratio: float | None,
    profile_options: ConversionProfile,
) -> tuple[int | None, str | None]:
    if not decimate:
        return None, None
    if requested_target is not None:
        return requested_target, "user"
    if requested_ratio is not None:
        return None, "ratio"
    if criterion == "quality":
        return None, "quality_tolerances"
    if profile_options.optimize is not None and profile_options.optimize.target_triangles is not None:
        return profile_options.optimize.target_triangles, "profile_budget"
    return None, "default_ratio"


def _read_pipeline_for_cli(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> PipelineSpec:
    from fascat.pipeline_file import PipelineSpec

    _require_existing_file(path, "pipeline", ctx, payload)
    try:
        return PipelineSpec.from_file(path)
    except Exception as exc:
        _fail(ctx, payload, f"Invalid pipeline file: {exc}", code=2)
    raise AssertionError("unreachable")


def _brep_heal_options(
    *,
    heal_tolerance: float,
    group_open_shells: bool,
    cleanup_overlapping_faces: bool,
    overlap_area_ratio: float,
    remove_sliver_faces: bool,
    max_sliver_area: float,
    fail_on_open_shells: bool = False,
) -> BrepHealOptions:
    return BrepHealOptions(
        tolerance=heal_tolerance,
        group_open_shells=group_open_shells,
        remove_overlapping_faces=cleanup_overlapping_faces,
        overlap_area_ratio=overlap_area_ratio,
        remove_sliver_faces=remove_sliver_faces,
        max_sliver_area=max_sliver_area,
        fail_on_open_shells=fail_on_open_shells,
    )


def _analyze_options(
    *,
    geometry_quality: bool,
    non_manifold_edges: bool,
    open_boundaries: bool,
    self_intersections: bool,
    sliver_triangles: bool,
    tiny_parts: bool,
    draw_call_estimate: bool,
    visual_risk: bool,
) -> AnalyzeOptions:
    return AnalyzeOptions(
        non_manifold_edges=geometry_quality or non_manifold_edges,
        open_boundaries=geometry_quality or open_boundaries,
        self_intersections=geometry_quality or self_intersections,
        sliver_triangles=geometry_quality or sliver_triangles,
        tiny_parts=geometry_quality or tiny_parts,
        draw_call_estimate=geometry_quality or draw_call_estimate,
        visual_risk=geometry_quality or visual_risk,
    )


def _analysis_requested(options: AnalyzeOptions) -> bool:
    return any(
        (
            options.non_manifold_edges,
            options.open_boundaries,
            options.self_intersections,
            options.sliver_triangles,
            options.tiny_parts,
            options.draw_call_estimate,
            options.visual_risk,
        )
    )


def _metadata_summary(asset: Any) -> dict[str, int]:
    return {
        "asset": len(asset.metadata),
        "nodes": sum(len(node.metadata) for node in asset.root.walk()),
        "parts": sum(len(part.metadata) for part in asset.parts.values()),
        "materials": sum(len(material.metadata) for material in asset.materials.values()),
    }


def _pmi_summary(asset: Any) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for annotation in asset.pmi:
        kinds[annotation.kind] = kinds.get(annotation.kind, 0) + 1
    return {"count": len(asset.pmi), **{f"kind_{kind}": count for kind, count in sorted(kinds.items())}}


def _validate_cad_input(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if not _is_stdio(path) and path.suffix.lower() not in CAD_SUFFIXES:
        _fail(
            ctx,
            payload,
            f"Unsupported CAD extension: {path.suffix or '<none>'}. Use .step, .stp, .igs, .iges, .brep, or .jt.",
            code=2,
        )


def _validate_export_output(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if not _is_stdio(path) and path.suffix.lower() not in EXPORT_SUFFIXES:
        _fail(
            ctx,
            payload,
            "Unsupported export extension: "
            f"{path.suffix or '<none>'}. Use .usd, .usda, .usdc, .usdz, .gltf, .glb, .obj, .stl, or .fbx.",
            code=2,
        )
