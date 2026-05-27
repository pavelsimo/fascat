from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from fascat import profiles
from fascat.asset import Asset
from fascat.filter import Filter
from fascat.io.gltf import GLTF_SUFFIXES, runtime_dependency_report, validate_gltf
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
    MetadataExportOptions,
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
    UV0Mode,
    UV1Mode,
)
from fascat.pipeline_file import PipelineSpec
from fascat.report import timed_step

if TYPE_CHECKING:
    from fascat.analysis import AnalysisReport

USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
ExportFormat = Literal["usd", "gltf", "obj", "stl"]
_LOAD_ESTIMATE_BYTES_PER_MS = 50_000
_LOAD_ESTIMATE_VERTEX_BYTES = 32
_LOAD_ESTIMATE_TRIANGLE_INDEX_BYTES = 12


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
    effective_import_options = import_options
    if effective_import_options is None and pipeline is not None:
        effective_import_options = pipeline.import_options
    asset = (
        read_step(input_path, options=effective_import_options)
        if effective_import_options is not None
        else read_step(input_path)
    )
    if progress is not None:
        progress("source", asset.stats())
    planned_tessellation: Tessellation | None = None
    planned_stage: StageOptions | None = None
    planned_optimize: OptimizeOptions | None = None
    planned_lods: LODOptions | None = None
    if pipeline is None:
        planned_tessellation = tessellation or selected.tessellation
        planned_stage = stage or selected.stage
        planned_optimize = optimize if optimize is not None else selected.optimize
        planned_lods = lods if lods is not None else selected.lods
    _add_preflight_report(
        asset,
        output_format,
        pipeline=pipeline,
        tessellation=planned_tessellation,
        stage=planned_stage,
        bake_materials=bake_materials,
        decimate=decimate,
        optimize=planned_optimize,
        lod_generator=lod_generator,
        lods=planned_lods,
        gltf_options=gltf_options,
    )
    if pipeline is not None:
        asset = pipeline.apply(asset, progress=progress)
        if pipeline.export_metadata is not None:
            gltf_options = _with_gltf_metadata(gltf_options, pipeline.export_metadata)
            usd_options = _with_usd_metadata(usd_options, pipeline.export_metadata)
    else:
        tessellation_options = planned_tessellation
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
        optimize_options = planned_optimize
        if optimize_options is not None:
            asset = asset.optimize(optimize_options, where=where)
            if progress is not None:
                progress("optimize", asset.stats())
        lod_options = planned_lods
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
    if output_format == "gltf":
        write_options["runtime_dependencies"] = runtime_dependency_report(asset, gltf_options)
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
    _add_conversion_manifest_report(
        asset,
        output_format,
        selected,
        import_options=effective_import_options,
        pipeline=pipeline,
        tessellation=planned_tessellation,
        heal_brep=heal_brep,
        stage=planned_stage,
        merge=merge,
        explode=explode,
        replace=replace,
        scene=scene,
        bake_materials=bake_materials,
        remove_holes=remove_holes,
        remove_occluded=remove_occluded,
        decimate=decimate,
        lod_generator=lod_generator,
        optimize=planned_optimize,
        lods=planned_lods,
        write_options=write_options,
    )
    _add_workflow_summary_report(asset, output_format, write_options)
    _add_profile_budget_report(asset, selected)
    asset.report.finish(_report_stats(asset))
    return asset


def _add_preflight_report(
    asset: Asset,
    output_format: ExportFormat,
    *,
    pipeline: PipelineSpec | None,
    tessellation: Tessellation | None,
    stage: StageOptions | None,
    bake_materials: BakeMaterialOptions | None,
    decimate: DecimateOptions | None,
    optimize: OptimizeOptions | None,
    lod_generator: LODGeneratorOptions | None,
    lods: LODOptions | None,
    gltf_options: GltfExportOptions | None,
) -> None:
    checks = _preflight_checks(
        output_format,
        pipeline=pipeline,
        tessellation=tessellation,
        stage=stage,
        bake_materials=bake_materials,
        decimate=decimate,
        optimize=optimize,
        lod_generator=lod_generator,
        lods=lods,
        gltf_options=gltf_options,
    )
    warnings = [str(item["message"]) for item in checks if item["status"] == "warning"]
    before = _report_stats(asset)
    after = dict(before)
    after["preflight_checks_total"] = len(checks)
    for status in ("ok", "warning", "skipped", "info"):
        after[f"preflight_checks_{status}"] = sum(1 for item in checks if item["status"] == status)
    for warning in warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "preflight",
        options={"style": "unity_asset_transformer", "checks": checks},
        before=before,
        after=after,
        warnings=warnings,
    )


def _preflight_checks(
    output_format: ExportFormat,
    *,
    pipeline: PipelineSpec | None,
    tessellation: Tessellation | None,
    stage: StageOptions | None,
    bake_materials: BakeMaterialOptions | None,
    decimate: DecimateOptions | None,
    optimize: OptimizeOptions | None,
    lod_generator: LODGeneratorOptions | None,
    lods: LODOptions | None,
    gltf_options: GltfExportOptions | None,
) -> list[dict[str, object]]:
    if pipeline is not None:
        checks = _pipeline_preflight_checks(pipeline)
        bake_planned = any(step.op == "bake_materials" for step in pipeline.steps)
    else:
        checks = _direct_preflight_checks(
            tessellation=tessellation,
            stage=stage,
            bake_materials=bake_materials,
            decimate=decimate,
            optimize=optimize,
            lod_generator=lod_generator,
            lods=lods,
        )
        bake_planned = bake_materials is not None
    checks.extend(_export_preflight_checks(output_format, gltf_options, bake_planned=bake_planned))
    return checks


def _pipeline_preflight_checks(pipeline: PipelineSpec) -> list[dict[str, object]]:
    checks = [
        _preflight_item(
            code=str(advisory["code"]),
            status="warning",
            stage="workflow",
            operation=str(advisory["operation"]),
            message=str(advisory["message"]),
            step=cast(int, advisory["step"]),
        )
        for advisory in pipeline.advisories()
    ]
    tessellate_steps = [step for step in pipeline.steps if step.op == "tessellate"]
    if not tessellate_steps:
        checks.append(
            _preflight_item(
                code="brep_patch_cleanup_not_planned",
                status="skipped",
                stage="import_cleanup",
                operation="tessellate",
                message="no tessellation step is planned, so BREP patch cleanup does not apply",
            )
        )
    elif any(bool(step.values.get("keep_brep", False)) for step in tessellate_steps):
        checks.append(
            _preflight_item(
                code="brep_patches_retained",
                status="warning",
                stage="import_cleanup",
                operation="tessellate",
                message="BREP patch data is retained after tessellation; disable keep_brep before runtime export unless CAD surfaces are still needed",
            )
        )
    else:
        checks.append(
            _preflight_item(
                code="brep_patch_cleanup_planned",
                status="ok",
                stage="import_cleanup",
                operation="tessellate",
                message="tessellation is planned with BREP patch cleanup enabled",
            )
        )

    stage_steps = [step for step in pipeline.steps if step.op == "stage"]
    if not stage_steps:
        checks.append(
            _preflight_item(
                code="orientation_not_planned",
                status="warning",
                stage="orientation",
                operation="stage",
                message="no stage step is planned; normal generation and orientation-sensitive attributes will not be prepared",
            )
        )
    elif any(_stage_values_prepare_normals(step.values) for step in stage_steps):
        checks.append(
            _preflight_item(
                code="orientation_preparation_planned",
                status="ok",
                stage="orientation",
                operation="stage",
                message="stage is planned with normal preparation enabled",
            )
        )
    else:
        checks.append(
            _preflight_item(
                code="orientation_not_planned",
                status="warning",
                stage="orientation",
                operation="stage",
                message="stage disables normals; face and normal orientation should be verified before export",
            )
        )
    return checks


def _direct_preflight_checks(
    *,
    tessellation: Tessellation | None,
    stage: StageOptions | None,
    bake_materials: BakeMaterialOptions | None,
    decimate: DecimateOptions | None,
    optimize: OptimizeOptions | None,
    lod_generator: LODGeneratorOptions | None,
    lods: LODOptions | None,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    if tessellation is None:
        checks.append(
            _preflight_item(
                code="brep_patch_cleanup_not_planned",
                status="skipped",
                stage="import_cleanup",
                operation="tessellate",
                message="no tessellation step is planned, so BREP patch cleanup does not apply",
            )
        )
    elif tessellation.keep_brep:
        checks.append(
            _preflight_item(
                code="brep_patches_retained",
                status="warning",
                stage="import_cleanup",
                operation="tessellate",
                message="BREP patch data is retained after tessellation; disable keep_brep before runtime export unless CAD surfaces are still needed",
            )
        )
    else:
        checks.append(
            _preflight_item(
                code="brep_patch_cleanup_planned",
                status="ok",
                stage="import_cleanup",
                operation="tessellate",
                message="tessellation is planned with BREP patch cleanup enabled",
            )
        )

    checks.append(_stage_orientation_preflight(stage))
    checks.append(_stage_tangent_preflight(stage))
    checks.append(_ao_bake_preflight(stage, bake_materials))

    lod_planned = lod_generator is not None or lods is not None
    if lod_planned and optimize is None and decimate is None:
        checks.append(
            _preflight_item(
                code="lods_without_lod0_optimization",
                status="warning",
                stage="lod_generation",
                operation="lods",
                message="LOD generation is planned without LOD0 optimization",
            )
        )
    elif lod_planned:
        checks.append(
            _preflight_item(
                code="lod0_optimization_before_lods",
                status="ok",
                stage="lod_generation",
                operation="optimize,lods",
                message="LOD0 optimization is planned before LOD generation",
            )
        )
    else:
        checks.append(
            _preflight_item(
                code="lod_generation_not_requested",
                status="skipped",
                stage="lod_generation",
                operation="lods",
                message="LOD generation was not requested",
            )
        )
    return checks


def _export_preflight_checks(
    output_format: ExportFormat,
    gltf_options: GltfExportOptions | None,
    *,
    bake_planned: bool,
) -> list[dict[str, object]]:
    if output_format != "gltf":
        return [
            _preflight_item(
                code="gltf_geometry_compression_not_applicable",
                status="skipped",
                stage="export",
                operation="export",
                message="glTF geometry compression checks do not apply to this export format",
            ),
            _preflight_item(
                code="texture_compression_not_applicable",
                status="skipped",
                stage="export",
                operation="export",
                message="KTX2/Basis texture export checks do not apply to this export format",
            ),
        ]

    geometry_compression = gltf_options is not None and (gltf_options.quantize or gltf_options.meshopt)
    texture_message = (
        "baked textures are planned for glTF export, but KTX2/Basis output is unavailable and requests are rejected"
        if bake_planned
        else "KTX2/Basis texture output is unavailable; no texture-producing bake step is planned"
    )
    return [
        _preflight_item(
            code="gltf_geometry_compression_planned"
            if geometry_compression
            else "gltf_geometry_compression_not_requested",
            status="ok" if geometry_compression else "info",
            stage="export",
            operation="write",
            message="glTF geometry compression or quantization is planned"
            if geometry_compression
            else "glTF export has no runtime geometry compression requested",
        ),
        _preflight_item(
            code="texture_compression_backend_missing",
            status="warning" if bake_planned else "info",
            stage="export",
            operation="write",
            message=texture_message,
        ),
    ]


def _stage_orientation_preflight(stage: StageOptions | None) -> dict[str, object]:
    if stage is None:
        return _preflight_item(
            code="orientation_not_planned",
            status="warning",
            stage="orientation",
            operation="stage",
            message="no stage step is planned; normal generation and orientation-sensitive attributes will not be prepared",
        )
    if not stage.normals or stage.normal_mode == "none":
        return _preflight_item(
            code="orientation_not_planned",
            status="warning",
            stage="orientation",
            operation="stage",
            message="stage disables normals; face and normal orientation should be verified before export",
        )
    return _preflight_item(
        code="orientation_preparation_planned",
        status="ok",
        stage="orientation",
        operation="stage",
        message="stage is planned with normal preparation enabled",
    )


def _stage_tangent_preflight(stage: StageOptions | None) -> dict[str, object]:
    if stage is None or not stage.tangents:
        return _preflight_item(
            code="tangent_generation_not_requested",
            status="skipped",
            stage="uv_preparation",
            operation="stage",
            message="tangent generation was not requested",
        )
    uv_channel = stage.tangent_uv_channel
    if _stage_has_uv_channel(stage, uv_channel):
        return _preflight_item(
            code="tangent_uv_available",
            status="ok",
            stage="uv_preparation",
            operation="stage",
            message=f"tangent generation has UV{uv_channel} available",
        )
    return _preflight_item(
        code=f"tangents_without_uv{uv_channel}",
        status="warning",
        stage="uv_preparation",
        operation="stage",
        message=f"tangents are requested before UV{uv_channel} is available",
    )


def _ao_bake_preflight(stage: StageOptions | None, bake_materials: BakeMaterialOptions | None) -> dict[str, object]:
    if bake_materials is None or "ao" not in bake_materials.bake:
        return _preflight_item(
            code="ao_bake_not_requested",
            status="skipped",
            stage="material_baking",
            operation="bake_materials",
            message="ambient occlusion baking was not requested",
        )
    generates_bake_uv = bake_materials.force_uv_generation and bake_materials.uv_channel == 1
    if _stage_has_uv_channel(stage, 1) or generates_bake_uv:
        return _preflight_item(
            code="ao_bake_uv1_available",
            status="ok",
            stage="material_baking",
            operation="bake_materials",
            message="ambient occlusion baking has UV1 available or explicitly generated",
        )
    return _preflight_item(
        code="ao_bake_without_uv1",
        status="warning",
        stage="material_baking",
        operation="bake_materials",
        message="ambient occlusion baking is requested before UV1 is available",
    )


def _stage_has_uv_channel(stage: StageOptions | None, channel: int) -> bool:
    if stage is None:
        return False
    if channel == 0:
        return stage.uv0 not in {None, "none"}
    if channel == 1:
        return stage.uv1 not in {None, "none"} and (stage.uv1 != "copy_uv0" or stage.uv0 not in {None, "none"})
    return False


def _stage_values_prepare_normals(values: dict[str, object]) -> bool:
    normal_mode = str(values.get("normal_mode", "smooth")).replace("-", "_")
    return bool(values.get("normals", True)) and normal_mode != "none"


def _preflight_item(
    *,
    code: str,
    status: str,
    stage: str,
    operation: str,
    message: str,
    step: int | None = None,
) -> dict[str, object]:
    item: dict[str, object] = {
        "code": code,
        "status": status,
        "stage": stage,
        "operation": operation,
        "message": message,
    }
    if step is not None:
        item["step"] = step
    return item


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


def _add_profile_budget_report(asset: Asset, profile: ConversionProfile) -> None:
    budget = profile.budget
    if budget is None:
        return
    before = {**_report_stats(asset), **asset.draw_call_breakdown()}
    after = dict(before)
    options = {"profile": profile.name, **budget.to_dict()}
    warnings: list[str] = []

    violations = 0
    if budget.target_fps is not None:
        after["profile_target_fps"] = budget.target_fps
    if budget.max_triangles is not None:
        after["profile_triangle_budget"] = budget.max_triangles
        over = max(0, asset.triangle_count - budget.max_triangles)
        after["profile_triangles_over_budget"] = over
        if over:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: triangles {asset.triangle_count} > {budget.max_triangles}"
            )
    if budget.unity_reference_triangles is not None:
        reference_min, reference_max = budget.unity_reference_triangles
        after["profile_unity_reference_triangle_min"] = reference_min
        after["profile_unity_reference_triangle_max"] = reference_max
        if budget.max_triangles is not None:
            after["profile_triangle_budget_below_unity_reference_min"] = max(0, reference_min - budget.max_triangles)
            after["profile_triangle_budget_over_unity_reference_max"] = max(0, budget.max_triangles - reference_max)
    if budget.max_vertices is not None:
        after["profile_vertex_budget"] = budget.max_vertices
        over = max(0, asset.vertex_count - budget.max_vertices)
        after["profile_vertices_over_budget"] = over
        if over:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: vertices {asset.vertex_count} > {budget.max_vertices}"
            )
    if budget.max_vertices_per_mesh is not None:
        mesh_vertex_counts = _mesh_vertex_counts(asset)
        largest = max(mesh_vertex_counts, default=0)
        over_count = sum(1 for vertex_count in mesh_vertex_counts if vertex_count > budget.max_vertices_per_mesh)
        after["profile_max_vertices_per_mesh_budget"] = budget.max_vertices_per_mesh
        after["profile_largest_mesh_vertices"] = largest
        after["profile_meshes_over_vertex_budget"] = over_count
        if over_count:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: {over_count} mesh(es) exceed "
                f"{budget.max_vertices_per_mesh} vertices (largest {largest})"
            )
    if budget.max_texture_resolution is not None:
        texture_resolutions = _material_texture_resolutions(asset)
        largest = max(texture_resolutions, default=0)
        over_count = sum(1 for resolution in texture_resolutions if resolution > budget.max_texture_resolution)
        after["profile_texture_resolution_budget"] = budget.max_texture_resolution
        after["profile_largest_texture_resolution"] = largest
        after["profile_texture_sets_with_resolution"] = len(texture_resolutions)
        after["profile_textures_over_resolution_budget"] = over_count
        if over_count:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: {over_count} texture set(s) exceed "
                f"{budget.max_texture_resolution}px (largest {largest}px)"
            )
    if budget.max_texture_memory_mb is not None:
        texture_count, estimated_bytes = _estimated_texture_memory(asset)
        budget_bytes = budget.max_texture_memory_mb * 1_000_000
        over = max(0, estimated_bytes - budget_bytes)
        after["profile_texture_memory_budget_bytes"] = budget_bytes
        after["profile_texture_memory_texture_count"] = texture_count
        after["profile_estimated_texture_memory_bytes"] = estimated_bytes
        after["profile_texture_memory_over_budget_bytes"] = over
        if over:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: estimated texture memory "
                f"{estimated_bytes} bytes > {budget_bytes} bytes"
            )
    if budget.max_load_time_ms is not None:
        load_time = _estimated_load_time(asset)
        over = max(0, load_time["estimated_ms"] - budget.max_load_time_ms)
        after["profile_load_time_budget_ms"] = budget.max_load_time_ms
        after["profile_estimated_load_time_ms"] = load_time["estimated_ms"]
        after["profile_load_time_file_bytes"] = load_time["file_bytes"]
        after["profile_load_time_geometry_bytes"] = load_time["geometry_bytes"]
        after["profile_load_time_texture_bytes"] = load_time["texture_bytes"]
        after["profile_load_time_over_budget_ms"] = over
        if over:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: estimated load time "
                f"{load_time['estimated_ms']}ms > {budget.max_load_time_ms}ms"
            )
    if budget.max_draw_calls is not None:
        after["profile_draw_call_budget"] = budget.max_draw_calls
        over = max(0, asset.draw_call_count - budget.max_draw_calls)
        after["profile_draw_calls_over_budget"] = over
        if over:
            violations += 1
            warnings.append(
                f"profile budget exceeded for {profile.name}: draw calls {asset.draw_call_count} > {budget.max_draw_calls}"
            )
    if budget.unity_reference_draw_calls is not None:
        after["profile_unity_reference_draw_call_budget"] = budget.unity_reference_draw_calls
        if budget.max_draw_calls is not None:
            after["profile_draw_call_budget_over_unity_reference"] = max(
                0,
                budget.max_draw_calls - budget.unity_reference_draw_calls,
            )

    after["profile_budget_violations"] = violations
    for warning in warnings:
        asset.report.add_warning(warning)
    asset.report.add_step("profile_budget", options=options, before=before, after=after, warnings=warnings)


def _add_workflow_summary_report(asset: Asset, output_format: ExportFormat, write_options: dict[str, object]) -> None:
    stages = _workflow_summary_stages(asset, output_format, write_options)
    before = _report_stats(asset)
    after = dict(before)
    after["workflow_stages_total"] = len(stages)
    for status in ("run", "skipped", "blocked"):
        after[f"workflow_stages_{status}"] = sum(1 for stage in stages if stage["status"] == status)
    for level in ("exact", "approximate", "metadata_only"):
        after[f"workflow_stages_{level}"] = sum(1 for stage in stages if stage["level"] == level)
    asset.report.add_step(
        "workflow_summary",
        options={"style": "unity_asset_transformer", "stages": stages},
        before=before,
        after=after,
    )


def _add_conversion_manifest_report(
    asset: Asset,
    output_format: ExportFormat,
    profile: ConversionProfile,
    *,
    import_options: StepReadOptions | None,
    pipeline: PipelineSpec | None,
    tessellation: Tessellation | None,
    heal_brep: BrepHealOptions | None,
    stage: StageOptions | None,
    merge: MergeOptions | None,
    explode: ExplodeOptions | None,
    replace: ReplaceOptions | None,
    scene: SceneOptimizeOptions | None,
    bake_materials: BakeMaterialOptions | None,
    remove_holes: RemoveHolesOptions | None,
    remove_occluded: RemoveOccludedOptions | None,
    decimate: DecimateOptions | None,
    lod_generator: LODGeneratorOptions | None,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    write_options: dict[str, object],
) -> None:
    direct_steps = {
        "tessellation": _manifest_options(tessellation),
        "brep_heal": _manifest_options(heal_brep),
        "repair": profile.repair.to_dict(),
        "stage": _manifest_options(stage),
        "merge": _manifest_options(merge),
        "explode": _manifest_options(explode),
        "replace": _manifest_options(replace),
        "scene": _manifest_options(scene),
        "bake_materials": _manifest_options(bake_materials),
        "remove_holes": _manifest_options(remove_holes),
        "remove_occluded": _manifest_options(remove_occluded),
        "decimate": _manifest_options(decimate),
        "lod_generator": _manifest_options(lod_generator),
        "optimize": _manifest_options(optimize),
        "lods": _manifest_options(lods),
    }
    manifest: dict[str, object] = {
        "style": "resolved_conversion_manifest",
        "mode": "pipeline" if pipeline is not None else "direct",
        "profile": profile.to_dict(),
        "import": (import_options or StepReadOptions()).to_dict(),
        "pipeline": None if pipeline is None else pipeline.to_dict(),
        "steps": direct_steps if pipeline is None else {},
        "export": {"output_format": output_format, "options": write_options},
    }
    before = _report_stats(asset)
    after = dict(before)
    after["conversion_manifest_sections"] = 5
    after["conversion_manifest_direct_steps"] = sum(1 for value in direct_steps.values() if value is not None)
    after["conversion_manifest_pipeline_steps"] = 0 if pipeline is None else len(pipeline.steps)
    asset.report.add_step(
        "conversion_manifest",
        options=manifest,
        before=before,
        after=after,
    )


def _manifest_options(options: object | None) -> dict[str, object] | None:
    if options is None:
        return None
    to_dict = getattr(options, "to_dict", None)
    if callable(to_dict):
        return cast(dict[str, object], to_dict())
    return cast(dict[str, object], options)


def _workflow_summary_stages(
    asset: Asset,
    output_format: ExportFormat,
    write_options: dict[str, object],
) -> list[dict[str, str]]:
    steps = {step.name: step for step in asset.report.steps}
    stages: list[dict[str, str]] = []

    def add(stage: str, status: str, level: str, operation: str, message: str) -> None:
        stages.append(
            {
                "stage": stage,
                "status": status,
                "level": level,
                "operation": operation,
                "message": message,
            }
        )

    add(
        "import",
        "run" if "import" in steps else "skipped",
        "exact" if "import" in steps else "not_applicable",
        "import",
        "STEP hierarchy, metadata, materials, and BREP handles were read when available"
        if "import" in steps
        else "input import did not run in this report",
    )

    cleanup_ops = [name for name in ("heal_brep", "repair") if name in steps]
    add(
        "import_cleanup",
        "run" if cleanup_ops else "skipped",
        _workflow_level_for_steps(steps, cleanup_ops),
        ",".join(cleanup_ops) if cleanup_ops else "heal_brep,repair",
        "BREP or mesh cleanup ran before staging" if cleanup_ops else "no BREP or mesh cleanup step ran",
    )

    add(
        "tessellation",
        "run" if "tessellate" in steps else "skipped",
        "exact" if "tessellate" in steps else "not_applicable",
        "tessellate",
        "BREP tessellation ran" if "tessellate" in steps else "no tessellation step ran",
    )

    stage_step = steps.get("stage")
    add(
        "orientation",
        "run" if stage_step is not None else "skipped",
        "exact" if stage_step is not None else "not_applicable",
        "stage",
        "staging handled materials, normals, tangents, or orientation-sensitive attributes"
        if stage_step is not None
        else "no staging step ran",
    )

    uv_status = stage_step is not None and _stage_step_prepares_uvs(stage_step.options)
    add(
        "uv_preparation",
        "run" if uv_status else "skipped",
        _uv_preparation_level(stage_step.options) if uv_status and stage_step is not None else "not_applicable",
        "stage",
        "staging prepared UV channels or atlas metadata" if uv_status else "no UV preparation was requested",
    )

    add(
        "material_baking",
        "run" if "bake_materials" in steps else "skipped",
        "approximate" if "bake_materials" in steps else "not_applicable",
        "bake_materials",
        "material baking emitted constant embedded texture maps"
        if "bake_materials" in steps
        else "material baking was not requested",
    )

    optimization_ops = [
        name
        for name in (
            "merge",
            "replace",
            "optimize_scene",
            "remove_holes",
            "remove_occluded",
            "decimate",
            "optimize",
        )
        if name in steps
    ]
    add(
        "optimization",
        "run" if optimization_ops else "skipped",
        _workflow_level_for_steps(steps, optimization_ops),
        ",".join(optimization_ops) if optimization_ops else "optimize",
        "LOD0 optimization or draw-call reduction ran" if optimization_ops else "no optimization step ran",
    )

    lod_ops = [name for name in ("run_lod_generators", "lods") if name in steps]
    add(
        "lod_generation",
        "run" if lod_ops else "skipped",
        _workflow_level_for_steps(steps, lod_ops),
        ",".join(lod_ops) if lod_ops else "lods",
        "LOD meshes were generated" if lod_ops else "LOD generation was not requested",
    )

    compression_ops = _export_compression_ops(output_format, write_options)
    add(
        "export_compression",
        "run" if compression_ops else "skipped",
        "exact" if compression_ops else "not_applicable",
        ",".join(compression_ops) if compression_ops else "export",
        "runtime export compression or quantization was requested"
        if compression_ops
        else "runtime export compression was not requested",
    )

    add(
        "export",
        "run" if "write" in steps else "skipped",
        "exact" if "write" in steps else "not_applicable",
        "write",
        f"{output_format} output was written" if "write" in steps else "output write did not run",
    )

    return stages


def _workflow_level_for_steps(steps: Mapping[str, object], names: list[str]) -> str:
    if not names:
        return "not_applicable"
    if any(_workflow_step_level(steps[name]) == "approximate" for name in names):
        return "approximate"
    if any(_workflow_step_level(steps[name]) == "metadata_only" for name in names):
        return "metadata_only"
    return "exact"


def _workflow_step_level(step: object) -> str:
    name = getattr(step, "name", "")
    options = getattr(step, "options", {})
    warnings = getattr(step, "warnings", [])
    if name in {"bake_materials", "remove_holes", "remove_occluded"}:
        return "approximate"
    if name == "heal_brep" and warnings:
        return "approximate"
    if name == "decimate" and isinstance(options, dict) and options.get("criterion") == "quality":
        return "approximate"
    if name == "stage" and isinstance(options, dict) and _uv_preparation_level(options) == "metadata_only":
        return "metadata_only"
    return "exact"


def _stage_step_prepares_uvs(options: dict[str, object]) -> bool:
    atlas = options.get("atlas")
    unwrap = options.get("unwrap")
    return (
        options.get("uv0") != "none"
        or options.get("uv1") not in {None, "none"}
        or bool(options.get("normalize_uvs"))
        or (isinstance(atlas, dict) and bool(atlas.get("enabled")))
        or (
            isinstance(unwrap, dict)
            and any(
                unwrap.get(key) not in {None, "default"}
                for key in ("texel_density", "max_stretch", "method", "iterations", "tolerance")
            )
        )
    )


def _uv_preparation_level(options: dict[str, object]) -> str:
    atlas = options.get("atlas")
    if isinstance(atlas, dict) and bool(atlas.get("enabled")):
        return "metadata_only"
    return "exact"


def _export_compression_ops(output_format: ExportFormat, write_options: dict[str, object]) -> list[str]:
    if output_format != "gltf":
        return []
    result: list[str] = []
    if bool(write_options.get("quantize")):
        result.append("KHR_mesh_quantization")
    if bool(write_options.get("meshopt")):
        result.append("EXT_meshopt_compression")
    return result


def _mesh_vertex_counts(asset: Asset) -> list[int]:
    counts: list[int] = []
    for part in asset.parts.values():
        if part.mesh is not None:
            counts.append(part.mesh.vertex_count)
        for lod_mesh in part.lod_meshes:
            counts.append(lod_mesh.vertex_count)
    return counts


def _material_texture_resolutions(asset: Asset) -> list[int]:
    return [resolution for resolution, _texture_count in _material_texture_summaries(asset)]


def _estimated_texture_memory(asset: Asset) -> tuple[int, int]:
    texture_count = 0
    estimated_bytes = 0
    for resolution, count in _material_texture_summaries(asset):
        texture_count += count
        estimated_bytes += resolution * resolution * 4 * count
    return texture_count, estimated_bytes


def _estimated_load_time(asset: Asset) -> dict[str, int]:
    _texture_count, texture_bytes = _estimated_texture_memory(asset)
    geometry_bytes = _estimated_geometry_bytes(asset)
    file_bytes = _output_file_size_bytes(asset)
    estimated_ms = _ceil_div(file_bytes + geometry_bytes + texture_bytes, _LOAD_ESTIMATE_BYTES_PER_MS)
    estimated_ms += asset.draw_call_count
    return {
        "estimated_ms": estimated_ms,
        "file_bytes": file_bytes,
        "geometry_bytes": geometry_bytes,
        "texture_bytes": texture_bytes,
    }


def _estimated_geometry_bytes(asset: Asset) -> int:
    vertex_count = 0
    triangle_count = 0
    for part in asset.parts.values():
        if part.mesh is not None:
            vertex_count += part.mesh.vertex_count
            triangle_count += part.mesh.triangle_count
        for lod_mesh in part.lod_meshes:
            vertex_count += lod_mesh.vertex_count
            triangle_count += lod_mesh.triangle_count
    return vertex_count * _LOAD_ESTIMATE_VERTEX_BYTES + triangle_count * _LOAD_ESTIMATE_TRIANGLE_INDEX_BYTES


def _output_file_size_bytes(asset: Asset) -> int:
    for step in reversed(asset.report.steps):
        if step.name == "write":
            size = step.after.get("file_size_bytes")
            return size if isinstance(size, int) and not isinstance(size, bool) and size > 0 else 0
    return 0


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _material_texture_summaries(asset: Asset) -> list[tuple[int, int]]:
    summaries: list[tuple[int, int]] = []
    for material in asset.materials.values():
        for key in ("baked_texture_resolution", "maps_resolution"):
            resolution = _metadata_positive_int(material.metadata.get(key))
            if resolution is not None:
                texture_count = max(1, _baked_texture_count(material.metadata.get("baked_maps")))
                summaries.append((resolution, texture_count))
                break
    return summaries


def _baked_texture_count(value: object) -> int:
    if not isinstance(value, str):
        return 0
    maps = {item.strip() for item in value.split(",") if item.strip()}
    count = 0
    if {"base_color", "opacity"} & maps:
        count += 1
    if {"metallic", "roughness"} & maps:
        count += 1
    if "normal" in maps:
        count += 1
    if "ao" in maps:
        count += 1
    if "emissive" in maps:
        count += 1
    return count


def _metadata_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


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


def _with_gltf_metadata(
    options: GltfExportOptions | None,
    metadata: MetadataExportOptions,
) -> GltfExportOptions:
    if options is None:
        return GltfExportOptions(metadata=metadata)
    return GltfExportOptions(
        quantize=options.quantize,
        meshopt=options.meshopt,
        draco=options.draco,
        texture_compression=options.texture_compression,
        file_size_budget_mb=options.file_size_budget_mb,
        metadata=metadata,
    )


def _with_usd_metadata(
    options: UsdExportOptions | None,
    metadata: MetadataExportOptions,
) -> UsdExportOptions:
    if options is None:
        return UsdExportOptions(metadata=metadata)
    return UsdExportOptions(
        package=options.package,
        file_size_budget_mb=options.file_size_budget_mb,
        metadata=metadata,
    )


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
    return UsdExportOptions(
        package="usdz",
        file_size_budget_mb=options.file_size_budget_mb,
        metadata=options.metadata,
    )


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
    sag_ratio: float | None = None,
    angle: float = 15.0,
    relative: bool = True,
    min_edge_length: float | None = None,
    max_edge_length: float | None = None,
    max_polygon_length: float | None = None,
    preserve_boundaries: bool = True,
    curvature_adaptive: bool = False,
    avoid_skinny_triangles: bool = False,
    quality_report: bool = False,
    free_edge_report: bool = False,
    create_normals: bool = True,
    keep_brep: bool = False,
    reuse_existing_meshes: bool = True,
    part_settings: dict[str, dict[str, object]] | None = None,
    where: Filter | None = None,
) -> Asset:
    return asset.tessellate(
        Tessellation(
            sag=sag,
            sag_ratio=sag_ratio,
            angle=angle,
            relative=relative,
            min_edge_length=min_edge_length,
            max_edge_length=max_edge_length,
            max_polygon_length=max_polygon_length,
            preserve_boundaries=preserve_boundaries,
            curvature_adaptive=curvature_adaptive,
            avoid_skinny_triangles=avoid_skinny_triangles,
            quality_report=quality_report,
            free_edge_report=free_edge_report,
            create_normals=create_normals,
            keep_brep=keep_brep,
            reuse_existing_meshes=reuse_existing_meshes,
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
    tangent_uv_channel: int = 0,
    override_tangents: bool = False,
    validate_normals: bool = False,
    unwrap: UnwrapOptions | None = None,
    atlas: AtlasOptions | None = None,
    uv0: UV0Mode = "box",
    uv1: UV1Mode | None = None,
    normalize_uvs: tuple[int, ...] = (),
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
            tangent_uv_channel=tangent_uv_channel,
            override_tangents=override_tangents,
            validate_normals=validate_normals,
            unwrap=unwrap or UnwrapOptions(),
            atlas=atlas or AtlasOptions(),
            uv0=uv0,
            uv1=uv1,
            normalize_uvs=normalize_uvs,
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
