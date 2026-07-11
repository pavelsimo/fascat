from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from fascat.asset import Asset, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.ops.parallel import parallel_map
from fascat.options import LODGeneratorOptions, LODOptions

_RECOMMENDED_MAX_LOD_LEVELS = 4
_CLOSE_VIEW_LOD1_MIN_RATIO = 0.4
_CLOSE_VIEW_LOD2_MIN_RATIO = 0.2
_FAR_LOD_RATIO_THRESHOLD = 0.15
_FAR_LOD_SCREEN_COVERAGE_THRESHOLD = 0.1
_MIN_RECOMMENDED_LOD_RATIO = 0.01


def run_lod_generators_asset(
    asset: Asset,
    options: LODGeneratorOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    ratios = tuple(level.target_ratio for level in options.levels)
    screen_coverage = tuple(level.screen_coverage for level in options.levels)
    switch_distance_overrides = tuple(level.switch_distance_override for level in options.levels)
    result = build_lods(
        asset,
        LODOptions(
            ratios=ratios,
            screen_coverage=screen_coverage,
            switch_distance_overrides=switch_distance_overrides,
            jobs=options.jobs,
        ),
        selected_part_ids=selected_part_ids,
    )
    coverage = ",".join(f"{level.screen_coverage:.9g}" for level in options.levels)
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.lod_meshes:
            part.metadata = {
                **part.metadata,
                "lod_generator_preset": options.preset,
                "lod_screen_coverage": coverage,
                "lod_output": options.output,
            }
    result.metadata["lod_generator_preset"] = options.preset
    result.metadata["lod_generator_output"] = options.output
    if options.validate:
        _validate_lod_monotonicity(result, selected_part_ids=selected_part_ids, allow=options.allow_non_monotonic)
    return result


def _validate_lod_monotonicity(
    asset: Asset,
    *,
    selected_part_ids: set[str] | None,
    allow: bool,
) -> None:
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None or not part.lod_meshes:
            continue
        counts = [part.mesh.triangle_count, *[mesh.triangle_count for mesh in part.lod_meshes]]
        if counts != sorted(counts, reverse=True):
            message = f"LOD triangles are not monotonic for part {part.id}"
            if allow:
                asset.report.add_warning(message)
            else:
                raise ValueError(message)


@dataclass(frozen=True)
class _LodPartResult:
    part_id: str
    lod_meshes: list[Mesh]
    metadata: dict[str, object]
    warnings: tuple[str, ...] = ()
    generated: bool = False
    skipped: bool = False
    source_vertices: int = 0
    source_triangles: int = 0
    source_mesh_bytes: int = 0
    added_vertices: int = 0
    added_triangles: int = 0
    added_mesh_bytes: int = 0
    omitted_tiny_parts: int = 0
    reused_instance_levels: int = 0
    material_merged_levels: int = 0
    texture_baked_levels: int = 0
    culling_changed_levels: int = 0


def build_lods(asset: Asset, options: LODOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    screen_coverage = _screen_coverage(options)
    export_mode = _lod_export_mode(options)
    level_policy_advisories = _level_policy_advisories(options, screen_coverage)
    occurrence_counts = _occurrence_counts_by_part(result)
    generated_parts = 0
    skipped_parts = 0
    source_vertices = 0
    source_triangles = 0
    source_mesh_bytes = 0
    added_vertices = 0
    added_triangles = 0
    added_mesh_bytes = 0
    retained_vertices = 0
    retained_triangles = 0
    retained_mesh_bytes = 0
    omitted_tiny_parts = 0
    reused_instance_levels = 0
    material_merged_levels = 0
    texture_baked_levels = 0
    culling_changed_levels = 0
    part_ids = [part.id for part in result.parts.values() if selected_part_ids is None or part.id in selected_part_ids]

    generate_part_ids: list[str] = []
    for part_id in part_ids:
        part = result.parts[part_id]
        if options.source == "generated" or not part.lod_meshes:
            if options.source != "imported":
                generate_part_ids.append(part_id)
            continue
        if options.source == "auto" and not _usable_imported_chain(part):
            result.report.add_warning(
                f"Imported LOD chain is invalid for part {part.name}; generating a fallback chain"
            )
            generate_part_ids.append(part_id)
            continue
        _apply_imported_coverage(part, options, result)
        part.metadata["lod_status"] = "retained_imported"
        part.metadata["lod_source"] = "imported"
        if part.mesh is not None:
            source_vertices += part.mesh.vertex_count
            source_triangles += part.mesh.triangle_count
            source_mesh_bytes += _mesh_payload_bytes(part.mesh)
        part_retained_vertices = sum(mesh.vertex_count for mesh in part.lod_meshes)
        part_retained_triangles = sum(mesh.triangle_count for mesh in part.lod_meshes)
        part_retained_mesh_bytes = sum(_mesh_payload_bytes(mesh) for mesh in part.lod_meshes)
        retained_vertices += part_retained_vertices
        retained_triangles += part_retained_triangles
        retained_mesh_bytes += part_retained_mesh_bytes
        part.metadata["lod_retained_vertices"] = str(part_retained_vertices)
        part.metadata["lod_retained_triangles"] = str(part_retained_triangles)
        part.metadata["lod_retained_mesh_bytes"] = str(part_retained_mesh_bytes)
        part.metadata["lod_added_vertices"] = "0"
        part.metadata["lod_added_triangles"] = "0"
        part.metadata["lod_added_mesh_bytes"] = "0"

    payloads = [
        _LodBuildPayload(
            part=result.parts[part_id].copy(keep_source=False),
            options=options,
            screen_coverage=screen_coverage,
            export_mode=export_mode,
            level_policy_advisories=level_policy_advisories,
            part_occurrences=occurrence_counts.get(part_id, 0),
        )
        for part_id in generate_part_ids
    ]

    for built in parallel_map(payloads, _lod_build_worker, jobs=options.jobs, executor="process"):
        part = result.parts[built.part_id]
        part.lod_meshes = built.lod_meshes
        part.metadata = built.metadata
        for warning in built.warnings:
            result.report.add_warning(warning)
        generated_parts += int(built.generated)
        skipped_parts += int(built.skipped)
        source_vertices += built.source_vertices
        source_triangles += built.source_triangles
        source_mesh_bytes += built.source_mesh_bytes
        added_vertices += built.added_vertices
        added_triangles += built.added_triangles
        added_mesh_bytes += built.added_mesh_bytes
        omitted_tiny_parts += built.omitted_tiny_parts
        reused_instance_levels += built.reused_instance_levels
        material_merged_levels += built.material_merged_levels
        texture_baked_levels += built.texture_baked_levels
        culling_changed_levels += built.culling_changed_levels
    scene_proxy = (
        _build_scene_far_proxy(result, options, selected_part_ids=selected_part_ids)
        if options.scene_far_proxy
        else None
    )
    result.metadata["lod_mode"] = options.mode
    result.metadata["lod_export_mode"] = export_mode
    result.metadata["lod_engine_profile"] = options.engine_profile
    result.metadata["lod_source_policy"] = options.source
    result.metadata["lod_far_lod_bake"] = str(options.far_lod_bake).lower()
    result.metadata["lod_scene_far_proxy"] = (
        "created" if scene_proxy is not None else "no_meshes" if options.scene_far_proxy else "not_requested"
    )
    result.metadata["lod_screen_coverage"] = ",".join(f"{value:.9g}" for value in screen_coverage)
    result.metadata["lod_level_switch_distance_sources"] = ",".join(_switch_distance_sources(options))
    result.metadata["lod_generated_parts"] = str(generated_parts)
    result.metadata["lod_skipped_no_mesh_parts"] = str(skipped_parts)
    result.metadata["lod_source_vertices"] = str(source_vertices)
    result.metadata["lod_source_triangles"] = str(source_triangles)
    result.metadata["lod_source_mesh_bytes"] = str(source_mesh_bytes)
    result.metadata["lod_added_vertices"] = str(added_vertices)
    result.metadata["lod_added_triangles"] = str(added_triangles)
    result.metadata["lod_added_mesh_bytes"] = str(added_mesh_bytes)
    result.metadata["lod_retained_vertices"] = str(retained_vertices)
    result.metadata["lod_retained_triangles"] = str(retained_triangles)
    result.metadata["lod_retained_mesh_bytes"] = str(retained_mesh_bytes)
    result.metadata["lod_chain_vertices"] = str(source_vertices + added_vertices + retained_vertices)
    result.metadata["lod_chain_triangles"] = str(source_triangles + added_triangles + retained_triangles)
    result.metadata["lod_chain_mesh_bytes"] = str(source_mesh_bytes + added_mesh_bytes + retained_mesh_bytes)
    result.metadata["lod_omitted_tiny_part_meshes"] = str(omitted_tiny_parts)
    result.metadata["lod_triangle_multiplier"] = _ratio_text(
        source_triangles + added_triangles + retained_triangles, source_triangles
    )
    result.metadata["lod_mesh_byte_multiplier"] = _ratio_text(
        source_mesh_bytes + added_mesh_bytes + retained_mesh_bytes, source_mesh_bytes
    )
    result.metadata["lod_reused_instance_levels"] = str(reused_instance_levels)
    result.metadata["lod_material_merged_levels"] = str(material_merged_levels)
    result.metadata["lod_texture_baked_levels"] = str(texture_baked_levels)
    result.metadata["lod_culling_changed_levels"] = str(culling_changed_levels)
    result.metadata["lod_level_policy_advisory"] = ",".join(level_policy_advisories)
    if scene_proxy is not None:
        result.metadata.update(scene_proxy)
    if generated_parts:
        _record_lod_chain_advisories(result, options, screen_coverage)
    else:
        result.metadata["lod_advisory_count"] = "0"
        result.metadata["lod_advisory_codes"] = ""
    if generated_parts == 0 and skipped_parts:
        result.report.add_warning("LOD generation matched no tessellated mesh-bearing parts")
    if options.validate:
        _validate_lods(result, selected_part_ids=selected_part_ids)
    return result


def _usable_imported_chain(part: Part) -> bool:
    if part.mesh is None or not part.lod_meshes:
        return False
    counts = [part.mesh.triangle_count, *(mesh.triangle_count for mesh in part.lod_meshes)]
    return all(current > following for current, following in pairwise(counts))


def _apply_imported_coverage(part: Part, options: LODOptions, asset: Asset) -> None:
    if options.screen_coverage is None:
        return
    if len(options.screen_coverage) != len(part.lod_meshes):
        asset.report.add_warning(
            f"LOD screen coverage was not applied to imported chain for part {part.name}: "
            f"expected {len(part.lod_meshes)} values, received {len(options.screen_coverage)}"
        )
        return
    values = tuple(float(value) for value in options.screen_coverage)
    part.metadata["lod_screen_coverage"] = ",".join(f"{value:.9g}" for value in values)
    for mesh, coverage in zip(part.lod_meshes, values, strict=True):
        mesh.metadata["lod_screen_coverage"] = f"{coverage:.9g}"


@dataclass(frozen=True)
class _LodBuildPayload:
    part: Part
    options: LODOptions
    screen_coverage: tuple[float, ...]
    export_mode: str
    level_policy_advisories: tuple[str, ...]
    part_occurrences: int


def _lod_build_worker(payload: _LodBuildPayload) -> _LodPartResult:
    return _build_part_lods(
        payload.part,
        payload.options,
        payload.screen_coverage,
        payload.export_mode,
        payload.level_policy_advisories,
        part_occurrences=payload.part_occurrences,
    )


def _build_part_lods(
    part: Part,
    options: LODOptions,
    screen_coverage: tuple[float, ...],
    export_mode: str,
    level_policy_advisories: tuple[str, ...],
    *,
    part_occurrences: int,
) -> _LodPartResult:
    if part.mesh is None:
        return _LodPartResult(
            part_id=part.id,
            lod_meshes=[],
            metadata={**part.metadata, "lod_status": "skipped_no_mesh"},
            warnings=(f"LOD generation skipped part without tessellated mesh: {part.name}",),
            skipped=True,
        )

    mesh = part.mesh
    previous_count = mesh.triangle_count
    diagonal = _mesh_diagonal(mesh)
    part_source_vertices = mesh.vertex_count
    part_source_triangles = mesh.triangle_count
    part_source_bytes = _mesh_payload_bytes(mesh)
    part_lod_vertices = 0
    part_lod_triangles = 0
    part_lod_bytes = 0
    part_omitted_tiny = 0
    part_reused_instance_levels = 0
    part_material_merged_levels = 0
    part_texture_baked_levels = 0
    part_culling_changed_levels = 0
    previous_mesh = mesh
    lod_meshes: list[Mesh] = []
    level_instance_reuse: list[str] = []
    level_material_merge: list[str] = []
    level_texture_bake: list[str] = []
    level_culling_granularity: list[str] = []
    level_switch_distances: list[float] = []
    level_switch_distance_sources: list[str] = []
    level_policy_advisory_values: list[str] = []
    level_simplification_sources: list[str] = []
    for index, ratio in enumerate(options.ratios):
        coverage = screen_coverage[index]
        far_lod = _is_far_lod(ratio, coverage)
        switch_distance, switch_distance_source = _switch_distance_for_level(
            diagonal,
            coverage,
            options.engine_profile,
            _switch_distance_override(options, index),
        )
        level_switch_distances.append(switch_distance)
        level_switch_distance_sources.append(switch_distance_source)
        simplification_source = "source" if index == 0 else "previous"
        policy_metadata = _level_policy_metadata(
            part_occurrences=part_occurrences,
            culling_granularity="omitted_tiny_part"
            if options.drop_tiny_parts and diagonal * coverage < options.tiny_part_screen_size
            else "part",
            policy_advisory=level_policy_advisories[index],
            far_lod=far_lod,
            far_lod_bake=options.far_lod_bake,
        )
        level_instance_reuse.append(policy_metadata["lod_instance_reuse"])
        level_material_merge.append(policy_metadata["lod_material_merge"])
        level_texture_bake.append(policy_metadata["lod_texture_bake"])
        level_culling_granularity.append(policy_metadata["lod_culling_granularity"])
        level_policy_advisory_values.append(policy_metadata["lod_policy_advisory"])
        if policy_metadata["lod_instance_reuse"] == "preserved":
            part_reused_instance_levels += 1
        if policy_metadata["lod_material_merge"] == "merged":
            part_material_merged_levels += 1
        if policy_metadata["lod_texture_bake"] == "baked":
            part_texture_baked_levels += 1
        if policy_metadata["lod_culling_granularity"] != "part":
            part_culling_changed_levels += 1
        if options.drop_tiny_parts and diagonal * coverage < options.tiny_part_screen_size:
            lod = _empty_lod(mesh)
            lod.metadata = {
                **lod.metadata,
                "lod_ratio": f"{ratio:.9g}",
                "lod_screen_coverage": f"{coverage:.9g}",
                "lod_omitted": "tiny_part",
                "lod_simplification_source": simplification_source,
                "lod_engine_profile": options.engine_profile,
                "lod_export_mode": export_mode,
                "lod_switch_distance": f"{switch_distance:.9g}",
                "lod_switch_distance_source": switch_distance_source,
                **policy_metadata,
            }
            lod_meshes.append(lod)
            previous_count = 0
            previous_mesh = lod
            part_omitted_tiny += 1
            level_simplification_sources.append(simplification_source)
            continue

        actual_simplification_source = simplification_source
        lod = previous_mesh.simplify(target_triangles=_target_lod_triangles(part_source_triangles, ratio))
        if lod.triangle_count > previous_count:
            lod = lod.simplify(target_triangles=previous_count)
            actual_simplification_source = f"{simplification_source}_retry"
        if options.far_lod_bake and far_lod:
            lod = _bake_far_lod_mesh(lod)
        lod.metadata = {
            **lod.metadata,
            "lod_ratio": f"{ratio:.9g}",
            "lod_screen_coverage": f"{coverage:.9g}",
            "lod_mode": options.mode,
            "lod_export_mode": export_mode,
            "lod_engine_profile": options.engine_profile,
            "lod_switch_distance": f"{switch_distance:.9g}",
            "lod_switch_distance_source": switch_distance_source,
            "lod_per_part_budget": str(options.per_part_budget).lower(),
            "lod_simplification_source": actual_simplification_source,
            **policy_metadata,
        }
        lod.validate()
        previous_count = lod.triangle_count
        previous_mesh = lod
        lod_meshes.append(lod)
        part_lod_vertices += lod.vertex_count
        part_lod_triangles += lod.triangle_count
        part_lod_bytes += _mesh_payload_bytes(lod)
        level_simplification_sources.append(actual_simplification_source)

    level_vertices = ",".join(str(mesh.vertex_count) for mesh in lod_meshes)
    level_triangles = ",".join(str(mesh.triangle_count) for mesh in lod_meshes)
    metadata = {
        **part.metadata,
        "lod_ratios": ",".join(f"{ratio:.9g}" for ratio in options.ratios),
        "lod_screen_coverage": ",".join(f"{value:.9g}" for value in screen_coverage),
        "lod_mode": options.mode,
        "lod_export_mode": export_mode,
        "lod_engine_profile": options.engine_profile,
        "lod_far_lod_bake": str(options.far_lod_bake).lower(),
        "lod_per_part_budget": str(options.per_part_budget).lower(),
        "lod_drop_tiny_parts": str(options.drop_tiny_parts).lower(),
        "lod_occurrences": str(part_occurrences),
        "lod_source_vertices": str(part_source_vertices),
        "lod_source_triangles": str(part_source_triangles),
        "lod_source_mesh_bytes": str(part_source_bytes),
        "lod_added_vertices": str(part_lod_vertices),
        "lod_added_triangles": str(part_lod_triangles),
        "lod_added_mesh_bytes": str(part_lod_bytes),
        "lod_retained_vertices": "0",
        "lod_retained_triangles": "0",
        "lod_retained_mesh_bytes": "0",
        "lod_chain_vertices": str(part_source_vertices + part_lod_vertices),
        "lod_chain_triangles": str(part_source_triangles + part_lod_triangles),
        "lod_chain_mesh_bytes": str(part_source_bytes + part_lod_bytes),
        "lod_level_vertices": level_vertices,
        "lod_level_triangles": level_triangles,
        "lod_omitted_tiny_part_meshes": str(part_omitted_tiny),
        "lod_triangle_multiplier": _ratio_text(part_source_triangles + part_lod_triangles, part_source_triangles),
        "lod_mesh_byte_multiplier": _ratio_text(part_source_bytes + part_lod_bytes, part_source_bytes),
        "lod_level_instance_reuse": ",".join(level_instance_reuse),
        "lod_level_material_merge": ",".join(level_material_merge),
        "lod_level_texture_bake": ",".join(level_texture_bake),
        "lod_level_culling_granularity": ",".join(level_culling_granularity),
        "lod_level_switch_distances": ",".join(f"{value:.9g}" for value in level_switch_distances),
        "lod_level_switch_distance_sources": ",".join(level_switch_distance_sources),
        "lod_switching_validation_status": _switching_validation_status(level_switch_distances),
        "lod_level_policy_advisory": ",".join(level_policy_advisory_values),
        "lod_level_simplification_source": ",".join(level_simplification_sources),
        "lod_reused_instance_levels": str(part_reused_instance_levels),
        "lod_material_merged_levels": str(part_material_merged_levels),
        "lod_texture_baked_levels": str(part_texture_baked_levels),
        "lod_culling_changed_levels": str(part_culling_changed_levels),
    }
    return _LodPartResult(
        part_id=part.id,
        lod_meshes=lod_meshes,
        metadata=metadata,
        generated=True,
        source_vertices=part_source_vertices,
        source_triangles=part_source_triangles,
        source_mesh_bytes=part_source_bytes,
        added_vertices=part_lod_vertices,
        added_triangles=part_lod_triangles,
        added_mesh_bytes=part_lod_bytes,
        omitted_tiny_parts=part_omitted_tiny,
        reused_instance_levels=part_reused_instance_levels,
        material_merged_levels=part_material_merged_levels,
        texture_baked_levels=part_texture_baked_levels,
        culling_changed_levels=part_culling_changed_levels,
    )


def _occurrence_counts_by_part(asset: Asset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in asset.root.walk():
        if node.part_id is None:
            continue
        counts[node.part_id] = counts.get(node.part_id, 0) + 1
    return counts


def _build_scene_far_proxy(
    asset: Asset,
    options: LODOptions,
    *,
    selected_part_ids: set[str] | None,
) -> dict[str, object] | None:
    occurrences = _world_part_occurrences(asset, selected_part_ids)
    points: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    source_parts: set[str] = set()
    source_occurrences = 0
    for part, world_transform in occurrences:
        if not part.lod_meshes:
            continue
        mesh = part.lod_meshes[-1]
        if mesh.vertex_count == 0 or mesh.triangle_count == 0:
            continue
        transformed_points = _transform_points(mesh.points, world_transform)
        points.append(transformed_points)
        faces.append(mesh.faces.astype(np.int64, copy=True) + offset)
        offset += mesh.vertex_count
        source_parts.add(part.id)
        source_occurrences += 1
    if not points or not faces:
        return None

    material_id = _unique_id("lod_scene_far_proxy_material", asset.materials)
    part_id = _unique_id("lod_scene_far_proxy", asset.parts)
    merged_points = np.vstack(points).astype(np.float64)
    merged_faces = np.vstack(faces).astype(np.int64)
    material_indices = np.zeros(merged_faces.shape[0], dtype=np.int64)
    screen_coverage = _screen_coverage(options)
    proxy_metadata: dict[str, object] = {
        "lod_scene_far_proxy": "true",
        "lod_level": str(len(options.ratios)),
        "lod_export_mode": _lod_export_mode(options),
        "lod_engine_profile": options.engine_profile,
        "lod_ratio": f"{options.ratios[-1]:.9g}",
        "lod_screen_coverage": f"{screen_coverage[-1]:.9g}",
        "lod_source_parts": str(len(source_parts)),
        "lod_source_occurrences": str(source_occurrences),
        "lod_material_merge": "scene_merged",
        "lod_texture_bake": "scene_baked" if options.far_lod_bake else "not_run",
        "lod_draw_calls": "1",
    }
    proxy_mesh = Mesh(
        points=merged_points,
        faces=merged_faces,
        material_indices=material_indices,
        metadata=proxy_metadata,
    )
    proxy_mesh.validate()
    asset.materials[material_id] = Material(
        id=material_id,
        name="Scene far proxy material",
        base_color=(0.7, 0.7, 0.7, 1.0),
        roughness=0.6,
        metadata={
            "lod_scene_far_proxy": "true",
            "lod_material_merge": "scene_merged",
            "lod_source_material_policy": "one_material",
        },
    )
    asset.parts[part_id] = Part(
        id=part_id,
        name="Scene Far Proxy",
        mesh=proxy_mesh,
        material_ids=[material_id],
        metadata=proxy_metadata,
        fingerprint=proxy_mesh.fingerprint(),
    )
    return {
        "lod_scene_far_proxy_part_id": part_id,
        "lod_scene_far_proxy_material_id": material_id,
        "lod_scene_far_proxy_vertices": str(proxy_mesh.vertex_count),
        "lod_scene_far_proxy_triangles": str(proxy_mesh.triangle_count),
        "lod_scene_far_proxy_draw_calls": "1",
        "lod_scene_far_proxy_source_parts": str(len(source_parts)),
        "lod_scene_far_proxy_source_occurrences": str(source_occurrences),
    }


def _world_part_occurrences(asset: Asset, selected_part_ids: set[str] | None) -> list[tuple[Part, np.ndarray]]:
    occurrences: list[tuple[Part, np.ndarray]] = []
    for node, current in asset.root.walk_world(np.eye(4, dtype=np.float64)):
        if node.part_id is not None and (selected_part_ids is None or node.part_id in selected_part_ids):
            part = asset.parts.get(node.part_id)
            if part is not None:
                occurrences.append((part, current))
    return occurrences


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return np.asarray(points.copy(), dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return np.asarray((transform @ homogeneous.T).T[:, :3], dtype=np.float64)


def _unique_id(base: str, values: Mapping[str, object]) -> str:
    if base not in values:
        return base
    index = 1
    while f"{base}_{index}" in values:
        index += 1
    return f"{base}_{index}"


def _level_policy_metadata(
    *,
    part_occurrences: int,
    culling_granularity: str,
    policy_advisory: str,
    far_lod: bool,
    far_lod_bake: bool,
) -> dict[str, str]:
    if culling_granularity == "omitted_tiny_part":
        instance_reuse = "omitted"
    else:
        instance_reuse = "preserved" if part_occurrences > 1 else "not_applicable"
    baked_far_lod = far_lod and far_lod_bake and culling_granularity != "omitted_tiny_part"
    return {
        "lod_instance_reuse": instance_reuse,
        "lod_material_merge": "merged" if baked_far_lod else "not_run",
        "lod_texture_bake": "baked" if baked_far_lod else "not_run",
        "lod_culling_granularity": culling_granularity,
        "lod_policy_advisory": policy_advisory,
    }


def _level_policy_advisories(options: LODOptions, screen_coverage: tuple[float, ...]) -> tuple[str, ...]:
    values: list[str] = []
    last_index = len(options.ratios) - 1
    for index, ratio in enumerate(options.ratios):
        coverage = screen_coverage[index]
        if index == 0 and ratio < _CLOSE_VIEW_LOD1_MIN_RATIO:
            values.append("close_view_too_aggressive")
        elif index == 1 and ratio < _CLOSE_VIEW_LOD2_MIN_RATIO:
            values.append("mid_view_too_aggressive")
        elif index == last_index and _is_far_lod(ratio, coverage):
            values.append("far_proxy_recommended")
        elif index <= 1:
            values.append("conservative_geometry")
        else:
            values.append("progressive_geometry")
    return tuple(values)


def _lod_export_mode(options: LODOptions) -> str:
    if options.engine_profile == "unity":
        return "variants"
    if options.engine_profile == "unreal":
        return "separate"
    return options.mode


def _screen_coverage(options: LODOptions) -> tuple[float, ...]:
    if options.screen_coverage is not None:
        return tuple(options.screen_coverage)
    return tuple(0.5 / (index + 1) for index, _ratio in enumerate(options.ratios))


def _record_lod_chain_advisories(
    asset: Asset,
    options: LODOptions,
    screen_coverage: tuple[float, ...],
) -> None:
    advisories = _lod_chain_advisories(options, screen_coverage)
    if not advisories:
        asset.metadata["lod_advisory_count"] = "0"
        asset.metadata["lod_advisory_codes"] = ""
        return
    encoded = json.dumps(advisories, sort_keys=True)
    codes = ",".join(str(item["code"]) for item in advisories)
    asset.metadata["lod_advisories"] = encoded
    asset.metadata["lod_advisory_count"] = str(len(advisories))
    asset.metadata["lod_advisory_codes"] = codes
    for advisory in advisories:
        if advisory.get("severity") == "warning":
            asset.report.add_warning(str(advisory["message"]))


def _lod_chain_advisories(options: LODOptions, screen_coverage: tuple[float, ...]) -> list[dict[str, object]]:
    advisories: list[dict[str, object]] = []
    if len(options.ratios) > _RECOMMENDED_MAX_LOD_LEVELS:
        advisories.append(
            {
                "code": "excessive_lod_levels",
                "severity": "warning",
                "levels": len(options.ratios),
                "recommended_max": _RECOMMENDED_MAX_LOD_LEVELS,
                "message": (
                    f"LOD chain has {len(options.ratios)} generated levels; "
                    "3-4 levels are usually enough, and extra meshes increase memory and export size"
                ),
            }
        )

    minimum_ratio_warnings = [
        {
            "level": index + 1,
            "ratio": ratio,
            "minimum_recommended_ratio": _MIN_RECOMMENDED_LOD_RATIO,
            "screen_coverage": screen_coverage[index],
        }
        for index, ratio in enumerate(options.ratios)
        if ratio <= _MIN_RECOMMENDED_LOD_RATIO
    ]
    if minimum_ratio_warnings:
        levels = ",".join(f"LOD{item['level']}" for item in minimum_ratio_warnings)
        advisories.append(
            {
                "code": "destructive_lod_ratio_floor",
                "severity": "warning",
                "levels": minimum_ratio_warnings,
                "message": (
                    f"{levels} use ratios at or below {_MIN_RECOMMENDED_LOD_RATIO:g}; "
                    "ratios this small commonly collapse to a one-triangle LOD, so prefer far_lod_bake, "
                    "scene_far_proxy, or drop_tiny_parts for distant runtime levels"
                ),
            }
        )

    close_view_warnings: list[dict[str, object]] = []
    for index, ratio in enumerate(options.ratios[:2]):
        threshold = _CLOSE_VIEW_LOD1_MIN_RATIO if index == 0 else _CLOSE_VIEW_LOD2_MIN_RATIO
        if ratio < threshold:
            close_view_warnings.append(
                {
                    "level": index + 1,
                    "ratio": ratio,
                    "minimum_recommended_ratio": threshold,
                    "screen_coverage": screen_coverage[index],
                }
            )
    if close_view_warnings:
        levels = ",".join(f"LOD{item['level']}" for item in close_view_warnings)
        advisories.append(
            {
                "code": "aggressive_close_view_lods",
                "severity": "warning",
                "levels": close_view_warnings,
                "message": (
                    f"{levels} use aggressive reduction for close or mid-view LODs; "
                    "keep LOD1 and LOD2 visually conservative and reserve destructive ratios for distant levels"
                ),
            }
        )

    far_index = len(options.ratios) - 1
    if far_index >= 0 and _is_far_lod(options.ratios[far_index], screen_coverage[far_index]):
        if options.far_lod_bake:
            advisories.append(
                {
                    "code": "far_lod_bake_enabled",
                    "severity": "info",
                    "level": far_index + 1,
                    "ratio": options.ratios[far_index],
                    "screen_coverage": screen_coverage[far_index],
                    "message": f"LOD{far_index + 1} uses far-distance one-material bake policy",
                }
            )
        else:
            advisories.append(
                {
                    "code": "far_lod_proxy_recommended",
                    "severity": "warning",
                    "level": far_index + 1,
                    "ratio": options.ratios[far_index],
                    "screen_coverage": screen_coverage[far_index],
                    "message": (
                        f"LOD{far_index + 1} is a far-distance geometry-only LOD; "
                        "enable far_lod_bake for a one-material far runtime level"
                    ),
                }
            )
    return advisories


def _is_far_lod(ratio: float, screen_coverage: float) -> bool:
    return ratio <= _FAR_LOD_RATIO_THRESHOLD or screen_coverage <= _FAR_LOD_SCREEN_COVERAGE_THRESHOLD


def _mesh_diagonal(mesh: Mesh) -> float:
    mins, maxs = mesh.bounds()
    return float(np.linalg.norm(maxs - mins))


def _empty_lod(mesh: Mesh) -> Mesh:
    return Mesh(
        points=np.empty((0, 3), dtype=np.float64),
        faces=np.empty((0, 3), dtype=np.int64),
        metadata={**mesh.metadata, "lod_omitted": "tiny_part"},
    )


def _bake_far_lod_mesh(mesh: Mesh) -> Mesh:
    result = mesh.copy()
    if result.triangle_count > 0:
        result.material_indices = np.zeros(result.triangle_count, dtype=np.int64)
    result.metadata = {
        **result.metadata,
        "lod_far_bake": "one_material",
        "lod_material_merge": "merged",
        "lod_texture_bake": "baked",
    }
    return result


def _switch_distance(diagonal: float, screen_coverage: float, engine_profile: str) -> float:
    if diagonal <= 0.0:
        return 0.0
    coverage = max(float(screen_coverage), 1e-6)
    if engine_profile == "unity":
        return diagonal / (2.0 * coverage)
    if engine_profile == "unreal":
        return diagonal / math.sqrt(coverage)
    return diagonal / coverage


def _switch_distance_override(options: LODOptions, index: int) -> float | None:
    if options.switch_distance_overrides is None:
        return None
    return options.switch_distance_overrides[index]


def _switch_distance_for_level(
    diagonal: float,
    screen_coverage: float,
    engine_profile: str,
    override: float | None,
) -> tuple[float, str]:
    if override is not None:
        return override, "override"
    return _switch_distance(diagonal, screen_coverage, engine_profile), "formula"


def _switch_distance_sources(options: LODOptions) -> tuple[str, ...]:
    return tuple(
        "override" if _switch_distance_override(options, index) is not None else "formula"
        for index in range(len(options.ratios))
    )


def _switching_validation_status(distances: list[float]) -> str:
    if not distances:
        return "not_applicable"
    return "monotonic" if distances == sorted(distances) else "non_monotonic"


def _target_lod_triangles(source_triangles: int, ratio: float) -> int:
    if source_triangles <= 0:
        return 0
    return max(1, int(round(source_triangles * ratio)))


def _mesh_payload_bytes(mesh: Mesh) -> int:
    total = mesh.points.nbytes + mesh.faces.nbytes
    if mesh.normals is not None:
        total += mesh.normals.nbytes
    if mesh.tangents is not None:
        total += mesh.tangents.nbytes
    if mesh.material_indices is not None:
        total += mesh.material_indices.nbytes
    for uv_values in mesh.uvs.values():
        total += uv_values.nbytes
    for face_group_values in mesh.face_groups.values():
        total += face_group_values.nbytes
    return int(total)


def _ratio_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    return f"{(numerator / denominator):.9g}"


def _validate_lods(asset: Asset, *, selected_part_ids: set[str] | None) -> None:
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None or not part.lod_meshes:
            continue
        counts = [part.mesh.triangle_count, *[mesh.triangle_count for mesh in part.lod_meshes]]
        if counts != sorted(counts, reverse=True):
            raise ValueError(f"LOD triangles are not monotonic for part {part.id}")
        for lod in part.lod_meshes:
            metrics = lod.quality_metrics()
            if metrics["degenerate_triangles"]:
                asset.report.add_warning(f"LOD contains degenerate triangles for part {part.id}")
