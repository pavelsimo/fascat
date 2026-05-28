from __future__ import annotations

import json

import numpy as np

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.options import LODOptions

_RECOMMENDED_MAX_LOD_LEVELS = 4
_CLOSE_VIEW_LOD1_MIN_RATIO = 0.4
_CLOSE_VIEW_LOD2_MIN_RATIO = 0.2
_FAR_LOD_RATIO_THRESHOLD = 0.15
_FAR_LOD_SCREEN_COVERAGE_THRESHOLD = 0.1


def build_lods(asset: Asset, options: LODOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    screen_coverage = _screen_coverage(options)
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
    omitted_tiny_parts = 0
    reused_instance_levels = 0
    material_merged_levels = 0
    texture_baked_levels = 0
    culling_changed_levels = 0
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part.lod_meshes = []
        if part.mesh is None:
            skipped_parts += 1
            part.metadata["lod_status"] = "skipped_no_mesh"
            result.report.add_warning(f"LOD generation skipped part without tessellated mesh: {part.name}")
            continue
        generated_parts += 1
        part_occurrences = occurrence_counts.get(part.id, 0)
        previous_count = part.mesh.triangle_count
        diagonal = _mesh_diagonal(part.mesh)
        part_source_vertices = part.mesh.vertex_count
        part_source_triangles = part.mesh.triangle_count
        part_source_bytes = _mesh_payload_bytes(part.mesh)
        part_lod_vertices = 0
        part_lod_triangles = 0
        part_lod_bytes = 0
        part_omitted_tiny = 0
        part_reused_instance_levels = 0
        part_material_merged_levels = 0
        part_texture_baked_levels = 0
        part_culling_changed_levels = 0
        level_instance_reuse: list[str] = []
        level_material_merge: list[str] = []
        level_texture_bake: list[str] = []
        level_culling_granularity: list[str] = []
        level_policy_advisory_values: list[str] = []
        for index, ratio in enumerate(options.ratios):
            coverage = screen_coverage[index]
            policy_metadata = _level_policy_metadata(
                part_occurrences=part_occurrences,
                culling_granularity="omitted_tiny_part"
                if options.drop_tiny_parts and diagonal * coverage < options.tiny_part_screen_size
                else "part",
                policy_advisory=level_policy_advisories[index],
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
                lod = _empty_lod(part.mesh)
                lod.metadata = {
                    **lod.metadata,
                    "lod_ratio": f"{ratio:.9g}",
                    "lod_screen_coverage": f"{coverage:.9g}",
                    "lod_omitted": "tiny_part",
                    **policy_metadata,
                }
                part.lod_meshes.append(lod)
                previous_count = 0
                part_omitted_tiny += 1
                continue

            lod = part.mesh.simplify(ratio=ratio)
            if lod.triangle_count > previous_count:
                lod = lod.simplify(target_triangles=previous_count)
            lod.metadata = {
                **lod.metadata,
                "lod_ratio": f"{ratio:.9g}",
                "lod_screen_coverage": f"{coverage:.9g}",
                "lod_mode": options.mode,
                "lod_per_part_budget": str(options.per_part_budget).lower(),
                **policy_metadata,
            }
            lod.validate()
            previous_count = lod.triangle_count
            part.lod_meshes.append(lod)
            part_lod_vertices += lod.vertex_count
            part_lod_triangles += lod.triangle_count
            part_lod_bytes += _mesh_payload_bytes(lod)
        source_vertices += part_source_vertices
        source_triangles += part_source_triangles
        source_mesh_bytes += part_source_bytes
        added_vertices += part_lod_vertices
        added_triangles += part_lod_triangles
        added_mesh_bytes += part_lod_bytes
        omitted_tiny_parts += part_omitted_tiny
        reused_instance_levels += part_reused_instance_levels
        material_merged_levels += part_material_merged_levels
        texture_baked_levels += part_texture_baked_levels
        culling_changed_levels += part_culling_changed_levels
        level_vertices = ",".join(str(mesh.vertex_count) for mesh in part.lod_meshes)
        level_triangles = ",".join(str(mesh.triangle_count) for mesh in part.lod_meshes)
        part.metadata = {
            **part.metadata,
            "lod_ratios": ",".join(f"{ratio:.9g}" for ratio in options.ratios),
            "lod_screen_coverage": ",".join(f"{value:.9g}" for value in screen_coverage),
            "lod_mode": options.mode,
            "lod_per_part_budget": str(options.per_part_budget).lower(),
            "lod_drop_tiny_parts": str(options.drop_tiny_parts).lower(),
            "lod_occurrences": str(part_occurrences),
            "lod_source_vertices": str(part_source_vertices),
            "lod_source_triangles": str(part_source_triangles),
            "lod_source_mesh_bytes": str(part_source_bytes),
            "lod_added_vertices": str(part_lod_vertices),
            "lod_added_triangles": str(part_lod_triangles),
            "lod_added_mesh_bytes": str(part_lod_bytes),
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
            "lod_level_policy_advisory": ",".join(level_policy_advisory_values),
            "lod_reused_instance_levels": str(part_reused_instance_levels),
            "lod_material_merged_levels": str(part_material_merged_levels),
            "lod_texture_baked_levels": str(part_texture_baked_levels),
            "lod_culling_changed_levels": str(part_culling_changed_levels),
        }
    result.metadata["lod_mode"] = options.mode
    result.metadata["lod_screen_coverage"] = ",".join(f"{value:.9g}" for value in screen_coverage)
    result.metadata["lod_generated_parts"] = str(generated_parts)
    result.metadata["lod_skipped_no_mesh_parts"] = str(skipped_parts)
    result.metadata["lod_source_vertices"] = str(source_vertices)
    result.metadata["lod_source_triangles"] = str(source_triangles)
    result.metadata["lod_source_mesh_bytes"] = str(source_mesh_bytes)
    result.metadata["lod_added_vertices"] = str(added_vertices)
    result.metadata["lod_added_triangles"] = str(added_triangles)
    result.metadata["lod_added_mesh_bytes"] = str(added_mesh_bytes)
    result.metadata["lod_chain_vertices"] = str(source_vertices + added_vertices)
    result.metadata["lod_chain_triangles"] = str(source_triangles + added_triangles)
    result.metadata["lod_chain_mesh_bytes"] = str(source_mesh_bytes + added_mesh_bytes)
    result.metadata["lod_omitted_tiny_part_meshes"] = str(omitted_tiny_parts)
    result.metadata["lod_triangle_multiplier"] = _ratio_text(source_triangles + added_triangles, source_triangles)
    result.metadata["lod_mesh_byte_multiplier"] = _ratio_text(source_mesh_bytes + added_mesh_bytes, source_mesh_bytes)
    result.metadata["lod_reused_instance_levels"] = str(reused_instance_levels)
    result.metadata["lod_material_merged_levels"] = str(material_merged_levels)
    result.metadata["lod_texture_baked_levels"] = str(texture_baked_levels)
    result.metadata["lod_culling_changed_levels"] = str(culling_changed_levels)
    result.metadata["lod_level_policy_advisory"] = ",".join(level_policy_advisories)
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


def _occurrence_counts_by_part(asset: Asset) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in asset.root.walk():
        if node.part_id is None:
            continue
        counts[node.part_id] = counts.get(node.part_id, 0) + 1
    return counts


def _level_policy_metadata(
    *,
    part_occurrences: int,
    culling_granularity: str,
    policy_advisory: str,
) -> dict[str, str]:
    if culling_granularity == "omitted_tiny_part":
        instance_reuse = "omitted"
    else:
        instance_reuse = "preserved" if part_occurrences > 1 else "not_applicable"
    return {
        "lod_instance_reuse": instance_reuse,
        "lod_material_merge": "not_run",
        "lod_texture_bake": "not_run",
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
        advisories.append(
            {
                "code": "far_lod_proxy_recommended",
                "severity": "warning",
                "level": far_index + 1,
                "ratio": options.ratios[far_index],
                "screen_coverage": screen_coverage[far_index],
                "message": (
                    f"LOD{far_index + 1} is a far-distance geometry-only LOD; "
                    "consider one-mesh and one-material baking when a future far-LOD backend is available"
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
