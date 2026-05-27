from __future__ import annotations

import base64
import binascii
import struct
import zlib
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import (
    BakeMaterialOptions,
    DecimateOptions,
    LODGeneratorOptions,
    LODOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_AGGRESSIVE_LOD0_RATIO = 0.2
_DECIMATION_ITERATIVE_THRESHOLD_TRIANGLES = 1_000_000
_DECIMATION_MEMORY_BYTES_PER_MILLION_TRIANGLES = 5_000_000_000
_DECIMATION_MEMORY_GB_PER_MILLION_TRIANGLES = 5.0


def bake_materials_asset(
    asset: Asset,
    options: BakeMaterialOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    result = asset.copy(keep_source=True)
    part_ids = _selected_mesh_part_ids(result, selected_part_ids)
    if not part_ids:
        result.report.add_warning("bake_materials matched no mesh-bearing parts")
        return result
    result.report.add_warning(
        "bake_materials emits constant embedded texture maps from material factors; raster texture baking is not implemented"
    )

    source_material_ids = sorted(
        {
            material_id
            for part_id in part_ids
            for material_id in result.parts[part_id].material_ids
            if material_id in result.materials
        }
    )
    baked_id = _unique_material_id(result.materials, "baked_material")
    baked = _baked_material(result, baked_id, source_material_ids, options)
    result.materials[baked.id] = baked

    for part_id in part_ids:
        part = result.parts[part_id]
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.force_uv_generation or options.uv_channel not in mesh.uvs:
            mesh = mesh.box_uv(options.uv_channel)
        mesh.metadata = {
            **mesh.metadata,
            "baked_material": baked.id,
            "baked_maps": ",".join(options.bake),
            "baked_maps_resolution": str(options.maps_resolution),
            "baked_uv_channel": str(options.uv_channel),
            "baked_padding": str(options.padding),
        }
        if options.merge_output:
            part.material_ids = [baked.id]
            mesh.material_indices = None
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "baked_material": baked.id,
            "source_material_ids": ",".join(source_material_ids),
        }
        part.fingerprint = mesh.fingerprint()

    if options.merge_output:
        _drop_unreferenced_materials(result)
    result.metadata["baked_material_count"] = "1"
    result.metadata["baked_source_material_count"] = str(len(source_material_ids))
    return result


def decimate_asset(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    source_meshes = _source_meshes(asset, selected_part_ids)
    working_asset = _prepare_decimation_asset(asset, options, selected_part_ids)
    if options.budget_scope == "selection":
        from fascat.ops.optimize import optimize_asset

        result = optimize_asset(
            working_asset,
            OptimizeOptions(
                target_triangles=options.target_triangles,
                ratio=_decimate_ratio(options),
                preserve_instances=True,
                simplify=True,
                optimize_buffers=True,
                preserve_hard_edges=True,
                hard_edge_angle=options.normal_tolerance,
                preserve_holes=options.protect_topology,
                preserve_material_boundaries=True,
                preserve_uv_seams=_preserve_uv_seams(options),
                preserve_silhouette=options.protect_topology,
            ),
            selected_part_ids=selected_part_ids,
        )
        _warn_aggressive_lod0_decimation(result, source_meshes, options)
        if options.criterion == "quality":
            result.report.add_warning(_quality_decimation_warning())
        _enforce_triangle_budget(result, options, selected_part_ids=selected_part_ids)
        _finalize_decimation_uv_importance(result, options, selected_part_ids=selected_part_ids)
        _annotate_decimation_result(result, source_meshes, options, selected_part_ids=selected_part_ids)
        return result

    result = working_asset.copy(keep_source=True)
    ratio = _decimate_ratio(options)
    _warn_aggressive_lod0_decimation(result, source_meshes, options)
    if options.criterion == "quality":
        result.report.add_warning(_quality_decimation_warning())
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        target = options.target_triangles
        if target is not None:
            target = min(target, part.mesh.triangle_count)
        mesh = part.mesh.simplify(
            target_triangles=target,
            ratio=None if target is not None else ratio,
            preserve_hard_edges=True,
            hard_edge_angle=options.normal_tolerance,
            preserve_holes=options.protect_topology,
            preserve_material_boundaries=True,
            preserve_uv_seams=_preserve_uv_seams(options),
            preserve_silhouette=options.protect_topology,
        )
        mesh = mesh.optimize_buffers().repair()
        target_budget = target if target is not None else _ratio_target(part.mesh, ratio)
        if target_budget is not None and mesh.triangle_count > target_budget:
            mesh = _sample_mesh_faces(mesh, target_budget).compute_normals()
        mesh = _finalize_decimated_mesh_uvs(mesh, options)
        mesh.metadata = {
            **mesh.metadata,
            "decimate_criterion": options.criterion,
            "decimate_budget_scope": options.budget_scope,
            "decimate_uv_importance": options.uv_importance,
        }
        mesh.validate()
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    _annotate_decimation_result(result, source_meshes, options, selected_part_ids=selected_part_ids)
    return result


def remove_holes_asset(
    asset: Asset,
    options: RemoveHolesOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    result = asset.copy(keep_source=True)
    if options.prefer_brep:
        result.report.add_warning(
            "BREP feature-level hole removal is not implemented; using mesh boundary classification and fill"
        )
    else:
        result.report.add_warning(
            "remove_holes uses mesh boundary classification and fill; closed BREP feature removal is not implemented"
        )
    removed_count = 0
    diameters: list[float] = []
    kinds: list[str] = []
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        mesh, stats = _fill_small_holes(part.mesh, options)
        if stats.count == 0:
            continue
        removed_count += stats.count
        diameters.extend(stats.diameters)
        kinds.extend(stats.kinds)
        mesh = mesh.compute_normals()
        mesh.validate()
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "removed_holes": str(stats.count),
            "removed_hole_types": _hole_kind_summary(stats.kinds),
            "removed_hole_diameter_method": "planar_span",
        }
        part.fingerprint = mesh.fingerprint()

    result.metadata["removed_holes"] = str(removed_count)
    for kind in ("through", "blind", "surface"):
        result.metadata[f"removed_{kind}_holes"] = str(kinds.count(kind))
    if diameters:
        result.metadata["removed_hole_min_diameter"] = f"{min(diameters):.9g}"
        result.metadata["removed_hole_max_diameter"] = f"{max(diameters):.9g}"
        result.metadata["removed_hole_diameter_method"] = "planar_span"
    if options.prefer_brep and removed_count:
        result.metadata["remove_holes_backend"] = "mesh"
    return result


def remove_occluded_asset(
    asset: Asset,
    options: RemoveOccludedOptions,
    *,
    selected_node_ids: set[str],
) -> Asset:
    result = asset.copy(keep_source=True)
    if options.level != "parts":
        _isolate_selected_occurrence_parts(result, selected_node_ids)
    result.report.add_warning(
        "remove_occluded uses deterministic sampled visibility; thin occluders may require higher precision"
    )
    occurrences = _world_occurrences(result)
    selected_occurrences = [item for item in occurrences if item.node.id in selected_node_ids]
    directions = _occlusion_directions(options)
    ray_distance = _occlusion_ray_distance(occurrences)
    removed_node_ids: set[str] = set()
    trims: list[_OcclusionTrim] = []
    measurements: list[_OcclusionMeasurement] = []
    for candidate in selected_occurrences:
        if _preserve_candidate_cavity(result, candidate, options):
            continue
        occluders = _candidate_occluders(result, candidate, occurrences, options)
        if options.level == "parts":
            measurement = _occurrence_visibility_measurement(candidate, occluders, directions, ray_distance, options)
            measurements.append(measurement)
            if measurement.visible_sample_count == 0:
                removed_node_ids.add(candidate.node.id)
            continue
        visible_faces, measurement = _visible_face_measurement(candidate, occluders, directions, ray_distance)
        measurements.append(measurement)
        part = result.parts.get(candidate.part_id)
        mesh = None if part is None else part.mesh
        if mesh is None or mesh.triangle_count == 0 or bool(np.all(visible_faces)):
            continue
        if options.level == "submeshes":
            keep_mask = _submesh_keep_mask(mesh, visible_faces)
        else:
            keep_mask = _expand_face_mask(mesh, visible_faces, options.neighbors_preservation)
        if bool(np.all(keep_mask)):
            continue
        keep_faces = np.flatnonzero(keep_mask)
        removed_faces = int(mesh.triangle_count - keep_faces.shape[0])
        if keep_faces.size == 0:
            removed_node_ids.add(candidate.node.id)
        else:
            trims.append(
                _OcclusionTrim(
                    node_id=candidate.node.id,
                    part_id=candidate.part_id,
                    keep_faces=keep_faces.astype(np.int64),
                    removed_faces=removed_faces,
                )
            )

    if removed_node_ids:
        _remove_part_nodes(result.root, removed_node_ids)
    removed_faces_total = _removed_node_triangle_count(result, selected_occurrences, removed_node_ids)
    removed_faces_total += _apply_occlusion_trims(result, trims, removed_node_ids, options)
    if removed_node_ids or removed_faces_total:
        _drop_unreferenced_parts(result)
    result.metadata["removed_occluded_nodes"] = str(len(removed_node_ids))
    result.metadata["removed_occluded_triangles"] = str(removed_faces_total)
    result.metadata["occlusion_strategy"] = options.strategy
    result.metadata["occlusion_level"] = options.level
    result.metadata["occlusion_direction_count"] = str(len(directions))
    result.metadata["occlusion_hemi_evaluation"] = str(options.hemi_evaluation).lower()
    _record_occlusion_confidence(result, measurements, len(selected_occurrences), len(directions), options)
    return result


def run_lod_generators_asset(
    asset: Asset,
    options: LODGeneratorOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    from fascat.ops.lod import build_lods

    ratios = tuple(level.target_ratio for level in options.levels)
    result = build_lods(asset, LODOptions(ratios=ratios), selected_part_ids=selected_part_ids)
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


@dataclass(frozen=True)
class _HoleFillStats:
    count: int
    diameters: tuple[float, ...]
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class _DecimationMetrics:
    source_vertices: int
    source_triangles: int
    output_triangles: int
    triangle_reduction: float
    max_vertex_error: float
    mean_vertex_error: float


@dataclass(frozen=True)
class _HoleLoop:
    loop: list[int]
    diameter: float
    centroid: FloatArray
    normal: FloatArray
    adjacent_faces: tuple[int, ...]
    kind: str = "surface"


@dataclass(frozen=True)
class _WorldOccurrence:
    node: Node
    part_id: str
    world_points: FloatArray
    faces: IntArray
    bounds_min: FloatArray
    bounds_max: FloatArray
    volume: float


@dataclass(frozen=True)
class _OcclusionTrim:
    node_id: str
    part_id: str
    keep_faces: IntArray
    removed_faces: int


@dataclass(frozen=True)
class _OcclusionMeasurement:
    face_count: int
    sample_count: int
    visible_sample_count: int
    hidden_sample_count: int


def _selected_mesh_part_ids(asset: Asset, selected_part_ids: set[str] | None) -> set[str]:
    return {
        part.id
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    }


def _baked_material(
    asset: Asset,
    material_id: str,
    source_material_ids: list[str],
    options: BakeMaterialOptions,
) -> Material:
    materials = [asset.materials[material_id] for material_id in source_material_ids if material_id in asset.materials]
    if not materials:
        base_color = (1.0, 1.0, 1.0, 1.0)
        metallic = 0.0
        roughness = 0.5
        opacity = 1.0
    else:
        base_color_values = np.asarray([material.base_color for material in materials], dtype=np.float64)
        base_color = cast(tuple[float, float, float, float], tuple(base_color_values.mean(axis=0).tolist()))
        metallic = float(np.mean([material.metallic for material in materials]))
        roughness = float(np.mean([material.roughness for material in materials]))
        opacity = float(np.mean([material.opacity for material in materials]))
    maps = {str(item) for item in options.bake}
    metadata: dict[str, object] = {
        "baked": "true",
        "baked_maps": ",".join(options.bake),
        "maps_resolution": str(options.maps_resolution),
        "padding": str(options.padding),
        "source_material_ids": ",".join(source_material_ids),
        "baked_texture_kind": "constant",
        "baked_texture_resolution": str(options.maps_resolution),
    }
    metadata.update(_baked_texture_metadata(base_color, metallic, roughness, maps))
    return Material(
        id=material_id,
        name="Baked Material",
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        opacity=opacity,
        metadata=metadata,
    )


def _baked_texture_metadata(
    base_color: tuple[float, float, float, float],
    metallic: float,
    roughness: float,
    maps: set[str],
) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if {"base_color", "opacity"} & maps:
        metadata["baked_texture_base_color_uri"] = _solid_png_data_uri(base_color)
    if {"metallic", "roughness"} & maps:
        metadata["baked_texture_metallic_roughness_uri"] = _solid_png_data_uri((1.0, roughness, metallic, 1.0))
    if "normal" in maps:
        metadata["baked_texture_normal_uri"] = _solid_png_data_uri((0.5, 0.5, 1.0, 1.0))
    if "ao" in maps:
        metadata["baked_texture_occlusion_uri"] = _solid_png_data_uri((1.0, 1.0, 1.0, 1.0))
    if "emissive" in maps:
        metadata["baked_texture_emissive_uri"] = _solid_png_data_uri((0.0, 0.0, 0.0, 1.0))
    return metadata


def _solid_png_data_uri(color: tuple[float, float, float, float]) -> str:
    pixel = bytes(_color_byte(value) for value in color)
    raw = b"\x00" + pixel
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _color_byte(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))


def _decimate_ratio(options: DecimateOptions) -> float | None:
    if options.target_triangles is not None:
        return None
    if options.target_ratio is not None:
        return options.target_ratio
    if options.criterion == "quality":
        tolerance = max(options.surface_tolerance or 0.0, options.line_tolerance or 0.0, options.uv_tolerance or 0.0)
        return max(0.1, min(0.95, 1.0 - tolerance))
    return 0.5


def _preserve_uv_seams(options: DecimateOptions) -> bool:
    return options.uv_importance in {"preserve_islands", "preserve_seams"}


def _prepare_decimation_asset(
    asset: Asset,
    options: DecimateOptions,
    selected_part_ids: set[str] | None,
) -> Asset:
    if options.uv_importance != "ignore":
        return asset
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        part.mesh = _mesh_without_texture_coordinates(part.mesh)
        part.fingerprint = part.mesh.fingerprint()
    return result


def _finalize_decimation_uv_importance(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None,
) -> None:
    if options.uv_importance == "preserve_islands":
        return
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        part.mesh = _finalize_decimated_mesh_uvs(part.mesh, options)
        part.fingerprint = part.mesh.fingerprint()


def _finalize_decimated_mesh_uvs(mesh: Mesh, options: DecimateOptions) -> Mesh:
    if options.uv_importance == "preserve_islands":
        return mesh
    return _mesh_without_texture_coordinates(mesh)


def _mesh_without_texture_coordinates(mesh: Mesh) -> Mesh:
    if not mesh.uvs and mesh.tangents is None:
        return mesh
    result = mesh.copy()
    result.uvs = {}
    result.tangents = None
    return result


def _quality_decimation_warning() -> str:
    return (
        "decimate quality criterion maps tolerances to a target ratio and records measured vertex error; "
        "tolerance-bounded simplification is not enforced"
    )


def _warn_aggressive_lod0_decimation(
    asset: Asset,
    source_meshes: dict[str, Mesh],
    options: DecimateOptions,
) -> None:
    ratio = _requested_decimation_keep_ratio(source_meshes, options)
    if ratio is None or ratio >= _AGGRESSIVE_LOD0_RATIO:
        return
    percent = f"{ratio:.1%}"
    asset.report.add_warning(
        f"decimation target keeps only {percent} of source triangles; ratios below 20% can visibly distort "
        "close-view LOD0 assets and are usually better reserved for distant LODs"
    )


def _requested_decimation_keep_ratio(source_meshes: dict[str, Mesh], options: DecimateOptions) -> float | None:
    source_counts = [mesh.triangle_count for mesh in source_meshes.values() if mesh.triangle_count > 0]
    if not source_counts:
        return None
    if options.target_triangles is not None:
        if options.budget_scope == "selection":
            return min(1.0, options.target_triangles / sum(source_counts))
        return min(1.0, min(options.target_triangles / count for count in source_counts))
    return _decimate_ratio(options)


def _source_meshes(asset: Asset, selected_part_ids: set[str] | None) -> dict[str, Mesh]:
    return {
        part.id: part.mesh.copy()
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    }


def _annotate_decimation_result(
    asset: Asset,
    source_meshes: dict[str, Mesh],
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None,
) -> None:
    source_total = 0
    output_total = 0
    max_error = 0.0
    weighted_error = 0.0
    measured_parts = 0
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        source = source_meshes.get(part.id)
        if source is None or part.mesh is None:
            continue
        metrics = _decimation_metrics(source, part.mesh)
        source_total += metrics.source_triangles
        output_total += metrics.output_triangles
        max_error = max(max_error, metrics.max_vertex_error)
        weighted_error += metrics.mean_vertex_error * max(metrics.source_vertices, 1)
        measured_parts += max(metrics.source_vertices, 1)
        part.metadata = {
            **part.metadata,
            "decimate_criterion": options.criterion,
            "decimate_budget_scope": options.budget_scope,
            "decimate_uv_importance": options.uv_importance,
            "decimate_source_triangles": str(metrics.source_triangles),
            "decimate_output_triangles": str(metrics.output_triangles),
            "decimate_triangle_reduction": f"{metrics.triangle_reduction:.9g}",
            "decimate_max_vertex_error": f"{metrics.max_vertex_error:.9g}",
            "decimate_mean_vertex_error": f"{metrics.mean_vertex_error:.9g}",
            "decimate_error_metric": "symmetric_vertex_nearest_distance",
        }
        removed_uv_channels = sorted(set(source.uvs) - set(part.mesh.uvs))
        if removed_uv_channels:
            part.metadata["decimate_removed_uv_channels"] = ",".join(str(channel) for channel in removed_uv_channels)
    if source_total == 0:
        return
    memory_plan = _decimation_memory_plan(source_total)
    reduction = (source_total - output_total) / source_total
    requested_ratio = _requested_decimation_keep_ratio(source_meshes, options)
    if requested_ratio is not None:
        asset.metadata["decimate_requested_keep_ratio"] = f"{requested_ratio:.9g}"
    asset.metadata["decimate_source_triangles"] = str(source_total)
    asset.metadata["decimate_output_triangles"] = str(output_total)
    asset.metadata["decimate_triangle_reduction"] = f"{reduction:.9g}"
    asset.metadata["decimate_max_vertex_error"] = f"{max_error:.9g}"
    asset.metadata["decimate_mean_vertex_error"] = f"{(weighted_error / measured_parts):.9g}"
    asset.metadata["decimate_error_metric"] = "symmetric_vertex_nearest_distance"
    asset.metadata["decimate_uv_importance"] = options.uv_importance
    asset.metadata["decimate_budget_allocation"] = (
        "global_selection" if options.budget_scope == "selection" else "per_part"
    )
    asset.metadata["decimate_estimated_memory_bytes"] = str(memory_plan.estimated_bytes)
    asset.metadata["decimate_estimated_memory_gb"] = f"{memory_plan.estimated_gb:.9g}"
    asset.metadata["decimate_memory_rule_gb_per_million_triangles"] = (
        f"{_DECIMATION_MEMORY_GB_PER_MILLION_TRIANGLES:.9g}"
    )
    asset.metadata["decimate_iterative_threshold_triangles"] = str(_DECIMATION_ITERATIVE_THRESHOLD_TRIANGLES)
    asset.metadata["decimate_iterative_recommended"] = str(memory_plan.iterative_recommended).lower()
    if memory_plan.iterative_recommended:
        asset.report.add_warning(
            f"decimation estimates {memory_plan.estimated_gb:.3g} GB RAM for {source_total} source triangles; "
            f"iterative decimation is recommended above {_DECIMATION_ITERATIVE_THRESHOLD_TRIANGLES} triangles"
        )
    removed_asset_uv_channels: set[int] = set()
    for part_id, source in source_meshes.items():
        output_part = asset.parts.get(part_id)
        if output_part is None or output_part.mesh is None:
            continue
        removed_asset_uv_channels.update(channel for channel in source.uvs if channel not in output_part.mesh.uvs)
    if removed_asset_uv_channels:
        asset.metadata["decimate_removed_uv_channels"] = ",".join(
            str(channel) for channel in sorted(removed_asset_uv_channels)
        )


@dataclass(frozen=True)
class _DecimationMemoryPlan:
    estimated_bytes: int
    estimated_gb: float
    iterative_recommended: bool


def _decimation_memory_plan(source_triangles: int) -> _DecimationMemoryPlan:
    estimated_bytes = int(np.ceil(source_triangles * _DECIMATION_MEMORY_BYTES_PER_MILLION_TRIANGLES / 1_000_000))
    return _DecimationMemoryPlan(
        estimated_bytes=max(estimated_bytes, 0),
        estimated_gb=source_triangles * _DECIMATION_MEMORY_GB_PER_MILLION_TRIANGLES / 1_000_000,
        iterative_recommended=source_triangles >= _DECIMATION_ITERATIVE_THRESHOLD_TRIANGLES,
    )


def _decimation_metrics(source: Mesh, output: Mesh) -> _DecimationMetrics:
    max_error, mean_error = _symmetric_vertex_error(source.points, output.points)
    reduction = 0.0
    if source.triangle_count:
        reduction = (source.triangle_count - output.triangle_count) / source.triangle_count
    return _DecimationMetrics(
        source_vertices=source.vertex_count,
        source_triangles=source.triangle_count,
        output_triangles=output.triangle_count,
        triangle_reduction=max(0.0, reduction),
        max_vertex_error=max_error,
        mean_vertex_error=mean_error,
    )


def _symmetric_vertex_error(left: FloatArray, right: FloatArray) -> tuple[float, float]:
    if left.size == 0 or right.size == 0:
        return 0.0, 0.0
    left_distances = _nearest_distances(left, right)
    right_distances = _nearest_distances(right, left)
    max_error = max(float(left_distances.max(initial=0.0)), float(right_distances.max(initial=0.0)))
    mean_error = float((left_distances.mean() + right_distances.mean()) * 0.5)
    return max_error, mean_error


def _nearest_distances(points: FloatArray, targets: FloatArray) -> FloatArray:
    distances = np.empty(points.shape[0], dtype=np.float64)
    chunk_size = 2048
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        delta = points[start:end, None, :] - targets[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        distances[start:end] = np.sqrt(squared.min(axis=1))
    return distances


def _enforce_triangle_budget(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None,
) -> None:
    eligible = [
        part
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    ]
    if not eligible:
        return
    if options.target_triangles is not None:
        current_total = sum(cast(Mesh, part.mesh).triangle_count for part in eligible)
        if current_total <= options.target_triangles:
            return
        assigned = 0
        budgets: dict[str, int] = {}
        for part in eligible:
            mesh = cast(Mesh, part.mesh)
            exact = options.target_triangles * (mesh.triangle_count / current_total)
            budget = max(1, min(mesh.triangle_count, int(round(exact))))
            budgets[part.id] = budget
            assigned += budget
        while assigned > options.target_triangles:
            reducible = [part_id for part_id, budget in budgets.items() if budget > 1]
            if not reducible:
                break
            part_id = max(reducible, key=lambda item: budgets[item])
            budgets[part_id] -= 1
            assigned -= 1
        for part in eligible:
            mesh = cast(Mesh, part.mesh)
            if mesh.triangle_count > budgets[part.id]:
                part.mesh = _sample_mesh_faces(mesh, budgets[part.id]).compute_normals()
                part.fingerprint = part.mesh.fingerprint()
        return

    ratio = _decimate_ratio(options)
    for part in eligible:
        mesh = cast(Mesh, part.mesh)
        target = _ratio_target(mesh, ratio)
        if target is not None and mesh.triangle_count > target:
            part.mesh = _sample_mesh_faces(mesh, target).compute_normals()
            part.fingerprint = part.mesh.fingerprint()


def _ratio_target(mesh: Mesh | None, ratio: float | None) -> int | None:
    if mesh is None or ratio is None:
        return None
    return max(1, int(round(mesh.triangle_count * ratio)))


def _sample_mesh_faces(mesh: Mesh, target_triangles: int) -> Mesh:
    target = max(1, min(target_triangles, mesh.triangle_count))
    if target >= mesh.triangle_count:
        return mesh.copy()
    face_indices = np.unique(np.linspace(0, mesh.triangle_count - 1, target, dtype=np.int64))
    while face_indices.shape[0] < target:
        missing = [index for index in range(mesh.triangle_count) if index not in set(face_indices.astype(int).tolist())]
        face_indices = np.sort(np.concatenate([face_indices, np.asarray(missing[: target - face_indices.shape[0]])]))
    return _slice_faces(mesh, face_indices).remove_unreferenced_vertices()


def _slice_faces(mesh: Mesh, face_indices: IntArray) -> Mesh:
    face_lookup = {int(face_index): local_index for local_index, face_index in enumerate(face_indices.tolist())}
    return Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces[face_indices].copy(),
        normals=None if mesh.normals is None else mesh.normals.copy(),
        tangents=None if mesh.tangents is None else mesh.tangents.copy(),
        uvs={channel: values.copy() for channel, values in mesh.uvs.items()},
        material_indices=None if mesh.material_indices is None else mesh.material_indices[face_indices].copy(),
        face_groups={
            name: np.asarray(
                [face_lookup[int(face_index)] for face_index in group.tolist() if int(face_index) in face_lookup],
                dtype=np.int64,
            )
            for name, group in mesh.face_groups.items()
        },
        metadata=dict(mesh.metadata),
    )


def _enabled_hole_kinds(options: RemoveHolesOptions) -> set[str]:
    values: set[str] = set()
    if options.through:
        values.add("through")
    if options.blind:
        values.add("blind")
    if options.surface:
        values.add("surface")
    return values


def _hole_kind_summary(kinds: tuple[str, ...]) -> str:
    return ",".join(kind for kind in ("through", "blind", "surface") if kind in set(kinds))


def _fill_small_holes(mesh: Mesh, options: RemoveHolesOptions) -> tuple[Mesh, _HoleFillStats]:
    candidates = _classified_hole_loops(mesh)
    enabled = _enabled_hole_kinds(options)
    fill_faces: list[list[int]] = []
    diameters: list[float] = []
    kinds: list[str] = []
    for candidate in candidates:
        loop = candidate.loop
        if len(loop) < 3 or len(loop) > 8:
            continue
        if candidate.kind not in enabled:
            continue
        if options.max_diameter is not None and candidate.diameter > options.max_diameter:
            continue
        anchor = loop[0]
        for index in range(1, len(loop) - 1):
            fill_faces.append([anchor, loop[index], loop[index + 1]])
        diameters.append(candidate.diameter)
        kinds.append(candidate.kind)

    if not fill_faces:
        return mesh.copy(), _HoleFillStats(count=0, diameters=(), kinds=())

    filled = mesh.copy()
    filled.faces = np.vstack([mesh.faces, np.asarray(fill_faces, dtype=np.int64)])
    if mesh.material_indices is not None:
        fill_material = int(mesh.material_indices[0]) if mesh.material_indices.size else 0
        filled.material_indices = np.concatenate(
            [mesh.material_indices.copy(), np.full(len(fill_faces), fill_material, dtype=np.int64)]
        )
    return filled, _HoleFillStats(count=len(diameters), diameters=tuple(diameters), kinds=tuple(kinds))


def _classified_hole_loops(mesh: Mesh) -> list[_HoleLoop]:
    loops = _boundary_loops(mesh)
    if not loops:
        return []
    edge_faces = _edge_faces(mesh)
    candidates = [
        _HoleLoop(
            loop=loop,
            diameter=_loop_diameter(mesh.points, loop),
            centroid=cast(FloatArray, mesh.points[np.asarray(loop, dtype=np.int64)].mean(axis=0)),
            normal=_loop_normal(mesh.points, loop),
            adjacent_faces=_loop_adjacent_faces(loop, edge_faces),
        )
        for loop in loops
        if len(loop) >= 3
    ]
    through_indices = _through_loop_indices(candidates)
    classified: list[_HoleLoop] = []
    for index, candidate in enumerate(candidates):
        kind = "through" if index in through_indices else _unpaired_hole_kind(mesh, candidate)
        classified.append(
            _HoleLoop(
                loop=candidate.loop,
                diameter=candidate.diameter,
                centroid=candidate.centroid,
                normal=candidate.normal,
                adjacent_faces=candidate.adjacent_faces,
                kind=kind,
            )
        )
    return classified


def _edge_faces(mesh: Mesh) -> dict[tuple[int, int], list[int]]:
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(mesh.faces.astype(int).tolist()):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(_edge_key(start, end), []).append(face_index)
    return edge_faces


def _loop_adjacent_faces(loop: list[int], edge_faces: dict[tuple[int, int], list[int]]) -> tuple[int, ...]:
    faces: set[int] = set()
    for index, start in enumerate(loop):
        end = loop[(index + 1) % len(loop)]
        faces.update(edge_faces.get(_edge_key(start, end), []))
    return tuple(sorted(faces))


def _through_loop_indices(candidates: list[_HoleLoop]) -> set[int]:
    paired: set[int] = set()
    for left_index, left in enumerate(candidates):
        if left_index in paired:
            continue
        best_index: int | None = None
        best_projected_delta = float("inf")
        for right_index in range(left_index + 1, len(candidates)):
            if right_index in paired:
                continue
            right = candidates[right_index]
            if not _loops_form_through_pair(left, right):
                continue
            delta = right.centroid - left.centroid
            projected = delta - left.normal * float(np.dot(delta, left.normal))
            projected_delta = float(np.linalg.norm(projected))
            if projected_delta < best_projected_delta:
                best_projected_delta = projected_delta
                best_index = right_index
        if best_index is not None:
            paired.add(left_index)
            paired.add(best_index)
    return paired


def _loops_form_through_pair(left: _HoleLoop, right: _HoleLoop) -> bool:
    diameter = max(left.diameter, right.diameter, 1e-9)
    if abs(left.diameter - right.diameter) > diameter * 0.25:
        return False
    if abs(float(np.dot(left.normal, right.normal))) < 0.85:
        return False
    delta = right.centroid - left.centroid
    separation = abs(float(np.dot(delta, left.normal)))
    projected = delta - left.normal * float(np.dot(delta, left.normal))
    projected_delta = float(np.linalg.norm(projected))
    return separation > diameter * 0.05 and projected_delta <= diameter * 0.25


def _unpaired_hole_kind(mesh: Mesh, candidate: _HoleLoop) -> str:
    normals = _face_normals(mesh, candidate.adjacent_faces)
    if normals.size == 0:
        return "surface"
    alignment = np.abs(normals @ candidate.normal)
    if float(np.median(alignment)) < 0.2:
        return "blind"
    return "surface"


def _face_normals(mesh: Mesh, face_indices: tuple[int, ...]) -> FloatArray:
    if not face_indices:
        return np.empty((0, 3), dtype=np.float64)
    faces = mesh.faces[np.asarray(face_indices, dtype=np.int64)]
    p0 = mesh.points[faces[:, 0]]
    p1 = mesh.points[faces[:, 1]]
    p2 = mesh.points[faces[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] = normals[valid] / lengths[valid, None]
    return cast(FloatArray, normals[valid])


def _boundary_loops(mesh: Mesh) -> list[list[int]]:
    if mesh.triangle_count == 0:
        return []
    edge_counts: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    for face in mesh.faces.astype(int).tolist():
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(start, end)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            directed.append((start, end))

    adjacency: dict[int, list[int]] = {}
    for start, end in directed:
        if edge_counts[_edge_key(start, end)] != 1:
            continue
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)

    loops: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for start, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = _edge_key(start, neighbor)
            if edge in visited:
                continue
            loop = [start, neighbor]
            visited.add(edge)
            previous = start
            current = neighbor
            while current != start:
                candidates = [item for item in adjacency.get(current, []) if item != previous]
                if not candidates:
                    break
                next_vertex = candidates[0]
                next_edge = _edge_key(current, next_vertex)
                if next_edge in visited and next_vertex != start:
                    break
                if next_vertex == start:
                    visited.add(next_edge)
                    break
                loop.append(next_vertex)
                visited.add(next_edge)
                previous, current = current, next_vertex
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _edge_key(start: int, end: int) -> tuple[int, int]:
    return (start, end) if start <= end else (end, start)


def _loop_diameter(points: FloatArray, loop: list[int]) -> float:
    loop_points = points[np.asarray(loop, dtype=np.int64)]
    if loop_points.shape[0] < 2:
        return 0.0
    if loop_points.shape[0] >= 3:
        normal = _loop_normal(points, loop)
        first_axis, second_axis = _loop_plane_basis(normal)
        centered = loop_points - loop_points.mean(axis=0)
        projected = np.column_stack([centered @ first_axis, centered @ second_axis])
        spans = projected.max(axis=0) - projected.min(axis=0)
        planar_span = float(spans.max()) if spans.size else 0.0
        if planar_span > 0.0:
            return planar_span
    delta = loop_points[:, None, :] - loop_points[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    return float(distances.max())


def _loop_normal(points: FloatArray, loop: list[int]) -> FloatArray:
    loop_points = points[np.asarray(loop, dtype=np.int64)]
    if loop_points.shape[0] < 3:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    centered = loop_points - loop_points.mean(axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    normal = vh[-1]
    length = float(np.linalg.norm(normal))
    if length <= 0.0:
        return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return cast(FloatArray, normal / length)


def _loop_plane_basis(normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(normal, reference))) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    first = np.cross(normal, reference)
    first_length = float(np.linalg.norm(first))
    first = np.asarray([1.0, 0.0, 0.0], dtype=np.float64) if first_length <= 0.0 else first / first_length
    second = np.cross(normal, first)
    second_length = float(np.linalg.norm(second))
    second = np.asarray([0.0, 1.0, 0.0], dtype=np.float64) if second_length <= 0.0 else second / second_length
    return first, second


def _isolate_selected_occurrence_parts(asset: Asset, selected_node_ids: set[str]) -> None:
    references: dict[str, list[Node]] = {}
    for node in asset.root.walk():
        if node.part_id is not None and node.part_id in asset.parts:
            references.setdefault(node.part_id, []).append(node)

    for part_id, nodes in list(references.items()):
        selected_nodes = [node for node in nodes if node.id in selected_node_ids]
        if not selected_nodes:
            continue
        for node in selected_nodes[:-1] if len(selected_nodes) == len(nodes) else selected_nodes:
            new_part_id = _unique_part_id(asset.parts, f"{part_id}_{node.id}")
            part = asset.parts[part_id].copy(keep_source=True)
            part.id = new_part_id
            part.metadata = {**part.metadata, "source_part_id": part_id}
            asset.parts[new_part_id] = part
            node.part_id = new_part_id


def _candidate_occluders(
    asset: Asset,
    candidate: _WorldOccurrence,
    occurrences: list[_WorldOccurrence],
    options: RemoveOccludedOptions,
) -> list[_WorldOccurrence]:
    return [
        occluder
        for occluder in occurrences
        if occluder.node.id != candidate.node.id
        and (options.consider_transparency_opaque or not _part_is_transparent(asset, occluder.part_id))
    ]


def _occlusion_directions(options: RemoveOccludedOptions) -> list[FloatArray]:
    if options.strategy == "conservative":
        vectors = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
    elif options.strategy == "exterior":
        vectors = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)] + [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
    else:
        vectors = [
            (x, y, z)
            for x in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for z in (-1.0, 0.0, 1.0)
            if (x, y, z) != (0.0, 0.0, 0.0)
        ]
    if options.hemi_evaluation:
        vectors = [vector for vector in vectors if vector[2] >= 0.0]
    directions: list[FloatArray] = []
    for vector in vectors:
        direction = np.asarray(vector, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length > 0.0:
            directions.append(direction / length)
    return directions


def _occlusion_ray_distance(occurrences: list[_WorldOccurrence]) -> float:
    if not occurrences:
        return 1.0
    bounds_min = np.vstack([occurrence.bounds_min for occurrence in occurrences]).min(axis=0)
    bounds_max = np.vstack([occurrence.bounds_max for occurrence in occurrences]).max(axis=0)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    return max(diagonal * 2.0, 1.0)


def _record_occlusion_confidence(
    asset: Asset,
    measurements: list[_OcclusionMeasurement],
    candidate_count: int,
    direction_count: int,
    options: RemoveOccludedOptions,
) -> None:
    face_count = sum(measurement.face_count for measurement in measurements)
    sample_count = sum(measurement.sample_count for measurement in measurements)
    visible_samples = sum(measurement.visible_sample_count for measurement in measurements)
    hidden_samples = sum(measurement.hidden_sample_count for measurement in measurements)
    sample_coverage = 1.0 if face_count == 0 else min(1.0, sample_count / face_count)
    max_directions = _max_occlusion_direction_count(options)
    direction_coverage = 1.0 if max_directions == 0 else min(1.0, direction_count / max_directions)
    confidence = min(sample_coverage, direction_coverage)

    asset.metadata["occlusion_candidate_count"] = str(candidate_count)
    asset.metadata["occlusion_face_count"] = str(face_count)
    asset.metadata["occlusion_sample_count"] = str(sample_count)
    asset.metadata["occlusion_visible_sample_count"] = str(visible_samples)
    asset.metadata["occlusion_hidden_sample_count"] = str(hidden_samples)
    asset.metadata["occlusion_sample_coverage"] = f"{sample_coverage:.6g}"
    asset.metadata["occlusion_direction_coverage"] = f"{direction_coverage:.6g}"
    asset.metadata["occlusion_confidence"] = f"{confidence:.6g}"


def _max_occlusion_direction_count(options: RemoveOccludedOptions) -> int:
    return len(
        _occlusion_directions(RemoveOccludedOptions(strategy="advanced", hemi_evaluation=options.hemi_evaluation))
    )


def _occurrence_visibility_measurement(
    candidate: _WorldOccurrence,
    occluders: list[_WorldOccurrence],
    directions: list[FloatArray],
    ray_distance: float,
    options: RemoveOccludedOptions,
) -> _OcclusionMeasurement:
    samples = _occurrence_visibility_samples(candidate, options.precision)
    face_count = int(candidate.faces.shape[0])
    if samples.size == 0 or not occluders:
        return _OcclusionMeasurement(
            face_count=face_count,
            sample_count=int(samples.shape[0]),
            visible_sample_count=int(samples.shape[0]),
            hidden_sample_count=0,
        )
    visible = np.asarray(
        [_sample_is_visible(sample, occluders, directions, ray_distance) for sample in samples],
        dtype=np.bool_,
    )
    visible_count = int(np.count_nonzero(visible))
    sample_count = int(visible.shape[0])
    return _OcclusionMeasurement(
        face_count=face_count,
        sample_count=sample_count,
        visible_sample_count=visible_count,
        hidden_sample_count=sample_count - visible_count,
    )


def _occurrence_visibility_samples(candidate: _WorldOccurrence, precision: int) -> FloatArray:
    face_samples = _face_centers(candidate)
    if face_samples.shape[0] > precision:
        indices = np.unique(np.linspace(0, face_samples.shape[0] - 1, precision, dtype=np.int64))
        face_samples = face_samples[indices]
    mins = candidate.bounds_min
    maxs = candidate.bounds_max
    box_samples = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
            [(mins[0] + maxs[0]) * 0.5, mins[1], (mins[2] + maxs[2]) * 0.5],
            [(mins[0] + maxs[0]) * 0.5, maxs[1], (mins[2] + maxs[2]) * 0.5],
            [mins[0], (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5],
            [maxs[0], (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5],
            [(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, mins[2]],
            [(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, maxs[2]],
        ],
        dtype=np.float64,
    )
    if face_samples.size == 0:
        return box_samples
    return cast(FloatArray, np.vstack([face_samples, box_samples]))


def _visible_face_measurement(
    candidate: _WorldOccurrence,
    occluders: list[_WorldOccurrence],
    directions: list[FloatArray],
    ray_distance: float,
) -> tuple[NDArray[np.bool_], _OcclusionMeasurement]:
    centers = _face_centers(candidate)
    if centers.size == 0 or not occluders:
        visible_faces = np.ones(candidate.faces.shape[0], dtype=np.bool_)
        sample_count = int(visible_faces.shape[0])
        return visible_faces, _OcclusionMeasurement(
            face_count=int(candidate.faces.shape[0]),
            sample_count=sample_count,
            visible_sample_count=sample_count,
            hidden_sample_count=0,
        )
    visible_faces = np.asarray(
        [_sample_is_visible(center, occluders, directions, ray_distance) for center in centers],
        dtype=np.bool_,
    )
    visible_count = int(np.count_nonzero(visible_faces))
    sample_count = int(visible_faces.shape[0])
    return visible_faces, _OcclusionMeasurement(
        face_count=int(candidate.faces.shape[0]),
        sample_count=sample_count,
        visible_sample_count=visible_count,
        hidden_sample_count=sample_count - visible_count,
    )


def _face_centers(occurrence: _WorldOccurrence) -> FloatArray:
    if occurrence.faces.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    triangles = occurrence.world_points[occurrence.faces]
    return cast(FloatArray, triangles.mean(axis=1))


def _sample_is_visible(
    sample: FloatArray,
    occluders: list[_WorldOccurrence],
    directions: list[FloatArray],
    ray_distance: float,
) -> bool:
    for direction in directions:
        origin = sample + direction * ray_distance
        if not _segment_blocked(origin, sample, occluders):
            return True
    return False


def _segment_blocked(start: FloatArray, end: FloatArray, occluders: list[_WorldOccurrence]) -> bool:
    for occluder in occluders:
        if not _segment_intersects_bounds(start, end, occluder.bounds_min, occluder.bounds_max):
            continue
        if _segment_intersects_mesh(start, end, occluder.world_points, occluder.faces):
            return True
    return False


def _segment_intersects_mesh(start: FloatArray, end: FloatArray, points: FloatArray, faces: IntArray) -> bool:
    for face in faces.astype(int).tolist():
        triangle = points[np.asarray(face, dtype=np.int64)]
        hit = _segment_triangle_t(start, end, triangle)
        if hit is not None and 1e-8 < hit < 1.0 - 1e-8:
            return True
    return False


def _segment_triangle_t(start: FloatArray, end: FloatArray, triangle: FloatArray) -> float | None:
    epsilon = 1e-12
    direction = end - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) <= epsilon:
        return None
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < -epsilon or u > 1.0 + epsilon:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < -epsilon or u + v > 1.0 + epsilon:
        return None
    t = float(np.dot(edge2, qvec)) * inverse
    if t < -epsilon or t > 1.0 + epsilon:
        return None
    return t


def _segment_intersects_bounds(
    start: FloatArray,
    end: FloatArray,
    bounds_min: FloatArray,
    bounds_max: FloatArray,
) -> bool:
    epsilon = 1e-12
    direction = end - start
    tmin = 0.0
    tmax = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= epsilon:
            if start[axis] < bounds_min[axis] - epsilon or start[axis] > bounds_max[axis] + epsilon:
                return False
            continue
        inverse = 1.0 / float(direction[axis])
        near = (float(bounds_min[axis]) - float(start[axis])) * inverse
        far = (float(bounds_max[axis]) - float(start[axis])) * inverse
        if near > far:
            near, far = far, near
        tmin = max(tmin, near)
        tmax = min(tmax, far)
        if tmin > tmax:
            return False
    return True


def _submesh_keep_mask(mesh: Mesh, visible_faces: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if mesh.material_indices is None:
        return cast(NDArray[np.bool_], visible_faces.copy())
    keep = np.zeros(mesh.triangle_count, dtype=np.bool_)
    for material_index in np.unique(mesh.material_indices):
        group = mesh.material_indices == material_index
        if bool(np.any(visible_faces[group])):
            keep[group] = True
    return keep


def _expand_face_mask(mesh: Mesh, visible_faces: NDArray[np.bool_], rings: int) -> NDArray[np.bool_]:
    keep = visible_faces.copy()
    if rings <= 0 or not bool(np.any(keep)):
        return cast(NDArray[np.bool_], keep)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(mesh.faces.astype(int).tolist()):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(_edge_key(start, end), []).append(face_index)
    neighbors: list[set[int]] = [set() for _ in range(mesh.triangle_count)]
    for face_indices in edge_faces.values():
        if len(face_indices) < 2:
            continue
        for face_index in face_indices:
            neighbors[face_index].update(index for index in face_indices if index != face_index)
    frontier = set(np.flatnonzero(keep).astype(int).tolist())
    for _ in range(rings):
        next_frontier: set[int] = set()
        for face_index in frontier:
            next_frontier.update(neighbors[face_index])
        next_frontier.difference_update(np.flatnonzero(keep).astype(int).tolist())
        if not next_frontier:
            break
        keep[np.asarray(sorted(next_frontier), dtype=np.int64)] = True
        frontier = next_frontier
    return cast(NDArray[np.bool_], keep)


def _apply_occlusion_trims(
    asset: Asset,
    trims: list[_OcclusionTrim],
    removed_node_ids: set[str],
    options: RemoveOccludedOptions,
) -> int:
    removed_faces_total = 0
    for trim in trims:
        if trim.node_id in removed_node_ids:
            continue
        part = asset.parts.get(trim.part_id)
        if part is None or part.mesh is None:
            continue
        mesh = _slice_faces(part.mesh, trim.keep_faces).remove_unreferenced_vertices()
        if mesh.triangle_count:
            mesh = mesh.compute_normals()
        _compact_material_slots(part, mesh)
        mesh.metadata = {
            **mesh.metadata,
            "occlusion_level": options.level,
            "occlusion_removed_faces": str(trim.removed_faces),
        }
        mesh.validate()
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "occlusion_level": options.level,
            "occlusion_removed_faces": str(trim.removed_faces),
        }
        part.fingerprint = mesh.fingerprint()
        removed_faces_total += trim.removed_faces
    return removed_faces_total


def _removed_node_triangle_count(
    asset: Asset,
    selected_occurrences: list[_WorldOccurrence],
    removed_node_ids: set[str],
) -> int:
    total = 0
    for occurrence in selected_occurrences:
        if occurrence.node.id not in removed_node_ids:
            continue
        part = asset.parts.get(occurrence.part_id)
        if part is not None and part.mesh is not None:
            total += part.mesh.triangle_count
    return total


def _compact_material_slots(part: Part, mesh: Mesh) -> None:
    if mesh.material_indices is None:
        return
    material_ids = part.material_ids
    used = sorted({int(index) for index in mesh.material_indices.astype(int).tolist()})
    if not used or any(index < 0 or index >= len(material_ids) for index in used):
        return
    remap = {old: new for new, old in enumerate(used)}
    mesh.material_indices = np.asarray([remap[int(index)] for index in mesh.material_indices], dtype=np.int64)
    part.material_ids = [material_ids[index] for index in used]


def _world_occurrences(asset: Asset) -> list[_WorldOccurrence]:
    occurrences: list[_WorldOccurrence] = []

    def walk(node: Node, world: FloatArray) -> None:
        current = world @ node.transform
        if node.part_id is not None and node.part_id in asset.parts:
            part = asset.parts[node.part_id]
            if part.mesh is not None:
                world_points = _transform_points(part.mesh.points, current)
                if world_points.shape[0] == 0:
                    mins, maxs = part.mesh.bounds()
                    world_min, world_max = _transform_bounds(mins, maxs, current)
                else:
                    world_min = cast(FloatArray, world_points.min(axis=0))
                    world_max = cast(FloatArray, world_points.max(axis=0))
                volume = float(np.prod(np.maximum(world_max - world_min, 0.0)))
                occurrences.append(
                    _WorldOccurrence(
                        node=node,
                        part_id=node.part_id,
                        world_points=world_points,
                        faces=part.mesh.faces.copy(),
                        bounds_min=world_min,
                        bounds_max=world_max,
                        volume=volume,
                    )
                )
        for child in node.children:
            walk(child, current)

    walk(asset.root, np.eye(4, dtype=np.float64))
    return occurrences


def _transform_points(points: FloatArray, transform: FloatArray) -> FloatArray:
    if points.shape[0] == 0:
        return cast(FloatArray, points.copy())
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return cast(FloatArray, (transform @ homogeneous.T).T[:, :3])


def _transform_bounds(mins: FloatArray, maxs: FloatArray, transform: FloatArray) -> tuple[FloatArray, FloatArray]:
    corners = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([corners, np.ones(corners.shape[0], dtype=np.float64)])
    transformed = (transform @ homogeneous.T).T[:, :3]
    return cast(FloatArray, transformed.min(axis=0)), cast(FloatArray, transformed.max(axis=0))


def _bbox_contains(outer_min: FloatArray, outer_max: FloatArray, inner_min: FloatArray, inner_max: FloatArray) -> bool:
    epsilon = 1e-9
    return bool(np.all(inner_min >= outer_min - epsilon) and np.all(inner_max <= outer_max + epsilon))


def _preserve_candidate_cavity(asset: Asset, candidate: _WorldOccurrence, options: RemoveOccludedOptions) -> bool:
    if not options.preserve_cavities:
        return False
    volume_m3 = candidate.volume * asset.meters_per_unit**3
    return volume_m3 >= options.minimum_cavity_volume_m3


def _part_is_transparent(asset: Asset, part_id: str) -> bool:
    part = asset.parts.get(part_id)
    if part is None:
        return False
    return any(
        asset.materials[material_id].opacity < 1.0
        for material_id in part.material_ids
        if material_id in asset.materials
    )


def _remove_part_nodes(node: Node, remove_node_ids: set[str]) -> bool:
    kept: list[Node] = []
    for child in node.children:
        keep_child = _remove_part_nodes(child, remove_node_ids)
        if child.id in remove_node_ids:
            child.part_id = None
            if child.children:
                kept.append(child)
            continue
        if keep_child:
            kept.append(child)
    node.children = kept
    return node.part_id is not None or bool(node.children)


def _drop_unreferenced_parts(asset: Asset) -> None:
    referenced = {node.part_id for node in asset.root.walk() if node.part_id is not None}
    asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id in referenced}


def _drop_unreferenced_materials(asset: Asset) -> None:
    referenced = {material_id for part in asset.parts.values() for material_id in part.material_ids}
    asset.materials = {
        material_id: material for material_id, material in asset.materials.items() if material_id in referenced
    }


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


def _unique_material_id(materials: dict[str, Material], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in materials:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _unique_part_id(parts: dict[str, Part], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in parts:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
