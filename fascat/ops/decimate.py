from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Part
from fascat.mesh import Mesh
from fascat.ops._mesh_utils import selected_mesh_part_id_list, slice_faces
from fascat.ops._visibility import face_ambient_occlusion
from fascat.ops.parallel import parallel_map
from fascat.options import DecimateOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_AGGRESSIVE_LOD0_RATIO = 0.2
_AO_IMPORTANCE_THRESHOLD = 0.5
_DECIMATION_MEMORY_BYTES_PER_MILLION_TRIANGLES = 5_000_000_000
_DECIMATION_MEMORY_GB_PER_MILLION_TRIANGLES = 5.0
_PAINTED_FACE_GROUP_TOKENS = (
    "paint",
    "painted",
    "protect",
    "protected",
    "preserve",
    "keep",
    "weight",
    "weighted",
    "importance",
)
_PAINTED_FACE_METADATA_KEYS = (
    "painted_faces",
    "protected_faces",
    "decimate_protected_faces",
    "simplification_protected_faces",
    "importance_faces",
)


def decimate_asset(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    source_meshes = _source_meshes(asset, selected_part_ids)
    working_asset = _prepare_decimation_asset(asset, options, selected_part_ids)
    pass_counts: dict[str, _DecimationPassCount] = {}
    part_targets: dict[str, int] = {}
    if options.budget_scope == "selection":
        result, pass_counts, part_targets = _decimate_selection_budget(
            working_asset,
            options,
            selected_part_ids=selected_part_ids,
        )
        _warn_aggressive_lod0_decimation(result, source_meshes, options)
        _enforce_triangle_budget(result, options, selected_part_ids=selected_part_ids)
        _finalize_decimation_uv_importance(result, options, selected_part_ids=selected_part_ids)
        _annotate_decimation_result(
            result,
            source_meshes,
            options,
            selected_part_ids=selected_part_ids,
            pass_counts=pass_counts,
            part_targets=part_targets,
        )
        return result

    result = working_asset.copy(keep_source=True)
    ratio = _decimate_ratio(options)
    _warn_aggressive_lod0_decimation(result, source_meshes, options)
    part_ids = selected_mesh_part_id_list(result, selected_part_ids)

    payloads = [
        _DecimateBudgetPayload(part=result.parts[part_id].copy(keep_source=False), options=options, ratio=ratio)
        for part_id in part_ids
    ]

    for decimated in parallel_map(payloads, _decimate_budget_worker, jobs=options.jobs, executor="process"):
        part = result.parts[decimated.part_id]
        part.mesh = decimated.mesh
        part.fingerprint = decimated.fingerprint
        pass_counts[part.id] = decimated.pass_count
        if decimated.target_budget is not None:
            part_targets[part.id] = decimated.target_budget
    _annotate_decimation_result(
        result,
        source_meshes,
        options,
        selected_part_ids=selected_part_ids,
        pass_counts=pass_counts,
        part_targets=part_targets,
    )
    return result


def decimation_target_strategy(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> dict[str, object]:
    return _decimation_target_strategy(_source_meshes(asset, selected_part_ids), options)


@dataclass(frozen=True)
class _DecimationMetrics:
    source_vertices: int
    source_triangles: int
    output_triangles: int
    triangle_reduction: float
    max_vertex_error: float
    mean_vertex_error: float


@dataclass(frozen=True)
class _DecimatedPart:
    part_id: str
    mesh: Mesh
    fingerprint: str
    pass_count: _DecimationPassCount
    target_budget: int | None


def _decimate_selection_part(
    part: Part,
    options: DecimateOptions,
    *,
    target: int | None,
    ratio: float | None,
) -> _DecimatedPart:
    if part.mesh is None:
        raise AssertionError("selected decimation part must have a mesh")
    target_budget = target if target is not None else _ratio_target(part.mesh, ratio)
    mesh, counts = _simplify_mesh_for_decimation(
        part.mesh,
        options,
        target_triangles=target,
        ratio=None if target is not None else ratio,
    )
    mesh.validate()
    mesh = mesh.optimize_buffers()
    mesh.validate()
    mesh = mesh.repair()
    return _DecimatedPart(
        part_id=part.id,
        mesh=mesh,
        fingerprint=mesh.fingerprint(),
        pass_count=counts,
        target_budget=target_budget,
    )


@dataclass(frozen=True)
class _DecimateBudgetPayload:
    part: Part
    options: DecimateOptions
    ratio: float | None


def _decimate_budget_worker(payload: _DecimateBudgetPayload) -> _DecimatedPart:
    return _decimate_part_budget(payload.part, payload.options, payload.ratio)


@dataclass(frozen=True)
class _DecimateSelectionPayload:
    part: Part
    options: DecimateOptions
    target: int | None
    ratio: float | None


def _decimate_selection_worker(payload: _DecimateSelectionPayload) -> _DecimatedPart:
    return _decimate_selection_part(payload.part, payload.options, target=payload.target, ratio=payload.ratio)


def _decimate_part_budget(part: Part, options: DecimateOptions, ratio: float | None) -> _DecimatedPart:
    if part.mesh is None:
        raise AssertionError("selected decimation part must have a mesh")
    target = options.target_triangles
    if target is not None:
        target = min(target, part.mesh.triangle_count)
    target_budget = target if target is not None else _ratio_target(part.mesh, ratio)
    mesh, counts = _simplify_mesh_for_decimation(
        part.mesh,
        options,
        target_triangles=target,
        ratio=None if target is not None else ratio,
    )
    mesh = mesh.optimize_buffers().repair()
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
    return _DecimatedPart(
        part_id=part.id,
        mesh=mesh,
        fingerprint=mesh.fingerprint(),
        pass_count=counts,
        target_budget=target_budget,
    )


def _decimate_ratio(options: DecimateOptions) -> float | None:
    if options.target_triangles is not None:
        return None
    if options.target_ratio is not None:
        return options.target_ratio
    if options.criterion == "quality":
        return None
    return 0.5


def _decimation_target_strategy(source_meshes: dict[str, Mesh], options: DecimateOptions) -> dict[str, object]:
    kind, source, workflow = _decimation_target_strategy_kind(options)
    source_triangles = sum(mesh.triangle_count for mesh in source_meshes.values() if mesh.triangle_count > 0)
    effective_ratio = _requested_decimation_keep_ratio(source_meshes, options)
    strategy: dict[str, object] = {
        "kind": kind,
        "source": source,
        "workflow": workflow,
        "backend_mode": "mesh_simplify",
        "criterion": options.criterion,
        "budget_scope": options.budget_scope,
        "source_triangles": source_triangles,
        "target_triangles": options.target_triangles,
        "target_ratio": options.target_ratio,
        "effective_keep_ratio": effective_ratio,
        "quality_bound_policy": "hint" if kind == "quality_error" else "not_applicable",
        "quality_bound_enforced": False,
        "quality_bound_status": "hint" if kind == "quality_error" else "not_applicable",
    }
    if kind == "quality_error":
        strategy.update(
            {
                "surface_tolerance": options.surface_tolerance,
                "line_tolerance": options.line_tolerance,
                "uv_tolerance": options.uv_tolerance,
                "quality_error_bound": _quality_error_bound(options),
            }
        )
    return strategy


def _decimation_target_strategy_kind(options: DecimateOptions) -> tuple[str, str, str]:
    if options.target_triangles is not None:
        return "target_count", "explicit_target_triangles", "unity_target_polygon_count"
    if options.target_ratio is not None:
        return "target_ratio", "explicit_target_ratio", "unity_target_ratio"
    if options.criterion == "quality":
        return "quality_error", "meshoptimizer_target_error", "meshoptimizer_target_error_hint"
    return "target_ratio", "default_target_ratio", "unity_target_ratio"


def _decimation_target_strategy_metadata(strategy: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        "decimate_target_strategy": str(strategy["kind"]),
        "decimate_target_strategy_source": str(strategy["source"]),
        "decimate_target_strategy_workflow": str(strategy["workflow"]),
        "decimate_target_strategy_backend": str(strategy["backend_mode"]),
        "decimate_quality_bound_policy": str(strategy["quality_bound_policy"]),
        "decimate_quality_bound_status": str(strategy["quality_bound_status"]),
        "decimate_quality_bound_enforced": str(strategy["quality_bound_enforced"]).lower(),
    }
    for key in ("target_triangles", "target_ratio", "effective_keep_ratio"):
        value = strategy.get(key)
        if value is not None:
            metadata[f"decimate_{key}"] = _format_metadata_value(value)
    if strategy.get("quality_error_bound") is not None:
        metadata["decimate_quality_error_bound"] = _format_metadata_value(strategy["quality_error_bound"])
    return metadata


def _format_metadata_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float):
        return f"{value:.9g}"
    return str(value)


def _preserve_uv_seams(options: DecimateOptions) -> bool:
    return options.uv_importance in {"preserve_islands", "preserve_seams"}


def _decimation_importance_faces(mesh: Mesh, options: DecimateOptions) -> IntArray:
    groups = _decimation_importance_face_groups(mesh, options)
    if not groups:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(list(groups.values())))


def _decimation_importance_face_groups(mesh: Mesh, options: DecimateOptions) -> dict[str, IntArray]:
    groups: dict[str, IntArray] = {}
    if options.preserve_painted_areas:
        painted = _painted_face_indices(mesh)
        if painted.size:
            groups["painted_area_faces"] = painted
    if options.preserve_ambient_occlusion:
        ao = _ambient_occlusion_importance_faces(mesh, options.ambient_occlusion_strategy)
        if ao.size:
            groups["ambient_occlusion_faces"] = ao
    return groups


def _painted_face_indices(mesh: Mesh) -> IntArray:
    protected: list[IntArray] = []
    for name, values in mesh.face_groups.items():
        normalized = name.lower().replace("-", "_").replace(" ", "_")
        if any(token in normalized for token in _PAINTED_FACE_GROUP_TOKENS):
            protected.append(values.astype(np.int64, copy=False))
    for key in _PAINTED_FACE_METADATA_KEYS:
        parsed = _metadata_face_indices(mesh.metadata.get(key), mesh.triangle_count)
        if parsed.size:
            protected.append(parsed)
    if not protected:
        return np.empty(0, dtype=np.int64)
    faces = np.unique(np.concatenate(protected))
    return faces[(faces >= 0) & (faces < mesh.triangle_count)]


def _metadata_face_indices(value: object, triangle_count: int) -> IntArray:
    if value is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(value, str):
        tokens: list[object] = [token for token in re.split(r"[\s,;|]+", value) if token]
    elif isinstance(value, list | tuple | np.ndarray):
        tokens = list(value)
    else:
        tokens = [value]
    indices: list[int] = []
    for item in tokens:
        if isinstance(item, bool):
            continue
        try:
            index = int(cast(Any, item))
        except (TypeError, ValueError):
            continue
        if 0 <= index < triangle_count:
            indices.append(index)
    return np.asarray(sorted(set(indices)), dtype=np.int64)


def _ambient_occlusion_importance_faces(mesh: Mesh, strategy: str) -> IntArray:
    values = face_ambient_occlusion(mesh, strategy)
    if values.size == 0:
        return np.empty(0, dtype=np.int64)
    return np.flatnonzero(values <= _AO_IMPORTANCE_THRESHOLD).astype(np.int64)


def _prepare_decimation_asset(
    asset: Asset,
    options: DecimateOptions,
    selected_part_ids: set[str] | None,
) -> Asset:
    if options.uv_importance != "ignore" and not options.cleanup_attributes:
        return asset
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        plan = _decimation_pre_cleanup_plan(part.mesh, options)
        if not plan.removes_attributes:
            continue
        part.mesh = _apply_decimation_pre_cleanup(part.mesh, plan)
        part.fingerprint = part.mesh.fingerprint()
    return result


def _decimate_selection_budget(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None,
) -> tuple[Asset, dict[str, _DecimationPassCount], dict[str, int]]:
    from fascat.ops.optimize import _selected_triangle_count, _targets_for_parts

    result = asset.copy(keep_source=True)
    total_triangles = _selected_triangle_count(result.parts, selected_part_ids)
    targets = _targets_for_parts(
        result.parts,
        total_triangles,
        options.target_triangles,
        selected_part_ids=selected_part_ids,
    )
    if targets is not None and sum(targets.values()) > (options.target_triangles or 0):
        result.report.add_warning(
            "target_triangles is lower than the number of non-empty unique meshes; using one triangle per mesh"
        )

    ratio = _decimate_ratio(options)
    pass_counts: dict[str, _DecimationPassCount] = {}
    part_targets: dict[str, int] = {}
    part_ids = selected_mesh_part_id_list(result, selected_part_ids)

    payloads = [
        _DecimateSelectionPayload(
            part=result.parts[part_id].copy(keep_source=False),
            options=options,
            target=targets.get(part_id) if targets is not None else None,
            ratio=ratio,
        )
        for part_id in part_ids
    ]

    for decimated in parallel_map(payloads, _decimate_selection_worker, jobs=options.jobs, executor="process"):
        part = result.parts[decimated.part_id]
        part.mesh = decimated.mesh
        part.fingerprint = decimated.fingerprint
        pass_counts[part.id] = decimated.pass_count
        if decimated.target_budget is not None:
            part_targets[part.id] = decimated.target_budget
    return result, pass_counts, part_targets


def _simplify_mesh_for_decimation(
    mesh: Mesh,
    options: DecimateOptions,
    *,
    target_triangles: int | None,
    ratio: float | None,
) -> tuple[Mesh, _DecimationPassCount]:
    final_target = target_triangles
    if final_target is None and ratio is not None:
        final_target = _ratio_target(mesh, ratio)
    if final_target is None and options.criterion == "quality":
        return _simplify_mesh_error_bounded(mesh, options)
    if final_target is None:
        return mesh.copy(), _DecimationPassCount()

    final_target = max(1, min(int(final_target), mesh.triangle_count))
    if final_target >= mesh.triangle_count:
        return mesh.copy(), _DecimationPassCount()

    current = mesh
    simplification_passes = 0
    iterative_passes = 0
    while current.triangle_count > options.iterative_threshold and final_target < options.iterative_threshold:
        next_target = max(
            final_target,
            options.iterative_threshold,
            int(math.ceil(current.triangle_count * 0.5)),
        )
        if next_target >= current.triangle_count:
            break
        previous_count = current.triangle_count
        current = _simplify_decimation_once(current, options, next_target)
        simplification_passes += 1
        iterative_passes += 1
        if current.triangle_count >= previous_count:
            break

    if current.triangle_count > final_target:
        current = _simplify_decimation_once(current, options, final_target)
        simplification_passes += 1

    return current, _DecimationPassCount(
        simplification_passes=simplification_passes,
        iterative_passes=iterative_passes,
    )


def _simplify_decimation_once(mesh: Mesh, options: DecimateOptions, target_triangles: int) -> Mesh:
    return mesh.simplify(
        target_triangles=target_triangles,
        target_error=_quality_error_bound(options) if options.criterion == "quality" else None,
        preserve_hard_edges=True,
        hard_edge_angle=options.normal_tolerance,
        preserve_holes=options.protect_topology,
        preserve_material_boundaries=True,
        preserve_uv_seams=_preserve_uv_seams(options),
        preserve_silhouette=options.protect_topology,
        protected_faces=_decimation_importance_faces(mesh, options),
    )


def _simplify_mesh_error_bounded(mesh: Mesh, options: DecimateOptions) -> tuple[Mesh, _DecimationPassCount]:
    simplified = mesh.simplify(
        target_error=_quality_error_bound(options),
        preserve_hard_edges=True,
        hard_edge_angle=options.normal_tolerance,
        preserve_holes=options.protect_topology,
        preserve_material_boundaries=True,
        preserve_uv_seams=_preserve_uv_seams(options),
        preserve_silhouette=options.protect_topology,
        protected_faces=_decimation_importance_faces(mesh, options),
    )
    passes = 1 if simplified.triangle_count < mesh.triangle_count else 0
    return simplified, _DecimationPassCount(simplification_passes=passes, iterative_passes=0)


def _quality_error_bound(options: DecimateOptions) -> float:
    tolerances = [value for value in (options.surface_tolerance, options.line_tolerance, options.uv_tolerance) if value]
    return max(tolerances) if tolerances else 0.01


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


def _apply_decimation_pre_cleanup(mesh: Mesh, plan: _DecimationPreCleanupPlan) -> Mesh:
    if not plan.removes_attributes:
        return mesh
    result = mesh.copy()
    if plan.removed_uv_channels:
        result.uvs = {
            channel: values for channel, values in result.uvs.items() if channel not in plan.removed_uv_channels
        }
    if plan.removed_tangents:
        result.tangents = None
    return result


def _decimation_pre_cleanup_plan(mesh: Mesh, options: DecimateOptions) -> _DecimationPreCleanupPlan:
    requested = tuple(options.cleanup_attributes)
    source_uv_channels = tuple(sorted(mesh.uvs))
    removed_uv_channels: set[int] = set()
    if options.uv_importance == "ignore":
        removed_uv_channels.update(source_uv_channels)
    elif "unused_uvs" in requested:
        removed_uv_channels.update(channel for channel in source_uv_channels if _is_unused_uv_channel(mesh, channel))

    preserved_uv_channels = tuple(channel for channel in source_uv_channels if channel not in removed_uv_channels)
    removed_tangents = bool(
        mesh.tangents is not None
        and (
            "tangents" in requested
            or options.uv_importance == "ignore"
            or (source_uv_channels and len(removed_uv_channels) == len(source_uv_channels))
        )
    )
    uv_constraint_status = "none"
    if options.uv_importance == "ignore":
        uv_constraint_status = "ignored"
    elif preserved_uv_channels:
        uv_constraint_status = "preserved_for_simplification"
    elif removed_uv_channels:
        uv_constraint_status = "cleanup_removed_unused_uvs"

    return _DecimationPreCleanupPlan(
        requested=requested,
        removed_uv_channels=tuple(sorted(removed_uv_channels)),
        preserved_uv_channels=preserved_uv_channels,
        removed_tangents=removed_tangents,
        uv_constraint_status=uv_constraint_status,
        uv_seam_vertices=len(mesh._uv_seam_vertices()) if preserved_uv_channels else 0,
    )


def _is_unused_uv_channel(mesh: Mesh, channel: int) -> bool:
    uv = mesh.uvs[channel]
    if uv.shape[0] == 0 or uv.shape[0] < mesh.vertex_count or mesh.triangle_count == 0:
        return True
    span = uv.max(axis=0) - uv.min(axis=0)
    if float(np.max(np.abs(span))) <= 1e-12:
        return True
    triangles = uv[mesh.faces]
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    areas = np.abs(0.5 * (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]))
    return not bool(np.any(areas > 1e-12))


def _mesh_without_texture_coordinates(mesh: Mesh) -> Mesh:
    if not mesh.uvs and mesh.tangents is None:
        return mesh
    result = mesh.copy()
    result.uvs = {}
    result.tangents = None
    return result


def _quality_decimation_warning() -> str:
    return (
        "decimate quality criterion passes tolerances as a target-error hint and records measured vertex error; "
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
    pass_counts: dict[str, _DecimationPassCount],
    part_targets: dict[str, int],
) -> None:
    target_strategy = _decimation_target_strategy(source_meshes, options)
    target_strategy_metadata = _decimation_target_strategy_metadata(target_strategy)
    source_total = 0
    output_total = 0
    max_error = 0.0
    weighted_error = 0.0
    measured_parts = 0
    simplification_passes = 0
    iterative_passes = 0
    max_part_passes = 0
    pre_cleanup_removed_uv_channels: set[int] = set()
    pre_cleanup_removed_tangent_parts = 0
    uv_constrained_parts = 0
    uv_seam_constraint_vertices = 0
    protected_feature_parts = 0
    allocated_target_total = 0
    allocation_part_count = 0
    allocation_preserved_parts = 0
    allocation_reduced_parts = 0
    allocation_targets: list[int] = []
    feature_totals = {
        "hard_edge_faces": 0,
        "hole_boundary_faces": 0,
        "material_boundary_faces": 0,
        "uv_seam_faces": 0,
        "silhouette_faces": 0,
        "total_feature_faces": 0,
        "painted_area_faces": 0,
        "ambient_occlusion_faces": 0,
        "importance_faces": 0,
    }
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        source = source_meshes.get(part.id)
        if source is None or part.mesh is None:
            continue
        pre_cleanup = _decimation_pre_cleanup_plan(source, options)
        feature_counts = _decimation_feature_counts(source, options)
        metrics = _decimation_metrics(source, part.mesh)
        counts = pass_counts.get(part.id, _DecimationPassCount())
        source_total += metrics.source_triangles
        output_total += metrics.output_triangles
        max_error = max(max_error, metrics.max_vertex_error)
        weighted_error += metrics.mean_vertex_error * max(metrics.source_vertices, 1)
        measured_parts += max(metrics.source_vertices, 1)
        simplification_passes += counts.simplification_passes
        iterative_passes += counts.iterative_passes
        max_part_passes = max(max_part_passes, counts.simplification_passes)
        metadata = {
            **part.metadata,
            **target_strategy_metadata,
            "decimate_criterion": options.criterion,
            "decimate_budget_scope": options.budget_scope,
            "decimate_uv_importance": options.uv_importance,
            "decimate_pre_cleanup_attributes": ",".join(options.cleanup_attributes) or "none",
            "decimate_uv_constraint_status": pre_cleanup.uv_constraint_status,
            "decimate_protect_hard_edge_faces": str(feature_counts["hard_edge_faces"]),
            "decimate_protect_hole_boundary_faces": str(feature_counts["hole_boundary_faces"]),
            "decimate_protect_material_boundary_faces": str(feature_counts["material_boundary_faces"]),
            "decimate_protect_uv_seam_faces": str(feature_counts["uv_seam_faces"]),
            "decimate_protect_silhouette_faces": str(feature_counts["silhouette_faces"]),
            "decimate_protect_total_feature_faces": str(feature_counts["total_feature_faces"]),
            "decimate_protect_painted_area_faces": str(feature_counts["painted_area_faces"]),
            "decimate_protect_ambient_occlusion_faces": str(feature_counts["ambient_occlusion_faces"]),
            "decimate_protect_importance_faces": str(feature_counts["importance_faces"]),
            "decimate_iterative_threshold_triangles": str(options.iterative_threshold),
            "decimate_simplification_passes": str(counts.simplification_passes),
            "decimate_iterative_passes": str(counts.iterative_passes),
            "decimate_source_triangles": str(metrics.source_triangles),
            "decimate_output_triangles": str(metrics.output_triangles),
            "decimate_triangle_reduction": f"{metrics.triangle_reduction:.9g}",
            "decimate_max_vertex_error": f"{metrics.max_vertex_error:.9g}",
            "decimate_mean_vertex_error": f"{metrics.mean_vertex_error:.9g}",
            "decimate_error_metric": "symmetric_vertex_nearest_distance",
        }
        allocated_target = part_targets.get(part.id)
        if allocated_target is not None:
            allocation_part_count += 1
            allocated_target_total += allocated_target
            allocation_targets.append(allocated_target)
            if allocated_target >= metrics.source_triangles:
                allocation_preserved_parts += 1
            else:
                allocation_reduced_parts += 1
            metadata["decimate_allocated_target_triangles"] = str(allocated_target)
            metadata["decimate_allocation_target_reduction"] = (
                f"{((metrics.source_triangles - allocated_target) / metrics.source_triangles):.9g}"
            )
        if pre_cleanup.removed_uv_channels:
            metadata["decimate_pre_cleanup_removed_uv_channels"] = ",".join(
                str(channel) for channel in pre_cleanup.removed_uv_channels
            )
            pre_cleanup_removed_uv_channels.update(pre_cleanup.removed_uv_channels)
        if pre_cleanup.preserved_uv_channels:
            metadata["decimate_preserved_uv_channels"] = ",".join(
                str(channel) for channel in pre_cleanup.preserved_uv_channels
            )
        if pre_cleanup.removed_tangents:
            metadata["decimate_pre_cleanup_removed_tangents"] = "true"
            pre_cleanup_removed_tangent_parts += 1
        if pre_cleanup.uv_constraint_status == "preserved_for_simplification":
            uv_constrained_parts += 1
            uv_seam_constraint_vertices += pre_cleanup.uv_seam_vertices
            metadata["decimate_uv_seam_constraint_vertices"] = str(pre_cleanup.uv_seam_vertices)
        if feature_counts["total_feature_faces"]:
            protected_feature_parts += 1
        for key, value in feature_counts.items():
            feature_totals[key] += value
        part.metadata = metadata
        removed_uv_channels = sorted(set(source.uvs) - set(part.mesh.uvs))
        if removed_uv_channels:
            part.metadata["decimate_removed_uv_channels"] = ",".join(str(channel) for channel in removed_uv_channels)
    if source_total == 0:
        return
    memory_plan = _decimation_memory_plan(source_total, options.iterative_threshold)
    reduction = (source_total - output_total) / source_total
    requested_ratio = _requested_decimation_keep_ratio(source_meshes, options)
    if requested_ratio is not None:
        asset.metadata["decimate_requested_keep_ratio"] = f"{requested_ratio:.9g}"
    asset.metadata.update(target_strategy_metadata)
    asset.metadata["decimate_source_triangles"] = str(source_total)
    asset.metadata["decimate_output_triangles"] = str(output_total)
    asset.metadata["decimate_triangle_reduction"] = f"{reduction:.9g}"
    asset.metadata["decimate_max_vertex_error"] = f"{max_error:.9g}"
    asset.metadata["decimate_mean_vertex_error"] = f"{(weighted_error / measured_parts):.9g}"
    asset.metadata["decimate_error_metric"] = "symmetric_vertex_nearest_distance"
    asset.metadata["decimate_uv_importance"] = options.uv_importance
    asset.metadata["decimate_pre_cleanup_attributes"] = ",".join(options.cleanup_attributes) or "none"
    asset.metadata["decimate_pre_cleanup_removed_tangent_parts"] = str(pre_cleanup_removed_tangent_parts)
    asset.metadata["decimate_uv_constrained_parts"] = str(uv_constrained_parts)
    asset.metadata["decimate_uv_seam_constraint_vertices"] = str(uv_seam_constraint_vertices)
    asset.metadata["decimate_protected_feature_parts"] = str(protected_feature_parts)
    asset.metadata["decimate_protect_hard_edge_faces"] = str(feature_totals["hard_edge_faces"])
    asset.metadata["decimate_protect_hole_boundary_faces"] = str(feature_totals["hole_boundary_faces"])
    asset.metadata["decimate_protect_material_boundary_faces"] = str(feature_totals["material_boundary_faces"])
    asset.metadata["decimate_protect_uv_seam_faces"] = str(feature_totals["uv_seam_faces"])
    asset.metadata["decimate_protect_silhouette_faces"] = str(feature_totals["silhouette_faces"])
    asset.metadata["decimate_protect_total_feature_faces"] = str(feature_totals["total_feature_faces"])
    asset.metadata["decimate_protect_painted_area_faces"] = str(feature_totals["painted_area_faces"])
    asset.metadata["decimate_protect_ambient_occlusion_faces"] = str(feature_totals["ambient_occlusion_faces"])
    asset.metadata["decimate_protect_importance_faces"] = str(feature_totals["importance_faces"])
    if pre_cleanup_removed_uv_channels:
        asset.metadata["decimate_pre_cleanup_removed_uv_channels"] = ",".join(
            str(channel) for channel in sorted(pre_cleanup_removed_uv_channels)
        )
    asset.metadata["decimate_budget_allocation"] = (
        "global_selection" if options.budget_scope == "selection" else "per_part"
    )
    if allocation_targets:
        asset.metadata["decimate_allocation_targets"] = ",".join(
            f"{part_id}:{part_targets[part_id]}" for part_id in sorted(part_targets)
        )
        asset.metadata["decimate_allocated_target_triangles"] = str(allocated_target_total)
        asset.metadata["decimate_allocation_part_count"] = str(allocation_part_count)
        asset.metadata["decimate_allocation_preserved_parts"] = str(allocation_preserved_parts)
        asset.metadata["decimate_allocation_reduced_parts"] = str(allocation_reduced_parts)
        asset.metadata["decimate_allocation_min_target_triangles"] = str(min(allocation_targets))
        asset.metadata["decimate_allocation_max_target_triangles"] = str(max(allocation_targets))
    asset.metadata["decimate_estimated_memory_bytes"] = str(memory_plan.estimated_bytes)
    asset.metadata["decimate_estimated_memory_gb"] = f"{memory_plan.estimated_gb:.9g}"
    asset.metadata["decimate_memory_rule_gb_per_million_triangles"] = (
        f"{_DECIMATION_MEMORY_GB_PER_MILLION_TRIANGLES:.9g}"
    )
    asset.metadata["decimate_iterative_threshold_triangles"] = str(options.iterative_threshold)
    asset.metadata["decimate_iterative_recommended"] = str(memory_plan.iterative_recommended).lower()
    asset.metadata["decimate_simplification_passes"] = str(simplification_passes)
    asset.metadata["decimate_iterative_passes"] = str(iterative_passes)
    asset.metadata["decimate_max_part_simplification_passes"] = str(max_part_passes)
    if memory_plan.iterative_recommended:
        asset.report.add_warning(
            f"decimation estimates {memory_plan.estimated_gb:.3g} GB RAM for {source_total} source triangles; "
            f"iterative decimation is recommended at or above {options.iterative_threshold} triangles"
        )
    if uv_constrained_parts:
        asset.report.add_warning(
            f"decimation preserved UV seam/island data on {uv_constrained_parts} part(s); "
            "preserved texture coordinates can reduce simplification efficiency. "
            "Use uv_importance='ignore' or cleanup_attributes=('unused_uvs', 'tangents') when UVs are not needed"
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


def _decimation_feature_counts(mesh: Mesh, options: DecimateOptions) -> dict[str, int]:
    counts = mesh.feature_preservation_counts(
        preserve_hard_edges=True,
        hard_edge_angle=options.normal_tolerance,
        preserve_holes=options.protect_topology,
        preserve_material_boundaries=True,
        preserve_uv_seams=_preserve_uv_seams(options),
        preserve_silhouette=options.protect_topology,
    )
    importance_groups = _decimation_importance_face_groups(mesh, options)
    painted = importance_groups.get("painted_area_faces", np.empty(0, dtype=np.int64))
    ambient_occlusion = importance_groups.get("ambient_occlusion_faces", np.empty(0, dtype=np.int64))
    importance = (
        np.unique(np.concatenate(list(importance_groups.values())))
        if importance_groups
        else np.empty(0, dtype=np.int64)
    )
    counts["painted_area_faces"] = int(painted.size)
    counts["ambient_occlusion_faces"] = int(ambient_occlusion.size)
    counts["importance_faces"] = int(importance.size)
    return counts


@dataclass(frozen=True)
class _DecimationPassCount:
    simplification_passes: int = 0
    iterative_passes: int = 0


@dataclass(frozen=True)
class _DecimationPreCleanupPlan:
    requested: tuple[str, ...] = ()
    removed_uv_channels: tuple[int, ...] = ()
    preserved_uv_channels: tuple[int, ...] = ()
    removed_tangents: bool = False
    uv_constraint_status: str = "none"
    uv_seam_vertices: int = 0

    @property
    def removes_attributes(self) -> bool:
        return bool(self.removed_uv_channels or self.removed_tangents)


@dataclass(frozen=True)
class _DecimationMemoryPlan:
    estimated_bytes: int
    estimated_gb: float
    iterative_recommended: bool


def _decimation_memory_plan(source_triangles: int, iterative_threshold: int) -> _DecimationMemoryPlan:
    estimated_bytes = int(np.ceil(source_triangles * _DECIMATION_MEMORY_BYTES_PER_MILLION_TRIANGLES / 1_000_000))
    return _DecimationMemoryPlan(
        estimated_bytes=max(estimated_bytes, 0),
        estimated_gb=source_triangles * _DECIMATION_MEMORY_GB_PER_MILLION_TRIANGLES / 1_000_000,
        iterative_recommended=source_triangles >= iterative_threshold,
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
        missing = np.setdiff1d(
            np.arange(mesh.triangle_count, dtype=np.int64),
            face_indices,
            assume_unique=True,
        )
        face_indices = np.sort(np.concatenate([face_indices, missing[: target - face_indices.shape[0]]]))
    return slice_faces(mesh, face_indices).remove_unreferenced_vertices()
