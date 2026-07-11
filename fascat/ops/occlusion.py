from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.ops._ids import unique_id
from fascat.ops._mesh_utils import edge_faces as mesh_edge_faces
from fascat.ops._mesh_utils import slice_faces
from fascat.ops._visibility import occlusion_directions
from fascat.options import RemoveOccludedOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_OCCLUSION_BVH_LEAF_OCCURRENCES = 8
_OCCLUSION_BVH_LEAF_TRIANGLES = 64
_OCCLUSION_BVH_MAX_DEPTH = 32


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
    operation_occluders = _candidate_occluder_pool(result, occurrences, options)
    directions = occlusion_directions(options)
    ray_distance = _occlusion_ray_distance(occurrences)
    removed_node_ids: set[str] = set()
    trims: list[_OcclusionTrim] = []
    measurements: list[_OcclusionMeasurement] = []
    for candidate in selected_occurrences:
        if _preserve_candidate_cavity(result, candidate, options):
            continue
        occluders = _candidate_occluders(candidate, operation_occluders)
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


@dataclass(frozen=True)
class _TriangleBvhNode:
    bounds_min: FloatArray
    bounds_max: FloatArray
    indices: IntArray | None = None
    left: _TriangleBvhNode | None = None
    right: _TriangleBvhNode | None = None


@dataclass(frozen=True)
class _OccurrenceBvhNode:
    bounds_min: FloatArray
    bounds_max: FloatArray
    indices: IntArray | None = None
    left: _OccurrenceBvhNode | None = None
    right: _OccurrenceBvhNode | None = None


@dataclass(frozen=True)
class _WorldOccurrence:
    node: Node
    part_id: str
    world_points: FloatArray
    faces: IntArray
    triangles: FloatArray
    triangle_edges1: FloatArray
    triangle_edges2: FloatArray
    triangle_bounds_min: FloatArray
    triangle_bounds_max: FloatArray
    triangle_bvh: _TriangleBvhNode | None
    bounds_min: FloatArray
    bounds_max: FloatArray
    volume: float


@dataclass(frozen=True)
class _OccluderSet:
    occurrences: tuple[_WorldOccurrence, ...]
    bvh: _OccurrenceBvhNode | None
    excluded_node_id: str | None = None

    def __bool__(self) -> bool:
        if self.excluded_node_id is None:
            return bool(self.occurrences)
        return any(occurrence.node.id != self.excluded_node_id for occurrence in self.occurrences)


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
            new_part_id = unique_id(asset.parts, f"{part_id}_{node.id}")
            part = asset.parts[part_id].copy(keep_source=True)
            part.id = new_part_id
            part.metadata = {**part.metadata, "source_part_id": part_id}
            asset.parts[new_part_id] = part
            node.part_id = new_part_id


def _candidate_occluder_pool(
    asset: Asset,
    occurrences: list[_WorldOccurrence],
    options: RemoveOccludedOptions,
) -> _OccluderSet:
    candidates = tuple(
        occluder
        for occluder in occurrences
        if options.consider_transparency_opaque or not _part_is_transparent(asset, occluder.part_id)
    )
    if not candidates:
        return _OccluderSet(occurrences=(), bvh=None)
    bounds_min = np.vstack([occluder.bounds_min for occluder in candidates])
    bounds_max = np.vstack([occluder.bounds_max for occluder in candidates])
    return _OccluderSet(
        occurrences=candidates,
        bvh=_build_occurrence_bvh(bounds_min, bounds_max),
    )


def _candidate_occluders(candidate: _WorldOccurrence, occluders: _OccluderSet) -> _OccluderSet:
    if not occluders.occurrences:
        return occluders
    return _OccluderSet(
        occurrences=occluders.occurrences,
        bvh=occluders.bvh,
        excluded_node_id=candidate.node.id,
    )


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
        occlusion_directions(RemoveOccludedOptions(strategy="advanced", hemi_evaluation=options.hemi_evaluation))
    )


def _occurrence_visibility_measurement(
    candidate: _WorldOccurrence,
    occluders: _OccluderSet,
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
    occluders: _OccluderSet,
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
    occluders: _OccluderSet,
    directions: list[FloatArray],
    ray_distance: float,
) -> bool:
    for direction in directions:
        origin = sample + direction * ray_distance
        if not _segment_blocked(origin, sample, occluders):
            return True
    return False


def _segment_blocked(start: FloatArray, end: FloatArray, occluders: _OccluderSet) -> bool:
    if occluders.bvh is not None:
        return _segment_intersects_occluder_bvh(
            start,
            end,
            occluders.bvh,
            occluders.occurrences,
            occluders.excluded_node_id,
        )
    for occluder in occluders.occurrences:
        if occluder.node.id == occluders.excluded_node_id:
            continue
        if not _segment_intersects_bounds(start, end, occluder.bounds_min, occluder.bounds_max):
            continue
        if _segment_intersects_occurrence(start, end, occluder):
            return True
    return False


def _segment_intersects_occluder_bvh(
    start: FloatArray,
    end: FloatArray,
    node: _OccurrenceBvhNode,
    occurrences: tuple[_WorldOccurrence, ...],
    excluded_node_id: str | None = None,
) -> bool:
    if not _segment_intersects_bounds(start, end, node.bounds_min, node.bounds_max):
        return False
    if node.indices is not None:
        for index in node.indices:
            occurrence = occurrences[int(index)]
            if occurrence.node.id == excluded_node_id:
                continue
            if not _segment_intersects_bounds(start, end, occurrence.bounds_min, occurrence.bounds_max):
                continue
            if _segment_intersects_occurrence(start, end, occurrence):
                return True
        return False
    if node.left is not None and _segment_intersects_occluder_bvh(
        start,
        end,
        node.left,
        occurrences,
        excluded_node_id,
    ):
        return True
    return node.right is not None and _segment_intersects_occluder_bvh(
        start,
        end,
        node.right,
        occurrences,
        excluded_node_id,
    )


def _segment_intersects_occurrence(start: FloatArray, end: FloatArray, occurrence: _WorldOccurrence) -> bool:
    if occurrence.triangle_bvh is not None:
        return _segment_intersects_bvh(start, end, occurrence.triangle_bvh, occurrence)
    return _segment_intersects_triangles(
        start,
        end,
        occurrence.triangles,
        occurrence.triangle_edges1,
        occurrence.triangle_edges2,
        occurrence.triangle_bounds_min,
        occurrence.triangle_bounds_max,
    )


def _segment_intersects_bvh(
    start: FloatArray,
    end: FloatArray,
    node: _TriangleBvhNode,
    occurrence: _WorldOccurrence,
) -> bool:
    if not _segment_intersects_bounds(start, end, node.bounds_min, node.bounds_max):
        return False
    if node.indices is not None:
        indices = node.indices
        return _segment_intersects_triangles(
            start,
            end,
            occurrence.triangles[indices],
            occurrence.triangle_edges1[indices],
            occurrence.triangle_edges2[indices],
            occurrence.triangle_bounds_min[indices],
            occurrence.triangle_bounds_max[indices],
        )
    if node.left is not None and _segment_intersects_bvh(start, end, node.left, occurrence):
        return True
    return node.right is not None and _segment_intersects_bvh(start, end, node.right, occurrence)


def _segment_intersects_mesh(start: FloatArray, end: FloatArray, points: FloatArray, faces: IntArray) -> bool:
    if faces.size == 0:
        return False
    triangles = points[faces]
    return _segment_intersects_triangles(
        start,
        end,
        triangles,
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
        triangles.min(axis=1),
        triangles.max(axis=1),
    )


def _build_triangle_bvh(
    triangle_bounds_min: FloatArray,
    triangle_bounds_max: FloatArray,
    indices: IntArray | None = None,
    *,
    depth: int = 0,
) -> _TriangleBvhNode | None:
    if triangle_bounds_min.shape[0] == 0:
        return None
    if indices is None:
        indices = np.arange(triangle_bounds_min.shape[0], dtype=np.int64)
    if indices.size == 0:
        return None

    bounds_min = cast(FloatArray, triangle_bounds_min[indices].min(axis=0))
    bounds_max = cast(FloatArray, triangle_bounds_max[indices].max(axis=0))
    if indices.size <= _OCCLUSION_BVH_LEAF_TRIANGLES or depth >= _OCCLUSION_BVH_MAX_DEPTH:
        return _TriangleBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, indices=indices.copy())

    centers = (triangle_bounds_min[indices] + triangle_bounds_max[indices]) * 0.5
    spans = np.ptp(centers, axis=0)
    axis = int(np.argmax(spans))
    if float(spans[axis]) <= 0.0:
        return _TriangleBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, indices=indices.copy())

    order = np.argsort(centers[:, axis], kind="mergesort")
    midpoint = indices.size // 2
    left = _build_triangle_bvh(
        triangle_bounds_min,
        triangle_bounds_max,
        indices[order[:midpoint]],
        depth=depth + 1,
    )
    right = _build_triangle_bvh(
        triangle_bounds_min,
        triangle_bounds_max,
        indices[order[midpoint:]],
        depth=depth + 1,
    )
    if left is None or right is None:
        return _TriangleBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, indices=indices.copy())
    return _TriangleBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, left=left, right=right)


def _build_occurrence_bvh(
    bounds_min_all: FloatArray,
    bounds_max_all: FloatArray,
    indices: IntArray | None = None,
    *,
    depth: int = 0,
) -> _OccurrenceBvhNode | None:
    if bounds_min_all.shape[0] == 0:
        return None
    if indices is None:
        indices = np.arange(bounds_min_all.shape[0], dtype=np.int64)
    if indices.size == 0:
        return None

    bounds_min = cast(FloatArray, bounds_min_all[indices].min(axis=0))
    bounds_max = cast(FloatArray, bounds_max_all[indices].max(axis=0))
    if indices.size <= _OCCLUSION_BVH_LEAF_OCCURRENCES or depth >= _OCCLUSION_BVH_MAX_DEPTH:
        return _OccurrenceBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, indices=indices.copy())

    centers = (bounds_min_all[indices] + bounds_max_all[indices]) * 0.5
    spans = np.ptp(centers, axis=0)
    axis = int(np.argmax(spans))
    if float(spans[axis]) <= 0.0:
        return _OccurrenceBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, indices=indices.copy())

    order = np.argsort(centers[:, axis], kind="mergesort")
    midpoint = indices.size // 2
    left = _build_occurrence_bvh(
        bounds_min_all,
        bounds_max_all,
        indices[order[:midpoint]],
        depth=depth + 1,
    )
    right = _build_occurrence_bvh(
        bounds_min_all,
        bounds_max_all,
        indices[order[midpoint:]],
        depth=depth + 1,
    )
    if left is None or right is None:
        return _OccurrenceBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, indices=indices.copy())
    return _OccurrenceBvhNode(bounds_min=bounds_min, bounds_max=bounds_max, left=left, right=right)


def _segment_intersects_triangles(
    start: FloatArray,
    end: FloatArray,
    triangles: FloatArray,
    edge1: FloatArray,
    edge2: FloatArray,
    triangle_bounds_min: FloatArray,
    triangle_bounds_max: FloatArray,
) -> bool:
    if triangles.shape[0] == 0:
        return False
    epsilon = 1e-12
    direction = end - start
    segment_min = np.minimum(start, end) - epsilon
    segment_max = np.maximum(start, end) + epsilon
    candidates = np.all((triangle_bounds_max >= segment_min) & (triangle_bounds_min <= segment_max), axis=1)
    if not np.any(candidates):
        return False
    triangles = triangles[candidates]
    edge1 = edge1[candidates]
    edge2 = edge2[candidates]
    pvec = np.cross(np.broadcast_to(direction, edge2.shape), edge2)
    determinant = np.einsum("ij,ij->i", edge1, pvec)
    active = np.abs(determinant) > epsilon
    if not np.any(active):
        return False

    inverse = np.zeros_like(determinant)
    inverse[active] = 1.0 / determinant[active]
    tvec = start - triangles[:, 0]
    u = np.einsum("ij,ij->i", tvec, pvec) * inverse
    active &= (u >= -epsilon) & (u <= 1.0 + epsilon)
    if not np.any(active):
        return False

    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inverse
    active &= (v >= -epsilon) & (u + v <= 1.0 + epsilon)
    if not np.any(active):
        return False

    t = np.einsum("ij,ij->i", edge2, qvec) * inverse
    active &= (t >= -epsilon) & (t <= 1.0 + epsilon)
    return bool(np.any(active & (t > 1e-8) & (t < 1.0 - 1e-8)))


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
    edge_faces = mesh_edge_faces(mesh)
    neighbors: list[set[int]] = [set() for _ in range(mesh.triangle_count)]
    for face_indices in edge_faces.values():
        if len(face_indices) < 2:
            continue
        for face_index in face_indices:
            neighbors[face_index].update(index for index in face_indices if index != face_index)
    kept_faces = {int(face_index) for face_index in np.flatnonzero(keep)}
    frontier = set(kept_faces)
    for _ in range(rings):
        next_frontier: set[int] = set()
        for face_index in frontier:
            next_frontier.update(neighbors[face_index])
        next_frontier.difference_update(kept_faces)
        if not next_frontier:
            break
        keep[np.asarray(sorted(next_frontier), dtype=np.int64)] = True
        kept_faces.update(next_frontier)
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
        mesh = slice_faces(part.mesh, trim.keep_faces).remove_unreferenced_vertices()
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
    indices = mesh.material_indices.astype(np.int64, copy=False)
    used = np.unique(indices)
    if used.size == 0 or bool(np.any(used < 0)) or bool(np.any(used >= len(material_ids))):
        return
    lookup = np.empty(len(material_ids), dtype=np.int64)
    lookup[used] = np.arange(len(used), dtype=np.int64)
    mesh.material_indices = lookup[indices]
    part.material_ids = [material_ids[int(index)] for index in used]


def _world_occurrences(asset: Asset) -> list[_WorldOccurrence]:
    occurrences: list[_WorldOccurrence] = []
    for node, current in asset.root.walk_world(np.eye(4, dtype=np.float64)):
        if node.part_id is not None and node.part_id in asset.parts:
            part = asset.parts[node.part_id]
            if part.mesh is not None:
                world_points = _transform_points(part.mesh.points, current)
                triangles = world_points[part.mesh.faces]
                if world_points.shape[0] == 0:
                    mins, maxs = part.mesh.bounds()
                    world_min, world_max = _transform_bounds(mins, maxs, current)
                else:
                    world_min = cast(FloatArray, world_points.min(axis=0))
                    world_max = cast(FloatArray, world_points.max(axis=0))
                triangle_edges1 = triangles[:, 1] - triangles[:, 0]
                triangle_edges2 = triangles[:, 2] - triangles[:, 0]
                triangle_bounds_min = (
                    triangles.min(axis=1) if triangles.shape[0] else np.empty((0, 3), dtype=np.float64)
                )
                triangle_bounds_max = (
                    triangles.max(axis=1) if triangles.shape[0] else np.empty((0, 3), dtype=np.float64)
                )
                volume = float(np.prod(np.maximum(world_max - world_min, 0.0)))
                occurrences.append(
                    _WorldOccurrence(
                        node=node,
                        part_id=node.part_id,
                        world_points=world_points,
                        faces=part.mesh.faces.copy(),
                        triangles=triangles,
                        triangle_edges1=triangle_edges1,
                        triangle_edges2=triangle_edges2,
                        triangle_bounds_min=triangle_bounds_min,
                        triangle_bounds_max=triangle_bounds_max,
                        triangle_bvh=_build_triangle_bvh(triangle_bounds_min, triangle_bounds_max),
                        bounds_min=world_min,
                        bounds_max=world_max,
                        volume=volume,
                    )
                )
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
