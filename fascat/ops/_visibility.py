from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fascat.mesh import Mesh
from fascat.options import RemoveOccludedOptions

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

_AO_RAY_DIRECTION_BATCH_SIZE = 8
_AO_RAY_TRIANGLE_BATCH_SIZE = 65_536
_AO_BVH_LEAF_SIZE = 32


@dataclass(frozen=True)
class _RayBvhNode:
    bounds_min: FloatArray
    bounds_max: FloatArray
    indices: IntArray
    children: tuple[_RayBvhNode, ...] = ()


def _build_ray_bvh(bounds_min: FloatArray, bounds_max: FloatArray, indices: IntArray) -> _RayBvhNode:
    lower = bounds_min[indices].min(axis=0)
    upper = bounds_max[indices].max(axis=0)
    # Expand the broad phase slightly so rounding at edges does not reject hits
    # accepted by the unchanged triangle intersection predicate.
    padding = 16.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.maximum(np.abs(lower), np.abs(upper)))
    if len(indices) <= _AO_BVH_LEAF_SIZE:
        return _RayBvhNode(lower - padding, upper + padding, indices)
    centers = bounds_min[indices] * 0.5 + bounds_max[indices] * 0.5
    axis = int(np.argmax(np.ptp(centers, axis=0)))
    midpoint = len(indices) // 2
    order = np.argpartition(centers[:, axis], midpoint)
    children = tuple(
        _build_ray_bvh(bounds_min, bounds_max, indices[part]) for part in (order[:midpoint], order[midpoint:])
    )
    return _RayBvhNode(lower - padding, upper + padding, np.empty(0, dtype=np.int64), children)


def _rays_intersect_bounds(origin: FloatArray, directions: FloatArray, node: _RayBvhNode, max_t: float) -> BoolArray:
    parallel = directions == 0.0
    near = np.full(directions.shape, -np.inf)
    far = np.full(directions.shape, np.inf)
    np.divide(node.bounds_min - origin, directions, out=near, where=~parallel)
    np.divide(node.bounds_max - origin, directions, out=far, where=~parallel)
    outside = np.any(parallel & ((origin < node.bounds_min) | (origin > node.bounds_max)), axis=1)
    enter = np.maximum(np.minimum(near, far).max(axis=1), 0.0)
    leave = np.minimum(np.maximum(near, far).min(axis=1), max_t)
    return cast(BoolArray, ~outside & (enter <= leave))


class _RayMeshIndex:
    """A per-call BVH; leaves use the public ray predicate and original face IDs."""

    def __init__(self, triangles: FloatArray) -> None:
        self.triangles = triangles
        self.root = (
            _build_ray_bvh(triangles.min(axis=1), triangles.max(axis=1), np.arange(len(triangles), dtype=np.int64))
            if len(triangles)
            else None
        )

    def ray_hits(self, origin: FloatArray, directions: FloatArray, *, ignore_face: int, max_t: float) -> BoolArray:
        directions = np.asarray(directions, dtype=np.float64).reshape((-1, 3))
        hits = np.zeros(len(directions), dtype=np.bool_)
        if self.root is None or len(directions) == 0:
            return hits
        pending = [(self.root, np.arange(len(directions), dtype=np.int64))]
        while pending:
            node, active = pending.pop()
            active = active[~hits[active]]
            if len(active) == 0:
                continue
            active = active[_rays_intersect_bounds(origin, directions[active], node, max_t)]
            if len(active) == 0:
                continue
            if node.children:
                pending.extend((child, active) for child in node.children)
                continue
            ignored = np.flatnonzero(node.indices == ignore_face)
            hits[active] = ray_hits_mesh_batch(
                origin,
                directions[active],
                self.triangles[node.indices],
                ignore_face=int(ignored[0]) if len(ignored) else -1,
                max_t=max_t,
            )
        return hits


def face_ambient_occlusion(mesh: Mesh, strategy: str = "conservative") -> FloatArray:
    if mesh.triangle_count == 0:
        return np.empty(0, dtype=np.float64)
    triangles = mesh.points[mesh.faces]
    ray_index = _RayMeshIndex(triangles)
    centroids = triangles.mean(axis=1)
    normals = _face_normals(mesh)
    directions_arr = np.asarray(_ambient_occlusion_directions(strategy), dtype=np.float64).reshape((-1, 3))
    dots = directions_arr @ normals.T
    mins, maxs = mesh.bounds()
    ray_length = max(float(np.linalg.norm(maxs - mins)), 1.0) * 2.0
    epsilon = ray_length * 1e-6
    values = np.ones(mesh.triangle_count, dtype=np.float64)
    for face_index, (centroid, normal) in enumerate(zip(centroids, normals, strict=True)):
        origin = centroid + normal * epsilon
        valid_directions = directions_arr[dots[:, face_index] > 0.0]
        tested = valid_directions.shape[0]
        hits = (
            0
            if tested == 0
            else int(
                np.count_nonzero(ray_index.ray_hits(origin, valid_directions, ignore_face=face_index, max_t=ray_length))
            )
        )
        values[face_index] = 1.0 if tested == 0 else 1.0 - (hits / tested)
    return values


def _face_normals(mesh: Mesh) -> FloatArray:
    if mesh.triangle_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    triangles = mesh.points[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] = normals[valid] / lengths[valid, None]
    normals[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return normals


def _ambient_occlusion_directions(strategy: str) -> tuple[FloatArray, ...]:
    options = RemoveOccludedOptions(strategy=cast(Any, strategy), hemi_evaluation=False)
    return tuple(occlusion_directions(options))


def occlusion_directions(options: RemoveOccludedOptions) -> list[FloatArray]:
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


def ray_hits_mesh(
    origin: FloatArray,
    direction: FloatArray,
    triangles: FloatArray,
    *,
    ignore_face: int,
    max_t: float,
) -> bool:
    directions = np.asarray(direction, dtype=np.float64).reshape((1, 3))
    return bool(ray_hits_mesh_batch(origin, directions, triangles, ignore_face=ignore_face, max_t=max_t)[0])


def ray_hits_mesh_batch(
    origin: FloatArray,
    directions: FloatArray,
    triangles: FloatArray,
    *,
    ignore_face: int,
    max_t: float,
    direction_chunk_size: int = _AO_RAY_DIRECTION_BATCH_SIZE,
    triangle_chunk_size: int = _AO_RAY_TRIANGLE_BATCH_SIZE,
) -> BoolArray:
    directions_arr = np.asarray(directions, dtype=np.float64).reshape((-1, 3))
    direction_count = len(directions_arr)
    triangle_count = len(triangles)
    hits_by_direction = np.zeros(direction_count, dtype=np.bool_)
    if triangles.size == 0 or directions_arr.size == 0:
        return hits_by_direction

    direction_step = max(direction_chunk_size, 1)
    triangle_step = max(triangle_chunk_size, 1)
    for direction_start in range(0, direction_count, direction_step):
        direction_end = min(direction_start + direction_step, direction_count)
        direction_chunk = directions_arr[direction_start:direction_end]
        direction_hits = hits_by_direction[direction_start:direction_end]
        for triangle_start in range(0, triangle_count, triangle_step):
            active_mask = ~direction_hits
            if not bool(np.any(active_mask)):
                break
            active_directions = direction_chunk[active_mask]
            triangle_end = min(triangle_start + triangle_step, triangle_count)
            triangle_chunk = triangles[triangle_start:triangle_end]
            edge1 = triangle_chunk[:, 1] - triangle_chunk[:, 0]
            edge2 = triangle_chunk[:, 2] - triangle_chunk[:, 0]
            h = np.cross(active_directions[:, None, :], edge2[None, :, :])
            determinant = np.einsum("tj,dtj->dt", edge1, h)
            valid = np.abs(determinant) > 1e-12
            inv_det = np.zeros_like(determinant)
            np.divide(1.0, determinant, out=inv_det, where=valid)
            s = origin - triangle_chunk[:, 0]
            u = inv_det * np.einsum("tj,dtj->dt", s, h)
            q = np.cross(s, edge1)
            v = inv_det * np.einsum("dj,tj->dt", active_directions, q)
            t = inv_det * np.einsum("tj,tj->t", edge2, q)[None, :]
            hits = valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-8) & (t < max_t)
            ignore_local = ignore_face - triangle_start
            if 0 <= ignore_local < hits.shape[1]:
                hits[:, ignore_local] = False
            direction_hits[active_mask] = direction_hits[active_mask] | np.any(hits, axis=1)
    return hits_by_direction
