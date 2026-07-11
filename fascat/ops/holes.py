from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.ops._mesh_utils import edge_faces as mesh_edge_faces
from fascat.ops._mesh_utils import face_major_edges, grouped_edge_faces
from fascat.options import RemoveHolesOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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


@dataclass(frozen=True)
class _HoleFillStats:
    count: int
    diameters: tuple[float, ...]
    kinds: tuple[str, ...]


@dataclass(frozen=True)
class _HoleLoop:
    loop: list[int]
    diameter: float
    centroid: FloatArray
    normal: FloatArray
    adjacent_faces: tuple[int, ...]
    kind: str = "surface"


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
    edge_faces = mesh_edge_faces(mesh)
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
    directed = face_major_edges(mesh)
    edge_faces = grouped_edge_faces(directed, mesh.triangle_count)

    adjacency: dict[int, list[int]] = {}
    for edge in directed:
        start = int(edge[0])
        end = int(edge[1])
        if len(edge_faces[_edge_key(start, end)]) != 1:
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
