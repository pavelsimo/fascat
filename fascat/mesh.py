from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fascat.options import RepairOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class MeshValidationError(ValueError):
    """Raised when mesh arrays are not usable by the pipeline."""


def _vector_angles(left: FloatArray, right: FloatArray) -> FloatArray:
    left_lengths = np.linalg.norm(left, axis=1)
    right_lengths = np.linalg.norm(right, axis=1)
    denom = left_lengths * right_lengths
    angles = np.zeros(left.shape[0], dtype=np.float64)
    valid = denom > 0.0
    if np.any(valid):
        cosines = np.einsum("ij,ij->i", left[valid], right[valid]) / denom[valid]
        angles[valid] = np.arccos(np.clip(cosines, -1.0, 1.0))
    return angles


@dataclass
class Mesh:
    points: FloatArray
    faces: IntArray
    normals: FloatArray | None = None
    uvs: dict[int, FloatArray] = field(default_factory=dict)
    material_indices: IntArray | None = None
    face_groups: dict[str, IntArray] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int64)
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
        self.uvs = {channel: np.asarray(values, dtype=np.float64) for channel, values in self.uvs.items()}
        if self.material_indices is not None:
            self.material_indices = np.asarray(self.material_indices, dtype=np.int64)
        self.face_groups = {name: np.asarray(values, dtype=np.int64) for name, values in self.face_groups.items()}

    @property
    def vertex_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.faces.shape[0])

    def copy(self) -> Mesh:
        return Mesh(
            points=self.points.copy(),
            faces=self.faces.copy(),
            normals=None if self.normals is None else self.normals.copy(),
            uvs={channel: values.copy() for channel, values in self.uvs.items()},
            material_indices=None if self.material_indices is None else self.material_indices.copy(),
            face_groups={name: values.copy() for name, values in self.face_groups.items()},
            metadata=dict(self.metadata),
        )

    def validate(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise MeshValidationError("points must have shape (N, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise MeshValidationError("faces must have shape (M, 3)")
        if not np.isfinite(self.points).all():
            raise MeshValidationError("points must not contain NaN or Inf values")
        if self.faces.size and int(self.faces.min()) < 0:
            raise MeshValidationError("faces must not contain negative vertex indices")
        if self.faces.size and int(self.faces.max()) >= self.vertex_count:
            raise MeshValidationError("faces must not contain out-of-range vertex indices")
        if self.normals is not None:
            if self.normals.shape != self.points.shape:
                raise MeshValidationError("normals must match points shape")
            if not np.isfinite(self.normals).all():
                raise MeshValidationError("normals must not contain NaN or Inf values")
        for channel, values in self.uvs.items():
            if values.ndim != 2 or values.shape[1] != 2:
                raise MeshValidationError(f"uv channel {channel} must have shape (N, 2)")
            if values.shape[0] != self.vertex_count:
                raise MeshValidationError(f"uv channel {channel} must match vertex count")
            if not np.isfinite(values).all():
                raise MeshValidationError(f"uv channel {channel} must not contain NaN or Inf values")
        if self.material_indices is not None and self.material_indices.shape != (self.triangle_count,):
            raise MeshValidationError("material_indices must match triangle count")

    def stats(self) -> dict[str, int]:
        return {"vertices": self.vertex_count, "triangles": self.triangle_count}

    def fingerprint(self) -> str:
        points = np.ascontiguousarray(np.round(self.points, 9))
        faces = np.ascontiguousarray(self.faces)
        digest = hashlib.sha1()
        digest.update(points.tobytes())
        digest.update(faces.tobytes())
        return digest.hexdigest()

    def bounds(self) -> tuple[FloatArray, FloatArray]:
        if self.vertex_count == 0:
            zero = np.zeros(3, dtype=np.float64)
            return zero.copy(), zero.copy()
        return self.points.min(axis=0), self.points.max(axis=0)

    def repair(self, options: RepairOptions | None = None) -> Mesh:
        opts = options or RepairOptions()
        mesh = self.copy()
        mesh = mesh._drop_non_finite()
        mesh = mesh.remove_unreferenced_vertices()
        if opts.merge_vertices and opts.tolerance > 0.0:
            mesh = mesh.merge_close_vertices(opts.tolerance)
        mesh = mesh.remove_duplicate_faces()
        if opts.delete_degenerate:
            mesh = mesh.remove_degenerate_faces(opts.area_epsilon)
        if opts.fill_small_holes:
            mesh = mesh.fill_holes()
        if opts.fix_winding:
            mesh = mesh.fix_winding()
        mesh = mesh.compute_normals()
        mesh.validate()
        return mesh

    def remove_unreferenced_vertices(self) -> Mesh:
        if self.vertex_count == 0 or self.triangle_count == 0:
            return Mesh(
                points=np.empty((0, 3), dtype=np.float64),
                faces=np.empty((0, 3), dtype=np.int64),
                metadata=dict(self.metadata),
            )
        used = np.unique(self.faces.reshape(-1))
        remap = np.full(self.vertex_count, -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0], dtype=np.int64)
        mesh = self.copy()
        mesh.points = self.points[used].copy()
        mesh.faces = remap[self.faces]
        if self.normals is not None:
            mesh.normals = self.normals[used].copy()
        mesh.uvs = {channel: values[used].copy() for channel, values in self.uvs.items()}
        return mesh

    def merge_close_vertices(self, tolerance: float) -> Mesh:
        if tolerance <= 0.0 or self.vertex_count == 0:
            return self.copy()
        quantized = np.round(self.points / tolerance).astype(np.int64)
        _, first_indices, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
        order = np.argsort(first_indices)
        old_to_new_cluster = np.empty_like(order)
        old_to_new_cluster[order] = np.arange(order.shape[0], dtype=np.int64)
        new_points = self.points[np.sort(first_indices)]
        new_faces = old_to_new_cluster[inverse][self.faces]
        mesh = self.copy()
        mesh.points = new_points
        mesh.faces = new_faces
        mesh.normals = None
        mesh.uvs = {}
        return mesh.remove_degenerate_faces()

    def remove_duplicate_faces(self) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        keys = np.sort(self.faces, axis=1)
        _, keep = np.unique(keys, axis=0, return_index=True)
        keep.sort()
        return self._filter_faces(keep)

    def remove_degenerate_faces(self, area_epsilon: float = 1e-12) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        p0 = self.points[self.faces[:, 0]]
        p1 = self.points[self.faces[:, 1]]
        p2 = self.points[self.faces[:, 2]]
        areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) * 0.5
        keep = np.flatnonzero(areas > area_epsilon)
        return self._filter_faces(keep).remove_unreferenced_vertices()

    def compute_normals(self, *, angle_weighted: bool = True) -> Mesh:
        normals = np.zeros_like(self.points, dtype=np.float64)
        if self.triangle_count > 0 and self.vertex_count > 0:
            p0 = self.points[self.faces[:, 0]]
            p1 = self.points[self.faces[:, 1]]
            p2 = self.points[self.faces[:, 2]]
            face_normals = np.cross(p1 - p0, p2 - p0)
            if angle_weighted:
                face_lengths = np.linalg.norm(face_normals, axis=1)
                valid = face_lengths > 0.0
                unit_normals = np.zeros_like(face_normals)
                unit_normals[valid] = face_normals[valid] / face_lengths[valid, None]
                for corner in range(3):
                    origin = self.points[self.faces[:, corner]]
                    left = self.points[self.faces[:, (corner + 1) % 3]] - origin
                    right = self.points[self.faces[:, (corner + 2) % 3]] - origin
                    angles = _vector_angles(left, right)
                    weighted = unit_normals * angles[:, None]
                    np.add.at(normals, self.faces[:, corner], weighted)
            else:
                for corner in range(3):
                    np.add.at(normals, self.faces[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        nonzero = lengths > 0.0
        normals[nonzero] = normals[nonzero] / lengths[nonzero, None]
        normals[~nonzero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        mesh = self.copy()
        mesh.normals = normals
        return mesh

    def subdivide_long_edges(self, max_edge_length: float) -> Mesh:
        if max_edge_length <= 0.0:
            raise ValueError("max_edge_length must be greater than 0")
        mesh = self.copy()
        if mesh.triangle_count == 0:
            return mesh

        points = mesh.points.tolist()
        faces: list[list[int]] = mesh.faces.astype(int).tolist()
        material_indices = None if mesh.material_indices is None else mesh.material_indices.astype(int).tolist()

        changed = True
        iterations = 0
        while changed and iterations < 32:
            changed = False
            iterations += 1
            next_faces: list[list[int]] = []
            next_materials: list[int] = []
            for face_index, face in enumerate(faces):
                triangle = np.asarray([points[index] for index in face], dtype=np.float64)
                edge_pairs = ((0, 1), (1, 2), (2, 0))
                lengths = [float(np.linalg.norm(triangle[b] - triangle[a])) for a, b in edge_pairs]
                longest = int(np.argmax(lengths))
                if lengths[longest] <= max_edge_length:
                    next_faces.append(face)
                    if material_indices is not None:
                        next_materials.append(material_indices[face_index])
                    continue
                changed = True
                a_corner, b_corner = edge_pairs[longest]
                c_corner = next(corner for corner in range(3) if corner not in {a_corner, b_corner})
                a = face[a_corner]
                b = face[b_corner]
                c = face[c_corner]
                midpoint = ((np.asarray(points[a]) + np.asarray(points[b])) * 0.5).tolist()
                midpoint_index = len(points)
                points.append(midpoint)
                next_faces.append([a, midpoint_index, c])
                next_faces.append([midpoint_index, b, c])
                if material_indices is not None:
                    next_materials.extend([material_indices[face_index], material_indices[face_index]])
            faces = next_faces
            material_indices = next_materials if material_indices is not None else None

        result = Mesh(
            points=np.asarray(points, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            material_indices=None if material_indices is None else np.asarray(material_indices, dtype=np.int64),
            metadata={**mesh.metadata, "max_edge_length": str(max_edge_length)},
        )
        if 0 in mesh.uvs or 1 in mesh.uvs:
            for channel in mesh.uvs:
                result = result.box_uv(channel)
        return result.compute_normals()

    def box_uv(self, channel: int = 0) -> Mesh:
        mesh = self.copy()
        if self.vertex_count == 0:
            mesh.uvs[channel] = np.empty((0, 2), dtype=np.float64)
            return mesh
        mins, maxs = self.bounds()
        size = maxs - mins
        axes = np.argsort(size)[-2:]
        denom = size[axes]
        denom[denom == 0.0] = 1.0
        uv = (self.points[:, axes] - mins[axes]) / denom
        mesh.uvs[channel] = uv.astype(np.float64)
        return mesh

    def unwrap_uv(self, channel: int = 0) -> Mesh:
        try:
            import xatlas
        except ImportError as exc:
            raise RuntimeError("UV unwrap requires the optional xatlas dependency") from exc

        if self.triangle_count == 0:
            mesh = self.copy()
            mesh.uvs[channel] = np.empty((0, 2), dtype=np.float64)
            return mesh

        vertex_mapping, faces, uvs = xatlas.parametrize(
            self.points.astype(np.float32),
            self.faces.astype(np.uint32),
            None if self.normals is None else self.normals.astype(np.float32),
        )
        mapping = np.asarray(vertex_mapping, dtype=np.int64)
        mesh = Mesh(
            points=self.points[mapping].copy(),
            faces=np.asarray(faces, dtype=np.int64),
            normals=None if self.normals is None else self.normals[mapping].copy(),
            uvs={**{key: values[mapping].copy() for key, values in self.uvs.items() if key != channel}},
            material_indices=None if self.material_indices is None else self.material_indices.copy(),
            face_groups={name: values.copy() for name, values in self.face_groups.items()},
            metadata={**self.metadata, f"uv{channel}": "xatlas"},
        )
        mesh.uvs[channel] = np.asarray(uvs, dtype=np.float64)
        mesh.validate()
        return mesh

    def simplify(self, *, target_triangles: int | None = None, ratio: float | None = None) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        if target_triangles is None:
            if ratio is None:
                return self.copy()
            target_triangles = max(1, int(round(self.triangle_count * ratio)))
        target_triangles = max(1, min(int(target_triangles), self.triangle_count))
        if target_triangles >= self.triangle_count:
            return self.copy()

        try:
            import meshoptimizer

            indices = np.ascontiguousarray(self.faces.reshape(-1), dtype=np.uint32)
            destination = np.empty_like(indices)
            index_count = meshoptimizer.simplify(
                destination,
                indices,
                np.ascontiguousarray(self.points.astype(np.float32)),
                vertex_count=self.vertex_count,
                target_index_count=target_triangles * 3,
            )
            simplified_indices = np.asarray(destination[:index_count], dtype=np.int64)
            if simplified_indices.size < 3:
                return self.copy()
            face_count = simplified_indices.size // 3
            mesh = self.copy()
            mesh.faces = simplified_indices[: face_count * 3].reshape((-1, 3))
            mesh.material_indices = self._assign_materials_by_nearest_centroid(mesh.points, mesh.faces)
            mesh = mesh.remove_unreferenced_vertices().compute_normals()
            mesh.validate()
            return mesh
        except Exception:
            try:
                from fast_simplification import simplify

                points, faces = simplify(
                    self.points.astype(np.float64), self.faces.astype(np.int64), target_count=target_triangles
                )
                points_array = np.asarray(points, dtype=np.float64)
                faces_array = np.asarray(faces, dtype=np.int64)
                mesh = Mesh(
                    points=points_array,
                    faces=faces_array,
                    material_indices=self._assign_materials_by_nearest_centroid(points_array, faces_array),
                )
                if self.uvs:
                    for channel in self.uvs:
                        mesh = mesh.box_uv(channel)
                mesh = mesh.repair(RepairOptions())
                mesh.validate()
                return mesh
            except Exception:
                stride = max(1, int(np.ceil(self.triangle_count / target_triangles)))
                keep = np.arange(0, self.triangle_count, stride, dtype=np.int64)[:target_triangles]
                mesh = self._filter_faces(keep).remove_unreferenced_vertices().compute_normals()
                mesh.validate()
                return mesh

    def optimize_buffers(self) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        try:
            import meshoptimizer

            indices = np.ascontiguousarray(self.faces.reshape(-1), dtype=np.uint32)
            cache_optimized = np.empty_like(indices)
            meshoptimizer.optimize_vertex_cache(cache_optimized, indices, vertex_count=self.vertex_count)
            reordered_face_indices: np.ndarray | None = None
            if self.material_indices is not None or self.face_groups:
                old_face_lookup = {
                    tuple(sorted(face)): index for index, face in enumerate(self.faces.astype(int).tolist())
                }
                reordered_face_indices = np.asarray(
                    [
                        old_face_lookup.get(tuple(sorted(face)), index)
                        for index, face in enumerate(cache_optimized.reshape((-1, 3)).astype(int).tolist())
                    ],
                    dtype=np.int64,
                )

            vertex_attributes = [self.points.astype(np.float32)]
            if self.normals is not None:
                vertex_attributes.append(self.normals.astype(np.float32))
            for channel in sorted(self.uvs):
                vertex_attributes.append(self.uvs[channel].astype(np.float32))
            vertex_stream = np.ascontiguousarray(np.column_stack(vertex_attributes))
            remap = np.empty(self.vertex_count, dtype=np.uint32)
            unique_vertices = meshoptimizer.generate_vertex_remap(
                remap,
                cache_optimized,
                vertices=vertex_stream,
            )
            remapped_indices = np.empty_like(cache_optimized)
            meshoptimizer.remap_index_buffer(remapped_indices, cache_optimized, remap=remap)
            old_for_new = np.empty(int(unique_vertices), dtype=np.int64)
            for old_index, new_index in enumerate(remap.astype(np.int64)):
                if new_index < unique_vertices:
                    old_for_new[new_index] = old_index

            mesh = self.copy()
            mesh.points = self.points[old_for_new].copy()
            mesh.faces = np.asarray(remapped_indices, dtype=np.int64).reshape((-1, 3))
            if self.normals is not None:
                mesh.normals = self.normals[old_for_new].copy()
            mesh.uvs = {channel: values[old_for_new].copy() for channel, values in self.uvs.items()}
            if self.material_indices is not None and reordered_face_indices is not None:
                mesh.material_indices = self.material_indices[reordered_face_indices].copy()
            if self.face_groups and reordered_face_indices is not None:
                inverse_face_order = np.empty_like(reordered_face_indices)
                inverse_face_order[reordered_face_indices] = np.arange(reordered_face_indices.shape[0])
                mesh.face_groups = {
                    name: inverse_face_order[values]
                    for name, values in self.face_groups.items()
                    if np.isin(values, reordered_face_indices).all()
                }
            return mesh
        except Exception:
            return self.copy()

    def _assign_materials_by_nearest_centroid(self, points: FloatArray, faces: IntArray) -> IntArray | None:
        if self.material_indices is None or self.triangle_count == 0 or faces.size == 0:
            return None
        source_centroids = self.points[self.faces].mean(axis=1)
        target_centroids = points[faces].mean(axis=1)
        nearest = np.empty(target_centroids.shape[0], dtype=np.int64)
        chunk_size = 4096
        for start in range(0, target_centroids.shape[0], chunk_size):
            end = min(start + chunk_size, target_centroids.shape[0])
            delta = target_centroids[start:end, None, :] - source_centroids[None, :, :]
            nearest[start:end] = np.argmin(np.einsum("ijk,ijk->ij", delta, delta), axis=1)
        return cast(IntArray, np.asarray(self.material_indices[nearest], dtype=np.int64).copy())

    def fill_holes(self) -> Mesh:
        loops = self._boundary_loops()
        if not loops or max(len(loop) for loop in loops) > 8:
            return self.copy()
        if np.linalg.matrix_rank(self.points - self.points.mean(axis=0), tol=1e-9) < 3:
            return self.copy()

        fill_faces: list[list[int]] = []
        for loop in loops:
            if len(loop) < 3:
                continue
            anchor = loop[0]
            for index in range(1, len(loop) - 1):
                fill_faces.append([anchor, loop[index], loop[index + 1]])
        if not fill_faces:
            return self.copy()

        mesh = self.copy()
        mesh.faces = np.vstack([self.faces, np.asarray(fill_faces, dtype=np.int64)])
        if self.material_indices is not None:
            fill_material = int(self.material_indices[0]) if self.material_indices.size else 0
            mesh.material_indices = np.concatenate(
                [self.material_indices.copy(), np.full(len(fill_faces), fill_material, dtype=np.int64)]
            )
        return mesh

    def fix_winding(self) -> Mesh:
        try:
            import trimesh

            tri = trimesh.Trimesh(vertices=self.points, faces=self.faces, process=False)
            fix_normals = cast(Callable[..., None], trimesh.repair.fix_normals)
            fix_normals(tri, multibody=True)
            mesh = self.copy()
            mesh.points = np.asarray(tri.vertices, dtype=np.float64)
            mesh.faces = np.asarray(tri.faces, dtype=np.int64)
            return mesh
        except Exception:
            return self.copy()

    def _boundary_loops(self) -> list[list[int]]:
        if self.triangle_count == 0:
            return []
        directed_edges = np.concatenate(
            [
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            ]
        )
        undirected = np.sort(directed_edges, axis=1)
        edges, counts = np.unique(undirected, axis=0, return_counts=True)
        boundary_edges = edges[counts == 1]
        if boundary_edges.size == 0:
            return []

        adjacency: dict[int, set[int]] = {}
        for start, end in boundary_edges.astype(int).tolist():
            adjacency.setdefault(start, set()).add(end)
            adjacency.setdefault(end, set()).add(start)

        loops: list[list[int]] = []
        visited: set[tuple[int, int]] = set()
        for start, neighbors in adjacency.items():
            for neighbor in sorted(neighbors):
                edge = (min(start, neighbor), max(start, neighbor))
                if edge in visited:
                    continue
                loop = [start]
                previous = start
                current = neighbor
                while True:
                    loop.append(current)
                    visited.add(edge)
                    next_candidates = sorted(item for item in adjacency[current] if item != previous)
                    if not next_candidates:
                        break
                    next_vertex = next_candidates[0]
                    next_edge = (min(current, next_vertex), max(current, next_vertex))
                    if next_vertex == start:
                        visited.add(next_edge)
                        break
                    if next_edge in visited:
                        break
                    edge = next_edge
                    previous, current = current, next_vertex
                if len(loop) >= 3:
                    loops.append(loop)
        return loops

    def _drop_non_finite(self) -> Mesh:
        finite = np.asarray(np.isfinite(self.points).all(axis=1), dtype=np.bool_)
        if finite.all():
            return self.copy()
        keep = np.flatnonzero(finite[self.faces].all(axis=1))
        return self._filter_faces(keep).remove_unreferenced_vertices()

    def _filter_faces(self, keep: IntArray) -> Mesh:
        mesh = self.copy()
        mesh.faces = self.faces[keep].copy()
        if self.material_indices is not None:
            mesh.material_indices = self.material_indices[keep].copy()
        old_to_new = {int(old_index): new_index for new_index, old_index in enumerate(keep.astype(int).tolist())}
        mesh.face_groups = {
            name: np.asarray(
                [old_to_new[int(value)] for value in values.astype(int).tolist() if int(value) in old_to_new],
                dtype=np.int64,
            )
            for name, values in self.face_groups.items()
        }
        return mesh

    def _with_geometry(self, points: FloatArray, faces: IntArray) -> Mesh:
        return Mesh(
            points=np.asarray(points, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            material_indices=None,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "uv_channels": sorted(self.uvs),
            "has_normals": self.normals is not None,
            "metadata": dict(self.metadata),
        }
