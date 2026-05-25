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
        return self._with_geometry(self.points[used], remap[self.faces])

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
        return self._with_geometry(new_points, new_faces).remove_degenerate_faces()

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

    def compute_normals(self) -> Mesh:
        normals = np.zeros_like(self.points, dtype=np.float64)
        if self.triangle_count > 0 and self.vertex_count > 0:
            p0 = self.points[self.faces[:, 0]]
            p1 = self.points[self.faces[:, 1]]
            p2 = self.points[self.faces[:, 2]]
            face_normals = np.cross(p1 - p0, p2 - p0)
            for corner in range(3):
                np.add.at(normals, self.faces[:, corner], face_normals)
        lengths = np.linalg.norm(normals, axis=1)
        nonzero = lengths > 0.0
        normals[nonzero] = normals[nonzero] / lengths[nonzero, None]
        normals[~nonzero] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        mesh = self.copy()
        mesh.normals = normals
        return mesh

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
            from fast_simplification import simplify

            points, faces = simplify(
                self.points.astype(np.float64), self.faces.astype(np.int64), target_count=target_triangles
            )
            return Mesh(points=np.asarray(points), faces=np.asarray(faces)).repair(RepairOptions())
        except Exception:
            stride = max(1, int(np.ceil(self.triangle_count / target_triangles)))
            keep = np.arange(0, self.triangle_count, stride, dtype=np.int64)[:target_triangles]
            return self._filter_faces(keep).remove_unreferenced_vertices().compute_normals()

    def optimize_buffers(self) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        try:
            import meshoptimizer

            indices = np.ascontiguousarray(self.faces.reshape(-1), dtype=np.uint32)
            optimized = meshoptimizer.optimize_vertex_cache(indices, self.vertex_count)
            faces = np.asarray(optimized, dtype=np.int64).reshape((-1, 3))
            return self._with_geometry(self.points.copy(), faces)
        except Exception:
            return self.copy()

    def fill_holes(self) -> Mesh:
        try:
            import trimesh

            tri = trimesh.Trimesh(vertices=self.points, faces=self.faces, process=False)
            trimesh.repair.fill_holes(tri)
            return Mesh(points=np.asarray(tri.vertices), faces=np.asarray(tri.faces), metadata=dict(self.metadata))
        except Exception:
            return self.copy()

    def fix_winding(self) -> Mesh:
        try:
            import trimesh

            tri = trimesh.Trimesh(vertices=self.points, faces=self.faces, process=False)
            fix_normals = cast(Callable[..., None], trimesh.repair.fix_normals)
            fix_normals(tri, multibody=True)
            return Mesh(points=np.asarray(tri.vertices), faces=np.asarray(tri.faces), metadata=dict(self.metadata))
        except Exception:
            return self.copy()

    def _drop_non_finite(self) -> Mesh:
        finite = np.asarray(np.isfinite(self.points).all(axis=1), dtype=np.bool_)
        if finite.all():
            return self.copy()
        remap = np.full(self.vertex_count, -1, dtype=np.int64)
        remap[finite] = np.arange(int(finite.sum()), dtype=np.int64)
        face_mask = finite[self.faces].all(axis=1)
        points = self.points[finite]
        faces = remap[self.faces[face_mask]]
        return self._with_geometry(points, faces)

    def _filter_faces(self, keep: IntArray) -> Mesh:
        mesh = self.copy()
        mesh.faces = self.faces[keep].copy()
        if self.material_indices is not None:
            mesh.material_indices = self.material_indices[keep].copy()
        mesh.face_groups = {
            name: np.intersect1d(values, keep).astype(np.int64) for name, values in self.face_groups.items()
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
