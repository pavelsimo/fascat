from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fascat.metadata import Metadata
from fascat.options import DeleteDegeneratePolygonsOptions, MergeVerticesOptions, RepairOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Point2D = tuple[float, float]


class MeshValidationError(ValueError):
    """Raised when mesh arrays are not usable by the pipeline."""


def _should_repair_winding(options: RepairOptions) -> bool:
    return options.fix_winding and options.face_orientation == "exterior"


def _repair_face_orientation_status(options: RepairOptions) -> str:
    if not options.fix_winding:
        return "disabled"
    if options.face_orientation == "exterior":
        return "closed_exterior"
    if options.face_orientation == "source_trusted":
        return "trusted_source"
    if options.face_orientation == "preserve":
        return "preserved"
    return "intent_not_implemented"


def _apply_repair_normal_orientation(
    mesh: Mesh,
    options: RepairOptions,
    input_normals: FloatArray | None,
    *,
    face_orientation_status: str,
) -> tuple[Mesh, str]:
    if options.normal_orientation == "viewer_standpoint":
        return mesh.compute_normals(), "intent_not_implemented"
    if options.normal_orientation == "from_faces":
        return mesh.compute_normals(), "generated_from_faces"
    if (
        input_normals is not None
        and input_normals.shape == (mesh.vertex_count, 3)
        and face_orientation_status in {"disabled", "preserved", "trusted_source"}
    ):
        result = mesh.copy()
        result.normals = input_normals.copy()
        return result, "preserved_existing"
    if face_orientation_status == "closed_exterior":
        return mesh.compute_normals(), "generated_after_face_orientation"
    return mesh.compute_normals(), "generated_missing_source"


def _format_float_value(value: float) -> str:
    return f"{float(value):.9g}"


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


def _triangle_corner_angles(triangles: FloatArray) -> FloatArray:
    if triangles.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    angles = np.empty((triangles.shape[0], 3), dtype=np.float64)
    for corner in range(3):
        origin = triangles[:, corner]
        left = triangles[:, (corner + 1) % 3] - origin
        right = triangles[:, (corner + 2) % 3] - origin
        angles[:, corner] = _vector_angles(left, right)
    return angles


def _polygon_area_2d(points: list[Point2D]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    previous = points[-1]
    for point in points:
        area += previous[0] * point[1] - point[0] * previous[1]
        previous = point
    return area * 0.5


def _inside_clip_edge(point: Point2D, start: Point2D, end: Point2D, orientation: float, tolerance: float) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
    return cross * orientation >= -tolerance


def _line_intersection_2d(start: Point2D, end: Point2D, clip_start: Point2D, clip_end: Point2D) -> Point2D:
    direction = (end[0] - start[0], end[1] - start[1])
    clip_direction = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
    denom = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(denom) <= 1e-15:
        return end
    offset = (clip_start[0] - start[0], clip_start[1] - start[1])
    factor = (offset[0] * clip_direction[1] - offset[1] * clip_direction[0]) / denom
    return (start[0] + factor * direction[0], start[1] + factor * direction[1])


def _clip_polygon_to_triangle(subject: list[Point2D], clip: list[Point2D], *, tolerance: float) -> list[Point2D]:
    orientation = 1.0 if _polygon_area_2d(clip) >= 0.0 else -1.0
    output = subject
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        input_points = output
        output = []
        if not input_points:
            break
        previous = input_points[-1]
        previous_inside = _inside_clip_edge(previous, clip_start, clip_end, orientation, tolerance)
        for current in input_points:
            current_inside = _inside_clip_edge(current, clip_start, clip_end, orientation, tolerance)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection_2d(previous, current, clip_start, clip_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection_2d(previous, current, clip_start, clip_end))
            previous = current
            previous_inside = current_inside
    return output


def _triangle_overlap_area_2d(left: FloatArray, right: FloatArray, *, tolerance: float) -> float:
    subject = [(float(point[0]), float(point[1])) for point in left]
    clip = [(float(point[0]), float(point[1])) for point in right]
    if abs(_polygon_area_2d(subject)) <= tolerance or abs(_polygon_area_2d(clip)) <= tolerance:
        return 0.0
    intersection = _clip_polygon_to_triangle(subject, clip, tolerance=tolerance)
    return abs(_polygon_area_2d(intersection))


@dataclass
class Mesh:
    points: FloatArray
    faces: IntArray
    normals: FloatArray | None = None
    tangents: FloatArray | None = None
    uvs: dict[int, FloatArray] = field(default_factory=dict)
    material_indices: IntArray | None = None
    face_groups: dict[str, IntArray] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.points = np.array(self.points, dtype=np.float64, copy=True)
        self.faces = np.array(self.faces, dtype=np.int64, copy=True)
        if self.normals is not None:
            self.normals = np.array(self.normals, dtype=np.float64, copy=True)
        if self.tangents is not None:
            self.tangents = np.array(self.tangents, dtype=np.float64, copy=True)
        self.uvs = {channel: np.array(values, dtype=np.float64, copy=True) for channel, values in self.uvs.items()}
        if self.material_indices is not None:
            self.material_indices = np.array(self.material_indices, dtype=np.int64, copy=True)
        self.face_groups = {
            name: np.array(values, dtype=np.int64, copy=True) for name, values in self.face_groups.items()
        }
        self.metadata = dict(self.metadata)

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
            tangents=None if self.tangents is None else self.tangents.copy(),
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
        if self.tangents is not None:
            if self.tangents.shape != (self.vertex_count, 4):
                raise MeshValidationError("tangents must have shape (N, 4)")
            if not np.isfinite(self.tangents).all():
                raise MeshValidationError("tangents must not contain NaN or Inf values")
        for channel, values in self.uvs.items():
            if values.ndim != 2 or values.shape[1] != 2:
                raise MeshValidationError(f"uv channel {channel} must have shape (N, 2)")
            if values.shape[0] != self.vertex_count:
                raise MeshValidationError(f"uv channel {channel} must match vertex count")
            if not np.isfinite(values).all():
                raise MeshValidationError(f"uv channel {channel} must not contain NaN or Inf values")
        if self.material_indices is not None and self.material_indices.shape != (self.triangle_count,):
            raise MeshValidationError("material_indices must match triangle count")
        if self.material_indices is not None and self.material_indices.size and int(self.material_indices.min()) < 0:
            raise MeshValidationError("material_indices must not contain negative values")
        for name, face_indices in self.face_groups.items():
            if face_indices.ndim != 1:
                raise MeshValidationError(f"face group {name} must be a one-dimensional array")
            if face_indices.size and int(face_indices.min()) < 0:
                raise MeshValidationError(f"face group {name} must not contain negative face indices")
            if face_indices.size and int(face_indices.max()) >= self.triangle_count:
                raise MeshValidationError(f"face group {name} must not contain out-of-range face indices")

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
        input_normals = None if self.normals is None else self.normals.copy()
        mesh = self.copy()
        mesh = mesh._drop_invalid_faces()
        mesh = mesh._drop_non_finite()
        mesh = mesh.remove_unreferenced_vertices()
        t_junction_tolerance = max(opts.tolerance, 1e-9)
        boundary_gap_tolerance = max(opts.tolerance, 1e-9)
        before_metrics = mesh.quality_metrics(area_epsilon=opts.area_epsilon)
        before_t_junctions = mesh.t_junction_count(tolerance=t_junction_tolerance)
        before_boundary_gaps = mesh.boundary_gap_count(tolerance=boundary_gap_tolerance)
        if opts.merge_vertices and opts.tolerance > 0.0:
            mesh = mesh.merge_close_vertices(opts.tolerance)
        mesh = mesh.remove_duplicate_faces()
        if opts.delete_degenerate:
            mesh = mesh.remove_degenerate_faces(opts.area_epsilon)
        orientation_metrics = mesh.orientability_metrics()
        face_orientation_status = _repair_face_orientation_status(opts)
        if _should_repair_winding(opts):
            mesh = mesh.fix_winding()
        mesh, normal_orientation_status = _apply_repair_normal_orientation(
            mesh,
            opts,
            input_normals,
            face_orientation_status=face_orientation_status,
        )
        if opts.fill_small_holes:
            previous_triangle_count = mesh.triangle_count
            mesh = mesh.fill_holes()
            if mesh.triangle_count != previous_triangle_count:
                mesh = mesh.compute_normals()
                normal_orientation_status = "generated_after_hole_fill"
        after_orientation_metrics = mesh.orientability_metrics()
        after_metrics = mesh.quality_metrics(area_epsilon=opts.area_epsilon)
        after_t_junctions = mesh.t_junction_count(tolerance=t_junction_tolerance)
        after_boundary_gaps = mesh.boundary_gap_count(tolerance=boundary_gap_tolerance)
        repair_metadata = {
            **mesh.metadata,
            "repair_duplicate_polygons_before": str(int(before_metrics["duplicate_polygons"])),
            "repair_duplicate_polygons_after": str(int(after_metrics["duplicate_polygons"])),
            "repair_degenerate_triangles_before": str(int(before_metrics["degenerate_triangles"])),
            "repair_degenerate_triangles_after": str(int(after_metrics["degenerate_triangles"])),
            "repair_boundary_edges_before": str(int(before_metrics["boundary_edges"])),
            "repair_boundary_edges_after": str(int(after_metrics["boundary_edges"])),
            "repair_non_manifold_edges_before": str(int(before_metrics["non_manifold_edges"])),
            "repair_non_manifold_edges_after": str(int(after_metrics["non_manifold_edges"])),
            "repair_t_junctions_before": str(before_t_junctions),
            "repair_t_junctions_after": str(after_t_junctions),
            "repair_boundary_gaps_before": str(before_boundary_gaps),
            "repair_boundary_gaps_after": str(after_boundary_gaps),
            "repair_orientation_components_before_orientation": str(int(orientation_metrics["orientation_components"])),
            "repair_non_orientable_edges_before_orientation": str(int(orientation_metrics["non_orientable_edges"])),
            "repair_closed_orientation_components_before_orientation": str(
                int(orientation_metrics["closed_orientation_components"])
            ),
            "repair_closed_orientation_components_after_orientation": str(
                int(after_orientation_metrics["closed_orientation_components"])
            ),
            "repair_flipped_components_before_orientation": str(
                int(orientation_metrics["flipped_orientation_components"])
            ),
            "repair_flipped_components_after_orientation": str(
                int(after_orientation_metrics["flipped_orientation_components"])
            ),
            "repair_face_orientation_strategy": opts.face_orientation,
            "repair_face_orientation_status": face_orientation_status,
            "repair_normal_orientation_strategy": opts.normal_orientation,
            "repair_normal_orientation_status": normal_orientation_status,
        }
        if opts.viewer_position is not None:
            repair_metadata["repair_orientation_viewer_position"] = ",".join(
                _format_float_value(value) for value in opts.viewer_position
            )
        mesh.metadata = repair_metadata
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
        if self.tangents is not None:
            mesh.tangents = self.tangents[used].copy()
        mesh.uvs = {channel: values[used].copy() for channel, values in self.uvs.items()}
        return mesh

    def merge_close_vertices(self, tolerance: float) -> Mesh:
        if tolerance <= 0.0 or self.vertex_count == 0:
            return self.copy()
        components = self._distance_connected_components(tolerance=tolerance)
        old_to_new = np.empty(self.vertex_count, dtype=np.int64)
        representative_indices: list[int] = []
        for new_index, vertices in enumerate(components):
            representative_indices.append(vertices[0])
            for vertex in vertices:
                old_to_new[vertex] = new_index
        mesh = self.copy()
        mesh.points = self.points[np.asarray(representative_indices, dtype=np.int64)].copy()
        mesh.faces = old_to_new[self.faces]
        mesh.normals = None
        mesh.tangents = None
        mesh.uvs = {}
        return mesh.remove_degenerate_faces()

    def merge_vertices(self, options: MergeVerticesOptions | None = None) -> Mesh:
        opts = options or MergeVerticesOptions()
        if self.vertex_count == 0:
            return self.copy()
        before_vertex_count = self.vertex_count
        before_triangle_count = self.triangle_count
        material_signatures = self._vertex_material_signatures() if opts.preserve_material_boundaries else None
        skip_diagnostics = self._merge_vertex_skip_diagnostics(opts, material_signatures)
        tolerance_diagnostics = self._merge_vertex_tolerance_diagnostics(opts.tolerance)
        components = self._merge_vertex_components(opts, material_signatures)
        old_to_new = np.empty(self.vertex_count, dtype=np.int64)
        representative_indices: list[int] = []
        for new_index, vertices in enumerate(components):
            representative_indices.append(vertices[0])
            for vertex_index in vertices:
                old_to_new[vertex_index] = new_index

        representatives = np.asarray(representative_indices, dtype=np.int64)
        mesh = self.copy()
        mesh.points = self.points[representatives].copy()
        mesh.faces = old_to_new[self.faces]
        mesh.normals = (
            None if self.normals is None or not opts.preserve_normals else self.normals[representatives].copy()
        )
        mesh.tangents = (
            None if self.tangents is None or not opts.preserve_tangents else self.tangents[representatives].copy()
        )
        mesh.uvs = (
            {}
            if not opts.preserve_uvs
            else {channel: values[representatives].copy() for channel, values in self.uvs.items()}
        )
        if opts.delete_degenerate:
            mesh = mesh.remove_degenerate_faces(opts.area_epsilon)
        removed_vertices = before_vertex_count - mesh.vertex_count
        removed_triangles = before_triangle_count - mesh.triangle_count
        mesh.metadata = {
            **mesh.metadata,
            "merge_vertices_tolerance": f"{opts.tolerance:g}",
            "merge_vertices_before": str(before_vertex_count),
            "merge_vertices_after": str(mesh.vertex_count),
            "merge_vertices_removed": str(removed_vertices),
            "merge_vertices_triangles_before": str(before_triangle_count),
            "merge_vertices_triangles_after": str(mesh.triangle_count),
            "merge_vertices_degenerate_triangles_removed": str(removed_triangles),
            "merge_vertices_preserve_normals": str(opts.preserve_normals).lower(),
            "merge_vertices_preserve_tangents": str(opts.preserve_tangents).lower(),
            "merge_vertices_preserve_uvs": str(opts.preserve_uvs).lower(),
            "merge_vertices_preserve_material_boundaries": str(opts.preserve_material_boundaries).lower(),
            **{key: str(value) for key, value in skip_diagnostics.items()},
            **tolerance_diagnostics,
        }
        mesh.validate()
        return mesh

    def _merge_vertex_tolerance_diagnostics(self, tolerance: float) -> dict[str, str]:
        mins, maxs = self.bounds()
        diagonal = float(np.linalg.norm(maxs - mins))
        edge_lengths = self._triangle_edge_lengths().reshape(-1)
        positive_edges = edge_lengths[edge_lengths > 0.0]
        min_edge = float(positive_edges.min()) if positive_edges.size else 0.0
        near_duplicate_pairs, nearest_near_duplicate = self._near_duplicate_unmerged_stats(
            tolerance=tolerance,
            diagonal=diagonal,
            min_edge=min_edge,
        )
        bbox_ratio = tolerance / diagonal if tolerance > 0.0 and diagonal > 0.0 else 0.0
        min_edge_ratio = tolerance / min_edge if tolerance > 0.0 and min_edge > 0.0 else 0.0
        risk = "exact_only"
        if tolerance > 0.0:
            if min_edge_ratio >= 0.25:
                risk = "high_relative_to_min_edge"
            elif bbox_ratio >= 0.01:
                risk = "high_relative_to_bbox"
            else:
                risk = "nominal"
        return {
            "merge_vertices_bbox_diagonal": f"{diagonal:.9g}",
            "merge_vertices_min_edge_length": f"{min_edge:.9g}",
            "merge_vertices_tolerance_bbox_ratio": f"{bbox_ratio:.9g}",
            "merge_vertices_tolerance_min_edge_ratio": f"{min_edge_ratio:.9g}",
            "merge_vertices_tolerance_risk": risk,
            "merge_vertices_near_duplicate_pairs": str(near_duplicate_pairs),
            "merge_vertices_nearest_near_duplicate_distance": f"{nearest_near_duplicate:.9g}",
            "merge_vertices_tolerance_advisory": "near_duplicates_unmerged" if near_duplicate_pairs else "none",
        }

    def _near_duplicate_unmerged_stats(
        self,
        *,
        tolerance: float,
        diagonal: float,
        min_edge: float,
    ) -> tuple[int, float]:
        if self.vertex_count < 2:
            return 0, 0.0
        lower = max(float(tolerance), 0.0)
        upper = self._near_duplicate_advisory_limit(tolerance=tolerance, diagonal=diagonal, min_edge=min_edge)
        if upper <= lower or upper <= 0.0:
            return 0, 0.0
        connected_edges = self._connected_edge_pairs()
        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        near_pairs = 0
        nearest = math.inf
        for vertex, point in enumerate(self.points):
            key = self._spatial_bucket_key(point, upper)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                        for other in buckets.get(neighbor_key, []):
                            pair = (min(vertex, other), max(vertex, other))
                            if pair in connected_edges:
                                continue
                            distance = float(np.linalg.norm(point - self.points[other]))
                            if lower < distance <= upper:
                                near_pairs += 1
                                nearest = min(nearest, distance)
            buckets[key].append(vertex)
        return near_pairs, 0.0 if math.isinf(nearest) else nearest

    def _near_duplicate_advisory_limit(self, *, tolerance: float, diagonal: float, min_edge: float) -> float:
        scale_limit = 0.0
        if min_edge > 0.0:
            scale_limit = min_edge * 0.05
        elif diagonal > 0.0:
            scale_limit = diagonal * 1e-6
        tolerance_limit = tolerance * 10.0 if tolerance > 0.0 else 0.0
        upper = max(scale_limit, tolerance_limit)
        if diagonal > 0.0:
            upper = min(upper, diagonal * 0.01)
        return upper

    def _connected_edge_pairs(self) -> set[tuple[int, int]]:
        edges, _counts = self._undirected_edges_and_counts()
        return {(int(edge[0]), int(edge[1])) for edge in edges.astype(int).tolist()}

    def _spatial_bucket_key(self, point: FloatArray, cell_size: float) -> tuple[int, int, int]:
        key = np.floor(point / cell_size).astype(np.int64)
        return (int(key[0]), int(key[1]), int(key[2]))

    def _distance_connected_components(
        self,
        *,
        tolerance: float,
        component_keys: Sequence[tuple[object, ...]] | None = None,
    ) -> list[list[int]]:
        if self.vertex_count == 0:
            return []
        if tolerance <= 0.0:
            exact_groups: dict[tuple[object, ...], list[int]] = {}
            for vertex, point in enumerate(self.points):
                key: tuple[object, ...] = (("position", tuple(float(value) for value in point)),)
                if component_keys is not None:
                    key = key + component_keys[vertex]
                exact_groups.setdefault(key, []).append(vertex)
            return sorted(exact_groups.values(), key=lambda vertices: vertices[0])

        parent = list(range(self.vertex_count))
        rank = [0] * self.vertex_count

        def find(vertex: int) -> int:
            while parent[vertex] != vertex:
                parent[vertex] = parent[parent[vertex]]
                vertex = parent[vertex]
            return vertex

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if rank[left_root] < rank[right_root]:
                parent[left_root] = right_root
            elif rank[left_root] > rank[right_root]:
                parent[right_root] = left_root
            else:
                parent[right_root] = left_root
                rank[left_root] += 1

        buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for vertex, point in enumerate(self.points):
            key = self._spatial_bucket_key(point, tolerance)
            component_key = None if component_keys is None else component_keys[vertex]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                        for other in buckets.get(neighbor_key, []):
                            if component_keys is not None and component_keys[other] != component_key:
                                continue
                            if float(np.linalg.norm(point - self.points[other])) <= tolerance:
                                union(vertex, other)
            buckets[key].append(vertex)

        components: dict[int, list[int]] = {}
        for vertex in range(self.vertex_count):
            components.setdefault(find(vertex), []).append(vertex)
        return sorted(components.values(), key=lambda vertices: vertices[0])

    def _merge_vertex_components(
        self,
        options: MergeVerticesOptions,
        material_signatures: tuple[tuple[int, ...], ...] | None,
    ) -> list[list[int]]:
        attribute_keys = [
            self._merge_vertex_attribute_key(vertex_index, options, material_signatures)
            for vertex_index in range(self.vertex_count)
        ]
        return self._distance_connected_components(tolerance=options.tolerance, component_keys=attribute_keys)

    def _merge_vertex_skip_diagnostics(
        self,
        options: MergeVerticesOptions,
        material_signatures: tuple[tuple[int, ...], ...] | None,
    ) -> dict[str, int]:
        candidate_components = self._distance_connected_components(tolerance=options.tolerance)

        diagnostics = {
            "merge_vertices_candidate_position_buckets": 0,
            "merge_vertices_candidate_vertices": 0,
            "merge_vertices_candidate_exact_duplicate_buckets": 0,
            "merge_vertices_candidate_boundary_buckets": 0,
            "merge_vertices_candidate_non_manifold_buckets": 0,
            "merge_vertices_candidate_hard_edge_buckets": 0,
            "merge_vertices_candidate_t_junctions": self.t_junction_count(tolerance=max(options.tolerance, 1e-9)),
            "merge_vertices_candidate_boundary_gaps": self.boundary_gap_count(tolerance=max(options.tolerance, 1e-9)),
            "merge_vertices_skipped_by_protection": 0,
            "merge_vertices_skipped_by_normals": 0,
            "merge_vertices_skipped_by_tangents": 0,
            "merge_vertices_skipped_by_uvs": 0,
            "merge_vertices_skipped_by_material_boundaries": 0,
        }
        boundary_vertices, non_manifold_vertices = self._merge_vertex_topology_vertices()
        for vertices in candidate_components:
            if len(vertices) <= 1:
                continue
            diagnostics["merge_vertices_candidate_position_buckets"] += 1
            diagnostics["merge_vertices_candidate_vertices"] += len(vertices) - 1
            exact_keys = {
                self._merge_vertex_exact_identity_key(vertex, options, material_signatures) for vertex in vertices
            }
            if len(exact_keys) < len(vertices):
                diagnostics["merge_vertices_candidate_exact_duplicate_buckets"] += 1
            if any(vertex in boundary_vertices for vertex in vertices):
                diagnostics["merge_vertices_candidate_boundary_buckets"] += 1
            if any(vertex in non_manifold_vertices for vertex in vertices):
                diagnostics["merge_vertices_candidate_non_manifold_buckets"] += 1
            if self.normals is not None and self._has_distinct_normals(vertices):
                diagnostics["merge_vertices_candidate_hard_edge_buckets"] += 1
            attribute_keys = {
                self._merge_vertex_attribute_key(vertex, options, material_signatures) for vertex in vertices
            }
            skipped = len(attribute_keys) - 1
            if skipped <= 0:
                continue
            diagnostics["merge_vertices_skipped_by_protection"] += skipped
            if options.preserve_normals and self.normals is not None and self._has_distinct_normals(vertices):
                diagnostics["merge_vertices_skipped_by_normals"] += skipped
            if options.preserve_tangents and self.tangents is not None and self._has_distinct_tangents(vertices):
                diagnostics["merge_vertices_skipped_by_tangents"] += skipped
            if options.preserve_uvs and self._has_distinct_uvs(vertices):
                diagnostics["merge_vertices_skipped_by_uvs"] += skipped
            if material_signatures is not None and self._has_distinct_material_signatures(
                vertices, material_signatures
            ):
                diagnostics["merge_vertices_skipped_by_material_boundaries"] += skipped
        return diagnostics

    def _merge_vertex_topology_vertices(self) -> tuple[set[int], set[int]]:
        boundary_vertices: set[int] = set()
        non_manifold_vertices: set[int] = set()
        for edge, faces in self._edge_faces_map().items():
            if len(faces) == 1:
                boundary_vertices.update(edge)
            elif len(faces) > 2:
                non_manifold_vertices.update(edge)
        return boundary_vertices, non_manifold_vertices

    def _has_distinct_normals(self, vertices: list[int]) -> bool:
        assert self.normals is not None
        return len({tuple(float(value) for value in self.normals[vertex]) for vertex in vertices}) > 1

    def _has_distinct_tangents(self, vertices: list[int]) -> bool:
        assert self.tangents is not None
        return len({tuple(float(value) for value in self.tangents[vertex]) for vertex in vertices}) > 1

    def _has_distinct_uvs(self, vertices: list[int]) -> bool:
        for values in self.uvs.values():
            if len({tuple(float(value) for value in values[vertex]) for vertex in vertices}) > 1:
                return True
        return False

    def _has_distinct_material_signatures(
        self,
        vertices: list[int],
        material_signatures: tuple[tuple[int, ...], ...],
    ) -> bool:
        return len({material_signatures[vertex] for vertex in vertices}) > 1

    def _merge_vertex_exact_identity_key(
        self,
        vertex_index: int,
        options: MergeVerticesOptions,
        material_signatures: tuple[tuple[int, ...], ...] | None,
    ) -> tuple[object, ...]:
        return (
            ("position", tuple(float(value) for value in self.points[vertex_index])),
            *self._merge_vertex_attribute_key(vertex_index, options, material_signatures),
        )

    def _merge_vertex_attribute_key(
        self,
        vertex_index: int,
        options: MergeVerticesOptions,
        material_signatures: tuple[tuple[int, ...], ...] | None,
    ) -> tuple[object, ...]:
        key: list[object] = []
        if options.preserve_normals and self.normals is not None:
            key.append(tuple(float(value) for value in self.normals[vertex_index]))
        if options.preserve_tangents and self.tangents is not None:
            key.append(tuple(float(value) for value in self.tangents[vertex_index]))
        if options.preserve_uvs:
            for channel in sorted(self.uvs):
                key.append((channel, tuple(float(value) for value in self.uvs[channel][vertex_index])))
        if material_signatures is not None:
            key.append(("materials", material_signatures[vertex_index]))
        return tuple(key)

    def _vertex_material_signatures(self) -> tuple[tuple[int, ...], ...]:
        if self.material_indices is None:
            return tuple(() for _index in range(self.vertex_count))
        signatures: list[set[int]] = [set() for _index in range(self.vertex_count)]
        for face_index, face in enumerate(self.faces):
            material_index = int(self.material_indices[face_index])
            for vertex_index in face.astype(int).tolist():
                signatures[vertex_index].add(material_index)
        return tuple(tuple(sorted(signature)) for signature in signatures)

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

    def delete_degenerate_polygons(self, options: DeleteDegeneratePolygonsOptions | None = None) -> Mesh:
        opts = options or DeleteDegeneratePolygonsOptions()
        before_vertex_count = self.vertex_count
        before_triangle_count = self.triangle_count
        before_degenerate_count = int(self.quality_metrics(area_epsilon=opts.area_epsilon)["degenerate_triangles"])
        reason_counts = self._degenerate_polygon_reason_counts(opts.area_epsilon)
        mesh = self.remove_degenerate_faces(opts.area_epsilon)
        duplicate_polygons_before = int(mesh.quality_metrics(area_epsilon=opts.area_epsilon)["duplicate_polygons"])
        if opts.delete_duplicates:
            mesh = mesh.remove_duplicate_faces().remove_unreferenced_vertices()
        duplicate_polygons_after = int(mesh.quality_metrics(area_epsilon=opts.area_epsilon)["duplicate_polygons"])
        after_degenerate_count = int(mesh.quality_metrics(area_epsilon=opts.area_epsilon)["degenerate_triangles"])
        mesh.metadata = {
            **mesh.metadata,
            "delete_degenerate_polygons_area_epsilon": f"{opts.area_epsilon:g}",
            "delete_degenerate_polygons_delete_duplicates": str(opts.delete_duplicates).lower(),
            "delete_degenerate_polygons_vertices_before": str(before_vertex_count),
            "delete_degenerate_polygons_vertices_after": str(mesh.vertex_count),
            "delete_degenerate_polygons_vertices_removed": str(before_vertex_count - mesh.vertex_count),
            "delete_degenerate_polygons_triangles_before": str(before_triangle_count),
            "delete_degenerate_polygons_triangles_after": str(mesh.triangle_count),
            "delete_degenerate_polygons_removed": str(before_triangle_count - mesh.triangle_count),
            "delete_degenerate_polygons_before": str(before_degenerate_count),
            "delete_degenerate_polygons_after": str(after_degenerate_count),
            "delete_degenerate_polygons_duplicate_polygons_before": str(duplicate_polygons_before),
            "delete_degenerate_polygons_duplicate_polygons_after": str(duplicate_polygons_after),
            "delete_degenerate_polygons_removed_duplicate_polygons": str(
                max(0, duplicate_polygons_before - duplicate_polygons_after)
            ),
            **{key: str(value) for key, value in reason_counts.items()},
        }
        mesh.validate()
        return mesh

    def _degenerate_polygon_reason_counts(self, area_epsilon: float) -> dict[str, int]:
        counts = {
            "delete_degenerate_polygons_removed_duplicate_vertices": 0,
            "delete_degenerate_polygons_removed_collapsed_edges": 0,
            "delete_degenerate_polygons_removed_near_flat_area": 0,
        }
        if self.triangle_count == 0:
            return counts
        p0 = self.points[self.faces[:, 0]]
        p1 = self.points[self.faces[:, 1]]
        p2 = self.points[self.faces[:, 2]]
        areas = np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) * 0.5
        edge_epsilon = math.sqrt(area_epsilon) if area_epsilon > 0.0 else 0.0
        for face_index in np.flatnonzero(areas <= area_epsilon).astype(int).tolist():
            face = self.faces[face_index].astype(int).tolist()
            if len(set(face)) < 3:
                counts["delete_degenerate_polygons_removed_duplicate_vertices"] += 1
                continue
            triangle = self.points[face]
            edge_lengths = (
                float(np.linalg.norm(triangle[1] - triangle[0])),
                float(np.linalg.norm(triangle[2] - triangle[1])),
                float(np.linalg.norm(triangle[0] - triangle[2])),
            )
            if any(length <= edge_epsilon for length in edge_lengths):
                counts["delete_degenerate_polygons_removed_collapsed_edges"] += 1
            else:
                counts["delete_degenerate_polygons_removed_near_flat_area"] += 1
        return counts

    def quality_metrics(
        self,
        *,
        min_edge_length: float | None = None,
        max_edge_length: float | None = None,
        skinny_aspect_ratio: float = 20.0,
        area_epsilon: float = 1e-12,
    ) -> dict[str, int | float]:
        if self.triangle_count == 0:
            return {
                "vertices": self.vertex_count,
                "triangles": 0,
                "min_edge_length": 0.0,
                "max_edge_length": 0.0,
                "mean_edge_length": 0.0,
                "min_triangle_area": 0.0,
                "max_triangle_area": 0.0,
                "mean_triangle_area": 0.0,
                "max_aspect_ratio": 0.0,
                "skinny_triangles": 0,
                "degenerate_triangles": 0,
                "duplicate_polygons": 0,
                "short_edges": 0,
                "long_edges": 0,
                "boundary_edges": 0,
                "non_manifold_edges": 0,
            }

        lengths = self._triangle_edge_lengths()
        flat_lengths = lengths.reshape(-1)
        areas = self._triangle_areas()
        min_lengths = lengths.min(axis=1)
        max_lengths = lengths.max(axis=1)
        aspect_ratios = np.divide(
            max_lengths,
            min_lengths,
            out=np.full(max_lengths.shape, np.inf, dtype=np.float64),
            where=min_lengths > 0.0,
        )
        finite_aspects = aspect_ratios[np.isfinite(aspect_ratios)]
        _edges, counts = self._undirected_edges_and_counts()
        polygon_keys = np.sort(self.faces, axis=1)
        _unique_polygons, polygon_counts = np.unique(polygon_keys, axis=0, return_counts=True)

        return {
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "min_edge_length": float(flat_lengths.min()) if flat_lengths.size else 0.0,
            "max_edge_length": float(flat_lengths.max()) if flat_lengths.size else 0.0,
            "mean_edge_length": float(flat_lengths.mean()) if flat_lengths.size else 0.0,
            "min_triangle_area": float(areas.min()) if areas.size else 0.0,
            "max_triangle_area": float(areas.max()) if areas.size else 0.0,
            "mean_triangle_area": float(areas.mean()) if areas.size else 0.0,
            "max_aspect_ratio": float(finite_aspects.max()) if finite_aspects.size else 0.0,
            "skinny_triangles": int(np.count_nonzero(aspect_ratios > skinny_aspect_ratio)),
            "degenerate_triangles": int(np.count_nonzero(areas <= area_epsilon)),
            "duplicate_polygons": int(np.sum(np.maximum(polygon_counts - 1, 0))),
            "short_edges": 0 if min_edge_length is None else int(np.count_nonzero(flat_lengths < min_edge_length)),
            "long_edges": 0 if max_edge_length is None else int(np.count_nonzero(flat_lengths > max_edge_length)),
            "boundary_edges": int(np.count_nonzero(counts == 1)),
            "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        }

    def t_junction_count(self, *, tolerance: float = 1e-9) -> int:
        if self.triangle_count == 0 or self.vertex_count < 3:
            return 0
        distance_tolerance = max(float(tolerance), 1e-12)
        edges, _counts = self._undirected_edges_and_counts()
        conflicts: set[tuple[int, int, int]] = set()
        for start_index, end_index in edges.astype(int).tolist():
            start = self.points[start_index]
            end = self.points[end_index]
            vector = end - start
            length_squared = float(np.dot(vector, vector))
            if length_squared <= distance_tolerance * distance_tolerance:
                continue
            minimum = np.minimum(start, end) - distance_tolerance
            maximum = np.maximum(start, end) + distance_tolerance
            candidates = np.flatnonzero(np.all((self.points >= minimum) & (self.points <= maximum), axis=1))
            if candidates.size == 0:
                continue
            candidate_points = self.points[candidates]
            projection = np.asarray(((candidate_points - start) @ vector) / length_squared, dtype=np.float64)
            length = math.sqrt(length_squared)
            endpoint_margin = distance_tolerance / length
            interior = (projection > endpoint_margin) & (projection < 1.0 - endpoint_margin)
            if not np.any(interior):
                continue
            projected = start + (projection[:, None] * vector)
            distances = np.linalg.norm(candidate_points - projected, axis=1)
            on_edge = interior & (distances <= distance_tolerance)
            for candidate in candidates[on_edge].astype(int).tolist():
                if candidate in {start_index, end_index}:
                    continue
                conflicts.add((min(start_index, end_index), max(start_index, end_index), candidate))
        return len(conflicts)

    def boundary_gap_count(self, *, tolerance: float = 1e-9) -> int:
        if self.triangle_count == 0 or self.vertex_count < 2:
            return 0
        distance_tolerance = max(float(tolerance), 1e-12)
        boundary_edges = self._boundary_edges_set()
        if not boundary_edges:
            return 0
        boundary_vertices = sorted({vertex for edge in boundary_edges for vertex in edge})
        all_edges, _counts = self._undirected_edges_and_counts()
        connected_edges = {(int(edge[0]), int(edge[1])) for edge in all_edges.astype(int).tolist()}
        buckets: dict[tuple[int, int, int], list[int]] = {}
        gaps: set[tuple[int, int]] = set()

        def bucket_key(point: FloatArray) -> tuple[int, int, int]:
            key = np.floor(point / distance_tolerance).astype(np.int64)
            return (int(key[0]), int(key[1]), int(key[2]))

        for vertex in boundary_vertices:
            point = self.points[vertex]
            key = bucket_key(point)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                        for other in buckets.get(neighbor_key, []):
                            pair = (min(vertex, other), max(vertex, other))
                            if pair in connected_edges:
                                continue
                            if float(np.linalg.norm(point - self.points[other])) <= distance_tolerance:
                                gaps.add(pair)
            buckets.setdefault(key, []).append(vertex)
        return len(gaps)

    def orientability_metrics(self) -> dict[str, int]:
        if self.triangle_count == 0:
            return {
                "orientation_components": 0,
                "non_orientable_edges": 0,
                "closed_orientation_components": 0,
                "flipped_orientation_components": 0,
            }

        edge_incidents: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        face_edges: list[list[tuple[int, int]]] = []
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            edges: list[tuple[int, int]] = []
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (min(start, end), max(start, end))
                direction = 1 if (start, end) == key else -1
                edge_incidents[key].append((face_index, direction))
                edges.append(key)
            face_edges.append(edges)

        adjacency: dict[int, list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
        for edge, incidents in edge_incidents.items():
            if len(incidents) != 2:
                continue
            (left_face, left_direction), (right_face, right_direction) = incidents
            required_relation = -left_direction * right_direction
            adjacency[left_face].append((right_face, required_relation, edge))
            adjacency[right_face].append((left_face, required_relation, edge))

        face_signs: dict[int, int] = {}
        conflict_edges: set[tuple[int, int]] = set()
        component_faces: list[set[int]] = []
        components = 0
        for face_index in range(self.triangle_count):
            if face_index in face_signs:
                continue
            components += 1
            faces = {face_index}
            face_signs[face_index] = 1
            queue: deque[int] = deque([face_index])
            while queue:
                current = queue.popleft()
                for neighbor, required_relation, edge in adjacency[current]:
                    expected_sign = face_signs[current] * required_relation
                    if neighbor not in face_signs:
                        face_signs[neighbor] = expected_sign
                        faces.add(neighbor)
                        queue.append(neighbor)
                    elif face_signs[neighbor] != expected_sign:
                        conflict_edges.add(edge)
            component_faces.append(faces)

        closed_components = 0
        flipped_components = 0
        volume_epsilon = self._orientation_volume_epsilon()
        for faces in component_faces:
            component_edges = {edge for face_index in faces for edge in face_edges[face_index]}
            if any(edge in conflict_edges or len(edge_incidents[edge]) != 2 for edge in component_edges):
                continue
            closed_components += 1
            if any(face_signs[face_index] != 1 for face_index in faces):
                continue
            if self._signed_volume_for_faces(faces) < -volume_epsilon:
                flipped_components += 1

        return {
            "orientation_components": components,
            "non_orientable_edges": len(conflict_edges),
            "closed_orientation_components": closed_components,
            "flipped_orientation_components": flipped_components,
        }

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
        mesh.tangents = None
        return mesh

    def compute_flat_normals(self) -> Mesh:
        if self.triangle_count == 0:
            return self.compute_normals()
        points: list[list[float]] = []
        normals: list[list[float]] = []
        faces: list[list[int]] = []
        uvs: dict[int, list[list[float]]] = {channel: [] for channel in self.uvs}
        face_normals = self._face_unit_normals()
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            new_face: list[int] = []
            for vertex in face:
                new_face.append(len(points))
                points.append(self.points[vertex].astype(float).tolist())
                normals.append(face_normals[face_index].astype(float).tolist())
                for channel, values in self.uvs.items():
                    uvs[channel].append(values[vertex].astype(float).tolist())
            faces.append(new_face)
        mesh = Mesh(
            points=np.asarray(points, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            normals=np.asarray(normals, dtype=np.float64),
            uvs={channel: np.asarray(values, dtype=np.float64) for channel, values in uvs.items()},
            material_indices=None if self.material_indices is None else self.material_indices.copy(),
            face_groups={name: values.copy() for name, values in self.face_groups.items()},
            metadata={**self.metadata, "normal_mode": "flat"},
        )
        mesh.validate()
        return mesh

    def compute_hard_edge_normals(
        self,
        *,
        hard_edge_angle: float = 30.0,
        preserve_face_boundaries: bool = False,
        angle_weighted: bool = True,
    ) -> Mesh:
        if self.triangle_count == 0:
            return self.compute_normals(angle_weighted=angle_weighted)
        hard_edges = self._hard_normal_edges(
            hard_edge_angle=hard_edge_angle,
            preserve_face_boundaries=preserve_face_boundaries,
        )
        if not hard_edges:
            mesh = self.compute_normals(angle_weighted=angle_weighted)
            mesh.metadata = {
                **mesh.metadata,
                "normal_mode": "hard_edges",
                "normal_weighting": "angle" if angle_weighted else "area",
            }
            return mesh

        edge_faces = self._edge_faces_map()
        face_normals = self._face_unit_normals()
        triangle_points = self.points[self.faces]
        corner_angles = _triangle_corner_angles(triangle_points)
        p0 = triangle_points[:, 0]
        p1 = triangle_points[:, 1]
        p2 = triangle_points[:, 2]
        raw_face_normals = np.cross(p1 - p0, p2 - p0)
        incident_faces: dict[int, set[int]] = {index: set() for index in range(self.vertex_count)}
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            for vertex in face:
                incident_faces[vertex].add(face_index)

        component_by_vertex_face: dict[tuple[int, int], int] = {}
        component_normals: list[FloatArray] = []
        for vertex, faces in incident_faces.items():
            remaining = set(faces)
            while remaining:
                seed = remaining.pop()
                component = {seed}
                stack = [seed]
                while stack:
                    current = stack.pop()
                    current_face = self.faces[current].astype(int).tolist()
                    for edge in (
                        (current_face[0], current_face[1]),
                        (current_face[1], current_face[2]),
                        (current_face[2], current_face[0]),
                    ):
                        if vertex not in edge:
                            continue
                        key = (min(edge), max(edge))
                        if key in hard_edges:
                            continue
                        for neighbor in edge_faces.get(key, []):
                            if neighbor in remaining:
                                remaining.remove(neighbor)
                                component.add(neighbor)
                                stack.append(neighbor)
                if angle_weighted:
                    normal = np.zeros(3, dtype=np.float64)
                    for face_index in component:
                        face_vertices = self.faces[face_index].astype(int).tolist()
                        corner = face_vertices.index(vertex)
                        normal += face_normals[face_index] * corner_angles[face_index, corner]
                else:
                    normal = raw_face_normals[list(component)].sum(axis=0)
                length = float(np.linalg.norm(normal))
                if length > 0.0:
                    normal = normal / length
                component_index = len(component_normals)
                component_normals.append(normal)
                for face_index in component:
                    component_by_vertex_face[(vertex, face_index)] = component_index

        points: list[list[float]] = []
        normals: list[list[float]] = []
        uvs: dict[int, list[list[float]]] = {channel: [] for channel in self.uvs}
        vertex_map: dict[tuple[int, int], int] = {}
        new_faces: list[list[int]] = []
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            new_face: list[int] = []
            for vertex in face:
                component_index = component_by_vertex_face[(vertex, face_index)]
                key = (vertex, component_index)
                new_index = vertex_map.get(key)
                if new_index is None:
                    new_index = len(points)
                    vertex_map[key] = new_index
                    points.append(self.points[vertex].astype(float).tolist())
                    normals.append(component_normals[component_index].astype(float).tolist())
                    for channel, values in self.uvs.items():
                        uvs[channel].append(values[vertex].astype(float).tolist())
                new_face.append(new_index)
            new_faces.append(new_face)

        mesh = Mesh(
            points=np.asarray(points, dtype=np.float64),
            faces=np.asarray(new_faces, dtype=np.int64),
            normals=np.asarray(normals, dtype=np.float64),
            uvs={channel: np.asarray(values, dtype=np.float64) for channel, values in uvs.items()},
            material_indices=None if self.material_indices is None else self.material_indices.copy(),
            face_groups={name: values.copy() for name, values in self.face_groups.items()},
            metadata={
                **self.metadata,
                "normal_mode": "hard_edges",
                "hard_edge_angle": str(hard_edge_angle),
                "preserve_face_boundaries": str(preserve_face_boundaries).lower(),
                "normal_weighting": "angle" if angle_weighted else "area",
            },
        )
        mesh.validate_normals()
        return mesh

    def compute_tangents(self, channel: int = 0) -> Mesh:
        if channel < 0:
            raise ValueError("tangent UV channel must be greater than or equal to 0")
        if channel not in self.uvs or self.triangle_count == 0:
            mesh = self.copy()
            mesh.tangents = None
            return mesh
        mesh = self if self.normals is not None else self.compute_normals()
        assert mesh.normals is not None
        uv = mesh.uvs[channel]
        tangents = np.zeros((mesh.vertex_count, 3), dtype=np.float64)
        bitangents = np.zeros((mesh.vertex_count, 3), dtype=np.float64)
        for face in mesh.faces.astype(int).tolist():
            i0, i1, i2 = face
            p0, p1, p2 = mesh.points[[i0, i1, i2]]
            uv0, uv1, uv2 = uv[[i0, i1, i2]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            duv1 = uv1 - uv0
            duv2 = uv2 - uv0
            denom = float(duv1[0] * duv2[1] - duv2[0] * duv1[1])
            if abs(denom) <= 1e-12:
                continue
            tangent = (edge1 * duv2[1] - edge2 * duv1[1]) / denom
            bitangent = (edge2 * duv1[0] - edge1 * duv2[0]) / denom
            np.add.at(tangents, face, tangent)
            np.add.at(bitangents, face, bitangent)
        handedness = np.ones(mesh.vertex_count, dtype=np.float64)
        for index, normal in enumerate(mesh.normals):
            tangent = tangents[index]
            tangent = tangent - normal * float(np.dot(normal, tangent))
            length = float(np.linalg.norm(tangent))
            tangent = _fallback_tangent(normal) if length <= 0.0 else tangent / length
            tangents[index] = tangent
            bitangent = bitangents[index]
            if float(np.linalg.norm(bitangent)) > 0.0 and float(np.dot(np.cross(normal, tangent), bitangent)) < 0.0:
                handedness[index] = -1.0
        result = mesh.copy()
        result.tangents = np.column_stack([tangents, handedness])
        result.metadata = {**result.metadata, "tangents": "mikktspace_like"}
        result.validate_normals(require_tangents=True)
        return result

    def validate_normals(self, *, require_tangents: bool = False) -> None:
        self.validate()
        if self.normals is None:
            raise MeshValidationError("mesh has no normals")
        lengths = np.linalg.norm(self.normals, axis=1)
        if not np.allclose(lengths, 1.0, atol=1e-4):
            raise MeshValidationError("normals must be unit length")
        if require_tangents and self.tangents is None:
            raise MeshValidationError("mesh has no tangents")

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

    def collapse_short_edges(self, min_edge_length: float, *, preserve_boundaries: bool = True) -> Mesh:
        if min_edge_length <= 0.0:
            raise ValueError("min_edge_length must be greater than 0")
        if self.triangle_count == 0 or self.vertex_count == 0:
            return self.copy()

        boundary_edges = self._boundary_edges_set() if preserve_boundaries else set()
        parent = np.arange(self.vertex_count, dtype=np.int64)
        rank = np.zeros(self.vertex_count, dtype=np.int64)
        merged = False

        def find(index: int) -> int:
            while int(parent[index]) != index:
                parent[index] = parent[int(parent[index])]
                index = int(parent[index])
            return index

        def union(left: int, right: int) -> None:
            nonlocal merged
            left_root = find(left)
            right_root = find(right)
            if left_root == right_root:
                return
            if rank[left_root] < rank[right_root]:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root
            if rank[left_root] == rank[right_root]:
                rank[left_root] += 1
            merged = True

        for start, end in self._face_edges().astype(int).tolist():
            edge = (min(start, end), max(start, end))
            if edge in boundary_edges:
                continue
            if float(np.linalg.norm(self.points[end] - self.points[start])) < min_edge_length:
                union(start, end)

        if not merged:
            return self.copy()

        roots = np.asarray([find(index) for index in range(self.vertex_count)], dtype=np.int64)
        unique_roots, inverse = np.unique(roots, return_inverse=True)
        new_points = np.empty((unique_roots.shape[0], 3), dtype=np.float64)
        for new_index, root in enumerate(unique_roots.astype(int).tolist()):
            new_points[new_index] = self.points[roots == root].mean(axis=0)

        mesh = self.copy()
        mesh.points = new_points
        mesh.faces = inverse[self.faces]
        mesh.normals = None
        mesh.uvs = {}
        mesh = mesh.remove_degenerate_faces().remove_unreferenced_vertices()
        mesh.metadata = {**self.metadata, "min_edge_length": str(min_edge_length)}
        return mesh

    def improve_skinny_triangles(
        self,
        *,
        max_aspect_ratio: float = 20.0,
        preserve_boundaries: bool = True,
        max_iterations: int = 4,
    ) -> Mesh:
        if max_aspect_ratio <= 1.0:
            raise ValueError("max_aspect_ratio must be greater than 1")
        mesh = self.copy()
        if mesh.triangle_count == 0:
            return mesh

        points = mesh.points.tolist()
        faces: list[list[int]] = mesh.faces.astype(int).tolist()
        material_indices = None if mesh.material_indices is None else mesh.material_indices.astype(int).tolist()

        for _iteration in range(max_iterations):
            changed = False
            boundary_edges = _boundary_edges_from_faces(faces) if preserve_boundaries else set()
            next_faces: list[list[int]] = []
            next_materials: list[int] = []
            for face_index, face in enumerate(faces):
                triangle = np.asarray([points[index] for index in face], dtype=np.float64)
                edge_pairs = ((0, 1), (1, 2), (2, 0))
                lengths = [float(np.linalg.norm(triangle[b] - triangle[a])) for a, b in edge_pairs]
                shortest = min(lengths)
                longest = max(lengths)
                if shortest <= 0.0 or longest / shortest <= max_aspect_ratio:
                    next_faces.append(face)
                    if material_indices is not None:
                        next_materials.append(material_indices[face_index])
                    continue
                longest_index = int(np.argmax(lengths))
                a_corner, b_corner = edge_pairs[longest_index]
                a = face[a_corner]
                b = face[b_corner]
                edge = (min(a, b), max(a, b))
                if edge in boundary_edges:
                    next_faces.append(face)
                    if material_indices is not None:
                        next_materials.append(material_indices[face_index])
                    continue
                changed = True
                c_corner = next(corner for corner in range(3) if corner not in {a_corner, b_corner})
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
            if not changed:
                break

        result = Mesh(
            points=np.asarray(points, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            material_indices=None if material_indices is None else np.asarray(material_indices, dtype=np.int64),
            metadata={**mesh.metadata, "skinny_triangle_max_aspect_ratio": str(max_aspect_ratio)},
        )
        return result.remove_degenerate_faces().remove_unreferenced_vertices()

    def box_uv(self, channel: int = 0) -> Mesh:
        mesh = self.copy()
        if self.vertex_count == 0:
            mesh.uvs[channel] = np.empty((0, 2), dtype=np.float64)
            mesh.tangents = None
            return mesh
        mins, maxs = self.bounds()
        size = maxs - mins
        axes = np.argsort(size)[-2:]
        denom = size[axes]
        denom[denom == 0.0] = 1.0
        uv = (self.points[:, axes] - mins[axes]) / denom
        mesh.uvs[channel] = uv.astype(np.float64)
        mesh.tangents = None
        return mesh

    def uv_layout_stats(self, channel: int = 0, *, tolerance: float = 1e-9) -> dict[str, int]:
        if tolerance < 0.0:
            raise ValueError("tolerance must be greater than or equal to 0")
        if channel not in self.uvs:
            raise ValueError(f"uv channel {channel} is not present")

        uv = self.uvs[channel]
        out_of_unit_vertices = int(np.count_nonzero(np.any((uv < -tolerance) | (uv > 1.0 + tolerance), axis=1)))
        if self.triangle_count == 0:
            return {
                "vertices": self.vertex_count,
                "faces": 0,
                "out_of_unit_vertices": out_of_unit_vertices,
                "degenerate_faces": 0,
                "overlapping_face_pairs": 0,
            }

        triangles = uv[self.faces]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        signed_areas = 0.5 * (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0])
        degenerate = np.abs(signed_areas) <= tolerance
        min_uv = triangles.min(axis=1)
        max_uv = triangles.max(axis=1)
        order = np.argsort(min_uv[:, 0], kind="mergesort")

        overlapping_pairs = 0
        for position, left_index_value in enumerate(order):
            left_index = int(left_index_value)
            if bool(degenerate[left_index]):
                continue
            left_min = min_uv[left_index]
            left_max = max_uv[left_index]
            for right_index_value in order[position + 1 :]:
                right_index = int(right_index_value)
                if min_uv[right_index, 0] > left_max[0] + tolerance:
                    break
                if bool(degenerate[right_index]):
                    continue
                if min_uv[right_index, 1] > left_max[1] + tolerance:
                    continue
                if max_uv[right_index, 1] + tolerance < left_min[1]:
                    continue
                if (
                    _triangle_overlap_area_2d(triangles[left_index], triangles[right_index], tolerance=tolerance)
                    > tolerance
                ):
                    overlapping_pairs += 1

        return {
            "vertices": self.vertex_count,
            "faces": self.triangle_count,
            "out_of_unit_vertices": out_of_unit_vertices,
            "degenerate_faces": int(np.count_nonzero(degenerate)),
            "overlapping_face_pairs": overlapping_pairs,
        }

    def uv_distortion_metrics(self, channel: int = 0, *, tolerance: float = 1e-9) -> dict[str, int | float]:
        if tolerance < 0.0:
            raise ValueError("tolerance must be greater than or equal to 0")
        if channel not in self.uvs:
            raise ValueError(f"uv channel {channel} is not present")

        uv = self.uvs[channel]
        if self.triangle_count == 0:
            return {
                "island_count": 0,
                "uv_area": 0.0,
                "uv_bounds_area": 0.0,
                "pack_efficiency": 0.0,
                "normalized_pack_efficiency": 0.0,
                "max_angle_distortion_degrees": 0.0,
                "mean_angle_distortion_degrees": 0.0,
                "max_edge_length_distortion": 0.0,
                "mean_edge_length_distortion": 0.0,
            }

        uv_triangles = uv[self.faces]
        uv_edge_a = uv_triangles[:, 1] - uv_triangles[:, 0]
        uv_edge_b = uv_triangles[:, 2] - uv_triangles[:, 0]
        uv_signed_areas = 0.5 * (uv_edge_a[:, 0] * uv_edge_b[:, 1] - uv_edge_a[:, 1] * uv_edge_b[:, 0])
        uv_abs_areas = np.abs(uv_signed_areas)
        uv_area = float(uv_abs_areas.sum())
        uv_bounds_area = self._uv_bounds_area(channel)
        pack_efficiency = min(1.0, uv_area / uv_bounds_area) if uv_bounds_area > tolerance else 0.0
        normalized_pack_efficiency = min(1.0, uv_area)

        geometry_areas = self._triangle_areas()
        valid_faces = (geometry_areas > tolerance) & (uv_abs_areas > tolerance)
        max_angle_distortion = 0.0
        mean_angle_distortion = 0.0
        max_edge_distortion = 0.0
        mean_edge_distortion = 0.0
        if np.any(valid_faces):
            geometry_angles = _triangle_corner_angles(self.points[self.faces][valid_faces])
            uv_angles = _triangle_corner_angles(uv_triangles[valid_faces])
            angle_distortion = np.degrees(np.abs(geometry_angles - uv_angles))
            max_angle_distortion = float(angle_distortion.max()) if angle_distortion.size else 0.0
            mean_angle_distortion = float(angle_distortion.mean()) if angle_distortion.size else 0.0

            geometry_edges = self._triangle_edge_lengths()[valid_faces]
            uv_edges = self._uv_triangle_edge_lengths(channel)[valid_faces]
            geometry_perimeters = geometry_edges.sum(axis=1)
            uv_perimeters = uv_edges.sum(axis=1)
            valid_edges = (geometry_perimeters > tolerance) & (uv_perimeters > tolerance)
            if np.any(valid_edges):
                scale = uv_perimeters[valid_edges] / geometry_perimeters[valid_edges]
                expected_uv_edges = geometry_edges[valid_edges] * scale[:, None]
                edge_distortion = np.abs(uv_edges[valid_edges] - expected_uv_edges) / np.maximum(
                    expected_uv_edges,
                    tolerance,
                )
                max_edge_distortion = float(edge_distortion.max()) if edge_distortion.size else 0.0
                mean_edge_distortion = float(edge_distortion.mean()) if edge_distortion.size else 0.0

        return {
            "island_count": self._uv_island_count(channel, tolerance=tolerance),
            "uv_area": uv_area,
            "uv_bounds_area": uv_bounds_area,
            "pack_efficiency": pack_efficiency,
            "normalized_pack_efficiency": normalized_pack_efficiency,
            "max_angle_distortion_degrees": max_angle_distortion,
            "mean_angle_distortion_degrees": mean_angle_distortion,
            "max_edge_length_distortion": max_edge_distortion,
            "mean_edge_length_distortion": mean_edge_distortion,
        }

    def unwrap_uv(self, channel: int = 0) -> Mesh:
        try:
            import xatlas
        except ImportError as exc:
            raise RuntimeError("UV unwrap requires the optional xatlas dependency") from exc

        if self.triangle_count == 0:
            mesh = self.copy()
            mesh.uvs[channel] = np.empty((0, 2), dtype=np.float64)
            mesh.tangents = None
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
            tangents=None,
            uvs={**{key: values[mapping].copy() for key, values in self.uvs.items() if key != channel}},
            material_indices=None if self.material_indices is None else self.material_indices.copy(),
            face_groups={name: values.copy() for name, values in self.face_groups.items()},
            metadata={**self.metadata, f"uv{channel}": "xatlas"},
        )
        mesh.uvs[channel] = np.asarray(uvs, dtype=np.float64)
        mesh.validate()
        return mesh

    def simplify(
        self,
        *,
        target_triangles: int | None = None,
        ratio: float | None = None,
        preserve_hard_edges: bool = False,
        hard_edge_angle: float = 30.0,
        preserve_holes: bool = False,
        preserve_material_boundaries: bool = False,
        preserve_uv_seams: bool = False,
        preserve_silhouette: bool = False,
    ) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        if target_triangles is None:
            if ratio is None:
                return self.copy()
            target_triangles = max(1, int(round(self.triangle_count * ratio)))
        target_triangles = max(1, min(int(target_triangles), self.triangle_count))
        if target_triangles >= self.triangle_count:
            return self.copy()

        if any(
            (
                preserve_hard_edges,
                preserve_holes,
                preserve_material_boundaries,
                preserve_uv_seams,
                preserve_silhouette,
            )
        ):
            protected = self._feature_face_indices(
                preserve_hard_edges=preserve_hard_edges,
                hard_edge_angle=hard_edge_angle,
                preserve_holes=preserve_holes,
                preserve_material_boundaries=preserve_material_boundaries,
                preserve_uv_seams=preserve_uv_seams,
                preserve_silhouette=preserve_silhouette,
            )
            if protected.size:
                mesh = self._simplify_preserving_faces(protected, max(target_triangles, int(protected.shape[0])))
                mesh.metadata = {
                    **mesh.metadata,
                    "simplification_preserved_feature_faces": str(int(protected.shape[0])),
                }
                mesh.validate()
                return mesh

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
            if self.tangents is not None:
                vertex_attributes.append(self.tangents.astype(np.float32))
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
            if self.tangents is not None:
                mesh.tangents = self.tangents[old_for_new].copy()
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

    def feature_preservation_counts(
        self,
        *,
        preserve_hard_edges: bool = False,
        hard_edge_angle: float = 30.0,
        preserve_holes: bool = False,
        preserve_material_boundaries: bool = False,
        preserve_uv_seams: bool = False,
        preserve_silhouette: bool = False,
    ) -> dict[str, int]:
        groups = self._feature_face_groups(
            preserve_hard_edges=preserve_hard_edges,
            hard_edge_angle=hard_edge_angle,
            preserve_holes=preserve_holes,
            preserve_material_boundaries=preserve_material_boundaries,
            preserve_uv_seams=preserve_uv_seams,
            preserve_silhouette=preserve_silhouette,
        )
        union: set[int] = set()
        for values in groups.values():
            union.update(values)
        return {
            "hard_edge_faces": len(groups["hard_edges"]),
            "hole_boundary_faces": len(groups["holes"]),
            "material_boundary_faces": len(groups["material_boundaries"]),
            "uv_seam_faces": len(groups["uv_seams"]),
            "silhouette_faces": len(groups["silhouette"]),
            "total_feature_faces": len(union),
        }

    def _simplify_preserving_faces(self, protected_faces: IntArray, target_triangles: int) -> Mesh:
        protected = {int(index) for index in protected_faces.astype(int).tolist()}
        keep = set(protected)
        remaining = max(0, min(target_triangles, self.triangle_count) - len(keep))
        if remaining:
            unprotected = [index for index in range(self.triangle_count) if index not in protected]
            if unprotected:
                stride = max(1, int(np.ceil(len(unprotected) / remaining)))
                keep.update(unprotected[::stride][:remaining])
        keep_indices = np.asarray(sorted(keep), dtype=np.int64)
        if keep_indices.shape[0] >= self.triangle_count:
            return self.copy()
        return self._filter_faces(keep_indices).remove_unreferenced_vertices().compute_normals()

    def _feature_face_indices(
        self,
        *,
        preserve_hard_edges: bool,
        hard_edge_angle: float,
        preserve_holes: bool,
        preserve_material_boundaries: bool,
        preserve_uv_seams: bool,
        preserve_silhouette: bool,
    ) -> IntArray:
        groups = self._feature_face_groups(
            preserve_hard_edges=preserve_hard_edges,
            hard_edge_angle=hard_edge_angle,
            preserve_holes=preserve_holes,
            preserve_material_boundaries=preserve_material_boundaries,
            preserve_uv_seams=preserve_uv_seams,
            preserve_silhouette=preserve_silhouette,
        )
        protected: set[int] = set()
        for values in groups.values():
            protected.update(values)
        return np.asarray(sorted(protected), dtype=np.int64)

    def _feature_face_groups(
        self,
        *,
        preserve_hard_edges: bool,
        hard_edge_angle: float,
        preserve_holes: bool,
        preserve_material_boundaries: bool,
        preserve_uv_seams: bool,
        preserve_silhouette: bool,
    ) -> dict[str, set[int]]:
        groups: dict[str, set[int]] = {
            "hard_edges": set(),
            "holes": set(),
            "material_boundaries": set(),
            "uv_seams": set(),
            "silhouette": set(),
        }
        if self.triangle_count == 0:
            return groups

        edge_faces = self._edge_faces_map()
        if preserve_holes:
            for faces in edge_faces.values():
                if len(faces) == 1:
                    groups["holes"].update(faces)
        if preserve_material_boundaries and self.material_indices is not None:
            for faces in edge_faces.values():
                if len(faces) == 2 and self.material_indices[faces[0]] != self.material_indices[faces[1]]:
                    groups["material_boundaries"].update(faces)
        if preserve_hard_edges:
            face_normals = self._face_unit_normals()
            limit = math.cos(math.radians(hard_edge_angle))
            for faces in edge_faces.values():
                if len(faces) != 2:
                    continue
                cosine = float(np.dot(face_normals[faces[0]], face_normals[faces[1]]))
                if cosine < limit:
                    groups["hard_edges"].update(faces)
        if preserve_uv_seams:
            seam_vertices = self._uv_seam_vertices()
            if seam_vertices:
                for face_index, face in enumerate(self.faces.astype(int).tolist()):
                    if any(vertex in seam_vertices for vertex in face):
                        groups["uv_seams"].add(face_index)
        if preserve_silhouette:
            groups["silhouette"].update(self._silhouette_face_indices())
        return groups

    def _hard_normal_edges(
        self,
        *,
        hard_edge_angle: float,
        preserve_face_boundaries: bool,
    ) -> set[tuple[int, int]]:
        hard_edges: set[tuple[int, int]] = set()
        edge_faces = self._edge_faces_map()
        face_normals = self._face_unit_normals()
        limit = math.cos(math.radians(hard_edge_angle))
        face_groups = self._face_group_by_face()
        for edge, faces in edge_faces.items():
            if len(faces) != 2:
                hard_edges.add(edge)
                continue
            left, right = faces
            if self.material_indices is not None and self.material_indices[left] != self.material_indices[right]:
                hard_edges.add(edge)
                continue
            if preserve_face_boundaries and face_groups.get(left) != face_groups.get(right):
                hard_edges.add(edge)
                continue
            cosine = float(np.dot(face_normals[left], face_normals[right]))
            if cosine < limit:
                hard_edges.add(edge)
        return hard_edges

    def _face_group_by_face(self) -> dict[int, str]:
        result: dict[int, str] = {}
        for name, values in self.face_groups.items():
            for face_index in values.astype(int).tolist():
                result[int(face_index)] = name
        return result

    def _edge_faces_map(self) -> dict[tuple[int, int], list[int]]:
        edge_faces: dict[tuple[int, int], list[int]] = {}
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = (min(start, end), max(start, end))
                edge_faces.setdefault(edge, []).append(face_index)
        return edge_faces

    def _face_unit_normals(self) -> FloatArray:
        if self.triangle_count == 0:
            return np.empty((0, 3), dtype=np.float64)
        p0 = self.points[self.faces[:, 0]]
        p1 = self.points[self.faces[:, 1]]
        p2 = self.points[self.faces[:, 2]]
        normals = np.cross(p1 - p0, p2 - p0)
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 0.0
        unit = np.zeros_like(normals, dtype=np.float64)
        unit[valid] = normals[valid] / lengths[valid, None]
        return unit

    def _orientation_volume_epsilon(self) -> float:
        mins, maxs = self.bounds()
        scale = max(float(np.linalg.norm(maxs - mins)), 1e-9)
        return scale**3 * 1e-12

    def _signed_volume_for_faces(self, face_indices: set[int]) -> float:
        if not face_indices:
            return 0.0
        faces = self.faces[np.asarray(sorted(face_indices), dtype=np.int64)]
        p0 = self.points[faces[:, 0]]
        p1 = self.points[faces[:, 1]]
        p2 = self.points[faces[:, 2]]
        return float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6.0)

    def _uv_bounds_area(self, channel: int) -> float:
        uv = self.uvs[channel]
        if uv.shape[0] == 0:
            return 0.0
        span = uv.max(axis=0) - uv.min(axis=0)
        return float(max(span[0], 0.0) * max(span[1], 0.0))

    def _uv_triangle_edge_lengths(self, channel: int) -> FloatArray:
        if self.triangle_count == 0:
            return np.empty((0, 3), dtype=np.float64)
        triangles = self.uvs[channel][self.faces]
        return cast(
            FloatArray,
            np.linalg.norm(
                np.stack(
                    [
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 1],
                        triangles[:, 0] - triangles[:, 2],
                    ],
                    axis=1,
                ),
                axis=2,
            ),
        )

    def _uv_island_count(self, channel: int, *, tolerance: float) -> int:
        if self.triangle_count == 0:
            return 0
        uv = self.uvs[channel]
        uv_triangles = uv[self.faces]
        edge_a = uv_triangles[:, 1] - uv_triangles[:, 0]
        edge_b = uv_triangles[:, 2] - uv_triangles[:, 0]
        uv_areas = np.abs(0.5 * (edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]))
        valid_faces = set(np.flatnonzero(uv_areas > tolerance).astype(int).tolist())
        if not valid_faces:
            return 0

        parent = np.arange(self.triangle_count, dtype=np.int64)

        def find(index: int) -> int:
            while int(parent[index]) != index:
                parent[index] = parent[int(parent[index])]
                index = int(parent[index])
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        edge_incidents: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            if face_index not in valid_faces:
                continue
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge_incidents[(min(start, end), max(start, end))].append((face_index, start, end))

        for incidents in edge_incidents.values():
            if len(incidents) != 2:
                continue
            left_face, left_start, left_end = incidents[0]
            right_face, right_start, right_end = incidents[1]
            same_direction = bool(
                np.linalg.norm(uv[left_start] - uv[right_start]) <= tolerance
                and np.linalg.norm(uv[left_end] - uv[right_end]) <= tolerance
            )
            opposite_direction = bool(
                np.linalg.norm(uv[left_start] - uv[right_end]) <= tolerance
                and np.linalg.norm(uv[left_end] - uv[right_start]) <= tolerance
            )
            if same_direction or opposite_direction:
                union(left_face, right_face)

        return len({find(face_index) for face_index in valid_faces})

    def _flip_inward_closed_components(self) -> Mesh:
        flipped_components = self._flipped_closed_orientation_component_faces()
        if not flipped_components:
            return self.copy()

        mesh = self.copy()
        for face_indices in flipped_components:
            indices = np.asarray(sorted(face_indices), dtype=np.int64)
            mesh.faces[indices] = mesh.faces[indices][:, [0, 2, 1]]
        mesh.normals = None
        mesh.tangents = None
        return mesh

    def _flipped_closed_orientation_component_faces(self) -> list[set[int]]:
        if self.triangle_count == 0:
            return []

        edge_incidents: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        face_edges: list[list[tuple[int, int]]] = []
        for face_index, face in enumerate(self.faces.astype(int).tolist()):
            edges: list[tuple[int, int]] = []
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (min(start, end), max(start, end))
                direction = 1 if (start, end) == key else -1
                edge_incidents[key].append((face_index, direction))
                edges.append(key)
            face_edges.append(edges)

        adjacency: dict[int, list[int]] = defaultdict(list)
        for incidents in edge_incidents.values():
            if len(incidents) != 2:
                continue
            left_face = incidents[0][0]
            right_face = incidents[1][0]
            adjacency[left_face].append(right_face)
            adjacency[right_face].append(left_face)

        visited: set[int] = set()
        flipped_components: list[set[int]] = []
        volume_epsilon = self._orientation_volume_epsilon()
        for face_index in range(self.triangle_count):
            if face_index in visited:
                continue
            faces = {face_index}
            visited.add(face_index)
            queue: deque[int] = deque([face_index])
            while queue:
                current = queue.popleft()
                for neighbor in adjacency[current]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    faces.add(neighbor)
                    queue.append(neighbor)

            component_edges = {edge for component_face in faces for edge in face_edges[component_face]}
            if any(len(edge_incidents[edge]) != 2 for edge in component_edges):
                continue
            if any(edge_incidents[edge][0][1] == edge_incidents[edge][1][1] for edge in component_edges):
                continue
            if self._signed_volume_for_faces(faces) < -volume_epsilon:
                flipped_components.append(faces)

        return flipped_components

    def _uv_seam_vertices(self) -> set[int]:
        if not self.uvs or self.vertex_count == 0:
            return set()
        by_position: dict[tuple[float, float, float], list[int]] = {}
        rounded = np.round(self.points, 9)
        for index, point in enumerate(rounded.tolist()):
            by_position.setdefault((float(point[0]), float(point[1]), float(point[2])), []).append(index)

        seam_vertices: set[int] = set()
        for indices in by_position.values():
            if len(indices) < 2:
                continue
            for channel_values in self.uvs.values():
                rounded_uvs = np.round(channel_values[indices], 9)
                if np.unique(rounded_uvs, axis=0).shape[0] > 1:
                    seam_vertices.update(indices)
                    break
        return seam_vertices

    def _silhouette_face_indices(self) -> set[int]:
        if self.vertex_count == 0 or self.triangle_count == 0:
            return set()
        mins, maxs = self.bounds()
        span = maxs - mins
        tolerance = max(float(span.max()) * 1e-6, 1e-9)
        on_extents = np.any(np.isclose(self.points, mins, atol=tolerance), axis=1) | np.any(
            np.isclose(self.points, maxs, atol=tolerance),
            axis=1,
        )
        return {
            face_index
            for face_index, face in enumerate(self.faces.astype(int).tolist())
            if any(on_extents[vertex] for vertex in face)
        }

    def _assign_materials_by_nearest_centroid(self, points: FloatArray, faces: IntArray) -> IntArray | None:
        if self.material_indices is None or self.triangle_count == 0 or faces.size == 0:
            return None
        source_centroids = self.points[self.faces].mean(axis=1)
        target_centroids = points[faces].mean(axis=1)
        n_target = target_centroids.shape[0]
        n_source = source_centroids.shape[0]
        nearest = np.zeros(n_target, dtype=np.int64)
        # Chunk both dimensions so the (target_block x source_block x 3) working
        # array stays bounded (~tens of MB) regardless of mesh size; otherwise a
        # full-resolution source mesh would allocate tens of GB and exhaust RAM.
        max_block_elems = 4_000_000
        target_block = max(1, min(n_target, max_block_elems // max(1, n_source)))
        if target_block * n_source > max_block_elems:
            target_block = max(1, min(n_target, 2048))
        source_block = max(1, min(n_source, max_block_elems // target_block))
        for t0 in range(0, n_target, target_block):
            t1 = min(t0 + target_block, n_target)
            tgt = target_centroids[t0:t1]
            best_dist = np.full(t1 - t0, np.inf)
            best_idx = np.zeros(t1 - t0, dtype=np.int64)
            for s0 in range(0, n_source, source_block):
                s1 = min(s0 + source_block, n_source)
                delta = tgt[:, None, :] - source_centroids[None, s0:s1, :]
                dist = np.einsum("ijk,ijk->ij", delta, delta)
                local_idx = np.argmin(dist, axis=1)
                local_dist = dist[np.arange(t1 - t0), local_idx]
                improved = local_dist < best_dist
                best_dist[improved] = local_dist[improved]
                best_idx[improved] = local_idx[improved] + s0
            nearest[t0:t1] = best_idx
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
            mesh._remap_face_attributes_from(self)
            return mesh._flip_inward_closed_components()
        except Exception:
            return self._flip_inward_closed_components()

    def _boundary_loops(self) -> list[list[int]]:
        if self.triangle_count == 0:
            return []
        edges, counts = self._undirected_edges_and_counts()
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

    def _triangle_edge_lengths(self) -> FloatArray:
        if self.triangle_count == 0:
            return np.empty((0, 3), dtype=np.float64)
        triangles = self.points[self.faces]
        return cast(
            FloatArray,
            np.linalg.norm(
                np.stack(
                    [
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 1],
                        triangles[:, 0] - triangles[:, 2],
                    ],
                    axis=1,
                ),
                axis=2,
            ),
        )

    def _triangle_areas(self) -> FloatArray:
        if self.triangle_count == 0:
            return np.empty((0,), dtype=np.float64)
        p0 = self.points[self.faces[:, 0]]
        p1 = self.points[self.faces[:, 1]]
        p2 = self.points[self.faces[:, 2]]
        return cast(FloatArray, np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1) * 0.5)

    def _face_edges(self) -> IntArray:
        if self.triangle_count == 0:
            return np.empty((0, 2), dtype=np.int64)
        return cast(
            IntArray,
            np.concatenate(
                [
                    self.faces[:, [0, 1]],
                    self.faces[:, [1, 2]],
                    self.faces[:, [2, 0]],
                ]
            ),
        )

    def _undirected_edges_and_counts(self) -> tuple[IntArray, IntArray]:
        if self.triangle_count == 0:
            return np.empty((0, 2), dtype=np.int64), np.empty((0,), dtype=np.int64)
        undirected = np.sort(self._face_edges(), axis=1)
        edges, counts = np.unique(undirected, axis=0, return_counts=True)
        return edges, counts

    def _boundary_edges_set(self) -> set[tuple[int, int]]:
        edges, counts = self._undirected_edges_and_counts()
        return {(int(edge[0]), int(edge[1])) for edge in edges[counts == 1].astype(int).tolist()}

    def _drop_non_finite(self) -> Mesh:
        finite = np.asarray(np.isfinite(self.points).all(axis=1), dtype=np.bool_)
        if finite.all():
            return self.copy()
        keep = np.flatnonzero(finite[self.faces].all(axis=1))
        return self._filter_faces(keep).remove_unreferenced_vertices()

    def _drop_invalid_faces(self) -> Mesh:
        if self.triangle_count == 0:
            return self.copy()
        valid = np.asarray((self.faces >= 0).all(axis=1) & (self.faces < self.vertex_count).all(axis=1), dtype=np.bool_)
        if valid.all():
            return self.copy()
        keep = np.flatnonzero(valid)
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

    def _remap_face_attributes_from(self, source: Mesh) -> None:
        if source.triangle_count != self.triangle_count or (source.material_indices is None and not source.face_groups):
            return

        old_face_indices_by_key: dict[tuple[int, int, int], list[int]] = {}
        for index, face in enumerate(source.faces.astype(int).tolist()):
            old_face_indices_by_key.setdefault(tuple(sorted(face)), []).append(index)

        new_to_old: list[int] = []
        for face in self.faces.astype(int).tolist():
            candidates = old_face_indices_by_key.get(tuple(sorted(face)))
            if not candidates:
                return
            new_to_old.append(candidates.pop(0))

        mapping = np.asarray(new_to_old, dtype=np.int64)
        old_to_new = {int(old_index): new_index for new_index, old_index in enumerate(mapping.tolist())}
        if source.material_indices is not None:
            self.material_indices = source.material_indices[mapping].copy()
        if source.face_groups:
            self.face_groups = {
                name: np.asarray(
                    [old_to_new[int(value)] for value in values.astype(int).tolist() if int(value) in old_to_new],
                    dtype=np.int64,
                )
                for name, values in source.face_groups.items()
            }

    def _with_geometry(self, points: FloatArray, faces: IntArray) -> Mesh:
        return Mesh(
            points=np.asarray(points, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            material_indices=None,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        material_indices = None
        if self.material_indices is not None:
            material_indices = {
                "count": int(self.material_indices.shape[0]),
                "unique": sorted({int(value) for value in self.material_indices.tolist()}),
            }
        return {
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
            "uv_channels": sorted(self.uvs),
            "has_normals": self.normals is not None,
            "has_tangents": self.tangents is not None,
            "material_indices": material_indices,
            "face_groups": {
                name: {"count": int(values.shape[0]), "indices": values.astype(int).tolist()}
                for name, values in self.face_groups.items()
            },
            "metadata": dict(self.metadata),
        }


def _boundary_edges_from_faces(faces: list[list[int]]) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (min(start, end), max(start, end))
            counts[edge] = counts.get(edge, 0) + 1
    return {edge for edge, count in counts.items() if count == 1}


def _fallback_tangent(normal: FloatArray) -> FloatArray:
    axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(axis, normal))) > 0.9:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    tangent = axis - normal * float(np.dot(axis, normal))
    length = float(np.linalg.norm(tangent))
    if length <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return tangent / length
