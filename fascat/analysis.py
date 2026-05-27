from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.filter import Filter
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import AnalyzeOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_GLTF_ACCESSOR_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_GLTF_DTYPES: dict[int, np.dtype[Any]] = {
    5120: np.dtype("i1"),
    5121: np.dtype("u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}
_INTERSECTION_EPSILON = 1e-9


@dataclass
class AnalysisReport:
    source_path: str | None = None
    options: dict[str, object] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    parts: list[dict[str, object]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.options = dict(self.options)
        self.stats = dict(self.stats)
        self.summary = dict(self.summary)
        self.parts = [dict(part) for part in self.parts]
        self.warnings = list(self.warnings)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_path": self.source_path,
            "options": dict(self.options),
            "stats": dict(self.stats),
            "summary": dict(self.summary),
            "parts": [dict(part) for part in self.parts],
            "warnings": list(self.warnings),
        }

    def write_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def analyze_asset(
    asset: Asset,
    options: AnalyzeOptions | None = None,
    *,
    source_path: str | Path | None = None,
) -> AnalysisReport:
    opts = options or AnalyzeOptions()
    stats = asset.stats(include_lods=any(part.lod_meshes for part in asset.parts.values()))
    summary: dict[str, object] = {
        "parts": asset.part_count,
        "vertices": asset.vertex_count,
        "triangles": asset.triangle_count,
        "material_count": asset.material_count,
    }
    if opts.draw_call_estimate or opts.visual_risk:
        draw_call_breakdown = asset.draw_call_breakdown()
        summary["draw_call_estimate"] = draw_call_breakdown["draw_calls"]
        summary.update(draw_call_breakdown)

    parts: list[dict[str, object]] = []
    totals = _QualityTotals()
    check_topology = opts.non_manifold_edges or opts.open_boundaries or opts.visual_risk
    check_slivers = opts.sliver_triangles or opts.visual_risk
    check_tiny = opts.tiny_parts or opts.visual_risk
    check_self_intersections = opts.self_intersections or opts.visual_risk
    warnings: list[str] = []

    for part in asset.parts.values():
        if part.mesh is None:
            continue
        entry = _mesh_quality_entry(
            part,
            part.mesh,
            opts,
            include_topology=check_topology,
            include_slivers=check_slivers,
            include_tiny=check_tiny,
            include_self_intersections=check_self_intersections,
        )
        parts.append(entry.values)
        totals.add(entry.totals)
        warnings.extend(entry.warnings)

    if opts.non_manifold_edges or opts.visual_risk:
        summary["non_manifold_edges"] = totals.non_manifold_edges
    if opts.open_boundaries or opts.visual_risk:
        summary["open_boundaries"] = totals.open_boundaries
        summary["boundary_edges"] = totals.boundary_edges
    if opts.self_intersections or opts.visual_risk:
        summary["self_intersections"] = totals.self_intersection_warnings
        summary["self_intersection_warnings"] = totals.self_intersection_warnings
        summary["self_intersections_lower_bound"] = totals.self_intersections_lower_bound
        summary["self_intersection_pairs_checked"] = totals.self_intersection_pairs_checked
        summary["self_intersection_pair_limit"] = totals.self_intersection_pair_limit
    if opts.sliver_triangles or opts.visual_risk:
        summary["degenerate_triangles"] = totals.degenerate_triangles
        summary["sliver_triangles"] = totals.sliver_triangles
        summary["max_aspect_ratio"] = totals.max_aspect_ratio
    if opts.tiny_parts or opts.visual_risk:
        summary["tiny_parts"] = totals.tiny_parts
        summary["tiny_part_triangles"] = totals.tiny_part_triangles

    if opts.visual_risk:
        risk_warnings = _visual_risk_warnings(asset, totals)
        summary["visual_risk_warnings"] = len(risk_warnings)
        warnings.extend(risk_warnings)

    return AnalysisReport(
        source_path=str(source_path)
        if source_path is not None
        else (str(asset.source_path) if asset.source_path else None),
        options=opts.to_dict(),
        stats=stats,
        summary=summary,
        parts=parts,
        warnings=_dedupe(warnings),
    )


def analyze_output(
    path: str | Path,
    options: AnalyzeOptions | None = None,
    *,
    where: Filter | None = None,
    validation_stats: dict[str, int] | None = None,
    source_path: str | Path | None = None,
) -> AnalysisReport:
    opts = options or AnalyzeOptions()
    output_path = Path(path)
    stats = dict(validation_stats or _validate_output(output_path))
    try:
        asset = _asset_from_output(output_path)
    except Exception as exc:
        if where is not None:
            raise RuntimeError(f"filtered validation requires readable mesh data: {exc}") from exc
        return _validation_only_report(output_path, opts, stats, source_path=source_path, warning=str(exc))

    selection = asset.select(where) if where is not None else None
    report = analyze_asset(
        asset if selection is None else _scoped_asset(asset, selection.part_ids),
        opts,
        source_path=source_path or output_path,
    )
    if selection is not None:
        report.summary["selection"] = selection.to_dict()
    report.stats = {
        **report.stats,
        "validated_meshes": stats["meshes"],
        "validated_points": stats["points"],
        "validated_triangles": stats["triangles"],
    }
    return report


@dataclass
class _QualityTotals:
    non_manifold_edges: int = 0
    open_boundaries: int = 0
    boundary_edges: int = 0
    self_intersection_warnings: int = 0
    self_intersections_lower_bound: bool = False
    self_intersection_pairs_checked: int = 0
    self_intersection_pair_limit: int = 0
    degenerate_triangles: int = 0
    sliver_triangles: int = 0
    max_aspect_ratio: float = 0.0
    tiny_parts: int = 0
    tiny_part_triangles: int = 0

    def add(self, other: _QualityTotals) -> None:
        self.non_manifold_edges += other.non_manifold_edges
        self.open_boundaries += other.open_boundaries
        self.boundary_edges += other.boundary_edges
        self.self_intersection_warnings += other.self_intersection_warnings
        self.self_intersections_lower_bound = (
            self.self_intersections_lower_bound or other.self_intersections_lower_bound
        )
        self.self_intersection_pairs_checked += other.self_intersection_pairs_checked
        self.self_intersection_pair_limit = max(self.self_intersection_pair_limit, other.self_intersection_pair_limit)
        self.degenerate_triangles += other.degenerate_triangles
        self.sliver_triangles += other.sliver_triangles
        self.max_aspect_ratio = max(self.max_aspect_ratio, other.max_aspect_ratio)
        self.tiny_parts += other.tiny_parts
        self.tiny_part_triangles += other.tiny_part_triangles


@dataclass(frozen=True)
class _MeshQualityEntry:
    values: dict[str, object]
    totals: _QualityTotals
    warnings: list[str]


@dataclass(frozen=True)
class _SelfIntersectionResult:
    intersections: int
    truncated: bool
    pairs_checked: int
    pair_limit: int


def _mesh_quality_entry(
    part: Part,
    mesh: Mesh,
    options: AnalyzeOptions,
    *,
    include_topology: bool,
    include_slivers: bool,
    include_tiny: bool,
    include_self_intersections: bool,
) -> _MeshQualityEntry:
    metrics = mesh.quality_metrics(
        skinny_aspect_ratio=options.sliver_aspect_ratio,
        area_epsilon=options.degenerate_area_epsilon,
    )
    bbox = _bounds_payload(mesh)
    totals = _QualityTotals()
    warnings: list[str] = []
    values: dict[str, object] = {
        "part_id": part.id,
        "name": part.name,
        "vertices": mesh.vertex_count,
        "triangles": mesh.triangle_count,
        "bounds": bbox,
    }

    if include_topology:
        totals.non_manifold_edges = int(metrics["non_manifold_edges"])
        totals.boundary_edges = int(metrics["boundary_edges"])
        totals.open_boundaries = _open_boundary_count(mesh)
        values["non_manifold_edges"] = totals.non_manifold_edges
        values["boundary_edges"] = totals.boundary_edges
        values["open_boundaries"] = totals.open_boundaries

    if include_slivers:
        totals.degenerate_triangles = int(metrics["degenerate_triangles"])
        totals.sliver_triangles = int(metrics["skinny_triangles"])
        totals.max_aspect_ratio = float(metrics["max_aspect_ratio"])
        values["degenerate_triangles"] = totals.degenerate_triangles
        values["sliver_triangles"] = totals.sliver_triangles
        values["max_aspect_ratio"] = totals.max_aspect_ratio

    if include_tiny:
        diagonal = cast(float, bbox["diagonal"])
        tiny = bool(mesh.triangle_count > 0 and diagonal <= options.tiny_part_diagonal)
        if tiny:
            totals.tiny_parts = 1
            totals.tiny_part_triangles = mesh.triangle_count
        values["tiny"] = tiny
        values["tiny_part_diagonal_threshold"] = options.tiny_part_diagonal

    if include_self_intersections:
        result = _self_intersection_count(mesh, options.max_self_intersection_pairs)
        totals.self_intersection_warnings = result.intersections
        totals.self_intersections_lower_bound = result.truncated
        totals.self_intersection_pairs_checked = result.pairs_checked
        totals.self_intersection_pair_limit = result.pair_limit
        values["self_intersections"] = result.intersections
        values["self_intersection_warnings"] = result.intersections
        values["self_intersections_lower_bound"] = result.truncated
        values["self_intersection_pairs_checked"] = result.pairs_checked
        values["self_intersection_pair_limit"] = result.pair_limit
        if result.truncated:
            warnings.append(
                f"self-intersection check for part {part.id} reached "
                f"{result.pair_limit} triangle pairs; self_intersections={result.intersections} is a lower bound"
            )

    return _MeshQualityEntry(values=values, totals=totals, warnings=warnings)


def _bounds_payload(mesh: Mesh) -> dict[str, object]:
    mins, maxs = mesh.bounds()
    diagonal = float(np.linalg.norm(maxs - mins))
    return {
        "min": [float(value) for value in mins.tolist()],
        "max": [float(value) for value in maxs.tolist()],
        "diagonal": diagonal,
    }


def _open_boundary_count(mesh: Mesh) -> int:
    edges, counts = mesh._undirected_edges_and_counts()
    boundary_edges = edges[counts == 1].astype(int)
    if boundary_edges.size == 0:
        return 0

    adjacency: dict[int, set[int]] = {}
    unvisited = {(int(left), int(right)) for left, right in boundary_edges.tolist()}
    for left, right in unvisited:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)

    count = 0
    while unvisited:
        count += 1
        start = next(iter(unvisited))[0]
        stack = [start]
        seen_vertices: set[int] = set()
        while stack:
            vertex = stack.pop()
            if vertex in seen_vertices:
                continue
            seen_vertices.add(vertex)
            for neighbor in adjacency.get(vertex, set()):
                edge = (min(vertex, neighbor), max(vertex, neighbor))
                unvisited.discard(edge)
                if neighbor not in seen_vertices:
                    stack.append(neighbor)
    return count


def _self_intersection_count(mesh: Mesh, max_pairs: int) -> _SelfIntersectionResult:
    if mesh.triangle_count < 2:
        return _SelfIntersectionResult(intersections=0, truncated=False, pairs_checked=0, pair_limit=max_pairs)
    triangles = mesh.points[mesh.faces]
    mins = triangles.min(axis=1)
    maxs = triangles.max(axis=1)
    face_vertices = [set(face) for face in mesh.faces.astype(int).tolist()]
    intersections = 0
    checked = 0
    for left in range(mesh.triangle_count - 1):
        for right in range(left + 1, mesh.triangle_count):
            if face_vertices[left] & face_vertices[right]:
                continue
            if checked >= max_pairs:
                return _SelfIntersectionResult(
                    intersections=intersections,
                    truncated=True,
                    pairs_checked=checked,
                    pair_limit=max_pairs,
                )
            checked += 1
            if not bool(np.all(maxs[left] >= mins[right]) and np.all(maxs[right] >= mins[left])):
                continue
            if _triangles_intersect(triangles[left], triangles[right]):
                intersections += 1
    return _SelfIntersectionResult(
        intersections=intersections,
        truncated=False,
        pairs_checked=checked,
        pair_limit=max_pairs,
    )


def _triangles_intersect(left: FloatArray, right: FloatArray) -> bool:
    if _triangle_area(left) <= _INTERSECTION_EPSILON or _triangle_area(right) <= _INTERSECTION_EPSILON:
        return False
    if _triangles_coplanar(left, right):
        return _coplanar_triangles_intersect(left, right)
    for start, end in _triangle_edges(left):
        if _segment_intersects_triangle(start, end, right):
            return True
    return any(_segment_intersects_triangle(start, end, left) for start, end in _triangle_edges(right))


def _triangle_area(triangle: FloatArray) -> float:
    return float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) * 0.5)


def _triangles_coplanar(left: FloatArray, right: FloatArray) -> bool:
    normal = np.cross(left[1] - left[0], left[2] - left[0])
    length = float(np.linalg.norm(normal))
    if length <= _INTERSECTION_EPSILON:
        return False
    distances = (right - left[0]) @ (normal / length)
    return bool(np.all(np.abs(distances) <= _INTERSECTION_EPSILON))


def _triangle_edges(triangle: FloatArray) -> tuple[tuple[FloatArray, FloatArray], ...]:
    return ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0]))


def _segment_intersects_triangle(start: FloatArray, end: FloatArray, triangle: FloatArray) -> bool:
    direction = end - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) <= _INTERSECTION_EPSILON:
        return False
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < -_INTERSECTION_EPSILON or u > 1.0 + _INTERSECTION_EPSILON:
        return False
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < -_INTERSECTION_EPSILON or u + v > 1.0 + _INTERSECTION_EPSILON:
        return False
    t = float(np.dot(edge2, qvec)) * inverse
    return -_INTERSECTION_EPSILON <= t <= 1.0 + _INTERSECTION_EPSILON


def _coplanar_triangles_intersect(left: FloatArray, right: FloatArray) -> bool:
    normal = np.cross(left[1] - left[0], left[2] - left[0])
    axis = int(np.argmax(np.abs(normal)))
    left_2d = np.delete(left, axis, axis=1)
    right_2d = np.delete(right, axis, axis=1)
    return _convex_polygon_intersection_area(left_2d, right_2d) > _INTERSECTION_EPSILON


def _convex_polygon_intersection_area(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    clip = [np.asarray(point, dtype=np.float64) for point in right]
    if _polygon_area_2d(clip) < 0.0:
        clip.reverse()
    output = [np.asarray(point, dtype=np.float64) for point in left]
    if _polygon_area_2d(output) < 0.0:
        output.reverse()

    for edge_start, edge_end in zip(clip, clip[1:] + clip[:1], strict=True):
        if not output:
            return 0.0
        current_input = output
        output = []
        previous = current_input[-1]
        previous_inside = _inside_half_plane(previous, edge_start, edge_end)
        for current in current_input:
            current_inside = _inside_half_plane(current, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(_line_intersection_2d(previous, current, edge_start, edge_end))
                output.append(current)
            elif previous_inside:
                output.append(_line_intersection_2d(previous, current, edge_start, edge_end))
            previous = current
            previous_inside = current_inside

    return abs(_polygon_area_2d(output))


def _inside_half_plane(
    point: NDArray[np.float64],
    edge_start: NDArray[np.float64],
    edge_end: NDArray[np.float64],
) -> bool:
    return _cross_2d(edge_end - edge_start, point - edge_start) >= -_INTERSECTION_EPSILON


def _line_intersection_2d(
    left_start: NDArray[np.float64],
    left_end: NDArray[np.float64],
    right_start: NDArray[np.float64],
    right_end: NDArray[np.float64],
) -> NDArray[np.float64]:
    left_delta = left_end - left_start
    right_delta = right_end - right_start
    denominator = _cross_2d(left_delta, right_delta)
    if abs(denominator) <= _INTERSECTION_EPSILON:
        return cast(NDArray[np.float64], np.asarray(left_end, dtype=np.float64).copy())
    t = _cross_2d(right_start - left_start, right_delta) / denominator
    return left_start + t * left_delta


def _polygon_area_2d(points: list[NDArray[np.float64]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        area += _cross_2d(start, end)
    return area * 0.5


def _cross_2d(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    return float(left[0] * right[1] - left[1] * right[0])


def _visual_risk_warnings(asset: Asset, totals: _QualityTotals) -> list[str]:
    warnings: list[str] = []
    if totals.non_manifold_edges:
        warnings.append(
            f"non-manifold edges may create cracks or invalid collision meshes: {totals.non_manifold_edges}"
        )
    if totals.open_boundaries:
        warnings.append(f"open boundaries may reveal holes or missing shell faces: {totals.open_boundaries}")
    if totals.degenerate_triangles:
        warnings.append(f"degenerate triangles may cause renderer or physics artifacts: {totals.degenerate_triangles}")
    if totals.sliver_triangles:
        warnings.append(f"sliver triangles may produce shading and simplification artifacts: {totals.sliver_triangles}")
    if totals.self_intersection_warnings:
        warnings.append(f"self-intersections need visual inspection: {totals.self_intersection_warnings}")
    if totals.tiny_parts:
        warnings.append(f"tiny parts may disappear after optimization or LOD generation: {totals.tiny_parts}")

    for step in asset.report.steps:
        before_triangles = step.before.get("triangles", 0)
        after_triangles = step.after.get("triangles", before_triangles)
        if before_triangles > 0 and after_triangles < before_triangles:
            reduction = 1.0 - (after_triangles / before_triangles)
            if reduction >= 0.75:
                warnings.append(
                    f"{step.name} reduced triangles by {reduction:.0%}; inspect silhouette and small features"
                )
        before_materials = step.before.get("materials", 0)
        after_materials = step.after.get("materials", before_materials)
        if before_materials > 0 and after_materials < before_materials:
            warnings.append(f"{step.name} reduced material count from {before_materials} to {after_materials}")
        before_draws = step.before.get("draw_calls", 0)
        after_draws = step.after.get("draw_calls", before_draws)
        if before_draws > 0 and after_draws < before_draws:
            warnings.append(f"{step.name} reduced draw calls from {before_draws} to {after_draws}")
    return warnings


def _validate_output(path: Path) -> dict[str, int]:
    suffix = path.suffix.lower()
    if suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        from fascat.io.usd import validate_usd

        return validate_usd(path)
    if suffix in {".gltf", ".glb"}:
        from fascat.io.gltf import validate_gltf

        return validate_gltf(path)
    if suffix == ".obj":
        from fascat.io.obj import validate_obj

        return validate_obj(path)
    if suffix == ".stl":
        from fascat.io.stl import validate_stl

        return validate_stl(path)
    raise ValueError(f"unsupported export extension: {suffix or '<none>'}")


def _asset_from_output(path: Path) -> Asset:
    suffix = path.suffix.lower()
    if suffix in {".gltf", ".glb"}:
        return _asset_from_gltf(path)
    if suffix == ".obj":
        return _asset_from_obj(path)
    if suffix == ".stl":
        return _asset_from_stl(path)
    if suffix in {".usd", ".usda", ".usdc", ".usdz"}:
        return _asset_from_usd(path)
    raise ValueError(f"geometry quality analysis is not supported for {suffix or '<none>'}")


def _asset_from_gltf(path: Path) -> Asset:
    from fascat.io import gltf as gltf_io

    document, buffers = gltf_io._read_document(path)
    gltf_io._validate_buffers(document, buffers)
    parts: dict[str, Part] = {}
    nodes: list[Node] = []
    materials, material_ids_by_index = _gltf_materials(document)
    scenes = gltf_io._array(document.get("scenes"), "scenes")
    scene_index = gltf_io._int(document.get("scene", 0), "default scene")
    scene = gltf_io._object(scenes[scene_index], f"scene {scene_index}")
    for node_index in gltf_io._array(scene.get("nodes", []), "scene nodes"):
        _append_gltf_node_parts(
            document,
            buffers,
            gltf_io._int(node_index, "scene node"),
            parts,
            nodes,
            material_ids_by_index,
            transform=np.eye(4, dtype=np.float64),
            stack=set(),
        )
    return Asset(
        root=Node(id="root", name=scene.get("name", "Scene"), children=nodes), parts=parts, materials=materials
    )


def _append_gltf_node_parts(
    document: dict[str, Any],
    buffers: list[bytes],
    node_index: int,
    parts: dict[str, Part],
    nodes: list[Node],
    material_ids_by_index: dict[int, str],
    *,
    transform: NDArray[np.float64],
    stack: set[int],
) -> None:
    from fascat.io import gltf as gltf_io

    if node_index in stack:
        raise RuntimeError("glTF node hierarchy contains a cycle")
    node_values = gltf_io._array(document.get("nodes"), "nodes")
    node = gltf_io._object(node_values[node_index], f"node {node_index}")
    node_transform = transform @ _gltf_node_transform(node)
    mesh_index = node.get("mesh")
    if mesh_index is not None:
        part_node = _gltf_mesh_part(
            document,
            buffers,
            gltf_io._int(mesh_index, f"node {node_index} mesh"),
            parts,
            material_ids_by_index,
            node_name=str(node.get("name", f"Node {node_index}")),
            node_metadata=_fascat_metadata(node),
            node_transform=node_transform,
        )
        nodes.append(part_node)
    for child_index in gltf_io._array(node.get("children", []), f"node {node_index} children"):
        _append_gltf_node_parts(
            document,
            buffers,
            gltf_io._int(child_index, f"node {node_index} child"),
            parts,
            nodes,
            material_ids_by_index,
            transform=node_transform,
            stack=stack | {node_index},
        )


def _gltf_node_transform(node: dict[str, Any]) -> NDArray[np.float64]:
    matrix = node.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        matrix_values = np.asarray(matrix, dtype=np.float64).reshape((4, 4)).T
        return cast(NDArray[np.float64], matrix_values)
    translation_matrix = np.eye(4, dtype=np.float64)
    translation = node.get("translation")
    if isinstance(translation, list) and len(translation) == 3:
        translation_matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    rotation_matrix = np.eye(4, dtype=np.float64)
    rotation = node.get("rotation")
    if isinstance(rotation, list) and len(rotation) == 4:
        rotation_matrix[:3, :3] = _quaternion_matrix([float(value) for value in rotation])
    scale_matrix = np.eye(4, dtype=np.float64)
    scale = node.get("scale")
    if isinstance(scale, list) and len(scale) == 3:
        scale_matrix = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0])
    return translation_matrix @ rotation_matrix @ scale_matrix


def _quaternion_matrix(rotation: list[float]) -> NDArray[np.float64]:
    x, y, z, w = rotation
    length = float(np.linalg.norm(np.asarray([x, y, z, w], dtype=np.float64)))
    if length == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / length, y / length, z / length, w / length
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_points(points: FloatArray, matrix: NDArray[np.float64]) -> FloatArray:
    if not len(points):
        return points
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return cast(FloatArray, (matrix @ homogeneous.T).T[:, :3])


def _gltf_mesh_part(
    document: dict[str, Any],
    buffers: list[bytes],
    mesh_index: int,
    parts: dict[str, Part],
    material_ids_by_index: dict[int, str],
    *,
    node_name: str,
    node_metadata: dict[str, object],
    node_transform: NDArray[np.float64],
) -> Node:
    from fascat.io import gltf as gltf_io

    meshes = gltf_io._array(document.get("meshes"), "meshes")
    mesh = gltf_io._object(meshes[mesh_index], f"mesh {mesh_index}")
    mesh_metadata = _fascat_metadata(mesh)
    points_chunks: list[FloatArray] = []
    face_chunks: list[IntArray] = []
    face_material_indices: list[int] = []
    material_ids: list[str] = []
    material_slots: dict[str, int] = {}
    offset = 0
    for primitive_index, primitive_value in enumerate(
        gltf_io._array(mesh.get("primitives"), f"mesh {mesh_index} primitives")
    ):
        primitive = gltf_io._object(primitive_value, f"mesh {mesh_index} primitive {primitive_index}")
        attributes = gltf_io._object(primitive.get("attributes"), f"mesh {mesh_index} primitive attributes")
        position_index = gltf_io._int(attributes.get("POSITION"), f"mesh {mesh_index} POSITION accessor")
        points = _transform_points(_read_gltf_float_accessor(document, buffers, position_index), node_transform)
        indices = primitive.get("indices")
        if indices is None:
            faces = np.arange(points.shape[0], dtype=np.int64).reshape((-1, 3))
        else:
            index_values = _read_gltf_int_accessor(document, buffers, gltf_io._int(indices, "primitive indices"))
            faces = index_values.reshape((-1, 3))
        material_index = primitive.get("material")
        material_slot: int | None = None
        if isinstance(material_index, int):
            material_id = material_ids_by_index.get(material_index, f"material_{material_index}")
            if material_id not in material_slots:
                material_slots[material_id] = len(material_ids)
                material_ids.append(material_id)
            material_slot = material_slots[material_id]
        points_chunks.append(points)
        face_chunks.append(faces + offset)
        if material_slot is not None:
            face_material_indices.extend([material_slot] * len(faces))
        offset += points.shape[0]
    points = np.vstack(points_chunks) if points_chunks else np.empty((0, 3), dtype=np.float64)
    faces = np.vstack(face_chunks) if face_chunks else np.empty((0, 3), dtype=np.int64)
    material_indices = (
        np.asarray(face_material_indices, dtype=np.int64)
        if face_material_indices and len(face_material_indices) == len(faces)
        else None
    )
    source_part_id = _fascat_string(mesh, "partId")
    part_id = _unique_part_id(parts, source_part_id or f"mesh_{mesh_index}_{len(parts)}")
    part_name = _fascat_string(mesh, "originalName") or str(mesh.get("name", node_name))
    mesh_value = Mesh(points=points, faces=faces, material_indices=material_indices, metadata=dict(mesh_metadata))
    parts[part_id] = Part(
        id=part_id, name=part_name, mesh=mesh_value, material_ids=material_ids, metadata=mesh_metadata
    )
    return Node(id=f"node_{part_id}", name=node_name, part_id=part_id, metadata=node_metadata)


def _read_gltf_float_accessor(document: dict[str, Any], buffers: list[bytes], accessor_index: int) -> FloatArray:
    from fascat.io import gltf as gltf_io

    accessor = gltf_io._require_accessor(document, accessor_index)
    component_type = gltf_io._int(accessor.get("componentType"), f"accessor {accessor_index} componentType")
    values = _read_gltf_accessor(document, buffers, accessor_index).astype(np.float64)
    if accessor.get("normalized") is True:
        values = _normalize_gltf_accessor(values, component_type)
    return values


def _read_gltf_int_accessor(document: dict[str, Any], buffers: list[bytes], accessor_index: int) -> IntArray:
    return cast(IntArray, _read_gltf_accessor(document, buffers, accessor_index).astype(np.int64).reshape((-1,)))


def _read_gltf_accessor(document: dict[str, Any], buffers: list[bytes], accessor_index: int) -> NDArray[Any]:
    from fascat.io import gltf as gltf_io

    accessor = gltf_io._require_accessor(document, accessor_index)
    buffer_views = gltf_io._array(document.get("bufferViews"), "bufferViews")
    view_index = gltf_io._int(accessor.get("bufferView"), f"accessor {accessor_index} bufferView")
    view = gltf_io._object(buffer_views[view_index], f"bufferView {view_index}")
    component_type = gltf_io._int(accessor.get("componentType"), f"accessor {accessor_index} componentType")
    accessor_type = accessor.get("type")
    if not isinstance(accessor_type, str) or accessor_type not in _GLTF_ACCESSOR_WIDTHS:
        raise RuntimeError(f"glTF accessor {accessor_index} has unsupported type")
    if component_type not in _GLTF_DTYPES:
        raise RuntimeError(f"glTF accessor {accessor_index} has unsupported component type")
    count = gltf_io._int(accessor.get("count"), f"accessor {accessor_index} count")
    width = _GLTF_ACCESSOR_WIDTHS[accessor_type]
    dtype = _GLTF_DTYPES[component_type]
    item_size = dtype.itemsize * width
    buffer_index = gltf_io._int(view.get("buffer", 0), f"bufferView {view_index} buffer")
    offset = gltf_io._int(view.get("byteOffset", 0), f"bufferView {view_index} byteOffset") + gltf_io._int(
        accessor.get("byteOffset", 0), f"accessor {accessor_index} byteOffset"
    )
    stride = (
        gltf_io._int(view.get("byteStride", 0), f"bufferView {view_index} byteStride") if "byteStride" in view else 0
    )
    buffer = buffers[buffer_index]
    if not stride or stride == item_size:
        return cast(
            NDArray[Any],
            np.frombuffer(buffer, dtype=dtype, count=count * width, offset=offset).reshape((count, width)).copy(),
        )

    values = np.empty((count, width), dtype=dtype)
    for row in range(count):
        row_offset = offset + row * stride
        values[row] = np.frombuffer(buffer, dtype=dtype, count=width, offset=row_offset)
    return cast(NDArray[Any], values.copy())


def _normalize_gltf_accessor(values: NDArray[np.float64], component_type: int) -> NDArray[np.float64]:
    if component_type == 5120:
        return np.maximum(values / 127.0, -1.0)
    if component_type == 5121:
        return values / 255.0
    if component_type == 5122:
        return np.maximum(values / 32767.0, -1.0)
    if component_type == 5123:
        return values / 65535.0
    return values


def _gltf_materials(document: dict[str, Any]) -> tuple[dict[str, Material], dict[int, str]]:
    from fascat.io import gltf as gltf_io

    if "materials" not in document:
        return {}, {}
    result: dict[str, Material] = {}
    ids_by_index: dict[int, str] = {}
    for index, value in enumerate(gltf_io._array(document.get("materials"), "materials")):
        material = gltf_io._object(value, f"material {index}")
        name = str(material.get("name", f"Material {index}"))
        fascat = _fascat_extras(material)
        material_id = str(fascat.get("materialId") or f"material_{index}")
        base_color = (0.8, 0.8, 0.8, 1.0)
        pbr = material.get("pbrMetallicRoughness")
        if isinstance(pbr, dict) and isinstance(pbr.get("baseColorFactor"), list) and len(pbr["baseColorFactor"]) == 4:
            color = [float(item) for item in pbr["baseColorFactor"]]
            base_color = (color[0], color[1], color[2], color[3])
        metadata = fascat.get("metadata") if isinstance(fascat.get("metadata"), dict) else {}
        result[material_id] = Material(
            id=material_id,
            name=name,
            base_color=base_color,
            metadata=cast(dict[str, object], metadata),
        )
        ids_by_index[index] = material_id
    return result, ids_by_index


def _scoped_asset(asset: Asset, part_ids: set[str]) -> Asset:
    scoped = asset.copy(keep_source=True)
    scoped.parts = {part_id: part for part_id, part in scoped.parts.items() if part_id in part_ids}
    _prune_unselected_nodes(scoped.root, part_ids)
    return scoped


def _prune_unselected_nodes(node: Node, part_ids: set[str]) -> bool:
    kept: list[Node] = []
    for child in node.children:
        if _prune_unselected_nodes(child, part_ids):
            kept.append(child)
    node.children = kept
    return node.part_id in part_ids if node.part_id is not None else bool(node.children)


def _fascat_extras(value: dict[str, Any]) -> dict[str, object]:
    extras = value.get("extras")
    if not isinstance(extras, dict):
        return {}
    fascat = extras.get("fascat")
    return dict(fascat) if isinstance(fascat, dict) else {}


def _fascat_metadata(value: dict[str, Any]) -> dict[str, object]:
    fascat = _fascat_extras(value)
    metadata = fascat.get("metadata")
    return dict(metadata) if isinstance(metadata, dict) else {}


def _fascat_string(value: dict[str, Any], key: str) -> str | None:
    item = _fascat_extras(value).get(key)
    return item if isinstance(item, str) and item else None


def _unique_part_id(parts: dict[str, Part], candidate: str) -> str:
    if candidate not in parts:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in parts:
        suffix += 1
    return f"{candidate}_{suffix}"


def _asset_from_obj(path: Path) -> Asset:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    material_ids: list[str] = []
    material_lookup: dict[str, int] = {}
    material_indices: list[int] = []
    current_material: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("v "):
            values = stripped.split()
            vertices.append([float(values[1]), float(values[2]), float(values[3])])
        elif stripped.startswith("usemtl "):
            current_material = stripped.split(maxsplit=1)[1]
            if current_material not in material_lookup:
                material_lookup[current_material] = len(material_ids)
                material_ids.append(current_material)
        elif stripped.startswith("f "):
            indices = [_obj_vertex_index(token, len(vertices)) for token in stripped.split()[1:]]
            for index in range(1, len(indices) - 1):
                faces.append([indices[0], indices[index], indices[index + 1]])
                if current_material is not None:
                    material_indices.append(material_lookup[current_material])
    points = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    mesh_material_indices = (
        np.asarray(material_indices, dtype=np.int64)
        if material_indices and len(material_indices) == len(faces)
        else None
    )
    materials = {
        material_id: Material(id=material_id, name=material_id, base_color=(0.8, 0.8, 0.8, 1.0))
        for material_id in material_ids
    }
    part = Part(
        id="obj_mesh",
        name=path.stem,
        mesh=Mesh(points=points, faces=face_array, material_indices=mesh_material_indices),
        material_ids=material_ids,
    )
    return Asset(
        root=Node(id="root", name=path.stem, children=[Node(id="obj_mesh_node", name=path.stem, part_id="obj_mesh")]),
        parts={"obj_mesh": part},
        materials=materials,
    )


def _obj_vertex_index(token: str, vertex_count: int) -> int:
    value = int(token.split("/", 1)[0])
    return value - 1 if value > 0 else vertex_count + value


def _asset_from_stl(path: Path) -> Asset:
    payload = path.read_bytes()
    triangles: list[list[list[float]]]
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected = 84 + triangle_count * 50
        if expected == len(payload):
            triangles = []
            offset = 84
            for _index in range(triangle_count):
                offset += 12
                values = struct.unpack_from("<9f", payload, offset)
                offset += 38
                triangles.append(
                    [
                        [float(values[0]), float(values[1]), float(values[2])],
                        [float(values[3]), float(values[4]), float(values[5])],
                        [float(values[6]), float(values[7]), float(values[8])],
                    ]
                )
            return _asset_from_triangles(path.stem, triangles)

    vertices: list[list[float]] = []
    for line in payload.decode("utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex "):
            tokens = stripped.split()
            vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
    triangles = [
        vertices[index : index + 3] for index in range(0, len(vertices), 3) if len(vertices[index : index + 3]) == 3
    ]
    return _asset_from_triangles(path.stem, triangles)


def _asset_from_triangles(name: str, triangles: list[list[list[float]]]) -> Asset:
    points = np.asarray([point for triangle in triangles for point in triangle], dtype=np.float64)
    faces = np.arange(points.shape[0], dtype=np.int64).reshape((-1, 3))
    part = Part(id="stl_mesh", name=name, mesh=Mesh(points=points, faces=faces))
    return Asset(
        root=Node(id="root", name=name, children=[Node(id="stl_mesh_node", name=name, part_id="stl_mesh")]),
        parts={"stl_mesh": part},
    )


def _asset_from_usd(path: Path) -> Asset:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("USD geometry quality analysis requires usd-core") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("USD stage has no defaultPrim")

    parts: dict[str, Part] = {}
    nodes: list[Node] = []
    for prim in Usd.PrimRange(default_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        usd_mesh = UsdGeom.Mesh(prim)
        points_value = usd_mesh.GetPointsAttr().Get() or []
        counts = [int(value) for value in (usd_mesh.GetFaceVertexCountsAttr().Get() or [])]
        indices = [int(value) for value in (usd_mesh.GetFaceVertexIndicesAttr().Get() or [])]
        if any(count != 3 for count in counts):
            continue
        points = np.asarray(
            [[float(coord[0]), float(coord[1]), float(coord[2])] for coord in points_value], dtype=np.float64
        )
        faces = np.asarray(indices, dtype=np.int64).reshape((-1, 3))
        part_id = f"usd_mesh_{len(parts)}"
        name = prim.GetName() or part_id
        parts[part_id] = Part(id=part_id, name=name, mesh=Mesh(points=points, faces=faces))
        nodes.append(Node(id=f"node_{part_id}", name=name, part_id=part_id))

    if not parts:
        raise RuntimeError("USD geometry quality analysis found no mesh prims")
    return Asset(root=Node(id="root", name=default_prim.GetName(), children=nodes), parts=parts)


def _validation_only_report(
    path: Path,
    options: AnalyzeOptions,
    stats: dict[str, int],
    *,
    source_path: str | Path | None,
    warning: str,
) -> AnalysisReport:
    summary: dict[str, object] = {
        "validated_meshes": stats["meshes"],
        "validated_points": stats["points"],
        "validated_triangles": stats["triangles"],
    }
    if options.draw_call_estimate or options.visual_risk:
        summary["draw_call_estimate"] = None
        summary.update(
            {
                "draw_calls": None,
                "draw_call_meshes": None,
                "draw_call_materials": None,
                "draw_call_submesh_slots": None,
                "draw_call_material_slots": None,
                "draw_call_mesh_instances": None,
                "draw_call_reused_instances": None,
                "draw_call_instanced_meshes": None,
                "draw_call_merged_batches": None,
            }
        )
    return AnalysisReport(
        source_path=str(source_path or path),
        options=options.to_dict(),
        stats={
            "validated_meshes": stats["meshes"],
            "validated_points": stats["points"],
            "validated_triangles": stats["triangles"],
        },
        summary=summary,
        warnings=[f"geometry quality analysis used validation stats only: {warning}"],
    )


def _dedupe(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
