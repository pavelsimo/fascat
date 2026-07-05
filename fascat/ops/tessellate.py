from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from fascat._ocp import shape_fingerprint
from fascat.asset import Asset, Part
from fascat.errors import FascatError
from fascat.mesh import Mesh
from fascat.metadata import Metadata
from fascat.ops._arrays import array_digest_required
from fascat.options import TessellationOptions

_FACE_GROUP_RISK_THRESHOLD = 64
_DRAW_CALL_RISK_THRESHOLD = 16
_COARSE_ABSOLUTE_SAG_RATIO = 0.02
_AGGRESSIVE_MAX_LENGTH_RATIO = 0.02
_SUSPICIOUS_COARSE_SAG_METERS = 0.1
_SUSPICIOUS_FINE_TOLERANCE_METERS = 1e-9
_SUSPICIOUS_COARSE_MAX_LENGTH_METERS = 100.0
_LONG_OBJECT_AXIS_RATIO = 8.0
_SHINY_ROUGHNESS_THRESHOLD = 0.25
_METALLIC_DETAIL_ROUGHNESS_THRESHOLD = 0.5
_METALLIC_DETAIL_THRESHOLD = 0.5
_DETAIL_ADAPTIVE_SAG_RATIO = 0.01
_DETAIL_METADATA_KEYS = frozenset(
    {
        "critical_detail",
        "detail_level",
        "high_detail",
        "inspection_surface",
        "surface_detail",
        "tessellation_detail",
        "tessellation_priority",
        "visual_priority",
    }
)
_DETAIL_METADATA_VALUES = frozenset(
    {"1", "critical", "detailed", "fine", "high", "high_detail", "inspection", "true", "yes"}
)
_SHINY_MATERIAL_VALUES = frozenset({"chrome", "gloss", "glossy", "mirror", "polished", "shiny"})
_CONSTRUCTION_CURVE_TUBE_SIDES = 8
_CONSTRUCTION_CURVE_SAMPLE_SEGMENTS = 8
_CONSTRUCTION_CURVE_MIN_SEGMENT_LENGTH = 1e-12


@dataclass(frozen=True)
class _FaceTriangulation:
    face_index: int
    triangulation: Any
    transform: Any
    reversed_face: bool
    material_index: int | None
    point_offset: int
    triangle_offset: int
    node_count: int
    triangle_count: int


def tessellate_asset(asset: Asset, options: TessellationOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    mesh_by_source: dict[tuple[str, tuple[str, ...], tuple[int, ...] | None, tuple[object, ...]], Mesh] = {}
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part_options = _options_for_part(options, part, result)
        if part.mesh is not None and part_options.reuse_existing_meshes:
            _guard_tessellation_output(part, part_options)
            _record_detail_adaptive_selection(result, part, options, part_options)
            _record_tessellation_attribute_sources(result, part, part_options, geometry_source="imported_mesh")
            _record_tessellation_diagnostics(result, part, part_options)
            _record_brep_patch_cleanup(result, part, part_options)
            if not part_options.keep_brep:
                part.source_shape = None
            continue
        if part.source_shape is None:
            if part.mesh is None:
                result.report.add_warning(f"part has no source shape and cannot be tessellated: {part.name}")
            else:
                result.report.add_warning(
                    f"part has existing mesh but no source shape and cannot be retessellated: {part.name}"
                )
            continue
        construction_curve_policy = _part_construction_curve_policy(part)
        if construction_curve_policy == "preserve_metadata":
            part.metadata["tessellation_construction_curve_policy"] = "preserve_metadata"
            result.report.add_warning(
                f"construction curve part preserved as metadata without mesh geometry: {part.name}"
            )
            continue
        if construction_curve_policy == "tessellate_tubes":
            radius = _part_construction_curve_tube_radius(part)
            construction_cache_key = (
                shape_fingerprint(part.source_shape),
                tuple(part.material_ids),
                None,
                ("construction_curve_tubes", radius, part_options.create_normals),
            )
            cached_mesh = mesh_by_source.get(construction_cache_key)
            if cached_mesh is None:
                part.mesh = tessellate_construction_curve_shape(
                    part.source_shape,
                    radius=radius,
                    sides=_CONSTRUCTION_CURVE_TUBE_SIDES,
                    create_normals=part_options.create_normals,
                )
                if part.material_ids and part.mesh.material_indices is None and part.mesh.triangle_count:
                    part.mesh.material_indices = np.zeros(part.mesh.triangle_count, dtype=np.int64)
                part.mesh.validate()
                mesh_by_source[construction_cache_key] = part.mesh.copy()
            else:
                part.mesh = cached_mesh.copy()
            _guard_tessellation_output(part, part_options)
            if part.mesh.triangle_count == 0:
                result.report.add_warning(f"construction curve part did not produce tube mesh geometry: {part.name}")
            part.fingerprint = part.mesh.fingerprint()
            _record_detail_adaptive_selection(result, part, options, part_options)
            _record_tessellation_attribute_sources(result, part, part_options, geometry_source="tessellation")
            _record_tessellation_diagnostics(result, part, part_options)
            part.metadata["tessellation_construction_curve_policy"] = "tessellate_tubes"
            part.metadata["tessellation_construction_curve_tube_radius"] = _format_metadata_value(radius)
            part.metadata["tessellation_construction_curve_tube_sides"] = str(_CONSTRUCTION_CURVE_TUBE_SIDES)
            part.mesh.metadata["tessellation_construction_curve_policy"] = "tessellate_tubes"
            if not part_options.keep_brep:
                part.source_shape = None
            continue
        face_material_indices = _face_material_indices_from_metadata(part.metadata)
        cache_key = _tessellation_cache_key(
            part.source_shape,
            part.material_ids,
            face_material_indices,
            part_options,
        )
        cached_mesh = mesh_by_source.get(cache_key)
        if cached_mesh is None:
            part.mesh = tessellate_shape(
                part.source_shape,
                part_options,
                face_material_indices=face_material_indices,
            )
            if part.material_ids and part.mesh.material_indices is None:
                part.mesh.material_indices = np.zeros(part.mesh.triangle_count, dtype=np.int64)
            part.mesh.validate()
            mesh_by_source[cache_key] = part.mesh.copy()
        else:
            part.mesh = cached_mesh.copy()
        _guard_tessellation_output(part, part_options)
        part.fingerprint = part.mesh.fingerprint()
        _record_detail_adaptive_selection(result, part, options, part_options)
        _record_tessellation_attribute_sources(result, part, part_options, geometry_source="tessellation")
        _record_tessellation_diagnostics(result, part, part_options)
        _record_brep_patch_cleanup(result, part, part_options)
        if not part_options.keep_brep:
            part.source_shape = None
    if selected_part_ids is not None:
        return result
    return _deduplicate_parts_by_fingerprint(result)


def _guard_tessellation_output(part: Part, options: TessellationOptions) -> None:
    limit = options.max_triangles_per_part
    if limit is None or part.mesh is None or part.mesh.triangle_count <= limit:
        return
    raise FascatError(
        f"part {part.name!r} tessellated to {part.mesh.triangle_count} triangles "
        f"(limit {limit}); increase sag or raise max_triangles_per_part"
    )


def tessellate_shape(
    shape: object,
    options: TessellationOptions,
    *,
    face_material_indices: list[int] | None = None,
) -> Mesh:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.IMeshTools import IMeshTools_Parameters
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise RuntimeError("STEP tessellation requires cadquery-ocp") from exc

    brep_shape = shape
    parameters = _occt_mesh_parameters(options, IMeshTools_Parameters)
    mesher = BRepMesh_IncrementalMesh(brep_shape, parameters)
    mesher.Perform()

    face_triangulations: list[_FaceTriangulation] = []
    point_count = 0
    triangle_count = 0
    face_index = 0
    explorer = TopExp_Explorer(brep_shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        material_index = None
        if face_material_indices is not None and face_index < len(face_material_indices):
            material_index = face_material_indices[face_index]
        if triangulation is not None:
            transform = location.Transformation()
            node_count = int(triangulation.NbNodes())
            face_triangle_count = int(triangulation.NbTriangles())
            face_triangulations.append(
                _FaceTriangulation(
                    face_index=face_index,
                    triangulation=triangulation,
                    transform=transform,
                    reversed_face=face.Orientation() == TopAbs_REVERSED,
                    material_index=material_index,
                    point_offset=point_count,
                    triangle_offset=triangle_count,
                    node_count=node_count,
                    triangle_count=face_triangle_count,
                )
            )
            point_count += node_count
            triangle_count += face_triangle_count
        face_index += 1
        explorer.Next()

    points = np.empty((point_count, 3), dtype=np.float64)
    faces = np.empty((triangle_count, 3), dtype=np.int64)
    cad_uv_values = np.full((point_count, 2), np.nan, dtype=np.float64) if options.cad_uvs else None
    material_values = None if face_material_indices is None else np.full(triangle_count, -1, dtype=np.int64)
    face_groups: dict[str, np.ndarray] = {}
    for payload in face_triangulations:
        nodes = payload.triangulation.MapNodeArray()
        node_lower = int(nodes.Lower())
        points[payload.point_offset : payload.point_offset + payload.node_count] = _transformed_occt_nodes(
            nodes,
            node_lower,
            payload.node_count,
            payload.transform,
        )
        if cad_uv_values is not None:
            surface_uvs = _triangulation_uv_nodes(payload.triangulation, payload.node_count)
            if surface_uvs is not None:
                cad_uv_values[payload.point_offset : payload.point_offset + payload.node_count] = surface_uvs

        triangles = payload.triangulation.MapTriangleArray()
        triangle_lower = int(triangles.Lower())
        face_group = np.arange(
            payload.triangle_offset,
            payload.triangle_offset + payload.triangle_count,
            dtype=np.int64,
        )
        if face_group.size:
            face_groups[f"occt_face_{payload.face_index}"] = face_group
        local_faces = _triangulation_faces(triangles, triangle_lower, payload.triangle_count)
        local_faces += payload.point_offset - 1
        if payload.reversed_face:
            local_faces = local_faces[:, ::-1]
        faces[payload.triangle_offset : payload.triangle_offset + payload.triangle_count] = local_faces
        if material_values is not None and payload.material_index is not None:
            material_values[payload.triangle_offset : payload.triangle_offset + payload.triangle_count] = (
                payload.material_index
            )

    mesh = Mesh(
        points=points,
        faces=faces,
        material_indices=(
            material_values
            if material_values is not None and material_values.size > 0 and bool(np.all(material_values >= 0))
            else None
        ),
        face_groups=face_groups,
        metadata={"occt_faces": str(face_index)},
    )
    mesh = _apply_mesh_tessellation_controls(mesh, options)
    if options.cad_uvs:
        mesh = _apply_cad_uvs(mesh, cad_uv_values)
    if options.create_normals:
        mesh = mesh.compute_normals()
    else:
        mesh.normals = None
    if options.tessellate_tangents:
        mesh = mesh.compute_tangents(0)
        mesh.metadata["tessellation_tangents"] = "generated" if mesh.tangents is not None else "missing_uv0"
    if options.free_edge_geometry:
        mesh = _store_free_edge_geometry(mesh)
    mesh.validate()
    return mesh


def _transformed_occt_nodes(nodes: Any, node_lower: int, node_count: int, transform: Any) -> np.ndarray:
    raw_points = np.empty((node_count, 3), dtype=np.float64)
    for local_index in range(node_count):
        point = nodes.Value(node_lower + local_index)
        raw_points[local_index] = (float(point.X()), float(point.Y()), float(point.Z()))
    matrix = _occt_transform_matrix(transform)
    return cast(np.ndarray, raw_points @ matrix[:, :3].T + matrix[:, 3])


def _occt_transform_matrix(transform: Any) -> np.ndarray:
    return np.asarray(
        [
            [transform.Value(1, 1), transform.Value(1, 2), transform.Value(1, 3), transform.Value(1, 4)],
            [transform.Value(2, 1), transform.Value(2, 2), transform.Value(2, 3), transform.Value(2, 4)],
            [transform.Value(3, 1), transform.Value(3, 2), transform.Value(3, 3), transform.Value(3, 4)],
        ],
        dtype=np.float64,
    )


def _triangulation_faces(triangles: Any, triangle_lower: int, triangle_count: int) -> np.ndarray:
    faces = np.empty((triangle_count, 3), dtype=np.int64)
    for local_index in range(triangle_count):
        faces[local_index] = triangles.Value(triangle_lower + local_index).Get()
    return faces


def tessellate_construction_curve_shape(
    shape: object,
    *,
    radius: float,
    sides: int = _CONSTRUCTION_CURVE_TUBE_SIDES,
    create_normals: bool = True,
) -> Mesh:
    segments = _construction_curve_segments(shape)
    mesh = _tube_mesh_from_segments(segments, radius=radius, sides=sides)
    if create_normals:
        mesh = mesh.compute_normals()
    mesh.validate()
    return mesh


def _construction_curve_segments(shape: object) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.TopAbs import TopAbs_EDGE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise RuntimeError("STEP construction curve tessellation requires cadquery-ocp") from exc

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        first, last = BRep_Tool.Range_s(edge)
        curve = BRep_Tool.Curve_s(edge, first, last)
        if curve is None:
            explorer.Next()
            continue
        samples = [
            _occt_curve_point(curve, first + (last - first) * index / _CONSTRUCTION_CURVE_SAMPLE_SEGMENTS)
            for index in range(_CONSTRUCTION_CURVE_SAMPLE_SEGMENTS + 1)
        ]
        samples_array = np.asarray(samples, dtype=np.float64)
        lengths = np.linalg.norm(np.diff(samples_array, axis=0), axis=1)
        keep = lengths > _CONSTRUCTION_CURVE_MIN_SEGMENT_LENGTH
        segments.extend(zip(samples_array[:-1][keep], samples_array[1:][keep], strict=True))
        explorer.Next()
    return segments


def _occt_curve_point(curve: object, parameter: float) -> np.ndarray:
    point = cast(Any, curve).Value(float(parameter))
    return np.asarray([point.X(), point.Y(), point.Z()], dtype=np.float64)


def _tube_mesh_from_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
    *,
    radius: float,
    sides: int,
) -> Mesh:
    if radius <= 0.0:
        raise ValueError("construction curve tube radius must be greater than 0")
    if sides < 3:
        raise ValueError("construction curve tube sides must be at least 3")

    side_indices = np.arange(sides, dtype=np.int64)
    next_side_indices = (side_indices + 1) % sides
    face_template = np.empty((sides, 4, 3), dtype=np.int64)
    face_template[:, 0, :] = np.column_stack([side_indices, next_side_indices, sides + next_side_indices])
    face_template[:, 1, :] = np.column_stack([side_indices, sides + next_side_indices, sides + side_indices])
    face_template[:, 2, :] = np.column_stack([np.full(sides, 2 * sides), side_indices, next_side_indices])
    face_template[:, 3, :] = np.column_stack(
        [np.full(sides, 2 * sides + 1), sides + next_side_indices, sides + side_indices]
    )
    face_template = face_template.reshape((-1, 3))
    angles = 2.0 * math.pi * side_indices.astype(np.float64) / sides
    cos_angles = np.cos(angles)[:, None]
    sin_angles = np.sin(angles)[:, None]

    point_blocks: list[np.ndarray] = []
    face_blocks: list[np.ndarray] = []
    face_groups: dict[str, np.ndarray] = {}
    point_offset = 0
    face_offset = 0
    for segment_index, (start, end) in enumerate(segments):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length <= _CONSTRUCTION_CURVE_MIN_SEGMENT_LENGTH:
            continue
        direction = direction / length
        u_axis, v_axis = _tube_frame(direction)
        offsets = radius * (cos_angles * u_axis + sin_angles * v_axis)
        point_blocks.append(
            np.concatenate(
                [
                    start + offsets,
                    end + offsets,
                    np.asarray(start, dtype=np.float64).reshape((1, 3)),
                    np.asarray(end, dtype=np.float64).reshape((1, 3)),
                ],
                axis=0,
            )
        )
        face_blocks.append(face_template + point_offset)
        face_groups[f"construction_curve_segment_{segment_index}"] = np.arange(
            face_offset,
            face_offset + face_template.shape[0],
            dtype=np.int64,
        )
        point_offset += 2 * sides + 2
        face_offset += face_template.shape[0]

    points_array = np.vstack(point_blocks) if point_blocks else np.empty((0, 3), dtype=np.float64)
    faces_array = np.vstack(face_blocks) if face_blocks else np.empty((0, 3), dtype=np.int64)
    return Mesh(
        points=points_array,
        faces=faces_array,
        face_groups=face_groups,
        metadata={
            "construction_curve_tube_segments": str(len(segments)),
            "construction_curve_tube_radius": _format_metadata_value(radius),
            "construction_curve_tube_sides": str(sides),
        },
    )


def _tube_frame(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(direction, helper))) > 0.9:
        helper = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    u_axis = np.cross(direction, helper)
    u_axis = u_axis / float(np.linalg.norm(u_axis))
    v_axis = np.cross(direction, u_axis)
    v_axis = v_axis / float(np.linalg.norm(v_axis))
    return u_axis, v_axis


def _triangulation_uv_nodes(triangulation: Any, node_count: int) -> np.ndarray | None:
    try:
        has_uv_nodes = triangulation.HasUVNodes()
    except Exception:
        return None
    if not has_uv_nodes:
        return None
    try:
        uv_nodes = triangulation.MapUVNodeArray()
        lower = int(uv_nodes.Lower())
        values = np.empty((node_count, 2), dtype=np.float64)
        for local_index in range(node_count):
            point = uv_nodes.Value(lower + local_index)
            values[local_index] = (float(point.X()), float(point.Y()))
        return values
    except Exception:
        return None


def _apply_cad_uvs(mesh: Mesh, cad_uv_values: np.ndarray | None) -> Mesh:
    if cad_uv_values is not None and cad_uv_values.shape == (mesh.vertex_count, 2) and np.isfinite(cad_uv_values).all():
        result = mesh.copy()
        result.uvs[0] = _normalize_uv_values(cad_uv_values)
        result.tangents = None
        result.metadata = {**result.metadata, "uv0": "cad_surface_uv", "uv0_source": "occt_surface_parameters"}
        return result
    return mesh.cad_project_uvs(0)


def _normalize_uv_values(values: np.ndarray) -> np.ndarray:
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    span = maxs - mins
    span[span == 0.0] = 1.0
    return cast(np.ndarray, ((values - mins) / span).astype(np.float64))


def _store_free_edge_geometry(mesh: Mesh) -> Mesh:
    edges, counts = mesh._undirected_edges_and_counts()
    boundary_edges = edges[counts == 1].astype(np.int64, copy=False)
    max_segments = 1024
    clipped_edges = boundary_edges[:max_segments]
    segments = np.stack([mesh.points[clipped_edges[:, 0]], mesh.points[clipped_edges[:, 1]]], axis=1).tolist()
    result = mesh.copy()
    result.metadata = {
        **result.metadata,
        "tessellation_free_edge_geometry": "stored",
        "tessellation_free_edge_segment_count": str(int(boundary_edges.shape[0])),
        "tessellation_free_edge_segments": json.dumps(segments, sort_keys=True),
        "tessellation_free_edge_segments_truncated": str(boundary_edges.shape[0] > max_segments).lower(),
    }
    return result


def _face_material_indices_from_metadata(metadata: Metadata) -> list[int] | None:
    value = metadata.get("occt_face_material_indices")
    if not value:
        return None
    return [int(item) for item in str(value).split(",") if item]


def _part_construction_curve_policy(part: Part) -> str | None:
    if part.metadata.get("loaded_representation") != "construction_lines":
        return None
    return str(part.metadata.get("construction_curve_policy", "preserve_metadata"))


def _part_construction_curve_tube_radius(part: Part) -> float:
    value = part.metadata.get("construction_curve_tube_radius", 0.01)
    if isinstance(value, (int, float)):
        radius = float(value)
    else:
        try:
            radius = float(str(value))
        except ValueError:
            radius = 0.01
    return radius if radius > 0.0 else 0.01


def _options_for_part(options: TessellationOptions, part: Part, asset: Asset) -> TessellationOptions:
    overrides = options.part_settings.get(part.id) or options.part_settings.get(part.name)
    detail_contexts = _adaptive_detail_contexts(asset, part) if options.detail_adaptive else []
    if not overrides and not detail_contexts:
        return options
    values = options.to_dict()
    values["part_settings"] = {}
    if detail_contexts:
        if values["sag_ratio"] is None:
            values["sag_ratio"] = _DETAIL_ADAPTIVE_SAG_RATIO
        values["curvature_adaptive"] = True
    if overrides:
        values.update(overrides)
    return TessellationOptions(**cast(Any, values))


def _record_detail_adaptive_selection(
    asset: Asset,
    part: Part,
    requested: TessellationOptions,
    effective: TessellationOptions,
) -> None:
    if not requested.detail_adaptive:
        return
    contexts = _adaptive_detail_contexts(asset, part)
    state = "not_applicable"
    if contexts:
        state = "applied" if effective.sag_ratio is not None or effective.curvature_adaptive else "overridden"
    metadata = {
        "tessellation_detail_adaptive": state,
        "tessellation_detail_contexts": ",".join(contexts),
        "tessellation_detail_adaptive_curvature": str(effective.curvature_adaptive).lower(),
    }
    if effective.sag_ratio is not None:
        metadata["tessellation_detail_adaptive_sag_ratio"] = _format_metadata_value(effective.sag_ratio)
    part.metadata.update(metadata)
    if part.mesh is not None:
        part.mesh.metadata.update(metadata)


def _occt_mesh_parameters(options: TessellationOptions, parameters_factory: Any) -> Any:
    deflection, relative = _deflection_settings(options)
    parameters = parameters_factory()
    parameters.Deflection = deflection
    parameters.Relative = relative
    parameters.Angle = math.radians(float(options.angle))
    parameters.InParallel = True
    parameters.InternalVerticesMode = bool(options.preserve_boundaries)
    if options.min_edge_length is not None:
        parameters.MinSize = float(options.min_edge_length)
        parameters.AdjustMinSize = True
    if options.curvature_adaptive:
        parameters.ControlSurfaceDeflection = True
        parameters.ForceFaceDeflection = True
        parameters.DeflectionInterior = deflection * 0.5
        parameters.AngleInterior = math.radians(max(0.1, float(options.angle) * 0.5))
    return parameters


def _deflection_settings(options: TessellationOptions) -> tuple[float, bool]:
    if options.sag_ratio is not None:
        return float(options.sag_ratio), True
    if options.sag is None:
        raise ValueError("tessellation sag or sag_ratio must be set")
    return float(options.sag), bool(options.relative)


def _apply_mesh_tessellation_controls(mesh: Mesh, options: TessellationOptions) -> Mesh:
    result = mesh
    edge_control_changed = False
    if options.max_edge_length is not None:
        before_counts = (result.vertex_count, result.triangle_count)
        result = result.subdivide_long_edges(options.max_edge_length)
        edge_control_changed = edge_control_changed or (result.vertex_count, result.triangle_count) != before_counts
    if options.min_edge_length is not None:
        before_counts = (result.vertex_count, result.triangle_count)
        result = result.collapse_short_edges(
            options.min_edge_length,
            preserve_boundaries=options.preserve_boundaries,
        )
        edge_control_changed = edge_control_changed or (result.vertex_count, result.triangle_count) != before_counts
    if options.avoid_skinny_triangles:
        before_counts = (result.vertex_count, result.triangle_count)
        result = result.improve_skinny_triangles(preserve_boundaries=options.preserve_boundaries)
        edge_control_changed = edge_control_changed or (result.vertex_count, result.triangle_count) != before_counts
    if edge_control_changed:
        if options.max_edge_length is not None:
            result = result.subdivide_long_edges(options.max_edge_length)
        if options.min_edge_length is not None:
            result = result.collapse_short_edges(
                options.min_edge_length,
                preserve_boundaries=options.preserve_boundaries,
            )
    result.metadata = {
        **result.metadata,
        "tessellation_feature_aware": str(options.curvature_adaptive or options.preserve_boundaries).lower(),
        "preserve_boundaries": str(options.preserve_boundaries).lower(),
        "tessellation_edge_control_passes": "2" if edge_control_changed else "1",
    }
    return result


def _record_tessellation_diagnostics(asset: Asset, part: Part, options: TessellationOptions) -> None:
    if part.mesh is None:
        return
    _record_tessellation_tolerance_policy(asset, part, options)
    _record_submesh_risk(asset, part)
    metrics: dict[str, int | float] | None = None
    if (options.free_edge_report or options.max_polygon_length is not None) and metrics is None:
        metrics = part.mesh.quality_metrics(
            min_edge_length=options.min_edge_length,
            max_edge_length=_quality_max_edge_length(options),
        )
    advisories = _tessellation_quality_advisories(asset, part, options)
    if options.quality_report:
        if metrics is None:
            metrics = part.mesh.quality_metrics(
                min_edge_length=options.min_edge_length,
                max_edge_length=_quality_max_edge_length(options),
            )
        _store_quality_report(part, options, metrics, advisories)
    _store_tessellation_quality_advisories(asset, part, advisories)
    if options.free_edge_report:
        _store_free_edge_report(asset, part, metrics)
    if options.max_polygon_length is not None:
        _warn_long_polygons(asset, part, options, metrics)


def tessellation_tolerance_policy(asset: Asset, options: TessellationOptions) -> dict[str, object]:
    source_units = _metadata_str(asset.metadata.get("source_units"), asset.units)
    source_meters_per_unit = _metadata_float(asset.metadata.get("source_meters_per_unit"), asset.meters_per_unit)
    target_meters_per_unit = asset.meters_per_unit
    coordinate_space = (
        "source_local"
        if source_units != asset.units or not np.isclose(source_meters_per_unit, asset.meters_per_unit)
        else "asset"
    )
    active_deflection, active_relative = _deflection_settings(options)
    active_kind = _active_deflection_kind(options)
    policy: dict[str, object] = {
        "coordinate_space": coordinate_space,
        "effective_units": source_units,
        "effective_meters_per_unit": source_meters_per_unit,
        "source_units": source_units,
        "source_meters_per_unit": source_meters_per_unit,
        "target_units": asset.units,
        "target_meters_per_unit": target_meters_per_unit,
        "angle_degrees": float(options.angle),
        "active_deflection": active_deflection,
        "active_deflection_relative": active_relative,
        "active_deflection_kind": active_kind,
        "relative": bool(options.relative),
        "sag": options.sag,
        "sag_ratio": options.sag_ratio,
        "curvature_adaptive": bool(options.curvature_adaptive),
        "detail_adaptive": bool(options.detail_adaptive),
        "preserve_boundaries": bool(options.preserve_boundaries),
    }
    if active_kind == "absolute_sag" and options.sag is not None:
        _add_length_policy_fields(policy, "sag", float(options.sag), source_meters_per_unit, target_meters_per_unit)
    for key, value in (
        ("min_edge_length", options.min_edge_length),
        ("max_edge_length", options.max_edge_length),
        ("max_polygon_length", options.max_polygon_length),
    ):
        if value is not None:
            _add_length_policy_fields(policy, key, float(value), source_meters_per_unit, target_meters_per_unit)
    return policy


def _record_tessellation_tolerance_policy(asset: Asset, part: Part, options: TessellationOptions) -> None:
    policy = tessellation_tolerance_policy(asset, options)
    advisories = _tessellation_tolerance_policy_advisories(part, policy)
    if advisories:
        policy["advisories"] = advisories
        part.metadata["tessellation_tolerance_advisory_count"] = str(len(advisories))
        part.metadata["tessellation_tolerance_advisory_codes"] = ",".join(
            str(advisory["code"]) for advisory in advisories
        )
        if part.mesh is not None:
            part.mesh.metadata["tessellation_tolerance_advisory_count"] = str(len(advisories))
            part.mesh.metadata["tessellation_tolerance_advisory_codes"] = part.metadata[
                "tessellation_tolerance_advisory_codes"
            ]
        for advisory in advisories:
            asset.report.add_warning(str(advisory["message"]))
    encoded = json.dumps(policy, sort_keys=True)
    metadata = _tessellation_tolerance_policy_metadata(policy)
    part.metadata["tessellation_tolerance_policy"] = encoded
    part.metadata.update(metadata)
    if part.mesh is not None:
        part.mesh.metadata["tessellation_tolerance_policy"] = encoded
        part.mesh.metadata.update(metadata)


def _add_length_policy_fields(
    policy: dict[str, object],
    key: str,
    value: float,
    source_meters_per_unit: float,
    target_meters_per_unit: float,
) -> None:
    value_meters = value * source_meters_per_unit
    policy[key] = value
    policy[f"{key}_meters"] = value_meters
    policy[f"{key}_target_units"] = value_meters / target_meters_per_unit if target_meters_per_unit > 0.0 else value


def _active_deflection_kind(options: TessellationOptions) -> str:
    if options.sag_ratio is not None:
        return "sag_ratio"
    return "relative_sag" if options.relative else "absolute_sag"


def _tessellation_tolerance_policy_advisories(part: Part, policy: dict[str, object]) -> list[dict[str, object]]:
    if policy.get("coordinate_space") != "source_local":
        return []
    advisories: list[dict[str, object]] = []
    if policy.get("active_deflection_kind") == "absolute_sag":
        sag_meters = _policy_float(policy.get("sag_meters"))
        if sag_meters is not None:
            if sag_meters >= _SUSPICIOUS_COARSE_SAG_METERS:
                advisories.append(
                    _tessellation_tolerance_advisory(
                        part,
                        policy,
                        code="coarse_normalized_sag",
                        key="sag",
                        message=(
                            "tessellation sag converts to a very large target-space tolerance after unit "
                            f"normalization; verify sag is specified in source/local units: {part.name}"
                        ),
                    )
                )
            elif 0.0 < sag_meters <= _SUSPICIOUS_FINE_TOLERANCE_METERS:
                advisories.append(
                    _tessellation_tolerance_advisory(
                        part,
                        policy,
                        code="fine_normalized_sag",
                        key="sag",
                        message=(
                            "tessellation sag converts to a sub-nanometer target-space tolerance after unit "
                            f"normalization; verify sag is not accidentally specified in target units: {part.name}"
                        ),
                    )
                )
    for key in ("min_edge_length", "max_edge_length", "max_polygon_length"):
        value_meters = _policy_float(policy.get(f"{key}_meters"))
        if value_meters is None or value_meters <= 0.0:
            continue
        if value_meters <= _SUSPICIOUS_FINE_TOLERANCE_METERS:
            advisories.append(
                _tessellation_tolerance_advisory(
                    part,
                    policy,
                    code=f"fine_normalized_{key}",
                    key=key,
                    message=(
                        f"tessellation {key} converts to a sub-nanometer target-space length after unit "
                        f"normalization; verify it is specified in source/local units: {part.name}"
                    ),
                )
            )
        elif key in {"max_edge_length", "max_polygon_length"} and value_meters >= _SUSPICIOUS_COARSE_MAX_LENGTH_METERS:
            advisories.append(
                _tessellation_tolerance_advisory(
                    part,
                    policy,
                    code=f"coarse_normalized_{key}",
                    key=key,
                    message=(
                        f"tessellation {key} converts to a very large target-space length after unit "
                        f"normalization; verify it is specified in source/local units: {part.name}"
                    ),
                )
            )
    return advisories


def _tessellation_tolerance_advisory(
    part: Part,
    policy: dict[str, object],
    *,
    code: str,
    key: str,
    message: str,
) -> dict[str, object]:
    return {
        "code": code,
        "severity": "warning",
        "part_id": part.id,
        "part_name": part.name,
        "parameter": key,
        "value": policy.get(key),
        "value_meters": policy.get(f"{key}_meters"),
        "value_target_units": policy.get(f"{key}_target_units"),
        "source_units": policy["source_units"],
        "target_units": policy["target_units"],
        "message": message,
    }


def _tessellation_tolerance_policy_metadata(policy: dict[str, object]) -> dict[str, str]:
    keys = (
        "coordinate_space",
        "effective_units",
        "effective_meters_per_unit",
        "source_units",
        "source_meters_per_unit",
        "target_units",
        "target_meters_per_unit",
        "active_deflection_kind",
        "active_deflection",
        "active_deflection_relative",
        "sag",
        "sag_ratio",
        "angle_degrees",
        "curvature_adaptive",
        "preserve_boundaries",
        "min_edge_length",
        "max_edge_length",
        "max_polygon_length",
        "sag_meters",
        "sag_target_units",
        "min_edge_length_meters",
        "min_edge_length_target_units",
        "max_edge_length_meters",
        "max_edge_length_target_units",
        "max_polygon_length_meters",
        "max_polygon_length_target_units",
    )
    metadata: dict[str, str] = {}
    for key in keys:
        value = policy.get(key)
        if value is None:
            continue
        metadata[f"tessellation_{key}"] = _format_metadata_value(value)
    return metadata


def _policy_float(value: object) -> float | None:
    return value if isinstance(value, float) else None


def _record_tessellation_attribute_sources(
    asset: Asset,
    part: Part,
    options: TessellationOptions,
    *,
    geometry_source: str,
) -> None:
    mesh = part.mesh
    if mesh is None:
        return
    sources: dict[str, object] = {
        "positions": geometry_source,
        "triangles": geometry_source,
        "normals": _normal_attribute_source(mesh, options, geometry_source),
        "tangents": _tangent_attribute_source(mesh, options, geometry_source),
        "uvs": _uv_attribute_sources(mesh, options, geometry_source),
        "face_groups": _face_group_attribute_source(mesh, geometry_source),
        "free_edges": _free_edge_attribute_source(options),
        "brep_patches": _brep_patch_attribute_source(part, options, geometry_source),
    }
    encoded = json.dumps(sources, sort_keys=True)
    part.metadata["tessellation_attribute_sources"] = encoded
    mesh.metadata["tessellation_attribute_sources"] = encoded


def _normal_attribute_source(mesh: Mesh, options: TessellationOptions, geometry_source: str) -> str:
    if geometry_source == "tessellation":
        if options.create_normals and mesh.normals is not None:
            return "tessellation"
        if not options.create_normals:
            return "disabled"
        return "missing"
    return "imported_mesh" if mesh.normals is not None else "missing"


def _tangent_attribute_source(mesh: Mesh, options: TessellationOptions, geometry_source: str) -> str:
    if mesh.tangents is not None:
        if geometry_source == "tessellation" and options.tessellate_tangents:
            return "tessellation"
        return geometry_source
    if options.tessellate_tangents:
        return "requested_missing_uvs"
    return "not_generated_by_tessellation" if geometry_source == "tessellation" else "missing"


def _uv_attribute_sources(mesh: Mesh, options: TessellationOptions, geometry_source: str) -> dict[str, str]:
    if not mesh.uvs:
        return {"status": "not_generated_by_tessellation" if geometry_source == "tessellation" else "missing"}
    if geometry_source == "tessellation" and options.cad_uvs:
        return {str(channel): str(mesh.metadata.get(f"uv{channel}", "cad_projected")) for channel in sorted(mesh.uvs)}
    source = "tessellation" if geometry_source == "tessellation" else "imported_mesh"
    return {str(channel): source for channel in sorted(mesh.uvs)}


def _free_edge_attribute_source(options: TessellationOptions) -> str:
    if options.free_edge_geometry:
        return "geometry_output"
    if options.free_edge_report:
        return "diagnostic_only"
    return "not_requested"


def _face_group_attribute_source(mesh: Mesh, geometry_source: str) -> str:
    if not mesh.face_groups:
        return "missing"
    return "cad_face_groups" if geometry_source == "tessellation" else "imported_mesh"


def _brep_patch_attribute_source(part: Part, options: TessellationOptions, geometry_source: str) -> str:
    if part.source_shape is None:
        return "not_available"
    return "retained" if options.keep_brep else "deleted"


def _store_quality_report(
    part: Part,
    options: TessellationOptions,
    metrics: dict[str, int | float],
    advisories: list[dict[str, object]],
) -> None:
    if part.mesh is None:
        return
    payload = {
        "part_id": part.id,
        "part_name": part.name,
        "options": _tessellation_mesh_options(options),
        "metrics": metrics,
        "advisories": advisories,
    }
    encoded = json.dumps(payload, sort_keys=True)
    part.metadata["tessellation_quality"] = encoded
    part.metadata["tessellation_short_edges"] = str(metrics["short_edges"])
    part.metadata["tessellation_long_edges"] = str(metrics["long_edges"])
    part.metadata["tessellation_skinny_triangles"] = str(metrics["skinny_triangles"])
    part.mesh.metadata["tessellation_quality"] = encoded


def _quality_max_edge_length(options: TessellationOptions) -> float | None:
    return options.max_polygon_length if options.max_polygon_length is not None else options.max_edge_length


def _tessellation_quality_advisories(asset: Asset, part: Part, options: TessellationOptions) -> list[dict[str, object]]:
    mesh = part.mesh
    if mesh is None:
        return []
    mins, maxs = mesh.bounds()
    extents = maxs - mins
    diagonal = float(np.linalg.norm(extents))
    if diagonal <= 0.0:
        return []

    advisories: list[dict[str, object]] = []
    detail_contexts = _adaptive_detail_contexts(asset, part)
    if detail_contexts and options.sag_ratio is None and not options.curvature_adaptive:
        advisories.append(
            {
                "code": "detail_sensitive_tessellation",
                "severity": "warning",
                "part_id": part.id,
                "part_name": part.name,
                "detail_contexts": detail_contexts,
                "recommendation": "set per-part sag_ratio or enable curvature_adaptive for this part",
                "message": (
                    "part has shiny, high-detail, or curved BREP context but tessellation uses bulk criteria "
                    f"without sag_ratio or curvature_adaptive; consider finer per-part tessellation: {part.name}"
                ),
            }
        )

    if options.sag is not None and options.sag_ratio is None and not options.relative:
        sag_ratio = float(options.sag) / diagonal
        if sag_ratio >= _COARSE_ABSOLUTE_SAG_RATIO:
            advisories.append(
                {
                    "code": "coarse_absolute_sag",
                    "severity": "warning",
                    "part_id": part.id,
                    "part_name": part.name,
                    "sag": float(options.sag),
                    "bbox_diagonal": diagonal,
                    "ratio": sag_ratio,
                    "message": (
                        f"tessellation sag is {sag_ratio:.1%} of the part bounding-box diagonal; "
                        f"small or high-detail features may be undersampled: {part.name}"
                    ),
                }
            )

    length_limit, length_kind = _active_max_length(options)
    if length_limit is not None:
        length_ratio = float(length_limit) / diagonal
        if length_ratio <= _AGGRESSIVE_MAX_LENGTH_RATIO and not _is_long_object(extents):
            advisories.append(
                {
                    "code": "aggressive_max_length",
                    "severity": "warning",
                    "part_id": part.id,
                    "part_name": part.name,
                    "length_kind": length_kind,
                    "length": float(length_limit),
                    "bbox_diagonal": diagonal,
                    "ratio": length_ratio,
                    "message": (
                        f"{length_kind} is very small relative to the part bounding box; "
                        "reserve aggressive polygon-length limits for long planar objects with lighting artifacts: "
                        f"{part.name}"
                    ),
                }
            )
    return advisories


def _adaptive_detail_contexts(asset: Asset, part: Part) -> list[str]:
    contexts = set(_detail_sensitive_contexts(asset, part))
    contexts.update(_curvature_sensitive_contexts(part))
    return sorted(contexts)


def _detail_sensitive_contexts(asset: Asset, part: Part) -> list[str]:
    contexts: set[str] = set()
    if _has_high_detail_metadata(part.metadata):
        contexts.add("high_detail_metadata")
    for material_id in part.material_ids:
        material = asset.materials.get(material_id)
        if material is None:
            continue
        if (
            material.roughness <= _SHINY_ROUGHNESS_THRESHOLD
            or (
                material.metallic >= _METALLIC_DETAIL_THRESHOLD
                and material.roughness <= _METALLIC_DETAIL_ROUGHNESS_THRESHOLD
            )
            or _has_shiny_material_metadata(material.metadata)
        ):
            contexts.add("shiny_material")
        if _has_high_detail_metadata(material.metadata):
            contexts.add("high_detail_material_metadata")
    return sorted(contexts)


def _curvature_sensitive_contexts(part: Part) -> list[str]:
    if part.source_shape is None:
        return []
    return ["curved_brep_faces"] if _source_shape_has_curved_faces(part.source_shape) else []


def _source_shape_has_curved_faces(shape: object) -> bool:
    try:
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.GeomAbs import GeomAbs_Plane
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError:
        return False

    try:
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
    except Exception:
        return False
    while explorer.More():
        try:
            face = TopoDS.Face_s(explorer.Current())
            surface = BRepAdaptor_Surface(face)
            if surface.GetType() != GeomAbs_Plane:
                return True
        except Exception:
            pass
        finally:
            explorer.Next()
    return False


def _has_high_detail_metadata(metadata: Metadata) -> bool:
    for key, value in metadata.items():
        normalized_key = str(key).strip().lower().replace("-", "_")
        if normalized_key in _DETAIL_METADATA_KEYS and _metadata_truthy(value):
            return True
    return False


def _has_shiny_material_metadata(metadata: Metadata) -> bool:
    for key, value in metadata.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in {"finish", "material_finish", "surface_finish"} and (
            _normalized_metadata_value(value) in _SHINY_MATERIAL_VALUES
        ):
            return True
    return False


def _metadata_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return _normalized_metadata_value(value) in _DETAIL_METADATA_VALUES


def _normalized_metadata_value(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def _active_max_length(options: TessellationOptions) -> tuple[float | None, str]:
    if options.max_edge_length is None:
        return options.max_polygon_length, "max_polygon_length"
    if options.max_polygon_length is None:
        return options.max_edge_length, "max_edge_length"
    if options.max_edge_length <= options.max_polygon_length:
        return options.max_edge_length, "max_edge_length"
    return options.max_polygon_length, "max_polygon_length"


def _is_long_object(extents: np.ndarray) -> bool:
    positive = sorted((float(value) for value in extents if value > 0.0), reverse=True)
    if len(positive) < 2:
        return False
    return positive[0] / positive[1] >= _LONG_OBJECT_AXIS_RATIO


def _store_tessellation_quality_advisories(
    asset: Asset,
    part: Part,
    advisories: list[dict[str, object]],
) -> None:
    if not advisories:
        return
    encoded = json.dumps(advisories, sort_keys=True)
    codes = ",".join(str(item["code"]) for item in advisories)
    part.metadata["tessellation_quality_advisories"] = encoded
    part.metadata["tessellation_quality_advisory_count"] = str(len(advisories))
    part.metadata["tessellation_quality_advisory_codes"] = codes
    if part.mesh is not None:
        part.mesh.metadata["tessellation_quality_advisories"] = encoded
        part.mesh.metadata["tessellation_quality_advisory_count"] = str(len(advisories))
        part.mesh.metadata["tessellation_quality_advisory_codes"] = codes
    for advisory in advisories:
        if advisory.get("severity") == "warning":
            asset.report.add_warning(str(advisory["message"]))


def _store_free_edge_report(
    asset: Asset,
    part: Part,
    metrics: dict[str, int | float] | None,
) -> None:
    if metrics is None:
        return
    boundary_edges = int(metrics["boundary_edges"])
    non_manifold_edges = int(metrics["non_manifold_edges"])
    part.metadata["tessellation_free_edges"] = str(boundary_edges)
    part.metadata["tessellation_non_manifold_edges"] = str(non_manifold_edges)
    if part.mesh is not None:
        part.mesh.metadata["tessellation_free_edges"] = str(boundary_edges)
        part.mesh.metadata["tessellation_non_manifold_edges"] = str(non_manifold_edges)
    if boundary_edges > 0:
        asset.report.add_warning(f"part has {boundary_edges} free tessellation edges: {part.name}")
    if non_manifold_edges > 0:
        asset.report.add_warning(f"part has {non_manifold_edges} non-manifold tessellation edges: {part.name}")


def _warn_long_polygons(
    asset: Asset,
    part: Part,
    options: TessellationOptions,
    metrics: dict[str, int | float] | None,
) -> None:
    if metrics is None or options.max_polygon_length is None:
        return
    long_edges = int(metrics["long_edges"])
    part.metadata["tessellation_long_polygon_edges"] = str(long_edges)
    part.metadata["tessellation_max_polygon_length"] = str(options.max_polygon_length)
    if long_edges > 0:
        asset.report.add_warning(f"part has {long_edges} tessellated edges longer than max_polygon_length: {part.name}")


def _record_brep_patch_cleanup(asset: Asset, part: Part, options: TessellationOptions) -> None:
    if part.source_shape is None:
        return
    cleanup = "retained" if options.keep_brep else "deleted"
    part.metadata["brep_patch_cleanup"] = cleanup
    part.metadata["source_shape_retained"] = str(options.keep_brep).lower()
    if part.mesh is not None:
        part.mesh.metadata["brep_patch_cleanup"] = cleanup
    if options.keep_brep:
        patch_count = _source_patch_count(part)
        part.metadata["brep_retained_patch_count"] = str(patch_count)
        if part.mesh is not None:
            part.mesh.metadata["brep_retained_patch_count"] = str(patch_count)
        if patch_count >= _FACE_GROUP_RISK_THRESHOLD:
            part.metadata["brep_patch_export_risk"] = "high"
            if part.mesh is not None:
                part.mesh.metadata["brep_patch_export_risk"] = "high"
            asset.report.add_warning(
                f"part retains {patch_count} BREP patch(es) after tessellation; "
                f"review draw-call and export-size risk before runtime export: {part.name}"
            )


def _record_submesh_risk(asset: Asset, part: Part) -> None:
    mesh = part.mesh
    if mesh is None:
        return
    face_group_count = len(mesh.face_groups)
    estimated_draw_calls = _estimated_part_draw_calls(part)
    part.metadata["tessellation_face_groups"] = str(face_group_count)
    part.metadata["tessellation_estimated_draw_calls"] = str(estimated_draw_calls)
    mesh.metadata["tessellation_face_groups"] = str(face_group_count)
    mesh.metadata["tessellation_estimated_draw_calls"] = str(estimated_draw_calls)
    if face_group_count >= _FACE_GROUP_RISK_THRESHOLD:
        part.metadata["tessellation_face_group_export_risk"] = "high"
        mesh.metadata["tessellation_face_group_export_risk"] = "high"
        asset.report.add_warning(
            f"part has {face_group_count} CAD face group(s) after tessellation; "
            f"per-face grouping can increase submesh or draw-call pressure: {part.name}"
        )
    if estimated_draw_calls >= _DRAW_CALL_RISK_THRESHOLD:
        part.metadata["tessellation_draw_call_export_risk"] = "high"
        mesh.metadata["tessellation_draw_call_export_risk"] = "high"
        asset.report.add_warning(
            f"part is estimated to emit {estimated_draw_calls} material draw call(s) after tessellation: {part.name}"
        )


def _estimated_part_draw_calls(part: Part) -> int:
    mesh = part.mesh
    if mesh is None or mesh.triangle_count == 0:
        return 0
    if mesh.material_indices is None:
        return 1
    return int(np.unique(mesh.material_indices.astype(np.int64, copy=False)).shape[0])


def _source_patch_count(part: Part) -> int:
    for value in (
        part.metadata.get("source_faces"),
        None if part.mesh is None else part.mesh.metadata.get("occt_faces"),
        None if part.mesh is None else part.mesh.metadata.get("tessellation_face_groups"),
    ):
        count = _metadata_int(value)
        if count is not None:
            return count
    return 0


def _metadata_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _metadata_str(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _metadata_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _format_metadata_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.9g}"
    return str(value)


def _tessellation_mesh_options(options: TessellationOptions) -> dict[str, object]:
    data = options.to_dict()
    data.pop("part_settings", None)
    data.pop("keep_brep", None)
    return data


def _tessellation_cache_key(
    shape: object,
    material_ids: list[str],
    face_material_indices: list[int] | None,
    options: TessellationOptions,
) -> tuple[str, tuple[str, ...], tuple[int, ...] | None, tuple[object, ...]]:
    indices = None if face_material_indices is None else tuple(face_material_indices)
    return (shape_fingerprint(shape), tuple(material_ids), indices, _tessellation_settings_key(options))


def _tessellation_settings_key(options: TessellationOptions) -> tuple[object, ...]:
    return (
        options.sag,
        options.sag_ratio,
        options.angle,
        options.relative,
        options.min_edge_length,
        options.max_edge_length,
        options.max_polygon_length,
        options.preserve_boundaries,
        options.curvature_adaptive,
        options.avoid_skinny_triangles,
        options.cad_uvs,
        options.tessellate_tangents,
        options.free_edge_geometry,
        options.create_normals,
        options.free_edge_report,
        options.reuse_existing_meshes,
        options.max_triangles_per_part,
    )


def build_tessellation_quality_report(asset: Asset) -> dict[str, object]:
    parts: list[dict[str, object]] = []
    for part in asset.parts.values():
        payload = _stored_quality_payload(part)
        if payload is None and part.mesh is not None:
            payload = {
                "part_id": part.id,
                "part_name": part.name,
                "options": {},
                "metrics": part.mesh.quality_metrics(),
            }
        if payload is not None:
            parts.append(payload)

    metrics = [cast(dict[str, int | float], part["metrics"]) for part in parts]
    summary = {
        "parts": len(parts),
        "triangles": int(sum(int(item["triangles"]) for item in metrics)),
        "vertices": int(sum(int(item["vertices"]) for item in metrics)),
        "short_edges": int(sum(int(item["short_edges"]) for item in metrics)),
        "long_edges": int(sum(int(item["long_edges"]) for item in metrics)),
        "skinny_triangles": int(sum(int(item["skinny_triangles"]) for item in metrics)),
        "degenerate_triangles": int(sum(int(item["degenerate_triangles"]) for item in metrics)),
        "boundary_edges": int(sum(int(item["boundary_edges"]) for item in metrics)),
        "non_manifold_edges": int(sum(int(item["non_manifold_edges"]) for item in metrics)),
        "min_edge_length": min((float(item["min_edge_length"]) for item in metrics), default=0.0),
        "max_edge_length": max((float(item["max_edge_length"]) for item in metrics), default=0.0),
        "max_aspect_ratio": max((float(item["max_aspect_ratio"]) for item in metrics), default=0.0),
    }
    return {"summary": summary, "parts": parts}


def _stored_quality_payload(part: Part) -> dict[str, object] | None:
    value = part.metadata.get("tessellation_quality")
    if value is None:
        return None
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _deduplicate_parts_by_fingerprint(asset: Asset) -> Asset:
    canonical_by_key: dict[tuple[str, tuple[str, ...], str | None, str], str] = {}
    replacements: dict[str, str] = {}
    for part_id, part in asset.parts.items():
        if part.fingerprint is None or part.mesh is None:
            continue
        material_indices = None
        if part.mesh.material_indices is not None:
            material_indices = array_digest_required(part.mesh.material_indices.astype(np.int64, copy=False))
        key = (part.fingerprint, tuple(part.material_ids), material_indices, _metadata_key(part.metadata))
        canonical_id = canonical_by_key.get(key)
        if canonical_id is None:
            canonical_by_key[key] = part_id
            continue
        replacements[part_id] = canonical_id
    if not replacements:
        return asset

    for node in asset.root.walk():
        if node.part_id in replacements:
            node.part_id = replacements[node.part_id]
    asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id not in replacements}
    return asset


def _metadata_key(metadata: Metadata) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)
