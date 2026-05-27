from __future__ import annotations

import json
import math
from typing import Any, cast

import numpy as np

from fascat._ocp import shape_fingerprint
from fascat.asset import Asset, Part
from fascat.mesh import Mesh
from fascat.metadata import Metadata
from fascat.options import Tessellation


def tessellate_asset(asset: Asset, options: Tessellation, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    mesh_by_source: dict[tuple[str, tuple[str, ...], tuple[int, ...] | None, tuple[object, ...]], Mesh] = {}
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part_options = _options_for_part(options, part)
        if part.mesh is not None and part_options.reuse_existing_meshes:
            _record_tessellation_diagnostics(result, part, part_options)
            continue
        if part.source_shape is None:
            if part.mesh is None:
                result.report.add_warning(f"part has no source shape and cannot be tessellated: {part.name}")
            else:
                result.report.add_warning(
                    f"part has existing mesh but no source shape and cannot be retessellated: {part.name}"
                )
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
        part.fingerprint = part.mesh.fingerprint()
        _record_tessellation_diagnostics(result, part, part_options)
        if not part_options.keep_brep:
            part.source_shape = None
    if selected_part_ids is not None:
        return result
    return _deduplicate_parts_by_fingerprint(result)


def tessellate_shape(
    shape: object,
    options: Tessellation,
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

    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    material_indices: list[int] = []
    face_groups: dict[str, list[int]] = {}
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
            offset = len(points)
            transform = location.Transformation()
            for node_index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_index).Transformed(transform)
                points.append((point.X(), point.Y(), point.Z()))
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for triangle_index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(triangle_index).Get()
                mesh_face_index = len(faces)
                if reversed_face:
                    faces.append((offset + c - 1, offset + b - 1, offset + a - 1))
                else:
                    faces.append((offset + a - 1, offset + b - 1, offset + c - 1))
                face_groups.setdefault(f"occt_face_{face_index}", []).append(mesh_face_index)
                if material_index is not None:
                    material_indices.append(material_index)
        face_index += 1
        explorer.Next()

    mesh = Mesh(
        points=np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        material_indices=(
            np.asarray(material_indices, dtype=np.int64)
            if material_indices and len(material_indices) == len(faces)
            else None
        ),
        face_groups={name: np.asarray(values, dtype=np.int64) for name, values in face_groups.items()},
        metadata={"occt_faces": str(face_index)},
    )
    mesh = _apply_mesh_tessellation_controls(mesh, options)
    if options.create_normals:
        mesh = mesh.compute_normals()
    else:
        mesh.normals = None
    mesh.validate()
    return mesh


def _face_material_indices_from_metadata(metadata: Metadata) -> list[int] | None:
    value = metadata.get("occt_face_material_indices")
    if not value:
        return None
    return [int(item) for item in str(value).split(",") if item]


def _options_for_part(options: Tessellation, part: Part) -> Tessellation:
    overrides = options.part_settings.get(part.id) or options.part_settings.get(part.name)
    if not overrides:
        return options
    values = options.to_dict()
    values["part_settings"] = {}
    values.update(overrides)
    return Tessellation(**cast(Any, values))


def _occt_mesh_parameters(options: Tessellation, parameters_factory: Any) -> Any:
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


def _deflection_settings(options: Tessellation) -> tuple[float, bool]:
    if options.sag_ratio is not None:
        return float(options.sag_ratio), True
    return float(options.sag), bool(options.relative)


def _apply_mesh_tessellation_controls(mesh: Mesh, options: Tessellation) -> Mesh:
    result = mesh
    if options.max_edge_length is not None:
        result = result.subdivide_long_edges(options.max_edge_length)
    if options.min_edge_length is not None:
        result = result.collapse_short_edges(
            options.min_edge_length,
            preserve_boundaries=options.preserve_boundaries,
        )
    if options.avoid_skinny_triangles:
        result = result.improve_skinny_triangles(preserve_boundaries=options.preserve_boundaries)
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
    }
    return result


def _record_tessellation_diagnostics(asset: Asset, part: Part, options: Tessellation) -> None:
    if part.mesh is None:
        return
    metrics: dict[str, int | float] | None = None
    if options.quality_report:
        metrics = _store_quality_report(part, options)
    if (options.free_edge_report or options.max_polygon_length is not None) and metrics is None:
        metrics = part.mesh.quality_metrics(
            min_edge_length=options.min_edge_length,
            max_edge_length=_quality_max_edge_length(options),
        )
    if options.free_edge_report:
        _store_free_edge_report(asset, part, metrics)
    if options.max_polygon_length is not None:
        _warn_long_polygons(asset, part, options, metrics)


def _store_quality_report(part: Part, options: Tessellation) -> dict[str, int | float] | None:
    if part.mesh is None:
        return None
    metrics = part.mesh.quality_metrics(
        min_edge_length=options.min_edge_length,
        max_edge_length=_quality_max_edge_length(options),
    )
    payload = {
        "part_id": part.id,
        "part_name": part.name,
        "options": _tessellation_mesh_options(options),
        "metrics": metrics,
    }
    encoded = json.dumps(payload, sort_keys=True)
    part.metadata["tessellation_quality"] = encoded
    part.metadata["tessellation_short_edges"] = str(metrics["short_edges"])
    part.metadata["tessellation_long_edges"] = str(metrics["long_edges"])
    part.metadata["tessellation_skinny_triangles"] = str(metrics["skinny_triangles"])
    part.mesh.metadata["tessellation_quality"] = encoded
    return metrics


def _quality_max_edge_length(options: Tessellation) -> float | None:
    return options.max_polygon_length if options.max_polygon_length is not None else options.max_edge_length


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
    options: Tessellation,
    metrics: dict[str, int | float] | None,
) -> None:
    if metrics is None or options.max_polygon_length is None:
        return
    long_edges = int(metrics["long_edges"])
    part.metadata["tessellation_long_polygon_edges"] = str(long_edges)
    part.metadata["tessellation_max_polygon_length"] = str(options.max_polygon_length)
    if long_edges > 0:
        asset.report.add_warning(f"part has {long_edges} tessellated edges longer than max_polygon_length: {part.name}")


def _tessellation_mesh_options(options: Tessellation) -> dict[str, object]:
    data = options.to_dict()
    data.pop("part_settings", None)
    data.pop("keep_brep", None)
    return data


def _tessellation_cache_key(
    shape: object,
    material_ids: list[str],
    face_material_indices: list[int] | None,
    options: Tessellation,
) -> tuple[str, tuple[str, ...], tuple[int, ...] | None, tuple[object, ...]]:
    indices = None if face_material_indices is None else tuple(face_material_indices)
    return (shape_fingerprint(shape), tuple(material_ids), indices, _tessellation_settings_key(options))


def _tessellation_settings_key(options: Tessellation) -> tuple[object, ...]:
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
        options.create_normals,
        options.free_edge_report,
        options.reuse_existing_meshes,
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
    canonical_by_key: dict[tuple[str, tuple[str, ...], tuple[int, ...] | None, str], str] = {}
    replacements: dict[str, str] = {}
    for part_id, part in asset.parts.items():
        if part.fingerprint is None or part.mesh is None:
            continue
        material_indices = None
        if part.mesh.material_indices is not None:
            material_indices = tuple(int(value) for value in part.mesh.material_indices.tolist())
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
