from __future__ import annotations

import json
import math
from typing import Any, cast

import numpy as np

from fascat._ocp import shape_fingerprint
from fascat.asset import Asset, Part
from fascat.mesh import Mesh
from fascat.metadata import Metadata
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


def tessellate_asset(asset: Asset, options: TessellationOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    mesh_by_source: dict[tuple[str, tuple[str, ...], tuple[int, ...] | None, tuple[object, ...]], Mesh] = {}
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part_options = _options_for_part(options, part)
        if part.mesh is not None and part_options.reuse_existing_meshes:
            _record_tessellation_attribute_sources(result, part, part_options, geometry_source="imported_mesh")
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
        _record_tessellation_attribute_sources(result, part, part_options, geometry_source="tessellation")
        _record_tessellation_diagnostics(result, part, part_options)
        _record_brep_patch_cleanup(result, part, part_options)
        if not part_options.keep_brep:
            part.source_shape = None
    if selected_part_ids is not None:
        return result
    return _deduplicate_parts_by_fingerprint(result)


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


def _options_for_part(options: TessellationOptions, part: Part) -> TessellationOptions:
    overrides = options.part_settings.get(part.id) or options.part_settings.get(part.name)
    if not overrides:
        return options
    values = options.to_dict()
    values["part_settings"] = {}
    values.update(overrides)
    return TessellationOptions(**cast(Any, values))


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
    return float(options.sag), bool(options.relative)


def _apply_mesh_tessellation_controls(mesh: Mesh, options: TessellationOptions) -> Mesh:
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
        "sag": float(options.sag),
        "sag_ratio": options.sag_ratio,
        "curvature_adaptive": bool(options.curvature_adaptive),
        "preserve_boundaries": bool(options.preserve_boundaries),
    }
    if active_kind == "absolute_sag":
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
        "tangents": _tangent_attribute_source(mesh, geometry_source),
        "uvs": _uv_attribute_sources(mesh, geometry_source),
        "face_groups": _face_group_attribute_source(mesh, geometry_source),
        "free_edges": "diagnostic_only" if options.free_edge_report else "not_requested",
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


def _tangent_attribute_source(mesh: Mesh, geometry_source: str) -> str:
    if mesh.tangents is not None:
        return geometry_source
    return "not_generated_by_tessellation" if geometry_source == "tessellation" else "missing"


def _uv_attribute_sources(mesh: Mesh, geometry_source: str) -> dict[str, str]:
    if not mesh.uvs:
        return {"status": "not_generated_by_tessellation" if geometry_source == "tessellation" else "missing"}
    source = "tessellation" if geometry_source == "tessellation" else "imported_mesh"
    return {str(channel): source for channel in sorted(mesh.uvs)}


def _face_group_attribute_source(mesh: Mesh, geometry_source: str) -> str:
    if not mesh.face_groups:
        return "missing"
    return "cad_face_groups" if geometry_source == "tessellation" else "imported_mesh"


def _brep_patch_attribute_source(part: Part, options: TessellationOptions, geometry_source: str) -> str:
    if part.source_shape is None:
        return "not_available"
    if geometry_source == "imported_mesh":
        return "unchanged_existing_mesh_reuse"
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
    detail_contexts = _detail_sensitive_contexts(asset, part)
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
                    "part has shiny or high-detail material/metadata but tessellation uses bulk criteria "
                    f"without sag_ratio or curvature_adaptive; consider finer per-part tessellation: {part.name}"
                ),
            }
        )

    if options.sag_ratio is None and not options.relative:
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
    return len(set(mesh.material_indices.astype(int).tolist()))


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
