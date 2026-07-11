from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.metadata import Metadata
from fascat.options import StepReadOptions

_PartIndex = dict[tuple[str, str, str, str], str]


_UNIT_FACTORS = {
    "metre": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "centimetre": 0.01,
    "centimeter": 0.01,
    "cm": 0.01,
    "millimetre": 0.001,
    "millimeter": 0.001,
    "mm": 0.001,
    "inch": 0.0254,
    "in": 0.0254,
    "foot": 0.3048,
    "feet": 0.3048,
    "ft": 0.3048,
}


_UNIT_NAMES = {
    "meter": "metre",
    "m": "metre",
    "centimeter": "centimetre",
    "cm": "centimetre",
    "millimeter": "millimetre",
    "mm": "millimetre",
    "in": "inch",
    "feet": "foot",
    "ft": "foot",
}


_MIRRORED_TRANSFORM_DETERMINANT_EPSILON = 1e-12


@dataclass(frozen=True)
class CadHeaderInfo:
    schema: str = ""
    pmi_present: bool = False


@dataclass(frozen=True)
class _ShapeTopologyCounts:
    vertices: int = 0
    edges: int = 0
    faces: int = 0


@dataclass
class _ImportCleanupStats:
    brep_parts: int = 0
    construction_point_parts: int = 0
    construction_line_parts: int = 0
    empty_shape_parts: int = 0
    deleted_free_vertex_parts: int = 0
    deleted_free_vertices: int = 0
    deleted_line_parts: int = 0
    deleted_line_edges: int = 0
    deleted_line_vertices: int = 0

    def record_loaded(self, representation: str) -> None:
        if representation == "brep":
            self.brep_parts += 1
        elif representation == "construction_points":
            self.construction_point_parts += 1
        elif representation == "construction_lines":
            self.construction_line_parts += 1
        elif representation == "empty_shape":
            self.empty_shape_parts += 1

    def record_deleted(self, action: str, counts: _ShapeTopologyCounts) -> None:
        if action == "delete_free_vertices":
            self.deleted_free_vertex_parts += 1
            self.deleted_free_vertices += counts.vertices
        elif action == "delete_lines":
            self.deleted_line_parts += 1
            self.deleted_line_edges += counts.edges
            self.deleted_line_vertices += counts.vertices

    def to_dict(self) -> dict[str, int]:
        return {
            "brep_parts": self.brep_parts,
            "construction_point_parts": self.construction_point_parts,
            "construction_line_parts": self.construction_line_parts,
            "empty_shape_parts": self.empty_shape_parts,
            "deleted_free_vertex_parts": self.deleted_free_vertex_parts,
            "deleted_free_vertices": self.deleted_free_vertices,
            "deleted_line_parts": self.deleted_line_parts,
            "deleted_line_edges": self.deleted_line_edges,
            "deleted_line_vertices": self.deleted_line_vertices,
        }


@dataclass(frozen=True)
class _SpaceNormalization:
    source_units: str
    source_meters_per_unit: float
    source_up_axis: str
    source_handedness: str
    target_units: str
    target_meters_per_unit: float
    target_up_axis: str
    target_handedness: str
    transform: np.ndarray

    @property
    def changed(self) -> bool:
        return not np.allclose(self.transform, np.eye(4, dtype=np.float64))

    @property
    def determinant(self) -> float:
        return _transform_determinant(self.transform)

    @property
    def mirrored(self) -> bool:
        return _is_mirrored_determinant(self.determinant)

    def metadata(self) -> dict[str, object]:
        return {
            "source_units": self.source_units,
            "source_meters_per_unit": self.source_meters_per_unit,
            "source_up_axis": self.source_up_axis,
            "source_handedness": self.source_handedness,
            "target_units": self.target_units,
            "target_meters_per_unit": self.target_meters_per_unit,
            "target_up_axis": self.target_up_axis,
            "target_handedness": self.target_handedness,
            "transform": self.transform.tolist(),
            "determinant": self.determinant,
            "mirrored": self.mirrored,
            "changed": self.changed,
        }


def _space_normalization(unit_name: str, meters_per_unit: float, options: StepReadOptions) -> _SpaceNormalization:
    source_units, source_meters_per_unit = _space_units(
        unit_name,
        meters_per_unit,
        override_units=options.source_units,
        override_meters_per_unit=options.source_meters_per_unit,
    )
    target_units, target_meters_per_unit = _space_units(
        source_units,
        source_meters_per_unit,
        override_units=options.target_units,
        override_meters_per_unit=options.target_meters_per_unit,
    )
    target_up_axis = options.target_up_axis or options.source_up_axis
    target_handedness = options.target_handedness or options.source_handedness
    transform = _space_transform(
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=options.source_up_axis,
        source_handedness=options.source_handedness,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=target_up_axis,
        target_handedness=target_handedness,
    )
    return _SpaceNormalization(
        source_units=source_units,
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=options.source_up_axis,
        source_handedness=options.source_handedness,
        target_units=target_units,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=target_up_axis,
        target_handedness=target_handedness,
        transform=transform,
    )


def _space_units(
    default_units: str,
    default_meters_per_unit: float,
    *,
    override_units: str | None,
    override_meters_per_unit: float | None,
) -> tuple[str, float]:
    unit_name = _canonical_unit_name(default_units)
    meters_per_unit = float(default_meters_per_unit)
    if override_units is not None:
        unit_name = _canonical_unit_name(override_units)
        meters_per_unit = _unit_factor(unit_name)
    if override_meters_per_unit is not None:
        meters_per_unit = float(override_meters_per_unit)
        if override_units is None:
            unit_name = "custom"
    return unit_name, meters_per_unit


def _canonical_unit_name(value: str) -> str:
    key = value.strip().lower()
    return _UNIT_NAMES.get(key, key or "unit")


def _unit_factor(unit_name: str) -> float:
    factor = _UNIT_FACTORS.get(unit_name)
    if factor is None:
        known = ", ".join(sorted({"metre", "centimetre", "millimetre", "inch", "foot"}))
        raise ValueError(f"unsupported unit name for space normalization: {unit_name}; known units: {known}")
    return factor


def _space_transform(
    *,
    source_meters_per_unit: float,
    source_up_axis: str,
    source_handedness: str,
    target_meters_per_unit: float,
    target_up_axis: str,
    target_handedness: str,
) -> np.ndarray:
    linear = (
        np.linalg.inv(_to_canonical_space(target_up_axis, target_handedness))
        @ _to_canonical_space(source_up_axis, source_handedness)
        * (source_meters_per_unit / target_meters_per_unit)
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    return transform


def _empty_mirrored_transform_summary() -> dict[str, int]:
    return {"local_mirrored_nodes": 0, "world_mirrored_nodes": 0, "mirrored_part_occurrences": 0}


def _transform_determinant(transform: np.ndarray) -> float:
    linear = np.asarray(transform, dtype=np.float64)[:3, :3]
    return float(np.linalg.det(linear))


def _is_mirrored_determinant(determinant: float) -> bool:
    return np.isfinite(determinant) and determinant < -_MIRRORED_TRANSFORM_DETERMINANT_EPSILON


def _annotate_mirrored_transforms(root: Node) -> dict[str, int]:
    summary = _empty_mirrored_transform_summary()

    def walk(node: Node, parent_world: np.ndarray) -> None:
        local_determinant = _transform_determinant(node.transform)
        world = parent_world @ node.transform
        world_determinant = _transform_determinant(world)
        if _is_mirrored_determinant(local_determinant):
            summary["local_mirrored_nodes"] += 1
            node.metadata["local_transform_determinant"] = local_determinant
            node.metadata["local_transform_mirrored"] = "true"
        if _is_mirrored_determinant(world_determinant):
            summary["world_mirrored_nodes"] += 1
            node.metadata["world_transform_determinant"] = world_determinant
            node.metadata["world_transform_mirrored"] = "true"
            if node.part_id is not None:
                summary["mirrored_part_occurrences"] += 1
        for child in node.children:
            walk(child, world)

    walk(root, np.eye(4, dtype=np.float64))
    return summary


def _mirrored_transform_warnings(summary: dict[str, int]) -> list[str]:
    if summary["local_mirrored_nodes"] == 0 and summary["world_mirrored_nodes"] == 0:
        return []
    return [
        "STEP import detected "
        f"{summary['local_mirrored_nodes']} local mirrored transform(s) and "
        f"{summary['world_mirrored_nodes']} mirrored world transform(s) with negative determinants; "
        f"{summary['mirrored_part_occurrences']} part occurrence(s) may need normal/winding compensation "
        "in downstream viewers"
    ]


def _to_canonical_space(up_axis: str, handedness: str) -> np.ndarray:
    if up_axis == "Z":
        axis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        axis = np.eye(3, dtype=np.float64)
    if handedness == "left":
        return np.diag([-1.0, 1.0, 1.0]) @ axis
    return axis


def _asset_metadata(
    source: Path,
    source_identity: str,
    options: StepReadOptions,
    header_info: CadHeaderInfo,
    cleanup: _ImportCleanupStats,
    space: _SpaceNormalization,
    pmi_count: int = 0,
    design_variant_summary: dict[str, int] | None = None,
    source_texture_summary: dict[str, int] | None = None,
    texture_binding_summary: dict[str, int] | None = None,
    material_library_summary: dict[str, int] | None = None,
    material_library_binding_summary: dict[str, int] | None = None,
    mirrored_transform_summary: dict[str, int] | None = None,
) -> Metadata:
    if not options.metadata:
        return {}
    metadata: Metadata = {
        "source": str(source),
        "source_identity": source_identity,
        "units": space.target_units,
        "meters_per_unit": space.target_meters_per_unit,
        "source_units": space.source_units,
        "source_meters_per_unit": space.source_meters_per_unit,
        "up_axis": space.target_up_axis,
        "source_up_axis": space.source_up_axis,
        "handedness": space.target_handedness,
        "source_handedness": space.source_handedness,
        "space_normalization": space.metadata(),
        "metadata_options": options.to_dict(),
        "import_cleanup": cleanup.to_dict(),
        "mirrored_transforms": mirrored_transform_summary or _empty_mirrored_transform_summary(),
    }
    if source_texture_summary is not None:
        metadata["source_texture_import"] = source_texture_summary
    if design_variant_summary is not None:
        metadata["design_variant_import"] = design_variant_summary
    if texture_binding_summary is not None:
        metadata["source_texture_bindings"] = texture_binding_summary
    if material_library_summary is not None:
        metadata["material_library_import"] = material_library_summary
    if material_library_binding_summary is not None:
        metadata["material_library_bindings"] = material_library_binding_summary
    if header_info.schema:
        metadata["step_schema"] = header_info.schema
    if header_info.pmi_present or pmi_count:
        metadata["pmi_present"] = "true"
        metadata["pmi_import_status"] = "imported" if pmi_count else "unsupported" if options.pmi else "disabled"
        metadata["pmi_import_count"] = pmi_count
    return metadata


def _loaded_representation_report(asset: Asset) -> dict[str, object]:
    parts = [_part_representation_record(part) for part in sorted(asset.parts.values(), key=lambda item: item.id)]
    deleted_nodes = [
        _deleted_node_representation_record(node)
        for node in asset.root.walk()
        if "import_cleanup" in node.metadata and node.part_id is None
    ]
    return {
        "summary": _representation_summary(parts, deleted_nodes),
        "parts": parts,
        "deleted_nodes": deleted_nodes,
    }


def _part_representation_record(part: Part) -> dict[str, object]:
    record: dict[str, object] = {
        "part_id": part.id,
        "name": part.name,
        "loaded_representation": str(part.metadata.get("loaded_representation", "unknown")),
        "cleanup_action": str(part.metadata.get("import_cleanup", "preserved")),
        "source_vertices": _metadata_int(part.metadata.get("source_vertices")),
        "source_edges": _metadata_int(part.metadata.get("source_edges")),
        "source_faces": _metadata_int(part.metadata.get("source_faces")),
        "source_name": str(part.metadata.get("source_name", "")),
    }
    if "construction_curve_policy" in part.metadata:
        record["construction_curve_policy"] = str(part.metadata["construction_curve_policy"])
    if "construction_curve_tube_radius" in part.metadata:
        record["construction_curve_tube_radius"] = _metadata_float(part.metadata["construction_curve_tube_radius"])
    if "mixed_construction_curve_action" in part.metadata:
        record["mixed_construction_curve_action"] = str(part.metadata["mixed_construction_curve_action"])
        record["mixed_construction_curve_edges"] = _metadata_int(part.metadata.get("mixed_construction_curve_edges"))
    return record


def _deleted_node_representation_record(node: Node) -> dict[str, object]:
    record: dict[str, object] = {
        "node_id": node.id,
        "name": node.name,
        "loaded_representation": str(node.metadata.get("loaded_representation", "unknown")),
        "cleanup_action": str(node.metadata.get("import_cleanup", "preserved")),
        "source_vertices": _metadata_int(node.metadata.get("source_vertices")),
        "source_edges": _metadata_int(node.metadata.get("source_edges")),
        "source_faces": _metadata_int(node.metadata.get("source_faces")),
    }
    if "construction_curve_policy" in node.metadata:
        record["construction_curve_policy"] = str(node.metadata["construction_curve_policy"])
    if "mixed_construction_curve_split" in node.metadata:
        record["mixed_construction_curve_split"] = str(node.metadata["mixed_construction_curve_split"])
    return record


def _representation_summary(
    parts: list[dict[str, object]],
    deleted_nodes: list[dict[str, object]],
) -> dict[str, int]:
    summary = {
        "brep_parts": 0,
        "construction_point_parts": 0,
        "construction_line_parts": 0,
        "empty_shape_parts": 0,
        "unknown_parts": 0,
        "deleted_nodes": len(deleted_nodes),
        "deleted_free_vertex_nodes": 0,
        "deleted_line_nodes": 0,
    }
    for part in parts:
        representation = part.get("loaded_representation")
        if representation == "brep":
            summary["brep_parts"] += 1
        elif representation == "construction_points":
            summary["construction_point_parts"] += 1
        elif representation == "construction_lines":
            summary["construction_line_parts"] += 1
        elif representation == "empty_shape":
            summary["empty_shape_parts"] += 1
        else:
            summary["unknown_parts"] += 1
    for node in deleted_nodes:
        cleanup_action = node.get("cleanup_action")
        if cleanup_action == "delete_free_vertices":
            summary["deleted_free_vertex_nodes"] += 1
        elif cleanup_action == "delete_lines":
            summary["deleted_line_nodes"] += 1
    return summary


def _metadata_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _metadata_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _loaded_representation(counts: _ShapeTopologyCounts) -> str:
    if counts.faces > 0:
        return "brep"
    if counts.edges > 0:
        return "construction_lines"
    if counts.vertices > 0:
        return "construction_points"
    return "empty_shape"


def _cleanup_action(counts: _ShapeTopologyCounts, options: StepReadOptions) -> str | None:
    if counts.faces > 0:
        return None
    if counts.edges > 0 and _construction_curve_policy(options) == "delete":
        return "delete_lines"
    if counts.edges == 0 and counts.vertices > 0 and options.delete_free_vertices:
        return "delete_free_vertices"
    return None


def _construction_curve_policy(options: StepReadOptions) -> str:
    return "delete" if options.delete_lines else options.construction_curve_policy


def _construction_curve_metadata(options: StepReadOptions, representation: str) -> dict[str, str]:
    if representation != "construction_lines":
        return {}
    metadata = {"construction_curve_policy": _construction_curve_policy(options)}
    if metadata["construction_curve_policy"] == "tessellate_tubes":
        metadata["construction_curve_tube_radius"] = str(options.construction_curve_tube_radius)
    return metadata


def _metadata_count(asset: Asset) -> int:
    return (
        len(asset.metadata)
        + sum(len(node.metadata) for node in asset.root.walk())
        + sum(len(part.metadata) for part in asset.parts.values())
        + sum(len(material.metadata) for material in asset.materials.values())
    )


def _stable_id(prefix: str, value: str) -> str:
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
