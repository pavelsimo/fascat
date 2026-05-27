from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fascat._ocp import shape_fingerprint as _shape_fingerprint
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.metadata import Metadata
from fascat.options import StepReadOptions
from fascat.report import Report, timed_step

_PartIndex = dict[tuple[str, str, str, str], str]


@dataclass(frozen=True)
class _StepHeaderInfo:
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


def read_step(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    source = Path(path)
    return _read_step_path(source, source_identity=str(source.resolve()), options=options or StepReadOptions())


def _read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing STEP file: {source}")
    if source.suffix.lower() not in {".step", ".stp"}:
        raise ValueError(f"unsupported STEP extension: {source.suffix or '<none>'}")

    header_info = _step_header_info(source)
    cleanup = _ImportCleanupStats()
    with timed_step() as timer:
        document, shape_tool, color_tool, unit_name, meters_per_unit = _read_xde_document(source, options)
        free_labels = _free_shape_labels(shape_tool)
        root = Node(
            id=_stable_id("node", f"{source_identity}:root"),
            name=source.stem,
            metadata={"source": str(source), "source_identity": source_identity},
        )
        parts: dict[str, Part] = {}
        part_index: _PartIndex = {}
        materials: dict[str, Material] = {}
        for index, label in enumerate(free_labels, start=1):
            root.children.append(
                _build_node(
                    label,
                    f"root/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )

    report = Report(source_path=str(source))
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        units=unit_name,
        meters_per_unit=meters_per_unit,
        up_axis="Z",
        source_path=source,
        metadata=_asset_metadata(source, source_identity, unit_name, meters_per_unit, options, header_info, cleanup),
        pmi=[],
        report=report,
    )
    asset.report.input_stats = asset.stats()
    metadata_count = _metadata_count(asset)
    unsupported_pmi_count = _unsupported_pmi_count(options, header_info, pmi_count=len(asset.pmi))
    import_warnings = _import_warnings(options, header_info, unsupported_pmi_count)
    for warning in import_warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "STEP",
            "backend": "OCP",
            "read_options": options.to_dict(),
            "metadata_count": metadata_count,
            "pmi_count": len(asset.pmi),
            "unsupported_pmi_count": unsupported_pmi_count,
            "pmi_schema": header_info.schema,
            "pmi_present": header_info.pmi_present,
            "cleanup": cleanup.to_dict(),
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
        warnings=import_warnings,
    )
    _ = document
    return asset


def read_step_bytes(data: bytes, *, name: str = "stdin.step", options: StepReadOptions | None = None) -> Asset:
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".step") as handle:
        handle.write(data)
        handle.flush()
        asset = _read_step_path(Path(handle.name), source_identity=name, options=options or StepReadOptions())
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    if asset.metadata:
        asset.metadata["source"] = name
        asset.metadata["source_identity"] = name
    return asset


def _read_xde_document(path: Path, options: StepReadOptions) -> tuple[Any, Any, Any, str, float]:
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except ImportError as exc:
        raise RuntimeError("STEP import requires cadquery-ocp") from exc

    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("fascat"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(options.metadata)
    reader.SetColorMode(True)
    reader.SetMatMode(True)
    reader.SetMetaMode(options.metadata or options.properties)
    reader.SetProductMetaMode(options.product_metadata)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"failed to read STEP file: {path}")

    unit_name, meters_per_unit = _reader_units(reader)
    if not reader.Transfer(document):
        raise RuntimeError(f"failed to transfer STEP data into XDE document: {path}")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    return document, shape_tool, color_tool, unit_name, meters_per_unit


def _free_shape_labels(shape_tool: Any) -> list[Any]:
    from OCP.TDF import TDF_LabelSequence

    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)
    return [labels.Value(index) for index in range(labels.Lower(), labels.Upper() + 1)]


def _shape_topology_counts(shape: Any) -> _ShapeTopologyCounts:
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    return _ShapeTopologyCounts(
        vertices=_count_subshapes(shape, TopAbs_VERTEX, TopExp_Explorer),
        edges=_count_subshapes(shape, TopAbs_EDGE, TopExp_Explorer),
        faces=_count_subshapes(shape, TopAbs_FACE, TopExp_Explorer),
    )


def _count_subshapes(shape: Any, shape_type: Any, explorer_factory: Any) -> int:
    explorer = explorer_factory(shape, shape_type)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _asset_metadata(
    source: Path,
    source_identity: str,
    unit_name: str,
    meters_per_unit: float,
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    cleanup: _ImportCleanupStats,
) -> Metadata:
    if not options.metadata:
        return {}
    metadata: Metadata = {
        "source": str(source),
        "source_identity": source_identity,
        "units": unit_name,
        "meters_per_unit": meters_per_unit,
        "metadata_options": options.to_dict(),
        "import_cleanup": cleanup.to_dict(),
    }
    if header_info.schema:
        metadata["step_schema"] = header_info.schema
    if header_info.pmi_present:
        metadata["pmi_present"] = "true"
        metadata["pmi_import_status"] = "unsupported" if options.pmi else "disabled"
    return metadata


def _step_header_info(source: Path) -> _StepHeaderInfo:
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(131_072)
    header = text.split("ENDSEC;", 1)[0]
    schema_match = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", header, flags=re.IGNORECASE | re.DOTALL)
    schema = " ".join(schema_match.group(1).split()) if schema_match else ""
    upper_header = header.upper()
    pmi_present = "AP242" in schema.upper() and (
        "PRODUCT MANUFACTURING INFORMATION" in upper_header or "PMI" in upper_header
    )
    return _StepHeaderInfo(schema=schema, pmi_present=pmi_present)


def _unsupported_pmi_count(options: StepReadOptions, header_info: _StepHeaderInfo, *, pmi_count: int) -> int:
    if not options.pmi or not header_info.pmi_present or pmi_count:
        return 0
    return 1


def _import_warnings(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    unsupported_pmi_count: int,
) -> list[str]:
    warnings: list[str] = []
    if options.pmi and unsupported_pmi_count:
        warnings.append(
            "STEP file advertises AP242 PMI, but PMI entity import is not implemented; annotations are omitted"
        )
    if options.design_variants:
        warnings.append("STEP design variant import is not implemented; variants are omitted")
    if options.multi_file:
        warnings.append("multi-file STEP assembly import is not implemented; external references are not loaded")
    return warnings


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
    if counts.edges > 0 and options.delete_lines:
        return "delete_lines"
    if counts.edges == 0 and counts.vertices > 0 and options.delete_free_vertices:
        return "delete_free_vertices"
    return None


def _metadata_count(asset: Asset) -> int:
    return (
        len(asset.metadata)
        + sum(len(node.metadata) for node in asset.root.walk())
        + sum(len(part.metadata) for part in asset.parts.values())
        + sum(len(material.metadata) for material in asset.materials.values())
    )


def _build_node(
    label: Any,
    occurrence_path: str,
    source_identity: str,
    shape_tool: Any,
    color_tool: Any,
    parts: dict[str, Part],
    part_index: _PartIndex,
    materials: dict[str, Material],
    options: StepReadOptions,
    cleanup: _ImportCleanupStats,
) -> Node:
    from OCP.TDF import TDF_LabelSequence
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    label_entry = _label_entry(label)
    node = Node(
        id=_stable_id("node", f"{source_identity}:{occurrence_path}"),
        name=_label_name(label) or f"Node {label_entry}",
        transform=_label_transform(label),
        metadata={"step_label": label_entry},
    )

    if XCAFDoc_ShapeTool.IsAssembly_s(label):
        children = TDF_LabelSequence()
        XCAFDoc_ShapeTool.GetComponents_s(label, children, False)
        for index in range(children.Lower(), children.Upper() + 1):
            child = children.Value(index)
            node.children.append(
                _build_node(
                    child,
                    f"{occurrence_path}/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )
        return node

    shape_label = _shape_definition_label(label)
    shape = XCAFDoc_ShapeTool.GetShape_s(shape_label)
    if shape.IsNull():
        return node
    topology = _shape_topology_counts(shape)
    representation = _loaded_representation(topology)
    cleanup_action = _cleanup_action(topology, options)
    if cleanup_action is not None:
        cleanup.record_deleted(cleanup_action, topology)
        node.metadata.update(
            {
                "loaded_representation": representation,
                "import_cleanup": cleanup_action,
                "source_vertices": str(topology.vertices),
                "source_edges": str(topology.edges),
                "source_faces": str(topology.faces),
            }
        )
        return node
    cleanup.record_loaded(representation)

    part_entry = _label_entry(shape_label)
    color = _label_color(label) or _label_color(shape_label) or (0.75, 0.75, 0.75, 1.0)
    material_id = _material_id(color)
    face_material_ids, face_material_colors = _face_material_ids(
        shape_tool,
        color_tool,
        shape_label,
        shape,
        base_material_id=material_id,
    )
    material_ids, face_material_indices = _material_binding_plan(material_id, face_material_ids)
    material_signature = "|".join(material_ids)
    if any(index != 0 for index in face_material_indices):
        material_signature = f"{material_signature}:{','.join(str(index) for index in face_material_indices)}"
    shape_hash = _shape_fingerprint(shape)
    part_id, is_new_part = _canonical_part_id(
        source_identity=source_identity,
        part_entry=part_entry,
        shape_hash=shape_hash,
        material_signature=material_signature,
        part_index=part_index,
    )
    node.part_id = part_id
    if is_new_part:
        _ensure_material(materials, material_id, color)
        for face_material_id, face_color in face_material_colors.items():
            _ensure_material(materials, face_material_id, face_color)
        metadata: Metadata = {
            "step_label": part_entry,
            "occurrence_label": label_entry,
            "source_identity": source_identity,
            "source_name": _label_name(shape_label) or "",
            "shape_fingerprint": shape_hash,
            "loaded_representation": representation,
            "source_vertices": str(topology.vertices),
            "source_edges": str(topology.edges),
            "source_faces": str(topology.faces),
        }
        if any(index != 0 for index in face_material_indices):
            metadata["occt_face_material_indices"] = ",".join(str(index) for index in face_material_indices)
        parts[part_id] = Part(
            id=part_id,
            name=_label_name(shape_label) or _label_name(label) or f"Part {part_entry}",
            source_shape=shape,
            material_ids=material_ids,
            metadata=metadata,
            fingerprint=shape_hash,
        )
    return node


def _canonical_part_id(
    *,
    source_identity: str,
    part_entry: str,
    shape_hash: str,
    material_signature: str,
    part_index: _PartIndex,
) -> tuple[str, bool]:
    label_key = ("label", source_identity, part_entry, material_signature)
    existing = part_index.get(label_key)
    if existing is not None:
        return existing, False

    shape_key = ("shape", source_identity, shape_hash, material_signature)
    existing = part_index.get(shape_key)
    if existing is not None:
        part_index[label_key] = existing
        return existing, False

    part_id = _stable_id("part", f"{source_identity}:{part_entry}")
    part_index[label_key] = part_id
    part_index[shape_key] = part_id
    return part_id, True


def _material_binding_plan(base_material_id: str, face_material_ids: list[str]) -> tuple[list[str], list[int]]:
    material_ids = [base_material_id]
    material_indices: list[int] = []
    for face_material_id in face_material_ids:
        if face_material_id not in material_ids:
            material_ids.append(face_material_id)
        material_indices.append(material_ids.index(face_material_id))
    return material_ids, material_indices


def _ensure_material(
    materials: dict[str, Material],
    material_id: str,
    color: tuple[float, float, float, float],
) -> None:
    if material_id not in materials:
        materials[material_id] = Material(id=material_id, name=f"CAD color {material_id[-8:]}", base_color=color)


def _shape_definition_label(label: Any) -> Any:
    from OCP.TDF import TDF_Label
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    if XCAFDoc_ShapeTool.IsReference_s(label):
        referred = TDF_Label()
        if XCAFDoc_ShapeTool.GetReferredShape_s(label, referred):
            return referred
    return label


def _label_entry(label: Any) -> str:
    from OCP.TCollection import TCollection_AsciiString
    from OCP.TDF import TDF_Tool

    value = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, value)
    return str(value.ToCString())


def _label_name(label: Any) -> str | None:
    from OCP.TDataStd import TDataStd_Name

    attribute = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        return None
    value = str(attribute.Get().ToExtString()).strip()
    return value or None


def _label_color(label: Any) -> tuple[float, float, float, float] | None:
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool

    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        color = Quantity_Color()
        if XCAFDoc_ColorTool.GetColor_s(label, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
    return None


def _face_material_ids(
    shape_tool: Any,
    color_tool: Any,
    shape_label: Any,
    shape: Any,
    *,
    base_material_id: str,
) -> tuple[list[str], dict[str, tuple[float, float, float, float]]]:
    from OCP.TDF import TDF_Label
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    material_ids: list[str] = []
    colors: dict[str, tuple[float, float, float, float]] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        color = _shape_color(color_tool, face)
        if color is None:
            sub_label = TDF_Label()
            if shape_tool.FindSubShape(shape_label, face, sub_label):
                color = _label_color(sub_label)
        if color is None:
            material_ids.append(base_material_id)
        else:
            material_id = _material_id(color)
            material_ids.append(material_id)
            colors[material_id] = color
        explorer.Next()
    return material_ids, colors


def _shape_color(color_tool: Any, shape: Any) -> tuple[float, float, float, float] | None:
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf

    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        color = Quantity_Color()
        if color_tool.GetColor(shape, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
        if color_tool.GetInstanceColor(shape, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
    return None


def _label_transform(label: Any) -> np.ndarray:
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    location = XCAFDoc_ShapeTool.GetLocation_s(label)
    transform = location.Transformation()
    matrix = np.eye(4, dtype=np.float64)
    for row in range(1, 4):
        for column in range(1, 5):
            matrix[row - 1, column - 1] = float(transform.Value(row, column))
    return matrix


def _reader_units(reader: Any) -> tuple[str, float]:
    from OCP.TColStd import TColStd_SequenceOfAsciiString

    length_units = TColStd_SequenceOfAsciiString()
    angle_units = TColStd_SequenceOfAsciiString()
    solid_angle_units = TColStd_SequenceOfAsciiString()
    reader.Reader().FileUnits(length_units, angle_units, solid_angle_units)
    if length_units.Length() == 0:
        return "millimetre", 0.001
    unit = str(length_units.Value(length_units.Lower()).ToCString()).lower()
    return unit, _meters_per_unit(unit)


def _meters_per_unit(unit: str) -> float:
    normalized = unit.lower().replace("meter", "metre")
    if "inch" in normalized:
        return 0.0254
    if "foot" in normalized or "feet" in normalized:
        return 0.3048
    if "centimetre" in normalized:
        return 0.01
    if "millimetre" in normalized:
        return 0.001
    if "metre" in normalized:
        return 1.0
    return 0.001


def _material_id(color: tuple[float, float, float, float]) -> str:
    encoded = ",".join(f"{component:.6f}" for component in color)
    return _stable_id("mat", encoded)


def _stable_id(prefix: str, value: str) -> str:
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
