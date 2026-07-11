from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fascat._ocp import shape_fingerprint as _shape_fingerprint
from fascat.asset import Node, Part
from fascat.io._import_base import (
    _cleanup_action,
    _construction_curve_metadata,
    _construction_curve_policy,
    _ImportCleanupStats,
    _loaded_representation,
    _PartIndex,
    _ShapeTopologyCounts,
    _stable_id,
)
from fascat.io.step.materials import (
    _CadMaterialSpec,
    _color_material_spec,
    _ensure_material,
    _label_visual_material_spec,
    _material_binding_plan,
    _material_id_from_spec,
    _shape_visual_material_spec,
)
from fascat.material import Material
from fascat.metadata import Metadata
from fascat.options import StepReadOptions


def _read_xde_document(path: Path, options: StepReadOptions) -> tuple[Any, Any, Any, Any, str, float]:
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
    vis_material_tool = XCAFDoc_DocumentTool.VisMaterialTool_s(document.Main())
    return document, shape_tool, color_tool, vis_material_tool, unit_name, meters_per_unit


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


def _build_node(
    label: Any,
    occurrence_path: str,
    source_identity: str,
    shape_tool: Any,
    color_tool: Any,
    vis_material_tool: Any,
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
                    vis_material_tool,
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
    mixed_construction_shape = _mixed_construction_curve_shape(shape, topology)
    mixed_construction_counts = (
        _shape_topology_counts(mixed_construction_shape) if mixed_construction_shape is not None else None
    )
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
                **_construction_curve_metadata(options, representation),
            }
        )
        return node
    cleanup.record_loaded(representation)

    part_entry = _label_entry(shape_label)
    color = _label_color(label) or _label_color(shape_label) or (0.75, 0.75, 0.75, 1.0)
    base_spec = (
        _label_visual_material_spec(vis_material_tool, label, options)
        or _label_visual_material_spec(vis_material_tool, shape_label, options)
        or _color_material_spec(color)
    )
    material_id = _material_id_from_spec(base_spec)
    face_material_ids, face_material_specs = _face_material_ids(
        shape_tool,
        color_tool,
        vis_material_tool,
        shape_label,
        shape,
        base_material_id=material_id,
        options=options,
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
        _ensure_material(materials, material_id, base_spec)
        for face_material_id, face_spec in face_material_specs.items():
            _ensure_material(materials, face_material_id, face_spec)
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
            **_construction_curve_metadata(options, representation),
        }
        if mixed_construction_counts is not None:
            metadata.update(
                _mixed_construction_curve_metadata(
                    options,
                    "deleted" if _construction_curve_policy(options) == "delete" else "split",
                    mixed_construction_counts,
                )
            )
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
    if mixed_construction_shape is not None and mixed_construction_counts is not None:
        if _construction_curve_policy(options) == "delete":
            cleanup.record_deleted("delete_lines", mixed_construction_counts)
        else:
            curve_node = _build_mixed_construction_curve_node(
                source_identity=source_identity,
                occurrence_path=occurrence_path,
                label_entry=label_entry,
                part_entry=part_entry,
                source_name=_label_name(shape_label) or _label_name(label) or f"Part {part_entry}",
                shape=mixed_construction_shape,
                counts=mixed_construction_counts,
                material_ids=material_ids,
                part_index=part_index,
                parts=parts,
                options=options,
                cleanup=cleanup,
            )
            node.children.append(curve_node)
    return node


def _mixed_construction_curve_shape(shape: Any, counts: _ShapeTopologyCounts) -> Any | None:
    if counts.faces == 0 or counts.edges == 0:
        return None
    try:
        from OCP.BRep import BRep_Builder
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound
        from OCP.TopTools import TopTools_IndexedMapOfShape
    except ImportError:
        return None

    face_edge_map = TopTools_IndexedMapOfShape()
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        TopExp.MapShapes_s(face_explorer.Current(), TopAbs_EDGE, face_edge_map)
        face_explorer.Next()
    if face_edge_map.IsEmpty():
        return None

    free_edges: list[Any] = []
    free_edge_map = TopTools_IndexedMapOfShape()
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge_s(edge_explorer.Current())
        if not face_edge_map.Contains(edge) and not free_edge_map.Contains(edge):
            free_edge_map.Add(edge)
            free_edges.append(edge)
        edge_explorer.Next()
    if not free_edges:
        return None

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for edge in free_edges:
        builder.Add(compound, edge)
    return compound


def _build_mixed_construction_curve_node(
    *,
    source_identity: str,
    occurrence_path: str,
    label_entry: str,
    part_entry: str,
    source_name: str,
    shape: Any,
    counts: _ShapeTopologyCounts,
    material_ids: list[str],
    part_index: _PartIndex,
    parts: dict[str, Part],
    options: StepReadOptions,
    cleanup: _ImportCleanupStats,
) -> Node:
    cleanup.record_loaded("construction_lines")
    curve_entry = f"{part_entry}:construction_curves"
    shape_hash = _shape_fingerprint(shape)
    material_signature = "|".join(material_ids)
    part_id, is_new_part = _canonical_part_id(
        source_identity=source_identity,
        part_entry=curve_entry,
        shape_hash=shape_hash,
        material_signature=f"{material_signature}:construction_curves",
        part_index=part_index,
    )
    if is_new_part:
        metadata: Metadata = {
            "step_label": curve_entry,
            "occurrence_label": label_entry,
            "source_identity": source_identity,
            "source_name": f"{source_name} construction curves",
            "shape_fingerprint": shape_hash,
            "loaded_representation": "construction_lines",
            "source_vertices": str(counts.vertices),
            "source_edges": str(counts.edges),
            "source_faces": str(counts.faces),
            "mixed_construction_curve_split": "true",
            **_construction_curve_metadata(options, "construction_lines"),
        }
        parts[part_id] = Part(
            id=part_id,
            name=f"{source_name} Construction Curves",
            source_shape=shape,
            material_ids=list(material_ids),
            metadata=metadata,
            fingerprint=shape_hash,
        )
    return Node(
        id=_stable_id("node", f"{source_identity}:{occurrence_path}:construction_curves"),
        name=f"{source_name} Construction Curves",
        part_id=part_id,
        metadata={
            "step_label": curve_entry,
            "occurrence_label": label_entry,
            "loaded_representation": "construction_lines",
            "mixed_construction_curve_split": "true",
            "source_vertices": str(counts.vertices),
            "source_edges": str(counts.edges),
            "source_faces": str(counts.faces),
            **_construction_curve_metadata(options, "construction_lines"),
        },
    )


def _mixed_construction_curve_metadata(
    options: StepReadOptions,
    action: str,
    counts: _ShapeTopologyCounts,
) -> dict[str, str]:
    metadata = {
        "mixed_construction_curve_policy": _construction_curve_policy(options),
        "mixed_construction_curve_action": action,
        "mixed_construction_curve_vertices": str(counts.vertices),
        "mixed_construction_curve_edges": str(counts.edges),
    }
    if action == "split":
        metadata["mixed_construction_curve_split"] = "true"
    return metadata


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


def _label_color(label: Any, reusable_color: Any | None = None) -> tuple[float, float, float, float] | None:
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool

    color = reusable_color if reusable_color is not None else Quantity_Color()
    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        if XCAFDoc_ColorTool.GetColor_s(label, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
    return None


def _face_material_ids(
    shape_tool: Any,
    color_tool: Any,
    vis_material_tool: Any,
    shape_label: Any,
    shape: Any,
    *,
    base_material_id: str,
    options: StepReadOptions,
) -> tuple[list[str], dict[str, _CadMaterialSpec]]:
    from OCP.TDF import TDF_Label
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    material_ids: list[str] = []
    specs: dict[str, _CadMaterialSpec] = {}
    reusable_color: Any | None = None
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        spec = _shape_visual_material_spec(vis_material_tool, face, options)
        found_sub_label = False
        sub_label: Any | None = None
        if spec is None:
            sub_label = TDF_Label()
            found_sub_label = shape_tool.FindSubShape(shape_label, face, sub_label)
            if found_sub_label:
                spec = _label_visual_material_spec(vis_material_tool, sub_label, options)
        if spec is None:
            if reusable_color is None:
                from OCP.Quantity import Quantity_Color

                reusable_color = Quantity_Color()
            color = _shape_color(color_tool, face, reusable_color)
            if color is None and found_sub_label and sub_label is not None:
                color = _label_color(sub_label, reusable_color)
            if color is not None:
                spec = _color_material_spec(color)
        if spec is None:
            material_ids.append(base_material_id)
        else:
            material_id = _material_id_from_spec(spec)
            material_ids.append(material_id)
            specs[material_id] = spec
        explorer.Next()
    return material_ids, specs


def _shape_color(
    color_tool: Any,
    shape: Any,
    reusable_color: Any | None = None,
) -> tuple[float, float, float, float] | None:
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf

    color = reusable_color if reusable_color is not None else Quantity_Color()
    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
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
