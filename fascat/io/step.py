from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from fascat._ocp import shape_fingerprint as _shape_fingerprint
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.report import Report, timed_step

_PartIndex = dict[tuple[str, str, str, str], str]


def read_step(path: str | Path) -> Asset:
    source = Path(path)
    return _read_step_path(source, source_identity=str(source.resolve()))


def _read_step_path(source: Path, *, source_identity: str, display_name: str | None = None) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing STEP file: {source}")
    if source.suffix.lower() not in {".step", ".stp"}:
        raise ValueError(f"unsupported STEP extension: {source.suffix or '<none>'}")

    return _read_xde_path(
        source,
        source_identity=source_identity,
        display_name=display_name,
        import_format="STEP",
        backend="OCP",
        reader=_read_xde_document,
    )


def _read_xde_path(
    source: Path,
    *,
    source_identity: str,
    import_format: str,
    backend: str,
    reader: Callable[[Path], tuple[Any, Any, Any, str, float]],
    display_name: str | None = None,
) -> Asset:
    with timed_step() as timer:
        document, shape_tool, color_tool, unit_name, meters_per_unit = reader(source)
        free_labels = _free_shape_labels(shape_tool)
        root = Node(
            id=_stable_id("node", f"{source_identity}:root"),
            name=display_name or source.stem,
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
        report=report,
    )
    asset.report.input_stats = asset.stats()
    asset.report.add_step(
        "import",
        options={"format": import_format, "backend": backend},
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
    )
    _ = document
    return asset


def read_step_bytes(data: bytes, *, name: str = "stdin.step") -> Asset:
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".step") as handle:
        handle.write(data)
        handle.flush()
        asset = _read_step_path(Path(handle.name), source_identity=name)
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    return asset


def _read_xde_document(path: Path) -> tuple[Any, Any, Any, str, float]:
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
    reader.SetNameMode(True)
    reader.SetColorMode(True)
    reader.SetMatMode(True)
    reader.SetMetaMode(True)
    reader.SetProductMetaMode(True)
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


def _build_node(
    label: Any,
    occurrence_path: str,
    source_identity: str,
    shape_tool: Any,
    color_tool: Any,
    parts: dict[str, Part],
    part_index: _PartIndex,
    materials: dict[str, Material],
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
                )
            )
        return node

    shape_label = _shape_definition_label(label)
    shape = XCAFDoc_ShapeTool.GetShape_s(shape_label)
    if shape.IsNull():
        return node

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
        metadata = {
            "step_label": part_entry,
            "occurrence_label": label_entry,
            "source_identity": source_identity,
            "source_name": _label_name(shape_label) or "",
            "shape_fingerprint": shape_hash,
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
