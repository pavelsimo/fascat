from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.export_report import referenced_materials
from fascat.io._atomic import atomic_output
from fascat.io._errors import wrap_io_errors
from fascat.io._geometry import face_normals
from fascat.io._suffixes import FBX_SUFFIXES
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import FbxExportOptions

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class _FbxObjectIds:
    root_model: int
    model_by_node_id: dict[str, int]
    geometry_by_node_id: dict[str, int]
    material_by_id: dict[str, int]


@wrap_io_errors("write FBX")
def write_fbx(asset: Asset, path: str | Path, *, options: FbxExportOptions | None = None) -> None:
    _write_fbx(asset, path, options=options, collect_stats=False)


@wrap_io_errors("write FBX with validation")
def write_fbx_with_validation_stats(
    asset: Asset,
    path: str | Path,
    *,
    options: FbxExportOptions | None = None,
) -> dict[str, int] | None:
    return _write_fbx(asset, path, options=options, collect_stats=True)


def _write_fbx(
    asset: Asset,
    path: str | Path,
    *,
    options: FbxExportOptions | None,
    collect_stats: bool,
) -> dict[str, int] | None:
    opts = options or FbxExportOptions()
    output_path = Path(path)
    if output_path.suffix.lower() not in FBX_SUFFIXES:
        raise ValueError(f"unsupported FBX extension: {output_path.suffix or '<none>'}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ids = _object_ids(asset, opts)
    text = _fbx_document(asset, opts, ids)
    with atomic_output(output_path) as temp:
        temp.write_text(text, encoding="utf-8")
        stats = validate_fbx(temp) if collect_stats else None
    return stats


def validate_fbx(path: str | Path) -> dict[str, int]:
    text = Path(path).read_text(encoding="utf-8")
    if "FBXHeaderExtension:" not in text or "Objects:" not in text or "Connections:" not in text:
        raise RuntimeError("FBX asset is missing required sections")
    mesh_count = text.count("GeometryVersion: 124")
    vertices = _count_array_values(text, "Vertices") // 3
    polygon_indices = _array_values(text, "PolygonVertexIndex")
    triangles = sum(1 for value in polygon_indices if value < 0)
    if mesh_count == 0 or vertices == 0 or triangles == 0:
        raise RuntimeError("FBX asset contains no mesh faces")
    return {"meshes": mesh_count, "points": vertices, "triangles": triangles}


def _object_ids(asset: Asset, options: FbxExportOptions) -> _FbxObjectIds:
    next_id = 1000
    root_model = next_id
    next_id += 1
    model_by_node_id: dict[str, int] = {}
    geometry_by_node_id: dict[str, int] = {}
    for node in asset.root.walk():
        model_by_node_id[node.id] = next_id
        next_id += 1
        if node.part_id is not None and node.part_id in asset.parts and asset.parts[node.part_id].mesh is not None:
            geometry_by_node_id[node.id] = next_id
            next_id += 1
    material_by_id: dict[str, int] = {}
    if options.materials:
        for material_id in referenced_materials(asset):
            material_by_id[material_id] = next_id
            next_id += 1
    return _FbxObjectIds(
        root_model=root_model,
        model_by_node_id=model_by_node_id,
        geometry_by_node_id=geometry_by_node_id,
        material_by_id=material_by_id,
    )


def _fbx_document(asset: Asset, options: FbxExportOptions, ids: _FbxObjectIds) -> str:
    lines: list[str] = []
    lines.extend(_header())
    lines.extend(_global_settings(asset))
    lines.append("Objects:  {\n")
    lines.extend(_model_object(ids.root_model, "Scene", np.eye(4, dtype=np.float64), kind="Null"))
    for material_id, material_object_id in ids.material_by_id.items():
        material = asset.materials.get(material_id)
        if material is not None:
            lines.extend(_material_object(material_object_id, material))
    for node in asset.root.walk():
        model_id = ids.model_by_node_id[node.id]
        kind = "Mesh" if node.id in ids.geometry_by_node_id else "Null"
        lines.extend(_model_object(model_id, node.name or node.id, node.transform, kind=kind))
        geometry_id = ids.geometry_by_node_id.get(node.id)
        if geometry_id is None or node.part_id is None:
            continue
        part = asset.parts.get(node.part_id)
        if part is None or part.mesh is None:
            continue
        lines.extend(_geometry_object(geometry_id, part, part.mesh, options))
    lines.append("}\n")
    lines.append("Connections:  {\n")
    lines.append(f'\tC: "OO",{ids.root_model},0\n')
    _connection_lines(lines, asset.root, ids, parent_model_id=ids.root_model, asset=asset, options=options)
    lines.append("}\n")
    return "".join(lines)


def _header() -> list[str]:
    return [
        "; FBX 7.4.0 project file\n",
        "; Generated by fascat\n",
        "FBXHeaderExtension:  {\n",
        "\tFBXHeaderVersion: 1003\n",
        "\tFBXVersion: 7400\n",
        '\tCreator: "fascat"\n',
        "}\n",
        'FileId: "fascat"\n',
        'CreationTime: "1970-01-01 00:00:00:000"\n',
        'Creator: "fascat"\n',
    ]


def _global_settings(asset: Asset) -> list[str]:
    unit_scale = max(asset.meters_per_unit * 100.0, 1e-12)
    axis = 1 if asset.up_axis == "Y" else 2
    return [
        "GlobalSettings:  {\n",
        "\tVersion: 1000\n",
        "\tProperties70:  {\n",
        f'\t\tP: "UpAxis", "int", "Integer", "",{axis}\n',
        '\t\tP: "UpAxisSign", "int", "Integer", "",1\n',
        '\t\tP: "FrontAxis", "int", "Integer", "",1\n',
        '\t\tP: "FrontAxisSign", "int", "Integer", "",1\n',
        '\t\tP: "CoordAxis", "int", "Integer", "",0\n',
        '\t\tP: "CoordAxisSign", "int", "Integer", "",1\n',
        f'\t\tP: "UnitScaleFactor", "double", "Number", "",{_num(unit_scale)}\n',
        "\t}\n",
        "}\n",
    ]


def _model_object(object_id: int, name: str, transform: FloatArray, *, kind: str) -> list[str]:
    translation, rotation, scale = _decompose_transform(transform)
    return [
        f'\tModel: {object_id}, "Model::{_escape_name(name)}", "{kind}" {{\n',
        "\t\tVersion: 232\n",
        "\t\tProperties70:  {\n",
        f'\t\t\tP: "Lcl Translation", "Lcl Translation", "", "A",{_array(translation)}\n',
        f'\t\t\tP: "Lcl Rotation", "Lcl Rotation", "", "A",{_array(rotation)}\n',
        f'\t\t\tP: "Lcl Scaling", "Lcl Scaling", "", "A",{_array(scale)}\n',
        "\t\t}\n",
        "\t}\n",
    ]


def _material_object(object_id: int, material: Material) -> list[str]:
    r, g, b, _alpha = material.base_color
    opacity = material.effective_opacity
    shininess = max(1.0, (1.0 - material.roughness) * 100.0)
    reflectivity = max(0.0, min(1.0, material.metallic))
    return [
        f'\tMaterial: {object_id}, "Material::{_escape_name(material.name or material.id)}", "" {{\n',
        "\t\tVersion: 102\n",
        '\t\tShadingModel: "phong"\n',
        "\t\tMultiLayer: 0\n",
        "\t\tProperties70:  {\n",
        f'\t\t\tP: "DiffuseColor", "Color", "", "A",{_num(r)},{_num(g)},{_num(b)}\n',
        f'\t\t\tP: "Opacity", "double", "Number", "A",{_num(opacity)}\n',
        f'\t\t\tP: "Shininess", "double", "Number", "A",{_num(shininess)}\n',
        f'\t\t\tP: "ReflectionFactor", "double", "Number", "A",{_num(reflectivity)}\n',
        "\t\t}\n",
        "\t}\n",
    ]


def _geometry_object(object_id: int, part: Part, mesh: Mesh, options: FbxExportOptions) -> list[str]:
    faces = mesh.faces.astype(np.int64, copy=False)
    polygon_indices = faces.copy()
    if polygon_indices.size:
        polygon_indices[:, -1] = -(polygon_indices[:, -1] + 1)
    lines = [
        f'\tGeometry: {object_id}, "Geometry::{_escape_name(part.name or part.id)}", "Mesh" {{\n',
        "\t\tGeometryVersion: 124\n",
        f"\t\tVertices: *{mesh.vertex_count * 3} {{\n",
        f"\t\t\ta: {_flat_array(mesh.points)}\n",
        "\t\t}\n",
        f"\t\tPolygonVertexIndex: *{mesh.triangle_count * 3} {{\n",
        f"\t\t\ta: {_flat_array(polygon_indices)}\n",
        "\t\t}\n",
    ]
    if options.normals:
        lines.extend(_normal_layer(mesh))
    if options.tangents and mesh.tangents is not None:
        lines.extend(_tangent_layer(mesh))
    if options.uvs:
        for channel in sorted(mesh.uvs):
            lines.extend(_uv_layer(mesh, channel))
    write_materials = options.materials and bool(part.material_ids)
    if write_materials:
        lines.extend(_material_layer(mesh, material_count=len(part.material_ids)))
    lines.extend(_layer_stack(mesh, options, write_materials=write_materials))
    lines.append("\t}\n")
    return lines


def _normal_layer(mesh: Mesh) -> list[str]:
    values = mesh.normals[mesh.faces] if mesh.normals is not None else face_normals(mesh.points, mesh.faces)[:, None, :]
    if mesh.normals is None:
        values = np.repeat(values, 3, axis=1)
    flat = values.reshape((-1, 3))
    return [
        "\t\tLayerElementNormal: 0 {\n",
        "\t\t\tVersion: 101\n",
        '\t\t\tName: ""\n',
        '\t\t\tMappingInformationType: "ByPolygonVertex"\n',
        '\t\t\tReferenceInformationType: "Direct"\n',
        f"\t\t\tNormals: *{len(flat) * 3} {{\n",
        f"\t\t\t\ta: {_flat_array(flat)}\n",
        "\t\t\t}\n",
        "\t\t}\n",
    ]


def _tangent_layer(mesh: Mesh) -> list[str]:
    if mesh.tangents is None:
        return []
    tangents = mesh.tangents[:, :3][mesh.faces].reshape((-1, 3))
    return [
        "\t\tLayerElementTangent: 0 {\n",
        "\t\t\tVersion: 101\n",
        '\t\t\tName: ""\n',
        '\t\t\tMappingInformationType: "ByPolygonVertex"\n',
        '\t\t\tReferenceInformationType: "Direct"\n',
        f"\t\t\tTangents: *{len(tangents) * 3} {{\n",
        f"\t\t\t\ta: {_flat_array(tangents)}\n",
        "\t\t\t}\n",
        "\t\t}\n",
    ]


def _uv_layer(mesh: Mesh, channel: int) -> list[str]:
    uv = mesh.uvs[channel]
    indices = mesh.faces.reshape(-1)
    return [
        f"\t\tLayerElementUV: {channel} {{\n",
        "\t\t\tVersion: 101\n",
        f'\t\t\tName: "UVChannel_{channel}"\n',
        '\t\t\tMappingInformationType: "ByPolygonVertex"\n',
        '\t\t\tReferenceInformationType: "IndexToDirect"\n',
        f"\t\t\tUV: *{uv.shape[0] * 2} {{\n",
        f"\t\t\t\ta: {_flat_array(uv)}\n",
        "\t\t\t}\n",
        f"\t\t\tUVIndex: *{indices.shape[0]} {{\n",
        f"\t\t\t\ta: {_flat_array(indices)}\n",
        "\t\t\t}\n",
        "\t\t}\n",
    ]


def _material_layer(mesh: Mesh, *, material_count: int) -> list[str]:
    indices = (
        mesh.material_indices if mesh.material_indices is not None else np.zeros(mesh.triangle_count, dtype=np.int64)
    )
    indices = np.where(indices < material_count, indices, -1)
    return [
        "\t\tLayerElementMaterial: 0 {\n",
        "\t\t\tVersion: 101\n",
        '\t\t\tName: ""\n',
        '\t\t\tMappingInformationType: "ByPolygon"\n',
        '\t\t\tReferenceInformationType: "IndexToDirect"\n',
        f"\t\t\tMaterials: *{indices.shape[0]} {{\n",
        f"\t\t\t\ta: {_flat_array(indices)}\n",
        "\t\t\t}\n",
        "\t\t}\n",
    ]


def _layer_stack(mesh: Mesh, options: FbxExportOptions, *, write_materials: bool) -> list[str]:
    elements = (
        ['\t\t\tLayerElement:  {\n\t\t\t\tType: "LayerElementNormal"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n']
        if options.normals
        else []
    )
    if options.tangents and mesh.tangents is not None:
        elements.append('\t\t\tLayerElement:  {\n\t\t\t\tType: "LayerElementTangent"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n')
    if options.uvs:
        for channel in sorted(mesh.uvs):
            elements.append(
                f'\t\t\tLayerElement:  {{\n\t\t\t\tType: "LayerElementUV"\n\t\t\t\tTypedIndex: {channel}\n\t\t\t}}\n'
            )
    if write_materials and mesh.triangle_count:
        elements.append(
            '\t\t\tLayerElement:  {\n\t\t\t\tType: "LayerElementMaterial"\n\t\t\t\tTypedIndex: 0\n\t\t\t}\n'
        )
    if not elements:
        return []
    return ["\t\tLayer: 0 {\n", *elements, "\t\t}\n"]


def _connection_lines(
    lines: list[str],
    node: Node,
    ids: _FbxObjectIds,
    *,
    parent_model_id: int,
    asset: Asset,
    options: FbxExportOptions,
) -> None:
    model_id = ids.model_by_node_id[node.id]
    lines.append(f'\tC: "OO",{model_id},{parent_model_id}\n')
    geometry_id = ids.geometry_by_node_id.get(node.id)
    if geometry_id is not None:
        lines.append(f'\tC: "OO",{geometry_id},{model_id}\n')
    if options.materials and node.part_id is not None:
        part = asset.parts.get(node.part_id)
        if part is not None:
            for material_id in part.material_ids:
                material_object_id = ids.material_by_id.get(material_id)
                if material_object_id is not None:
                    lines.append(f'\tC: "OO",{material_object_id},{model_id}\n')
    for child in node.children:
        _connection_lines(lines, child, ids, parent_model_id=model_id, asset=asset, options=options)


def _decompose_transform(transform: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    translation = np.asarray(transform[:3, 3], dtype=np.float64)
    linear = np.asarray(transform[:3, :3], dtype=np.float64)
    scale = np.linalg.norm(linear, axis=0)
    safe_scale = scale.copy()
    safe_scale[safe_scale == 0.0] = 1.0
    rotation_matrix = linear / safe_scale
    rotation = _matrix_to_euler_xyz(rotation_matrix)
    return translation, rotation, scale


def _matrix_to_euler_xyz(matrix: FloatArray) -> FloatArray:
    sy = float(np.clip(matrix[0, 2], -1.0, 1.0))
    y = np.arcsin(sy)
    cy = np.cos(y)
    if abs(cy) > 1e-8:
        x = np.arctan2(-matrix[1, 2], matrix[2, 2])
        z = np.arctan2(-matrix[0, 1], matrix[0, 0])
    else:
        x = np.arctan2(matrix[2, 1], matrix[1, 1])
        z = 0.0
    return np.degrees(np.array([x, y, z], dtype=np.float64))


def _array(values: FloatArray) -> str:
    return ",".join(_num(float(value)) for value in values)


def _flat_array(values: np.ndarray) -> str:
    flat = np.asarray(values).reshape(-1)
    if np.issubdtype(flat.dtype, np.integer):
        return ",".join(str(int(value)) for value in flat)
    return ",".join(_num(float(value)) for value in flat)


def _num(value: float) -> str:
    return f"{float(value):.9g}"


def _escape_name(value: str) -> str:
    return value.replace("\\", "_").replace('"', "_").replace("\x00", "_")


def _count_array_values(text: str, name: str) -> int:
    return len(_array_values(text, name))


def _array_values(text: str, name: str) -> list[int]:
    values: list[int] = []
    pattern = rf"{re.escape(name)}: \*\d+ \{{\s*a: ([^}}]*)\s*\}}"
    for match in re.finditer(pattern, text, flags=re.MULTILINE):
        for item in match.group(1).replace("\n", "").split(","):
            item = item.strip()
            if not item:
                continue
            values.append(int(float(item)))
    return values
