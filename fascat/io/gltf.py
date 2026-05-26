from __future__ import annotations

import base64
import json
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh

GLTF_SUFFIXES = {".gltf", ".glb"}

_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125

_COMPONENT_SIZES = {
    5120: 1,
    5121: 1,
    _UNSIGNED_SHORT: 2,
    5122: 2,
    _UNSIGNED_INT: 4,
    _FLOAT: 4,
}
_ACCESSOR_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def write_gltf(asset: Asset, path: str | Path) -> None:
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in GLTF_SUFFIXES:
        raise ValueError(f"unsupported glTF extension: {output_path.suffix or '<none>'}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document, binary = _build_document(asset, binary_uri=suffix == ".gltf")
    if suffix == ".glb":
        output_path.write_bytes(_pack_glb(document, binary))
        return
    output_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def validate_gltf(path: str | Path) -> dict[str, int]:
    document, buffers = _read_document(Path(path))
    asset_info = _object(document.get("asset"), "glTF asset metadata")
    if asset_info.get("version") != "2.0":
        raise RuntimeError("glTF asset version must be 2.0")

    _validate_buffers(document, buffers)
    stats = _validate_default_scene(document)
    if stats["meshes"] == 0:
        raise RuntimeError("glTF asset contains no meshes in default scene")
    return stats


@dataclass
class _BufferBuilder:
    data: bytearray = field(default_factory=bytearray)
    buffer_views: list[dict[str, object]] = field(default_factory=list)
    accessors: list[dict[str, object]] = field(default_factory=list)

    def add_accessor(
        self,
        values: NDArray[Any],
        *,
        component_type: int,
        accessor_type: Literal["SCALAR", "VEC2", "VEC3"],
        target: int,
        minimum: list[float] | None = None,
        maximum: list[float] | None = None,
    ) -> int:
        self._align()
        contiguous = np.ascontiguousarray(values)
        byte_offset = len(self.data)
        payload = contiguous.tobytes()
        self.data.extend(payload)
        buffer_view: dict[str, object] = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(payload),
            "target": target,
        }
        self.buffer_views.append(buffer_view)
        accessor: dict[str, object] = {
            "bufferView": len(self.buffer_views) - 1,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(contiguous.shape[0]),
            "type": accessor_type,
        }
        if minimum is not None:
            accessor["min"] = minimum
        if maximum is not None:
            accessor["max"] = maximum
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def _align(self) -> None:
        padding = (-len(self.data)) % 4
        if padding:
            self.data.extend(b"\x00" * padding)


@dataclass(frozen=True)
class _ExportSpace:
    linear: NDArray[np.float64]
    normal_linear: NDArray[np.float64]
    matrix: NDArray[np.float64]
    inverse_matrix: NDArray[np.float64]


def _build_document(asset: Asset, *, binary_uri: bool) -> tuple[dict[str, Any], bytes]:
    export_space = _export_space(asset)
    builder = _BufferBuilder()
    material_indices = _write_materials(asset.materials)
    meshes: list[dict[str, Any]] = []
    part_meshes: dict[str, int] = {}
    part_lods: dict[str, list[int]] = {}

    for part in asset.parts.values():
        if part.mesh is None:
            continue
        part_meshes[part.id] = _append_mesh(builder, meshes, part, part.mesh, material_indices, export_space, lod=0)
        lod_indices: list[int] = []
        for lod, lod_mesh in enumerate(part.lod_meshes, start=1):
            lod_indices.append(_append_mesh(builder, meshes, part, lod_mesh, material_indices, export_space, lod=lod))
        if lod_indices:
            part_lods[part.id] = lod_indices

    nodes: list[dict[str, Any]] = []
    root_node = _append_node(nodes, asset.root, part_meshes, part_lods, export_space)
    binary = bytes(builder.data)
    buffers: list[dict[str, object]] = [{"byteLength": len(binary)}]
    if binary_uri:
        buffers[0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binary).decode("ascii")

    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "fascat"},
        "scene": 0,
        "scenes": [{"name": asset.root.name or "Scene", "nodes": [root_node]}],
        "nodes": nodes,
        "buffers": buffers,
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
        "meshes": meshes,
        "extras": {
            "fascat": {
                "units": asset.units,
                "metersPerUnit": asset.meters_per_unit,
                "sourceUpAxis": asset.up_axis,
                "exportUnits": "metre",
                "exportUpAxis": "Y",
            }
        },
    }
    if material_indices:
        document["materials"] = [
            {key: value for key, value in material.items() if key != "_fascat_index"}
            for material in material_indices.values()
        ]
    return document, binary


def _write_materials(materials: dict[str, Material]) -> dict[str, dict[str, Any]]:
    written: dict[str, dict[str, Any]] = {}
    for index, material in enumerate(materials.values()):
        gltf_material: dict[str, Any] = {
            "name": material.name or material.id,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(material.base_color),
                "metallicFactor": material.metallic,
                "roughnessFactor": material.roughness,
            },
            "extras": {"fascat": {"materialId": material.id}},
        }
        if material.opacity < 1.0 or material.base_color[3] < 1.0:
            gltf_material["alphaMode"] = "BLEND"
        gltf_material["_fascat_index"] = index
        written[material.id] = gltf_material
    return written


def _append_mesh(
    builder: _BufferBuilder,
    meshes: list[dict[str, Any]],
    part: Part,
    mesh: Mesh,
    material_indices: dict[str, dict[str, Any]],
    export_space: _ExportSpace,
    *,
    lod: int,
) -> int:
    mesh.validate()
    points = _points_to_export_space(mesh.points, export_space.linear).astype(np.float32)
    position_accessor = builder.add_accessor(
        points,
        component_type=_FLOAT,
        accessor_type="VEC3",
        target=_ARRAY_BUFFER,
        minimum=points.min(axis=0).astype(float).tolist() if len(points) else [0.0, 0.0, 0.0],
        maximum=points.max(axis=0).astype(float).tolist() if len(points) else [0.0, 0.0, 0.0],
    )
    attributes: dict[str, int] = {"POSITION": position_accessor}
    if mesh.normals is not None:
        normals = _normals_to_export_space(mesh.normals, export_space.normal_linear).astype(np.float32)
        attributes["NORMAL"] = builder.add_accessor(
            normals,
            component_type=_FLOAT,
            accessor_type="VEC3",
            target=_ARRAY_BUFFER,
        )
    if 0 in mesh.uvs:
        attributes["TEXCOORD_0"] = builder.add_accessor(
            mesh.uvs[0].astype(np.float32),
            component_type=_FLOAT,
            accessor_type="VEC2",
            target=_ARRAY_BUFFER,
        )
    if 1 in mesh.uvs:
        attributes["TEXCOORD_1"] = builder.add_accessor(
            mesh.uvs[1].astype(np.float32),
            component_type=_FLOAT,
            accessor_type="VEC2",
            target=_ARRAY_BUFFER,
        )

    primitives: list[dict[str, Any]] = []
    for material_id, faces in _face_groups(part, mesh):
        if faces.size == 0:
            continue
        indices = mesh.faces[faces].reshape(-1)
        if mesh.vertex_count <= np.iinfo(np.uint16).max:
            index_values = indices.astype(np.uint16)
            component_type = _UNSIGNED_SHORT
        else:
            index_values = indices.astype(np.uint32)
            component_type = _UNSIGNED_INT
        primitive: dict[str, Any] = {
            "attributes": attributes,
            "indices": builder.add_accessor(
                index_values.reshape(-1, 1),
                component_type=component_type,
                accessor_type="SCALAR",
                target=_ELEMENT_ARRAY_BUFFER,
            ),
            "mode": 4,
        }
        if material_id is not None and material_id in material_indices:
            primitive["material"] = int(material_indices[material_id]["_fascat_index"])
        primitives.append(primitive)

    gltf_mesh: dict[str, Any] = {
        "name": f"{part.name or part.id}_lod{lod}",
        "primitives": primitives,
        "extras": {"fascat": {"partId": part.id, "originalName": part.name, "lod": lod}},
    }
    meshes.append(gltf_mesh)
    return len(meshes) - 1


def _face_groups(part: Part, mesh: Mesh) -> list[tuple[str | None, NDArray[np.int64]]]:
    all_faces = np.arange(mesh.triangle_count, dtype=np.int64)
    if mesh.material_indices is None:
        material_id = part.material_ids[0] if part.material_ids else None
        return [(material_id, all_faces)]
    groups: list[tuple[str | None, NDArray[np.int64]]] = []
    for material_index in sorted(set(mesh.material_indices.astype(int).tolist())):
        material_id = part.material_ids[material_index] if material_index < len(part.material_ids) else None
        groups.append((material_id, np.flatnonzero(mesh.material_indices == material_index).astype(np.int64)))
    return groups


def _append_node(
    nodes: list[dict[str, Any]],
    node: Node,
    part_meshes: dict[str, int],
    part_lods: dict[str, list[int]],
    export_space: _ExportSpace,
) -> int:
    gltf_node: dict[str, Any] = {
        "name": node.name or node.id,
        "extras": {"fascat": {"nodeId": node.id, **node.metadata}},
    }
    transform = _matrix_to_export_space(node.transform, export_space)
    if not np.allclose(transform, np.eye(4)):
        gltf_node["matrix"] = transform.T.reshape(-1).astype(float).tolist()
    if node.part_id is not None and node.part_id in part_meshes:
        gltf_node["mesh"] = part_meshes[node.part_id]
        lods = part_lods.get(node.part_id)
        if lods:
            gltf_node["extras"]["fascat"]["lodMeshIndices"] = lods
    index = len(nodes)
    nodes.append(gltf_node)
    children = [_append_node(nodes, child, part_meshes, part_lods, export_space) for child in node.children]
    if children:
        gltf_node["children"] = children
    return index


def _export_space(asset: Asset) -> _ExportSpace:
    axis = np.eye(3, dtype=np.float64)
    if asset.up_axis == "Z":
        axis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
    linear = axis * float(asset.meters_per_unit)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    inverse = np.asarray(np.linalg.inv(matrix), dtype=np.float64)
    return _ExportSpace(
        linear=linear,
        normal_linear=axis,
        matrix=matrix,
        inverse_matrix=inverse,
    )


def _points_to_export_space(points: NDArray[np.float64], linear: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], points @ linear.T)


def _normals_to_export_space(normals: NDArray[np.float64], linear: NDArray[np.float64]) -> NDArray[np.float64]:
    transformed = cast(NDArray[np.float64], normals @ linear.T)
    lengths = np.linalg.norm(transformed, axis=1)
    valid = lengths > 0.0
    transformed[valid] = transformed[valid] / lengths[valid, None]
    transformed[~valid] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return transformed


def _matrix_to_export_space(matrix: NDArray[np.float64], export_space: _ExportSpace) -> NDArray[np.float64]:
    return export_space.matrix @ matrix @ export_space.inverse_matrix


def _pack_glb(document: dict[str, Any], binary: bytes) -> bytes:
    json_payload = json.dumps(document, separators=(",", ":"), sort_keys=False).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    bin_payload = binary + b"\x00" * ((-len(binary)) % 4)
    length = 12 + 8 + len(json_payload) + 8 + len(bin_payload)
    return b"".join(
        [
            struct.pack("<4sII", _GLB_MAGIC, _GLB_VERSION, length),
            struct.pack("<II", len(json_payload), _JSON_CHUNK),
            json_payload,
            struct.pack("<II", len(bin_payload), _BIN_CHUNK),
            bin_payload,
        ]
    )


def _read_document(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return _read_glb(path)
    if suffix == ".gltf":
        return _read_gltf(path)
    raise RuntimeError(f"unsupported glTF extension: {path.suffix or '<none>'}")


def _read_glb(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise RuntimeError("invalid GLB header")
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    if magic != _GLB_MAGIC or version != _GLB_VERSION:
        raise RuntimeError("invalid GLB header")
    if length != len(payload):
        raise RuntimeError("invalid GLB length")
    offset = 12
    document: dict[str, Any] | None = None
    binary = b""
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise RuntimeError("invalid GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise RuntimeError("invalid GLB chunk length")
        offset += chunk_length
        if chunk_type == _JSON_CHUNK:
            document = cast(dict[str, Any], json.loads(chunk.decode("utf-8").rstrip(" \x00")))
        elif chunk_type == _BIN_CHUNK:
            binary = chunk
    if document is None:
        raise RuntimeError("GLB contains no JSON chunk")
    return document, [binary]


def _read_gltf(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    buffers: list[bytes] = []
    for index, buffer in enumerate(_array(document.get("buffers"), "buffers")):
        buffer_object = _object(buffer, f"buffer {index}")
        uri = buffer_object.get("uri")
        if not isinstance(uri, str) or not uri.startswith("data:"):
            raise RuntimeError("glTF validation only supports embedded data URI buffers")
        try:
            _, encoded = uri.split(",", 1)
            buffers.append(base64.b64decode(encoded))
        except ValueError as exc:
            raise RuntimeError("invalid glTF data URI buffer") from exc
    return document, buffers


def _validate_buffers(document: dict[str, Any], buffers: list[bytes]) -> None:
    buffer_objects = _array(document.get("buffers"), "buffers")
    if len(buffer_objects) != len(buffers):
        raise RuntimeError("glTF buffer count does not match payload count")
    for index, buffer in enumerate(buffer_objects):
        buffer_length = _int(_object(buffer, f"buffer {index}").get("byteLength"), f"buffer {index} byteLength")
        if buffer_length > len(buffers[index]):
            raise RuntimeError(f"glTF buffer {index} is shorter than its byteLength")
    for index, buffer_view in enumerate(_array(document.get("bufferViews"), "bufferViews")):
        view = _object(buffer_view, f"bufferView {index}")
        buffer_index = _int(view.get("buffer", 0), f"bufferView {index} buffer")
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise RuntimeError(f"glTF bufferView {index} references an invalid buffer")
        offset = _int(view.get("byteOffset", 0), f"bufferView {index} byteOffset")
        length = _int(view.get("byteLength"), f"bufferView {index} byteLength")
        if offset < 0 or length < 0 or offset + length > len(buffers[buffer_index]):
            raise RuntimeError(f"glTF bufferView {index} is out of range")
    for index, accessor in enumerate(_array(document.get("accessors"), "accessors")):
        _validate_accessor_storage(document, index, _object(accessor, f"accessor {index}"))


def _validate_accessor_storage(document: dict[str, Any], index: int, accessor: dict[str, Any]) -> None:
    buffer_views = _array(document.get("bufferViews"), "bufferViews")
    view_index = _int(accessor.get("bufferView"), f"accessor {index} bufferView")
    if view_index < 0 or view_index >= len(buffer_views):
        raise RuntimeError(f"glTF accessor {index} references an invalid bufferView")
    view = _object(buffer_views[view_index], f"bufferView {view_index}")
    count = _int(accessor.get("count"), f"accessor {index} count")
    component_type = _int(accessor.get("componentType"), f"accessor {index} componentType")
    accessor_type = accessor.get("type")
    if component_type not in _COMPONENT_SIZES:
        raise RuntimeError(f"glTF accessor {index} uses unsupported componentType")
    if accessor_type not in _ACCESSOR_WIDTHS:
        raise RuntimeError(f"glTF accessor {index} uses unsupported type")
    byte_offset = _int(accessor.get("byteOffset", 0), f"accessor {index} byteOffset")
    stride = _int(view.get("byteStride", 0), f"bufferView {view_index} byteStride") if "byteStride" in view else 0
    item_size = _COMPONENT_SIZES[component_type] * _ACCESSOR_WIDTHS[cast(str, accessor_type)]
    needed = byte_offset
    if count > 0:
        needed += (count - 1) * stride + item_size if stride else count * item_size
    if count < 0 or byte_offset < 0 or needed > _int(view.get("byteLength"), f"bufferView {view_index} byteLength"):
        raise RuntimeError(f"glTF accessor {index} is out of range")


def _validate_default_scene(document: dict[str, Any]) -> dict[str, int]:
    scenes = _array(document.get("scenes"), "scenes")
    nodes = _array(document.get("nodes"), "nodes")
    scene_index = _int(document.get("scene", 0), "default scene")
    if scene_index < 0 or scene_index >= len(scenes):
        raise RuntimeError("glTF default scene index is invalid")
    stats = {"meshes": 0, "points": 0, "triangles": 0}
    for node_index in _array(_object(scenes[scene_index], f"scene {scene_index}").get("nodes", []), "scene nodes"):
        _walk_node(document, _int(node_index, "scene node"), stats, stack=set())
    if len(nodes) == 0:
        raise RuntimeError("glTF asset contains no nodes")
    return stats


def _walk_node(document: dict[str, Any], node_index: int, stats: dict[str, int], *, stack: set[int]) -> None:
    nodes = _array(document.get("nodes"), "nodes")
    if node_index < 0 or node_index >= len(nodes):
        raise RuntimeError("glTF node index is invalid")
    if node_index in stack:
        raise RuntimeError("glTF node hierarchy contains a cycle")
    node = _object(nodes[node_index], f"node {node_index}")
    mesh_index = node.get("mesh")
    if mesh_index is not None:
        _validate_mesh(document, _int(mesh_index, f"node {node_index} mesh"), stats)
    for child_index in _array(node.get("children", []), f"node {node_index} children"):
        _walk_node(document, _int(child_index, f"node {node_index} child"), stats, stack=stack | {node_index})


def _validate_mesh(document: dict[str, Any], mesh_index: int, stats: dict[str, int]) -> None:
    meshes = _array(document.get("meshes"), "meshes")
    if mesh_index < 0 or mesh_index >= len(meshes):
        raise RuntimeError("glTF mesh index is invalid")
    mesh = _object(meshes[mesh_index], f"mesh {mesh_index}")
    position_accessors: set[int] = set()
    triangles = 0
    for primitive_index, primitive_value in enumerate(_array(mesh.get("primitives"), f"mesh {mesh_index} primitives")):
        primitive = _object(primitive_value, f"mesh {mesh_index} primitive {primitive_index}")
        if _int(primitive.get("mode", 4), f"mesh {mesh_index} primitive mode") != 4:
            raise RuntimeError("glTF validation only supports triangle primitives")
        attributes = _object(primitive.get("attributes"), f"mesh {mesh_index} primitive attributes")
        position_index = _int(attributes.get("POSITION"), f"mesh {mesh_index} POSITION accessor")
        _require_accessor(document, position_index, component_type=_FLOAT, accessor_type="VEC3")
        position_accessors.add(position_index)
        indices = primitive.get("indices")
        if indices is None:
            position_count = _accessor_count(document, position_index)
            if position_count % 3:
                raise RuntimeError("glTF non-indexed triangle primitive has invalid vertex count")
            triangles += position_count // 3
        else:
            index = _int(indices, f"mesh {mesh_index} indices accessor")
            accessor = _require_accessor(document, index, accessor_type="SCALAR")
            component_type = _int(accessor.get("componentType"), f"accessor {index} componentType")
            if component_type not in {_UNSIGNED_SHORT, _UNSIGNED_INT}:
                raise RuntimeError("glTF index accessor must use unsigned integer components")
            index_count = _int(accessor.get("count"), f"accessor {index} count")
            if index_count % 3:
                raise RuntimeError("glTF indexed triangle primitive has invalid index count")
            triangles += index_count // 3
    stats["meshes"] += 1
    stats["points"] += sum(_accessor_count(document, accessor) for accessor in position_accessors)
    stats["triangles"] += triangles


def _require_accessor(
    document: dict[str, Any],
    accessor_index: int,
    *,
    component_type: int | None = None,
    accessor_type: str | None = None,
) -> dict[str, Any]:
    accessors = _array(document.get("accessors"), "accessors")
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise RuntimeError("glTF accessor index is invalid")
    accessor = _object(accessors[accessor_index], f"accessor {accessor_index}")
    if (
        component_type is not None
        and _int(accessor.get("componentType"), f"accessor {accessor_index} componentType") != component_type
    ):
        raise RuntimeError("glTF accessor has the wrong component type")
    if accessor_type is not None and accessor.get("type") != accessor_type:
        raise RuntimeError("glTF accessor has the wrong type")
    return accessor


def _accessor_count(document: dict[str, Any], accessor_index: int) -> int:
    accessor = _require_accessor(document, accessor_index)
    return _int(accessor.get("count"), f"accessor {accessor_index} count")


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"glTF {label} must be an array")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"glTF {label} must be an object")
    return value


def _int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise RuntimeError(f"glTF {label} must be an integer")
    return value
