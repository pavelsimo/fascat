"""JT Logical Scene Graph parsing: elements, attributes, property atoms, property table.

Element layouts and object-type GUIDs are transcribed from the public Siemens
*JT File Format Reference* v9.5 sections 7.1.3.2 and 7.2.1 and Appendix A.
Unknown or out-of-scope elements are skipped by their length field and counted,
so vendor extensions never hard-fail the import.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from fascat.io.jt.container import ByteReader


def _guid(part1: int, part2: int, part3: int, *tail: int) -> bytes:
    return struct.pack("<IHH", part1, part2, part3) + bytes(tail)


END_OF_ELEMENTS = b"\xff" * 16

GUID_BASE_NODE = _guid(0x10DD1035, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_GROUP_NODE = _guid(0x10DD101B, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_INSTANCE_NODE = _guid(0x10DD102A, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_LOD_NODE = _guid(0x10DD102C, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_META_DATA_NODE = _guid(0xCE357245, 0x38FB, 0x11D1, 0xA5, 0x06, 0x00, 0x60, 0x97, 0xBD, 0xC6, 0xE1)
GUID_NULL_SHAPE_NODE = _guid(0xD239E7B6, 0xDD77, 0x4289, 0xA0, 0x7D, 0xB0, 0xEE, 0x79, 0xF7, 0x94, 0x94)
GUID_PART_NODE = _guid(0xCE357244, 0x38FB, 0x11D1, 0xA5, 0x06, 0x00, 0x60, 0x97, 0xBD, 0xC6, 0xE1)
GUID_PARTITION_NODE = _guid(0x10DD103E, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_RANGE_LOD_NODE = _guid(0x10DD104C, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_SWITCH_NODE = _guid(0x10DD10F3, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_BASE_SHAPE_NODE = _guid(0x10DD1059, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_POINT_SET_SHAPE_NODE = _guid(0x98134716, 0x0010, 0x0818, 0x19, 0x98, 0x08, 0x00, 0x09, 0x83, 0x5D, 0x5A)
GUID_POLYGON_SET_SHAPE_NODE = _guid(0x10DD1048, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_POLYLINE_SET_SHAPE_NODE = _guid(0x10DD1046, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_TRISTRIP_SET_SHAPE_NODE = _guid(0x10DD1077, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_VERTEX_SHAPE_NODE = _guid(0x10DD107F, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)

GUID_MATERIAL_ATTRIBUTE = _guid(0x10DD1030, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_TRANSFORM_ATTRIBUTE = _guid(0x10DD1083, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)

GUID_BASE_PROPERTY_ATOM = _guid(0x10DD104B, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_STRING_PROPERTY_ATOM = _guid(0x10DD106E, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_INTEGER_PROPERTY_ATOM = _guid(0x10DD102B, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_FLOAT_PROPERTY_ATOM = _guid(0x10DD1019, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_OBJECT_REF_PROPERTY_ATOM = _guid(0x10DD1004, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_DATE_PROPERTY_ATOM = _guid(0xCE357246, 0x38FB, 0x11D1, 0xA5, 0x06, 0x00, 0x60, 0x97, 0xBD, 0xC6, 0xE1)
GUID_LATE_LOADED_PROPERTY_ATOM = _guid(0xE0B05BE5, 0xFBBD, 0x11D1, 0xA3, 0xA7, 0x00, 0xAA, 0x00, 0xD1, 0x09, 0x54)
GUID_VECTOR4F_PROPERTY_ATOM = _guid(0x2E7DB4BE, 0xC71A, 0x4B18, 0x9D, 0x07, 0xC7, 0x22, 0x7E, 0x9F, 0xEF, 0x76)

GUID_TRISTRIP_SET_SHAPE_LOD = _guid(0x10DD10AB, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_TOPOMESH_TOPO_COMPRESSED_LOD = _guid(0xF830A5AD, 0xBE4C, 0x4FBC, 0x9B, 0x5F, 0xB9, 0x26, 0x92, 0x78, 0xD2, 0xE1)
GUID_POLYLINE_SET_SHAPE_LOD = _guid(0x10DD10A1, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
GUID_POINT_SET_SHAPE_LOD = _guid(0x98134716, 0x0011, 0x0818, 0x19, 0x98, 0x08, 0x00, 0x09, 0x83, 0x5D, 0x5A)
GUID_VERTEX_SHAPE_LOD = _guid(0x10DD10B0, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)

_SHAPE_NODE_KINDS = {
    GUID_BASE_SHAPE_NODE: "base_shape",
    GUID_NULL_SHAPE_NODE: "null_shape",
    GUID_POINT_SET_SHAPE_NODE: "point_shape",
    GUID_POLYGON_SET_SHAPE_NODE: "polygon_shape",
    GUID_POLYLINE_SET_SHAPE_NODE: "polyline_shape",
    GUID_TRISTRIP_SET_SHAPE_NODE: "tristrip_shape",
    GUID_VERTEX_SHAPE_NODE: "vertex_shape",
}

SHAPE_NODE_KINDS = frozenset(_SHAPE_NODE_KINDS.values())


@dataclass
class LsgNode:
    object_id: int
    kind: str
    attribute_ids: tuple[int, ...] = ()
    child_ids: tuple[int, ...] = ()
    file_name: str | None = None  # partitions: external (shattered) file reference
    range_limits: tuple[float, ...] = ()


@dataclass(frozen=True)
class JtMaterial:
    object_id: int
    ambient: tuple[float, float, float, float]
    diffuse: tuple[float, float, float, float]
    specular: tuple[float, float, float, float]
    emission: tuple[float, float, float, float]
    shininess: float
    reflectivity: float | None = None


@dataclass(frozen=True)
class JtTransform:
    object_id: int
    matrix: npt.NDArray[np.float64]  # 4x4, row-vector convention (p' = pAM)


@dataclass(frozen=True)
class LateLoadedRef:
    segment_id: bytes
    segment_type: int
    payload_object_id: int


@dataclass(frozen=True)
class PropertyAtom:
    object_id: int
    value: object


@dataclass
class Lsg:
    root_id: int | None = None
    nodes: dict[int, LsgNode] = field(default_factory=dict)
    materials: dict[int, JtMaterial] = field(default_factory=dict)
    transforms: dict[int, JtTransform] = field(default_factory=dict)
    atoms: dict[int, PropertyAtom] = field(default_factory=dict)
    properties: dict[int, dict[str, object]] = field(default_factory=dict)
    late_loaded: dict[int, list[LateLoadedRef]] = field(default_factory=dict)
    skipped_elements: Counter[str] = field(default_factory=Counter)


def parse_lsg(payload: bytes, *, version: tuple[int, int], byte_order: str) -> Lsg:
    """Parse a decompressed LSG segment payload into an intermediate graph."""
    reader = ByteReader(payload, byte_order=byte_order)
    lsg = Lsg()
    _parse_graph_section(reader, version, lsg)
    _parse_property_atom_section(reader, version, lsg)
    _parse_property_table(reader, version, lsg)
    return lsg


def _element_version(reader: ByteReader, version: tuple[int, int]) -> int:
    """In-element version fields are absent in JT 8, I16 in JT 9, U8 in JT 10."""
    if version[0] < 9:
        return 0
    return reader.i16() if version[0] < 10 else reader.u8()


def _parse_graph_section(reader: ByteReader, version: tuple[int, int], lsg: Lsg) -> None:
    while True:
        length = reader.i32()
        start = reader.tell()
        guid = reader.guid()
        if guid == END_OF_ELEMENTS:
            reader.seek(start + length)
            return
        reader.u8()  # object base type
        object_id = reader.i32()
        end = start + length
        if guid == GUID_MATERIAL_ATTRIBUTE:
            lsg.materials[object_id] = _parse_material(reader, version, object_id)
        elif guid == GUID_TRANSFORM_ATTRIBUTE:
            lsg.transforms[object_id] = _parse_transform(reader, version, object_id)
        else:
            node = _parse_node(reader, version, guid, object_id)
            if node is not None:
                lsg.nodes[object_id] = node
                if lsg.root_id is None:
                    lsg.root_id = object_id
            else:
                lsg.skipped_elements[guid.hex()] += 1
        if reader.tell() > end:
            raise RuntimeError(f"corrupt JT data: element {guid.hex()} overran its length")
        reader.seek(end)


def _parse_node(reader: ByteReader, version: tuple[int, int], guid: bytes, object_id: int) -> LsgNode | None:
    if guid == GUID_BASE_NODE:
        attributes = _parse_base_node_data(reader, version)
        return LsgNode(object_id, "base_node", attributes)
    if guid == GUID_GROUP_NODE:
        attributes, children = _parse_group_node_data(reader, version)
        return LsgNode(object_id, "group", attributes, children)
    if guid == GUID_PARTITION_NODE:
        attributes, children = _parse_group_node_data(reader, version)
        if version[0] >= 10:
            reader.u8()  # Partition Node Element version (new in JT 10)
        reader.i32()  # partition flags
        file_name = reader.mbstring()
        return LsgNode(object_id, "partition", attributes, children, file_name=file_name or None)
    if guid == GUID_INSTANCE_NODE:
        attributes = _parse_base_node_data(reader, version)
        _element_version(reader, version)
        child = reader.i32()
        return LsgNode(object_id, "instance", attributes, (child,))
    if guid == GUID_PART_NODE:
        attributes, children = _parse_meta_data_node_data(reader, version)
        return LsgNode(object_id, "part", attributes, children)
    if guid == GUID_META_DATA_NODE:
        attributes, children = _parse_meta_data_node_data(reader, version)
        return LsgNode(object_id, "metadata", attributes, children)
    if guid == GUID_LOD_NODE:
        attributes, children = _parse_lod_node_data(reader, version)
        return LsgNode(object_id, "lod", attributes, children)
    if guid == GUID_RANGE_LOD_NODE:
        attributes, children = _parse_lod_node_data(reader, version)
        _element_version(reader, version)
        limits = tuple(float(value) for value in reader.vec_f32())
        return LsgNode(object_id, "range_lod", attributes, children, range_limits=limits)
    if guid == GUID_SWITCH_NODE:
        attributes, children = _parse_group_node_data(reader, version)
        return LsgNode(object_id, "switch", attributes, children)
    kind = _SHAPE_NODE_KINDS.get(guid)
    if kind is not None:
        attributes = _parse_base_node_data(reader, version)
        return LsgNode(object_id, kind, attributes)
    return None


def _parse_base_node_data(reader: ByteReader, version: tuple[int, int]) -> tuple[int, ...]:
    _element_version(reader, version)
    reader.u32()  # node flags
    count = reader.i32()
    if count < 0:
        raise RuntimeError(f"corrupt JT data: negative attribute count {count}")
    return tuple(reader.i32() for _ in range(count))


def _parse_group_node_data(reader: ByteReader, version: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    attributes = _parse_base_node_data(reader, version)
    _element_version(reader, version)
    count = reader.i32()
    if count < 0:
        raise RuntimeError(f"corrupt JT data: negative child count {count}")
    children = tuple(reader.i32() for _ in range(count))
    return attributes, children


def _parse_meta_data_node_data(reader: ByteReader, version: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    attributes, children = _parse_group_node_data(reader, version)
    _element_version(reader, version)
    return attributes, children


def _parse_lod_node_data(reader: ByteReader, version: tuple[int, int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    attributes, children = _parse_group_node_data(reader, version)
    _element_version(reader, version)
    reader.vec_f32()  # reserved
    reader.i32()  # reserved
    return attributes, children


def _parse_base_attribute_data(reader: ByteReader, version: tuple[int, int]) -> None:
    _element_version(reader, version)
    reader.u8()  # state flags
    reader.u32()  # field inhibit flags


def _rgba(reader: ByteReader) -> tuple[float, float, float, float]:
    return (reader.f32(), reader.f32(), reader.f32(), reader.f32())


def _parse_material(reader: ByteReader, version: tuple[int, int], object_id: int) -> JtMaterial:
    _parse_base_attribute_data(reader, version)
    material_version = _element_version(reader, version)
    data_flags = reader.u16()
    if version[0] < 9:
        return _parse_material_channels_v8(reader, object_id, data_flags)
    ambient = _rgba(reader)
    diffuse = _rgba(reader)
    specular = _rgba(reader)
    emission = _rgba(reader)
    shininess = reader.f32()
    reflectivity = reader.f32() if material_version >= 2 else None
    return JtMaterial(object_id, ambient, diffuse, specular, emission, shininess, reflectivity)


def _parse_material_channels_v8(reader: ByteReader, object_id: int, data_flags: int) -> JtMaterial:
    # JT 8 (spec 7.2.1.1.2.2): when data-flag bit 0x0001 is set, the pattern
    # bits mark ambient (0x0002), emission (0x0004), and specular (0x0008)
    # channels stored as a single common F32 meaning (c, c, c, 1.0); diffuse is
    # always full RGBA. JT 8 has no reflectivity field.
    patterns_valid = bool(data_flags & 0x0001)

    def channel(pattern_bit: int) -> tuple[float, float, float, float]:
        if patterns_valid and data_flags & pattern_bit:
            common = reader.f32()
            return (common, common, common, 1.0)
        return _rgba(reader)

    ambient = channel(0x0002)
    diffuse = _rgba(reader)
    specular = channel(0x0008)
    emission = channel(0x0004)
    shininess = reader.f32()
    return JtMaterial(object_id, ambient, diffuse, specular, emission, shininess, None)


def _parse_transform(reader: ByteReader, version: tuple[int, int], object_id: int) -> JtTransform:
    _parse_base_attribute_data(reader, version)
    _element_version(reader, version)
    mask = reader.u16()
    # Stored element values are F64 from JT 9 on (F32 in JT 8); the v9.5 Rev-A
    # prose is ambiguous but real Siemens 9.5/10 files store F64.
    read_value = reader.f64 if version[0] >= 9 else reader.f32
    matrix = np.identity(4, dtype=np.float64)
    for index in range(16):
        if mask & 0x8000:
            matrix[index // 4, index % 4] = read_value()
        mask = (mask << 1) & 0xFFFF
    # Real files (notably F32-era JT 8) carry rounding noise in the projective
    # column, which must be exactly (0, 0, 0, 1) for downstream exporters to
    # emit TRS-decomposable matrices.
    if np.all(np.abs(matrix[:, 3] - (0.0, 0.0, 0.0, 1.0)) < 1e-6):
        matrix[:, 3] = (0.0, 0.0, 0.0, 1.0)
    return JtTransform(object_id, matrix)


def _parse_property_atom_section(reader: ByteReader, version: tuple[int, int], lsg: Lsg) -> None:
    while True:
        length = reader.i32()
        start = reader.tell()
        guid = reader.guid()
        if guid == END_OF_ELEMENTS:
            reader.seek(start + length)
            return
        reader.u8()  # object base type
        object_id = reader.i32()
        end = start + length
        value = _parse_property_atom(reader, version, guid)
        if value is _UNKNOWN_ATOM:
            lsg.skipped_elements[guid.hex()] += 1
        else:
            lsg.atoms[object_id] = PropertyAtom(object_id, value)
        if reader.tell() > end:
            raise RuntimeError(f"corrupt JT data: property atom {guid.hex()} overran its length")
        reader.seek(end)


_UNKNOWN_ATOM = object()


def _parse_property_atom(reader: ByteReader, version: tuple[int, int], guid: bytes) -> object:
    if guid == GUID_BASE_PROPERTY_ATOM:
        return None
    _element_version(reader, version)
    reader.u32()  # state flags
    if guid == GUID_STRING_PROPERTY_ATOM:
        _element_version(reader, version)
        return reader.mbstring()
    if guid == GUID_INTEGER_PROPERTY_ATOM:
        _element_version(reader, version)
        return reader.i32()
    if guid == GUID_FLOAT_PROPERTY_ATOM:
        _element_version(reader, version)
        return reader.f32()
    if guid == GUID_OBJECT_REF_PROPERTY_ATOM:
        _element_version(reader, version)
        return ObjectRef(reader.i32())
    if guid == GUID_DATE_PROPERTY_ATOM:
        _element_version(reader, version)
        year, month, day, hour, minute, second = (reader.i16() for _ in range(6))
        return f"{year:04d}-{month + 1:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    if guid == GUID_LATE_LOADED_PROPERTY_ATOM:
        _element_version(reader, version)
        segment_id = reader.guid()
        segment_type = reader.i32()
        # The payload object id field was added in JT 9.
        payload_object_id = reader.i32() if version[0] >= 9 else -1
        return LateLoadedRef(segment_id, segment_type, payload_object_id)
    if guid == GUID_VECTOR4F_PROPERTY_ATOM:
        _element_version(reader, version)
        return (reader.f32(), reader.f32(), reader.f32(), reader.f32())
    return _UNKNOWN_ATOM


@dataclass(frozen=True)
class ObjectRef:
    object_id: int


def _parse_property_table(reader: ByteReader, version: tuple[int, int], lsg: Lsg) -> None:
    if reader.remaining() < 6:
        return  # tolerate files without a property table
    reader.i16()  # property table version: I16 in both JT 9 and JT 10 (spec 6.3)
    table_count = reader.i32()
    for _ in range(table_count):
        element_id = reader.i32()
        while True:
            key_id = reader.i32()
            if key_id == 0:
                break
            value_id = reader.i32()
            key_atom = lsg.atoms.get(key_id)
            value_atom = lsg.atoms.get(value_id)
            if key_atom is None or value_atom is None or not isinstance(key_atom.value, str):
                continue
            value = value_atom.value
            if isinstance(value, LateLoadedRef):
                lsg.late_loaded.setdefault(element_id, []).append(value)
            else:
                lsg.properties.setdefault(element_id, {})[key_atom.value] = value


def display_name(raw_name: str) -> str:
    """Strip the ``name;version;instance:`` encoding used by JT_PROP_NAME values."""
    trimmed = raw_name[:-1] if raw_name.endswith(":") else raw_name
    return trimmed.split(";", 1)[0]
