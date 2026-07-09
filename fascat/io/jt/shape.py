"""JT 9.5 Shape LOD decode: topologically compressed tri-strip set meshes.

JT 9 stores TriStripSet geometry as a topologically compressed dual mesh
(sections 7.2.2.1.2.4-7.2.2.1.2.6 of the reference); the decoder below is a
faithful Python translation of the reference implementation in Appendix E
(DualVFMesh / MeshCoderDriver / MeshCodec / MeshDecoder). Dual vertices are
primal triangles, dual faces are primal (topological) vertices; visit order
defines the vertex-coordinate and attribute-record numbering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fascat.io.jt import lsg as _lsg
from fascat.io.jt.codec import (
    combine_float_bits,
    decode_deering_normals,
    decode_int32_cdp2,
    dequantize_uniform,
    hash_float32_scalars,
    jt_hash16,
    jt_hash32,
)
from fascat.io.jt.container import ByteReader

_COORD_BINDING_MASK = 0x7  # bits 1-3: 2/3/4-component vertex coordinates
_NORMAL_BINDING_MASK = 0x8  # bit 4: 3-component normals


@dataclass
class DecodedShape:
    points: npt.NDArray[np.float64]  # (N, 3)
    faces: npt.NDArray[np.int64]  # (M, 3)
    normals: npt.NDArray[np.float64] | None  # (N, 3)
    face_groups: npt.NDArray[np.int64]  # (M,)


def decode_shape_lod(payload: bytes, *, version: tuple[int, int], byte_order: str) -> DecodedShape:
    """Decode a Shape LOD segment payload (element header included) into a triangle mesh."""
    if version[0] >= 10:
        raise RuntimeError("JT 10 mesh coding not yet supported")
    reader = ByteReader(payload, byte_order=byte_order)
    reader.i32()  # element length
    guid = reader.guid()
    if guid != _lsg.GUID_TRISTRIP_SET_SHAPE_LOD:
        raise RuntimeError(f"unsupported JT shape LOD element type: {guid.hex()}")
    reader.u8()  # object base type
    reader.i32()  # object id
    _lsg._element_version(reader, version)  # Vertex Shape LOD Data version
    reader.u64()  # vertex bindings (restated in the vertex records)
    _lsg._element_version(reader, version)  # TopoMesh LOD Data version
    reader.i32()  # vertex records object id
    _lsg._element_version(reader, version)  # Topologically Compressed LOD Data version
    symbols = _read_topological_symbols(reader)
    coordinates, normal_records = _read_vertex_records(reader)
    mesh = _TopologyDecoder(symbols).run()
    return _extract_primal_mesh(mesh, coordinates, normal_records)


# --- Topologically Compressed Rep Data (7.2.2.1.2.5) ---


@dataclass
class _TopologicalSymbols:
    degrees: list[list[int]]  # face degree symbols per compression context
    valences: list[int]
    groups: list[int]
    flags: list[int]
    attr_masks: list[list[int]]  # per context; context 7 combined to full 64-bit values
    attr_masks_large: npt.NDArray[np.uint32]
    split_faces: list[int]
    split_positions: list[int]


def _read_topological_symbols(reader: ByteReader) -> _TopologicalSymbols:
    degrees = [decode_int32_cdp2(reader) for _ in range(8)]
    valences = decode_int32_cdp2(reader)
    groups = decode_int32_cdp2(reader)
    flags = decode_int32_cdp2(reader, "lag1")
    masks = [decode_int32_cdp2(reader) for _ in range(8)]
    mask8_mid = decode_int32_cdp2(reader)
    mask8_top = decode_int32_cdp2(reader)
    large = reader.vec_u32()
    split_faces = decode_int32_cdp2(reader, "lag1")
    split_positions = decode_int32_cdp2(reader)
    stored_hash = reader.u32()
    computed = _composite_hash(
        degrees, valences, groups, flags, masks, mask8_mid, mask8_top, large, split_faces, split_positions
    )
    if computed != stored_hash:
        raise RuntimeError(
            f"corrupt JT data: topology hash mismatch (stored {stored_hash:#010x}, computed {computed:#010x})"
        )
    if len(mask8_mid) != len(masks[7]) or len(mask8_top) != len(masks[7]):
        raise RuntimeError("corrupt JT data: context-8 attribute mask field length mismatch")
    combined_mask8 = [
        (int(top) << 60) | ((int(mid) & 0x3FFFFFFF) << 30) | (int(low) & 0x3FFFFFFF)
        for low, mid, top in zip(masks[7], mask8_mid, mask8_top, strict=True)
    ]
    return _TopologicalSymbols(
        degrees=[array.tolist() for array in degrees],
        valences=valences.tolist(),
        groups=groups.tolist(),
        flags=flags.tolist(),
        attr_masks=[array.tolist() for array in masks[:7]] + [combined_mask8],
        attr_masks_large=large,
        split_faces=split_faces.tolist(),
        split_positions=split_positions.tolist(),
    )


def _masked_u32(values: npt.NDArray[np.int64]) -> npt.NDArray[np.uint32]:
    return (values & 0xFFFFFFFF).astype(np.uint32)


def _composite_hash(
    degrees: list[npt.NDArray[np.int64]],
    valences: npt.NDArray[np.int64],
    groups: npt.NDArray[np.int64],
    flags: npt.NDArray[np.int64],
    masks: list[npt.NDArray[np.int64]],
    mask8_mid: npt.NDArray[np.int64],
    mask8_top: npt.NDArray[np.int64],
    large: npt.NDArray[np.uint32],
    split_faces: npt.NDArray[np.int64],
    split_positions: npt.NDArray[np.int64],
) -> int:
    """Composite topology hash, exactly as the 7.2.2.1.2.5 pseudocode computes it.

    Note the reference casts the in-memory UInt64 attribute-mask arrays for
    contexts 1-7 to UInt32 pointers while passing the mask *count* as the word
    count, so only the first `count` 32-bit words of each 64-bit array are
    hashed; that behavior is replicated bit-for-bit here.
    """
    value = 0
    for context in range(8):
        value = jt_hash32(_masked_u32(degrees[context]), value)
    value = jt_hash32(_masked_u32(valences), value)
    value = jt_hash32(_masked_u32(groups), value)
    value = jt_hash16((flags & 0xFFFF).astype(np.uint16), value)
    for context in range(7):
        as_u64 = (masks[context] & 0xFFFFFFFF).astype("<u8")
        words = as_u64.view("<u4")[: len(masks[context])].astype(np.uint32)
        value = jt_hash32(words, value)
    value = jt_hash32(_masked_u32(masks[7] & 0x3FFFFFFF), value)
    value = jt_hash32(_masked_u32(mask8_mid & 0x3FFFFFFF), value)
    value = jt_hash32(_masked_u32(mask8_top & 0xF), value)
    value = jt_hash32(large, value)
    value = jt_hash32(_masked_u32(split_faces), value)
    value = jt_hash32(_masked_u32(split_positions), value)
    return value


# --- Topologically Compressed Vertex Records (7.2.2.1.2.6) ---


def _read_vertex_records(
    reader: ByteReader,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64] | None]:
    bindings = reader.u64()
    for _ in range(4):
        reader.u8()  # quantization parameters (restated per array)
    topological_vertices = reader.i32()
    if topological_vertices <= 0 or not bindings & _COORD_BINDING_MASK:
        raise RuntimeError("JT shape has no vertex coordinate data")
    coordinates = _read_coordinate_array(reader)
    if len(coordinates) != topological_vertices:
        raise RuntimeError("corrupt JT data: vertex coordinate count mismatch")
    normal_records = _read_normal_array(reader) if bindings & _NORMAL_BINDING_MASK else None
    return coordinates, normal_records


def _read_coordinate_array(reader: ByteReader) -> npt.NDArray[np.float64]:
    count = reader.i32()
    components = reader.u8()
    if count < 0 or components < 3:
        raise RuntimeError("corrupt JT data: invalid vertex coordinate array")
    quantizers = [(reader.f32(), reader.f32(), reader.u8()) for _ in range(3)]
    bits = quantizers[0][2]
    columns: list[npt.NDArray[np.float64]] = []
    hash_value = 0
    if bits == 0:
        for _ in range(components):
            exponents = decode_int32_cdp2(reader, "lag1")
            mantissae = decode_int32_cdp2(reader, "lag1")
            column = combine_float_bits(exponents, mantissae)
            hash_value = hash_float32_scalars(column.astype(np.float32), hash_value)
            columns.append(column)
    else:
        for component in range(components):
            codes = decode_int32_cdp2(reader, "lag1")
            hash_value = jt_hash32(_masked_u32(codes), hash_value)
            minimum, maximum, component_bits = quantizers[min(component, 2)]
            columns.append(dequantize_uniform(codes, minimum, maximum, component_bits))
    stored_hash = reader.i32() & 0xFFFFFFFF
    if stored_hash != hash_value:
        raise RuntimeError("corrupt JT data: vertex coordinate hash mismatch")
    if any(len(column) != count for column in columns):
        raise RuntimeError("corrupt JT data: vertex coordinate column length mismatch")
    return np.stack(columns[:3], axis=1)


def _read_normal_array(reader: ByteReader) -> npt.NDArray[np.float64]:
    count = reader.i32()
    components = reader.u8()
    bits = reader.u8()
    if count < 0 or components < 3:
        raise RuntimeError("corrupt JT data: invalid vertex normal array")
    hash_value = 0
    if bits == 0:
        columns = []
        for _ in range(components):
            exponents = decode_int32_cdp2(reader)
            mantissae = decode_int32_cdp2(reader)
            column = combine_float_bits(exponents, mantissae)
            hash_value = hash_float32_scalars(column.astype(np.float32), hash_value)
            columns.append(column)
        normals = np.stack(columns[:3], axis=1)
    else:
        sextants = decode_int32_cdp2(reader)
        octants = decode_int32_cdp2(reader)
        thetas = decode_int32_cdp2(reader)
        psis = decode_int32_cdp2(reader)
        for codes in (sextants, octants, thetas, psis):
            hash_value = jt_hash32(_masked_u32(codes), hash_value)
        normals = decode_deering_normals(sextants, octants, thetas, psis, bits)
    stored_hash = reader.u32()
    if stored_hash != hash_value:
        raise RuntimeError("corrupt JT data: vertex normal hash mismatch")
    if len(normals) != count:
        raise RuntimeError("corrupt JT data: vertex normal count mismatch")
    return normals


# --- Appendix E: DualVFMesh and the topology decoder ---


class _DualMesh:
    """The dual vertex-facet mesh built by the topology decoder."""

    def __init__(self) -> None:
        self.vtx_valence: list[int] = []
        self.vtx_flags: list[int] = []
        self.vtx_group: list[int] = []
        self.vtx_faces: list[list[int]] = []
        self.face_degree: list[int] = []
        self.face_empty: list[int] = []
        self.face_attr_mask: list[int] = []
        self.face_attr_records: list[list[int]] = []
        self.face_vts: list[list[int]] = []

    def new_vtx(self, valence: int, group: int, flags: int) -> int:
        self.vtx_valence.append(valence)
        self.vtx_group.append(group)
        self.vtx_flags.append(flags)
        self.vtx_faces.append([-1] * valence)
        return len(self.vtx_valence) - 1

    def new_face(self, degree: int, attr_mask: int, attr_records: list[int]) -> int:
        self.face_degree.append(degree)
        self.face_empty.append(degree)
        self.face_attr_mask.append(attr_mask)
        self.face_attr_records.append(attr_records)
        self.face_vts.append([-1] * degree)
        return len(self.face_degree) - 1

    def is_valid_face(self, face: int) -> bool:
        return 0 <= face < len(self.face_degree)

    def set_vtx_face(self, vtx: int, slot: int, face: int) -> None:
        self.vtx_faces[vtx][slot] = face

    def set_face_vtx(self, face: int, slot: int, vtx: int) -> None:
        if self.face_vts[face][slot] != vtx:
            self.face_empty[face] -= 1
        self.face_vts[face][slot] = vtx

    def find_vtx_slot(self, face: int, target: int) -> int:
        try:
            return self.face_vts[face].index(target)
        except ValueError:
            return -1

    def find_face_slot(self, vtx: int, target: int) -> int:
        try:
            return self.vtx_faces[vtx].index(target)
        except ValueError:
            return -1


class _TopologyDecoder:
    """Literal translation of the Appendix E MeshCodec/MeshDecoder machine."""

    def __init__(self, symbols: _TopologicalSymbols) -> None:
        self._symbols = symbols
        self._degree_pos = [0] * 8
        self._valence_pos = 0
        self._group_pos = 0
        self._flag_pos = 0
        self._mask_pos = [0] * 8
        self._mask_large_pos = 0
        self._split_face_pos = 0
        self._split_pos_pos = 0
        self._face_attr_counter = 0
        self.mesh = _DualMesh()
        self._active: list[int] = []
        self._removed: set[int] = set()

    # -- symbol streams --

    def _next_degree(self, context: int) -> int:
        stream = self._symbols.degrees[context]
        if self._degree_pos[context] >= len(stream):
            return -1
        value = stream[self._degree_pos[context]]
        self._degree_pos[context] += 1
        return value

    def _next_valence(self) -> int:
        if self._valence_pos >= len(self._symbols.valences):
            return -1
        value = self._symbols.valences[self._valence_pos]
        self._valence_pos += 1
        return value

    def _next_group(self) -> int:
        if self._group_pos >= len(self._symbols.groups):
            return -1
        value = self._symbols.groups[self._group_pos]
        self._group_pos += 1
        return value

    def _next_flag(self) -> int:
        if self._flag_pos >= len(self._symbols.flags):
            return 0
        value = self._symbols.flags[self._flag_pos]
        self._flag_pos += 1
        return value

    def _next_attr_mask(self, context: int) -> int:
        stream = self._symbols.attr_masks[context]
        if self._mask_pos[context] >= len(stream):
            return 0
        value = stream[self._mask_pos[context]]
        self._mask_pos[context] += 1
        return value

    def _next_attr_mask_large(self, degree: int) -> int:
        words = self._symbols.attr_masks_large
        count = (degree + 31) >> 5
        if self._mask_large_pos + count > len(words):
            return 0
        mask = 0
        for index in range(count):
            mask |= int(words[self._mask_large_pos + index]) << (32 * index)
        self._mask_large_pos += count
        return mask

    def _next_split_face(self) -> int:
        if self._split_face_pos >= len(self._symbols.split_faces):
            return -1
        value = self._symbols.split_faces[self._split_face_pos]
        self._split_face_pos += 1
        return value

    def _next_split_position(self) -> int:
        if self._split_pos_pos >= len(self._symbols.split_positions):
            return -1
        value = self._symbols.split_positions[self._split_pos_pos]
        self._split_pos_pos += 1
        return value

    def _face_context(self, vtx: int) -> int:
        mesh = self.mesh
        valence = mesh.vtx_valence[vtx]
        known_faces = 0
        known_degree_total = 0
        for face in mesh.vtx_faces[vtx]:
            if not mesh.is_valid_face(face):
                continue
            known_faces += 1
            known_degree_total += mesh.face_degree[face]
        if valence == 3:
            if known_degree_total < known_faces * 6:
                return 0
            return 1 if known_degree_total == known_faces * 6 else 2
        if valence == 4:
            if known_degree_total < known_faces * 4:
                return 3
            return 4 if known_degree_total == known_faces * 4 else 5
        return 6 if valence == 5 else 7

    # -- polymorphic I/O (decoder realization) --

    def _io_vtx(self) -> int:
        valence = self._next_valence()
        if valence <= -1:
            return -1
        group = self._next_group()
        flags = self._next_flag()
        return self.mesh.new_vtx(valence, group, flags)

    def _io_face(self, vtx: int) -> int:
        context = self._face_context(vtx)
        symbol = self._next_degree(context)
        if symbol == 0:
            return -1
        if symbol < 0:
            raise RuntimeError("corrupt JT data: face degree symbols exhausted")
        degree = symbol
        if degree <= 64:  # noqa: SIM108 - mirrors the reference decoder's branch
            mask = self._next_attr_mask(min(7, max(0, degree - 2)))
        else:
            mask = self._next_attr_mask_large(degree)
        attr_count = bin(mask).count("1")
        if attr_count > degree:
            raise RuntimeError("corrupt JT data: attribute mask wider than face degree")
        records = list(range(self._face_attr_counter, self._face_attr_counter + attr_count))
        self._face_attr_counter += attr_count
        return self.mesh.new_face(degree, mask, records)

    def _io_split_face(self) -> int:
        offset = self._next_split_face()
        if offset <= 0 or offset > len(self._active):
            raise RuntimeError("corrupt JT data: split-face offset out of range")
        return self._active[len(self._active) - offset]

    def _io_split_position(self) -> int:
        position = self._next_split_position()
        if position < 0:
            raise RuntimeError("corrupt JT data: split-position symbols exhausted")
        return position

    # -- MeshCodec driver chain --

    def run(self) -> _DualMesh:
        found = True
        while found:
            found = self._run_component()
        if self._valence_pos != len(self._symbols.valences):
            raise RuntimeError("corrupt JT data: unconsumed vertex valence symbols")
        return self.mesh

    def _run_component(self) -> bool:
        vtx = self._io_vtx()
        if vtx == -1:
            return False
        for slot in range(self.mesh.vtx_valence[vtx]):
            self._activate_face(vtx, slot)
        while (face := self._next_active_face()) != -1:
            self._complete_face(face)
            self._removed.add(face)
        return True

    def _complete_face(self, face: int) -> None:
        mesh = self.mesh
        while (slot := mesh.find_vtx_slot(face, -1)) != -1:
            vtx = self._activate_vtx(face, slot)
            self._complete_vtx(vtx, slot)

    def _activate_vtx(self, face: int, face_slot: int) -> int:
        vtx = self._io_vtx()
        if vtx == -1:
            raise RuntimeError("corrupt JT data: vertex symbols exhausted mid-face")
        self.mesh.set_vtx_face(vtx, 0, face)
        self._add_vtx_to_face(vtx, 0, face, face_slot)
        return vtx

    def _activate_face(self, vtx: int, vtx_slot: int) -> int:
        face = self._io_face(vtx)
        if face >= 0:
            self.mesh.set_vtx_face(vtx, vtx_slot, face)
            self.mesh.set_face_vtx(face, 0, vtx)
            self._active.append(face)
        elif face == -1:
            face = self._io_split_face()
            face_slot = self._io_split_position()
            self.mesh.set_vtx_face(vtx, vtx_slot, face)
            self._add_vtx_to_face(vtx, vtx_slot, face, face_slot)
        return face

    def _complete_vtx(self, vtx: int, vtx_slot_on_face0: int) -> None:
        mesh = self.mesh
        valence = mesh.vtx_valence[vtx]
        # Walk counter-clockwise from face slot 0, linking already-reachable faces.
        vp = mesh.vtx_faces[vtx][0]
        jp = vtx_slot_on_face0
        i = 1
        while (vn := mesh.vtx_faces[vtx][i]) != -1:
            jp = (jp - 1) % mesh.face_degree[vp]
            vtx2 = mesh.face_vts[vp][jp]
            if vtx2 == -1:
                break
            jn = mesh.find_vtx_slot(vn, vtx2)
            if jn < 0:
                raise RuntimeError("corrupt JT data: inconsistent dual mesh topology")
            jn = (jn - 1) % mesh.face_degree[vn]
            self._add_vtx_to_face(vtx, i, vn, jn)
            vp, jp = vn, jn
            i += 1
            if i >= valence:
                return
        # Walk clockwise from face slot 0.
        i_last = i
        vp = mesh.vtx_faces[vtx][0]
        jp = vtx_slot_on_face0
        i = valence - 1
        while (vn := mesh.vtx_faces[vtx][i]) != -1:
            jp = (jp + 1) % mesh.face_degree[vp]
            vtx2 = mesh.face_vts[vp][jp]
            if vtx2 == -1:
                break
            jn = mesh.find_vtx_slot(vn, vtx2)
            if jn < 0:
                raise RuntimeError("corrupt JT data: inconsistent dual mesh topology")
            jn = (jn + 1) % mesh.face_degree[vn]
            self._add_vtx_to_face(vtx, i, vn, jn)
            vp, jp = vn, jn
            i -= 1
            if i < i_last:
                return
        # Activate the remaining faces that could not be deduced topologically.
        while i_last <= i:
            self._activate_face(vtx, i_last)
            i_last += 1

    def _add_vtx_to_face(self, vtx: int, vtx_face_slot: int, face: int, face_slot: int) -> None:
        mesh = self.mesh
        degree = mesh.face_degree[face]
        slot_cw = (face_slot - 1) % degree
        slot_ccw = (face_slot + 1) % degree
        mesh.set_face_vtx(face, face_slot, vtx)
        valence = mesh.vtx_valence[vtx]
        neighbor = mesh.face_vts[face][slot_cw]
        if neighbor != -1:
            shared = mesh.find_face_slot(neighbor, face)
            vtx_slot_ccw = (vtx_face_slot + 1) % valence
            if mesh.vtx_faces[vtx][vtx_slot_ccw] == -1:
                shared = (shared - 1) % mesh.vtx_valence[neighbor]
                mesh.set_vtx_face(vtx, vtx_slot_ccw, mesh.vtx_faces[neighbor][shared])
        neighbor = mesh.face_vts[face][slot_ccw]
        if neighbor != -1:
            shared = mesh.find_face_slot(neighbor, face)
            vtx_slot_cw = (vtx_face_slot - 1) % valence
            if mesh.vtx_faces[vtx][vtx_slot_cw] == -1:
                shared = (shared + 1) % mesh.vtx_valence[neighbor]
                mesh.set_vtx_face(vtx, vtx_slot_cw, mesh.vtx_faces[neighbor][shared])

    def _next_active_face(self) -> int:
        active = self._active
        removed = self._removed
        while active and active[-1] in removed:
            active.pop()
        best = -1
        lowest = 9999999
        i = len(active) - 1
        while i >= max(0, len(active) - 16):
            face = active[i]
            if face in removed:
                del active[i]
                i -= 1
                continue
            empty = self.mesh.face_empty[face]
            if empty < lowest:
                lowest = empty
                best = face
            i -= 1
        return best


# --- Primal mesh extraction ---


def _extract_primal_mesh(
    mesh: _DualMesh,
    coordinates: npt.NDArray[np.float64],
    normal_records: npt.NDArray[np.float64] | None,
) -> DecodedShape:
    triangles: list[tuple[int, int, int]] = []
    groups: list[int] = []
    corner_records: list[tuple[int, int, int]] = []
    for vtx in range(len(mesh.vtx_valence)):
        if mesh.vtx_flags[vtx]:
            continue  # cover face added to close the mesh
        faces = mesh.vtx_faces[vtx]
        if mesh.vtx_valence[vtx] != 3:
            raise RuntimeError("corrupt JT data: non-triangular primal face in tri-strip set")
        if any(face < 0 for face in faces):
            raise RuntimeError("corrupt JT data: incomplete dual vertex after decode")
        triangles.append((faces[0], faces[1], faces[2]))
        groups.append(mesh.vtx_group[vtx])
        if normal_records is not None:
            corner_records.append(tuple(_corner_record(mesh, face, vtx) for face in faces))  # type: ignore[arg-type]
    if not triangles:
        raise RuntimeError("JT shape decoded to zero triangles")
    faces_array = np.array(triangles, dtype=np.int64)
    groups_array = np.array(groups, dtype=np.int64)
    if normal_records is None:
        return DecodedShape(points=coordinates, faces=faces_array, normals=None, face_groups=groups_array)
    return _split_vertices_by_record(coordinates, faces_array, groups_array, corner_records, normal_records)


def _corner_record(mesh: _DualMesh, face: int, vtx: int) -> int:
    """Attribute record used by primal vertex `face` at primal triangle `vtx`."""
    ring = mesh.face_vts[face]
    position = ring.index(vtx)
    mask = mesh.face_attr_mask[face]
    records = mesh.face_attr_records[face]
    if not records:
        raise RuntimeError("corrupt JT data: face without attribute records")
    ones_through = bin(mask & ((2 << position) - 1)).count("1")
    if ones_through == 0:
        return records[-1]  # wrap to the last record counter-clockwise
    return records[ones_through - 1]


def _split_vertices_by_record(
    coordinates: npt.NDArray[np.float64],
    faces: npt.NDArray[np.int64],
    groups: npt.NDArray[np.int64],
    corner_records: list[tuple[int, int, int]],
    normal_records: npt.NDArray[np.float64],
) -> DecodedShape:
    """Duplicate topological vertices per distinct attribute record (hard edges)."""
    remap: dict[tuple[int, int], int] = {}
    out_faces = np.empty_like(faces)
    for triangle_index, corners in enumerate(corner_records):
        for corner_index in range(3):
            key = (int(faces[triangle_index, corner_index]), corners[corner_index])
            index = remap.setdefault(key, len(remap))
            out_faces[triangle_index, corner_index] = index
    points = np.empty((len(remap), 3), dtype=np.float64)
    normals = np.empty((len(remap), 3), dtype=np.float64)
    for (vertex, record), index in remap.items():
        points[index] = coordinates[vertex]
        if record >= len(normal_records):
            raise RuntimeError("corrupt JT data: attribute record index out of range")
        normals[index] = normal_records[record]
    return DecodedShape(points=points, faces=out_faces, normals=normals, face_groups=groups)
