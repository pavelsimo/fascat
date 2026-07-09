"""Synthetic JT byte-stream builder for hermetic tests.

An independent encoder written from the public Siemens *JT File Format
Reference*. It intentionally never imports ``fascat.io.jt`` so a reader bug
cannot be mirrored by a matching builder bug.
"""

from __future__ import annotations

import lzma
import math
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

_VERSION_LINE_BYTES = 80

SEGMENT_LOGICAL_SCENE_GRAPH = 1
SEGMENT_META_DATA = 4
SEGMENT_SHAPE_LOD0 = 7
SEGMENT_XT_BREP = 17

# Segment types in this range carry raw (codec-compressed) payloads.
_SHAPE_SEGMENT_RANGE = range(6, 17)

COMPRESSION_NONE = 1
COMPRESSION_ZLIB = 2
COMPRESSION_LZMA = 3


def make_guid(seed: int) -> tuple[int, int, int, bytes]:
    """A deterministic synthetic GUID as (u32, u16, u16, 8 bytes)."""
    return (seed & 0xFFFFFFFF, (seed >> 4) & 0xFFFF, (seed >> 8) & 0xFFFF, bytes([seed & 0xFF] * 8))


def guid_bytes(guid: tuple[int, int, int, bytes]) -> bytes:
    """Canonical little-endian encoding of a GUID (matches the reader's key form)."""
    return struct.pack("<IHH", guid[0], guid[1], guid[2]) + guid[3]


class JtWriter:
    """Packs scalars into a bytearray with deferred length/offset backpatching."""

    def __init__(self, byte_order: str = "<") -> None:
        assert byte_order in ("<", ">")
        self.byte_order = byte_order
        self.buffer = bytearray()

    def tell(self) -> int:
        return len(self.buffer)

    def raw(self, data: bytes) -> None:
        self.buffer.extend(data)

    def _pack(self, code: str, value: float) -> None:
        self.buffer.extend(struct.pack(f"{self.byte_order}{code}", value))

    def u8(self, value: int) -> None:
        self._pack("B", value)

    def i16(self, value: int) -> None:
        self._pack("h", value)

    def u16(self, value: int) -> None:
        self._pack("H", value)

    def i32(self, value: int) -> None:
        self._pack("i", value)

    def u32(self, value: int) -> None:
        self._pack("I", value)

    def u64(self, value: int) -> None:
        self._pack("Q", value)

    def f32(self, value: float) -> None:
        self._pack("f", value)

    def f64(self, value: float) -> None:
        self._pack("d", value)

    def guid(self, guid: tuple[int, int, int, bytes]) -> None:
        self.u32(guid[0])
        self.u16(guid[1])
        self.u16(guid[2])
        self.raw(guid[3])

    def string(self, text: str) -> None:
        encoded = text.encode("latin-1")
        self.i32(len(encoded))
        self.raw(encoded)

    def mbstring(self, text: str) -> None:
        encoding = "utf-16-le" if self.byte_order == "<" else "utf-16-be"
        self.i32(len(text))
        self.raw(text.encode(encoding))

    def reserve(self, code: str) -> tuple[int, str]:
        """Write a placeholder scalar and return a slot for later patching."""
        position = self.tell()
        self._pack(code, 0)
        return (position, code)

    def patch(self, slot: tuple[int, str], value: int) -> None:
        position, code = slot
        packed = struct.pack(f"{self.byte_order}{code}", value)
        self.buffer[position : position + len(packed)] = packed


@dataclass
class SegmentSpec:
    guid: tuple[int, int, int, bytes]
    segment_type: int
    payload: bytes
    compression: int = COMPRESSION_NONE


@dataclass
class ContainerSpec:
    segments: list[SegmentSpec] = field(default_factory=list)
    version: tuple[int, int] = (9, 5)
    byte_order: str = "<"
    lsg_guid: tuple[int, int, int, bytes] | None = None
    version_line: str | None = None


def build_container(spec: ContainerSpec) -> bytes:
    """Emit a complete JT file: header, segments, and trailing TOC."""
    major, minor = spec.version
    writer = JtWriter(spec.byte_order)
    line = spec.version_line if spec.version_line is not None else f"Version {major}.{minor} JT"
    encoded_line = line.encode("ascii")[: _VERSION_LINE_BYTES - 2].ljust(_VERSION_LINE_BYTES - 2) + b"\r\n"
    writer.raw(encoded_line)
    writer.u8(0 if spec.byte_order == "<" else 1)
    writer.i32(0)  # reserved
    toc_slot = writer.reserve("Q" if major >= 10 else "I")
    lsg_guid = spec.lsg_guid
    if lsg_guid is None:
        lsg_guid = next(
            (seg.guid for seg in spec.segments if seg.segment_type == SEGMENT_LOGICAL_SCENE_GRAPH),
            make_guid(0),
        )
    writer.guid(lsg_guid)

    placements: list[tuple[SegmentSpec, int, int]] = []
    for segment in spec.segments:
        offset = writer.tell()
        _write_segment(writer, segment, major)
        placements.append((segment, offset, writer.tell() - offset))

    writer.patch(toc_slot, writer.tell())
    writer.i32(len(placements))
    for segment, offset, length in placements:
        writer.guid(segment.guid)
        if major >= 10:
            writer.u64(offset)
        else:
            writer.u32(offset)
        writer.u32(length)
        writer.u32((segment.segment_type & 0xFF) << 24)
    return bytes(writer.buffer)


def _write_segment(writer: JtWriter, segment: SegmentSpec, major: int = 9) -> None:
    writer.guid(segment.guid)
    writer.i32(segment.segment_type)
    length_slot = writer.reserve("i")
    start = length_slot[0] - 20  # segment length counts from the segment header GUID
    if segment.segment_type in _SHAPE_SEGMENT_RANGE:
        writer.raw(segment.payload)
    else:
        # JT 9 writes compression flag 2 (ZLIB block); JT 10 writes flag 3
        # (LZMA block) with XZ-container payloads.
        writer.i32(3 if major >= 10 else 2)
        if segment.compression == COMPRESSION_ZLIB:
            data = zlib.compress(segment.payload)
        elif segment.compression == COMPRESSION_LZMA:
            fmt = lzma.FORMAT_XZ if major >= 10 else lzma.FORMAT_ALONE
            data = lzma.compress(segment.payload, format=fmt)
        else:
            data = segment.payload
        writer.i32(len(data) + 1)
        writer.u8(segment.compression)
        writer.raw(data)
    writer.patch(length_slot, writer.tell() - start)


def build_jt7_bytes() -> bytes:
    """A JT 7.0 header stub — just enough to trip the version gate."""
    line = b"Version 7.0 JT".ljust(_VERSION_LINE_BYTES - 2) + b"\r\n"
    return line + bytes([0]) + bytes(28)


# --- Int32 CDP Mk.2 encoding (JT 9.5 reference sections 8.1.2 / 8.2 / Appendix C) ---


class BitWriter:
    """Collects MSB-first bits and folds them into big-endian 32-bit words."""

    def __init__(self) -> None:
        self.bits: list[int] = []

    def write_bit(self, bit: int) -> None:
        self.bits.append(bit & 1)

    def write(self, value: int, nbits: int) -> None:
        for shift in range(nbits - 1, -1, -1):
            self.bits.append((value >> shift) & 1)

    def write_signed(self, value: int, nbits: int) -> None:
        self.write(value & ((1 << nbits) - 1) if nbits else 0, nbits)

    def to_words(self) -> list[int]:
        words = []
        for start in range(0, len(self.bits), 32):
            chunk = self.bits[start : start + 32]
            word = 0
            for bit in chunk:
                word = (word << 1) | bit
            word <<= 32 - len(chunk)
            words.append(word)
        return words

    def to_bytes(self) -> bytes:
        out = bytearray()
        for start in range(0, len(self.bits), 8):
            chunk = self.bits[start : start + 8]
            byte = 0
            for bit in chunk:
                byte = (byte << 1) | bit
            byte <<= 8 - len(chunk)
            out.append(byte)
        return bytes(out)


def pack_residuals(values: list[int], predictor: str) -> list[int]:
    """Inverse of the spec's unpackResiduals: first four values are primers."""
    out = []
    for index, value in enumerate(values):
        if index < 4 or predictor == "null":
            out.append(value)
            continue
        v1, v2 = values[index - 1], values[index - 2]
        v4 = values[index - 4]
        if predictor in ("lag1", "xor1"):
            predicted = v1
        elif predictor in ("lag2", "xor2"):
            predicted = v2
        elif predictor == "stride1":
            predicted = v1 + (v1 - v2)
        elif predictor == "stride2":
            predicted = v2 + (v2 - v4)
        elif predictor == "stripindex":
            predicted = v2 + ((v2 - v4) if -8 < v2 - v4 < 8 else 2)
        elif predictor == "ramp":
            predicted = index
        else:
            raise ValueError(f"unknown predictor {predictor}")
        if predictor in ("xor1", "xor2"):
            residual = (value & 0xFFFFFFFF) ^ (predicted & 0xFFFFFFFF)
            if residual >= 1 << 31:
                residual -= 1 << 32
        else:
            residual = value - predicted
            residual = (residual + (1 << 31)) % (1 << 32) - (1 << 31)
        out.append(residual)
    return out


def _signed_bitsize(value: int) -> int:
    """Spec bitsize(): smallest signed field width that holds `value` (0 -> 1)."""
    magnitude = value if value >= 0 else ~value
    return magnitude.bit_length() + 1


def _encode_bitlength_fixed(residuals: list[int]) -> BitWriter:
    """BitLengthCodec2 fixed-width mode: signed min/max header, then offset fields."""
    writer = BitWriter()
    writer.write_bit(0)
    minimum = min(residuals)
    maximum = max(residuals)
    min_bits = _signed_bitsize(minimum)
    max_bits = _signed_bitsize(maximum)
    writer.write(min_bits, 6)
    writer.write(max_bits, 6)
    writer.write_signed(minimum, min_bits)
    writer.write_signed(maximum, max_bits)
    if maximum - minimum > 0:
        width = (maximum - minimum).bit_length()
        for value in residuals:
            writer.write(value - minimum, width)
    return writer


_VARIABLE_DELTA_BITS = 6
_VARIABLE_RUN_BITS = 7


def _encode_bitlength_variable(residuals: list[int]) -> BitWriter:
    """BitLengthCodec2 variable-width mode: runs of signed fields around a mean."""
    writer = BitWriter()
    writer.write_bit(1)
    mean = sorted(residuals)[len(residuals) // 2]
    writer.write_signed(mean, 32)
    writer.write(_VARIABLE_DELTA_BITS, 3)
    writer.write(_VARIABLE_RUN_BITS, 3)
    saturate_low = -(1 << (_VARIABLE_DELTA_BITS - 1))
    saturate_high = (1 << (_VARIABLE_DELTA_BITS - 1)) - 1
    max_run = (1 << _VARIABLE_RUN_BITS) - 1
    width = 0
    position = 0
    while position < len(residuals):
        run = residuals[position : position + max_run]
        target = max(_signed_bitsize(value - mean) for value in run)
        delta = target - width
        while delta >= saturate_high:
            writer.write_signed(saturate_high, _VARIABLE_DELTA_BITS)
            delta -= saturate_high
        while delta <= saturate_low:
            writer.write_signed(saturate_low, _VARIABLE_DELTA_BITS)
            delta -= saturate_low
        writer.write_signed(delta, _VARIABLE_DELTA_BITS)
        width = target
        writer.write(len(run), _VARIABLE_RUN_BITS)
        for value in run:
            writer.write_signed(value - mean, width)
        position += len(run)
    return writer


def _encode_bitlength_codetext(residuals: list[int], mode: str = "bitlength") -> BitWriter:
    fixed = _encode_bitlength_fixed(residuals)
    if mode == "bitlength-fixed":
        return fixed
    variable = _encode_bitlength_variable(residuals)
    if mode == "bitlength-variable":
        return variable
    return fixed if len(fixed.bits) <= len(variable.bits) else variable


def _encode_arithmetic_codetext(
    symbol_stream: list[int],
    cumulative: dict[int, tuple[int, int]],
    total: int,
) -> BitWriter:
    """The classic 16-bit arithmetic encoder mirroring the spec's Appendix C decoder."""
    writer = BitWriter()
    pending = 0

    def emit(bit: int) -> None:
        nonlocal pending
        writer.write_bit(bit)
        while pending:
            writer.write_bit(bit ^ 1)
            pending -= 1

    low, high = 0, 0xFFFF
    for symbol in symbol_stream:
        low_count, high_count = cumulative[symbol]
        span = high - low + 1
        high = low + (span * high_count) // total - 1
        low = low + (span * low_count) // total
        while True:
            if (~(high ^ low)) & 0x8000:
                emit((high >> 15) & 1)
            elif (low & 0x4000) and not (high & 0x4000):
                pending += 1
                low &= 0x3FFF
                high |= 0x4000
            else:
                break
            low = (low << 1) & 0xFFFF
            high = ((high << 1) | 1) & 0xFFFF
    for shift in range(15, -1, -1):
        emit((low >> shift) & 1)
    return writer


def encode_cdp2(
    values: list[int],
    *,
    codec: str = "null",
    predictor: str = "null",
    byte_order: str = "<",
    oob_values: set[int] | None = None,
) -> bytes:
    """Encode an Int32 Compressed Data Packet Mk.2.

    `oob_values` (arithmetic only) forces the given residual values out-of-band
    behind an escape symbol; by default every residual gets a context entry.
    """
    writer = JtWriter(byte_order)
    writer.i32(len(values))
    if not values:
        return bytes(writer.buffer)
    residuals = pack_residuals(values, predictor)
    if codec == "null":
        writer.u8(0)
        writer.i32(4 * len(residuals))  # the Null CODEC length field counts bytes
        for value in residuals:
            writer.u32(value & 0xFFFFFFFF)
        return bytes(writer.buffer)
    if codec in ("bitlength", "bitlength-fixed", "bitlength-variable"):
        bits = _encode_bitlength_codetext(residuals, codec)
        writer.u8(1)
        _write_codetext(writer, bits)
        return bytes(writer.buffer)
    if codec != "arithmetic":
        raise ValueError(f"unknown codec {codec}")
    oob_values = oob_values or set()
    escaped = [value for value in residuals if value in oob_values]
    in_band = sorted({value for value in residuals if value not in oob_values})
    if not in_band and not escaped:
        raise ValueError("arithmetic packet needs at least one value")
    entries: list[tuple[int, int, int]] = []  # (symbol, count, value)
    symbol_of: dict[int, int] = {}
    if escaped:
        entries.append((-2, len(escaped), in_band[0] if in_band else 0))
    for index, value in enumerate(in_band):
        symbol_of[value] = index
        entries.append((index, sum(1 for r in residuals if r == value), value))
    total = sum(count for _, count, _ in entries)
    assert total < 1 << 14, "arithmetic total count must stay below 2^14"
    min_value = min(value for _, _, value in entries)
    symbol_bits = max(2, max(symbol + 2 for symbol, _, _ in entries).bit_length())
    count_bits = max(1, max(count for _, count, _ in entries).bit_length())
    value_bits = max(value - min_value for _, _, value in entries).bit_length()
    cumulative: dict[int, tuple[int, int]] = {}
    running = 0
    for symbol, count, _ in entries:
        cumulative[symbol] = (running, running + count)
        running += count
    symbol_stream = [symbol_of.get(value, -2) for value in residuals]
    code_bits = _encode_arithmetic_codetext(symbol_stream, cumulative, total)
    writer.u8(3)
    _write_codetext(writer, code_bits)
    context_bits = BitWriter()
    context_bits.write(len(entries), 16)
    context_bits.write(symbol_bits, 6)
    context_bits.write(count_bits, 6)
    context_bits.write(value_bits, 6)
    context_bits.write(min_value & 0xFFFFFFFF, 32)
    for symbol, count, value in entries:
        context_bits.write(symbol + 2, symbol_bits)
        context_bits.write(count, count_bits)
        context_bits.write(value - min_value, value_bits)
    writer.raw(context_bits.to_bytes())
    writer.raw(encode_cdp2(escaped, codec="null", byte_order=byte_order))
    return bytes(writer.buffer)


def _write_codetext(writer: JtWriter, bits: BitWriter) -> None:
    writer.i32(len(bits.bits))
    for word in bits.to_words():
        writer.u32(word)


def encode_cdp3(
    values: list[int],
    *,
    predictor: str = "null",
    byte_order: str = "<",
) -> bytes:
    """Encode an Int32 Compressed Data Packet Mk.3 (JT 10) with the Null CODEC.

    Entropy-coded Mk.3 paths (bitlength, arithmetic, move-to-front, chopper)
    are pinned by real-file golden packets in test_jt_codec.py instead of a
    synthetic encoder.
    """
    writer = JtWriter(byte_order)
    writer.i32(len(values))
    if not values:
        return bytes(writer.buffer)
    residuals = pack_residuals(values, predictor)
    writer.u8(0)
    writer.i32(4 * len(residuals))  # the Null CODEC length field counts bytes
    for value in residuals:
        writer.u32(value & 0xFFFFFFFF)
    return bytes(writer.buffer)


# --- Int32 CDP Mk.1 encoding (JT 8, v8.1 reference section 8.1) ---


def _encode_bitlength1(residuals: list[int]) -> BitWriter:
    """Mk.1 adaptive-width bitlength: escalate the field width in 2-bit steps.

    Widths only ever grow here (the decoder supports both directions); a width
    change of k steps is coded as '1', the direction bit, k-1 repeats of the
    direction bit, and one terminating complement bit.
    """
    writer = BitWriter()
    width = 0
    for value in residuals:
        needed = 0 if value == 0 else _signed_bitsize(value)
        if needed > width:
            target = needed + (needed & 1)  # widths move in 2s from 0, so stay even
            steps = (target - width) // 2
            writer.write_bit(1)
            for _ in range(steps):
                writer.write_bit(1)
            writer.write_bit(0)
            width = target
        else:
            writer.write_bit(0)
        if width:
            writer.write_signed(value, width)
    return writer


def encode_cdp1(
    values: list[int],
    *,
    codec: str = "bitlength",
    predictor: str = "null",
    byte_order: str = "<",
) -> bytes:
    """Encode an Int32 Compressed Data Packet Mk.1 (Null and Bitlength CODECs).

    Huffman and arithmetic are decode-only, pinned by real-file golden vectors.
    """
    writer = JtWriter(byte_order)
    residuals = pack_residuals([int(value) for value in values], predictor)
    if codec == "null":
        writer.u8(0)
        writer.i32(len(residuals))  # the Mk.1 Null length field counts values
        for value in residuals:
            writer.i32(value)
        return bytes(writer.buffer)
    if codec != "bitlength":
        raise ValueError(f"unsupported Mk.1 encoder codec {codec!r}")
    bits = _encode_bitlength1(residuals)
    writer.u8(1)
    writer.i32(len(bits.bits))
    writer.i32(len(residuals))
    words = bits.to_words()
    writer.i32(len(words))
    for word in words:
        writer.u32(word)
    return bytes(writer.buffer)


# --- Deering normal encoding (spec 8.2.4 / Appendix C 4 inverse) ---

_DEERING_PSI_MAX = 0.615479709
# (max axis, axis mapped to base[1], axis mapped to base[2]) -> sextant, the
# inverse of the decoder's sextant permutation table.
_DEERING_SEXTANT_BY_AXES = {
    (0, 1, 2): 0,
    (0, 2, 1): 5,
    (2, 1, 0): 1,
    (2, 0, 1): 2,
    (1, 0, 2): 3,
    (1, 2, 0): 4,
}


def encode_deering_normals(normals, bits: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Quantize unit normals into (sextant, octant, theta, psi) code streams."""
    bit_range = float(1 << bits)
    sextants: list[int] = []
    octants: list[int] = []
    thetas: list[int] = []
    psis: list[int] = []
    for normal in np.asarray(normals, dtype=np.float64):
        octant = (4 if normal[0] >= 0 else 0) | (2 if normal[1] >= 0 else 0) | (1 if normal[2] >= 0 else 0)
        magnitudes = np.abs(normal)
        max_axis = int(np.argmax(magnitudes))
        rest = [axis for axis in range(3) if axis != max_axis]
        # base[1] holds sin(psi), which the code range caps at sin(psi_max) ~=
        # 0.577 -- route the smaller remaining component there.
        axis1, axis2 = sorted(rest, key=lambda axis: magnitudes[axis])
        sextant = _DEERING_SEXTANT_BY_AXES[(max_axis, axis1, axis2)]
        psi = math.asin(min(float(magnitudes[axis1]), 1.0))
        theta = math.atan2(float(magnitudes[axis2]), float(magnitudes[max_axis]))
        theta_adjusted = bit_range * (1.0 - math.atan(math.sin(theta)) / _DEERING_PSI_MAX)
        theta_code = max(round(theta_adjusted) - (sextant & 1), 0)
        psi_code = min(round(psi * bit_range / _DEERING_PSI_MAX), int(bit_range))
        sextants.append(sextant)
        octants.append(octant)
        thetas.append(theta_code)
        psis.append(psi_code)
    return sextants, octants, thetas, psis


# --- LSG segment emission (JT 9.5 reference sections 7.1.3.2 / 7.2.1 / Appendix A) ---


def _lsg_guid(part1: int, part2: int, part3: int, *tail: int) -> tuple[int, int, int, bytes]:
    return (part1, part2, part3, bytes(tail))


END_OF_ELEMENTS_GUID = _lsg_guid(0xFFFFFFFF, 0xFFFF, 0xFFFF, *([0xFF] * 8))
GROUP_NODE_GUID = _lsg_guid(0x10DD101B, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
INSTANCE_NODE_GUID = _lsg_guid(0x10DD102A, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
LOD_NODE_GUID = _lsg_guid(0x10DD102C, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
PART_NODE_GUID = _lsg_guid(0xCE357244, 0x38FB, 0x11D1, 0xA5, 0x06, 0x00, 0x60, 0x97, 0xBD, 0xC6, 0xE1)
PARTITION_NODE_GUID = _lsg_guid(0x10DD103E, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
RANGE_LOD_NODE_GUID = _lsg_guid(0x10DD104C, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
TRISTRIP_SHAPE_NODE_GUID = _lsg_guid(0x10DD1077, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
MATERIAL_ATTRIBUTE_GUID = _lsg_guid(0x10DD1030, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
TRANSFORM_ATTRIBUTE_GUID = _lsg_guid(0x10DD1083, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
STRING_ATOM_GUID = _lsg_guid(0x10DD106E, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
INTEGER_ATOM_GUID = _lsg_guid(0x10DD102B, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
FLOAT_ATOM_GUID = _lsg_guid(0x10DD1019, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
LATE_LOADED_ATOM_GUID = _lsg_guid(0xE0B05BE5, 0xFBBD, 0x11D1, 0xA3, 0xA7, 0x00, 0xAA, 0x00, 0xD1, 0x09, 0x54)

TRISTRIP_SHAPE_LOD_ELEMENT_GUID = _lsg_guid(0x10DD10AB, 0x2AC8, 0x11D1, 0x9B, 0x6B, 0x00, 0x80, 0xC7, 0xBB, 0x59, 0x97)
TOPOMESH_TOPO_COMPRESSED_LOD_GUID = _lsg_guid(
    0xF830A5AD, 0xBE4C, 0x4FBC, 0x9B, 0x5F, 0xB9, 0x26, 0x92, 0x78, 0xD2, 0xE1
)


# --- Topology encoder + Shape LOD emission (spec 7.2.2.1.2.4-6 / Appendix E mirror) ---


def _hash_mix(a: int, b: int, c: int) -> tuple[int, int, int]:
    m = 0xFFFFFFFF
    a = (a - b - c) & m ^ (c >> 13)
    b = (b - c - a) & m ^ ((a << 8) & m)
    c = (c - a - b) & m ^ (b >> 13)
    a = (a - b - c) & m ^ (c >> 12)
    b = (b - c - a) & m ^ ((a << 16) & m)
    c = (c - a - b) & m ^ (b >> 5)
    a = (a - b - c) & m ^ (c >> 3)
    b = (b - c - a) & m ^ ((a << 10) & m)
    c = (c - a - b) & m ^ (b >> 15)
    return a, b, c


def _hash32_words(words: list[int], seed: int) -> int:
    a = b = 0x9E3779B9
    c = seed & 0xFFFFFFFF
    i, length = 0, len(words)
    while length - i >= 3:
        a = (a + words[i]) & 0xFFFFFFFF
        b = (b + words[i + 1]) & 0xFFFFFFFF
        c = (c + words[i + 2]) & 0xFFFFFFFF
        a, b, c = _hash_mix(a, b, c)
        i += 3
    c = (c + length) & 0xFFFFFFFF
    if length - i == 2:
        b = (b + words[i + 1]) & 0xFFFFFFFF
    if length - i >= 1:
        a = (a + words[i]) & 0xFFFFFFFF
    return _hash_mix(a, b, c)[2]


def _hash16_words(halves: list[int], seed: int) -> int:
    a = b = 0x9E3779B9
    c = seed & 0xFFFFFFFF
    i, length = 0, len(halves)
    while length - i >= 6:
        a = (a + halves[i] + (halves[i + 1] << 16)) & 0xFFFFFFFF
        b = (b + halves[i + 2] + (halves[i + 3] << 16)) & 0xFFFFFFFF
        c = (c + halves[i + 4] + (halves[i + 5] << 16)) & 0xFFFFFFFF
        a, b, c = _hash_mix(a, b, c)
        i += 6
    c = (c + length) & 0xFFFFFFFF
    remaining = length - i
    if remaining == 5:
        c = (c + (halves[i + 4] << 16)) & 0xFFFFFFFF
    if remaining >= 4:
        b = (b + (halves[i + 3] << 16)) & 0xFFFFFFFF
    if remaining >= 3:
        b = (b + halves[i + 2]) & 0xFFFFFFFF
    if remaining >= 2:
        a = (a + (halves[i + 1] << 16)) & 0xFFFFFFFF
    if remaining >= 1:
        a = (a + halves[i]) & 0xFFFFFFFF
    return _hash_mix(a, b, c)[2]


def _hash_float32_column(values, seed: int) -> int:
    words = np.asarray(values, dtype="<f4").view("<u4").tolist()
    c = seed & 0xFFFFFFFF
    for word in words:
        a = (0x9E3779B9 + word) & 0xFFFFFFFF
        c = (c + 1) & 0xFFFFFFFF
        _, _, c = _hash_mix(a, 0x9E3779B9, c)
    return c


class _SourceDualMesh:
    """The dual of a closed manifold triangle mesh: rings for the encoder to consult."""

    def __init__(self, triangles: list[tuple[int, int, int]], vertex_count: int) -> None:
        edge_to_triangle: dict[tuple[int, int], int] = {}
        for index, (a, b, c) in enumerate(triangles):
            for edge in ((a, b), (b, c), (c, a)):
                if edge in edge_to_triangle:
                    raise ValueError("input mesh is not manifold: duplicate directed edge")
                edge_to_triangle[edge] = index
        incident: dict[int, list[int]] = {v: [] for v in range(vertex_count)}
        for index, triangle in enumerate(triangles):
            for vertex in triangle:
                incident[vertex].append(index)
        self.triangle_rings = [tuple(tri) for tri in triangles]  # dual vtx -> dual faces (CCW)
        self.vertex_rings: list[tuple[int, ...]] = []  # dual face -> dual vts (CCW)
        for vertex in range(vertex_count):
            triangles_at = incident[vertex]
            if not triangles_at:
                raise ValueError("input mesh has an unreferenced vertex")
            ring = [triangles_at[0]]
            while True:
                current = triangles[ring[-1]]
                position = current.index(vertex)
                previous_vertex = current[(position - 1) % 3]
                next_triangle = edge_to_triangle.get((vertex, previous_vertex))
                if next_triangle is None:
                    raise ValueError("input mesh is not closed: boundary edge found")
                if next_triangle == ring[0]:
                    break
                ring.append(next_triangle)
            if len(ring) != len(triangles_at):
                raise ValueError("input mesh is not manifold: split vertex fan")
            self.vertex_rings.append(tuple(ring))


class _TopologyEncoder:
    """Mirror of the Appendix E mesh coder: same machine, io methods emit symbols."""

    def __init__(self, source: _SourceDualMesh, face_groups: list[int]) -> None:
        self.source = source
        self.face_groups = face_groups
        # emitted symbol streams
        self.degrees: list[list[int]] = [[] for _ in range(8)]
        self.valences: list[int] = []
        self.groups: list[int] = []
        self.flags: list[int] = []
        self.attr_masks: list[list[int]] = [[] for _ in range(8)]
        self.split_faces: list[int] = []
        self.split_positions: list[int] = []
        # dst bookkeeping (mirrors the decoder's DualVFMesh)
        self.vtx_valence: list[int] = []
        self.vtx_faces: list[list[int]] = []
        self.face_degree: list[int] = []
        self.face_empty: list[int] = []
        self.face_vts: list[list[int]] = []
        # src <-> dst mapping with ring offsets
        self.vtx_src: list[int] = []  # dst vtx -> src triangle
        self.vtx_offset: list[int] = []
        self.face_src: list[int] = []  # dst face -> src vertex
        self.face_offset: list[int] = []
        self.src_vtx_visited = [False] * len(source.triangle_rings)
        self.src_face_to_dst = [-1] * len(source.vertex_rings)
        self._active: list[int] = []
        self._removed: set[int] = set()
        self._seed_cursor = 0

    # -- dst mesh mirroring --

    def _new_vtx(self, src_vtx: int) -> int:
        valence = 3
        self.vtx_valence.append(valence)
        self.vtx_faces.append([-1] * valence)
        self.vtx_src.append(src_vtx)
        self.vtx_offset.append(0)
        self.src_vtx_visited[src_vtx] = True
        self.valences.append(valence)
        self.groups.append(self.face_groups[src_vtx])
        self.flags.append(0)
        return len(self.vtx_valence) - 1

    def _new_face(self, src_face: int) -> int:
        degree = len(self.source.vertex_rings[src_face])
        assert degree <= 64, "builder only emits <=64 degree faces"
        self.face_degree.append(degree)
        self.face_empty.append(degree)
        self.face_vts.append([-1] * degree)
        self.face_src.append(src_face)
        self.face_offset.append(0)
        self.src_face_to_dst[src_face] = len(self.face_degree) - 1
        return len(self.face_degree) - 1

    def _set_vtx_face(self, vtx: int, slot: int, face: int) -> None:
        self.vtx_faces[vtx][slot] = face

    def _set_face_vtx(self, face: int, slot: int, vtx: int) -> None:
        if self.face_vts[face][slot] != vtx:
            self.face_empty[face] -= 1
        self.face_vts[face][slot] = vtx

    def _face_context(self, vtx: int) -> int:
        valence = self.vtx_valence[vtx]
        known = 0
        total_degree = 0
        for face in self.vtx_faces[vtx]:
            if face < 0:
                continue
            known += 1
            total_degree += self.face_degree[face]
        if valence == 3:
            if total_degree < known * 6:
                return 0
            return 1 if total_degree == known * 6 else 2
        if valence == 4:
            if total_degree < known * 4:
                return 3
            return 4 if total_degree == known * 4 else 5
        return 6 if valence == 5 else 7

    # -- io methods (encoder realization) --

    def _io_vtx_seed(self) -> int:
        while self._seed_cursor < len(self.src_vtx_visited) and self.src_vtx_visited[self._seed_cursor]:
            self._seed_cursor += 1
        if self._seed_cursor >= len(self.src_vtx_visited):
            return -1
        return self._new_vtx(self._seed_cursor)

    def _io_vtx(self, face: int, face_slot: int) -> int:
        src_face = self.face_src[face]
        ring = self.source.vertex_rings[src_face]
        src_vtx = ring[(self.face_offset[face] + face_slot) % len(ring)]
        vtx = self._new_vtx(src_vtx)
        # dst vtx slot 0 corresponds to `face`, i.e. to src_face's position in the triangle ring.
        self.vtx_offset[vtx] = self.source.triangle_rings[src_vtx].index(src_face)
        return vtx

    def _io_face(self, vtx: int, vtx_slot: int) -> int:
        src_vtx = self.vtx_src[vtx]
        ring = self.source.triangle_rings[src_vtx]
        src_face = ring[(self.vtx_offset[vtx] + vtx_slot) % len(ring)]
        context = self._face_context(vtx)
        if self.src_face_to_dst[src_face] == -1:
            degree = len(self.source.vertex_rings[src_face])
            self.degrees[context].append(degree)
            self.attr_masks[min(7, max(0, degree - 2))].append(1)  # one record per face, bit 0
            face = self._new_face(src_face)
            # dst face slot 0 corresponds to `vtx`, i.e. to src_vtx's position in the vertex ring.
            self.face_offset[face] = self.source.vertex_rings[src_face].index(src_vtx)
            return face
        self.degrees[context].append(0)  # SPLIT
        return -1

    def _io_split_face(self, vtx: int, vtx_slot: int) -> int:
        src_vtx = self.vtx_src[vtx]
        ring = self.source.triangle_rings[src_vtx]
        src_face = ring[(self.vtx_offset[vtx] + vtx_slot) % len(ring)]
        face = self.src_face_to_dst[src_face]
        offset = -1
        for index in range(len(self._active) - 1, -1, -1):
            if self._active[index] == face:
                offset = len(self._active) - index
                break
        assert offset > 0, "split face must be in the active queue"
        self.split_faces.append(offset)
        return face

    def _io_split_position(self, vtx: int, face: int) -> int:
        src_vtx = self.vtx_src[vtx]
        src_face = self.face_src[face]
        ring = self.source.vertex_rings[src_face]
        position = (ring.index(src_vtx) - self.face_offset[face]) % len(ring)
        self.split_positions.append(position)
        return position

    # -- the shared machine (identical control flow to the decoder) --

    def run(self) -> None:
        while self._run_component():
            pass

    def _run_component(self) -> bool:
        vtx = self._io_vtx_seed()
        if vtx == -1:
            return False
        for slot in range(self.vtx_valence[vtx]):
            self._activate_face(vtx, slot)
        while (face := self._next_active_face()) != -1:
            self._complete_face(face)
            self._removed.add(face)
        return True

    def _complete_face(self, face: int) -> None:
        while True:
            try:
                slot = self.face_vts[face].index(-1)
            except ValueError:
                return
            vtx = self._activate_vtx(face, slot)
            self._complete_vtx(vtx, slot)

    def _activate_vtx(self, face: int, face_slot: int) -> int:
        vtx = self._io_vtx(face, face_slot)
        self._set_vtx_face(vtx, 0, face)
        self._add_vtx_to_face(vtx, 0, face, face_slot)
        return vtx

    def _activate_face(self, vtx: int, vtx_slot: int) -> int:
        face = self._io_face(vtx, vtx_slot)
        if face >= 0:
            self._set_vtx_face(vtx, vtx_slot, face)
            self._set_face_vtx(face, 0, vtx)
            self._active.append(face)
        else:
            face = self._io_split_face(vtx, vtx_slot)
            face_slot = self._io_split_position(vtx, face)
            self._set_vtx_face(vtx, vtx_slot, face)
            self._add_vtx_to_face(vtx, vtx_slot, face, face_slot)
        return face

    def _complete_vtx(self, vtx: int, vtx_slot_on_face0: int) -> None:
        valence = self.vtx_valence[vtx]
        vp = self.vtx_faces[vtx][0]
        jp = vtx_slot_on_face0
        i = 1
        while (vn := self.vtx_faces[vtx][i]) != -1:
            jp = (jp - 1) % self.face_degree[vp]
            vtx2 = self.face_vts[vp][jp]
            if vtx2 == -1:
                break
            jn = self.face_vts[vn].index(vtx2)
            jn = (jn - 1) % self.face_degree[vn]
            self._add_vtx_to_face(vtx, i, vn, jn)
            vp, jp = vn, jn
            i += 1
            if i >= valence:
                return
        i_last = i
        vp = self.vtx_faces[vtx][0]
        jp = vtx_slot_on_face0
        i = valence - 1
        while (vn := self.vtx_faces[vtx][i]) != -1:
            jp = (jp + 1) % self.face_degree[vp]
            vtx2 = self.face_vts[vp][jp]
            if vtx2 == -1:
                break
            jn = self.face_vts[vn].index(vtx2)
            jn = (jn + 1) % self.face_degree[vn]
            self._add_vtx_to_face(vtx, i, vn, jn)
            vp, jp = vn, jn
            i -= 1
            if i < i_last:
                return
        while i_last <= i:
            self._activate_face(vtx, i_last)
            i_last += 1

    def _add_vtx_to_face(self, vtx: int, vtx_face_slot: int, face: int, face_slot: int) -> None:
        degree = self.face_degree[face]
        slot_cw = (face_slot - 1) % degree
        slot_ccw = (face_slot + 1) % degree
        self._set_face_vtx(face, face_slot, vtx)
        valence = self.vtx_valence[vtx]
        neighbor = self.face_vts[face][slot_cw]
        if neighbor != -1:
            shared = self.vtx_faces[neighbor].index(face)
            vtx_slot_ccw = (vtx_face_slot + 1) % valence
            if self.vtx_faces[vtx][vtx_slot_ccw] == -1:
                shared = (shared - 1) % self.vtx_valence[neighbor]
                self._set_vtx_face(vtx, vtx_slot_ccw, self.vtx_faces[neighbor][shared])
        neighbor = self.face_vts[face][slot_ccw]
        if neighbor != -1:
            shared = self.vtx_faces[neighbor].index(face)
            vtx_slot_cw = (vtx_face_slot - 1) % valence
            if self.vtx_faces[vtx][vtx_slot_cw] == -1:
                shared = (shared + 1) % self.vtx_valence[neighbor]
                self._set_vtx_face(vtx, vtx_slot_cw, self.vtx_faces[neighbor][shared])

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
            empty = self.face_empty[face]
            if empty < lowest:
                lowest = empty
                best = face
            i -= 1
        return best


def build_tristrip_shape_lod_payload(
    points,
    triangles,
    *,
    normals=None,
    face_groups=None,
    byte_order: str = "<",
    version: tuple[int, int] = (9, 5),
    quant_bits: int = 0,
    cdp_codec: str = "null",
    corrupt_topology_hash: bool = False,
) -> bytes:
    """Emit a Tri-Strip Set Shape LOD element (topologically compressed, JT 9.5).

    `points` is (N, 3) float, `triangles` (M, 3) int CCW over a closed manifold
    mesh, `normals` optional (N, 3) per-vertex (written losslessly, one
    attribute record per topological vertex).
    """
    points = np.asarray(points, dtype=np.float32)
    triangle_list = [tuple(int(v) for v in triangle) for triangle in np.asarray(triangles)]
    if version[0] < 9:
        return _build_tristrip_shape_lod_payload_v8(
            points,
            triangle_list,
            normals,
            byte_order=byte_order,
            quant_bits=quant_bits,
        )
    groups = [0] * len(triangle_list) if face_groups is None else [int(g) for g in face_groups]
    source = _SourceDualMesh(triangle_list, len(points))
    encoder = _TopologyEncoder(source, groups)
    encoder.run()
    if version[0] >= 10:
        return _build_tristrip_shape_lod_payload_v10(
            points,
            encoder,
            normals,
            byte_order=byte_order,
            quant_bits=quant_bits,
            corrupt_topology_hash=corrupt_topology_hash,
        )

    def cdp(values: list[int], predictor: str = "null") -> bytes:
        return encode_cdp2(values, codec=cdp_codec, predictor=predictor, byte_order=byte_order)

    writer = JtWriter(byte_order)

    def version_field(target: JtWriter, value: int = 1) -> None:
        if version[0] < 10:
            target.i16(value)
        else:
            target.u8(value)

    bindings = 0x2 | (0x8 if normals is not None else 0)  # 3-component coords (+ normals)
    version_field(writer)  # Base Shape LOD Data version
    version_field(writer)  # Vertex Shape LOD Data version
    writer.u64(bindings)
    version_field(writer)  # TopoMesh LOD Data version
    writer.i32(1)  # vertex records object id
    version_field(writer)  # Topologically Compressed LOD Data version
    for context in range(8):
        writer.raw(cdp(encoder.degrees[context]))
    writer.raw(cdp(encoder.valences))
    writer.raw(cdp(encoder.groups))
    writer.raw(cdp(encoder.flags, "lag1"))
    masks8 = encoder.attr_masks[7]
    for context in range(7):
        writer.raw(cdp([mask & 0x3FFFFFFF for mask in encoder.attr_masks[context]]))
    writer.raw(cdp([mask & 0x3FFFFFFF for mask in masks8]))
    writer.raw(cdp([(mask >> 30) & 0x3FFFFFFF for mask in masks8]))
    writer.raw(cdp([(mask >> 60) & 0xF for mask in masks8]))
    writer.i32(0)  # high-degree attribute masks (none)
    writer.raw(cdp(encoder.split_faces, "lag1"))
    writer.raw(cdp(encoder.split_positions))
    writer.u32(_topology_hash(encoder) ^ (0xDEAD if corrupt_topology_hash else 0))

    # Topologically Compressed Vertex Records: coordinates (and normals) in face visit order.
    order = encoder.face_src  # dst face id -> src vertex id
    permuted_points = points[order]
    writer.u64(bindings)
    for _ in range(4):
        writer.u8(quant_bits)
    writer.i32(len(order))
    writer.i32(len(order) if normals is not None else 0)  # number of vertex attributes
    _write_coordinate_array(writer, permuted_points, quant_bits, cdp_codec, byte_order)
    if normals is not None:
        permuted_normals = np.asarray(normals, dtype=np.float32)[order]
        _write_lossless_normal_array(writer, permuted_normals, cdp_codec, byte_order)

    element = JtWriter(byte_order)
    element.i32(16 + 1 + 4 + len(writer.buffer))
    element.guid(TRISTRIP_SHAPE_LOD_ELEMENT_GUID)
    element.u8(4)  # Shape LOD base type
    element.i32(1)  # object id
    element.raw(bytes(writer.buffer))
    return bytes(element.buffer)


def _build_tristrip_shape_lod_payload_v8(
    points,
    triangle_list: list[tuple[int, ...]],
    normals,
    *,
    byte_order: str,
    quant_bits: int,
) -> bytes:
    """JT 8 layout: plain tri-strips over lossy uniform-quantized coordinates.

    One three-corner strip per triangle keeps the encoder trivial; the decoder
    handles arbitrary strip lengths. The element header carries no object id.
    """
    quant_bits = quant_bits or 12
    normal_bits = 10
    writer = JtWriter(byte_order)
    writer.i16(1)  # Vertex Shape LOD Data version
    writer.i32(0x2 | (0x8 if normals is not None else 0))  # binding attributes
    writer.raw(bytes(4))  # quantization parameters (restated below)
    writer.i16(1)  # TriStripSet Shape LOD version
    writer.i16(1)  # Vertex Based Shape Compressed Rep Data version
    writer.u8(1 if normals is not None else 0)  # normal binding (per vertex)
    writer.u8(0)  # texture-coordinate binding
    writer.u8(0)  # color binding
    writer.u8(quant_bits)  # bits per vertex
    writer.u8(normal_bits)  # normal bits factor
    writer.u8(0)  # bits per texture coordinate
    writer.u8(0)  # bits per color

    corners = [int(vertex) for triangle in triangle_list for vertex in triangle]
    strip_offsets = list(range(0, len(corners) + 1, 3))
    writer.raw(encode_cdp1(strip_offsets, predictor="stride1", byte_order=byte_order))

    points = np.asarray(points, dtype=np.float64)
    max_code = (1 << quant_bits) - 1
    quantized_columns = []
    for axis in range(3):
        minimum = float(points[:, axis].min())
        maximum = float(points[:, axis].max())
        writer.f32(minimum)
        writer.f32(maximum)
        writer.u8(quant_bits)
        if maximum == minimum:
            codes = np.zeros(len(points), dtype=np.int64)
        else:
            codes = np.rint((points[:, axis] - minimum) * max_code / (maximum - minimum)).astype(np.int64)
        quantized_columns.append([int(code) for code in codes])
    writer.i32(len(points))
    for column in quantized_columns:
        writer.raw(encode_cdp1(column, predictor="lag1", byte_order=byte_order))

    if normals is not None:
        writer.u8(normal_bits)
        writer.i32(len(points))
        for stream in encode_deering_normals(normals, normal_bits):
            writer.raw(encode_cdp1(stream, predictor="lag1", byte_order=byte_order))

    writer.raw(encode_cdp1(corners, predictor="stripindex", byte_order=byte_order))

    element = JtWriter(byte_order)
    element.i32(16 + 1 + len(writer.buffer))  # JT 8 headers carry no object id
    element.guid(TRISTRIP_SHAPE_LOD_ELEMENT_GUID)
    element.u8(4)  # Shape LOD base type
    element.raw(bytes(writer.buffer))
    return bytes(element.buffer)


def _topology_hash(encoder: _TopologyEncoder) -> int:
    value = 0
    for context in range(8):
        value = _hash32_words([d & 0xFFFFFFFF for d in encoder.degrees[context]], value)
    value = _hash32_words([v & 0xFFFFFFFF for v in encoder.valences], value)
    value = _hash32_words([g & 0xFFFFFFFF for g in encoder.groups], value)
    value = _hash16_words([f & 0xFFFF for f in encoder.flags], value)
    for context in range(7):
        value = _hash32_words([mask & 0x3FFFFFFF for mask in encoder.attr_masks[context]], value)
    masks8 = encoder.attr_masks[7]
    value = _hash32_words([m & 0x3FFFFFFF for m in masks8], value)
    value = _hash32_words([(m >> 30) & 0x3FFFFFFF for m in masks8], value)
    value = _hash32_words([(m >> 60) & 0xF for m in masks8], value)
    value = _hash32_words([], value)  # high-degree masks
    value = _hash32_words([s & 0xFFFFFFFF for s in encoder.split_faces], value)
    value = _hash32_words([p & 0xFFFFFFFF for p in encoder.split_positions], value)
    return value


def _build_tristrip_shape_lod_payload_v10(
    points,
    encoder: _TopologyEncoder,
    normals,
    *,
    byte_order: str,
    quant_bits: int,
    corrupt_topology_hash: bool,
) -> bytes:
    """JT 10 layout: the compressed LOD data sits in a nested logical element."""

    def cdp(values: list[int], predictor: str = "null") -> bytes:
        return encode_cdp3(values, predictor=predictor, byte_order=byte_order)

    bindings = 0x2 | (0x8 if normals is not None else 0)
    nested = JtWriter(byte_order)
    nested.u8(1)  # TopoMesh LOD Data version
    nested.u32(1)  # vertex records object id
    nested.u8(1)  # TopoMesh Topologically Compressed LOD Data version
    for context in range(8):
        nested.raw(cdp(encoder.degrees[context]))
    nested.raw(cdp(encoder.valences))
    nested.raw(cdp(encoder.groups))
    nested.raw(cdp(encoder.flags, "lag1"))
    masks8 = encoder.attr_masks[7]
    for context in range(7):
        nested.raw(cdp([mask & 0xFFFFFFFF for mask in encoder.attr_masks[context]]))
    nested.raw(cdp([mask & 0xFFFFFFFF for mask in masks8]))
    nested.raw(cdp([(mask >> 32) & 0xFFFFFFFF for mask in masks8]))
    nested.i32(0)  # high-degree attribute masks (none)
    nested.raw(cdp(encoder.split_faces, "lag1"))
    nested.raw(cdp(encoder.split_positions))
    nested.u32(_topology_hash_v10(encoder) ^ (0xDEAD if corrupt_topology_hash else 0))

    order = encoder.face_src  # dst face id -> src vertex id
    permuted_points = points[order]
    nested.u64(bindings)
    for _ in range(4):
        nested.u8(quant_bits)
    nested.i32(len(order))
    nested.i32(len(order) if normals is not None else 0)  # number of vertex attributes
    _write_coordinate_array_v10(nested, permuted_points, quant_bits, byte_order)
    if normals is not None:
        permuted_normals = np.asarray(normals, dtype=np.float32)[order]
        _write_lossless_normal_array_v10(nested, permuted_normals, byte_order)

    body = JtWriter(byte_order)
    body.u8(1)  # Base Shape LOD Data version
    body.u8(1)  # Vertex Shape LOD Data version
    body.u64(bindings)
    body.i32(16 + 1 + 4 + len(nested.buffer))
    body.guid(TOPOMESH_TOPO_COMPRESSED_LOD_GUID)
    body.u8(9)  # TopoMesh Topologically Compressed LOD Data base type
    body.i32(1)  # object id
    body.raw(bytes(nested.buffer))

    element = JtWriter(byte_order)
    element.i32(16 + 1 + 4 + len(body.buffer))
    element.guid(TRISTRIP_SHAPE_LOD_ELEMENT_GUID)
    element.u8(4)  # Shape LOD base type
    element.i32(1)  # object id
    element.raw(bytes(body.buffer))
    return bytes(element.buffer)


def _topology_hash_v10(encoder: _TopologyEncoder) -> int:
    value = 0
    for context in range(8):
        value = _hash32_words([d & 0xFFFFFFFF for d in encoder.degrees[context]], value)
    value = _hash32_words([v & 0xFFFFFFFF for v in encoder.valences], value)
    value = _hash32_words([g & 0xFFFFFFFF for g in encoder.groups], value)
    value = _hash16_words([f & 0xFFFF for f in encoder.flags], value)
    for context in range(7):
        value = _hash32_words([m & 0xFFFFFFFF for m in encoder.attr_masks[context]], value)
    masks8 = encoder.attr_masks[7]
    value = _hash32_words([m & 0xFFFFFFFF for m in masks8], value)
    value = _hash32_words([(m >> 32) & 0xFFFFFFFF for m in masks8], value)
    value = _hash32_words([], value)  # high-degree masks
    value = _hash32_words([s & 0xFFFFFFFF for s in encoder.split_faces], value)
    value = _hash32_words([p & 0xFFFFFFFF for p in encoder.split_positions], value)
    return value


def _float_bit_patterns(column) -> list[int]:
    """Raw IEEE-754 float32 bit patterns as signed int32 values."""
    bits = np.ascontiguousarray(column, dtype="<f4").view("<u4").astype(np.int64)
    return np.where(bits >= 1 << 31, bits - (1 << 32), bits).tolist()


def _write_coordinate_array_v10(writer: JtWriter, points, quant_bits: int, byte_order: str) -> None:
    writer.i32(len(points))
    writer.u8(3)
    hash_value = 0
    if quant_bits == 0:
        for _axis in range(3):
            writer.f32(0.0)
            writer.f32(0.0)
            writer.u8(0)
        for axis in range(3):
            codes = _float_bit_patterns(points[:, axis])
            writer.raw(encode_cdp3(codes, predictor="lag1", byte_order=byte_order))
            hash_value = _hash32_words([c & 0xFFFFFFFF for c in codes], hash_value)
    else:
        max_code = (1 << quant_bits) - 1
        ranges = []
        for axis in range(3):
            column = points[:, axis].astype(np.float64)
            minimum, maximum = float(column.min()), float(column.max())
            if maximum == minimum:
                maximum = minimum + 1.0
            ranges.append((minimum, maximum))
        for minimum, maximum in ranges:
            writer.f32(minimum)
            writer.f32(maximum)
            writer.u8(quant_bits)
        for axis in range(3):
            minimum, maximum = ranges[axis]
            column = points[:, axis].astype(np.float64)
            multiplier = max_code / (maximum - minimum)
            codes = np.clip((column - minimum) * multiplier + 0.5, 0, max_code).astype(np.int64).tolist()
            writer.raw(encode_cdp3(codes, predictor="lag1", byte_order=byte_order))
            hash_value = _hash32_words([c & 0xFFFFFFFF for c in codes], hash_value)
    writer.u32(hash_value)


def _write_lossless_normal_array_v10(writer: JtWriter, normals, byte_order: str) -> None:
    writer.i32(len(normals))
    writer.u8(3)
    writer.u8(0)  # quantization disabled: raw float bit patterns
    hash_value = 0
    for axis in range(3):
        codes = _float_bit_patterns(normals[:, axis])
        writer.raw(encode_cdp3(codes, byte_order=byte_order))
        hash_value = _hash32_words([c & 0xFFFFFFFF for c in codes], hash_value)
    writer.u32(hash_value)


def _write_coordinate_array(writer: JtWriter, points, quant_bits: int, cdp_codec: str, byte_order: str) -> None:
    writer.i32(len(points))
    writer.u8(3)
    hash_value = 0
    if quant_bits == 0:
        for _axis in range(3):
            writer.f32(0.0)
            writer.f32(0.0)
            writer.u8(0)
        payload = bytearray()
        for axis in range(3):
            column = np.ascontiguousarray(points[:, axis], dtype="<f4")
            bits = column.view("<u4")
            exponents = (bits >> 23).astype(np.int64).tolist()
            mantissae = (bits & 0x7FFFFF).astype(np.int64).tolist()
            payload += encode_cdp2(exponents, codec=cdp_codec, predictor="lag1", byte_order=byte_order)
            payload += encode_cdp2(mantissae, codec=cdp_codec, predictor="lag1", byte_order=byte_order)
            hash_value = _hash_float32_column(column, hash_value)
        writer.raw(bytes(payload))
    else:
        max_code = (1 << quant_bits) - 1
        payload = bytearray()
        ranges = []
        for axis in range(3):
            column = points[:, axis].astype(np.float64)
            minimum, maximum = float(column.min()), float(column.max())
            if maximum == minimum:
                maximum = minimum + 1.0
            ranges.append((minimum, maximum))
        for minimum, maximum in ranges:
            writer.f32(minimum)
            writer.f32(maximum)
            writer.u8(quant_bits)
        for axis in range(3):
            minimum, maximum = ranges[axis]
            column = points[:, axis].astype(np.float64)
            multiplier = max_code / (maximum - minimum)
            codes = np.clip((column - minimum) * multiplier + 0.5, 0, max_code).astype(np.int64)
            payload += encode_cdp2(codes.tolist(), codec=cdp_codec, predictor="lag1", byte_order=byte_order)
            hash_value = _hash32_words([c & 0xFFFFFFFF for c in codes.tolist()], hash_value)
        writer.raw(bytes(payload))
    writer.i32(hash_value - (1 << 32) if hash_value >= 1 << 31 else hash_value)


def _write_lossless_normal_array(writer: JtWriter, normals, cdp_codec: str, byte_order: str) -> None:
    writer.i32(len(normals))
    writer.u8(3)
    writer.u8(0)  # quantization disabled: lossless exponent/mantissa path
    hash_value = 0
    for axis in range(3):
        column = np.ascontiguousarray(normals[:, axis], dtype="<f4")
        bits = column.view("<u4")
        exponents = (bits >> 23).astype(np.int64).tolist()
        mantissae = (bits & 0x7FFFFF).astype(np.int64).tolist()
        writer.raw(encode_cdp2(exponents, codec=cdp_codec, byte_order=byte_order))
        writer.raw(encode_cdp2(mantissae, codec=cdp_codec, byte_order=byte_order))
        hash_value = _hash_float32_column(column, hash_value)
    writer.u32(hash_value)


class LsgBuilder:
    """Assembles an (uncompressed) LSG segment payload element by element."""

    def __init__(self, byte_order: str = "<", version: tuple[int, int] = (9, 5)) -> None:
        self.byte_order = byte_order
        self.version = version
        self._graph = JtWriter(byte_order)
        self._atoms = JtWriter(byte_order)
        self._tables: list[tuple[int, list[tuple[int, int]]]] = []
        self._next_id = 1

    def new_id(self) -> int:
        object_id = self._next_id
        self._next_id += 1
        return object_id

    def _version_field(self, writer: JtWriter, value: int = 1) -> None:
        if self.version[0] < 9:
            return  # JT 8 element bodies carry no per-element version fields
        if self.version[0] < 10:
            writer.i16(value)
        else:
            writer.u8(value)

    def _element(
        self,
        target: JtWriter,
        guid: tuple[int, int, int, bytes],
        base_type: int,
        object_id: int,
        body: bytes,
        pad: int = 0,
    ) -> None:
        target.i32(16 + 1 + 4 + len(body) + pad)
        target.guid(guid)
        target.u8(base_type)
        target.i32(object_id)
        target.raw(body)
        target.raw(bytes(pad))

    def _body(self) -> JtWriter:
        return JtWriter(self.byte_order)

    def _base_node_data(self, writer: JtWriter, attribute_ids: list[int]) -> None:
        self._version_field(writer)
        writer.u32(0)  # node flags
        writer.i32(len(attribute_ids))
        for attribute_id in attribute_ids:
            writer.i32(attribute_id)

    def _group_node_data(self, writer: JtWriter, attribute_ids: list[int], child_ids: list[int]) -> None:
        self._base_node_data(writer, attribute_ids)
        self._version_field(writer)
        writer.i32(len(child_ids))
        for child_id in child_ids:
            writer.i32(child_id)

    def add_partition(
        self,
        child_ids: list[int],
        *,
        attribute_ids: list[int] | None = None,
        file_name: str = "",
        object_id: int | None = None,
    ) -> int:
        object_id = self.new_id() if object_id is None else object_id
        body = self._body()
        self._group_node_data(body, attribute_ids or [], child_ids)
        if self.version[0] >= 10:
            body.u8(1)  # Partition Node Element version (new in JT 10)
        body.i32(0)  # partition flags
        body.mbstring(file_name)
        self._element(self._graph, PARTITION_NODE_GUID, 0, object_id, bytes(body.buffer))
        return object_id

    def add_group(self, child_ids: list[int], *, attribute_ids: list[int] | None = None) -> int:
        object_id = self.new_id()
        body = self._body()
        self._group_node_data(body, attribute_ids or [], child_ids)
        self._element(self._graph, GROUP_NODE_GUID, 1, object_id, bytes(body.buffer))
        return object_id

    def add_part(self, child_ids: list[int], *, attribute_ids: list[int] | None = None) -> int:
        object_id = self.new_id()
        body = self._body()
        self._group_node_data(body, attribute_ids or [], child_ids)
        self._version_field(body)  # meta data node data version
        self._version_field(body)  # part node version
        body.i32(0)  # reserved
        self._element(self._graph, PART_NODE_GUID, 1, object_id, bytes(body.buffer))
        return object_id

    def add_instance(self, child_id: int, *, attribute_ids: list[int] | None = None) -> int:
        object_id = self.new_id()
        body = self._body()
        self._base_node_data(body, attribute_ids or [])
        self._version_field(body)
        body.i32(child_id)
        self._element(self._graph, INSTANCE_NODE_GUID, 0, object_id, bytes(body.buffer))
        return object_id

    def _lod_node_data(self, writer: JtWriter, attribute_ids: list[int], child_ids: list[int]) -> None:
        self._group_node_data(writer, attribute_ids, child_ids)
        self._version_field(writer)
        writer.i32(0)  # reserved VecF32 (empty)
        writer.i32(0)  # reserved I32

    def add_lod(self, child_ids: list[int], *, attribute_ids: list[int] | None = None) -> int:
        object_id = self.new_id()
        body = self._body()
        self._lod_node_data(body, attribute_ids or [], child_ids)
        self._element(self._graph, LOD_NODE_GUID, 1, object_id, bytes(body.buffer))
        return object_id

    def add_range_lod(
        self,
        child_ids: list[int],
        *,
        range_limits: list[float] | None = None,
        attribute_ids: list[int] | None = None,
    ) -> int:
        object_id = self.new_id()
        body = self._body()
        self._lod_node_data(body, attribute_ids or [], child_ids)
        self._version_field(body)
        limits = range_limits or []
        body.i32(len(limits))
        for limit in limits:
            body.f32(limit)
        for _ in range(3):
            body.f32(0.0)  # center
        self._element(self._graph, RANGE_LOD_NODE_GUID, 1, object_id, bytes(body.buffer))
        return object_id

    def add_tristrip_shape_node(self, *, attribute_ids: list[int] | None = None) -> int:
        object_id = self.new_id()
        body = self._body()
        self._base_node_data(body, attribute_ids or [])
        self._element(self._graph, TRISTRIP_SHAPE_NODE_GUID, 2, object_id, bytes(body.buffer))
        return object_id

    def add_material(
        self,
        diffuse: tuple[float, float, float, float],
        *,
        shininess: float = 32.0,
        specular: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
        ambient: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0),
        emission: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        reflectivity: float = 0.5,
    ) -> int:
        object_id = self.new_id()
        body = self._body()
        self._version_field(body)  # base attribute version
        body.u8(0)  # state flags
        body.u32(0)  # field inhibit flags
        self._version_field(body, 2)  # material version 2
        if self.version[0] < 9:
            # JT 8 flag-driven channels (spec 7.2.1.1.2.2): a scalar
            # (c, c, c, 1) channel collapses to one F32 behind its pattern
            # bit; diffuse is always full RGBA and there is no reflectivity.
            def _is_scalar(color: tuple[float, float, float, float]) -> bool:
                return color[0] == color[1] == color[2] and color[3] == 1.0

            data_flags = 0
            for pattern_bit, color in ((0x0002, ambient), (0x0004, emission), (0x0008, specular)):
                if _is_scalar(color):
                    data_flags |= 0x0001 | pattern_bit
            body.u16(data_flags)

            def _channel(pattern_bit: int, color: tuple[float, float, float, float]) -> None:
                if data_flags & 0x0001 and data_flags & pattern_bit:
                    body.f32(color[0])
                else:
                    for component in color:
                        body.f32(component)

            _channel(0x0002, ambient)
            for component in diffuse:
                body.f32(component)
            _channel(0x0008, specular)
            _channel(0x0004, emission)
            body.f32(shininess)
        else:
            body.u16(0)  # data flags
            for color in (ambient, diffuse, specular, emission):
                for component in color:
                    body.f32(component)
            body.f32(shininess)
            body.f32(reflectivity)
        self._element(self._graph, MATERIAL_ATTRIBUTE_GUID, 3, object_id, bytes(body.buffer))
        return object_id

    def add_transform(self, matrix: list[list[float]]) -> int:
        """`matrix` is a 4x4 in JT row-vector convention (translation in the last row)."""
        object_id = self.new_id()
        body = self._body()
        self._version_field(body)
        body.u8(0)
        body.u32(0)
        self._version_field(body)
        identity = [[1.0 if row == col else 0.0 for col in range(4)] for row in range(4)]
        mask = 0
        stored = []
        for index in range(16):
            value = matrix[index // 4][index % 4]
            if value != identity[index // 4][index % 4]:
                mask |= 0x8000 >> index
                stored.append(value)
        body.u16(mask)
        for value in stored:
            if self.version[0] >= 9:
                body.f64(value)
            else:
                body.f32(value)
        self._element(self._graph, TRANSFORM_ATTRIBUTE_GUID, 3, object_id, bytes(body.buffer))
        return object_id

    def add_unknown_element(self, *, seed: int = 0xBAD, body: bytes = b"\xde\xad\xbe\xef" * 5) -> int:
        object_id = self.new_id()
        self._element(self._graph, make_guid(seed), 255, object_id, body)
        return object_id

    def add_string_atom(self, value: str) -> int:
        object_id = self.new_id()
        body = self._body()
        self._version_field(body)  # base property atom version
        body.u32(0)  # state flags
        self._version_field(body)
        body.mbstring(value)
        self._element(self._atoms, STRING_ATOM_GUID, 5, object_id, bytes(body.buffer))
        return object_id

    def add_integer_atom(self, value: int) -> int:
        object_id = self.new_id()
        body = self._body()
        self._version_field(body)
        body.u32(0)
        self._version_field(body)
        body.i32(value)
        self._element(self._atoms, INTEGER_ATOM_GUID, 5, object_id, bytes(body.buffer))
        return object_id

    def add_float_atom(self, value: float) -> int:
        object_id = self.new_id()
        body = self._body()
        self._version_field(body)
        body.u32(0)
        self._version_field(body)
        body.f32(value)
        self._element(self._atoms, FLOAT_ATOM_GUID, 5, object_id, bytes(body.buffer))
        return object_id

    def add_late_loaded_atom(self, segment_guid: tuple[int, int, int, bytes], segment_type: int) -> int:
        object_id = self.new_id()
        body = self._body()
        self._version_field(body)
        body.u32(0)
        self._version_field(body)
        body.guid(segment_guid)
        body.i32(segment_type)
        if self.version[0] >= 9:
            body.i32(object_id)  # payload object id (added in JT 9)
        body.i32(1)  # reserved, always >= 1
        self._element(self._atoms, LATE_LOADED_ATOM_GUID, 8, object_id, bytes(body.buffer))
        return object_id

    def set_properties(self, element_id: int, pairs: list[tuple[int, int]]) -> None:
        self._tables.append((element_id, pairs))

    def set_string_property(self, element_id: int, key: str, value: str) -> None:
        self.set_properties(element_id, [(self.add_string_atom(key), self.add_string_atom(value))])

    def attach_late_loaded(
        self, element_id: int, key: str, segment_guid: tuple[int, int, int, bytes], segment_type: int
    ) -> None:
        self.set_properties(
            element_id, [(self.add_string_atom(key), self.add_late_loaded_atom(segment_guid, segment_type))]
        )

    def payload(self) -> bytes:
        writer = JtWriter(self.byte_order)
        writer.raw(bytes(self._graph.buffer))
        self._end_of_elements(writer)
        writer.raw(bytes(self._atoms.buffer))
        self._end_of_elements(writer)
        writer.i16(1)  # property table version: I16 in both JT 9 and JT 10
        merged: dict[int, list[tuple[int, int]]] = {}
        for element_id, pairs in self._tables:
            merged.setdefault(element_id, []).extend(pairs)
        writer.i32(len(merged))
        for element_id, pairs in merged.items():
            writer.i32(element_id)
            for key_id, value_id in pairs:
                writer.i32(key_id)
                writer.i32(value_id)
            writer.i32(0)
        return bytes(writer.buffer)

    def _end_of_elements(self, writer: JtWriter) -> None:
        writer.i32(16)
        writer.guid(END_OF_ELEMENTS_GUID)


# --- Full-file assembly ---


@dataclass
class SyntheticPart:
    name: str
    points: object  # (N, 3) float array-like
    triangles: object  # (M, 3) int array-like, CCW, closed manifold
    normals: object | None = None  # optional (N, 3) per-vertex normals
    diffuse: tuple[float, float, float, float] | None = None
    shininess: float = 32.0
    jt_transform: list[list[float]] | None = None  # JT row-vector convention
    instances: int = 0  # extra Instance nodes referencing this part
    lod_meshes: list[tuple[object, object]] = field(default_factory=list)  # coarser (points, triangles)


def build_jt(
    parts: list[SyntheticPart],
    *,
    version: tuple[int, int] = (9, 5),
    byte_order: str = "<",
    units: str | None = "Millimeters",
    compress_lsg: bool = True,
    quant_bits: int = 0,
    cdp_codec: str = "null",
    brep_only: bool = False,
    external_ref: str | None = None,
    attach_shape_refs: bool = True,
    root_file_name: str = "",
    inject_unknown_element: bool = False,
    corrupt_shape: bool = False,
) -> bytes:
    """Assemble a complete synthetic JT file (header, LSG segment, shape segments, TOC)."""
    lsg = LsgBuilder(byte_order=byte_order, version=version)
    segments: list[SegmentSpec] = []
    guid_counter = 100
    if inject_unknown_element:
        lsg.add_unknown_element()
    root_children: list[int] = []
    for part in parts:
        attribute_ids: list[int] = []
        if part.diffuse is not None:
            attribute_ids.append(lsg.add_material(part.diffuse, shininess=part.shininess))
        if part.jt_transform is not None:
            attribute_ids.append(lsg.add_transform(part.jt_transform))
        lod_specs = [(part.points, part.triangles, part.normals)] + [
            (points, triangles, None) for points, triangles in part.lod_meshes
        ]
        shape_node_ids = []
        for lod_index, (points, triangles, normals) in enumerate(lod_specs):
            shape_node = lsg.add_tristrip_shape_node()
            segment_guid = make_guid(guid_counter)
            guid_counter += 1
            payload = build_tristrip_shape_lod_payload(
                points,
                triangles,
                normals=normals,
                byte_order=byte_order,
                version=version,
                quant_bits=quant_bits,
                cdp_codec=cdp_codec,
                corrupt_topology_hash=corrupt_shape,
            )
            segment_type = SEGMENT_SHAPE_LOD0 + min(lod_index, 9)
            segments.append(SegmentSpec(guid=segment_guid, segment_type=segment_type, payload=payload))
            if attach_shape_refs:
                lsg.attach_late_loaded(shape_node, "JT_LLPROP_SHAPEDATA", segment_guid, segment_type)
            shape_node_ids.append(shape_node)
        if len(shape_node_ids) > 1:
            limits = [100.0 * (index + 1) for index in range(len(shape_node_ids) - 1)]
            geometry_node = lsg.add_range_lod(shape_node_ids, range_limits=limits)
        else:
            geometry_node = shape_node_ids[0]
        part_node = lsg.add_part([geometry_node], attribute_ids=attribute_ids)
        lsg.set_string_property(part_node, "JT_PROP_NAME", f"{part.name};0;1:")
        root_children.append(part_node)
        for _ in range(part.instances):
            root_children.append(lsg.add_instance(part_node))
    if brep_only:
        brep_guid = make_guid(guid_counter)
        guid_counter += 1
        segments.append(SegmentSpec(guid=brep_guid, segment_type=SEGMENT_XT_BREP, payload=b"XT placeholder"))
        shape_node = lsg.add_tristrip_shape_node()
        lsg.attach_late_loaded(shape_node, "JT_LLPROP_BREPDATA", brep_guid, SEGMENT_XT_BREP)
        part_node = lsg.add_part([shape_node])
        lsg.set_string_property(part_node, "JT_PROP_NAME", "brep-only.part;0;1:")
        root_children.append(part_node)
    if external_ref is not None:
        root_children.append(lsg.add_partition([], file_name=external_ref))
    assembly = lsg.add_group(root_children)
    root = lsg.add_partition([assembly], file_name=root_file_name)
    if units is not None:
        lsg.set_string_property(root, "JT_PROP_MEASUREMENT_UNITS", units)
    lsg_guid = make_guid(1)
    segments.insert(
        0,
        SegmentSpec(
            guid=lsg_guid,
            segment_type=SEGMENT_LOGICAL_SCENE_GRAPH,
            payload=lsg.payload(),
            compression=COMPRESSION_ZLIB if compress_lsg else COMPRESSION_NONE,
        ),
    )
    return build_container(ContainerSpec(segments=segments, version=version, byte_order=byte_order, lsg_guid=lsg_guid))


def build_jt10_mismatched_shape() -> bytes:
    """A JT 10 file whose shape segment carries a JT 9.5-layout payload.

    The container and LSG parse, but shape decode fails on the nested-element
    GUID check, exercising the per-part skip path for JT 10 files.
    """
    tetra_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    tetra_faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    lsg = LsgBuilder(byte_order="<", version=(10, 0))
    shape_node = lsg.add_tristrip_shape_node()
    segment_guid = make_guid(200)
    payload = build_tristrip_shape_lod_payload(tetra_points, tetra_faces, version=(9, 5))
    lsg.attach_late_loaded(shape_node, "JT_LLPROP_SHAPEDATA", segment_guid, SEGMENT_SHAPE_LOD0)
    part_node = lsg.add_part([shape_node])
    lsg.set_string_property(part_node, "JT_PROP_NAME", "ten.part;0;1:")
    root = lsg.add_partition([part_node])
    lsg.set_string_property(root, "JT_PROP_MEASUREMENT_UNITS", "Millimeters")
    lsg_guid = make_guid(1)
    segments = [
        SegmentSpec(
            guid=lsg_guid,
            segment_type=SEGMENT_LOGICAL_SCENE_GRAPH,
            payload=lsg.payload(),
            compression=COMPRESSION_LZMA,
        ),
        SegmentSpec(guid=segment_guid, segment_type=SEGMENT_SHAPE_LOD0, payload=payload),
    ]
    return build_container(ContainerSpec(segments=segments, version=(10, 0), byte_order="<", lsg_guid=lsg_guid))
