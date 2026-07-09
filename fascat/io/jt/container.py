"""JT file container: version header, TOC, segment framing, and decompression.

Field layouts follow the public Siemens *JT File Format Reference* (v9.5 and
v10.5 editions). All multi-byte fields honor the byte-order flag from the file
header; GUID keys are normalized to a canonical little-endian byte string so
lookups are byte-order independent.
"""

from __future__ import annotations

import lzma
import re
import struct
import zlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

JT_HEADER_VERSION_BYTES = 80

# Segment types (JT File Format Reference, segment-type table).
SEGMENT_LOGICAL_SCENE_GRAPH = 1
SEGMENT_JT_BREP = 2
SEGMENT_PMI = 3
SEGMENT_META_DATA = 4
SEGMENT_SHAPE = 6
SEGMENT_SHAPE_LOD0 = 7
SEGMENT_SHAPE_LOD9 = 16
SEGMENT_XT_BREP = 17

# Shape and Shape LOD segments carry codec-compressed payloads and are never
# segment-level compressed; every other segment type carries the compression block.
SHAPE_SEGMENT_TYPES = frozenset(range(SEGMENT_SHAPE, SEGMENT_SHAPE_LOD9 + 1))
BREP_SEGMENT_TYPES = frozenset({SEGMENT_JT_BREP, SEGMENT_XT_BREP})

COMPRESSION_NONE = 1
COMPRESSION_ZLIB = 2
COMPRESSION_LZMA = 3

_VERSION_PATTERN = re.compile(r"Version\s+(\d+)\.(\d+)")


class ByteReader:
    """Bounds-checked scalar/array reader over a bytes buffer with explicit endianness."""

    def __init__(self, data: bytes | bytearray | memoryview, *, byte_order: str = "<", offset: int = 0) -> None:
        if byte_order not in ("<", ">"):
            raise ValueError(f"invalid byte order: {byte_order!r}")
        self._view = memoryview(data).toreadonly()
        self.byte_order = byte_order
        self._pos = offset

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int) -> None:
        if offset < 0 or offset > len(self._view):
            raise RuntimeError(f"truncated JT data: seek to {offset} outside buffer of {len(self._view)} bytes")
        self._pos = offset

    def remaining(self) -> int:
        return len(self._view) - self._pos

    def remaining_view(self) -> memoryview:
        return self._view[self._pos :]

    def _require(self, count: int) -> None:
        if count < 0 or self._pos + count > len(self._view):
            raise RuntimeError(f"truncated JT data at offset {self._pos}: need {count} bytes, have {self.remaining()}")

    def read_bytes(self, count: int) -> bytes:
        self._require(count)
        value = bytes(self._view[self._pos : self._pos + count])
        self._pos += count
        return value

    def skip(self, count: int) -> None:
        self._require(count)
        self._pos += count

    def _scalar(self, code: str, size: int) -> int | float:
        self._require(size)
        value = struct.unpack_from(f"{self.byte_order}{code}", self._view, self._pos)[0]
        self._pos += size
        return value  # type: ignore[no-any-return]

    def u8(self) -> int:
        return int(self._scalar("B", 1))

    def i16(self) -> int:
        return int(self._scalar("h", 2))

    def u16(self) -> int:
        return int(self._scalar("H", 2))

    def i32(self) -> int:
        return int(self._scalar("i", 4))

    def u32(self) -> int:
        return int(self._scalar("I", 4))

    def i64(self) -> int:
        return int(self._scalar("q", 8))

    def u64(self) -> int:
        return int(self._scalar("Q", 8))

    def f32(self) -> float:
        return float(self._scalar("f", 4))

    def f64(self) -> float:
        return float(self._scalar("d", 8))

    def guid(self) -> bytes:
        """Read a GUID (U32/U16/U16/U8[8]) and normalize to canonical little-endian bytes."""
        part1 = self.u32()
        part2 = self.u16()
        part3 = self.u16()
        tail = self.read_bytes(8)
        return struct.pack("<IHH", part1, part2, part3) + tail

    def string(self) -> str:
        """I32 count followed by single-byte characters."""
        count = self.i32()
        if count < 0:
            raise RuntimeError(f"truncated JT data: negative string length {count}")
        return self.read_bytes(count).decode("latin-1")

    def mbstring(self) -> str:
        """I32 count followed by UTF-16 code units."""
        count = self.i32()
        if count < 0:
            raise RuntimeError(f"truncated JT data: negative string length {count}")
        raw = self.read_bytes(count * 2)
        encoding = "utf-16-le" if self.byte_order == "<" else "utf-16-be"
        return raw.decode(encoding)

    def _vector(self, dtype: str, count: int, itemsize: int) -> npt.NDArray[np.generic]:
        if count < 0:
            raise RuntimeError(f"truncated JT data: negative array length {count}")
        self._require(count * itemsize)
        array = np.frombuffer(self._view, dtype=np.dtype(f"{self.byte_order}{dtype}"), count=count, offset=self._pos)
        self._pos += count * itemsize
        return array

    def read_i32_array(self, count: int) -> npt.NDArray[np.int32]:
        return self._vector("i4", count, 4).astype(np.int32, copy=False)

    def read_u32_array(self, count: int) -> npt.NDArray[np.uint32]:
        return self._vector("u4", count, 4).astype(np.uint32, copy=False)

    def read_f32_array(self, count: int) -> npt.NDArray[np.float32]:
        return self._vector("f4", count, 4).astype(np.float32, copy=False)

    def read_f64_array(self, count: int) -> npt.NDArray[np.float64]:
        return self._vector("f8", count, 8).astype(np.float64, copy=False)

    def vec_i32(self) -> npt.NDArray[np.int32]:
        """I32 count followed by that many I32 values."""
        return self.read_i32_array(self.i32())

    def vec_u32(self) -> npt.NDArray[np.uint32]:
        return self.read_u32_array(self.i32())

    def vec_f32(self) -> npt.NDArray[np.float32]:
        return self.read_f32_array(self.i32())

    def vec_f64(self) -> npt.NDArray[np.float64]:
        return self.read_f64_array(self.i32())


class BitReader:
    """MSB-first bit reader over U32 code-text words."""

    def __init__(self, words: npt.NDArray[np.uint32]) -> None:
        self._bits: npt.NDArray[np.uint8] = np.unpackbits(
            np.ascontiguousarray(words, dtype=np.uint32).astype(">u4").view(np.uint8)
        )
        self._pos = 0

    @classmethod
    def from_bytes(cls, data: bytes | memoryview) -> BitReader:
        reader = cls.__new__(cls)
        reader._bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        reader._pos = 0
        return reader

    @property
    def bit_count(self) -> int:
        return int(self._bits.size)

    def tell(self) -> int:
        return self._pos

    def remaining(self) -> int:
        return int(self._bits.size) - self._pos

    def _require(self, nbits: int) -> None:
        if nbits < 0 or self._pos + nbits > self._bits.size:
            raise RuntimeError(
                f"truncated JT bit stream at bit {self._pos}: need {nbits} bits, have {self.remaining()}"
            )

    def read(self, nbits: int) -> int:
        if nbits == 0:
            return 0
        self._require(nbits)
        chunk = self._bits[self._pos : self._pos + nbits]
        self._pos += nbits
        value = 0
        for bit in chunk:
            value = (value << 1) | int(bit)
        return value

    def read_signed(self, nbits: int) -> int:
        value = self.read(nbits)
        if nbits > 0 and value >= 1 << (nbits - 1):
            value -= 1 << nbits
        return value

    def read_bit(self) -> int:
        self._require(1)
        value = int(self._bits[self._pos])
        self._pos += 1
        return value

    def read_fixed(self, nbits: int, count: int) -> npt.NDArray[np.int64]:
        """Bulk-read `count` unsigned fields of `nbits` bits each."""
        if count == 0:
            return np.zeros(0, dtype=np.int64)
        if nbits == 0:
            return np.zeros(count, dtype=np.int64)
        self._require(nbits * count)
        chunk = self._bits[self._pos : self._pos + nbits * count]
        self._pos += nbits * count
        weights = 1 << np.arange(nbits - 1, -1, -1, dtype=np.int64)
        return chunk.reshape(count, nbits).astype(np.int64) @ weights

    def read_fixed_signed(self, nbits: int, count: int) -> npt.NDArray[np.int64]:
        values = self.read_fixed(nbits, count)
        if nbits > 0:
            sign = np.int64(1) << (nbits - 1)
            values = np.where(values >= sign, values - (np.int64(1) << nbits), values)
        return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class FileHeader:
    version: tuple[int, int]
    byte_order: str
    toc_offset: int
    lsg_segment_id: bytes


@dataclass(frozen=True)
class TocEntry:
    guid: bytes
    offset: int
    length: int
    attributes: int

    @property
    def segment_type(self) -> int:
        return (self.attributes >> 24) & 0xFF


@dataclass(frozen=True)
class Toc:
    entries: tuple[TocEntry, ...]

    def find(self, guid: bytes) -> TocEntry | None:
        for entry in self.entries:
            if entry.guid == guid:
                return entry
        return None


def read_file_header(data: bytes) -> FileHeader:
    if len(data) < JT_HEADER_VERSION_BYTES + 1:
        raise RuntimeError("not a JT file: missing version header")
    version_line = data[:JT_HEADER_VERSION_BYTES].decode("ascii", errors="replace")
    match = _VERSION_PATTERN.search(version_line)
    if match is None:
        raise RuntimeError("not a JT file: missing version header")
    version = (int(match.group(1)), int(match.group(2)))
    if version[0] < 9:
        raise RuntimeError(f"unsupported JT version {version[0]}.{version[1]}: fascat supports JT 9.x and 10.x")
    order_flag = data[JT_HEADER_VERSION_BYTES]
    if order_flag not in (0, 1):
        raise RuntimeError(f"not a JT file: invalid byte-order flag {order_flag}")
    byte_order = "<" if order_flag == 0 else ">"
    reader = ByteReader(data, byte_order=byte_order, offset=JT_HEADER_VERSION_BYTES + 1)
    reader.i32()  # reserved field
    # JT 10 widened the TOC offset (and TOC entry offsets) to 64 bits.
    toc_offset = reader.u64() if version[0] >= 10 else reader.u32()
    lsg_segment_id = reader.guid()
    return FileHeader(version=version, byte_order=byte_order, toc_offset=toc_offset, lsg_segment_id=lsg_segment_id)


def read_toc(data: bytes, header: FileHeader) -> Toc:
    reader = ByteReader(data, byte_order=header.byte_order, offset=header.toc_offset)
    count = reader.i32()
    if count < 0:
        raise RuntimeError(f"truncated JT data: negative TOC entry count {count}")
    wide = header.version[0] >= 10
    entries = []
    for _ in range(count):
        guid = reader.guid()
        offset = reader.u64() if wide else reader.u32()
        length = reader.u32()
        attributes = reader.u32()
        entries.append(TocEntry(guid=guid, offset=offset, length=length, attributes=attributes))
    return Toc(entries=tuple(entries))


def load_segment(data: bytes, entry: TocEntry, *, header: FileHeader) -> bytes:
    """Return a segment's payload with segment-level compression undone.

    Shape/Shape-LOD segments are returned raw (their compression lives inside
    the codecs); all other segment types carry the compression block, whose
    per-segment algorithm field is honored rather than assumed by version.
    """
    reader = ByteReader(data, byte_order=header.byte_order, offset=entry.offset)
    guid = reader.guid()
    if guid != entry.guid:
        raise RuntimeError("corrupt JT segment: header GUID does not match TOC entry")
    reader.i32()  # segment type (also present in the TOC attributes)
    reader.i32()  # segment length
    end = entry.offset + entry.length
    if end > len(data):
        raise RuntimeError(f"truncated JT data: segment at {entry.offset} extends past end of file")
    if entry.segment_type in SHAPE_SEGMENT_TYPES:
        return bytes(data[reader.tell() : end])
    # Logical Element Header ZLIB block: flag, compressed length (which counts
    # the algorithm byte), and algorithm are always present in these segments.
    compression_flag = reader.i32()
    compressed_length = reader.i32()
    algorithm = reader.u8()
    if compression_flag != 2 or algorithm == COMPRESSION_NONE:
        return bytes(data[reader.tell() : end])
    payload = reader.read_bytes(compressed_length - 1)
    if algorithm == COMPRESSION_ZLIB:
        return zlib.decompress(payload)
    if algorithm == COMPRESSION_LZMA:
        return _decompress_lzma(payload)
    raise RuntimeError(f"unsupported JT segment compression algorithm: {algorithm}")


def _decompress_lzma(payload: bytes) -> bytes:
    """Decompress an LZMA payload: 5-byte properties header followed by the stream."""
    try:
        return lzma.decompress(payload, format=lzma.FORMAT_ALONE)
    except lzma.LZMAError:
        pass
    if len(payload) < 5:
        raise RuntimeError("truncated JT data: LZMA payload shorter than its properties header")
    props = payload[0]
    literal_context_bits = props % 9
    literal_position_bits = (props // 9) % 5
    position_bits = props // 45
    dict_size = int.from_bytes(payload[1:5], "little")
    filters = [
        {
            "id": lzma.FILTER_LZMA1,
            "dict_size": max(dict_size, 4096),
            "lc": literal_context_bits,
            "lp": literal_position_bits,
            "pb": position_bits,
        }
    ]
    try:
        return lzma.decompress(payload[5:], format=lzma.FORMAT_RAW, filters=filters)
    except lzma.LZMAError as exc:
        raise RuntimeError(f"failed to decompress JT LZMA segment: {exc}") from exc
