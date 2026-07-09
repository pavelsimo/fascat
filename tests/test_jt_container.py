from __future__ import annotations

import numpy as np
import pytest

from fascat.io.jt.container import (
    COMPRESSION_LZMA,
    COMPRESSION_NONE,
    COMPRESSION_ZLIB,
    SEGMENT_LOGICAL_SCENE_GRAPH,
    SEGMENT_SHAPE_LOD0,
    BitReader,
    ByteReader,
    load_segment,
    read_file_header,
    read_toc,
)
from tests._jt_builder import (
    ContainerSpec,
    SegmentSpec,
    build_container,
    build_jt8_bytes,
    guid_bytes,
    make_guid,
)


def _single_segment_file(
    payload: bytes,
    *,
    segment_type: int = SEGMENT_LOGICAL_SCENE_GRAPH,
    compression: int = COMPRESSION_NONE,
    version: tuple[int, int] = (9, 5),
    byte_order: str = "<",
) -> tuple[bytes, bytes]:
    guid = make_guid(7)
    spec = ContainerSpec(
        segments=[SegmentSpec(guid=guid, segment_type=segment_type, payload=payload, compression=compression)],
        version=version,
        byte_order=byte_order,
    )
    return build_container(spec), guid_bytes(guid)


class TestFileHeader:
    def test_parses_version_byte_order_and_lsg_guid(self) -> None:
        data, guid = _single_segment_file(b"payload")
        header = read_file_header(data)
        assert header.version == (9, 5)
        assert header.byte_order == "<"
        assert header.lsg_segment_id == guid
        assert 0 < header.toc_offset < len(data)

    def test_parses_big_endian_flag(self) -> None:
        data, _ = _single_segment_file(b"payload", byte_order=">")
        header = read_file_header(data)
        assert header.byte_order == ">"

    def test_parses_jt10_wide_toc_offset(self) -> None:
        data, _ = _single_segment_file(b"payload", version=(10, 5))
        header = read_file_header(data)
        assert header.version == (10, 5)
        assert 0 < header.toc_offset < len(data)

    def test_rejects_jt8(self) -> None:
        with pytest.raises(RuntimeError, match="unsupported JT version 8.1"):
            read_file_header(build_jt8_bytes())

    def test_rejects_garbage(self) -> None:
        with pytest.raises(RuntimeError, match="not a JT file"):
            read_file_header(b"\x00" * 200)

    def test_rejects_short_data(self) -> None:
        with pytest.raises(RuntimeError, match="not a JT file"):
            read_file_header(b"Version 9.5")

    def test_rejects_invalid_byte_order_flag(self) -> None:
        line = b"Version 9.5 JT".ljust(78) + b"\r\n"
        with pytest.raises(RuntimeError, match="byte-order flag"):
            read_file_header(line + bytes([9]) + bytes(28))


class TestToc:
    @pytest.mark.parametrize("byte_order", ["<", ">"])
    @pytest.mark.parametrize("version", [(9, 5), (10, 0)])
    def test_roundtrips_entries(self, byte_order: str, version: tuple[int, int]) -> None:
        data, guid = _single_segment_file(b"payload", byte_order=byte_order, version=version)
        header = read_file_header(data)
        toc = read_toc(data, header)
        assert len(toc.entries) == 1
        entry = toc.entries[0]
        assert entry.guid == guid
        assert entry.segment_type == SEGMENT_LOGICAL_SCENE_GRAPH
        assert entry.offset + entry.length <= header.toc_offset

    def test_find_returns_none_for_unknown_guid(self) -> None:
        data, guid = _single_segment_file(b"payload")
        toc = read_toc(data, read_file_header(data))
        assert toc.find(guid) is not None
        assert toc.find(b"\x00" * 16) is None

    def test_truncated_toc_raises(self) -> None:
        data, _ = _single_segment_file(b"payload")
        with pytest.raises(RuntimeError, match="truncated JT data"):
            read_toc(data[:-10], read_file_header(data))


class TestLoadSegment:
    @pytest.mark.parametrize("compression", [COMPRESSION_NONE, COMPRESSION_ZLIB, COMPRESSION_LZMA])
    def test_decompresses_per_algorithm_field(self, compression: int) -> None:
        payload = bytes(range(256)) * 8
        data, guid = _single_segment_file(payload, compression=compression)
        header = read_file_header(data)
        entry = read_toc(data, header).find(guid)
        assert entry is not None
        assert load_segment(data, entry, header=header) == payload

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    def test_shape_segments_are_returned_raw(self, byte_order: str) -> None:
        payload = b"\x01\x02\x03\x04raw shape bytes"
        data, guid = _single_segment_file(payload, segment_type=SEGMENT_SHAPE_LOD0, byte_order=byte_order)
        header = read_file_header(data)
        entry = read_toc(data, header).find(guid)
        assert entry is not None
        assert entry.segment_type == SEGMENT_SHAPE_LOD0
        assert load_segment(data, entry, header=header) == payload

    def test_unknown_compression_algorithm_raises(self) -> None:
        payload = b"payload"
        data, guid = _single_segment_file(payload)
        header = read_file_header(data)
        entry = read_toc(data, header).find(guid)
        assert entry is not None
        # The algorithm byte sits right after the segment header (24 bytes) and
        # the compression flag + compressed length fields (8 bytes).
        broken = bytearray(data)
        broken[entry.offset + 32] = 99
        with pytest.raises(RuntimeError, match="compression algorithm"):
            load_segment(bytes(broken), entry, header=header)

    def test_guid_mismatch_raises(self) -> None:
        data, guid = _single_segment_file(b"payload")
        header = read_file_header(data)
        entry = read_toc(data, header).find(guid)
        assert entry is not None
        broken = bytearray(data)
        broken[entry.offset] ^= 0xFF
        with pytest.raises(RuntimeError, match="GUID"):
            load_segment(bytes(broken), entry, header=header)

    def test_truncated_segment_raises(self) -> None:
        payload = bytes(1024)
        data, guid = _single_segment_file(payload, compression=COMPRESSION_ZLIB)
        header = read_file_header(data)
        entry = read_toc(data, header).find(guid)
        assert entry is not None
        with pytest.raises(RuntimeError, match="truncated JT data"):
            load_segment(data[: entry.offset + 30], entry, header=header)


class TestByteReader:
    def test_scalars_and_arrays_little_endian(self) -> None:
        reader = ByteReader(b"\x01\x00\x00\x00" + b"\x02\x00" + b"\x03\x00\x00\x00\x04\x00\x00\x00")
        assert reader.i32() == 1
        assert reader.u16() == 2
        assert np.array_equal(reader.read_i32_array(2), np.array([3, 4], dtype=np.int32))

    def test_scalars_big_endian(self) -> None:
        reader = ByteReader(b"\x00\x00\x00\x01\x00\x02", byte_order=">")
        assert reader.i32() == 1
        assert reader.u16() == 2

    def test_guid_is_byte_order_independent(self) -> None:
        little = ByteReader(bytes.fromhex("7856341234123412") + bytes(range(8)))
        big = ByteReader(bytes.fromhex("1234567812341234") + bytes(range(8)), byte_order=">")
        assert little.guid() == big.guid()

    def test_string_and_mbstring(self) -> None:
        writer_bytes = b"\x03\x00\x00\x00abc" + b"\x02\x00\x00\x00" + "hi".encode("utf-16-le")
        reader = ByteReader(writer_bytes)
        assert reader.string() == "abc"
        assert reader.mbstring() == "hi"

    def test_out_of_bounds_raises(self) -> None:
        reader = ByteReader(b"\x00\x00")
        with pytest.raises(RuntimeError, match="truncated JT data"):
            reader.i32()


class TestBitReader:
    def test_reads_msb_first_within_words(self) -> None:
        # 0x80000001: first bit set, last bit set within one 32-bit word.
        reader = BitReader(np.array([0x80000001], dtype=np.uint32))
        assert reader.read(1) == 1
        assert reader.read(30) == 0
        assert reader.read(1) == 1

    def test_reads_across_word_boundary(self) -> None:
        # Last 4 bits of word 0 = 0xA, first 4 bits of word 1 = 0xB → 0xAB.
        reader = BitReader(np.array([0x0000000A, 0xB0000000], dtype=np.uint32))
        assert reader.read(28) == 0
        assert reader.read(8) == 0xAB

    def test_read_fixed_matches_serial_reads(self) -> None:
        words = np.array([0xDEADBEEF, 0x12345678, 0x9ABCDEF0], dtype=np.uint32)
        serial = BitReader(words)
        expected = [serial.read(6) for _ in range(16)]
        bulk = BitReader(words)
        assert bulk.read_fixed(6, 16).tolist() == expected

    def test_read_signed(self) -> None:
        # 4-bit two's complement: 0b1111 = -1, 0b0111 = 7.
        reader = BitReader(np.array([0xF7000000], dtype=np.uint32))
        assert reader.read_signed(4) == -1
        assert reader.read_signed(4) == 7

    def test_read_fixed_signed(self) -> None:
        reader = BitReader(np.array([0xF7000000], dtype=np.uint32))
        assert reader.read_fixed_signed(4, 2).tolist() == [-1, 7]

    def test_exhausted_stream_raises(self) -> None:
        reader = BitReader(np.array([0], dtype=np.uint32))
        reader.read(32)
        with pytest.raises(RuntimeError, match="truncated JT bit stream"):
            reader.read(1)
