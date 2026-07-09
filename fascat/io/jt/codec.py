"""JT data codecs: Int32 CDP Mk.1/Mk.2/Mk.3, predictors, quantization, Deering normals, hashes.

Algorithms are transcribed from the public Siemens *JT File Format Reference*:
the Mk.2 packet layout from v9.5 Rev-A (sections 8.1-8.2 and Appendices C/D),
and the Bitlength CODEC plus the Mk.3 packet from v10 Rev-B (sections
13.1-13.2 / Appendix B) — the v9.5 document's prose section 8.2.2 still
describes the obsolete JT 8 Mk.1 bitlength algorithm, which real Mk.2 packets
do not use. JT 9.5 shape data uses the Mk.2 Int32 Compressed Data Packet
exclusively; JT 10 shape data uses the Mk.3 packet (nibbler-coded bitlength
headers, post-code-text probability contexts, and a Move-to-Front CODEC).

JT 8 shape data uses the Mk.1 packet (v9.5 Rev-A section 8.1.1, whose layout
is carried over from the v8.1 reference): pre-code-text probability contexts
with up to two tables and per-entry next-context switching, the adaptive-width
bitlength algorithm from section 8.2.2, and a Huffman CODEC (type 2) that JT 9
removed. The Huffman tree build must reproduce the writer's heap ordering
bit-exactly; the tie-breaking behavior here follows the MIT-licensed jcadlib
reader (consulted for details the spec leaves implicit, as disclosed in
specs/JT.md) and is validated against real Siemens-written JT 8 files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fascat.io.jt.container import BitReader, ByteReader

CODEC_NULL = 0
CODEC_BITLENGTH = 1
CODEC_HUFFMAN = 2  # Mk.1 only
CODEC_ARITHMETIC = 3
CODEC_CHOPPER = 4
CODEC_MOVE_TO_FRONT = 5  # Mk.3 only

_MOVE_TO_FRONT_WINDOW = 16

_ESCAPE_SYMBOL = -2
_MAX_CDP_RECURSION = 8
_INT32_SPAN = 1 << 32
_INT32_MIN = -(1 << 31)

PREDICTOR_NULL = "null"
PREDICTOR_LAG1 = "lag1"
PREDICTOR_LAG2 = "lag2"
PREDICTOR_STRIDE1 = "stride1"
PREDICTOR_STRIDE2 = "stride2"
PREDICTOR_STRIP_INDEX = "stripindex"
PREDICTOR_RAMP = "ramp"
PREDICTOR_XOR1 = "xor1"
PREDICTOR_XOR2 = "xor2"


def decode_int32_cdp2(reader: ByteReader, predictor: str = PREDICTOR_NULL) -> npt.NDArray[np.int64]:
    """Decode an Int32 Compressed Data Packet Mk.2 and unpack its predictor residuals."""
    residuals = _decode_cdp2_residuals(reader, 0)
    return unpack_residuals(residuals, predictor)


def _decode_cdp2_residuals(reader: ByteReader, depth: int) -> npt.NDArray[np.int64]:
    if depth > _MAX_CDP_RECURSION:
        raise RuntimeError("corrupt JT data: CDP recursion depth exceeded")
    count = reader.i32()
    if count < 0:
        raise RuntimeError(f"corrupt JT data: negative CDP value count {count}")
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    codec = reader.u8()
    if codec == CODEC_CHOPPER:
        chop_bits = reader.u8()
        if chop_bits == 0:
            # No chopping happened: a single nested packet holds the values.
            values = _decode_cdp2_residuals(reader, depth + 1)
            if len(values) != count:
                raise RuntimeError("corrupt JT data: chopper field length mismatch")
            return values
        bias = reader.i32()
        span_bits = reader.u8()
        msb = _decode_cdp2_residuals(reader, depth + 1)
        lsb = _decode_cdp2_residuals(reader, depth + 1)
        if len(msb) != count or len(lsb) != count:
            raise RuntimeError("corrupt JT data: chopper field length mismatch")
        return _wrap_int32((msb << (span_bits - chop_bits) | lsb) + bias)
    if codec not in (CODEC_NULL, CODEC_BITLENGTH, CODEC_ARITHMETIC):
        raise RuntimeError(f"unsupported JT CODEC type: {codec}")
    code_text_length = reader.i32()
    if code_text_length < 0:
        raise RuntimeError(f"corrupt JT data: negative code-text length {code_text_length}")
    if codec == CODEC_NULL:
        # The Null CODEC's length field counts bytes, not bits.
        words = reader.read_u32_array(code_text_length // 4)
        if len(words) != count:
            raise RuntimeError("corrupt JT data: Null CODEC word count mismatch")
        return words.astype(np.int32).astype(np.int64)
    words = reader.read_u32_array((code_text_length + 31) // 32)
    if codec == CODEC_BITLENGTH:
        return _decode_bitlength(BitReader(words), count, code_text_length)
    context = _read_probability_context(reader)
    out_of_band = _decode_cdp2_residuals(reader, depth + 1)
    if code_text_length == 0 and len(out_of_band) == count:
        return out_of_band
    return _decode_arithmetic(BitReader(words), count, context, out_of_band)


def _decode_bitlength(bits: BitReader, count: int, code_text_bits: int) -> npt.NDArray[np.int64]:
    """BitLengthCodec2 (v10 spec 13.2.2 / Appendix B.2): fixed- or variable-width fields."""
    out = np.empty(count, dtype=np.int64)
    if bits.read_bit() == 0:
        # Fixed-width mode: signed min/max prefixed by their own bit sizes.
        min_bits = bits.read(6)
        max_bits = bits.read(6)
        minimum = bits.read_signed(min_bits)
        maximum = bits.read_signed(max_bits)
        if maximum - minimum <= 0:
            out[:] = minimum
        else:
            width = int(maximum - minimum).bit_length()
            out[:] = bits.read_fixed(width, count) + minimum
    else:
        # Variable-width mode: runs of fields around a mean, with saturating
        # width-delta escapes.
        mean = bits.read_signed(32)
        delta_bits = bits.read(3)
        run_bits = bits.read(3)
        if delta_bits == 0:
            raise RuntimeError("corrupt JT data: zero bitlength delta field size")
        saturate_low = -(1 << (delta_bits - 1))
        saturate_high = (1 << (delta_bits - 1)) - 1
        width = 0
        produced = 0
        while produced < count:
            while True:
                delta = bits.read_signed(delta_bits)
                width += delta
                if delta != saturate_low and delta != saturate_high:
                    break
            if width < 0 or width > 32:
                raise RuntimeError(f"corrupt JT data: bitlength field width {width} out of range")
            run = bits.read(run_bits)
            if produced + run > count:
                raise RuntimeError("corrupt JT data: bitlength run overruns value count")
            out[produced : produced + run] = bits.read_fixed_signed(width, run) + mean
            produced += run
    if bits.tell() != code_text_bits:
        raise RuntimeError(f"corrupt JT data: bitlength code text consumed {bits.tell()} of {code_text_bits} bits")
    return _wrap_int32(out)


# --- Int32 Compressed Data Packet Mk.3 (JT 10, spec v10 Rev-B section 13.1) ---


def decode_int32_cdp3(reader: ByteReader, predictor: str = PREDICTOR_NULL) -> npt.NDArray[np.int64]:
    """Decode an Int32 Compressed Data Packet Mk.3 and unpack its predictor residuals."""
    residuals = _decode_cdp3_residuals(reader, 0)
    return unpack_residuals(residuals, predictor)


def _decode_cdp3_residuals(reader: ByteReader, depth: int) -> npt.NDArray[np.int64]:
    if depth > _MAX_CDP_RECURSION:
        raise RuntimeError("corrupt JT data: CDP recursion depth exceeded")
    count = reader.i32()
    if count <= 0:
        return np.zeros(0, dtype=np.int64)
    codec = reader.u8()
    if codec == CODEC_CHOPPER:
        chop_bits = reader.u8()
        bias = reader.i32()
        span_bits = reader.u8()
        msb = _decode_cdp3_residuals(reader, depth + 1)
        lsb = _decode_cdp3_residuals(reader, depth + 1)
        if len(msb) != count or len(lsb) != count:
            raise RuntimeError("corrupt JT data: chopper field length mismatch")
        return _wrap_int32((msb << (span_bits - chop_bits) | lsb) + bias)
    if codec == CODEC_MOVE_TO_FRONT:
        literals = _decode_cdp3_residuals(reader, depth + 1)
        offsets = _decode_cdp3_residuals(reader, depth + 1)
        if len(offsets) != count:
            raise RuntimeError("corrupt JT data: move-to-front offset count mismatch")
        return _decode_move_to_front(literals, offsets)
    if codec not in (CODEC_NULL, CODEC_BITLENGTH, CODEC_ARITHMETIC):
        raise RuntimeError(f"unsupported JT CODEC type: {codec}")
    code_text_length = reader.i32()
    if code_text_length < 0:
        raise RuntimeError(f"corrupt JT data: negative code-text length {code_text_length}")
    if codec == CODEC_NULL:
        # As in Mk.2, the Null CODEC's length field counts bytes.
        words = reader.read_u32_array(code_text_length // 4)
        if len(words) != count:
            raise RuntimeError("corrupt JT data: Null CODEC word count mismatch")
        return words.astype(np.int32).astype(np.int64)
    words = reader.read_u32_array((code_text_length + 31) // 32)
    if codec == CODEC_BITLENGTH:
        return _decode_bitlength3(BitReader(words), count, code_text_length)
    # Mk.3 stores the probability context after the code text, and the
    # out-of-band packet is present only when the context has an escape entry.
    context, has_escape = _read_probability_context3(reader)
    out_of_band = _decode_cdp3_residuals(reader, depth + 1) if has_escape else np.zeros(0, dtype=np.int64)
    if code_text_length == 0 and len(out_of_band) == count:
        return out_of_band
    return _decode_arithmetic(BitReader(words), count, context, out_of_band)


def _nibbler_get_signed(bits: BitReader) -> int:
    """Nibbler decoding (v10 Appendix B): 4-bit chunks, LSB first, each followed by a continue bit."""
    value = 0
    shift = 0
    while True:
        value |= bits.read(4) << shift
        shift += 4
        if bits.read_bit() == 0:
            break
    sign_bits = min(shift, 32)
    if value >= 1 << (sign_bits - 1):
        value -= 1 << sign_bits
    return value


def _decode_bitlength3(bits: BitReader, count: int, code_text_bits: int) -> npt.NDArray[np.int64]:
    """Mk.3 Bitlength CODEC: as BitLengthCodec2 but with nibbler headers and 4/4 field sizes."""
    out = np.empty(count, dtype=np.int64)
    if bits.read_bit() == 0:
        minimum = _nibbler_get_signed(bits)
        maximum = _nibbler_get_signed(bits)
        width = int((maximum - minimum) & 0xFFFFFFFF).bit_length()
        out[:] = bits.read_fixed(width, count) + minimum
    else:
        mean = _nibbler_get_signed(bits)
        saturate_low = -8
        saturate_high = 7
        width = 0
        produced = 0
        while produced < count:
            while True:
                delta = bits.read_signed(4)
                width += delta
                if delta != saturate_low and delta != saturate_high:
                    break
            if width < 0 or width > 32:
                raise RuntimeError(f"corrupt JT data: bitlength field width {width} out of range")
            run = bits.read(4)
            if produced + run > count:
                raise RuntimeError("corrupt JT data: bitlength run overruns value count")
            out[produced : produced + run] = bits.read_fixed_signed(width, run) + mean
            produced += run
    if bits.tell() != code_text_bits:
        raise RuntimeError(f"corrupt JT data: bitlength code text consumed {bits.tell()} of {code_text_bits} bits")
    return _wrap_int32(out)


def _decode_move_to_front(literals: npt.NDArray[np.int64], offsets: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    window: list[int] = []
    out = np.empty(len(offsets), dtype=np.int64)
    position = 0
    for index, offset in enumerate(offsets.tolist()):
        if offset == -1:
            if position >= len(literals):
                raise RuntimeError("corrupt JT data: move-to-front literals exhausted")
            value = int(literals[position])
            position += 1
        elif 0 <= offset < len(window):
            value = window.pop(offset)
        else:
            raise RuntimeError(f"corrupt JT data: move-to-front offset {offset} out of range")
        window.insert(0, value)
        del window[_MOVE_TO_FRONT_WINDOW:]
        out[index] = value
    return out


# --- Int32 Compressed Data Packet Mk.1 (JT 8, spec v9.5 Rev-A section 8.1.1) ---


def decode_int32_cdp1(reader: ByteReader, predictor: str = PREDICTOR_NULL) -> npt.NDArray[np.int64]:
    """Decode an Int32 Compressed Data Packet Mk.1 and unpack its predictor residuals."""
    residuals = _decode_cdp1_residuals(reader, 0)
    return unpack_residuals(residuals, predictor)


def _decode_cdp1_residuals(reader: ByteReader, depth: int) -> npt.NDArray[np.int64]:
    if depth > _MAX_CDP_RECURSION:
        raise RuntimeError("corrupt JT data: CDP recursion depth exceeded")
    codec = reader.u8()
    contexts: list[_ProbabilityContext1] | None = None
    out_of_band = np.zeros(0, dtype=np.int64)
    if codec in (CODEC_HUFFMAN, CODEC_ARITHMETIC):
        contexts = _read_probability_contexts1(reader)
        out_of_band_count = reader.i32()
        if out_of_band_count < 0:
            raise RuntimeError(f"corrupt JT data: negative out-of-band count {out_of_band_count}")
        if out_of_band_count > 0:
            out_of_band = _decode_cdp1_residuals(reader, depth + 1)
    if codec == CODEC_NULL:
        # The Null CODEC stores a plain VecI32: count then raw values.
        count = reader.i32()
        if count < 0:
            raise RuntimeError(f"corrupt JT data: negative CDP value count {count}")
        return reader.read_i32_array(count).astype(np.int64)
    if codec not in (CODEC_BITLENGTH, CODEC_HUFFMAN, CODEC_ARITHMETIC):
        raise RuntimeError(f"unsupported JT CODEC type: {codec}")
    code_text_length = reader.i32()
    if code_text_length < 0:
        raise RuntimeError(f"corrupt JT data: negative code-text length {code_text_length}")
    count = reader.i32()
    if count < 0:
        raise RuntimeError(f"corrupt JT data: negative CDP value count {count}")
    symbol_count = count
    if contexts is not None and len(contexts) > 1:
        # With two probability context tables the symbol count exceeds the
        # value count: escapes decoded in table 1+ emit no value.
        symbol_count = reader.i32()
    word_count = reader.i32()
    if word_count < 0 or code_text_length > word_count * 32:
        raise RuntimeError("corrupt JT data: CDP code text shorter than its bit length")
    words = reader.read_u32_array(word_count)
    bits = BitReader(words)
    if codec == CODEC_BITLENGTH:
        return _decode_bitlength1(bits, count, code_text_length)
    assert contexts is not None
    if code_text_length == 0 and len(out_of_band) == count:
        return out_of_band
    if codec == CODEC_HUFFMAN:
        if len(contexts) != 1:
            raise RuntimeError("unsupported JT data: multi-context Huffman CODEC")
        return _decode_huffman(bits, count, code_text_length, contexts[0], out_of_band)
    return _decode_arithmetic1(bits, count, symbol_count, contexts, out_of_band)


@dataclass(frozen=True)
class _ProbabilityContext1:
    symbols: npt.NDArray[np.int64]
    counts: npt.NDArray[np.int64]
    values: npt.NDArray[np.int64]
    next_contexts: npt.NDArray[np.int64]
    cumulative: npt.NDArray[np.int64]  # cumulative count preceding each entry
    total: int


def _read_probability_contexts1(reader: ByteReader) -> list[_ProbabilityContext1]:
    """Int32 Probability Contexts Mk.1: one or two bit-packed tables, byte-aligned at the end.

    Only the first table stores associated values; the second table's entries
    inherit them by symbol from the first (spec 8.1.1.1).
    """
    table_count = reader.u8()
    if table_count not in (1, 2):
        raise RuntimeError(f"corrupt JT data: probability context table count {table_count}")
    bits = BitReader.from_bytes(reader.remaining_view())
    tables: list[_ProbabilityContext1] = []
    value_by_symbol: dict[int, int] = {}
    for table_index in range(table_count):
        entry_count = bits.read(32)
        symbol_bits = bits.read(6)
        occurrence_bits = bits.read(6)
        if table_index == 0:
            value_bits = bits.read(6)
            next_context_bits = bits.read(6)
            min_value = bits.read(32)
            if min_value >= 1 << 31:
                min_value -= _INT32_SPAN
        else:
            value_bits = 0
            next_context_bits = bits.read(6)
        symbols = np.empty(entry_count, dtype=np.int64)
        counts = np.empty(entry_count, dtype=np.int64)
        values = np.empty(entry_count, dtype=np.int64)
        next_contexts = np.empty(entry_count, dtype=np.int64)
        for index in range(entry_count):
            symbol = bits.read(symbol_bits) - 2
            symbols[index] = symbol
            counts[index] = bits.read(occurrence_bits)
            if table_index == 0:
                value = bits.read(value_bits) + min_value
                value_by_symbol[symbol] = value
            else:
                value = value_by_symbol.get(symbol, 0)
            values[index] = value
            next_contexts[index] = bits.read(next_context_bits) if next_context_bits else 0
        cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
        total = int(cumulative[-1])
        if total <= 0:
            raise RuntimeError("corrupt JT data: empty probability context")
        tables.append(
            _ProbabilityContext1(
                symbols=symbols,
                counts=counts,
                values=values,
                next_contexts=next_contexts,
                cumulative=cumulative,
                total=total,
            )
        )
    reader.skip((bits.tell() + 7) // 8)
    return tables


def _decode_bitlength1(bits: BitReader, count: int, code_text_bits: int) -> npt.NDArray[np.int64]:
    """Mk.1 Bitlength CODEC (v9.5 Rev-A section 8.2.2): adaptive width in 2-bit steps."""
    out = np.empty(count, dtype=np.int64)
    width = 0
    produced = 0
    while bits.tell() < code_text_bits:
        if produced >= count:
            raise RuntimeError("corrupt JT data: bitlength code text overruns value count")
        if bits.read_bit():
            adjust = bits.read_bit()
            step = 2 if adjust else -2
            width += step
            while bits.read_bit() == adjust:
                width += step
            if width < 0 or width > 32:
                raise RuntimeError(f"corrupt JT data: bitlength field width {width} out of range")
        out[produced] = bits.read_signed(width) if width else 0
        produced += 1
    if produced != count:
        raise RuntimeError(f"corrupt JT data: bitlength decoded {produced} of {count} values")
    return _wrap_int32(out)


@dataclass
class _HuffmanNode:
    count: int
    symbol: int
    value: int
    left: _HuffmanNode | None = None
    right: _HuffmanNode | None = None


def _huffman_heap_push(heap: list[_HuffmanNode], node: _HuffmanNode) -> None:
    # Sift-up with a strict comparison: ties keep the earlier entry above the
    # newcomer, which is what the JT 8 writer's heap does.
    heap.append(node)
    index = len(heap)
    while index != 1 and heap[index // 2 - 1].count > node.count:
        heap[index - 1] = heap[index // 2 - 1]
        index //= 2
    heap[index - 1] = node


def _huffman_heap_pop(heap: list[_HuffmanNode]) -> _HuffmanNode:
    top = heap[0]
    last = heap[-1]
    size = len(heap) - 1
    index = 1
    child = 2
    while child <= size:
        if child < size and heap[child - 1].count > heap[child].count:
            child += 1
        if last.count < heap[child - 1].count:
            break
        heap[index - 1] = heap[child - 1]
        index = child
        child *= 2
    heap[index - 1] = last
    heap.pop()
    return top


def _build_huffman_tree(context: _ProbabilityContext1) -> _HuffmanNode:
    heap: list[_HuffmanNode] = []
    for index in range(len(context.symbols)):
        node = _HuffmanNode(
            count=int(context.counts[index]),
            symbol=int(context.symbols[index]),
            value=int(context.values[index]),
        )
        _huffman_heap_push(heap, node)
    while len(heap) > 1:
        left = _huffman_heap_pop(heap)
        right = _huffman_heap_pop(heap)
        _huffman_heap_push(
            heap, _HuffmanNode(count=left.count + right.count, symbol=0, value=0, left=left, right=right)
        )
    return _huffman_heap_pop(heap)


def _decode_huffman(
    bits: BitReader,
    count: int,
    code_text_bits: int,
    context: _ProbabilityContext1,
    out_of_band: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """Mk.1 Huffman CODEC: rebuild the writer's tree and walk it bit by bit (1 = left)."""
    root = _build_huffman_tree(context)
    if root.left is None:
        raise RuntimeError("corrupt JT data: degenerate single-leaf Huffman tree")
    out = np.empty(count, dtype=np.int64)
    produced = 0
    oob_position = 0
    node = root
    while bits.tell() < code_text_bits:
        child = node.left if bits.read_bit() else node.right
        if child is None:
            raise RuntimeError("corrupt JT data: Huffman walk fell off the tree")
        node = child
        if node.left is None and node.right is None:
            if produced >= count:
                raise RuntimeError("corrupt JT data: Huffman code text overruns value count")
            if node.symbol == _ESCAPE_SYMBOL:
                if oob_position >= len(out_of_band):
                    raise RuntimeError("corrupt JT data: out-of-band values exhausted")
                out[produced] = out_of_band[oob_position]
                oob_position += 1
            else:
                out[produced] = node.value
            produced += 1
            node = root
    if produced != count:
        raise RuntimeError(f"corrupt JT data: Huffman decoded {produced} of {count} values")
    return out


def _decode_arithmetic1(
    bits: BitReader,
    count: int,
    symbol_count: int,
    contexts: list[_ProbabilityContext1],
    out_of_band: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """The Mk.1 arithmetic decoder: the Appendix C core plus per-symbol context switching.

    Escapes decoded while in a table other than 0 emit no value (spec 8.1.1,
    "Symbol Count" subtlety); every entry names the table for the next symbol.
    """

    def next_bit() -> int:
        return bits.read_bit() if bits.remaining() > 0 else 0

    code = 0
    for _ in range(16):
        code = (code << 1) | next_bit()
    low = 0
    high = 0xFFFF
    out = np.empty(count, dtype=np.int64)
    produced = 0
    oob_position = 0
    current = 0
    for _ in range(symbol_count):
        context = contexts[current]
        value_range = high - low + 1
        rescaled = ((code - low + 1) * context.total - 1) // value_range
        index = int(np.searchsorted(context.cumulative, rescaled, side="right")) - 1
        if index < 0 or index >= len(context.symbols):
            raise RuntimeError("corrupt JT data: arithmetic symbol out of range")
        low_count = int(context.cumulative[index])
        high_count = int(context.cumulative[index + 1])
        high = low + (value_range * high_count) // context.total - 1
        low = low + (value_range * low_count) // context.total
        while True:
            if (~(high ^ low)) & 0x8000:
                pass
            elif (low & 0x4000) and not (high & 0x4000):
                code ^= 0x4000
                low &= 0x3FFF
                high |= 0x4000
            else:
                break
            low = (low << 1) & 0xFFFF
            high = ((high << 1) | 1) & 0xFFFF
            code = ((code << 1) | next_bit()) & 0xFFFF
        symbol = int(context.symbols[index])
        if symbol != _ESCAPE_SYMBOL or current == 0:
            if produced >= count:
                raise RuntimeError("corrupt JT data: arithmetic symbols overrun value count")
            if symbol == _ESCAPE_SYMBOL:
                if oob_position >= len(out_of_band):
                    raise RuntimeError("corrupt JT data: out-of-band values exhausted")
                out[produced] = out_of_band[oob_position]
                oob_position += 1
            else:
                out[produced] = context.values[index]
            produced += 1
        current = int(context.next_contexts[index])
        if not 0 <= current < len(contexts):
            raise RuntimeError(f"corrupt JT data: arithmetic next-context {current} out of range")
    if produced != count:
        raise RuntimeError(f"corrupt JT data: arithmetic decoded {produced} of {count} values")
    return out


@dataclass(frozen=True)
class _ProbabilityContext:
    symbols: npt.NDArray[np.int64]
    values: npt.NDArray[np.int64]
    cumulative: npt.NDArray[np.int64]  # cumulative count preceding each entry
    total: int


def _read_probability_context(reader: ByteReader) -> _ProbabilityContext:
    """Int32 Probability Contexts Mk.2: a single bit-packed table, byte-aligned at the end."""
    bits = BitReader.from_bytes(reader.remaining_view())
    entry_count = bits.read(16)
    symbol_bits = bits.read(6)
    occurrence_bits = bits.read(6)
    value_bits = bits.read(6)
    min_value = bits.read(32)
    if min_value >= 1 << 31:
        min_value -= _INT32_SPAN
    symbols = np.empty(entry_count, dtype=np.int64)
    counts = np.empty(entry_count, dtype=np.int64)
    values = np.empty(entry_count, dtype=np.int64)
    for index in range(entry_count):
        symbols[index] = bits.read(symbol_bits) - 2
        counts[index] = bits.read(occurrence_bits)
        values[index] = bits.read(value_bits) + min_value
    reader.skip((bits.tell() + 7) // 8)
    cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    total = int(cumulative[-1])
    if total <= 0:
        raise RuntimeError("corrupt JT data: empty probability context")
    return _ProbabilityContext(symbols=symbols, values=values, cumulative=cumulative, total=total)


def _read_probability_context3(reader: ByteReader) -> tuple[_ProbabilityContext, bool]:
    """Mk.3 probability context: 7-bit value sizes and per-entry escape flags."""
    bits = BitReader.from_bytes(reader.remaining_view())
    entry_count = bits.read(16)
    occurrence_bits = bits.read(6)
    value_bits = bits.read(7)
    min_value = bits.read(32)
    if min_value >= 1 << 31:
        min_value -= _INT32_SPAN
    symbols = np.empty(entry_count, dtype=np.int64)
    counts = np.empty(entry_count, dtype=np.int64)
    values = np.empty(entry_count, dtype=np.int64)
    has_escape = False
    for index in range(entry_count):
        is_escape = bits.read_bit()
        counts[index] = bits.read(occurrence_bits)
        values[index] = bits.read(value_bits) + min_value
        symbols[index] = _ESCAPE_SYMBOL if is_escape else index
        has_escape = has_escape or bool(is_escape)
    reader.skip((bits.tell() + 7) // 8)
    cumulative = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    total = int(cumulative[-1])
    if total <= 0:
        raise RuntimeError("corrupt JT data: empty probability context")
    return (
        _ProbabilityContext(symbols=symbols, values=values, cumulative=cumulative, total=total),
        has_escape,
    )


def _decode_arithmetic(
    bits: BitReader,
    count: int,
    context: _ProbabilityContext,
    out_of_band: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    """The spec's 16-bit integer arithmetic decoder (Appendix C 3.2), bit-exact."""

    def next_bit() -> int:
        return bits.read_bit() if bits.remaining() > 0 else 0

    code = 0
    for _ in range(16):
        code = (code << 1) | next_bit()
    low = 0
    high = 0xFFFF
    total = context.total
    cumulative = context.cumulative
    out = np.empty(count, dtype=np.int64)
    oob_position = 0
    for produced in range(count):
        value_range = high - low + 1
        rescaled = ((code - low + 1) * total - 1) // value_range
        index = int(np.searchsorted(cumulative, rescaled, side="right")) - 1
        if index < 0 or index >= len(context.symbols):
            raise RuntimeError("corrupt JT data: arithmetic symbol out of range")
        low_count = int(cumulative[index])
        high_count = int(cumulative[index + 1])
        high = low + (value_range * high_count) // total - 1
        low = low + (value_range * low_count) // total
        while True:
            if (~(high ^ low)) & 0x8000:
                pass
            elif (low & 0x4000) and not (high & 0x4000):
                code ^= 0x4000
                low &= 0x3FFF
                high |= 0x4000
            else:
                break
            low = (low << 1) & 0xFFFF
            high = ((high << 1) | 1) & 0xFFFF
            code = ((code << 1) | next_bit()) & 0xFFFF
        symbol = int(context.symbols[index])
        if symbol == _ESCAPE_SYMBOL:
            if oob_position >= len(out_of_band):
                raise RuntimeError("corrupt JT data: out-of-band values exhausted")
            out[produced] = out_of_band[oob_position]
            oob_position += 1
        else:
            out[produced] = context.values[index]
    return out


def _wrap_int32(values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Reduce to Int32 two's-complement range, matching the reference C++ arithmetic."""
    wrapped: npt.NDArray[np.int64] = ((values - _INT32_MIN) % _INT32_SPAN + _INT32_MIN).astype(np.int64)
    return wrapped


def unpack_residuals(residuals: npt.NDArray[np.int64], predictor: str) -> npt.NDArray[np.int64]:
    """Reconstruct primal values from predictor residuals (Appendix C 1.3).

    The first four values are unpredicted primers; from index 4 on, each value
    is the residual plus (or XOR) the predicted value.
    """
    if predictor == PREDICTOR_NULL:
        return residuals
    length = len(residuals)
    out = residuals.astype(np.int64, copy=True)
    if length <= 4:
        return out
    if predictor == PREDICTOR_LAG1:
        out[3:] = np.cumsum(residuals[3:])
    elif predictor == PREDICTOR_LAG2:
        out[2::2] = np.cumsum(residuals[2::2])
        out[3::2] = np.cumsum(residuals[3::2])
    elif predictor == PREDICTOR_STRIDE1:
        deltas = np.cumsum(residuals[4:]) + (int(residuals[3]) - int(residuals[2]))
        out[4:] = int(residuals[3]) + np.cumsum(deltas)
    elif predictor == PREDICTOR_STRIDE2:
        even = np.cumsum(residuals[4::2]) + (int(residuals[2]) - int(residuals[0]))
        out[4::2] = int(residuals[2]) + np.cumsum(even)
        if length > 5:
            odd = np.cumsum(residuals[5::2]) + (int(residuals[3]) - int(residuals[1]))
            out[5::2] = int(residuals[3]) + np.cumsum(odd)
    elif predictor == PREDICTOR_STRIP_INDEX:
        for index in range(4, length):
            stride = int(out[index - 2]) - int(out[index - 4])
            predicted = int(out[index - 2]) + (stride if -8 < stride < 8 else 2)
            out[index] = int(residuals[index]) + predicted
    elif predictor == PREDICTOR_RAMP:
        out[4:] = residuals[4:] + np.arange(4, length, dtype=np.int64)
    elif predictor == PREDICTOR_XOR1:
        chain = np.concatenate([out[3:4], residuals[4:]]).astype(np.int64) & 0xFFFFFFFF
        out[3:] = np.bitwise_xor.accumulate(chain)
    elif predictor == PREDICTOR_XOR2:
        even = np.concatenate([out[2:3], residuals[4::2]]).astype(np.int64) & 0xFFFFFFFF
        out[2::2] = np.bitwise_xor.accumulate(even)
        odd = np.concatenate([out[3:4], residuals[5::2]]).astype(np.int64) & 0xFFFFFFFF
        out[3::2] = np.bitwise_xor.accumulate(odd)
    else:
        raise RuntimeError(f"unsupported JT predictor: {predictor}")
    if predictor in (PREDICTOR_XOR1, PREDICTOR_XOR2):
        out = np.where(out >= 1 << 31, out - _INT32_SPAN, out).astype(np.int64)
        return out
    return _wrap_int32(out)


def dequantize_uniform(
    codes: npt.NDArray[np.int64],
    minimum: float,
    maximum: float,
    bits: int,
) -> npt.NDArray[np.float64]:
    """Invert the spec 8.2.1 uniform quantizer: code = round((v - min) * maxCode / (max - min))."""
    max_code = (1 << bits) - 1 if bits < 32 else 0xFFFFFFFF
    if maximum == minimum:
        return np.full(len(codes), minimum, dtype=np.float64)
    return minimum + codes.astype(np.float64) * ((maximum - minimum) / float(max_code))


def combine_float_bits(
    exponents: npt.NDArray[np.int64],
    mantissae: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """Reassemble IEEE-754 float32 values from sign+exponent (9 bits) and mantissa (23 bits)."""
    pattern = ((exponents.astype(np.uint32) & np.uint32(0x1FF)) << np.uint32(23)) | (
        mantissae.astype(np.uint32) & np.uint32(0x7FFFFF)
    )
    return pattern.view(np.float32).astype(np.float64)


_DEERING_PSI_MAX = 0.615479709

# Decode-side sextant coordinate permutations (Appendix C 4.2 convertCodeToVec):
# output (x, y, z) drawn from the base vector (xx, yy, zz) by index.
_DEERING_PERMUTATIONS = np.array(
    [
        [0, 1, 2],  # sextant 0: no-op
        [2, 1, 0],  # sextant 1: mirror about x=z plane
        [1, 2, 0],  # sextant 2: rotate clockwise
        [1, 0, 2],  # sextant 3: mirror about x=y plane
        [2, 0, 1],  # sextant 4: rotate counter-clockwise
        [0, 2, 1],  # sextant 5: mirror about y=z plane
    ],
    dtype=np.int64,
)


def decode_deering_normals(
    sextants: npt.NDArray[np.int64],
    octants: npt.NDArray[np.int64],
    thetas: npt.NDArray[np.int64],
    psis: npt.NDArray[np.int64],
    bits: int,
) -> npt.NDArray[np.float64]:
    """Decode Deering-coded normals (spec 8.2.4 / Appendix C 4) into unit vectors."""
    bit_range = float(1 << bits)
    theta = thetas.astype(np.float64) + (sextants & 1)
    f_theta = np.arcsin(np.tan(_DEERING_PSI_MAX * (bit_range - theta) / bit_range))
    f_psi = _DEERING_PSI_MAX * psis.astype(np.float64) / bit_range
    cos_psi = np.cos(f_psi)
    base = np.stack(
        [np.cos(f_theta) * cos_psi, np.sin(f_psi), np.sin(f_theta) * cos_psi],
        axis=1,
    )
    out: npt.NDArray[np.float64] = np.take_along_axis(base, _DEERING_PERMUTATIONS[sextants], axis=1)
    out[:, 0] = np.where(octants & 4, out[:, 0], -out[:, 0])
    out[:, 1] = np.where(octants & 2, out[:, 1], -out[:, 1])
    out[:, 2] = np.where(octants & 1, out[:, 2], -out[:, 2])
    return out


_HASH_GOLDEN_RATIO = 0x9E3779B9
_U32 = 0xFFFFFFFF


def _mix(a: int, b: int, c: int) -> tuple[int, int, int]:
    a = (a - b - c) & _U32
    a ^= c >> 13
    b = (b - c - a) & _U32
    b ^= (a << 8) & _U32
    c = (c - a - b) & _U32
    c ^= b >> 13
    a = (a - b - c) & _U32
    a ^= c >> 12
    b = (b - c - a) & _U32
    b ^= (a << 16) & _U32
    c = (c - a - b) & _U32
    c ^= b >> 5
    a = (a - b - c) & _U32
    a ^= c >> 3
    b = (b - c - a) & _U32
    b ^= (a << 10) & _U32
    c = (c - a - b) & _U32
    c ^= b >> 15
    return a, b, c


def jt_hash32(words: npt.NDArray[np.uint32], seed: int = 0) -> int:
    """Bob Jenkins lookup2 hash2 over 32-bit words (spec Appendix D hash32)."""
    values: list[int] = np.ascontiguousarray(words, dtype=np.uint32).tolist()
    length = len(values)
    a = b = _HASH_GOLDEN_RATIO
    c = seed & _U32
    position = 0
    remaining = length
    while remaining >= 3:
        a = (a + values[position]) & _U32
        b = (b + values[position + 1]) & _U32
        c = (c + values[position + 2]) & _U32
        a, b, c = _mix(a, b, c)
        position += 3
        remaining -= 3
    c = (c + length) & _U32
    if remaining == 2:
        b = (b + values[position + 1]) & _U32
    if remaining >= 1:
        a = (a + values[position]) & _U32
    _, _, c = _mix(a, b, c)
    return c


def jt_hash16(halfwords: npt.NDArray[np.uint16], seed: int = 0) -> int:
    """Bob Jenkins lookup2 hash3 over 16-bit values (spec Appendix D jthash16)."""
    values: list[int] = np.ascontiguousarray(halfwords, dtype=np.uint16).tolist()
    length = len(values)
    a = b = _HASH_GOLDEN_RATIO
    c = seed & _U32
    position = 0
    remaining = length
    while remaining >= 6:
        a = (a + values[position] + (values[position + 1] << 16)) & _U32
        b = (b + values[position + 2] + (values[position + 3] << 16)) & _U32
        c = (c + values[position + 4] + (values[position + 5] << 16)) & _U32
        a, b, c = _mix(a, b, c)
        position += 6
        remaining -= 6
    c = (c + length) & _U32
    if remaining == 5:
        c = (c + (values[position + 4] << 16)) & _U32
    if remaining >= 4:
        b = (b + (values[position + 3] << 16)) & _U32
    if remaining >= 3:
        b = (b + values[position + 2]) & _U32
    if remaining >= 2:
        a = (a + (values[position + 1] << 16)) & _U32
    if remaining >= 1:
        a = (a + values[position]) & _U32
    _, _, c = _mix(a, b, c)
    return c


def hash_float32_scalars(values: npt.NDArray[np.float32], seed: int = 0) -> int:
    """Hash float32 values one word at a time, as the vertex-array hash pseudocode does."""
    words: list[int] = np.ascontiguousarray(values, dtype="<f4").view(np.uint32).tolist()
    c = seed & _U32
    for word in words:
        a = (_HASH_GOLDEN_RATIO + word) & _U32
        c = (c + 1) & _U32
        _, _, c = _mix(a, _HASH_GOLDEN_RATIO, c)
    return c
