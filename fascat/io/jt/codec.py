"""JT 9.5 data codecs: Int32 CDP Mk.2, predictors, quantization, Deering normals, hashes.

Algorithms are transcribed from the public Siemens *JT File Format Reference*
v9.5 (sections 8.1-8.2 and Appendices C/D). JT 9.5 shape data uses the Mk.2
Int32 Compressed Data Packet exclusively; the JT 10 packet is a third,
incompatible generation and is rejected upstream with a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fascat.io.jt.container import BitReader, ByteReader

CODEC_NULL = 0
CODEC_BITLENGTH = 1
CODEC_ARITHMETIC = 3
CODEC_CHOPPER = 4

_ESCAPE_SYMBOL = -2
_MAX_CDP_RECURSION = 3
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
            return np.zeros(count, dtype=np.int64)
        bias = reader.i32()
        span_bits = reader.u8()
        msb = _decode_cdp2_residuals(reader, depth + 1)
        lsb = _decode_cdp2_residuals(reader, depth + 1)
        if len(msb) != count or len(lsb) != count:
            raise RuntimeError("corrupt JT data: chopper field length mismatch")
        return _wrap_int32((msb << (span_bits - chop_bits) | lsb) + bias)
    if codec not in (CODEC_NULL, CODEC_BITLENGTH, CODEC_ARITHMETIC):
        raise RuntimeError(f"unsupported JT CODEC type: {codec}")
    reader.i32()  # code-text length in bits
    words = reader.vec_u32()
    if codec == CODEC_NULL:
        if len(words) != count:
            raise RuntimeError("corrupt JT data: Null CODEC word count mismatch")
        return words.astype(np.int32).astype(np.int64)
    if codec == CODEC_BITLENGTH:
        return _decode_bitlength(BitReader(words), count)
    context = _read_probability_context(reader)
    out_of_band = _decode_cdp2_residuals(reader, depth + 1)
    return _decode_arithmetic(BitReader(words), count, context, out_of_band)


def _decode_bitlength(bits: BitReader, count: int) -> npt.NDArray[np.int64]:
    """Adaptive-width bit field decoding (spec 8.2.2 / Appendix C 2.1); step size 2."""
    out = np.empty(count, dtype=np.int64)
    width = 0
    for index in range(count):
        if bits.read_bit() == 1:
            direction = bits.read_bit()
            step = 2 if direction == 1 else -2
            width += step
            while bits.read_bit() == direction:
                width += step
            if width < 0:
                raise RuntimeError("corrupt JT data: negative bit field width")
        out[index] = bits.read_signed(width)
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
