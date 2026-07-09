from __future__ import annotations

import numpy as np
import pytest

from fascat.io.jt import codec
from fascat.io.jt.container import ByteReader
from tests._jt_builder import encode_cdp2

_PREDICTORS = ["null", "lag1", "lag2", "stride1", "stride2", "stripindex", "ramp", "xor1", "xor2"]


def _decode(data: bytes, predictor: str = "null", byte_order: str = "<") -> np.ndarray:
    return codec.decode_int32_cdp2(ByteReader(data, byte_order=byte_order), predictor)


class TestNullCodec:
    def test_round_trips_raw_values(self) -> None:
        values = [0, 1, -1, 123456, -123456, 2**31 - 1, -(2**31)]
        assert _decode(encode_cdp2(values)).tolist() == values

    def test_empty_packet(self) -> None:
        assert _decode(encode_cdp2([])).tolist() == []

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    def test_byte_orders(self, byte_order: str) -> None:
        values = [7, -9, 1000]
        data = encode_cdp2(values, byte_order=byte_order)
        assert _decode(data, byte_order=byte_order).tolist() == values


class TestPredictors:
    @pytest.mark.parametrize("predictor", _PREDICTORS)
    def test_round_trip_through_null_codec(self, predictor: str) -> None:
        values = [10, 12, 14, 16, 18, 20, 25, 33, 32, 31, 5, -6, 90, 91, 92, 93]
        data = encode_cdp2(values, predictor=predictor)
        assert _decode(data, predictor).tolist() == values

    @pytest.mark.parametrize("predictor", _PREDICTORS)
    def test_short_arrays_are_primers_only(self, predictor: str) -> None:
        values = [5, -3, 8]
        assert codec.unpack_residuals(np.array(values, dtype=np.int64), predictor).tolist() == values

    def test_lag1_matches_reference_loop(self) -> None:
        residuals = np.array([3, 1, 4, 1, 5, 9, 2, 6], dtype=np.int64)
        expected = residuals.copy()
        for i in range(4, len(expected)):
            expected[i] = residuals[i] + expected[i - 1]
        assert codec.unpack_residuals(residuals, "lag1").tolist() == expected.tolist()

    def test_stride1_matches_reference_loop(self) -> None:
        residuals = np.array([3, 1, 4, 1, 5, 9, 2, 6, -7, 0], dtype=np.int64)
        expected = residuals.copy()
        for i in range(4, len(expected)):
            predicted = expected[i - 1] + (expected[i - 1] - expected[i - 2])
            expected[i] = residuals[i] + predicted
        assert codec.unpack_residuals(residuals, "stride1").tolist() == expected.tolist()

    def test_stride2_matches_reference_loop(self) -> None:
        residuals = np.array([3, 1, 4, 1, 5, 9, 2, 6, -7, 0, 11], dtype=np.int64)
        expected = residuals.copy()
        for i in range(4, len(expected)):
            predicted = expected[i - 2] + (expected[i - 2] - expected[i - 4])
            expected[i] = residuals[i] + predicted
        assert codec.unpack_residuals(residuals, "stride2").tolist() == expected.tolist()

    def test_xor1_matches_reference_loop(self) -> None:
        residuals = np.array([3, 1, 4, 1, 5, 9, -2, 6], dtype=np.int64)
        expected = residuals.copy()
        for i in range(4, len(expected)):
            expected[i] = np.int64(
                np.int32(np.uint32(residuals[i] & 0xFFFFFFFF) ^ np.uint32(expected[i - 1] & 0xFFFFFFFF))
            )
        assert codec.unpack_residuals(residuals, "xor1").tolist() == expected.tolist()

    def test_unknown_predictor_raises(self) -> None:
        with pytest.raises(RuntimeError, match="unsupported JT predictor"):
            codec.unpack_residuals(np.zeros(8, dtype=np.int64), "bogus")


class TestBitlengthCodec:
    def test_hand_derived_golden_vector(self) -> None:
        # Values [3, -2]. Width starts at 0, so 3 forces two increments
        # (prefix 1 11 0), then 3 as 4-bit two's complement (0011); -2 fits the
        # current 4-bit width (prefix 0, value 1110). Bit stream:
        # 1 1 1 0 0 0 1 1 0 1 1 1 0 -> 13 bits, one word 0xE3700000.
        golden = (
            b"\x02\x00\x00\x00"  # value count
            b"\x01"  # Bitlength CODEC
            b"\x0d\x00\x00\x00"  # code text length in bits
            b"\x01\x00\x00\x00"  # code text word count
            b"\x00\x00\x70\xe3"  # 0xE3700000 little-endian
        )
        assert _decode(golden).tolist() == [3, -2]

    def test_round_trips_mixed_widths(self) -> None:
        values = [0, 0, 1, -1, 3, 700, -700, 2, 0, 65000, -65000, 12, 5, 0]
        data = encode_cdp2(values, codec="bitlength")
        assert _decode(data).tolist() == values

    def test_round_trips_beyond_one_word(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.integers(-(2**20), 2**20, size=300).tolist()
        data = encode_cdp2(values, codec="bitlength")
        assert _decode(data).tolist() == values

    @pytest.mark.parametrize("predictor", ["lag1", "stride1", "stripindex"])
    def test_with_predictors(self, predictor: str) -> None:
        values = list(range(0, 60, 3)) + [1000, 4, 1000, 4]
        data = encode_cdp2(values, codec="bitlength", predictor=predictor)
        assert _decode(data, predictor).tolist() == values


class TestArithmeticCodec:
    def test_single_symbol_context(self) -> None:
        values = [42] * 50
        data = encode_cdp2(values, codec="arithmetic")
        assert _decode(data).tolist() == values

    def test_skewed_distribution_forces_renormalization(self) -> None:
        values = [0] * 400 + [1] * 30 + [0] * 400 + [2] * 3 + [0] * 100
        data = encode_cdp2(values, codec="arithmetic")
        assert _decode(data).tolist() == values

    def test_escape_symbols_pull_from_out_of_band(self) -> None:
        values = [5, 5, 5, 99999, 5, 5, -12345, 5, 5, 99999, 5]
        data = encode_cdp2(values, codec="arithmetic", oob_values={99999, -12345})
        assert _decode(data).tolist() == values

    def test_with_lag1_predictor(self) -> None:
        values = [100 + i for i in range(200)]  # constant residual of 1 after primers
        data = encode_cdp2(values, codec="arithmetic", predictor="lag1")
        assert _decode(data, "lag1").tolist() == values

    def test_random_small_alphabet(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.choice([3, 7, -2, 900], size=500, p=[0.7, 0.2, 0.05, 0.05]).tolist()
        data = encode_cdp2(values, codec="arithmetic")
        assert _decode(data).tolist() == values


class TestChopperCodec:
    def test_zero_chop_bits_decodes_zeros(self) -> None:
        data = b"\x05\x00\x00\x00" + b"\x04" + b"\x00"  # count 5, chopper, chop bits 0
        assert _decode(data).tolist() == [0, 0, 0, 0, 0]

    def test_chopped_msb_lsb_reassembly(self) -> None:
        # values = (msb << (span - chop)) | lsb, then + bias.
        # span=8, chop=3 -> lsb holds 5 bits.
        msb = [1, 2, 3, 4, 5]
        lsb = [0, 31, 7, 1, 16]
        bias = -100
        expected = [(m << 5 | v) + bias for m, v in zip(msb, lsb, strict=True)]
        data = (
            b"\x05\x00\x00\x00"  # count
            b"\x04"  # chopper
            b"\x03"  # chop bits
            + (-100 & 0xFFFFFFFF).to_bytes(4, "little")  # value bias
            + b"\x08"  # value span bits
            + encode_cdp2(msb)
            + encode_cdp2(lsb)
        )
        assert _decode(data).tolist() == expected


class TestCdpErrors:
    def test_unsupported_codec_type(self) -> None:
        with pytest.raises(RuntimeError, match="unsupported JT CODEC type"):
            _decode(b"\x01\x00\x00\x00" + b"\x07")

    def test_negative_count(self) -> None:
        with pytest.raises(RuntimeError, match="negative CDP value count"):
            _decode(b"\xff\xff\xff\xff")

    def test_truncated_packet(self) -> None:
        data = encode_cdp2([1, 2, 3])
        with pytest.raises(RuntimeError, match="truncated JT data"):
            _decode(data[:-2])


class TestDequantizeUniform:
    def test_endpoints_are_exact(self) -> None:
        codes = np.array([0, (1 << 10) - 1], dtype=np.int64)
        out = codec.dequantize_uniform(codes, -4.0, 12.0, 10)
        assert out.tolist() == [-4.0, 12.0]

    def test_inverts_spec_encoder_within_half_step(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.uniform(-5.0, 5.0, size=1000)
        bits = 12
        max_code = (1 << bits) - 1
        multiplier = max_code / 10.0
        codes = np.clip((values + 5.0) * multiplier + 0.5, 0, max_code).astype(np.int64)
        decoded = codec.dequantize_uniform(codes, -5.0, 5.0, bits)
        assert np.allclose(decoded, values, atol=10.0 / max_code)

    def test_degenerate_range(self) -> None:
        out = codec.dequantize_uniform(np.array([0, 5], dtype=np.int64), 2.5, 2.5, 8)
        assert out.tolist() == [2.5, 2.5]


class TestCombineFloatBits:
    def test_reassembles_exact_float32(self) -> None:
        values = np.array([0.0, 1.0, -1.5, 3.14159, -1e-20, 1e20], dtype=np.float32)
        bits = values.view(np.uint32)
        exponents = (bits >> 23).astype(np.int64)
        mantissae = (bits & 0x7FFFFF).astype(np.int64)
        out = codec.combine_float_bits(exponents, mantissae)
        assert np.array_equal(out, values.astype(np.float64))


class TestDeeringNormals:
    def test_all_outputs_are_unit_vectors(self) -> None:
        rng = np.random.default_rng(5)
        n = 500
        bits = 8
        sextants = rng.integers(0, 6, n)
        octants = rng.integers(0, 8, n)
        thetas = rng.integers(0, (1 << bits) - 1, n)
        psis = rng.integers(0, 1 << bits, n)
        normals = codec.decode_deering_normals(sextants, octants, thetas, psis, bits)
        assert np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)

    def test_sextant_zero_octant_seven_reference_case(self) -> None:
        # Direct transcription of the Appendix C formulas for a single code.
        bits = 6
        theta_code, psi_code = 20, 9
        bit_range = float(1 << bits)
        f_theta = np.arcsin(np.tan(0.615479709 * (bit_range - theta_code) / bit_range))
        f_psi = 0.615479709 * psi_code / bit_range
        expected = [
            np.cos(f_theta) * np.cos(f_psi),
            np.sin(f_psi),
            np.sin(f_theta) * np.cos(f_psi),
        ]
        out = codec.decode_deering_normals(
            np.array([0]), np.array([7]), np.array([theta_code]), np.array([psi_code]), bits
        )
        assert np.allclose(out[0], expected)

    def test_octant_bits_flip_signs(self) -> None:
        args = (np.array([0]), np.array([20]), np.array([9]))
        base = codec.decode_deering_normals(args[0], np.array([7]), args[1], args[2], 6)
        flipped = codec.decode_deering_normals(args[0], np.array([0]), args[1], args[2], 6)
        assert np.allclose(flipped, -base)

    def test_odd_sextants_increment_theta(self) -> None:
        # Sextant 1 mirrors x/z and shifts theta by one code.
        bits = 6
        s0 = codec.decode_deering_normals(np.array([0]), np.array([7]), np.array([21]), np.array([9]), bits)
        s1 = codec.decode_deering_normals(np.array([1]), np.array([7]), np.array([20]), np.array([9]), bits)
        assert np.allclose(s1[0], s0[0, ::-1])


def _reference_lookup2(words: list[int], seed: int) -> int:
    """Independent transcription of Bob Jenkins' lookup2 hash2 for cross-checking."""

    def mix(a: int, b: int, c: int) -> tuple[int, int, int]:
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

    a = b = 0x9E3779B9
    c = seed
    i, length = 0, len(words)
    while length - i >= 3:
        a = (a + words[i]) & 0xFFFFFFFF
        b = (b + words[i + 1]) & 0xFFFFFFFF
        c = (c + words[i + 2]) & 0xFFFFFFFF
        a, b, c = mix(a, b, c)
        i += 3
    c = (c + len(words)) & 0xFFFFFFFF
    if length - i == 2:
        b = (b + words[i + 1]) & 0xFFFFFFFF
    if length - i >= 1:
        a = (a + words[i]) & 0xFFFFFFFF
    return mix(a, b, c)[2]


class TestJenkinsHash:
    @pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 11, 100])
    def test_hash32_matches_independent_reference(self, size: int) -> None:
        rng = np.random.default_rng(size)
        words = rng.integers(0, 2**32, size=size, dtype=np.uint32)
        assert codec.jt_hash32(words, 17) == _reference_lookup2(words.tolist(), 17)

    def test_hash32_chains_seed(self) -> None:
        words = np.arange(10, dtype=np.uint32)
        once = codec.jt_hash32(words, 0)
        chained = codec.jt_hash32(words[5:], codec.jt_hash32(words[:5], 0))
        assert once != chained  # chaining is order/segment sensitive

    def test_hash_float32_scalars_matches_per_word_hashing(self) -> None:
        values = np.array([1.5, -2.25, 3e7], dtype=np.float32)
        expected = 0
        for word in values.view(np.uint32).tolist():
            expected = _reference_lookup2([word], expected)
        assert codec.hash_float32_scalars(values, 0) == expected
