from __future__ import annotations

import numpy as np
import pytest

from fascat.io.jt import codec
from fascat.io.jt.container import ByteReader
from tests._jt_builder import encode_cdp2, encode_cdp3

_PREDICTORS = ["null", "lag1", "lag2", "stride1", "stride2", "stripindex", "ramp", "xor1", "xor2"]


def _decode(data: bytes, predictor: str = "null", byte_order: str = "<") -> np.ndarray:
    return codec.decode_int32_cdp2(ByteReader(data, byte_order=byte_order), predictor)


class TestNullCodec:
    def test_round_trips_raw_values(self) -> None:
        values = [0, 1, -1, 123456, -123456, 2**31 - 1, -(2**31)]
        assert _decode(encode_cdp2(values)).tolist() == values

    def test_empty_packet(self) -> None:
        assert _decode(encode_cdp2([])).tolist() == []

    def test_length_field_counts_bytes(self) -> None:
        # The Null CODEC's code-text length field is a byte count, unlike the
        # bit counts used by the entropy codecs.
        data = encode_cdp2([1, 2, 3])
        assert int.from_bytes(data[5:9], "little") == 12

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


_BITLENGTH_MODES = ["bitlength", "bitlength-fixed", "bitlength-variable"]


class TestBitlengthCodec:
    def test_hand_derived_fixed_mode_golden_vector(self) -> None:
        # Values [3, -2], fixed-width mode. Mode bit 0; min = -2 (signed bitsize
        # 2), max = 3 (signed bitsize 3); header 0 000010 000011 10 011; range 5
        # -> 3-bit unsigned offsets 101 and 000. 24 bits total:
        # 00000100 00011100 11101000 -> word 0x041CE800.
        golden = (
            b"\x02\x00\x00\x00"  # value count
            b"\x01"  # Bitlength CODEC
            b"\x18\x00\x00\x00"  # code text length in bits (24)
            b"\x00\xe8\x1c\x04"  # 0x041CE800 little-endian
        )
        assert _decode(golden).tolist() == [3, -2]

    def test_hand_derived_variable_mode_golden_vector(self) -> None:
        # Values [7, 7, 6], variable-width mode. Mode bit 1; mean 7 (32 bits);
        # delta field size 3 (011), run field size 2 (010); width delta +1
        # (001), run 3 (11), then 1-bit signed fields 0, 0, -1 (0 0 1).
        # 47 bits total -> words 0x80000003, 0xB4720000.
        golden = (
            b"\x03\x00\x00\x00"  # value count
            b"\x01"  # Bitlength CODEC
            b"\x2f\x00\x00\x00"  # code text length in bits (47)
            b"\x03\x00\x00\x80"  # 0x80000003 little-endian
            b"\x00\x00\x72\xb4"  # 0xB4720000 little-endian
        )
        assert _decode(golden).tolist() == [7, 7, 6]

    def test_constant_array_is_count_independent(self) -> None:
        # A fixed-mode packet with min == max carries no per-value fields, so
        # the same code text serves any value count.
        for count in (4, 2900):
            data = count.to_bytes(4, "little") + b"\x01\x13\x00\x00\x00\x00\x60\x1b\x06"
            assert _decode(data).tolist() == [3] * count

    @pytest.mark.parametrize("mode", _BITLENGTH_MODES)
    def test_round_trips_mixed_widths(self, mode: str) -> None:
        values = [0, 0, 1, -1, 3, 700, -700, 2, 0, 65000, -65000, 12, 5, 0]
        data = encode_cdp2(values, codec=mode)
        assert _decode(data).tolist() == values

    @pytest.mark.parametrize("mode", _BITLENGTH_MODES)
    def test_round_trips_beyond_one_word(self, mode: str) -> None:
        rng = np.random.default_rng(7)
        values = rng.integers(-(2**20), 2**20, size=300).tolist()
        data = encode_cdp2(values, codec=mode)
        assert _decode(data).tolist() == values

    @pytest.mark.parametrize("predictor", ["lag1", "stride1", "stripindex"])
    def test_with_predictors(self, predictor: str) -> None:
        values = list(range(0, 60, 3)) + [1000, 4, 1000, 4]
        data = encode_cdp2(values, codec="bitlength", predictor=predictor)
        assert _decode(data, predictor).tolist() == values

    def test_unconsumed_code_text_raises(self) -> None:
        # The [3, -2] fixed-mode golden consumes 24 bits; claiming 30 must fail.
        data = b"\x02\x00\x00\x00\x01\x1e\x00\x00\x00\x00\xe8\x1c\x04"
        with pytest.raises(RuntimeError, match="consumed 24 of 30 bits"):
            _decode(data)

    def test_run_overrunning_value_count_raises(self) -> None:
        # The variable-mode golden's run of 3 against a claimed count of 2.
        data = b"\x02\x00\x00\x00\x01\x2f\x00\x00\x00\x03\x00\x00\x80\x00\x00\x72\xb4"
        with pytest.raises(RuntimeError, match="run overruns value count"):
            _decode(data)


class TestBitlengthRealFileGoldens:
    """Packets extracted verbatim from Siemens-written JT 9.5 files.

    These pin the decoder to real DM output independently of the synthetic
    encoder in tests/_jt_builder.py, so a mirrored spec misreading in builder
    and decoder cannot cancel out.
    """

    def test_constant_valences_packet(self) -> None:
        # 2900 triangle valences, all 3, in 19 bits of code text.
        data = bytes.fromhex("540b0000011300000000601b06")
        assert _decode(data).tolist() == [3] * 2900

    def test_constant_zero_packet(self) -> None:
        # 2900 vertex group ids, all 0, in 15 bits.
        data = bytes.fromhex("540b0000010f00000000000802")
        assert _decode(data).tolist() == [0] * 2900

    def test_short_constant_packet(self) -> None:
        # Two attribute masks of 0b111 in 21 bits.
        data = bytes.fromhex("02000000011500000000b82308")
        assert _decode(data).tolist() == [7, 7]

    def test_fixed_width_field_packet(self) -> None:
        # 19 face degrees spanning 3..14: 21-bit header plus 19 4-bit fields.
        data = bytes.fromhex("130000000161000000d5752b06541a181b25401d3000000000")
        expected = [14, 13, 13, 6, 9, 6, 3, 6, 7, 13, 11, 9, 3, 6, 13, 11, 3, 7, 13]
        assert _decode(data).tolist() == expected


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
    def test_zero_chop_bits_reads_nested_packet(self) -> None:
        # chop bits 0 means no chopping happened: the values arrive in one
        # nested packet instead of msb/lsb halves.
        values = [12, -7, 0, 400, 3]
        data = b"\x05\x00\x00\x00" + b"\x04" + b"\x00" + encode_cdp2(values)
        assert _decode(data).tolist() == values

    def test_zero_chop_bits_length_mismatch_raises(self) -> None:
        data = b"\x05\x00\x00\x00" + b"\x04" + b"\x00" + encode_cdp2([1, 2])
        with pytest.raises(RuntimeError, match="chopper field length mismatch"):
            _decode(data)

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


def _decode3(data: bytes, predictor: str = "null", byte_order: str = "<") -> np.ndarray:
    return codec.decode_int32_cdp3(ByteReader(data, byte_order=byte_order), predictor)


class TestCdp3NullCodec:
    @pytest.mark.parametrize("predictor", _PREDICTORS)
    def test_round_trips_through_predictors(self, predictor: str) -> None:
        values = [10, 12, 14, 16, 18, 20, 25, 33, 32, 31, 5, -6, 90, 91, 92, 93]
        assert _decode3(encode_cdp3(values, predictor=predictor), predictor).tolist() == values

    def test_empty_packet(self) -> None:
        assert _decode3(encode_cdp3([])).tolist() == []


class TestCdp3Chopper:
    def test_fields_always_read_and_reassembled(self) -> None:
        # Unlike Mk.2, the Mk.3 chopper stores bias and span even for
        # chop_bits == 0; this golden exercises the ordinary chopped path.
        msb = [1, 2, 3, 4, 5]
        lsb = [0, 31, 7, 1, 16]
        bias = -100
        expected = [(m << 5 | v) + bias for m, v in zip(msb, lsb, strict=True)]
        data = (
            b"\x05\x00\x00\x00"
            b"\x04"
            b"\x03"  # chop bits
            + (-100 & 0xFFFFFFFF).to_bytes(4, "little")
            + b"\x08"  # span bits
            + encode_cdp3(msb)
            + encode_cdp3(lsb)
        )
        assert _decode3(data).tolist() == expected


class TestCdp3RealFileGoldens:
    """Packets extracted verbatim from JT 10.0 files written by DM 8.1.3.3.

    They pin the Mk.3 decoder (nibbler headers, post-code-text probability
    contexts, escape-driven out-of-band data, move-to-front) to real output.
    """

    def test_bitlength_fixed_constant(self) -> None:
        # 448 triangle valences, all 3: nibbler-coded min == max, no fields.
        data = bytes.fromhex("c0010000010b0000000000c018")
        assert _decode3(data).tolist() == [3] * 448

    def test_bitlength_variable(self) -> None:
        data = bytes.fromhex("2700000001ab000000da659181c488a073039b9f8c8410ae8df9ffcd3b00008012")
        expected = [
            22,
            23,
            26,
            28,
            2,
            2,
            3,
            1,
            2,
            3,
            2,
            7,
            -1,
            0,
            0,
            0,
            -1,
            -2,
            -9,
            1,
            1,
            1,
            1,
            -5,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            2,
            -4,
        ]
        assert _decode3(data).tolist() == expected

    def test_arithmetic_with_out_of_band(self) -> None:
        # 95 face degrees; the context carries an escape entry whose two
        # occurrences pull from a nested out-of-band packet.
        data = bytes.fromhex(
            "5f00000003b200000089fa6584871bbd7a5d961c87321aadaad79c1085000000"
            "40000418180000000440b29bc53804000000012000000000842002"
        )
        values = _decode3(data).tolist()
        assert len(values) == 95
        assert values[:17] == [6, 8, 7, 7, 7, 6, 6, 7, 6, 6, 5, 4, 6, 6, 6, 6, 7]
        assert values[17] == 0 and values[24] == 0  # out-of-band escapes
        assert values[-8:] == [6, 6, 6, 6, 6, 6, 5, 5]

    def test_move_to_front(self) -> None:
        data = bytes.fromhex(
            "310000000509000000014f000000eb87ee0b042064f30000222331000000019e"
            "0000004922c07824119224840448844992240124119348"
        )
        expected = [
            68,
            123,
            68,
            68,
            68,
            68,
            68,
            68,
            68,
            68,
            68,
            68,
            127,
            68,
            68,
            68,
            68,
            68,
            55,
            68,
            68,
            68,
            34,
            1,
            1,
            1,
            9,
            1,
            36,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            18,
            36,
            18,
            18,
            18,
            18,
            9,
            18,
            18,
            18,
            18,
        ]
        assert _decode3(data).tolist() == expected


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

    def test_recursion_depth_capped(self) -> None:
        # Nine nested zero-chop choppers exceed the depth-8 cap.
        data = encode_cdp2([1])
        for _ in range(9):
            data = b"\x01\x00\x00\x00" + b"\x04" + b"\x00" + data
        with pytest.raises(RuntimeError, match="recursion depth exceeded"):
            _decode(data)

    def test_deep_but_legal_recursion_decodes(self) -> None:
        data = encode_cdp2([1])
        for _ in range(7):
            data = b"\x01\x00\x00\x00" + b"\x04" + b"\x00" + data
        assert _decode(data).tolist() == [1]


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
