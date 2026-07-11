from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fascat.io.step import pmi as step_pmi
from fascat.io.step import records as step_records
from fascat.io.step import single as step_single
from fascat.io.step.textures import (
    _extract_source_textures,
)
from fascat.options import StepReadOptions


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain", "plain"),
        (r"a\\b", "a\\b"),
        (r"caf\S\i", "café"),
        (r"\PG\\S\d", "δ"),
        (r"\X\C4", "Ä"),
        (r"\X2\00E9\X0\ acute", "é acute"),
        ("\\X2\\004100E9\\X0\\", "Aé"),
        ("\\X4\\0001F600\\X0\\", "😀"),
        (r"\PE\\S\T\S\5", "дЕ"),
    ],
)
def test_decode_step_string_directives(raw: str, expected: str) -> None:
    assert step_records._decode_step_string(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "\\X2\\12\\X0\\",
        r"\X2\00E9",
        r"\X\ZZ",
        "trailing\\",
        r"bare \S",
        r"\PZ\page",
        "\\X4\\FFFFFFFF\\X0\\",
    ],
)
def test_decode_step_string_keeps_malformed_sequences_literal(raw: str) -> None:
    assert step_records._decode_step_string(raw) == raw


def test_step_string_values_decodes_iso_escapes() -> None:
    assert step_records._step_string_values(r"('NAME \X2\00E9\X0\')") == ["NAME é"]


def test_find_step_record_args_end_bounds_unterminated_string() -> None:
    text = "('unterminated " + "x" * 2_000_000

    assert step_records._find_step_record_args_end(text, 0) is None


def test_find_step_record_args_end_respects_explicit_bound() -> None:
    text = "('payload value')"

    assert step_records._find_step_record_args_end(text, 0, max_scan=8) is None
    assert step_records._find_step_record_args_end(text, 0) == len(text) - 1


def test_iter_step_records_recovers_after_unterminated_record() -> None:
    text = "#1=BROKEN_ENTITY('unterminated\n#2=DIMENSIONAL_SIZE(#9,'diameter 5.0');\n"

    records = step_records._iter_step_records(text)

    assert any(record.entity == "DIMENSIONAL_SIZE" for record in records)


def test_oversized_step_skips_auxiliary_scans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_records, "_MAX_STEP_SCAN_BYTES", 16)
    texture = tmp_path / "panel.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(texture)
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=EXTERNAL_REFERENCE('panel.png');\n"
        "#2=DIMENSIONAL_SIZE(#9,'diameter 5.0');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    annotations = step_pmi._extract_step_pmi_annotations(source, StepReadOptions(pmi=True))
    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())
    warnings = step_single._import_warnings(
        StepReadOptions(),
        step_records._step_header_info(source),
        0,
        scan_capped=step_records._step_scan_capped(source),
    )

    assert annotations == []
    assert extraction.summary["references"] == 0
    assert any("auxiliary STEP text scans skipped" in warning for warning in warnings)


def test_step_string_values_preserve_decoded_non_breaking_space() -> None:
    values = step_records._step_string_values("('a\\S\\ b')")

    assert values == ["a\xa0b"]
