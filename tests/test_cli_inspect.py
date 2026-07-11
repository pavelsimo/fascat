import json
from pathlib import Path

import pytest

from fascat.cli import app

from ._cli_test_helpers import (
    block_imports,
    runner,
)


def test_inspect_help() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "STEP" in result.output


def test_inspect_dry_run() -> None:
    result = runner.invoke(app, ["--dry-run", "inspect", "input.step"])
    assert result.exit_code == 0
    assert "Would inspect input.step" in result.output


@pytest.mark.requires_ocp
def test_inspect_fixture_reports_stats() -> None:
    result = runner.invoke(app, ["inspect", "tests/fixtures/spool-clamp-lid.step"])
    assert result.exit_code == 0
    assert "1 parts" in result.output
    assert "units=millimetre" in result.output

    json_result = runner.invoke(app, ["--json", "inspect", "tests/fixtures/spool-clamp-lid.step"])
    assert json_result.exit_code == 0
    payload = json.loads(json_result.output)
    assert payload["options"]["name"] == "inspect-only"
    assert payload["root"]["id"]
    assert "children" in payload["root"]
    assert payload["root"]["transform"][3] == [0.0, 0.0, 0.0, 1.0]
    assert payload["parts"][0]["has_source_shape"] is True
    assert len(payload["materials"]) == 1
    assert len(payload["materials"][0]["base_color"]) == 4
    assert payload["report"]["input_stats"]["parts"] == 1
    assert payload["report"]["steps"][0]["name"] == "import"
    assert payload["report"]["steps"][0]["after"]["parts"] == 1


def test_inspect_missing_step_backend_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block_imports(monkeypatch, "OCP")
    step_file = tmp_path / "input.step"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")

    result = runner.invoke(app, ["inspect", str(step_file)])

    assert result.exit_code == 1
    assert "STEP import requires cadquery-ocp" in result.output


@pytest.mark.requires_ocp
def test_inspect_reads_step_from_stdin() -> None:
    step_data = Path("tests/fixtures/spool-clamp-lid.step").read_text(encoding="utf-8")

    result = runner.invoke(app, ["--json", "inspect", "-"], input=step_data)

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["input"] == "-"
    assert payload["stats"]["parts"] == 1
    assert payload["report"]["source_path"] is None
