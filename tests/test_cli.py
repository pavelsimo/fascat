from pathlib import Path
from typing import NamedTuple

from typer.testing import CliRunner

from fascat import __version__
from fascat.cli import app, run

runner = CliRunner()


class RunResult(NamedTuple):
    exit_code: int
    stdout: str
    stderr: str


def invoke_run(args: list[str], capsys) -> RunResult:  # type: ignore[no-untyped-def]
    exit_code = 0
    try:
        run(args)
    except SystemExit as exc:
        exit_code = int(exc.code or 0)
    captured = capsys.readouterr()
    return RunResult(exit_code=exit_code, stdout=captured.out, stderr=captured.err)


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_version_subcommand() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fascat" in result.output
    assert "Examples:" in result.output
    assert "https://pavelsimo.github.io/fascat" in result.output


def test_dry_run_flag() -> None:
    result = runner.invoke(app, ["--dry-run", "--help"])
    assert result.exit_code == 0


def test_no_color_flag() -> None:
    result = runner.invoke(app, ["--no-color", "--help"])
    assert result.exit_code == 0


def test_inspect_help() -> None:
    result = runner.invoke(app, ["inspect", "--help"])
    assert result.exit_code == 0
    assert "STEP" in result.output


def test_convert_help() -> None:
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--target-triangles" in result.output


def test_validate_help() -> None:
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "USD" in result.output


def test_convert_dry_run_json() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "output.usdc"])
    assert result.exit_code == 0
    assert '"command": "convert"' in result.output
    assert '"dry_run": true' in result.output


def test_inspect_dry_run() -> None:
    result = runner.invoke(app, ["--dry-run", "inspect", "input.step"])
    assert result.exit_code == 0
    assert "Would inspect input.step" in result.output


def test_validate_dry_run() -> None:
    result = runner.invoke(app, ["--dry-run", "validate", "output.usdc"])
    assert result.exit_code == 0
    assert "Would validate output.usdc" in result.output


def test_validate_missing_file_fails() -> None:
    result = runner.invoke(app, ["validate", "missing.usdc"])
    assert result.exit_code == 1
    assert "Missing output file" in result.output


def test_inspect_fixture_reports_stats() -> None:
    result = runner.invoke(app, ["inspect", "tests/fixtures/spool-clamp-lid.step"])
    assert result.exit_code == 0
    assert "1 parts" in result.output
    assert "units=millimetre" in result.output


def test_convert_fixture_writes_usd_and_report(tmp_path: Path) -> None:
    output_file = tmp_path / "output.usda"
    report_file = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "convert",
            "tests/fixtures/spool-clamp-lid.step",
            str(output_file),
            "--sag",
            "0.2",
            "--angle",
            "20",
            "--target-triangles",
            "120",
            "--lods",
            "0.5",
            "--report",
            str(report_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()
    assert report_file.exists()
    assert "Converted" in result.output
    assert '"steps"' in report_file.read_text(encoding="utf-8")


def test_convert_existing_output_requires_force(tmp_path: Path) -> None:
    step_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")
    output_file.write_text("#usdc", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(step_file), str(output_file)])
    assert result.exit_code == 1
    assert "Pass --force" in result.output


def test_validate_rejects_unknown_extension(tmp_path: Path) -> None:
    output_file = tmp_path / "output.txt"
    output_file.write_text("not usd", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(output_file)])
    assert result.exit_code == 2
    assert "Unsupported USD extension" in result.output


def test_validate_generated_usd(tmp_path: Path) -> None:
    output_file = tmp_path / "output.usda"
    convert_result = runner.invoke(
        app,
        [
            "convert",
            "tests/fixtures/spool-clamp-lid.step",
            str(output_file),
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
        ],
    )
    assert convert_result.exit_code == 0

    result = runner.invoke(app, ["validate", str(output_file)])
    assert result.exit_code == 0
    assert "valid USD" in result.output


def test_convert_rejects_invalid_lods() -> None:
    result = runner.invoke(app, ["--dry-run", "convert", "input.step", "output.usdc", "--lods", "1.5"])
    assert result.exit_code == 2
    assert "--lods ratios" in result.output


def test_convert_rejects_invalid_lods_as_json(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--json", "--dry-run", "convert", "input.step", "output.usdc", "--lods", "1.5"], capsys)
    assert result.exit_code == 2
    assert '"error": "--lods ratios must be greater than 0 and less than 1."' in result.stdout


def test_help_command_alias(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["help"], capsys)
    assert result.exit_code == 0
    assert "Usage: fascat" in result.stdout


def test_help_command_alias_for_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["help", "convert"], capsys)
    assert result.exit_code == 0
    assert "Usage: fascat convert" in result.stdout
    assert "--target-triangles" in result.stdout


def test_help_wins_with_invalid_tokens(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "input.step", "output.usdc", "--bad", "-h"], capsys)
    assert result.exit_code == 0
    assert "Usage: fascat convert" in result.stdout
    assert "No such option" not in result.stderr


def test_version_wins_after_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "input.step", "output.usdc", "--version"], capsys)
    assert result.exit_code == 0
    assert f"fascat {__version__}" in result.stdout
    assert result.stderr == ""


def test_global_flags_work_after_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "input.step", "output.usdc", "--json", "--dry-run"], capsys)
    assert result.exit_code == 0
    assert '"command": "convert"' in result.stdout
    assert '"dry_run": true' in result.stdout


def test_unknown_command_suggests_once(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["conv"], capsys)
    assert result.exit_code == 2
    assert result.stderr.count("Did you mean 'convert'?") == 1


def test_dash_input_and_output_are_accepted_for_dry_run(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "-", "-", "--json"], capsys)
    assert result.exit_code == 0
    assert '"input": "-"' in result.stdout
    assert '"output": "-"' in result.stdout


def test_json_error_payload_for_missing_file(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--json", "validate", "missing.usdc"], capsys)
    assert result.exit_code == 1
    payload = result.stdout
    assert '"command": "validate"' in payload
    assert '"error": "Missing output file: missing.usdc"' in payload


def test_convert_rejects_bad_input_suffix(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.txt", "output.usdc"], capsys)
    assert result.exit_code == 2
    assert "Unsupported STEP extension" in result.stderr


def test_convert_rejects_bad_output_suffix(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.txt"], capsys)
    assert result.exit_code == 2
    assert "Unsupported USD extension" in result.stderr


def test_convert_rejects_zero_ratio(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--ratio", "0"], capsys)
    assert result.exit_code == 2
    assert "--ratio must be greater than 0" in result.stderr


def test_no_color_help_has_no_ansi(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--no-color", "--help"], capsys)
    assert result.exit_code == 0
    assert "\x1b[" not in result.stdout


def test_no_color_env_has_no_ansi(capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("NO_COLOR", "1")
    result = invoke_run(["--help"], capsys)
    assert result.exit_code == 0
    assert "\x1b[" not in result.stdout


def test_dumb_term_has_no_ansi(capsys, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("TERM", "dumb")
    result = invoke_run(["--help"], capsys)
    assert result.exit_code == 0
    assert "\x1b[" not in result.stdout
