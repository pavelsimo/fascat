import builtins
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from typer.testing import CliRunner

from fascat import __version__
from fascat.cli import app, run
from fascat.report import Report

runner = CliRunner()
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


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


def plain(text: str) -> str:
    return ANSI_RE.sub("", text)


def compact(text: str) -> str:
    return " ".join(plain(text).split())


def block_imports(monkeypatch: pytest.MonkeyPatch, *prefixes: str) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


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
    assert "--target-triangles" in plain(result.output)
    assert "--max-edge-length" in plain(result.output)
    assert "--materials" in plain(result.output)
    assert "--uv1" in plain(result.output)
    assert "--no-preserve-instances" in plain(result.output)


def test_validate_help() -> None:
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "USD" in result.output


def test_convert_dry_run_json() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "output.usdc"])
    assert result.exit_code == 0
    assert '"command": "convert"' in result.output
    assert '"dry_run": true' in result.output


def test_convert_dry_run_defaults_output_to_usdc() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step"])
    assert result.exit_code == 0
    assert '"output": "input.usdc"' in result.output


def test_convert_dry_run_accepts_material_staging_mode() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "--materials", "display"])
    assert result.exit_code == 0
    assert '"materials": "display"' in result.output


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


def test_convert_missing_input_file_fails_before_processing(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "missing.step", "output.usdc"], capsys)

    assert result.exit_code == 1
    assert "Missing input file: missing.step" in result.stderr
    assert result.stdout == ""


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


def test_convert_missing_step_backend_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block_imports(monkeypatch, "OCP")
    step_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(step_file), str(output_file)])

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


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_defaults_to_binary_usdc_and_validates(tmp_path: Path) -> None:
    input_file = tmp_path / "spool.step"
    output_file = input_file.with_suffix(".usdc")
    shutil.copyfile("tests/fixtures/spool-clamp-lid.step", input_file)

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_file),
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert f"Converted {input_file} to {output_file}" in compact(result.output)

    validate_result = runner.invoke(app, ["validate", str(output_file)])
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
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
            "--max-edge-length",
            "1000",
            "--target-triangles",
            "120",
            "--lods",
            "0.5",
            "--uv1",
            "box",
            "--materials",
            "display",
            "--report",
            str(report_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()
    assert report_file.exists()
    assert "Converted" in result.output
    report = json.loads(report_file.read_text(encoding="utf-8"))
    step_names = [step["name"] for step in report["steps"]]
    assert "write" in step_names
    assert "validate" in step_names
    assert report["finished_at"] is not None
    assert report["output_stats"]["materials"] == 0
    assert report["output_stats"]["triangles"] <= 120


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_json_output_includes_stats_and_report(tmp_path: Path) -> None:
    output_file = tmp_path / "output.usda"

    result = runner.invoke(
        app,
        [
            "--json",
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

    assert result.exit_code == 0, result.output
    assert output_file.exists()
    payload = json.loads(result.output)
    step_names = [step["name"] for step in payload["report"]["steps"]]
    assert payload["command"] == "convert"
    assert payload["stats"]["parts"] == 1
    assert payload["stats"]["triangles"] > 0
    assert payload["report"]["input_stats"]["parts"] == 1
    assert payload["report"]["finished_at"] is not None
    assert step_names[-2:] == ["write", "validate"]


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_debug_usda_authors_debug_metadata(tmp_path: Path) -> None:
    from pxr import Usd

    output_file = tmp_path / "debug.usda"

    result = runner.invoke(
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
            "--debug",
        ],
    )

    assert result.exit_code == 0, result.output
    stage = Usd.Stage.Open(str(output_file))
    assert stage is not None
    assert stage.GetRootLayer().comment == "Generated by fascat debug mode"
    assert stage.GetDefaultPrim().GetCustomDataByKey("fascat:debug") is True

    validate_result = runner.invoke(app, ["validate", str(output_file)])
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_material_modes_write_cad_display_and_no_material_usd(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    fixture = "tests/fixtures/radial-fan-50x15.step"
    expected_display_color = pytest.approx(
        (0.009721217676997185, 0.009721217676997185, 0.009721217676997185),
        abs=1e-6,
    )
    cad_output = tmp_path / "cad.usda"
    display_output = tmp_path / "display.usda"
    none_output = tmp_path / "none.usda"

    for mode, output in (("cad", cad_output), ("display", display_output), ("none", none_output)):
        result = runner.invoke(
            app,
            [
                "convert",
                fixture,
                str(output),
                "--sag",
                "0.2",
                "--target-triangles",
                "80",
                "--lods",
                "0.5",
                "--materials",
                mode,
            ],
        )
        assert result.exit_code == 0

    cad_stage = Usd.Stage.Open(str(cad_output))
    display_stage = Usd.Stage.Open(str(display_output))
    none_stage = Usd.Stage.Open(str(none_output))
    assert cad_stage is not None
    assert display_stage is not None
    assert none_stage is not None
    cad_mesh = next(prim for prim in Usd.PrimRange(cad_stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    display_mesh = next(prim for prim in Usd.PrimRange(display_stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    none_mesh = next(prim for prim in Usd.PrimRange(none_stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))

    assert cad_stage.GetPrimAtPath("/Materials")
    assert "MaterialBindingAPI" in cad_mesh.GetAppliedSchemas()

    assert not display_stage.GetPrimAtPath("/Materials")
    assert "MaterialBindingAPI" not in display_mesh.GetAppliedSchemas()
    assert tuple(UsdGeom.Mesh(display_mesh).GetDisplayColorAttr().Get()[0]) == expected_display_color

    assert not none_stage.GetPrimAtPath("/Materials")
    assert "MaterialBindingAPI" not in none_mesh.GetAppliedSchemas()
    assert tuple(UsdGeom.Mesh(none_mesh).GetDisplayColorAttr().Get()[0]) == (0.75, 0.75, 0.75)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_reads_step_from_stdin_and_writes_usd_to_stdout() -> None:
    step_data = Path("tests/fixtures/spool-clamp-lid.step").read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "convert",
            "-",
            "-",
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
            "--debug",
        ],
        input=step_data,
    )

    assert result.exit_code == 0
    assert "#usda" in result.output
    assert 'def Xform "Scene"' in result.output
    assert "Converted" not in result.output

    validate_result = runner.invoke(app, ["validate", "-"], input=result.output)
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_cli_stdio_paths_use_real_process_streams() -> None:
    step_data = Path("tests/fixtures/spool-clamp-lid.step").read_text(encoding="utf-8")

    inspect_result = subprocess.run(
        [sys.executable, "-m", "fascat", "--json", "inspect", "-"],
        input=step_data,
        capture_output=True,
        check=False,
        text=True,
    )
    assert inspect_result.returncode == 0
    inspect_payload = json.loads(inspect_result.stdout)
    assert inspect_payload["input"] == "-"
    assert inspect_payload["stats"]["parts"] == 1

    convert_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fascat",
            "convert",
            "-",
            "-",
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
            "--debug",
        ],
        input=step_data,
        capture_output=True,
        check=False,
        text=True,
    )
    assert convert_result.returncode == 0
    assert "#usda" in convert_result.stdout
    assert convert_result.stderr == ""

    validate_result = subprocess.run(
        [sys.executable, "-m", "fascat", "validate", "-"],
        input=convert_result.stdout,
        capture_output=True,
        check=False,
        text=True,
    )
    assert validate_result.returncode == 0
    assert "valid USD" in compact(validate_result.stdout)


def test_convert_existing_output_requires_force(tmp_path: Path) -> None:
    step_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")
    output_file.write_text("#usdc", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(step_file), str(output_file)])
    assert result.exit_code == 1
    assert "Pass --force" in compact(result.output)


def test_convert_writes_failure_report_sidecar(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.cli as cli

    input_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    report_file = tmp_path / "report.json"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    failure_report = Report(source_path=str(input_file))
    failure_report.add_error("invalid usd")
    failure_report.finish({"parts": 1, "triangles": 2})
    error = RuntimeError("invalid usd")
    error.report = failure_report

    def fail_convert(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(cli, "_convert_for_cli", fail_convert)

    result = runner.invoke(app, ["convert", str(input_file), str(output_file), "--report", str(report_file)])

    assert result.exit_code == 1
    assert "invalid usd" in result.output
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["errors"] == ["invalid usd"]
    assert report["finished_at"] is not None


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_writes_failure_report_when_usd_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fascat.pipeline as pipeline

    output_file = tmp_path / "output.usda"
    report_file = tmp_path / "report.json"

    def fail_validate(_path: str | Path) -> dict[str, int]:
        raise RuntimeError("invalid generated USD")

    monkeypatch.setattr(pipeline, "validate_usd", fail_validate)

    result = runner.invoke(
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
            "--report",
            str(report_file),
        ],
    )

    assert result.exit_code == 1
    assert "invalid generated USD" in result.output
    assert output_file.exists()
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    step_names = [step["name"] for step in report["steps"]]
    assert report["errors"] == ["invalid generated USD"]
    assert step_names[-2:] == ["write", "validate"]
    assert report["finished_at"] is not None
    assert report["output_stats"]["triangles"] > 0


def test_validate_rejects_unknown_extension(tmp_path: Path) -> None:
    output_file = tmp_path / "output.txt"
    output_file.write_text("not usd", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(output_file)])
    assert result.exit_code == 2
    assert "Unsupported USD extension" in result.output


def test_validate_missing_usd_backend_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block_imports(monkeypatch, "pxr")
    output_file = tmp_path / "output.usda"
    output_file.write_text("#usda 1.0", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(output_file)])

    assert result.exit_code == 1
    assert "USD validation requires usd-core" in result.output


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
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
    assert "valid USD" in compact(result.output)

    stdin_result = runner.invoke(app, ["validate", "-"], input=output_file.read_text(encoding="utf-8"))
    assert stdin_result.exit_code == 0
    assert "valid USD" in compact(stdin_result.output)


def test_convert_rejects_invalid_lods() -> None:
    result = runner.invoke(app, ["--dry-run", "convert", "input.step", "output.usdc", "--lods", "1.5"])
    assert result.exit_code == 2
    assert "--lods ratios" in result.output


def test_convert_rejects_unsorted_lods_during_dry_run(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--lods", "0.25,0.5"], capsys)
    assert result.exit_code == 2
    assert "--lods ratios must be sorted from highest to lowest detail" in result.stderr


def test_convert_rejects_invalid_max_edge_length(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--max-edge-length", "0"], capsys)
    assert result.exit_code == 2
    assert "--max-edge-length must be greater than 0" in result.stderr


def test_debug_requires_text_usd_output(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--debug"], capsys)
    assert result.exit_code == 2
    assert "--debug requires .usd or .usda output" in result.stderr


def test_convert_rejects_invalid_lods_as_json(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--json", "--dry-run", "convert", "input.step", "output.usdc", "--lods", "1.5"], capsys)
    assert result.exit_code == 2
    assert '"error": "--lods ratios must be greater than 0 and less than 1."' in result.stdout


def test_help_command_alias(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["help"], capsys)
    assert result.exit_code == 0
    assert "Usage: fascat" in compact(result.stdout)


def test_help_command_alias_for_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["help", "convert"], capsys)
    assert result.exit_code == 0
    assert "Usage: fascat convert" in compact(result.stdout)
    assert "--target-triangles" in plain(result.stdout)


def test_help_wins_with_invalid_tokens(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "input.step", "output.usdc", "--bad", "-h"], capsys)
    assert result.exit_code == 0
    assert "Usage: fascat convert" in compact(result.stdout)
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


def test_dash_input_requires_explicit_output(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "-"], capsys)
    assert result.exit_code == 2
    assert "Output path is required" in result.stderr


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


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_reports_stage_progress_to_stderr(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output_file = tmp_path / "output.usda"

    result = invoke_run(
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
        capsys,
    )

    assert result.exit_code == 0
    assert "Converted" in result.stdout
    assert "source:" in result.stderr
    assert "tessellate:" in result.stderr
    assert "write:" in result.stderr
    assert "validate:" in result.stderr
