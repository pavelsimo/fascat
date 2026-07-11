import json
from pathlib import Path

import pytest

from fascat import __version__
from fascat.cli import app

from ._cli_test_helpers import (
    _FakeContext,
    _make_state,
    _patched_err,
    compact,
    invoke_run,
    plain,
    runner,
)


def test_cli_package_registers_public_commands_once_in_help_order() -> None:
    import fascat.cli as cli

    assert [command.name for command in cli.app.registered_commands] == [
        "inspect",
        "convert",
        "validate",
        "version",
        "help",
    ]
    assert {
        "CliState",
        "app",
        "main",
        "run",
        "by_name",
        "profile_from_file",
        "convert",
        "validate_export",
        "cmd_inspect",
        "cmd_convert",
        "cmd_validate",
        "cmd_version",
        "cmd_help",
    } <= set(cli.__all__)


@pytest.mark.parametrize(
    "overrides, output_path",
    [
        ({"quiet": True}, "out.glb"),
        ({"json_output": True}, "out.glb"),
        ({}, "-"),
    ],
)
def test_stage_reporter_disabled_yields_no_callback(
    monkeypatch: pytest.MonkeyPatch, overrides: dict[str, bool], output_path: str
) -> None:
    from fascat.cli._output import _stage_reporter

    _patched_err(monkeypatch, terminal=True)
    reporter = _stage_reporter(_FakeContext(_make_state(**overrides)), Path("in.step"), Path(output_path))
    assert reporter.callback is None
    assert reporter.live is False


def test_stage_reporter_plain_mode_matches_legacy_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    from fascat.cli._output import _stage_reporter

    buffer = _patched_err(monkeypatch, terminal=False)
    reporter = _stage_reporter(_FakeContext(_make_state()), Path("in.step"), Path("out.glb"))
    assert reporter.live is False
    with reporter:
        assert reporter.callback is not None
        reporter.callback("tessellate", {"parts": 1, "vertices": 89273, "triangles": 120884})
    text = plain(buffer.getvalue())
    assert "tessellate: 1 parts, 89273 vertices, 120884 triangles" in text
    assert "Done in" not in text
    assert "✓" not in text


def test_stage_reporter_live_mode_renders_log_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    from fascat.cli._output import _stage_reporter

    buffer = _patched_err(monkeypatch, terminal=True)
    reporter = _stage_reporter(_FakeContext(_make_state()), Path("in.step"), Path("out.glb"))
    assert reporter.live is True

    class _FakeAsset:
        def stats(self) -> dict[str, int]:
            return {"vertices": 84010, "triangles": 120884}

    with reporter:
        assert reporter.callback is not None
        reporter.callback("tessellate", {"parts": 1, "vertices": 89273, "triangles": 120884})
    reporter.summary(_FakeAsset())

    text = compact(buffer.getvalue())
    assert "Converting in.step" in text
    assert "✓ tessellate 89,273 verts · 120,884 tris" in text
    assert "Done in" in text
    assert "84,010 verts · 120,884 tris" in text


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


@pytest.mark.parametrize("args", [["--json", "--version"], ["--version", "--json"]])
def test_json_version_flag(args: list[str]) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert json.loads(result.output) == {"command": "version", "version": __version__}


def test_json_version_subcommand() -> None:
    result = runner.invoke(app, ["--json", "version"])
    assert result.exit_code == 0
    assert json.loads(result.output) == {"command": "version", "version": __version__}


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


def test_quiet_suppresses_nonessential_dry_run_output(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--quiet", "--dry-run", "convert", "input.step", "output.usdc"], capsys)

    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""


def test_verbose_dry_run_prints_operation_diagnostics(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--verbose", "--dry-run", "convert", "input.step", "output.usdc"], capsys)

    assert result.exit_code == 0
    assert "Would convert input.step to output.usdc" in result.stdout
    assert "operation diagnostics:" in result.stderr
    assert "import [exact]" in result.stderr
    assert "tessellate [exact]" in result.stderr
    assert "export [exact]" in result.stderr


def test_verbose_dry_run_preserves_json_stdout(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--json", "--verbose", "--dry-run", "convert", "input.step", "output.usdc"], capsys)

    assert result.exit_code == 0
    assert json.loads(result.stdout)["dry_run"] is True
    assert result.stderr == ""


def test_quiet_does_not_suppress_errors(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--quiet", "validate", "missing.usdc"], capsys)

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Missing output file: missing.usdc" in result.stderr


def test_debug_requires_text_usd_output(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--debug"], capsys)
    assert result.exit_code == 2
    assert "--debug requires .usd or .usda output" in result.stderr


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


def test_json_version_wins_after_subcommand(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "input.step", "output.usdc", "--json", "--version"], capsys)
    assert result.exit_code == 0
    assert json.loads(result.stdout) == {"command": "version", "version": __version__}
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


def test_run_converts_keyboard_interrupt_to_exit_130(
    monkeypatch: pytest.MonkeyPatch,
    capsys,  # type: ignore[no-untyped-def]
) -> None:
    from fascat.cli import _app

    def interrupt_app(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(_app, "app", interrupt_app)

    result = invoke_run(["version"], capsys)

    assert result.exit_code == 130
    assert "Interrupted." in result.stderr
