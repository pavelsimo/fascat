from __future__ import annotations

import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated

import typer
import typer.rich_utils as rich_utils
from rich.console import Console

from fascat import __version__

DOCS_URL = "https://pavelsimo.github.io/fascat"
ISSUES_URL = "https://github.com/pavelsimo/fascat/issues"
rich_utils.MAX_WIDTH = 120
COMMAND_NAMES = ("inspect", "convert", "validate", "runtime-fixtures", "version", "help")
GLOBAL_FLAG_ALIASES = {
    "--json",
    "--dry-run",
    "-n",
    "--quiet",
    "-q",
    "--verbose",
    "-v",
    "--no-color",
    "--no-input",
}
HELP_FLAGS = {"-h", "--help"}
VERSION_FLAGS = {"-V", "--version"}
TOP_LEVEL_EPILOG = f"""Examples:
  fascat inspect motor.step
  fascat inspect legacy.igs
  fascat convert motor.step motor.usdc --profile realtime-desktop
  fascat convert source.brep source.glb --profile realtime-web
  fascat convert motor.step motor.glb --profile virtual-reality
  fascat --json validate motor.usdc
  fascat runtime-fixtures runtime-parity/

Docs: {DOCS_URL}
Issues: {ISSUES_URL}"""

app = typer.Typer(
    name="fascat",
    help="convert CAD data into realtime-ready OpenUSD and glTF assets",
    epilog=TOP_LEVEL_EPILOG,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_show_locals=False,
)

out = Console()
err = Console(stderr=True)


@dataclass(frozen=True)
class CliState:
    verbose: bool
    quiet: bool
    json_output: bool
    no_color: bool
    dry_run: bool
    no_input: bool


def _version_payload() -> dict[str, str]:
    return {"command": "version", "version": __version__}


def _version_callback(ctx: typer.Context, value: bool) -> None:
    if value:
        if _json_requested_for_version_callback(ctx):
            out.print_json(json.dumps(_version_payload()))
        else:
            out.print(f"fascat {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON.", is_eager=True)] = False,
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=False,
            help="Show version and exit.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output.", is_eager=False),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-essential output.")] = False,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Disable ANSI color output.",
            envvar="NO_COLOR",
        ),
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Preview changes without applying them.")] = False,
    no_input: Annotated[bool, typer.Option("--no-input", help="Disable interactive prompts.")] = False,
) -> None:
    """convert CAD data into realtime-ready OpenUSD and glTF assets"""
    _configure_consoles(no_color)
    ctx.obj = CliState(
        verbose=verbose,
        quiet=quiet,
        json_output=json_output,
        no_color=no_color,
        dry_run=dry_run,
        no_input=no_input,
    )


def _is_tty() -> bool:
    return sys.stdin.isatty()


def _configure_consoles(no_color: bool) -> None:
    global out, err  # noqa: PLW0603
    disable_color = _color_disabled_requested(["--no-color"] if no_color else [])
    out = Console(no_color=disable_color)
    err = Console(stderr=True, no_color=disable_color)


def _state(ctx: typer.Context) -> CliState:
    if isinstance(ctx.obj, CliState):
        return ctx.obj
    return CliState(verbose=False, quiet=False, json_output=False, no_color=False, dry_run=False, no_input=False)


def _json_requested_for_version_callback(ctx: typer.Context) -> bool:
    return ctx.params.get("json_output") is True


def _color_disabled_requested(args: Sequence[str]) -> bool:
    return (
        "--no-color" in args
        or bool(os.environ.get("NO_COLOR"))
        or os.environ.get("TERM") == "dumb"
        or not sys.stdout.isatty()
        or not sys.stderr.isatty()
    )
