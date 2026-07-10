from __future__ import annotations

from typing import Annotated

import typer

from fascat import __version__

from ._app import COMMAND_NAMES, DOCS_URL, _color_disabled_requested, _version_payload, app
from ._output import _emit
from ._runner import _print_unknown_command


@app.command("version", epilog=f"Docs: {DOCS_URL}")
def cmd_version(ctx: typer.Context) -> None:
    """Show the version and exit."""
    _emit(ctx, _version_payload(), f"fascat {__version__}")


@app.command(
    "help",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    epilog=f"Docs: {DOCS_URL}",
)
def cmd_help(
    command: Annotated[str | None, typer.Argument(help="Command to show help for.")] = None,
) -> None:
    """Show help for fascat or one command."""
    if command is not None and command not in COMMAND_NAMES:
        _print_unknown_command(command)
        raise typer.Exit(2)
    args = ["--help"] if command is None else [command, "--help"]
    app(args=args, prog_name="fascat", color=not _color_disabled_requested([]))
