from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from difflib import get_close_matches

from . import _app
from ._app import COMMAND_NAMES, GLOBAL_FLAG_ALIASES, HELP_FLAGS, VERSION_FLAGS, _color_disabled_requested
from ._output import _INTERRUPT_EXIT_CODE


def run(args: Sequence[str] | None = None) -> None:
    """Console-script entry point with CLI-guideline argument normalization."""
    raw_args = list(sys.argv[1:] if args is None else args)
    normalized_args = _normalize_args(raw_args)
    unknown_command = _find_unknown_command(normalized_args)
    if unknown_command is not None:
        _print_unknown_command(unknown_command)
        raise SystemExit(2)

    color_enabled = not _color_disabled_requested(raw_args)
    try:
        with _temporary_no_color(not color_enabled):
            _app.app(args=normalized_args, prog_name="fascat", color=color_enabled)
    except KeyboardInterrupt:
        _app.err.print("Interrupted.")
        raise SystemExit(_INTERRUPT_EXIT_CODE) from None


def _normalize_args(args: Sequence[str]) -> list[str]:
    raw_args = list(args)
    if any(arg in VERSION_FLAGS for arg in raw_args):
        version_args = ["--version"]
        if "--json" in raw_args:
            version_args.insert(0, "--json")
        return version_args

    if any(arg in HELP_FLAGS for arg in raw_args):
        command = _first_command(raw_args)
        return [command, "--help"] if command is not None else ["--help"]

    if raw_args and raw_args[0] == "help":
        if len(raw_args) == 1:
            return ["--help"]
        return [raw_args[1], "--help"]

    global_flags = [arg for arg in raw_args if arg in GLOBAL_FLAG_ALIASES]
    remaining = [arg for arg in raw_args if arg not in GLOBAL_FLAG_ALIASES]
    return [*global_flags, *remaining]


def _first_command(args: Sequence[str]) -> str | None:
    for arg in args:
        if arg in COMMAND_NAMES and arg != "help":
            return arg
    return None


def _find_unknown_command(args: Sequence[str]) -> str | None:
    remaining = [arg for arg in args if arg not in GLOBAL_FLAG_ALIASES]
    if not remaining:
        return None
    candidate = remaining[0]
    if candidate.startswith("-") or candidate in COMMAND_NAMES:
        return None
    return candidate


def _print_unknown_command(command: str) -> None:
    suggestion = get_close_matches(command, COMMAND_NAMES, n=1)
    message = f"No such command '{command}'."
    if suggestion:
        message = f"{message} Did you mean '{suggestion[0]}'?"
    _app.err.print(message)
    _app.err.print("Run 'fascat --help' to see available commands.")


class _temporary_no_color:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.previous_value: str | None = None
        self.previous_color_system: object | None = None
        self.previous_force_terminal: object | None = None

    def __enter__(self) -> None:
        self.previous_value = os.environ.get("NO_COLOR")
        if self.enabled:
            os.environ["NO_COLOR"] = "1"
            import typer.rich_utils as rich_utils

            self.previous_color_system = rich_utils.COLOR_SYSTEM
            self.previous_force_terminal = rich_utils.FORCE_TERMINAL
            rich_utils.COLOR_SYSTEM = None
            rich_utils.FORCE_TERMINAL = False

    def __exit__(self, *_exc_info: object) -> None:
        if not self.enabled:
            return
        import typer.rich_utils as rich_utils

        rich_utils.COLOR_SYSTEM = self.previous_color_system  # type: ignore[assignment]
        rich_utils.FORCE_TERMINAL = self.previous_force_terminal  # type: ignore[assignment]
        if self.previous_value is None:
            os.environ.pop("NO_COLOR", None)
        else:
            os.environ["NO_COLOR"] = self.previous_value
