from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from difflib import get_close_matches
from functools import lru_cache

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

    color_enabled = not _color_disabled_requested(_before_separator(raw_args))
    try:
        with _temporary_no_color(not color_enabled):
            _app.app(args=normalized_args, prog_name="fascat", color=color_enabled)
    except KeyboardInterrupt:
        _app.err.print("Interrupted.")
        raise SystemExit(_INTERRUPT_EXIT_CODE) from None


def _split_at_separator(args: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split at the first ``--``; everything from it onward is passed through verbatim."""
    raw_args = list(args)
    if "--" not in raw_args:
        return raw_args, []
    index = raw_args.index("--")
    return raw_args[:index], raw_args[index:]


def _before_separator(args: Sequence[str]) -> list[str]:
    head, _ = _split_at_separator(args)
    return head


@lru_cache(maxsize=1)
def _value_option_flags() -> frozenset[str]:
    """Option spellings that consume the following token as their value."""
    import click
    import typer.main

    root = typer.main.get_command(_app.app)
    commands: list[click.Command] = [root]
    if isinstance(root, click.Group):
        commands.extend(root.commands.values())
    flags: set[str] = set()
    for command in commands:
        for param in command.params:
            if isinstance(param, click.Option) and not param.is_flag and param.nargs != 0:
                flags.update(param.opts)
                flags.update(param.secondary_opts)
    return frozenset(flags)


def _hoistable_flags(args: Sequence[str]) -> list[bool]:
    """Mark tokens that may be treated as global flags rather than option values."""
    value_flags = _value_option_flags()
    hoistable: list[bool] = []
    expects_value = False
    for arg in args:
        hoistable.append(not expects_value)
        expects_value = not expects_value and arg in value_flags
    return hoistable


def _normalize_args(args: Sequence[str]) -> list[str]:
    head, tail = _split_at_separator(args)
    hoistable = _hoistable_flags(head)

    def _is_global(index: int, arg: str, flags: set[str] | frozenset[str]) -> bool:
        return hoistable[index] and arg in flags

    if any(_is_global(index, arg, VERSION_FLAGS) for index, arg in enumerate(head)):
        version_args = ["--version"]
        if any(_is_global(index, arg, {"--json"}) for index, arg in enumerate(head)):
            version_args.insert(0, "--json")
        return version_args

    if any(_is_global(index, arg, HELP_FLAGS) for index, arg in enumerate(head)):
        command = _first_command(head, hoistable)
        return [command, "--help"] if command is not None else ["--help"]

    if head and head[0] == "help":
        if len(head) == 1:
            return ["--help"]
        return [head[1], "--help"]

    global_flags = [arg for index, arg in enumerate(head) if _is_global(index, arg, GLOBAL_FLAG_ALIASES)]
    remaining = [arg for index, arg in enumerate(head) if not _is_global(index, arg, GLOBAL_FLAG_ALIASES)]
    return [*global_flags, *remaining, *tail]


def _first_command(args: Sequence[str], hoistable: Sequence[bool]) -> str | None:
    for index, arg in enumerate(args):
        if hoistable[index] and arg in COMMAND_NAMES and arg != "help":
            return arg
    return None


def _find_unknown_command(args: Sequence[str]) -> str | None:
    remaining = list(args)
    while remaining and remaining[0] in GLOBAL_FLAG_ALIASES:
        remaining.pop(0)
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
