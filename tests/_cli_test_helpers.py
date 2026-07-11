from __future__ import annotations

import builtins
import re
from typing import Any, NamedTuple

import pytest
from typer.testing import CliRunner

from fascat.cli import run

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


class _FakeContext:
    def __init__(self, state: Any) -> None:
        self.obj = state


def _make_state(**overrides: bool) -> Any:
    from fascat.cli import CliState

    defaults = dict(verbose=False, quiet=False, json_output=False, no_color=False, dry_run=False, no_input=False)
    defaults.update(overrides)
    return CliState(**defaults)  # type: ignore[arg-type]


def _patched_err(monkeypatch: pytest.MonkeyPatch, *, terminal: bool) -> Any:
    import io

    from rich.console import Console

    from fascat.cli import _app

    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=terminal, force_interactive=False, width=120)
    monkeypatch.setattr(_app, "err", console)
    return buffer
