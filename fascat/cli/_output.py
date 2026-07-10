from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

import typer
from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn

from fascat.io._suffixes import (
    GLTF_SUFFIXES,
    OBJ_SUFFIXES,
    STL_SUFFIXES,
)
from fascat.report import Report

from . import _app
from ._app import _state

_WARNING_DISPLAY_LIMIT = 10
_INTERRUPT_EXIT_CODE = 130  # 128 + SIGINT
_STAGE_TITLES: dict[str, str] = {
    "source": "source",
    "heal_brep": "heal brep",
    "tessellate": "tessellate",
    "repair": "repair",
    "stage": "stage attributes",
    "merge_vertices": "merge vertices",
    "delete_degenerate_polygons": "delete degenerate polygons",
    "merge": "merge",
    "explode": "explode",
    "replace": "replace",
    "optimize_scene": "optimize scene",
    "scene": "optimize scene",
    "bake_materials": "bake materials",
    "remove_holes": "remove holes",
    "remove_occluded": "remove occluded",
    "decimate": "decimate",
    "run_lod_generators": "generate LODs",
    "optimize": "optimize",
    "lods": "generate LODs",
    "write": "write",
    "validate": "validate",
}


def _emit(ctx: typer.Context, payload: dict[str, Any], human_message: str) -> None:
    state = _state(ctx)
    if state.json_output:
        _app.out.print_json(json.dumps(payload))
    elif not state.quiet:
        _app.out.print(human_message, markup=False, soft_wrap=True)


def _require_existing_file(path: Path, label: str, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if _is_stdio(path):
        return
    if not path.exists():
        _fail(ctx, payload, f"Missing {label} file: {path}")
    if not path.is_file():
        _fail(ctx, payload, f"Expected {label} to be a file: {path}")


def _is_stdio(path: Path) -> bool:
    return str(path) == "-"


def _fail(ctx: typer.Context, payload: dict[str, Any], message: str, code: int = 1) -> NoReturn:
    if _state(ctx).json_output:
        _app.out.print_json(json.dumps({**payload, "error": message}))
    else:
        _app.err.print(message)
    raise typer.Exit(code)


def _print_report_warnings(ctx: typer.Context, report: Report, *, limit: int = _WARNING_DISPLAY_LIMIT) -> None:
    state = _state(ctx)
    if state.quiet or state.json_output or not report.warnings:
        return
    display_limit = len(report.warnings) if state.verbose else limit
    for warning in report.warnings[:display_limit]:
        _app.err.print(f"warning: {warning}", style="yellow", markup=False, soft_wrap=True)
    hidden = len(report.warnings) - display_limit
    if hidden > 0:
        _app.err.print(
            f"... and {hidden} more warning(s); pass --verbose or --report report.json for the full list",
            style="yellow",
            markup=False,
        )


def _print_verbose_operation_diagnostics(ctx: typer.Context, payload: dict[str, Any]) -> None:
    state = _state(ctx)
    if not state.verbose or state.quiet or state.json_output:
        return
    diagnostics = payload.get("operation_diagnostics")
    if not isinstance(diagnostics, list) or not diagnostics:
        return
    _app.err.print("operation diagnostics:", style="cyan", markup=False)
    for item in diagnostics:
        if not isinstance(item, dict):
            continue
        operation = str(item.get("operation", "unknown"))
        level = str(item.get("level", "unknown"))
        message = str(item.get("message", ""))
        suffix = f" — {message}" if message else ""
        _app.err.print(f"  {operation} [{level}]{suffix}", markup=False, soft_wrap=True)


def _interrupt(ctx: typer.Context, payload: dict[str, Any]) -> NoReturn:
    if _state(ctx).json_output:
        _app.out.print_json(json.dumps({**payload, "error": "interrupted"}))
    else:
        _app.err.print("Interrupted.")
    raise typer.Exit(_INTERRUPT_EXIT_CODE)


def _format_stats_compact(stats: dict[str, int]) -> str:
    """Condensed stats for the live step log (matches the conversion summary style)."""
    triangles = stats.get("triangles", 0)
    vertices = stats.get("vertices", 0)
    if triangles or vertices:
        return f"{vertices:,} verts · {triangles:,} tris"
    parts = stats.get("parts", 0)
    occurrences = stats.get("occurrences", 0)
    part_label = "part" if parts == 1 else "parts"
    return f"{parts} {part_label} · {occurrences} occ"


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _output_size_text(path: Path) -> str | None:
    try:
        return _human_size(path.stat().st_size)
    except OSError:
        return None


class _StageReporter:
    """Renders conversion stage progress on stderr.

    Three modes drive the same ``progress(step, stats)`` callback contract:

    - ``disabled`` — no output (``--quiet``/``--json``/stdio output).
    - ``plain`` — one line per finished stage, for non-TTY/CI streams. Matches
      the historical ``step: stats`` format byte-for-byte.
    - ``live`` — an animated spinner with a running clock while a stage runs,
      with finished stages dropping into a colorized log, plus a final summary.
    """

    def __init__(self, mode: str, input_path: Path, output_path: Path) -> None:
        self._mode = mode
        self._input_path = input_path
        self._output_path = output_path
        self._bar: Progress | None = None
        self._task: TaskID | None = None
        self._start = 0.0
        self._last = 0.0

    @property
    def live(self) -> bool:
        return self._mode == "live"

    @property
    def callback(self) -> Callable[[str, dict[str, int]], None] | None:
        return None if self._mode == "disabled" else self._on_step

    def __enter__(self) -> _StageReporter:
        self._start = self._last = time.monotonic()
        if self._mode == "live":
            _app.err.print(f"Converting [bold]{self._input_path}[/] → [bold]{self._output_path}[/]\n")
            self._bar = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                TimeElapsedColumn(),
                console=_app.err,
                transient=True,
            )
            self._bar.start()
            self._task = self._bar.add_task("working…", total=None)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._bar is not None:
            self._bar.stop()
            self._bar = None

    def _on_step(self, step: str, stats: dict[str, int]) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        self._last = now
        if self._mode == "plain":
            _app.err.print(f"{step}: {_format_stats(stats)}")
            return
        title = _STAGE_TITLES.get(step, step)
        line = f"  [green]✓[/] {title}  [dim]{_format_stats_compact(stats)} · {elapsed:.1f}s[/]"
        if self._bar is not None:
            self._bar.console.print(line)
            if self._task is not None:
                self._bar.reset(self._task, description="working…", total=None)

    def summary(self, asset: Any) -> None:
        if self._mode != "live":
            return
        total = time.monotonic() - self._start
        stats = asset.stats()
        verts = f"{stats.get('vertices', 0):,}"
        tris = f"{stats.get('triangles', 0):,}"
        size = _output_size_text(self._output_path)
        size_suffix = f" ({size})" if size else ""
        _app.err.print(
            f"[green]✓[/] [bold]Done[/] in {total:.1f}s — {verts} verts · {tris} tris — "
            f"{self._output_path}{size_suffix}"
        )


def _stage_reporter(ctx: typer.Context, input_path: Path, output_path: Path) -> _StageReporter:
    state = _state(ctx)
    if state.quiet or state.json_output or _is_stdio(output_path):
        mode = "disabled"
    elif _app.err.is_terminal:
        mode = "live"
    else:
        mode = "plain"
    return _StageReporter(mode, input_path, output_path)


def _export_label(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in GLTF_SUFFIXES:
        return "glTF"
    if suffix in OBJ_SUFFIXES:
        return "OBJ"
    if suffix in STL_SUFFIXES:
        return "STL"
    return "USD"


def _format_stats(stats: dict[str, int]) -> str:
    parts = []
    for key in ("parts", "occurrences", "materials", "meshes", "vertices", "points", "triangles"):
        if key in stats:
            parts.append(f"{stats[key]} {key}")
    return ", ".join(parts) if parts else json.dumps(stats, sort_keys=True)
