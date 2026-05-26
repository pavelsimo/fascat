from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import typer.rich_utils as rich_utils
from rich.console import Console

from fascat import __version__
from fascat.io.step import read_step, read_step_bytes
from fascat.io.usd import validate_usd
from fascat.options import LODOptions, OptimizeOptions, StageOptions, Tessellation
from fascat.pipeline import convert
from fascat.profiles import by_name

DOCS_URL = "https://pavelsimo.github.io/fascat"
ISSUES_URL = "https://github.com/pavelsimo/fascat/issues"
rich_utils.MAX_WIDTH = 120
STEP_SUFFIXES = {".step", ".stp"}
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
COMMAND_NAMES = ("inspect", "convert", "validate", "version", "help")
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
  fascat convert motor.step motor.usdc --profile realtime-desktop
  fascat --json validate motor.usdc

Docs: {DOCS_URL}
Issues: {ISSUES_URL}"""

app = typer.Typer(
    name="fascat",
    help="convert CAD STEP data into realtime-ready OpenUSD assets",
    epilog=TOP_LEVEL_EPILOG,
    no_args_is_help=True,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    pretty_exceptions_show_locals=False,
)

out = Console()
err = Console(stderr=True)


class Profile(str, Enum):
    INSPECT_ONLY = "inspect-only"
    REALTIME_DESKTOP = "realtime-desktop"
    REALTIME_WEB = "realtime-web"


class UVMode(str, Enum):
    NONE = "none"
    BOX = "box"
    UNWRAP = "unwrap"


class MaterialMode(str, Enum):
    CAD = "cad"
    DISPLAY = "display"
    NONE = "none"


@dataclass(frozen=True)
class CliState:
    verbose: bool
    quiet: bool
    json_output: bool
    no_color: bool
    dry_run: bool
    no_input: bool


def _version_callback(value: bool) -> None:
    if value:
        out.print(f"fascat {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    _version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output.", is_eager=False),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Suppress non-essential output.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output results as JSON.")] = False,
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
    """convert CAD STEP data into realtime-ready OpenUSD assets"""
    _configure_consoles(no_color)
    ctx.obj = CliState(
        verbose=verbose,
        quiet=quiet,
        json_output=json_output,
        no_color=no_color,
        dry_run=dry_run,
        no_input=no_input,
    )


@app.command(
    "inspect",
    epilog=f"""Examples:
  fascat inspect motor.step
  fascat --json inspect motor.step
  cat motor.step | fascat inspect -

Docs: {DOCS_URL}/reference.html""",
)
def cmd_inspect(
    ctx: typer.Context,
    input_path: Annotated[Path, typer.Argument(help="STEP file to inspect, or '-' for stdin.", allow_dash=True)],
    profile: Annotated[Profile, typer.Option("--profile", help="Inspection profile to apply.")] = Profile.INSPECT_ONLY,
) -> None:
    """Inspect STEP assembly metadata and planned conversion inputs."""
    state = _state(ctx)
    payload = {
        "command": "inspect",
        "input": str(input_path),
        "profile": profile.value,
        "dry_run": state.dry_run,
    }
    _validate_step_input(input_path, ctx, payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would inspect {input_path} with profile {profile.value}.")
        return

    asset = _read_step_for_cli(input_path, ctx, payload)
    profile_options = by_name(profile.value)
    result = {
        **payload,
        "units": asset.units,
        "meters_per_unit": asset.meters_per_unit,
        "up_axis": asset.up_axis,
        "stats": asset.stats(),
        "options": profile_options.to_dict(),
        "root": asset.root.to_dict(),
        "parts": [part.to_dict() for part in asset.parts.values()],
        "materials": [material.to_dict() for material in asset.materials.values()],
        "report": asset.report.to_dict(),
    }
    _emit(ctx, result, f"{input_path}: {_format_stats(asset.stats())}; units={asset.units}")


@app.command(
    "convert",
    epilog=f"""Examples:
  fascat convert motor.step motor.usdc
  fascat convert motor.step
  fascat convert motor.step motor.usda --debug --report report.json
  fascat --dry-run --json convert motor.step motor.usdc
  cat motor.step | fascat convert - - --profile realtime-web

Docs: {DOCS_URL}/reference.html""",
)
def cmd_convert(
    ctx: typer.Context,
    input_path: Annotated[Path, typer.Argument(help="Input STEP file, or '-' for stdin.", allow_dash=True)],
    output_path: Annotated[
        Path | None,
        typer.Argument(
            help="Output USD file, usually .usdc or .usda, or '-' for stdout. Defaults to input .usdc.",
            allow_dash=True,
        ),
    ] = None,
    profile: Annotated[Profile, typer.Option("--profile", help="Conversion profile.")] = Profile.REALTIME_DESKTOP,
    sag: Annotated[float | None, typer.Option("--sag", help="CAD tessellation sag tolerance.")] = None,
    angle: Annotated[
        float | None,
        typer.Option("--angle", help="CAD tessellation angle tolerance in degrees."),
    ] = None,
    target_triangles: Annotated[int | None, typer.Option("--target-triangles", help="LOD0 triangle budget.")] = None,
    ratio: Annotated[
        float | None,
        typer.Option("--ratio", help="Simplification ratio when no triangle target is set."),
    ] = None,
    max_edge_length: Annotated[
        float | None,
        typer.Option("--max-edge-length", help="Split tessellated triangles longer than this length."),
    ] = None,
    lods: Annotated[
        str | None,
        typer.Option("--lods", help="Comma-separated LOD ratios, for example 0.5,0.25,0.1."),
    ] = None,
    uv0: Annotated[UVMode, typer.Option("--uv0", help="UV0 generation mode.")] = UVMode.BOX,
    uv1: Annotated[UVMode, typer.Option("--uv1", help="UV1 generation mode.")] = UVMode.NONE,
    materials: Annotated[
        MaterialMode,
        typer.Option("--materials", help="Material staging mode: cad, display, or none."),
    ] = MaterialMode.CAD,
    preserve_instances: Annotated[
        bool,
        typer.Option(
            "--preserve-instances/--no-preserve-instances",
            help="Preserve repeated parts as shared instances.",
        ),
    ] = True,
    debug: Annotated[bool, typer.Option("--debug", help="Prefer debuggable USDA output conventions.")] = False,
    report: Annotated[Path | None, typer.Option("--report", help="Write a JSON conversion report sidecar.")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite an existing output file.")] = False,
) -> None:
    """Convert a STEP file into realtime-ready OpenUSD."""
    state = _state(ctx)
    payload: dict[str, Any] = {
        "command": "convert",
        "input": str(input_path),
        "output": str(output_path) if output_path is not None else None,
        "profile": profile.value,
        "sag": sag,
        "angle": angle,
        "target_triangles": target_triangles,
        "ratio": ratio,
        "max_edge_length": max_edge_length,
        "lods": None,
        "uv0": uv0.value,
        "uv1": uv1.value,
        "materials": materials.value,
        "preserve_instances": preserve_instances,
        "debug": debug,
        "report": str(report) if report else None,
        "force": force,
        "dry_run": state.dry_run,
    }
    lod_values = _parse_lods(lods, ctx, payload)
    payload["lods"] = lod_values
    _validate_step_input(input_path, ctx, payload)
    output_path = _resolve_convert_output(input_path, output_path, ctx, payload)
    payload["output"] = str(output_path)
    _validate_usd_output(output_path, ctx, payload)
    if ratio is not None and (ratio <= 0.0 or ratio >= 1.0):
        _fail(ctx, payload, "--ratio must be greater than 0 and less than 1.", code=2)
    if sag is not None and sag <= 0.0:
        _fail(ctx, payload, "--sag must be greater than 0.", code=2)
    if angle is not None and (angle <= 0.0 or angle > 180.0):
        _fail(ctx, payload, "--angle must be greater than 0 and no more than 180.", code=2)
    if target_triangles is not None and target_triangles <= 0:
        _fail(ctx, payload, "--target-triangles must be greater than 0.", code=2)
    if max_edge_length is not None and max_edge_length <= 0.0:
        _fail(ctx, payload, "--max-edge-length must be greater than 0.", code=2)
    if debug and not _is_stdio(output_path) and output_path.suffix.lower() == ".usdc":
        _fail(ctx, payload, "--debug requires .usd or .usda output, not binary .usdc.", code=2)

    if state.dry_run:
        _emit(ctx, payload, f"Would convert {input_path} to {output_path} with profile {profile.value}.")
        return

    _require_existing_file(input_path, "input", ctx, payload)
    if not _is_stdio(output_path) and output_path.exists() and not force:
        _fail(ctx, payload, f"Output already exists: {output_path}. Pass --force to overwrite.")

    try:
        profile_options = by_name(profile.value)
        base_tessellation = profile_options.tessellation
        if base_tessellation is None:
            _fail(ctx, payload, "The inspect-only profile cannot be used for conversion.", code=2)
        tessellation = replace(
            base_tessellation,
            sag=sag if sag is not None else base_tessellation.sag,
            angle=angle if angle is not None else base_tessellation.angle,
            max_edge_length=max_edge_length if max_edge_length is not None else base_tessellation.max_edge_length,
        )
        optimize_options = profile_options.optimize
        if optimize_options is not None:
            optimize_options = replace(
                optimize_options,
                target_triangles=target_triangles
                if target_triangles is not None
                else optimize_options.target_triangles,
                ratio=ratio,
                preserve_instances=preserve_instances,
            )
        stage_options = replace(profile_options.stage, materials=materials.value, uv0=uv0.value, uv1=uv1.value)
        lod_options = LODOptions(tuple(lod_values)) if lod_values is not None else profile_options.lods
        asset = _convert_for_cli(
            input_path,
            output_path,
            profile=profile.value,
            tessellation=tessellation,
            stage=stage_options,
            optimize=optimize_options,
            lods=lod_options,
            progress=_progress_callback(ctx, output_path),
            debug=debug,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc

    if report is not None:
        asset.report.write_json(report)

    if _is_stdio(output_path):
        return

    result = {
        **payload,
        "stats": asset.stats(),
        "report": asset.report.to_dict(),
    }
    _emit(ctx, result, f"Converted {input_path} to {output_path}: {_format_stats(asset.stats())}.")


@app.command(
    "validate",
    epilog=f"""Examples:
  fascat validate motor.usdc
  fascat --json validate motor.usda
  cat motor.usdc | fascat validate -

Docs: {DOCS_URL}/reference.html""",
)
def cmd_validate(
    ctx: typer.Context,
    output_path: Annotated[Path, typer.Argument(help="USD file to validate, or '-' for stdin.", allow_dash=True)],
) -> None:
    """Validate a generated USD file."""
    state = _state(ctx)
    payload = {
        "command": "validate",
        "output": str(output_path),
        "dry_run": state.dry_run,
    }
    _validate_usd_output(output_path, ctx, payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would validate {output_path}.")
        return

    _require_existing_file(output_path, "output", ctx, payload)
    try:
        stats = _validate_usd_for_cli(output_path)
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc
    _emit(ctx, {**payload, "stats": stats}, f"{output_path}: valid USD, {_format_stats(stats)}.")


@app.command("version", epilog=f"Docs: {DOCS_URL}")
def cmd_version() -> None:
    """Show the version and exit."""
    out.print(f"fascat {__version__}")


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


def run(args: Sequence[str] | None = None) -> None:
    """Console-script entry point with CLI-guideline argument normalization."""
    raw_args = list(sys.argv[1:] if args is None else args)
    normalized_args = _normalize_args(raw_args)
    unknown_command = _find_unknown_command(normalized_args)
    if unknown_command is not None:
        _print_unknown_command(unknown_command)
        raise SystemExit(2)

    color_enabled = not _color_disabled_requested(raw_args)
    with _temporary_no_color(not color_enabled):
        app(args=normalized_args, prog_name="fascat", color=color_enabled)


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


def _emit(ctx: typer.Context, payload: dict[str, Any], human_message: str) -> None:
    state = _state(ctx)
    if state.json_output:
        out.print_json(json.dumps(payload))
    elif not state.quiet:
        out.print(human_message)


def _require_existing_file(path: Path, label: str, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if _is_stdio(path):
        return
    if not path.exists():
        _fail(ctx, payload, f"Missing {label} file: {path}")
    if not path.is_file():
        _fail(ctx, payload, f"Expected {label} to be a file: {path}")


def _resolve_convert_output(
    input_path: Path,
    output_path: Path | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> Path:
    if output_path is not None:
        return output_path
    if _is_stdio(input_path):
        _fail(ctx, payload, "Output path is required when reading STEP data from stdin.", code=2)
    return input_path.with_suffix(".usdc")


def _parse_lods(value: str | None, ctx: typer.Context, payload: dict[str, Any]) -> list[float] | None:
    if value is None:
        return None
    try:
        ratios = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        _fail(ctx, payload, "--lods must be a comma-separated list of numbers.", code=2)
        raise AssertionError("unreachable") from exc
    if not ratios:
        _fail(ctx, payload, "--lods must include at least one ratio.", code=2)
    if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
        _fail(ctx, payload, "--lods ratios must be greater than 0 and less than 1.", code=2)
    return ratios


def _validate_step_input(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if not _is_stdio(path) and path.suffix.lower() not in STEP_SUFFIXES:
        _fail(ctx, payload, f"Unsupported STEP extension: {path.suffix or '<none>'}. Use .step or .stp.", code=2)


def _validate_usd_output(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if not _is_stdio(path) and path.suffix.lower() not in USD_SUFFIXES:
        _fail(ctx, payload, f"Unsupported USD extension: {path.suffix or '<none>'}. Use .usd, .usda, or .usdc.", code=2)


def _is_stdio(path: Path) -> bool:
    return str(path) == "-"


def _fail(ctx: typer.Context, payload: dict[str, Any], message: str, code: int = 1) -> NoReturn:
    if _state(ctx).json_output:
        out.print_json(json.dumps({**payload, "error": message}))
    else:
        err.print(message)
    raise typer.Exit(code)


def _read_step_for_cli(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> Any:
    if _is_stdio(path):
        data = sys.stdin.buffer.read()
        if not data:
            _fail(ctx, payload, "Missing input data on stdin.")
        return read_step_bytes(data)
    _require_existing_file(path, "input", ctx, payload)
    try:
        return read_step(path)
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc


def _convert_for_cli(
    input_path: Path,
    output_path: Path,
    *,
    profile: str,
    tessellation: Tessellation,
    stage: StageOptions,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    progress: Callable[[str, dict[str, int]], None] | None,
    debug: bool,
) -> Any:
    if _is_stdio(input_path):
        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing input data on stdin.")
        with _temporary_step_file(data) as temp_input:
            return _convert_output(
                temp_input, output_path, profile, tessellation, stage, optimize, lods, progress, debug
            )
    return _convert_output(input_path, output_path, profile, tessellation, stage, optimize, lods, progress, debug)


def _convert_output(
    input_path: Path,
    output_path: Path,
    profile: str,
    tessellation: Tessellation,
    stage: StageOptions,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    progress: Callable[[str, dict[str, int]], None] | None,
    debug: bool,
) -> Any:
    if _is_stdio(output_path):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".usda") as handle:
            asset = convert(
                input_path,
                handle.name,
                profile=profile,
                tessellation=tessellation,
                stage=stage,
                optimize=optimize,
                lods=lods,
                progress=progress,
                debug=debug,
            )
            handle.seek(0)
            sys.stdout.buffer.write(handle.read())
            return asset
    return convert(
        input_path,
        output_path,
        profile=profile,
        tessellation=tessellation,
        stage=stage,
        optimize=optimize,
        lods=lods,
        progress=progress,
        debug=debug,
    )


def _progress_callback(ctx: typer.Context, output_path: Path) -> Callable[[str, dict[str, int]], None] | None:
    state = _state(ctx)
    if state.quiet or state.json_output or _is_stdio(output_path):
        return None

    def progress(step: str, stats: dict[str, int]) -> None:
        err.print(f"{step}: {_format_stats(stats)}")

    return progress


class _temporary_step_file:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.path: Path | None = None
        self._handle: Any = None

    def __enter__(self) -> Path:
        import tempfile

        self._handle = tempfile.NamedTemporaryFile(suffix=".step")
        self._handle.write(self.data)
        self._handle.flush()
        self.path = Path(self._handle.name)
        return self.path

    def __exit__(self, *_exc_info: object) -> None:
        if self._handle is not None:
            self._handle.close()


def _validate_usd_for_cli(path: Path) -> dict[str, int]:
    if _is_stdio(path):
        import tempfile

        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing USD data on stdin.")
        with tempfile.NamedTemporaryFile(suffix=".usda") as handle:
            handle.write(data)
            handle.flush()
            return validate_usd(handle.name)
    return validate_usd(path)


def _format_stats(stats: dict[str, int]) -> str:
    parts = []
    for key in ("parts", "occurrences", "materials", "meshes", "vertices", "points", "triangles"):
        if key in stats:
            parts.append(f"{stats[key]} {key}")
    return ", ".join(parts) if parts else json.dumps(stats, sort_keys=True)


def _normalize_args(args: Sequence[str]) -> list[str]:
    raw_args = list(args)
    if any(arg in VERSION_FLAGS for arg in raw_args):
        return ["--version"]

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
    err.print(message)
    err.print("Run 'fascat --help' to see available commands.")


def _color_disabled_requested(args: Sequence[str]) -> bool:
    return (
        "--no-color" in args
        or bool(os.environ.get("NO_COLOR"))
        or os.environ.get("TERM") == "dumb"
        or not sys.stdout.isatty()
        or not sys.stderr.isatty()
    )


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
