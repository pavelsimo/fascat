from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import typer

from ._app import DOCS_URL, _state, app
from ._enums import RuntimeParityCaptureMode
from ._output import _emit, _fail


@app.command(
    "runtime-fixtures",
    epilog=f"""Examples:
  fascat runtime-fixtures runtime-parity/
  fascat --json runtime-fixtures runtime-parity/

Docs: {DOCS_URL}/reference.html""",
)
def cmd_runtime_fixtures(
    ctx: typer.Context,
    output_dir: Annotated[Path, typer.Argument(help="Directory to write runtime parity GLBs and baselines.")],
    capture: Annotated[
        list[RuntimeParityCaptureMode] | None,
        typer.Option("--capture", help="Capture previews after writing fixtures: browser, unity, or unreal."),
    ] = None,
    runtime_browser_command: Annotated[
        str | None,
        typer.Option("--runtime-browser-command", help="Browser executable to use for --capture browser."),
    ] = None,
    unity_command: Annotated[
        str | None,
        typer.Option("--unity-command", help="Unity executable to use for --capture unity."),
    ] = None,
    unreal_command: Annotated[
        str | None,
        typer.Option("--unreal-command", help="Unreal executable to use for --capture unreal."),
    ] = None,
    unity_project: Annotated[
        Path | None,
        typer.Option("--unity-project", help="Optional Unity harness project for --capture unity."),
    ] = None,
    unreal_project: Annotated[
        Path | None,
        typer.Option("--unreal-project", help="Optional Unreal .uproject for --capture unreal."),
    ] = None,
    runtime_engine_timeout: Annotated[
        float,
        typer.Option("--runtime-engine-timeout", help="Unity/Unreal parity capture timeout in seconds."),
    ] = 120.0,
    promote_goldens: Annotated[
        bool,
        typer.Option(
            "--promote-goldens/--no-promote-goldens",
            help="Copy rendered parity captures into goldens/<target>/ for review or future baselines.",
        ),
    ] = False,
    require_goldens: Annotated[
        bool,
        typer.Option(
            "--require-goldens/--no-require-goldens",
            help="Require existing goldens/<target>/<fixture>.png files for captured or checked parity targets.",
        ),
    ] = False,
    check_goldens: Annotated[
        bool,
        typer.Option(
            "--check-goldens/--no-check-goldens",
            help="Audit existing goldens/<target>/<fixture>.png coverage after writing fixtures.",
        ),
    ] = False,
) -> None:
    from fascat.runtime_fixtures import (
        audit_runtime_parity_goldens,
        capture_runtime_parity_suite,
        write_runtime_parity_suite,
    )

    """Write bundled runtime parity fixtures for browser, Unity, and Unreal preview checks."""
    state = _state(ctx)
    capture_targets = tuple(item.value for item in capture or [])
    payload: dict[str, Any] = {
        "command": "runtime-fixtures",
        "output": str(output_dir),
        "dry_run": state.dry_run,
        "capture": list(capture_targets),
        "runtime_browser_command": runtime_browser_command,
        "unity_command": unity_command,
        "unreal_command": unreal_command,
        "unity_project": str(unity_project) if unity_project else None,
        "unreal_project": str(unreal_project) if unreal_project else None,
        "runtime_engine_timeout": runtime_engine_timeout,
        "promote_goldens": promote_goldens,
        "require_goldens": require_goldens,
        "check_goldens": check_goldens,
    }
    if runtime_engine_timeout <= 0.0:
        _fail(ctx, payload, "--runtime-engine-timeout must be greater than 0.", code=2)
    if state.dry_run:
        action = "write and capture" if capture_targets else "write"
        _emit(ctx, payload, f"Would {action} runtime parity fixtures in {output_dir}.")
        return

    try:
        suite = write_runtime_parity_suite(output_dir)
        capture_report = (
            capture_runtime_parity_suite(
                output_dir,
                targets=cast(Any, capture_targets),
                browser_command=runtime_browser_command,
                unity_command=unity_command,
                unreal_command=unreal_command,
                unity_project=unity_project,
                unreal_project=unreal_project,
                engine_timeout_seconds=runtime_engine_timeout,
                promote_goldens=promote_goldens,
                require_goldens=require_goldens,
            )
            if capture_targets
            else None
        )
        golden_coverage = (
            audit_runtime_parity_goldens(
                output_dir,
                targets=cast(Any, capture_targets or ("browser", "unity", "unreal")),
            )
            if check_goldens
            else None
        )
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc

    json_payload = {**payload, "suite": suite.to_dict()}
    if capture_report is not None:
        json_payload["captures"] = capture_report.to_dict()
        if require_goldens and not capture_report.passed:
            _fail(ctx, json_payload, "runtime parity target captures failed required golden validation.")
    if golden_coverage is not None:
        json_payload["golden_coverage"] = golden_coverage.to_dict()
        if require_goldens and not golden_coverage.passed:
            _fail(ctx, json_payload, "runtime parity target goldens are missing or invalid.")
    message = f"Wrote runtime parity fixtures to {suite.directory} ({len(suite.fixtures)} fixtures)."
    if capture_report is not None:
        message = f"{message} Captured {len(capture_report.captures)} preview(s)."
    if golden_coverage is not None:
        message = (
            f"{message} Golden coverage: {golden_coverage.present_count} present, "
            f"{golden_coverage.missing_count} missing, {golden_coverage.invalid_count} invalid."
        )
    _emit(
        ctx,
        json_payload,
        message,
    )
