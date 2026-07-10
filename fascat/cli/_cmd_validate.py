from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, cast

import typer

from ._app import DOCS_URL, _state, app
from ._enums import RuntimeEngineMode
from ._io_helpers import _validate_and_analyze_output_for_cli
from ._output import _emit, _export_label, _fail, _format_stats, _is_stdio, _require_existing_file
from ._params import _analysis_requested, _analyze_options, _parse_filter_options, _validate_export_output


@app.command(
    "validate",
    epilog=f"""Examples:
  fascat validate motor.usdc
  fascat validate motor.glb
  fascat validate motor.glb --geometry-quality --report report.json
  fascat validate motor.glb --visual-preview preview.png --lod-preview-dir previews/
  fascat validate motor.glb --turntable-dir views/ --turntable-views 8 --turntable-elevations -30,30
  fascat validate motor.glb --turntable-dir views/ --turntable-baseline-dir reference-views/
  fascat validate motor.glb --filter 'material=painted' --geometry-quality
  fascat --json validate motor.usda
  cat motor.usdc | fascat validate -

Docs: {DOCS_URL}/reference.html""",
)
def cmd_validate(
    ctx: typer.Context,
    output_path: Annotated[
        Path,
        typer.Argument(
            help="Generated USD, glTF, OBJ, or STL file to validate, or '-' for USD stdin.", allow_dash=True
        ),
    ],
    geometry_quality: Annotated[
        bool,
        typer.Option("--geometry-quality", help="Enable all geometry quality checks in the validation report."),
    ] = False,
    non_manifold_edges: Annotated[
        bool,
        typer.Option("--non-manifold-edges", help="Report non-manifold edge counts."),
    ] = False,
    open_boundaries: Annotated[
        bool,
        typer.Option("--open-boundaries", help="Report open boundary counts."),
    ] = False,
    self_intersections: Annotated[
        bool,
        typer.Option("--self-intersections", help="Report detected self-intersections with bounded triangle checks."),
    ] = False,
    sliver_triangles: Annotated[
        bool,
        typer.Option("--sliver-triangles", help="Report degenerate and sliver triangle stats."),
    ] = False,
    tiny_parts: Annotated[
        bool,
        typer.Option("--tiny-parts", help="Report tiny part stats."),
    ] = False,
    draw_call_estimate: Annotated[
        bool,
        typer.Option("--draw-call-estimate", help="Report material count and draw-call estimate."),
    ] = False,
    visual_risk: Annotated[
        bool,
        typer.Option("--visual-risk", help="Report before/after visual risk warnings."),
    ] = False,
    runtime_browser: Annotated[
        bool,
        typer.Option("--runtime-browser", help="Run optional headless browser/WebGL runtime measurement for glTF/GLB."),
    ] = False,
    runtime_browser_command: Annotated[
        str | None,
        typer.Option("--runtime-browser-command", help="Browser executable to use for --runtime-browser."),
    ] = None,
    runtime_duration: Annotated[
        float,
        typer.Option("--runtime-duration", help="Browser runtime FPS measurement duration in seconds."),
    ] = 2.0,
    runtime_timeout: Annotated[
        float,
        typer.Option("--runtime-timeout", help="Browser runtime validation timeout in seconds."),
    ] = 15.0,
    runtime_engine: Annotated[
        RuntimeEngineMode | None,
        typer.Option("--runtime-engine", help="Run optional Unity or Unreal runtime harness measurement."),
    ] = None,
    runtime_engine_command: Annotated[
        str | None,
        typer.Option("--runtime-engine-command", help="Unity or Unreal executable for --runtime-engine."),
    ] = None,
    runtime_engine_project: Annotated[
        Path | None,
        typer.Option(
            "--runtime-engine-project",
            help="Custom Unity project folder or Unreal .uproject with Fascat harness.",
        ),
    ] = None,
    runtime_engine_preview: Annotated[
        Path | None,
        typer.Option(
            "--runtime-engine-preview",
            help="Preview PNG path requested from a Unity/Unreal runtime harness.",
        ),
    ] = None,
    runtime_engine_baseline: Annotated[
        Path | None,
        typer.Option(
            "--runtime-engine-baseline",
            help="Compare --runtime-engine-preview against this baseline PNG.",
        ),
    ] = None,
    runtime_engine_timeout: Annotated[
        float,
        typer.Option("--runtime-engine-timeout", help="Unity/Unreal runtime harness timeout in seconds."),
    ] = 120.0,
    visual_preview: Annotated[
        Path | None,
        typer.Option("--visual-preview", help="Write a stable software preview PNG during validation."),
    ] = None,
    runtime_browser_preview: Annotated[
        Path | None,
        typer.Option("--runtime-browser-preview", help="Write a browser/WebGL-rendered preview PNG during validation."),
    ] = None,
    visual_baseline: Annotated[
        Path | None,
        typer.Option("--visual-baseline", help="Compare --visual-preview against this baseline PNG."),
    ] = None,
    visual_diff_pixel_tolerance: Annotated[
        int,
        typer.Option("--visual-diff-pixel-tolerance", help="Per-channel byte tolerance for visual baseline diff."),
    ] = 0,
    visual_diff_mean_threshold: Annotated[
        float,
        typer.Option("--visual-diff-mean-threshold", help="Maximum allowed visual diff mean absolute error."),
    ] = 0.0,
    visual_diff_changed_pixel_ratio: Annotated[
        float,
        typer.Option(
            "--visual-diff-changed-pixel-ratio",
            help="Maximum allowed ratio of pixels that exceed the visual diff tolerance.",
        ),
    ] = 0.0,
    lod_preview_dir: Annotated[
        Path | None,
        typer.Option("--lod-preview-dir", help="Write LOD switching preview PNGs and a contact sheet."),
    ] = None,
    turntable_dir: Annotated[
        Path | None,
        typer.Option("--turntable-dir", help="Write multi-angle turntable preview PNGs and a contact sheet."),
    ] = None,
    turntable_views: Annotated[
        int,
        typer.Option("--turntable-views", help="Number of turntable azimuth views per elevation."),
    ] = 8,
    turntable_elevations: Annotated[
        str,
        typer.Option("--turntable-elevations", help="Comma-separated turntable camera elevations in degrees."),
    ] = "-30,30",
    turntable_baseline_dir: Annotated[
        Path | None,
        typer.Option(
            "--turntable-baseline-dir",
            help="Compare each turntable view against same-named PNGs in this directory.",
        ),
    ] = None,
    turntable_width: Annotated[
        int,
        typer.Option("--turntable-width", help="Turntable preview image width in pixels."),
    ] = 512,
    turntable_height: Annotated[
        int,
        typer.Option("--turntable-height", help="Turntable preview image height in pixels."),
    ] = 512,
    turntable_supersample: Annotated[
        int,
        typer.Option("--turntable-supersample", help="Turntable preview supersampling factor."),
    ] = 2,
    filters: Annotated[
        list[str] | None,
        typer.Option("--filter", help="Scope validation-time analysis with selectors such as path=*/Fasteners/*."),
    ] = None,
    exclude_filters: Annotated[
        list[str] | None,
        typer.Option("--exclude-filter", help="Exclude selector matches from --filter results."),
    ] = None,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Write validation and geometry quality report as JSON."),
    ] = None,
) -> None:
    from fascat.runtime import (
        RuntimeBrowserOptions,
        RuntimeBrowserRenderOptions,
        RuntimeEngineOptions,
        measure_browser_runtime,
        measure_engine_runtime,
        write_browser_render_preview,
    )
    from fascat.visual import (
        TurntableOptions,
        VisualDiffOptions,
        VisualPreviewOptions,
        compare_images,
        write_output_lod_switch_previews,
        write_output_preview,
        write_output_turntable_previews,
    )

    """Validate a generated USD, glTF, OBJ, or STL file."""
    state = _state(ctx)
    analyze_options = _analyze_options(
        geometry_quality=geometry_quality,
        non_manifold_edges=non_manifold_edges,
        open_boundaries=open_boundaries,
        self_intersections=self_intersections,
        sliver_triangles=sliver_triangles,
        tiny_parts=tiny_parts,
        draw_call_estimate=draw_call_estimate,
        visual_risk=visual_risk,
    )
    should_analyze = report is not None or _analysis_requested(analyze_options)
    payload = {
        "command": "validate",
        "output": str(output_path),
        "dry_run": state.dry_run,
        "geometry_quality": geometry_quality,
        "runtime_browser": runtime_browser,
        "runtime_browser_command": runtime_browser_command,
        "runtime_duration": runtime_duration,
        "runtime_timeout": runtime_timeout,
        "runtime_engine": None if runtime_engine is None else runtime_engine.value,
        "runtime_engine_command": runtime_engine_command,
        "runtime_engine_project": str(runtime_engine_project) if runtime_engine_project else None,
        "runtime_engine_preview": str(runtime_engine_preview) if runtime_engine_preview else None,
        "runtime_engine_baseline": str(runtime_engine_baseline) if runtime_engine_baseline else None,
        "runtime_engine_timeout": runtime_engine_timeout,
        "visual_preview": str(visual_preview) if visual_preview else None,
        "runtime_browser_preview": str(runtime_browser_preview) if runtime_browser_preview else None,
        "visual_baseline": str(visual_baseline) if visual_baseline else None,
        "visual_diff_pixel_tolerance": visual_diff_pixel_tolerance,
        "visual_diff_mean_threshold": visual_diff_mean_threshold,
        "visual_diff_changed_pixel_ratio": visual_diff_changed_pixel_ratio,
        "lod_preview_dir": str(lod_preview_dir) if lod_preview_dir else None,
        "turntable_dir": str(turntable_dir) if turntable_dir else None,
        "turntable_views": turntable_views,
        "turntable_elevations": turntable_elevations,
        "turntable_baseline_dir": str(turntable_baseline_dir) if turntable_baseline_dir else None,
        "turntable_width": turntable_width,
        "turntable_height": turntable_height,
        "turntable_supersample": turntable_supersample,
        "analysis_options": analyze_options.to_dict() if should_analyze else None,
        "filters": filters or [],
        "exclude_filters": exclude_filters or [],
        "report": str(report) if report else None,
    }
    where = _parse_filter_options(filters, exclude_filters, ctx, payload)
    should_analyze = should_analyze or where is not None
    payload["analysis_options"] = analyze_options.to_dict() if should_analyze else None
    _validate_export_output(output_path, ctx, payload)
    if runtime_duration <= 0.0:
        _fail(ctx, payload, "--runtime-duration must be greater than 0.", code=2)
    if runtime_timeout <= 0.0:
        _fail(ctx, payload, "--runtime-timeout must be greater than 0.", code=2)
    if runtime_engine_timeout <= 0.0:
        _fail(ctx, payload, "--runtime-engine-timeout must be greater than 0.", code=2)
    if visual_diff_pixel_tolerance < 0 or visual_diff_pixel_tolerance > 255:
        _fail(ctx, payload, "--visual-diff-pixel-tolerance must be between 0 and 255.", code=2)
    if visual_diff_mean_threshold < 0.0 or visual_diff_mean_threshold > 255.0:
        _fail(ctx, payload, "--visual-diff-mean-threshold must be between 0 and 255.", code=2)
    if visual_diff_changed_pixel_ratio < 0.0 or visual_diff_changed_pixel_ratio > 1.0:
        _fail(ctx, payload, "--visual-diff-changed-pixel-ratio must be between 0 and 1.", code=2)
    if visual_baseline is not None and visual_preview is None:
        _fail(ctx, payload, "--visual-baseline requires --visual-preview.", code=2)
    if turntable_views < 1:
        _fail(ctx, payload, "--turntable-views must be greater than 0.", code=2)
    parsed_turntable_elevations: tuple[float, ...] = ()
    try:
        parsed_turntable_elevations = tuple(
            float(value.strip()) for value in turntable_elevations.split(",") if value.strip()
        )
    except ValueError:
        _fail(ctx, payload, "--turntable-elevations must be comma-separated numbers.", code=2)
    if not parsed_turntable_elevations:
        _fail(ctx, payload, "--turntable-elevations must contain at least one value.", code=2)
    if any(value < -90.0 or value > 90.0 for value in parsed_turntable_elevations):
        _fail(ctx, payload, "--turntable-elevations values must be between -90 and 90.", code=2)
    if turntable_baseline_dir is not None and turntable_dir is None:
        _fail(ctx, payload, "--turntable-baseline-dir requires --turntable-dir.", code=2)
    if turntable_width <= 0 or turntable_height <= 0:
        _fail(ctx, payload, "--turntable-width and --turntable-height must be greater than 0.", code=2)
    if turntable_supersample < 1:
        _fail(ctx, payload, "--turntable-supersample must be greater than 0.", code=2)
    if runtime_engine_preview is not None and runtime_engine is None:
        _fail(ctx, payload, "--runtime-engine-preview requires --runtime-engine.", code=2)
    if runtime_engine_baseline is not None and runtime_engine_preview is None:
        _fail(ctx, payload, "--runtime-engine-baseline requires --runtime-engine-preview.", code=2)
    if _is_stdio(output_path) and (
        visual_preview is not None
        or runtime_browser_preview is not None
        or runtime_engine_preview is not None
        or lod_preview_dir is not None
        or turntable_dir is not None
    ):
        _fail(ctx, payload, "Visual preview validation requires a file path, not stdin.", code=2)
    if state.dry_run:
        _emit(ctx, payload, f"Would validate {output_path}.")
        return

    _require_existing_file(output_path, "output", ctx, payload)
    try:
        stats, analysis = _validate_and_analyze_output_for_cli(
            output_path,
            analyze_options if should_analyze else None,
            where=where,
        )
        runtime_report = (
            measure_browser_runtime(
                output_path,
                RuntimeBrowserOptions(
                    browser=runtime_browser_command,
                    duration_seconds=runtime_duration,
                    timeout_seconds=runtime_timeout,
                ),
            )
            if runtime_browser
            else None
        )
        runtime_engine_report = (
            measure_engine_runtime(
                output_path,
                RuntimeEngineOptions(
                    engine=cast(Any, runtime_engine.value),
                    executable=runtime_engine_command,
                    project=runtime_engine_project,
                    preview_path=runtime_engine_preview,
                    timeout_seconds=runtime_engine_timeout,
                ),
            )
            if runtime_engine is not None
            else None
        )
        runtime_browser_preview_report = (
            write_browser_render_preview(
                output_path,
                runtime_browser_preview,
                RuntimeBrowserRenderOptions(
                    browser=runtime_browser_command,
                    timeout_seconds=runtime_timeout,
                ),
            )
            if runtime_browser_preview is not None
            else None
        )
        visual_preview_report = (
            write_output_preview(output_path, visual_preview) if visual_preview is not None else None
        )
        visual_diff_report = (
            compare_images(
                visual_baseline,
                visual_preview,
                VisualDiffOptions(
                    pixel_tolerance=visual_diff_pixel_tolerance,
                    max_mean_absolute_error=visual_diff_mean_threshold,
                    max_changed_pixel_ratio=visual_diff_changed_pixel_ratio,
                ),
            )
            if visual_baseline is not None and visual_preview is not None
            else None
        )
        runtime_engine_diff_report = None
        runtime_engine_diff_error = None
        if runtime_engine_baseline is not None:
            if runtime_engine_report is None:
                runtime_engine_diff_error = "--runtime-engine-baseline requires --runtime-engine."
            elif runtime_engine_report.render_status != "rendered" or runtime_engine_report.preview_path is None:
                runtime_engine_diff_error = (
                    "runtime engine baseline requires a rendered engine preview"
                    f"; render_status={runtime_engine_report.render_status}"
                )
            else:
                runtime_engine_diff_report = compare_images(
                    runtime_engine_baseline,
                    runtime_engine_report.preview_path,
                    VisualDiffOptions(
                        pixel_tolerance=visual_diff_pixel_tolerance,
                        max_mean_absolute_error=visual_diff_mean_threshold,
                        max_changed_pixel_ratio=visual_diff_changed_pixel_ratio,
                    ),
                )
        lod_preview_report = (
            write_output_lod_switch_previews(output_path, lod_preview_dir) if lod_preview_dir is not None else None
        )
        turntable_report = (
            write_output_turntable_previews(
                output_path,
                turntable_dir,
                VisualPreviewOptions(
                    width=turntable_width,
                    height=turntable_height,
                    supersample=turntable_supersample,
                ),
                TurntableOptions(views=turntable_views, elevations=parsed_turntable_elevations),
                baseline_dir=turntable_baseline_dir,
                diff_options=VisualDiffOptions(
                    pixel_tolerance=visual_diff_pixel_tolerance,
                    max_mean_absolute_error=visual_diff_mean_threshold,
                    max_changed_pixel_ratio=visual_diff_changed_pixel_ratio,
                ),
            )
            if turntable_dir is not None
            else None
        )
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc
    if report is not None and analysis is not None:
        analysis.write_json(report)
    json_payload = {**payload, "stats": stats}
    if analysis is not None:
        json_payload["analysis"] = analysis.to_dict()
    if runtime_report is not None:
        json_payload["runtime_browser"] = runtime_report.to_dict()
    if runtime_engine_report is not None:
        json_payload["runtime_engine"] = runtime_engine_report.to_dict()
    if runtime_browser_preview_report is not None:
        json_payload["runtime_browser_preview"] = runtime_browser_preview_report.to_dict()
    if visual_preview_report is not None:
        json_payload["visual_preview"] = visual_preview_report.to_dict()
    if visual_diff_report is not None:
        json_payload["visual_diff"] = visual_diff_report.to_dict()
    if runtime_engine_diff_report is not None:
        json_payload["runtime_engine_diff"] = runtime_engine_diff_report.to_dict()
    if runtime_engine_diff_error is not None:
        json_payload["runtime_engine_diff_error"] = runtime_engine_diff_error
    if lod_preview_report is not None:
        json_payload["lod_preview"] = lod_preview_report.to_dict()
    if turntable_report is not None:
        json_payload["turntable"] = turntable_report.to_dict()
    message = f"{output_path}: valid {_export_label(output_path)}, {_format_stats(stats)}."
    if report is not None:
        message = f"{message} Wrote report {report}."
    if visual_preview_report is not None:
        message = f"{message} Wrote preview {visual_preview_report.path}."
    if runtime_browser_preview_report is not None:
        if runtime_browser_preview_report.status in {"rendered", "rendered_partial"}:
            message = f"{message} Wrote browser preview {runtime_browser_preview_report.preview_path}."
            if runtime_browser_preview_report.status == "rendered_partial":
                message = f"{message} Browser preview partial: {runtime_browser_preview_report.error}."
        else:
            message = (
                f"{message} Browser preview {runtime_browser_preview_report.status}: "
                f"{runtime_browser_preview_report.error}."
            )
    if runtime_engine_report is not None and runtime_engine_report.preview_path is not None:
        if runtime_engine_report.render_status in {"rendered", "rendered_partial"}:
            message = f"{message} Wrote engine preview {runtime_engine_report.preview_path}."
            if runtime_engine_report.render_status == "rendered_partial":
                detail = runtime_engine_report.render_error or "; ".join(runtime_engine_report.render_limitations)
                message = f"{message} Engine preview partial: {detail}."
        else:
            message = (
                f"{message} Engine preview {runtime_engine_report.render_status}: {runtime_engine_report.render_error}."
            )
    if visual_diff_report is not None:
        message = f"{message} Visual diff passed." if visual_diff_report.passed else f"{message} Visual diff failed."
    if runtime_engine_diff_report is not None:
        message = (
            f"{message} Engine preview diff passed."
            if runtime_engine_diff_report.passed
            else f"{message} Engine preview diff failed."
        )
    if runtime_engine_diff_error is not None:
        message = f"{message} Engine preview diff failed: {runtime_engine_diff_error}."
    if lod_preview_report is not None:
        message = f"{message} Wrote LOD previews {lod_preview_report.directory}."
    if turntable_report is not None:
        message = (
            f"{message} Wrote turntable previews {turntable_report.directory} ({len(turntable_report.views)} views)."
        )
        if turntable_report.diff_passed is True:
            message = f"{message} Turntable diff passed."
        elif turntable_report.diff_passed is False:
            worst = turntable_report.worst_diff_view()
            worst_label = f" (worst view {worst.name})" if worst is not None else ""
            message = f"{message} Turntable diff failed{worst_label}."
    if analysis is not None and "selection" in analysis.summary:
        selection = cast(dict[str, Any], analysis.summary["selection"])
        message = f"{message} Matched {_format_stats(cast(dict[str, int], selection['stats']))}."
    if runtime_report is not None:
        if runtime_report.status == "measured" and runtime_report.measured_fps is not None:
            message = f"{message} Browser runtime measured {runtime_report.measured_fps:.1f} FPS."
        else:
            message = f"{message} Browser runtime {runtime_report.status}: {runtime_report.error}."
    if visual_diff_report is not None and not visual_diff_report.passed:
        _emit(ctx, json_payload, message)
        raise typer.Exit(1)
    if runtime_engine_diff_error is not None or (
        runtime_engine_diff_report is not None and not runtime_engine_diff_report.passed
    ):
        _emit(ctx, json_payload, message)
        raise typer.Exit(1)
    if turntable_report is not None and turntable_report.diff_passed is False:
        _emit(ctx, json_payload, message)
        raise typer.Exit(1)
    _emit(
        ctx,
        json_payload,
        message,
    )
