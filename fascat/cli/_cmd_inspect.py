from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from . import _io_helpers
from ._app import DOCS_URL, _state, app
from ._enums import (
    AxisMode,
    ConstructionCurvePolicyMode,
    HandednessMode,
    MaterialLibraryColorSpaceMode,
    MetadataMode,
    PmiMode,
    Profile,
)
from ._io_helpers import by_name
from ._output import _emit, _fail, _format_stats
from ._params import (
    _brep_heal_options,
    _metadata_summary,
    _parse_filter_options,
    _pmi_summary,
    _step_read_options,
    _validate_cad_input,
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
    input_path: Annotated[Path, typer.Argument(help="CAD file to inspect, or '-' for stdin STEP.", allow_dash=True)],
    profile: Annotated[Profile, typer.Option("--profile", help="Inspection profile to apply.")] = Profile.INSPECT_ONLY,
    metadata: Annotated[
        MetadataMode,
        typer.Option("--metadata", help="Metadata output mode: none, summary, or full."),
    ] = MetadataMode.SUMMARY,
    pmi: Annotated[
        PmiMode,
        typer.Option("--pmi", help="PMI output mode: none, summary, full, metadata, or metadata-and-visuals."),
    ] = PmiMode.SUMMARY,
    design_variants: Annotated[
        bool,
        typer.Option("--design-variants/--no-design-variants", help="Request STEP design variant import."),
    ] = False,
    design_variant_selection: Annotated[
        list[str] | None,
        typer.Option(
            "--design-variant",
            help="STEP design variant label, record id, or referenced label to select. Can be repeated.",
        ),
    ] = None,
    import_existing_meshes: Annotated[
        bool,
        typer.Option(
            "--import-existing-meshes/--no-import-existing-meshes",
            help="Prefer existing STEP tessellation payloads when the importer exposes them.",
        ),
    ] = True,
    multi_file_import: Annotated[
        bool,
        typer.Option(
            "--multi-file-import/--single-file-import",
            help="Resolve quoted external STEP references from a master STEP file.",
        ),
    ] = False,
    material_libraries: Annotated[
        list[Path] | None,
        typer.Option(
            "--material-library", help="Vendor material-library JSON/MTL/ZIP file or folder to apply on import."
        ),
    ] = None,
    material_library_color_space: Annotated[
        MaterialLibraryColorSpaceMode,
        typer.Option(
            "--material-library-color-space",
            help="Numeric material-library color interpretation: auto, linear, or srgb255.",
        ),
    ] = MaterialLibraryColorSpaceMode.AUTO,
    delete_free_vertices: Annotated[
        bool,
        typer.Option(
            "--delete-free-vertices/--keep-free-vertices",
            help="Drop construction-only point shapes during STEP import.",
        ),
    ] = False,
    delete_lines: Annotated[
        bool,
        typer.Option("--delete-lines/--keep-lines", help="Drop construction-only line shapes during STEP import."),
    ] = False,
    construction_curve_policy: Annotated[
        ConstructionCurvePolicyMode,
        typer.Option(
            "--construction-curve-policy",
            help="Construction-only line policy: preserve-metadata, delete, or tessellate-tubes.",
        ),
    ] = ConstructionCurvePolicyMode.PRESERVE_METADATA,
    construction_curve_tube_radius: Annotated[
        float,
        typer.Option(
            "--construction-curve-tube-radius",
            help="Tube radius in source units when --construction-curve-policy=tessellate-tubes.",
        ),
    ] = 0.01,
    source_units: Annotated[
        str | None,
        typer.Option("--source-units", help="Override source STEP units for normalization, for example millimetre."),
    ] = None,
    source_meters_per_unit: Annotated[
        float | None,
        typer.Option("--source-meters-per-unit", help="Override source meters-per-unit for normalization."),
    ] = None,
    source_up_axis: Annotated[
        AxisMode,
        typer.Option("--source-up-axis", help="Declared source up axis: Y or Z."),
    ] = AxisMode.Z,
    source_handedness: Annotated[
        HandednessMode,
        typer.Option("--source-handedness", help="Declared source handedness: right or left."),
    ] = HandednessMode.RIGHT,
    target_units: Annotated[
        str | None,
        typer.Option("--target-units", help="Normalize asset units to this unit, for example metre."),
    ] = None,
    target_meters_per_unit: Annotated[
        float | None,
        typer.Option("--target-meters-per-unit", help="Normalize asset units to this meters-per-unit value."),
    ] = None,
    target_up_axis: Annotated[
        AxisMode | None,
        typer.Option("--target-up-axis", help="Normalize asset up axis to Y or Z."),
    ] = None,
    target_handedness: Annotated[
        HandednessMode | None,
        typer.Option("--target-handedness", help="Normalize asset handedness to right or left."),
    ] = None,
    heal_brep: Annotated[bool, typer.Option("--heal-brep", help="Run BREP healing before inspection output.")] = False,
    heal_tolerance: Annotated[float, typer.Option("--heal-tolerance", help="BREP healing tolerance.")] = 0.05,
    group_open_shells: Annotated[
        bool,
        typer.Option(
            "--group-open-shells/--no-group-open-shells",
            help="Group disconnected open BREP shells before healing.",
        ),
    ] = True,
    cleanup_overlapping_faces: Annotated[
        bool,
        typer.Option(
            "--cleanup-overlapping-faces/--keep-overlapping-faces",
            help="Remove redundant coplanar BREP faces that overlap enough to z-fight.",
        ),
    ] = True,
    overlap_area_ratio: Annotated[
        float,
        typer.Option("--overlap-area-ratio", help="Minimum smaller-face area ratio for BREP overlap cleanup."),
    ] = 0.995,
    remove_sliver_faces: Annotated[
        bool,
        typer.Option("--remove-sliver-faces", help="Request tiny sliver-face removal during BREP healing."),
    ] = False,
    max_sliver_area: Annotated[
        float,
        typer.Option("--max-sliver-area", help="Area threshold for sliver-face reporting."),
    ] = 1e-4,
    filters: Annotated[
        list[str] | None,
        typer.Option("--filter", help="Scope inspection with a selector such as path=*/Fasteners/* or triangles<=12."),
    ] = None,
    exclude_filters: Annotated[
        list[str] | None,
        typer.Option("--exclude-filter", help="Exclude selector matches from --filter results."),
    ] = None,
) -> None:
    """Inspect CAD assembly metadata and planned conversion inputs."""
    state = _state(ctx)
    payload = {
        "command": "inspect",
        "input": str(input_path),
        "profile": profile.value,
        "metadata": metadata.value,
        "pmi": pmi.value,
        "design_variants": design_variants,
        "design_variant_selection": design_variant_selection or [],
        "import_existing_meshes": import_existing_meshes,
        "multi_file_import": multi_file_import,
        "material_libraries": [str(path) for path in material_libraries or []],
        "material_library_color_space": material_library_color_space.value,
        "delete_free_vertices": delete_free_vertices,
        "delete_lines": delete_lines,
        "construction_curve_policy": construction_curve_policy.value,
        "construction_curve_tube_radius": construction_curve_tube_radius,
        "source_units": source_units,
        "source_meters_per_unit": source_meters_per_unit,
        "source_up_axis": source_up_axis.value,
        "source_handedness": source_handedness.value,
        "target_units": target_units,
        "target_meters_per_unit": target_meters_per_unit,
        "target_up_axis": None if target_up_axis is None else target_up_axis.value,
        "target_handedness": None if target_handedness is None else target_handedness.value,
        "heal_brep": heal_brep,
        "heal_tolerance": heal_tolerance,
        "group_open_shells": group_open_shells,
        "cleanup_overlapping_faces": cleanup_overlapping_faces,
        "overlap_area_ratio": overlap_area_ratio,
        "remove_sliver_faces": remove_sliver_faces,
        "max_sliver_area": max_sliver_area,
        "filters": filters or [],
        "exclude_filters": exclude_filters or [],
        "dry_run": state.dry_run,
    }
    where = _parse_filter_options(filters, exclude_filters, ctx, payload)
    if heal_tolerance <= 0.0:
        _fail(ctx, payload, "--heal-tolerance must be greater than 0.", code=2)
    if overlap_area_ratio <= 0.0 or overlap_area_ratio > 1.0:
        _fail(ctx, payload, "--overlap-area-ratio must be greater than 0 and no more than 1.", code=2)
    if max_sliver_area < 0.0:
        _fail(ctx, payload, "--max-sliver-area must be greater than or equal to 0.", code=2)
    if source_meters_per_unit is not None and source_meters_per_unit <= 0.0:
        _fail(ctx, payload, "--source-meters-per-unit must be greater than 0.", code=2)
    if target_meters_per_unit is not None and target_meters_per_unit <= 0.0:
        _fail(ctx, payload, "--target-meters-per-unit must be greater than 0.", code=2)
    if construction_curve_tube_radius <= 0.0:
        _fail(ctx, payload, "--construction-curve-tube-radius must be greater than 0.", code=2)
    _validate_cad_input(input_path, ctx, payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would inspect {input_path} with profile {profile.value}.")
        return

    import_options = _step_read_options(
        metadata,
        pmi,
        design_variants=design_variants,
        design_variant_selection=tuple(design_variant_selection or ()),
        existing_meshes=import_existing_meshes,
        multi_file=multi_file_import,
        material_library_paths=material_libraries,
        material_library_color_space=material_library_color_space.value,
        delete_free_vertices=delete_free_vertices,
        delete_lines=delete_lines,
        construction_curve_policy=construction_curve_policy.value,
        construction_curve_tube_radius=construction_curve_tube_radius,
        source_units=source_units,
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=source_up_axis.value,
        source_handedness=source_handedness.value,
        target_units=target_units,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=None if target_up_axis is None else target_up_axis.value,
        target_handedness=None if target_handedness is None else target_handedness.value,
    )
    asset = _io_helpers._read_cad_for_cli(input_path, ctx, payload, import_options=import_options)
    if heal_brep:
        asset = asset.heal_brep(
            _brep_heal_options(
                heal_tolerance=heal_tolerance,
                group_open_shells=group_open_shells,
                cleanup_overlapping_faces=cleanup_overlapping_faces,
                overlap_area_ratio=overlap_area_ratio,
                remove_sliver_faces=remove_sliver_faces,
                max_sliver_area=max_sliver_area,
            ),
            where=where,
        )
    profile_options = by_name(profile.value)
    selection = asset.select(where) if where is not None else None
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
        "metadata_summary": _metadata_summary(asset),
        "pmi_summary": _pmi_summary(asset),
        "report": asset.report.to_dict(),
    }
    if metadata == MetadataMode.FULL:
        result["asset_metadata"] = dict(asset.metadata)
    if pmi in {PmiMode.FULL, PmiMode.METADATA, PmiMode.METADATA_AND_VISUALS}:
        result["pmi"] = [annotation.to_dict() for annotation in asset.pmi]
    if selection is not None:
        result["selection"] = selection.to_dict()
    message = f"{input_path}: {_format_stats(asset.stats())}; units={asset.units}"
    if selection is not None:
        message = f"{message}; matched {_format_stats(selection.stats())}"
    _emit(ctx, result, message)
