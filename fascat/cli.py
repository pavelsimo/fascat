from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclass_replace
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import typer
import typer.rich_utils as rich_utils
from rich.console import Console

from fascat import __version__
from fascat.analysis import AnalysisReport, analyze_output
from fascat.filter import Filter, FilterExpressionError
from fascat.io.gltf import GLTF_SUFFIXES
from fascat.io.obj import OBJ_SUFFIXES
from fascat.io.step import read_step, read_step_bytes
from fascat.io.stl import STL_SUFFIXES
from fascat.options import (
    AnalyzeOptions,
    AtlasOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    ConversionProfile,
    DecimateOptions,
    DeleteDegeneratePolygonsOptions,
    ExplodeOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    LODLevel,
    LODOptions,
    MergeOptions,
    MergeVerticesOptions,
    MetadataExportOptions,
    ObjExportOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    ReplaceOptions,
    SceneOptimizeOptions,
    StageOptions,
    StepReadOptions,
    StlExportOptions,
    Tessellation,
    UnwrapOptions,
    UsdExportOptions,
)
from fascat.pipeline import convert
from fascat.pipeline import validate_output as validate_export
from fascat.pipeline_file import PipelineSpec
from fascat.profiles import by_name
from fascat.profiles import from_file as profile_from_file
from fascat.report import Report

DOCS_URL = "https://pavelsimo.github.io/fascat"
ISSUES_URL = "https://github.com/pavelsimo/fascat/issues"
rich_utils.MAX_WIDTH = 120
STEP_SUFFIXES = {".step", ".stp"}
USD_SUFFIXES = {".usd", ".usda", ".usdc", ".usdz"}
EXPORT_SUFFIXES = USD_SUFFIXES | GLTF_SUFFIXES | OBJ_SUFFIXES | STL_SUFFIXES
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
  fascat convert motor.step motor.glb --profile virtual-reality
  fascat --json validate motor.usdc

Docs: {DOCS_URL}
Issues: {ISSUES_URL}"""

app = typer.Typer(
    name="fascat",
    help="convert CAD STEP data into realtime-ready OpenUSD and glTF assets",
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
    REALTIME_MOBILE = "realtime-mobile"
    VIRTUAL_REALITY = "virtual-reality"
    AUGMENTED_REALITY = "augmented-reality"
    MIXED_REALITY = "mixed-reality"


class AxisMode(str, Enum):
    Y = "Y"
    Z = "Z"


class HandednessMode(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class UV0Mode(str, Enum):
    NONE = "none"
    BOX = "box"
    UNWRAP = "unwrap"
    LIGHTMAP = "lightmap"


class UV1Mode(str, Enum):
    NONE = "none"
    BOX = "box"
    UNWRAP = "unwrap"
    LIGHTMAP = "lightmap"
    COPY_UV0 = "copy-uv0"


class UnwrapMethod(str, Enum):
    DEFAULT = "default"
    CONFORMAL = "conformal"
    ISOMETRIC = "isometric"


class MaterialMode(str, Enum):
    CAD = "cad"
    DISPLAY = "display"
    NONE = "none"


class MaterialPipelineMode(str, Enum):
    CAD = "cad"
    PBR = "pbr"


class NormalMode(str, Enum):
    NONE = "none"
    SMOOTH = "smooth"
    HARD_EDGES = "hard-edges"
    FLAT = "flat"


class MergeMode(str, Enum):
    ALL = "all"
    BY_MATERIAL = "by-material"
    BY_NODE_NAME = "by-node-name"
    BY_PART_NAME = "by-part-name"
    HIERARCHY_LEVEL = "hierarchy-level"
    PARENT_CHILDREN = "parent-children"
    FINAL_LEVEL = "final-level"
    REGIONS = "regions"


class MergeMetadata(str, Enum):
    PRESERVE = "preserve"
    COMBINE = "combine"
    SUMMARIZE = "summarize"
    DROP = "drop"


class MergeStrategy(str, Enum):
    ALL = "all"
    BY_MATERIAL = "by-material"


class ExplodeMode(str, Enum):
    BY_MATERIAL = "by-material"
    CONNECTED_COMPONENTS = "connected-components"


class ReplaceMode(str, Enum):
    BOUNDING_BOX = "bounding-box"
    EXTERNAL_ASSET = "external-asset"


class IndexBufferMode(str, Enum):
    AUTO = "auto"
    UINT16 = "uint16"
    UINT32 = "uint32"


class FlattenMode(str, Enum):
    NONE = "none"
    SAFE = "safe"
    ALL = "all"


class InstancePolicy(str, Enum):
    AUTO = "auto"
    PRESERVE = "preserve"
    EXPAND = "expand"


class DecimateCriterion(str, Enum):
    TARGET = "target"
    QUALITY = "quality"


class BudgetScope(str, Enum):
    PART = "part"
    SELECTION = "selection"


class UVImportance(str, Enum):
    PRESERVE_ISLANDS = "preserve-islands"
    PRESERVE_SEAMS = "preserve-seams"
    IGNORE = "ignore"


class OcclusionStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    EXTERIOR = "exterior"
    ADVANCED = "advanced"


class OcclusionLevel(str, Enum):
    PARTS = "parts"
    SUBMESHES = "submeshes"
    TRIANGLES = "triangles"


class LODPreset(str, Enum):
    DESKTOP = "desktop"
    WEB = "web"
    MOBILE = "mobile"
    VR = "vr"


class LODMode(str, Enum):
    VARIANTS = "variants"
    EXTRAS = "extras"
    SEPARATE = "separate"


class UsdPackage(str, Enum):
    DEFAULT = "default"
    USDZ = "usdz"


class MetadataMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


class PmiMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"
    METADATA = "metadata"
    METADATA_AND_VISUALS = "metadata-and-visuals"


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
    """convert CAD STEP data into realtime-ready OpenUSD and glTF assets"""
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
            help="Request multi-file STEP assembly reference resolution when supported.",
        ),
    ] = False,
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
    """Inspect STEP assembly metadata and planned conversion inputs."""
    state = _state(ctx)
    payload = {
        "command": "inspect",
        "input": str(input_path),
        "profile": profile.value,
        "metadata": metadata.value,
        "pmi": pmi.value,
        "design_variants": design_variants,
        "import_existing_meshes": import_existing_meshes,
        "multi_file_import": multi_file_import,
        "delete_free_vertices": delete_free_vertices,
        "delete_lines": delete_lines,
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
        "remove_sliver_faces": remove_sliver_faces,
        "max_sliver_area": max_sliver_area,
        "filters": filters or [],
        "exclude_filters": exclude_filters or [],
        "dry_run": state.dry_run,
    }
    where = _parse_filter_options(filters, exclude_filters, ctx, payload)
    if heal_tolerance <= 0.0:
        _fail(ctx, payload, "--heal-tolerance must be greater than 0.", code=2)
    if max_sliver_area < 0.0:
        _fail(ctx, payload, "--max-sliver-area must be greater than or equal to 0.", code=2)
    if source_meters_per_unit is not None and source_meters_per_unit <= 0.0:
        _fail(ctx, payload, "--source-meters-per-unit must be greater than 0.", code=2)
    if target_meters_per_unit is not None and target_meters_per_unit <= 0.0:
        _fail(ctx, payload, "--target-meters-per-unit must be greater than 0.", code=2)
    _validate_step_input(input_path, ctx, payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would inspect {input_path} with profile {profile.value}.")
        return

    import_options = _step_read_options(
        metadata,
        pmi,
        design_variants=design_variants,
        existing_meshes=import_existing_meshes,
        multi_file=multi_file_import,
        delete_free_vertices=delete_free_vertices,
        delete_lines=delete_lines,
        source_units=source_units,
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=source_up_axis.value,
        source_handedness=source_handedness.value,
        target_units=target_units,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=None if target_up_axis is None else target_up_axis.value,
        target_handedness=None if target_handedness is None else target_handedness.value,
    )
    asset = _read_step_for_cli(input_path, ctx, payload, import_options=import_options)
    if heal_brep:
        asset = asset.heal_brep(
            _brep_heal_options(
                heal_tolerance=heal_tolerance,
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


@app.command(
    "convert",
    epilog=f"""Examples:
  fascat convert motor.step motor.usdc
  fascat convert motor.step motor.glb --profile virtual-reality
  fascat convert motor.step motor.glb --pipeline realtime.toml
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
            help="Output USD or glTF file, or '-' for stdout. Defaults to input .usdc.",
            allow_dash=True,
        ),
    ] = None,
    profile: Annotated[Profile, typer.Option("--profile", help="Conversion profile.")] = Profile.REALTIME_DESKTOP,
    target_device_profile: Annotated[
        Path | None,
        typer.Option("--target-device-profile", help="TOML or JSON file overriding the selected profile budget."),
    ] = None,
    pipeline: Annotated[
        Path | None,
        typer.Option("--pipeline", help="TOML pipeline file with named filters and ordered steps."),
    ] = None,
    sag: Annotated[float | None, typer.Option("--sag", help="CAD tessellation sag tolerance.")] = None,
    sag_ratio: Annotated[
        float | None,
        typer.Option("--sag-ratio", help="Relative CAD tessellation sag ratio."),
    ] = None,
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
    max_polygon_length: Annotated[
        float | None,
        typer.Option("--max-polygon-length", help="Report tessellated polygon edges longer than this length."),
    ] = None,
    min_edge_length: Annotated[
        float | None,
        typer.Option("--min-edge-length", help="Collapse tessellated edges shorter than this length."),
    ] = None,
    preserve_boundaries: Annotated[
        bool,
        typer.Option(
            "--preserve-boundaries/--no-preserve-boundaries",
            help="Preserve sharp/boundary edges during tessellation cleanup.",
        ),
    ] = True,
    curvature_adaptive: Annotated[
        bool,
        typer.Option("--curvature-adaptive", help="Use tighter interior meshing on curved CAD faces."),
    ] = False,
    avoid_skinny_triangles: Annotated[
        bool,
        typer.Option("--avoid-skinny-triangles", help="Refine long skinny triangles after tessellation."),
    ] = False,
    quality_report: Annotated[
        Path | None,
        typer.Option("--quality-report", help="Write per-part tessellation quality metrics as JSON."),
    ] = None,
    free_edge_report: Annotated[
        bool,
        typer.Option("--free-edge-report", help="Record and warn about free tessellation edges."),
    ] = False,
    reuse_existing_meshes: Annotated[
        bool,
        typer.Option(
            "--reuse-existing-meshes/--retessellate-existing-meshes",
            help="Reuse imported meshes instead of retessellating source BREP geometry.",
        ),
    ] = True,
    heal_brep: Annotated[bool, typer.Option("--heal-brep", help="Run BREP healing before tessellation.")] = False,
    heal_tolerance: Annotated[float, typer.Option("--heal-tolerance", help="BREP healing tolerance.")] = 0.05,
    remove_sliver_faces: Annotated[
        bool,
        typer.Option("--remove-sliver-faces", help="Detect tiny sliver faces during BREP healing."),
    ] = False,
    max_sliver_area: Annotated[
        float,
        typer.Option("--max-sliver-area", help="Area threshold for sliver-face reporting."),
    ] = 1e-4,
    fail_on_open_shells: Annotated[
        bool,
        typer.Option("--fail-on-open-shells", help="Fail if healed BREP still contains open shells."),
    ] = False,
    lods: Annotated[
        str | None,
        typer.Option("--lods", help="Comma-separated LOD ratios, for example 0.5,0.25,0.1."),
    ] = None,
    lod_mode: Annotated[
        LODMode,
        typer.Option("--lod-mode", help="LOD output mode: variants, extras, or separate."),
    ] = LODMode.VARIANTS,
    lod_per_part_budget: Annotated[
        bool,
        typer.Option("--lod-per-part-budget", help="Apply LOD budgets independently per part."),
    ] = False,
    lod_drop_tiny_parts: Annotated[
        bool,
        typer.Option("--lod-drop-tiny-parts", help="Omit tiny parts from lower LOD meshes."),
    ] = False,
    lod_tiny_part_screen_size: Annotated[
        float,
        typer.Option("--lod-tiny-part-screen-size", help="Screen-size threshold for tiny-part LOD omission."),
    ] = 2.0,
    normals: Annotated[
        NormalMode,
        typer.Option("--normals", help="Normal generation mode: none, smooth, hard-edges, or flat."),
    ] = NormalMode.SMOOTH,
    preserve_face_boundaries: Annotated[
        bool,
        typer.Option("--preserve-face-boundaries", help="Treat CAD face-group boundaries as hard normal edges."),
    ] = False,
    tangents: Annotated[
        bool,
        typer.Option("--tangents", help="Generate glTF-compatible vertex tangents from the selected UV channel."),
    ] = False,
    tangent_uv_channel: Annotated[
        int,
        typer.Option("--tangent-uv-channel", help="UV channel used for tangent generation."),
    ] = 0,
    override_tangents: Annotated[
        bool,
        typer.Option(
            "--override-tangents/--preserve-tangents",
            help="Regenerate existing tangents instead of preserving them when --tangents is used.",
        ),
    ] = False,
    validate_normals: Annotated[
        bool,
        typer.Option("--validate-normals", help="Validate staged normals and tangents."),
    ] = False,
    uv0: Annotated[UV0Mode, typer.Option("--uv0", help="UV0 generation mode.")] = UV0Mode.BOX,
    uv1: Annotated[UV1Mode, typer.Option("--uv1", help="UV1 generation mode.")] = UV1Mode.NONE,
    normalize_uvs: Annotated[
        str | None,
        typer.Option("--normalize-uvs", help="Comma-separated UV channels to normalize into 0..1, for example 1."),
    ] = None,
    materials: Annotated[
        MaterialMode,
        typer.Option("--materials", help="Material staging mode: cad, display, or none."),
    ] = MaterialMode.CAD,
    material_mode: Annotated[
        MaterialPipelineMode,
        typer.Option("--material-mode", help="Material normalization mode: cad or pbr."),
    ] = MaterialPipelineMode.CAD,
    merge_equivalent_materials: Annotated[
        bool,
        typer.Option("--merge-equivalent-materials", help="Merge CAD materials with matching PBR values."),
    ] = False,
    merge_vertices: Annotated[
        bool,
        typer.Option("--merge-vertices", help="Merge exact or tolerance-close vertices after staging."),
    ] = False,
    merge_vertex_tolerance: Annotated[
        float,
        typer.Option("--merge-vertex-tolerance", help="Position tolerance used by --merge-vertices."),
    ] = 0.0,
    preserve_merge_vertex_attributes: Annotated[
        bool,
        typer.Option(
            "--preserve-merge-vertex-attributes/--drop-merge-vertex-attributes",
            help="Protect normals, tangents, and UV seams when --merge-vertices is used.",
        ),
    ] = True,
    preserve_merge_vertex_material_boundaries: Annotated[
        bool,
        typer.Option(
            "--preserve-merge-vertex-material-boundaries/--ignore-merge-vertex-material-boundaries",
            help="Protect material-boundary vertices when --merge-vertices is used.",
        ),
    ] = True,
    delete_merge_vertex_degenerate: Annotated[
        bool,
        typer.Option(
            "--delete-merge-vertex-degenerate/--keep-merge-vertex-degenerate",
            help="Delete degenerate polygons created by --merge-vertices.",
        ),
    ] = True,
    merge_vertex_area_epsilon: Annotated[
        float,
        typer.Option("--merge-vertex-area-epsilon", help="Area threshold for degenerate polygons after merging."),
    ] = 1e-12,
    delete_degenerate_polygons: Annotated[
        bool,
        typer.Option("--delete-degenerate-polygons", help="Run standalone degenerate polygon cleanup."),
    ] = False,
    degenerate_area_epsilon: Annotated[
        float,
        typer.Option("--degenerate-area-epsilon", help="Area threshold for standalone degenerate polygon cleanup."),
    ] = 1e-12,
    texel_density: Annotated[
        float | None,
        typer.Option("--texel-density", help="UV texel density metadata for unwrap and atlas workflows."),
    ] = None,
    uv_padding: Annotated[
        int,
        typer.Option("--uv-padding", help="UV island padding metadata in pixels."),
    ] = 2,
    max_stretch: Annotated[
        float | None,
        typer.Option("--max-stretch", help="Maximum UV stretch metadata for unwrap workflows."),
    ] = None,
    unwrap_method: Annotated[
        UnwrapMethod,
        typer.Option("--unwrap-method", help="Unwrap solver intent: default, conformal, or isometric."),
    ] = UnwrapMethod.DEFAULT,
    unwrap_iterations: Annotated[
        int | None,
        typer.Option("--unwrap-iterations", help="Requested unwrap solver iteration budget metadata."),
    ] = None,
    unwrap_tolerance: Annotated[
        float | None,
        typer.Option("--unwrap-tolerance", help="Requested unwrap solver tolerance metadata."),
    ] = None,
    uv_sharp_to_seam: Annotated[
        bool,
        typer.Option(
            "--uv-sharp-to-seam/--uv-no-sharp-to-seam",
            help="Request sharp edges as UV seams for unwrap and lightmap channels.",
        ),
    ] = False,
    uv_forbid_overlapping: Annotated[
        bool,
        typer.Option(
            "--uv-forbid-overlapping/--uv-allow-overlapping",
            help="Request non-overlapping UV islands and report overlaps as policy violations.",
        ),
    ] = False,
    atlas: Annotated[bool, typer.Option("--atlas", help="Tag materials and UVs for a generated atlas.")] = False,
    atlas_size: Annotated[int, typer.Option("--atlas-size", help="Maximum atlas texture size.")] = 4096,
    metadata: Annotated[
        MetadataMode,
        typer.Option("--metadata", help="Metadata import/export mode: none, summary, or full."),
    ] = MetadataMode.FULL,
    pmi: Annotated[
        PmiMode,
        typer.Option("--pmi", help="PMI import/export mode: none, metadata, or metadata-and-visuals."),
    ] = PmiMode.METADATA,
    design_variants: Annotated[
        bool,
        typer.Option("--design-variants/--no-design-variants", help="Request STEP design variant import."),
    ] = False,
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
            help="Request multi-file STEP assembly reference resolution when supported.",
        ),
    ] = False,
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
    merge: Annotated[bool, typer.Option("--merge", help="Merge selected geometry before optimization.")] = False,
    merge_mode: Annotated[MergeMode, typer.Option("--merge-mode", help="Merge grouping mode.")] = MergeMode.ALL,
    keep_parent: Annotated[
        bool,
        typer.Option(
            "--keep-parent/--no-keep-parent", help="Attach merged nodes to a shared selected parent when possible."
        ),
    ] = True,
    merge_metadata: Annotated[
        MergeMetadata,
        typer.Option("--merge-metadata", help="Metadata policy for merged parts."),
    ] = MergeMetadata.PRESERVE,
    max_vertices_per_mesh: Annotated[
        int | None,
        typer.Option("--max-vertices-per-mesh", help="Split merged output above this vertex count."),
    ] = 65_535,
    region_size: Annotated[
        float | None,
        typer.Option("--region-size", help="Spatial region size for --merge-mode regions."),
    ] = None,
    merge_strategy: Annotated[
        MergeStrategy,
        typer.Option("--merge-strategy", help="Substrategy for region merging."),
    ] = MergeStrategy.ALL,
    hierarchy_level: Annotated[
        int,
        typer.Option("--hierarchy-level", help="Hierarchy level used by --merge-mode hierarchy-level."),
    ] = 1,
    explode: Annotated[
        ExplodeMode | None,
        typer.Option("--explode", help="Explode selected geometry by material or connected-components."),
    ] = None,
    replace: Annotated[
        ReplaceMode | None,
        typer.Option("--replace", help="Replace selected geometry with bounding-box or external-asset proxies."),
    ] = None,
    external_asset: Annotated[
        str | None,
        typer.Option("--external-asset", help="External asset path recorded by --replace external-asset."),
    ] = None,
    batch_by_material: Annotated[
        bool,
        typer.Option("--batch-by-material", help="Batch compatible scene geometry by material."),
    ] = False,
    merge_compatible_meshes: Annotated[
        bool,
        typer.Option("--merge-compatible-meshes", help="Merge compatible scene meshes to reduce draw calls."),
    ] = False,
    split_large_meshes: Annotated[
        bool,
        typer.Option("--split-large-meshes", help="Split scene-optimized meshes above the vertex limit."),
    ] = False,
    index_buffer: Annotated[
        IndexBufferMode,
        typer.Option("--index-buffer", help="Index buffer mode: auto, uint16, or uint32."),
    ] = IndexBufferMode.AUTO,
    flatten: Annotated[
        FlattenMode,
        typer.Option("--flatten", help="Hierarchy flattening mode: none, safe, or all."),
    ] = FlattenMode.SAFE,
    instance_policy: Annotated[
        InstancePolicy,
        typer.Option("--instance-policy", help="Instance policy: auto, preserve, or expand."),
    ] = InstancePolicy.AUTO,
    bake_materials: Annotated[
        bool,
        typer.Option("--bake-materials", help="Create a shared baked material with constant embedded textures."),
    ] = False,
    maps_resolution: Annotated[
        int,
        typer.Option("--maps-resolution", help="Requested bake texture resolution metadata in pixels."),
    ] = 2048,
    force_uv_generation: Annotated[
        bool,
        typer.Option("--force-uv-generation", help="Generate UVs before recording baked material textures."),
    ] = False,
    bake: Annotated[
        str,
        typer.Option("--bake", help="Comma-separated material maps to bake, for example base-color,opacity."),
    ] = "base-color",
    decimate: Annotated[
        bool,
        typer.Option("--decimate", help="Run the explicit decimation action before profile optimization."),
    ] = False,
    decimate_criterion: Annotated[
        DecimateCriterion,
        typer.Option("--decimate-criterion", help="Decimation criterion: target or quality."),
    ] = DecimateCriterion.TARGET,
    surface_tolerance: Annotated[
        float | None,
        typer.Option("--surface-tolerance", help="Surface deviation tolerance metadata for decimation."),
    ] = None,
    line_tolerance: Annotated[
        float | None,
        typer.Option("--line-tolerance", help="Hard-edge deviation tolerance metadata for decimation."),
    ] = None,
    normal_tolerance: Annotated[
        float,
        typer.Option("--normal-tolerance", help="Normal angle tolerance for decimation preservation."),
    ] = 15.0,
    uv_tolerance: Annotated[
        float | None,
        typer.Option("--uv-tolerance", help="UV deviation tolerance metadata for decimation."),
    ] = None,
    protect_topology: Annotated[
        bool,
        typer.Option("--protect-topology/--no-protect-topology", help="Preserve topology-sensitive faces."),
    ] = True,
    budget_scope: Annotated[
        BudgetScope,
        typer.Option("--budget-scope", help="Decimation budget scope: part or selection."),
    ] = BudgetScope.SELECTION,
    uv_importance: Annotated[
        UVImportance,
        typer.Option("--uv-importance", help="Decimation UV importance: preserve-islands, preserve-seams, or ignore."),
    ] = UVImportance.PRESERVE_ISLANDS,
    remove_holes: Annotated[
        bool,
        typer.Option("--remove-holes", help="Remove small hole features with mesh fallback."),
    ] = False,
    hole_types: Annotated[
        str,
        typer.Option("--hole-types", help="Comma-separated hole types: through, blind, surface."),
    ] = "through,blind,surface",
    max_hole_diameter: Annotated[
        float | None,
        typer.Option("--max-hole-diameter", help="Maximum hole diameter to remove."),
    ] = 3.0,
    prefer_brep: Annotated[
        bool,
        typer.Option("--prefer-brep/--no-prefer-brep", help="Prefer BREP feature removal when available."),
    ] = True,
    remove_occluded: Annotated[
        bool,
        typer.Option("--remove-occluded", help="Remove selected nodes hidden inside larger opaque bounds."),
    ] = False,
    occlusion_strategy: Annotated[
        OcclusionStrategy,
        typer.Option("--occlusion-strategy", help="Occlusion strategy: conservative, exterior, or advanced."),
    ] = OcclusionStrategy.ADVANCED,
    occlusion_level: Annotated[
        OcclusionLevel,
        typer.Option("--occlusion-level", help="Occlusion processing level: parts, submeshes, or triangles."),
    ] = OcclusionLevel.TRIANGLES,
    occlusion_precision: Annotated[
        int,
        typer.Option("--occlusion-precision", help="Occlusion precision preset or sample resolution."),
    ] = 2048,
    hemi_evaluation: Annotated[
        bool,
        typer.Option("--hemi-evaluation", help="Restrict occlusion visibility rays to top and side views."),
    ] = False,
    neighbors_preservation: Annotated[
        int,
        typer.Option("--neighbors-preservation", help="Visible-neighbor preservation rings for triangle occlusion."),
    ] = 1,
    consider_transparency_opaque: Annotated[
        bool,
        typer.Option("--consider-transparency-opaque", help="Treat transparent materials as occluders."),
    ] = False,
    preserve_cavities: Annotated[
        bool,
        typer.Option("--preserve-cavities/--no-preserve-cavities", help="Preserve large interior cavities."),
    ] = True,
    minimum_cavity_volume_m3: Annotated[
        float,
        typer.Option("--minimum-cavity-volume-m3", help="Minimum cavity volume to preserve."),
    ] = 0.5,
    run_lod_generators: Annotated[
        bool,
        typer.Option("--run-lod-generators", help="Run preset-driven LOD generation after optimization actions."),
    ] = False,
    lod_preset: Annotated[
        LODPreset,
        typer.Option("--lod-preset", help="LOD generator preset: desktop, web, mobile, or vr."),
    ] = LODPreset.DESKTOP,
    lod_screen_coverage: Annotated[
        str | None,
        typer.Option("--lod-screen-coverage", help="Comma-separated LOD screen coverage values."),
    ] = None,
    validate_lods: Annotated[
        bool,
        typer.Option("--validate-lods", help="Validate generated LOD monotonicity."),
    ] = False,
    filters: Annotated[
        list[str] | None,
        typer.Option("--filter", help="Scope optimization and LOD work with selectors such as path=*/Fasteners/*."),
    ] = None,
    exclude_filters: Annotated[
        list[str] | None,
        typer.Option("--exclude-filter", help="Exclude selector matches from --filter results."),
    ] = None,
    preserve_instances: Annotated[
        bool,
        typer.Option(
            "--preserve-instances/--no-preserve-instances",
            help="Preserve repeated parts as shared instances.",
        ),
    ] = True,
    preserve_hard_edges: Annotated[
        bool,
        typer.Option("--preserve-hard-edges", help="Protect faces adjacent to hard edges during simplification."),
    ] = False,
    hard_edge_angle: Annotated[
        float,
        typer.Option("--hard-edge-angle", help="Angle threshold for hard-edge preservation."),
    ] = 30.0,
    preserve_holes: Annotated[
        bool,
        typer.Option("--preserve-holes", help="Protect open boundary faces during simplification."),
    ] = False,
    preserve_material_boundaries: Annotated[
        bool,
        typer.Option("--preserve-material-boundaries", help="Protect faces along material boundaries."),
    ] = False,
    preserve_uv_seams: Annotated[
        bool,
        typer.Option("--preserve-uv-seams", help="Protect faces touching duplicated-position UV seams."),
    ] = False,
    preserve_small_parts: Annotated[
        bool,
        typer.Option("--preserve-small-parts", help="Skip simplification for small parts."),
    ] = False,
    small_part_triangle_threshold: Annotated[
        int,
        typer.Option("--small-part-triangle-threshold", help="Triangle threshold for --preserve-small-parts."),
    ] = 64,
    preserve_silhouette: Annotated[
        bool,
        typer.Option("--preserve-silhouette", help="Protect faces on bounding-box silhouette extremes."),
    ] = False,
    quantize: Annotated[
        bool,
        typer.Option("--quantize", help="Write glTF KHR_mesh_quantization accessors."),
    ] = False,
    meshopt: Annotated[
        bool,
        typer.Option("--meshopt", help="Write glTF EXT_meshopt_compression payloads with fallback data."),
    ] = False,
    draco: Annotated[
        bool,
        typer.Option("--draco", help="Unsupported until a Draco encoder backend is integrated."),
    ] = False,
    texture_compression: Annotated[
        str | None,
        typer.Option("--texture-compression", help="Unsupported until a KTX2/Basis encoder backend is integrated."),
    ] = None,
    package: Annotated[
        UsdPackage,
        typer.Option("--package", help="USD package mode: default or usdz."),
    ] = UsdPackage.DEFAULT,
    file_size_budget_mb: Annotated[
        float | None,
        typer.Option("--file-size-budget-mb", help="Warn in reports when output exceeds this size."),
    ] = None,
    obj_materials: Annotated[
        bool,
        typer.Option("--obj-materials/--no-obj-materials", help="Write OBJ material assignments."),
    ] = True,
    write_mtl: Annotated[
        bool,
        typer.Option("--write-mtl/--no-write-mtl", help="Write an OBJ MTL sidecar."),
    ] = True,
    preserve_groups: Annotated[
        bool,
        typer.Option("--preserve-groups/--no-preserve-groups", help="Preserve OBJ groups per occurrence."),
    ] = True,
    stl_binary: Annotated[
        bool,
        typer.Option("--stl-binary/--stl-ascii", help="Write binary STL instead of ASCII STL."),
    ] = True,
    stl_merge: Annotated[
        bool,
        typer.Option("--stl-merge/--no-stl-merge", help="Merge STL output into one triangle stream."),
    ] = True,
    debug: Annotated[bool, typer.Option("--debug", help="Prefer debuggable USDA output conventions.")] = False,
    report: Annotated[Path | None, typer.Option("--report", help="Write a JSON conversion report sidecar.")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite an existing output file.")] = False,
) -> None:
    """Convert a STEP file into a realtime-ready OpenUSD or glTF asset."""
    state = _state(ctx)
    payload: dict[str, Any] = {
        "command": "convert",
        "input": str(input_path),
        "output": str(output_path) if output_path is not None else None,
        "profile": profile.value,
        "base_profile": None,
        "target_device_profile": str(target_device_profile) if target_device_profile else None,
        "pipeline": str(pipeline) if pipeline else None,
        "sag": sag,
        "sag_ratio": sag_ratio,
        "angle": angle,
        "target_triangles": target_triangles,
        "ratio": ratio,
        "max_edge_length": max_edge_length,
        "max_polygon_length": max_polygon_length,
        "min_edge_length": min_edge_length,
        "preserve_boundaries": preserve_boundaries,
        "curvature_adaptive": curvature_adaptive,
        "avoid_skinny_triangles": avoid_skinny_triangles,
        "quality_report": str(quality_report) if quality_report else None,
        "free_edge_report": free_edge_report,
        "reuse_existing_meshes": reuse_existing_meshes,
        "heal_brep": heal_brep,
        "heal_tolerance": heal_tolerance,
        "remove_sliver_faces": remove_sliver_faces,
        "max_sliver_area": max_sliver_area,
        "fail_on_open_shells": fail_on_open_shells,
        "lods": None,
        "lod_mode": lod_mode.value,
        "lod_per_part_budget": lod_per_part_budget,
        "lod_drop_tiny_parts": lod_drop_tiny_parts,
        "lod_tiny_part_screen_size": lod_tiny_part_screen_size,
        "normals": normals.value,
        "preserve_face_boundaries": preserve_face_boundaries,
        "tangents": tangents,
        "tangent_uv_channel": tangent_uv_channel,
        "override_tangents": override_tangents,
        "validate_normals": validate_normals,
        "uv0": uv0.value,
        "uv1": uv1.value,
        "normalize_uvs": normalize_uvs,
        "materials": materials.value,
        "material_mode": material_mode.value,
        "merge_equivalent_materials": merge_equivalent_materials,
        "merge_vertices": merge_vertices,
        "merge_vertex_tolerance": merge_vertex_tolerance,
        "preserve_merge_vertex_attributes": preserve_merge_vertex_attributes,
        "preserve_merge_vertex_material_boundaries": preserve_merge_vertex_material_boundaries,
        "delete_merge_vertex_degenerate": delete_merge_vertex_degenerate,
        "merge_vertex_area_epsilon": merge_vertex_area_epsilon,
        "delete_degenerate_polygons": delete_degenerate_polygons,
        "degenerate_area_epsilon": degenerate_area_epsilon,
        "texel_density": texel_density,
        "uv_padding": uv_padding,
        "max_stretch": max_stretch,
        "unwrap_method": unwrap_method.value,
        "unwrap_iterations": unwrap_iterations,
        "unwrap_tolerance": unwrap_tolerance,
        "uv_sharp_to_seam": uv_sharp_to_seam,
        "uv_forbid_overlapping": uv_forbid_overlapping,
        "atlas": atlas,
        "atlas_size": atlas_size,
        "metadata": metadata.value,
        "pmi": pmi.value,
        "design_variants": design_variants,
        "import_existing_meshes": import_existing_meshes,
        "multi_file_import": multi_file_import,
        "delete_free_vertices": delete_free_vertices,
        "delete_lines": delete_lines,
        "source_units": source_units,
        "source_meters_per_unit": source_meters_per_unit,
        "source_up_axis": source_up_axis.value,
        "source_handedness": source_handedness.value,
        "target_units": target_units,
        "target_meters_per_unit": target_meters_per_unit,
        "target_up_axis": None if target_up_axis is None else target_up_axis.value,
        "target_handedness": None if target_handedness is None else target_handedness.value,
        "merge": merge,
        "merge_mode": merge_mode.value,
        "keep_parent": keep_parent,
        "merge_metadata": merge_metadata.value,
        "max_vertices_per_mesh": max_vertices_per_mesh,
        "region_size": region_size,
        "merge_strategy": merge_strategy.value,
        "hierarchy_level": hierarchy_level,
        "explode": None if explode is None else explode.value,
        "replace": None if replace is None else replace.value,
        "external_asset": external_asset,
        "batch_by_material": batch_by_material,
        "merge_compatible_meshes": merge_compatible_meshes,
        "split_large_meshes": split_large_meshes,
        "index_buffer": index_buffer.value,
        "flatten": flatten.value,
        "instance_policy": instance_policy.value,
        "bake_materials": bake_materials,
        "maps_resolution": maps_resolution,
        "force_uv_generation": force_uv_generation,
        "bake": bake,
        "decimate": decimate,
        "decimate_criterion": decimate_criterion.value,
        "surface_tolerance": surface_tolerance,
        "line_tolerance": line_tolerance,
        "normal_tolerance": normal_tolerance,
        "uv_tolerance": uv_tolerance,
        "protect_topology": protect_topology,
        "budget_scope": budget_scope.value,
        "uv_importance": uv_importance.value,
        "remove_holes": remove_holes,
        "hole_types": hole_types,
        "max_hole_diameter": max_hole_diameter,
        "prefer_brep": prefer_brep,
        "remove_occluded": remove_occluded,
        "occlusion_strategy": occlusion_strategy.value,
        "occlusion_level": occlusion_level.value,
        "occlusion_precision": occlusion_precision,
        "hemi_evaluation": hemi_evaluation,
        "neighbors_preservation": neighbors_preservation,
        "consider_transparency_opaque": consider_transparency_opaque,
        "preserve_cavities": preserve_cavities,
        "minimum_cavity_volume_m3": minimum_cavity_volume_m3,
        "run_lod_generators": run_lod_generators,
        "lod_preset": lod_preset.value,
        "lod_screen_coverage": lod_screen_coverage,
        "validate_lods": validate_lods,
        "filters": filters or [],
        "exclude_filters": exclude_filters or [],
        "preserve_instances": preserve_instances,
        "preserve_hard_edges": preserve_hard_edges,
        "hard_edge_angle": hard_edge_angle,
        "preserve_holes": preserve_holes,
        "preserve_material_boundaries": preserve_material_boundaries,
        "preserve_uv_seams": preserve_uv_seams,
        "preserve_small_parts": preserve_small_parts,
        "small_part_triangle_threshold": small_part_triangle_threshold,
        "preserve_silhouette": preserve_silhouette,
        "quantize": quantize,
        "meshopt": meshopt,
        "draco": draco,
        "texture_compression": texture_compression,
        "package": package.value,
        "file_size_budget_mb": file_size_budget_mb,
        "obj_materials": obj_materials,
        "write_mtl": write_mtl,
        "preserve_groups": preserve_groups,
        "stl_binary": stl_binary,
        "stl_merge": stl_merge,
        "debug": debug,
        "report": str(report) if report else None,
        "force": force,
        "dry_run": state.dry_run,
    }
    where = _parse_filter_options(filters, exclude_filters, ctx, payload)
    pipeline_spec = _read_pipeline_for_cli(pipeline, ctx, payload) if pipeline is not None else None
    if pipeline_spec is not None:
        payload["pipeline_steps"] = [step.to_dict() for step in pipeline_spec.steps]
        payload["pipeline_filters"] = sorted(pipeline_spec.filters)
        payload["pipeline_advisories"] = pipeline_spec.advisories()
        payload["pipeline_import"] = (
            None if pipeline_spec.import_options is None else pipeline_spec.import_options.to_dict()
        )
        payload["pipeline_export"] = (
            None if pipeline_spec.export_metadata is None else pipeline_spec.export_metadata.to_dict()
        )
    lod_values = _parse_lods(lods, ctx, payload)
    bake_maps = _parse_bake_maps(bake, ctx, payload)
    enabled_hole_types = _parse_hole_types(hole_types, ctx, payload)
    lod_coverages = _parse_lod_screen_coverage(lod_screen_coverage, ctx, payload)
    normalized_uv_channels = _parse_uv_channels(normalize_uvs, ctx, payload)
    if tangent_uv_channel < 0:
        _fail(ctx, payload, "--tangent-uv-channel must be greater than or equal to 0.", code=2)
    payload["lods"] = lod_values
    payload["bake"] = list(bake_maps)
    payload["hole_types"] = list(enabled_hole_types)
    payload["lod_screen_coverage"] = lod_coverages
    payload["normalize_uvs"] = list(normalized_uv_channels)
    _validate_step_input(input_path, ctx, payload)
    output_path = _resolve_convert_output(input_path, output_path, ctx, payload)
    payload["output"] = str(output_path)
    _validate_export_output(output_path, ctx, payload)
    if ratio is not None and (ratio <= 0.0 or ratio >= 1.0):
        _fail(ctx, payload, "--ratio must be greater than 0 and less than 1.", code=2)
    if sag is not None and sag <= 0.0:
        _fail(ctx, payload, "--sag must be greater than 0.", code=2)
    if sag_ratio is not None and sag_ratio <= 0.0:
        _fail(ctx, payload, "--sag-ratio must be greater than 0.", code=2)
    if angle is not None and (angle <= 0.0 or angle > 180.0):
        _fail(ctx, payload, "--angle must be greater than 0 and no more than 180.", code=2)
    if target_triangles is not None and target_triangles <= 0:
        _fail(ctx, payload, "--target-triangles must be greater than 0.", code=2)
    if min_edge_length is not None and min_edge_length <= 0.0:
        _fail(ctx, payload, "--min-edge-length must be greater than 0.", code=2)
    if max_edge_length is not None and max_edge_length <= 0.0:
        _fail(ctx, payload, "--max-edge-length must be greater than 0.", code=2)
    if max_polygon_length is not None and max_polygon_length <= 0.0:
        _fail(ctx, payload, "--max-polygon-length must be greater than 0.", code=2)
    if min_edge_length is not None and max_edge_length is not None and min_edge_length > max_edge_length:
        _fail(ctx, payload, "--min-edge-length must be less than or equal to --max-edge-length.", code=2)
    if heal_tolerance <= 0.0:
        _fail(ctx, payload, "--heal-tolerance must be greater than 0.", code=2)
    if max_sliver_area < 0.0:
        _fail(ctx, payload, "--max-sliver-area must be greater than or equal to 0.", code=2)
    if max_vertices_per_mesh is not None and max_vertices_per_mesh <= 0:
        _fail(ctx, payload, "--max-vertices-per-mesh must be greater than 0.", code=2)
    if split_large_meshes and max_vertices_per_mesh is not None and max_vertices_per_mesh < 3:
        _fail(ctx, payload, "--max-vertices-per-mesh must be at least 3 when splitting large meshes.", code=2)
    if hierarchy_level < 0:
        _fail(ctx, payload, "--hierarchy-level must be greater than or equal to 0.", code=2)
    if region_size is not None and region_size <= 0.0:
        _fail(ctx, payload, "--region-size must be greater than 0.", code=2)
    if merge and merge_mode == MergeMode.REGIONS and region_size is None:
        _fail(ctx, payload, "--merge-mode regions requires --region-size.", code=2)
    if replace == ReplaceMode.EXTERNAL_ASSET and not external_asset:
        _fail(ctx, payload, "--replace external-asset requires --external-asset.", code=2)
    if hard_edge_angle <= 0.0 or hard_edge_angle > 180.0:
        _fail(ctx, payload, "--hard-edge-angle must be greater than 0 and no more than 180.", code=2)
    if small_part_triangle_threshold < 0:
        _fail(ctx, payload, "--small-part-triangle-threshold must be greater than or equal to 0.", code=2)
    if texel_density is not None and texel_density <= 0.0:
        _fail(ctx, payload, "--texel-density must be greater than 0.", code=2)
    if uv_padding < 0:
        _fail(ctx, payload, "--uv-padding must be greater than or equal to 0.", code=2)
    if max_stretch is not None and max_stretch < 0.0:
        _fail(ctx, payload, "--max-stretch must be greater than or equal to 0.", code=2)
    if merge_vertex_tolerance < 0.0:
        _fail(ctx, payload, "--merge-vertex-tolerance must be greater than or equal to 0.", code=2)
    if merge_vertex_area_epsilon < 0.0:
        _fail(ctx, payload, "--merge-vertex-area-epsilon must be greater than or equal to 0.", code=2)
    if unwrap_iterations is not None and unwrap_iterations <= 0:
        _fail(ctx, payload, "--unwrap-iterations must be greater than 0.", code=2)
    if unwrap_tolerance is not None and unwrap_tolerance < 0.0:
        _fail(ctx, payload, "--unwrap-tolerance must be greater than or equal to 0.", code=2)
    if atlas_size <= 0:
        _fail(ctx, payload, "--atlas-size must be greater than 0.", code=2)
    if maps_resolution <= 0:
        _fail(ctx, payload, "--maps-resolution must be greater than 0.", code=2)
    for option_name, value in {
        "--surface-tolerance": surface_tolerance,
        "--line-tolerance": line_tolerance,
        "--uv-tolerance": uv_tolerance,
    }.items():
        if value is not None and value < 0.0:
            _fail(ctx, payload, f"{option_name} must be greater than or equal to 0.", code=2)
    if normal_tolerance <= 0.0 or normal_tolerance > 180.0:
        _fail(ctx, payload, "--normal-tolerance must be greater than 0 and no more than 180.", code=2)
    if max_hole_diameter is not None and max_hole_diameter <= 0.0:
        _fail(ctx, payload, "--max-hole-diameter must be greater than 0.", code=2)
    if source_meters_per_unit is not None and source_meters_per_unit <= 0.0:
        _fail(ctx, payload, "--source-meters-per-unit must be greater than 0.", code=2)
    if target_meters_per_unit is not None and target_meters_per_unit <= 0.0:
        _fail(ctx, payload, "--target-meters-per-unit must be greater than 0.", code=2)
    if occlusion_precision <= 0:
        _fail(ctx, payload, "--occlusion-precision must be greater than 0.", code=2)
    if neighbors_preservation < 0:
        _fail(ctx, payload, "--neighbors-preservation must be greater than or equal to 0.", code=2)
    if minimum_cavity_volume_m3 < 0.0:
        _fail(ctx, payload, "--minimum-cavity-volume-m3 must be greater than or equal to 0.", code=2)
    if (
        run_lod_generators
        and lod_coverages is not None
        and lod_values is not None
        and len(lod_coverages) != len(lod_values)
    ):
        _fail(ctx, payload, "--lod-screen-coverage and --lods must have the same number of values.", code=2)
    if (
        not run_lod_generators
        and lod_coverages is not None
        and lod_values is not None
        and len(lod_coverages) != len(lod_values)
    ):
        _fail(ctx, payload, "--lod-screen-coverage and --lods must have the same number of values.", code=2)
    if run_lod_generators and lod_coverages is not None and lod_values is None:
        default_lod_count = len(LODGeneratorOptions(preset=cast(Any, lod_preset.value)).levels)
        if len(lod_coverages) != default_lod_count:
            _fail(
                ctx, payload, "--lod-screen-coverage must match the preset LOD count or be paired with --lods.", code=2
            )
    if lod_tiny_part_screen_size < 0.0:
        _fail(ctx, payload, "--lod-tiny-part-screen-size must be greater than or equal to 0.", code=2)
    if texture_compression not in {None, "ktx2", "basisu"}:
        _fail(ctx, payload, "--texture-compression must be one of: ktx2, basisu.", code=2)
    if texture_compression is not None:
        _fail(
            ctx,
            payload,
            "--texture-compression is not supported because no KTX2/Basis encoder backend is integrated.",
            code=2,
        )
    if draco:
        _fail(ctx, payload, "--draco is not supported because no Draco encoder backend is integrated.", code=2)
    if file_size_budget_mb is not None and file_size_budget_mb <= 0.0:
        _fail(ctx, payload, "--file-size-budget-mb must be greater than 0.", code=2)
    if package == UsdPackage.USDZ and not _is_stdio(output_path) and output_path.suffix.lower() != ".usdz":
        _fail(ctx, payload, "--package usdz requires a .usdz output path.", code=2)
    if debug and not _is_stdio(output_path) and output_path.suffix.lower() not in {".usd", ".usda"}:
        _fail(ctx, payload, "--debug requires .usd or .usda output.", code=2)
    if quality_report is not None and report is not None and quality_report.resolve() == report.resolve():
        _fail(ctx, payload, "--quality-report must use a different path than --report.", code=2)

    profile_options = _profile_for_cli(profile, target_device_profile, ctx, payload)
    payload["profile"] = profile_options.name
    if target_device_profile is not None:
        payload["base_profile"] = profile.value
        payload["profile_options"] = profile_options.to_dict()

    payload["operation_diagnostics"] = _convert_operation_diagnostics(payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would convert {input_path} to {output_path} with profile {profile_options.name}.")
        return

    _require_existing_file(input_path, "input", ctx, payload)
    if not _is_stdio(output_path) and output_path.exists() and not force:
        _fail(ctx, payload, f"Output already exists: {output_path}. Pass --force to overwrite.")

    try:
        base_tessellation = profile_options.tessellation
        if base_tessellation is None:
            _fail(ctx, payload, "The inspect-only profile cannot be used for conversion.", code=2)
        tessellation = dataclass_replace(
            base_tessellation,
            sag=sag if sag is not None else base_tessellation.sag,
            sag_ratio=sag_ratio if sag_ratio is not None else base_tessellation.sag_ratio,
            angle=angle if angle is not None else base_tessellation.angle,
            min_edge_length=min_edge_length if min_edge_length is not None else base_tessellation.min_edge_length,
            max_edge_length=max_edge_length if max_edge_length is not None else base_tessellation.max_edge_length,
            max_polygon_length=max_polygon_length
            if max_polygon_length is not None
            else base_tessellation.max_polygon_length,
            preserve_boundaries=preserve_boundaries,
            curvature_adaptive=curvature_adaptive,
            avoid_skinny_triangles=avoid_skinny_triangles,
            quality_report=quality_report is not None or base_tessellation.quality_report,
            free_edge_report=free_edge_report or base_tessellation.free_edge_report,
            reuse_existing_meshes=reuse_existing_meshes,
        )
        optimize_options = profile_options.optimize
        if optimize_options is not None:
            optimize_options = dataclass_replace(
                optimize_options,
                target_triangles=target_triangles
                if target_triangles is not None
                else optimize_options.target_triangles,
                ratio=ratio,
                preserve_instances=preserve_instances,
                preserve_hard_edges=preserve_hard_edges,
                hard_edge_angle=hard_edge_angle,
                preserve_holes=preserve_holes,
                preserve_material_boundaries=preserve_material_boundaries,
                preserve_uv_seams=preserve_uv_seams,
                preserve_small_parts=preserve_small_parts,
                small_part_triangle_threshold=small_part_triangle_threshold,
                preserve_silhouette=preserve_silhouette,
            )
        stage_options = dataclass_replace(
            profile_options.stage,
            materials=materials.value,
            material_mode=material_mode.value,
            merge_equivalent_materials=merge_equivalent_materials,
            normals=normals != NormalMode.NONE,
            normal_mode=cast(Any, normals.value.replace("-", "_")),
            hard_edge_angle=hard_edge_angle,
            preserve_face_boundaries=preserve_face_boundaries,
            tangents=tangents,
            tangent_uv_channel=tangent_uv_channel,
            override_tangents=override_tangents,
            validate_normals=validate_normals,
            unwrap=UnwrapOptions(
                texel_density=texel_density,
                padding=uv_padding,
                max_stretch=max_stretch,
                method=unwrap_method.value,
                iterations=unwrap_iterations,
                tolerance=unwrap_tolerance,
                sharp_to_seam=uv_sharp_to_seam,
                forbid_overlapping=uv_forbid_overlapping,
            ),
            atlas=AtlasOptions(enabled=atlas, max_size=atlas_size),
            uv0=uv0.value,
            uv1=cast(Any, uv1.value.replace("-", "_")),
            normalize_uvs=normalized_uv_channels,
        )
        merge_vertices_options = (
            MergeVerticesOptions(
                tolerance=merge_vertex_tolerance,
                preserve_normals=preserve_merge_vertex_attributes,
                preserve_tangents=preserve_merge_vertex_attributes,
                preserve_uvs=preserve_merge_vertex_attributes,
                preserve_material_boundaries=preserve_merge_vertex_material_boundaries,
                delete_degenerate=delete_merge_vertex_degenerate,
                area_epsilon=merge_vertex_area_epsilon,
            )
            if merge_vertices
            else None
        )
        delete_degenerate_polygons_options = (
            DeleteDegeneratePolygonsOptions(area_epsilon=degenerate_area_epsilon)
            if delete_degenerate_polygons
            else None
        )
        import_options = (
            pipeline_spec.import_options
            if pipeline_spec and pipeline_spec.import_options
            else _step_read_options(
                metadata,
                pmi,
                design_variants=design_variants,
                existing_meshes=import_existing_meshes,
                multi_file=multi_file_import,
                delete_free_vertices=delete_free_vertices,
                delete_lines=delete_lines,
                source_units=source_units,
                source_meters_per_unit=source_meters_per_unit,
                source_up_axis=source_up_axis.value,
                source_handedness=source_handedness.value,
                target_units=target_units,
                target_meters_per_unit=target_meters_per_unit,
                target_up_axis=None if target_up_axis is None else target_up_axis.value,
                target_handedness=None if target_handedness is None else target_handedness.value,
            )
        )
        export_metadata = (
            pipeline_spec.export_metadata
            if pipeline_spec is not None and pipeline_spec.export_metadata is not None
            else _metadata_export_options(metadata, pmi)
        )
        heal_options = (
            _brep_heal_options(
                heal_tolerance=heal_tolerance,
                remove_sliver_faces=remove_sliver_faces,
                max_sliver_area=max_sliver_area,
                fail_on_open_shells=fail_on_open_shells,
            )
            if heal_brep
            else None
        )
        merge_options = (
            MergeOptions(
                mode=cast(Any, merge_mode.value.replace("-", "_")),
                keep_parent=keep_parent,
                metadata=merge_metadata.value,
                max_vertices_per_mesh=max_vertices_per_mesh,
                region_size=region_size,
                merge_strategy=cast(Any, merge_strategy.value.replace("-", "_")),
                hierarchy_level=hierarchy_level,
            )
            if merge
            else None
        )
        explode_options = (
            ExplodeOptions(mode=cast(Any, explode.value.replace("-", "_")), metadata=merge_metadata.value)
            if explode is not None
            else None
        )
        replace_options = (
            ReplaceOptions(
                mode=cast(Any, replace.value.replace("-", "_")),
                metadata=merge_metadata.value,
                external_path=external_asset,
            )
            if replace is not None
            else None
        )
        scene_options = (
            SceneOptimizeOptions(
                batch_by_material=batch_by_material,
                merge_compatible_meshes=merge_compatible_meshes,
                split_large_meshes=split_large_meshes,
                max_vertices_per_mesh=max_vertices_per_mesh,
                index_buffer=index_buffer.value,
                flatten=flatten.value,
                remove_empty_nodes=True,
                instance_policy=instance_policy.value,
            )
            if (
                batch_by_material
                or merge_compatible_meshes
                or split_large_meshes
                or flatten != FlattenMode.SAFE
                or index_buffer != IndexBufferMode.AUTO
                or instance_policy != InstancePolicy.AUTO
            )
            else None
        )
        bake_options = (
            BakeMaterialOptions(
                maps_resolution=maps_resolution,
                force_uv_generation=force_uv_generation,
                uv_channel=0,
                padding=uv_padding,
                bake=cast(Any, bake_maps),
                merge_output=True,
            )
            if bake_materials
            else None
        )
        decimate_options = (
            DecimateOptions(
                criterion=decimate_criterion.value,
                target_triangles=target_triangles,
                target_ratio=ratio,
                surface_tolerance=surface_tolerance,
                line_tolerance=line_tolerance,
                normal_tolerance=normal_tolerance,
                uv_tolerance=uv_tolerance,
                protect_topology=protect_topology,
                budget_scope=budget_scope.value,
                uv_importance=cast(Any, uv_importance.value.replace("-", "_")),
            )
            if decimate
            else None
        )
        remove_holes_options = (
            RemoveHolesOptions(
                through="through" in enabled_hole_types,
                blind="blind" in enabled_hole_types,
                surface="surface" in enabled_hole_types,
                max_diameter=max_hole_diameter,
                prefer_brep=prefer_brep,
            )
            if remove_holes
            else None
        )
        remove_occluded_options = (
            RemoveOccludedOptions(
                strategy=occlusion_strategy.value,
                level=occlusion_level.value,
                precision=occlusion_precision,
                hemi_evaluation=hemi_evaluation,
                neighbors_preservation=neighbors_preservation,
                consider_transparency_opaque=consider_transparency_opaque,
                preserve_cavities=preserve_cavities,
                minimum_cavity_volume_m3=minimum_cavity_volume_m3,
            )
            if remove_occluded
            else None
        )
        lod_generator_options = (
            _lod_generator_options(lod_preset.value, lod_values, lod_coverages, validate_lods)
            if run_lod_generators
            else None
        )
        lod_options = _lod_options_for_cli(
            profile_options.lods,
            lod_values,
            lod_coverages,
            lod_mode.value,
            lod_per_part_budget,
            lod_drop_tiny_parts,
            lod_tiny_part_screen_size,
            validate_lods,
        )
        usd_package = "usdz" if (package == UsdPackage.USDZ or output_path.suffix.lower() == ".usdz") else "default"
        gltf_options = GltfExportOptions(
            quantize=quantize,
            meshopt=meshopt,
            draco=draco,
            texture_compression=cast(Any, texture_compression),
            file_size_budget_mb=file_size_budget_mb,
            metadata=export_metadata,
        )
        usd_options = UsdExportOptions(
            package=cast(Any, usd_package),
            file_size_budget_mb=file_size_budget_mb,
            metadata=export_metadata,
        )
        obj_options = ObjExportOptions(
            materials=obj_materials,
            write_mtl=write_mtl,
            preserve_groups=preserve_groups,
            file_size_budget_mb=file_size_budget_mb,
        )
        stl_options = StlExportOptions(binary=stl_binary, merge=stl_merge, file_size_budget_mb=file_size_budget_mb)
        asset = _convert_for_cli(
            input_path,
            output_path,
            profile=profile_options,
            pipeline=pipeline_spec,
            tessellation=tessellation,
            stage=stage_options,
            import_options=import_options,
            heal_brep=heal_options,
            merge_vertices=merge_vertices_options,
            delete_degenerate_polygons=delete_degenerate_polygons_options,
            merge=merge_options,
            explode=explode_options,
            replace=replace_options,
            scene=scene_options,
            bake_materials=bake_options,
            remove_holes=remove_holes_options,
            remove_occluded=remove_occluded_options,
            decimate=decimate_options,
            lod_generator=lod_generator_options,
            optimize=optimize_options,
            lods=lod_options,
            where=where,
            progress=_progress_callback(ctx, output_path),
            debug=debug,
            gltf_options=gltf_options,
            usd_options=usd_options,
            obj_options=obj_options,
            stl_options=stl_options,
        )
    except typer.Exit:
        raise
    except Exception as exc:
        if report is not None:
            failure_report = getattr(exc, "report", None)
            if isinstance(failure_report, Report):
                failure_report.write_json(report)
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc

    if report is not None:
        asset.report.write_json(report)
    if quality_report is not None:
        _write_tessellation_quality_report(asset, quality_report)

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
  fascat validate motor.glb
  fascat validate motor.glb --geometry-quality --report report.json
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
        "analysis_options": analyze_options.to_dict() if should_analyze else None,
        "filters": filters or [],
        "exclude_filters": exclude_filters or [],
        "report": str(report) if report else None,
    }
    where = _parse_filter_options(filters, exclude_filters, ctx, payload)
    should_analyze = should_analyze or where is not None
    payload["analysis_options"] = analyze_options.to_dict() if should_analyze else None
    _validate_export_output(output_path, ctx, payload)
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
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc
    if report is not None and analysis is not None:
        analysis.write_json(report)
    json_payload = {**payload, "stats": stats}
    if analysis is not None:
        json_payload["analysis"] = analysis.to_dict()
    message = f"{output_path}: valid {_export_label(output_path)}, {_format_stats(stats)}."
    if report is not None:
        message = f"{message} Wrote report {report}."
    if analysis is not None and "selection" in analysis.summary:
        selection = cast(dict[str, Any], analysis.summary["selection"])
        message = f"{message} Matched {_format_stats(cast(dict[str, int], selection['stats']))}."
    _emit(
        ctx,
        json_payload,
        message,
    )


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
    if ratios != sorted(ratios, reverse=True):
        _fail(ctx, payload, "--lods ratios must be sorted from highest to lowest detail.", code=2)
    return ratios


def _convert_operation_diagnostics(payload: dict[str, Any]) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []

    def add(operation: str, level: str, message: str) -> None:
        diagnostics.append({"operation": operation, "level": level, "message": message})

    add("import", "exact", "STEP import reads hierarchy, metadata, materials, and BREP handles when available")
    if payload["heal_brep"]:
        if payload["remove_sliver_faces"]:
            add(
                "heal_brep",
                "approximate",
                "BREP healing runs, but sliver-face removal is reported only because backend removal is unavailable",
            )
        else:
            add("heal_brep", "exact", "BREP sewing, edge fixing, and tolerance unification are requested")
    add("tessellate", "exact", "BREP tessellation uses the selected sag, angle, and edge cleanup settings")
    add("repair", "exact", "mesh repair applies selected cleanup operations after tessellation")
    if payload["atlas"]:
        add("atlas", "metadata_only", "atlas settings are recorded as metadata; atlas images are not written")
    add("stage", "exact", "material, normal, tangent, and UV staging options are applied before optimization")
    if payload["merge_vertices"]:
        add(
            "merge_vertices",
            "exact",
            "exact or tolerance-close vertices are merged with selected attribute and material-boundary protections",
        )
    if payload["delete_degenerate_polygons"]:
        add(
            "delete_degenerate_polygons",
            "exact",
            "degenerate polygons are removed with the requested area threshold and before/after counts",
        )
    if payload["merge"]:
        add("merge", "exact", "selected hierarchy is merged according to the requested merge mode")
    if payload["explode"] is not None:
        add("explode", "exact", "selected meshes are split by material or connected components")
    if payload["replace"] is not None:
        add("replace", "exact", "selected geometry is replaced with the requested proxy mode")
    if (
        payload["batch_by_material"]
        or payload["merge_compatible_meshes"]
        or payload["split_large_meshes"]
        or payload["flatten"] != "safe"
        or payload["index_buffer"] != "auto"
        or payload["instance_policy"] != "auto"
    ):
        add("optimize_scene", "exact", "scene batching, splitting, flattening, and instance policy options are applied")
    if payload["bake_materials"]:
        add(
            "bake_materials",
            "approximate",
            "material baking emits constant embedded texture maps from material factors, not rasterized source textures",
        )
    if payload["remove_holes"]:
        add(
            "remove_holes",
            "approximate",
            "hole removal uses mesh boundary classification and filling when BREP feature removal is unavailable",
        )
    if payload["remove_occluded"]:
        add(
            "remove_occluded",
            "approximate",
            "occlusion removal uses deterministic sampled visibility; precision controls the sample budget",
        )
    if payload["decimate"]:
        if payload["decimate_criterion"] == "quality":
            add(
                "decimate",
                "approximate",
                "quality decimation maps tolerances to a target ratio and reports measured vertex error; bounds are not enforced",
            )
        else:
            add("decimate", "exact", "target decimation applies the requested triangle budget or ratio")
    add("optimize", "exact", "profile optimization applies triangle reduction and buffer optimization settings")
    if payload["run_lod_generators"]:
        add("run_lod_generators", "exact", "preset or explicit LOD levels are generated from optimized meshes")
    elif payload["lods"] is not None:
        add("lods", "exact", "ratio-based LOD meshes are generated from optimized meshes")
    add(
        "export",
        "exact",
        "the selected writer produces the requested output format and records file-size budget warnings",
    )
    return diagnostics


def _parse_bake_maps(value: str, ctx: typer.Context, payload: dict[str, Any]) -> tuple[str, ...]:
    maps = tuple(item.strip().replace("-", "_") for item in value.split(",") if item.strip())
    allowed = {"base_color", "opacity", "normal", "roughness", "metallic", "ao", "emissive"}
    if not maps:
        _fail(ctx, payload, "--bake must include at least one map.", code=2)
    unknown = set(maps) - allowed
    if unknown:
        _fail(ctx, payload, f"Unsupported --bake maps: {', '.join(sorted(unknown))}.", code=2)
    return maps


def _parse_hole_types(value: str, ctx: typer.Context, payload: dict[str, Any]) -> tuple[str, ...]:
    hole_types = tuple(item.strip().replace("-", "_") for item in value.split(",") if item.strip())
    allowed = {"through", "blind", "surface"}
    if not hole_types:
        _fail(ctx, payload, "--hole-types must include at least one type.", code=2)
    unknown = set(hole_types) - allowed
    if unknown:
        _fail(ctx, payload, f"Unsupported --hole-types values: {', '.join(sorted(unknown))}.", code=2)
    return hole_types


def _parse_uv_channels(value: str | None, ctx: typer.Context, payload: dict[str, Any]) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        channels = tuple(dict.fromkeys(int(item.strip()) for item in value.split(",") if item.strip()))
    except ValueError as exc:
        _fail(ctx, payload, "--normalize-uvs must be a comma-separated list of UV channel indices.", code=2)
        raise AssertionError("unreachable") from exc
    if not channels:
        _fail(ctx, payload, "--normalize-uvs must include at least one UV channel.", code=2)
    if any(channel < 0 for channel in channels):
        _fail(ctx, payload, "--normalize-uvs values must be greater than or equal to 0.", code=2)
    return channels


def _parse_lod_screen_coverage(
    value: str | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> list[float] | None:
    if value is None:
        return None
    try:
        coverages = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        _fail(ctx, payload, "--lod-screen-coverage must be a comma-separated list of numbers.", code=2)
        raise AssertionError("unreachable") from exc
    if not coverages:
        _fail(ctx, payload, "--lod-screen-coverage must include at least one value.", code=2)
    if any(coverage <= 0.0 or coverage > 1.0 for coverage in coverages):
        _fail(ctx, payload, "--lod-screen-coverage values must be greater than 0 and no more than 1.", code=2)
    if coverages != sorted(coverages, reverse=True):
        _fail(ctx, payload, "--lod-screen-coverage values must be sorted from highest to lowest.", code=2)
    return coverages


def _lod_generator_options(
    preset: str,
    lod_values: list[float] | None,
    lod_coverages: list[float] | None,
    validate_lods: bool,
) -> LODGeneratorOptions:
    if lod_values is None and lod_coverages is None:
        return LODGeneratorOptions(preset=cast(Any, preset), validate=validate_lods)
    default_levels = LODGeneratorOptions(preset=cast(Any, preset), validate=validate_lods).levels
    ratios = lod_values if lod_values is not None else [level.target_ratio for level in default_levels]
    if lod_coverages is None:
        if len(ratios) == len(default_levels):
            coverages = [level.screen_coverage for level in default_levels]
        else:
            coverages = [max(0.01, 0.5 / (index + 1)) for index in range(len(ratios))]
    else:
        coverages = lod_coverages
    levels = tuple(
        LODLevel(screen_coverage=coverage, target_ratio=ratio)
        for coverage, ratio in zip(coverages, ratios, strict=True)
    )
    return LODGeneratorOptions(preset=cast(Any, preset), levels=levels, validate=validate_lods)


def _lod_options_for_cli(
    profile_lods: LODOptions | None,
    lod_values: list[float] | None,
    lod_coverages: list[float] | None,
    lod_mode: str,
    lod_per_part_budget: bool,
    lod_drop_tiny_parts: bool,
    lod_tiny_part_screen_size: float,
    validate_lods: bool,
) -> LODOptions | None:
    ratios = tuple(lod_values) if lod_values is not None else None
    if ratios is None and profile_lods is not None:
        ratios = tuple(profile_lods.ratios)
    if ratios is None:
        return None
    if lod_coverages is not None and len(lod_coverages) != len(ratios):
        raise ValueError("screen_coverage must contain one value per LOD ratio")
    return LODOptions(
        ratios=ratios,
        mode=cast(Any, lod_mode),
        screen_coverage=None if lod_coverages is None else tuple(lod_coverages),
        per_part_budget=lod_per_part_budget,
        drop_tiny_parts=lod_drop_tiny_parts,
        tiny_part_screen_size=lod_tiny_part_screen_size,
        validate=validate_lods,
    )


def _parse_filter_options(
    filters: list[str] | None,
    exclude_filters: list[str] | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> Filter | None:
    try:
        return Filter.from_cli(filters or [], exclude=exclude_filters or [])
    except FilterExpressionError as exc:
        _fail(ctx, payload, str(exc), code=2)
    except ValueError as exc:
        _fail(ctx, payload, str(exc), code=2)
    raise AssertionError("unreachable")


def _step_read_options(
    metadata: MetadataMode,
    pmi: PmiMode,
    *,
    design_variants: bool = False,
    existing_meshes: bool = True,
    multi_file: bool = False,
    delete_free_vertices: bool = False,
    delete_lines: bool = False,
    source_units: str | None = None,
    source_meters_per_unit: float | None = None,
    source_up_axis: str = "Z",
    source_handedness: str = "right",
    target_units: str | None = None,
    target_meters_per_unit: float | None = None,
    target_up_axis: str | None = None,
    target_handedness: str | None = None,
) -> StepReadOptions:
    metadata_enabled = metadata != MetadataMode.NONE
    pmi_enabled = pmi != PmiMode.NONE
    return StepReadOptions(
        metadata=metadata_enabled,
        product_metadata=metadata_enabled,
        properties=metadata_enabled,
        layers=metadata_enabled,
        validation_properties=metadata_enabled,
        pmi=pmi_enabled,
        design_variants=design_variants,
        existing_meshes=existing_meshes,
        multi_file=multi_file,
        delete_free_vertices=delete_free_vertices,
        delete_lines=delete_lines,
        source_units=source_units,
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=cast(Any, source_up_axis),
        source_handedness=cast(Any, source_handedness),
        target_units=target_units,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=cast(Any, target_up_axis),
        target_handedness=cast(Any, target_handedness),
    )


def _metadata_export_options(metadata: MetadataMode, pmi: PmiMode) -> MetadataExportOptions:
    return MetadataExportOptions(
        mode=cast(Any, metadata.value),
        pmi=cast(Any, pmi.value.replace("-", "_")),
    )


def _profile_for_cli(
    profile: Profile,
    target_device_profile: Path | None,
    ctx: typer.Context,
    payload: dict[str, Any],
) -> ConversionProfile:
    if target_device_profile is None:
        return by_name(profile.value)
    _require_existing_file(target_device_profile, "target device profile", ctx, payload)
    try:
        return profile_from_file(target_device_profile, base=profile.value)
    except Exception as exc:
        _fail(ctx, payload, f"Invalid target device profile: {exc}", code=2)
    raise AssertionError("unreachable")


def _read_pipeline_for_cli(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> PipelineSpec:
    _require_existing_file(path, "pipeline", ctx, payload)
    try:
        return PipelineSpec.from_file(path)
    except Exception as exc:
        _fail(ctx, payload, f"Invalid pipeline file: {exc}", code=2)
    raise AssertionError("unreachable")


def _brep_heal_options(
    *,
    heal_tolerance: float,
    remove_sliver_faces: bool,
    max_sliver_area: float,
    fail_on_open_shells: bool = False,
) -> BrepHealOptions:
    return BrepHealOptions(
        tolerance=heal_tolerance,
        remove_sliver_faces=remove_sliver_faces,
        max_sliver_area=max_sliver_area,
        fail_on_open_shells=fail_on_open_shells,
    )


def _write_tessellation_quality_report(asset: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset.tessellation_quality_report(), indent=2, sort_keys=True), encoding="utf-8")


def _analyze_options(
    *,
    geometry_quality: bool,
    non_manifold_edges: bool,
    open_boundaries: bool,
    self_intersections: bool,
    sliver_triangles: bool,
    tiny_parts: bool,
    draw_call_estimate: bool,
    visual_risk: bool,
) -> AnalyzeOptions:
    return AnalyzeOptions(
        non_manifold_edges=geometry_quality or non_manifold_edges,
        open_boundaries=geometry_quality or open_boundaries,
        self_intersections=geometry_quality or self_intersections,
        sliver_triangles=geometry_quality or sliver_triangles,
        tiny_parts=geometry_quality or tiny_parts,
        draw_call_estimate=geometry_quality or draw_call_estimate,
        visual_risk=geometry_quality or visual_risk,
    )


def _analysis_requested(options: AnalyzeOptions) -> bool:
    return any(
        (
            options.non_manifold_edges,
            options.open_boundaries,
            options.self_intersections,
            options.sliver_triangles,
            options.tiny_parts,
            options.draw_call_estimate,
            options.visual_risk,
        )
    )


def _metadata_summary(asset: Any) -> dict[str, int]:
    return {
        "asset": len(asset.metadata),
        "nodes": sum(len(node.metadata) for node in asset.root.walk()),
        "parts": sum(len(part.metadata) for part in asset.parts.values()),
        "materials": sum(len(material.metadata) for material in asset.materials.values()),
    }


def _pmi_summary(asset: Any) -> dict[str, int]:
    kinds: dict[str, int] = {}
    for annotation in asset.pmi:
        kinds[annotation.kind] = kinds.get(annotation.kind, 0) + 1
    return {"count": len(asset.pmi), **{f"kind_{kind}": count for kind, count in sorted(kinds.items())}}


def _validate_step_input(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if not _is_stdio(path) and path.suffix.lower() not in STEP_SUFFIXES:
        _fail(ctx, payload, f"Unsupported STEP extension: {path.suffix or '<none>'}. Use .step or .stp.", code=2)


def _validate_export_output(path: Path, ctx: typer.Context, payload: dict[str, Any]) -> None:
    if not _is_stdio(path) and path.suffix.lower() not in EXPORT_SUFFIXES:
        _fail(
            ctx,
            payload,
            "Unsupported export extension: "
            f"{path.suffix or '<none>'}. Use .usd, .usda, .usdc, .usdz, .gltf, .glb, .obj, or .stl.",
            code=2,
        )


def _is_stdio(path: Path) -> bool:
    return str(path) == "-"


def _fail(ctx: typer.Context, payload: dict[str, Any], message: str, code: int = 1) -> NoReturn:
    if _state(ctx).json_output:
        out.print_json(json.dumps({**payload, "error": message}))
    else:
        err.print(message)
    raise typer.Exit(code)


def _read_step_for_cli(
    path: Path,
    ctx: typer.Context,
    payload: dict[str, Any],
    *,
    import_options: StepReadOptions | None = None,
) -> Any:
    if _is_stdio(path):
        data = sys.stdin.buffer.read()
        if not data:
            _fail(ctx, payload, "Missing input data on stdin.")
        return read_step_bytes(data, options=import_options)
    _require_existing_file(path, "input", ctx, payload)
    try:
        return read_step(path, options=import_options)
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc


def _convert_for_cli(
    input_path: Path,
    output_path: Path,
    *,
    profile: str | ConversionProfile,
    pipeline: PipelineSpec | None,
    tessellation: Tessellation,
    stage: StageOptions,
    import_options: StepReadOptions,
    heal_brep: BrepHealOptions | None,
    merge_vertices: MergeVerticesOptions | None,
    delete_degenerate_polygons: DeleteDegeneratePolygonsOptions | None,
    merge: MergeOptions | None,
    explode: ExplodeOptions | None,
    replace: ReplaceOptions | None,
    scene: SceneOptimizeOptions | None,
    bake_materials: BakeMaterialOptions | None,
    remove_holes: RemoveHolesOptions | None,
    remove_occluded: RemoveOccludedOptions | None,
    decimate: DecimateOptions | None,
    lod_generator: LODGeneratorOptions | None,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    where: Filter | None,
    progress: Callable[[str, dict[str, int]], None] | None,
    debug: bool,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
) -> Any:
    if _is_stdio(input_path):
        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing input data on stdin.")
        with _temporary_step_file(data) as temp_input:
            return _convert_output(
                temp_input,
                output_path,
                profile,
                pipeline,
                tessellation,
                stage,
                import_options,
                heal_brep,
                merge_vertices,
                delete_degenerate_polygons,
                merge,
                explode,
                replace,
                scene,
                bake_materials,
                remove_holes,
                remove_occluded,
                decimate,
                lod_generator,
                optimize,
                lods,
                where,
                progress,
                debug,
                gltf_options,
                usd_options,
                obj_options,
                stl_options,
            )
    return _convert_output(
        input_path,
        output_path,
        profile,
        pipeline,
        tessellation,
        stage,
        import_options,
        heal_brep,
        merge_vertices,
        delete_degenerate_polygons,
        merge,
        explode,
        replace,
        scene,
        bake_materials,
        remove_holes,
        remove_occluded,
        decimate,
        lod_generator,
        optimize,
        lods,
        where,
        progress,
        debug,
        gltf_options,
        usd_options,
        obj_options,
        stl_options,
    )


def _convert_output(
    input_path: Path,
    output_path: Path,
    profile: str | ConversionProfile,
    pipeline: PipelineSpec | None,
    tessellation: Tessellation,
    stage: StageOptions,
    import_options: StepReadOptions,
    heal_brep: BrepHealOptions | None,
    merge_vertices: MergeVerticesOptions | None,
    delete_degenerate_polygons: DeleteDegeneratePolygonsOptions | None,
    merge: MergeOptions | None,
    explode: ExplodeOptions | None,
    replace: ReplaceOptions | None,
    scene: SceneOptimizeOptions | None,
    bake_materials: BakeMaterialOptions | None,
    remove_holes: RemoveHolesOptions | None,
    remove_occluded: RemoveOccludedOptions | None,
    decimate: DecimateOptions | None,
    lod_generator: LODGeneratorOptions | None,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    where: Filter | None,
    progress: Callable[[str, dict[str, int]], None] | None,
    debug: bool,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
) -> Any:
    if _is_stdio(output_path):
        import tempfile

        import click

        with tempfile.NamedTemporaryFile(suffix=".usda") as handle:
            asset = convert(
                input_path,
                handle.name,
                profile=profile,
                pipeline=pipeline,
                import_options=import_options,
                tessellation=tessellation,
                heal_brep=heal_brep,
                merge_vertices=merge_vertices,
                delete_degenerate_polygons=delete_degenerate_polygons,
                stage=stage,
                merge=merge,
                explode=explode,
                replace=replace,
                scene=scene,
                bake_materials=bake_materials,
                remove_holes=remove_holes,
                remove_occluded=remove_occluded,
                decimate=decimate,
                lod_generator=lod_generator,
                optimize=optimize,
                lods=lods,
                where=where,
                progress=progress,
                debug=debug,
                gltf_options=gltf_options,
                usd_options=usd_options,
                obj_options=obj_options,
                stl_options=stl_options,
            )
            stdout = click.get_binary_stream("stdout")
            stdout.write(Path(handle.name).read_bytes())
            stdout.flush()
            return asset
    return convert(
        input_path,
        output_path,
        profile=profile,
        pipeline=pipeline,
        import_options=import_options,
        tessellation=tessellation,
        heal_brep=heal_brep,
        merge_vertices=merge_vertices,
        delete_degenerate_polygons=delete_degenerate_polygons,
        stage=stage,
        merge=merge,
        explode=explode,
        replace=replace,
        scene=scene,
        bake_materials=bake_materials,
        remove_holes=remove_holes,
        remove_occluded=remove_occluded,
        decimate=decimate,
        lod_generator=lod_generator,
        optimize=optimize,
        lods=lods,
        where=where,
        progress=progress,
        debug=debug,
        gltf_options=gltf_options,
        usd_options=usd_options,
        obj_options=obj_options,
        stl_options=stl_options,
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


def _validate_and_analyze_output_for_cli(
    path: Path,
    options: AnalyzeOptions | None,
    *,
    where: Filter | None = None,
) -> tuple[dict[str, int], AnalysisReport | None]:
    if _is_stdio(path):
        import tempfile

        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing USD data on stdin.")
        with tempfile.NamedTemporaryFile(suffix=".usda") as handle:
            handle.write(data)
            handle.flush()
            stats = validate_export(handle.name)
            analysis = (
                analyze_output(handle.name, options, where=where, validation_stats=stats, source_path="-")
                if options is not None
                else None
            )
            return stats, analysis
    stats = validate_export(path)
    analysis = analyze_output(path, options, where=where, validation_stats=stats) if options is not None else None
    return stats, analysis


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
