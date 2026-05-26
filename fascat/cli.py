from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from difflib import get_close_matches
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, NoReturn, cast

import typer
import typer.rich_utils as rich_utils
from rich.console import Console

from fascat import __version__
from fascat.filter import Filter, FilterExpressionError
from fascat.io.gltf import GLTF_SUFFIXES
from fascat.io.step import read_step, read_step_bytes
from fascat.options import (
    AtlasOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    DecimateOptions,
    LODGeneratorOptions,
    LODLevel,
    LODOptions,
    MergeOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    SceneOptimizeOptions,
    StageOptions,
    StepReadOptions,
    Tessellation,
    UnwrapOptions,
)
from fascat.pipeline import convert
from fascat.pipeline import validate_output as validate_export
from fascat.profiles import by_name
from fascat.report import Report

DOCS_URL = "https://pavelsimo.github.io/fascat"
ISSUES_URL = "https://github.com/pavelsimo/fascat/issues"
rich_utils.MAX_WIDTH = 120
STEP_SUFFIXES = {".step", ".stp"}
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
EXPORT_SUFFIXES = USD_SUFFIXES | GLTF_SUFFIXES
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
    VIRTUAL_REALITY = "virtual-reality"


class UVMode(str, Enum):
    NONE = "none"
    BOX = "box"
    UNWRAP = "unwrap"
    LIGHTMAP = "lightmap"


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
    heal_brep: Annotated[bool, typer.Option("--heal-brep", help="Run BREP healing before inspection output.")] = False,
    heal_tolerance: Annotated[float, typer.Option("--heal-tolerance", help="BREP healing tolerance.")] = 0.05,
    remove_sliver_faces: Annotated[
        bool,
        typer.Option("--remove-sliver-faces", help="Detect tiny sliver faces during BREP healing."),
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
    _validate_step_input(input_path, ctx, payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would inspect {input_path} with profile {profile.value}.")
        return

    import_options = _step_read_options(metadata, pmi)
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
        typer.Option("--tangents", help="Generate glTF-compatible vertex tangents from UV0."),
    ] = False,
    validate_normals: Annotated[
        bool,
        typer.Option("--validate-normals", help="Validate staged normals and tangents."),
    ] = False,
    uv0: Annotated[UVMode, typer.Option("--uv0", help="UV0 generation mode.")] = UVMode.BOX,
    uv1: Annotated[UVMode, typer.Option("--uv1", help="UV1 generation mode.")] = UVMode.NONE,
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
        typer.Option("--bake-materials", help="Bake selected materials into shared texture material metadata."),
    ] = False,
    maps_resolution: Annotated[
        int,
        typer.Option("--maps-resolution", help="Bake texture resolution in pixels."),
    ] = 2048,
    force_uv_generation: Annotated[
        bool,
        typer.Option("--force-uv-generation", help="Generate UVs before material baking when needed."),
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
        typer.Option("--hemi-evaluation", help="Use hemispherical top/side occlusion evaluation metadata."),
    ] = False,
    neighbors_preservation: Annotated[
        int,
        typer.Option("--neighbors-preservation", help="Visible-neighbor preservation rings for occlusion fallback."),
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
        "sag": sag,
        "angle": angle,
        "target_triangles": target_triangles,
        "ratio": ratio,
        "max_edge_length": max_edge_length,
        "min_edge_length": min_edge_length,
        "preserve_boundaries": preserve_boundaries,
        "curvature_adaptive": curvature_adaptive,
        "avoid_skinny_triangles": avoid_skinny_triangles,
        "quality_report": str(quality_report) if quality_report else None,
        "heal_brep": heal_brep,
        "heal_tolerance": heal_tolerance,
        "remove_sliver_faces": remove_sliver_faces,
        "max_sliver_area": max_sliver_area,
        "fail_on_open_shells": fail_on_open_shells,
        "lods": None,
        "normals": normals.value,
        "preserve_face_boundaries": preserve_face_boundaries,
        "tangents": tangents,
        "validate_normals": validate_normals,
        "uv0": uv0.value,
        "uv1": uv1.value,
        "materials": materials.value,
        "material_mode": material_mode.value,
        "merge_equivalent_materials": merge_equivalent_materials,
        "texel_density": texel_density,
        "uv_padding": uv_padding,
        "max_stretch": max_stretch,
        "atlas": atlas,
        "atlas_size": atlas_size,
        "metadata": metadata.value,
        "pmi": pmi.value,
        "merge": merge,
        "merge_mode": merge_mode.value,
        "keep_parent": keep_parent,
        "merge_metadata": merge_metadata.value,
        "max_vertices_per_mesh": max_vertices_per_mesh,
        "region_size": region_size,
        "merge_strategy": merge_strategy.value,
        "hierarchy_level": hierarchy_level,
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
        "debug": debug,
        "report": str(report) if report else None,
        "force": force,
        "dry_run": state.dry_run,
    }
    where = _parse_filter_options(filters, exclude_filters, ctx, payload)
    lod_values = _parse_lods(lods, ctx, payload)
    bake_maps = _parse_bake_maps(bake, ctx, payload)
    enabled_hole_types = _parse_hole_types(hole_types, ctx, payload)
    lod_coverages = _parse_lod_screen_coverage(lod_screen_coverage, ctx, payload)
    payload["lods"] = lod_values
    payload["bake"] = list(bake_maps)
    payload["hole_types"] = list(enabled_hole_types)
    payload["lod_screen_coverage"] = lod_coverages
    _validate_step_input(input_path, ctx, payload)
    output_path = _resolve_convert_output(input_path, output_path, ctx, payload)
    payload["output"] = str(output_path)
    _validate_export_output(output_path, ctx, payload)
    if ratio is not None and (ratio <= 0.0 or ratio >= 1.0):
        _fail(ctx, payload, "--ratio must be greater than 0 and less than 1.", code=2)
    if sag is not None and sag <= 0.0:
        _fail(ctx, payload, "--sag must be greater than 0.", code=2)
    if angle is not None and (angle <= 0.0 or angle > 180.0):
        _fail(ctx, payload, "--angle must be greater than 0 and no more than 180.", code=2)
    if target_triangles is not None and target_triangles <= 0:
        _fail(ctx, payload, "--target-triangles must be greater than 0.", code=2)
    if min_edge_length is not None and min_edge_length <= 0.0:
        _fail(ctx, payload, "--min-edge-length must be greater than 0.", code=2)
    if max_edge_length is not None and max_edge_length <= 0.0:
        _fail(ctx, payload, "--max-edge-length must be greater than 0.", code=2)
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
    if run_lod_generators and lod_coverages is not None and lod_values is None:
        default_lod_count = len(LODGeneratorOptions(preset=cast(Any, lod_preset.value)).levels)
        if len(lod_coverages) != default_lod_count:
            _fail(
                ctx, payload, "--lod-screen-coverage must match the preset LOD count or be paired with --lods.", code=2
            )
    if debug and not _is_stdio(output_path) and output_path.suffix.lower() not in {".usd", ".usda"}:
        _fail(ctx, payload, "--debug requires .usd or .usda output.", code=2)
    if quality_report is not None and report is not None and quality_report.resolve() == report.resolve():
        _fail(ctx, payload, "--quality-report must use a different path than --report.", code=2)

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
            min_edge_length=min_edge_length if min_edge_length is not None else base_tessellation.min_edge_length,
            max_edge_length=max_edge_length if max_edge_length is not None else base_tessellation.max_edge_length,
            preserve_boundaries=preserve_boundaries,
            curvature_adaptive=curvature_adaptive,
            avoid_skinny_triangles=avoid_skinny_triangles,
            quality_report=quality_report is not None or base_tessellation.quality_report,
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
                preserve_hard_edges=preserve_hard_edges,
                hard_edge_angle=hard_edge_angle,
                preserve_holes=preserve_holes,
                preserve_material_boundaries=preserve_material_boundaries,
                preserve_uv_seams=preserve_uv_seams,
                preserve_small_parts=preserve_small_parts,
                small_part_triangle_threshold=small_part_triangle_threshold,
                preserve_silhouette=preserve_silhouette,
            )
        stage_options = replace(
            profile_options.stage,
            materials=materials.value,
            material_mode=material_mode.value,
            merge_equivalent_materials=merge_equivalent_materials,
            normals=normals != NormalMode.NONE,
            normal_mode=cast(Any, normals.value.replace("-", "_")),
            hard_edge_angle=hard_edge_angle,
            preserve_face_boundaries=preserve_face_boundaries,
            tangents=tangents,
            validate_normals=validate_normals,
            unwrap=UnwrapOptions(texel_density=texel_density, padding=uv_padding, max_stretch=max_stretch),
            atlas=AtlasOptions(enabled=atlas, max_size=atlas_size),
            uv0=uv0.value,
            uv1=uv1.value,
        )
        import_options = _step_read_options(metadata, pmi)
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
        lod_options = LODOptions(tuple(lod_values)) if lod_values is not None else profile_options.lods
        asset = _convert_for_cli(
            input_path,
            output_path,
            profile=profile.value,
            tessellation=tessellation,
            stage=stage_options,
            import_options=import_options,
            heal_brep=heal_options,
            merge=merge_options,
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
  fascat --json validate motor.usda
  cat motor.usdc | fascat validate -

Docs: {DOCS_URL}/reference.html""",
)
def cmd_validate(
    ctx: typer.Context,
    output_path: Annotated[
        Path,
        typer.Argument(help="Generated USD or glTF file to validate, or '-' for USD stdin.", allow_dash=True),
    ],
) -> None:
    """Validate a generated USD or glTF file."""
    state = _state(ctx)
    payload = {
        "command": "validate",
        "output": str(output_path),
        "dry_run": state.dry_run,
    }
    _validate_export_output(output_path, ctx, payload)
    if state.dry_run:
        _emit(ctx, payload, f"Would validate {output_path}.")
        return

    _require_existing_file(output_path, "output", ctx, payload)
    try:
        stats = _validate_output_for_cli(output_path)
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc
    _emit(
        ctx, {**payload, "stats": stats}, f"{output_path}: valid {_export_label(output_path)}, {_format_stats(stats)}."
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


def _step_read_options(metadata: MetadataMode, pmi: PmiMode) -> StepReadOptions:
    metadata_enabled = metadata != MetadataMode.NONE
    pmi_enabled = pmi != PmiMode.NONE
    return StepReadOptions(
        metadata=metadata_enabled,
        product_metadata=metadata_enabled,
        properties=metadata_enabled,
        layers=metadata_enabled,
        validation_properties=metadata_enabled,
        pmi=pmi_enabled,
    )


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
            f"Unsupported export extension: {path.suffix or '<none>'}. Use .usd, .usda, .usdc, .gltf, or .glb.",
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
    profile: str,
    tessellation: Tessellation,
    stage: StageOptions,
    import_options: StepReadOptions,
    heal_brep: BrepHealOptions | None,
    merge: MergeOptions | None,
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
                tessellation,
                stage,
                import_options,
                heal_brep,
                merge,
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
            )
    return _convert_output(
        input_path,
        output_path,
        profile,
        tessellation,
        stage,
        import_options,
        heal_brep,
        merge,
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
    )


def _convert_output(
    input_path: Path,
    output_path: Path,
    profile: str,
    tessellation: Tessellation,
    stage: StageOptions,
    import_options: StepReadOptions,
    heal_brep: BrepHealOptions | None,
    merge: MergeOptions | None,
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
) -> Any:
    if _is_stdio(output_path):
        import tempfile

        import click

        with tempfile.NamedTemporaryFile(suffix=".usda") as handle:
            asset = convert(
                input_path,
                handle.name,
                profile=profile,
                import_options=import_options,
                tessellation=tessellation,
                heal_brep=heal_brep,
                stage=stage,
                merge=merge,
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
            )
            stdout = click.get_binary_stream("stdout")
            stdout.write(Path(handle.name).read_bytes())
            stdout.flush()
            return asset
    return convert(
        input_path,
        output_path,
        profile=profile,
        import_options=import_options,
        tessellation=tessellation,
        heal_brep=heal_brep,
        stage=stage,
        merge=merge,
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


def _validate_output_for_cli(path: Path) -> dict[str, int]:
    if _is_stdio(path):
        import tempfile

        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing USD data on stdin.")
        with tempfile.NamedTemporaryFile(suffix=".usda") as handle:
            handle.write(data)
            handle.flush()
            return validate_export(handle.name)
    return validate_export(path)


def _export_label(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in GLTF_SUFFIXES:
        return "glTF"
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
