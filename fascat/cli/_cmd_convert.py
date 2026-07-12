from __future__ import annotations

from dataclasses import replace as dataclass_replace
from pathlib import Path
from typing import Annotated, Any, cast

import typer

from fascat.io._suffixes import (
    JT_SUFFIXES,
    STEP_SUFFIXES,
)
from fascat.options import (
    AabbProjectionOptions,
    AtlasOptions,
    BakeMaterialOptions,
    DecimateOptions,
    DeleteDegeneratePolygonsOptions,
    ExplodeOptions,
    FbxExportOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    MergeOptions,
    MergeVerticesOptions,
    ObjExportOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    ReplaceOptions,
    SceneOptimizeOptions,
    StlExportOptions,
    UnwrapOptions,
    UsdExportOptions,
    default_jobs,
)
from fascat.report import Report

from . import _io_helpers
from ._app import DOCS_URL, _state, app
from ._enums import (
    AabbProjectionScope,
    AxisMode,
    BudgetScope,
    ConstructionCurvePolicyMode,
    DecimateCriterion,
    ExplodeMode,
    ExportPreset,
    FlattenMode,
    HandednessMode,
    IndexBufferMode,
    InstancePolicy,
    JtLodSelectionMode,
    LODEngineProfile,
    LODMode,
    LODPreset,
    LODSourceMode,
    MaterialLibraryColorSpaceMode,
    MaterialMode,
    MaterialPipelineMode,
    MergeMetadata,
    MergeMode,
    MergeStrategy,
    MetadataMode,
    NormalMode,
    NormalWeighting,
    OcclusionLevel,
    OcclusionStrategy,
    PmiMode,
    Profile,
    ReplaceMode,
    StdoutFormat,
    UnwrapMethod,
    UsdLayout,
    UsdPackage,
    UV0Mode,
    UV1Mode,
    UVImportance,
)
from ._io_helpers import _write_tessellation_quality_report
from ._output import (
    _emit,
    _fail,
    _format_stats,
    _interrupt,
    _is_stdio,
    _print_report_warnings,
    _print_verbose_operation_diagnostics,
    _require_existing_file,
    _stage_reporter,
)
from ._params import (
    _brep_heal_options,
    _convert_operation_diagnostics,
    _decimate_target_for_cli,
    _lod_generator_options,
    _lod_options_for_cli,
    _metadata_export_options,
    _parse_bake_maps,
    _parse_decimate_cleanup_attributes,
    _parse_filter_options,
    _parse_hole_types,
    _parse_lod_screen_coverage,
    _parse_lods,
    _parse_uv_channels,
    _profile_for_cli,
    _read_pipeline_for_cli,
    _resolve_convert_output,
    _step_read_options,
    _validate_cad_input,
    _validate_export_output,
)


@app.command(
    "convert",
    epilog=f"""Examples:
  fascat convert motor.step motor.usdc
  fascat convert legacy.igs legacy.glb
  fascat convert native.brep native.usdc
  fascat convert motor.step motor.glb --profile virtual-reality
  fascat convert motor.step motor.glb --pipeline realtime.toml
  fascat convert motor.step
  fascat convert motor.step motor.usda --debug --report report.json
  fascat --dry-run --json convert motor.step motor.usdc
  cat motor.step | fascat convert - - --stdout-format glb --profile realtime-web

Docs: {DOCS_URL}/reference.html""",
)
def cmd_convert(
    ctx: typer.Context,
    input_path: Annotated[Path, typer.Argument(help="Input CAD file, or '-' for stdin STEP.", allow_dash=True)],
    output_path: Annotated[
        Path | None,
        typer.Argument(
            help="Output USD, glTF, OBJ, STL, or FBX file, or '-' for stdout. Defaults to input .usdc.",
            allow_dash=True,
        ),
    ] = None,
    stdout_format: Annotated[
        StdoutFormat,
        typer.Option("--stdout-format", help="Format to use when output is '-'."),
    ] = StdoutFormat.USDA,
    extra_inputs: Annotated[
        list[Path] | None,
        typer.Option("--input", help="Additional STEP root input for explicit multi-root conversion."),
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
    max_triangles_per_part: Annotated[
        int | None,
        typer.Option("--max-triangles-per-part", help="Fail if tessellation produces more triangles for any part."),
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
    detail_adaptive: Annotated[
        bool,
        typer.Option(
            "--detail-adaptive",
            help="Auto-tighten tessellation for shiny or high-detail material/metadata parts.",
        ),
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
    lod_source: Annotated[
        LODSourceMode | None,
        typer.Option("--lod-source", help="LOD chain policy: imported, generated, or auto."),
    ] = None,
    lod_mode: Annotated[
        LODMode,
        typer.Option("--lod-mode", help="LOD output mode: variants, extras, or separate."),
    ] = LODMode.VARIANTS,
    lod_engine_profile: Annotated[
        LODEngineProfile,
        typer.Option("--lod-engine-profile", help="LOD export profile: generic, unity, or unreal."),
    ] = LODEngineProfile.GENERIC,
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
    normal_weighting: Annotated[
        NormalWeighting,
        typer.Option("--normal-weighting", help="Normal averaging weights for smooth or hard-edge normals."),
    ] = NormalWeighting.AREA,
    preserve_face_boundaries: Annotated[
        bool,
        typer.Option("--preserve-face-boundaries", help="Treat CAD face-group boundaries as hard normal edges."),
    ] = False,
    override_normals: Annotated[
        bool,
        typer.Option(
            "--override-normals/--preserve-normals",
            help="Regenerate existing normals instead of preserving them during staging.",
        ),
    ] = True,
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
    uv_aabb_scope: Annotated[
        AabbProjectionScope,
        typer.Option("--uv-aabb-scope", help="AABB projection scope for box UV generation: local or shared."),
    ] = AabbProjectionScope.LOCAL,
    uv3d_size: Annotated[
        float | None,
        typer.Option("--uv3d-size", help="World-space size per UV tile for box/AABB projection."),
    ] = None,
    uv_override_existing: Annotated[
        bool,
        typer.Option(
            "--uv-override-existing/--uv-preserve-existing",
            help="Override existing UV channels when box/AABB projection is requested.",
        ),
    ] = True,
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
    merge_vertex_quality_report: Annotated[
        bool,
        typer.Option(
            "--merge-vertex-quality-report",
            help="Run heavy merge-vertex candidate, topology, and near-duplicate diagnostics.",
        ),
    ] = False,
    merge_vertex_area_epsilon: Annotated[
        float | None,
        typer.Option(
            "--merge-vertex-area-epsilon",
            help="Area threshold for degenerate polygons after merging (default: derived from the mesh bounding box).",
        ),
    ] = None,
    delete_degenerate_polygons: Annotated[
        bool,
        typer.Option("--delete-degenerate-polygons", help="Run standalone degenerate polygon cleanup."),
    ] = False,
    degenerate_area_epsilon: Annotated[
        float | None,
        typer.Option(
            "--degenerate-area-epsilon",
            help=(
                "Area threshold for standalone degenerate polygon cleanup "
                "(default: derived from the mesh bounding box)."
            ),
        ),
    ] = None,
    delete_duplicate_polygons: Annotated[
        bool,
        typer.Option(
            "--delete-duplicate-polygons/--keep-duplicate-polygons",
            help="Delete exact duplicate polygons during standalone degenerate polygon cleanup.",
        ),
    ] = True,
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
    lod_selection: Annotated[
        JtLodSelectionMode | None,
        typer.Option("--lod-selection", help="Import the finest stored LOD or all stored LODs."),
    ] = None,
    jt_lod_selection: Annotated[
        JtLodSelectionMode | None,
        typer.Option(
            "--jt-lod-selection",
            help="Deprecated alias for --lod-selection.",
        ),
    ] = None,
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
    instance_similarity_tolerance: Annotated[
        float,
        typer.Option(
            "--instance-similarity-tolerance",
            help="Position tolerance for reconstructing similar mesh instances.",
        ),
    ] = 0.0,
    bake_materials: Annotated[
        bool,
        typer.Option("--bake-materials", help="Create a shared baked material with raster atlas textures."),
    ] = False,
    maps_resolution: Annotated[
        int,
        typer.Option("--maps-resolution", help="Requested bake texture resolution metadata in pixels."),
    ] = 2048,
    lightmap_resolution: Annotated[
        int,
        typer.Option("--lightmap-resolution", help="Resolution used for generated bake/lightmap UV packing."),
    ] = 1024,
    force_uv_generation: Annotated[
        bool,
        typer.Option("--force-uv-generation", help="Generate UVs before recording baked material textures."),
    ] = False,
    bake: Annotated[
        str,
        typer.Option("--bake", help="Comma-separated material maps to bake, for example base-color,opacity."),
    ] = "base-color",
    ambient_occlusion_strategy: Annotated[
        OcclusionStrategy,
        typer.Option(
            "--ambient-occlusion-strategy",
            help="AO sampling directions for baked AO maps and decimation AO protection.",
        ),
    ] = OcclusionStrategy.CONSERVATIVE,
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
    decimate_iterative_threshold: Annotated[
        int,
        typer.Option(
            "--decimate-iterative-threshold",
            help="Source triangle threshold above which explicit decimation runs intermediate passes.",
        ),
    ] = 1_000_000,
    protect_topology: Annotated[
        bool,
        typer.Option("--protect-topology/--no-protect-topology", help="Preserve topology-sensitive faces."),
    ] = True,
    preserve_painted_areas: Annotated[
        bool,
        typer.Option("--preserve-painted-areas", help="Preserve painted or protected face groups during decimation."),
    ] = False,
    preserve_ambient_occlusion: Annotated[
        bool,
        typer.Option("--preserve-ambient-occlusion", help="Preserve low-AO faces during decimation."),
    ] = False,
    budget_scope: Annotated[
        BudgetScope,
        typer.Option("--budget-scope", help="Decimation budget scope: part or selection."),
    ] = BudgetScope.SELECTION,
    uv_importance: Annotated[
        UVImportance,
        typer.Option("--uv-importance", help="Decimation UV importance: preserve-islands, preserve-seams, or ignore."),
    ] = UVImportance.PRESERVE_ISLANDS,
    decimate_cleanup_attributes: Annotated[
        str,
        typer.Option(
            "--decimate-cleanup-attributes",
            help="Comma-separated pre-decimation cleanup attributes: unused-uvs,tangents.",
        ),
    ] = "",
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
    jobs: Annotated[
        int | None,
        typer.Option(
            "--jobs",
            help="Worker count for independent per-part mesh operations (default: min(4, CPU count)).",
        ),
    ] = None,
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
    ] = 45.0,
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
    export_preset: Annotated[
        ExportPreset | None,
        typer.Option("--export-preset", help="glTF export preset: desktop, web, mobile, vr, or ar."),
    ] = None,
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
        typer.Option("--draco", help="Compress glTF geometry with KHR_draco_mesh_compression."),
    ] = False,
    draco_compression_level: Annotated[
        int,
        typer.Option("--draco-compression-level", help="Draco compression level, 0 (fastest) to 10 (smallest)."),
    ] = 5,
    draco_quantize_position: Annotated[
        int,
        typer.Option("--draco-quantize-position", help="Draco position quantization bits (1-30)."),
    ] = 14,
    draco_quantize_normal: Annotated[
        int,
        typer.Option("--draco-quantize-normal", help="Draco normal quantization bits (1-30)."),
    ] = 10,
    draco_quantize_texcoord: Annotated[
        int,
        typer.Option("--draco-quantize-texcoord", help="Draco texcoord quantization bits (1-30)."),
    ] = 12,
    draco_quantize_color: Annotated[
        int,
        typer.Option("--draco-quantize-color", help="Draco color quantization bits (1-30)."),
    ] = 8,
    texture_compression: Annotated[
        str | None,
        typer.Option("--texture-compression", help="Compress glTF textures with KTX2/Basis: ktx2 or basisu."),
    ] = None,
    ktx2_quality: Annotated[
        int,
        typer.Option("--ktx2-quality", help="KTX2/Basis encoder quality level (0-255)."),
    ] = 128,
    ktx2_effort: Annotated[
        int,
        typer.Option("--ktx2-effort", help="KTX2/Basis encoder compression effort (0-6)."),
    ] = 2,
    ktx2_uastc: Annotated[
        bool | None,
        typer.Option(
            "--ktx2-uastc/--ktx2-etc1s",
            help="Force UASTC or ETC1S encoding (default: derived from --texture-compression).",
        ),
    ] = None,
    texture_fallback_format: Annotated[
        str,
        typer.Option(
            "--texture-fallback-format",
            help="Fallback texture format when KTX2/Basis compression is not requested: auto, png, or jpeg.",
        ),
    ] = "auto",
    png_compression: Annotated[
        int,
        typer.Option("--png-compression", help="PNG fallback compression level from 0 to 9."),
    ] = 6,
    jpeg_quality: Annotated[
        int,
        typer.Option("--jpeg-quality", help="JPEG fallback quality from 0 to 100."),
    ] = 85,
    package: Annotated[
        UsdPackage,
        typer.Option("--package", help="USD package mode: default or usdz."),
    ] = UsdPackage.DEFAULT,
    usd_layout: Annotated[
        UsdLayout,
        typer.Option(
            "--usd-layout",
            help="USD scene layout: auto (flat for realtime-web, instanced otherwise), "
            "instanced (prototypes, internal references, LOD variants), or flat "
            "(inline meshes per occurrence for maximum viewer compatibility).",
        ),
    ] = UsdLayout.AUTO,
    file_size_budget_mb: Annotated[
        float | None,
        typer.Option("--file-size-budget-mb", help="Warn in reports when output exceeds this size."),
    ] = None,
    size_ladder: Annotated[
        bool,
        typer.Option("--size-ladder/--no-size-ladder", help="Measure baseline and compressed temporary GLB sizes."),
    ] = False,
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
    fbx_materials: Annotated[
        bool,
        typer.Option("--fbx-materials/--no-fbx-materials", help="Write FBX material nodes and connections."),
    ] = True,
    fbx_normals: Annotated[
        bool,
        typer.Option("--fbx-normals/--no-fbx-normals", help="Write FBX normal layers."),
    ] = True,
    fbx_tangents: Annotated[
        bool,
        typer.Option("--fbx-tangents/--no-fbx-tangents", help="Write FBX tangent layers when available."),
    ] = True,
    fbx_uvs: Annotated[
        bool,
        typer.Option("--fbx-uvs/--no-fbx-uvs", help="Write FBX UV layers when available."),
    ] = True,
    debug: Annotated[bool, typer.Option("--debug", help="Prefer debuggable USDA output conventions.")] = False,
    report: Annotated[Path | None, typer.Option("--report", help="Write a JSON conversion report sidecar.")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite an existing output file.")] = False,
) -> None:
    """Convert a CAD file into a realtime-ready OpenUSD or glTF asset."""
    state = _state(ctx)
    input_paths = [input_path, *(extra_inputs or [])]
    payload: dict[str, Any] = {
        "command": "convert",
        "input": str(input_path) if len(input_paths) == 1 else [str(path) for path in input_paths],
        "extra_inputs": [str(path) for path in input_paths[1:]],
        "output": str(output_path) if output_path is not None else None,
        "stdout_format": stdout_format.value,
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
        "max_triangles_per_part": max_triangles_per_part,
        "min_edge_length": min_edge_length,
        "preserve_boundaries": preserve_boundaries,
        "curvature_adaptive": curvature_adaptive,
        "detail_adaptive": detail_adaptive,
        "avoid_skinny_triangles": avoid_skinny_triangles,
        "quality_report": str(quality_report) if quality_report else None,
        "free_edge_report": free_edge_report,
        "reuse_existing_meshes": reuse_existing_meshes,
        "heal_brep": heal_brep,
        "heal_tolerance": heal_tolerance,
        "group_open_shells": group_open_shells,
        "cleanup_overlapping_faces": cleanup_overlapping_faces,
        "overlap_area_ratio": overlap_area_ratio,
        "remove_sliver_faces": remove_sliver_faces,
        "max_sliver_area": max_sliver_area,
        "fail_on_open_shells": fail_on_open_shells,
        "lods": None,
        "lod_source": None if lod_source is None else lod_source.value,
        "lod_selection": None if lod_selection is None else lod_selection.value,
        "jt_lod_selection": None if jt_lod_selection is None else jt_lod_selection.value,
        "lod_mode": lod_mode.value,
        "lod_engine_profile": lod_engine_profile.value,
        "lod_per_part_budget": lod_per_part_budget,
        "lod_drop_tiny_parts": lod_drop_tiny_parts,
        "lod_tiny_part_screen_size": lod_tiny_part_screen_size,
        "normals": normals.value,
        "normal_weighting": normal_weighting.value,
        "preserve_face_boundaries": preserve_face_boundaries,
        "override_normals": override_normals,
        "tangents": tangents,
        "tangent_uv_channel": tangent_uv_channel,
        "override_tangents": override_tangents,
        "validate_normals": validate_normals,
        "uv0": uv0.value,
        "uv1": uv1.value,
        "uv_aabb_scope": uv_aabb_scope.value,
        "uv3d_size": uv3d_size,
        "uv_override_existing": uv_override_existing,
        "normalize_uvs": normalize_uvs,
        "materials": materials.value,
        "material_mode": material_mode.value,
        "merge_equivalent_materials": merge_equivalent_materials,
        "merge_vertices": merge_vertices,
        "merge_vertex_tolerance": merge_vertex_tolerance,
        "preserve_merge_vertex_attributes": preserve_merge_vertex_attributes,
        "preserve_merge_vertex_material_boundaries": preserve_merge_vertex_material_boundaries,
        "delete_merge_vertex_degenerate": delete_merge_vertex_degenerate,
        "merge_vertex_quality_report": merge_vertex_quality_report,
        "merge_vertex_area_epsilon": merge_vertex_area_epsilon,
        "delete_degenerate_polygons": delete_degenerate_polygons,
        "degenerate_area_epsilon": degenerate_area_epsilon,
        "delete_duplicate_polygons": delete_duplicate_polygons,
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
        "instance_similarity_tolerance": instance_similarity_tolerance,
        "bake_materials": bake_materials,
        "maps_resolution": maps_resolution,
        "lightmap_resolution": lightmap_resolution,
        "force_uv_generation": force_uv_generation,
        "bake": bake,
        "ambient_occlusion_strategy": ambient_occlusion_strategy.value,
        "decimate": decimate,
        "decimate_criterion": decimate_criterion.value,
        "surface_tolerance": surface_tolerance,
        "line_tolerance": line_tolerance,
        "normal_tolerance": normal_tolerance,
        "uv_tolerance": uv_tolerance,
        "decimate_iterative_threshold": decimate_iterative_threshold,
        "protect_topology": protect_topology,
        "preserve_painted_areas": preserve_painted_areas,
        "preserve_ambient_occlusion": preserve_ambient_occlusion,
        "budget_scope": budget_scope.value,
        "uv_importance": uv_importance.value,
        "decimate_cleanup_attributes": decimate_cleanup_attributes,
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
        "jobs": jobs,
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
        "export_preset": None if export_preset is None else export_preset.value,
        "quantize": quantize,
        "meshopt": meshopt,
        "draco": draco,
        "texture_compression": texture_compression,
        "texture_fallback_format": texture_fallback_format,
        "png_compression": png_compression,
        "jpeg_quality": jpeg_quality,
        "package": package.value,
        "usd_layout": usd_layout.value,
        "file_size_budget_mb": file_size_budget_mb,
        "size_ladder": size_ladder,
        "obj_materials": obj_materials,
        "write_mtl": write_mtl,
        "preserve_groups": preserve_groups,
        "stl_binary": stl_binary,
        "stl_merge": stl_merge,
        "fbx_materials": fbx_materials,
        "fbx_normals": fbx_normals,
        "fbx_tangents": fbx_tangents,
        "fbx_uvs": fbx_uvs,
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
    if lod_selection is not None and jt_lod_selection is not None and lod_selection != jt_lod_selection:
        _fail(ctx, payload, "--lod-selection and --jt-lod-selection must not conflict.", code=2)
    if jt_lod_selection is not None:
        from ._app import err

        err.print("warning: --jt-lod-selection is deprecated; use --lod-selection instead.", style="yellow")
    if (lod_selection is not None or jt_lod_selection is not None) and all(
        not _is_stdio(path) and path.suffix.lower() != ".jt" for path in input_paths
    ):
        from ._app import err

        err.print("warning: --lod-selection is ignored for non-JT inputs.", style="yellow")
    effective_lod_selection = lod_selection or jt_lod_selection or JtLodSelectionMode.FINEST
    if lod_source == LODSourceMode.IMPORTED and lod_values is not None:
        _fail(ctx, payload, "--lod-source imported cannot be combined with --lods.", code=2)
    bake_maps = _parse_bake_maps(bake, ctx, payload)
    enabled_hole_types = _parse_hole_types(hole_types, ctx, payload)
    cleanup_attributes = _parse_decimate_cleanup_attributes(decimate_cleanup_attributes, ctx, payload)
    lod_coverages = _parse_lod_screen_coverage(lod_screen_coverage, ctx, payload)
    normalized_uv_channels = _parse_uv_channels(normalize_uvs, ctx, payload)
    if tangent_uv_channel < 0:
        _fail(ctx, payload, "--tangent-uv-channel must be greater than or equal to 0.", code=2)
    payload["lods"] = lod_values
    payload["bake"] = list(bake_maps)
    payload["hole_types"] = list(enabled_hole_types)
    payload["decimate_cleanup_attributes"] = list(cleanup_attributes)
    payload["lod_screen_coverage"] = lod_coverages
    payload["normalize_uvs"] = list(normalized_uv_channels)
    for path in input_paths:
        _validate_cad_input(path, ctx, payload)
    if len(input_paths) > 1:
        if any(_is_stdio(path) for path in input_paths):
            _fail(ctx, payload, "Explicit multi-root CLI import does not support stdin.", code=2)
        if any(path.suffix.lower() not in STEP_SUFFIXES for path in input_paths):
            _fail(ctx, payload, "Explicit multi-root CLI import currently supports only STEP inputs.", code=2)
    output_path = _resolve_convert_output(input_path, output_path, ctx, payload)
    payload["output"] = str(output_path)
    _validate_export_output(output_path, ctx, payload)
    effective_output_suffix = f".{stdout_format.value}" if _is_stdio(output_path) else output_path.suffix.lower()
    payload["effective_output_suffix"] = effective_output_suffix
    if not _is_stdio(output_path) and stdout_format != StdoutFormat.USDA:
        _fail(ctx, payload, "--stdout-format can only be used when output is '-'.", code=2)
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
    if max_triangles_per_part is not None and max_triangles_per_part <= 0:
        _fail(ctx, payload, "--max-triangles-per-part must be greater than 0.", code=2)
    if min_edge_length is not None and max_edge_length is not None and min_edge_length > max_edge_length:
        _fail(ctx, payload, "--min-edge-length must be less than or equal to --max-edge-length.", code=2)
    if heal_tolerance <= 0.0:
        _fail(ctx, payload, "--heal-tolerance must be greater than 0.", code=2)
    if overlap_area_ratio <= 0.0 or overlap_area_ratio > 1.0:
        _fail(ctx, payload, "--overlap-area-ratio must be greater than 0 and no more than 1.", code=2)
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
    if instance_similarity_tolerance < 0.0:
        _fail(ctx, payload, "--instance-similarity-tolerance must be greater than or equal to 0.", code=2)
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
    if uv3d_size is not None and uv3d_size <= 0.0:
        _fail(ctx, payload, "--uv3d-size must be greater than 0.", code=2)
    if merge_vertex_tolerance < 0.0:
        _fail(ctx, payload, "--merge-vertex-tolerance must be greater than or equal to 0.", code=2)
    if merge_vertex_area_epsilon is not None and merge_vertex_area_epsilon < 0.0:
        _fail(ctx, payload, "--merge-vertex-area-epsilon must be greater than or equal to 0.", code=2)
    if degenerate_area_epsilon is not None and degenerate_area_epsilon < 0.0:
        _fail(ctx, payload, "--degenerate-area-epsilon must be greater than or equal to 0.", code=2)
    if unwrap_iterations is not None and unwrap_iterations <= 0:
        _fail(ctx, payload, "--unwrap-iterations must be greater than 0.", code=2)
    if unwrap_tolerance is not None and unwrap_tolerance < 0.0:
        _fail(ctx, payload, "--unwrap-tolerance must be greater than or equal to 0.", code=2)
    if atlas_size <= 0:
        _fail(ctx, payload, "--atlas-size must be greater than 0.", code=2)
    if maps_resolution <= 0:
        _fail(ctx, payload, "--maps-resolution must be greater than 0.", code=2)
    if lightmap_resolution <= 0:
        _fail(ctx, payload, "--lightmap-resolution must be greater than 0.", code=2)
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
    if construction_curve_tube_radius <= 0.0:
        _fail(ctx, payload, "--construction-curve-tube-radius must be greater than 0.", code=2)
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
    if jobs is not None and jobs < 1:
        _fail(ctx, payload, "--jobs must be greater than or equal to 1.", code=2)
    if draco_compression_level < 0 or draco_compression_level > 10:
        _fail(ctx, payload, "--draco-compression-level must be between 0 and 10.", code=2)
    if ktx2_quality < 0 or ktx2_quality > 255:
        _fail(ctx, payload, "--ktx2-quality must be between 0 and 255.", code=2)
    if ktx2_effort < 0 or ktx2_effort > 6:
        _fail(ctx, payload, "--ktx2-effort must be between 0 and 6.", code=2)
    for draco_bits_flag, draco_bits in (
        ("--draco-quantize-position", draco_quantize_position),
        ("--draco-quantize-normal", draco_quantize_normal),
        ("--draco-quantize-texcoord", draco_quantize_texcoord),
        ("--draco-quantize-color", draco_quantize_color),
    ):
        if draco_bits < 1 or draco_bits > 30:
            _fail(ctx, payload, f"{draco_bits_flag} must be between 1 and 30.", code=2)
    if jobs is None:
        jobs = default_jobs()
    if texture_compression not in {None, "ktx2", "basisu"}:
        _fail(ctx, payload, "--texture-compression must be one of: ktx2, basisu.", code=2)
    texture_fallback_format = texture_fallback_format.replace("-", "_")
    payload["texture_fallback_format"] = texture_fallback_format
    if texture_fallback_format not in {"auto", "png", "jpeg"}:
        _fail(ctx, payload, "--texture-fallback-format must be one of: auto, png, jpeg.", code=2)
    if png_compression < 0 or png_compression > 9:
        _fail(ctx, payload, "--png-compression must be between 0 and 9.", code=2)
    if jpeg_quality < 0 or jpeg_quality > 100:
        _fail(ctx, payload, "--jpeg-quality must be between 0 and 100.", code=2)
    if file_size_budget_mb is not None and file_size_budget_mb <= 0.0:
        _fail(ctx, payload, "--file-size-budget-mb must be greater than 0.", code=2)
    if package == UsdPackage.USDZ and effective_output_suffix != ".usdz":
        _fail(ctx, payload, "--package usdz requires a .usdz output path.", code=2)
    if debug and effective_output_suffix not in {".usd", ".usda"}:
        _fail(ctx, payload, "--debug requires .usd or .usda output.", code=2)
    if quality_report is not None and report is not None and quality_report.resolve() == report.resolve():
        _fail(ctx, payload, "--quality-report must use a different path than --report.", code=2)

    profile_options = _profile_for_cli(profile, target_device_profile, ctx, payload)
    profile_options = dataclass_replace(profile_options, repair=dataclass_replace(profile_options.repair, jobs=jobs))
    effective_file_size_budget_mb = (
        file_size_budget_mb
        if file_size_budget_mb is not None
        else (None if profile_options.budget is None else profile_options.budget.max_file_size_mb)
    )
    payload["profile"] = profile_options.name
    if target_device_profile is not None:
        payload["base_profile"] = profile.value
        payload["profile_options"] = profile_options.to_dict()
    decimate_target_triangles, decimate_target_source = _decimate_target_for_cli(
        decimate=decimate,
        criterion=decimate_criterion.value,
        requested_target=target_triangles,
        requested_ratio=ratio,
        profile_options=profile_options,
    )
    payload["decimate_target_triangles"] = decimate_target_triangles
    payload["decimate_target_source"] = decimate_target_source

    payload["operation_diagnostics"] = _convert_operation_diagnostics(payload)
    if state.dry_run:
        input_label = ", ".join(str(path) for path in input_paths)
        _emit(ctx, payload, f"Would convert {input_label} to {output_path} with profile {profile_options.name}.")
        _print_verbose_operation_diagnostics(ctx, payload)
        return

    for path in input_paths:
        _require_existing_file(path, "input", ctx, payload)
    if not _is_stdio(output_path) and output_path.exists() and not force:
        _fail(ctx, payload, f"Output already exists: {output_path}. Pass --force to overwrite.")

    try:
        base_tessellation = profile_options.tessellation
        if base_tessellation is None:
            _fail(ctx, payload, "The inspect-only profile cannot be used for conversion.", code=2)
        tessellation = dataclass_replace(
            base_tessellation,
            sag=sag if sag is not None else base_tessellation.sag,
            sag_ratio=sag_ratio
            if sag_ratio is not None
            else (None if sag is not None else base_tessellation.sag_ratio),
            angle=angle if angle is not None else base_tessellation.angle,
            min_edge_length=min_edge_length if min_edge_length is not None else base_tessellation.min_edge_length,
            max_edge_length=max_edge_length if max_edge_length is not None else base_tessellation.max_edge_length,
            max_polygon_length=max_polygon_length
            if max_polygon_length is not None
            else base_tessellation.max_polygon_length,
            preserve_boundaries=preserve_boundaries,
            curvature_adaptive=curvature_adaptive,
            detail_adaptive=detail_adaptive or base_tessellation.detail_adaptive,
            avoid_skinny_triangles=avoid_skinny_triangles,
            quality_report=quality_report is not None or base_tessellation.quality_report,
            free_edge_report=free_edge_report or base_tessellation.free_edge_report,
            reuse_existing_meshes=reuse_existing_meshes,
            max_triangles_per_part=max_triangles_per_part
            if max_triangles_per_part is not None
            else base_tessellation.max_triangles_per_part,
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
                jobs=jobs,
            )
        stage_options = dataclass_replace(
            profile_options.stage,
            materials=materials.value,
            material_mode=material_mode.value,
            merge_equivalent_materials=merge_equivalent_materials,
            normals=normals != NormalMode.NONE,
            normal_mode=cast(Any, normals.value.replace("-", "_")),
            normal_weighting=normal_weighting.value,
            hard_edge_angle=hard_edge_angle,
            preserve_face_boundaries=preserve_face_boundaries,
            override_normals=override_normals,
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
            aabb_projection=AabbProjectionOptions(
                scope=uv_aabb_scope.value,
                uv3d_size=uv3d_size,
                override_existing=uv_override_existing,
            ),
            uv0=uv0.value,
            uv1=cast(Any, uv1.value.replace("-", "_")),
            normalize_uvs=normalized_uv_channels,
            jobs=jobs,
        )
        merge_vertices_options = (
            MergeVerticesOptions(
                tolerance=merge_vertex_tolerance,
                preserve_normals=preserve_merge_vertex_attributes,
                preserve_tangents=preserve_merge_vertex_attributes,
                preserve_uvs=preserve_merge_vertex_attributes,
                preserve_material_boundaries=preserve_merge_vertex_material_boundaries,
                delete_degenerate=delete_merge_vertex_degenerate,
                quality_report=merge_vertex_quality_report,
                area_epsilon=merge_vertex_area_epsilon,
                jobs=jobs,
            )
            if merge_vertices
            else None
        )
        delete_degenerate_polygons_options = (
            DeleteDegeneratePolygonsOptions(
                area_epsilon=degenerate_area_epsilon,
                delete_duplicates=delete_duplicate_polygons,
            )
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
        )
        if import_options is not None and input_path.suffix.lower() in JT_SUFFIXES:
            from fascat.options import JtReadOptions

            import_options = JtReadOptions(
                **cast(Any, import_options.to_dict()), lod_selection=effective_lod_selection.value
            )
        export_metadata = (
            pipeline_spec.export_metadata
            if pipeline_spec is not None and pipeline_spec.export_metadata is not None
            else _metadata_export_options(metadata, pmi)
        )
        heal_options = (
            _brep_heal_options(
                heal_tolerance=heal_tolerance,
                group_open_shells=group_open_shells,
                cleanup_overlapping_faces=cleanup_overlapping_faces,
                overlap_area_ratio=overlap_area_ratio,
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
                instance_similarity_tolerance=instance_similarity_tolerance,
            )
            if (
                batch_by_material
                or merge_compatible_meshes
                or split_large_meshes
                or flatten != FlattenMode.SAFE
                or index_buffer != IndexBufferMode.AUTO
                or instance_policy != InstancePolicy.AUTO
                or instance_similarity_tolerance > 0.0
            )
            else None
        )
        bake_options = (
            BakeMaterialOptions(
                maps_resolution=maps_resolution,
                lightmap_resolution=lightmap_resolution,
                force_uv_generation=force_uv_generation,
                uv_channel=0,
                padding=uv_padding,
                bake=cast(Any, bake_maps),
                merge_output=True,
                ambient_occlusion_strategy=ambient_occlusion_strategy.value,
            )
            if bake_materials
            else None
        )
        decimate_options = (
            DecimateOptions(
                criterion=decimate_criterion.value,
                target_triangles=decimate_target_triangles,
                target_ratio=ratio,
                surface_tolerance=surface_tolerance,
                line_tolerance=line_tolerance,
                normal_tolerance=normal_tolerance,
                uv_tolerance=uv_tolerance,
                iterative_threshold=decimate_iterative_threshold,
                protect_topology=protect_topology,
                preserve_painted_areas=preserve_painted_areas,
                preserve_ambient_occlusion=preserve_ambient_occlusion,
                ambient_occlusion_strategy=ambient_occlusion_strategy.value,
                budget_scope=budget_scope.value,
                uv_importance=cast(Any, uv_importance.value.replace("-", "_")),
                cleanup_attributes=cast(Any, cleanup_attributes),
                jobs=jobs,
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
            _lod_generator_options(lod_preset.value, lod_values, lod_coverages, validate_lods, jobs)
            if run_lod_generators
            else None
        )
        lod_options = _lod_options_for_cli(
            profile_options.lods,
            lod_values,
            lod_coverages,
            lod_mode.value,
            lod_engine_profile.value,
            lod_per_part_budget,
            lod_drop_tiny_parts,
            lod_tiny_part_screen_size,
            validate_lods,
            jobs,
            (lod_source or (LODSourceMode.GENERATED if lod_values is not None else LODSourceMode.AUTO)).value,
        )
        usd_package = "usdz" if (package == UsdPackage.USDZ or effective_output_suffix == ".usdz") else "default"
        gltf_options = GltfExportOptions(
            preset=None if export_preset is None else cast(Any, export_preset.value),
            quantize=quantize,
            meshopt=meshopt,
            draco=draco,
            draco_compression_level=draco_compression_level,
            draco_quantize_position=draco_quantize_position,
            draco_quantize_normal=draco_quantize_normal,
            draco_quantize_texcoord=draco_quantize_texcoord,
            draco_quantize_color=draco_quantize_color,
            ktx2_quality=ktx2_quality,
            ktx2_effort=ktx2_effort,
            ktx2_uastc=ktx2_uastc,
            texture_compression=cast(Any, texture_compression),
            texture_fallback_format=cast(Any, texture_fallback_format),
            png_compression=png_compression,
            jpeg_quality=jpeg_quality,
            file_size_budget_mb=effective_file_size_budget_mb,
            size_ladder=size_ladder,
            metadata=export_metadata,
        )
        usd_options = UsdExportOptions(
            package=cast(Any, usd_package),
            layout=cast(Any, usd_layout.value),
            file_size_budget_mb=effective_file_size_budget_mb,
            metadata=export_metadata,
        )
        obj_options = ObjExportOptions(
            materials=obj_materials,
            write_mtl=write_mtl,
            preserve_groups=preserve_groups,
            file_size_budget_mb=effective_file_size_budget_mb,
        )
        stl_options = StlExportOptions(
            binary=stl_binary,
            merge=stl_merge,
            file_size_budget_mb=effective_file_size_budget_mb,
        )
        fbx_options = FbxExportOptions(
            materials=fbx_materials,
            normals=fbx_normals,
            tangents=fbx_tangents,
            uvs=fbx_uvs,
            file_size_budget_mb=effective_file_size_budget_mb,
        )
        reporter = _stage_reporter(ctx, input_path, output_path)
        convert_input: Path | list[Path] = input_paths if len(input_paths) > 1 else input_path
        with reporter:
            asset = _io_helpers._convert_for_cli(
                convert_input,
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
                progress=reporter.callback,
                debug=debug,
                gltf_options=gltf_options,
                usd_options=usd_options,
                obj_options=obj_options,
                stl_options=stl_options,
                fbx_options=fbx_options,
                stdout_format=stdout_format,
            )
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        _interrupt(ctx, payload)
        raise AssertionError("unreachable") from None
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

    _print_report_warnings(ctx, asset.report)
    if _is_stdio(output_path):
        return

    result = {
        **payload,
        "stats": asset.stats(),
        "report": asset.report.to_dict(),
    }
    if reporter.live:
        reporter.summary(asset)
    else:
        _emit(ctx, result, f"Converted {input_path} to {output_path}: {_format_stats(asset.stats())}.")
