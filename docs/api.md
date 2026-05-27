---
title: Python API
description: Use fascat from Python
---

The Python API exposes the same pipeline as the CLI, but keeps each conversion step explicit and composable.

## End-to-end pipeline

```python
import fascat as fc

asset = fc.read_step("motor.step")

asset = asset.tessellate(
    fc.Tessellation(
        sag=0.1,
        sag_ratio=None,
        angle=15.0,
        relative=True,
        min_edge_length=None,
        max_edge_length=None,
        max_polygon_length=None,
        free_edge_report=False,
        reuse_existing_meshes=True,
        preserve_boundaries=True,
        curvature_adaptive=False,
        avoid_skinny_triangles=False,
        quality_report=False,
    )
)

asset = asset.repair(
    fc.RepairOptions(
        tolerance=0.05,
        merge_vertices=True,
        delete_degenerate=True,
        fix_winding=True,
        fill_small_holes=False,
    )
)

asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        material_mode="cad",
        merge_equivalent_materials=False,
        normals=True,
        normal_mode="smooth",
        hard_edge_angle=30.0,
        preserve_face_boundaries=False,
        tangents=False,
        tangent_uv_channel=0,
        override_tangents=False,
        validate_normals=False,
        unwrap=fc.UnwrapOptions(),
        atlas=fc.AtlasOptions(),
        uv0="box",
        uv1=None,
    )
)

asset = asset.optimize(
    fc.OptimizeOptions(
        target_triangles=500_000,
        preserve_instances=True,
        simplify=True,
        optimize_buffers=True,
        preserve_hard_edges=False,
        preserve_holes=False,
        preserve_material_boundaries=False,
        preserve_uv_seams=False,
        preserve_small_parts=False,
        preserve_silhouette=False,
    )
)

asset = asset.lods(
    fc.LODOptions(
        ratios=[0.5, 0.25, 0.1],
        mode="variants",
        screen_coverage=[0.5, 0.2, 0.05],
        per_part_budget=True,
        drop_tiny_parts=True,
        tiny_part_screen_size=2.0,
        validate=True,
    )
)

asset.write_usd("motor.usdc")
asset.write_gltf("motor.glb")
```

Pipeline operations return new `Asset` instances instead of mutating the previous asset. Write calls attach a final write step to the asset report.

Core pipeline calls:

| API | Parameters | Purpose |
|-----|------------|---------|
| `fc.read_step(path, options=None)` | `path` is a STEP file path or `-` for stdin. `options` is `StepReadOptions`. | Import STEP assembly hierarchy, metadata, materials, and source BREP handles when the backend exposes them. |
| `asset.tessellate(options, where=None)` | `options` is `Tessellation`. `where` optionally scopes the operation with a `Filter`. | Convert source BREP geometry into meshes. |
| `asset.repair(options, where=None)` | `options` is `RepairOptions`. `where` optionally scopes selected parts. | Clean mesh-level issues after tessellation. |
| `asset.stage(options, where=None)` | `options` is `StageOptions`. `where` optionally scopes selected parts. | Prepare materials, normals, tangents, and UV metadata for runtime export. |
| `asset.optimize(options, where=None)` | `options` is `OptimizeOptions`. `where` optionally scopes selected parts. | Reduce mesh complexity while preserving selected mechanical features. |
| `asset.lods(options, where=None)` | `options` is `LODOptions`. `where` optionally scopes selected parts. | Generate lower-detail runtime meshes. |
| `asset.write_usd(path, options=None)` | `path` ends in `.usd`, `.usda`, `.usdc`, or `.usdz`. `options` is `UsdExportOptions`. | Write OpenUSD output and append a write step to the report. |
| `asset.write_gltf(path, options=None)` | `path` ends in `.gltf` or `.glb`. `options` is `GltfExportOptions`. | Write glTF 2.0 output and append a write step to the report. |

## Assembly filters

Use `Filter` selectors to inspect or process one branch of an assembly while leaving the rest unchanged.

```python
import fascat as fc

asset = fc.read_step("motor.step").tessellate()

fasteners = fc.Filter(
    path="*/Fasteners/*",
    name=["Bolt*", "Nut*", "Washer*"],
)

large_castings = fc.Filter.all(
    fc.Filter.path("*/Housing/*"),
    fc.Filter.size(min_diagonal=50.0),
)

print(asset.select(fasteners).stats())

asset = asset.optimize(
    fc.OptimizeOptions(target_triangles=80_000),
    where=fasteners,
)

asset = asset.stage(
    fc.StageOptions(materials="display", uv0="none", uv1=None),
    where=large_castings,
)
```

Filters support node path, node name, part id, part name, material, metadata, bounding box, size, triangle count, vertex count, and logical `all`, `any`, and `not_` composition. If a selected occurrence shares a part with an unmatched occurrence, Fascat duplicates the selected occurrence's part before applying the operation so the unmatched branch stays intact. The scope planner skips that isolation copy when the selection already maps cleanly to whole unique parts. Report steps include `where` and `matched` fields when an operation is scoped.

Filter parameters:

| Parameter | Meaning |
|-----------|---------|
| `path` | Match the full assembly node path with shell-style patterns such as `*/Fasteners/*`. |
| `name` | Match node names. Accepts a string or list of patterns. |
| `part_name` | Match the source part name. |
| `part_id` | Match the stable Fascat part id. `Filter.part(value)` is shorthand for this. |
| `material` | Match any material assigned to the selected part. |
| `metadata` | Require metadata key/value matches on the node, part, material, or asset context. |
| `min_bounds`, `max_bounds` | Match parts whose bounding box lies inside the supplied coordinate bounds. |
| `min_diagonal`, `max_diagonal` | Match by bounding-box diagonal size. |
| `min_triangles`, `max_triangles` | Match by mesh triangle count. `Filter.triangle_count()` builds these criteria. |
| `min_vertices`, `max_vertices` | Match by mesh vertex count. `Filter.vertex_count()` builds these criteria. |
| `include` | Require at least one nested filter to match before criteria are accepted. |
| `exclude` | Drop matches selected by nested filters. |
| `Filter.all(...)` | Require every child filter to match. |
| `Filter.any(...)` | Require at least one child filter to match. |
| `Filter.not_(...)` | Invert one child filter. |
| `where` | Most pipeline methods accept `where=Filter(...)` to scope an operation without destroying unmatched hierarchy. |

## Hierarchy merge

Use `merge()` to reduce node count and draw calls before optimization.

```python
import fascat as fc

asset = fc.read_step("motor.step").tessellate().stage()

asset = asset.merge(
    fc.MergeOptions(
        mode="by_material",
        keep_parent=True,
        metadata="combine",
        max_vertices_per_mesh=65_535,
        preserve_materials=True,
    ),
    where=fc.Filter.path("*/Fasteners/*"),
)
```

Merge modes include `all`, `by_material`, `by_node_name`, `by_part_name`, `hierarchy_level`, `parent_children`, `final_level`, and `regions`. Merging bakes node transforms into merged vertex positions, keeps material slots when requested, removes replaced empty nodes, and records before/after `draw_calls` in the merge report step.

Use `explode()` when runtime tools need separate meshes by material or connected component, and `replace()` when a selected part should become a proxy.

```python
asset = asset.explode(
    fc.ExplodeOptions(mode="connected_components"),
    where=fc.Filter.material("rubber"),
)

asset = asset.replace(
    fc.ReplaceOptions(mode="bounding_box", preserve_transform=True),
    where=fc.Filter.triangle_count(max=12),
)
```

`ReplaceOptions(mode="external_asset", external_path="proxy.glb")` records an external proxy reference while keeping a bounding-box mesh fallback in the asset.

Hierarchy option parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `MergeOptions` | `mode` | Merge strategy: `all`, `by_material`, `by_node_name`, `by_part_name`, `hierarchy_level`, `parent_children`, `final_level`, or `regions`. |
| `MergeOptions` | `keep_parent` | Keep a selected parent node and place merged geometry under it instead of flattening the selected branch completely. |
| `MergeOptions` | `metadata` | Metadata policy: `preserve`, `combine`, `summarize`, or `drop`. |
| `MergeOptions` | `max_vertices_per_mesh` | Split merged output before it exceeds this vertex count. Use `65_535` for 16-bit index friendly meshes. |
| `MergeOptions` | `preserve_materials` | Keep material slots and face material assignments in merged geometry. |
| `MergeOptions` | `hierarchy_level` | Level used by `mode="hierarchy_level"`. `0` starts at the selected root. |
| `MergeOptions` | `region_size` | Spatial cell size used by `mode="regions"`. Required for region merging. |
| `MergeOptions` | `merge_strategy` | Sub-strategy inside region merging: `all` or `by_material`. |
| `MergeOptions` | `remove_empty_nodes` | Remove hierarchy nodes left empty after merging. |
| `ExplodeOptions` | `mode` | Split selected meshes by `by_material` or `connected_components`. |
| `ExplodeOptions` | `metadata` | Metadata policy applied to exploded parts. |
| `ExplodeOptions` | `remove_empty_nodes` | Remove empty source nodes after selected geometry is replaced by exploded children. |
| `ReplaceOptions` | `mode` | Replacement style: `bounding_box`, `proxy_mesh`, or `external_asset`. |
| `ReplaceOptions` | `preserve_transform` | Keep the selected occurrence transform on the replacement. |
| `ReplaceOptions` | `metadata` | Metadata policy applied to replacement parts. |
| `ReplaceOptions` | `proxy_mesh` | Mesh object required when `mode="proxy_mesh"`. |
| `ReplaceOptions` | `external_path` | External asset path recorded when `mode="external_asset"`. |

## Metadata and PMI

Fascat keeps top-level asset metadata and typed PMI records alongside node, part, material, and mesh metadata.

```python
import fascat as fc

asset = fc.read_step(
    "motor.step",
    options=fc.StepReadOptions(
        metadata=True,
        product_metadata=True,
        properties=True,
        layers=True,
        validation_properties=True,
        pmi=True,
        design_variants=False,
        existing_meshes=True,
        multi_file=False,
        delete_free_vertices=False,
        delete_lines=False,
        source_units=None,
        source_up_axis="Z",
        source_handedness="right",
        target_units="metre",
        target_up_axis="Y",
        target_handedness="right",
    ),
)

asset.metadata["review_state"] = "approved"
asset.pmi.append(
    fc.PmiAnnotation(
        id="pmi_001",
        kind="dimension",
        text="25.4 +/-0.1",
        value=25.4,
        unit="millimetre",
        tolerance=fc.Tolerance(upper=0.1, lower=0.0),
        applies_to=["part_123"],
    )
)
```

glTF export writes metadata and PMI into `extras.fascat`. USD export writes Fascat metadata into `customData` on the scene, nodes, prototypes, materials, meshes, and `/PMI/*` annotation prims. When merge, explode, or replace operations create new parts, exporters resolve PMI links through `source_part_id` and `source_part_ids` metadata so annotations that targeted the original part still attach to the derived output.

STEP AP242 files can advertise PMI even when the current OCP-backed importer cannot extract typed annotation entities. In that case the import report records `pmi_present=true`, `unsupported_pmi_count=1`, and a warning instead of silently implying that PMI was imported.

Metadata and PMI parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `StepReadOptions` | `metadata` | Enables general source metadata import. If `False`, the more specific metadata import groups are disabled by default. |
| `StepReadOptions` | `product_metadata` | Import product and assembly-level metadata where the STEP backend exposes it. |
| `StepReadOptions` | `properties` | Import user and product properties. |
| `StepReadOptions` | `layers` | Import layer assignments as metadata. |
| `StepReadOptions` | `validation_properties` | Import STEP validation properties such as source counts or checksums when available. |
| `StepReadOptions` | `pmi` | Import typed PMI records when the backend exposes them; AP242 PMI markers are reported when typed import is unavailable. |
| `StepReadOptions` | `design_variants` | Request STEP design variant import. Current backend support is limited and reports a warning when requested variants cannot be loaded. |
| `StepReadOptions` | `existing_meshes` | Prefer existing tessellation payloads from the source file when the importer exposes them. Tessellation `reuse_existing_meshes` still controls whether loaded meshes are retessellated later. |
| `StepReadOptions` | `multi_file` | Request multi-file STEP assembly import intent. Current single-path imports report a warning instead of silently claiming external references were loaded. |
| `StepReadOptions` | `delete_free_vertices` | Drop construction-only point shapes during import and record deletion counts in the import report. |
| `StepReadOptions` | `delete_lines` | Drop construction-only line shapes during import and record deleted edge and vertex counts. Mixed BREP parts with faces are preserved. |
| `StepReadOptions` | `source_units`, `source_meters_per_unit` | Override the source unit declaration when the STEP header is wrong or ambiguous. Known unit names include `metre`, `centimetre`, `millimetre`, `inch`, and `foot`; custom factors use meters per source unit. |
| `StepReadOptions` | `source_up_axis`, `source_handedness` | Declare the source coordinate basis before normalization. Defaults are STEP-style `Z` up and `right` handed. |
| `StepReadOptions` | `target_units`, `target_meters_per_unit` | Normalize the imported asset to a target unit by applying a root transform and updating the asset's declared units. |
| `StepReadOptions` | `target_up_axis`, `target_handedness` | Normalize the imported asset to a target up-axis or handedness. Import reports include the exact normalization transform and whether it changed the asset space. |
| `PmiAnnotation` | `id` | Stable annotation id used for references from parts or mesh groups. |
| `PmiAnnotation` | `kind` | Annotation type such as `dimension`, `datum`, `tolerance`, `note`, or backend-specific kinds. |
| `PmiAnnotation` | `text` | Human-readable annotation text. |
| `PmiAnnotation` | `value`, `unit` | Numeric measurement value and unit when available. |
| `PmiAnnotation` | `tolerance` | `Tolerance(upper=..., lower=...)` values for dimensional or GD&T annotations. |
| `PmiAnnotation` | `applies_to` | Target ids such as part ids, node ids, face groups, edge groups, or material ids. |
| `MetadataExportOptions` | `mode` | Export metadata as `full`, count-only `summary`, or `none`. |
| `MetadataExportOptions` | `pmi` | Export PMI as `none`, `summary`, `metadata`, `metadata_and_visuals`, or `full`. `metadata_and_visuals` currently emits metadata records and stable links; annotation geometry is still a planned backend. |

```python
asset.write_gltf(
    "motor.glb",
    options=fc.GltfExportOptions(
        metadata=fc.MetadataExportOptions(mode="full", pmi="metadata"),
    ),
)

asset.write_usd(
    "motor.usdc",
    options=fc.UsdExportOptions(
        metadata=fc.MetadataExportOptions(mode="full", pmi="metadata_and_visuals"),
    ),
)
```

## BREP Healing

Run BREP healing before tessellation when STEP topology needs sewing, edge fixing, tolerance unification, or open-shell and unstitched-edge reporting.

```python
asset = fc.read_step("motor.step").heal_brep(
    fc.BrepHealOptions(
        tolerance=0.05,
        sew_faces=True,
        fix_edges=True,
        remove_sliver_faces=True,
        max_sliver_area=1e-4,
        unify_tolerances=True,
        fail_on_open_shells=False,
    ),
    where=fc.Filter.path("*/Housing/*"),
)
```

The operation stores per-part `brep_*` metadata and records a `heal_brep` report step. Metadata includes BREP kind, solid/shell/wire/edge/face counts, open shells, free or unstitched edges, small edges at or below the healing tolerance, and sliver-face counts. The report step also includes `tolerance_policy`, which records the effective source/local units used by the BREP backend, declared target units, meters-per-unit conversions, tolerance values in meters, sliver area in square meters, and whether sewing, edge fixing, tolerance unification, sliver removal, T-junction sewing, and non-manifold cracking are enabled, disabled, requested, or not implemented. `fc.convert(..., heal_brep=fc.BrepHealOptions())` runs healing before tessellation. Sliver-face removal is requested through the BREP backend, but the current backend reports a warning when that removal path is unavailable instead of silently claiming that the source shape changed. Remaining open shells, free edges, and small edges are also surfaced as report warnings.

Brep healing parameters:

| Parameter | Meaning |
|-----------|---------|
| `tolerance` | Working tolerance used for sewing, edge fixes, and tolerance unification. Must be greater than zero. |
| `sew_faces` | Attempt to sew adjacent faces into shells before tessellation. |
| `fix_edges` | Attempt to repair bad trims and edge curves where supported by the backend. |
| `remove_sliver_faces` | Request tiny sliver-face removal before tessellation. Current backend support is limited and reports a warning when removal is unavailable. |
| `max_sliver_area` | Area threshold for sliver-face removal. |
| `unify_tolerances` | Normalize shape tolerances to the requested working tolerance. |
| `fail_on_open_shells` | Raise when healing detects open shells instead of reporting a warning. |
| `where` | Optional filter that limits healing to selected assembly occurrences. |

## Tessellation Controls

Tessellation supports global and per-part settings for edge limits, boundary preservation, curvature-adaptive OCCT meshing, skinny-triangle cleanup, and per-part quality metrics.

```python
asset = fc.read_step("motor.step").tessellate(
    fc.Tessellation(
        sag=0.05,
        sag_ratio=None,
        angle=10.0,
        min_edge_length=0.02,
        max_edge_length=2.0,
        max_polygon_length=4.0,
        preserve_boundaries=True,
        curvature_adaptive=True,
        avoid_skinny_triangles=True,
        quality_report=True,
        free_edge_report=True,
        reuse_existing_meshes=True,
        part_settings={
            "housing": {"sag": 0.03, "sag_ratio": 0.005, "max_edge_length": 1.0},
            "Fastener": {"sag": 0.15},
        },
    )
)

quality = asset.tessellation_quality_report()
```

`part_settings` keys match a part id or part name. Quality reports include per-part edge length, triangle area, aspect ratio, skinny triangle, duplicate polygon, boundary edge, and non-manifold edge counts. Tessellated parts also record `tessellation_face_groups`, `tessellation_estimated_draw_calls`, and retained-patch counts when available; the tessellation step warns when retained BREP patches, CAD face groups, or material splits are likely to increase submesh, draw-call, or export-size pressure.

Tessellation parameters:

| Parameter | Meaning |
|-----------|---------|
| `sag` | Maximum chordal deviation between source surface and tessellated mesh. Lower values produce more triangles. |
| `sag_ratio` | Relative chordal deviation ratio. When set, it becomes the backend deflection value and enables relative tessellation explicitly. |
| `angle` | Angular deviation limit in degrees. Lower values preserve curved surfaces with more triangles. |
| `relative` | Compatibility switch for interpreting `sag` as a relative backend deflection when `sag_ratio` is unset. Prefer `sag_ratio` for new relative-tolerance workflows. |
| `min_edge_length` | Collapse or avoid edges shorter than this length during post-processing. |
| `max_edge_length` | Split long triangle edges to keep mesh density bounded. |
| `max_polygon_length` | Report tessellated polygon edges longer than this threshold without subdividing geometry. Quality reports count these as `long_edges`; the tessellation step emits warnings when exceeded. |
| `preserve_boundaries` | Preserve CAD face and boundary edges during tessellation cleanup. |
| `curvature_adaptive` | Request curvature-aware meshing from the backend when available. |
| `avoid_skinny_triangles` | Run a cleanup pass that reduces long skinny triangles. |
| `quality_report` | Record per-part tessellation quality metrics for later reporting. |
| `free_edge_report` | Record free/boundary edge and non-manifold edge counts on tessellated parts and warn when free edges are present. |
| `create_normals` | Generate normals during tessellation when the backend can provide them. |
| `keep_brep` | Keep source BREP handles on parts after tessellation for later BREP-aware operations. Tessellated parts record `brep_patch_cleanup=retained` or `deleted` and warn when many retained patches could increase runtime/export risk. |
| `reuse_existing_meshes` | Reuse meshes already present on imported parts. Set to `False` to retessellate from source BREP where available. |
| `part_settings` | Per-part overrides keyed by part id or part name. Supports the same tessellation option names. |

Repair parameters:

| Parameter | Meaning |
|-----------|---------|
| `tolerance` | Merge tolerance for nearby vertices. `0.0` disables distance-based merging beyond exact duplicates. |
| `merge_vertices` | Deduplicate vertices after tessellation. |
| `delete_degenerate` | Remove triangles with repeated vertices or near-zero area. |
| `fix_winding` | Normalize triangle winding where a consistent orientation can be inferred. |
| `fill_small_holes` | Fill small mesh boundary loops as a fallback mesh repair step. |
| `area_epsilon` | Area threshold used to classify degenerate triangles. |

Repair metadata records before/after counts for `repair_duplicate_polygons`, `repair_degenerate_triangles`, `repair_boundary_edges`, and `repair_non_manifold_edges`. Duplicate polygons are triangles that reference the same three vertices, regardless of winding. The `repair` report step also records `tolerance_policy`, including effective source/local units, declared target units, meter conversions, vertex merge tolerance in meters, degenerate area epsilon in square meters, and the status of vertex merge, degenerate-polygon cleanup, T-junction sewing, and non-manifold edge cracking.

## Feature-Preserving Simplification

Optimization can protect mechanical features while reducing triangle count. Preservation flags keep protected faces from being dropped when a target would otherwise remove them.

```python
asset = asset.optimize(
    fc.OptimizeOptions(
        target_triangles=500_000,
        simplify=True,
        preserve_instances=True,
        preserve_hard_edges=True,
        hard_edge_angle=30.0,
        preserve_holes=True,
        preserve_material_boundaries=True,
        preserve_uv_seams=True,
        preserve_small_parts=True,
        small_part_triangle_threshold=64,
        preserve_silhouette=True,
    )
)
```

Protected-feature counts are stored as part metadata under `simplification_preserved_features`. Parts below `small_part_triangle_threshold` are left unsimplified when `preserve_small_parts=True`.

Optimization parameters:

| Parameter | Meaning |
|-----------|---------|
| `target_triangles` | Absolute triangle budget for selected geometry. |
| `ratio` | Fraction of original triangles to keep. Use this instead of `target_triangles` for proportional simplification. |
| `preserve_instances` | Keep repeated part instances sharing geometry instead of expanding them unnecessarily. |
| `simplify` | Enable triangle-count reduction. Disable to run only metadata and buffer optimization steps. |
| `optimize_buffers` | Reorder and compact mesh buffers after simplification. |
| `preserve_hard_edges` | Protect faces around hard normal edges from simplification. |
| `hard_edge_angle` | Edge angle threshold in degrees used to detect hard edges. |
| `preserve_holes` | Protect hole boundary loops and nearby faces. |
| `preserve_material_boundaries` | Avoid collapsing across material boundaries. |
| `preserve_uv_seams` | Avoid collapsing across UV seams. |
| `preserve_small_parts` | Leave small parts unsimplified instead of spending budget on them. |
| `small_part_triangle_threshold` | Parts at or below this triangle count are treated as small when preservation is enabled. |
| `preserve_silhouette` | Protect bounding-box silhouette extremes to reduce visible shape loss. |

## Hard-Edge Normals And Tangents

Staging can generate smooth, flat, or hard-edge normals and glTF-ready tangents. Hard-edge mode splits vertices across hard normal edges, material boundaries, and optional CAD face-group boundaries.

```python
asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        normals=True,
        normal_mode="hard_edges",
        hard_edge_angle=30.0,
        preserve_face_boundaries=True,
        tangents=True,
        tangent_uv_channel=0,
        override_tangents=False,
        validate_normals=True,
        uv0="box",
    )
)
```

Tangents require the selected UV channel when they are generated, which defaults to UV0. If tangents are requested without that channel, staging records missing-UV metadata and emits a stage warning instead of silently writing no tangent data. Existing tangents are preserved by default when staging has not invalidated them and the selected UV channel is still present; set `override_tangents=True` to force regeneration from `tangent_uv_channel`. When UV generation edits a mesh that already had tangents, staging invalidates the old tangent basis; if `tangents=True`, it regenerates tangents from the selected UV channel, otherwise it records the dropped tangent state. glTF export writes a `TANGENT` vertex attribute when staged meshes contain tangent data.

Normal and tangent parameters:

| Parameter | Meaning |
|-----------|---------|
| `normals` | Generate or preserve vertex normals. Automatically disabled when `normal_mode="none"`. |
| `normal_mode` | `smooth` averages face normals, `flat` keeps face normals, `hard_edges` splits vertices along hard edges, and `none` omits normals. |
| `hard_edge_angle` | Edge angle threshold in degrees for `normal_mode="hard_edges"`. |
| `preserve_face_boundaries` | Treat CAD face-group boundaries as hard normal boundaries. |
| `tangents` | Ensure glTF-ready tangent vectors exist. Existing valid tangents are preserved by default. |
| `tangent_uv_channel` | UV channel used when tangents need to be generated or regenerated. Defaults to `0`. |
| `override_tangents` | Regenerate existing tangents instead of preserving them when `tangents=True`. |
| `validate_normals` | Check for missing, zero-length, or invalid normals after staging. |

## UV And Material Pipeline

Staging can merge equivalent CAD materials, normalize simple CAD colors into PBR-friendly material values, tag UV unwrap settings, generate lightmap UV channels, and attach material-atlas metadata for later baking.

```python
asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        material_mode="pbr",
        merge_equivalent_materials=True,
        uv0="unwrap",
        uv1="lightmap",
        unwrap=fc.UnwrapOptions(
            texel_density=256.0,
            padding=4,
            max_stretch=0.15,
            method="conformal",
            iterations=32,
            tolerance=0.001,
        ),
        atlas=fc.AtlasOptions(
            enabled=True,
            max_size=4096,
        ),
        normalize_uvs=(1,),
    )
)
```

Atlas support currently records atlas and texture-bake metadata on materials and meshes. It does not write atlas images. Dedicated material baking is a separate optimization step; it emits constant embedded texture maps from material factors and glTF can export those maps as material textures.

Staged meshes also record UV layout quality metadata for each channel: `uvN_domain`, `uvN_bounds`, `uvN_unit_domain_status`, `uvN_validation_status`, `uvN_out_of_unit_vertices`, `uvN_degenerate_faces`, and `uvN_overlap_pairs`. UV0 defaults to the `tileable` domain, where overlaps and coordinates outside 0..1 are allowed but still counted. UV1 and `lightmap` channels use the `bake` domain, where overlaps, degenerate UV faces, or coordinates outside 0..1 set `uvN_validation_status` and add stage warnings. Use `uv1="copy_uv0"` when the secondary channel should reuse the generated or existing UV0 layout; staging records `uv1_source_channel="0"` and warns if UV0 is missing. Use `normalize_uvs=(1,)` to rescale selected channels into the 0..1 domain; staging records the original bounds and warns when a requested channel is absent.

When `uv0` or `uv1` uses `unwrap` or `lightmap`, fascat currently uses the optional xatlas backend. `method`, `iterations`, and `tolerance` record Unity-style solver intent for conformal or isometric unwrapping and solver stopping criteria. The current xatlas integration does not expose those controls directly, so non-default values are marked with `*_status="intent"` metadata and add a report warning instead of pretending the backend enforced them.

Staging, UV, and material parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `StageOptions` | `materials` | Material source policy: `cad` preserves CAD materials, `display` creates display materials, and `none` omits materials. |
| `StageOptions` | `material_mode` | `cad` keeps source-style materials. `pbr` normalizes simple CAD colors into PBR-friendly material values. |
| `StageOptions` | `merge_equivalent_materials` | Merge materials with equivalent visual values to reduce material count. |
| `StageOptions` | `uv0` | Primary UV channel mode: `none`, `box`, `unwrap`, or `lightmap`. |
| `StageOptions` | `uv1` | Secondary UV channel mode. Commonly `lightmap` for baked lighting, or `copy_uv0` to duplicate UV0 into UV1. |
| `StageOptions` | `normalize_uvs` | UV channels to rescale into 0..1 after generation/copy. Use explicitly because UV0 may intentionally tile outside 0..1. |
| `StageOptions` | `unwrap` | `UnwrapOptions` used when a UV channel uses `unwrap`. |
| `StageOptions` | `atlas` | `AtlasOptions` used to record atlas layout and baking intent. |
| `UnwrapOptions` | `texel_density` | Desired texture density for generated UVs. |
| `UnwrapOptions` | `padding` | Padding between UV islands in pixels. |
| `UnwrapOptions` | `max_stretch` | Maximum tolerated UV stretch before reporting unwrap risk. |
| `UnwrapOptions` | `method` | Requested unwrap solver intent: `default`, `conformal`, or `isometric`. Non-default values are recorded as intent with the xatlas backend. |
| `UnwrapOptions` | `iterations` | Requested unwrap solver iteration budget. Recorded as intent until a backend exposes this control. |
| `UnwrapOptions` | `tolerance` | Requested unwrap solver error threshold. Recorded as intent until a backend exposes this control. |
| `AtlasOptions` | `enabled` | Record atlas metadata and prepare materials for later baking. |
| `AtlasOptions` | `max_size` | Maximum atlas texture size in pixels. |

## Scene Optimization

Use scene optimization to reduce draw calls after staging and optional hierarchy merging. It batches compatible meshes, can batch by material, reconstructs exact repeated mesh instances when vertex attributes, materials, and metadata match, reports duplicate mesh payload savings, splits large merged meshes, simplifies empty hierarchy, and annotates the intended index-buffer width.

```python
asset = asset.optimize_scene(
    fc.SceneOptimizeOptions(
        batch_by_material=True,
        merge_compatible_meshes=True,
        split_large_meshes=True,
        max_vertices_per_mesh=65_535,
        index_buffer="auto",
        flatten="safe",
        remove_empty_nodes=True,
        instance_policy="auto",
    )
)
```

Scene optimization parameters:

| Parameter | Meaning |
|-----------|---------|
| `batch_by_material` | Group compatible geometry by material to reduce draw calls. |
| `merge_compatible_meshes` | Merge meshes that can share buffers and material assignments safely. |
| `split_large_meshes` | Split merged output that exceeds the configured vertex limit. |
| `max_vertices_per_mesh` | Vertex limit used for splitting and index-buffer planning. |
| `index_buffer` | `auto` chooses 16-bit or 32-bit indices. `uint16` and `uint32` force a width. |
| `flatten` | `none` preserves hierarchy, `safe` removes only safe empty structure, and `all` aggressively flattens. |
| `remove_empty_nodes` | Remove hierarchy nodes with no part and no children. |
| `instance_policy` | `auto` and `preserve` reconstruct exact repeated mesh instances when vertex attributes, material assignments, and metadata match. `expand` duplicates instances per occurrence. |

## Optimization Actions

Use explicit optimization actions when a realtime pipeline needs named preparation steps and separate report entries for each action.

```python
asset = asset.bake_materials(
    fc.BakeMaterialOptions(
        maps_resolution=2048,
        force_uv_generation=True,
        bake=("base_color", "opacity"),
    )
)

asset = asset.decimate(
    fc.DecimateOptions(
        criterion="target",
        target_triangles=250_000,
        surface_tolerance=0.1,
        line_tolerance=0.02,
        normal_tolerance=15.0,
        uv_tolerance=0.01,
        protect_topology=True,
        budget_scope="selection",
        uv_importance="preserve_islands",
    )
)

asset = asset.remove_holes(fc.RemoveHolesOptions(max_diameter=3.0, prefer_brep=True))
asset = asset.remove_occluded(fc.RemoveOccludedOptions(strategy="advanced", level="triangles"))
asset = asset.run_lod_generators(
    fc.LODGeneratorOptions(
        preset="vr",
        levels=(
            fc.LODLevel(screen_coverage=0.5, target_ratio=0.5),
            fc.LODLevel(screen_coverage=0.2, target_ratio=0.25),
            fc.LODLevel(screen_coverage=0.05, target_ratio=0.1),
        ),
        validate=True,
    )
)
```

Material baking currently creates a shared flat material and constant embedded texture maps from material factors; it does not rasterize source textures into atlases. Hole removal uses deterministic mesh boundary classification and filling when BREP feature editing is unavailable. Occlusion removal uses deterministic visibility sampling, so the report records that thin occluders can require higher precision and asset metadata records the measured sample coverage, direction coverage, and confidence score. Decimation records `decimate_requested_keep_ratio` metadata when a requested ratio can be derived, and warns when the request keeps less than 20% of source triangles because those settings are usually appropriate for distant LODs rather than close-view LOD0 assets. Decimation also records a memory estimate using the Unity rule of thumb of 5 GB RAM per million source triangles; the report marks `decimate_iterative_recommended` when the selected source triangle count reaches the iterative threshold. `uv_importance="ignore"` strips UV/tangent attributes before simplification; `"preserve_seams"` uses UVs for seam preservation and then strips them; `"preserve_islands"` keeps UVs through the output.

LOD generation skips parts that do not have tessellated meshes, records `lod_status="skipped_no_mesh"` on those parts, and adds report warnings so partial LOD chains are visible. Asset metadata records `lod_generated_parts` and `lod_skipped_no_mesh_parts`.

Optimization action parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `BakeMaterialOptions` | `maps_resolution` | Requested texture size recorded in bake metadata for downstream atlas generation. Current embedded maps are constant factor textures. |
| `BakeMaterialOptions` | `force_uv_generation` | Generate UVs first when selected meshes do not have the required UV channel. |
| `BakeMaterialOptions` | `uv_channel` | UV channel used for baking. |
| `BakeMaterialOptions` | `padding` | Texture padding between islands in pixels. |
| `BakeMaterialOptions` | `bake` | Maps to bake, such as `base_color`, `opacity`, `normal`, `roughness`, `metallic`, `ao`, or `emissive`. |
| `BakeMaterialOptions` | `merge_output` | Replace selected materials with a shared baked output material. |
| `DecimateOptions` | `criterion` | `target` prioritizes a triangle budget. `quality` maps tolerances to a target ratio, records measured vertex error, and warns because tolerance bounds are not enforced. |
| `DecimateOptions` | `target_triangles` | Absolute triangle target for selected geometry. |
| `DecimateOptions` | `target_ratio` | Fraction of source triangles to keep when no absolute target is set. Ratios below 20% produce an LOD0 distortion warning. |
| `DecimateOptions` | `surface_tolerance` | Tolerance input used by `criterion="quality"` to derive a reduction ratio; post-run metadata records measured vertex error but does not enforce this value. |
| `DecimateOptions` | `line_tolerance` | Line-feature tolerance input used by `criterion="quality"` ratio derivation and reporting. |
| `DecimateOptions` | `normal_tolerance` | Maximum normal deviation in degrees. |
| `DecimateOptions` | `uv_tolerance` | UV tolerance input used by `criterion="quality"` ratio derivation and reporting. |
| `DecimateOptions` | `protect_topology` | Avoid topology changes that would remove important boundaries. |
| `DecimateOptions` | `preserve_painted_areas` | Preserve metadata-marked or painted regions where present. |
| `DecimateOptions` | `budget_scope` | `part` budgets each part separately. `selection` uses a global selected-geometry target so sparse/simple parts can stay intact while dense parts absorb more reduction. Global selection decimation also reports estimated RAM and iterative-threshold status. |
| `DecimateOptions` | `uv_importance` | Texture-coordinate handling: `preserve_islands` keeps UVs, `preserve_seams` protects seam topology then drops UVs, and `ignore` strips UVs/tangents before decimation. |
| `RemoveHolesOptions` | `through`, `blind`, `surface` | Hole-type filters for boundary-loop classification. `through` matches paired aligned openings, `blind` matches open pocket mouths, and `surface` matches remaining surface openings. |
| `RemoveHolesOptions` | `max_diameter` | Only fill detected open boundary loops at or below the measured planar-span diameter. |
| `RemoveHolesOptions` | `prefer_brep` | Request BREP-level feature removal. Current implementation warns and uses mesh boundary classification and filling. |
| `RemoveOccludedOptions` | `strategy` | Visibility direction set: `conservative` checks cardinal views, `exterior` adds exterior diagonals, and `advanced` uses the densest deterministic direction set. |
| `RemoveOccludedOptions` | `level` | Removal granularity: `parts` removes fully hidden occurrences, `submeshes` removes fully hidden material groups, and `triangles` removes hidden faces. |
| `RemoveOccludedOptions` | `precision` | Maximum part-level face sample count before deterministic downsampling. Higher values can help thin occluders and large parts. |
| `RemoveOccludedOptions` | `hemi_evaluation` | Restrict visibility rays to the upper hemisphere and side views for top/side-oriented evaluation. |
| `RemoveOccludedOptions` | `neighbors_preservation` | Keep this many rings around visible triangles to reduce cracks. |
| `RemoveOccludedOptions` | `consider_transparency_opaque` | Treat transparent materials as opaque for conservative visibility. |
| `RemoveOccludedOptions` | `preserve_cavities` | Preserve interior cavities above the configured volume threshold. |
| `RemoveOccludedOptions` | `minimum_cavity_volume_m3` | Cavity volume threshold used when `preserve_cavities=True`. |
| `LODGeneratorOptions` | `preset` | Default LOD level set: `desktop`, `web`, `mobile`, or `vr`. |
| `LODGeneratorOptions` | `levels` | Explicit `LODLevel` entries overriding the preset. |
| `LODGeneratorOptions` | `validate` | Validate monotonic triangle, material, and draw-call counts after generation. |
| `LODGeneratorOptions` | `output` | LOD representation: `variants`, `extras`, or `separate`. |
| `LODGeneratorOptions` | `allow_non_monotonic` | Permit non-monotonic LODs without failing validation. |
| `LODLevel` | `screen_coverage` | Screen fraction at which this LOD becomes appropriate. |
| `LODLevel` | `target_ratio` | Fraction of source triangles to keep for this LOD. |

Occlusion metadata includes `occlusion_candidate_count`, `occlusion_face_count`, `occlusion_sample_count`, `occlusion_visible_sample_count`, `occlusion_hidden_sample_count`, `occlusion_sample_coverage`, `occlusion_direction_coverage`, and `occlusion_confidence`. The confidence score is the lower of sample coverage and direction coverage; lower values mean the result depends on sparse sampling or a reduced direction set.

Report examples for destructive and approximate operations:

```json
{
  "name": "merge",
  "before": {"parts": 42, "triangles": 120000, "draw_calls": 42},
  "after": {"parts": 8, "triangles": 120000, "draw_calls": 8},
  "warnings": []
}
```

```json
{
  "name": "bake_materials",
  "before": {"materials": 12, "draw_calls": 18},
  "after": {"materials": 1, "draw_calls": 1},
  "warnings": [
    "bake_materials emits constant embedded texture maps from material factors; raster texture baking is not implemented"
  ]
}
```

```json
{
  "name": "remove_holes",
  "before": {"triangles": 8400},
  "after": {"triangles": 8412},
  "warnings": [
    "BREP feature-level hole removal is not implemented; using mesh boundary classification and fill"
  ]
}
```

```json
{
  "name": "remove_occluded",
  "before": {"parts": 120, "triangles": 300000},
  "after": {"parts": 118, "triangles": 296000},
  "warnings": [
    "remove_occluded uses deterministic sampled visibility; thin occluders may require higher precision"
  ]
}
```

## One-shot conversion

Use `fc.convert()` when you want the full default pipeline and output validation in one call.

```python
import fascat as fc

asset = fc.convert(
    "motor.step",
    "motor.usdc",
    profile="realtime-desktop",
    where=fc.Filter.path("*/Fasteners/*"),
    merge=fc.MergeOptions(mode="by_material", metadata="combine"),
)

print(asset.stats())
print(asset.report.summary())
```

The output format is selected from the output suffix:

```python
fc.convert("motor.step", "motor.usdc")
fc.convert("motor.step", "motor.usda", debug=True)
fc.convert("motor.step", "motor.glb", profile="virtual-reality")
fc.convert("motor.step", "motor.glb", profile="realtime-mobile")
fc.convert("motor.step", "motor.gltf", profile="realtime-web")
```

`fc.convert()` validates generated output by default. Pass `validate_output=False` only when another step in your pipeline validates the asset.
When `where` is provided to `fc.convert()`, tessellation, repair, and staging still run for the full asset, while merge, scene optimization, optimization actions, optimization, and LOD generation are scoped to the matched assembly subset.

Conversion parameters:

| Parameter | Meaning |
|-----------|---------|
| `input_path` | STEP input path or `-` for stdin. |
| `output_path` | Output path. Suffix selects USD, glTF, OBJ, or STL. |
| `profile` | Profile name or `ConversionProfile` that supplies default tessellation, repair, stage, optimize, and LOD options. |
| `import_options` | `StepReadOptions` for STEP metadata and PMI import. |
| `tessellation` | Overrides the profile tessellation step. |
| `heal_brep` | Optional BREP healing step before tessellation. |
| `stage` | Overrides the profile staging step. |
| `merge`, `explode`, `replace` | Optional hierarchy operations run after staging. |
| `scene` | Optional scene optimization step. |
| `bake_materials`, `remove_holes`, `remove_occluded`, `decimate`, `lod_generator` | Optional explicit optimization actions. |
| `optimize` | Overrides the profile simplification step. |
| `lods` | Overrides the profile ratio-based LOD step. |
| `progress` | Callback receiving `(step_name, stats)` after major conversion steps. |
| `validate_output` | Reopen and validate generated output before returning. Defaults to `True`. |
| `debug` | Prefer debuggable USDA conventions. Only valid for `.usd` or `.usda` outputs. |
| `gltf_options`, `usd_options`, `obj_options`, `stl_options` | Format-specific write options. |
| `pipeline` | `PipelineSpec` loaded from TOML. When present, ordered pipeline steps drive the conversion. |
| `where` | Optional `Filter` applied to scoped hierarchy, optimization, and LOD steps. |

For multiple branch-specific steps, load the same TOML pipeline format used by `fascat convert --pipeline`:

```python
pipeline = fc.PipelineSpec.from_file("realtime.toml")
for advisory in pipeline.advisories():
    print(advisory["message"])
asset = fc.convert("motor.step", "motor.glb", pipeline=pipeline)
```

Pipeline files can also define import and export metadata policy:

```toml
[import]
metadata = "full"
pmi = true
design_variants = false
existing_meshes = true
multi_file = false
delete_free_vertices = false
delete_lines = false
target_units = "metre"
target_up_axis = "Y"
target_handedness = "right"

[export]
metadata = "summary"
pmi = "metadata"
```

## Runtime Export Options

glTF and USD exports accept runtime delivery options, and OBJ/STL are available for mesh-only handoff workflows.

```python
asset.write_gltf(
    "motor.glb",
    options=fc.GltfExportOptions(
        quantize=True,
        meshopt=True,
        draco=False,
        texture_compression=None,
        file_size_budget_mb=50,
        metadata=fc.MetadataExportOptions(mode="summary", pmi="metadata"),
    ),
)

asset.write_usd(
    "motor.usdz",
    options=fc.UsdExportOptions(package="usdz", file_size_budget_mb=100),
)

asset.write_obj("motor.obj", options=fc.ObjExportOptions(materials=True, write_mtl=True))
asset.write_stl("motor.stl", options=fc.StlExportOptions(binary=True, merge=True))
```

`quantize=True` writes `KHR_mesh_quantization` accessors and composes the dequantization transform into referencing nodes. `meshopt=True` writes `EXT_meshopt_compression` bufferView payloads while keeping fallback buffer data for validators and loaders that ignore the extension. USDZ output is built by writing a temporary USD stage and packaging it as `.usdz`. Draco and texture compression are not implemented yet, so `draco=True` and `texture_compression="ktx2"` or `"basisu"` raise instead of silently writing uncompressed runtime payloads. glTF write report steps include `runtime_dependencies`, listing emitted extensions, required extensions, `extras.fascat` metadata, unsupported Draco/KTX2 outputs, and expected runtime support. Write report steps also include output file size and file-size budget warnings when a budget is provided.

OBJ export writes vertex positions, normals, `f v//vn` face references, material assignments, and smoothing directives. Staged smooth normals export with smoothing enabled; flat, hard-edge, or generated face normals export with smoothing disabled.

Export option parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `GltfExportOptions` | `quantize` | Write `KHR_mesh_quantization` accessors and dequantization transforms. |
| `GltfExportOptions` | `meshopt` | Write `EXT_meshopt_compression` payloads with fallback uncompressed data. |
| `GltfExportOptions` | `draco` | Unsupported until a Draco encoder backend is integrated; `True` raises `ValueError`. |
| `GltfExportOptions` | `texture_compression` | Unsupported until a KTX2/Basis encoder and texture packaging backend is integrated; non-`None` values raise `ValueError`. |
| `GltfExportOptions` | `file_size_budget_mb` | Add report warnings when the output exceeds this size. |
| `GltfExportOptions` | `metadata` | `MetadataExportOptions` controlling metadata and PMI in `extras.fascat`. |
| `UsdExportOptions` | `package` | `default` writes normal USD. `usdz` writes a packaged `.usdz` file. |
| `UsdExportOptions` | `file_size_budget_mb` | Add report warnings when the output exceeds this size. |
| `UsdExportOptions` | `metadata` | `MetadataExportOptions` controlling USD custom data and PMI prims. |
| `ObjExportOptions` | `materials` | Write OBJ `usemtl` assignments when material data exists. |
| `ObjExportOptions` | `write_mtl` | Write an `.mtl` sidecar next to the OBJ. |
| `ObjExportOptions` | `preserve_groups` | Write OBJ group/object names from Fascat hierarchy and parts. |
| `ObjExportOptions` | `file_size_budget_mb` | Add report warnings when the output exceeds this size. |
| `StlExportOptions` | `binary` | Write binary STL when `True`; ASCII STL when `False`. |
| `StlExportOptions` | `merge` | Merge selected triangles into one STL stream. STL does not preserve hierarchy or materials. |
| `StlExportOptions` | `file_size_budget_mb` | Add report warnings when the output exceeds this size. |

## Profiles

Profiles provide practical defaults for tessellation, staging, optimization, LODs, and platform budget checks.

```python
profile = fc.profiles.realtime_web(
    tessellation_sag=0.2,
    angle=20.0,
    max_triangles=250_000,
    lod_ratios=(0.5, 0.25),
)

asset = fc.convert("motor.step", "motor.glb", profile=profile)
```

Available profiles:

| Profile | Use | Target FPS | Triangle budget | Per-mesh vertex budget | Texture resolution budget | Texture memory budget | Load-time budget | Draw-call budget | Unity reference range |
|---------|-----|------------|-----------------|------------------------|---------------------------|-----------------------|------------------|------------------|-----------------------|
| `inspect-only` | inspect STEP input without conversion | unset | unset | unset | unset | unset | unset | unset | unset |
| `realtime-desktop` | higher-detail OpenUSD or glTF output | 60 | 1,000,000 | 65,535 | 4,096px | 512 MB | 2,000 ms | 2,000 | 10M-100M triangles, under 10,000 draw calls |
| `realtime-web` | lower triangle budgets for web delivery | 60 | 250,000 | 65,535 | 2,048px | 128 MB | 3,000 ms | 500 | 100K-1M triangles, under 200 draw calls |
| `realtime-mobile` | tighter mobile runtime budget for app-store builds | 60 | 150,000 | 65,535 | 2,048px | 128 MB | 2,500 ms | 250 | 100K-500K triangles, under 1,000 draw calls |
| `virtual-reality` | balanced triangle budgets and LODs for VR runtimes | 90 | 500,000 | 65,535 | 2,048px | 256 MB | 1,500 ms | 250 | 500K-2M triangles, under 1,000 draw calls |

You can pass either a profile name or a `ConversionProfile` returned by `fc.profiles`. Conversion reports include a `profile_budget` step when the selected profile has a budget. That step records target FPS, triangle, vertex, per-mesh vertex, texture-resolution, texture-memory, estimated load-time, draw-call budgets, and Unity reference triangle/draw-call ranges when the profile has them. Fascat's defaults are intentionally stricter than Unity's broad reference ranges for repeatable export checks. Load time is a deterministic estimate based on output file size, geometry bytes, baked texture bytes, and draw-call overhead; it is not a measured engine runtime.

## Functional wrappers

The top-level functions mirror the fluent `Asset` methods.

```python
import fascat as fc

asset = fc.read_step("motor.step")
asset = fc.tessellate(asset, sag=0.1, angle=15.0)
asset = fc.repair(asset, tolerance=0.05)
asset = fc.stage(asset, materials="cad", uv0="box")
asset = fc.optimize(asset, target_triangles=500_000)
asset = fc.lods(asset, ratios=(0.5, 0.25, 0.1))

fc.write_usd(asset, "motor.usdc")
fc.write_gltf(asset, "motor.glb")
```

## Reports and stats

Every imported or converted asset carries a report.

```python
asset = fc.convert("motor.step", "motor.usdc")

print(asset.stats(include_lods=True))
print(asset.report.summary())

for step in asset.report.steps:
    print(step.name, step.duration, step.before, step.after)

asset.report.write_json("report.json")
```

The report records options, before/after counts, warnings, errors, and timings for each pipeline step. Approximate operations put the limitation on the step that produced it, so callers can distinguish exact geometry changes from fallbacks or metadata-only intent. Conversion reports include a `preflight` step before expensive operations start, with checklist warnings for missing patch cleanup, orientation preparation, UV/tangent ordering, AO bake UV1 prerequisites, LOD0 optimization, and unavailable glTF texture/compression backends. They also include a `workflow_summary` step that maps Unity-inspired preparation stages such as import cleanup, UV preparation, material baking, LOD generation, export compression, and export to run or skipped status.

Use `Asset.analyze()` when you need geometry quality risks beyond raw part and triangle totals.

```python
report = asset.analyze(
    fc.AnalyzeOptions(
        non_manifold_edges=True,
        open_boundaries=True,
        self_intersections=True,
        sliver_triangles=True,
        tiny_parts=True,
        draw_call_estimate=True,
        visual_risk=True,
    )
)

print(report.summary)
report.write_json("quality-report.json")
```

The analysis report includes per-part topology counts, actual triangle
self-intersection counts, degenerate and sliver triangle stats, tiny-part stats,
material count, draw-call estimate, and visual-risk warnings derived from mesh
quality and before/after pipeline report steps. Self-intersection checks ignore
adjacent triangles that share vertices. Coplanar overlaps count as intersections,
while point-only endpoint contact does not. If `max_self_intersection_pairs` is
reached, `self_intersections_lower_bound` is `true` and the report includes
`self_intersection_pairs_checked` and `self_intersection_pair_limit`;
`self_intersection_warnings` is kept as a compatibility alias for
`self_intersections`.

Analysis parameters:

| Parameter | Meaning |
|-----------|---------|
| `non_manifold_edges` | Count edges shared by more than two triangles. |
| `open_boundaries` | Count boundary loops and boundary edges. |
| `self_intersections` | Run bounded triangle-triangle intersection checks and report detected self-intersections. |
| `sliver_triangles` | Report degenerate and high-aspect-ratio triangles. |
| `tiny_parts` | Report parts below the configured diagonal threshold. |
| `draw_call_estimate` | Include material count and estimated draw calls. |
| `visual_risk` | Enable risk-oriented warnings from geometry quality and report steps. |
| `sliver_aspect_ratio` | Aspect-ratio threshold used to classify sliver triangles. |
| `degenerate_area_epsilon` | Triangle area threshold used to classify degenerates. |
| `tiny_part_diagonal` | Bounding-box diagonal threshold used to classify tiny parts. |
| `max_self_intersection_pairs` | Maximum non-adjacent triangle pairs to check before reporting a lower-bound result. |

## Validation

Direct write calls produce files but do not automatically reopen and validate them. Validate direct writes explicitly when you need the same safety as `fc.convert()`.

```python
asset.write_usd("motor.usdc")
usd_stats = fc.validate_usd("motor.usdc")

asset.write_gltf("motor.glb")
gltf_stats = fc.validate_gltf("motor.glb")

stats = fc.validate_output("motor.glb")
```

The CLI can write a validation-time quality report for exported assets:

```bash
fascat validate motor.glb \
  --filter 'material=Painted*' \
  --geometry-quality \
  --report quality-report.json
```

Validation-time geometry reports use the same filter selectors as conversion
when an exported format can be reconstructed for analysis.

## Inspecting assets

Use `to_dict()` for structured inspection or JSON serialization.

```python
asset = fc.read_step("motor.step")

print(asset.part_count)
print(asset.material_count)
print(asset.occurrence_count)

payload = asset.to_dict()
print(payload["root"])
print(payload["parts"])
```

The asset model preserves hierarchy, part records, material records, transforms, units, and source metadata where the STEP backend can read them.

## glTF notes

OpenUSD is the highest-fidelity export path for USD-style LOD variants and instance metadata.

glTF export writes valid glTF 2.0 files for runtime use:

- `.gltf` uses embedded binary buffers
- `.glb` writes a binary glTF container
- geometry is exported in metres and Y-up
- original units and source up-axis are preserved in top-level Fascat extras
- material subsets are exported as separate glTF primitives
- generated LOD meshes are included as Fascat extras and as node-level `MSFT_lod` extension references, with `MSFT_screencoverage` hints when coverage metadata is available
