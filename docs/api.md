---
title: Python API
description: Use fascat from Python
---

The Python API runs the same pipeline as the CLI, but keeps each conversion step
explicit and composable. The package is PEP 561 typed (`py.typed` ships in the
wheel), so mypy and IDEs see the full strict-mode annotations. `import fascat` is lazy — heavy stacks (numpy, Pillow,
runtime harnesses) load on first use, so importing the package costs almost nothing.

## Start here

The simplest path is one call. It runs the full default pipeline for a profile and
validates the output:

```python
import fascat as fc

asset = fc.convert("motor.step", "motor.glb", profile="realtime-web")
print(asset.report.summary())
```

When you need control over individual steps, build the pipeline yourself. Each
method returns a **new** `Asset` and accepts keyword arguments directly — set only
what you want to change:

```python
import fascat as fc

asset = (
    fc.read_step("motor.step")          # or read_iges(...) / read_brep(...)
    .tessellate(sag=0.1, angle=15.0)
    .repair(tolerance=0.05)
    .stage(materials="cad", uv0="box")
    .optimize(target_triangles=500_000)
    .lods([0.5, 0.25, 0.1])
)

asset.write_gltf("motor.glb")
asset.write_usd("motor.usdc")
```

Keyword arguments mirror the matching `*Options` dataclass field-for-field. For a
prebuilt configuration, pass `options=` (or the options object positionally)
instead — never both: `asset.repair(fc.RepairOptions(tolerance=0.05))`.

The rest of this page documents each step and every option. Two things apply
throughout:

- **Reports.** Write calls append a write step to the asset report, and every step
  records its options, before/after counts, and warnings. See [Reports and
  stats](#reports-and-stats).
- **Parallelism.** Mesh-heavy per-part operations fan independent parts out to a
  process pool and reassemble them in deterministic order. `jobs` defaults to
  `min(4, CPU count)` on `RepairOptions`, `MergeVerticesOptions`, `StageOptions`,
  `OptimizeOptions`, `DecimateOptions`, `LODOptions`, and `LODGeneratorOptions`;
  the `FASCAT_JOBS` environment variable overrides the default, and `jobs=1`
  disables pooling entirely. Worker processes start via spawn, so assemblies with
  only a handful of small parts can be faster with `jobs=1`.

Core pipeline calls:

| API | Parameters | Purpose |
|-----|------------|---------|
| `fc.read_step(path, options=None)` | `path` is a STEP file path or `-` for stdin. `options` is `StepReadOptions`. | Import STEP assembly hierarchy, metadata, materials, and source BREP handles when the backend exposes them. With `StepReadOptions(multi_file=True)`, quoted external `.step` / `.stp` references are recursively resolved from a master file and imported as deterministic member occurrences. |
| `fc.read_step_many(paths, options=None, continue_on_error=False)` | `paths` is an ordered list of `.step` / `.stp` files. | Import explicit multi-root STEP members into deterministic per-file namespaces, prefix member warnings, and preserve each member as a top-level root. |
| `fc.read_iges(path, options=None)` | `path` ends in `.igs` or `.iges`. `options` is `IgesReadOptions`. | Import IGES geometry through the same OCP/XDE hierarchy, transform, color, and material path used by STEP. |
| `fc.read_brep(path, options=None)` | `path` ends in `.brep`. `options` is `BrepReadOptions`. | Import a native OpenCASCADE BREP file as one root occurrence and one source-shape part. |
| `asset.tessellate(options=None, *, where=None, **kwargs)` | Keyword args mirror `TessellationOptions`. `where` optionally scopes the operation with a `Filter`. | Convert source BREP geometry into meshes. |
| `asset.repair(options=None, *, where=None, **kwargs)` | Keyword args mirror `RepairOptions`. `where` optionally scopes selected parts. | Clean mesh-level issues after tessellation. |
| `asset.merge_vertices(options=None, *, where=None, **kwargs)` | Keyword args mirror `MergeVerticesOptions`. `where` optionally scopes selected parts. | Merge exact or tolerance-close vertices with attribute and material-boundary protection. |
| `asset.delete_degenerate_polygons(options=None, *, where=None, **kwargs)` | Keyword args mirror `DeleteDegeneratePolygonsOptions`. `where` optionally scopes selected parts. | Remove repeated-vertex, duplicate, or near-zero-area triangles as a standalone cleanup step. |
| `asset.stage(options=None, *, where=None, **kwargs)` | Keyword args mirror `StageOptions`. `where` optionally scopes selected parts. | Prepare materials, normals, tangents, and UV metadata for runtime export. |
| `asset.optimize(options=None, *, where=None, **kwargs)` | Keyword args mirror `OptimizeOptions`. `where` optionally scopes selected parts. | Reduce mesh complexity while preserving selected mechanical features. |
| `asset.lods(options=None, *, where=None, **kwargs)` | `options` may also be a bare ratio sequence; keyword args mirror `LODOptions`. `where` optionally scopes selected parts. | Generate lower-detail runtime meshes. |
| `asset.write_usd(path, options=None)` | `path` ends in `.usd`, `.usda`, `.usdc`, or `.usdz`. `options` is `UsdExportOptions`. | Write OpenUSD output and append a write step to the report. |
| `asset.write_gltf(path, options=None)` | `path` ends in `.gltf` or `.glb`. `options` is `GltfExportOptions`. | Write glTF 2.0 output and append a write step to the report. |
| `asset.write_fbx(path, options=None)` | `path` ends in `.fbx`. `options` is `FbxExportOptions`. | Write ASCII FBX output and append a write step to the report. |

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

Merge modes are `all`, `by_material`, `by_node_name`, `by_part_name`,
`hierarchy_level`, `parent_children`, `final_level`, and `regions`. Merging bakes node
transforms into merged vertex positions, keeps material slots when requested, and
removes replaced empty nodes. The merge report step records a before/after draw-call
breakdown (`draw_calls`, `draw_call_meshes`, `draw_call_materials`,
`draw_call_submesh_slots`, `draw_call_material_slots`, `draw_call_mesh_instances`,
`draw_call_reused_instances`, `draw_call_instanced_meshes`,
`draw_call_merged_batches`). When merging reduces reusable instances, the step adds an
`export_advisor` entry and warning so the GLB file-size, memory, and culling tradeoff
is explicit.

Use `explode()` when runtime tools need separate meshes by material or connected component, and `replace()` when a selected part should become a proxy.

```python
asset = asset.explode(
    fc.options.ExplodeOptions(mode="connected_components"),
    where=fc.Filter.material("rubber"),
)

asset = asset.replace(
    fc.options.ReplaceOptions(mode="bounding_box", preserve_transform=True),
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
    options=fc.options.StepReadOptions(
        metadata=True,
        product_metadata=True,
        properties=True,
        layers=True,
        validation_properties=True,
        pmi=True,
        design_variants=False,
        design_variant_selection=(),
        existing_meshes=True,
        multi_file=False,
        source_textures=True,
        source_texture_search_paths=("textures",),
        material_library_mapping=True,
        material_library_paths=("vendor-materials.json",),
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

### PMI import

When `pmi=True`, STEP AP242 import runs a textual scan and turns supported records
into typed `PmiAnnotation` objects with source STEP entity ids, references, and
numeric tolerance bounds where available. ISO-10303-21 string escape directives
(`\X2\…\X0\`, `\X4\…\X0\`, `\X\HH`, `\S\`, `\P?\`, `\\`) are decoded in PMI text,
design-variant labels, and external/texture references; malformed sequences stay
literal. Supported record families:

- **Dimensions** — `DIMENSIONAL_SIZE`, `DIMENSIONAL_LOCATION`
- **Tolerances** — `PLUS_MINUS_TOLERANCE`, `GEOMETRIC_TOLERANCE`, and named subtypes such as `FLATNESS_TOLERANCE`, `POSITION_TOLERANCE`, `SURFACE_PROFILE_TOLERANCE`
- **Datums** — `DATUM`, `DATUM_REFERENCE`, `DATUM_TARGET`
- **Callouts** — `FEATURE_CONTROL_FRAME` and annotation text entities

Import reports and asset metadata also include `pmi_semantic_graph`: a textual STEP
reference graph of PMI entity nodes, referenced entities, shape-aspect/product
targets, inbound callout/associativity records, tolerance-zone and
annotation-presentation support records, reference edges, and missing-reference
counts.

If a file advertises AP242 PMI but no supported record is extracted, the import
report records `pmi_present=true`, `unsupported_pmi_count=1`, and a warning rather
than implying PMI was imported. `metadata_and_visuals` export adds deterministic
glTF/USD marker meshes with simple vector text glyphs linked to these records.

> Full AP242 graphical presentation reconstruction and semantic coverage beyond
> these textual records is planned backend work.

### Design variants

When `design_variants=True`, import scans STEP configuration, effectivity, and
condition records into asset metadata and the import report, with counts, STEP
references, resolved reference labels, effectivity values, condition operators, and
parsed literal values. Pass `design_variant_selection=(...)` to **prune geometry**
by selected variant.

A selection value can be a variant label, an effectivity value or range, a STEP
record id, a referenced label, or a `label=value` assignment — for example
`("left hand",)`, `("SN-A-050",)`, `("load rating=15",)`,
`("finish=black anodized",)`, or `("service enabled=false",)`. Conditions are
**evaluated before** their operand labels can drive pruning: an `AND_EXPRESSION`
selects only when all operands match, `EQUALS_EXPRESSION` / `COMPARISON_EQUAL`
only when operands resolve equal, `APPLIED_INEFFECTIVITY_ASSIGNMENT` suppresses its
targets, and wrappers gate their configured targets. Operand-only expression
labels are never promoted to geometry selectors.

Supported record families:

- **Configuration & effectivity** — `CONFIGURATION_ITEM`, `PRODUCT_CONCEPT_FEATURE`, `CONFIGURATION_DESIGN`, `CONFIGURATION_EFFECTIVITY`, `PRODUCT_DEFINITION_EFFECTIVITY`, `SERIAL_NUMBERED_EFFECTIVITY`, `LOT_EFFECTIVITY`, `DATED_EFFECTIVITY`, `EFFECTIVITY_RELATIONSHIP`
- **Boolean / comparison conditions** — `AND_EXPRESSION`, `OR_EXPRESSION`, `XOR_EXPRESSION`, `NOT_EXPRESSION`, `EQUALS_EXPRESSION`, `COMPARISON_EQUAL`, `COMPARISON_NOT_EQUAL`, `COMPARISON_GREATER`, `COMPARISON_GREATER_EQUAL`, `COMPARISON_LESS`, `COMPARISON_LESS_EQUAL`, `INTERVAL_EXPRESSION`, `LIKE_EXPRESSION`
- **Numeric arithmetic** — `PLUS_EXPRESSION`, `MINUS_EXPRESSION`, `MULT_EXPRESSION`, `DIV_EXPRESSION`, `SLASH_EXPRESSION`, `MOD_EXPRESSION`, `POWER_EXPRESSION`, `RATIONAL_REPRESENTATION_ITEM`, `EXPRESSION_EXTENSION_NUMERIC`
- **Numeric functions** — `ABS_FUNCTION`, `MINUS_FUNCTION`, `SQUARE_ROOT_FUNCTION`, `MAXIMUM_FUNCTION`, `MINIMUM_FUNCTION`, `SIN_FUNCTION`, `COS_FUNCTION`, `TAN_FUNCTION`, `ASIN_FUNCTION`, `ACOS_FUNCTION`, `ATAN_FUNCTION` (unary, or binary as `atan2`), `EXP_FUNCTION`, `LOG_FUNCTION`, `LOG2_FUNCTION`, `LOG10_FUNCTION`, `ODD_FUNCTION`
- **String expressions** — `CONCAT_EXPRESSION`, `SUBSTRING_EXPRESSION`, `INDEX_EXPRESSION`, `FORMAT_FUNCTION`, `EXPRESSION_EXTENSION_STRING`, and the string-to-numeric `LENGTH_FUNCTION`, `VALUE_FUNCTION`, `INT_VALUE_FUNCTION`
- **Literals & variables** — `BOOLEAN_LITERAL`, `BOOLEAN_REPRESENTATION_ITEM`, `LOGICAL_LITERAL`, `LOGICAL_REPRESENTATION_ITEM`, `BOOLEAN_VARIABLE`, `MATHS_BOOLEAN_VARIABLE`, `STRING_LITERAL`, `MATHS_STRING_VARIABLE` / `STRING_VARIABLE`, numeric/string literals, and named maths variables
- **Wrappers** — `CONDITIONAL_CONFIGURATION`, `CONDITIONAL_CONCEPT_FEATURE`, `CONDITIONAL_EFFECTIVITY`, `CONFIGURED_EFFECTIVITY_ASSIGNMENT`, `APPLIED_EFFECTIVITY_ASSIGNMENT` / `APPLIED_INEFFECTIVITY_ASSIGNMENT`, `APPLIED_EFFECTIVITY_CONTEXT_ASSIGNMENT`, `CONFIGURED_EFFECTIVITY_CONTEXT_ASSIGNMENT`, `CLASS_USAGE_EFFECTIVITY_CONTEXT_ASSIGNMENT`
- **Date bounds** (for serial/date/interval ranges) — `TIME_INTERVAL_WITH_BOUNDS`, `CALENDAR_DATE`, `ORDINAL_DATE`, `WEEK_OF_YEAR_AND_DAY_DATE`

How values resolve:

- **Numeric** comparisons and intervals match named numeric variables supplied as `label=value`, including values flowing through the arithmetic, function, and rational/extension records above (`ATAN_FUNCTION` evaluated as `atan2`; `ODD_FUNCTION` tests odd integers).
- **String** equality, not-equality, and `LIKE_EXPRESSION` match named string variables, with `*`/`%` and `?`/`_` wildcards, flowing through concat/substring/index/extension operands; `FORMAT_FUNCTION` can feed string matching from numeric variables, and `LENGTH`/`VALUE`/`INT_VALUE` functions feed numeric comparisons from strings (strict text parsing).
- **Boolean** literals parse STEP `.T.` / `.F.`; boolean variables act as named operands selected by label, record id, or explicit `label=true` / `label=false`.

> This matches configuration labels against loaded product names. Full AP242
> conditional/effectivity geometry evaluation remains planned backend work.

### Textures and materials

With `source_textures=True`, STEP/IGES import scans source-file string references
for sidecar PNG, JPEG, and KTX2 textures, loads them as first-class
`ImageResource` objects, and binds semantic names (`baseColor`, `normal`, `ao`,
`emissive`) to material metadata. XDE visual PBR values are preserved where exposed,
and common CAD material names (steel, aluminum, brass, copper, glass, plastic,
rubber, paint) map to deterministic PBR defaults.

Vendor material libraries can be supplied via `material_library_paths` or referenced
from the CAD source — as JSON/MTL files, ZIP packages, or folders. Imported records
update matching CAD materials with PBR factors and texture slots, and the import
report records resolved, missing, unreadable, matched, and unmatched counts.

Texture and material-library references found inside CAD file content are confined
to the CAD source directory plus the configured search paths: absolute paths, `..`
traversal, and symlinks that escape every search root are reported as missing instead
of being read. (External STEP assembly references are guarded separately by strict
`.step`/`.stp` extension validation.) Library
files passed explicitly via `material_library_paths` are trusted CLI/API input and
are not subject to confinement.

Auxiliary text scans are also size-capped: STEP files over 64 MiB skip the textual
PMI/variant/reference passes with a report warning (geometry import is unaffected),
and sidecar material libraries (16 MiB) or textures (64 MiB) over their caps are
reported unreadable instead of being loaded into memory.

### Construction curves

Construction-only line shapes follow the `construction_curve_policy`:

- `preserve_metadata` (default) — keep the source shape; report it has no triangle mesh
- `delete` — drop construction-only line nodes during import (legacy `delete_lines=True` is an alias)
- `tessellate_tubes` — convert curve segments into low-sided triangle tubes during tessellation, using `construction_curve_tube_radius` (source units)

STEP import also splits free construction edges out of mixed face+curve shapes so the
same policy applies. With `keep_brep=True`, the original mixed topology stays
available on the BREP part. Import reports record `import_decisions` (each toggle as
requested/effective plus a status) and `loaded_representations` (per-part input kind
and source topology counts).

### Multi-file assemblies

Two cases, both imported through the normal STEP path with deterministic member
namespacing:

- **Several root files** — `fc.read_step_many([...])`, `fc.convert([...], "out.glb")`, or `fascat convert root-a.step out.glb --input root-b.step`. Members are namespaced and kept under a shared root; warnings are prefixed with member index and path. `continue_on_error=True` keeps successful members and records failures.
- **One master file with external references** — `StepReadOptions(multi_file=True)` (or `--multi-file-import`). Quoted `.step`/`.stp` references are resolved relative to the referencing file and followed once per source; repeated references become separate member occurrences. The report includes `external_reference_graph` (resolved/missing/unsupported/unique-source/occurrence counts), and missing references warn rather than disappear.

> Multi-file import is graph-level loading, not deep reconstruction of every
> vendor-specific external-reference placement transform.

Metadata and PMI parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `StepReadOptions` | `metadata` | Enables general source metadata import. If `False`, the more specific metadata import groups are disabled by default. |
| `StepReadOptions` | `product_metadata` | Import product and assembly-level metadata where the STEP backend exposes it. |
| `StepReadOptions` | `properties` | Import user and product properties. |
| `StepReadOptions` | `layers` | Request layer assignments as metadata. Current normalized layer extraction is reported as unsupported in `import_decisions` when requested. |
| `StepReadOptions` | `validation_properties` | Request STEP validation properties. Current reports approximate this with source topology counts rather than typed validation-property entities. |
| `StepReadOptions` | `pmi` | Import typed AP242 PMI records into `PmiAnnotation` objects and report the `pmi_semantic_graph`. See [PMI import](#pmi-import). |
| `StepReadOptions` | `design_variants` | Scan STEP configuration, effectivity, and condition records into metadata and import reports. See [Design variants](#design-variants). |
| `StepReadOptions` | `design_variant_selection` | One or more selection values (variant labels, effectivity values/ranges, record ids, or `label=value` assignments) used to prune imported geometry. See [Design variants](#design-variants) for the full resolution rules. |
| `StepReadOptions` | `existing_meshes` | Prefer existing tessellation payloads from the source file when the importer exposes them. Tessellation `reuse_existing_meshes` still controls whether loaded meshes are retessellated later. |
| `StepReadOptions` | `multi_file` | Request multi-file STEP assembly import. `read_step_many()` honors explicit member lists; single-path STEP imports recursively resolve quoted external `.step` / `.stp` references, preserve repeated references as member occurrences, and report the `external_reference_graph`. |
| `StepReadOptions` | `source_textures` | Scan STEP/IGES source text for referenced sidecar PNG/JPEG/KTX2 texture files, load resolved files into `asset.images`, and report resolved/missing/unreadable counts. |
| `StepReadOptions` | `source_texture_search_paths` | Extra directories used to resolve relative source texture references in addition to the CAD file directory. |
| `StepReadOptions` | `material_library_mapping` | Apply deterministic CAD material-name mapping rules to PBR metallic, roughness, opacity, and default color values when source visual material names are available. |
| `StepReadOptions` | `material_library_paths` | Explicit vendor material-library JSON/MTL/ZIP files or folders to load during STEP/IGES import. Referenced library files are resolved relative to the CAD source and texture search paths. |
| `StepReadOptions` | `material_library_color_space` | Numeric material-library color interpretation: `auto` preserves 0-1 or 0-255 detection, `linear` clamps direct factors, and `srgb255` treats numeric colors as 0-255 values. |
| `StepReadOptions` | `delete_free_vertices` | Drop construction-only point shapes during import and record deletion counts in the import report. |
| `StepReadOptions` | `delete_lines` | Legacy alias for deleting construction-only line shapes during import. Free construction edges split from mixed face+curve shapes follow the same delete policy. |
| `StepReadOptions` | `construction_curve_policy` | Construction line policy for construction-only shapes and free construction edges split from mixed face+curve shapes: `preserve_metadata`, `delete`, or `tessellate_tubes`. Tube tessellation happens when the asset is tessellated. |
| `StepReadOptions` | `construction_curve_tube_radius` | Tube radius in source units for `construction_curve_policy="tessellate_tubes"`. |
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
| `MetadataExportOptions` | `pmi` | Export PMI as `none`, `summary`, `metadata`, `metadata_and_visuals`, or `full`. `metadata_and_visuals` emits metadata records, stable links, and deterministic glTF/USD marker geometry with simple vector text glyphs; full AP242 visual presentation reconstruction remains planned. |

```python
asset.write_gltf(
    "motor.glb",
    options=fc.GltfExportOptions(
        metadata=fc.options.MetadataExportOptions(mode="full", pmi="metadata"),
    ),
)

asset.write_usd(
    "motor.usdc",
    options=fc.UsdExportOptions(
        metadata=fc.options.MetadataExportOptions(mode="full", pmi="metadata_and_visuals"),
    ),
)
```

## BREP Healing

Run BREP healing before tessellation when STEP topology needs sewing, edge fixing, tolerance unification, or open-shell and unstitched-edge reporting.

```python
asset = fc.read_step("motor.step").heal_brep(
    fc.options.BrepHealOptions(
        tolerance=0.05,
        group_open_shells=True,
        sew_faces=True,
        fix_edges=True,
        unify_same_domain=True,
        remove_overlapping_faces=True,
        overlap_area_ratio=0.995,
        remove_sliver_faces=True,
        max_sliver_area=1e-4,
        unify_tolerances=True,
        fail_on_open_shells=False,
    ),
    where=fc.Filter.path("*/Housing/*"),
)
```

Healing stores per-part `brep_*` metadata and records a `heal_brep` report step.
You can also run it inside a one-shot conversion: `fc.convert(..., heal_brep=fc.options.BrepHealOptions())`.

What it does:

- **Open-shell grouping** processes disconnected shell groups independently, so unrelated surface patches aren't forced through one global sewing pass.
- **Same-domain cleanup** uses OCCT to merge neighboring faces/edges on coincident surfaces and curves.
- **Overlap cleanup** triangulates faces, measures coplanar overlap against `overlap_area_ratio`, and removes redundant z-fighting faces with OCCT `BRepTools_ReShape`.
- **Sliver removal** is requested through the backend; when unavailable it warns rather than claim the shape changed.

What it records:

- **Metadata** — BREP kind; solid/shell/wire/edge/face counts; open shells; free/unstitched edges; small edges at or below tolerance; sliver counts; overlap and resolved-overlap counts; open-shell grouping counts; same-domain reductions.
- **`tolerance_policy`** — effective source/local and target units, meters-per-unit conversions, tolerance and sliver area in metric units, and whether each cleanup stage is enabled, disabled, requested, or not implemented.
- **Warnings** for remaining open shells, free edges, small edges, or unresolved overlapping face pairs.

Brep healing parameters:

| Parameter | Meaning |
|-----------|---------|
| `tolerance` | Working tolerance used for sewing, edge fixes, and tolerance unification. Must be greater than zero. |
| `group_open_shells` | Group disconnected open shell patches before running the BREP cleanup stack. |
| `sew_faces` | Attempt to sew adjacent faces into shells before tessellation. |
| `fix_edges` | Attempt to repair bad trims and edge curves where supported by the backend. |
| `unify_same_domain` | Merge neighboring faces/edges that lie on the same OCCT surface or curve domain. |
| `remove_overlapping_faces` | Remove redundant coplanar faces whose projected overlap would cause z-fighting. |
| `overlap_area_ratio` | Minimum overlap ratio against the smaller face before an overlapping face is removed. |
| `remove_sliver_faces` | Request tiny sliver-face removal before tessellation. Current backend support is limited and reports a warning when removal is unavailable. |
| `max_sliver_area` | Area threshold for sliver-face removal. |
| `unify_tolerances` | Normalize shape tolerances to the requested working tolerance. |
| `fail_on_open_shells` | Raise when healing detects open shells instead of reporting a warning. |
| `where` | Optional filter that limits healing to selected assembly occurrences. |

## Tessellation Controls

Tessellation supports global and per-part settings for edge limits, boundary preservation, curvature-adaptive OCCT meshing, material/metadata-driven detail adaptation, skinny-triangle cleanup, and per-part quality metrics.

```python
asset = fc.read_step("motor.step").tessellate(
    fc.TessellationOptions(
        sag=0.05,
        sag_ratio=None,
        angle=10.0,
        min_edge_length=0.02,
        max_edge_length=2.0,
        max_polygon_length=4.0,
        preserve_boundaries=True,
        curvature_adaptive=True,
        detail_adaptive=True,
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

`part_settings` keys match a part id or part name. A few behaviors worth knowing:

- **Quality reports** (`quality_report=True`) record per-part edge length, triangle area, aspect ratio, skinny-triangle, duplicate-polygon, boundary-edge, and non-manifold-edge counts, plus advisories for coarse absolute sag, aggressive polygon-length limits, and shiny/high-detail/curved parts lacking `sag_ratio` or `curvature_adaptive`.
- **`detail_adaptive=True`** turns shiny/high-detail material metadata and curved-BREP detection into per-part settings — affected parts get `sag_ratio=0.01` (when unset) and `curvature_adaptive=True`. Explicit `part_settings` always win.
- **`tolerance_policy`** on the report step (and on part/mesh metadata) records the active deflection kind, source/target units, meters-per-unit conversions, and converted unit-bearing values; it warns when unit normalization makes sag or max length suspiciously coarse or fine.
- **Risk metadata** — parts record `tessellation_face_groups`, `tessellation_estimated_draw_calls`, and retained-patch counts; the step warns when retained patches, face groups, or material splits could raise submesh/draw-call/export pressure.
- **Provenance** — `tessellation_attribute_sources` records whether positions, triangles, normals, tangents, UVs, face groups, free-edge diagnostics, and BREP patches came from tessellation, an imported mesh, or weren't generated. `tessellation_edge_control_passes` is `2` only when a cleanup pass was rerun after the first edge-control pass changed the mesh.

Size-adaptive tessellation helpers can generate `part_settings` from part
bounding-box diagonals, using existing mesh bounds or source BREP bounds when
the OCCT backend is available:

```python
asset = fc.read_step("motor.step")
tessellation = fc.profiles.size_adaptive_tessellation(
    asset,
    base=fc.TessellationOptions(sag=0.1, angle=15.0, quality_report=True),
    bands=(
        fc.profiles.TessellationSizeBand(max_diagonal=25.0, sag=0.02, angle=8.0, max_polygon_length=1.0),
        fc.profiles.TessellationSizeBand(max_diagonal=None, sag=0.12, sag_ratio=0.01, angle=18.0),
    ),
)
asset = asset.tessellate(tessellation)
```

Explicit `part_settings` on `base` remain authoritative; the helper only fills
settings for parts that do not already have a part-id or part-name override.

Tessellation parameters:

| Parameter | Meaning |
|-----------|---------|
| `sag` | Maximum chordal deviation between source surface and tessellated mesh. Lower values produce more triangles. |
| `sag_ratio` | Relative chordal deviation ratio. When set, it becomes the backend deflection value and enables relative tessellation explicitly. |
| `angle` | Angular deviation limit in degrees. Lower values preserve curved surfaces with more triangles. |
| `relative` | Compatibility switch for interpreting `sag` as a relative backend deflection when `sag_ratio` is unset. Prefer `sag_ratio` for new relative-tolerance workflows. |
| `min_edge_length` | Collapse or avoid edges shorter than this length during post-processing. |
| `max_edge_length` | Split long triangle edges to keep mesh density bounded. Very small values on non-elongated parts emit a quality advisory because this control is most useful for long objects with lighting artifacts. |
| `max_polygon_length` | Report tessellated polygon edges longer than this threshold without subdividing geometry. Quality reports count these as `long_edges`; the tessellation step emits warnings when exceeded and advises when the limit is aggressive for ordinary parts. |
| `preserve_boundaries` | Preserve CAD face and boundary edges during tessellation cleanup. |
| `curvature_adaptive` | Request curvature-aware meshing from the backend when available. |
| `detail_adaptive` | Auto-apply finer per-part tessellation to shiny material, high-detail metadata, or curved BREP-face parts by enabling curvature-adaptive meshing and setting `sag_ratio=0.01` when unset. |
| `avoid_skinny_triangles` | Run a cleanup pass that reduces long skinny triangles. |
| `quality_report` | Record per-part tessellation quality metrics and advisories for later reporting. Coarse absolute sag relative to part size is flagged when `relative=False` and no `sag_ratio` is set; shiny, high-detail, or curved BREP parts also advise per-part `sag_ratio` or `curvature_adaptive` tuning. |
| `free_edge_report` | Record free/boundary edge and non-manifold edge counts on tessellated parts and warn when free edges are present. |
| `create_normals` | Generate normals during tessellation when the backend can provide them. Attribute provenance records `tessellation`, `disabled`, or `missing` for normals. |
| `keep_brep` | Keep source BREP handles on parts after tessellation for later BREP-aware operations. When `False`, source BREP handles are dropped even when an imported mesh is reused instead of retessellated. Tessellated parts record `brep_patch_cleanup=retained` or `deleted` and warn when many retained patches could increase runtime/export risk. |
| `reuse_existing_meshes` | Reuse meshes already present on imported parts. Set to `False` to retessellate from source BREP where available. |
| `part_settings` | Per-part overrides keyed by part id or part name. Supports the same tessellation option names. |

Repair parameters:

| Parameter | Meaning |
|-----------|---------|
| `tolerance` | Merge tolerance for nearby vertices. `0.0` disables distance-based merging beyond exact duplicates. |
| `merge_vertices` | Deduplicate vertices after tessellation. |
| `delete_degenerate` | Remove triangles with repeated vertices or near-zero area. |
| `fix_winding` | Normalize triangle winding where a consistent orientation can be inferred, including inward closed components detected by signed volume. |
| `quality_report` | Run heavier before/after repair diagnostics for duplicate polygons, degenerate triangles, boundary and non-manifold edges, T-junctions, boundary gaps, and orientability. Defaults to `False` so conversion repair does geometric cleanup without paying for report-only topology scans. |
| `face_orientation` | Face-orientation policy: `exterior` runs the current closed-component outward winding path; `source_trusted` and `preserve` keep source winding; `viewer_standpoint`, `single_sided_open_shell`, and `unstitched_groups` are recorded as intent until those strategies have a backend. |
| `normal_orientation` | Normal-orientation policy: `from_faces` regenerates normals from repaired faces; `source_trusted` and `preserve` keep compatible source normals when possible; `viewer_standpoint` is recorded as intent until implemented. |
| `viewer_position` | Three-number viewer position required when either orientation policy is `viewer_standpoint`. Recorded in metadata and reports. |
| `fill_small_holes` | Fill small mesh boundary loops as a fallback mesh repair step. Fill faces inherit the material of the nearest neighboring face. |
| `area_epsilon` | Area threshold used to classify degenerate triangles. Defaults to a scale-invariant value derived from the mesh bounding box (1e-12 × squared diagonal); pass a value to override. |
| `jobs` | Worker count for independent mesh-bearing parts. `1` keeps serial behavior. |

Repair always records orientation policy metadata
(`repair_face_orientation_strategy`/`_status`, `repair_normal_orientation_strategy`/`_status`,
and optional `repair_orientation_viewer_position`), so viewer-standpoint or
source-trusted choices stay visible even when the backend preserves source
orientation. The report step also records a unit-aware `tolerance_policy` covering
merge tolerance, degenerate area epsilon, and the status of each cleanup stage.

With `quality_report=True`, repair adds before/after counts for duplicate polygons,
degenerate triangles, boundary edges, non-manifold edges, T-junctions, boundary
gaps, and orientability. Note:

- **Duplicate polygons** are triangles referencing the same three vertices, regardless of winding.
- **T-junctions** and **boundary gaps** are reported by default but only fixed when the opt-in `fix_t_junctions` / `stitch_boundary_gaps` flags are set — non-zero counts after a report-only repair emit a warning. Stitched vertices keep the attributes (normals, tangents, UVs) of the surviving representative vertex; when merged vertices disagreed on UVs, the count of conflicting merges is recorded as `boundary_gap_stitching_uv_conflicts` metadata.

`MergeVerticesOptions` gives you vertex merging as a standalone step (rather than the
broad repair pass). `tolerance=0.0` merges exact duplicate positions; larger values
merge Euclidean-close positions, including across spatial bucket boundaries. By
default normals, tangents, UVs, and material-boundary signatures are part of the
merge key so hard edges and UV seams aren't collapsed — set `preserve_normals=False`,
`preserve_tangents=False`, or `preserve_uvs=False` to drop that protection. Reports
include removed counts, tolerance scale ratios (against bounding-box diagonal and
shortest edge), high-risk-tolerance warnings, and `tolerance_policy`;
`quality_report=True` (or `--merge-vertex-quality-report`) adds candidate counts and
skipped-merge reasons. Use `jobs` to process independent parts concurrently.

Use `DeleteDegeneratePolygonsOptions` when you want Unity-style degenerate
polygon cleanup as a standalone, reproducible step. `area_epsilon` controls the
near-zero-area threshold, and `delete_duplicates=True` also removes exact
duplicate polygons that reference the same three vertices regardless of winding.
The operation always writes a report step, even when no polygons are removed,
and per-part metadata records before/after degenerate and duplicate-polygon
counts, removed triangle counts, removed unreferenced vertices, primary removal
reasons for duplicate vertices, collapsed edges, near-flat area, and duplicate
polygons, plus the unit-aware area threshold.

`DeleteDegeneratePolygonsOptions` parameters:

| Parameter | Meaning |
|-----------|---------|
| `area_epsilon` | Area threshold used to classify near-flat triangles as degenerate. Defaults to a bounding-box-derived, scale-invariant value. |
| `delete_duplicates` | Remove exact duplicate polygons after degenerate triangles are removed. |

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
| `jobs` | Worker count for independent mesh-bearing parts. `1` keeps serial behavior. |

## Hard-Edge Normals And Tangents

Staging can generate smooth, flat, or hard-edge normals and glTF-ready tangents. Smooth and hard-edge normals can use angle or area weighting. Hard-edge mode splits vertices across hard normal edges, material boundaries, and optional CAD face-group boundaries.

```python
asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        normals=True,
        normal_mode="hard_edges",
        normal_weighting="angle",
        hard_edge_angle=30.0,
        preserve_face_boundaries=True,
        override_normals=True,
        tangents=True,
        tangent_uv_channel=0,
        override_tangents=False,
        validate_normals=True,
        uv0="box",
    )
)
```

Tangent generation needs the selected UV channel (`tangent_uv_channel`, default UV0).
If tangents are requested without it, staging records missing-UV metadata and warns
rather than silently writing none. Existing tangents are preserved unless staging
invalidated them and the channel is still present; set `override_tangents=True` to
force regeneration. When UV generation edits a mesh that already had tangents, the old
basis is invalidated — staging regenerates from the selected channel if `tangents=True`,
otherwise records the dropped state. glTF export writes a `TANGENT` attribute whenever
staged meshes carry tangent data.

Normal and tangent parameters:

| Parameter | Meaning |
|-----------|---------|
| `normals` | Generate or preserve vertex normals. Automatically disabled when `normal_mode="none"`. |
| `normal_mode` | `smooth` averages face normals, `flat` keeps face normals, `hard_edges` splits vertices along hard edges, and `none` omits normals. |
| `normal_weighting` | `angle` weights smooth or hard-edge normals by corner angle; `area` weights by triangle area. |
| `hard_edge_angle` | Edge angle threshold in degrees for `normal_mode="hard_edges"`. |
| `preserve_face_boundaries` | Treat CAD face-group boundaries as hard normal boundaries. |
| `override_normals` | Regenerate existing normals. Set `False` to preserve existing normals and only generate normals when missing. |
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
        unwrap=fc.options.UnwrapOptions(
            texel_density=256.0,
            padding=4,
            max_stretch=0.15,
            method="conformal",
            iterations=32,
            tolerance=0.001,
            sharp_to_seam=True,
            forbid_overlapping=True,
        ),
        atlas=fc.options.AtlasOptions(
            enabled=True,
            max_size=4096,
        ),
        normalize_uvs=(1,),
    )
)
```

Atlas options on staging record texture-bake intent and layout limits. Dedicated material baking is the step that writes raster atlas images: baked maps are stored as first-class `ImageResource` objects, mirrored into material metadata for compatibility, and bound by the glTF/USD exporters as material textures.

When `merge_equivalent_materials=True`, staging groups materials by PBR factors after rounding `base_color`, `metallic`, `roughness`, and `opacity` to six decimal places. This absorbs floating-point noise from importers while keeping visibly distinct values separate; the precision is fixed.

Staging records detailed per-channel UV metadata (fields are prefixed `uvN_`, where
`N` is the channel index):

- **Domains** — UV0 defaults to the `tileable` domain (overlaps and coordinates outside 0..1 are fine). UV1 and `lightmap` channels use the `bake` domain, where overlaps, degenerate faces, or out-of-unit coordinates set `uvN_validation_status` and add warnings.
- **Layout quality** — every channel records `uvN_domain`, `uvN_bounds`, `uvN_validation_status`, `uvN_out_of_unit_vertices`, `uvN_degenerate_faces`, and `uvN_overlap_check`. Expensive overlap checks run only for bake-domain UVs or when `forbid_overlapping=True`; skipped channels record `uvN_overlap_pairs="not_evaluated"`.
- **Seam graph** — duplicated-position UV discontinuities are reported per channel (`uvN_seam_edges`, `uvN_seam_components`, `uvN_seam_length`, …) and summarized on the asset as `stage_uv_seam_graph_channels` / `stage_uv_seam_graph_edges`.
- **Distortion** — bake-domain channels, or any channel staged with `max_stretch`, record `uvN_island_count`, `uvN_pack_efficiency`, and angle/edge distortion fields; other channels record `uvN_distortion_check="skipped"`.
- **Packing** — `unwrap`/`lightmap` bake channels are packed by xatlas with the configured padding/resolution and record `uvN_pack_status`, dimensions, and utilization.
- **Box projection** — `box` channels run an AABB projection and record `uvN_projection_*` fields for local/shared bounds, axes, destination, override policy, units, and `uv3d_size`.

Two convenience modes: `uv1="copy_uv0"` reuses the UV0 layout for the secondary
channel (warns if UV0 is missing), and `normalize_uvs=(1,)` rescales selected
channels into 0..1 (warns if a requested channel is absent). Sharp-to-seam and
forbid-overlap controls are recorded as intent and validated after generation,
since the current xatlas path doesn't expose them directly.

```python
asset = asset.stage(
    fc.StageOptions(
        uv0="box",
        aabb_projection=fc.options.AabbProjectionOptions(
            scope="shared",
            uv3d_size=1.0,
            override_existing=True,
        ),
    )
)
```

When `uv0` or `uv1` uses `unwrap` or `lightmap`, fascat uses the optional xatlas backend for flattening, packing, and padding. `method`, `iterations`, `tolerance`, `sharp_to_seam`, and `forbid_overlapping` still record Unity-style solver and policy intent when xatlas does not expose a direct equivalent; Fascat validates the generated UVs and reports the actual pack status instead of treating the request as silently honored.

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
| `StageOptions` | `aabb_projection` | `AabbProjectionOptions` used when a UV channel uses `box` projection. |
| `StageOptions` | `jobs` | Worker count for independent mesh-bearing parts. `1` keeps serial behavior. |
| `AabbProjectionOptions` | `scope` | `local` projects each part against its own AABB; `shared` projects selected parts against one shared AABB. |
| `AabbProjectionOptions` | `uv3d_size` | Optional real-world size per UV tile. When unset, coordinates are normalized to the chosen AABB axes. |
| `AabbProjectionOptions` | `override_existing` | Replace existing destination-channel UVs during box projection. Set `False` to preserve existing UVs and record that choice. |
| `UnwrapOptions` | `texel_density` | Desired texture density for generated UVs. |
| `UnwrapOptions` | `padding` | Padding between UV islands in pixels. |
| `UnwrapOptions` | `max_stretch` | Maximum tolerated UV stretch before reporting unwrap risk. |
| `UnwrapOptions` | `method` | Requested unwrap solver intent: `default`, `conformal`, or `isometric`. Non-default values are recorded as intent with the xatlas backend. |
| `UnwrapOptions` | `iterations` | Requested unwrap solver iteration budget. Recorded as intent until a backend exposes this control. |
| `UnwrapOptions` | `tolerance` | Requested unwrap solver error threshold. Recorded as intent until a backend exposes this control. |
| `UnwrapOptions` | `sharp_to_seam` | Request sharp edges as UV seams for unwrap/lightmap channels. Recorded as intent until a backend exposes explicit seam policy controls. |
| `UnwrapOptions` | `forbid_overlapping` | Require non-overlapping UV islands. When set explicitly, staging raises `UVOverlapError` if overlapping UV faces remain; bake-domain channels (UV1/lightmap) are always checked and warn loudly by default. |
| `AtlasOptions` | `enabled` | Record atlas metadata and prepare materials for later baking. |
| `AtlasOptions` | `max_size` | Maximum atlas texture size in pixels. |

## Scene Optimization

Use scene optimization to reduce draw calls after staging and optional hierarchy
merging. It:

- batches compatible meshes (optionally by material) and splits oversized merged meshes;
- reconstructs repeated mesh instances — exact matches, or near-identical ones within a position tolerance — when vertex attributes, materials, and metadata match;
- simplifies empty hierarchy and annotates the intended index-buffer width;
- reports duplicate vertex/triangle counts, estimated payload-byte savings, and how mesh/material counts, submesh/material slots, instances, and merged batches contributed to the draw-call estimate.

When batching removes reusable instances, the report includes the same export
advisor used by explicit merge operations.

```python
asset = asset.optimize_scene(
    fc.options.SceneOptimizeOptions(
        batch_by_material=True,
        merge_compatible_meshes=True,
        split_large_meshes=True,
        max_vertices_per_mesh=65_535,
        index_buffer="auto",
        flatten="safe",
        remove_empty_nodes=True,
        instance_policy="auto",
        instance_similarity_tolerance=0.0,
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
| `instance_similarity_tolerance` | Position tolerance for reconstructing near-identical repeated meshes with matching topology, vertex attributes, material assignments, and metadata. `0.0` keeps exact fingerprint matching only. |

## Optimization Actions

Use explicit optimization actions when a realtime pipeline needs named preparation steps and separate report entries for each action.

```python
asset = asset.bake_materials(
    fc.options.BakeMaterialOptions(
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
        preserve_painted_areas=True,
        preserve_ambient_occlusion=True,
        budget_scope="selection",
        uv_importance="preserve_islands",
        cleanup_attributes=("unused_uvs", "tangents"),
        iterative_threshold=1_000_000,
    )
)

asset = asset.remove_holes(fc.options.RemoveHolesOptions(max_diameter=3.0, prefer_brep=True))
asset = asset.remove_occluded(fc.options.RemoveOccludedOptions(strategy="advanced", level="triangles"))
asset = asset.run_lod_generators(
    fc.options.LODGeneratorOptions(
        preset="vr",
        levels=(
            fc.options.LODLevel(screen_coverage=0.5, target_ratio=0.5),
            fc.options.LODLevel(screen_coverage=0.2, target_ratio=0.25),
            fc.options.LODLevel(screen_coverage=0.05, target_ratio=0.1),
        ),
        validate=True,
    )
)
```

**Material baking** creates a shared flat material plus raster atlas images from
selected maps and per-face assignments. Images are stored as `ImageResource` objects
and bound by glTF/USD exports through texture slots or `UsdUVTexture` networks.
When multiple source materials are collapsed into the baked output material, Fascat uses a simple arithmetic mean for base color, metallic, roughness, and opacity. The average is not currently weighted by face area, texel coverage, or material usage.
`ambient_occlusion_strategy` selects conservative, exterior, or advanced direction
sets when baking AO maps or protecting low-AO faces during decimation.
Emissive bakes record `baked_emissive_source` plus material/fallback face counts
so explicit material emission can be distinguished from the black fallback.

**Hole removal** uses mesh boundary classification and filling when BREP feature
editing is unavailable. **Occlusion removal** uses deterministic visibility sampling;
asset metadata records sample coverage, direction coverage, and a confidence score,
and the report notes that thin occluders may need higher precision.

**Decimation** records what it did and how aggressively:

- **Strategy** — `target_strategy` (and `decimate_target_strategy` metadata) marks the run as explicit target count, target ratio, or quality/error-bounded; `decimate_requested_keep_ratio` is recorded when derivable. Keeping under 20% of source triangles warns, since that suits distant LODs more than LOD0.
- **Memory & passes** — a RAM estimate (Unity's ~5 GB per million source triangles), `iterative_threshold` control, and `decimate_simplification_passes` / `decimate_iterative_passes` / `decimate_iterative_recommended`.
- **Selection budgets** — selection-wide runs record per-part target allocation (assigned targets, reduced-vs-preserved counts, min/max), showing which dense parts absorbed the reduction.
- **Protected features** — counts for hard edges, hole boundaries, material boundaries, UV seams, and silhouette faces. `preserve_painted_areas` protects painted/protected/weighted/important face groups; `preserve_ambient_occlusion` protects low-AO faces.
- **UV handling** — `uv_importance` is `ignore` (strip UV/tangents first), `preserve_seams` (use then strip), or `preserve_islands` (keep through output); `cleanup_attributes` removes unused UV channels/tangents before simplification.

**LOD generation** simplifies progressively — LOD1 from the source mesh, later levels
from the previous LOD, each preserving its requested ratio against the source count.
Parts without tessellated meshes are skipped (`lod_status="skipped_no_mesh"`, plus
`lod_generated_parts` / `lod_skipped_no_mesh_parts` on the asset). Reports record
source/added/full-chain vertex/triangle counts and payload bytes, per-level
provenance (instance reuse, material merge, texture bake, culling-granularity
changes, resolved export representation), and chain advisories (more than four
levels; over-aggressive LOD1/LOD2; geometry-only far LODs that should bake to
one mesh/material). `engine_profile="unity"` exports LODs as `MSFT_lod` variant
nodes; `"unreal"` exports separate `_LOD#` scene nodes for pipelines that ignore
`MSFT_lod`.

Optimization action parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `BakeMaterialOptions` | `maps_resolution` | Raster atlas texture size in pixels for generated baked maps. |
| `BakeMaterialOptions` | `force_uv_generation` | Generate UVs first when selected meshes do not have the required UV channel. |
| `BakeMaterialOptions` | `uv_channel` | UV channel used for baking. |
| `BakeMaterialOptions` | `padding` | Texture padding between islands in pixels. |
| `BakeMaterialOptions` | `bake` | Maps to bake, such as `base_color`, `opacity`, `normal`, `roughness`, `metallic`, `ao`, or `emissive`. |
| `BakeMaterialOptions` | `merge_output` | Replace selected materials with a shared baked output material. |
| `BakeMaterialOptions` | `ambient_occlusion_strategy` | Direction set for baked AO maps: `conservative`, `exterior`, or `advanced`. |
| `DecimateOptions` | `criterion` | `target` prioritizes a triangle budget. `quality` passes the largest configured tolerance as a target error bound to the simplification backend and records bound/result metadata. Report `target_strategy` identifies whether the effective workflow was target count, target ratio, or quality/error-bounded simplification. |
| `DecimateOptions` | `target_triangles` | Absolute triangle target for selected geometry. In the CLI, `--decimate` uses the selected profile or target-device triangle budget when no explicit target or ratio is supplied. |
| `DecimateOptions` | `target_ratio` | Fraction of source triangles to keep when no absolute target is set. Ratios below 20% produce an LOD0 distortion warning. |
| `DecimateOptions` | `surface_tolerance` | Surface tolerance input used by `criterion="quality"` to derive the simplification target error bound. |
| `DecimateOptions` | `line_tolerance` | Line-feature tolerance input included in the quality target error bound. |
| `DecimateOptions` | `normal_tolerance` | Maximum normal deviation in degrees. |
| `DecimateOptions` | `uv_tolerance` | UV tolerance input included in the quality target error bound. |
| `DecimateOptions` | `protect_topology` | Avoid topology changes that would remove important boundaries. Reports include protected hard-edge, hole-boundary, material-boundary, UV-seam, silhouette, and total feature-face counts. |
| `DecimateOptions` | `preserve_painted_areas` | Preserve face groups or metadata-marked face indices named as painted, protected, weighted, or important. Reports include painted-area and combined importance-face counts. |
| `DecimateOptions` | `preserve_ambient_occlusion` | Preserve low-AO faces from the sampled AO estimator during decimation. Reports include ambient-occlusion and combined importance-face counts. |
| `DecimateOptions` | `ambient_occlusion_strategy` | Direction set for the low-AO estimator used by `preserve_ambient_occlusion`: `conservative`, `exterior`, or `advanced`. |
| `DecimateOptions` | `budget_scope` | `part` budgets each part separately. `selection` uses a global selected-geometry target so sparse/simple parts can stay intact while dense parts absorb more reduction. Global selection decimation reports per-part target allocation, estimated RAM, and iterative-threshold status. |
| `DecimateOptions` | `uv_importance` | Texture-coordinate handling: `preserve_islands` keeps UVs, `preserve_seams` protects seam topology then drops UVs, and `ignore` strips UVs/tangents before decimation. |
| `DecimateOptions` | `cleanup_attributes` | Pre-decimation cleanup for attribute streams that are not useful to simplification. `unused_uvs` removes empty, constant, or zero-area UV channels. `tangents` removes tangents before simplification. Reports record removed channels, removed tangent parts, preserved UV channels, and UV seam/island constraint status. |
| `DecimateOptions` | `iterative_threshold` | Source-triangle threshold above which decimation inserts intermediate simplification passes before the final target and reports actual pass counts. |
| `DecimateOptions` | `jobs` | Worker count for independent mesh-bearing parts. `1` keeps serial behavior. |
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
| `LODOptions` | `engine_profile` | Switch-distance and glTF export profile: `generic`, `unity`, or `unreal`. `unity` resolves to `MSFT_lod` variant export; `unreal` resolves to separate `_LOD#` scene nodes for import tools that ignore `MSFT_lod`. |
| `LODOptions` | `far_lod_bake` | For far-distance levels, collapse material indices to a one-material far LOD policy and record far texture-bake metadata. |
| `LODOptions` | `scene_far_proxy` | Build an optional scene-level far proxy part from the final LOD occurrence geometry as one mesh, one material, and one draw-call proxy. glTF export attaches it as root `MSFT_lod` metadata. |
| `LODOptions` / `LODGeneratorOptions` | `jobs` | Worker count for independent mesh-bearing parts. `1` keeps serial behavior. |
| `LODLevel` | `screen_coverage` | Screen fraction at which this LOD becomes appropriate. |
| `LODLevel` | `target_ratio` | Fraction of source triangles to keep for this LOD. |
| `LODOptions` / `LODGeneratorOptions` | report metadata | LOD steps record `lod_source_*`, `lod_added_*`, and `lod_chain_*` counts for vertices, triangles, and estimated mesh payload bytes, plus per-level vertex/triangle counts, simplification source, omitted tiny-part LOD counts, instance-reuse counts, material-merge counts, texture-bake counts, culling-granularity change counts, scene-far-proxy counts, resolved export mode, and LOD chain advisory counts/codes. |

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
  "warnings": []
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
fc.convert("motor.step", "motor.glb", profile="mixed-reality")
fc.convert("motor.step", "motor.gltf", profile="realtime-web")
fc.convert("legacy.igs", "legacy.glb")
fc.convert("native.brep", "native.usdc")
```

`fc.convert()` validates generated output by default. Pass `validate_output=False` only when another step in your pipeline validates the asset.
When `where` is provided to `fc.convert()`, tessellation, repair, and staging still run for the full asset, while standalone vertex merging, standalone degenerate-polygon cleanup, hierarchy merge, scene optimization, optimization actions, optimization, and LOD generation are scoped to the matched assembly subset.

Conversion parameters:

| Parameter | Meaning |
|-----------|---------|
| `input_path` | CAD input path ending in `.step`, `.stp`, `.igs`, `.iges`, or `.brep`; Python callers may pass a sequence of STEP paths for explicit multi-root import, and the CLI accepts repeated `--input` STEP roots. CLI stdin remains STEP-oriented because stdin has no suffix. |
| `output_path` | Output path. Suffix selects USD, glTF, OBJ, STL, or FBX. |
| `profile` | Profile name or `ConversionProfile` that supplies default tessellation, repair, stage, optimize, LOD, budget, and workflow-recipe metadata. |
| `import_options` | `StepReadOptions` for STEP metadata and PMI import. |
| `tessellation` | Overrides the profile tessellation step. |
| `heal_brep` | Optional BREP healing step before tessellation. |
| `stage` | Overrides the profile staging step. |
| `merge_vertices` | Optional standalone vertex merge step after staging. |
| `delete_degenerate_polygons` | Optional standalone degenerate-polygon cleanup step after vertex merging. |
| `merge`, `explode`, `replace` | Optional hierarchy operations run after staging. |
| `scene` | Optional scene optimization step. |
| `bake_materials`, `remove_holes`, `remove_occluded`, `decimate`, `lod_generator` | Optional explicit optimization actions. |
| `optimize` | Overrides the profile simplification step. |
| `lods` | Overrides the profile ratio-based LOD step. |
| `progress` | Callback receiving `(step_name, stats)` after major conversion steps. |
| `validate_output` | Reopen and validate generated output before returning. Defaults to `True`. |
| `debug` | Prefer debuggable USDA conventions. Only valid for `.usd` or `.usda` outputs. |
| `gltf_options`, `usd_options`, `obj_options`, `stl_options`, `fbx_options` | Format-specific write options. |
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
design_variant_selection = []
existing_meshes = true
multi_file = false
material_library_paths = ["vendor-materials.json"]
delete_free_vertices = false
delete_lines = false
construction_curve_policy = "preserve_metadata"
construction_curve_tube_radius = 0.01
target_units = "metre"
target_up_axis = "Y"
target_handedness = "right"

[export]
metadata = "summary"
pmi = "metadata"
```

## Runtime Export Options

glTF and USD exports accept runtime delivery options, and OBJ/STL/FBX are available for mesh and DCC handoff workflows.

```python
asset.write_gltf(
    "motor.glb",
    options=fc.GltfExportOptions(
        preset="web",
        quantize=True,
        meshopt=True,
        draco=False,
        texture_compression=None,
        texture_fallback_format="auto",
        png_compression=6,
        jpeg_quality=85,
        file_size_budget_mb=50,
        size_ladder=True,
        metadata=fc.options.MetadataExportOptions(mode="summary", pmi="metadata"),
    ),
)

asset.write_usd(
    "motor.usdz",
    options=fc.UsdExportOptions(package="usdz", file_size_budget_mb=100),
)

asset.write_obj("motor.obj", options=fc.options.ObjExportOptions(materials=True, write_mtl=True))
asset.write_stl("motor.stl", options=fc.options.StlExportOptions(binary=True, merge=True))
asset.write_fbx("motor.fbx", options=fc.options.FbxExportOptions(materials=True, normals=True, uvs=True))
```

**Presets** — `preset="web"` (also `mobile`, `desktop`, `vr`, `ar`) resolves to
concrete compression defaults and, during `fc.convert()`, runs texture
resize/dedupe cleanup with the preset's texture cap before writing.

**Geometry compression** — these can be combined:

- `quantize=True` writes `KHR_mesh_quantization` accessors and composes the dequantization transform into referencing nodes.
- `meshopt=True` writes `EXT_meshopt_compression` payloads, keeping fallback buffer data for loaders that ignore the extension.
- `draco=True` runs the glTF Transform Draco encoder and writes `KHR_draco_mesh_compression`.

**Textures** — `texture_compression="ktx2"`/`"basisu"` runs the KTX2/Basis encoder
and writes `KHR_texture_basisu`. Otherwise `texture_fallback_format` sets PNG/JPEG
policy: `auto` keeps alpha-bearing sets PNG and color-only sets JPEG; explicit `png`
or `jpeg` forces a format (`png_compression`/`jpeg_quality` tune it). Scalar
transparency uses effective opacity without double-counting duplicated CAD alpha.

Draco export requires the `gltf-transform` CLI on `PATH` or `FASCAT_GLTF_TRANSFORM`.
KTX2/Basis texture export requires Node.js plus `@gltf-transform/core`,
`@gltf-transform/extensions`, `ktx2-encoder`, and `sharp` installed in the working
directory or `FASCAT_NODE_MODULE_ROOT`.

**Report fields** — glTF write steps include `runtime_dependencies` (emitted/required
extensions, `extras.fascat`, a `runtime_compatibility` matrix for glTFast/web/mobile/XR,
and a `runtime_decision_matrix`). With `size_ladder=True`, a `gltf_size_ladder` step
writes temporary baseline/quantized/meshopt/Draco/texture-compressed/requested GLB
variants and records measured bytes plus unavailable-encoder warnings. All write
steps record output size, estimated geometry/texture/metadata bytes, material/image
counts, and budget warnings.

USDZ is built by writing a temporary USD stage and packaging it. glTF, USD, and OBJ
exports write only referenced materials (the in-memory asset is unchanged); glTF also
drops images used only by unused materials and reuses repeated embedded texture URIs.

OBJ export writes vertex positions, normals, `f v//vn` face references, material assignments, and smoothing directives. Staged smooth normals export with smoothing enabled; flat, hard-edge, or generated face normals export with smoothing disabled.

FBX export writes ASCII FBX 7.4 files with `Model`, `Geometry`, `Material`, `GlobalSettings`, and `Connections` sections. Geometry uses FBX polygon-end index bits, preserves hierarchy transforms, and can write normal, tangent, UV, and per-face material layers. PBR material factors are approximated through legacy Phong properties, including effective material opacity.

Export option parameters:

| Option | Parameter | Meaning |
|--------|-----------|---------|
| `GltfExportOptions` | `preset` | Named glTF export preset: `desktop`, `web`, `mobile`, `vr`, or `ar`. Presets request quantization, meshopt, KTX2/Basis texture compression, fallback quality, and `fc.convert()` texture resize/dedupe cleanup. |
| `GltfExportOptions` | `quantize` | Write `KHR_mesh_quantization` accessors and dequantization transforms. |
| `GltfExportOptions` | `meshopt` | Write `EXT_meshopt_compression` payloads with fallback uncompressed data. |
| `GltfExportOptions` | `draco` | Run Draco geometry compression and require `KHR_draco_mesh_compression` when mesh payloads are present. |
| `GltfExportOptions` | `draco_compression_level` | Draco compression level, 0 (fastest) to 10 (smallest). Default 5 matches the encoder default. |
| `GltfExportOptions` | `draco_quantize_position` / `draco_quantize_normal` / `draco_quantize_texcoord` / `draco_quantize_color` | Per-attribute Draco quantization bits (1-30). Defaults 14/10/12/8 match the encoder defaults. |
| `GltfExportOptions` | `ktx2_quality` | KTX2/Basis encoder quality level (0-255), default 128. |
| `GltfExportOptions` | `ktx2_effort` | KTX2/Basis encoder compression effort (0-6), default 2. |
| `GltfExportOptions` | `ktx2_uastc` | Force UASTC (`True`) or ETC1S (`False`); `None` derives from `texture_compression`. |
| `GltfExportOptions` | `texture_compression` | Run KTX2/Basis texture compression for referenced texture images: `ktx2` or `basisu`. |
| `GltfExportOptions` | `texture_fallback_format` | PNG/JPEG fallback policy when KTX2/Basis compression is not requested: `auto`, `png`, or `jpeg`. `auto` keeps alpha-bearing texture sets PNG-safe and color-only sets JPEG-compatible. |
| `GltfExportOptions` | `png_compression` | PNG fallback compression level, 0 through 9. |
| `GltfExportOptions` | `jpeg_quality` | JPEG fallback quality, 0 through 100. Reports warn when explicit JPEG fallback would discard alpha-bearing texture data. |
| `GltfExportOptions` | `file_size_budget_mb` | Add report warnings when the output exceeds this size. |
| `GltfExportOptions` | `size_ladder` | Add a measured `gltf_size_ladder` report comparing temporary baseline, optimized, compressed, and requested GLB variants. |
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
| `FbxExportOptions` | `materials` | Write FBX material nodes, per-face material indices, and model-material connections. |
| `FbxExportOptions` | `normals` | Write FBX normal layers. |
| `FbxExportOptions` | `tangents` | Write FBX tangent layers when mesh tangents exist. |
| `FbxExportOptions` | `uvs` | Write FBX UV layers when mesh UV channels exist. |
| `FbxExportOptions` | `file_size_budget_mb` | Add report warnings when the output exceeds this size. |

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
| `augmented-reality` | stricter phone and tablet AR runtime budget | 60 | 100,000 | 65,535 | 1,024px | 64 MB | 1,500 ms | 150 | 50K-250K triangles, under 500 draw calls |
| `mixed-reality` | stricter headset budget for mixed-reality runtimes | 60 | 75,000 | 65,535 | 1,024px | 64 MB | 1,200 ms | 100 | 50K-200K triangles, under 500 draw calls |

Pass either a profile name or a `ConversionProfile` from `fc.profiles`. Built-in
profiles carry a `WorkflowRecipe` naming the target (`web-glb`, `mobile-glb`,
`vr-glb`, `high-fidelity-desktop`), surfaced as a `workflow_recipe` report step that
marks each stage `honored`, `disabled`, `metadata_only`, or `unsupported`.

When the profile has a budget, conversion reports add:

- **`profile_budget`** — target FPS plus triangle, vertex, per-mesh vertex, texture-resolution, texture-memory, load-time, and draw-call budgets, draw-call breakdown, compression/extension caps, and Unity reference ranges. Fascat's defaults are intentionally stricter than Unity's broad ranges. Load time is a deterministic estimate (file size, geometry/texture bytes, draw-call overhead), not a measured runtime.
- **`texture_export_policy`** (when baked textures are referenced, before write) — source/referenced/unused texture-set and map counts, largest resolutions, estimated bytes, the profile's texture caps, resize candidates and estimated savings, KTX2/Basis request state, and PNG/JPEG fallback policy with transparency-loss warnings.

Custom target-device profiles can overlay a budget on any built-in base profile:

```toml
name = "factory-tablet-ar"

[budget]
target_fps = 60
max_triangles = 42000
max_texture_resolution = 512
max_draw_calls = 120
supported_compression = ["meshopt"]
supported_runtime_extensions = ["KHR_mesh_quantization", "EXT_meshopt_compression"]
unity_reference_profile = "tablet-ar"
unity_reference_triangles = [30000, 60000]
```

```python
profile = fc.profiles.from_file("factory-tablet.toml", base="realtime-mobile")
asset = fc.convert("motor.step", "motor.glb", profile=profile)
```

The CLI equivalent is `fascat convert motor.step motor.glb --profile realtime-mobile
--target-device-profile factory-tablet.toml`. These TOML/JSON files are **budget
overlays only** — tessellation, repair, staging, and LOD defaults still come from the
base profile. Notes:

- An overridden `max_triangles` becomes the optimization target (and the explicit decimation target when `--decimate` is used without `--target-triangles`/`--ratio`); `max_vertices` defaults to 3× the triangle budget unless set.
- `supported_compression` and `supported_runtime_extensions` are optional caps; the profile budget report records a violation when the write emits anything outside them.

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

The report records options, before/after counts, warnings, errors, and timings for
each pipeline step. Approximate operations attach the limitation to the step that
produced it, so you can tell exact geometry changes from fallbacks or metadata-only
intent. Conversion reports add four framing steps:

- **`preflight`** (before expensive work) — checklist warnings for missing patch cleanup, orientation prep, UV/tangent ordering, AO-bake UV1 prerequisites, and LOD0 optimization, plus glTF compression notes.
- **`workflow_recipe`** — for built-in profiles, the target recipe and honored/disabled/metadata-only/unsupported choice counts.
- **`conversion_manifest`** — the resolved profile, import options, operation settings, and export options.
- **`workflow_summary`** — preparation stages (import cleanup, UV prep, baking, LOD, export compression, export) mapped to run/skipped status.

Use `Asset.analyze()` when you need geometry quality risks beyond raw part and triangle totals.

```python
report = asset.analyze(
    fc.options.AnalyzeOptions(
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
material count, draw-call estimate, draw-call breakdown fields, and visual-risk
warnings derived from mesh quality and before/after pipeline report steps. Self-intersection checks ignore
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
| `draw_call_estimate` | Include estimated draw calls, mesh count, referenced material count, submesh/material slots, instances, and merged batch counts. |
| `visual_risk` | Enable risk-oriented warnings from geometry quality and report steps. |
| `sliver_aspect_ratio` | Aspect-ratio threshold used to classify sliver triangles. |
| `degenerate_area_epsilon` | Triangle area threshold used to classify degenerates. Defaults to a bounding-box-derived, scale-invariant value. |
| `tiny_part_diagonal` | Bounding-box diagonal threshold used to classify tiny parts. |
| `max_self_intersection_pairs` | Maximum non-adjacent triangle pairs to check before reporting a lower-bound result. |

Use the visual preview helpers when you need stable review artifacts in CI
or before handing an asset to a runtime viewer:

```python
from fascat import validation

preview = validation.write_preview(asset, "preview.png")
comparison = validation.write_before_after_previews(before_asset, after_asset, "visual-review/")
lod_contact_sheet = validation.write_lod_switch_previews(asset_with_lods, "lod-previews/")
diff = validation.compare_images("baseline.png", "preview.png", validation.VisualDiffOptions(pixel_tolerance=2))
suite = validation.write_runtime_parity_suite("runtime-parity/")
captures = validation.capture_runtime_parity_suite(
    "runtime-parity/",
    targets=("browser", "unity"),
    unity_command="Unity",
    promote_goldens=True,
    require_goldens=False,
)
```

The preview renderer is a local orthographic software renderer: it writes PNGs, uses
material base colors, respects node transforms, and can substitute each part's LOD
mesh into an LOD-switching contact sheet. It is repeatable for a fixed Python,
Pillow, and platform stack, but antialiasing and resampling can vary across
platform builds, so CI baselines should compare with explicit thresholds.
`compare_images()` is a general image-diff
primitive that reports mean absolute error, max channel error, changed-pixel counts
and ratio, and whether configured thresholds passed (the same thresholds gate engine
preview baselines through `fascat validate`).

For cross-renderer comparison, `write_runtime_parity_suite()` writes bundled GLB
fixtures, software baseline PNGs, and a manifest. The generated manifest
recommends regression-gating thresholds of 2 channel values of pixel tolerance,
4.0 mean absolute error, and a 2% changed-pixel ratio unless callers pass
explicit `diff_options`.
`capture_runtime_parity_suite()` runs selected browser/engine targets, writes
`runtime-parity-captures.json`, and can promote renders into `goldens/<target>/`.
When a `goldens/<target>/<fixture>.png` exists, later captures compare against that
golden instead of the software baseline; `require_goldens=True` fails CI if the
golden corpus is incomplete.

## Validation

Direct write calls produce files but do not automatically reopen and validate them. Validate direct writes explicitly when you need the same safety as `fc.convert()`. `fc.validate_output` dispatches by suffix; per-format validators live on the io modules, and the runtime/visual/parity harness machinery lives in `fascat.validation` — one import surface for everything measurement-related.

```python
from fascat import validation
from fascat.io.gltf import validate_gltf
from fascat.io.usd import validate_usd

asset.write_usd("motor.usdc")
usd_stats = validate_usd("motor.usdc")

asset.write_gltf("motor.glb")
gltf_stats = validate_gltf("motor.glb")

stats = fc.validate_output("motor.glb")

runtime = validation.measure_browser_runtime(
    "motor.glb",
    options=validation.RuntimeBrowserOptions(duration_seconds=2.0),
)

unity_project = validation.copy_engine_runtime_harness("unity", "FascatUnityHarness")
unity_runtime = validation.measure_engine_runtime(
    "motor.glb",
    options=validation.RuntimeEngineOptions(
        engine="unity",
        executable="Unity",
        project=unity_project,
        preview_path="motor-unity.png",
    ),
)

preview = validation.write_output_preview("motor.glb", "motor-preview.png")
browser_preview = validation.write_browser_render_preview("motor.glb", "motor-browser.png")
```

The CLI can write a validation-time quality report for exported assets:

```bash
fascat validate motor.glb \
  --filter 'material=Painted*' \
  --geometry-quality \
  --report quality-report.json

fascat validate motor.glb --runtime-browser

fascat validate motor.glb \
  --runtime-engine unity

fascat validate motor.glb \
  --runtime-engine unity \
  --runtime-engine-project FascatUnityHarness \
  --runtime-engine-preview motor-unity.png

fascat validate motor.glb \
  --runtime-browser-preview motor-browser.png \
  --visual-preview motor-preview.png \
  --lod-preview-dir lod-previews/

fascat runtime-fixtures runtime-parity/

fascat runtime-fixtures runtime-parity/ \
  --capture browser \
  --capture unity \
  --unity-command Unity \
  --promote-goldens

fascat runtime-fixtures runtime-parity/ \
  --capture unity \
  --unity-command Unity \
  --require-goldens

fascat runtime-fixtures runtime-parity/ \
  --check-goldens \
  --require-goldens
```

Validation-time geometry reports use the same filter selectors as conversion when an
exported format can be reconstructed for analysis.

**Software preview (always available).** `--visual-preview` writes a stable
PNG from the validated output mesh; `--visual-baseline` diffs it against a baseline
and exits non-zero when thresholds fail. `--lod-preview-dir` writes `lod0.png`, each
LOD level, and a `lod-switching.png` contact sheet (Fascat GLB exports preserve
enough LOD metadata to reconstruct these).

**Browser (glTF/GLB).** `--runtime-browser` launches a local Chromium-compatible
browser, runs a bounded WebGL workload, and reports load time, FPS, frame count,
memory, and workload scale. `--runtime-browser-preview` renders a real WebGL
screenshot — node transforms, base-color factors, quantized attributes, Draco
(via an installed glTF Transform CLI), meshopt (fallback or local `meshoptimizer`), KTX2/Basis
(Python `alktx2` with the `ktx2` extra installed, else installed glTF Transform + KTX-Software),
and base-color textures. Decoded payloads are listed in `decoded_extensions`; if
Draco/meshopt tooling is missing the preview is `unsupported` (no misleading image),
and missing KTX2/Basis tooling falls back to `status="rendered_partial"`. Set
`FASCAT_BROWSER` or `--runtime-browser-command` if the browser isn't on PATH; with no
browser, the report is `status="unavailable"` rather than an estimate.

**Engine (glTF/GLB).** `--runtime-engine unity|unreal` runs a packaged harness (copied
to a temp dir) or a custom one via `--runtime-engine-project`; use
`validation.copy_engine_runtime_harness(engine, path)` for a persistent project. It records
load/parse time, frame count, memory, engine version, and mesh/triangle counts.
`--runtime-engine-preview` requests a PNG and records `render_status`,
`render_backend`, `render_time_ms`, and limitations; `--runtime-engine-baseline`
diffs that render against a baseline with the `--visual-diff-*` thresholds. The Unity
template (glTFast) renders a fixed camera loop with `measured_fps`; the Unreal
commandlet rasterizes supported GLB geometry with `baseColorFactor` (falling back to
a count-based placeholder). Set `FASCAT_UNITY`/`UNITY_EDITOR`/`FASCAT_UNREAL`/`UNREAL_EDITOR`
or `--runtime-engine-command` if the executable isn't on PATH; missing executables or
projects are reported unavailable.

**Parity fixtures.** `fascat runtime-fixtures DIR` writes bundled material, texture,
KTX2/Basis-fallback, lighting, and Unity/Unreal LOD-profile GLBs with software
baselines and a manifest. `--capture browser|unity|unreal` renders captures into
`runtime-parity-captures.json`; `--promote-goldens` copies them into
`goldens/<target>/` (which then become comparison baselines); `--require-goldens`
fails on missing goldens; `--check-goldens` audits coverage without rendering.

> Full Unreal scene rendering, sparse accessors, and checked-in engine-specific
> golden corpora remain open.

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
- generated LOD meshes are included as Fascat extras; Unity-profile exports add node-level `MSFT_lod` references with `MSFT_screencoverage` hints, and Unreal-profile exports add separate `_LOD#` scene nodes
- write reports include runtime compatibility notes for `KHR_mesh_quantization`, `EXT_meshopt_compression`, `KHR_draco_mesh_compression`, `KHR_texture_basisu`, `MSFT_lod`, and `extras.fascat`
