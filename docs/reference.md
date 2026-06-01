---
title: Reference
description: Global flags, environment variables, exit codes, and shell completions
---

## Global flags

Global flags can be placed before or after the subcommand. These are equivalent:

```bash
fascat --json inspect input.step
fascat inspect input.step --json
```

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--help` | `-h` | — | Show help for the current command |
| `--version` | `-V` | — | Show version and exit |
| `--verbose` | `-v` | `false` | Enable verbose output |
| `--quiet` | `-q` | `false` | Suppress non-essential output |
| `--json` | — | `false` | Output results as JSON |
| `--no-color` | — | `false` | Disable ANSI color output |
| `--dry-run` | `-n` | `false` | Preview changes without applying them |
| `--no-input` | — | `false` | Disable interactive prompts |

`-h` / `--help` and `-V` / `--version` are invocation-wide controls. They work before or after subcommands and ignore other arguments.

Dry-run JSON for `convert` includes `operation_diagnostics`, a list of planned operations with `level` set to `exact`, `approximate`, or `metadata_only`. When `--pipeline` is used it also includes `pipeline_advisories`, warning about ordering issues such as decimation before repair, tangents without UV0, AO baking without UV1, and LOD generation before LOD0 optimization. Conversion reports include a `preflight` step before expensive work starts, followed near the end by `workflow_recipe`, `conversion_manifest`, and `workflow_summary` steps. The preflight checklist flags missing patch cleanup, orientation preparation, UV/tangent ordering, AO bake UV1 prerequisites, and LOD0 optimization, plus glTF compression applicability notes. The workflow recipe records the selected profile's Unity-inspired target recipe and counts honored, disabled, metadata-only, and unsupported choices. The conversion manifest records the resolved profile, import, direct or pipeline operation settings, and export options; the workflow summary maps Unity-inspired preparation stages to run/skipped status and exact, approximate, or metadata-only levels.

## Commands

| Command | Description |
|---------|-------------|
| `fascat inspect input.step` | Inspect CAD assembly metadata and planned conversion inputs |
| `fascat convert input.step [output.usdc]` | Convert STEP, IGES, or BREP CAD into OpenUSD, glTF, OBJ, STL, or FBX |
| `fascat validate output.usdc` | Validate generated USD, glTF, OBJ, STL, or FBX output |
| `fascat validate output.glb --runtime-browser` | Measure optional headless browser/WebGL load and FPS for glTF/GLB output |
| `fascat validate output.glb --runtime-browser-preview preview.png` | Write a browser/WebGL-rendered glTF/GLB PNG preview |
| `fascat validate output.glb --runtime-engine unity` | Measure optional Unity or Unreal harness load/parse metrics for glTF/GLB output |
| `fascat validate output.glb --runtime-engine unity --runtime-engine-preview preview.png` | Request a Unity or Unreal harness-rendered preview PNG and report render status |
| `fascat validate output.glb --runtime-engine unity --runtime-engine-preview preview.png --runtime-engine-baseline baseline.png` | Fail validation when an engine preview drifts beyond configured image-diff thresholds |
| `fascat validate output.glb --visual-preview preview.png` | Write a deterministic software-rendered PNG preview for visual review |
| `fascat runtime-fixtures runtime-parity/` | Write bundled browser/Unity/Unreal material, lighting, KTX2/Basis fallback, and LOD-profile parity GLBs, software baselines, and a manifest |
| `fascat runtime-fixtures runtime-parity/ --capture unity --promote-goldens` | Capture runtime parity previews and optionally promote rendered outputs into target golden directories |
| `fascat runtime-fixtures runtime-parity/ --capture unity --require-goldens` | Compare captures against existing `goldens/<target>/<fixture>.png` files and fail when required target goldens are missing |
| `fascat runtime-fixtures runtime-parity/ --check-goldens --require-goldens` | Audit checked-in target golden PNG coverage without rendering and fail on missing, invalid, or wrong-size goldens |
| `fascat help [command]` | Show top-level or command-specific help |
| `fascat version` | Print version and exit |

## Format support matrix

This matrix is scoped to Fascat's current importer/exporter surface. It is meant
to make Unity Asset Transformer-style format parity explicit, not to imply that
all listed formats are implemented.

| Format family | Input support | Output support | Current decision |
|---------------|---------------|----------------|------------------|
| STEP `.step`, `.stp` | Supported | Not emitted as CAD | Primary neutral CAD input path |
| IGES `.igs`, `.iges` | Supported | Not emitted as CAD | Legacy CAD geometry import through OCP/XDE |
| OpenCASCADE BREP `.brep` | Supported | Not emitted as CAD | Native-kernel shape import as a single source-shape part |
| OpenUSD `.usd`, `.usda`, `.usdc`, `.usdz` | Not imported | Supported | Runtime and scene-composition delivery |
| glTF `.gltf`, `.glb` | Not imported | Supported | Preferred web/mobile runtime delivery |
| OBJ `.obj` | Not imported | Supported | Mesh-only interchange output |
| STL `.stl` | Not imported | Supported | Mesh-only manufacturing or inspection output |
| FBX `.fbx` | Not imported | Supported | ASCII FBX DCC and engine handoff output |
| Parasolid `.x_t`, `.x_b` | Not supported | Not supported | Native-kernel CAD import candidate, not in scope yet |
| JT `.jt` | Not supported | Not supported | Visualization/CAD hybrid import candidate, not in scope yet |
| CATIA, NX, SolidWorks, Inventor | Not supported | Not supported | Native CAD coverage is deferred |
| IFC, 3MF, QIF | Not supported | Not supported | Adjacent workflow formats, deferred unless a user need changes priority |

STEP import report steps include `import_decisions`, which separates requested,
effective, `honored`, `approximated`, `unsupported`, `disabled`, `not_present`,
and `backend_default` states for Unity-style import toggles such as PMI,
variants, existing meshes, construction cleanup, and coordinate normalization.
PMI import uses a textual AP242 scan for common dimension, location, geometric
tolerance and named geometric-tolerance subtypes, plus/minus tolerance, datum,
datum-reference, feature-control-frame, and annotation text records and writes
typed `PmiAnnotation` metadata plus a `pmi_semantic_graph` of STEP entity nodes
and reference edges, including common target support records and inbound
callout/annotation associativity records.
`metadata_and_visuals` export writes deterministic glTF/USD PMI marker meshes
with simple vector text glyphs linked to those records; full AP242 visual
presentation reconstruction remains planned. Report steps also include detected
STEP design variant configuration/effectivity metadata when
`design_variants` is enabled; `design_variant_selection` can prune imported
geometry by selected variant labels, STEP record ids, exact effectivity values,
or values inside supported serial/date/time-interval effectivity ranges when
those records resolve to labels that match imported node, part, or source-name
metadata, including referenced `TIME_INTERVAL_WITH_BOUNDS` date bounds using
`CALENDAR_DATE`, `ORDINAL_DATE`, or `WEEK_OF_YEAR_AND_DAY_DATE` records. Simple
AP242 condition/expression records such as `AND_EXPRESSION`,
`OR_EXPRESSION`, `XOR_EXPRESSION`, `NOT_EXPRESSION`, `EQUALS_EXPRESSION`,
`COMPARISON_EQUAL`, `COMPARISON_NOT_EQUAL`, `COMPARISON_GREATER`,
`COMPARISON_GREATER_EQUAL`, `COMPARISON_LESS`, `COMPARISON_LESS_EQUAL`,
`INTERVAL_EXPRESSION`, `LIKE_EXPRESSION`, simple numeric arithmetic expressions
such as `PLUS_EXPRESSION`, `MINUS_EXPRESSION`, `MULT_EXPRESSION`,
`DIV_EXPRESSION`, `SLASH_EXPRESSION`, `MOD_EXPRESSION`, and
`POWER_EXPRESSION`, `RATIONAL_REPRESENTATION_ITEM`,
`EXPRESSION_EXTENSION_NUMERIC`, numeric functions such as
`ABS_FUNCTION`,
`MINUS_FUNCTION`, `SQUARE_ROOT_FUNCTION`, `MAXIMUM_FUNCTION`,
`MINIMUM_FUNCTION`, `SIN_FUNCTION`, `COS_FUNCTION`, `TAN_FUNCTION`,
`ASIN_FUNCTION`, `ACOS_FUNCTION`, unary or binary `ATAN_FUNCTION`, `EXP_FUNCTION`,
`LOG_FUNCTION`, `LOG2_FUNCTION`, and `LOG10_FUNCTION`, simple string
expressions such as `CONCAT_EXPRESSION`, `SUBSTRING_EXPRESSION`,
`INDEX_EXPRESSION`, `FORMAT_FUNCTION`, and `EXPRESSION_EXTENSION_STRING`,
and string-derived numeric
functions such as
`LENGTH_FUNCTION`, `VALUE_FUNCTION`, `INT_VALUE_FUNCTION`, `ODD_FUNCTION`,
`BOOLEAN_LITERAL`, `BOOLEAN_REPRESENTATION_ITEM`, `LOGICAL_LITERAL`,
`LOGICAL_REPRESENTATION_ITEM`, `BOOLEAN_VARIABLE`, and
`MATHS_BOOLEAN_VARIABLE` are reported with condition operators, and boolean or
logical literals include parsed true/false values. Numeric literals include
parsed numeric values, including expression-extension numeric values; string
literals include parsed text values, including expression-extension string
values. Supported
conditions gate their operand labels before those labels can drive geometry
pruning; expression-only operand labels are not promoted as geometry targets;
equality/not-equality records compare resolved numeric/string operand values
and fall back to selected boolean operand states for non-value operands. Named
maths numeric variables evaluate from selected `label=value` assignments for
numeric comparisons, intervals, equality/not-equality, numeric function
expressions including rational representation items, expression-extension
numeric values, and elementary trig/log/exp functions, with two-operand
`ATAN_FUNCTION` evaluated as `atan2`, and `ODD_FUNCTION` integer tests. String
variables evaluate from
selected `label=value` assignments for equality/not-equality,
`LIKE_EXPRESSION`, `CONCAT_EXPRESSION`, `SUBSTRING_EXPRESSION`,
`INDEX_EXPRESSION`, `EXPRESSION_EXTENSION_STRING`, `LENGTH_FUNCTION`,
`VALUE_FUNCTION`, and `INT_VALUE_FUNCTION` matching, while numeric variables
can feed conservative
`FORMAT_FUNCTION` string matching,
boolean variables act as named operands
selected by label, STEP record id, or explicit `label=true` / `label=false`
assignments, and conditional/effectivity-assignment wrappers, including
`CONDITIONAL_CONCEPT_FEATURE`,
`CONDITIONAL_EFFECTIVITY`, `CONFIGURED_EFFECTIVITY_ASSIGNMENT`, and
`APPLIED_EFFECTIVITY_ASSIGNMENT`, also gate their configured target labels.
`APPLIED_INEFFECTIVITY_ASSIGNMENT` records are reported and suppress assigned
target labels when the selected effectivity matches. Applied assignment targets
can resolve through referenced product-definition labels. Effectivity
context-assignment wrappers add their context target labels only when the
referenced effectivity assignment is satisfied. Product-definition and
configuration effectivity records add their usage relationship labels only when
the selected effectivity matches. Effectivity relationship records can bridge
selected effectivity labels to related gated usage records.
Report steps also include `loaded_representations`, a per-part
BREP/construction-shape summary plus deleted construction-only nodes and source
topology counts. Free construction edges split from mixed face+curve STEP shapes
appear as separate construction-line parts or construction cleanup counts,
depending on the construction-curve policy.
IGES import uses the same OCP/XDE shape tree traversal for hierarchy, transforms,
colors, and materials. BREP import creates one root occurrence and one part with
the native source shape retained for tessellation and healing.

## Convert flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `realtime-desktop` | Conversion profile: `inspect-only`, `realtime-desktop`, `realtime-web`, `realtime-mobile`, `virtual-reality`, `augmented-reality`, or `mixed-reality` |
| `--target-device-profile` | unset | TOML or JSON target-device budget overlay for the selected profile |
| `--pipeline` | unset | TOML pipeline file with named filters and ordered conversion steps |
| `--input` | unset | Additional STEP root input for explicit multi-root conversion; may be passed more than once |
| `--sag` | profile value | CAD tessellation sag tolerance |
| `--sag-ratio` | unset | Relative CAD tessellation sag ratio; enables explicit relative deflection when set |
| `--angle` | profile value | CAD tessellation angle tolerance in degrees |
| `--target-triangles` | profile value | Target triangle count for optimized LOD0 |
| `--ratio` | unset | Simplification ratio when no triangle target is set |
| `--min-edge-length` | unset | Collapse tessellated edges shorter than this length |
| `--max-edge-length` | profile value | Split tessellated triangles longer than this length |
| `--max-polygon-length` | unset | Report tessellated polygon edges longer than this length without subdividing geometry |
| `--preserve-boundaries / --no-preserve-boundaries` | `true` | Preserve sharp/boundary edges during tessellation cleanup |
| `--curvature-adaptive` | `false` | Use tighter interior meshing on curved CAD faces |
| `--detail-adaptive` | `false` | Auto-tighten tessellation for shiny or high-detail material/metadata parts |
| `--avoid-skinny-triangles` | `false` | Refine long skinny triangles after tessellation |
| `--quality-report` | unset | Write per-part tessellation quality metrics and quality advisories as JSON |
| `--free-edge-report` | `false` | Record free/boundary edge diagnostics and warn when tessellated parts have open boundaries |
| `--reuse-existing-meshes / --retessellate-existing-meshes` | `true` | Reuse imported mesh data or force retessellation from source BREP where available |
| `--heal-brep` | `false` | Run BREP healing before tessellation |
| `--heal-tolerance` | `0.05` | BREP healing tolerance |
| `--group-open-shells / --no-group-open-shells` | `true` | Group disconnected open BREP shells before healing |
| `--cleanup-overlapping-faces / --keep-overlapping-faces` | `true` | Remove redundant coplanar BREP faces that overlap enough to z-fight |
| `--overlap-area-ratio` | `0.995` | Minimum smaller-face area ratio for BREP overlap cleanup |
| `--remove-sliver-faces` | `false` | Request tiny sliver-face removal during BREP healing; current backend support is limited and reports warnings when unavailable |
| `--max-sliver-area` | `1e-4` | Area threshold for sliver-face reporting |
| `--fail-on-open-shells` | `false` | Fail if healed BREP still contains open shells |
| `--lods` | profile value | Comma-separated LOD ratios, for example `0.5,0.25,0.1` |
| `--lod-mode` | `variants` | LOD output mode: `variants`, `extras`, or `separate` |
| `--lod-engine-profile` | `generic` | Engine-specific LOD export profile: `generic`, `unity`, or `unreal` |
| `--lod-screen-coverage` | unset | Screen coverage values for generated LODs |
| `--lod-per-part-budget` | `false` | Apply LOD budgets independently per part |
| `--lod-drop-tiny-parts` | `false` | Omit tiny parts from lower LOD meshes |
| `--lod-tiny-part-screen-size` | `2.0` | Screen-size threshold for tiny-part LOD omission |
| `--validate-lods` | `false` | Validate generated LOD monotonicity |
| `--jobs` | `1` | Worker count for independent per-part repair, staging, optimization, decimation, and LOD mesh work |
| `--normals` | `smooth` | Normal generation mode: `none`, `smooth`, `hard-edges`, or `flat` |
| `--normal-weighting` | `angle` | Normal averaging weights for smooth or hard-edge normals: `angle` or `area` |
| `--preserve-face-boundaries` | `false` | Treat CAD face-group boundaries as hard normal edges |
| `--override-normals / --preserve-normals` | `true` | Regenerate existing normals, or preserve existing normals and only generate missing normals |
| `--tangents` | `false` | Ensure glTF-compatible vertex tangents exist; existing tangents are preserved unless invalidated or overridden |
| `--tangent-uv-channel` | `0` | UV channel used when tangent generation or regeneration is needed |
| `--override-tangents / --preserve-tangents` | `false` | Regenerate existing tangents instead of preserving them when `--tangents` is used |
| `--validate-normals` | `false` | Validate staged normals and tangents |
| `--uv0` | `box` | UV0 generation mode: `none`, `box`, `unwrap`, or `lightmap` |
| `--uv1` | `none` | UV1 generation mode: `none`, `box`, `unwrap`, `lightmap`, or `copy-uv0` |
| `--uv-aabb-scope` | `local` | AABB projection scope for `box` UVs: `local` per part or `shared` across selected parts |
| `--uv3d-size` | unset | World-space size per UV tile for `box`/AABB projection; unset normalizes to the AABB |
| `--uv-override-existing / --uv-preserve-existing` | `true` | Override or preserve existing destination-channel UVs when `box` projection is requested |
| `--normalize-uvs` | unset | Comma-separated UV channels to rescale into 0..1 after UV generation or copy |
| `--materials` | `cad` | Material staging mode: `cad`, `display`, or `none` |
| `--material-mode` | `cad` | Material normalization mode: `cad` or `pbr` |
| `--merge-equivalent-materials` | `false` | Merge CAD materials with matching PBR values |
| `--merge-vertices` | `false` | Merge exact or tolerance-close vertices after staging |
| `--merge-vertex-tolerance` | `0.0` | Position tolerance used by `--merge-vertices` |
| `--preserve-merge-vertex-attributes / --drop-merge-vertex-attributes` | `true` | Keep normals, tangents, and UV seams as merge constraints |
| `--preserve-merge-vertex-material-boundaries / --ignore-merge-vertex-material-boundaries` | `true` | Keep material-boundary signatures as merge constraints |
| `--delete-merge-vertex-degenerate / --keep-merge-vertex-degenerate` | `true` | Delete degenerate polygons created by vertex merging |
| `--merge-vertex-area-epsilon` | `1e-12` | Area threshold for degenerate polygons after vertex merging |
| `--delete-degenerate-polygons` | `false` | Run standalone degenerate polygon cleanup after vertex merging |
| `--degenerate-area-epsilon` | `1e-12` | Area threshold for standalone degenerate polygon cleanup |
| `--delete-duplicate-polygons / --keep-duplicate-polygons` | `true` | Remove exact duplicate polygons during standalone degenerate polygon cleanup |
| `--texel-density` | unset | UV texel density metadata for unwrap and atlas workflows |
| `--uv-padding` | `2` | UV island padding metadata in pixels |
| `--max-stretch` | unset | Maximum UV stretch metadata for unwrap workflows |
| `--unwrap-method` | `default` | Unwrap solver intent: `default`, `conformal`, or `isometric` |
| `--unwrap-iterations` | unset | Requested unwrap solver iteration budget metadata |
| `--unwrap-tolerance` | unset | Requested unwrap solver tolerance metadata |
| `--uv-sharp-to-seam / --uv-no-sharp-to-seam` | `false` | Request sharp edges as UV seams for unwrap and lightmap channels |
| `--uv-forbid-overlapping / --uv-allow-overlapping` | `false` | Request non-overlapping UV islands and report overlaps as policy violations |
| `--atlas` | `false` | Tag materials and UVs for a generated atlas |
| `--atlas-size` | `4096` | Maximum atlas texture size |
| `--metadata` | `full` | Metadata import/export mode: `none`, `summary`, or `full` |
| `--pmi` | `metadata` | PMI import/export mode: `none`, `metadata`, or `metadata-and-visuals` |
| `--design-variants / --no-design-variants` | `false` | Scan STEP design variant records into metadata and import reports |
| `--design-variant` | unset | Select a STEP design variant label, effectivity value, numeric/string assignment, record id, or referenced label and filter imported geometry; may be passed more than once |
| `--import-existing-meshes / --no-import-existing-meshes` | `true` | Prefer existing STEP tessellation payloads when the importer exposes them |
| `--multi-file-import / --single-file-import` | `false` | Resolve quoted external STEP references from a master STEP file |
| `--material-library` | unset | Vendor material-library JSON/MTL/ZIP file or folder to apply during import; may be passed more than once |
| `--delete-free-vertices / --keep-free-vertices` | `false` | Drop construction-only point shapes during STEP import |
| `--delete-lines / --keep-lines` | `false` | Drop construction-only line shapes during STEP import |
| `--construction-curve-policy` | `preserve-metadata` | Construction-only line policy: `preserve-metadata`, `delete`, or `tessellate-tubes` |
| `--construction-curve-tube-radius` | `0.01` | Tube radius in source units when construction curves are tessellated as tubes |
| `--source-units` | STEP header | Override source units for normalization |
| `--source-meters-per-unit` | STEP header | Override source meters per unit for normalization |
| `--source-up-axis` | `Z` | Declared source up axis: `Y` or `Z` |
| `--source-handedness` | `right` | Declared source handedness: `right` or `left` |
| `--target-units` | source units | Normalize imported asset units, for example `metre` |
| `--target-meters-per-unit` | source factor | Normalize imported asset meters per unit with a custom factor |
| `--target-up-axis` | source axis | Normalize imported asset up axis to `Y` or `Z` |
| `--target-handedness` | source handedness | Normalize imported asset handedness to `right` or `left` |
| `--filter` | unset | Scope optimization and LOD work with a selector such as `path=*/Fasteners/*` |
| `--exclude-filter` | unset | Exclude selector matches from `--filter` results |
| `--merge` | `false` | Merge selected geometry before optimization |
| `--merge-mode` | `all` | Merge mode: `all`, `by-material`, `by-node-name`, `by-part-name`, `hierarchy-level`, `parent-children`, `final-level`, or `regions` |
| `--keep-parent / --no-keep-parent` | `true` | Attach merged nodes to a shared selected parent when possible |
| `--merge-metadata` | `preserve` | Metadata policy: `preserve`, `combine`, `summarize`, or `drop` |
| `--max-vertices-per-mesh` | `65535` | Split merged output above this vertex count |
| `--region-size` | unset | Spatial region size for `--merge-mode regions` |
| `--merge-strategy` | `all` | Region merge strategy: `all` or `by-material` |
| `--hierarchy-level` | `1` | Hierarchy level used by `--merge-mode hierarchy-level` |
| `--explode` | unset | Explode selected geometry by `by-material` or `connected-components` |
| `--replace` | unset | Replace selected geometry with `bounding-box` or `external-asset` proxies |
| `--external-asset` | unset | External asset path recorded by `--replace external-asset` |
| `--batch-by-material` | `false` | Batch compatible scene geometry by material |
| `--merge-compatible-meshes` | `false` | Merge compatible scene meshes to reduce draw calls |
| `--split-large-meshes` | `false` | Split scene-optimized meshes above the vertex limit |
| `--index-buffer` | `auto` | Index buffer mode: `auto`, `uint16`, or `uint32` |
| `--flatten` | `safe` | Hierarchy flattening mode: `none`, `safe`, or `all` |
| `--instance-policy` | `auto` | Instance policy: `auto` and `preserve` reconstruct exact matching mesh instances; `expand` duplicates per occurrence |
| `--instance-similarity-tolerance` | `0.0` | Position tolerance for reconstructing near-identical mesh instances with matching topology, attributes, materials, and metadata |
| `--bake-materials` | `false` | Create a shared baked material with raster atlas textures |
| `--maps-resolution` | `2048` | Requested bake texture resolution in pixels, recorded for downstream atlas generation |
| `--force-uv-generation` | `false` | Generate UVs before material bake metadata and textures are recorded |
| `--bake` | `base-color` | Maps to bake into raster atlas textures, such as `base-color,opacity` |
| `--decimate` | `false` | Run explicit decimation before profile optimization |
| `--decimate-criterion` | `target` | Decimation criterion: `target` or `quality` |
| `--surface-tolerance` | unset | Surface deviation tolerance metadata for decimation |
| `--line-tolerance` | unset | Hard-edge deviation tolerance metadata for decimation |
| `--normal-tolerance` | `15` | Normal angle tolerance for decimation preservation |
| `--uv-tolerance` | unset | UV deviation tolerance metadata for decimation |
| `--decimate-iterative-threshold` | `1000000` | Source triangle threshold above which explicit decimation runs intermediate passes |
| `--protect-topology / --no-protect-topology` | `true` | Preserve topology-sensitive faces during decimation |
| `--preserve-painted-areas` | `false` | Preserve painted/protected/weighted face groups or metadata-marked face indices during decimation |
| `--preserve-ambient-occlusion` | `false` | Preserve low-AO faces from the sampled AO estimator during decimation |
| `--budget-scope` | `selection` | Decimation budget scope: `part` or `selection` |
| `--uv-importance` | `preserve-islands` | Decimation UV handling: preserve islands, preserve seams, or ignore UVs |
| `--decimate-cleanup-attributes` | unset | Comma-separated pre-decimation cleanup attributes: `unused-uvs,tangents` |
| `--remove-holes` | `false` | Remove small open hole loops with mesh boundary classification |
| `--hole-types` | `through,blind,surface` | Boundary hole types to remove |
| `--max-hole-diameter` | `3.0` | Maximum planar-span hole diameter to remove |
| `--prefer-brep / --no-prefer-brep` | `true` | Prefer BREP feature removal when available |
| `--remove-occluded` | `false` | Remove geometry hidden from sampled exterior visibility rays |
| `--occlusion-strategy` | `advanced` | Occlusion strategy: `conservative`, `exterior`, or `advanced` |
| `--occlusion-level` | `triangles` | Occlusion removal level: `parts`, `submeshes`, or `triangles` |
| `--occlusion-precision` | `2048` | Occlusion precision preset or sample resolution |
| `--hemi-evaluation` | `false` | Restrict occlusion visibility rays to upper-hemisphere and side views |
| `--neighbors-preservation` | `1` | Visible-neighbor preservation rings for triangle occlusion removal |
| `--consider-transparency-opaque` | `false` | Treat transparent materials as occluders |
| `--preserve-cavities / --no-preserve-cavities` | `true` | Preserve large interior cavities |
| `--minimum-cavity-volume-m3` | `0.5` | Minimum cavity volume to preserve |
| `--run-lod-generators` | `false` | Run preset-driven LOD generation after optimization actions |
| `--lod-preset` | `desktop` | LOD preset: `desktop`, `web`, `mobile`, or `vr` |
| `--preserve-instances / --no-preserve-instances` | `true` | Preserve repeated parts as shared instances, or duplicate per occurrence |
| `--preserve-hard-edges` | `false` | Protect faces adjacent to hard edges during simplification |
| `--hard-edge-angle` | `30` | Angle threshold for hard-edge preservation |
| `--preserve-holes` | `false` | Protect open boundary faces during simplification |
| `--preserve-material-boundaries` | `false` | Protect faces along material boundaries |
| `--preserve-uv-seams` | `false` | Protect faces touching duplicated-position UV seams |
| `--preserve-small-parts` | `false` | Skip simplification for small parts |
| `--small-part-triangle-threshold` | `64` | Triangle threshold for `--preserve-small-parts` |
| `--preserve-silhouette` | `false` | Protect faces on bounding-box silhouette extremes |
| `--export-preset` | unset | glTF export preset: `desktop`, `web`, `mobile`, `vr`, or `ar`; presets request compression plus texture resize/dedupe cleanup during conversion |
| `--quantize` | `false` | Write glTF `KHR_mesh_quantization` accessors and node dequantization transforms |
| `--meshopt` | `false` | Write glTF `EXT_meshopt_compression` bufferView payloads with fallback data |
| `--draco` | `false` | Compress glTF geometry with `KHR_draco_mesh_compression` |
| `--texture-compression` | unset | Compress glTF textures with KTX2/Basis: `ktx2` or `basisu` |
| `--texture-fallback-format` | `auto` | PNG/JPEG fallback policy when KTX2/Basis compression is not requested: `auto`, `png`, or `jpeg` |
| `--png-compression` | `6` | PNG fallback compression level, 0 through 9 |
| `--jpeg-quality` | `85` | JPEG fallback quality, 0 through 100 |
| `--package` | `default` | USD package mode: `default` or packaged `.usdz` |
| `--file-size-budget-mb` | unset | Warn in reports when output exceeds this size |
| `--size-ladder` | `false` | Measure baseline, optimized, compressed, and requested temporary GLB sizes in a `gltf_size_ladder` report |
| `--obj-materials / --no-obj-materials` | `true` | Write OBJ material assignments |
| `--write-mtl / --no-write-mtl` | `true` | Write an OBJ MTL sidecar |
| `--preserve-groups / --no-preserve-groups` | `true` | Preserve OBJ groups per occurrence |
| `--stl-binary / --stl-ascii` | `true` | Write binary STL instead of ASCII STL |
| `--stl-merge / --no-stl-merge` | `true` | Merge STL output into one triangle stream |
| `--fbx-materials / --no-fbx-materials` | `true` | Write FBX material nodes and connections |
| `--fbx-normals / --no-fbx-normals` | `true` | Write FBX normal layers |
| `--fbx-tangents / --no-fbx-tangents` | `true` | Write FBX tangent layers when available |
| `--fbx-uvs / --no-fbx-uvs` | `true` | Write FBX UV layers when available |
| `--debug` | `false` | Require text `.usd` or `.usda` output for debugging |
| `--report` | unset | Write a JSON conversion report sidecar |
| `--force` | `false` | Overwrite an existing output file |

Units and behavior notes:

- Linear tolerances and sizes such as `--sag`, `--min-edge-length`, `--max-edge-length`, `--max-polygon-length`, `--heal-tolerance`, `--max-sliver-area`, `--region-size`, and `--max-hole-diameter` use the source asset's working units unless the option explicitly says otherwise.
- Import space normalization uses a root transform: source BREP coordinates stay in source units, while the asset declares the target units, up-axis, and handedness and records the transform in the import report.
- `tessellate`, `heal_brep`, and `repair` report steps include `tolerance_policy`, which records effective source/local units, declared target units, meter conversions, active tessellation deflection kind, converted absolute tessellation lengths where applicable, and whether related repair cleanup backends such as T-junction sewing, boundary-gap stitching, or non-manifold cracking are implemented.
- `repair` detects non-orientable shared-edge cycles before winding normalization and warns when Mobius-like topology cannot be fixed by ordinary face flipping.
- `repair` records face and normal orientation policy. `face_orientation="exterior"` is the implemented closed-component winding path; source-trusted, preserve, viewer-standpoint, open-shell, and unstitched-group policies are explicit in metadata and report status instead of being hidden behind `fix_winding`.
- `repair` records before/after flipped closed-component counts. `fix_winding` flips coherent closed shells with inward signed volume when possible and warns if flipped components remain.
- `repair` records before/after T-junction counts and warns when T-junctions remain, because the current mesh repair path reports but does not sew them.
- `repair` records nearby unstitched boundary gap counts and warns when boundary gaps remain, because the current mesh repair path reports but does not stitch them.
- `--delete-degenerate-polygons` removes repeated-vertex, collapsed-edge, near-flat, and exact duplicate polygons by default, with separate report counts for each reason; use `--keep-duplicate-polygons` when duplicate triangles should only be reported.
- Angles such as `--angle`, `--normal-tolerance`, and `--hard-edge-angle` are degrees.
- Ratios such as `--ratio`, `--lods`, and decimation target ratios are fractions between `0` and `1`; LOD ratios must be sorted from highest to lowest detail.
- Explicit decimation requests that keep less than 20% of source triangles emit an LOD0 distortion warning. Use those aggressive ratios primarily for distant LODs unless visual validation says otherwise.
- Explicit decimation report metadata includes `target_strategy` data that distinguishes explicit target-count, target-ratio, and quality/error approximation workflows, plus estimated RAM using the Unity rule of thumb of 5 GB per million source triangles, the iterative threshold, whether iterative decimation is recommended, actual simplification and iterative pass counts, whether the target was allocated globally across the selection or per part, and per-part target allocation summaries.
- When `--decimate` is enabled without `--target-triangles` or `--ratio`, the selected profile or `--target-device-profile` triangle budget seeds the explicit decimation target when available.
- Screen coverage values are fractions between `0` and `1`; file-size budgets are megabytes; atlas and bake sizes are pixels.
- LOD report metadata separates source, added-LOD, and full-chain vertex/triangle counts plus estimated mesh payload bytes, making the memory and export-size tradeoff of extra LOD levels visible before runtime testing. Ratio LODs simplify progressively from the previous generated level while preserving each requested ratio against the source triangle count.
- LOD chain advisories warn when a chain has more than four generated levels, when LOD1 or LOD2 are too aggressive for close or mid views, and when the farthest LOD is still geometry-only and should eventually use one-mesh/one-material baking.
- LOD per-level policy metadata reports the simplification source, whether instances were reused, whether materials were merged, whether textures were baked, whether culling granularity changed, which advisory applies to each level, and the resolved export representation. `--lod-engine-profile unity` emits standard `MSFT_lod` variant nodes for glTFast-oriented import, while `--lod-engine-profile unreal` emits separate `_LOD#` scene nodes so import tooling that ignores `MSFT_lod` can still see the generated meshes.
- `fascat validate output.glb --visual-preview preview.png --lod-preview-dir previews/` writes local software-rendered PNG review artifacts. `--visual-preview` renders the validated output mesh with material base colors and node transforms; `--lod-preview-dir` writes `lod0.png`, each available LOD level, and `lod-switching.png` with monotonic triangle-count metadata in JSON mode.
- `fascat validate output.glb --runtime-browser-preview preview.png` writes a browser/WebGL-rendered PNG for supported glTF/GLB mesh primitives, including node transforms, material base-color factors, quantized vertex attributes, Draco geometry decoded through glTF Transform, meshopt exports that keep fallback buffer data, meshopt bufferViews without fallback buffer data decoded through local `meshoptimizer`, KTX2/Basis textures decoded through the default Python `alktx2` backend on supported Python 3.11+ Linux/Windows x86_64 installs or glTF Transform plus KTX-Software when available, optional KHR_texture_basisu textures with PNG fallbacks, and supported base-color image URI/data URI texture sampling. The browser preview preflights compressed runtime extensions: Draco or meshopt decode tooling failures return `status="unsupported"`, while unavailable KTX2/Basis texture decode tooling can still render fallback texture sources as `status="rendered_partial"` with `unsupported_extensions` and `preview_limitations`; assets without fallbacks render geometry only. Sparse accessors and full engine material/lighting parity remain outside this preview path.
- `fascat validate output.glb --runtime-browser` runs a Chromium-compatible headless browser when available, loads the GLB/GLTF bytes, executes a bounded WebGL triangle workload derived from asset triangle count, and reports measured browser load time, FPS, frame count, memory bytes, and workload scale. If no browser is available, the runtime report is marked unavailable instead of estimated.
- `fascat validate output.glb --runtime-engine unity` or `--runtime-engine unreal` launches a local engine harness and records measured engine-process load/parse time, frame count, memory bytes, engine version, mesh count, and triangle count. When `--runtime-engine-project` is omitted, Fascat runs a packaged temporary Unity or Unreal harness template. Pass `--runtime-engine-project Harness/` or `--runtime-engine-project Harness.uproject` to use a customized project, and use `--runtime-engine-command` or `FASCAT_UNITY` / `UNITY_EDITOR` / `FASCAT_UNREAL` / `UNREAL_EDITOR` when the engine executable is not on `PATH`; missing engines or explicit projects are reported as unavailable. `--runtime-engine-preview preview.png` reports `render_status`, `render_time_ms`, `rendered_frames`, `render_backend`, `render_limitations`, `render_error`, and `preview_path`; `--runtime-engine-baseline baseline.png` compares a rendered engine preview against a baseline with the `--visual-diff-*` thresholds and fails validation on drift. The packaged Unity template uses glTFast to instantiate the scene, render a fixed multi-frame camera loop with `measured_fps`, and write a camera-rendered PNG when graphics are available. The packaged Unreal commandlet now rasterizes supported GLB triangle geometry with material `baseColorFactor` into a deterministic PNG and fixed-frame software benchmark; unsupported assets fall back to `render_status="rendered_partial"` with explicit limitations. Full Unreal scene-rendered screenshots/FPS and checked-in engine-specific golden corpora remain open. `fascat runtime-fixtures DIR` writes bundled PBR material, texture-map, optional KTX2/Basis fallback, normal/lighting, Unity `MSFT_lod`, and Unreal separate-node LOD-profile GLB fixtures with software baseline PNGs, target-golden paths, and a manifest for browser, Unity, and Unreal preview comparisons; add `--capture browser`, `--capture unity`, or `--capture unreal` to write preview captures and `runtime-parity-captures.json`, `--promote-goldens` to copy rendered captures into `goldens/<target>/`, and `--require-goldens` to fail when requested target golden files are absent. `--check-goldens` writes `runtime-parity-golden-coverage.json` without rendering; combined with `--require-goldens`, it fails on missing, invalid, or wrong-size target PNGs.
- `--decimate-criterion quality` passes tolerance-derived target error bounds to the simplification backend and records bound/result metadata plus achieved triangle reduction.
- Explicit decimation reports protected hard-edge, hole-boundary, material-boundary, UV-seam, silhouette, and total feature-face counts so topology/material/UV preservation pressure is visible.
- `--uv-importance ignore` strips UV/tangent attributes before simplification; `preserve-seams` uses UVs for seam preservation and then strips them; `preserve-islands` keeps UVs through the output.
- `--preserve-painted-areas` and `--preserve-ambient-occlusion` add painted/protected face groups, metadata-marked face indices, and low-AO faces to the simplification constraints. Reports include painted-area, AO, and combined importance-face counts.
- `--decimate-cleanup-attributes unused-uvs,tangents` removes empty, constant, or zero-area UV channels and tangents before simplification. Reports record removed channels, removed tangent parts, preserved UV channels, and warn when preserved UV seams or islands can reduce simplification efficiency.
- Bake-domain UV channels generated with `--uv1 unwrap` or `--uv1 lightmap` are packed by xatlas with configured padding/resolution and report pack dimensions, utilization, and padding status.
- Box UV generation is reported as AABB projection. `--uv-aabb-scope`, `--uv3d-size`, and `--uv-preserve-existing` record local/shared bounds, real-world UV scale, destination channel, override policy, and unit metadata.
- `--uv-sharp-to-seam` and `--uv-forbid-overlapping` are recorded as Unity-style UV policy intent for unwrap/lightmap channels. The current xatlas backend does not expose those controls directly, so reports include per-channel requested/enforced fields and post-generation overlap validation.
- `--remove-holes` uses mesh boundary classification and filling when BREP hole removal is unavailable. `--hole-types` filters inferred through, blind, and surface boundary loops; closed BREP feature holes still require a BREP feature backend.
- `--remove-occluded` uses deterministic sampled visibility. Strategy changes the direction set, `--hemi-evaluation` restricts rays to upper-hemisphere and side views, and `--occlusion-level` controls whether fully hidden parts, material groups, or triangles are removed. Output metadata records sample coverage, direction coverage, and an occlusion confidence score.
- `--draco` runs the Draco encoder for glTF/GLB mesh payloads.
- `--texture-fallback-format auto` records PNG fallback for alpha-bearing texture sets and JPEG fallback for color-only texture sets when KTX2/Basis output is not requested. Explicit `jpeg` fallback reports a warning when referenced texture metadata indicates transparency would be discarded.

## Inspect flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `inspect-only` | Inspection profile to show in output |
| `--metadata` | `summary` | Metadata output mode: `none`, `summary`, or `full` |
| `--pmi` | `summary` | PMI output mode: `none`, `summary`, `full`, `metadata`, or `metadata-and-visuals` |
| `--design-variants / --no-design-variants` | `false` | Scan STEP design variant records into metadata and import reports |
| `--design-variant` | unset | Select a STEP design variant label, effectivity value, numeric/string assignment, record id, or referenced label and filter imported geometry; may be passed more than once |
| `--import-existing-meshes / --no-import-existing-meshes` | `true` | Prefer existing STEP tessellation payloads when the importer exposes them |
| `--multi-file-import / --single-file-import` | `false` | Resolve quoted external STEP references from a master STEP file |
| `--material-library` | unset | Vendor material-library JSON/MTL/ZIP file or folder to apply during import; may be passed more than once |
| `--delete-free-vertices / --keep-free-vertices` | `false` | Drop construction-only point shapes during STEP import |
| `--delete-lines / --keep-lines` | `false` | Drop construction-only line shapes during STEP import |
| `--construction-curve-policy` | `preserve-metadata` | Construction-only line policy: `preserve-metadata`, `delete`, or `tessellate-tubes` |
| `--construction-curve-tube-radius` | `0.01` | Tube radius in source units when construction curves are tessellated as tubes |
| `--source-units` | STEP header | Override source units for normalization |
| `--source-meters-per-unit` | STEP header | Override source meters per unit for normalization |
| `--source-up-axis` | `Z` | Declared source up axis: `Y` or `Z` |
| `--source-handedness` | `right` | Declared source handedness: `right` or `left` |
| `--target-units` | source units | Normalize imported asset units, for example `metre` |
| `--target-meters-per-unit` | source factor | Normalize imported asset meters per unit with a custom factor |
| `--target-up-axis` | source axis | Normalize imported asset up axis to `Y` or `Z` |
| `--target-handedness` | source handedness | Normalize imported asset handedness to `right` or `left` |
| `--heal-brep` | `false` | Run BREP healing before inspection output |
| `--heal-tolerance` | `0.05` | BREP healing tolerance |
| `--cleanup-overlapping-faces / --keep-overlapping-faces` | `true` | Remove redundant coplanar BREP faces that overlap enough to z-fight |
| `--overlap-area-ratio` | `0.995` | Minimum smaller-face area ratio for BREP overlap cleanup |
| `--remove-sliver-faces` | `false` | Request tiny sliver-face removal during BREP healing; current backend support is limited and reports warnings when unavailable |
| `--max-sliver-area` | `1e-4` | Area threshold for sliver-face reporting |
| `--filter` | unset | Report matched assembly nodes and parts |
| `--exclude-filter` | unset | Exclude selector matches from `--filter` results |

Supported filter expressions:

| Expression | Meaning |
|------------|---------|
| `path=*/Fasteners/*` | Match node paths |
| `name=Bolt*` | Match node names |
| `part=part_123` | Match part ids |
| `part-name=Housing*` | Match part names |
| `material=*Steel*` | Match material ids or names |
| `metadata.step_label=0:1:*` | Match metadata values |
| `triangles<=1200` | Match triangle counts |
| `vertices>=300` | Match vertex counts |
| `size>=50` | Match bounding-box diagonal |

Repeated `--filter` flags are combined with logical AND. Use `--exclude-filter` for negative selectors.

## Pipeline files

Use `--pipeline` when different assembly branches need different ordered steps.

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

[[filters]]
name = "fasteners"
path = "*/Fasteners/*"
names = ["Bolt*", "Nut*", "Washer*"]

[[filters]]
name = "large_castings"
path = "*/Housing/*"
min_diagonal = 50.0

[[steps]]
op = "tessellate"
where = "large_castings"
sag = 0.03
sag-ratio = 0.005
max-polygon-length = 5.0
free-edge-report = true
reuse-existing-meshes = false
angle = 10.0

[[steps]]
op = "tessellate"
where_not = "large_castings"
sag = 0.2
angle = 20.0

[[steps]]
op = "merge_vertices"
tolerance = 0.001
preserve_uvs = true
preserve_material_boundaries = true
quality_report = true

[[steps]]
op = "delete_degenerate_polygons"
area_epsilon = 1e-12

[[steps]]
op = "merge"
where = "fasteners"
mode = "by_material"
metadata = "combine"
```

```bash
fascat convert motor.step motor.glb --pipeline realtime.toml
```

Pipeline files are validated before conversion starts. Unknown top-level,
filter, import/export, or operation keys are rejected, and option constraints
such as invalid numeric ranges, missing `external_path`, conflicting `where` /
`where_not`, or unsupported operation names fail during parse. CLI errors include
`line N` when the source file location can be identified.

## Unity-inspired capability matrix

This matrix maps Unity Asset Transformer-style CAD-to-runtime capabilities to the
current Fascat behavior. Use dry-run `operation_diagnostics`, report
`workflow_summary`, and report step warnings to distinguish exact work from
fallbacks.

| Capability | Fascat status | Report or diagnostic | Next step |
|------------|---------------|----------------------|-----------|
| CAD import, hierarchy, names, transforms, colors, metadata | Implemented for STEP and IGES; native BREP imports as a single source-shape part; explicit multi-root STEP path lists and master STEP quoted external-reference graphs are imported with deterministic member namespaces and repeated external-file occurrences | `import` report stats, cleanup counts, STEP import decisions, pipeline import options, per-part loaded-representation reports, construction-curve delete/preserve/tube policy including free construction edges split from mixed face+curve shapes, space normalization transforms, source texture resolved/missing counts, CAD material PBR mapping metadata, AP242 PMI warnings plus textual semantic graph records for dimension/location/tolerance/datum/note families, shape-aspect/product target support records, and inbound callout/annotation associativity records, deterministic glTF/USD PMI marker and simple vector text geometry when `metadata_and_visuals` is requested, design-variant metadata counts/records, serial/lot/date effectivity values/ranges, simple boolean/numeric/string condition operators, numeric arithmetic/function expression operands including rational representation items, expression-extension numeric values, and elementary trig/log/exp functions including binary `ATAN_FUNCTION`, odd-function conditions, string concat/substring/index/format/expression-extension/length/value expression operands, boolean/logical literal values, explicit boolean variable assignments, effectivity relationship links, product-definition/configuration effectivity usage target labels, applied effectivity/ineffectivity-assignment target labels, gated effectivity context-assignment target labels, resolved-reference labels, and name/reference-based geometry-selection counts, `read_step_many` member records/warnings, and `external_reference_graph` unique-source/resolved-occurrence records for master STEP assemblies | Add full AP242 conditional/effectivity geometry evaluation, full AP242 PMI semantic coverage and graphical presentation reconstruction, richer vendor-specific external-reference placement transforms, and closed vendor material-library containers |
| BREP healing | Partial | `heal_brep`; records open shells, free/unstitched edges, small edges, sliver counts, open-shell grouping counts/status, same-domain face/edge reductions, overlap/z-fighting face-pair counts, resolved overlap counts, OCCT same-domain cleanup status, and OCCT overlap cleanup status; disconnected open shell groups are processed independently before cleanup; sliver removal warns that the backend leaves shapes unchanged | Implement sliver-face removal and deeper face/wire repair |
| Tessellation | Implemented | `tessellate` report options, unit-aware tolerance policy with source/local/target unit conversions and normalized-tolerance warnings, explicit sag-ratio, existing mesh reuse/retessellation controls, tessellation-time source BREP cleanup even when imported meshes are reused, size-adaptive `part_settings` helpers, material/metadata/curved-BREP-driven `detail_adaptive` per-part criteria, conditional edge-control cleanup pass metadata, max-polygon-length diagnostics, free-edge diagnostics, retained-patch/submesh risk warnings, quality advisories for coarse absolute sag and aggressive polygon-length settings, attribute-provenance metadata, and quality metadata | Add intrinsic/conformal CAD UV solving and deeper curvature-targeted profiles |
| Mesh repair | Implemented for core cleanup | `repair` report step; mesh metadata always records face-orientation strategy/status and normal-orientation strategy/status, while `RepairOptions.quality_report=True` adds before/after duplicate polygon, degenerate triangle, boundary edge, boundary gap, non-manifold edge, T-junction, flipped closed-component, and non-orientable shared-edge counts; standalone `merge_vertices` and `delete_degenerate_polygons` are available through Python, CLI flags, and TOML pipelines with before/after reports; vertex-merge reports use Euclidean tolerance matching across spatial bucket boundaries and always include removed counts and tolerance-risk warnings, while `MergeVerticesOptions.quality_report=True` adds same-position candidates, exact-duplicate, boundary, non-manifold, hard-edge, T-junction, boundary-gap, and near-duplicate candidate counts plus skipped normal, tangent, UV, and material-boundary protection reasons; degenerate-polygon reports include duplicate-vertex, collapsed-edge, near-flat, and exact duplicate-polygon removal reasons | Add T-junction sewing, boundary-gap stitching, non-manifold cracking, topology-only merge connectivity, tolerance-overlap cleanup, and backend implementation for viewer/open-shell orientation strategies |
| Staging, normals, tangents, UV metadata | Partial | `stage` report step; normal generation supports hard-edge angle, angle or area weighting, and preserve-versus-override behavior with generated/regenerated/preserved/disabled metadata; tangents require the selected UV channel when generated; existing tangents are preserved by default and mesh/asset metadata report generated, regenerated, preserved, missing, invalidated, or dropped tangent states; UV metadata records per-channel domain, bounds, validation status, degenerates, conditional overlap-check status/counts, seam graph edge/component/vertex/length metrics for duplicated-position UV discontinuities, conditional distortion-check status with island counts, pack efficiency, normalized-space utilization, xatlas pack status/dimensions/padding/utilization, and angle/edge distortion for bake or stretch-diagnostic channels, AABB box-projection scope/scale/destination/override/unit policy, UV0-to-UV1 copy status, explicit normalization status, unwrap solver intent, and sharp-to-seam/forbid-overlap policy intent | Add backend-enforced unwrap solver controls, island merge, and tileable UV alignment |
| Material baking | Implemented for factor/face atlas baking, imported sidecar texture slots, JSON/MTL material-library sidecars, and ZIP material-library packages | `bake_materials` emits first-class raster atlas images for selected maps, glTF exports baked or imported source images as material texture slots, USD exports them as `UsdUVTexture` shader networks, exporters use effective opacity consistently, glTF marks scalar or baked-opacity materials as alpha-blended, `process_textures` resizes/dedupes/fallback-converts first-class images, and import reports include material-library resolved/matched/unmatched counts | Add high-poly normal transfer and closed vendor material-library containers |
| Hole removal | Approximate | `remove_holes` warns when it falls back to mesh boundary classification and filling | Add BREP feature-level removal for closed cylindrical and pocket holes |
| Occlusion removal | Approximate | `remove_occluded` warns that sampled visibility may require higher precision and records candidate counts, sampled face coverage, direction coverage, and confidence metadata | Add acceleration structures and optional raster/GPU backends for high-poly production scenes |
| Decimation | Partial | `decimate`; reports target strategy as target count, target ratio, or quality target-error simplification; aggressive LOD0 ratios emit distortion warnings, UV importance modes control seam preservation or UV cleanup, pre-cleanup can remove unused UVs/tangents, painted/protected and low-AO faces can become simplification constraints, and report metadata includes RAM estimates plus configurable iterative-threshold status, actual pass counts, per-part target allocation summaries, UV constraint status, quality-bound metadata, protected feature-face counts, and importance-face counts | Add continuous weighted decimation and retopology |
| LOD generation | Partial | `run_lod_generators` / `lods` report steps; untessellated parts are skipped with `lod_status="skipped_no_mesh"`, generated/skipped counts, warnings, per-level counts, progressive simplification source metadata, source/added/full-chain mesh payload estimates, chain advisories for excessive levels, aggressive close-view ratios, far LOD bake metadata, scene-level far proxy counts, per-level instance/material/texture/culling policy metadata, Unity `MSFT_lod` glTF export, Unreal `_LOD#` separate-node glTF export, and extras-only LOD export mode | Add measured engine runtime validation for exported LOD profiles |
| Instance reconstruction | Partial | `optimize_scene` reconstructs exact matching mesh fingerprints and position-tolerant near-identical meshes when topology, vertex attributes, material assignments, and metadata match; metadata records reconstructed part/occurrence counts, similarity tolerance/candidate counts, vertex/triangle savings, and estimated mesh-payload byte savings; hierarchy reports separate draw calls by mesh count, referenced materials, submesh/material slots, instances, merged batches, and export-advisor warnings when merging removes reusable instances | Add transform-aware matching and export-format-specific compressed size estimates |
| Runtime compression | Implemented | glTF quantization, meshopt, Draco, and KTX2/Basis texture compression are implemented; `--export-preset desktop/web/mobile/vr/ar` resolves compression defaults and conversion-time texture resize/dedupe cleanup; glTF write reports list runtime extension dependencies, expected support, Unity glTFast/web/mobile/XR compatibility notes with fallback behavior, and a runtime decision matrix for quantization, meshopt, Draco, KTX2/Basis, and PNG/JPEG fallbacks | Add full renderer/material validation |
| Export and budgets | Implemented for USD, USDZ, glTF/GLB, OBJ, STL | `write` report includes glTF runtime dependencies, file size, estimated geometry/texture/metadata payload bytes, referenced/unused/written material counts, source/referenced/unused/duplicate-reference/written image counts, and optional budget warnings; `gltf_size_ladder` measures temporary baseline, quantized, meshopt, Draco, texture-compressed, and requested GLB variant sizes when requested; glTF, USD, and OBJ exports prune unused materials from the written artifact; glTF also omits images referenced only by unused materials and deduplicates repeated embedded texture/image references; `texture_export_policy` records profile texture caps, resize candidates, estimated resized bytes/savings, KTX2/Basis requested/supported state, and alpha-aware PNG/JPEG fallback policy before write, including PNG compression, JPEG quality, and transparency-loss warnings; `profile_budget` reports selected-profile target FPS plus triangle, vertex, per-mesh vertex, texture-resolution, texture-memory, estimated load-time, draw-call budget status, draw-call breakdown fields, supported compression/runtime-extension caps, and Unity reference triangle/draw-call ranges when present; `validate --runtime-browser` adds measured browser/WebGL load/FPS reports when a local browser is available; `validate --runtime-browser-preview` writes browser/WebGL screenshots for supported glTF/GLB primitives with quantized attribute support, Draco decode, meshopt fallback-buffer support, meshopt-only/no-fallback decode, default Python KTX2/Basis texture decode on supported platforms with KTX-Software fallback, optional KHR_texture_basisu fallback sampling, base-color texture sampling, and compressed-extension preflight diagnostics; `validate --runtime-engine unity|unreal` adds packaged or project-backed engine harness reports, and `--runtime-engine-preview` writes packaged Unity/glTFast PNG screenshots with fixed multi-frame FPS, packaged Unreal commandlet geometry-rasterized GLB previews with backend/limitation reporting, or custom harness screenshots with render-status fields; `--runtime-engine-baseline` applies image-diff thresholds to rendered engine preview artifacts; `validate --visual-preview` and `--lod-preview-dir` write deterministic software-rendered PNG review artifacts; `runtime-fixtures` writes bundled material/lighting/KTX2 fallback/LOD-profile parity GLBs, software baselines, target-golden paths, preview commands, capture reports, promoted target goldens, required-golden validation, and golden coverage audit reports | Add cross-platform fallback-free KTX2/Basis decode, full Unreal scene renderer screenshots/FPS, and checked-in engine-specific material/lighting golden corpora |
| PMI metadata export | Partial | STEP AP242 import extracts common dimension, location, generic and named geometric tolerance, plus/minus tolerance, datum, datum-reference, feature-control-frame, and annotation text records into typed `PmiAnnotation` metadata plus a textual STEP semantic reference graph with common target support records and inbound callout/annotation associativity records; glTF `extras.fascat` and USD `customData` include PMI records, resolve links through `source_part_ids` after merge/replace, and write deterministic marker meshes with simple vector text glyphs when `metadata_and_visuals` or `full` is requested | Add full AP242 semantic coverage and graphical visual presentation reconstruction |

## Validate flags

| Flag | Default | Description |
|------|---------|-------------|
| `--geometry-quality` | `false` | Enable all geometry quality checks in the validation report |
| `--non-manifold-edges` | `false` | Report non-manifold edge counts |
| `--open-boundaries` | `false` | Report open boundary counts |
| `--self-intersections` | `false` | Report detected self-intersections with bounded triangle-triangle checks and lower-bound fields when the pair limit is hit |
| `--sliver-triangles` | `false` | Report degenerate and sliver triangle stats |
| `--tiny-parts` | `false` | Report tiny part stats |
| `--draw-call-estimate` | `false` | Report material count, draw-call estimate, mesh/submesh slots, instances, and merged batch counts |
| `--visual-risk` | `false` | Report before/after visual risk warnings |
| `--visual-preview` | unset | Write a deterministic software-rendered PNG preview of the validated output mesh |
| `--runtime-browser-preview` | unset | Write a browser/WebGL-rendered PNG preview for supported glTF/GLB primitives |
| `--visual-baseline` | unset | Compare `--visual-preview` against a baseline PNG and fail validation when thresholds are exceeded |
| `--visual-diff-pixel-tolerance` | `0` | Per-channel byte tolerance ignored when counting changed visual diff pixels |
| `--visual-diff-mean-threshold` | `0.0` | Maximum allowed mean absolute error for the visual baseline diff |
| `--visual-diff-changed-pixel-ratio` | `0.0` | Maximum allowed ratio of changed visual diff pixels |
| `--lod-preview-dir` | unset | Write LOD switching preview PNGs and `lod-switching.png` into a directory |
| `--runtime-browser` | `false` | For glTF/GLB, run optional headless browser/WebGL load and FPS measurement |
| `--runtime-browser-command` | unset | Browser executable for `--runtime-browser` or `--runtime-browser-preview`; otherwise `FASCAT_BROWSER` or common Chromium/Chrome names are used |
| `--runtime-duration` | `2.0` | Browser FPS measurement duration in seconds |
| `--runtime-timeout` | `15.0` | Browser runtime validation timeout in seconds |
| `--runtime-engine` | unset | Optional engine runtime harness to run: `unity` or `unreal` |
| `--runtime-engine-command` | unset | Unity or Unreal executable for `--runtime-engine`; otherwise environment variables or common executable names are used |
| `--runtime-engine-project` | unset | Optional Unity project folder or Unreal `.uproject` containing a custom Fascat runtime harness; omitted uses a packaged temporary harness |
| `--runtime-engine-preview` | unset | Request a PNG preview from a Unity/Unreal runtime harness |
| `--runtime-engine-baseline` | unset | Compare `--runtime-engine-preview` against a baseline PNG with the `--visual-diff-*` thresholds |
| `--runtime-engine-timeout` | `120.0` | Unity/Unreal runtime harness timeout in seconds |
| `--filter` | unset | Scope validation-time geometry analysis with an assembly selector |
| `--exclude-filter` | unset | Exclude selector matches from validation-time analysis |
| `--report` | unset | Write validation and geometry quality report as JSON |

Example:

```bash
fascat validate motor.glb \
  --filter 'path=*/Fasteners/*' \
  --geometry-quality \
  --non-manifold-edges \
  --open-boundaries \
  --self-intersections \
  --sliver-triangles \
  --tiny-parts \
  --draw-call-estimate \
  --visual-risk \
  --visual-preview preview.png \
  --visual-baseline baseline.png \
  --lod-preview-dir preview-lods/ \
  --runtime-browser \
  --report report.json
```

## File arguments

Use `-` for standard streams:

```bash
cat input.step | fascat inspect -
cat input.step | fascat convert - - --profile realtime-web
cat output.usdc | fascat validate -
```

When the convert output argument is omitted for a file input, Fascat writes beside the input with a `.usdc` suffix. Stdin input requires an explicit output path or `-`.

When output is `-`, USD bytes are reserved for stdout and progress/errors stay on stderr.

Supported input suffixes are `.step`, `.stp`, `.igs`, `.iges`, and `.brep`.
Supported output suffixes are `.usd`, `.usda`, `.usdc`, `.usdz`, `.gltf`, `.glb`, `.obj`, `.stl`, and `.fbx`.

`--debug` is only valid with `.usd` or `.usda` output. Binary `.usdc`, `.gltf`, and `.glb` output is rejected when debug mode is enabled.

`convert` validates the generated asset before reporting success. If validation fails, the command exits non-zero.

## Benchmarking

Use the benchmark harness before performance-sensitive changes:

```bash
make benchmark
# or
uv run python scripts/benchmark.py tests/fixtures/vertical-screw.step --output-dir dist/benchmarks --output-suffix .glb
```

The harness writes JSON with total wall time, process peak RSS where the platform exposes it, per-report-step durations, output paths, and final mesh statistics. Pass `--repeat N` for repeated runs and `--validate-output` when the validation round trip should be included in the measured path.

## Output streams

| Stream | Contents |
|--------|----------|
| stdout | Primary command output and `--json` payloads |
| stderr | Errors, source counts, per-stage progress, warnings, and diagnostics |

When `--json` is active, expected runtime errors are reported as JSON payloads on stdout and still exit non-zero.

## Environment variables

| Variable | Description |
|----------|-------------|
| `NO_COLOR` | Set to any non-empty value to disable color output |

Color is also disabled when `--no-color` is passed, `TERM=dumb`, or the relevant stream is not a TTY.

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Runtime failure |
| `2` | Invalid usage |

## Shell completions

Typer provides built-in shell completion support:

```bash
# Install completion for your shell (auto-detects)
fascat --install-completion

# Show the completion script without installing
fascat --show-completion
```
