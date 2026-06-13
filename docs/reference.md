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
| `--verbose` | `-v` | `false` | Enable verbose diagnostics; conversion warnings are not truncated |
| `--quiet` | `-q` | `false` | Suppress non-essential output |
| `--json` | — | `false` | Output results as JSON |
| `--no-color` | — | `false` | Disable ANSI color output |
| `--dry-run` | `-n` | `false` | Preview changes without applying them |
| `--no-input` | — | `false` | Reserved guard for non-interactive runs; commands must not prompt when set |

`-h` / `--help` and `-V` / `--version` are invocation-wide controls. They work before or after subcommands and ignore other arguments.

`--verbose` expands diagnostic output on stderr. Conversion report warnings are normally capped after 10 messages; with `--verbose`, Fascat prints the full warning list. `--quiet` suppresses non-essential diagnostics even when `--verbose` is also set.

`--no-input` is reserved for non-interactive automation. Fascat does not prompt today, and future prompts must first check that stdin is a TTY and that `--no-input` is not set.

A `convert --dry-run` emits JSON with `operation_diagnostics` — planned operations
each tagged `exact`, `approximate`, or `metadata_only`. With `--pipeline` it also adds
`pipeline_advisories` warning about ordering issues (decimation before repair, tangents
without UV0, AO baking without UV1, LOD generation before LOD0 optimization).

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Command completed successfully |
| `1` | Runtime failure, validation failure, missing file, unsupported backend, or failed conversion |
| `2` | Usage error, invalid option combination, unsupported extension, or invalid argument value |
| `130` | Interrupted by Ctrl-C / SIGINT |

Errors are written to stderr in human mode. With `--json`, command errors are emitted as a single JSON object on stdout with the same command context plus an `error` string, while diagnostics remain on stderr.

## JSON output schemas

All `--json` payloads are one JSON object on stdout. Optional sections are omitted when the matching flag or workflow is not used.

### `version`

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Always `version` |
| `version` | string | Fascat package version |

### `inspect`

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Always `inspect` |
| `input` | string | Input path, or `-` for stdin |
| `profile` | string | Inspection profile name |
| `dry_run` | boolean | Whether execution was skipped |
| `stats` | object | Asset counts such as parts, occurrences, materials, vertices, and triangles |
| `options` | object | Resolved profile options |
| `root` | object | Root node dictionary |
| `parts` | array | Part dictionaries |
| `materials` | array | Material dictionaries |
| `metadata_summary` | object | Compact metadata counts and keys |
| `pmi_summary` | object | Compact PMI counts and types |
| `report` | object | Import/conversion warnings and report steps |
| `selection` | object | Present when filters are used |
| `error` | string | Present only on failure |

### `convert`

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Always `convert` |
| `input` | string or array | Input path, stdin marker, or multi-root input list |
| `output` | string | Output path, or `-` for stdout |
| `profile` | string | Resolved conversion profile |
| `dry_run` | boolean | Whether execution was skipped |
| `operation_diagnostics` | array | Dry-run operation entries with `operation`, `level`, and `message` |
| `pipeline_advisories` | array | Dry-run pipeline-ordering warnings when `--pipeline` is used |
| `stats` | object | Output asset counts on successful conversion |
| `report` | object | Report warnings and workflow steps on successful conversion |
| `error` | string | Present only on failure |

### `validate`

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Always `validate` |
| `output` | string | Output path, or `-` for stdin |
| `dry_run` | boolean | Whether execution was skipped |
| `stats` | object | Validation stats from the selected exporter validator |
| `analysis` | object | Present with geometry quality checks, reports, or filters |
| `runtime_browser` | object | Present with `--runtime-browser` |
| `runtime_browser_preview` | object | Present with `--runtime-browser-preview` |
| `runtime_engine` | object | Present with `--runtime-engine` |
| `runtime_engine_diff` | object | Present when engine preview baseline comparison runs |
| `visual_preview` | object | Present with `--visual-preview` |
| `visual_diff` | object | Present with `--visual-baseline` |
| `lod_preview` | object | Present with `--lod-preview-dir` |
| `error` | string | Present only on failure |

### `runtime-fixtures`

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Always `runtime-fixtures` |
| `output_dir` | string | Target suite directory |
| `dry_run` | boolean | Whether files or captures were skipped |
| `fixtures` | array | Fixture entries written or planned |
| `manifest` | string | Manifest path when a suite is written |
| `capture` | string or null | Requested capture target |
| `captures` | array | Capture results when a runtime target is used |
| `golden_coverage` | object | Golden-image coverage report when requested |
| `error` | string | Present only on failure |

Conversion reports wrap the run in four steps: `preflight` (before expensive work —
flags missing patch cleanup, orientation prep, UV/tangent ordering, AO-bake UV1
prerequisites, LOD0 optimization, and glTF compression notes), then near the end
`workflow_recipe` (the profile's target recipe with honored/disabled/metadata-only/unsupported
counts), `conversion_manifest` (resolved profile, import, operation, and export
settings), and `workflow_summary` (preparation stages mapped to run/skipped status).

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

STEP import reports record two diagnostic steps:

- **`import_decisions`** — each import toggle (PMI, variants, existing meshes, construction cleanup, coordinate normalization) as requested vs. effective, with a status of `honored`, `approximated`, `unsupported`, `disabled`, `not_present`, or `backend_default`.
- **`loaded_representations`** — a per-part summary of BREP/construction-shape inputs, deleted construction-only nodes, and source topology counts. Free construction edges split from mixed face+curve shapes appear as separate construction-line parts or cleanup counts, per the construction-curve policy.

**PMI** (`--pmi`) runs a textual AP242 scan, writes typed `PmiAnnotation` metadata,
and reports a `pmi_semantic_graph`. **Design variants** (`--design-variants`) scan
configuration, effectivity, and condition records; `--design-variant` then prunes
geometry by variant label, record id, effectivity value/range, or `label=value`
assignment. The full list of supported STEP entity families and selection-resolution
rules is documented once on the [Python API page](api.html#pmi-import) — the CLI flags
drive exactly the same importer. `metadata_and_visuals` export adds deterministic
glTF/USD marker meshes with simple vector text glyphs.

> Full AP242 conditional/effectivity geometry evaluation and graphical PMI
> presentation reconstruction remain planned backend work.

**IGES** import uses the same OCP/XDE shape-tree traversal for hierarchy, transforms,
colors, and materials. **BREP** import creates one root occurrence and one part,
keeping the native source shape for tessellation and healing.

## Convert flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `realtime-desktop` | Conversion profile: `inspect-only`, `realtime-desktop`, `realtime-web`, `realtime-mobile`, `virtual-reality`, `augmented-reality`, or `mixed-reality` |
| `--target-device-profile` | unset | TOML or JSON target-device budget overlay for the selected profile |
| `--pipeline` | unset | TOML pipeline file with named filters and ordered conversion steps |
| `--input` | unset | Additional STEP root input for explicit multi-root conversion; may be passed more than once |
| `--stdout-format` | `usda` | Output format used only when the output path is `-`: `usda`, `usdc`, `usdz`, `gltf`, `glb`, `obj`, `stl`, or `fbx` |
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
| `--merge-vertex-area-epsilon` | bbox-derived | Area threshold for degenerate polygons after vertex merging (default: 1e-12 × squared bbox diagonal) |
| `--delete-degenerate-polygons` | `false` | Run standalone degenerate polygon cleanup after vertex merging |
| `--degenerate-area-epsilon` | bbox-derived | Area threshold for standalone degenerate polygon cleanup (default: 1e-12 × squared bbox diagonal) |
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
| `--draco-compression-level` | `5` | Draco compression level, 0 (fastest) to 10 (smallest) |
| `--draco-quantize-position` | `14` | Draco position quantization bits (1-30) |
| `--draco-quantize-normal` | `10` | Draco normal quantization bits (1-30) |
| `--draco-quantize-texcoord` | `12` | Draco texcoord quantization bits (1-30) |
| `--draco-quantize-color` | `8` | Draco color quantization bits (1-30) |
| `--texture-compression` | unset | Compress glTF textures with KTX2/Basis: `ktx2` or `basisu` |
| `--ktx2-quality` | `128` | KTX2/Basis encoder quality level (0-255) |
| `--ktx2-effort` | `2` | KTX2/Basis encoder compression effort (0-6) |
| `--ktx2-uastc/--ktx2-etc1s` | derived | Force UASTC or ETC1S encoding (default derives from `--texture-compression`) |
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

### Units

- Linear tolerances and sizes (`--sag`, `--min-edge-length`, `--max-edge-length`, `--max-polygon-length`, `--heal-tolerance`, `--max-sliver-area`, `--region-size`, `--max-hole-diameter`) use the source asset's working units unless stated otherwise.
- Angles (`--angle`, `--normal-tolerance`, `--hard-edge-angle`) are degrees.
- Ratios (`--ratio`, `--lods`, decimation ratios) and screen-coverage values are fractions between `0` and `1`; LOD ratios must be sorted highest-to-lowest detail. File-size budgets are megabytes; atlas/bake sizes are pixels.
- Import space normalization uses a root transform: source coordinates stay in source units while the asset declares the target units, up-axis, and handedness, and the import report records the transform.
- `tessellate`, `heal_brep`, and `repair` report a `tolerance_policy` with effective source/target units, meter conversions, the active deflection kind, converted absolute lengths, and which cleanup backends are implemented.

### Repair

- `face_orientation="exterior"` is the implemented closed-component winding path; source-trusted, preserve, viewer-standpoint, open-shell, and unstitched-group policies are explicit in metadata rather than hidden behind `fix_winding`.
- `fix_winding` flips coherent closed shells with inward signed volume and records before/after flipped-component counts; it detects non-orientable shared-edge cycles first and warns on Möbius-like topology it cannot fix.
- T-junctions and boundary gaps are reported by default and fixed only with the opt-in `fix_t_junctions` / `stitch_boundary_gaps` flags; stitched vertices keep the surviving representative vertex's normals, tangents, and UVs, and UV-conflicting merges are counted in metadata.
- `--delete-degenerate-polygons` removes repeated-vertex, collapsed-edge, near-flat, and exact-duplicate polygons (separate report counts per reason); use `--keep-duplicate-polygons` to only report duplicates.

### Decimation

- Keeping under 20% of source triangles emits an LOD0 distortion warning — prefer aggressive ratios for distant LODs.
- `--decimate` without `--target-triangles`/`--ratio` seeds its target from the profile or `--target-device-profile` triangle budget.
- `--decimate-criterion quality` passes tolerance-derived error bounds to the backend and records bound/result metadata.
- Reports include `target_strategy` (target count / ratio / quality), estimated RAM (Unity's ~5 GB per million source triangles), iterative pass counts, and per-part target allocation; plus protected hard-edge, hole-boundary, material-boundary, UV-seam, and silhouette face counts.
- `--uv-importance`: `ignore` strips UV/tangents first, `preserve-seams` uses then strips them, `preserve-islands` keeps them. `--preserve-painted-areas` and `--preserve-ambient-occlusion` add painted/protected and low-AO faces as constraints. `--decimate-cleanup-attributes unused-uvs,tangents` removes unused UV channels/tangents first.

### UVs

- `--uv1 unwrap`/`lightmap` bake channels are packed by xatlas with configured padding/resolution and report pack dimensions, utilization, and padding status.
- `box` UV generation is reported as AABB projection; `--uv-aabb-scope`, `--uv3d-size`, and `--uv-preserve-existing` record local/shared bounds, scale, destination, override policy, and units.
- `--uv-sharp-to-seam` and `--uv-forbid-overlapping` are recorded as intent (the xatlas backend doesn't expose them directly) and validated after generation.

### LODs

- LOD reports separate source, added-LOD, and full-chain vertex/triangle counts and payload bytes, so the memory/size cost of extra levels is visible. Ratio LODs simplify progressively from the previous level while preserving each ratio against the source count.
- Chain advisories warn on more than four levels, over-aggressive LOD1/LOD2, and geometry-only far LODs that should bake to one mesh/material. Per-level metadata records simplification source, instance reuse, material merge, texture bake, culling-granularity changes, and resolved export representation.
- `--lod-engine-profile unity` emits `MSFT_lod` variant nodes; `unreal` emits separate `_LOD#` scene nodes for tools that ignore `MSFT_lod`.

### Holes, occlusion, and compression

- `--remove-holes` uses mesh boundary classification and filling when BREP hole removal is unavailable; `--hole-types` filters inferred through/blind/surface loops.
- `--remove-occluded` uses deterministic sampled visibility — `--occlusion-strategy` sets the direction set, `--hemi-evaluation` restricts to upper-hemisphere/side views, and `--occlusion-level` picks parts/submeshes/triangles. Metadata records sample/direction coverage and a confidence score.
- `--draco` runs the Draco encoder for glTF/GLB. `--texture-fallback-format auto` keeps alpha-bearing sets PNG and color-only sets JPEG when KTX2/Basis isn't requested; explicit `jpeg` warns when it would discard transparency.

### Validation previews

`fascat validate` can write review artifacts (full detail on the
[Python API page](api.html#validation)):

- `--visual-preview` / `--lod-preview-dir` — software-rendered PNGs of the output mesh and per-LOD contact sheet.
- `--runtime-browser` / `--runtime-browser-preview` — headless Chromium WebGL load/FPS measurement and screenshots for supported glTF/GLB primitives.
- `--runtime-engine unity|unreal` (+ `--runtime-engine-preview` / `--runtime-engine-baseline`) — packaged or custom engine-harness load metrics and rendered previews. Set `FASCAT_UNITY`/`UNITY_EDITOR`/`FASCAT_UNREAL`/`UNREAL_EDITOR` or `--runtime-engine-command` if the executable isn't on PATH.
- `fascat runtime-fixtures DIR` — bundled parity GLBs with software baselines and a manifest; `--capture`, `--promote-goldens`, `--require-goldens`, and `--check-goldens` manage target golden comparison.

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

The **Report step** column names where to look for the detail; the linked sections
above and on the [Python API page](api.html) document each field.

| Capability | Status | Report step | Next step |
|------------|--------|-------------|-----------|
| CAD import (hierarchy, names, transforms, colors, metadata, PMI, variants) | Implemented for STEP/IGES; BREP as a single part; multi-root and master-file external references supported | `import` (decisions, loaded representations, PMI semantic graph, design-variant records, `external_reference_graph`) | Full AP242 conditional/effectivity geometry evaluation; full PMI semantic + graphical coverage; richer external-reference transforms |
| BREP healing | Partial | `heal_brep` (open shells, edges, slivers, same-domain/overlap cleanup status) | Sliver-face removal and deeper face/wire repair |
| Tessellation | Implemented | `tessellate` (tolerance policy, detail-adaptive criteria, quality metrics and advisories) | Conformal CAD UV solving; deeper curvature-targeted profiles |
| Mesh repair | Implemented for core cleanup | `repair`, plus standalone `merge_vertices` / `delete_degenerate_polygons` reports | T-junction sewing, boundary-gap stitching, non-manifold cracking, viewer/open-shell orientation backends |
| Staging (normals, tangents, UV metadata) | Partial | `stage` (normal/tangent provenance, per-channel UV domain/seam/distortion/pack metadata) | Backend-enforced unwrap controls, island merge, tileable UV alignment |
| Material baking | Implemented for factor/face atlas, sidecar textures, JSON/MTL/ZIP libraries | `bake_materials`; import material-library match counts | High-poly normal transfer; closed vendor library containers |
| Hole removal | Approximate | `remove_holes` (warns on mesh-fill fallback) | BREP feature-level removal for closed holes |
| Occlusion removal | Approximate | `remove_occluded` (sample/direction coverage, confidence) | Acceleration structures and raster/GPU backends |
| Decimation | Partial | `decimate` (target strategy, RAM/pass estimates, protected/importance counts) | Continuous weighted decimation and retopology |
| LOD generation | Partial | `lods` / `run_lod_generators` (per-level counts, chain advisories, engine export mode) | Measured engine runtime validation of LOD profiles |
| Instance reconstruction | Partial | `optimize_scene` (reconstructed counts, savings, draw-call breakdown) | Transform-aware matching; compressed size estimates |
| Runtime compression | Implemented | `write` `runtime_dependencies` / `runtime_decision_matrix` (quantize, meshopt, Draco, KTX2/Basis) | Full renderer/material validation |
| Export and budgets | Implemented for USD, USDZ, glTF/GLB, OBJ, STL | `write`, `gltf_size_ladder`, `texture_export_policy`, `profile_budget`; `validate` runtime/preview reports | Fallback-free KTX2/Basis decode; full Unreal scene rendering; engine golden corpora |
| PMI metadata export | Partial | `PmiAnnotation` metadata + `pmi_semantic_graph`; glTF `extras.fascat` / USD `customData`; markers when `metadata_and_visuals` | Full AP242 semantic + graphical presentation |

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
cat input.step | fascat convert - - --stdout-format glb --profile realtime-web
cat output.usdc | fascat validate -
```

When the convert output argument is omitted for a file input, Fascat writes beside the input with a `.usdc` suffix. Stdin input requires an explicit output path or `-`.

When output is `-`, `--stdout-format` selects the emitted format and defaults to `usda`. Progress/errors stay on stderr.

Supported input suffixes are `.step`, `.stp`, `.igs`, `.iges`, and `.brep`.
Supported output suffixes are `.usd`, `.usda`, `.usdc`, `.usdz`, `.gltf`, `.glb`, `.obj`, `.stl`, and `.fbx`.

`--debug` is only valid with `.usd` or `.usda` output. Binary `.usdc`, `.gltf`, and `.glb` output is rejected when debug mode is enabled.

`convert` validates the generated asset before reporting success. If validation fails, the command exits non-zero.

`convert` prints up to 10 report warnings (budget violations, tessellation and UV
advisories, export policy notes) on stderr after a successful run; `--quiet`
suppresses them and `--json` keeps them in the JSON payload instead. Pass
`--report report.json` for the full list.

All exporters write transactionally: content is produced at a hidden temp file in the
destination directory, validated there, and atomically renamed into place only on success.
A failed or interrupted export never leaves a partial or corrupt file at the output path
(sidecar files from a previous differently-named export are not garbage-collected).

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
| `130` | Interrupted (Ctrl-C) — no partial output file is left behind |

## Shell completions

Typer provides built-in shell completion support:

```bash
# Install completion for your shell (auto-detects)
fascat --install-completion

# Show the completion script without installing
fascat --show-completion
```
