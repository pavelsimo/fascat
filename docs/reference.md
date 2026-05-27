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

Dry-run JSON for `convert` includes `operation_diagnostics`, a list of planned operations with `level` set to `exact`, `approximate`, or `metadata_only`. When `--pipeline` is used it also includes `pipeline_advisories`, warning about ordering issues such as decimation before repair, tangents without UV0, AO baking without UV1, and LOD generation before LOD0 optimization.

## Commands

| Command | Description |
|---------|-------------|
| `fascat inspect input.step` | Inspect STEP assembly metadata and planned conversion inputs |
| `fascat convert input.step [output.usdc]` | Convert STEP CAD into OpenUSD, glTF, OBJ, or STL |
| `fascat validate output.usdc` | Validate generated USD, glTF, OBJ, or STL output |
| `fascat help [command]` | Show top-level or command-specific help |
| `fascat version` | Print version and exit |

## Convert flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `realtime-desktop` | Conversion profile: `inspect-only`, `realtime-desktop`, `realtime-web`, `realtime-mobile`, or `virtual-reality` |
| `--pipeline` | unset | TOML pipeline file with named filters and ordered conversion steps |
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
| `--avoid-skinny-triangles` | `false` | Refine long skinny triangles after tessellation |
| `--quality-report` | unset | Write per-part tessellation quality metrics as JSON |
| `--free-edge-report` | `false` | Record free/boundary edge diagnostics and warn when tessellated parts have open boundaries |
| `--reuse-existing-meshes / --retessellate-existing-meshes` | `true` | Reuse imported mesh data or force retessellation from source BREP where available |
| `--heal-brep` | `false` | Run BREP healing before tessellation |
| `--heal-tolerance` | `0.05` | BREP healing tolerance |
| `--remove-sliver-faces` | `false` | Request tiny sliver-face removal during BREP healing; current backend support is limited and reports warnings when unavailable |
| `--max-sliver-area` | `1e-4` | Area threshold for sliver-face reporting |
| `--fail-on-open-shells` | `false` | Fail if healed BREP still contains open shells |
| `--lods` | profile value | Comma-separated LOD ratios, for example `0.5,0.25,0.1` |
| `--lod-mode` | `variants` | LOD output mode: `variants`, `extras`, or `separate` |
| `--lod-screen-coverage` | unset | Screen coverage values for generated LODs |
| `--lod-per-part-budget` | `false` | Apply LOD budgets independently per part |
| `--lod-drop-tiny-parts` | `false` | Omit tiny parts from lower LOD meshes |
| `--lod-tiny-part-screen-size` | `2.0` | Screen-size threshold for tiny-part LOD omission |
| `--validate-lods` | `false` | Validate generated LOD monotonicity |
| `--normals` | `smooth` | Normal generation mode: `none`, `smooth`, `hard-edges`, or `flat` |
| `--preserve-face-boundaries` | `false` | Treat CAD face-group boundaries as hard normal edges |
| `--tangents` | `false` | Ensure glTF-compatible vertex tangents exist; existing tangents are preserved unless invalidated or overridden |
| `--tangent-uv-channel` | `0` | UV channel used when tangent generation or regeneration is needed |
| `--override-tangents / --preserve-tangents` | `false` | Regenerate existing tangents instead of preserving them when `--tangents` is used |
| `--validate-normals` | `false` | Validate staged normals and tangents |
| `--uv0` | `box` | UV0 generation mode: `none`, `box`, `unwrap`, or `lightmap` |
| `--uv1` | `none` | UV1 generation mode: `none`, `box`, `unwrap`, `lightmap`, or `copy-uv0` |
| `--normalize-uvs` | unset | Comma-separated UV channels to rescale into 0..1 after UV generation or copy |
| `--materials` | `cad` | Material staging mode: `cad`, `display`, or `none` |
| `--material-mode` | `cad` | Material normalization mode: `cad` or `pbr` |
| `--merge-equivalent-materials` | `false` | Merge CAD materials with matching PBR values |
| `--texel-density` | unset | UV texel density metadata for unwrap and atlas workflows |
| `--uv-padding` | `2` | UV island padding metadata in pixels |
| `--max-stretch` | unset | Maximum UV stretch metadata for unwrap workflows |
| `--unwrap-method` | `default` | Unwrap solver intent: `default`, `conformal`, or `isometric` |
| `--unwrap-iterations` | unset | Requested unwrap solver iteration budget metadata |
| `--unwrap-tolerance` | unset | Requested unwrap solver tolerance metadata |
| `--atlas` | `false` | Tag materials and UVs for a generated atlas |
| `--atlas-size` | `4096` | Maximum atlas texture size |
| `--metadata` | `full` | Metadata import/export mode: `none`, `summary`, or `full` |
| `--pmi` | `metadata` | PMI import/export mode: `none`, `metadata`, or `metadata-and-visuals` |
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
| `--bake-materials` | `false` | Create a shared baked material with constant embedded texture maps |
| `--maps-resolution` | `2048` | Requested bake texture resolution in pixels, recorded for downstream atlas generation |
| `--force-uv-generation` | `false` | Generate UVs before material bake metadata and textures are recorded |
| `--bake` | `base-color` | Maps to bake as constant embedded textures, such as `base-color,opacity` |
| `--decimate` | `false` | Run explicit decimation before profile optimization |
| `--decimate-criterion` | `target` | Decimation criterion: `target` or `quality` |
| `--surface-tolerance` | unset | Surface deviation tolerance metadata for decimation |
| `--line-tolerance` | unset | Hard-edge deviation tolerance metadata for decimation |
| `--normal-tolerance` | `15` | Normal angle tolerance for decimation preservation |
| `--uv-tolerance` | unset | UV deviation tolerance metadata for decimation |
| `--protect-topology / --no-protect-topology` | `true` | Preserve topology-sensitive faces during decimation |
| `--budget-scope` | `selection` | Decimation budget scope: `part` or `selection` |
| `--uv-importance` | `preserve-islands` | Decimation UV handling: preserve islands, preserve seams, or ignore UVs |
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
| `--quantize` | `false` | Write glTF `KHR_mesh_quantization` accessors and node dequantization transforms |
| `--meshopt` | `false` | Write glTF `EXT_meshopt_compression` bufferView payloads with fallback data |
| `--draco` | `false` | Unsupported until a Draco encoder backend is integrated |
| `--texture-compression` | unset | Unsupported until a KTX2/Basis encoder and texture packaging backend is integrated |
| `--package` | `default` | USD package mode: `default` or packaged `.usdz` |
| `--file-size-budget-mb` | unset | Warn in reports when output exceeds this size |
| `--obj-materials / --no-obj-materials` | `true` | Write OBJ material assignments |
| `--write-mtl / --no-write-mtl` | `true` | Write an OBJ MTL sidecar |
| `--preserve-groups / --no-preserve-groups` | `true` | Preserve OBJ groups per occurrence |
| `--stl-binary / --stl-ascii` | `true` | Write binary STL instead of ASCII STL |
| `--stl-merge / --no-stl-merge` | `true` | Merge STL output into one triangle stream |
| `--debug` | `false` | Require text `.usd` or `.usda` output for debugging |
| `--report` | unset | Write a JSON conversion report sidecar |
| `--force` | `false` | Overwrite an existing output file |

Units and behavior notes:

- Linear tolerances and sizes such as `--sag`, `--min-edge-length`, `--max-edge-length`, `--max-polygon-length`, `--heal-tolerance`, `--max-sliver-area`, `--region-size`, and `--max-hole-diameter` use the source asset's working units unless the option explicitly says otherwise.
- Angles such as `--angle`, `--normal-tolerance`, and `--hard-edge-angle` are degrees.
- Ratios such as `--ratio`, `--lods`, and decimation target ratios are fractions between `0` and `1`; LOD ratios must be sorted from highest to lowest detail.
- Explicit decimation requests that keep less than 20% of source triangles emit an LOD0 distortion warning. Use those aggressive ratios primarily for distant LODs unless visual validation says otherwise.
- Screen coverage values are fractions between `0` and `1`; file-size budgets are megabytes; atlas and bake sizes are pixels.
- `--decimate-criterion quality` maps tolerances to a target ratio, records measured nearest-vertex error and achieved triangle reduction, and reports a warning because tolerance bounds are not enforced.
- `--uv-importance ignore` strips UV/tangent attributes before simplification; `preserve-seams` uses UVs for seam preservation and then strips them; `preserve-islands` keeps UVs through the output.
- `--remove-holes` uses mesh boundary classification and filling when BREP hole removal is unavailable. `--hole-types` filters inferred through, blind, and surface boundary loops; closed BREP feature holes still require a BREP feature backend.
- `--remove-occluded` uses deterministic sampled visibility. Strategy changes the direction set, `--hemi-evaluation` restricts rays to upper-hemisphere and side views, and `--occlusion-level` controls whether fully hidden parts, material groups, or triangles are removed. Output metadata records sample coverage, direction coverage, and an occlusion confidence score.
- `--draco` is rejected until a Draco encoder backend is integrated.

## Inspect flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `inspect-only` | Inspection profile to show in output |
| `--metadata` | `summary` | Metadata output mode: `none`, `summary`, or `full` |
| `--pmi` | `summary` | PMI output mode: `none`, `summary`, `full`, `metadata`, or `metadata-and-visuals` |
| `--heal-brep` | `false` | Run BREP healing before inspection output |
| `--heal-tolerance` | `0.05` | BREP healing tolerance |
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
current Fascat behavior. Use dry-run `operation_diagnostics` and report step
warnings to distinguish exact work from fallbacks.

| Capability | Fascat status | Report or diagnostic | Next step |
|------------|---------------|----------------------|-----------|
| STEP import, hierarchy, names, transforms, colors, metadata | Implemented for STEP | `import` report stats and pipeline import options; AP242 PMI markers warn when typed PMI import is unavailable | Add design variants, typed PMI entity extraction, existing mesh preference, and multi-file import |
| BREP healing | Partial | `heal_brep`; records open shells, free/unstitched edges, small edges, and sliver counts; sliver removal warns that the backend leaves shapes unchanged | Implement sliver-face removal, duplicate-face cleanup, and deeper face/wire repair |
| Tessellation | Implemented | `tessellate` report options, explicit sag-ratio, existing mesh reuse/retessellation controls, max-polygon-length diagnostics, free-edge diagnostics, and quality metadata | Add CAD UV/tangent extraction |
| Mesh repair | Implemented for core cleanup | `repair` report step; mesh metadata records before/after duplicate polygon, degenerate triangle, boundary edge, and non-manifold edge counts | Add T-junction sewing, non-manifold cracking, and configurable orientation strategies |
| Staging, normals, tangents, UV metadata | Partial | `stage` report step; tangents require the selected UV channel when generated; existing tangents are preserved by default and mesh/asset metadata report generated, regenerated, preserved, missing, invalidated, or dropped tangent states; UV metadata records per-channel domain, bounds, validation status, degenerates, overlap counts, UV0-to-UV1 copy status, explicit normalization status, and unwrap solver intent, with warnings for UV1/lightmap bake violations and unsupported solver controls | Add seam planning, backend-enforced unwrap solver controls, island merge, repack, padding controls, and distortion metrics |
| Material baking | Approximate | `bake_materials` emits constant embedded texture maps from material factors and warns that raster baking is not implemented | Generate real atlas textures from source texture/material inputs |
| Hole removal | Approximate | `remove_holes` warns when it falls back to mesh boundary classification and filling | Add BREP feature-level removal for closed cylindrical and pocket holes |
| Occlusion removal | Approximate | `remove_occluded` warns that sampled visibility may require higher precision and records candidate counts, sampled face coverage, direction coverage, and confidence metadata | Add acceleration structures and optional raster/GPU backends for high-poly production scenes |
| Decimation | Partial | `decimate`; quality criterion records measured vertex error but still uses a ratio heuristic, aggressive LOD0 ratios emit distortion warnings, and UV importance modes control seam preservation or UV cleanup | Enforce geometric error bounds and add topology protection metrics, iterative limits, and AO/user-weight importance modes |
| LOD generation | Partial | `run_lod_generators` / `lods` report steps; untessellated parts are skipped with `lod_status="skipped_no_mesh"`, generated/skipped counts, and warnings | Preserve occurrence-level LOD chains and add far-LOD merge plus validation |
| Instance reconstruction | Partial | `optimize_scene` reconstructs exact matching mesh fingerprints when vertex attributes, material assignments, and metadata match; metadata records reconstructed part/occurrence counts and vertex/triangle savings | Add tolerance-based similarity detection and richer file-size savings estimates |
| Runtime compression | Partial | glTF quantization and meshopt are implemented; Draco and texture compression are rejected until real encoders are integrated; glTF write reports list runtime extension dependencies and expected support | Add real KTX2/Basis output and a Draco path only if reliable encoders are integrated |
| Export and budgets | Implemented for USD, USDZ, glTF/GLB, OBJ, STL | `write` report includes glTF runtime dependencies, file size, and optional budget warnings; `profile_budget` reports selected-profile target FPS plus triangle, vertex, per-mesh vertex/index-buffer, texture-resolution, texture-memory, and draw-call budget status | Add geometry/texture/metadata size breakdowns and export cleanup for unused resources |
| PMI metadata export | Partial | glTF `extras.fascat` and USD `customData` include PMI records and resolve links through `source_part_ids` after merge/replace | Add visual annotation geometry for `metadata_and_visuals` |

## Validate flags

| Flag | Default | Description |
|------|---------|-------------|
| `--geometry-quality` | `false` | Enable all geometry quality checks in the validation report |
| `--non-manifold-edges` | `false` | Report non-manifold edge counts |
| `--open-boundaries` | `false` | Report open boundary counts |
| `--self-intersections` | `false` | Report detected self-intersections with bounded triangle-triangle checks and lower-bound fields when the pair limit is hit |
| `--sliver-triangles` | `false` | Report degenerate and sliver triangle stats |
| `--tiny-parts` | `false` | Report tiny part stats |
| `--draw-call-estimate` | `false` | Report material count and draw-call estimate |
| `--visual-risk` | `false` | Report before/after visual risk warnings |
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

Supported output suffixes are `.usd`, `.usda`, `.usdc`, `.usdz`, `.gltf`, `.glb`, `.obj`, and `.stl`.

`--debug` is only valid with `.usd` or `.usda` output. Binary `.usdc`, `.gltf`, and `.glb` output is rejected when debug mode is enabled.

`convert` validates the generated asset before reporting success. If validation fails, the command exits non-zero.

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
