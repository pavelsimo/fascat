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

## Commands

| Command | Description |
|---------|-------------|
| `fascat inspect input.step` | Inspect STEP assembly metadata and planned conversion inputs |
| `fascat convert input.step [output.usdc]` | Convert STEP CAD into OpenUSD or glTF |
| `fascat validate output.usdc` | Validate generated USD or glTF output |
| `fascat help [command]` | Show top-level or command-specific help |
| `fascat version` | Print version and exit |

## Convert flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `realtime-desktop` | Conversion profile: `inspect-only`, `realtime-desktop`, `realtime-web`, or `virtual-reality` |
| `--sag` | profile value | CAD tessellation sag tolerance |
| `--angle` | profile value | CAD tessellation angle tolerance in degrees |
| `--target-triangles` | profile value | Target triangle count for optimized LOD0 |
| `--ratio` | unset | Simplification ratio when no triangle target is set |
| `--min-edge-length` | unset | Collapse tessellated edges shorter than this length |
| `--max-edge-length` | profile value | Split tessellated triangles longer than this length |
| `--preserve-boundaries / --no-preserve-boundaries` | `true` | Preserve sharp/boundary edges during tessellation cleanup |
| `--curvature-adaptive` | `false` | Use tighter interior meshing on curved CAD faces |
| `--avoid-skinny-triangles` | `false` | Refine long skinny triangles after tessellation |
| `--quality-report` | unset | Write per-part tessellation quality metrics as JSON |
| `--heal-brep` | `false` | Run BREP healing before tessellation |
| `--heal-tolerance` | `0.05` | BREP healing tolerance |
| `--remove-sliver-faces` | `false` | Detect tiny sliver faces during BREP healing |
| `--max-sliver-area` | `1e-4` | Area threshold for sliver-face reporting |
| `--fail-on-open-shells` | `false` | Fail if healed BREP still contains open shells |
| `--lods` | profile value | Comma-separated LOD ratios, for example `0.5,0.25,0.1` |
| `--normals` | `smooth` | Normal generation mode: `none`, `smooth`, `hard-edges`, or `flat` |
| `--preserve-face-boundaries` | `false` | Treat CAD face-group boundaries as hard normal edges |
| `--tangents` | `false` | Generate glTF-compatible vertex tangents from UV0 |
| `--validate-normals` | `false` | Validate staged normals and tangents |
| `--uv0` | `box` | UV0 generation mode: `none`, `box`, `unwrap`, or `lightmap` |
| `--uv1` | `none` | UV1 generation mode: `none`, `box`, `unwrap`, or `lightmap` |
| `--materials` | `cad` | Material staging mode: `cad`, `display`, or `none` |
| `--material-mode` | `cad` | Material normalization mode: `cad` or `pbr` |
| `--merge-equivalent-materials` | `false` | Merge CAD materials with matching PBR values |
| `--texel-density` | unset | UV texel density metadata for unwrap and atlas workflows |
| `--uv-padding` | `2` | UV island padding metadata in pixels |
| `--max-stretch` | unset | Maximum UV stretch metadata for unwrap workflows |
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
| `--batch-by-material` | `false` | Batch compatible scene geometry by material |
| `--merge-compatible-meshes` | `false` | Merge compatible scene meshes to reduce draw calls |
| `--split-large-meshes` | `false` | Split scene-optimized meshes above the vertex limit |
| `--index-buffer` | `auto` | Index buffer mode: `auto`, `uint16`, or `uint32` |
| `--flatten` | `safe` | Hierarchy flattening mode: `none`, `safe`, or `all` |
| `--instance-policy` | `auto` | Instance policy: `auto`, `preserve`, or `expand` |
| `--preserve-instances / --no-preserve-instances` | `true` | Preserve repeated parts as shared instances, or duplicate per occurrence |
| `--preserve-hard-edges` | `false` | Protect faces adjacent to hard edges during simplification |
| `--hard-edge-angle` | `30` | Angle threshold for hard-edge preservation |
| `--preserve-holes` | `false` | Protect open boundary faces during simplification |
| `--preserve-material-boundaries` | `false` | Protect faces along material boundaries |
| `--preserve-uv-seams` | `false` | Protect faces touching duplicated-position UV seams |
| `--preserve-small-parts` | `false` | Skip simplification for small parts |
| `--small-part-triangle-threshold` | `64` | Triangle threshold for `--preserve-small-parts` |
| `--preserve-silhouette` | `false` | Protect faces on bounding-box silhouette extremes |
| `--debug` | `false` | Require text `.usd` or `.usda` output for debugging |
| `--report` | unset | Write a JSON conversion report sidecar |
| `--force` | `false` | Overwrite an existing output file |

## Inspect flags

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` | `inspect-only` | Inspection profile to show in output |
| `--metadata` | `summary` | Metadata output mode: `none`, `summary`, or `full` |
| `--pmi` | `summary` | PMI output mode: `none`, `summary`, `full`, `metadata`, or `metadata-and-visuals` |
| `--heal-brep` | `false` | Run BREP healing before inspection output |
| `--heal-tolerance` | `0.05` | BREP healing tolerance |
| `--remove-sliver-faces` | `false` | Detect tiny sliver faces during BREP healing |
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

## File arguments

Use `-` for standard streams:

```bash
cat input.step | fascat inspect -
cat input.step | fascat convert - - --profile realtime-web
cat output.usdc | fascat validate -
```

When the convert output argument is omitted for a file input, Fascat writes beside the input with a `.usdc` suffix. Stdin input requires an explicit output path or `-`.

When output is `-`, USD bytes are reserved for stdout and progress/errors stay on stderr.

Supported output suffixes are `.usd`, `.usda`, `.usdc`, `.gltf`, and `.glb`.

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
