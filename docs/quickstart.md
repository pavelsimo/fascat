---
title: Quick Start
description: Get productive with fascat in 60 seconds
---

## Show help

```bash
fascat --help
fascat help convert
```

## Check version

```bash
fascat version
```

## Inspect STEP input

```bash
fascat inspect motor.step
fascat --json inspect motor.step
cat motor.step | fascat inspect -
```

## Convert to OpenUSD or glTF

```bash
fascat convert motor.step
fascat convert motor.step motor.usdc --profile realtime-desktop
fascat convert motor.step motor.glb --profile virtual-reality
fascat convert motor.step motor.glb --profile realtime-mobile
fascat convert motor.step motor.usda --debug --report report.json
fascat convert motor.step - --dry-run
```

## Scope by assembly filter

```bash
fascat inspect motor.step \
  --filter 'path=*/Fasteners/*' \
  --filter 'name=Bolt*' \
  --metadata full \
  --pmi summary \
  --json

fascat convert motor.step motor.glb \
  --filter 'path=*/Fasteners/*' \
  --merge \
  --merge-mode by-material \
  --explode connected-components \
  --replace bounding-box \
  --batch-by-material \
  --merge-compatible-meshes \
  --split-large-meshes \
  --bake-materials \
  --bake base-color,opacity \
  --decimate \
  --remove-holes \
  --max-hole-diameter 3.0 \
  --remove-occluded \
  --run-lod-generators \
  --lod-preset vr \
  --target-triangles 80000 \
  --report report.json
```

For branch-specific conversion settings, put named filters and ordered steps in a pipeline file:

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
names = ["Bolt*"]

[[steps]]
op = "merge"
where = "fasteners"
mode = "by_material"
```

```bash
fascat convert motor.step motor.glb --pipeline realtime.toml
```

## Tune tessellation and LODs

```bash
fascat convert input.step output.usdc \
  --heal-brep \
  --heal-tolerance 0.05 \
  --sag 0.1 \
  --angle 15 \
  --min-edge-length 0.02 \
  --max-edge-length 25 \
  --curvature-adaptive \
  --quality-report tessellation-quality.json \
  --target-triangles 500000 \
  --preserve-hard-edges \
  --preserve-material-boundaries \
  --preserve-small-parts \
  --normals hard-edges \
  --tangents \
  --validate-normals \
  --materials cad \
  --material-mode pbr \
  --merge-equivalent-materials \
  --uv1 lightmap \
  --texel-density 256 \
  --uv-padding 4 \
  --atlas \
  --atlas-size 4096 \
  --lods 0.5,0.25,0.1 \
  --lod-mode variants \
  --lod-screen-coverage 0.5,0.2,0.05 \
  --lod-per-part-budget \
  --lod-drop-tiny-parts \
  --lod-tiny-part-screen-size 2 \
  --validate-lods
```

Use `.usda` or `.usd` with `--debug` when you want inspectable text output. Binary `.usdc`, `.gltf`, and `.glb` output is rejected in debug mode.

## Runtime export handoff

```bash
fascat convert input.step output.glb \
  --quantize \
  --meshopt \
  --file-size-budget-mb 50

fascat convert input.step output.obj --obj-materials --write-mtl --force
fascat convert input.step output.stl --stl-binary --force
```

## Validate output

```bash
fascat validate output.usdc
fascat validate output.glb
fascat validate output.glb --geometry-quality --report quality-report.json
```

## JSON output

Every command supports `--json` for scripting:

```bash
fascat --json <subcommand> | jq '.'
fascat <subcommand> --json | jq '.'
```

## Dry run

Preview changes before applying them:

```bash
fascat --dry-run <subcommand>
fascat <subcommand> --dry-run
```

## Shell completions

```bash
# bash
fascat --install-completion bash

# zsh
fascat --install-completion zsh

# fish
fascat --install-completion fish
```
