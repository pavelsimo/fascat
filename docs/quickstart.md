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
fascat convert motor.step motor.usda --debug --report report.json
fascat convert motor.step - --dry-run
```

## Scope by assembly filter

```bash
fascat inspect motor.step \
  --filter 'path=*/Fasteners/*' \
  --filter 'name=Bolt*' \
  --json

fascat convert motor.step motor.glb \
  --filter 'path=*/Fasteners/*' \
  --target-triangles 80000 \
  --report report.json
```

## Tune tessellation and LODs

```bash
fascat convert input.step output.usdc \
  --sag 0.1 \
  --angle 15 \
  --max-edge-length 25 \
  --target-triangles 500000 \
  --materials display \
  --uv1 box \
  --lods 0.5,0.25,0.1
```

Use `.usda` or `.usd` with `--debug` when you want inspectable text output. Binary `.usdc`, `.gltf`, and `.glb` output is rejected in debug mode.

## Validate output

```bash
fascat validate output.usdc
fascat validate output.glb
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
