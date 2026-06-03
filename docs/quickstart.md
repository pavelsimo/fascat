---
title: Quick Start
description: Get productive with fascat in 60 seconds
---

## The three commands

Almost everything you do with fascat is one of these:

```bash
fascat inspect motor.step              # look at a CAD assembly
fascat convert motor.step motor.glb    # convert it to a runtime asset
fascat validate motor.glb              # check the result
```

If you omit the output path, `convert` writes a `.usdc` file beside the input.
The output suffix selects the format (`.usdc`, `.glb`, `.obj`, `.stl`, `.fbx`, …).
Not sure which to pick? See [Which output format?](index.html#which-output-format).

## Pick a target profile

Profiles set sensible tessellation, optimization, and LOD defaults for a platform:

```bash
fascat convert motor.step motor.usdc --profile realtime-desktop
fascat convert motor.step motor.glb  --profile realtime-web
fascat convert motor.step motor.glb  --profile realtime-mobile
fascat convert motor.step motor.glb  --profile virtual-reality
fascat convert motor.step motor.glb  --profile mixed-reality
```

See the [profiles table](api.html#profiles) for each profile's budgets.

## Inspect before converting

```bash
fascat inspect motor.step
fascat --json inspect motor.step          # machine-readable
cat motor.step | fascat inspect -          # read from stdin

fascat inspect motor.step --filter 'path=*/Fasteners/*' --pmi summary
```

## Scope work to part of the assembly

Use `--filter` to apply optimization and LOD work to one branch only:

```bash
fascat convert motor.step motor.glb \
  --filter 'path=*/Fasteners/*' \
  --merge --merge-mode by-material \
  --target-triangles 80000
```

## Tune tessellation

```bash
fascat convert motor.step motor.usdc \
  --sag 0.1 --angle 15 \
  --curvature-adaptive \
  --target-triangles 500000
```

## Prepare for runtime delivery

```bash
fascat convert motor.step motor.glb \
  --export-preset web --quantize --meshopt \
  --file-size-budget-mb 50
```

## Validate output

```bash
fascat validate motor.glb
fascat validate motor.glb --geometry-quality --report quality-report.json
fascat validate motor.glb --visual-preview preview.png
```

## Going further

- **Combine many steps** — long flag lists work, but for repeatable
  branch-specific recipes use a pipeline file instead. See
  [Pipeline files](reference.html#pipeline-files).
- **Every flag** is documented in the [CLI Reference](reference.html).
- **Scripting from Python** — the same pipeline is available as a fluent API.
  See the [Python API](api.html).

A pipeline file keeps named filters and ordered steps in TOML:

```toml
[[filters]]
name = "fasteners"
path = "*/Fasteners/*"

[[steps]]
op = "merge"
where = "fasteners"
mode = "by_material"
```

```bash
fascat convert motor.step motor.glb --pipeline realtime.toml
```

## Useful global behavior

```bash
fascat --json <subcommand>           # JSON output, pipe to jq
fascat <subcommand> --dry-run        # preview without writing
fascat --install-completion bash     # shell completion (bash/zsh/fish)
fascat --help                        # help works anywhere in the command
```
