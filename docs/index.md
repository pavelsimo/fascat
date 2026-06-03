---
title: fascat
description: convert CAD data into realtime-ready OpenUSD, glTF, OBJ, STL, and FBX assets
---

Fascat converts STEP, IGES, and OpenCASCADE BREP files into realtime-ready
OpenUSD, glTF, OBJ, STL, and FBX assets. It preserves hierarchy, materials,
transforms, and repeated parts wherever the source format exposes them.

```bash
fascat convert motor.step motor.glb --profile realtime-web
```

## Why fascat

- **Focused** — CAD import from STEP, IGES, and native BREP; output to OpenUSD, glTF, OBJ, STL, or FBX
- **Scriptable** — every command can emit JSON for pipeline use
- **Inspectable** — conversion reports expose options, warnings, timings, and mesh statistics
- **Conservative** — lossy operations are explicit and controlled by profiles or flags

## Which output format?

| Output | Use it for |
|--------|------------|
| **OpenUSD** (`.usd`, `.usdc`, `.usda`, `.usdz`) | Highest-fidelity delivery, scene composition, and LOD variants |
| **glTF** (`.gltf`, `.glb`) | Web, mobile, and XR runtime delivery — the usual realtime target |
| **OBJ** / **STL** | Mesh-only interchange, manufacturing, or inspection |
| **FBX** | Handoff to DCC tools and engines |

## Quick links

- [Install](install.html) — Homebrew, pip, pipx, or PyPI
- [Quick Start](quickstart.html) — the common commands in 60 seconds
- [Python API](api.html) — fluent assets, profiles, reports, validation, and export
- [CLI Reference](reference.html) — global flags, env vars, exit codes, completions
