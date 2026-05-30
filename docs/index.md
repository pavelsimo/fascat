---
title: fascat
description: convert CAD data into realtime-ready OpenUSD and glTF assets
---

Fascat converts STEP, IGES, and OpenCASCADE BREP assets into realtime-ready OpenUSD and glTF assets while preserving hierarchy, materials, transforms, and repeated parts where the source format exposes them.

## Why fascat

- **Focused** — V1 supports CAD import through STEP, IGES, and native BREP, with OpenUSD or glTF output
- **Scriptable** — command results can be emitted as JSON for pipeline use
- **Inspectable** — conversion reports expose options, warnings, timings, and mesh statistics
- **Conservative** — lossy operations are explicit and controlled by profiles or flags

## Quick links

- [Install](install.html) — Homebrew, pip, pipx, or PyPI
- [Quick Start](quickstart.html) — common patterns in 60 seconds
- [Python API](api.html) — fluent assets, profiles, reports, validation, and export
- [Format Guidelines](format-guidelines.html) — what to ask for and what to export
- [Reference](reference.html) — global flags, env vars, exit codes, completions
