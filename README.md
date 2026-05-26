# 🐱 fascat

Fascat is a Python library and CLI for converting CAD STEP data into realtime-ready OpenUSD assets.

[![release](https://img.shields.io/github/v/release/pavelsimo/fascat?style=flat-square&color=4d9e4d&logoColor=white)](https://github.com/pavelsimo/fascat/releases)
[![license MIT](https://img.shields.io/badge/license-MIT-ffd60a?style=flat-square&logoColor=white)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=flat-square&logoColor=white)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/fascat?style=flat-square&color=3775a9&logoColor=white)](https://pypi.org/project/fascat)
[![Homebrew](https://img.shields.io/badge/Homebrew-b28f62?style=flat-square&logoColor=white)](https://github.com/pavelsimo/homebrew-tap)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-0088cc?style=flat-square&logoColor=white)](https://deepwiki.com/pavelsimo/fascat)

The V1 pipeline is intentionally narrow:

```text
STEP CAD -> imported assembly -> tessellated meshes -> repaired meshes -> staged materials and UVs -> optimized LODs -> OpenUSD
```

## Installation

### Homebrew (macOS / Linux)

```bash
brew tap pavelsimo/homebrew-tap
brew install fascat
```

### pip / pipx

```bash
pipx install fascat
# or
pip install fascat
```

## Quick Start

```bash
# Show help
fascat --help
fascat help convert

# Print version
fascat version

# Inspect a STEP assembly
fascat inspect motor.step
fascat --json inspect motor.step

# Convert STEP to binary OpenUSD
fascat convert motor.step motor.usdc --profile realtime-desktop

# Tune tessellation, UVs, optimization, and LODs
fascat convert motor.step motor.usdc \
  --sag 0.1 \
  --angle 15 \
  --max-edge-length 25 \
  --target-triangles 500000 \
  --uv1 box \
  --lods 0.5,0.25,0.1

# Preview a conversion without writing files
fascat convert motor.step motor.usdc --dry-run

# Emit a debuggable ASCII USD and report
fascat convert motor.step motor.usda --debug --report report.json

# Validate generated USD
fascat validate motor.usdc
```

## Commands

| Command | Description |
|---------|-------------|
| `fascat inspect input.step` | Inspect a STEP assembly before conversion |
| `fascat convert input.step output.usdc` | Convert STEP CAD into OpenUSD |
| `fascat validate output.usdc` | Validate generated USD output |
| `fascat help [command]` | Show top-level or command-specific help |
| `fascat version` | Print version and exit |

Fascat follows standard CLI stream conventions: primary output and JSON go to stdout, while errors and per-stage conversion progress go to stderr. File arguments accept `-` for stdin/stdout where meaningful.

## Python API

```python
import fascat as fc

fc.convert(
    "pump.step",
    "pump.usdc",
    tessellation=fc.Tessellation(sag=0.1, angle=15, max_edge_length=25),
    stage=fc.StageOptions(uv0="box", uv1="unwrap"),
    optimize=fc.OptimizeOptions(target_triangles=1_000_000, preserve_instances=True),
    lods=fc.LODOptions((0.5, 0.25, 0.1)),
)
```

## Docs

Full documentation at **[pavelsimo.github.io/fascat](https://pavelsimo.github.io/fascat)**.

## License

MIT
