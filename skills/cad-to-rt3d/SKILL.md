---
name: cad-to-rt3d
description: Convert a CAD file (STEP/IGES/BREP/JT) into a real-time 3D asset (glTF/GLB/USD) that passes geometry-quality and visual-fidelity gates for a target profile. Runs an iterative fascat convert/validate loop, inspecting turntable renders against a high-quality reference and adjusting conversion flags until the result looks good or the iteration budget is exhausted.
---

# CAD → RT3D conversion loop

Automate `fascat` conversion tuning: convert, validate, render the result from
many angles, compare against a high-fidelity reference, fix the biggest problem,
repeat. Stop when all gates pass or after 5 iterations.

## Inputs

- **CAD file** (required): `.step`/`.stp`, `.igs`/`.iges`, `.brep`, or `.jt`.
- **Profile** (default `realtime-web`): one of `realtime-desktop`, `realtime-web`,
  `realtime-mobile`, `virtual-reality`, `augmented-reality`, `mixed-reality`.
- **Output path** (default `<input stem>.glb`). Any fascat-supported export
  format works; `.glb` is best for runtime targets.

Run every fascat command with the global `--json` flag and redirect stdout to a
file in the work directory — the JSON payloads are the data the loop runs on.
Diagnostics go to stderr, so still check stderr for warnings.

## Artifacts layout

Create a work directory next to the output and keep every iteration:

```
<output>.fascat-work/
├── inspect.json                  # fascat --json inspect
├── reference/
│   ├── reference.glb             # high-fidelity conversion (visual ground truth)
│   ├── validate.json
│   └── turntable/                # reference turntable renders
├── iter-1/ … iter-N/
│   ├── convert.json              # fascat --json convert stdout
│   ├── report.json               # fascat convert --report (budgets, warnings)
│   ├── validate.json             # fascat --json validate stdout (incl. gate results)
│   └── turntable/                # this iteration's renders + diff vs reference
└── summary.md                    # final report
```

## Step 1 — Inspect

```bash
fascat --json inspect input.step > work/inspect.json
```

Note the part count, materials, units, and bounding-box size. Derive the
reference tessellation sag from the model scale: `sag ≈ bbox_diagonal / 2000`,
clamped to something sane for the units (e.g. 0.01–0.5 mm for millimetre
models). If inspect reports open shells or healing warnings, plan to add
`--heal-brep` from the first conversion onward.

## Step 2 — Build the visual reference

Convert once at high fidelity — tight sag, curvature-adaptive, **no**
decimation, LODs, or compression — and render its turntable. This is the ground
truth every iteration is compared against:

```bash
fascat --json convert input.step work/reference/reference.glb \
  --sag <reference_sag> --angle 10 --curvature-adaptive \
  > work/reference/convert.json

fascat --json validate work/reference/reference.glb \
  --geometry-quality \
  --turntable-dir work/reference/turntable \
  --turntable-views 8 --turntable-elevations -30,30 \
  > work/reference/validate.json
```

**Never gate against a broken reference.** If the reference itself reports
non-manifold edges or open boundaries, re-convert it with `--heal-brep`
(raise `--heal-tolerance` if needed) before proceeding. Read the reference
contact sheet (`work/reference/turntable/turntable.png`) once to confirm the
model renders sensibly.

## Step 3 — Iterate (max 5)

Start iteration 1 with just the profile defaults (plus `--heal-brep` if Step 1
called for it). For each iteration N:

```bash
fascat --json convert input.step output.glb \
  --profile realtime-web [adjusted flags] \
  --report work/iter-N/report.json \
  > work/iter-N/convert.json

fascat --json validate output.glb \
  --strict-geometry --profile realtime-web \
  --turntable-dir work/iter-N/turntable \
  --turntable-views 8 --turntable-elevations -30,30 \
  --turntable-baseline-dir work/reference/turntable \
  --visual-diff-pixel-tolerance 8 \
  --visual-diff-mean-threshold 3.0 \
  --visual-diff-changed-pixel-ratio 0.05 \
  > work/iter-N/validate.json
```

Validate evaluates every gate itself and exits 1 when any gate fails — that is
a gate result, not an error; still read the JSON. The `gates` object in
validate.json is the authoritative verdict (`overall`, `failed`, `evaluated`,
and one entry per gate in `results`); the same PASS/FAIL/SKIP lines appear on
stdout without `--json`. Add `--max-file-size-mb <budget>` when a file-size
budget is known, and relax an individual geometry gate with an explicit
`--max-*` flag (for example `--max-slivers 12`) when the source geometry
forces it — explicit flags override `--strict-geometry` and `--profile`.

**Visual inspection is mandatory every iteration**, even when all gates pass.
Read `work/iter-N/turntable/turntable.png` and compare the 2–3 views with the
worst diff metrics (see `turntable.diff.worst_view` in validate.json) side by
side with the same-named reference images. Look for:

- missing parts or holes that weren't in the reference
- faceted silhouettes on curved surfaces
- shading seams or hard edges across smooth regions
- z-fighting stripes on coplanar surfaces
- collapsed or paper-thin features from over-decimation

## Gates

All deterministic gates are evaluated by `fascat validate` itself; the results
live in the `gates` object of validate.json.

| Gate | Threshold | Set by |
|------|-----------|--------|
| `non_manifold_edges` | 0 | `--strict-geometry` |
| `self_intersections` | 0 | `--strict-geometry` |
| `open_boundaries` | 0 | `--strict-geometry` |
| `sliver_triangles` | 0 (relax with an explicit `--max-slivers <count>` if the source geometry forces them) | `--strict-geometry` |
| `triangles` | profile triangle budget | `--profile <name>` (override with `--max-triangles`) |
| `file_size_bytes` | file-size budget in MiB | `--max-file-size-mb <budget>` (SKIP when unset) |
| `turntable_diff` | all views pass | `--turntable-baseline-dir` |
| Visual read | no artifacts from the list above | your eyes on the renders |

## Troubleshooting: symptom → flags

Change **1–2 flags per iteration**, targeting the biggest failing gate first,
so you can attribute improvements. Record what changed and why in summary.md.

| Symptom | Fix |
|---------|-----|
| Faceted silhouettes, polygonal curves | Lower `--sag`, add `--curvature-adaptive`, tighten `--angle` |
| Cracks, open boundaries, non-manifold edges | `--heal-brep` (+ raise `--heal-tolerance`), `--group-open-shells` |
| Sliver triangles | `--avoid-skinny-triangles`, `--remove-sliver-faces`, `--merge-vertices --delete-merge-vertex-degenerate` |
| Over triangle budget | `--decimate --target-triangles <budget>`; or coarser `--sag` if fidelity allows |
| Decimation ate small features | Raise `--target-triangles`, add `--protect-topology`, use `--budget-scope part` |
| Z-fighting stripes | `--cleanup-overlapping-faces` (tune `--overlap-area-ratio`) |
| Too many draw calls | `--merge-compatible-meshes`, `--batch-by-material`, `--merge --merge-mode by-material` |
| File too big (geometry) | `--draco` or `--meshopt`, `--quantize` |
| File too big (textures) | `--texture-compression ktx2`, lower `--maps-resolution` |
| Hard shading on smooth surfaces | `--normals smooth`, tune `--normal-weighting angle` |
| Tiny interior junk parts | `--remove-occluded`, or `--exclude-filter` on known-hidden paths |
| Needs LODs (VR/mobile profiles) | `--lods 0.5,0.25,0.1`, check `lod_preview.monotonic_triangles` |

## Stop conditions and summary

Stop when every gate passes, or after 5 iterations. If iterations run out, keep
the **best** iteration's output (fewest failed gates; break ties by better
turntable diff), copy it to the requested output path, and say plainly which
gates still fail.

Write `work/summary.md` with: the flags used per iteration, each iteration's
gate results, the final metrics (triangles, file size, draw calls, diff
metrics), and paths to the final turntable renders. Report the same summary in
your final message, including explicit residual failures if any.

## Performance notes

The turntable renderer is CPU software rasterization — cost scales with
triangle count × views × resolution. For multi-million-triangle assets, use
`--turntable-views 4 --turntable-supersample 1 --turntable-width 256
--turntable-height 256` during early iterations and run the full 8-view grid
only for the final check. Keep the reference turntable settings identical to
the iteration settings — baseline diffing matches views by filename and image
size.
