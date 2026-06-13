# Fascat Plan

The single planning document for Fascat. For the log of shipped work, see
[CHANGELOG.md](../CHANGELOG.md). This file tracks **what blocks production readiness**
and **what is still open**, as prioritized TODO checklists.

Provenance: full codebase audit on 2026-06-12 (every subsystem read end-to-end; every
bug claim adversarially re-verified before inclusion). The Unity Asset Transformer
guidelines are used as a reference checklist — matching them 100% is explicitly a
non-goal. Shapely is the bar for the Python API.

Status: every P0 and P1 item from that audit shipped in **0.4.0** (2026-06-13),
including the two items corrected during implementation (budget-warning surfacing
and forbid-overlapping failure semantics — see the inline notes). Open work below
is P2/P3 and the carried-forward roadmap.

## What Fascat Is

A Python library and CLI that converts CAD (STEP, IGES, BREP) into realtime-ready OpenUSD,
glTF/GLB, OBJ, STL, and FBX assets. The end-to-end V1 pipeline is implemented and produces
real geometry. The goal is to be the best production **CAD → realtime 3D (RT3D)** converter:
correct by default, safe on untrusted input, fast on large assemblies, with a small,
beautiful, Pythonic API.

Backends: OCCT/OCP (CAD + tessellation + BREP healing), xatlas (UV unwrap/packing),
meshoptimizer / fast-simplification (decimation + compression), Pillow (texture baking),
glTF Transform + KTX2 encoders (Draco and KTX2/Basis), usd-core (USD), built-in
glTF/OBJ/STL/FBX writers, trimesh + numpy (mesh ops).

## How to read this plan

- States: `[ ]` open · `[~]` partial · `[x]` done.
- Priorities: **P0** correctness/security blocker · **P1** production requirement ·
  **P2** quality/competitive · **P3** later.
- Every code claim cites `file:line` as of the audit commit; lines drift as code changes.

## 1. Correctness bugs

### Import (STEP/IGES/BREP)

- [x] **P0** Path traversal in texture/material-library resolution — `_resolve_material_library_reference` / `_resolve_source_texture` don't confine resolved paths to the search roots (`fascat/io/step.py:5315`, `fascat/io/step.py:6215`); validate with `Path.is_relative_to()` after resolving
- [x] **P1** ISO-10303-21 string escape directives (`\X2\`, `\X4\`, `\S\`) are not decoded → garbled part names and PMI text on standards-conformant files (`fascat/io/step.py:3984`)
- [ ] **P2** Color range heuristic (any component > 1.0 ⇒ assume 0–255) misreads HDR-ish inputs; make the color space explicit (`fascat/io/step.py:5694`)
- [ ] **P2** Texture slot matching can bind the same slot twice — dedupe by slot (`fascat/io/step.py:5753`)
- [ ] **P2** Mirrored transforms (negative determinant) are neither detected nor documented; risk of inverted normals on mirrored instances (`fascat/io/step.py:1476`, `fascat/io/step.py:4551`)
- [ ] **P3** Stale "multi-file import not implemented" warning contradicts the shipped external-reference graph (`fascat/io/step.py:4087`)

### Mesh repair / tessellation

- [x] **P0** `fill_holes` assigns `material_indices[0]` to every fill face — wrong materials on filled regions (`fascat/mesh.py:2883`); assign per-hole from neighboring faces (nearest centroid)
- [x] **P0** `stitch_boundary_gaps` wipes normals, tangents, and ALL UV channels (`fascat/mesh.py:1384-1386`); normals regenerate downstream but UVs are unrecoverable — interpolate attributes across stitched vertices instead
- [x] **P1** Silent `return` when post-edit attribute remapping is incomplete leaves stale `material_indices` (`fascat/mesh.py:3047`); warn or invalidate the channel
- [x] **P1** Fixed `area_epsilon=1e-12` in degenerate-face removal is scale-dependent — wrong for meter- or micron-scale models (`fascat/mesh.py:939`); derive from bbox diagonal
- [ ] **P2** `simplify` with `target_error`: bound violations are recorded as `exceeded` in metadata but never enforced; document target_error as a hint or add a retry fallback (`fascat/mesh.py:2354-2412`)

### Stage / materials / textures

Audit verdict: healthy. xatlas vertex-split bookkeeping, opacity math, and bake color-space
handling were each independently challenged and held up.

- [ ] **P2** Warn in `asset.report` when textures are silently downsampled by `max_resolution` (`fascat/ops/textures.py:79`)
- [ ] **P3** Document the 6-decimal PBR rounding in material dedup keys, or make precision configurable (`fascat/ops/stage.py:857`)
- [ ] **P3** Mean-based PBR averaging in baked materials: document, or use a weighted strategy (`fascat/ops/actions.py:882`)
- [ ] **P3** Expose/adapt the fixed 6-direction AO sampling (`fascat/ops/actions.py:736`)
- [ ] **P3** Record whether baked emissive came from the material or the (0,0,0) fallback (`fascat/ops/actions.py:704`)

### Exporters (glTF / USD / OBJ / STL / FBX)

- [x] **P0** Draco/KTX2 export leaves a corrupt file on disk when validation fails after `_write_gltf_with_external_compression` (`fascat/io/gltf.py:189-190`); write to a temp file + atomic rename on success — and make **all** exporters transactional
- [x] **P1** A mesh whose face groups are all empty can emit invalid empty-primitive glTF (`fascat/io/gltf.py:1261-1284`)
- [x] **P1** Out-of-bounds material index silently drops the material binding (`fascat/io/gltf.py:1577-1586`); add a report warning
- [x] **P1** KTX2 encoder subprocess has no timeout — export can hang forever (`fascat/io/gltf.py:1353-1375`)
- [ ] **P2** Empty accessors get dummy `[0,0,0]` min/max — skip empty meshes or omit the accessor (`fascat/io/gltf.py:1175`, `fascat/io/gltf.py:1185`)
- [ ] **P2** Draco-compressed output is not re-validated before success is reported (`fascat/io/gltf.py:1316-1338`)
- [ ] **P2** Quantized SNORM8 NORMAL accessors are not covered by the KHR_mesh_quantization validation pass (`fascat/io/gltf.py:1200`, validator `fascat/io/gltf.py:2089`)
- [x] **P2** USD `display_color` metadata parse failures fall back silently (`fascat/io/usd.py:512-525`)
- [x] **P2** USD UV conversion lacks (N,2) shape validation (`fascat/io/usd.py:504`)
- [ ] **P3** Warn when ASCII STL is chosen for >10k-triangle meshes (`fascat/io/stl.py:17`)
- [ ] **P3** USD name sanitization collision policy documented/enforced (`fascat/io/usd.py:681`)

### Optimize / LOD

- [ ] **P2** LOD re-simplification retry doesn't update `simplification_source` metadata — misleading provenance (`fascat/ops/lod.py:229-231`)
- [ ] **P3** LOD ratio ≤ 0.01 produces a 1-triangle LOD with no warning; validate ratios (`fascat/ops/lod.py:631`, `fascat/ops/lod.py:553`)
- [ ] **P3** `_flatten_safe` near-identity check relies on default `np.allclose` tolerances; make the tolerance explicit (`fascat/ops/scene.py:312`)
- [ ] **P3** Merge without a `max_vertices_per_mesh` default can exceed engine index limits; default 65536 + warning (`fascat/ops/hierarchy.py:298`)

### Runtime / validation harnesses

- [x] **P0** Subprocess timeouts don't kill process groups — orphaned Chromium/engine children on Unix (`fascat/runtime.py:289`, `fascat/runtime.py:383`, `fascat/runtime.py:451`, `fascat/runtime.py:556`); launch with a new session/process group and kill the group on timeout
- [x] **P0** `TemporaryDirectory` is deleted while a timed-out subprocess may still read from it (`fascat/runtime.py:322-480`, `fascat/runtime.py:537-600`); extend the directory lifetime past subprocess teardown
- [ ] **P2** glTF-Transform invocations lack preflight `shutil.which` checks and contextual errors (`fascat/runtime.py:1058`, `fascat/runtime.py:1131`)
- [ ] **P2** Default visual-diff thresholds are too permissive for regression gating (pixel tolerance 8/255, 35% changed-pixel ratio) (`fascat/runtime_fixtures.py:245`)
- [ ] **P3** Pillow-based preview rendering is platform-dependent — soften the "deterministic" claim or pin/render differently (`fascat/visual.py:382`)
- [ ] **P3** Cap browser screenshot data-URI payload size (`fascat/runtime.py:2226`)

## 2. Robustness & input hardening

- [x] **P1** Whole STEP file loaded via `read_text()` at ≥6 call sites with no size guard — 1 GB file ⇒ 1 GB+ allocation (`fascat/io/step.py:942` et al.); stream or mmap with size limits
- [x] **P1** Unterminated STEP string ⇒ unbounded forward scan per record lookup (`fascat/io/step.py:3956`); bound the scan distance
- [x] **P1** Malformed-input behavior pass: truncated/garbage STEP, empty assemblies, zero-vertex parts must produce clean errors, never tracebacks (tests in §7)
- [ ] **P2** ZIP material libraries: entry-count/size caps before extraction (`fascat/io/step.py:5424`)
- [ ] **P2** JSON material library recursion depth cap (`fascat/io/step.py:5537`)
- [ ] **P2** KTX2 dimension parsing must handle truncated files (`fascat/io/step.py:6291`)
- [ ] **P2** Report broken external-reference cycles instead of silently dropping them (`fascat/io/step.py:873`)
- [ ] **P3** PMI semantic graph cycle detection (`fascat/io/step.py:1897`); domain checks for SQRT/LOG condition operands (`fascat/io/step.py:2797`)

## 3. Performance

- [x] **P1** Parallelism is effectively off: every op defaults `jobs=1`, and `parallel_map` uses `ThreadPoolExecutor` for CPU-bound (GIL-bound) mesh work (`fascat/ops/parallel.py:11-24`); default `jobs=min(4, cpu_count())` and use process pools for tessellate/simplify-class stages
- [ ] **P2** Cache topology metrics: `orientability_metrics` (O(F) BFS, `fascat/mesh.py:1516`) is recomputed via `stats()` after every stage whenever a progress callback is set — and the CLI always sets one (`fascat/pipeline.py:134+`, `fascat/asset.py:368`)
- [ ] **P2** Cache `Node.walk()` results per operation scope — repeated full-tree walks in stats/draw-call/occurrence paths (`fascat/asset.py:100`, `fascat/asset.py:335`)
- [ ] **P2** `read_step_many` doesn't dedupe identical parts across member files — shared library parts are tessellated and stored N times (`fascat/io/step.py:809`)
- [ ] **P2** Memoize design-variant selector term resolution (`fascat/io/step.py:2548`)
- [ ] **P2** Audit `Asset.__post_init__` deep-copy amplification: the constructor deep-copies root/parts/materials/images (`fascat/asset.py:205-212`); verify all hot paths use `_adopt` and document the contract
- [ ] **P3** Cache the nearest-centroid KD-tree across repeated simplifications of the same source (`fascat/mesh.py:2098`)
- [ ] **P3** Cache per-channel UV layout/seam-graph stats in `_tag_uv_layout_quality` (`fascat/ops/stage.py:620-641`)
- [ ] **P3** Restrict the instance-reconstruction second tree walk to nodes referencing replaced parts (`fascat/ops/scene.py:154`)
- [ ] **P3** Early-exit `orientability_metrics` for manifestly non-manifold meshes (`fascat/mesh.py:1516`)
- [ ] **P3** Profile on real large CAD corpora (10k+ parts) and record numbers in [PERFORMANCE.md](PERFORMANCE.md)

## 4. Python API redesign — 0.4.0, breaking (Shapely-inspired)

Decision: break the 0.3.0 API freely; no deprecation shims; docs and tests updated in the
same release. Target feel:

```python
import fascat as fc

asset = fc.read("motor.step")                       # sniffs format; read_step/read_iges/read_brep stay
asset = (asset
    .tessellate(sag=0.1, angle=15)                  # kwargs, not Options ceremony
    .repair(tolerance=0.05)
    .stage(materials="cad", uv0="box")
    .optimize(target_triangles=500_000)
    .lods([0.5, 0.25, 0.1]))
asset.write("motor.glb")                            # extension-dispatched; write_usd/write_gltf stay
asset                                                # <Asset: 412 parts, 1.2M triangles, 38 materials>
```

- [x] **P1** kwargs-first operations: every `Asset` method accepts keyword args directly (`asset.tessellate(sag=0.1)`); the `*Options` dataclasses remain as the typed backing store and for power users (`fascat/options.py`, `fascat/asset.py`)
- [x] **P1** Shrink the top-level namespace from 123 exports to ~40 core names: `Asset`/`Node`/`Part`/`Mesh`/`Material`/`Filter`, `read_*`, `write_*`, `convert`, core options; move runtime/parity/visual/benchmark machinery to a `fascat.validation` (or `fascat.runtime`) submodule (`fascat/__init__.py:126-250`)
- [x] **P1** Delete module-level operation duplicates (`fc.tessellate(asset, …)` etc., `fascat/pipeline.py:1820-2100`) — the dual surface has already drifted (module `repair()` is missing 10 of `RepairOptions`' parameters, `fascat/pipeline.py:1866`); keep only `read_*`, `write_*`, `convert`
- [x] **P1** Lazy submodule imports — core `import fascat` must not pay for the runtime/visual/Pillow stacks (~500 ms today; also taxes every CLI invocation) (`fascat/__init__.py:84-122`)
- [x] **P1** `__repr__` for `Asset`/`Node`/`Part`/`Mesh` and options (showing non-default fields only) (`fascat/asset.py:54`, `fascat/asset.py:117`, `fascat/asset.py:191`, `fascat/mesh.py:210`)
- [ ] **P2** `where: Filter | str | dict | None` instead of `Any` across all ops (`fascat/asset.py:370` et al.)
- [ ] **P2** Mutability contract: document copy-on-operation semantics; guard or document direct mutation of `asset.parts` / mesh arrays (`fascat/asset.py:191`, `fascat/mesh.py:210`)
- [ ] **P2** `convert()` becomes profile-first: `convert(src, dst, profile="realtime-web", **overrides)`; `Literal` profile names for IDE/typo safety (`fascat/pipeline.py:86`)
- [ ] **P2** Consistency pass: properties vs methods (`stats()`, `draw_call_breakdown()` vs `triangle_count`); reconcile `select()`-for-inspection vs `where=`-for-scoping; rename/fold `run_lod_generators` into `lods`; hide `copy(keep_source=…)`
- [ ] **P2** Docstrings on all public classes/methods (document the asset tree model: nodes/parts/occurrences); error-handling examples in docs
- [ ] **P2** Fail fast on unwritable output paths in `write_*`; add `dry_run` to `write_*` for parity with `convert`
- [ ] **P3** Convenience predicates (`is_empty`, `has_meshes`, `has_lods`); `to_trimesh()` interop helpers
- [ ] **P3** Document/normalize implicit conversions in `Filter` (fnmatch patterns) (`fascat/filter.py:101`)

## 5. CLI

Audit verdict: the AGENTS.md contract is largely honored — exit codes 0/1/2 are consistent,
JSON output is clean, NO_COLOR/TTY handling is correct.

- [x] **P1** Ctrl-C mid-conversion leaves a partial output file: `KeyboardInterrupt` bypasses the `except Exception` handlers (`fascat/cli.py:2166`, `fascat/pipeline.py:275`) and `_StageReporter.__exit__` only stops the progress bar (`fascat/cli.py:3612`); fixed by the §1 transactional temp-file + atomic-rename writes (`fascat/cli.py:2131`)
- [ ] **P2** `fascat --json --version` emits plain text — the version callback ignores JSON mode (`fascat/cli.py:362`)
- [ ] **P2** `--verbose`/`-v` is registered and stored but never read anywhere — implement or remove (`fascat/cli.py:381`)
- [ ] **P2** Document the exit-code table in docs/reference.md (0 success / 1 runtime / 2 usage)
- [ ] **P2** Document `--json` output schemas (success/failure payloads per subcommand)
- [ ] **P3** `--no-input` is inert (no prompts exist anywhere) — document as reserved or remove (`fascat/cli.py:396`)

## 6. Production defaults & guideline alignment

- [x] **P1** Expose Draco parameters (compression level 0–10, per-attribute quantization bits) through `GltfExportOptions` — currently hardcoded defaults (`fascat/io/gltf.py:1328`)
- [x] **P1** Expose KTX2/Basis quality (0–255) and effort levels — currently hardcoded UASTC + mips (`fascat/io/gltf.py:1397`)
- [x] **P1** Platform budget validation in profiles/preflight: warn when output exceeds per-target triangle and draw-call budgets (desktop/mobile/VR/WebGL) — budgets are computed today but never enforced *(correction: budget violations already produced report warnings; the real gap was that the CLI never surfaced warning texts — convert now prints report warnings on stderr)*
- [x] **P1** Enforce `forbid_overlapping=True` automatically when UV1 is a bake domain; validate and fail clearly (`fascat/options.py:414`) *(correction: bake-domain auto-enable already existed; the gap was failure semantics — explicit `forbid_overlapping=True` now raises `UVOverlapError` on residual overlaps, bake-domain defaults keep a loud warning)*
- [ ] **P2** Repair `tolerance` default 0.0 means "no vertex merge" — derive a default from the bbox (≈0.1 mm baseline) or warn loudly (`fascat/options.py:191`)
- [ ] **P2** Make relative `sag_ratio` (≈0.0002 of bbox) the default tessellation strategy; absolute sag as fallback (`fascat/options.py:143`)
- [ ] **P2** Re-evaluate normal defaults against industry practice: `hard_edge_angle` 30°→45°, `normal_weighting` angle→area (`fascat/options.py:471-472`)
- [ ] **P2** Enforce 2 px padding for any UV1 bake-domain packing (unwrap or projection) (`fascat/options.py:409`, `fascat/ops/stage.py:178`)
- [ ] **P2** Atlas baking on by default in production export presets (`fascat/options.py:436`)
- [ ] **P2** Warn when the decimation target is <10–20% of source triangles (quality cliff); suggest the proxy/retopo path (`fascat/options.py:784`)
- [ ] **P2** Merge guardrails: warn when merging destroys N instances; prefer instancing messaging (`fascat/options.py:608`)
- [ ] **P3** `lightmap_resolution` option distinct from atlas resolution (default 1024) in `BakeMaterialOptions`
- [ ] **P3** Open-shell-separate face orientation mode (`fascat/options.py:196`)
- [ ] **P3** Texel-density preset for AABB projection (e.g. 1000 units/UV) (`fascat/options.py:449`)
- [ ] **P3** Document switch-distance formula derivations; allow per-level `switch_distance_override` (`fascat/ops/lod.py:614`)

## 7. Testing & CI

- [x] **P1** CI matrix: ubuntu + windows + macos × Python 3.10–3.13 (currently Linux-only, `.github/workflows/ci.yml:15`)
- [x] **P1** Round-trip regression tests: STEP→GLB/USD→re-validate triangle/material/transform fidelity (new `tests/test_round_trip.py`)
- [x] **P1** Malformed-input corpus: truncated STEP, garbage bytes, empty assemblies, degenerate parts, >1M-vertex stress (new `tests/test_robustness.py`)
- [ ] **P2** Real IGES/BREP fixtures (zero exist today; only synthetic geometry is tested)
- [ ] **P2** stdin→stdout conversion integration tests (`-` paths)
- [ ] **P2** Consistent `requires_xatlas` markers (only 3 tests are marked today)
- [ ] **P3** mypy over `tests/`
- [ ] **P3** Golden-image corpora for engine previews (carried from the previous plan)

## 8. Packaging, docs & release

- [x] **P1** Remove `alktx2` from core dependencies (keep only in the `[ktx2]` extra) (`pyproject.toml:27`, `pyproject.toml:43`)
- [x] **P1** Ship `fascat/py.typed` (PEP 561) — strict-mode type hints are currently unusable downstream
- [ ] **P2** Resolve the meshoptimizer pre-release pin `>=0.2.30a0,<0.3`, or document why it's required (`pyproject.toml:30`)
- [ ] **P2** Rename the `uv` extra → `unwrap` (collides with the uv package manager and UV-coordinate jargon) (`pyproject.toml:45`)
- [ ] **P2** Docs: error-handling section + exit codes + JSON schemas (pairs with §5)
- [ ] **P3** FBX epoch CreationTime: document the reproducibility rationale (`fascat/io/fbx.py:140`)

## 9. Carried-forward feature roadmap

Still-open feature work from the previous plan, unchanged in intent:

- [ ] Conformal/intrinsic CAD UV solver (also a top guideline gap); ConformalScaledUV-style mode in tessellation
- [ ] UV island merge + alignment for tileable UV0; backend-enforced solver controls (conformal/isometric, seam, overlap policies)
- [ ] Topology-only connectivity vertex merge with split render attributes
- [ ] Non-orientable strip cracking before face orientation
- [ ] High-poly → proxy normal-map baking
- [ ] Raster/GPU occlusion backend; loose/precise + symmetry/mirror-aware instance reconstruction; retopology/proxy-mesh paths
- [ ] Per-LOD texture-resolution policy beyond advisory
- [ ] Full Unreal scene renderer; cross-platform fallback-free KTX2/Basis decode; checked-in engine-specific material/lighting and LOD-profile golden corpora
- [ ] Vendor material-library containers; deeper vendor-specific external-reference placement transforms; full AP242 conditional/effectivity geometry evaluation; deeper curvature-tiered tessellation profiles; open-shell standalone patch-cleanup expert operations

## Principles

- Keep the public API small, explicit, and Pythonic — Shapely is the bar.
- Preserve CAD hierarchy, transforms, names, colors, metadata, and instancing by default.
- Make lossy or approximate steps explicit in options, docs, and reports.
- Prefer warnings and partial success over silent data loss.
- Never leave a partial or corrupt output file on disk — writes are transactional.
- Treat input files as untrusted: bounded memory, bounded scans, confined path resolution.
- Use proven geometry libraries for CAD kernels, tessellation, simplification, UV packing, and USD authoring.
- Work one feature at a time: implement, test, document, commit, push, verify CI/docs.

## Deferrals

Intentionally out of scope until a user need changes the priority:

- Non-STEP CAD formats beyond the supported IGES and native BREP paths: Parasolid, JT, CATIA, NX, SolidWorks, IFC, 3MF, QIF.
- Animation / skinning / morph targets / animated GLB passthrough.
- Convex decomposition and physics proxy generation.
- Advanced retopology and subdivision workflows.
- GPU-specific runtime packaging beyond standards-aligned glTF/USD output.

## Operating Checklist

For each planned item:

1. Confirm the intended behavior in docs or tests first.
2. Keep the change scoped to one user-visible outcome.
3. Add or update focused tests.
4. Update API/reference docs when public behavior changes.
5. Run `make fmt-check`, `make lint`, `make docs`, and `make ci`.
6. Commit with the repo convention.
7. Push and verify GitHub CI and Docs workflows are green.

## Audit notes — claims investigated and excluded

Recorded so future audits don't re-litigate them:

- xatlas remap claims (face_groups / material_indices "not remapped") — refuted: xatlas splits vertices, not faces; face count and per-face data are preserved (`fascat/mesh.py:2274-2275`).
- Opacity-bake fallback and alpha-flatten color-space claims — refuted: the math matches `Material.effective_opacity`; the Pillow pipeline operates in sRGB throughout.
- External-reference path traversal — blocked by extension validation (`fascat/io/step.py:964`); only the texture/material-library resolvers are vulnerable (kept as the §1 P0).
- "Tangents generated before UVs" — refuted: `_stage_tangents` runs after all UV setup (`fascat/ops/stage.py:190`).
- "Unconditional stats() in progress reporting" — overstated: gated on `progress is not None` (`fascat/pipeline.py:134`); kept only as the §3 caching item because the CLI always passes a callback.
- FPS-target validation — not meaningfully checkable at conversion time; superseded by the §6 budget-validation item.
