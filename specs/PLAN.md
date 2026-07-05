# Fascat Plan

The single planning document for Fascat. For the log of shipped work, see
[CHANGELOG.md](../CHANGELOG.md). This file tracks **what blocks production readiness**
and **what is still open**, as prioritized TODO checklists.

Provenance: full codebase audit on 2026-06-12 (every subsystem read end-to-end; every
bug claim adversarially re-verified before inclusion). The Unity Asset Transformer
guidelines are used as a reference checklist — matching them 100% is explicitly a
non-goal. Shapely is the bar for the Python API.

Status: every P0 and P1 item from that audit shipped in **0.4.0** (2026-06-13).
Open work below is P2/P3 and the carried-forward roadmap.

Re-audit 2026-07-01: identified the exception hierarchy, tessellation output guard,
library logging, and merge/explode hot-path gaps (F1–F7), plus the merged performance
baselines.

Re-audit 2026-07-03 (three parallel passes over io/, ops/+mesh, and CLI/options/tests,
with every contested claim re-verified by hand): the 2026-07-01 batch is **implemented**
— exception hierarchy, tessellation guard, F1–F7, scipy connected components, library
logging, bbox-derived repair tolerance, sag_ratio-primary tessellation, 45°/area normal
defaults, decimation-cliff and merge-instance-loss warnings, `lightmap_resolution`, the
meshoptimizer pin comment, and the `unwrap` extra rename — those items have left this
plan. New this audit: one correctness item (§1), two robustness items (§2), five
performance findings F8–F12 (§3) — the F1/F4/F6 patterns were fixed only in
`hierarchy.py`/`scene.py` and survive verbatim in the export/tessellate/LOD paths —
plus docs-staleness and test-coverage items (§6, §7). Two agent claims were refuted and
recorded in the audit notes.

Re-audit 2026-07-05 (five parallel passes over mesh/analysis, pipeline/runtime/asset,
CLI/options, `io/`, and `ops/`, every high-impact claim re-verified by hand): the F8–F12
batch is **implemented** (shipped with the mesh/scene vectorization commit) and has left
the open list. New this audit: twelve performance findings F13–F24 (§3), spanning the
ambient-occlusion bake, the tessellation OCP boundary, STEP import scaling, CLI startup
imports, and the Asset copy discipline. Three agent claims were refuted and recorded in
the audit notes.

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

- States: `[ ]` open · `[~]` partial · `[x]` implemented.
- Priorities: **P0** correctness/security blocker · **P1** production requirement ·
  **P2** quality/competitive · **P3** later.
- Every code claim cites `file:line` as of the audit commit; lines drift as code changes.

## 1. Correctness bugs

- [x] **P2** `Mesh.optimize_buffers` face-group remap reads uninitialized memory — added
  by audit 2026-07-03. The inverse-permutation buffer is allocated with
  `np.empty_like(reordered_face_indices)` (`fascat/mesh.py:2691-2692`) and then scattered
  into via `inverse_face_order[reordered_face_indices] = np.arange(...)`.
  `reordered_face_indices` is built by looking each cache-optimized face up in a
  `tuple(sorted(face)) → old index` dict (`fascat/mesh.py:2645-2657`); when the mesh
  contains **duplicate faces** (same vertex set), the dict collapses them to one entry, so
  `reordered_face_indices` is no longer a permutation of `0..F-1` — some slots of the
  `np.empty` buffer are never written and leak garbage indices into the remapped
  `face_groups`. This does not crash (see the refuted-claims note below), but corrupted
  face-group indices silently poison downstream material-boundary and feature-preservation
  logic. Compounding it, the entire method body is wrapped in
  `except Exception: return self.copy()` (`fascat/mesh.py:2699-2700`), so *any* future bug
  in this path degrades to a silent no-op with no report entry.

  ```python
  # fascat/mesh.py — replace the np.empty_like scatter with a sentinel-filled
  # lookup and drop unmatched entries, mirroring _remap_sliced_face_group in ops/scene.py:
  inverse_face_order = np.full(self.triangle_count, -1, dtype=np.int64)
  inverse_face_order[reordered_face_indices] = np.arange(
      reordered_face_indices.shape[0], dtype=np.int64
  )
  face_groups: dict[str, IntArray] = {}
  for name, values in self.face_groups.items():
      if not np.isin(values, reordered_face_indices).all():
          continue  # unchanged policy: drop groups that reference reordered-away faces
      remapped = inverse_face_order[values]
      face_groups[name] = remapped[remapped >= 0]
  mesh.face_groups = face_groups

  # and make the blanket fallback observable instead of silent:
  except Exception:
      logger.warning("optimize_buffers failed; returning unoptimized copy", exc_info=True)
      return self.copy()
  ```

  Test to add: a mesh with two identical faces plus a `face_groups` entry, assert
  `optimize_buffers()` returns in-range group indices (today it can return garbage).

## 2. Robustness & input hardening

- [x] **P2** I/O readers/writers bypass the exception hierarchy — added by audit
  2026-07-03. The 2026-07-01 item shipped for the *custom* exceptions
  (`FascatError` base in `fascat/errors.py`; `MeshValidationError` `fascat/mesh.py:70`,
  `UVOverlapError` `fascat/ops/stage.py:16`, `FilterExpressionError` `fascat/filter.py:18`
  all re-based and exported), but the I/O layer still raises bare stdlib exceptions, so
  `except fascat.FascatError` misses every reader/writer failure. Verified examples:
  `RuntimeError` for glTF Transform timeouts (`fascat/io/gltf.py:1417-1419`), `ValueError`
  for oversized inputs (`fascat/io/step.py:918`), `RuntimeError` for malformed FBX/OBJ/STL
  (`fascat/io/fbx.py:67`, `fascat/io/obj.py:97`, `fascat/io/stl.py:70`). Library consumers
  embedding fascat must catch `(RuntimeError, ValueError, OSError)` alongside
  `FascatError`, which defeats the point of the hierarchy. Fix at the public entry points
  (`read_step`/`read_iges`/`read_brep`/`read`/`write_*`), not at every raise site:

  ```python
  # fascat/errors.py
  class FascatIOError(FascatError):
      """Raised when reading or writing an asset file fails."""

  # fascat/io/_errors.py (new) — applied to every public read_*/write_* entry point
  def wrap_io_errors(operation: str) -> Callable[[F], F]:
      def decorate(func: F) -> F:
          @functools.wraps(func)
          def wrapper(*args: Any, **kwargs: Any) -> Any:
              try:
                  return func(*args, **kwargs)
              except FascatError:
                  raise  # already ours — never double-wrap
              except (RuntimeError, ValueError, OSError) as exc:
                  raise FascatIOError(f"{operation} failed: {exc}") from exc
          return cast(F, wrapper)
      return decorate
  ```

  Deliberately narrow: do **not** catch `Exception` (keeps `KeyboardInterrupt`,
  `MemoryError`, and genuine bugs loud). The CLI's own error handling is unaffected — this
  is for library embedders. Docs: add a "catching errors" section to `docs/api.md`
  showing `except fc.FascatError`, and state that `fc.Error` is an alias kept for
  brevity (`fascat/errors.py:10`) with `FascatError` as the documented canonical name.

- [x] **P2** Bbox-derived repair tolerance silently degrades to 0.0 on mesh-less
  selections — added by audit 2026-07-03. The bbox-derived default shipped
  (`RepairOptions.resolved_tolerance`, `fascat/options.py:303-307`, wired at
  `fascat/asset.py:644`), but `_mesh_selection_bbox_diagonal` returns `0.0` when no
  selected part carries a mesh (`fascat/asset.py:1679-1695`), so
  `resolved_tolerance(0.0)` yields `0.0` — i.e. vertex merge silently disabled, which is
  exactly the failure mode the 2026-07-01 item was written to eliminate. A user who runs
  `repair` before `tessellate` (or filters down to mesh-less construction parts) gets the
  old no-merge behavior with nothing in the report. Warn at the resolution site:

  ```python
  # fascat/asset.py — repair op entry (asset.py:644)
  diagonal = _mesh_selection_bbox_diagonal(scope.asset, part_ids)
  resolved_tolerance = opts.resolved_tolerance(diagonal)
  if opts.tolerance == 0.0 and resolved_tolerance == 0.0:
      scope.asset.report.add_warning(
          "repair tolerance auto-derivation found no mesh geometry in the selection; "
          "vertex merge is disabled (tolerance=0) - run tessellate first or pass an "
          "explicit tolerance"
      )
  ```

## 3. Performance

- [x] **P3** Vectorize the remaining export/tessellate/LOD hot paths — audit 2026-07-03:
  the F1/F4/F6 patterns fixed in `hierarchy.py`/`scene.py` survive verbatim in five other
  places, including two on the *main* pipeline (glTF export, tessellation dedupe) that the
  merge/explode-only F1–F6 batch never touched; findings F8–F12 with ready-to-apply
  snippets live in the detailed findings below — audit 2026-07-05: F8–F12 verified
  **shipped** (see verification below)
- [ ] **P2** Fix the 2026-07-05 hot-path findings — the AO bake's O(F² × D) Python ray
  casting (F13), the per-vertex OCP round-trips on the main tessellate path (F14), and
  the per-construction Asset geometry deep-copies (F20, profile first); details below
- [ ] **P3** Fix the 2026-07-05 scaling and cleanup findings F15–F19 and F21–F24 —
  analysis O(n²) pair scans, STEP import list scans, CLI startup imports, and the
  remaining `.tolist()`/per-face Python-loop sites; details below
- [ ] **P3** Profile on real large CAD corpora (10k+ parts) and record numbers in the
  baseline list below — fixture baselines recorded there (2026-06-14 corpus, 2026-07-01
  re-run); actual 10k+ part corpus profiling still needed

### Performance Findings

Audit of 2026-07-05: the 2026-07-03 findings F8–F12 were verified **shipped** (see
verification below) and are gone from the open list, joining F1–F7 and the scipy
connected-components rewrite. This section now holds the **new open findings F13–F24**,
from a five-pass sweep of the whole package (mesh/analysis, pipeline/runtime/asset,
CLI/options, `io/`, `ops/`) with every high-impact claim re-verified by hand. The batch
splits into algorithmic rewrites (F13–F16, F18), scaling fixes (F17, F21), startup
work (F19), one architectural item to profile before acting (F20), and the recurring
`.tolist()`/per-face-Python-loop sweep (F22–F24). Findings are independent unless noted;
implement, test, and commit them one at a time per the operating checklist.

#### Baseline

- Fixture baseline (2026-06-14, 8 real STEP fixtures): **75.19 s total / 665.3 MiB peak RSS**
  — import 31.80 s, LOD 20.53 s, tessellation 8.39 s, repair 5.36 s, stage 5.01 s,
  optimize 3.84 s, write 0.11 s.
- Re-run 2026-07-01 (`make benchmark`, `tests/fixtures/vertical-screw.step` → GLB):
  **3.69 s total / 370.5 MiB peak RSS** — import 0.87 s, tessellate 1.57 s, LOD 0.49 s,
  repair 0.34 s, stage 0.21 s, optimize 0.19 s, write 0.004 s.
- Still open (§3 above): profiling on a real 10k+ part corpus.

#### Verification of prior findings (2026-07-03)

All seven 2026-07-01 findings plus the connected-components rewrite are implemented:

- F1 — `np.unique` material enumeration: `fascat/ops/hierarchy.py:262` and `:514`.
- F2 — vectorized `_face_material_ids` gather: `fascat/ops/hierarchy.py:405-416`.
- F3 — BFS `face_rows` hoisted once: `fascat/ops/hierarchy.py:561-573`.
- F4 — `_part_material_key` via `_array_digest_required` with int64 normalization:
  `fascat/ops/scene.py:274-278`; the digest-count test moved to 10 as predicted
  (`tests/test_scene.py:161`).
- F5 — `_face_chunks` counts only new vertices (quadratic removed):
  `fascat/ops/scene.py:448`.
- F6 — `_slice_mesh` face-group remap via lookup array:
  `fascat/ops/scene.py:464-473` + `_remap_sliced_face_group` at `:480-486`.
- F7 — per-channel UV stats **are cached**, in `fascat/mesh.py` rather than the op:
  `uv_layout_stats` (`fascat/mesh.py:2205-2273`) and `uv_distortion_metrics`
  (`fascat/mesh.py:2383-2458`) both go through `_cached_value` keyed on
  channel + tolerance + geometry cache token, so `_tag_uv_layout_quality` calling them per
  channel is correct. The former `[~]` §3 item is closed (see audit notes).
- scipy connected components: `_connected_face_components_scipy`
  (`fascat/ops/hierarchy.py:537-556`) builds a face×vertex incidence matrix and runs
  `scipy.sparse.csgraph.connected_components` on `incidence @ incidence.T`, with the
  Python BFS kept as fallback (`:559-580`).

#### Verification of prior findings (2026-07-05)

All five 2026-07-03 findings are implemented (current lines):

- F8 — the `np.unique(...astype(np.int64, copy=False))` idiom is in place at all four
  sites: `fascat/io/gltf.py:1689`, `fascat/export_report.py:111`,
  `fascat/asset.py:1764,1779`, `fascat/ops/actions.py:2684`.
- F9 — tessellation dedupe keys on `array_digest_required(...)`:
  `fascat/ops/tessellate.py:1382-1385`.
- F10 — `_slice_faces` uses the shared `sliced_face_lookup` +
  `remap_sliced_face_group` helpers: `fascat/ops/actions.py:1820-1831`.
- F11 — `_sample_mesh_faces` tops up via `np.setdiff1d`:
  `fascat/ops/actions.py:1805-1817`.
- F12 — `Mesh.optimize_buffers` uses the sorted-rows void-view + `searchsorted` remap
  and the masked-scatter remap inversion: `fascat/mesh.py:2650-2661`.

#### Open findings

F13 and F14 sit on user-facing hot paths (the `bake-materials` action and the main
tessellate step — 1.57 s of the 3.69 s fixture baseline); F20 multiplies with pipeline
depth on every run; the rest bite on large imports, analysis of large meshes, or CLI
startup.

##### F13 — AO bake is O(F² × D) Python with a full-mesh gather per ray

`_face_ambient_occlusion` (`fascat/ops/actions.py:809-831`) loops every face × every
direction and calls `_ray_hits_mesh` (`:839-855`), which re-gathers
`triangles = points[faces]` — a copy of the entire mesh — **on every ray** (`:848`),
then runs a pure-Python per-triangle loop over `_ray_triangle_t`. A 100k-face mesh with
the conservative direction set is ~2.6M full-mesh gathers and on the order of 10¹¹
Python-level ray-triangle tests; the bake is unusable beyond toy meshes. Fix, in order
of payoff:

1. Hoist the `points[faces]` gather out of `_ray_hits_mesh` (one line; removes the
   dominant allocation).
2. Vectorize `_ray_triangle_t` over all triangles per ray — batched Möller–Trumbore:
   the edge/`np.cross`/determinant math is already array-shaped; mask the ignored face
   and finish with `np.any`.
3. If still hot, reuse the `_build_occurrence_bvh` AABB idiom to prefilter triangles
   per ray.

Call context: per part in `bake_materials`.

##### F14 — tessellation reads triangulation node-by-node through OCP

On the **main tessellate path**: per vertex,
`nodes.Value(i).Transformed(payload.transform)` plus three `.X()/.Y()/.Z()` calls
(`fascat/ops/tessellate.py:238-240`); per triangle, `triangles.Value(i).Get()`
(`:255-268`); per UV node, `uv_nodes.Value(i)` + `.X()/.Y()`
(`_triangulation_uv_nodes`, `:431-437`). That is ~6 OCP round-trips per vertex and 4 per
triangle, and it dominates tessellation of dense meshes. Fix: read the untransformed
node coordinates once per triangulation and apply the 4×4 transform vectorized in numpy
(halves the per-vertex OCP calls and moves the arithmetic out of Python); then
investigate batch/array access on `Poly_Triangulation` in OCP for the remaining
per-element reads. Tessellate was 1.57 s of the 3.69 s `vertical-screw.step` baseline,
so `make benchmark` must move.

##### F15 — `_self_intersection_count` scans O(n²) pairs in Python

`fascat/analysis.py:339-369`: full double loop over triangle pairs; the AABB reject
inside runs two scalar `np.all` calls per pair (`:360`), so even rejected pairs are
expensive. `max_pairs` caps the work, but the cap is consumed scanning near-diagonal
pairs, so large meshes get truncated, low-value results rather than fast ones. Fix:
generate candidate pairs vectorized first — sort by min-x and broadcast the AABB overlap
test per row (the same sweep shape as `uv_layout_stats`), or query
`scipy.spatial.cKDTree` on triangle centroids (scipy is already a dependency) — then run
`_triangles_intersect` only on survivors, spending `max_pairs` on real candidates.

##### F16 — UV overlap detection degrades to O(n²) with Python polygon clipping

`uv_layout_stats` (`fascat/mesh.py:2236-2265`): the overlap sweep sorts on U and prunes
with `break` on the U axis only (`:2249-2250`); atlas-packed meshes, where most
triangles share the U range, degrade to near-quadratic, and every surviving pair pays
`_triangle_overlap_area_2d` — pure-Python polygon clipping. The result is cached (F7),
but the first computation is the cost. Fix: bucket triangles on both UV axes (2D grid),
prefilter pairs with a vectorized AABB test, and clip only the survivors.

##### F17 — STEP material binding uses `list.index()` per face

`_material_binding_plan` (`fascat/io/step.py:5306-5313`) runs
`face_material_id not in material_ids` plus `material_ids.index(face_material_id)` — two
linear scans — per face: O(faces × distinct materials) per part. Fix: keep a
`{material_id: index}` dict beside the list:

```python
index_by_id = {base_material_id: 0}
for face_material_id in face_material_ids:
    index = index_by_id.get(face_material_id)
    if index is None:
        index = len(material_ids)
        index_by_id[face_material_id] = index
        material_ids.append(face_material_id)
    material_indices.append(index)
```

##### F18 — STEP free-edge detection is O(edges × face_edges) via `IsSame` scans

`fascat/io/step.py:5180-5188`: every candidate edge is checked with
`any(edge.IsSame(face_edge) for face_edge in face_edges)` plus a growing scan of
`free_edges` — nested linear scans of OCP calls, quadratic on edge-rich shapes. Fix:
collect the face edges into a `TopTools_IndexedMapOfShape` (one
`TopExp.MapShapes_s(shape, TopAbs_EDGE, map)` per face pass) and use `Contains()` — a
hash lookup honoring `IsSame` semantics — for both membership tests. Do **not** key on
Python `id()`: OCP wrapper objects are transient (see audit notes).

##### F19 — CLI startup imports numpy, `io/step.py`, and `fascat.analysis` eagerly

Measured 2026-07-05: `import fascat.cli` takes ~129 ms, with the `fascat.analysis`
subtree alone ~59 ms of it (`python -X importtime`). Three eager chains, all in
`fascat/cli.py`:

- `:22` — `AnalysisReport` is used only inside a stringified annotation (`:3889`;
  `from __future__ import annotations` is active), yet the import loads
  `fascat.analysis` → `fascat.asset` → `fascat.mesh` → numpy. `analyze_output` runs only
  on the analyze path — move both into the function / a `TYPE_CHECKING` block.
- `:24-30` — `read_step`/`read_brep`/`read_iges` are imported at module top from
  `fascat/io/step.py` (7k lines, `import numpy` at `:16`); defer into the command
  bodies.
- `:24-30` — the `*_SUFFIXES` constants come from `io/` modules that import numpy at top
  (`fascat/io/gltf.py:16`, `io/fbx.py:7`, `io/obj.py:6`, `io/stl.py:7`); move them to a
  lightweight constants module (e.g. `fascat/io/_suffixes.py`) re-exported by the io
  modules.

The lazy-import contract already forbids numpy after `import fascat`
(`tests/test_lazy_imports.py:15-23`), but the CLI test (`:26-33`) only excludes
runtime/visual/PIL — extend it to forbid `numpy`, `fascat.io.step`, and
`fascat.analysis` after `import fascat.cli` so this can't regress.

##### F20 — every `Asset(...)` construction deep-copies all geometry

`Asset.__post_init__` (`fascat/asset.py:327-334`) copies the node tree, every part,
every material, and every image; `Part.copy(keep_source=True)` → `Mesh.copy()`
(`fascat/mesh.py:306-320`) copies every numpy array (points, faces, normals, tangents,
UVs, material indices, face groups). A pipeline run constructs Assets at every stage, so
total geometry copied scales with pipeline depth × scene size. The `_adopt` classmethod
(`fascat/asset.py:336`) already bypasses the copies for trusted construction. This is a
deliberate immutability design, so treat it as architectural: profile on the 10k+ part
corpus first (§3 open item), then route internally-constructed, invariant-preserving
results through `_adopt` — or add copy-on-write to `Mesh`. Behavior-sensitive; needs
dedicated aliasing tests before any change.

##### F21 — spatial-bucket neighbor search is per-vertex Python with scalar norms

Four sites in `fascat/mesh.py` share the hand-rolled 3×3×3-cell grid walk
(`for dx in (-1, 0, 1): ...` at `:800`, `:878`, `:1356`, `:1483` — the merge-vertices,
boundary-gap, and stitch paths): per vertex, 27 dict lookups plus
`float(np.linalg.norm(point - self.points[other]))` per neighbor — one scalar numpy call
each. Fix: `scipy.spatial.cKDTree(points).query_pairs(radius)` (scipy is already a
dependency) replaces each grid with a single vectorized call; keep a grid only where a
site needs its exact tie-break order (verify against the existing merge/stitch tests).

##### F22 — per-face `.astype(int).tolist()` loops survive in mesh repair/analysis

The same shape F8/F12 removed elsewhere, still present at:

- `fascat/mesh.py:1029-1032` — material-signature build converts each face row with
  `face.astype(int).tolist()` inside the per-face loop; faces are already int64 —
  iterate the array directly or group vectorized.
- `fascat/mesh.py:1378-1383` — `split_t_junctions` round-trips `points`, all UV
  channels, and `faces` through Python lists up front, then appends per split; keep the
  numpy arrays and concatenate the (rare) additions once at the end.
- `fascat/mesh.py:1684-1686` — orientability edge map: `.astype(int).tolist()` over all
  faces to build per-edge incidence; build the edge array with `np.stack` + the F12
  void-view dedupe instead.
- `fascat/mesh.py:2306-2327` — UV-seam metrics build `_rounded_key` tuples per vertex in
  list comprehensions and iterate `.tolist()` faces; and
  `edge_lengths.setdefault(edge_key, float(np.linalg.norm(...)))` (`:2327`) computes the
  norm even on a dict hit. Round keys with one `np.round` pass and precompute all edge
  lengths vectorized (`np.linalg.norm(..., axis=1)`).
- `fascat/ops/actions.py:1917-1919` (`_edge_faces`), `:1998-2002` (`_boundary_loops`),
  `:2609-2611` (keep-frontier edge map) — three copies of the same
  `.astype(int).tolist()` + per-edge `setdefault` build; share one vectorized
  edge-incidence helper.

##### F23 — export-writer conversion churn

Small allocations repeated per accessor/node/face on every export:

- `fascat/io/gltf.py:894-895`, `:1212-1213`, `:1222-1223` — `.astype(float).tolist()` /
  `.astype(int).tolist()` on accessor min/max where `.tolist()` alone suffices (the
  `astype` allocates a full copy first); `:1721`, `:1810` — same on per-node 4×4
  transforms.
- `fascat/io/usd.py:569` — `len(set(mesh.material_indices.astype(int).tolist())) <= 1`
  → `np.unique(mesh.material_indices).size <= 1`; `:574` — flatnonzero result through
  `.astype(int).tolist()` per material; `:450` — `Gf.Matrix4d(child.transform.tolist())`
  per node (check whether the Gf/Vt APIs accept buffers directly).
- `fascat/io/obj.py:206` — `_face_material_id(part, mesh, face_index)` called once per
  triangle; compute the material-index → id mapping once per mesh and gather.
- `fascat/ops/tessellate.py:334-340` — construction-curve sample segment lengths one
  scalar `np.linalg.norm` at a time (stack the samples, vectorize with `axis=1`);
  `:356-410` — `_tube_mesh_from_segments` builds points/faces via Python list appends
  with per-side trig in the inner loop (precompute the ring once, broadcast per
  segment); `:465-466` — free-edge JSON via per-point `.astype(float).tolist()`.

##### F24 — small per-run overheads

- `fascat/export_report.py:82` — recomputes `_source_image_ids(asset)` even though
  `source_unique` (`:79`) already contains it; use `len(source_unique)`.
- `fascat/pipeline.py:1501-1504` — `_measured_report_timings` makes three passes over
  `report.steps`; one pass with three accumulators.
- `fascat/pipeline.py:1467-1472` + callers at `:1013`, `:1027`, `:1476` — the
  profile-budget report recomputes `_material_texture_summaries(asset)` (a full
  materials iteration) per metric; compute once per report and share.
- `fascat/runtime.py:1091`, `:1148` — the preview decoders `deepcopy` the whole glTF
  JSON document where copying only the mutated `images`/`bufferViews` lists suffices;
  and `:1121-1124` writes the document to disk then re-reads it just to validate —
  validate the in-memory document before writing. Preview harness only; lowest priority.
- `fascat/ops/actions.py:261` + `:2111-2130` — `_candidate_occluders` rebuilds the
  occluder tuple, the bounds vstacks, **and a full BVH** for every candidate, though the
  only per-candidate difference is excluding the candidate itself (the transparency
  filter is candidate-invariant). Build one BVH over all opaque occurrences and skip
  self-hits by node id at query time — turns O(N² log N) BVH construction into
  O(N log N).

#### Verification recipe (when implementing)

1. Focused tests per finding: `tests/test_actions.py` (F13, F22, F24),
   `tests/test_tessellate.py` (F14, F23), `tests/test_analysis.py` (F15),
   `tests/test_mesh.py` (F16, F21, F22), the STEP/BREP/IGES import tests (F17, F18),
   `tests/test_lazy_imports.py` + `tests/test_cli.py` (F19 — add the numpy/step/analysis
   exclusions to the CLI lazy test first so the fix is pinned), `tests/test_asset.py`
   (F20 — add aliasing tests before touching the copy discipline),
   `tests/test_gltf.py` / OBJ / USD writer tests (F23).
2. `make ci` for the full gate.
3. `make benchmark` before/after — F14 is on the fixture's critical path (tessellate was
   1.57 s of 3.69 s on `vertical-screw.step`), so expect a visible improvement; F13
   needs a bake-materials scenario to measure; F19 is measured with
   `python -X importtime -c "import fascat.cli"` (baseline 2026-07-05: ~129 ms);
   F15–F18 and F20–F24 show on large imports/meshes only.

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


## 5. CLI

Audit verdict: the AGENTS.md contract is largely honored — exit codes 0/1/2 are consistent,
JSON output is clean, NO_COLOR/TTY handling is correct. The library logging item shipped
(NullHandler-backed `logging.getLogger("fascat")`, `fascat/report.py:10-11`, warnings and
errors mirrored at `:135-139`, covered by `tests/test_report.py`).

No open items.

## 6. Production defaults & guideline alignment

- [x] **P2** Enforce 2 px padding for any UV1 bake-domain packing (unwrap or projection) —
  `UnwrapOptions.padding` defaults to 2 (`fascat/options.py:476`) but nothing enforces a
  floor when a bake-domain channel is packed with a smaller explicit value; a 0/1 px
  padding on a lightmap atlas bleeds neighboring charts into AO and lightmap bakes. Clamp
  (with a report note) at the point where the bake domain is resolved
  (`fascat/ops/stage.py`, the `_uv_domain(...) == "bake"` branch):

  ```python
  if domain == "bake" and effective_padding < 2:
      asset.report.add_warning(
          f"part {part_id} uv{channel}: padding {effective_padding}px is below the 2px "
          "bake-domain minimum; clamping to 2px to prevent chart bleed in AO/lightmap bakes"
      )
      effective_padding = 2
  ```

- [x] **P2** Atlas baking on by default in production export presets —
  `AtlasOptions.enabled` is still `False` (`fascat/options.py:504`) and no built-in
  profile flips it, so the "production" presets ship without atlas consolidation; either
  enable it in the realtime/production profiles (`fascat/profiles.py`) or document why
  default-off is the deliberate choice

- [x] **P2** Docs drifted behind the 2026-07 defaults batch — added by audit 2026-07-03;
  re-verify each cite while fixing since docs lines drift fast:
  - `docs/api.md:629-632` describes the `sag_ratio=0.0002` default but not the mutual
    exclusivity: `TessellationOptions(sag=0.1)` silently sets `sag_ratio=None`
    (`fascat/options.py:215-218`) — say so, it changes what "default" means for any
    caller passing `sag`
  - `docs/reference.md` flag tables: no `--sag-ratio` row alongside the documented sag
    flag, and no upgrade note that `--hard-edge-angle` (30→45) and `--normal-weighting`
    (angle→area) defaults changed — assets re-converted with 0.5.0 defaults will shade
    differently, which deserves a migration callout
  - `docs/api.md`: document `fc.Error` vs `fc.FascatError` (alias, `fascat/errors.py:10`)
    and the "catching errors" story from the §2 I/O item

- [x] **P3** Open-shell-separate face orientation mode — implemented and documented
  (`single_sided_open_shell` accepted in `fascat/options.py:275`, wired at
  `fascat/asset.py:1827` and `fascat/mesh.py:96-97`, described in `docs/api.md`) but has
  **zero test coverage** (`grep -r single_sided_open_shell tests/` is empty); add an
  open-shell fixture test asserting per-component consistent orientation before calling
  this done
- [x] **P3** Texel-density preset for AABB projection — `UnwrapOptions.texel_density`
  exists and validates (`fascat/options.py:475,485-486`) but is only *recorded* as
  material metadata (`fascat/ops/stage.py:393-394`); it does not scale the AABB
  projection, so the guideline behavior (e.g. 1000 units/UV) is still unimplemented
- [x] **P3** Document switch-distance formula derivations; allow per-level
  `switch_distance_override` (`_switch_distance`, `fascat/ops/lod.py:665`)

## 7. Testing & CI

- [ ] **P2** Real IGES/BREP fixtures (zero exist today; only synthetic geometry is tested)
- [x] **P2** Modules with no dedicated test file — added by audit 2026-07-03:
  `fascat/export_report.py`, `fascat/material.py`, `fascat/pipeline_file.py`,
  `fascat/pmi_visuals.py`, `fascat/size_ladder.py`, `fascat/validation.py` have no
  `tests/test_<module>.py`; before writing new tests, check the coverage report
  (`make coverage`) per module — several are exercised indirectly through pipeline tests,
  so target the genuinely uncovered branches rather than duplicating integration coverage
- [~] **P2** stdin→stdout conversion integration tests (`-` paths) — the CLI supports `-`
  and two basic tests exist (`tests/test_cli.py:96,1447`); still missing: round-trip
  stdin→stdout for each writer format and the error paths (binary garbage on stdin,
  closed stdout)
- [x] **P2** Consistent `requires_xatlas` markers — still only 3 marked tests
  (`tests/test_stage.py:744,817` + one more); every unwrap/lightmap/bake test that
  soft-skips without xatlas should carry the marker so `-m requires_xatlas` selects them
- [ ] **P3** mypy over `tests/` (`pyproject.toml:98` scopes mypy to `fascat/` only)
- [ ] **P3** Golden-image corpora for engine previews (carried from the previous plan)

## 8. Packaging, docs & release

- [x] **P3** FBX epoch CreationTime: document the reproducibility rationale (`fascat/io/fbx.py:140`)

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
- External-reference path traversal — blocked by extension validation (`fascat/io/step.py:964`); only the texture/material-library resolvers were vulnerable.
- "Tangents generated before UVs" — refuted: `_stage_tangents` runs after all UV setup (`fascat/ops/stage.py:190`).
- "Unconditional stats() in progress reporting" — overstated: gated on `progress is not None` (`fascat/pipeline.py:134`); kept only as the §3 caching item because the CLI always passes a callback.
- FPS-target validation — not meaningfully checkable at conversion time; superseded by the §6 budget-validation item.
- Audit 2026-07-03: "`optimize_buffers` face-group remap is a P0 IndexError crash" —
  refuted: `reordered_face_indices` always has length equal to the triangle count with
  values in `[0, F)` (buffer optimization never changes face count), so the scatter cannot
  raise; and the enclosing `except Exception` (`fascat/mesh.py:2699`) would swallow it
  anyway. Retained only as the §1 P2 uninitialized-memory item.
- Audit 2026-07-03: "`_tag_uv_layout_quality` recomputes per-channel UV stats with no
  cache" (former F7, was `[~]` in §3) — refuted for the working tree: `uv_layout_stats`
  and `uv_distortion_metrics` are cached in `fascat/mesh.py:2205-2273,2383-2458` via
  `_cached_value` keyed on channel + tolerance + geometry cache token; the op calling them
  per channel hits the cache. Closed.
- Audit 2026-07-03, excluded pending verification (reported by audit tooling, **not**
  adversarially verified — re-check before acting): `--normalize-uvs` allegedly dedupes
  channel lists silently; the global-flag normalizer allegedly ignores global flags placed
  after the subcommand (e.g. `fascat convert in.step --verbose out.glb`). Neither claim
  was confirmed against `fascat/cli.py`; verify first, then either add as §5 items or move
  up here as refuted.
- Audit 2026-07-05: "`_context_for` material membership test is O(n²)"
  (`fascat/filter.py:532`) — refuted: `material_id in asset.materials` is an O(1) dict
  lookup; the loop is linear in `part.material_ids`.
- Audit 2026-07-05: "`np.unique(...).tolist()` in the glTF writer is wasteful churn"
  (`fascat/io/gltf.py:1689`) — refuted: this is the sanctioned F8 idiom; the `.tolist()`
  on the (small) unique array avoids boxing a numpy scalar per loop iteration.
- Audit 2026-07-05: an agent-suggested F18 fix keyed OCP edges on Python `id()` —
  rejected: OCP wrapper objects are transient, so `id()` identity is meaningless across
  explorer iterations; shape identity must come from `IsSame`/`TopTools` hashing.
