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

`_face_ambient_occlusion` (`fascat/ops/actions.py:809-831`) shades every face by casting
one ray per hemisphere direction and testing it against every other triangle in the
mesh, in Python:

```python
for face_index, (centroid, normal) in enumerate(zip(centroids, normals, strict=True)):
    hits = 0
    tested = 0
    origin = centroid + normal * epsilon
    for direction in directions:
        if float(np.dot(direction, normal)) <= 0.0:
            continue
        tested += 1
        if _ray_hits_mesh(origin, direction, mesh.points, mesh.faces, ignore_face=face_index, max_t=ray_length):
            hits += 1
```

and `_ray_hits_mesh` (`:839-855`) re-gathers the whole mesh **on every ray** before its
per-triangle scan:

```python
def _ray_hits_mesh(origin, direction, points, faces, *, ignore_face, max_t):
    triangles = points[faces]                    # full-mesh (F, 3, 3) copy, once per ray
    for face_index, triangle in enumerate(triangles):
        if face_index == ignore_face:
            continue
        hit = _ray_triangle_t(origin, direction, triangle)   # scalar Möller–Trumbore
        if hit is not None and 1e-8 < hit < max_t:
            return True
    return False
```

Three compounding costs: (1) the `points[faces]` gather allocates an F×3×3 float array
per ray — F × D times per bake; (2) the inner loop is O(F) Python iterations per ray,
each running ~6 scalar `np.dot`/`np.cross` calls inside `_ray_triangle_t` (`:858-876`),
so the total is O(F² × D) interpreter-level work — a 100k-face mesh at ~26 directions is
on the order of 10¹¹ scalar numpy calls; (3) there is no spatial pruning, so rays that
leave the scene immediately still pay the full scan. The bake is unusable beyond toy
meshes. Fix, in order of payoff:

1. Hoist the gather out of `_ray_hits_mesh` (one line; removes F × D full-mesh
   allocations):

   ```python
   triangles = mesh.points[mesh.faces]          # once, before the face loop
   ```

2. Vectorize Möller–Trumbore over all triangles per ray. The math in `_ray_triangle_t`
   is already array-shaped — broadcast it and reduce with `np.any`; `hit` short-circuits
   nothing today anyway because Python-loop iterations dominate:

   ```python
   def _ray_hits_any(origin, direction, tri, *, ignore_face, max_t):
       edge1 = tri[:, 1] - tri[:, 0]
       edge2 = tri[:, 2] - tri[:, 0]
       h = np.cross(direction, edge2)
       det = np.einsum("ij,ij->i", edge1, h)
       valid = np.abs(det) > 1e-12
       inv_det = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
       s = origin - tri[:, 0]
       u = inv_det * np.einsum("ij,ij->i", s, h)
       q = np.cross(s, edge1)
       v = inv_det * (q @ direction)
       t = inv_det * np.einsum("ij,ij->i", edge2, q)
       hit = valid & (u >= 0.0) & (u <= 1.0) & (v >= 0.0) & (u + v <= 1.0)
       hit &= (t > 1e-8) & (t < max_t)
       hit[ignore_face] = False
       return bool(np.any(hit))
   ```

   This turns the per-ray cost from O(F) Python iterations into a handful of vectorized
   passes; the remaining face × direction loop can later be batched into a single
   (D, F) broadcast per face if needed.

3. If still hot on very large meshes, add spatial pruning: reuse the
   `_build_occurrence_bvh` AABB idiom on per-triangle bounds to cull candidates per ray.

Call context: per part in `bake_materials` — this loop is the entire cost of the AO
channel. Verify against the existing bake tests; the vectorized form must keep the
`1e-8 < t < max_t` window and the ignore-self semantics exactly.

##### F14 — tessellation reads triangulation node-by-node through OCP

On the **main tessellate path**, the mesh-assembly loop extracts every vertex and every
triangle through individual OCP method calls (`fascat/ops/tessellate.py:235-268`):

```python
for payload in face_triangulations:
    nodes = payload.triangulation.MapNodeArray()
    node_lower = int(nodes.Lower())
    for local_index in range(payload.node_count):
        point = nodes.Value(node_lower + local_index).Transformed(payload.transform)
        points[payload.point_offset + local_index] = (point.X(), point.Y(), point.Z())
    ...
    for local_index in range(payload.triangle_count):
        a, b, c = triangles.Value(triangle_lower + local_index).Get()
        if payload.reversed_face:
            faces[payload.triangle_offset + local_index] = (
                payload.point_offset + c - 1, payload.point_offset + b - 1, payload.point_offset + a - 1,
            )
        else:
            ...
```

with the same shape for UVs in `_triangulation_uv_nodes` (`:431-437`):

```python
for local_index in range(node_count):
    point = uv_nodes.Value(lower + local_index)
    values[local_index] = (float(point.X()), float(point.Y()))
```

Per vertex that is a `Value()` call, a `Transformed()` call (which constructs a new
`gp_Pnt` and multiplies through the `gp_Trsf` in C++ but pays a Python round-trip), and
three coordinate getters — ~6 OCP boundary crossings per vertex, plus 4 per triangle
(`Value` + `Get` + the per-row Python branch on `reversed_face`). Each crossing is
~microseconds of wrapper overhead; on dense tessellations this dominates the step
(tessellate was 1.57 s of the 3.69 s `vertical-screw.step` baseline). Fix in two stages:

1. Drop `.Transformed()` — read raw coordinates once and apply the transform vectorized;
   and hoist the `reversed_face` branch out of the triangle loop as a column flip:

   ```python
   raw = np.empty((payload.node_count, 3), dtype=np.float64)
   for local_index in range(payload.node_count):
       point = nodes.Value(node_lower + local_index)          # no .Transformed()
       raw[local_index] = (point.X(), point.Y(), point.Z())
   trsf = payload.transform
   matrix = np.array(
       [[trsf.Value(r, c) for c in (1, 2, 3, 4)] for r in (1, 2, 3)], dtype=np.float64
   )                                                          # 3×4, once per payload
   points[offset : offset + payload.node_count] = raw @ matrix[:, :3].T + matrix[:, 3]

   rows = np.empty((payload.triangle_count, 3), dtype=np.int64)
   for local_index in range(payload.triangle_count):
       rows[local_index] = triangles.Value(triangle_lower + local_index).Get()
   if payload.reversed_face:
       rows = rows[:, ::-1]
   faces[t_off : t_off + payload.triangle_count] = payload.point_offset + rows - 1
   ```

   This removes the per-vertex transform round-trip and moves all arithmetic and the
   winding flip out of Python; the material fill at `:269-272` is already vectorized.

2. Investigate OCP batch access on `Poly_Triangulation` to eliminate the remaining
   per-element `Value()` calls entirely (e.g. whether the node/triangle arrays expose a
   buffer, or whether `RWStl`/`BRepTools`-style bulk copies apply). If OCP offers no
   batch path, stage 1 is still roughly a 3× reduction in boundary crossings.

`make benchmark` must move on this finding; it is the fixture's largest step.

##### F15 — `_self_intersection_count` scans O(n²) pairs in Python

`fascat/analysis.py:339-369` enumerates every triangle pair in a Python double loop:

```python
face_vertices = [set(face) for face in mesh.faces.astype(int).tolist()]
...
for left in range(mesh.triangle_count - 1):
    for right in range(left + 1, mesh.triangle_count):
        if face_vertices[left] & face_vertices[right]:
            continue
        if checked >= max_pairs:
            return _SelfIntersectionResult(..., truncated=True, ...)
        checked += 1
        if not bool(np.all(maxs[left] >= mins[right]) and np.all(maxs[right] >= mins[left])):
            continue
        if _triangles_intersect(triangles[left], triangles[right]):
            intersections += 1
```

Two problems. First, the AABB reject runs two scalar `np.all` calls per pair (`:360`),
so even rejected pairs cost ~microseconds each — F²/2 of them. Second, `max_pairs` was
meant as a safety valve, but because pairs are visited in row order, the budget is spent
entirely on the near-diagonal neighborhood of the first few hundred triangles; large
meshes get a **truncated result that sampled almost none of the mesh**, i.e. slow *and*
wrong-ish. Fix: generate candidate pairs vectorized first, then test only survivors:

```python
order = np.argsort(mins[:, 0], kind="mergesort")
smin, smax = mins[order], maxs[order]
for pos in range(order.size - 1):
    # sweep window: only triangles whose min-x is inside [.., smax[pos, 0]]
    end = pos + 1 + int(np.searchsorted(smin[pos + 1 :, 0], smax[pos, 0], side="right"))
    if end == pos + 1:
        continue
    window = slice(pos + 1, end)
    overlap = np.all(smax[window] >= smin[pos], axis=1) & np.all(smin[window] <= smax[pos], axis=1)
    for right in order[window][overlap]:                     # AABB-confirmed candidates only
        ...  # shared-vertex check, budget check, _triangles_intersect
```

The per-row AABB test is one vectorized pass over the window instead of two scalar calls
per pair, and `max_pairs` is then spent on genuine AABB-overlapping candidates.
(`scipy.spatial.cKDTree` on triangle centroids with a radius of the max triangle extent
is an equivalent alternative; scipy is already a dependency.) Preserve the exact
`truncated`/`pairs_checked` reporting semantics — tests assert on them. Keep the scalar
`_triangles_intersect` as-is: after the prefilter it runs on a tiny survivor set and
vectorizing it buys little.

##### F16 — UV overlap detection degrades to O(n²) with Python polygon clipping

The overlap sweep in `uv_layout_stats` (`fascat/mesh.py:2236-2265`) sorts triangles on U
and prunes only on the U axis:

```python
order = np.argsort(min_uv[:, 0], kind="mergesort")
for position, left_index_value in enumerate(order):
    ...
    for right_index_value in order[position + 1 :]:
        right_index = int(right_index_value)
        if min_uv[right_index, 0] > left_max[0] + tolerance:
            break                                            # U-axis pruning only
        ...
        if _triangle_overlap_area_2d(triangles[left_index], triangles[right_index],
                                     tolerance=tolerance) > tolerance:
            overlapping_pairs += 1
```

On atlas-packed meshes — the normal case after unwrap, where charts tile the full 0–1
square and most triangles' U intervals overlap somebody's — the `break` rarely fires
early and the sweep degrades toward F²/2 iterations, each surviving pair paying
`_triangle_overlap_area_2d`, a pure-Python Sutherland–Hodgman polygon clip. The result
is cached per channel + tolerance + geometry token (F7), but the *first* computation on
every analyzed mesh is the cost. Fix: prune on both axes with a 2D grid keyed by UV
AABB, then AABB-prefilter within buckets and clip only survivors:

```python
cell = max(float(np.median(max_uv - min_uv)), tolerance)     # ~triangle-sized cells
grid: dict[tuple[int, int], list[int]] = defaultdict(list)
for index in np.flatnonzero(~degenerate).tolist():
    lo = np.floor(min_uv[index] / cell).astype(int)
    hi = np.floor(max_uv[index] / cell).astype(int)
    for gx in range(lo[0], hi[0] + 1):
        for gy in range(lo[1], hi[1] + 1):
            grid[(gx, gy)].append(index)
seen: set[tuple[int, int]] = set()
for bucket in grid.values():
    for i, left in enumerate(bucket):
        for right in bucket[i + 1 :]:
            pair = (left, right) if left < right else (right, left)
            if pair in seen:
                continue
            seen.add(pair)
            # vectorizable AABB test, then _triangle_overlap_area_2d on survivors
```

Expected cost drops to O(F × occupancy): pairs are only generated within cells, and
genuinely overlapping charts are the only thing that still pays the Python clipper.
Counting semantics (`overlapping_pairs` counts unordered pairs once) must be preserved —
hence the `seen` dedupe for triangles spanning multiple cells.

##### F17 — STEP material binding uses `list.index()` per face

`_material_binding_plan` (`fascat/io/step.py:5306-5313`) runs two linear scans of the
materials list for **every face** of a part:

```python
def _material_binding_plan(base_material_id: str, face_material_ids: list[str]) -> tuple[list[str], list[int]]:
    material_ids = [base_material_id]
    material_indices: list[int] = []
    for face_material_id in face_material_ids:
        if face_material_id not in material_ids:             # O(M) scan
            material_ids.append(face_material_id)
        material_indices.append(material_ids.index(face_material_id))  # O(M) scan again
    return material_ids, material_indices
```

That is O(faces × distinct materials) with two string-comparison scans per face — on a
10k-face part with dozens of face colors (common in styled STEP assemblies), millions of
string comparisons during import. Fix: keep a `{material_id: index}` dict beside the
list; identical return values, O(faces) total:

```python
def _material_binding_plan(base_material_id: str, face_material_ids: list[str]) -> tuple[list[str], list[int]]:
    material_ids = [base_material_id]
    index_by_id = {base_material_id: 0}
    material_indices: list[int] = []
    for face_material_id in face_material_ids:
        index = index_by_id.get(face_material_id)
        if index is None:
            index = len(material_ids)
            index_by_id[face_material_id] = index
            material_ids.append(face_material_id)
        material_indices.append(index)
    return material_ids, material_indices
```

Call context: per part during STEP import, whenever per-face materials are bound.

##### F18 — STEP free-edge detection is O(edges × face_edges) via `IsSame` scans

The mixed construction-curve detector (`fascat/io/step.py:5170-5197`) first collects
every edge of every face into a Python list, then checks each candidate edge against
that list — and against the growing result list — one `IsSame` call at a time:

```python
free_edges: list[Any] = []
edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
while edge_explorer.More():
    edge = TopoDS.Edge_s(edge_explorer.Current())
    if not any(edge.IsSame(face_edge) for face_edge in face_edges) and not any(
        edge.IsSame(existing) for existing in free_edges
    ):
        free_edges.append(edge)
    edge_explorer.Next()
```

Every `IsSame` is an OCP boundary crossing, and shared edges (the common case — interior
edges belong to two faces, so `face_edges` holds duplicates) make the list longer than
the edge count. A shape with E edges pays O(E × |face_edges|) OCP calls; on edge-rich
imported geometry this is quadratic in practice. Fix: use OCCT's own shape-hashing map,
whose `Contains()` honors `IsSame` semantics as an O(1) hash lookup:

```python
from OCP.TopTools import TopTools_IndexedMapOfShape
from OCP.TopExp import TopExp

face_edge_map = TopTools_IndexedMapOfShape()
face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
while face_explorer.More():
    TopExp.MapShapes_s(face_explorer.Current(), TopAbs_EDGE, face_edge_map)
    face_explorer.Next()
if face_edge_map.IsEmpty():
    return None

free_edges: list[Any] = []
free_edge_map = TopTools_IndexedMapOfShape()
edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
while edge_explorer.More():
    edge = TopoDS.Edge_s(edge_explorer.Current())
    if not face_edge_map.Contains(edge) and not free_edge_map.Contains(edge):
        free_edge_map.Add(edge)
        free_edges.append(edge)
    edge_explorer.Next()
```

This also deduplicates `face_edges` for free (the map stores each shape once). Do
**not** key on Python `id()`: OCP wrapper objects are transient across explorer
iterations (see audit notes). Call context: per part, when a shape mixes faces and
standalone construction edges.

##### F19 — CLI startup imports numpy, `io/step.py`, and `fascat.analysis` eagerly

Measured 2026-07-05: `import fascat.cli` takes ~129 ms, and `python -X importtime`
attributes ~59 ms of it to the `fascat.analysis` subtree alone
(analysis → asset → export_report → mesh → options, plus numpy at ~26 ms). Every
`fascat` invocation — including `--help` and `--version` — pays this. Three eager
chains, all in the `fascat/cli.py` header:

```python
from fascat.analysis import AnalysisReport, analyze_output   # :22
from fascat.io.brep import BREP_SUFFIXES, read_brep          # :24
from fascat.io.fbx import FBX_SUFFIXES                       # :25
from fascat.io.gltf import GLTF_SUFFIXES                     # :26
from fascat.io.iges import IGES_SUFFIXES, read_iges          # :27
from fascat.io.obj import OBJ_SUFFIXES                       # :28
from fascat.io.step import read_step, read_step_bytes        # :29
from fascat.io.stl import STL_SUFFIXES                       # :30
```

- `AnalysisReport` appears **only** inside a return-type annotation (`:3889`), and
  `from __future__ import annotations` is active, so the annotation is never evaluated —
  the import exists purely for the type checker. `analyze_output` runs only on the
  analyze path.
- `read_step`/`read_brep`/`read_iges` pull in `fascat/io/step.py` — 7k lines with
  `import numpy` at `:16` — before any command has been chosen.
- The `*_SUFFIXES` constants are small frozen sets, but they live in modules that import
  numpy at top (`fascat/io/gltf.py:16`, `io/fbx.py:7`, `io/obj.py:6`, `io/stl.py:7`),
  so importing the constant loads the writer.

Fix:

```python
# fascat/io/_suffixes.py — new, zero-dependency
BREP_SUFFIXES = frozenset({".brep", ...})    # moved verbatim from each io module,
FBX_SUFFIXES = frozenset({".fbx"})           # re-exported there for compatibility
...

# fascat/cli.py
from typing import TYPE_CHECKING
from fascat.io._suffixes import (BREP_SUFFIXES, FBX_SUFFIXES, GLTF_SUFFIXES,
                                 IGES_SUFFIXES, OBJ_SUFFIXES, STL_SUFFIXES)
if TYPE_CHECKING:
    from fascat.analysis import AnalysisReport

def _validate_and_analyze_output_for_cli(...) -> tuple[dict[str, int], AnalysisReport | None]:
    from fascat.analysis import analyze_output               # deferred to the analyze path
    ...

def _read_cad_input(path: Path, ...):
    from fascat.io.step import read_step                     # deferred to command bodies
    ...
```

Pin the fix in the existing contract: `tests/test_lazy_imports.py` already forbids numpy
after `import fascat` (`:15-23`), but the CLI test (`:26-33`) excludes only
runtime/visual/PIL — extend its exclusion list:

```python
"heavy = [name for name in ('fascat.runtime', 'fascat.visual', 'PIL',"
" 'numpy', 'fascat.io.step', 'fascat.analysis') if name in sys.modules]\n"
```

Expected result: `import fascat.cli` drops to typer + rich + stdlib (~40 ms), a
~90 ms/invocation saving, and heavy imports move to first use inside commands.

##### F20 — every `Asset(...)` construction deep-copies all geometry

`Asset.__post_init__` (`fascat/asset.py:327-334`) defensively copies the entire object
graph on every construction:

```python
def __post_init__(self) -> None:
    self.root = self.root.copy()
    self.parts = {part_id: part.copy(keep_source=True) for part_id, part in self.parts.items()}
    self.materials = {material_id: material.copy() for material_id, material in self.materials.items()}
    self.images = {image_id: image.copy() for image_id, image in self.images.items()}
    self.metadata = dict(self.metadata)
    self.pmi = [annotation for annotation in self.pmi]
    self.report = self.report.copy()
```

and `Part.copy(keep_source=True)` reaches `Mesh.copy()` (`fascat/mesh.py:306-320`),
which clones **every** numpy array — points, faces, normals, tangents, all UV channels,
material indices, and all face groups:

```python
mesh = Mesh._adopt(
    points=self.points.copy(),
    faces=self.faces.copy(),
    normals=None if self.normals is None else self.normals.copy(),
    tangents=None if self.tangents is None else self.tangents.copy(),
    uvs={channel: values.copy() for channel, values in self.uvs.items()},
    ...
)
```

So constructing an `Asset` costs a full copy of all scene geometry, and a pipeline run
constructs Assets at every stage — total bytes copied scale with **pipeline depth ×
scene size**. For a 500 MB-geometry assembly through a ~10-stage pipeline that is
gigabytes of pure memcpy plus the allocator pressure that comes with it. The escape
hatch already exists: `Asset._adopt` (`fascat/asset.py:336`) constructs without copying,
exactly like `Mesh._adopt` above.

This is a deliberate immutability/ownership design, so treat it as architectural rather
than a drop-in fix:

1. First profile it on the 10k+ part corpus (§3 open item) to size the win — if
   conversion time is dominated by tessellation/import, this can stay P3.
2. Then migrate internal pipeline stages that already own their intermediates (built a
   fresh `parts` dict, never alias caller data) from `Asset(...)` to `Asset._adopt(...)`
   — the same pattern `Mesh.copy` itself uses.
3. Alternatively add copy-on-write to `Mesh`: mark arrays read-only at adopt time
   (`array.setflags(write=False)`) and copy lazily in mutators.

Behavior-sensitive: write aliasing tests first (mutating a returned Asset must not
corrupt the input Asset) so the `_adopt` migration can't silently introduce sharing
bugs.

##### F21 — spatial-bucket neighbor search is per-vertex Python with scalar norms

Four sites in `fascat/mesh.py` share a hand-rolled 3×3×3-cell grid walk — `:800`
(`_near_duplicate_unmerged_stats`), `:878`, `:1356`, `:1483` (the merge-vertices,
boundary-gap, and stitch paths). Representative shape:

```python
buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
for vertex, point in enumerate(self.points):
    key = self._spatial_bucket_key(point, upper)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbor_key = (key[0] + dx, key[1] + dy, key[2] + dz)
                for other in buckets.get(neighbor_key, []):
                    pair = (min(vertex, other), max(vertex, other))
                    if pair in connected_edges:
                        continue
                    distance = float(np.linalg.norm(point - self.points[other]))
                    if lower < distance <= upper:
                        near_pairs += 1
    buckets[key].append(vertex)
```

Per vertex: a Python bucket-key computation, 27 dict probes, and one **scalar**
`np.linalg.norm` call (~µs each) per candidate neighbor — all interpreter-bound. The
algorithm is the right complexity class; the constant factor is 100–1000× a vectorized
equivalent. Fix: scipy is already a dependency (connected components); its cKDTree
returns all close pairs in one call, and the distance filter vectorizes:

```python
from scipy.spatial import cKDTree

pairs = cKDTree(self.points).query_pairs(upper, output_type="ndarray")   # (P, 2) int64
distances = np.linalg.norm(self.points[pairs[:, 0]] - self.points[pairs[:, 1]], axis=1)
mask = distances > lower
# _near_duplicate_unmerged_stats: drop connected pairs, then count/min vectorized
keys = pairs[mask]
unconnected = np.array([tuple(row) not in connected_edges for row in keys.tolist()])
near_pairs = int(np.count_nonzero(unconnected))
nearest = float(distances[mask][unconnected].min()) if near_pairs else 0.0
```

(`query_pairs` returns each unordered pair once with `i < j`, matching the
`(min, max)` canonicalization above; if `connected_edges` grows large, replace the
membership comprehension with a sorted structured-view `np.isin`.) Apply per site — the
merge/stitch variants differ in what they do with a found pair, not in the search — and
keep a grid only if a site's tests pin an exact visit order.

##### F22 — per-face `.astype(int).tolist()` loops survive in mesh repair/analysis

The same shape F8/F12 removed elsewhere. The dominant pattern is the per-face edge-map
build, copied three times in `fascat/ops/actions.py` — `:1917-1919` (`_edge_faces`),
`:1998-2002` (`_boundary_loops`), `:2609-2611` (keep-frontier edge map):

```python
for face_index, face in enumerate(mesh.faces.astype(int).tolist()):
    for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
        edge_faces.setdefault(_edge_key(start, end), []).append(face_index)
```

That is O(F) list materialization plus 3F tuple allocations and dict probes in Python.
Build the edge table once, vectorized, in a shared helper (same home as the F9/F10
helpers) and derive all three consumers from its grouped form:

```python
def edge_incidence(faces: IntArray) -> tuple[IntArray, IntArray]:
    """Sorted (E*3, 2) canonical edges and the face index of each row."""
    faces64 = faces.astype(np.int64, copy=False)
    edges = np.concatenate([faces64[:, [0, 1]], faces64[:, [1, 2]], faces64[:, [2, 0]]])
    edges.sort(axis=1)                                   # canonical (min, max) per row
    face_ids = np.tile(np.arange(faces64.shape[0], dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    return edges[order], face_ids[order]
```

Consumers then use `np.diff`-based group boundaries: boundary edges are groups of size
1 (`_boundary_loops`), face adjacency reads face ids within a group (`_edge_faces`, the
keep-frontier map). Remaining sites, same disease, individual cures:

- `fascat/mesh.py:1029-1032` — the material-signature build converts each face row
  inside the loop (`for vertex_index in face.astype(int).tolist():`); faces are already
  int64, so at minimum iterate the row directly, or vectorize the whole build as unique
  (vertex, material) pairs: `np.unique(np.stack([self.faces.reshape(-1),
  np.repeat(self.material_indices, 3)], axis=1), axis=0)`.
- `fascat/mesh.py:1378-1383` — `split_t_junctions` round-trips `points`, every UV
  channel, and `faces` through Python lists **up front**, before knowing whether any
  T-junction exists; splits are rare. Keep the arrays, collect only the appended
  vertices/faces in small Python lists, and `np.concatenate` once at the end.
- `fascat/mesh.py:1684-1686` — orientability edge map iterates
  `self.faces.astype(int).tolist()`; use `edge_incidence` above (it preserves the
  per-face directed order needed for the orientation check via `face_ids`).
- `fascat/mesh.py:2306-2327` — UV-seam metrics: `_rounded_key` is called once per vertex
  and per UV in list comprehensions, and the edge loop runs
  `edge_lengths.setdefault(edge_key, float(np.linalg.norm(...)))` (`:2327`), which
  evaluates the norm argument **even when the key is already present**. Round in one
  vectorized pass and precompute all lengths:

  ```python
  position_keys = list(map(tuple, np.round(self.points, decimals).tolist()))
  uv_keys = list(map(tuple, np.round(self.uvs[channel], decimals).tolist()))
  edges, _ = edge_incidence(self.faces)
  lengths = np.linalg.norm(self.points[edges[:, 0]] - self.points[edges[:, 1]], axis=1)
  ```

Behavior-preserving throughout; the existing repair/analysis tests pin the outputs.

##### F23 — export-writer conversion churn

Small allocations repeated per accessor/node/face on every export. Individually minor;
together they are the writers' Python overhead. Per site:

- `fascat/io/gltf.py:894-895`, `:1212-1213`, `:1222-1223` — accessor min/max run
  `.astype(...)` before `.tolist()`; the `astype` allocates a full intermediate copy and
  `.tolist()` already yields Python floats/ints of the right type. Same on per-node 4×4
  transforms at `:1721`, `:1810`:

  ```python
  # before
  minimum=points.min(axis=0).astype(float).tolist(),
  gltf_node["matrix"] = transform.T.reshape(-1).astype(float).tolist()
  # after — drop the astype; dtype is already float64 (quantized path: int via .tolist())
  minimum=points.min(axis=0).tolist(),
  gltf_node["matrix"] = transform.T.reshape(-1).tolist()
  ```

- `fascat/io/usd.py:569` — uniqueness check materializes every face index as a Python
  int; `:574` — the per-material face list round-trips through `.astype(int).tolist()`
  (USD's `Vt` arrays accept numpy/int sequences; keep `.tolist()` only if the binding
  demands it); `:450` — `Gf.Matrix4d(child.transform.tolist())` per node (check the
  Gf/Vt buffer-protocol constructors):

  ```python
  # before
  if mesh.material_indices is None or len(set(mesh.material_indices.astype(int).tolist())) <= 1:
  # after
  if mesh.material_indices is None or np.unique(mesh.material_indices).size <= 1:
  ```

- `fascat/io/obj.py:206` — one Python function call (`_face_material_id`, `:260-266`:
  bounds check + numpy scalar indexing + `int()`) per triangle:

  ```python
  # before
  material_ids = [_face_material_id(part, mesh, face_index) for face_index in range(mesh.triangle_count)]
  # after — resolve the lookup table once, index in bulk
  lookup = list(part.material_ids)
  if not lookup:
      material_ids = [None] * mesh.triangle_count
  elif mesh.material_indices is None:
      material_ids = [lookup[0]] * mesh.triangle_count
  else:
      material_ids = [
          lookup[index] if index < len(lookup) else None
          for index in mesh.material_indices.tolist()
      ]
  ```

- `fascat/ops/tessellate.py:334-340` — construction-curve segment lengths are computed
  one scalar `np.linalg.norm` per sample pair; stack and vectorize:

  ```python
  # after
  samples_array = np.asarray(samples, dtype=np.float64)          # (N+1, 3)
  lengths = np.linalg.norm(np.diff(samples_array, axis=0), axis=1)
  keep = lengths > _CONSTRUCTION_CURVE_MIN_SEGMENT_LENGTH
  segments.extend(zip(samples_array[:-1][keep], samples_array[1:][keep], strict=True))
  ```

- `fascat/ops/tessellate.py:356-410` — `_tube_mesh_from_segments` appends per-point
  numpy arrays with `math.cos`/`math.sin` per side in the inner loop. The ring is
  identical for every segment; compute it once and broadcast:

  ```python
  angles = 2.0 * math.pi * np.arange(sides, dtype=np.float64) / sides
  cos_a, sin_a = np.cos(angles)[:, None], np.sin(angles)[:, None]      # (sides, 1)
  # per segment (u_axis/v_axis depend on direction):
  offsets = radius * (cos_a * u_axis + sin_a * v_axis)                 # (sides, 3)
  ring_points = np.concatenate([start + offsets, end + offsets])       # (2*sides, 3)
  ```

  and build the two-triangles-per-side index block once as a template, offset per
  segment.

- `fascat/ops/tessellate.py:465-466` — free-edge JSON converts each endpoint separately;
  gather both sides in bulk:

  ```python
  # before
  for start, end in boundary_edges[:max_segments].tolist():
      segments.append([mesh.points[start].astype(float).tolist(), mesh.points[end].astype(float).tolist()])
  # after — one gather, one conversion
  clipped = boundary_edges[:max_segments]
  segments = np.stack([mesh.points[clipped[:, 0]], mesh.points[clipped[:, 1]]], axis=1).tolist()
  ```

##### F24 — small per-run overheads

Each is once-per-conversion (or once-per-candidate) rather than per-element; listed
together because every fix is a few lines.

- `fascat/export_report.py:82` — `export_image_counts` recomputes
  `_source_image_ids(asset)` (a full pass over `asset.images`) inside a union with
  `source_unique`, which was built from that same call three lines up (`:79`), so the
  union is a no-op:

  ```python
  # before
  source_unique = _source_image_ids(asset) | set(source_refs)          # :79
  ...
  "export_source_image_count": len(_source_image_ids(asset) | source_unique),  # :82
  # after
  "export_source_image_count": len(source_unique),
  ```

- `fascat/pipeline.py:1501-1504` — `_measured_report_timings` makes three full passes
  over `report.steps`:

  ```python
  # before
  pipeline_ms = int(round(sum(step.duration for step in asset.report.steps) * 1000.0))
  write_ms = int(round(sum(step.duration for step in asset.report.steps if step.name == "write") * 1000.0))
  validate_ms = int(round(sum(step.duration for step in asset.report.steps if step.name == "validate") * 1000.0))
  # after — one pass, same rounding (round the float sums at the end)
  pipeline = write = validate = 0.0
  for step in asset.report.steps:
      pipeline += step.duration
      if step.name == "write":
          write += step.duration
      elif step.name == "validate":
          validate += step.duration
  ```

- `fascat/pipeline.py:1467-1472` + callers at `:1013`, `:1027`, `:1476` — every
  budget-report metric independently calls `_material_texture_summaries(asset)`, a full
  iteration over all materials (`:1537-1538`):

  ```python
  # after — compute once in _add_profile_budget_report and thread through
  summaries = _material_texture_summaries(asset)
  texture_resolutions = [resolution for resolution, _count in summaries]
  texture_count, estimated_bytes = _texture_summary_totals(summaries)
  ```

- `fascat/runtime.py:1091`, `:1148` — the preview decoders `deepcopy` the entire glTF
  JSON document, but the subsequent mutations touch only the `images` list entries,
  `bufferViews`, and the extension arrays:

  ```python
  # before
  document = deepcopy(source_document)
  # after — copy exactly the containers the decoder mutates
  document = dict(source_document)
  document["images"] = [dict(image) for image in source_document.get("images", [])]
  document["bufferViews"] = [dict(view) for view in source_document.get("bufferViews", [])]
  ```

  (audit `_promote_basis_texture_sources` / `_remove_gltf_extension` /
  `_embed_preview_buffers` / `_rewrite_external_image_uris` for the full mutated-key
  set before landing). And `:1121-1124` writes the document to disk then immediately
  re-reads and re-parses it just to validate:

  ```python
  # before
  output_path.write_text(json.dumps(document), encoding="utf-8")
  decoded_document = _read_gltf_json_document(output_path)
  if _document_uses_basis_textures(decoded_document):
      raise RuntimeError("alktx2 KTX2 decode preserved KHR_texture_basisu")
  # after — validate in memory, then write
  if _document_uses_basis_textures(document):
      raise RuntimeError("alktx2 KTX2 decode preserved KHR_texture_basisu")
  output_path.write_text(json.dumps(document), encoding="utf-8")
  ```

  Preview harness only; lowest priority of the batch.

- `fascat/ops/actions.py:261` + `:2111-2130` — the occlusion-removal loop calls
  `_candidate_occluders` once **per candidate**, and that function rebuilds the filtered
  occluder tuple, both bounds vstacks, and a **full BVH** each time:

  ```python
  for candidate in selected_occurrences:                     # :261 — N times
      occluders = _candidate_occluders(result, candidate, occurrences, options)
      # inside: tuple filter over all occurrences, two np.vstack, _build_occurrence_bvh
  ```

  The only per-candidate difference is excluding the candidate itself — the
  transparency filter is candidate-invariant. Build the opaque set and one BVH before
  the loop, and skip self-hits by node id at query time:

  ```python
  opaque = tuple(
      occluder for occluder in occurrences
      if options.consider_transparency_opaque or not _part_is_transparent(asset, occluder.part_id)
  )
  bvh = _build_occurrence_bvh(
      np.vstack([occluder.bounds_min for occluder in opaque]),
      np.vstack([occluder.bounds_max for occluder in opaque]),
  )
  for candidate in selected_occurrences:
      ...  # query bvh; discard hits where occurrence.node.id == candidate.node.id
  ```

  Turns O(N² log N) BVH construction into O(N log N); the self-exclusion moves from
  set-construction time to query time with identical results.

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
