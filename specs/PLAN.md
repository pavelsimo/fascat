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
and left this plan; new that audit: one correctness item, two robustness items, five
performance findings F8–F12, plus docs-staleness and test-coverage items.

Re-audit 2026-07-05 (five parallel passes over mesh/analysis, pipeline/runtime/asset,
CLI/options, `io/`, and `ops/`, every high-impact claim re-verified by hand): the F8–F12
batch is **implemented** and left the open list. New that audit: twelve performance
findings F13–F24. Implementation pass 2026-07-05: F13–F19 and F21–F24 shipped in
separate focused commits; the 2026-07-03 correctness, robustness, docs, and coverage
items shipped alongside.

Re-audit 2026-07-07 (three parallel performance-only passes over mesh/analysis/asset,
`ops/`, and io/+pipeline/runtime/CLI, every high-impact claim re-verified by hand): the
F13–F24 batch is **implemented** (⚡️ commits `b969a41`…`bcd4cd7`) and has left the plan,
as have the last open correctness (§1), robustness (§2), production-defaults (§6), and
packaging (§8) items. New this audit: performance findings F25–F35 (§3) — including two
surviving instances of previously fixed patterns (the pre-F14 OCP extraction in
`ops/heal.py`, the unshipped F22 sub-item in the UV-seam metrics) and the carried-forward
stage-2 follow-ups to F13/F14. Six agent claims were refuted and recorded in the audit
notes.

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

No open items.

## 2. Robustness & input hardening

No open items.

## 3. Performance

- [ ] **P2** Fix the 2026-07-07 hot-path findings — heal's pre-F14 OCP extraction
  (F25), the bake-materials per-face Python fills (F28), and the STEP per-face
  material/color loop (F29); details below
- [x] **P3** Close OCP batch triangulation access (F26) as pure-Python infeasible —
  evidence recorded below; a small compiled helper remains possible future work
- [ ] **P3** F13 stage-2 follow-up — AO direction-batching/BVH pruning (F27);
  details below
- [ ] **P3** Fix the 2026-07-07 scaling and cleanup findings F30–F35 — STEP BFS
  queues, the surviving UV-seam-metrics loops, material-index remap, UV-seam vertex
  grouping, export-report double pass, and small writer cleanups; details below
- [~] **P2** Profile the per-construction Asset geometry deep-copies (F20) before
  changing copy ownership — synthetic profile recorded below; real 10k+ corpus profile
  and ownership redesign still open
- [ ] **P3** Profile on real large CAD corpora (10k+ parts) and record numbers in the
  baseline list below — fixture baselines recorded there (2026-06-14 corpus, 2026-07-01
  re-run); actual 10k+ part corpus profiling still needed

### Performance Findings

Audit of 2026-07-07: the 2026-07-05 findings F13–F24 were verified **shipped** (see
verification below) and are gone from the open list, joining F1–F12 and the scipy
connected-components rewrite. New findings F25–F35 below; F25 and F31 are surviving
instances of previously fixed patterns in paths the earlier batches never touched. F26
has since been closed as pure-Python infeasible; F27 remains the explicitly deferred
stage-2 half of F13.

#### Baseline

- Fixture baseline (2026-06-14, 8 real STEP fixtures): **75.19 s total / 665.3 MiB peak RSS**
  — import 31.80 s, LOD 20.53 s, tessellation 8.39 s, repair 5.36 s, stage 5.01 s,
  optimize 3.84 s, write 0.11 s.
- Re-run 2026-07-01 (`make benchmark`, `tests/fixtures/vertical-screw.step` → GLB):
  **3.69 s total / 370.5 MiB peak RSS** — import 0.87 s, tessellate 1.57 s, LOD 0.49 s,
  repair 0.34 s, stage 0.21 s, optimize 0.19 s, write 0.004 s.
- Still open (§3 above): profiling on a real 10k+ part corpus, and a fresh
  `make benchmark` re-run to capture the F13–F24 batch (the 2026-07-01 numbers predate
  it).

#### Verification of prior findings (2026-07-07)

All twelve 2026-07-05 findings are implemented (current locations / commits):

- F13 — `_ray_hits_mesh` vectorized Möller–Trumbore: `fascat/ops/actions.py:843-869`
  (`1aaf2f3`); the triangle gather is hoisted out of the ray loop.
- F14 (stage 1) — `_transformed_occt_nodes` / `_occt_transform_matrix` /
  `_triangulation_faces`: `fascat/ops/tessellate.py:295-319` (`09d697f`); no more
  per-vertex `.Transformed()`, winding flip hoisted.
- F15 — self-intersection candidate pruning: `463f79d`.
- F16 — UV-overlap grid pruning in `uv_layout_stats`: `fa6272c`.
- F17 — STEP material binding via `{material_id: index}` dict: `b969a41`.
- F18 — STEP free-edge detection via `TopTools_IndexedMapOfShape`: `1f8a785`.
- F19 — CLI heavy imports deferred (`fascat/io/_suffixes.py`, TYPE_CHECKING analysis
  import, command-body io imports): `209a678`.
- F21 — `_scipy_close_point_pairs` cKDTree fast paths, gated ahead of the bucket walk
  (`fascat/mesh.py:1074`): `f69ce7d`.
- F22 — shared `edge_incidence` helper replacing the per-face `.astype(int).tolist()`
  edge-map builds: `3a0bf4f`. (One listed sub-site did not ship — see F31.)
- F23 — export-writer conversion churn: `374f81c`.
- F24 — per-run reporting overhead (duplicate `_source_image_ids`, single-pass report
  timings, hoisted occluder BVH, preview-decoder copies): `2a7774a`.

F20 remains `[~]`: synthetic profile recorded (see below), real-corpus profile open.

#### Open findings

F25 sits on the heal/repair path; F27/F28 are the entire cost of the `bake-materials`
action; F29/F30 bite on large styled or PMI-rich STEP imports; the rest bite on analysis
of large meshes or add up as per-run overhead. F26 is retained below as closure evidence.

##### F25 — heal face-overlap extraction still uses the pre-F14 per-vertex OCP pattern

`_face_overlap_descriptor` (`fascat/ops/heal.py:578-593`) is a surviving verbatim
instance of the pattern F14 removed from tessellate.py:

```python
nodes = triangulation.MapNodeArray()
node_lower = int(nodes.Lower())
points = np.empty((node_count, 3), dtype=np.float64)
for local_index in range(node_count):
    point = nodes.Value(node_lower + local_index).Transformed(transform)
    points[local_index] = (point.X(), point.Y(), point.Z())
...
for local_index in range(triangle_count):
    a, b, c = triangles.Value(triangle_lower + local_index).Get()
    if reversed_face:
        face_triangles[local_index] = (c - 1, b - 1, a - 1)
    else:
        face_triangles[local_index] = (a - 1, b - 1, c - 1)
```

~6 OCP boundary crossings per vertex — `.Transformed()` constructs a new `gp_Pnt` and
round-trips through the wrapper per vertex — plus a Python branch per triangle. Called
once **per face** by `_face_overlap_descriptors` (`fascat/ops/heal.py:551`) on the heal
path, so dense tessellations pay the full pre-F14 cost during repair. Fix: reuse the F14
helpers that already exist — `_transformed_occt_nodes`, `_occt_transform_matrix`,
`_triangulation_faces` (`fascat/ops/tessellate.py:295-319`). Move them to a shared
module (e.g. `fascat/ops/_occt_mesh.py`), re-export from tessellate for compatibility,
and apply the `reversed_face` flip as a `[:, ::-1]` column slice after the bulk read.
Impact: HIGH on repair/heal for dense tessellations; behavior-preserving (the heal tests
pin the descriptors).

##### F26 — OCP batch triangulation access is closed as pure-Python infeasible

`fascat/ops/tessellate.py:297-299` (nodes), `:317-318` (triangles), and the UV-node loop
(`_triangulation_uv_nodes`, `:~468`). Stage 1 shipped — the transform is applied
vectorized and the winding flip is hoisted — but one `Value()` call plus coordinate
getters per vertex (~4 crossings) and one `Value().Get()` per triangle (2 crossings)
remain on the **main tessellate path**. Investigation in the current environment closes
the pure-Python stage-2 path as infeasible:

- `cadquery-ocp` is 7.9.3.1.
- `Poly_Triangulation.MapNodeArray()` returns `TColgp_HArray1OfPnt`;
  `MapTriangleArray()` returns `Poly_HArray1OfTriangle`.
- `InternalNodes()` and `InternalUVNodes()` still expose wrapper arrays with per-index
  `Value()` access.
- The node, triangle, and internal arrays expose no NumPy/buffer bridge: no `__array__`,
  no `__buffer__`, `memoryview(...)` fails, and `np.asarray(...)` produces a scalar
  `object` array containing the wrapper rather than numeric coordinates.
- `RWMesh`/`RWStl` APIs are reader/writer/iterator oriented (`RWMesh_FaceIterator`,
  `RWMesh_TriangulationReader`, `RWStl.ReadFile_s`/`Write*`) and are not a safe
  substitute for Fascat's tessellation extraction: they do not preserve the per-face
  groups, material indices, shape transforms, and CAD UVs that the current path carries
  through `Poly_Triangulation`.
- `BRepTools` exposes triangulation helpers such as `Triangulation_s`,
  `LoadTriangulation_s`, `LoadAllTriangulations_s`, and `ActivateTriangulation_s`, but
  these verify, load, or activate existing face triangulation data; they do not expose a
  numeric bulk-copy path from OCCT wrapper arrays into Python/NumPy.

Conclusion: keep the stage-1 Python extraction as the supported pure-Python path. A small
compiled helper could still bulk-copy OCCT arrays in a future performance pass, but that
is out of scope for F26.

##### F27 — AO bake outer loop is still O(F² × D) (F13 stage 2/3)

`_face_ambient_occlusion` (`fascat/ops/actions.py:824-834`). F13 vectorized the inner
ray test, but the outer loop is still one Python iteration per (face × direction) with a
scalar `np.dot` gate:

```python
for face_index, (centroid, normal) in enumerate(zip(centroids, normals, strict=True)):
    ...
    for direction in directions:
        if float(np.dot(direction, normal)) <= 0.0:
            continue
        tested += 1
        if _ray_hits_mesh(origin, direction, triangles, ignore_face=face_index, max_t=ray_length):
            hits += 1
```

and each `_ray_hits_mesh` call is a full O(F) vectorized pass over all triangles — total
O(F² × D) numpy work with F × D Python iterations and no spatial pruning. Fix in two
steps: (1) batch directions per face — precompute `dots = directions_arr @ normals.T`
once (D × F), and evaluate all valid directions for a face in one broadcast
Möller–Trumbore (D_valid, F) pass, reducing per face with `np.any(..., axis=1)`;
(2) if still hot on very large meshes, add AABB/BVH pruning per ray (reuse the
`_build_occurrence_bvh` idiom on per-triangle bounds). Keep the `1e-8 < t < max_t`
window and the ignore-self semantics exactly — the bake tests pin them. Impact: HIGH for
`bake-materials` on non-toy meshes; not on the default pipeline.

##### F28 — bake-materials per-face Python fills

Two sites, same cure — compute per unique material index, not per face:

- `fascat/ops/actions.py:724-737` — the base_color / metallic_roughness / emissive fill
  calls `_face_material(...)` once per face and allocates per face:

  ```python
  for face_index in range(mesh.triangle_count):
      material = _face_material(asset, part, mesh, face_index)
      ...
      result[face_index] = np.asarray([_color_byte(value) for value in values], dtype=np.uint8)
  ```

  Fix: build a small `(len(part.material_ids) + 1, 4)` uint8 LUT per kind (one
  `_color_byte` pass per material, last row = the None-material fallback), map
  out-of-range/missing indices to the fallback row, then
  `result = lut[safe_indices]` — an O(F) numpy gather instead of O(F) Python calls.

- `fascat/ops/actions.py:771-798` — `_emissive_provenance_metadata` loops over **every
  face of every part** calling `_face_material` + `_emissive_color_with_source`, which
  **string-parses `material.metadata["emissive_color"]` per face**. Fix: classify each
  unique material once (material vs fallback), then count faces per material index via
  `np.unique(mesh.material_indices, return_counts=True)` (all faces of a part share the
  base material when `material_indices is None`).

Impact: MEDIUM-HIGH — this is per-face work on every `bake-materials` run.

##### F29 — STEP per-face material/color extraction does unconditional OCP work

`_face_material_ids` (`fascat/io/step.py:5574-5594`), per part on the styled-STEP import
path:

```python
while explorer.More():
    face = TopoDS.Face_s(explorer.Current())
    spec = _shape_visual_material_spec(vis_material_tool, face, options)
    sub_label = TDF_Label()
    found_sub_label = shape_tool.FindSubShape(shape_label, face, sub_label)
    if spec is None and found_sub_label:
        spec = _label_visual_material_spec(vis_material_tool, sub_label, options)
    color = _shape_color(color_tool, face)
    if spec is None and color is None and found_sub_label:
        color = _label_color(sub_label)
```

`FindSubShape` is called **unconditionally** per face even though its result is only
consumed when `spec is None`; if it scans the label tree per call, a part with F faces
pays O(F × labels). `_shape_color` (`:5597-5604`) additionally constructs a fresh
`Quantity_Color` and calls `GetColor` up to twice per face. Fix: (1) defer
`FindSubShape` — call it only when `spec is None`; (2) hoist one reusable
`Quantity_Color` out of the loop; (3) investigate whether XCAF exposes a sub-shape label
map per shape label (e.g. iterating the shape label's children once into a
`TopTools`-keyed dict) to replace per-face `FindSubShape` entirely. Impact: MEDIUM-HIGH
on import of styled assemblies (thousands of faces per part). Behavior-preserving; the
STEP color/material tests pin the outputs.

##### F30 — `list.pop(0)` BFS queues in STEP scanning

Two sites shift the whole list on every dequeue:

- `fascat/io/step.py:936` — external-reference traversal:
  `current, path_keys = queue.pop(0)`
- `fascat/io/step.py:2194` — PMI semantic-graph traversal:
  `record_id = pending_ids.pop(0)` (this queue can hold every reachable record id on
  PMI-rich files)

O(n) per pop → O(n²) on deep reference chains / large PMI graphs. Fix: two-line change
per site — `collections.deque` + `popleft()`. Both are FIFO today; preserve visit order.

##### F31 — UV-seam metrics: the F22 sub-item for this site never shipped

`fascat/mesh.py:2516-2517` and `:2538`, inside the (cached) seam-metrics computation.
The original F22 entry specified this fix, but the shipped batch only landed the
`edge_incidence` half:

```python
position_keys = [_rounded_key(point, decimals) for point in self.points]
uv_keys = [_rounded_key(uv, decimals) for uv in self.uvs[channel]]
...
edge_lengths.setdefault(edge_key, float(np.linalg.norm(self.points[start] - self.points[end])))
```

`_rounded_key` (`fascat/mesh.py:72-73`) runs its own `np.round` on a 3-vector plus
`.tolist()` per call (~µs each), once per vertex and per UV — O(V) scalar-shaped numpy
calls; and the `setdefault` evaluates the norm argument even when the key already
exists. Fix as originally specified: round in one vectorized pass —
`position_keys = list(map(tuple, np.round(self.points, decimals).tolist()))`, same for
UVs — and precompute all edge lengths in one
`np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)` pass, indexed by
edge row. The result is cached per channel + tolerance + geometry token (F7), so this is
first-computation cost per analyzed mesh. Impact: MEDIUM on analysis of large UV'd
meshes.

##### F32 — `_compact_material_slots` remaps material indices in Python, per face

`fascat/ops/actions.py:2752-2761`:

```python
remap = {old: new for new, old in enumerate(used)}
mesh.material_indices = np.asarray([remap[int(index)] for index in mesh.material_indices], dtype=np.int64)
```

Iterates the numpy array directly — boxing a numpy scalar per element, the exact
anti-pattern the audit notes call out — with a dict lookup per face. Fix: lookup array —
`lookup = np.empty(len(material_ids), dtype=np.int64)`,
`lookup[used] = np.arange(len(used), dtype=np.int64)`,
`mesh.material_indices = lookup[mesh.material_indices]` (bounds already validated at
`:2757`). Called on the occlusion-removal/compaction paths; O(F) numpy gather replaces
O(F) Python. Impact: MEDIUM.

##### F33 — `_uv_seam_vertices` groups vertices via per-vertex Python dict inserts

`fascat/mesh.py:3257-3268`:

```python
rounded = np.round(self.points, 9)
for index, point in enumerate(rounded.tolist()):
    by_position.setdefault((float(point[0]), float(point[1]), float(point[2])), []).append(index)
```

The rounding is vectorized but the grouping is O(V) Python dict work, followed by
per-group UV comparisons. Fix: vectorize the grouping —
`np.unique(rounded, axis=0, return_inverse=True)`, take groups from
`np.argsort(inverse)` split at count boundaries (`np.cumsum` of the unique counts); the
per-group UV-mismatch check becomes
`np.unique(np.round(channel_values[group], 9), axis=0).shape[0] > 1` per channel.
Callers: feature-group preservation for decimation (`fascat/mesh.py:3031`) and repair
stats (`fascat/ops/actions.py:1400`). Impact: MEDIUM on large meshes entering
feature-preserving decimation.

##### F34 — `referenced_material_ids` computed twice per export-stats call

`fascat/export_report.py:67` (`export_material_counts`) and `:104` (via
`referenced_materials`, called from `export_image_counts` at `:78`). Each call is a full
pass over all parts × LOD meshes with an `np.unique` per mesh — the same disease F24
fixed for `_source_image_ids`, one function over. Fix: compute once in the shared caller
and thread through, or give `export_image_counts`/`referenced_materials` an optional
`referenced: set[str] | None = None` parameter defaulting to recompute. Impact:
LOW-MEDIUM (per export-stats call, scales with part × LOD count).

##### F35 — small per-site cleanups

Each is a few lines; listed together:

- `fascat/io/gltf.py:1689-1697` — `_face_groups` runs one `np.flatnonzero` full scan per
  unique material (O(k × n)); a single `np.argsort(material_indices, kind="stable")` +
  boundary split makes it one O(n log n) pass. Only worth it for heavily multi-material
  meshes; keep the sanctioned `np.unique(...).tolist()` idiom for the loop itself.
- `fascat/io/step.py:1614` — `for key, value in list(result.items())` copies the items
  list; only values are reassigned in the body, so iterate `result.items()` directly
  (verify no key mutation first).
- `fascat/io/usd.py:576` — `np.flatnonzero(...).tolist()` per material subset; check
  whether `Vt.IntArray` / `CreateGeomSubset` accepts numpy via the buffer protocol and
  drop the `.tolist()` if so.
- `fascat/mesh.py:1363-1364` — the degenerate-face duplicate check builds
  `set(face.astype(int).tolist())` per degenerate face;
  `np.unique(self.faces[face_index]).size < 3` avoids both allocations (rare path,
  cheap fix).

##### F20 — every `Asset(...)` construction deep-copies all geometry (carried, `[~]`)

`Asset.__post_init__` (`fascat/asset.py:327-334`) copies the entire object graph —
root, all parts (through `Mesh.copy()`, which clones every numpy array), materials,
images, report — on every construction, so total bytes copied scale with **pipeline
depth × scene size**. The escape hatch exists: `Asset._adopt` (`fascat/asset.py:336`)
constructs without copying. This is a deliberate immutability/ownership design — treat
it as architectural, not a drop-in fix.

Synthetic profile (2026-07-05): a 10,000-part / 20,000-triangle minimal-mesh case copied
1.8 MiB in 28.4 ms median (`_adopt` baseline 4.4 µs); a 64-part / 524,288-triangle
full-attribute case copied 188.0 MiB in 26.1 ms median (~7.2 GiB/s; `_adopt` baseline
5.7 µs). `cProfile` attributed the cost to `Asset.__post_init__ → Part.copy →
Mesh.copy` (577 `ndarray.copy` calls). This confirms the copy cost but does not yet
justify changing copy-ownership semantics without the real 10k+ corpus profile.

Remaining work: (1) profile on the 10k+ part corpus (§3 open item); (2) if justified,
migrate internal pipeline stages that already own their intermediates to
`Asset._adopt(...)`, or add copy-on-write to `Mesh` (`array.setflags(write=False)` at
adopt time, lazy copy in mutators). Behavior-sensitive: write aliasing tests first.

#### Verification recipe (when implementing)

1. Focused tests per finding: the heal/repair tests (F25), `tests/test_actions.py` (F27,
   F28, F32), the STEP import tests (F29, F30), `tests/test_mesh.py` (F31, F33),
   `tests/test_export_report.py` (F34), the glTF/USD/writer tests (F35),
   `tests/test_asset.py` aliasing tests before touching F20. F26 is evidence-only
   closure; `tests/test_tessellate.py` remains the regression suite for the stage-1
   Python extraction path.
2. `make ci` for the full gate.
3. `make benchmark` before/after — F25 shows on repair of dense tessellations; F27/F28
   need a bake-materials scenario to measure; F29/F30 show on large styled/PMI-rich STEP
   imports only. Record a fresh baseline first — the 2026-07-01 numbers predate the
   F13–F24 batch.

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

No open items.

## 7. Testing & CI

- [ ] **P2** Real IGES/BREP fixtures (zero exist today; only synthetic geometry is tested)
- [~] **P2** stdin→stdout conversion integration tests (`-` paths) — the CLI supports `-`
  and two basic tests exist (`tests/test_cli.py:96,1447`); still missing: round-trip
  stdin→stdout for each writer format and the error paths (binary garbage on stdin,
  closed stdout)
- [ ] **P3** mypy over `tests/` (`pyproject.toml:98` scopes mypy to `fascat/` only)
- [ ] **P3** Golden-image corpora for engine previews (carried from the previous plan)

## 8. Packaging, docs & release

No open items.

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
  anyway. Retained only as the §1 P2 uninitialized-memory item (since shipped).
- Audit 2026-07-03: "`_tag_uv_layout_quality` recomputes per-channel UV stats with no
  cache" (former F7, was `[~]` in §3) — refuted for the working tree: `uv_layout_stats`
  and `uv_distortion_metrics` are cached in `fascat/mesh.py` via `_cached_value` keyed on
  channel + tolerance + geometry cache token; the op calling them per channel hits the
  cache. Closed.
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
- Audit 2026-07-07: "`_bucket_close_vertex_pairs` 27-cell grid walk is a HIGH hot path"
  (`fascat/mesh.py:1079-1098`) — refuted as HIGH: since F21, `_close_vertex_pairs` takes
  the `_scipy_close_point_pairs` cKDTree fast path first (`fascat/mesh.py:1074`); the
  bucket walk only runs when scipy is unavailable. Kept as fallback; not worth
  optimizing.
- Audit 2026-07-07: "`_connected_face_components_bfs` `.tolist()` edge map is HIGH"
  (`fascat/ops/hierarchy.py:561-562`) — refuted: it is the scipy-unavailable fallback;
  the scipy incidence-matrix path runs first (`:532`).
- Audit 2026-07-07: "remove `.tolist()` before Python loops" (`fascat/io/obj.py:265`,
  the mesh.py t-junction loops at `:1463/:1485/:1503/:1527`, `_rounded_key`) — refuted
  as stated: iterating a numpy array directly boxes a numpy scalar per element and is
  *slower* than `.tolist()`; `.tolist()`-then-iterate is the sanctioned idiom (see the
  F8 note above). The actionable form is vectorizing the *loop itself* where possible
  (F31/F33); the t-junction loops carry early-exit/stateful logic and stay Python.
- Audit 2026-07-07: "`faces.astype(np.int64, copy=False)` is wasteful"
  (`fascat/analysis.py:344`) — refuted: no-op-cost dtype guard; sanctioned idiom.
