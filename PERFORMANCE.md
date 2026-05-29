# Performance Audit

A living checklist of performance issues in the fascat conversion pipeline
(STEP → tessellate → repair → stage → optimize/decimate → LOD → export).

Status: **identified, not yet fixed.** We will work these one at a time. Each item has a
category, severity, code location, why it is slow, and a fix direction (not a full solution).

> **Measure before fixing (P19).** There is per-step timing in the report but no profiling or
> benchmark harness. Profile a representative STEP file first so effort lands on the real
> bottleneck rather than a guessed one.

## Priority summary

| #   | Item                                                            | Category      | Severity |
| --- | --------------------------------------------------------------- | ------------- | -------- |
| ~~P1~~ | ~~`t_junction_count` is O(edges × vertices)~~ ✅ **done**     | CPU           | High     |
| P2  | `repair()` runs a full before/after diagnostic suite, always-on | CPU           | High     |
| P3  | Occlusion ray cast has no acceleration structure                | CPU           | High     |
| P4  | Per-element Python loops in core mesh kernels                   | CPU           | High     |
| P11 | `Asset.copy()` duplicates every mesh array ~4–8×, ~10×/pipeline | Memory        | High     |
| P15 | Exporters build one Python object/string per vertex/triangle    | CPU / I/O     | High     |
| P18 | `stats()` / `walk()` / draw-call recomputed many times/stage    | System design | Medium   |
| P5  | Edge/adjacency maps rebuilt from `.tolist()` repeatedly         | CPU           | Medium   |
| P6  | `merge_vertices` recomputes components + heavy diagnostics      | CPU           | Medium   |
| P7  | Staging computes O(F²) UV overlap diagnostics per part          | CPU           | Medium   |
| P8  | Nearest-centroid material assignment is O(target × source)      | CPU           | Medium   |
| P9  | LOD levels simplified from full-res mesh, not progressively     | CPU           | Medium   |
| P10 | Tessellation: Python-list extraction + duplicated edge passes   | CPU           | Medium   |
| P12 | Whole-mesh fingerprint/digest recomputed after every op         | Memory        | Medium   |
| P13 | Export binary buffer copied several times                       | Memory        | Medium   |
| P14 | Default `validate_output` re-reads & re-parses the output       | I/O           | Medium   |
| P16 | Pipeline processes independent parts serially                   | Concurrency   | Medium   |
| P17 | No memoization of derived mesh topology across stages           | System design | Medium   |
| P19 | Metrics are heuristic estimates; no benchmark/profiling harness | Measurement   | High*    |

\* High as an *enabler* — do this first so the rest is data-driven.

---

## CPU

### P1 — `t_junction_count` is O(edges × vertices) — ✅ done (2026-05-29)
- **Where:** `fascat/mesh.py` (`t_junction_count`); called inside `repair()` and in merge-vertex
  diagnostics.
- **Why:** For every undirected edge it scanned **all** points with
  `np.all((self.points >= minimum) & (self.points <= maximum), axis=1)` — a full O(V) array
  pass per edge, so O(E·V) ≈ O(V²). Because `repair()` calls it **twice** (before *and* after)
  and `repair()` is in the default pipeline, this dominated on medium/large meshes.
- **Resolution:** Vertices are now bucketed into a spatial grid once (cell size = median edge
  length) via `_point_cell_buckets`; each edge probes only the grid cells overlapping its
  tolerance-expanded AABB via `_segment_candidate_vertices`, with a full-scan fallback for rare
  oversized edges (`_T_JUNCTION_CELL_BUDGET`). The candidate set is a provable superset of the old
  AABB scan, so the projection/distance/conflict logic is unchanged and **the returned count is
  identical** (pinned by `test_t_junction_count_matches_bruteforce`, which asserts equality against
  the original implementation across random/structured/degenerate meshes × 6 tolerances). Measured
  on grid-plane meshes: ~5× at 10k verts, ~12× at 25.6k, growing with size; tractable at 57.6k
  where the old code was effectively O(V²). Remaining per-edge Python/NumPy overhead could be
  vectorized later (out of P1 scope).

### P2 — `repair()` always runs a full before/after diagnostic suite
- **Where:** `fascat/mesh.py:256` (`repair`), metrics at `:265-267` (before) and `:289-292`
  (after): `quality_metrics`, `t_junction_count`, `boundary_gap_count`, `orientability_metrics`
  — each computed twice. Invoked unconditionally at `fascat/pipeline.py:157`, and again after
  `decimate`/`optimize` (`fascat/ops/actions.py:148`, `fascat/ops/optimize.py:73`).
- **Why:** These diagnostics exist to populate report metadata, but they run even when the
  caller only needs geometric cleanup, and the before/after pairs double the cost (P1 is one
  of them).
- **Fix:** Gate the expensive metrics behind a flag (e.g. only when a repair/quality report is
  requested); compute shared structures once and reuse for before/after deltas.

### P3 — Occlusion removal is a brute-force ray cast with no acceleration structure
- **Where:** `fascat/ops/actions.py:1653` (`_sample_is_visible`), `:1666` (`_segment_blocked`),
  `:1675` (`_segment_intersects_mesh`), `:1684` (`_segment_triangle_t`); occluder set built at
  `:1462` (`_candidate_occluders`, all other occurrences).
- **Why:** Complexity is ~O(samples × directions × occluders × occluder_faces). The inner
  triangle test is pure-Python Möller–Trumbore with scalar `np.dot`/`np.cross` per triangle
  (NumPy scalar overhead is worse than plain floats), and there is no BVH/grid — only a
  per-ray AABB reject. PLAN.md confirms "no acceleration structure / GPU."
- **Fix:** Build a BVH/uniform grid per occluder (or use `trimesh.ray`/embree), vectorize the
  ray–triangle test across all candidate faces at once, and spatially cull occluders per
  candidate instead of testing all occurrences.

### P4 — Per-element Python loops in core mesh kernels
- **Where (all `fascat/mesh.py`):** `compute_flat_normals:1026`, `compute_hard_edge_normals:1055`,
  `compute_tangents:1171`, `subdivide_long_edges:1224`, `improve_skinny_triangles:1334`,
  `collapse_short_edges:1279`; plus `np.add.at` scatter at `:1013`, `:1016`, `:1196`, `:1197`.
- **Why:** These iterate faces/vertices in Python (`for face in self.faces.astype(int).tolist()`,
  per-vertex list building, per-face `np.linalg.norm`). `np.add.at` is an unbuffered scatter and
  is far slower than vectorized alternatives. `compute_normals`/`compute_tangents` run after
  almost every stage (repair, stage, simplify, decimate sampling, edge controls), so this is hot.
- **Fix:** Vectorize: expand per-corner contributions with `np.repeat`, accumulate with
  `np.bincount` (per component) or `np.add.reduceat` on sorted indices instead of `np.add.at`;
  rebuild flat/hard-edge vertices with array ops rather than Python lists.

### P5 — Edge / adjacency maps rebuilt from `.tolist()` repeatedly
- **Where:** `fascat/mesh.py:1886` (`_edge_faces_map`), `:2255` (`_undirected_edges_and_counts`),
  `:923` (`orientability_metrics`), `:2172` (`_boundary_loops`); duplicate boundary-loop builder
  at `fascat/ops/actions.py:1344`.
- **Why:** Each rebuilds Python dict/set edge structures from `self.faces.astype(int).tolist()`
  from scratch. Several are recomputed multiple times within a single `repair()` / quality pass
  and again during feature-preservation in `simplify`.
- **Fix:** Compute the undirected-edge / edge→faces / boundary structures once per mesh state and
  pass them into the metrics that need them (ties into P17).

### P6 — `merge_vertices` recomputes connected components and per-vertex keys, plus heavy diagnostics
- **Where:** `fascat/mesh.py:372` (`merge_vertices`), `:572` (`_merge_vertex_components`),
  `:583` (`_merge_vertex_skip_diagnostics`), `:459` (`_near_duplicate_unmerged_stats`).
- **Why:** `_distance_connected_components` is run for the diagnostics **and** again for the
  actual merge; per-vertex attribute-key tuples are rebuilt several times; and the diagnostics
  also invoke `t_junction_count`/`boundary_gap_count` (P1). The near-duplicate scan loops all
  vertices with per-pair `np.linalg.norm`.
- **Fix:** Compute components and attribute keys once and reuse; make the skip/near-duplicate
  diagnostics opt-in.

### P7 — Staging computes O(F²) UV overlap diagnostics for every part
- **Where:** `fascat/ops/stage.py:545-546` (`_tag_uv_layout_quality`) → `fascat/mesh.py:1416`
  (`uv_layout_stats`), `:1473` (`uv_distortion_metrics`), `:1947` (`_uv_island_count`).
- **Why:** `uv_layout_stats` does a sweep with an inner loop and pure-Python polygon-clipping
  (`_triangle_overlap_area_2d`, `mesh.py:144`) → worst case O(F²) triangle-pair tests, run on
  every staged part with UVs just to fill metadata. Island count is a per-edge Python union-find.
- **Fix:** Make overlap/distortion diagnostics opt-in or sampled; skip overlap detection unless a
  UV policy (`forbid_overlapping`/bake domain) actually requires it.

### P8 — Nearest-centroid material assignment is O(target × source)
- **Where:** `fascat/mesh.py:2098` (`_assign_materials_by_nearest_centroid`), called from
  `simplify` (`:1641`, `:1657`).
- **Why:** All-pairs distance between target and source face centroids (chunked to bound memory,
  but still quadratic compute). Runs on every `simplify()` that carries materials — i.e. every
  decimate/optimize/LOD level.
- **Fix:** Use a KD-tree (`scipy.spatial.cKDTree`) for nearest centroid, or carry material indices
  through meshoptimizer's vertex remap instead of re-deriving them.

### P9 — LOD levels are simplified from the full-resolution mesh, not progressively
- **Where:** `fascat/ops/lod.py:101` (`part.mesh.simplify(ratio=ratio)` inside the per-level loop).
- **Why:** N LOD levels = N independent simplifications of the original high-poly mesh (each also
  re-runs nearest-centroid material assignment + `compute_normals`), instead of chaining LOD2 off
  LOD1, etc.
- **Fix:** Simplify progressively from the previous (already reduced) level.

### P10 — Tessellation: Python-list extraction and duplicated edge-control passes
- **Where:** `fascat/ops/tessellate.py:116-147` (per-OCCT-node / per-triangle list building with
  `.Transformed()` per node) and `:211-228` (`_apply_mesh_tessellation_controls` runs
  subdivide→collapse→skinny→**subdivide→collapse again**).
- **Why:** Vertex/triangle extraction appends tuples in Python; and the long-edge/short-edge
  passes (themselves Python loops, see P4) run twice unconditionally.
- **Fix:** Only re-run the second subdivide/collapse pass if the first changed the mesh; batch
  triangulation reads where the OCCT API allows.

## Memory

### P11 — `Asset.copy()` duplicates every mesh array ~4–8× per call (~10×/pipeline)
- **Where:** `fascat/mesh.py:164-177` (`__post_init__` re-copies arrays already copied by
  `copy()` at `:187-197`); `fascat/asset.py:104-121` (`Part.copy` + `Part.__post_init__`),
  `:149-155`/`:223-235` (`Asset.copy` + `Asset.__post_init__`). Same pattern in
  `fascat/report.py:61-78` (every step's option dicts copied twice).
- **Why:** `.copy()` copies arrays, then the constructor's `__post_init__` copies them again;
  this nests Asset→Part→Mesh, so one `Asset.copy()` duplicates each mesh array several times.
  Most op functions begin with `asset.copy(keep_source=True)`, and a full conversion chains ~10+
  stages → large redundant allocation + memcpy proportional to total geometry.
- **Fix:** Drop the redundant defensive copy in `__post_init__` (or add an internal
  "adopt-without-copy" path) so `.copy()` copies exactly once; consider copy-on-write / sharing
  unchanged parts and the report by reference across stages.

### P12 — Whole-mesh fingerprint / array digests recomputed after every op
- **Where:** `fascat/mesh.py:242` (`fingerprint` — `np.round` of all points + SHA1 over
  points+faces), set after tessellate/repair/merge/decimate/optimize/etc.; `fascat/ops/scene.py:283`
  (`_array_digest`) hashes points/normals/UVs/faces, computed ~2× per part in
  `_reconstruct_instances` (`:101-117`).
- **Why:** Hashing entire meshes is memory-bandwidth heavy and repeated each stage even when the
  mesh is unchanged.
- **Fix:** Cache the fingerprint on the mesh and invalidate on mutation; compute scene attribute
  digests once per part and reuse.

### P13 — Export binary buffer is copied several times
- **Where:** `fascat/io/gltf.py:604` (`bytes(builder.data)`), `:982` (`bytearray(binary)` for
  meshopt), `:1304-1317` (`_pack_glb` concatenation).
- **Why:** The full geometry buffer (can be the largest single allocation) is duplicated for the
  immutable copy, again for meshopt, and again when packing the GLB.
- **Fix:** Encode/repack in place or stream into the final buffer; avoid the intermediate full-copy.

## I/O

### P14 — Default `validate_output` re-reads and re-parses the just-written file
- **Where:** `fascat/pipeline.py:262-293` (validate step, `validate_output=True` by default);
  glTF validator at `fascat/io/gltf.py:1433-1526` re-fetches and re-typechecks
  `document["nodes"]`/`["accessors"]` arrays on **every** recursion (`_walk_node`,
  `_require_accessor`).
- **Why:** Every conversion writes the file, then reads the whole thing back and re-parses it;
  the validator's repeated `_array(document.get(...))` lookups add quadratic-ish overhead on big
  documents.
- **Fix:** Validate the in-memory document/arrays before/at write time instead of round-tripping
  through disk; cache the parsed `nodes`/`accessors` lists once; allow skipping validation for
  large assets.

### P15 — Exporters build one Python object/string per vertex/triangle
- **Where:** glTF strided path `fascat/io/gltf.py:514-519` (`_accessor_payload` loops every vertex
  row doing `payload[...] = row.tobytes()` whenever `byte_stride` is set, i.e. quantized export);
  OBJ `fascat/io/obj.py:39-51` (per-vertex `v`, per-normal `vn`, per-face `f` f-strings);
  STL `fascat/io/stl.py:43-68` (`_triangles` makes one ndarray per triangle, `_binary_stl`/
  `_ascii_stl` loop per triangle + per vertex with `struct.pack`); USD `fascat/io/usd.py:395`,
  `:401`, `:409` (one `Gf.Vec3f`/`Gf.Vec2f` per point/normal/UV).
- **Why:** Millions of tiny Python objects / f-strings / `struct.pack` calls dominate export time
  for large meshes; this is the export-side mirror of P4.
- **Fix:** Vectorize: build strided buffers with a NumPy `(rows, stride)` uint8 view; format
  OBJ/STL with `np.savetxt`-style bulk formatting or a structured dtype + `tobytes()`; feed USD
  via `Vt.*Array` from NumPy buffers instead of Python comprehensions.

## Concurrency

### P16 — The pipeline processes independent parts serially
- **Where:** Per-part loops in `fascat/ops/optimize.py:39`, `fascat/ops/lod.py:36`,
  `fascat/ops/stage.py:51`, `fascat/ops/actions.py` (decimate), and `Asset.repair/merge_vertices`
  in `fascat/asset.py`. Only OCCT meshing sets `InParallel=True` (`fascat/ops/tessellate.py:192`).
- **Why:** Tessellation, repair, staging, decimation, and LOD generation are independent per part
  but run on one core; the per-part Python loops (P4) hold the GIL, so they don't overlap.
- **Fix:** Parallelize per-part work — a process pool for CPU-bound stages (tessellate/decimate/
  LOD), thread pool where the work is NumPy/native (GIL-releasing) — with deterministic ordering.

## System design

### P17 — No memoization of derived mesh topology across stages
- **Where:** cross-cutting; see P5 structures plus `quality_metrics` (`fascat/mesh.py:793`),
  `_face_unit_normals` (`:1894`), `fingerprint` (`:242`).
- **Why:** Edge maps, boundary loops, face normals, quality metrics, and fingerprints are
  recomputed from scratch each time they are needed — within one stage, across
  repair→stage→optimize, and across LOD levels. (Tessellation *does* cache by shape via
  `mesh_by_source`, `fascat/ops/tessellate.py:46`, and dedups identical parts — good prior art.)
- **Fix:** Attach a lazily-computed topology cache to `Mesh` (undirected edges, edge→faces,
  boundary loops, face normals) invalidated on geometry mutation.

### P18 — `stats()` / `walk()` / draw-call breakdown recomputed many times per stage
- **Where:** `fascat/asset.py:76-80` (`Node.walk`), `:243-257` (`stats`), `:181-221`
  (`draw_call_breakdown`); every op records `before`/`after` stats, the `progress` callback calls
  `stats()` again (`fascat/pipeline.py:118-216`), and `_report_stats`/`_hierarchy_report_stats`
  add `stats(include_lods=True)` + `draw_call_breakdown()`.
- **Why:** Each `stats()`/`draw_call_breakdown()` rebuilds the entire node list via recursive
  `walk()` (list concatenation) and re-sums all meshes — several times per stage.
- **Fix:** Walk once per stage and pass node lists/counts around; cache stats between unchanged
  steps.

## Measurement

### P19 — Reported metrics are heuristic estimates; there is no benchmark/profiling harness
- **Where:** fixed "5 GB per million triangles" decimation rule
  (`fascat/ops/actions.py:30-31`, `:1055`), `_estimated_load_time` (`fascat/pipeline.py:1352`),
  and PLAN.md ("the budgets are numeric only"; missing "measured … load-time, memory, and FPS
  profiling").
- **Why:** Memory/load-time/throughput numbers are formulas, not measurements, so they can point
  optimization at the wrong place. There is per-step `timed_step` timing in the report, but no
  per-hotspot profiling and no regression benchmark for conversion throughput.
- **Fix:** Add a small benchmark suite (a few representative STEP files) recording per-stage wall
  time + peak RSS, and profile (`cProfile`/`py-spy`/`scalene`) before optimizing. Expected top
  costs to confirm: tessellation, `repair` T-junctions (P1/P2), occlusion (P3), and export
  serialization (P15). **Do this first** so the rest of the list is data-driven.
