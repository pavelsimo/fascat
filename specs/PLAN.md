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
  snippets live in the detailed findings below
- [ ] **P3** Profile on real large CAD corpora (10k+ parts) and record numbers in the
  baseline list below — fixture baselines recorded there (2026-06-14 corpus, 2026-07-01
  re-run); actual 10k+ part corpus profiling still needed

### Performance Findings

Audit of 2026-07-03: the 2026-07-01 findings F1–F7 and the scipy connected-components
rewrite were verified **shipped** (see verification below) and are gone from the open
list. This section now holds the **new open findings F8–F12** — the same per-face
Python-loop/`.tolist()`/dict-remap patterns, found by sweeping the rest of the package
after the hierarchy/scene fixes landed. Each is scoped, behavior-preserving (one
documented tie-break caveat in F12), and independent of the others; implement, test, and
commit them one at a time per the operating checklist.

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

#### Open findings

Unlike F1–F6 (merge/explode-only), **F8 and F9 sit on the main export and tessellate
paths**, so `make benchmark` on `vertical-screw.step` should move slightly; F10–F12 bite
on LOD sampling and buffer optimization of large meshes.

##### F8 — F1's material-enumeration pattern survives in four more files

The exact `sorted(set(mesh.material_indices.astype(int).tolist()))` shape that F1 removed
from `hierarchy.py` still allocates O(F) Python ints per call in:

- `fascat/io/gltf.py:1686` — **glTF writer primitive splitting, on every export**
- `fascat/export_report.py:109` — export report material summary
- `fascat/asset.py:1772` — asset-level material usage scan
- `fascat/ops/actions.py:2679` — `used = sorted({int(index) for index in ...tolist()})`,
  same allocation shape via a set comprehension

```python
# before (all four sites)
for material_index in sorted(set(mesh.material_indices.astype(int).tolist())):
# after — np.unique is already sorted, no per-face Python objects
for material_index in np.unique(mesh.material_indices.astype(np.int64)).tolist():
```

##### F9 — tessellation dedupe builds an O(F)-int tuple key per part

`_deduplicate_parts_by_fingerprint` (`fascat/ops/tessellate.py:1375-1384`) keys its
canonical-part dict on
`tuple(int(value) for value in part.mesh.material_indices.tolist())` — one Python int per
face, per part, **on the main tessellate path** (it runs at the end of every
`tessellate_asset` call, `fascat/ops/tessellate.py:160`). This is exactly the pattern F4
replaced in instance reconstruction. Reuse the digest idiom: hoist
`_array_digest_required` (`fascat/ops/scene.py:296`) into a shared helper (e.g.
`fascat/ops/_digest.py`, or a `Mesh` staticmethod) so scene and tessellate share one
implementation, and normalize dtype first — the digest hashes dtype and the tuple key was
dtype-agnostic:

```python
material_indices = None
if part.mesh.material_indices is not None:
    material_indices = _array_digest_required(
        part.mesh.material_indices.astype(np.int64, copy=False)
    )
key = (part.fingerprint, tuple(part.material_ids), material_indices, _metadata_key(part.metadata))
```

Update the `canonical_by_key` annotation (`fascat/ops/tessellate.py:1376`) from
`tuple[int, ...] | None` to `str | None` to match.

##### F10 — F6's dict-based face-group remap survives in two more places

- `_slice_faces` (`fascat/ops/actions.py:1809-1826`): builds a
  `{int(face_index): local_index}` dict from `.tolist()` and probes it per group entry —
  byte-for-byte the pattern F6 replaced in `_slice_mesh`.
- `fascat/ops/hierarchy.py:640-642`: the `face_position` dict + per-entry
  `[face_position[int(value)] for value in values.astype(int).tolist() ...]`
  comprehension, same shape.

Fix: hoist `_remap_sliced_face_group` (`fascat/ops/scene.py:480-486`) into the same
shared home as F9's digest helper and call it from all three sites:

```python
face_lookup = np.full(mesh.triangle_count, -1, dtype=np.int64)
face_lookup[face_indices.astype(np.int64)] = np.arange(face_indices.shape[0], dtype=np.int64)
face_groups = {name: _remap_sliced_face_group(face_lookup, group) for name, group in mesh.face_groups.items()}
```

The bounds/sentinel masking in `_remap_sliced_face_group` preserves the current behavior
of silently skipping out-of-slice group entries at all three call sites.

##### F11 — accidental O(F²) in `_sample_mesh_faces`

`fascat/ops/actions.py:1798-1806`: the top-up loop rebuilds the exclusion set **inside
the comprehension condition**, i.e. once per candidate index:

```python
missing = [index for index in range(mesh.triangle_count) if index not in set(face_indices.astype(int).tolist())]
```

Python evaluates the `if` clause per element, so this constructs an O(F) set F times —
O(F²) with O(F²) allocations. The loop is rarely entered today (`np.linspace` over
`0..F-1` with `target ≤ F` steps by ≥ 1, so the truncated ints are strictly increasing
and `np.unique` removes nothing), which is precisely why it has survived: it is a
landmine, not a hot path. Replace the whole while-loop body with the vectorized
difference:

```python
while face_indices.shape[0] < target:
    missing = np.setdiff1d(
        np.arange(mesh.triangle_count, dtype=np.int64), face_indices, assume_unique=True
    )
    face_indices = np.sort(
        np.concatenate([face_indices, missing[: target - face_indices.shape[0]]])
    )
```

##### F12 — `Mesh.optimize_buffers` runs per-face and per-vertex Python loops

`fascat/mesh.py:2645-2678`, on the optimize path (`fascat/ops/optimize.py` calls it for
every part when buffer optimization is enabled):

1. **Per-face dict build** (`:2645-2657`): `old_face_lookup` maps
   `tuple(sorted(face)) → old index` via `.tolist()` — O(F) tuple allocations — then
   `reordered_face_indices` probes it once per cache-optimized face. Vectorize with a
   sorted-rows structured view + `searchsorted`:

   ```python
   old_sorted = np.sort(self.faces.astype(np.int64, copy=False), axis=1)
   new_sorted = np.sort(cache_optimized.reshape((-1, 3)).astype(np.int64), axis=1)
   row_dtype = np.dtype((np.void, old_sorted.dtype.itemsize * 3))
   old_view = np.ascontiguousarray(old_sorted).view(row_dtype).ravel()
   new_view = np.ascontiguousarray(new_sorted).view(row_dtype).ravel()
   order = np.argsort(old_view)
   pos = np.searchsorted(old_view[order], new_view)
   pos_clamped = np.minimum(pos, old_view.size - 1)
   matched = old_view[order[pos_clamped]] == new_view
   reordered_face_indices = np.where(
       matched, order[pos_clamped], np.arange(new_view.size, dtype=np.int64)
   )
   ```

   **Tie-break caveat:** for duplicate faces the dict keeps the *last* old index while
   `argsort` + `searchsorted` yields the *first*. This only matters when duplicate faces
   carry different `material_indices` — cover it with a test and document the
   first-occurrence semantics (it is the more predictable of the two).

2. **Per-vertex remap inversion** (`:2675-2678`): `for old_index, new_index in
   enumerate(remap.astype(np.int64))` is a pure-Python loop over every vertex. Replace
   with a masked scatter — duplicate `new_index` targets came from vertices with
   byte-identical attribute streams (that is why meshoptimizer merged them), so which
   representative wins is immaterial:

   ```python
   remap64 = remap.astype(np.int64)
   valid = remap64 < int(unique_vertices)
   old_for_new = np.zeros(int(unique_vertices), dtype=np.int64)
   old_for_new[remap64[valid]] = np.flatnonzero(valid)
   ```

Implement together with the §1 correctness item (same function, same tests).

#### Verification recipe (when implementing)

1. Focused tests per finding: `uv run pytest tests/test_gltf.py tests/test_tessellate.py
   tests/test_actions.py tests/test_hierarchy.py tests/test_mesh.py --no-cov -q`. F9
   changes a dict-key type — check no test asserts on the tuple form; F12 needs the new
   duplicate-face tests from §1.
2. `make ci` for the full gate.
3. `make benchmark` before/after — F8 (glTF export) and F9 (tessellate dedupe) are on the
   fixture's critical path, so expect a small improvement on `vertical-screw.step`;
   F10–F12 show on large meshes and LOD-heavy runs only.

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
