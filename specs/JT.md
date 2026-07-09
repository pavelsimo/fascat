# JT Import

The planning document for adding JT (Siemens Jupiter Tessellation, ISO 14306) as a CAD
**input** format. JT is currently listed under Deferrals in [PLAN.md](PLAN.md); this spec
un-defers it with a concrete, phased design. Scope: **import only** — JT joins STEP, IGES,
and BREP as an input; fascat does not write JT.

Provenance: ecosystem and codebase investigation on 2026-07-09. Key decisions were made
up front and are recorded here so they are not relitigated per-phase.

## Why an in-house pure-Python reader

There is no MIT-compatible, pip-installable JT reader:

- **OpenCascade TKJT** (JT Assistant sources) and its forks
  ([oce-jt](https://github.com/cbsghost/oce-jt)) are **GPL-2.0** — incompatible with
  fascat's MIT license as a dependency — C++, dormant, and not packaged for Python.
- **[PyOpenJt](https://github.com/jriegel/PyOpenJt)** (TKJT + Python bindings) is also
  **GPL-2.0**, Windows-focused, not on PyPI, and early-stage (last release 2024-02).
- **Siemens JT Open Toolkit** is commercial and not redistributable.
- OCP/`cadquery-ocp` (fascat's existing CAD backend) has no JT support in core OCCT.

The JT file format itself is publicly documented (Siemens *JT File Format Reference*;
ISO 14306:2017 for JT 10.x, ISO 14306:2012 for 8.1), so a clean-room pure-Python reader
is feasible: stdlib `zlib`/`lzma` + the existing numpy dependency, **no new runtime deps**.
This matches how fascat already owns its glTF/OBJ/STL/FBX writers.

Development-time oracle hygiene: GPL tools (TKJT, PyOpenJt) may be **run** externally to
produce reference outputs for cross-checking decoded arrays; their source is never read
into, ported to, or vendored in this codebase.

## What JT import adds

JT files carry **pre-tessellated LOD meshes**, assembly structure, materials, and
properties. Embedded precise geometry is usually Parasolid XT B-rep — proprietary and
undecodable without a Parasolid license — so JT import is **tessellation-only**: imported
parts have `mesh` set and `source_shape=None`. The pipeline already supports this — the
`reuse_existing_meshes` path (`fascat/ops/tessellate.py:73-88`) passes mesh-only parts
through, and repair/stage/optimize/LOD/export all operate on meshes.

v1 extracts: tessellated LOD meshes · assembly hierarchy, transforms, and instances ·
colors/materials · part names and properties/metadata. Deferred (see Deferrals): PMI,
B-rep, textures, external-reference (shattered) assemblies, JT export, JT 8.x.

Target versions: **JT 9.5 and 10.x** (what NX/Teamcenter emit). JT 8.x files fail with a
clear "unsupported JT version" error.

## How to read this plan

- States: `[ ]` open · `[~]` partial · `[x]` implemented.
- Priorities: **P0** correctness/security blocker · **P1** production requirement ·
  **P2** quality/competitive · **P3** later.
- Every code claim cites `file:line` as of 2026-07-09 (`01c2487`); lines drift as code
  changes.

## 1. Reader architecture — `fascat/io/jt/`

A package, not a single module — the codec layer alone justifies it.

```
fascat/io/jt/
  __init__.py      # re-export read_jt, read_jt_bytes
  container.py     # file header, TOC, segment framing, decompression
  codec.py         # Int32 CDP, bitlength/Huffman/arithmetic codecs, predictors, dequant
  lsg.py           # Logical Scene Graph elements, property tables, traversal
  shape.py         # Shape LOD segment decode -> (points, faces, normals) numpy arrays
  reader.py        # read_jt orchestration: container -> lsg -> shapes -> Asset
```

### `container.py`

- Parse the 80-byte version header (`"Version 9.5 ..."` / `"Version 10.x ..."`),
  byte-order flag, TOC offset. **Version gate here**: anything below 9 raises
  `RuntimeError("unsupported JT version 8.x: fascat supports JT 9.x and 10.x")`,
  wrapped into `FascatIOError` by `@wrap_io_errors("read JT")`.
- TOC parsing: entries of (segment GUID, offset, length, attributes); segment type from
  the attribute bits. Expose `Toc.find(guid)` so late-loaded segments resolve lazily —
  unused LOD segments are never decompressed or decoded.
- Segment framing with per-segment decompression: zlib (JT 9) and LZMA (JT 10, raw
  stream with props byte), both stdlib. Honor the per-segment compression-algorithm
  field rather than assuming by version.
- A shared `BitReader`/`ByteReader` (numpy-backed where possible), reused by `codec.py`.

### `codec.py`

The Int32 Compressed Data Packet (CDP) is the universal primitive — vertex indices,
quantized coordinates, normals, and property references all decode through it.

- CODECs: Null (raw), Bitlength (fixed/variable width), Huffman, Arithmetic. JT 10 uses
  a revised CDP/arithmetic variant — keep both behind a version-dispatched interface
  (`decode_int32_cdp(reader, version) -> np.ndarray`).
- Predictors (Lag1, Lag2, Stride1/2, StripIndex, Ramp, Xor1/2, Null) as vectorized
  numpy post-passes.
- Dequantization: uniform quantizer ranges → float64; **Deering normal codec**
  (sextant/octant + theta/psi) with numpy lookup tables so whole normal arrays decode
  in one shot.
- The bit-serial parts (arithmetic decode loop, Huffman tree walk) cannot be fully
  vectorized; keep them tight, correct, and profiled (see §7). The codec API must not
  preclude a future compiled accelerator, which is itself out of scope.

### `lsg.py`

- Element registry mapping object-type GUIDs → parser functions. v1 element set:
  Partition, Assembly, Part, Instance, Group, LOD/Range-LOD nodes, Tri-Strip Set /
  Vertex Shape nodes; Material and Geometric Transform attributes; property atoms
  (String/Integer/Float/Date/Object-reference/**Late-Loaded**); the Property Table.
- **Unknown elements are skipped by length, counted, and reported** (element headers
  carry lengths precisely for this) — real-world files are full of vendor elements and
  must not hard-fail.
- Output is an intermediate dataclass graph with resolved per-node properties, not
  `Asset` — keeps `reader.py` thin and the traversal testable.

### `shape.py`

- Decode Shape LOD segments: JT 9 Tri-Strip Set / vertex-based compressed reps; JT 10
  mesh coder variant. Output plain `(points (N,3) f64, faces (M,3) i64,
  normals (N,3) f64 | None)` tuples. Vectorized tri-strip → triangle expansion;
  degenerates dropped.
- A shape whose LOD segments are missing/undecodable: warn and continue per part. If
  **no part in the file yields any mesh** (B-rep-only export), raise
  `RuntimeError("JT file contains no tessellated LOD data (B-rep only); re-export with
  tessellation")` — wrong geometry or a silently empty asset is worse than a clear error.

### `reader.py`

Mirrors `fascat/io/iges.py` / `fascat/io/brep.py` exactly:

- `read_jt(path, *, options: JtReadOptions | StepReadOptions | None = None) -> Asset`
  decorated `@wrap_io_errors("read JT")`; `read_jt_bytes(data, *, name="stdin.jt",
  options=None)` for API parity (parses from `BytesIO`, no temp file).
- Suffix validation against `JT_SUFFIXES`, `timed_step()` timing, `_stable_id`-style
  sha1 ids (reuse the `fascat/io/step.py` helpers — `step.py` has no module-level OCP
  import, so the JT reader stays importable without `cadquery-ocp`), `report.add_step(
  "import", options={"format": "JT", "jt_version": …, "lod_summary": …,
  "skipped_elements": …, "read_options": …, "space_normalization": …}, …)`.
- Parts: `mesh=<finest LOD>`, `source_shape=None`, `fingerprint=mesh.fingerprint()`,
  `metadata` with `source_identity`, `source_name`, `loaded_representation: "mesh"`,
  flattened JT node properties, and LOD counts. JT Instance nodes → multiple `Node`s
  sharing one `part_id` (fascat's existing occurrence model).
- Materials: JT Material Attribute → `Material(base_color=diffuse RGBA,
  opacity=diffuse alpha, metallic=0.0, roughness=clamp(sqrt(2/(shininess+2)), 0, 1))`;
  raw JT values (`jt_specular`, `jt_shininess`, …) kept in `material.metadata`.
  Value-hash dedupe per `brep.py:_material_id`; parts without a material attribute get
  the shared default (`brep.py` `_DEFAULT_MATERIAL_COLOR` pattern).

## 2. Integration points

- [x] **P1** `fascat/io/_suffixes.py` — `JT_SUFFIXES = frozenset({".jt"})`; add to
  `CAD_SUFFIXES` (`_suffixes.py:13`)
- [x] **P1** `fascat/pipeline.py:_read_input` (~397) — `if suffix in JT_SUFFIXES:
  return read_jt(path, options=options)`; the multi-file list branch stays STEP-only
- [x] **P1** `fascat/cli.py:_read_cad_for_cli` (~3501) — add the `read_jt` branch;
  `_validate_cad_input` (~3425) — extend the hardcoded error string with `.jt`
- [x] **P1** `fascat/options.py` (~445, after `BrepReadOptions`) —
  `JtReadOptions(StepReadOptions)` overriding `pmi=False`, `design_variants=False`,
  `multi_file=False`, `source_textures=False`, `material_library_mapping=False`; new
  field `lod_selection: str = "finest"` (`finest` | `all`) with `__post_init__`
  validation. The CLI's single `StepReadOptions` coerces via `to_dict()` (default on
  the new field), so all existing flags (`--source-units`, `--target-up-axis`,
  `--metadata`, …) work for JT with no new plumbing
- [x] **P1** `fascat/__init__.py` — lazy-export `read_jt` (TYPE_CHECKING block, lazy
  map, `__all__`); `tests/test_public_api.py` (~430) gains `"read_jt"`
- [x] **P1** Docs — `docs/index.md`, `docs/reference.md` (format lists, I/O support
  table, capability matrix, input-suffix sentence), `docs/api.md` (`fc.read_jt` +
  `JtReadOptions`), `README.md`, `CHANGELOG.md`
- [x] **P2** `--jt-lod-selection finest|all` CLI flag threaded into `JtReadOptions`
  when the input suffix is `.jt`

Stdin (`-`) stays STEP-only in v1 (`_read_cad_for_cli` unconditionally uses
`read_step_bytes` for `-`); documented. External-reference (shattered) partitions are
**not resolved** in v1: emit a placeholder `Node` with
`metadata["external_reference"] = <name>` plus a report warning; a follow-up can mirror
`StepReadOptions.multi_file` semantics.

## 3. LOD mapping

Default: **import only the finest LOD into `Part.mesh` and discard coarser LODs**.
`JtReadOptions(lod_selection="all")` opts into filling `Part.lod_meshes` fine → coarse,
tagging each with `metadata["lod_source"]="imported"`.

Rationale, grounded in how `lod_meshes` is consumed:

- `Part.mesh` is the canonical geometry for repair/stage/optimize/stats;
  `Part.lod_meshes` is strictly "coarser variants, LOD1..LODn"
  (`fascat/io/gltf.py:451` `meshes = (part.mesh, *part.lod_meshes)`;
  `fascat/io/usd.py:414` variant sets). Finest → `mesh` is the only consistent mapping.
- `fascat/ops/lod.py:80` **unconditionally overwrites** `part.lod_meshes` when the
  pipeline generates LODs — and default convert profiles do — so importing all JT LODs
  by default is wasted decode time and memory (JT files commonly ship 3–5 LODs/part).
- `asset.stats(include_lods=…)` (`fascat/asset.py:574-577`) would count imported LODs
  and skew budget/report numbers versus STEP inputs.
- Late-loaded segments make finest-only genuinely cheap: coarser Shape LOD segments are
  never even decompressed.

## 4. Units and up-axis

- **Units**: JT geometry is unitless; the conventional carrier is the
  `JT_PROP_MEASUREMENT_UNITS` string property (`"Millimeters"`, `"Inches"`, …) on the
  root partition/part. Rule: read it, map to `(unit_name, meters_per_unit)`, then call
  `_step._space_normalization(…)` exactly as `iges.py` does — all existing
  `--source-units` / `--source-meters-per-unit` / `--target-*` overrides work
  unchanged. Absent property → default `("millimetre", 0.001)` (NX convention, matches
  the IGES/BREP fallback) plus a report warning.
- **Up axis / handedness**: JT declares neither; NX/Teamcenter output is Z-up
  right-handed, which is exactly the `StepReadOptions` defaults. No new code; document
  the assumption and point at the existing override flags.

## 5. Testing & fixtures

- [x] **P1** `tests/test_jt_import.py` modeled on `tests/test_iges_import.py`:
  part/occurrence counts, units, `report.steps[-1].options["format"] == "JT"`,
  `part.mesh is not None`, `part.source_shape is None`,
  `metadata["loaded_representation"] == "mesh"`, wrong-extension raises
  `FascatIOError`, JT-8 header raises `"unsupported JT version"`, B-rep-only file
  raises `"no tessellated LOD data"`, external-ref file warns
- [x] **P1** Synthetic fixture writer `tests/_jt_builder.py` emitting minimal valid
  JT 9.5 (later 10.x) byte streams — header, TOC, LSG with a two-part assembly +
  instance + material, one tri-strip Shape LOD using the simplest codecs. Test-owned,
  hermetic, MIT-clean; directly exercises container/codec framing
- [x] **P1** Golden byte-vectors per CODEC (bitlength, Huffman, arithmetic, predictors,
  Deering normals) as inline `bytes` literals with expected `np.ndarray` outputs — the
  arithmetic coder especially needs golden vectors
- [x] **P1** Round-trip in the `tests/test_round_trip.py` style: synthetic JT →
  `fc.convert(…, "out.glb")` → `validate_gltf` + triangle/transform/material asserts;
  same for `.usdc` under `requires_usd`
- [x] **P2** Real-world fixture, licensing-gated: prostep ivip JT Implementor Forum
  suites and Siemens JT2Go samples are likely **not** redistributable — assume no.
  Support an untracked `tests/fixtures/local/*.jt` path behind `pytest.mark.skipif` so
  developers can drop files in. Do not take PyOpenJt repo samples without a license
  check
- [x] **P1** `tests/test_cli.py`: one `convert` on a synthetic `.jt`, plus the updated
  `_validate_cad_input` message

No new pytest marker: the reader is pure stdlib + numpy, making JT the first CAD input
testable without `requires_ocp`. Verify during implementation that `fc.convert` on a
mesh-only asset trips no OCP import in the `reuse_existing_meshes` path.

## 6. Milestones

- [x] **P1** Phase 1 — `container.py`: header/version gate, TOC, segment framing,
  zlib + LZMA; JT-8 rejection; unit tests on synthetic bytes (S)
- [x] **P1** Phase 2 — `lsg.py` + minimal `codec.py` (Null/bitlength CDP): traversal,
  properties, transforms, instances → `Asset` with empty meshes; `reader.py` skeleton
  wired to `Report` (M)
- [x] **P1** Phase 3 — full `codec.py`: Huffman + arithmetic CODECs, predictors,
  dequantization, Deering normals, golden-vector tests (L — highest correctness risk)
- [x] **P1** Phase 4 — `shape.py` JT 9.5: Tri-Strip Set / vertex-based Shape LOD decode
  → real meshes; finest-LOD default + `lod_selection="all"`; B-rep-only error (L)
- [ ] **P2** Phase 5 — JT 10.x mesh coder (revised CDP/arithmetic, v10 topology
  coding). Until landed, 10.x files fail with `"JT 10 mesh coding not yet supported"`
  rather than emit wrong geometry (L–XL; budget for spec ambiguity)
- [x] **P2** Phase 6 — polish: material mapping, property/metadata flattening,
  external-ref warnings, `--jt-lod-selection` flag (S–M)
- [x] **P1** Phase 7 — integration + docs: `_suffixes`/`pipeline`/`cli`/`__init__`,
  docs/README/CHANGELOG, round-trip tests (S)

Phases 1–4 + 7 constitute a shippable **JT 9.5 milestone**; phase 5 lands separately
behind the same entry point.

## Implementation notes (2026-07-09, v1 landed)

The JT 9.5 milestone (phases 1–4, 6, 7) is implemented in `fascat/io/jt/`
(`container.py`, `codec.py`, `lsg.py`, `shape.py`, `reader.py`), with the
synthetic encoder in `tests/_jt_builder.py`. During implementation the public
*JT File Format Reference* v9.5 Rev-A was transcribed directly, which corrected
three assumptions in this plan:

- **No Huffman codec in JT 9.5.** The Int32 CDP codec set is Null, Bitlength,
  Arithmetic, and Chopper (Huffman existed only in JT 8.x). JT 9.5 shape data
  uses the Mk.2 CDP exclusively (single probability context, trailing
  out-of-band packet), which is what `codec.py` implements; JT 10 uses a third,
  incompatible CDP generation and stays behind the phase-5 error.
- **Tri-strip sets are topologically compressed.** JT 9 stores TriStripSet
  geometry as a dual-mesh topology coding (spec 7.2.2.1.2.4–6 and Appendix E),
  not as strip indices — strip data would only appear in polyline/point sets.
  `shape.py` implements the full Appendix E DualVFMesh topology decoder plus
  the Jenkins lookup2 composite/vertex hashes, and the builder mirrors it with
  a topology *encoder* (closed manifold inputs) so round trips are hermetic.
- **Per-corner normals.** Normals bind through face attribute masks and
  attribute records; the reader splits topological vertices per distinct record
  so hard edges survive into `Mesh.normals`.

Remaining spec ambiguities are marked in code comments (Mk.2 chopper with
`chop bits == 0` decodes to zeros; arithmetic probability contexts are read
after the code text). Bit-exactness against real NX/Teamcenter exports is
unvalidated until a real fixture lands in `tests/fixtures/local/` (untracked,
picked up automatically by `tests/test_jt_import.py`).

## 7. Risks

- **Arithmetic-codec bit-exactness** — a one-bit error corrupts everything downstream.
  Mitigation: golden vectors; cross-check decoded arrays against a GPL oracle run as an
  external tool (outputs are not derivative works; code is never read).
- **JT 10 mesh-coder divergence** — v10 revised the CDP and topology coding and the
  public reference has known ambiguities. Mitigated by phase separation and an explicit
  "not yet supported" error.
- **Fixture availability/licensing** — real NX exports are rarely redistributable; the
  synthetic writer carries the test burden, real files are optional local fixtures.
- **B-rep-only files** — must fail with the informative `FascatIOError`, test-covered,
  never an empty asset.
- **Pure-Python decode performance** on multi-million-triangle assemblies — bit-serial
  loops are slow in Python. Mitigations: numpy-vectorized predictors/dequant/strip
  expansion, lazy finest-only decode, profiling in phase 4. A compiled accelerator is
  possible future work; the codec API must not preclude it.
- **Element zoo** — real files contain undocumented/vendor elements; skip-by-length +
  the `skipped_elements` report counter prevent hard failures and give field
  diagnostics.

## Principles

- MIT-clean: no GPL code read, ported, or vendored; GPL tools only as external
  black-box oracles during development.
- Wrong geometry is worse than a clear error — unsupported versions and B-rep-only
  files fail loudly and informatively.
- No new runtime dependencies: stdlib `zlib`/`lzma` + existing numpy.
- Follow the existing reader contract (`iges.py`/`brep.py` shape: options coercion,
  `wrap_io_errors`, report steps, stable ids) so JT is unsurprising to maintain.
- Decode lazily: late-loaded segments mean the default path never touches bytes it
  does not need.
- Tests are hermetic: synthetic, test-owned fixtures first; real files optional.

## Deferrals

Intentionally out of scope for v1 (revisit on user priority):

- PMI (GD&T) annotation extraction into `Asset.pmi`.
- Precise B-rep: XT/Parasolid (proprietary, undecodable), JT B-rep, STEP B-rep segments
  (ISO 14306:2017).
- Textures / texture-coordinate channels from JT shape data.
- External-reference (shattered) assembly resolution — v1 warns and placeholders.
- JT export.
- JT 8.x input (clear error instead).
- JT via stdin (`-`).
- Compiled codec accelerator (Cython/Rust).

## Sources

- Siemens *JT File Format Reference* (public PDF; v9.5 and v10.5 editions).
- [ISO 14306:2017](https://standards.iteh.ai/catalog/standards/iso/18fd19fd-36e6-452c-8494-aebda2332a93/iso-14306-2017)
  (JT 10.x) and ISO 14306:2012 (JT 8.1).
- [PyOpenJt](https://github.com/jriegel/PyOpenJt), [oce-jt](https://github.com/cbsghost/oce-jt)
  — GPL-2.0 licensing evidence.
- [Siemens JT Open Toolkit](https://plm.sw.siemens.com/en-US/plm-components/jt/jt-open-toolkit/)
  — commercial alternative, rejected.
- [JT (visualization format), Wikipedia](https://en.wikipedia.org/wiki/JT_(visualization_format))
  — format overview and standards history.
