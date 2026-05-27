# Fascat Plan

This is the single planning document for Fascat. It replaces the older long-form
implementation plan, TODO list, and gap analysis.

## Current Status

Fascat is a Python library and CLI for converting CAD data into realtime-ready
OpenUSD, glTF, OBJ, and STL assets.

The V1 pipeline is implemented:

```text
STEP CAD -> assembly -> tessellation -> healing/repair -> staging -> optimization -> LODs -> export
```

Working baseline:

- STEP import with assembly hierarchy, names, transforms, colors, metadata, and repeated-part handling.
- Mesh repair, staging, UV generation, normals, tangents, optimization, LODs, and scene operations.
- Export to USD, USDZ, glTF/GLB, OBJ, and STL.
- glTF quantization, meshopt compression, `MSFT_lod` metadata, and file-size budget reporting.
- OBJ vertex normals and smoothing directives.
- Analysis reports with topology, sliver, tiny-part, draw-call, and actual triangle self-intersection checks.
- CLI, Python API, TOML pipelines, docs site, release/build workflows, and CI coverage.

The project is no longer in "scaffold V1" mode. The next phase is polish:
make option behavior exact, document limitations clearly, and improve algorithms
that are currently conservative approximations.

## Principles

- Keep the public API small, explicit, and Pythonic.
- Preserve CAD hierarchy, transforms, names, colors, metadata, and instancing by default.
- Make lossy or approximate steps explicit in options, docs, and reports.
- Prefer warnings and partial success over silent data loss.
- Use proven geometry libraries for CAD kernels, tessellation, simplification, UV packing, and USD authoring.
- Work one feature at a time: implement, test, document, commit, push, and verify GitHub CI/docs before moving on.

## Recently Finished

- API parameter documentation was expanded so examples are backed by option descriptions.
- Docs rendering issues were fixed, including Python highlighting, heading anchors, and sidebar branding.
- Approximate or metadata-only operations now surface public report warnings, and dry-run diagnostics classify planned operations as exact, approximate, or metadata-only.
- Pipeline TOML files now validate supported keys, operation options, filter conflicts, and line-number diagnostics before conversion starts.
- Tangent handedness and staged normal behavior are covered by regressions for mirrored UVs, material-index remapping, OBJ smoothing, and glTF tangent export.
- Unsupported Draco compression now raises instead of silently writing uncompressed output.
- Export file-size budgets are recorded and warn when outputs exceed the budget.
- glTF LODs now include node-level `MSFT_lod` references in addition to Fascat extras.
- OBJ export writes normals, `f v//vn` faces, and smoothing directives.
- Self-intersection analysis now performs bounded triangle-triangle checks instead of counting AABB candidates.

## Unity Asset Transformer Parity

References reviewed on 2026-05-27:

- Import: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/sdktips/import-guidelines
- Stage: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/sdktips/stage-guidelines
- Optimize: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/sdktips/optimization-guidelines
- LODs: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/sdktips/lod-guidelines
- Export: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/sdktips/export-guidelines
- Tessellate: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/functions/tessellate
- Repair meshes: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/functions/repairmeshes
- Merge vertices: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/functions/mergevertices
- Delete degenerate polygons: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/functions/deletedegeneratepolygons
- Decimate to target: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/functions/decimatetotarget
- Unwrap UV: https://docs.unity.com/en-us/asset-transformer-sdk/2026.1/manual/functions/unwrap-uv

Fascat is aligned with the broad Unity workflow: import, repair, tessellate,
orient/stage attributes, optimize LOD0, generate LODs, and export. The parity
gap is depth: Unity exposes more CAD import toggles, BREP repair paths, UV
preparation tools, material/AO baking, real visibility processing, richer
decimation controls, and final export compression.

Parity gaps to track:

1. Import controls
   - Add explicit import toggles for design variants, PMI, product metadata, existing mesh preference, and multi-file imports.
   - Add cleanup operations for free points, line geometry, and post-tessellation BREP patch deletion.
   - Decide whether line geometry should be deleted, preserved as metadata, or tessellated into renderable tubes.

2. CAD and mesh repair depth
   - Add an open-shell repair workflow: detect single open-shell parts, merge or group them before BREP healing, and keep separate warnings for unstitched faces.
   - Improve BREP healing beyond the current sewing/fix-edge path: sliver-face removal, duplicate face handling, tolerance unification, and visible report warnings for unsupported backend work.
   - Extend mesh repair with true T-junction sewing, non-manifold edge cracking, and configurable face-orientation strategies for closed solids versus open shells.

3. Tessellation controls
   - Expose a separate sag-ratio option instead of overloading `relative=True`.
   - Support explicit override/reuse of existing tessellation when imported data already contains meshes.
   - Investigate CAD-parametric UV and tangent generation during tessellation, plus free-edge extraction for diagnostics.

4. UV staging
   - Add UV segmentation and seam planning, including sharp-edge seams and lines of interest.
   - Expose unwrap solver intent where the backend supports it: conformal versus isometric.
   - Add UV island merge, alignment, overlap checks, repack, normalize, and per-channel validation.
   - Make UV0 tileable and UV1 baking requirements explicit: UV0 may overlap; UV1 must fit in `[0,1]` with padding and no overlaps.

5. Materials and baking
   - Add material-library import and CAD-material-to-PBR mapping, including CSV or TOML mapping tables.
   - Replace current bake metadata with actual texture output for base color, opacity, roughness, metallic, normal, AO, and emissive maps.
   - Add ambient occlusion baking to textures and optionally to vertex colors for downstream decimation weights.
   - Add image cleanup: merge duplicate images, remove unused images, resize textures to platform budgets.

6. Optimization and draw-call reduction
   - Replace AABB containment with real occlusion/visibility removal.
   - Add loose and precise instance reconstruction for similar, separately modeled parts.
   - Improve merge planning so reports show draw-call savings, instance loss, memory growth, and culling impact.
   - Add retopology or proxy-mesh paths for cases where decimation and occlusion are not enough.

7. Decimation parity
   - Add iterative decimation thresholds for large meshes to control memory use.
   - Replace quality-criterion heuristics with measured geometric error.
   - Add texture-coordinate importance modes: preserve islands, preserve seams only, or ignore UVs.
   - Make topology protection explicit and measured, especially for holes and singularities.
   - Support AO or user-painted vertex weights as simplification constraints.

8. LOD parity
   - Preserve occurrence-level LOD chains and instance relationships across all LOD levels.
   - Add far-LOD generation that can merge to one mesh and one baked material for one-draw-call distant rendering.
   - Add LOD validation for screen coverage, monotonic triangle reduction, material simplification, and export runtime behavior.

9. Export parity
   - Add a real Draco encoder path with compression level and quantization settings, or keep `draco=True` rejected.
   - Add real KTX2/Basis texture output with quality, compression level, and max-resolution controls.
   - Add export cleanup for unused images/materials and file-size reports broken down by geometry, textures, and metadata.
   - Keep GLB as the preferred web/mobile runtime target while preserving USD/USDZ for OpenUSD workflows.

10. Platform budgets
   - Turn desktop, mobile, VR, and WebGL target triangle/draw-call budgets into documented profile checks.
   - Report when output exceeds the selected profile budget, not only an optional file-size budget.

## Near-Term Polish

These are the next small-to-medium tasks. They should be handled before larger
algorithmic work because they improve trust in the current tool.

1. Analysis polish
   - Add clearer lower-bound reporting when self-intersection checks hit `max_self_intersection_pairs`.
   - Add tests for coplanar overlap, endpoint contact, and adjacent-triangle exclusions.
   - Keep compatibility keys only where needed and document their migration path.

2. Unity parity matrix
   - Add a compact docs table that maps Unity-inspired capabilities to Fascat status: implemented, approximate, unsupported, deferred.
   - Link each approximate feature to its report warning and next implementation step.

## Larger Algorithmic Work

These need more design and should not be mixed into documentation or diagnostics commits.

1. True occlusion removal
   - Replace AABB containment with visibility testing.
   - Support part, submesh, and triangle granularity.
   - Make strategy and hemispherical evaluation change the algorithm, not just metadata.

2. Better hole removal
   - Add a real BREP path for cylindrical or feature-level holes when source shape data is available.
   - Respect through, blind, and surface hole options.
   - Improve diameter measurement beyond boundary-loop max distance.

3. Material baking
   - Generate actual texture atlas files, not only flat merged materials.
   - Reuse xatlas UVs where possible.
   - Export atlas references through glTF and USD material bindings.
   - Include AO baking and texture resizing/compression prep.

4. Error-bounded simplification
   - Replace `criterion="quality"` ratio heuristics with measured geometric error.
   - Preserve hard edges, boundary edges, material seams, UV seams, and selected CAD features.
   - Report achieved error and triangle reduction.
   - Add iterative processing and vertex-weight constraints for very large meshes.

5. BREP healing depth
   - Implement or delegate sliver-face removal.
   - Improve sewing, small edge handling, and face/wire repair before tessellation.
   - Add open-shell detection and unstitched-face handling before repair.
   - Keep warnings visible when a backend cannot perform a requested operation.

6. PMI and metadata output
   - Add STEP AP242 PMI import tests.
   - Decide how PMI should appear in USD and glTF: metadata only, annotation geometry, or both.
   - Preserve stable metadata paths through merge, replace, and export operations.

7. Large assembly scaling
   - Reduce full-asset copying in operations that only touch selected parts.
   - Add memory and time benchmarks for large assemblies.
   - Consider streaming or lazy mesh payloads for heavy STEP imports.

8. Runtime compression
   - Add a real Draco backend only if there is a reliable Python encoder path.
   - Treat texture compression as a packaging step with emitted files, not metadata-only intent.
   - Add KTX2/Basis output only after texture assets are real files in the export graph.

9. UV pipeline depth
   - Add seam segmentation, unwrap method selection, island merging, packing, normalization, and overlap checks.
   - Keep UV1 baking constraints separate from UV0 tileable texture constraints.

10. Instance reconstruction
   - Detect similar separately modeled parts and rebuild shared mesh instances where safe.
   - Report memory/file-size savings and any metadata or material differences that prevent instancing.

## Correct Deferrals

These are intentionally outside the immediate plan unless a user need changes the priority:

- Full Unity-level CAD format coverage such as IGES, Parasolid, JT, CATIA, NX, and native SolidWorks.
- Convex decomposition and physics proxy generation.
- Advanced retopology and subdivision workflows.
- GPU-specific runtime packaging beyond standards-aligned glTF/USD output.
- Animation and time-varying CAD data.

## Operating Checklist

For each planned feature:

1. Confirm the intended behavior in docs or tests first.
2. Keep the change scoped to one user-visible outcome.
3. Add or update focused tests.
4. Update API/reference docs when public behavior changes.
5. Run `make fmt-check`, `make lint`, `make docs`, and `make ci`.
6. Commit with the repo convention.
7. Push and verify GitHub CI and Docs workflows are green.
