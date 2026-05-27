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
- Analysis reports now mark truncated self-intersection checks as lower bounds and cover coplanar overlap, endpoint contact, and adjacent-triangle exclusions.
- Reference docs now include a Unity-inspired capability matrix with implemented, partial, approximate, and metadata-only behavior plus the next parity step.
- Unsupported Draco compression now raises instead of silently writing uncompressed output.
- Export file-size budgets are recorded and warn when outputs exceed the budget.
- glTF LODs now include node-level `MSFT_lod` references in addition to Fascat extras.
- OBJ export writes normals, `f v//vn` faces, and smoothing directives.
- Self-intersection analysis now performs bounded triangle-triangle checks instead of counting AABB candidates.
- Conversion reports now include a Unity-style `preflight` checklist before
  expensive operations run, covering patch cleanup, orientation, UV/tangent
  ordering, AO bake prerequisites, LOD0 optimization, and export backend gaps.
- BREP healing and mesh repair reports now include unit-aware tolerance policies
  with effective source/local units, declared target units, meter conversions,
  and implemented versus missing repair backend operations.
- Tessellation now warns when retained BREP patches, CAD face groups, or
  material splits are likely to increase submesh, draw-call, or export-size
  pressure.
- Tessellation profiles now include a size-adaptive helper that generates
  per-part sag, sag-ratio, angle, and max-polygon-length settings from
  bounding-box bands.
- Platform budgets now record Unity reference triangle and draw-call ranges in
  profile definitions, conversion reports, and documentation tables.
- Decimation now records RAM estimates, budget-allocation mode,
  configurable iterative-threshold runtime controls, and actual simplification
  pass counts in metadata and report fields.
- Staging now warns when bake-domain UVs are only unwrapped without a separate
  repack/padding pass and records that missing repack status in metadata.
- Staging now records UV island counts, pack efficiency, normalized-space
  utilization, and conformal/isometric distortion metrics per channel.
- Mesh repair now detects non-orientable shared-edge cycles before face
  orientation and warns when Mobius-like topology cannot be fixed by winding.
- Draw-call reports now separate mesh count, referenced material count,
  submesh/material slots, instances, reused instances, and merged batches.
- Merge and scene-batching reports now warn when draw-call reduction removes
  reusable instances and record export-advisor metrics for GLB size, memory, and
  culling tradeoffs.
- Mesh repair now records before/after T-junction counts and warns that
  remaining T-junctions still need a sewing backend.
- Mesh repair now records before/after nearby boundary-gap counts and warns
  that remaining gaps still need a stitching backend.
- Mesh repair now records before/after flipped closed-component counts and
  flips coherent inward shells during winding repair when possible.
- glTF write reports now include a runtime compatibility matrix for Unity
  glTFast, web, mobile, and XR targets with extension state, support, and
  fallback notes.
- Augmented-reality and mixed-reality profiles now expose stricter AR/XR device
  budgets through Python, CLI selection, tests, and docs.
- Custom target-device profile files now load TOML/JSON budget overlays on a
  built-in base profile and surface the resolved budget through Python, CLI
  dry-run output, and conversion report budget checks.
- Custom target-device triangle budgets now seed the profile optimization target
  and derive a matching vertex budget when one is not provided.
- Custom target-device budgets now record supported compression methods and
  runtime glTF extension caps, and profile budget reports warn when emitted
  runtime dependencies exceed those caps.
- Vertex merging is now available as a standalone Unity-style operation through
  Python, CLI flags, and TOML pipelines with attribute/material-boundary
  protection, unit-aware tolerance reporting, and before/after merge counts.
- Degenerate-polygon cleanup is now available as a standalone Unity-style
  operation through Python, CLI flags, and TOML pipelines with area-threshold
  controls, scoped selection support, no-op reports, and before/after counts.
- Unity-style UV policy controls for sharp edges as seams and forbidden overlap
  are now exposed through Python, CLI flags, and TOML pipelines, with
  per-channel requested/enforced metadata and warnings when the xatlas backend
  can only record policy intent.
- Conversion reports now include a resolved conversion manifest with the
  effective profile, import options, direct or pipeline operation settings, and
  export settings needed to reproduce a run.
- STEP import reports now include Unity-style `import_decisions` for requested,
  effective, `honored`, `approximated`, `unsupported`, `disabled`,
  `not_present`, and `backend_default` import choices, plus per-part
  loaded-representation records and deleted construction-only node records.
- Tessellated parts now record attribute provenance for positions, triangles,
  normals, tangents, UVs, face groups, free-edge diagnostics, and BREP patch
  state so users can tell what came from tessellation versus imported meshes or
  later staging.
- Explicit decimation now exposes pre-cleanup for unused UV channels and
  tangents, records removed/preserved attribute metadata, and warns when
  preserved UV seams or islands can reduce simplification efficiency.
- Explicit decimation now reports topology/material/UV protection pressure with
  protected hard-edge, hole-boundary, material-boundary, UV-seam, silhouette,
  and total feature-face counts.
- Explicit decimation now uses the selected profile or target-device triangle
  budget as its target when `--decimate` is enabled without a manual
  `--target-triangles` or `--ratio`.
- LOD generation now reports source, added-LOD, and full-chain vertex/triangle
  counts plus estimated mesh payload bytes, making extra-level memory and
  export-size tradeoffs visible.

## Unity Asset Transformer Parity

References reviewed and re-audited on 2026-05-27:

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
preparation tools, normal/orientation controls, material/AO baking, real
visibility processing, richer decimation controls, staged LOD workflows, and
final export compression.

Comparison snapshot:

| Area | Fascat today | Missing for closer Unity parity |
| --- | --- | --- |
| Import | STEP-centric import with hierarchy, transforms, metadata, colors, repeated-part handling, PMI presence reporting, existing-mesh reuse intent, construction-only point/line cleanup controls, source-space normalization reporting, import-decision reports, per-part loaded-representation reports, and BREP patch cleanup reporting after tessellation. | True multi-file/multi-root import semantics, design-variant import, typed/visual PMI, mixed BREP construction-curve cleanup, native CAD/JT/IFC/Parasolid/IGES coverage, and richer loaded-representation coverage for existing tessellations, typed PMI, variants, and product metadata. |
| Repair and tessellation | BREP sewing/fix-edge path, mesh duplicate/degenerate/T-junction/boundary-gap/flipped-component diagnostics, unit-aware repair tolerance reporting, sag/sag-ratio/angle/max-length controls, bounding-box-derived tessellation helpers, free-edge diagnostics, reusable existing mesh control, retained patch / submesh risk warnings, and tessellation attribute-provenance metadata. | Open-shell grouping, unstitched-face handling, T-junction sewing, boundary-gap stitching, non-manifold edge cracking, tolerance-based overlapping-surface/z-fighting cleanup, non-orientable strip cracking, topology-only vertex connectivity with split render attributes, selectable face/normal orientation strategies, real tessellation-time tangent/UV/free-edge geometry generation controls, CAD-derived UV modes, targeted tessellation by material/metadata/curvature, and optional free-edge geometry output. |
| Staging | Normal/tangent generation, box/unwrap/lightmap UV modes, UV copy/normalization, UV validation, UV island/distortion/packing diagnostics, material normalization, duplicate-material merge, and metadata-only atlas intent. | Unity-style UV0 tileable versus UV1 bake workflows with segmentation, sharp-edge seam and forbid-overlap UV policies, lines of interest, island merge/alignment, real repack/padding/share-map controls, material-library mapping, real atlas textures, AO/lightmap baking, and texture cleanup. |
| Optimization | Mesh simplification, measured error reporting, sampled occlusion removal, exact instance reconstruction, scene merge/split utilities, draw-call breakdown reports, UV-importance modes, and pre-decimation cleanup for unused UVs/tangents. | Global assembly target allocation with iterative memory thresholds, real geometric-error bounded simplification, AO/user-weighted decimation, cleanup for vertex colors/weights, standard/advanced occlusion backends, retopology/proxy mesh generation with normal-map transfer, symmetry-aware loose/precise instance reconstruction, duplicate image/material cleanup, and merge reports that quantify culling, memory, and file-size tradeoffs. |
| LODs | LOD ratios, screen-coverage metadata, validation, skipped-part reporting, per-level mesh-payload tradeoff reports, and glTF `MSFT_lod` metadata. | Occurrence-level LOD group authoring with preserved instance relationships, optimized LOD0 as master asset, explicit conservative LOD0 versus destructive distant-LOD policy, far-LOD one-mesh/one-material baking, switching-distance validation, and engine-specific runtime export profiles. |
| Export | USD/USDZ, glTF/GLB, OBJ, STL, glTF quantization, meshopt, extension reporting, file-size budgets, and rejection of unsupported Draco/KTX2 requests. | Real Draco compression settings, KTX2/Basis texture output, texture resize and PNG/JPEG fallback controls, unused texture cleanup, baseline-versus-optimized size comparisons, expected-versus-measured export size ladders, Unity/glTFast-oriented profiles, and web/mobile/VR/XR budget presets backed by runtime measurements. |

Function-level parity notes from the linked Unity pages:

| Unity reference | Fascat today | Gap to track |
| --- | --- | --- |
| Tessellate models | Sag, sag-ratio, angle, max-polygon-length, per-part overrides, size-adaptive helpers, and attribute-provenance metadata are represented. | Add real tessellation-time tangent/UV/free-edge geometry generation controls, CAD-derived UV modes, optional free-edge geometry output, and material/metadata/curvature-driven tessellation profiles. |
| Repair meshes | Duplicate and degenerate cleanup plus standalone degenerate-polygon deletion, T-junction, boundary-gap, non-manifold, and orientation diagnostics are reported. | Implement true T-junction sewing, boundary stitching, non-manifold edge cracking, tolerance-based overlap/z-fighting cleanup, non-orientable strip cracking, and explicit face/normal orientation strategies. |
| Merge vertices | Standalone `merge_vertices` is exposed across Python, CLI, and TOML with normals, tangents, UV, and material-boundary protection plus before/after reports. | Add topology-only connectivity merging that can preserve hard-edge, UV, and material seams as split render attributes; also add stronger cross-bucket tolerance merging and richer reports for skipped merges by protection reason. |
| Delete degenerate polygons | Standalone `delete_degenerate_polygons` is exposed across Python, CLI, and TOML with area-threshold controls, selection support, no-op reports, unit-aware area reporting, and before/after counts. | Extend cleanup beyond zero-area triangles to tolerance-based overlapping or z-fighting polygons. |
| Decimate to target | Target count, ratio, UV-importance modes, topology protection counts, RAM estimates, configurable iterative threshold/pass reports, measured-error reports, and pre-cleanup for unused UVs/tangents exist. | Add enforced geometric error bounds, AO/user-weighted decimation, and cleanup for future vertex colors/weights. |
| Unwrap UV | UV0/UV1 unwrap intent, solver method, iteration, tolerance, sharp-edge seam and forbid-overlap policy intent, distortion, and packing diagnostics are represented. | Add destination-channel control, channel-as-destination behavior when lines of interest define islands, backend-enforced seam policies, create-seams-from-lines-of-interest, seam graph metadata, island merge/alignment, and real repack/padding/share-map controls. |

Second-pass gaps from the Unity references:

- Distinguish UV unwrapping from bake-ready UV packing everywhere. Unity's
  unwrap function only flattens islands; bake/lightmap UVs still require repack,
  padding, overlap checks, and normalization. Fascat should warn when UV1 is
  unwrapped but not packed before AO, lightmap, or material baking.
- Mesh repair now includes orientability diagnostics for face-orientation
  repair, including non-orientable/Mobius-strip-like polygon strips, so
  orientation warnings are not limited to non-manifold edge counts.
- Make decimation memory planning explicit: estimate RAM from polygon count,
  report when iterative decimation should be used, and explain how a global
  target is allocated across parts so sparse walls/simple parts stay intact.
- Decimation memory planning now has runtime controls, not just diagnostics:
  `iterative_threshold` triggers intermediate simplification passes and reports
  actual simplification and iterative pass counts.
- Use Unity's broad desktop, mobile, VR, and WebGL ranges as report context
  when tuning future target-device presets and measured runtime profiles.
- Add export comparison reports that show unoptimized GLB, optimized GLB,
  geometry-compressed GLB, and geometry-plus-texture-compressed GLB deltas once
  real Draco and KTX2 outputs exist.
- Target-device/profile triangle budgets now seed explicit decimation targets
  when `--decimate` is enabled without a manual target or ratio. Remaining
  work: selected devices should also drive LOD ratios, texture resize limits,
  compression choices, and cleanup defaults.
- Conversion reports now include a resolved conversion manifest that records the
  effective import, tessellation, staging, optimization, LOD, and export
  settings so Unity-style module-property choices are reproducible from a
  report.
- Treat orientation as its own post-repair stage, not just a side effect of mesh
  cleanup or normal generation. Unity separates polygon orientation, normal
  orientation, and open-shell/unstitched-face handling; Fascat should report
  those decisions separately.
- Model Unity's connectivity-oriented vertex merge more accurately. Unity can
  merge same-position vertices with different render attributes to recreate
  connectivity; Fascat currently avoids collapsing those seams by default. Add
  split topology/render-attribute support or corner attributes so hard edges,
  UV seams, and material boundaries survive while topology is still connected.
- Extend repair beyond exact duplicate polygons. Unity repair targets
  overlapping/z-fighting surfaces and can crack non-orientable strips before
  orientation; Fascat should add tolerance-based overlap cleanup and optional
  non-orientable strip cracking instead of only reporting those risks.
- Track Unity's lines-of-interest UV workflow explicitly. Segmenting seams,
  optionally creating seams from LoI, unwrapping, merging, aligning, repacking,
  and normalizing should be modeled as distinct UV steps with per-channel
  metadata.
- Unity-style automatic UV mapping controls are now explicit: sharp edges as
  seams and forbidden overlap are first-class UV policies with per-channel
  requested/enforced metadata. Remaining work is backend-enforced seam and
  overlap prevention rather than intent plus validation.
- Expose Unity-style function-level repair steps where useful. `repair` can stay
  the high-level default, but face orientation, normal orientation, and patch
  cleanup still need standalone operations for reproducible expert pipelines.
- Add an export-aware merge-versus-instance advisor. Unity's export guidance
  favors preserving instances for file size even when merging can reduce draw
  calls, so Fascat should warn when a merge helps batching but hurts GLB size,
  memory, or culling.
- Unity-style import module-property decisions are now first-class report data:
  load PMI, load variants, prefer existing meshes, delete free vertices, delete
  lines, metadata groups, and space normalization each report requested,
  effective, `honored`, `approximated`, `unsupported`, `disabled`,
  `not_present`, or `backend_default` state. Remaining work is to connect
  delete-patch decisions to tessellation-time BREP retention and cleanup.
- Tessellation-time attribute generation is now explicit in metadata. Positions,
  triangles, normals, tangents, UVs, face groups, free-edge diagnostics, and
  BREP patch state report whether they came from tessellation, an imported mesh,
  were disabled, missing, diagnostic-only, or deferred to staging. Remaining
  work is real CAD-derived UVs, tessellation-time tangents, and optional
  free-edge geometry output.
- Pre-decimation cleanup now removes unused UV channels and tangents before
  simplification and reports removed/preserved attribute streams plus UV
  constraint warnings. Remaining work is vertex-color/weight cleanup and
  measured simplification-efficiency deltas.
- Separate LOD policy from ordinary optimization policy. LOD0 should stay
  conservative and close-view safe, while LOD1+ can be progressively more
  destructive and the farthest levels can simplify materials, culling
  granularity, and hierarchy when the export profile asks for a proxy LOD.
- Add export size ladders based on runtime assets, not source CAD files. Reports
  should compare unoptimized GLB, optimized GLB, geometry-compressed GLB, and
  geometry-plus-texture-compressed GLB when those artifacts exist, and should
  keep warning that CAD source size is not a meaningful baseline for mesh export
  size.

Parity gaps to track:

1. Workflow validation
   - Pipeline files now expose Unity-style ordering advisories through `PipelineSpec.advisories()`, dry-run `pipeline_advisories`, and conversion report warnings.
   - The advisor warns when a pipeline decimates before repair, computes tangents before UV0, bakes AO without UV1, or generates LODs before LOD0 optimization. Compression backend requests are still rejected by the CLI/options layer until real encoders exist.
   - Conversion reports now include a `workflow_summary` step that maps Unity-inspired preparation stages to run/skipped status and exact, approximate, or metadata-only levels, including import cleanup, orientation, UV preparation, material baking, LOD generation, export compression, and export.
   - Conversion reports now include a `preflight` step before pipeline or profile operations run, with checklist warnings for missing patch cleanup, face/normal orientation, UV-before-tangent ordering, AO bake UV1 prerequisites, LOD generation without LOD0 optimization, and glTF texture/compression backend gaps.
   - Conversion reports now include a `conversion_manifest` step with the resolved profile, import options, direct or pipeline operation settings, and export options needed to reproduce a run.

2. Import controls
   - Reference docs now include a supported-format parity matrix. Unity's baseline covers many CAD and mesh formats; Fascat currently centers on STEP input and USD/glTF/OBJ/STL output, with IGES, Parasolid, JT, native CAD, IFC, 3MF, and QIF explicitly deferred.
   - Explicit import toggles now cover product metadata, properties, layers, validation properties, PMI, design variants, existing mesh preference, and multi-file import intent across Python, CLI, and TOML. Unsupported design-variant and multi-file import requests report warnings instead of silently claiming support, and `import_decisions` records requested/effective state for each import choice.
   - Define true multi-file import semantics: multiple input paths should produce deterministic multi-root assemblies, shared material/image namespaces, stable source-file metadata, and warnings for failed members instead of all-or-nothing failure.
   - Import cleanup now exposes `delete_free_vertices` and `delete_lines` for construction-only point and line shapes across Python, CLI, and TOML. Import reports include cleanup counts, and preserved parts record loaded representation plus source topology counts.
   - Tessellated parts now record `brep_patch_cleanup=deleted` or `retained` and `source_shape_retained`, matching `keep_brep` behavior.
   - Remaining work: decide whether mixed BREP construction curves should be deleted, preserved as metadata, or tessellated into renderable tubes.
   - Source unit, source up-axis, source handedness, target unit, target up-axis, and target handedness normalization controls now apply a root transform, update the asset's declared working space, and record the exact transform in import metadata and reports.
   - Import reports now include per-part loaded representation records for BREP, construction points/lines, empty shapes, source topology counts, and deleted construction-only nodes. Remaining work: existing tessellation payloads, typed PMI, variants, and richer product metadata are still limited by importer support.
   - Tessellation now reports when retained BREP patches, CAD face groups, or material splits are likely to increase submesh, draw-call, or export-size pressure.

3. CAD and mesh repair depth
   - Add an open-shell repair workflow: detect single open-shell parts, merge or group them before BREP healing, and keep separate warnings for unstitched faces.
   - Improve BREP healing beyond the current sewing/fix-edge path: sliver-face removal, duplicate face handling, tolerance unification, and visible report warnings for unsupported backend work.
   - Mesh repair now deletes duplicate polygons and records before/after duplicate, degenerate, boundary-edge, and non-manifold metrics. Standalone degenerate-polygon cleanup now exposes the degenerate deletion path as a reproducible operation with before/after no-op reports.
   - Add tolerance-based overlapping-surface and z-fighting cleanup, not only exact duplicate-polygon deletion.
   - Extend mesh repair with true T-junction sewing, non-manifold edge cracking, and configurable face-orientation strategies for closed solids versus open shells.
   - Mesh repair now detects non-orientable strips before face orientation so Mobius-like topology is reported separately from ordinary flipped faces.
   - Add optional cracking of non-orientable/Mobius-like strips before face orientation when a backend can split them safely.
   - Add explicit face-orientation and normal-orientation report steps with selectable strategies for exterior solids, single-sided open shells, unstitched-face groups, and preserved two-sided surfaces.
   - Add missing-normal generation controls for sharp-edge angle, area weighting, and override behavior.
   - Standalone vertex merging now rebuilds connectivity without collapsing intentional material, normal, tangent, or UV seams by default. Remaining work: support topology-only connectivity merging with split render attributes, report skipped merge reasons, and improve cross-bucket tolerance matching.
   - Mesh repair now records before/after T-junction, nearby boundary-gap, and flipped closed-component counts. It warns that sewing/stitching remains unavailable and warns when outward orientation is still not produced.
   - BREP healing and mesh repair now report unit-aware tolerance policy: effective source/local units, declared target units, meters-per-unit conversions, vertex-merge and degenerate-polygon cleanup status, and missing T-junction/non-manifold backend operations.

4. Tessellation controls
   - Sag-ratio is now a first-class tessellation option across Python, CLI, TOML pipelines, per-part overrides, reports, and OCCT backend parameter mapping. `relative=True` remains for compatibility when `sag_ratio` is unset.
   - Existing imported meshes now have an explicit `reuse_existing_meshes` control across Python, CLI, TOML pipelines, per-part overrides, and reports. The default preserves imported meshes; disabling it retessellates from source BREP where available.
   - Free-edge tessellation diagnostics are now available through `free_edge_report` across Python, CLI, TOML pipelines, per-part overrides, metadata, and report warnings. Tessellation attribute provenance now records whether positions, triangles, normals, tangents, UVs, face groups, free-edge diagnostics, and BREP patches came from tessellation or imported meshes. Remaining work: investigate CAD-parametric UV and tangent generation during tessellation.
   - Add optional free-edge geometry output or retention, separate from diagnostics, for wire overlays, boundary inspection, and import cleanup validation.
   - Size-adaptive tessellation helpers now generate per-part `part_settings` from bounding-box diagonal bands, so sag, sag-ratio, angle, and polygon-length defaults can vary by part size.
   - Remaining targeted-profile work: material, metadata, curvature, or filter driven tessellation so shiny/high-detail parts can use finer criteria than bulk structural parts.
   - Expose CAD-derived UV generation modes, including none, intrinsic surface UVs, and conformal/scaled UVs, instead of only post-mesh unwrapping.
   - Max polygon length is now exposed separately from cleanup subdivision. `max_edge_length` still subdivides geometry; `max_polygon_length` drives quality-report `long_edges`, metadata, and warnings for long tessellated edges that may cause lighting artifacts.

5. UV staging
   - Extend existing box UVs into Unity-style AABB projection controls: local versus shared/global AABB, real-world UV scale or `uv3dSize`, destination channel, override policy, and unit reporting.
   - Add UV segmentation and seam planning, including sharp-edge seams, material-boundary seams, user-supplied seam curves, and lines of interest.
   - Automatic UV mapping policy controls equivalent to Unity's `sharpToSeam` and `forbidOverlapping` are now exposed across Python, CLI, TOML pipelines, metadata, and reports. The current xatlas backend records them as intent and validates overlaps after generation; remaining work is backend-enforced seam and overlap prevention.
   - Add a complete UV workflow model: segment, unwrap, optionally merge islands, align tileable UV0 islands, repack UV1, normalize, validate overlaps, and record which steps ran per channel.
   - Add lines-of-interest seam controls equivalent to Unity's create-seams-from-LoI path, with persisted seam graph metadata and warnings when the backend falls back to existing UV islands.
   - Staging now warns when bake-domain UVs were only unwrapped and not repacked, because unwrap alone does not prove islands were packed into `[0,1]` with padding for lightmap, AO, or material baking.
   - Add affine UV island merge and alignment controls for tileable UV0 workflows, including allowed transforms, polygon weighting, and rotation-step quantization.
   - Unwrap solver intent is now accepted across Python, CLI, TOML pipelines, metadata, and reports with `default`, `conformal`, and `isometric` values. The current xatlas backend records non-default solver methods as intent and warns that it cannot enforce them directly.
   - Unwrap iteration and tolerance controls are now accepted across Python, CLI, TOML pipelines, metadata, and reports. Remaining work: add a backend that enforces those controls and reports solver failure or excessive distortion.
   - UV layout diagnostics now record island count, pack efficiency, normalized-space utilization, conformal angle distortion, and isometric edge-length distortion per channel.
   - UV0-to-UV1 copy is now supported through `uv1="copy_uv0"` / `--uv1 copy-uv0`, with copied/missing-source metadata and warnings. UV normalization is now explicit through `normalize_uvs=(...)` / `--normalize-uvs`, with original-bounds metadata and missing-channel warnings. UV validation now records per-channel domains, bounds, unit-domain status, validation status, distortion metrics, and packing efficiency. Remaining work: UV island merge, alignment, repack, padding/resolution/share-map controls, overlap removal, uniform versus non-uniform normalization, shared versus per-part UV space, destination-channel controls, and null-island handling.
   - Make UV0 tileable and UV1 baking requirements explicit: UV0 may overlap; UV1 must fit in `[0,1]` with padding and no overlaps.
   - Add UV1 bake packing controls for atlas resolution, pixel padding, share-map behavior, uniform versus non-uniform scaling, shared versus per-part UV space, overlap removal, destination channel, null-island handling, and normalized-space utilization reports.
   - Tangent lifecycle validation now warns when UV0 is missing, invalidates tangents after UV edits, and records generated, regenerated, preserved, invalidated, missing, or dropped tangent states on mesh and asset metadata.
   - Tangent generation can now use an explicit source UV channel and preserves existing tangents by default, with `override_tangents` available across Python, CLI, TOML pipelines, metadata, and reports when regeneration is required.
   - Explicit decimation can now strip UV/tangent attributes through `uv_importance="ignore"` or preserve seams and then drop UVs with `uv_importance="preserve_seams"`.

6. Materials and baking
   - Add material-library import from glTF/GLB or native material-library assets and CAD-material-to-PBR mapping, including CSV or TOML mapping tables.
   - Add material mapping diagnostics that report replacements, missing library materials, source materials with no mapping, and unresolved texture dependencies.
   - Add material-combine planning that can bake many materials into one atlas when draw-call reduction is more important than material editability.
   - Replace constant embedded factor maps with real atlas/raster texture output for base color, opacity, roughness, metallic, normal, AO, and emissive maps.
   - Add high-poly-to-proxy normal map baking for retopology or aggressive far-LOD workflows.
   - Add real ambient occlusion baking to textures and optionally to vertex colors for downstream decimation weights, with explicit resolution, padding, sample-count, target-channel, bent-normal, and denoise/filter controls.
   - Add material and image cleanup: merge duplicate materials/images, remove unused images, resize textures to platform budgets, and keep PNG/JPEG fallbacks when KTX2 is unavailable.

7. Optimization and draw-call reduction
   - Add acceleration structures, confidence metrics, and optional raster/GPU backends to the new sampled occlusion removal.
   - Expose standard versus advanced occlusion-removal parameters such as resolution, sphere count or ray direction set, adjacency depth, hemisphere-only evaluation, cavity preservation, and GPU/backend requirements.
   - Add loose and precise instance reconstruction for similar, separately modeled parts, including configurable similarity thresholds and symmetry/mirror handling.
   - Merge and scene-batching reports now show draw-call savings, lost reusable instances, added merged batches, and export-advisor warnings when merging destroys repeated geometry.
   - Export-aware merge advisors now recommend preserving or reconstructing instances when file size, memory, or culling is more important than reducing draw calls.
   - Draw-call budget analysis now separates mesh count, referenced material count, submesh/material slots, instances, reused instances, and merged batches.
   - Add retopology or proxy-mesh paths for cases where decimation and occlusion are not enough.
   - Add dedicated cleanup for unused texture coordinates, duplicate materials, and duplicate images before draw-call and file-size optimization.

8. Decimation parity
   - Add Unity-style global target allocation across a selected assembly while decimating at part level, so sparse/simple parts stay intact and dense parts carry most of the reduction, with before/after allocation reports.
   - Decimation now records RAM estimates using the Unity 5 GB per million polygons rule of thumb, reports global versus per-part budget allocation, exposes a configurable iterative threshold, and records actual simplification and iterative pass counts.
   - Target-device/profile triangle budgets now seed explicit decimation targets when `--decimate` has no manual target or ratio. Remaining work: named XR/HoloLens-style decimation presets and more device-specific simplification policy.
   - Replace quality-criterion heuristics with measured geometric error.
   - Explicit decimation now supports UV importance modes: preserve full UV islands, preserve seam topology only, or ignore UVs by stripping UV/tangent attributes before simplification.
   - Pre-decimation cleanup now removes unused UV channels and tangents, records removed/preserved attribute metadata, and reports when preserved UVs can make simplification less efficient. Remaining work is vertex-color/weight cleanup and measured efficiency deltas.
   - Topology/material/UV protection metrics now record protected hard-edge, hole-boundary, material-boundary, UV-seam, silhouette, and total feature-face counts. Remaining work is singularity-specific protection and protection-versus-reduction efficiency deltas.
   - Support AO or user-painted vertex weights as simplification constraints.
   - Explicit decimation now records the requested keep ratio when derivable and warns when the request keeps less than 20% of source triangles for close-view LOD0 assets.
   - Keep skinning, bones, and animation preservation out of scope until Fascat supports animated mesh imports.

9. LOD parity
   - Represent LOD0 as the optimized master asset and LOD1+ as occurrence-level LOD groups with stable naming, parentage, and instance references.
   - Preserve occurrence-level LOD chains and instance relationships across all LOD levels.
   - Add far-LOD generation that can merge to one mesh and one baked material for one-draw-call distant rendering.
   - Add LOD validation for screen coverage, monotonic triangle reduction, material simplification, switching distances, and export runtime behavior.
   - LOD generation now reports source, added-LOD, and full-chain vertex/triangle counts plus estimated mesh payload bytes so users can choose useful levels instead of generating wasteful chains.
   - Add LOD generation reports that show whether each level reused instances, merged materials, baked textures, or changed culling granularity.
   - Non-mesh or untessellated selections are now skipped with part metadata, generated/skipped counts, and report warnings instead of quietly producing partial chains.
   - Add engine-specific LOD export metadata or profiles for Unity, Unreal, and standards-based glTF runtimes, including switching-distance validation.

10. Export parity
   - Add a real Draco encoder path with compression level and quantization settings, or keep `draco=True` rejected.
   - Add real KTX2/Basis texture output with quality, compression level, and max-resolution controls.
   - Add export cleanup for unused images/materials and file-size reports broken down by geometry, textures, and metadata.
   - Add texture-resize preprocessing with before/after dimensions, byte estimates, and per-profile maximums before KTX2/PNG/JPEG export decisions.
   - glTF write reports now list emitted runtime extensions, required extensions, `extras.fascat` metadata, unsupported Draco/KTX2 outputs, expected runtime support, and target compatibility notes with fallback behavior.
   - Add Unity/glTFast-oriented GLB export profiles that combine extension support notes, Draco/KTX2 settings, fallback choices, and runtime compatibility warnings.
   - Runtime extension compatibility reports now cover Unity glTFast, web, mobile, and XR targets for `MSFT_lod`, `EXT_meshopt_compression`, `KHR_draco_mesh_compression`, `KHR_texture_basisu`, quantization, and fallback behavior.
   - Add baseline-versus-optimized export comparisons so reports show how much each preparation step changed file size, and warn when draw-call merging increases export size by breaking instancing.
   - Add format-aware texture export policy and reporting: prefer KTX2/Basis for glTF/GLB, use PNG/JPEG fallbacks for texture-capable non-glTF exports, remove unused images before export, and warn when users compare source CAD file size directly against runtime mesh exports.
   - Add named web, mobile, desktop, VR, AR/XR, and custom-device export presets that combine geometry compression, texture compression, texture resizing, and cleanup choices.
   - Keep GLB as the preferred web/mobile runtime target while preserving USD/USDZ for OpenUSD workflows.
   - Expose Draco quantization bits for positions, normals, UVs, and vertex colors once a real encoder is available.
   - Expose PNG/JPEG fallback texture export settings, including PNG compression and JPEG quality, for formats or environments where KTX2 is not available.

11. Platform budgets
   - Desktop, WebGL/web, mobile, and VR profiles now include documented target-FPS, triangle, vertex, per-mesh vertex/index-buffer, texture-resolution, texture-memory, estimated load-time, and draw-call budgets.
   - Conversion reports now include a `profile_budget` step for selected-profile budget status and warnings when output exceeds profile triangle, vertex, per-mesh vertex/index-buffer, texture-resolution, texture-memory, estimated load-time, or draw-call budgets.
   - Profile budgets now include explicit Unity reference ranges for each broad profile so users can see how Fascat's stricter defaults compare with Unity's desktop, mobile, VR, and WebGL guideline ranges.
   - Augmented-reality and mixed-reality profiles now model stricter AR/XR device caps, and custom target-device overrides are supported through profile files.
   - Custom target-device profiles can now be loaded from TOML/JSON as budget overlays and surfaced in reports with resolved FPS, triangle, vertex, draw-call, texture, load-time, compression-support, and runtime-extension caps.
   - Custom target-device triangle budgets now seed profile optimization targets and explicit decimation targets instead of only warning after conversion. Remaining work: use selected platform budgets to seed LOD choices, texture-resize choices, and export-compression defaults.
   - The platform-budget checklist is complete at diagnostic-report level; future work is measured engine/runtime load profiling.

## Near-Term Polish

The current near-term polish list is complete. Choose the next scoped item from
larger algorithmic work below, then implement, test, document, commit, push, and
verify CI/docs before moving on.

## Larger Algorithmic Work

These need more design and should not be mixed into documentation or diagnostics commits.

1. True occlusion removal - measured confidence pass complete
   - Replaced AABB containment with deterministic sampled visibility rays.
   - Supports part, submesh/material-group, and triangle granularity.
   - Strategy changes the direction set, and hemispherical evaluation restricts rays to top/side views.
   - Output metadata now records candidate counts, face counts, sample counts, visible/hidden samples, sample coverage, direction coverage, and an occlusion confidence score.
   - Remaining polish: add acceleration structures and optional raster/GPU backends for very large production meshes.

2. Better hole removal - first mesh-classification pass complete
   - Mesh fallback now classifies boundary loops as through, blind, or surface and respects the enabled hole types.
   - Diameter filtering now uses planar-span measurement instead of boundary-loop diagonal distance.
   - Remaining polish: add real BREP feature-level removal for closed cylindrical holes and blind pockets when source shape data is available.

3. Material baking - first embedded-texture pass complete
   - `bake_materials` now emits constant embedded texture maps from material factors.
   - glTF export writes baked base-color/opacity, metallic-roughness, normal, AO, and emissive texture bindings when present.
   - Remaining polish: generate real atlas textures from source texture/material inputs, reuse xatlas UVs where possible, add USD texture bindings, AO baking, and texture resizing/compression prep.

4. Error-bounded simplification - first reporting pass complete
   - Decimation now records achieved triangle reduction and measured symmetric nearest-vertex error on parts and asset metadata.
   - Explicit decimation now records `decimate_requested_keep_ratio` when derivable and warns when the requested keep ratio is below 20% for close-view LOD0 assets.
   - Explicit decimation now supports UV importance modes for preserving islands, preserving seams only, or ignoring UV/tangent attributes before simplification.
   - Explicit decimation now records estimated RAM, budget-allocation mode, configurable iterative-threshold controls, and actual simplification pass counts.
   - Explicit decimation now supports pre-cleanup for unused UV channels and tangents, reports removed/preserved attributes, and warns when preserved UV seams or islands can reduce simplification efficiency.
   - Explicit decimation now reports protected hard-edge, hole-boundary, material-boundary, UV-seam, silhouette, and total feature-face counts.
   - `criterion="quality"` now reports measured error, but still maps tolerances to a target ratio.
   - Remaining polish: enforce geometric error bounds, preserve selected CAD features, add singularity-specific protection metrics, add vertex-color/weight cleanup, and add AO/user-weight constraints for very large meshes.

5. BREP healing depth - first topology-risk reporting pass complete
   - BREP status now records wire, edge, free/unstitched-edge, small-edge, open-shell, and sliver-face counts.
   - `heal_brep` now stores those counts in per-part metadata and warns when open shells, free edges, or small edges remain after healing.
   - Unsupported sliver-face removal still reports a visible warning instead of claiming geometry was changed.
   - Mesh repair now handles duplicate polygon cleanup after tessellation.
   - Mesh repair now reports non-orientable shared-edge cycles before winding normalization.
   - Mesh repair now reports before/after T-junction counts and warns when T-junctions remain.
   - Mesh repair now reports nearby boundary gaps and warns when they remain unstitched.
   - Mesh repair now reports flipped closed-component counts before and after orientation repair.
   - Remaining polish: implement or delegate sliver-face removal, BREP duplicate-face cleanup, and deeper face/wire repair before tessellation.

6. PMI and metadata output - first stability pass complete
   - STEP AP242 fixtures now test that advertised PMI is reported as present and unsupported when typed PMI import is unavailable.
   - glTF and USD exporters keep PMI as metadata records for now and resolve links through `source_part_id` / `source_part_ids` after merge, explode, or replace operations.
   - Remaining polish: implement typed AP242 PMI entity extraction and visual annotation geometry for `metadata_and_visuals`.

7. Large assembly scaling - first scoped-copy pass complete
   - Filtered operations now skip occurrence-isolation asset copies when the selection already maps cleanly to whole unique parts.
   - Shared repeated parts are still copied only when a selected occurrence must be isolated from unmatched occurrences.
   - Remaining polish: reduce the operation-level full copy for selected part edits, add memory/time benchmarks, and consider streaming or lazy mesh payloads for heavy STEP imports.

8. Runtime compression - unsupported texture compression now rejected
   - Draco remains rejected until a reliable encoder backend is integrated.
   - KTX2/Basis texture compression is now rejected instead of recorded as metadata-only intent.
   - Remaining polish: add real KTX2/Basis output only after texture assets are real files in the export graph, and add a Draco path only if a reliable Python encoder exists.

9. UV pipeline depth - validation status pass complete
   - Stage now records per-channel UV domain, bounds, unit-domain status, validation status, degenerate UV face counts, and overlap-pair counts on mesh metadata.
   - Stage now records UV island counts, pack efficiency, normalized-space utilization, conformal angle distortion, and isometric edge-length distortion.
   - UV1 or `lightmap` channels warn on bake-domain violations, while UV0 overlaps remain metadata-only for tileable texture workflows.
   - Bake-domain `unwrap` and `lightmap` channels now record `missing_repack` status and warn that no separate repack/padding backend ran.
   - Unwrap method, iteration, and tolerance controls are now represented as solver intent; non-default values warn when xatlas cannot enforce them directly.
   - Tangent lifecycle validation now reports generated, regenerated, preserved, invalidated, missing-UV0, and dropped tangent states, with explicit override support for forced regeneration.
   - UV0-to-UV1 copy now records source-channel and missing-source metadata, and emits a warning when the source channel is unavailable.
   - UV normalization now rescales selected channels into 0..1 and records original bounds plus missing-channel warnings.
   - Remaining polish: add seam segmentation, backend-enforced solver/policy controls, island merging, and real packing.

10. Instance reconstruction - exact mesh pass complete
   - `optimize_scene(instance_policy="auto"|"preserve")` now reconstructs shared instances for separately modeled parts with matching mesh fingerprints, vertex attributes, material assignments, and metadata.
   - Scene metadata records reconstructed part/occurrence counts plus duplicate vertex/triangle payload savings.
   - Vertex attribute, material, or metadata differences now emit warnings when they prevent full reconstruction.
   - Remaining polish: add tolerance-based similarity detection, transform-aware matching, and richer file-size savings estimates.

## Correct Deferrals

These are intentionally outside the immediate plan unless a user need changes the priority:

- Full Unity-level CAD format coverage such as IGES, Parasolid, JT, CATIA, NX, and native SolidWorks.
- Convex decomposition and physics proxy generation.
- Advanced retopology and subdivision workflows.
- GPU-specific runtime packaging beyond standards-aligned glTF/USD output.
- Animation and time-varying CAD data, including skeletal animation, morph targets, and animated GLB passthrough validation.

## Operating Checklist

For each planned feature:

1. Confirm the intended behavior in docs or tests first.
2. Keep the change scoped to one user-visible outcome.
3. Add or update focused tests.
4. Update API/reference docs when public behavior changes.
5. Run `make fmt-check`, `make lint`, `make docs`, and `make ci`.
6. Commit with the repo convention.
7. Push and verify GitHub CI and Docs workflows are green.
