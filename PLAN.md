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
- Platform budgets now record Unity reference triangle and draw-call ranges in
  profile definitions, conversion reports, and documentation tables.
- Decimation now records RAM estimates, budget-allocation mode, and
  iterative-threshold recommendations in metadata and report fields.
- Staging now warns when bake-domain UVs are only unwrapped without a separate
  repack/padding pass and records that missing repack status in metadata.
- Mesh repair now detects non-orientable shared-edge cycles before face
  orientation and warns when Mobius-like topology cannot be fixed by winding.

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
preparation tools, normal/orientation controls, material/AO baking, real
visibility processing, richer decimation controls, staged LOD workflows, and
final export compression.

Comparison snapshot:

| Area | Fascat today | Missing for closer Unity parity |
| --- | --- | --- |
| Import | STEP-centric import with hierarchy, transforms, metadata, colors, repeated-part handling, PMI presence reporting, existing-mesh reuse intent, construction-only point/line cleanup controls, source-space normalization reporting, and BREP patch cleanup reporting after tessellation. | True multi-file/multi-root import semantics, design-variant import, typed/visual PMI, mixed BREP construction-curve cleanup, native CAD/JT/IFC/Parasolid/IGES coverage, and richer per-part loaded-representation reports. |
| Repair and tessellation | BREP sewing/fix-edge path, mesh duplicate/degenerate cleanup, unit-aware repair tolerance reporting, sag/sag-ratio/angle/max-length controls, free-edge diagnostics, reusable existing mesh control, and retained patch / submesh risk warnings. | Open-shell grouping, unstitched-face handling, T-junction sewing, non-manifold edge cracking, selectable face/normal orientation strategies, CAD-derived UV modes, targeted tessellation by part/material/metadata/curvature, and optional free-edge geometry output. |
| Staging | Normal/tangent generation, box/unwrap/lightmap UV modes, UV copy/normalization, UV validation, material normalization, duplicate-material merge, and metadata-only atlas intent. | Unity-style UV0 tileable versus UV1 bake workflows with segmentation, lines of interest, island merge/alignment, repack/padding/share-map controls, distortion and pack-efficiency metrics, material-library mapping, real atlas textures, AO/lightmap baking, and texture cleanup. |
| Optimization | Mesh simplification, measured error reporting, sampled occlusion removal, exact instance reconstruction, scene merge/split utilities, draw-call estimates, and UV-importance modes. | Global assembly target allocation with iterative memory thresholds, real geometric-error bounded simplification, AO/user-weighted decimation, standard/advanced occlusion backends, retopology/proxy mesh generation, duplicate image/material cleanup, and merge reports that quantify culling, instancing, memory, and file-size tradeoffs. |
| LODs | LOD ratios, screen-coverage metadata, validation, skipped-part reporting, and glTF `MSFT_lod` metadata. | Occurrence-level LOD group authoring with preserved instance relationships, optimized LOD0 as master asset, far-LOD one-mesh/one-material baking, switching-distance validation, and engine-specific runtime export profiles. |
| Export | USD/USDZ, glTF/GLB, OBJ, STL, glTF quantization, meshopt, extension reporting, file-size budgets, and rejection of unsupported Draco/KTX2 requests. | Real Draco compression settings, KTX2/Basis texture output, texture resize and PNG/JPEG fallback controls, unused texture cleanup, baseline-versus-optimized size comparisons, Unity/glTFast-oriented profiles, and web/mobile/VR/XR budget presets backed by runtime measurements. |

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
- Use Unity's broad desktop, mobile, VR, and WebGL ranges as report context
  when tuning future target-device presets and measured runtime profiles.
- Add export comparison reports that show unoptimized GLB, optimized GLB,
  geometry-compressed GLB, and geometry-plus-texture-compressed GLB deltas once
  real Draco and KTX2 outputs exist.

Parity gaps to track:

1. Workflow validation
   - Pipeline files now expose Unity-style ordering advisories through `PipelineSpec.advisories()`, dry-run `pipeline_advisories`, and conversion report warnings.
   - The advisor warns when a pipeline decimates before repair, computes tangents before UV0, bakes AO without UV1, or generates LODs before LOD0 optimization. Compression backend requests are still rejected by the CLI/options layer until real encoders exist.
   - Conversion reports now include a `workflow_summary` step that maps Unity-inspired preparation stages to run/skipped status and exact, approximate, or metadata-only levels, including import cleanup, orientation, UV preparation, material baking, LOD generation, export compression, and export.
   - Conversion reports now include a `preflight` step before pipeline or profile operations run, with checklist warnings for missing patch cleanup, face/normal orientation, UV-before-tangent ordering, AO bake UV1 prerequisites, LOD generation without LOD0 optimization, and glTF texture/compression backend gaps.

2. Import controls
   - Reference docs now include a supported-format parity matrix. Unity's baseline covers many CAD and mesh formats; Fascat currently centers on STEP input and USD/glTF/OBJ/STL output, with IGES, Parasolid, JT, native CAD, IFC, 3MF, and QIF explicitly deferred.
   - Explicit import toggles now cover product metadata, properties, layers, validation properties, PMI, design variants, existing mesh preference, and multi-file import intent across Python, CLI, and TOML. Unsupported design-variant and multi-file import requests report warnings instead of silently claiming support.
   - Define true multi-file import semantics: multiple input paths should produce deterministic multi-root assemblies, shared material/image namespaces, stable source-file metadata, and warnings for failed members instead of all-or-nothing failure.
   - Import cleanup now exposes `delete_free_vertices` and `delete_lines` for construction-only point and line shapes across Python, CLI, and TOML. Import reports include cleanup counts, and preserved parts record loaded representation plus source topology counts.
   - Tessellated parts now record `brep_patch_cleanup=deleted` or `retained` and `source_shape_retained`, matching `keep_brep` behavior.
   - Remaining work: decide whether mixed BREP construction curves should be deleted, preserved as metadata, or tessellated into renderable tubes.
   - Source unit, source up-axis, source handedness, target unit, target up-axis, and target handedness normalization controls now apply a root transform, update the asset's declared working space, and record the exact transform in import metadata and reports.
   - Report the loaded representation for each part: BREP, existing tessellation, construction points/lines, PMI, variants, product metadata, and the cleanup action applied.
   - Tessellation now reports when retained BREP patches, CAD face groups, or material splits are likely to increase submesh, draw-call, or export-size pressure.

3. CAD and mesh repair depth
   - Add an open-shell repair workflow: detect single open-shell parts, merge or group them before BREP healing, and keep separate warnings for unstitched faces.
   - Improve BREP healing beyond the current sewing/fix-edge path: sliver-face removal, duplicate face handling, tolerance unification, and visible report warnings for unsupported backend work.
   - Mesh repair now deletes duplicate polygons and records before/after duplicate, degenerate, boundary-edge, and non-manifold metrics.
   - Extend mesh repair with true T-junction sewing, non-manifold edge cracking, and configurable face-orientation strategies for closed solids versus open shells.
   - Mesh repair now detects non-orientable strips before face orientation so Mobius-like topology is reported separately from ordinary flipped faces.
   - Add explicit face and normal orientation passes with selectable strategies for exterior solids, single-sided open shells, and preserved two-sided surfaces.
   - Add missing-normal generation controls for sharp-edge angle, area weighting, override behavior, and flipped-component reporting.
   - Add attribute-aware tolerance vertex merging that rebuilds connectivity across hard-edge and non-manifold borders without collapsing intentional material, normal, or UV seams.
   - Add before/after repair metrics for T-junctions, boundary gaps, and flipped components.
   - BREP healing and mesh repair now report unit-aware tolerance policy: effective source/local units, declared target units, meters-per-unit conversions, vertex-merge and degenerate-polygon cleanup status, and missing T-junction/non-manifold backend operations.

4. Tessellation controls
   - Sag-ratio is now a first-class tessellation option across Python, CLI, TOML pipelines, per-part overrides, reports, and OCCT backend parameter mapping. `relative=True` remains for compatibility when `sag_ratio` is unset.
   - Existing imported meshes now have an explicit `reuse_existing_meshes` control across Python, CLI, TOML pipelines, per-part overrides, and reports. The default preserves imported meshes; disabling it retessellates from source BREP where available.
   - Free-edge tessellation diagnostics are now available through `free_edge_report` across Python, CLI, TOML pipelines, per-part overrides, metadata, and report warnings. Remaining work: investigate CAD-parametric UV and tangent generation during tessellation.
   - Add optional free-edge geometry output or retention, separate from diagnostics, for wire overlays, boundary inspection, and import cleanup validation.
   - Add targeted tessellation profiles by part size, material, metadata, curvature, or filter so shiny/high-detail parts can use finer criteria than bulk structural parts.
   - Add bounding-box-derived tessellation profile helpers so sag, sag-ratio, angle, and polygon-length defaults can be selected per part instead of only globally.
   - Expose CAD-derived UV generation modes, including none, intrinsic surface UVs, and conformal/scaled UVs, instead of only post-mesh unwrapping.
   - Max polygon length is now exposed separately from cleanup subdivision. `max_edge_length` still subdivides geometry; `max_polygon_length` drives quality-report `long_edges`, metadata, and warnings for long tessellated edges that may cause lighting artifacts.

5. UV staging
   - Extend existing box UVs into Unity-style AABB projection controls: local versus shared/global AABB, real-world UV scale or `uv3dSize`, destination channel, override policy, and unit reporting.
   - Add UV segmentation and seam planning, including sharp-edge seams and lines of interest.
   - Add a complete UV workflow model: segment, unwrap, optionally merge islands, align tileable UV0 islands, repack UV1, normalize, validate overlaps, and record which steps ran per channel.
   - Staging now warns when bake-domain UVs were only unwrapped and not repacked, because unwrap alone does not prove islands were packed into `[0,1]` with padding for lightmap, AO, or material baking.
   - Add affine UV island merge and alignment controls for tileable UV0 workflows, including allowed transforms, polygon weighting, and rotation-step quantization.
   - Unwrap solver intent is now accepted across Python, CLI, TOML pipelines, metadata, and reports with `default`, `conformal`, and `isometric` values. The current xatlas backend records non-default solver methods as intent and warns that it cannot enforce them directly.
   - Unwrap iteration and tolerance controls are now accepted across Python, CLI, TOML pipelines, metadata, and reports. Remaining work: add a backend that enforces those controls and reports solver failure or excessive distortion.
   - Add unwrap distortion metrics per channel and island, including conformal angle distortion, isometric edge-length distortion, island count, and pack efficiency.
   - UV0-to-UV1 copy is now supported through `uv1="copy_uv0"` / `--uv1 copy-uv0`, with copied/missing-source metadata and warnings. UV normalization is now explicit through `normalize_uvs=(...)` / `--normalize-uvs`, with original-bounds metadata and missing-channel warnings. UV validation now records per-channel domains, bounds, unit-domain status, and validation status. Remaining work: UV island merge, alignment, repack, padding/resolution/share-map controls, overlap removal, uniform versus non-uniform normalization, shared versus per-part UV space, null-island handling, distortion metrics, and packing efficiency reports.
   - Make UV0 tileable and UV1 baking requirements explicit: UV0 may overlap; UV1 must fit in `[0,1]` with padding and no overlaps.
   - Add UV1 bake packing controls for atlas resolution, pixel padding, shared versus per-part UV space, overlap removal, and normalized-space utilization reports.
   - Tangent lifecycle validation now warns when UV0 is missing, invalidates tangents after UV edits, and records generated, regenerated, preserved, invalidated, missing, or dropped tangent states on mesh and asset metadata.
   - Tangent generation can now use an explicit source UV channel and preserves existing tangents by default, with `override_tangents` available across Python, CLI, TOML pipelines, metadata, and reports when regeneration is required.
   - Explicit decimation can now strip UV/tangent attributes through `uv_importance="ignore"` or preserve seams and then drop UVs with `uv_importance="preserve_seams"`.

6. Materials and baking
   - Add material-library import and CAD-material-to-PBR mapping, including CSV or TOML mapping tables.
   - Add material mapping diagnostics that report replacements, missing library materials, source materials with no mapping, and unresolved texture dependencies.
   - Add material-combine planning that can bake many materials into one atlas when draw-call reduction is more important than material editability.
   - Replace constant embedded factor maps with real atlas/raster texture output for base color, opacity, roughness, metallic, normal, AO, and emissive maps.
   - Add high-poly-to-proxy normal map baking for retopology or aggressive far-LOD workflows.
   - Add real ambient occlusion baking to textures and optionally to vertex colors for downstream decimation weights, with explicit resolution, padding, sample-count, target-channel, bent-normal, and denoise/filter controls.
   - Add material and image cleanup: merge duplicate materials/images, remove unused images, resize textures to platform budgets, and keep PNG/JPEG fallbacks when KTX2 is unavailable.

7. Optimization and draw-call reduction
   - Add acceleration structures, confidence metrics, and optional raster/GPU backends to the new sampled occlusion removal.
   - Expose standard versus advanced occlusion-removal parameters such as resolution, sphere count or ray direction set, adjacency depth, hemisphere-only evaluation, cavity preservation, and GPU/backend requirements.
   - Add loose and precise instance reconstruction for similar, separately modeled parts.
   - Improve merge planning so reports show draw-call savings, instance loss, memory growth, culling impact, and export file-size risk when merging destroys repeated geometry.
   - Add draw-call budget analysis that separates mesh count, material count, submesh/material slots, instances, and merged batches.
   - Add retopology or proxy-mesh paths for cases where decimation and occlusion are not enough.
   - Add dedicated cleanup for unused texture coordinates, duplicate materials, and duplicate images before draw-call and file-size optimization.

8. Decimation parity
   - Add global target allocation across a selected assembly while decimating at part level, so sparse parts stay intact and dense parts carry most of the reduction.
   - Decimation now records RAM estimates using the Unity 5 GB per million polygons rule of thumb, reports global versus per-part budget allocation, and warns when the selected source triangle count reaches the iterative threshold.
   - Add target-device decimation presets, including XR/HoloLens-style triangle caps, so platform targets can drive simplification before export.
   - Replace quality-criterion heuristics with measured geometric error.
   - Explicit decimation now supports UV importance modes: preserve full UV islands, preserve seam topology only, or ignore UVs by stripping UV/tangent attributes before simplification.
   - Add a pre-decimation cleanup path for unused texture coordinates and report when preserved UVs make simplification less efficient.
   - Make topology protection explicit and measured, especially for holes, boundary loops, singularities, and material/UV seam preservation.
   - Support AO or user-painted vertex weights as simplification constraints.
   - Explicit decimation now records the requested keep ratio when derivable and warns when the request keeps less than 20% of source triangles for close-view LOD0 assets.
   - Keep skinning, bones, and animation preservation out of scope until Fascat supports animated mesh imports.

9. LOD parity
   - Represent LOD0 as the optimized master asset and LOD1+ as occurrence-level LOD groups with stable naming, parentage, and instance references.
   - Preserve occurrence-level LOD chains and instance relationships across all LOD levels.
   - Add far-LOD generation that can merge to one mesh and one baked material for one-draw-call distant rendering.
   - Add LOD validation for screen coverage, monotonic triangle reduction, material simplification, switching distances, and export runtime behavior.
   - Add LOD generation reports that show whether each level reused instances, merged materials, baked textures, or changed culling granularity.
   - Non-mesh or untessellated selections are now skipped with part metadata, generated/skipped counts, and report warnings instead of quietly producing partial chains.
   - Add engine-specific LOD export metadata or profiles for Unity, Unreal, and standards-based glTF runtimes, including switching-distance validation.

10. Export parity
   - Add a real Draco encoder path with compression level and quantization settings, or keep `draco=True` rejected.
   - Add real KTX2/Basis texture output with quality, compression level, and max-resolution controls.
   - Add export cleanup for unused images/materials and file-size reports broken down by geometry, textures, and metadata.
   - Add texture-resize preprocessing with before/after dimensions, byte estimates, and per-profile maximums before KTX2/PNG/JPEG export decisions.
   - glTF write reports now list emitted runtime extensions, required extensions, `extras.fascat` metadata, unsupported Draco/KTX2 outputs, and expected runtime support.
   - Add Unity/glTFast-oriented GLB export profiles that combine extension support notes, Draco/KTX2 settings, fallback choices, and runtime compatibility warnings.
   - Add baseline-versus-optimized export comparisons so reports show how much each preparation step changed file size, and warn when draw-call merging increases export size by breaking instancing.
   - Add format-aware texture export policy and reporting: prefer KTX2/Basis for glTF/GLB, use PNG/JPEG fallbacks for texture-capable non-glTF exports, remove unused images before export, and warn when users compare source CAD file size directly against runtime mesh exports.
   - Add named web, mobile, desktop, and VR export presets that combine geometry compression, texture compression, texture resizing, and cleanup choices.
   - Keep GLB as the preferred web/mobile runtime target while preserving USD/USDZ for OpenUSD workflows.
   - Expose Draco quantization bits for positions, normals, UVs, and vertex colors once a real encoder is available.
   - Expose PNG/JPEG fallback texture export settings, including PNG compression and JPEG quality, for formats or environments where KTX2 is not available.

11. Platform budgets
   - Desktop, WebGL/web, mobile, and VR profiles now include documented target-FPS, triangle, vertex, per-mesh vertex/index-buffer, texture-resolution, texture-memory, estimated load-time, and draw-call budgets.
   - Conversion reports now include a `profile_budget` step for selected-profile budget status and warnings when output exceeds profile triangle, vertex, per-mesh vertex/index-buffer, texture-resolution, texture-memory, estimated load-time, or draw-call budgets.
   - Profile budgets now include explicit Unity reference ranges for each broad profile so users can see how Fascat's stricter defaults compare with Unity's desktop, mobile, VR, and WebGL guideline ranges.
   - Add XR/AR device profiles and custom target-device overrides so budgets can model device-specific caps instead of only broad desktop/web/mobile/VR classes.
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
   - Explicit decimation now records estimated RAM, budget-allocation mode, and iterative-threshold recommendations.
   - `criterion="quality"` now reports measured error, but still maps tolerances to a target ratio.
   - Remaining polish: enforce geometric error bounds, preserve selected CAD features, add richer topology protection metrics, configurable iterative processing, and AO/user-weight constraints for very large meshes.

5. BREP healing depth - first topology-risk reporting pass complete
   - BREP status now records wire, edge, free/unstitched-edge, small-edge, open-shell, and sliver-face counts.
   - `heal_brep` now stores those counts in per-part metadata and warns when open shells, free edges, or small edges remain after healing.
   - Unsupported sliver-face removal still reports a visible warning instead of claiming geometry was changed.
   - Mesh repair now handles duplicate polygon cleanup after tessellation.
   - Mesh repair now reports non-orientable shared-edge cycles before winding normalization.
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
   - UV1 or `lightmap` channels warn on bake-domain violations, while UV0 overlaps remain metadata-only for tileable texture workflows.
   - Bake-domain `unwrap` and `lightmap` channels now record `missing_repack` status and warn that no separate repack/padding backend ran.
   - Unwrap method, iteration, and tolerance controls are now represented as solver intent; non-default values warn when xatlas cannot enforce them directly.
   - Tangent lifecycle validation now reports generated, regenerated, preserved, invalidated, missing-UV0, and dropped tangent states, with explicit override support for forced regeneration.
   - UV0-to-UV1 copy now records source-channel and missing-source metadata, and emits a warning when the source channel is unavailable.
   - UV normalization now rescales selected channels into 0..1 and records original bounds plus missing-channel warnings.
   - Remaining polish: add seam segmentation, backend-enforced solver controls, island merging, packing, distortion metrics, and packing efficiency reports.

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
