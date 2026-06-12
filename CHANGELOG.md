# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- add stdout format selection for streaming USD, glTF, GLB, OBJ, STL, and FBX conversions

### Changed
- require preinstalled glTF compression tooling instead of installing Node packages during export
- document glTF compression tooling requirements and prefer pipx for installation guidance

### Removed
- remove unsupported Homebrew formula, install instructions, and release automation

### Fixed
- validate final written USD, glTF, OBJ, and STL artifacts after export
- validate glTF files that use external binary buffers or Draco-compressed accessors
- avoid locked temporary files during stdin import and stdout conversion workflows

## [0.3.0] - 2026-06-06

### Added
- add IGES and native BREP import support for CAD conversion workflows
- add benchmark tooling for measuring CAD import and conversion performance
- add configurable per-part worker jobs for repair, staging, optimization, decimation, and LOD generation
- add FBX export support for realtime asset workflows
- add texture baking, compression, and source-texture import support
- add vendor material-library sidecar and zipped material-container imports
- add browser, Unity, and Unreal runtime validation previews with packaged engine harnesses
- add visual baseline diff checks and runtime parity fixtures for exported assets
- add multi-root STEP import for assemblies spread across multiple input files
- add STEP external reference graph resolution and preservation of external occurrences
- add glTF export presets and asset-size ladder guidance
- add detail-adaptive tessellation for curved BREP geometry
- add AP242 PMI extraction, visualization, semantic graph reporting, tolerance zones, and annotation presentation support
- add STEP design variant, effectivity, and conditional feature evaluation
- add construction-curve import controls and mixed construction-curve splitting
- add BREP same-domain cleanup, overlap cleanup, open-shell grouping, and curved-BREP tessellation handling
- add engine LOD profile export and LOD runtime parity fixtures

### Changed
- improve conversion performance across mesh repair, staging, occlusion, decimation, tessellation, LOD generation, and USD/glTF/OBJ/STL export paths
- reduce repeated mesh topology, validation, fingerprint, and export-buffer work across pipeline stages
- accelerate occlusion removal with triangle and occurrence BVH culling
- preserve decimation importance faces during optimization
- normalize exported material opacity for more consistent runtime previews

### Fixed
- bound memory usage during nearest-centroid material assignment on large meshes
- delete stale BREP patches when reusing existing meshes

## [0.2.0] - 2026-05-28

### Added
- add import controls for product metadata, PMI, design variants, existing-mesh preference, multi-file intent, and construction-only cleanup of free vertices and lines, with import-decision and per-part loaded-representation reports
- add source unit, up-axis, and handedness normalization that applies a root transform and records it in import metadata
- add tessellation controls for sag-ratio, per-part size-adaptive settings, existing-mesh reuse, and max-polygon-length, with unit-aware tolerance reporting, free-edge diagnostics, attribute provenance, and quality advisories
- add standalone merge-vertices and delete-degenerate-polygons repair operations with duplicate-polygon deletion and before/after counts
- add mesh repair diagnostics for T-junctions, boundary gaps, flipped components, non-manifold edges, and non-orientable strips, with explicit face and normal orientation policies and unit-aware tolerance reporting
- add UV staging controls for AABB projection, UV0-to-UV1 copy, normalization, sharp-edge seams, and forbidden overlap, with per-channel validation, island, distortion, and packing diagnostics, tangent lifecycle validation, and angle-versus-area normal weighting
- add optimization for sampled occlusion removal, exact and tolerance-based instance reconstruction, scene merge and split, draw-call breakdown reports, pre-decimation cleanup, and decimation memory, target-strategy, UV-importance, global-allocation, and protection reporting
- add LOD per-level mesh-payload and policy reports, chain advisories, node-level MSFT_lod references, and skipped-part reporting
- add export support for glTF quantization and meshopt compression, file-size budgets with payload estimates, unused-material pruning, embedded-texture dedupe, alpha-aware texture-export policy, runtime compatibility and decision matrices, USD baked-texture shader bindings, OBJ normals and smoothing, and rejection of unsupported Draco and KTX2 requests
- add desktop, web, mobile, VR, AR, and MR platform budgets with Unity reference ranges, custom target-device profile overlays, and named workflow recipes
- add conversion reporting with dry-run operation classification, pipeline ordering advisories, workflow-summary, preflight, and conversion-manifest steps, pipeline TOML validation, and analysis reports for topology, slivers, tiny parts, draw calls, and self-intersections
- add expanded API parameter docs, a Unity capability matrix, and a supported-format parity matrix
- show live per-stage conversion progress with an animated spinner and elapsed timing on interactive terminals, plain step lines on non-interactive streams, and a one-line completion summary

### Changed
- **breaking:** rename the `Tessellation` options dataclass to `TessellationOptions` for consistency with the other `*Options` classes; update callers to `fc.TessellationOptions(...)` (low impact pre-1.0 with no published consumers)

## [0.1.0] - 2026-05-26

### Added
- add CAD STEP inspection and conversion CLI
- add the STEP-to-realtime pipeline with tessellation, repair, staging, optimization, and LOD generation
- add OpenUSD export with hierarchy, materials, instancing, LOD variants, and validation
- add glTF and GLB export with materials, UVs, transforms, and validation
- add conversion profiles for inspection, desktop, web, and virtual reality
- add JSON reports, sidecar reports, dry runs, stdin/stdout support, and validation commands

### Fixed
- preserve CAD transforms, metadata, face materials, repeated parts, and material bindings through conversion
- improve CLI behavior for help, color handling, quiet mode, backend failures, and validation errors
- keep asset, mesh, material, report, and node models isolated from caller-owned mutable inputs

[Unreleased]: https://github.com/pavelsimo/fascat/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/pavelsimo/fascat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pavelsimo/fascat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pavelsimo/fascat/releases/tag/v0.1.0
