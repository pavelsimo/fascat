# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/pavelsimo/fascat/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pavelsimo/fascat/releases/tag/v0.1.0
