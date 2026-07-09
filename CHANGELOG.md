# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- import JT (ISO 14306) files: a pure-Python, clean-room JT 8.x/9.x/10.x reader (`fc.read_jt`, `JtReadOptions`) covering pre-tessellated LOD meshes (plain tri-strips in JT 8, topologically compressed tri-strip sets in JT 9/10), assembly hierarchy, instances, transforms, materials, and properties, plus a `--jt-lod-selection finest|all` convert flag; JT 7 and older fail with a clear error
- render multi-angle turntable previews from `fascat validate` (`--turntable-dir`, `--turntable-views`, `--turntable-elevations`, `--turntable-width/height/supersample`) with per-view baseline diffing via `--turntable-baseline-dir`; Python API `validation.write_turntable_previews` / `write_output_turntable_previews`
- add a `skills/cad-to-rt3d` Claude Code skill that automates the CAD-to-RT3D convert/validate loop with deterministic gate checking (`skills/cad-to-rt3d/scripts/gates.py`) and turntable-based visual comparison against a high-quality reference conversion

## [0.4.0] - 2026-06-13

### Added
- ship the PEP 561 `py.typed` marker so downstream type checkers consume fascat's strict-mode hints
- print conversion report warnings (including platform budget violations) on stderr, capped at 10 lines with an overflow pointer to `--report`
- expose KTX2/Basis encoder quality (0-255), compression effort (0-6), and a UASTC/ETC1S override on `GltfExportOptions` plus `--ktx2-*` convert flags
- expose Draco compression level and per-attribute quantization bits on `GltfExportOptions` and matching `--draco-*` convert flags (defaults reproduce previous output)
- add concise reprs for Asset, Node, Part, Mesh, and options (options show non-default fields only)
- accept keyword arguments directly on every Asset operation (`asset.tessellate(sag=0.1)`); `options=` stays as the prebuilt escape hatch, and `asset.lods` accepts a bare ratio sequence
- add stdout format selection for streaming USD, glTF, GLB, OBJ, STL, and FBX conversions

### Changed
- expand CI to a lint job plus a test matrix: ubuntu across Python 3.10-3.13 with windows and macos spot cells
- move `alktx2` out of core dependencies; install the `ktx2` extra for the Python KTX2 preview decoder
- **breaking:** explicit `forbid_overlapping=True` now raises `UVOverlapError` when overlapping UV faces remain after staging; bake-domain defaults keep warnings
- run per-part mesh operations in process pools with `jobs` defaulting to `min(4, CPU count)` (override with `FASCAT_JOBS` or `jobs=1`); thread-based execution previously serialized CPU-bound work behind the GIL
- load the public api lazily: `import fascat` drops from ~190ms to ~30ms and no longer pulls numpy, Pillow, or the runtime harness stacks until first use
- require preinstalled glTF compression tooling instead of installing Node packages during export
- document glTF compression tooling requirements and prefer pipx for installation guidance

### Removed
- **breaking:** shrink the top-level namespace to 41 core names; the validation/runtime/visual harness surface moves to `fascat.validation`, remaining options classes stay importable from `fascat.options`, and per-format validators live on `fascat.io.*`
- **breaking:** remove the 18 module-level operation wrappers (`fc.tessellate(asset, ...)` etc.); use the fluent `Asset` methods
- remove unsupported Homebrew formula, install instructions, and release automation

### Security
- cap auxiliary STEP text scans and sidecar library/texture loads with size limits
- bound per-record STEP argument scanning so unterminated strings cannot trigger whole-file rescans
- confine STEP texture and material-library references to the CAD source directory and configured search paths; escaping references are reported as missing

### Fixed
- fill holes with the material of the nearest neighboring face instead of stamping `material_indices[0]`
- preserve normals, tangents, and UV channels when stitching boundary gaps (representative-vertex carry; UV-conflicting merges are counted in metadata)
- invalidate stale per-face materials and face groups when the winding remap cannot match the new faces, instead of leaving silently misaligned assignments
- make all exporters write transactionally via same-directory temp files and atomic renames — failed validation or Ctrl-C never leaves a partial or corrupt output file
- kill the whole subprocess process group when browser/engine/encoder tools time out, and bound gltf-transform and KTX2 encoder invocations with timeouts
- derive the degenerate-face area epsilon from the mesh bounding box by default (scale-invariant; explicit values stay authoritative)
- skip zero-triangle meshes during glTF export instead of emitting invalid empty primitives, and report out-of-bounds material indices as warnings
- exit with code 130 and a clean message on Ctrl-C instead of a traceback
- decode ISO-10303-21 string escape directives in PMI text, design-variant labels, and sidecar references
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

[Unreleased]: https://github.com/pavelsimo/fascat/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/pavelsimo/fascat/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pavelsimo/fascat/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pavelsimo/fascat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pavelsimo/fascat/releases/tag/v0.1.0
