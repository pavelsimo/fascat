# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
