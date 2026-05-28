---
title: Format Guidelines
description: Recommended source and delivery formats for CAD, visualization, manufacturing, BIM, and inspection workflows
---

The core rule: keep exact engineering data separate from runtime visualization data.
Use STEP AP242 as the single source of truth, then add one delivery format for the
downstream job.

Fascat currently reads STEP and writes USD, glTF/GLB, OBJ, and STL. The other formats
below are workflow recommendations, not paths Fascat implements today.

## Quick Reference

| Need | Use | Why |
|------|-----|-----|
| Exact mechanical CAD geometry, assemblies, PMI | STEP AP242 | Neutral, open-source-friendly engineering exchange |
| Web / realtime / preview visualization | glTF / GLB | Lightweight, widely supported runtime format |
| Large scene composition, digital twins | USD | Scene graph with references, variants, and layering |
| 3D printing / additive manufacturing | 3MF | Carries materials, colors, and print metadata |
| Buildings / BIM | IFC | Vendor-neutral construction and facilities standard |
| Inspection / metrology / quality | STEP AP242 + QIF | AP242 for geometry, QIF for measurement data |
| Final mesh handoff only | STL / OBJ | Last-mile meshes, never the source of truth |

## When To Use Each Format

### STEP AP242 — engineering source of truth
Use when you need exact B-Rep solids/surfaces, assembly structure, part names, colors,
layers, metadata, PMI/GD&T, or long-term archival. Ask suppliers for "STEP AP242"
specifically, not just "a STEP file."

### glTF / GLB — visualization
Use for web viewers, Three.js/Babylon.js apps, AR/VR previews, product configurators,
and sharing a lightweight visual model. It stores meshes, materials, transforms, and
hierarchy — not exact CAD surfaces. Pair it with STEP; don't use it as the CAD source.

### USD — large scenes
Use for big assemblies as scene graphs, simulation and visualization pipelines, digital
twins, Omniverse-style workflows, asset variants, and layered composition. It is not a
mechanical B-Rep exchange format.

### 3MF — additive manufacturing
Use instead of STL for 3D printing and slicer workflows when you need materials, colors,
and print metadata. It is not for preserving parametric CAD solids.

### IFC — BIM / buildings
Use for architecture, construction, infrastructure, and BIM coordination — walls, doors,
spaces, systems, and quantities. It is not for machined mechanical parts.

### QIF — inspection / metrology
Use for inspection planning, CMM workflows, metrology, and quality reporting. Pair it
with STEP AP242 as the geometry source rather than using it alone.
