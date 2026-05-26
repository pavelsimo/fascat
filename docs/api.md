---
title: Python API
description: Use fascat from Python
---

The Python API exposes the same pipeline as the CLI, but keeps each conversion step explicit and composable.

## End-to-end pipeline

```python
import fascat as fc

asset = fc.read_step("motor.step")

asset = asset.tessellate(
    fc.Tessellation(
        sag=0.1,
        angle=15.0,
        relative=True,
        min_edge_length=None,
        max_edge_length=None,
        preserve_boundaries=True,
        curvature_adaptive=False,
        avoid_skinny_triangles=False,
        quality_report=False,
    )
)

asset = asset.repair(
    fc.RepairOptions(
        tolerance=0.05,
        merge_vertices=True,
        delete_degenerate=True,
        fix_winding=True,
        fill_small_holes=False,
    )
)

asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        material_mode="cad",
        merge_equivalent_materials=False,
        normals=True,
        normal_mode="smooth",
        hard_edge_angle=30.0,
        preserve_face_boundaries=False,
        tangents=False,
        validate_normals=False,
        unwrap=fc.UnwrapOptions(),
        atlas=fc.AtlasOptions(),
        uv0="box",
        uv1=None,
    )
)

asset = asset.optimize(
    fc.OptimizeOptions(
        target_triangles=500_000,
        preserve_instances=True,
        simplify=True,
        optimize_buffers=True,
        preserve_hard_edges=False,
        preserve_holes=False,
        preserve_material_boundaries=False,
        preserve_uv_seams=False,
        preserve_small_parts=False,
        preserve_silhouette=False,
    )
)

asset = asset.lods(
    fc.LODOptions(
        ratios=[0.5, 0.25, 0.1],
        mode="variants",
    )
)

asset.write_usd("motor.usdc")
asset.write_gltf("motor.glb")
```

Pipeline operations return new `Asset` instances instead of mutating the previous asset. Write calls attach a final write step to the asset report.

## Assembly filters

Use `Filter` selectors to inspect or process one branch of an assembly while leaving the rest unchanged.

```python
import fascat as fc

asset = fc.read_step("motor.step").tessellate()

fasteners = fc.Filter(
    path="*/Fasteners/*",
    name=["Bolt*", "Nut*", "Washer*"],
)

large_castings = fc.Filter.all(
    fc.Filter.path("*/Housing/*"),
    fc.Filter.size(min_diagonal=50.0),
)

print(asset.select(fasteners).stats())

asset = asset.optimize(
    fc.OptimizeOptions(target_triangles=80_000),
    where=fasteners,
)

asset = asset.stage(
    fc.StageOptions(materials="display", uv0="none", uv1=None),
    where=large_castings,
)
```

Filters support node path, node name, part id, part name, material, metadata, bounding box, size, triangle count, vertex count, and logical `all`, `any`, and `not_` composition. If a selected occurrence shares a part with an unmatched occurrence, Fascat duplicates the selected occurrence's part before applying the operation so the unmatched branch stays intact. Report steps include `where` and `matched` fields when an operation is scoped.

## Hierarchy merge

Use `merge()` to reduce node count and draw calls before optimization.

```python
import fascat as fc

asset = fc.read_step("motor.step").tessellate().stage()

asset = asset.merge(
    fc.MergeOptions(
        mode="by_material",
        keep_parent=True,
        metadata="combine",
        max_vertices_per_mesh=65_535,
        preserve_materials=True,
    ),
    where=fc.Filter.path("*/Fasteners/*"),
)
```

Merge modes include `all`, `by_material`, `by_node_name`, `by_part_name`, `hierarchy_level`, `parent_children`, `final_level`, and `regions`. Merging bakes node transforms into merged vertex positions, keeps material slots when requested, removes replaced empty nodes, and records before/after `draw_calls` in the merge report step.

## Metadata and PMI

Fascat keeps top-level asset metadata and typed PMI records alongside node, part, material, and mesh metadata.

```python
import fascat as fc

asset = fc.read_step(
    "motor.step",
    options=fc.StepReadOptions(
        metadata=True,
        product_metadata=True,
        properties=True,
        layers=True,
        validation_properties=True,
        pmi=True,
    ),
)

asset.metadata["review_state"] = "approved"
asset.pmi.append(
    fc.PmiAnnotation(
        id="pmi_001",
        kind="dimension",
        text="25.4 +/-0.1",
        value=25.4,
        unit="millimetre",
        tolerance=fc.Tolerance(upper=0.1, lower=0.0),
        applies_to=["part_123"],
    )
)
```

glTF export writes metadata and PMI into `extras.fascat`. USD export writes Fascat metadata into `customData` on the scene, nodes, prototypes, materials, meshes, and `/PMI/*` annotation prims.

## BREP Healing

Run BREP healing before tessellation when STEP topology needs sewing, edge fixing, tolerance unification, or open-shell reporting.

```python
asset = fc.read_step("motor.step").heal_brep(
    fc.BrepHealOptions(
        tolerance=0.05,
        sew_faces=True,
        fix_edges=True,
        remove_sliver_faces=True,
        max_sliver_area=1e-4,
        unify_tolerances=True,
        fail_on_open_shells=False,
    ),
    where=fc.Filter.path("*/Housing/*"),
)
```

The operation stores per-part `brep_*` metadata and records a `heal_brep` report step. `fc.convert(..., heal_brep=fc.BrepHealOptions())` runs healing before tessellation.

## Tessellation Controls

Tessellation supports global and per-part settings for edge limits, boundary preservation, curvature-adaptive OCCT meshing, skinny-triangle cleanup, and per-part quality metrics.

```python
asset = fc.read_step("motor.step").tessellate(
    fc.Tessellation(
        sag=0.05,
        angle=10.0,
        min_edge_length=0.02,
        max_edge_length=2.0,
        preserve_boundaries=True,
        curvature_adaptive=True,
        avoid_skinny_triangles=True,
        quality_report=True,
        part_settings={
            "housing": {"sag": 0.03, "max_edge_length": 1.0},
            "Fastener": {"sag": 0.15},
        },
    )
)

quality = asset.tessellation_quality_report()
```

`part_settings` keys match a part id or part name. Quality reports include per-part edge length, triangle area, aspect ratio, skinny triangle, boundary edge, and non-manifold edge counts.

## Feature-Preserving Simplification

Optimization can protect mechanical features while reducing triangle count. Preservation flags keep protected faces from being dropped when a target would otherwise remove them.

```python
asset = asset.optimize(
    fc.OptimizeOptions(
        target_triangles=500_000,
        simplify=True,
        preserve_instances=True,
        preserve_hard_edges=True,
        hard_edge_angle=30.0,
        preserve_holes=True,
        preserve_material_boundaries=True,
        preserve_uv_seams=True,
        preserve_small_parts=True,
        small_part_triangle_threshold=64,
        preserve_silhouette=True,
    )
)
```

Protected-feature counts are stored as part metadata under `simplification_preserved_features`. Parts below `small_part_triangle_threshold` are left unsimplified when `preserve_small_parts=True`.

## Hard-Edge Normals And Tangents

Staging can generate smooth, flat, or hard-edge normals and glTF-ready tangents. Hard-edge mode splits vertices across hard normal edges, material boundaries, and optional CAD face-group boundaries.

```python
asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        normals=True,
        normal_mode="hard_edges",
        hard_edge_angle=30.0,
        preserve_face_boundaries=True,
        tangents=True,
        validate_normals=True,
        uv0="box",
    )
)
```

Tangents require UV0. glTF export writes a `TANGENT` vertex attribute when staged meshes contain tangent data.

## UV And Material Pipeline

Staging can merge equivalent CAD materials, normalize simple CAD colors into PBR-friendly material values, tag UV unwrap settings, generate lightmap UV channels, and attach material-atlas metadata for later baking.

```python
asset = asset.stage(
    fc.StageOptions(
        materials="cad",
        material_mode="pbr",
        merge_equivalent_materials=True,
        uv0="unwrap",
        uv1="lightmap",
        unwrap=fc.UnwrapOptions(
            texel_density=256.0,
            padding=4,
            max_stretch=0.15,
        ),
        atlas=fc.AtlasOptions(
            enabled=True,
            max_size=4096,
        ),
    )
)
```

Atlas support currently records atlas and texture-bake metadata on materials and meshes. Dedicated material baking is a separate optimization step.

## One-shot conversion

Use `fc.convert()` when you want the full default pipeline and output validation in one call.

```python
import fascat as fc

asset = fc.convert(
    "motor.step",
    "motor.usdc",
    profile="realtime-desktop",
    where=fc.Filter.path("*/Fasteners/*"),
    merge=fc.MergeOptions(mode="by_material", metadata="combine"),
)

print(asset.stats())
print(asset.report.summary())
```

The output format is selected from the output suffix:

```python
fc.convert("motor.step", "motor.usdc")
fc.convert("motor.step", "motor.usda", debug=True)
fc.convert("motor.step", "motor.glb", profile="virtual-reality")
fc.convert("motor.step", "motor.gltf", profile="realtime-web")
```

`fc.convert()` validates generated output by default. Pass `validate_output=False` only when another step in your pipeline validates the asset.
When `where` is provided to `fc.convert()`, tessellation, repair, and staging still run for the full asset, while merge, optimization, and LOD generation are scoped to the matched assembly subset.

## Profiles

Profiles provide practical defaults for tessellation, staging, optimization, and LODs.

```python
profile = fc.profiles.realtime_web(
    tessellation_sag=0.2,
    angle=20.0,
    max_triangles=250_000,
    lod_ratios=(0.5, 0.25),
)

asset = fc.convert("motor.step", "motor.glb", profile=profile)
```

Available profiles:

| Profile | Use |
|---------|-----|
| `inspect-only` | inspect STEP input without conversion |
| `realtime-desktop` | higher-detail OpenUSD or glTF output |
| `realtime-web` | lower triangle budgets for web delivery |
| `virtual-reality` | balanced triangle budgets and LODs for VR runtimes |

You can pass either a profile name or a `ConversionProfile` returned by `fc.profiles`.

## Functional wrappers

The top-level functions mirror the fluent `Asset` methods.

```python
import fascat as fc

asset = fc.read_step("motor.step")
asset = fc.tessellate(asset, sag=0.1, angle=15.0)
asset = fc.repair(asset, tolerance=0.05)
asset = fc.stage(asset, materials="cad", uv0="box")
asset = fc.optimize(asset, target_triangles=500_000)
asset = fc.lods(asset, ratios=(0.5, 0.25, 0.1))

fc.write_usd(asset, "motor.usdc")
fc.write_gltf(asset, "motor.glb")
```

## Reports and stats

Every imported or converted asset carries a report.

```python
asset = fc.convert("motor.step", "motor.usdc")

print(asset.stats(include_lods=True))
print(asset.report.summary())

for step in asset.report.steps:
    print(step.name, step.duration, step.before, step.after)

asset.report.write_json("report.json")
```

The report records options, before/after counts, warnings, errors, and timings for each pipeline step.

## Validation

Direct write calls produce files but do not automatically reopen and validate them. Validate direct writes explicitly when you need the same safety as `fc.convert()`.

```python
asset.write_usd("motor.usdc")
usd_stats = fc.validate_usd("motor.usdc")

asset.write_gltf("motor.glb")
gltf_stats = fc.validate_gltf("motor.glb")

stats = fc.validate_output("motor.glb")
```

## Inspecting assets

Use `to_dict()` for structured inspection or JSON serialization.

```python
asset = fc.read_step("motor.step")

print(asset.part_count)
print(asset.material_count)
print(asset.occurrence_count)

payload = asset.to_dict()
print(payload["root"])
print(payload["parts"])
```

The asset model preserves hierarchy, part records, material records, transforms, units, and source metadata where the STEP backend can read them.

## glTF notes

OpenUSD is the highest-fidelity export path for USD-style LOD variants and instance metadata.

glTF export writes valid glTF 2.0 files for runtime use:

- `.gltf` uses embedded binary buffers
- `.glb` writes a binary glTF container
- geometry is exported in metres and Y-up
- original units and source up-axis are preserved in top-level Fascat extras
- material subsets are exported as separate glTF primitives
- generated LOD meshes are included and referenced from Fascat node extras
