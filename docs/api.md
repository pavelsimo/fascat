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
        max_edge_length=None,
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
        normals=True,
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

## One-shot conversion

Use `fc.convert()` when you want the full default pipeline and output validation in one call.

```python
import fascat as fc

asset = fc.convert(
    "motor.step",
    "motor.usdc",
    profile="realtime-desktop",
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
