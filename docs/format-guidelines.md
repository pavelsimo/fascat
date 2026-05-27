---
title: Format Guidelines
description: Recommended source and delivery formats for CAD, visualization, manufacturing, BIM, and inspection workflows
---

Fascat's default data guideline is simple: keep exact engineering data separate
from runtime visualization data.

For mechanical CAD, do not replace JT with another visualization container as
the source of truth. Use STEP AP242 as the primary neutral CAD exchange format,
then add a second format for the downstream job.

```text
Exact CAD / engineering exchange                   -> STEP AP242
Fast visualization / web / preview                 -> glTF / GLB
Large scene / digital twin / Omniverse workflows   -> USD
3D printing / additive manufacturing               -> 3MF
Buildings / BIM                                    -> IFC
Inspection / metrology / quality                   -> STEP AP242 + QIF
Simple mesh only                                   -> STL / OBJ as last-mile meshes
```

Fascat currently supports STEP input and USD/glTF output. The other formats in
this guide are recommendations for complete data handoff workflows, not claims
that Fascat implements every path today.

## Default Recommendation

Use this as the baseline contract for general mechanical CAD pipelines:

```text
Primary engineering exchange:
    STEP AP242

Visualization / web preview:
    GLB or glTF

Large scene composition:
    USD

Manufacturing / 3D printing:
    3MF

Inspection / metrology:
    QIF, usually alongside STEP AP242

BIM / buildings:
    IFC

Do not rely on:
    JT as the only source
    STL as the source of truth
    OBJ as the source of truth
    IGES unless legacy compatibility requires it
```

## STEP AP242

Use STEP AP242 as the primary format when you care about real mechanical CAD
data.

STEP AP242 is the best default replacement for JT when the source data needs to
remain engineering-grade. ISO 10303-242:2025 covers managed model-based 3D
engineering, including mechanical products, assemblies, geometry models,
tessellated geometry, dimensional and geometrical tolerance data, annotated 3D
models, assembly mating information, kinematics, composites, harnesses,
additive manufacturing data, and requirements-management data. See
[ISO 10303-242:2025](https://www.iso.org/standard/84300.html).

Use STEP AP242 when you need:

```text
exact solid/surface geometry
assemblies
part names
colors
layers
metadata
PMI / GD&T, depending on exporter and importer support
long-term archival
interoperability across CAD systems
open-source reader support
```

The practical open-source advantage is important. Open Cascade lists STEP
AP203, AP214, and AP242 among its standard data-exchange interfaces, and its XDE
layer translates extra attributes such as colors, layers, names, and materials.
See [Open Cascade Data Exchange](https://dev.opencascade.org/about/data_exchange).

Ask suppliers for:

```text
STEP AP242
precise B-Rep geometry
assembly structure
colors, names, layers, and metadata where available
PMI/GD&T when the model is MBD-based
validation properties when their exporter supports them
```

Do not just ask for "a STEP file." Ask for STEP AP242 specifically.

## glTF / GLB

Use glTF or GLB when you want a lightweight, easy-to-render visual asset.

Khronos describes glTF as an API-neutral runtime delivery format for efficient,
extensible, interoperable transmission and loading of 3D content. See the
[glTF 2.0 specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html).

Use glTF/GLB for:

```text
web viewers
Three.js / Babylon.js apps
quick previews
AR/VR-style visualization
product configurators
rendering pipelines
sharing a lightweight visual model
```

Do not treat glTF as a CAD exchange replacement. It usually stores meshes,
materials, transforms, and scene hierarchy. It does not preserve exact CAD
surfaces the way STEP does.

```text
STEP AP242 -> engineering source of truth
GLB/glTF   -> visual preview/export
```

## USD

Use USD when the problem is large-scale scene composition rather than exact CAD
exchange.

OpenUSD is a scalable hierarchical scene-description system with references,
variants, composition operators, C++ libraries, and Python bindings. See the
[OpenUSD FAQ](https://openusd.org/release/usdfaq.html).

Use USD for:

```text
large assemblies as scene graphs
simulation and visualization pipelines
robotics and digital twin workflows
Omniverse-style pipelines
asset variants
layered composition
large-scale rendering
```

USD is not primarily a mechanical CAD B-Rep standard. It is excellent for scene
structure, references, variants, animation, and visualization. It is not the
first choice when the main requirement is editable CAD solids.

```text
STEP AP242 -> exact CAD
USD        -> large visualization / simulation scene
```

## 3MF

Use 3MF instead of STL when the target is additive manufacturing.

3MF is a 3D printing format for moving full-fidelity 3D models between design
applications, platforms, services, and printers. The 3MF Consortium also lists
the specification as recognized as an international standard. See the
[3MF specification](https://3mf.io/spec/).

Use 3MF for:

```text
3D printing
slicer workflows
materials
colors
print metadata
additive manufacturing handoff
```

The open-source ecosystem is stronger than JT for this use case. `lib3mf`
provides reading, writing, conversion, and validation support across Windows,
Linux, and macOS, with APIs for several languages. See
[lib3mf](https://github.com/3MFConsortium/lib3mf/).

3MF is still mainly an additive-manufacturing package. It is not the best choice
for preserving exact parametric CAD solids.

```text
STEP AP242 -> design geometry
3MF        -> print/manufacturing package
```

## IFC

Use IFC for BIM, not STEP or JT.

IFC is the vendor-neutral building and infrastructure data standard.
buildingSMART lists IFC 4.3.2.0 as the latest official version and says it is
published by ISO as ISO 16739-1:2024. See
[buildingSMART IFC](https://www.buildingsmart.org/standards/bsi-standards/industry-foundation-classes/).

Use IFC for:

```text
buildings
architecture
construction
infrastructure
BIM coordination
facilities data
walls, doors, spaces, systems, quantities
```

The open-source ecosystem is strong here. IfcOpenShell reads, writes, and
modifies BIM models using IFC, supports C++ and Python, and supports multiple
IFC schemas and serializations. See [IfcOpenShell](https://ifcopenshell.org/).

IFC is not the best choice for a gearbox, bracket, casting, turbine blade, or
machined mechanical part.

```text
Mechanical CAD  -> STEP AP242
BIM / buildings -> IFC
```

## QIF

Use QIF when the goal is measurement, inspection, and quality data.

QIF is an XML-based, CAD-agnostic quality-information standard that associates
quality information such as measurement plans, measurement results, part
geometry, and product manufacturing information. See the
[QIF overview](https://qifstandards.org/overview/).

Use QIF for:

```text
inspection planning
CMM workflows
metrology
quality reporting
PMI-heavy workflows
measurement results
digital thread / traceability
```

Do not use QIF alone as the main CAD geometry exchange. Use it as a companion to
STEP AP242:

```text
STEP AP242 -> geometry + assembly + PMI source
QIF        -> inspection and measurement data
```

## Formats To Avoid As The Source

### STL

STL is acceptable for simple 3D printing, quick mesh export, and slicer input.
It is a poor CAD data source because it usually loses:

```text
exact surfaces
assemblies
part names
metadata
PMI
materials
feature information
```

Use STL only as a final mesh handoff, not as the engineering source.

### OBJ

OBJ is acceptable for some visual mesh workflows, but it is not serious CAD
interchange. Use glTF/GLB instead for most modern visualization workflows.

### IGES

IGES is useful as a legacy fallback, especially for surfaces. Open Cascade lists
IGES support up to 5.3. Prefer STEP when possible:

```text
STEP AP242 > STEP AP214/AP203 > IGES
```

### Parasolid X_T / X_B

Parasolid files can be excellent if everyone in the workflow uses
Parasolid-based tools, but they are not as open-source-friendly as STEP. Use
Parasolid when CAD vendors specifically require it, not as the default neutral
open format.

## Decision Table

| Need | Best format | Why |
|------|-------------|-----|
| Exact mechanical CAD geometry | STEP AP242 | Best neutral open-source-friendly CAD exchange choice |
| Assemblies and part structure | STEP AP242 | Better open-source support than JT |
| PMI / GD&T | STEP AP242, sometimes plus QIF | AP242 for model-based definition, QIF for inspection/metrology |
| Web viewer / browser preview | glTF / GLB | Fast, widely supported visualization format |
| Massive scene composition | USD | Strong scene graph, references, variants, Python bindings |
| 3D printing | 3MF | Better modern additive-manufacturing package than STL |
| Buildings / BIM | IFC | Domain-specific standard for construction and BIM |
| Legacy surface exchange | IGES | Use only when STEP is not available |
| Simple mesh handoff | STL / OBJ | Only for final mesh output, not source CAD |

## Contract Language

When asking a supplier or customer for files, use language like this:

```text
Please provide:
1. STEP AP242 for engineering exchange, with exact B-Rep geometry.
2. Preserve assembly structure, part names, colors, and metadata where available.
3. Include PMI/GD&T in the STEP AP242 export if the model is MBD-based.
4. Provide a GLB/glTF file for lightweight visualization.
5. Provide 3MF only if the deliverable is for additive manufacturing.
6. Provide QIF if inspection/metrology data is part of the handoff.
```

This avoids the JT trap: a file can be good for visualization while still being
poor as the engineering source of truth. STEP AP242 plus glTF/GLB separates the
engineering model from the visualization asset.
