from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import fascat as fc

pytestmark = [pytest.mark.requires_ocp, pytest.mark.requires_usd]


def _write_repeated_box_step(path: Path) -> None:
    from OCP.BRep import BRep_Builder
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Trsf, gp_Vec
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS_Compound
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    def located_shape(x: float, y: float, z: float) -> object:
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(x, y, z))
        return box.Located(TopLoc_Location(transform))

    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("fascat-test"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    box = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
    assembly_shape = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(assembly_shape)
    builder.Add(assembly_shape, located_shape(0.0, 0.0, 0.0))
    builder.Add(assembly_shape, located_shape(10.0, 0.0, 0.0))
    assembly_label = shape_tool.AddShape(assembly_shape, True, True)
    TDataStd_Name.Set_s(assembly_label, TCollection_ExtendedString("Generated Assembly"))

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    writer.SetColorMode(True)
    assert writer.Transfer(document, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone


@pytest.mark.parametrize("fixture", sorted(Path("tests/fixtures").glob("*.step")))
def test_step_fixtures_import_with_names_units_and_parts(fixture: Path) -> None:
    asset = fc.read_step(fixture)
    nodes = asset.root.walk()

    assert asset.part_count >= 1
    assert asset.occurrence_count >= asset.part_count
    assert asset.units
    assert asset.meters_per_unit > 0.0
    assert asset.root.name == fixture.stem
    assert all(node.name for node in nodes)
    assert all("step_label" in node.metadata for node in nodes[1:])
    assert all(part.name for part in asset.parts.values())
    assert all(part.metadata["source_identity"].endswith(str(fixture)) for part in asset.parts.values())
    assert all(part.metadata["shape_fingerprint"] == part.fingerprint for part in asset.parts.values())
    assert asset.root.children
    assert asset.report.steps[0].name == "import"
    assert asset.report.steps[0].before == {
        "nodes": 0,
        "parts": 0,
        "occurrences": 0,
        "materials": 0,
        "vertices": 0,
        "triangles": 0,
    }
    assert asset.report.steps[0].after == asset.stats()


def test_generated_step_assembly_preserves_repeated_occurrences_and_transforms_in_usd(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    step_file = tmp_path / "generated-repeated-assembly.step"
    output = tmp_path / "generated-repeated-assembly.usda"
    _write_repeated_box_step(step_file)

    imported = fc.read_step(step_file)
    occurrences = [node for node in imported.root.walk() if node.part_id is not None]

    assert imported.part_count == 1
    assert imported.occurrence_count == 2
    assert len({node.part_id for node in occurrences}) == 1
    assert any(np.allclose(node.transform[:3, 3], [10.0, 0.0, 0.0]) for node in occurrences)

    converted = fc.convert(
        step_file,
        output,
        tessellation=fc.TessellationOptions(sag=0.2, angle=20),
        optimize=fc.OptimizeOptions(simplify=False, optimize_buffers=False),
        lods=None,
    )
    validation_stats = fc.validate_usd(output)
    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    instance_prims = [prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsInstanceable()]
    translated_instances = [prim for prim in instance_prims if UsdGeom.Xformable(prim).GetOrderedXformOps()]
    transform_ops = UsdGeom.Xformable(translated_instances[0]).GetOrderedXformOps()

    assert converted.part_count == 1
    assert converted.occurrence_count == 2
    assert validation_stats["meshes"] == 2
    assert validation_stats["triangles"] == converted.triangle_count * 2
    assert len(instance_prims) == 2
    assert len(translated_instances) == 1
    assert all(prim.IsInstance() for prim in instance_prims)
    assert np.allclose(np.asarray(transform_ops[0].Get())[:3, 3], [10.0, 0.0, 0.0])


def test_step_ids_include_source_identity(tmp_path: Path) -> None:
    fixture = Path("tests/fixtures/spool-clamp-lid.step")
    copied = tmp_path / "spool-copy.step"
    shutil.copyfile(fixture, copied)

    original_a = fc.read_step(fixture)
    original_b = fc.read_step(fixture)
    copied_asset = fc.read_step(copied)

    assert set(original_a.parts) == set(original_b.parts)
    assert set(original_a.parts) != set(copied_asset.parts)
    assert all("source_identity" in part.metadata for part in original_a.parts.values())


def test_step_cad_color_imports_and_exports_as_visible_usd_material(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    fixture = Path("tests/fixtures/radial-fan-50x15.step")
    expected_color = pytest.approx((0.009721217676997185, 0.009721217676997185, 0.009721217676997185), abs=1e-6)

    imported = fc.read_step(fixture)
    material = next(iter(imported.materials.values()))
    part = next(iter(imported.parts.values()))

    assert imported.units == "metre"
    assert imported.meters_per_unit == pytest.approx(1.0)
    assert part.material_ids == [material.id]
    assert material.base_color[:3] == expected_color

    output = tmp_path / "radial.usda"
    fc.convert(
        fixture,
        output,
        tessellation=fc.TessellationOptions(sag=0.002, angle=20),
        optimize=fc.OptimizeOptions(target_triangles=80),
        lods=None,
    )

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    bound_material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]

    assert tuple(usd_mesh.GetDisplayColorAttr().Get()[0]) == expected_color
    assert bound_material.GetPrim().GetCustomDataByKey("fascat:materialId") == material.id


def test_step_fixture_converts_to_valid_usd_with_report(tmp_path: Path) -> None:
    output = tmp_path / "spool.usda"

    asset = fc.convert(
        "tests/fixtures/spool-clamp-lid.step",
        output,
        tessellation=fc.TessellationOptions(sag=0.2, angle=20),
        optimize=fc.OptimizeOptions(target_triangles=120),
        lods=fc.LODOptions((0.5,)),
    )

    assert output.exists()
    assert asset.triangle_count <= 120
    assert {step.name for step in asset.report.steps} >= {
        "import",
        "tessellate",
        "repair",
        "stage",
        "optimize",
        "lods",
        "write",
        "validate",
    }
    assert asset.report.finished_at is not None
    assert fc.validate_usd(output)["triangles"] == asset.triangle_count


def test_tessellation_max_edge_length_limits_fixture_edges() -> None:
    asset = fc.read_step("tests/fixtures/spool-clamp-lid.step").tessellate(
        fc.TessellationOptions(sag=0.2, angle=20, max_edge_length=10.0, create_normals=False)
    )
    mesh = next(part.mesh for part in asset.parts.values() if part.mesh is not None)
    edge_lengths = []
    for face in mesh.faces:
        corners = mesh.points[face]
        edge_lengths.extend(
            [
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[1]),
                np.linalg.norm(corners[0] - corners[2]),
            ]
        )

    assert max(edge_lengths) <= 10.0
    assert mesh.normals is None


def test_convert_progress_callback_receives_stage_stats(tmp_path: Path) -> None:
    output = tmp_path / "spool.usda"
    progress: list[tuple[str, dict[str, int]]] = []

    fc.convert(
        "tests/fixtures/spool-clamp-lid.step",
        output,
        tessellation=fc.TessellationOptions(sag=0.2, angle=20),
        optimize=fc.OptimizeOptions(target_triangles=120),
        lods=fc.LODOptions((0.5,)),
        progress=lambda step, stats: progress.append((step, stats)),
    )

    assert [step for step, _stats in progress] == [
        "source",
        "tessellate",
        "repair",
        "stage",
        "optimize",
        "lods",
        "write",
        "validate",
    ]
    assert all("triangles" in stats for _step, stats in progress)
