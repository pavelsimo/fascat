"""Round-trip regression: STEP → GLB/USD output must preserve geometry, materials, and transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fascat as fc
from fascat.io.gltf import _read_document, validate_gltf
from fascat.io.usd import validate_usd

pytestmark = [pytest.mark.requires_ocp, pytest.mark.requires_usd]

_FIXTURE = Path("tests/fixtures/spool-clamp-body.step")


def _exact_count_convert(source: Path, output: Path) -> fc.Asset:
    # simplify/LODs off so output triangle counts match the converted asset exactly.
    return fc.convert(
        source,
        output,
        tessellation=fc.TessellationOptions(sag=0.2, angle=20),
        optimize=fc.OptimizeOptions(simplify=False, optimize_buffers=False),
        lods=None,
    )


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
    document = TDocStd_Document(TCollection_ExtendedString("fascat-round-trip"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    box = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
    assembly_shape = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(assembly_shape)
    builder.Add(assembly_shape, located_shape(0.0, 0.0, 0.0))
    builder.Add(assembly_shape, located_shape(10.0, 0.0, 0.0))
    assembly_label = shape_tool.AddShape(assembly_shape, True, True)
    TDataStd_Name.Set_s(assembly_label, TCollection_ExtendedString("Round Trip Assembly"))

    writer = STEPCAFControl_Writer()
    writer.SetNameMode(True)
    assert writer.Transfer(document, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_step_to_glb_round_trip_preserves_triangles_and_materials(tmp_path: Path) -> None:
    output = tmp_path / "round-trip.glb"

    asset = _exact_count_convert(_FIXTURE, output)
    stats = validate_gltf(output)
    document, _buffers = _read_document(output)

    assert stats["triangles"] == asset.triangle_count
    assert stats["meshes"] == asset.occurrence_count
    exported_materials = document.get("materials", [])
    assert len(exported_materials) == len(asset.materials)
    assert {material.get("name") for material in exported_materials} == {
        material.name for material in asset.materials.values()
    }


def test_step_to_usd_round_trip_preserves_triangles_and_materials(tmp_path: Path) -> None:
    from pxr import Usd, UsdShade

    output = tmp_path / "round-trip.usda"

    asset = _exact_count_convert(_FIXTURE, output)
    stats = validate_usd(output)

    assert stats["triangles"] == asset.triangle_count
    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    material_prims = [prim for prim in Usd.PrimRange(stage.GetPseudoRoot()) if prim.IsA(UsdShade.Material)]
    assert len(material_prims) == len(asset.materials)


def test_step_assembly_round_trip_preserves_instance_transforms(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    step_file = tmp_path / "assembly.step"
    _write_repeated_box_step(step_file)

    imported = fc.read_step(step_file)
    occurrences = [node for node in imported.root.walk() if node.part_id is not None]
    assert imported.part_count == 1
    assert imported.occurrence_count == 2
    assert any(np.allclose(node.transform[:3, 3], [10.0, 0.0, 0.0]) for node in occurrences)

    glb_output = tmp_path / "assembly.glb"
    converted = _exact_count_convert(step_file, glb_output)
    document, _buffers = _read_document(glb_output)
    base_mesh_indices = {
        index
        for index, mesh in enumerate(document.get("meshes", []))
        if mesh.get("extras", {}).get("fascat", {}).get("lod") == 0
    }
    mesh_nodes = [node for node in document["nodes"] if node.get("mesh") in base_mesh_indices]
    assert len(mesh_nodes) == 2
    assert len({node["mesh"] for node in mesh_nodes}) == 1
    translations = []
    for node in mesh_nodes:
        if "translation" in node:
            translations.append(node["translation"])
        elif "matrix" in node:
            translations.append(node["matrix"][12:15])
    # glTF output is normalized to metres; the source offset is 10 source units.
    expected_offset = [10.0 * converted.meters_per_unit, 0.0, 0.0]
    assert any(np.allclose(translation, expected_offset, atol=1e-9) for translation in translations)

    usd_output = tmp_path / "assembly.usda"
    _exact_count_convert(step_file, usd_output)
    stage = Usd.Stage.Open(str(usd_output))
    assert stage is not None
    instance_prims = [prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsInstanceable()]
    assert len(instance_prims) == 2
    translated = [ops[0].Get() for prim in instance_prims if (ops := UsdGeom.Xformable(prim).GetOrderedXformOps())]
    # Gf.Matrix4d is row-vector: translation lives in the last row.
    assert any(np.allclose(np.asarray(value)[3, :3], [10.0, 0.0, 0.0]) for value in translated)

    assert converted.part_count == 1
    assert converted.occurrence_count == 2
