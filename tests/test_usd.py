from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fascat as fc
from fascat.asset import Asset, Node, Part
from fascat.io.usd import _usd_custom_data, validate_usd, write_usd, write_usd_with_validation_stats
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import OptimizeOptions, UsdExportOptions

pytestmark = pytest.mark.requires_usd
pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdShade  # noqa: E402


def cube_mesh() -> Mesh:
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    return Mesh(points=points, faces=faces).compute_normals().box_uv()


def test_usd_custom_data_serializes_nested_transform_lists() -> None:
    payload = _usd_custom_data({"transform": [[1.0, 0.0], [0.0, 1.0]], "unset": None})

    assert payload == {"transform": "[[1.0, 0.0], [0.0, 1.0]]"}


def author_triangle_mesh(stage: object, path: str, indices: list[int]) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr(indices)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("no_default_prim", "no defaultPrim"),
        ("no_meshes", "contains no meshes"),
        ("non_triangle", "non-triangle faces"),
        ("invalid_index_count", "invalid face index count"),
        ("out_of_range_indices", "out-of-range face indices"),
        ("bad_subdivision", "subdivisionScheme"),
    ],
)
def test_validate_usd_rejects_invalid_stage_structure(tmp_path: Path, case: str, message: str) -> None:
    output = tmp_path / f"{case}.usda"
    stage = Usd.Stage.CreateNew(str(output))
    assert stage is not None

    if case == "no_default_prim":
        author_triangle_mesh(stage, "/Scene/Mesh", [0, 1, 2])
    else:
        scene = UsdGeom.Xform.Define(stage, "/Scene")
        stage.SetDefaultPrim(scene.GetPrim())
        if case == "non_triangle":
            mesh = UsdGeom.Mesh.Define(stage, "/Scene/Mesh")
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreatePointsAttr(
                [
                    (0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (0.0, 1.0, 0.0),
                ]
            )
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        elif case == "invalid_index_count":
            author_triangle_mesh(stage, "/Scene/Mesh", [0, 1])
        elif case == "out_of_range_indices":
            author_triangle_mesh(stage, "/Scene/Mesh", [0, 1, 9])
        elif case == "bad_subdivision":
            mesh = UsdGeom.Mesh.Define(stage, "/Scene/Mesh")
            mesh.CreateSubdivisionSchemeAttr("catmullClark")
            mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
            mesh.CreateFaceVertexCountsAttr([3])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    assert stage.GetRootLayer().Save()
    with pytest.raises(RuntimeError, match=message):
        validate_usd(output)


def test_usd_export_authors_mesh_material_units_and_lods(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(id="root", name="root", children=[Node(id="n1", name="Cube Occurrence", part_id="cube")])
    material = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))
    asset = Asset(
        root=root,
        parts={
            "cube": Part(id="cube", name="Cube", mesh=mesh, material_ids=["red"], lod_meshes=[mesh.simplify(ratio=0.5)])
        },
        materials={"red": material},
    )
    output = tmp_path / "cube.usda"

    write_usd(asset, output)
    stats = validate_usd(output)

    assert stats["meshes"] == 1
    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert UsdGeom.GetStageMetersPerUnit(stage) == 0.001
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    prim = stage.GetPrimAtPath("/Scene/Cube_Occurrence")
    variant_set = prim.GetVariantSets().GetVariantSet("lod")
    assert variant_set.GetVariantSelection() == "lod0"
    assert variant_set.GetVariantNames() == ["lod0", "lod1"]
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    face_counts = usd_mesh.GetFaceVertexCountsAttr().Get()
    face_indices = usd_mesh.GetFaceVertexIndicesAttr().Get()
    points = usd_mesh.GetPointsAttr().Get()

    assert usd_mesh.GetSubdivisionSchemeAttr().Get() == "none"
    assert len(points) == mesh.vertex_count
    assert all(count == 3 for count in face_counts)
    assert len(face_counts) == mesh.triangle_count
    assert len(face_indices) == mesh.triangle_count * 3
    assert min(face_indices) >= 0
    assert max(face_indices) < len(points)
    assert usd_mesh.GetDisplayColorAttr().Get()[0] == (1.0, 0.0, 0.0)
    assert "MaterialBindingAPI" in mesh_prim.GetAppliedSchemas()
    extent = usd_mesh.GetExtentAttr().Get()
    assert len(extent) == 2
    assert tuple(extent[0]) == (-1.0, -1.0, -1.0)
    assert tuple(extent[1]) == (1.0, 1.0, 1.0)


def test_usd_export_authors_metadata_and_pmi_custom_data(tmp_path: Path) -> None:
    mesh = cube_mesh()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            metadata={"assembly": "demo"},
            children=[Node(id="node", name="Cube", part_id="cube", metadata={"step_label": "0:1"})],
        ),
        parts={
            "cube": Part(
                id="cube",
                name="Cube",
                mesh=mesh,
                metadata={"layer": "A"},
            )
        },
        materials={},
        metadata={"document": "demo.step"},
        pmi=[fc.PmiAnnotation(id="pmi_001", kind="dimension", text="10", applies_to=["cube"])],
    )
    output = tmp_path / "metadata.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    scene = stage.GetPrimAtPath("/Scene")
    prototype = next(
        prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/__Prototypes")) if prim.GetName() != "__Prototypes"
    )
    node = stage.GetPrimAtPath("/Scene/Cube")
    pmi = stage.GetPrimAtPath("/PMI/pmi_001")

    assert scene.GetCustomDataByKey("fascat:metadata")["document"] == "demo.step"
    assert node.GetCustomDataByKey("fascat:metadata")["step_label"] == "0:1"
    assert prototype.GetCustomDataByKey("fascat:metadata")["layer"] == "A"
    assert prototype.GetCustomDataByKey("fascat:pmiIds") == ["pmi_001"]
    assert pmi.GetCustomDataByKey("fascat:type") == "dimension"
    assert pmi.GetCustomDataByKey("fascat:appliesTo") == ["cube"]


def test_usd_export_resolves_pmi_links_through_source_part_metadata(tmp_path: Path) -> None:
    mesh = cube_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh, metadata={"layer": "A"})},
        materials={},
        pmi=[fc.PmiAnnotation(id="pmi_001", kind="dimension", text="10", applies_to=["cube"])],
    ).replace(fc.ReplaceOptions(mode="bounding_box"), where=fc.Filter.part("cube"))
    output = tmp_path / "metadata-replaced.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    prototype = next(
        prim for prim in Usd.PrimRange(stage.GetPrimAtPath("/__Prototypes")) if prim.GetName() != "__Prototypes"
    )

    assert prototype.GetCustomDataByKey("fascat:metadata")["source_part_ids"] == "cube"
    assert prototype.GetCustomDataByKey("fascat:pmiIds") == ["pmi_001"]


def test_usd_export_writes_binary_usdc(tmp_path: Path) -> None:
    mesh = cube_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
        materials={},
    )
    output = tmp_path / "cube.usdc"

    write_usd(asset, output)

    assert output.read_bytes().startswith(b"PXR-USDC")
    assert validate_usd(output)["triangles"] == mesh.triangle_count


def test_usdz_export_packages_stage_and_validates(tmp_path: Path) -> None:
    mesh = cube_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
        materials={},
    )
    output = tmp_path / "cube.usdz"

    asset.write_usd(output, options=UsdExportOptions(package="usdz"))

    assert output.read_bytes()[:2] == b"PK"
    assert validate_usd(output)["triangles"] == mesh.triangle_count


def test_validate_usd_rejects_invalid_lod_variant_mesh(tmp_path: Path) -> None:
    output = tmp_path / "invalid-lod.usda"
    stage = Usd.Stage.CreateNew(str(output))
    assert stage is not None
    scene = UsdGeom.Xform.Define(stage, "/Scene")
    stage.SetDefaultPrim(scene.GetPrim())
    thing_prim = UsdGeom.Xform.Define(stage, "/Scene/Thing").GetPrim()
    variant_set = thing_prim.GetVariantSets().AddVariantSet("lod")

    variant_set.AddVariant("lod0")
    variant_set.SetVariantSelection("lod0")
    with variant_set.GetVariantEditContext():
        author_triangle_mesh(stage, "/Scene/Thing/Mesh", [0, 1, 2])

    variant_set.AddVariant("lod1")
    variant_set.SetVariantSelection("lod1")
    with variant_set.GetVariantEditContext():
        author_triangle_mesh(stage, "/Scene/Thing/Mesh", [0, 1, 9])

    variant_set.SetVariantSelection("lod0")
    assert stage.GetRootLayer().Save()

    with pytest.raises(RuntimeError, match="out-of-range face indices"):
        validate_usd(output)


def test_public_usd_write_apis_export_valid_stages(tmp_path: Path) -> None:
    mesh = cube_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
        materials={},
    )
    method_output = tmp_path / "method.usda"
    function_output = tmp_path / "function.usda"

    asset.write_usd(method_output)
    fc.write_usd(asset, function_output)

    assert validate_usd(method_output)["triangles"] == mesh.triangle_count
    assert validate_usd(function_output)["triangles"] == mesh.triangle_count
    assert [step.name for step in asset.report.steps] == ["write", "write"]


def test_usd_export_authors_preview_surface_material_inputs(tmp_path: Path) -> None:
    mesh = cube_mesh()
    material = Material(
        id="pbr",
        name="PBR Material",
        base_color=(0.2, 0.4, 0.6, 0.8),
        metallic=0.25,
        roughness=0.7,
        opacity=0.35,
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh, material_ids=[material.id])},
        materials={material.id: material},
    )
    output = tmp_path / "preview-surface.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    shader = UsdShade.Shader(stage.GetPrimAtPath("/Materials/pbr/PreviewSurface"))
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    bound_material = UsdShade.MaterialBindingAPI(mesh_prim).ComputeBoundMaterial()[0]

    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert tuple(shader.GetInput("diffuseColor").Get()) == pytest.approx((0.2, 0.4, 0.6))
    assert shader.GetInput("opacity").Get() == pytest.approx(0.35)
    assert shader.GetInput("metallic").Get() == pytest.approx(0.25)
    assert shader.GetInput("roughness").Get() == pytest.approx(0.7)
    assert bound_material.GetPath().pathString == "/Materials/pbr"


def test_usd_export_authors_baked_texture_shader_bindings(tmp_path: Path) -> None:
    mesh = cube_mesh()
    uris = {
        "baked_texture_base_color_uri": "data:image/png;base64,BASE",
        "baked_texture_metallic_roughness_uri": "data:image/png;base64,MR",
        "baked_texture_normal_uri": "data:image/png;base64,NORMAL",
        "baked_texture_occlusion_uri": "data:image/png;base64,OCC",
        "baked_texture_emissive_uri": "data:image/png;base64,EMISSIVE",
    }
    material = Material(
        id="baked",
        name="Baked",
        base_color=(0.2, 0.4, 0.6, 0.8),
        metadata=uris,
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh, material_ids=[material.id])},
        materials={material.id: material},
    )
    output = tmp_path / "baked-textures.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    preview = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/PreviewSurface"))
    base_texture = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/BaseColorTexture"))
    mr_texture = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/MetallicRoughnessTexture"))
    normal_texture = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/NormalTexture"))
    occlusion_texture = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/OcclusionTexture"))
    emissive_texture = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/EmissiveTexture"))
    st_reader = UsdShade.Shader(stage.GetPrimAtPath("/Materials/baked/StReader"))

    assert st_reader.GetIdAttr().Get() == "UsdPrimvarReader_float2"
    assert st_reader.GetInput("varname").Get() == "st"
    assert base_texture.GetIdAttr().Get() == "UsdUVTexture"
    assert base_texture.GetInput("file").Get().path == uris["baked_texture_base_color_uri"]
    assert base_texture.GetInput("sourceColorSpace").Get() == "sRGB"
    assert mr_texture.GetInput("file").Get().path == uris["baked_texture_metallic_roughness_uri"]
    assert mr_texture.GetInput("sourceColorSpace").Get() == "raw"
    assert normal_texture.GetInput("file").Get().path == uris["baked_texture_normal_uri"]
    assert occlusion_texture.GetInput("file").Get().path == uris["baked_texture_occlusion_uri"]
    assert emissive_texture.GetInput("file").Get().path == uris["baked_texture_emissive_uri"]
    assert emissive_texture.GetInput("sourceColorSpace").Get() == "sRGB"
    assert _connected_source(preview, "diffuseColor") == ("/Materials/baked/BaseColorTexture", "rgb")
    assert _connected_source(preview, "opacity") == ("/Materials/baked/BaseColorTexture", "a")
    assert _connected_source(preview, "roughness") == ("/Materials/baked/MetallicRoughnessTexture", "g")
    assert _connected_source(preview, "metallic") == ("/Materials/baked/MetallicRoughnessTexture", "b")
    assert _connected_source(preview, "normal") == ("/Materials/baked/NormalTexture", "rgb")
    assert _connected_source(preview, "occlusion") == ("/Materials/baked/OcclusionTexture", "r")
    assert _connected_source(preview, "emissiveColor") == ("/Materials/baked/EmissiveTexture", "rgb")
    assert _connected_source(base_texture, "st") == ("/Materials/baked/StReader", "result")


def _connected_source(shader: object, input_name: str) -> tuple[str, str]:
    source, source_name, _source_type = shader.GetInput(input_name).GetConnectedSource()
    return source.GetPath().pathString, source_name


def test_usd_export_authors_uv0_normals_and_original_names(tmp_path: Path) -> None:
    mesh = cube_mesh()
    transform = np.eye(4, dtype=float)
    transform[0, 0] = 2.0
    transform[1, 1] = 3.0
    transform[2, 2] = 4.0
    root = Node(
        id="root",
        name="root",
        children=[Node(id="node", name="123 motor housing!", part_id="part", transform=transform)],
    )
    asset = Asset(
        root=root,
        parts={"part": Part(id="part", name="housing source name", mesh=mesh)},
        materials={},
    )
    output = tmp_path / "attributes.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    xform_prim = stage.GetPrimAtPath("/Scene/_123_motor_housing")
    assert xform_prim
    assert xform_prim.GetCustomDataByKey("fascat:originalName") == "123 motor housing!"
    assert xform_prim.GetCustomDataByKey("fascat:nodeId") == "node"
    xform_ops = UsdGeom.Xformable(xform_prim).GetOrderedXformOps()
    assert len(xform_ops) == 1
    assert np.allclose(np.asarray(xform_ops[0].Get()), transform)

    mesh_prim = stage.GetPrimAtPath("/Scene/_123_motor_housing/Mesh")
    usd_mesh = UsdGeom.Mesh(mesh_prim)
    normals = usd_mesh.GetNormalsAttr().Get()
    st = UsdGeom.PrimvarsAPI(usd_mesh).GetPrimvar("st")

    assert mesh_prim.GetCustomDataByKey("fascat:originalName") == "housing source name"
    assert normals is not None
    assert len(normals) == mesh.vertex_count
    assert usd_mesh.GetNormalsInterpolation() == UsdGeom.Tokens.vertex
    assert st
    assert st.GetInterpolation() == UsdGeom.Tokens.vertex
    assert len(st.Get()) == mesh.vertex_count


def test_usd_export_preserves_original_names_on_sanitized_prototypes_and_materials(tmp_path: Path) -> None:
    mesh = cube_mesh()
    material = Material(id="12 red material!", name="12 red material!", base_color=(1.0, 0.0, 0.0, 1.0))
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[Node(id="node", name="Occurrence", part_id="123 part id!")],
        ),
        parts={
            "123 part id!": Part(
                id="123 part id!",
                name="123 part source!",
                mesh=mesh,
                material_ids=[material.id],
            )
        },
        materials={material.id: material},
    )
    output = tmp_path / "sanitized.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    material_prim = stage.GetPrimAtPath("/Materials/_12_red_material")
    prototype_prim = stage.GetPrimAtPath("/__Prototypes/_123_part_id_lod0")

    assert material_prim
    assert material_prim.GetCustomDataByKey("fascat:materialId") == "12 red material!"
    assert material_prim.GetCustomDataByKey("fascat:originalName") == "12 red material!"
    assert prototype_prim
    assert prototype_prim.GetCustomDataByKey("fascat:partId") == "123 part id!"
    assert prototype_prim.GetCustomDataByKey("fascat:originalName") == "123 part source!"


def test_usd_export_disambiguates_sanitized_name_collisions(tmp_path: Path) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )
    materials = {
        "red!": Material(id="red!", name="red bang", base_color=(1.0, 0.0, 0.0, 1.0)),
        "red?": Material(id="red?", name="red question", base_color=(0.8, 0.0, 0.0, 1.0)),
    }
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node-a", name="panel-a", part_id="part-a"),
                Node(id="node-b", name="panel_a", part_id="part_a"),
            ],
        ),
        parts={
            "part-a": Part(id="part-a", name="part dash", mesh=mesh, material_ids=["red!", "red?"]),
            "part_a": Part(id="part_a", name="part underscore", mesh=mesh.copy()),
        },
        materials=materials,
    )
    output = tmp_path / "collisions.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    first_material = stage.GetPrimAtPath("/Materials/red")
    second_material = stage.GetPrimAtPath("/Materials/red_2")
    first_prototype = stage.GetPrimAtPath("/__Prototypes/part_a_lod0")
    second_prototype = stage.GetPrimAtPath("/__Prototypes/part_a_lod0_2")
    first_subset = stage.GetPrimAtPath("/__Prototypes/part_a_lod0/Mesh/red")
    second_subset = stage.GetPrimAtPath("/__Prototypes/part_a_lod0/Mesh/red_2")
    first_node = stage.GetPrimAtPath("/Scene/panel_a")
    second_node = stage.GetPrimAtPath("/Scene/panel_a_2")

    assert first_node.GetCustomDataByKey("fascat:originalName") == "panel-a"
    assert first_node.GetCustomDataByKey("fascat:nodeId") == "node-a"
    assert second_node.GetCustomDataByKey("fascat:originalName") == "panel_a"
    assert second_node.GetCustomDataByKey("fascat:nodeId") == "node-b"
    assert first_material.GetCustomDataByKey("fascat:materialId") == "red!"
    assert second_material.GetCustomDataByKey("fascat:materialId") == "red?"
    assert first_prototype.GetCustomDataByKey("fascat:partId") == "part-a"
    assert second_prototype.GetCustomDataByKey("fascat:partId") == "part_a"
    assert first_subset.GetCustomDataByKey("fascat:materialId") == "red!"
    assert second_subset.GetCustomDataByKey("fascat:materialId") == "red?"


def test_usd_export_uses_instanceable_references_for_repeated_parts(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="n1", name="Cube A", part_id="cube"),
            Node(id="n2", name="Cube B", part_id="cube"),
        ],
    )
    asset = Asset(root=root, parts={"cube": Part(id="cube", name="Cube", mesh=mesh)}, materials={})
    output = tmp_path / "instances.usda"

    written_stats = write_usd_with_validation_stats(asset, output)
    stats = validate_usd(output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetPrimAtPath("/Scene/Cube_A").IsInstanceable()
    assert stage.GetPrimAtPath("/Scene/Cube_B").IsInstanceable()
    assert stage.GetPrimAtPath("/__Prototypes/cube_lod0/Mesh")
    assert written_stats == stats
    assert stats["meshes"] == 2
    assert stats["triangles"] == mesh.triangle_count * 2


def test_usd_export_marks_repeated_parts_instanceable_across_assemblies(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="g1", name="Group A", children=[Node(id="n1", name="Cube", part_id="cube")]),
            Node(id="g2", name="Group B", children=[Node(id="n2", name="Cube", part_id="cube")]),
        ],
    )
    asset = Asset(root=root, parts={"cube": Part(id="cube", name="Cube", mesh=mesh)}, materials={})
    output = tmp_path / "nested-instances.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetPrimAtPath("/Scene/Group_A/Cube").IsInstanceable()
    assert stage.GetPrimAtPath("/Scene/Group_B/Cube").IsInstanceable()


def test_usd_export_authors_face_material_subsets(tmp_path: Path) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )
    root = Node(id="root", name="root", children=[Node(id="node", name="Panel", part_id="panel")])
    red = Material(id="red metal!", name="red material source!", base_color=(1.0, 0.0, 0.0, 1.0))
    blue = Material(id="2 blue paint", name="blue material source!", base_color=(0.0, 0.0, 1.0, 1.0))
    asset = Asset(
        root=root,
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=[red.id, blue.id])},
        materials={red.id: red, blue.id: blue},
    )
    output = tmp_path / "subsets.usda"

    write_usd(asset, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    subsets = [prim for prim in Usd.PrimRange(mesh_prim) if prim.GetTypeName() == "GeomSubset"]
    subset_indices = sorted(UsdGeom.Subset(prim).GetIndicesAttr().Get()[0] for prim in subsets)
    subset_bindings = [
        UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0].GetPath().pathString for prim in subsets
    ]
    subset_metadata = {
        prim.GetCustomDataByKey("fascat:materialId"): prim.GetCustomDataByKey("fascat:originalName") for prim in subsets
    }

    assert len(subsets) == 2
    assert subset_indices == [0, 1]
    assert all("MaterialBindingAPI" in prim.GetAppliedSchemas() for prim in subsets)
    assert sorted(subset_bindings) == ["/Materials/_2_blue_paint", "/Materials/red_metal"]
    assert subset_metadata == {
        "red metal!": "red material source!",
        "2 blue paint": "blue material source!",
    }


def test_usd_export_does_not_instance_when_instances_are_not_preserved(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="n1", name="Cube A", part_id="cube"),
            Node(id="n2", name="Cube B", part_id="cube"),
        ],
    )
    asset = Asset(root=root, parts={"cube": Part(id="cube", name="Cube", mesh=mesh)}, materials={})
    optimized = asset.optimize(OptimizeOptions(simplify=False, optimize_buffers=False, preserve_instances=False))
    output = tmp_path / "duplicates.usda"

    write_usd(optimized, output)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert not stage.GetPrimAtPath("/Scene/Cube_A").IsInstanceable()
    assert not stage.GetPrimAtPath("/Scene/Cube_B").IsInstanceable()


def test_usd_export_authors_debug_metadata_and_display_color_only_materials(tmp_path: Path) -> None:
    mesh = cube_mesh()
    root = Node(id="root", name="root", children=[Node(id="n1", name="Cube", part_id="cube")])
    part = Part(
        id="cube",
        name="Cube",
        mesh=mesh,
        metadata={"display_color": "0.100000,0.200000,0.300000,0.400000"},
    )
    asset = Asset(root=root, parts={"cube": part}, materials={})
    output = tmp_path / "debug.usda"

    write_usd(asset, output, debug=True)

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetDefaultPrim().GetCustomDataByKey("fascat:debug") is True
    mesh_prim = next(prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    usd_mesh = UsdGeom.Mesh(mesh_prim)

    assert usd_mesh.GetDisplayColorAttr().Get()[0] == (0.1, 0.2, 0.3)
    assert usd_mesh.GetDisplayOpacityAttr().Get()[0] == pytest.approx(0.4)
