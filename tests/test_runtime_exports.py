from __future__ import annotations

import json
import struct

import numpy as np
import pytest
from typer.testing import CliRunner

from fascat.analysis import analyze_output
from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.io.gltf import validate_gltf
from fascat.io.obj import validate_obj
from fascat.io.stl import validate_stl
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import GltfExportOptions, ObjExportOptions, StlExportOptions

runner = CliRunner()


def _asset() -> Asset:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="tri", name="Triangle", part_id="tri")]),
        parts={"tri": Part(id="tri", name="Triangle", mesh=mesh, material_ids=["mat"])},
        materials={"mat": Material(id="mat", name="Mat", base_color=(0.2, 0.4, 0.6, 1.0))},
        units="metre",
        meters_per_unit=1.0,
        up_axis="Y",
    )


def test_gltf_export_options_write_meshopt_extension_and_file_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    asset = _asset()
    output = tmp_path / "triangle.gltf"

    asset.write_gltf(output, options=GltfExportOptions(quantize=True, meshopt=True, file_size_budget_mb=0.000001))

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["extras"]["fascat"]["compression"] == {
        "quantize": True,
        "meshopt": True,
    }
    assert "KHR_mesh_quantization" in document["extensionsUsed"]
    assert "KHR_mesh_quantization" in document["extensionsRequired"]
    assert "EXT_meshopt_compression" in document["extensionsUsed"]
    primitive = document["meshes"][0]["primitives"][0]
    position_accessor = document["accessors"][primitive["attributes"]["POSITION"]]
    quantized_node = next(node for node in document["nodes"] if node.get("mesh") == 0)
    assert position_accessor["componentType"] == 5123
    assert position_accessor["max"] == [65535, 65535, 0]
    assert quantized_node["matrix"][0] == pytest.approx(1.0 / 65535.0)
    compressed_views = [
        view["extensions"]["EXT_meshopt_compression"]
        for view in document["bufferViews"]
        if "EXT_meshopt_compression" in view.get("extensions", {})
    ]
    assert compressed_views
    assert {view["mode"] for view in compressed_views} >= {"ATTRIBUTES", "TRIANGLES"}
    assert validate_gltf(output)["triangles"] == 1
    assert analyze_output(output).parts[0]["bounds"]["max"] == pytest.approx([1.0, 1.0, 0.0])
    runtime_dependencies = asset.report.steps[-1].options["runtime_dependencies"]
    assert runtime_dependencies["extensions_used"] == ["KHR_mesh_quantization", "EXT_meshopt_compression"]
    assert runtime_dependencies["extensions_required"] == ["KHR_mesh_quantization"]
    assert "EXT_meshopt_compression" in runtime_dependencies["expected_runtime_support"]
    compatibility = runtime_dependencies["runtime_compatibility"]
    assert set(compatibility) == {"unity_gltfast", "web", "mobile", "xr"}
    web_extensions = compatibility["web"]["extensions"]
    assert web_extensions["KHR_mesh_quantization"]["state"] == "required"
    assert web_extensions["KHR_mesh_quantization"]["fallback"].startswith("no fallback")
    assert web_extensions["EXT_meshopt_compression"]["state"] == "optional"
    assert "fallback buffer data" in web_extensions["EXT_meshopt_compression"]["fallback"]
    assert web_extensions["KHR_texture_basisu"]["state"] == "not_written"
    decision_matrix = runtime_dependencies["runtime_decision_matrix"]
    geometry_policy = decision_matrix["geometry"]
    assert geometry_policy["quantization"]["state"] == "enabled_required"
    assert geometry_policy["meshopt"]["state"] == "enabled_optional"
    assert geometry_policy["draco"]["state"] == "unsupported"
    assert "prefer meshopt" in decision_matrix["targets"]["web"]["geometry"]
    assert asset.report.steps[-1].after["file_size_bytes"] > 0
    assert asset.report.steps[-1].after["export_estimated_geometry_bytes"] == 96
    assert asset.report.steps[-1].after["export_estimated_texture_bytes"] == 0
    assert asset.report.steps[-1].after["export_estimated_metadata_bytes"] == 0
    assert asset.report.steps[-1].after["export_estimated_payload_bytes"] == 96
    assert asset.report.steps[-1].after["export_source_material_count"] == 1
    assert asset.report.steps[-1].after["export_referenced_material_count"] == 1
    assert asset.report.steps[-1].after["export_unused_material_count"] == 0
    assert asset.report.steps[-1].after["export_written_material_count"] == 1
    assert asset.report.steps[-1].after["file_size_budget_bytes"] == 1
    assert "file size budget exceeded" in asset.report.warnings[-1]


def test_exports_prune_unused_materials_and_report_counts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    gltf_asset = _asset()
    gltf_asset.materials["unused"] = Material(
        id="unused",
        name="Unused",
        base_color=(1.0, 0.0, 0.0, 1.0),
        metadata={"baked_texture_base_color_uri": "data:image/png;base64,VU5VU0VE"},
    )
    gltf_output = tmp_path / "used.gltf"

    gltf_asset.write_gltf(gltf_output)
    document = json.loads(gltf_output.read_text(encoding="utf-8"))
    after = gltf_asset.report.steps[-1].after

    assert [material["name"] for material in document["materials"]] == ["Mat"]
    assert "images" not in document
    assert after["export_source_material_count"] == 2
    assert after["export_referenced_material_count"] == 1
    assert after["export_unused_material_count"] == 1
    assert after["export_written_material_count"] == 1
    assert after["export_source_image_count"] == 1
    assert after["export_referenced_image_count"] == 0
    assert after["export_unused_image_count"] == 1
    assert after["export_duplicate_image_reference_count"] == 0
    assert after["export_written_image_count"] == 0
    assert after["export_estimated_texture_bytes"] == 0

    obj_asset = _asset()
    obj_asset.materials["unused"] = Material(id="unused", name="Unused", base_color=(1.0, 0.0, 0.0, 1.0))
    obj_output = tmp_path / "used.obj"

    obj_asset.write_obj(obj_output, options=ObjExportOptions(materials=True, write_mtl=True))
    mtl = obj_output.with_suffix(".mtl").read_text(encoding="utf-8")

    assert "newmtl mat" in mtl
    assert "newmtl unused" not in mtl


def test_gltf_export_deduplicates_referenced_texture_images_and_reports_counts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    shared_uri = "data:image/png;base64,QUJD"
    unused_uri = "data:image/png;base64,VU5VU0VE"
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.asarray([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="quad", name="Quad", part_id="quad")]),
        parts={"quad": Part(id="quad", name="Quad", mesh=mesh, material_ids=["red", "blue"])},
        materials={
            "red": Material(
                id="red",
                name="Red",
                base_color=(1.0, 0.0, 0.0, 1.0),
                metadata={"baked_texture_base_color_uri": shared_uri},
            ),
            "blue": Material(
                id="blue",
                name="Blue",
                base_color=(0.0, 0.0, 1.0, 1.0),
                metadata={"baked_texture_base_color_uri": shared_uri},
            ),
            "unused": Material(
                id="unused",
                name="Unused",
                base_color=(0.0, 1.0, 0.0, 1.0),
                metadata={"baked_texture_base_color_uri": unused_uri},
            ),
        },
    )
    output = tmp_path / "dedup.gltf"

    asset.write_gltf(output)
    document = json.loads(output.read_text(encoding="utf-8"))
    after = asset.report.steps[-1].after

    assert [image["uri"] for image in document["images"]] == [shared_uri]
    assert document["textures"] == [{"source": 0}]
    assert document["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert document["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert after["export_source_image_count"] == 2
    assert after["export_source_image_reference_count"] == 3
    assert after["export_referenced_image_count"] == 1
    assert after["export_referenced_image_reference_count"] == 2
    assert after["export_unused_image_count"] == 1
    assert after["export_duplicate_image_reference_count"] == 1
    assert after["export_written_image_count"] == 1
    assert after["export_estimated_texture_bytes"] == 3


def test_write_report_estimates_geometry_texture_and_metadata_payloads(tmp_path) -> None:  # type: ignore[no-untyped-def]
    asset = _asset()
    asset.metadata["asset_note"] = "qa"
    asset.parts["tri"].metadata["part_note"] = "runtime"
    asset.materials["mat"].metadata.update(
        {
            "material_note": "paint",
            "baked_texture_base_color_uri": "data:image/png;base64,QUJD",
        }
    )
    output = tmp_path / "payloads.gltf"

    asset.write_gltf(output)
    after = asset.report.steps[-1].after
    runtime_dependencies = asset.report.steps[-1].options["runtime_dependencies"]

    assert after["export_estimated_geometry_bytes"] == 96
    assert after["export_estimated_texture_bytes"] == 3
    assert after["export_estimated_metadata_bytes"] > 0
    assert after["export_estimated_payload_bytes"] == (
        after["export_estimated_geometry_bytes"]
        + after["export_estimated_texture_bytes"]
        + after["export_estimated_metadata_bytes"]
    )
    assert after["export_source_image_count"] == 1
    assert after["export_referenced_image_count"] == 1
    assert after["export_duplicate_image_reference_count"] == 0
    assert after["export_written_image_count"] == 1
    texture_policy = runtime_dependencies["runtime_decision_matrix"]["textures"]
    assert texture_policy["ktx2_basisu"]["state"] == "unsupported"
    assert texture_policy["png_jpeg_fallbacks"]["state"] == "source_textures_present"
    assert texture_policy["png_jpeg_fallbacks"]["fallback_format"] == "auto"
    assert texture_policy["png_jpeg_fallbacks"]["resolved_format"] == "PNG/JPEG"
    assert texture_policy["png_jpeg_fallbacks"]["png_compression"] == 6
    assert texture_policy["png_jpeg_fallbacks"]["jpeg_quality"] == 85
    assert texture_policy["png_jpeg_fallbacks"]["jpeg_alpha_risk_sets"] == 0
    assert "keep PNG/JPEG fallbacks" in texture_policy["png_jpeg_fallbacks"]["recommendation"]


def test_gltf_runtime_texture_fallback_reports_alpha_risk(tmp_path) -> None:  # type: ignore[no-untyped-def]
    asset = _asset()
    asset.materials["mat"] = Material(
        id="mat",
        name="Glass",
        base_color=(1.0, 1.0, 1.0, 0.5),
        opacity=0.5,
        metadata={
            "baked_texture_base_color_uri": "data:image/png;base64,QUJD",
            "baked_maps": "base_color,opacity",
        },
    )
    output = tmp_path / "glass.gltf"

    asset.write_gltf(output, options=GltfExportOptions(texture_fallback_format="jpeg", jpeg_quality=70))
    runtime_dependencies = asset.report.steps[-1].options["runtime_dependencies"]
    fallback_policy = runtime_dependencies["runtime_decision_matrix"]["textures"]["png_jpeg_fallbacks"]

    assert fallback_policy["fallback_format"] == "jpeg"
    assert fallback_policy["resolved_format"] == "JPEG"
    assert fallback_policy["jpeg_quality"] == 70
    assert fallback_policy["alpha_texture_sets"] == 1
    assert fallback_policy["jpeg_alpha_risk_sets"] == 1
    assert "avoid JPEG fallback" in fallback_policy["recommendation"]


def test_gltf_write_reports_lod_and_metadata_runtime_dependencies(tmp_path) -> None:  # type: ignore[no-untyped-def]
    asset = _asset()
    part = asset.parts["tri"]
    assert part.mesh is not None
    part.lod_meshes = [part.mesh.copy()]
    output = tmp_path / "triangle_lod.gltf"

    asset.write_gltf(output, options=GltfExportOptions(quantize=True))

    document = json.loads(output.read_text(encoding="utf-8"))
    runtime_dependencies = asset.report.steps[-1].options["runtime_dependencies"]

    assert "MSFT_lod" in document["extensionsUsed"]
    assert runtime_dependencies["extensions_used"] == ["KHR_mesh_quantization", "MSFT_lod"]
    assert runtime_dependencies["extensions_required"] == ["KHR_mesh_quantization"]
    assert runtime_dependencies["extras"] == {"fascat": True, "metadata": "full", "pmi": "metadata"}
    assert "extras.fascat" in runtime_dependencies["expected_runtime_support"]
    assert runtime_dependencies["not_written"]["KHR_draco_mesh_compression"].startswith("unsupported")
    assert runtime_dependencies["not_written"]["KHR_texture_basisu"].startswith("unsupported")
    unity_extensions = runtime_dependencies["runtime_compatibility"]["unity_gltfast"]["extensions"]
    assert unity_extensions["MSFT_lod"]["state"] == "optional"
    assert unity_extensions["EXT_meshopt_compression"]["state"] == "not_used"
    assert unity_extensions["extras.fascat"]["state"] == "metadata"
    assert runtime_dependencies["runtime_decision_matrix"]["geometry"]["meshopt"]["state"] == "available_not_requested"


def test_obj_export_writes_mesh_and_mtl_sidecar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "triangle.obj"

    _asset().write_obj(output, options=ObjExportOptions(materials=True, write_mtl=True, preserve_groups=True))
    text = output.read_text(encoding="utf-8")

    assert validate_obj(output) == {"meshes": 1, "points": 3, "triangles": 1}
    assert "usemtl mat" in text
    assert "vn 0 0 1" in text
    assert "s off" in text
    assert "f 1//1 2//1 3//1" in text
    assert (tmp_path / "triangle.mtl").exists()


def test_obj_export_writes_staged_vertex_normals_and_smoothing_groups(tmp_path) -> None:  # type: ignore[no-untyped-def]
    smooth_mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    ).compute_normals()
    hard_mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.asarray([[0, 1, 2], [0, 3, 1]], dtype=int),
    ).compute_hard_edge_normals(hard_edge_angle=30.0)
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="smooth", name="Smooth", part_id="smooth"),
                Node(id="hard", name="Hard", part_id="hard"),
            ],
        ),
        parts={
            "smooth": Part(id="smooth", name="Smooth", mesh=smooth_mesh),
            "hard": Part(id="hard", name="Hard", mesh=hard_mesh),
        },
    )
    output = tmp_path / "normals.obj"

    asset.write_obj(output, options=ObjExportOptions(materials=False, write_mtl=False))

    lines = output.read_text(encoding="utf-8").splitlines()
    smooth_group = lines.index("g Smooth")
    hard_group = lines.index("g Hard")
    assert "s 1" in lines[smooth_group:hard_group]
    assert "s off" in lines[hard_group:]
    assert sum(1 for line in lines if line.startswith("vn ")) == smooth_mesh.vertex_count + hard_mesh.vertex_count
    assert all("//" in line for line in lines if line.startswith("f "))


def test_mesh_only_exports_report_file_size_budget_warnings(tmp_path) -> None:  # type: ignore[no-untyped-def]
    obj_asset = _asset()
    stl_asset = _asset()

    obj_asset.write_obj(tmp_path / "budget.obj", options=ObjExportOptions(file_size_budget_mb=0.000001))
    stl_asset.write_stl(tmp_path / "budget.stl", options=StlExportOptions(file_size_budget_mb=0.000001))

    assert "file size budget exceeded" in obj_asset.report.warnings[-1]
    assert "file size budget exceeded" in stl_asset.report.warnings[-1]


def test_stl_export_writes_binary_mesh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "triangle.stl"

    _asset().write_stl(output, options=StlExportOptions(binary=True))

    payload = output.read_bytes()
    assert struct.unpack_from("<I", payload, 80)[0] == 1
    assert validate_stl(output) == {"meshes": 1, "points": 3, "triangles": 1}


def test_stl_export_writes_ascii_mesh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "triangle_ascii.stl"

    _asset().write_stl(output, options=StlExportOptions(binary=False))

    text = output.read_text(encoding="utf-8")
    assert text.startswith("solid fascat\n")
    assert "facet normal 0 0 1" in text
    assert "vertex 1 0 0" in text
    assert text.endswith("endsolid fascat\n")
    assert validate_stl(output) == {"meshes": 1, "points": 3, "triangles": 1}


def test_cli_convert_accepts_runtime_export_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.obj",
            "--quantize",
            "--meshopt",
            "--file-size-budget-mb",
            "50",
            "--texture-fallback-format",
            "png",
            "--png-compression",
            "9",
            "--jpeg-quality",
            "70",
            "--obj-materials",
            "--write-mtl",
            "--preserve-groups",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["output"] == "output.obj"
    assert payload["quantize"] is True
    assert payload["meshopt"] is True
    assert payload["file_size_budget_mb"] == 50
    assert payload["texture_fallback_format"] == "png"
    assert payload["png_compression"] == 9
    assert payload["jpeg_quality"] == 70


def test_cli_convert_rejects_unsupported_draco_option_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--draco",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert "draco" in payload["error"]


def test_cli_convert_rejects_unsupported_texture_compression_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--texture-compression",
            "ktx2",
        ],
    )

    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert "texture-compression" in payload["error"]


def test_cli_validate_writes_geometry_quality_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "triangle.gltf"
    report_path = tmp_path / "quality.json"
    _asset().write_gltf(output)

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output),
            "--geometry-quality",
            "--report",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["analysis"]["summary"]["open_boundaries"] == 1
    assert payload["analysis"]["summary"]["draw_call_estimate"] == 1
    assert report["summary"]["boundary_edges"] == 3
    assert report["stats"]["validated_triangles"] == 1
