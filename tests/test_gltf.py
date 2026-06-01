from __future__ import annotations

import base64
import io
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.image import ImageResource
from fascat.io.gltf import _apply_meshopt_compression, validate_gltf, write_gltf
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import BakeMaterialOptions, LODOptions


def _asset_with_materials_and_lods() -> Asset:
    mesh = Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        normals=np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        uvs={0: np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)},
        material_indices=np.array([0, 1], dtype=int),
    ).compute_tangents()
    lod_mesh = Mesh(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        metadata={"lod_ratio": "0.5", "lod_screen_coverage": "0.35"},
    )
    transform = np.eye(4, dtype=float)
    transform[0, 3] = 2.0
    return Asset(
        root=Node(
            id="root",
            name="Root",
            children=[Node(id="node", name="Occurrence", part_id="part", transform=transform)],
        ),
        parts={
            "part": Part(
                id="part",
                name="Panel",
                mesh=mesh,
                material_ids=["red", "blue"],
                lod_meshes=[lod_mesh],
            )
        },
        materials={
            "red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0)),
            "blue": Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 0.5), opacity=0.5),
        },
        units="metre",
        meters_per_unit=1.0,
        up_axis="Y",
    )


def test_meshopt_compression_reuses_mutable_binary_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    document: dict[str, Any] = {
        "buffers": [{"byteLength": 6}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 6, "target": 34963}],
        "accessors": [{"bufferView": 0, "count": 3, "componentType": 5123, "type": "SCALAR"}],
    }
    binary = bytearray(np.asarray([0, 1, 2], dtype="<u2").tobytes())

    fake_meshoptimizer = SimpleNamespace(
        encode_index_buffer=lambda indices, *, index_count: b"abc",
        encode_index_sequence=lambda indices, *, index_count: b"unused",
        encode_vertex_buffer=lambda vertices, *, vertex_count, vertex_size: b"unused",
    )
    monkeypatch.setitem(sys.modules, "meshoptimizer", fake_meshoptimizer)

    compressed = _apply_meshopt_compression(document, binary)

    assert compressed is binary
    assert len(compressed) > 6
    assert document["buffers"][0]["byteLength"] == len(compressed)
    assert document["bufferViews"][0]["extensions"]["EXT_meshopt_compression"]["byteOffset"] == 8


def _read_glb(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(payload[20 : 20 + json_length].decode("utf-8").rstrip(" \x00"))
    bin_offset = 20 + json_length
    bin_length, bin_type = struct.unpack_from("<II", payload, bin_offset)
    assert bin_type == 0x004E4942
    return document, payload[bin_offset + 8 : bin_offset + 8 + bin_length]


def _accessor_array(document: dict[str, Any], binary: bytes, accessor_index: int) -> np.ndarray:
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
    dtypes = {5126: np.float32}
    accessor = document["accessors"][accessor_index]
    buffer_view = document["bufferViews"][accessor["bufferView"]]
    width = widths[accessor["type"]]
    dtype = dtypes[accessor["componentType"]]
    offset = buffer_view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"] * width
    return np.frombuffer(binary, dtype=dtype, count=count, offset=offset).reshape((accessor["count"], width)).copy()


def test_glb_export_writes_valid_scene_materials_uvs_and_lod_metadata(tmp_path: Path) -> None:
    output = tmp_path / "panel.glb"

    write_gltf(_asset_with_materials_and_lods(), output)

    document, binary = _read_glb(output)
    stats = validate_gltf(output)
    occurrence = next(node for node in document["nodes"] if node.get("mesh") == 0)

    assert stats == {"meshes": 1, "points": 4, "triangles": 2}
    assert document["asset"]["version"] == "2.0"
    assert document["extras"]["fascat"]["exportUnits"] == "metre"
    assert len(document["meshes"]) == 2
    assert len(document["meshes"][0]["primitives"]) == 2
    assert document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] >= 0
    assert document["meshes"][0]["primitives"][0]["attributes"]["TANGENT"] >= 0
    assert document["materials"][0]["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 0.0, 0.0, 1.0]
    assert document["materials"][1]["pbrMetallicRoughness"]["baseColorFactor"] == [0.0, 0.0, 1.0, 0.5]
    assert document["materials"][1]["alphaMode"] == "BLEND"
    assert "_fascat_index" not in document["materials"][0]
    assert occurrence["matrix"][12:15] == [2.0, 0.0, 0.0]
    assert occurrence["extras"]["fascat"]["lodMeshIndices"] == [1]
    assert occurrence["extras"]["fascat"]["lods"] == [{"level": 1, "mesh": 1, "ratio": 0.5, "screenCoverage": 0.35}]
    assert "MSFT_lod" in document["extensionsUsed"]
    assert "MSFT_lod" not in document.get("extensionsRequired", [])
    assert occurrence["extras"]["MSFT_screencoverage"] == [0.35]
    lod_node_index = occurrence["extensions"]["MSFT_lod"]["ids"][0]
    lod_node = document["nodes"][lod_node_index]
    assert lod_node["mesh"] == 1
    assert lod_node["name"] == "Occurrence_lod1"
    assert lod_node["matrix"] == occurrence["matrix"]
    assert lod_node["extras"]["fascat"] == {"nodeId": "node_lod1", "sourceNodeId": "node", "lod": 1}
    assert len(binary) >= document["buffers"][0]["byteLength"]


def test_glb_export_writes_unreal_lods_as_separate_scene_nodes(tmp_path: Path) -> None:
    output = tmp_path / "unreal-lods.glb"
    asset = _asset_with_materials_and_lods().lods(
        LODOptions(ratios=(0.5,), screen_coverage=(0.35,), engine_profile="unreal")
    )

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    root = document["nodes"][document["scenes"][0]["nodes"][0]]
    occurrence_index = root["children"][0]
    separate_lod_index = root["children"][1]
    occurrence = document["nodes"][occurrence_index]
    separate_lod = document["nodes"][separate_lod_index]

    assert "MSFT_lod" not in occurrence.get("extensions", {})
    assert "MSFT_lod" not in document.get("extensionsUsed", [])
    assert occurrence["extras"]["fascat"]["lodExportMode"] == "separate"
    assert occurrence["extras"]["fascat"]["lodEngineProfile"] == "unreal"
    assert occurrence["extras"]["fascat"]["lods"][0]["engineProfile"] == "unreal"
    assert occurrence["extras"]["fascat"]["lods"][0]["exportMode"] == "separate"
    assert separate_lod["name"] == "Occurrence_LOD1"
    assert separate_lod["mesh"] == occurrence["extras"]["fascat"]["lodMeshIndices"][0]
    assert separate_lod["matrix"] == occurrence["matrix"]
    assert separate_lod["extras"]["fascat"]["lodExportMode"] == "separate"
    assert separate_lod["extras"]["fascat"]["lodEngineProfile"] == "unreal"


def test_glb_export_can_keep_lods_as_extras_only(tmp_path: Path) -> None:
    output = tmp_path / "extras-lods.glb"
    asset = _asset_with_materials_and_lods().lods(LODOptions(ratios=(0.5,), mode="extras"))

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    occurrence = next(node for node in document["nodes"] if node.get("mesh") == 0)

    assert occurrence["extras"]["fascat"]["lodExportMode"] == "extras"
    assert "MSFT_lod" not in occurrence.get("extensions", {})
    assert "MSFT_lod" not in document.get("extensionsUsed", [])


def test_glb_export_writes_embedded_baked_material_textures(tmp_path: Path) -> None:
    output = tmp_path / "baked.glb"
    asset = _asset_with_materials_and_lods().bake_materials(
        BakeMaterialOptions(
            bake=("base_color", "roughness", "metallic", "normal", "ao", "emissive"),
            force_uv_generation=False,
        )
    )

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    material = document["materials"][0]
    pbr = material["pbrMetallicRoughness"]

    assert len(document["images"]) == 5
    assert len(document["textures"]) == 5
    assert all(image["uri"].startswith("data:image/png;base64,") for image in document["images"])
    assert pbr["baseColorTexture"]["index"] == 0
    assert pbr["metallicRoughnessTexture"]["index"] == 1
    assert material["normalTexture"]["index"] == 2
    assert material["occlusionTexture"]["index"] == 3
    assert material["emissiveTexture"]["index"] == 4


def test_glb_export_uses_source_texture_image_bindings(tmp_path: Path) -> None:
    from PIL import Image

    output = tmp_path / "source-texture.glb"
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (200, 100, 50)).save(buffer, format="PNG")
    asset = _asset_with_materials_and_lods()
    image = ImageResource(
        id="panel_base",
        name="panel_baseColor.png",
        mime_type="image/png",
        data=buffer.getvalue(),
        width=2,
        height=2,
    )
    asset.images[image.id] = image
    asset.materials["red"].metadata["source_texture_base_color_image"] = image.id

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    material = document["materials"][0]
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0


def test_glb_export_marks_baked_opacity_textures_blend(tmp_path: Path) -> None:
    output = tmp_path / "opacity-map.glb"
    asset = _asset_with_materials_and_lods()
    asset.images["paint_alpha"] = ImageResource(
        id="paint_alpha",
        name="paint-alpha.png",
        mime_type="image/png",
        data=b"png",
        width=1,
        height=1,
    )
    asset.materials["red"] = Material(
        id="red",
        name="Red",
        base_color=(1.0, 0.0, 0.0, 0.6),
        opacity=0.6,
        metadata={
            "baked_maps": "base_color,opacity",
            "baked_texture_base_color_image": "paint_alpha",
        },
    )

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    material = document["materials"][0]

    assert material["alphaMode"] == "BLEND"
    assert material["pbrMetallicRoughness"]["baseColorFactor"] == [1.0, 0.0, 0.0, 0.6]
    assert material["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0
    assert document["images"][0]["uri"].startswith("data:image/png;base64,")


def test_glb_export_attaches_scene_far_proxy_as_root_lod(tmp_path: Path) -> None:
    output = tmp_path / "scene-proxy.glb"
    asset = _asset_with_materials_and_lods().lods(
        LODOptions(
            ratios=(0.5,),
            screen_coverage=(0.05,),
            scene_far_proxy=True,
        )
    )

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    root = document["nodes"][document["scenes"][0]["nodes"][0]]
    proxy_node_index = root["extensions"]["MSFT_lod"]["ids"][0]
    proxy_node = document["nodes"][proxy_node_index]
    assert root["extras"]["fascat"]["sceneFarProxyPartId"] == asset.metadata["lod_scene_far_proxy_part_id"]
    assert root["extras"]["MSFT_screencoverage"] == [0.05]
    assert proxy_node["extras"]["fascat"]["sceneFarProxy"] is True
    assert proxy_node["mesh"] == root["extras"]["fascat"]["sceneFarProxyMeshIndex"]
    assert "MSFT_lod" in document["extensionsUsed"]


def test_glb_export_attaches_unreal_scene_far_proxy_as_separate_node(tmp_path: Path) -> None:
    output = tmp_path / "scene-proxy-unreal.glb"
    asset = _asset_with_materials_and_lods().lods(
        LODOptions(
            ratios=(0.5,),
            screen_coverage=(0.05,),
            engine_profile="unreal",
            scene_far_proxy=True,
        )
    )

    write_gltf(asset, output)

    document, _binary = _read_glb(output)
    root = document["nodes"][document["scenes"][0]["nodes"][0]]
    scene_nodes = document["scenes"][0]["nodes"]
    proxy_node_index = root["extras"]["fascat"]["sceneFarProxyNodeIndex"]
    proxy_node = document["nodes"][proxy_node_index]
    assert proxy_node_index in scene_nodes
    assert "MSFT_lod" not in root.get("extensions", {})
    assert "MSFT_lod" not in document.get("extensionsUsed", [])
    assert root["extras"]["fascat"]["sceneFarProxyExportMode"] == "separate"
    assert root["extras"]["fascat"]["sceneFarProxyScreenCoverage"] == 0.05
    assert proxy_node["extras"]["fascat"]["sceneFarProxy"] is True


def test_glb_export_preserves_normals_and_tangent_handedness(tmp_path: Path) -> None:
    mesh = Mesh(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=float)},
    ).compute_normals()
    mesh = mesh.compute_tangents()
    asset = Asset(
        root=Node(id="root", name="Root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    )
    output = tmp_path / "mirrored.glb"

    write_gltf(asset, output)

    document, binary = _read_glb(output)
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    normals = _accessor_array(document, binary, attributes["NORMAL"])
    tangents = _accessor_array(document, binary, attributes["TANGENT"])

    assert np.allclose(normals, np.array([[0.0, 0.0, 1.0]] * 3, dtype=np.float32))
    assert np.allclose(np.linalg.norm(tangents[:, :3], axis=1), 1.0)
    assert np.all(tangents[:, 3] == -1.0)


def test_gltf_export_embeds_buffer_data_uri_and_validates(tmp_path: Path) -> None:
    output = tmp_path / "panel.gltf"

    write_gltf(_asset_with_materials_and_lods(), output)

    document = json.loads(output.read_text(encoding="utf-8"))
    uri = document["buffers"][0]["uri"]

    assert validate_gltf(output)["triangles"] == 2
    assert uri.startswith("data:application/octet-stream;base64,")
    assert len(base64.b64decode(uri.split(",", 1)[1])) == document["buffers"][0]["byteLength"]


def test_gltf_export_rejects_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported glTF extension"):
        write_gltf(_asset_with_materials_and_lods(), tmp_path / "panel.txt")


def test_gltf_validation_rejects_assets_without_scene_meshes(tmp_path: Path) -> None:
    output = tmp_path / "empty.gltf"
    output.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "scene": 0,
                "scenes": [{"nodes": [0]}],
                "nodes": [{"name": "empty"}],
                "buffers": [{"byteLength": 0, "uri": "data:application/octet-stream;base64,"}],
                "bufferViews": [],
                "accessors": [],
                "meshes": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="contains no meshes"):
        validate_gltf(output)
