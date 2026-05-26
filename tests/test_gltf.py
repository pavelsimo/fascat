from __future__ import annotations

import base64
import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.io.gltf import validate_gltf, write_gltf
from fascat.material import Material
from fascat.mesh import Mesh


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
    assert document["materials"][1]["alphaMode"] == "BLEND"
    assert "_fascat_index" not in document["materials"][0]
    assert occurrence["matrix"][12:15] == [2.0, 0.0, 0.0]
    assert occurrence["extras"]["fascat"]["lodMeshIndices"] == [1]
    assert occurrence["extras"]["fascat"]["lods"] == [{"level": 1, "mesh": 1}]
    assert len(binary) >= document["buffers"][0]["byteLength"]


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
