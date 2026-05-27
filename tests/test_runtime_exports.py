from __future__ import annotations

import json
import struct

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
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
    )


def test_gltf_export_options_record_compression_intent_and_file_budget(tmp_path) -> None:  # type: ignore[no-untyped-def]
    asset = _asset()
    output = tmp_path / "triangle.gltf"

    asset.write_gltf(
        output,
        options=GltfExportOptions(
            quantize=True, meshopt=True, texture_compression="ktx2", file_size_budget_mb=0.000001
        ),
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["extras"]["fascat"]["compression"] == {
        "quantize": True,
        "meshopt": True,
        "textureCompression": "ktx2",
    }
    assert asset.report.steps[-1].after["file_size_bytes"] > 0
    assert asset.report.steps[-1].after["file_size_budget_bytes"] == 1
    assert "file size budget exceeded" in asset.report.warnings[-1]


def test_obj_export_writes_mesh_and_mtl_sidecar(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "triangle.obj"

    _asset().write_obj(output, options=ObjExportOptions(materials=True, write_mtl=True, preserve_groups=True))

    assert validate_obj(output) == {"meshes": 1, "points": 3, "triangles": 1}
    assert "usemtl mat" in output.read_text(encoding="utf-8")
    assert (tmp_path / "triangle.mtl").exists()


def test_stl_export_writes_binary_mesh(tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = tmp_path / "triangle.stl"

    _asset().write_stl(output, options=StlExportOptions(binary=True))

    payload = output.read_bytes()
    assert struct.unpack_from("<I", payload, 80)[0] == 1
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
