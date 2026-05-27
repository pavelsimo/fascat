from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.io.gltf import write_gltf
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.metadata import PmiAnnotation, Tolerance
from fascat.options import GltfExportOptions, MetadataExportOptions, ReplaceOptions

runner = CliRunner()


def _asset_with_metadata() -> Asset:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    return Asset(
        root=Node(
            id="root",
            name="Root",
            metadata={"assembly": "demo"},
            children=[Node(id="node", name="Panel Node", part_id="part", metadata={"step_label": "0:1"})],
        ),
        parts={
            "part": Part(
                id="part",
                name="Panel",
                mesh=mesh,
                material_ids=["mat"],
                metadata={"source_name": "Panel", "layer": "A"},
            )
        },
        materials={
            "mat": Material(
                id="mat",
                name="Paint",
                base_color=(0.1, 0.2, 0.3, 1.0),
                metadata={"finish": "matte"},
            )
        },
        metadata={"document": "demo.step", "author": "qa"},
        pmi=[
            PmiAnnotation(
                id="pmi_001",
                kind="dimension",
                text="25.4 +/-0.1",
                value=25.4,
                unit="millimetre",
                tolerance=Tolerance(upper=0.1, lower=0.0),
                applies_to=["part"],
                view="front",
                source={"step_label": "0:2"},
            )
        ],
    )


def test_gltf_export_writes_metadata_and_pmi_extras(tmp_path: Path) -> None:
    output = tmp_path / "metadata.gltf"

    write_gltf(_asset_with_metadata(), output)

    document = json.loads(output.read_text(encoding="utf-8"))
    fascat = document["extras"]["fascat"]
    mesh_extras = document["meshes"][0]["extras"]["fascat"]
    node_extras = next(node["extras"]["fascat"] for node in document["nodes"] if node["name"] == "Panel Node")

    assert fascat["metadata"]["document"] == "demo.step"
    assert fascat["pmi"][0]["id"] == "pmi_001"
    assert fascat["pmi"][0]["tolerance"]["upper"] == 0.1
    assert mesh_extras["metadata"]["layer"] == "A"
    assert mesh_extras["pmiIds"] == ["pmi_001"]
    assert node_extras["metadata"]["step_label"] == "0:1"
    assert document["materials"][0]["extras"]["fascat"]["metadata"]["finish"] == "matte"


def test_gltf_export_resolves_pmi_links_through_source_part_metadata(tmp_path: Path) -> None:
    asset = _asset_with_metadata().replace(ReplaceOptions(mode="bounding_box"), where=Filter.part("part"))
    output = tmp_path / "metadata-replaced.gltf"

    write_gltf(asset, output)

    document = json.loads(output.read_text(encoding="utf-8"))
    mesh_extras = document["meshes"][0]["extras"]["fascat"]

    assert mesh_extras["metadata"]["source_part_ids"] == "part"
    assert mesh_extras["pmiIds"] == ["pmi_001"]


def test_gltf_export_can_suppress_metadata_and_pmi(tmp_path: Path) -> None:
    output = tmp_path / "metadata-none.gltf"

    write_gltf(
        _asset_with_metadata(),
        output,
        options=GltfExportOptions(metadata=MetadataExportOptions(mode="none", pmi="none")),
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    fascat = document["extras"]["fascat"]
    mesh_extras = document["meshes"][0]["extras"]["fascat"]
    node_extras = next(node["extras"]["fascat"] for node in document["nodes"] if node["name"] == "Panel Node")

    assert "metadata" not in fascat
    assert "metadataSummary" not in fascat
    assert "pmi" not in fascat
    assert "metadata" not in mesh_extras
    assert "pmiIds" not in mesh_extras
    assert "metadata" not in node_extras
    assert "metadata" not in document["materials"][0]["extras"]["fascat"]


def test_asset_copy_preserves_top_level_metadata_and_pmi() -> None:
    asset = _asset_with_metadata()

    copied = asset.copy()
    copied.metadata["document"] = "copy.step"
    copied.pmi.append(PmiAnnotation(id="pmi_002", kind="note", text="copy"))

    assert asset.metadata["document"] == "demo.step"
    assert [annotation.id for annotation in asset.pmi] == ["pmi_001"]
    assert copied.metadata["document"] == "copy.step"
    assert [annotation.id for annotation in copied.pmi] == ["pmi_001", "pmi_002"]


def test_cli_inspect_can_emit_metadata_and_pmi(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.cli as cli

    captured = {}

    def fake_read_step(_path, _ctx, _payload, *, import_options=None):  # type: ignore[no-untyped-def]
        captured["options"] = import_options
        return _asset_with_metadata()

    monkeypatch.setattr(cli, "_read_step_for_cli", fake_read_step)

    result = runner.invoke(
        app,
        [
            "--json",
            "inspect",
            "input.step",
            "--metadata",
            "full",
            "--pmi",
            "full",
            "--design-variants",
            "--no-import-existing-meshes",
            "--multi-file-import",
            "--delete-free-vertices",
            "--delete-lines",
            "--source-units",
            "millimetre",
            "--source-up-axis",
            "Z",
            "--source-handedness",
            "right",
            "--target-units",
            "metre",
            "--target-up-axis",
            "Y",
            "--target-handedness",
            "right",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert captured["options"].metadata is True
    assert captured["options"].pmi is True
    assert captured["options"].design_variants is True
    assert captured["options"].existing_meshes is False
    assert captured["options"].multi_file is True
    assert captured["options"].delete_free_vertices is True
    assert captured["options"].delete_lines is True
    assert captured["options"].source_units == "millimetre"
    assert captured["options"].source_up_axis == "Z"
    assert captured["options"].source_handedness == "right"
    assert captured["options"].target_units == "metre"
    assert captured["options"].target_up_axis == "Y"
    assert captured["options"].target_handedness == "right"
    assert payload["design_variants"] is True
    assert payload["import_existing_meshes"] is False
    assert payload["multi_file_import"] is True
    assert payload["delete_free_vertices"] is True
    assert payload["delete_lines"] is True
    assert payload["source_units"] == "millimetre"
    assert payload["target_units"] == "metre"
    assert payload["target_up_axis"] == "Y"
    assert payload["metadata_summary"] == {"asset": 2, "nodes": 2, "parts": 2, "materials": 1}
    assert payload["asset_metadata"]["author"] == "qa"
    assert payload["pmi_summary"]["count"] == 1
    assert payload["pmi"][0]["applies_to"] == ["part"]


def test_cli_convert_accepts_metadata_and_pmi_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--metadata",
            "none",
            "--pmi",
            "none",
            "--design-variants",
            "--no-import-existing-meshes",
            "--multi-file-import",
            "--delete-free-vertices",
            "--delete-lines",
            "--source-units",
            "millimetre",
            "--target-units",
            "metre",
            "--target-up-axis",
            "Y",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["metadata"] == "none"
    assert payload["pmi"] == "none"
    assert payload["design_variants"] is True
    assert payload["import_existing_meshes"] is False
    assert payload["multi_file_import"] is True
    assert payload["delete_free_vertices"] is True
    assert payload["delete_lines"] is True
    assert payload["source_units"] == "millimetre"
    assert payload["target_units"] == "metre"
    assert payload["target_up_axis"] == "Y"
