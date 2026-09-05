from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.image import ImageResource
from fascat.io.gltf import validate_gltf, write_gltf
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.metadata import PmiAnnotation, Tolerance
from fascat.options import GltfExportOptions, MetadataExportOptions, ReplaceOptions
from fascat.report import Report, ReportStep

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


def test_gltf_export_writes_pmi_visual_marker_geometry(tmp_path: Path) -> None:
    output = tmp_path / "metadata-visuals.gltf"

    write_gltf(
        _asset_with_metadata(),
        output,
        options=GltfExportOptions(metadata=MetadataExportOptions(mode="full", pmi="metadata_and_visuals")),
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    stats = validate_gltf(output)
    fascat = document["extras"]["fascat"]
    visual_group = next(node for node in document["nodes"] if node["name"] == "PMIVisuals")
    visual_node = document["nodes"][visual_group["children"][0]]
    visual_mesh = document["meshes"][visual_node["mesh"]]
    visual_primitive = visual_mesh["primitives"][0]
    visual_material = document["materials"][visual_primitive["material"]]

    assert fascat["pmiVisuals"]["count"] == 1
    assert visual_group["extras"]["fascat"]["pmiVisualCount"] == 1
    assert visual_node["extras"]["fascat"]["pmiId"] == "pmi_001"
    assert visual_node["extras"]["fascat"]["currentPartIds"] == ["part"]
    assert visual_node["extras"]["fascat"]["textGeometry"] == "block_glyphs"
    assert visual_node["extras"]["fascat"]["textGlyphCount"] > 0
    assert visual_mesh["extras"]["fascat"]["pmiVisual"] is True
    assert visual_mesh["extras"]["fascat"]["representation"] == "marker_geometry"
    assert visual_mesh["extras"]["fascat"]["textGeometry"] == "block_glyphs"
    assert visual_mesh["extras"]["fascat"]["textGlyphCount"] == visual_node["extras"]["fascat"]["textGlyphCount"]
    assert visual_primitive["mode"] == 4
    assert visual_material["doubleSided"] is True
    assert visual_material["extras"]["fascat"]["pmiVisualMaterial"] is True
    assert stats["meshes"] == 2
    assert stats["triangles"] > 1


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
    captured = {}

    def fake_read_cad(_path, _ctx, _payload, *, import_options=None):  # type: ignore[no-untyped-def]
        captured["options"] = import_options
        return _asset_with_metadata()

    monkeypatch.setattr("fascat.cli._io_helpers._read_cad_for_cli", fake_read_cad)

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
            "--design-variant",
            "left hand",
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
    assert captured["options"].design_variant_selection == ("left hand",)
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
    assert payload["design_variant_selection"] == ["left hand"]
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
            "--design-variant",
            "left hand",
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
    assert payload["design_variant_selection"] == ["left hand"]
    assert payload["import_existing_meshes"] is False
    assert payload["multi_file_import"] is True
    assert payload["delete_free_vertices"] is True
    assert payload["delete_lines"] is True
    assert payload["source_units"] == "millimetre"
    assert payload["target_units"] == "metre"
    assert payload["target_up_axis"] == "Y"


@pytest.mark.parametrize(
    "factory",
    [
        lambda metadata: Asset(root=Node(id="root", name="Root"), metadata=metadata),
        lambda metadata: Node(id="node", name="Node", metadata=metadata),
        lambda metadata: Part(id="part", name="Part", metadata=metadata),
        lambda metadata: Mesh(points=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]), metadata=metadata),
        lambda metadata: Material(id="mat", name="Material", base_color=(1, 1, 1, 1), metadata=metadata),
        lambda metadata: ImageResource(
            id="image", name="Image", mime_type="image/png", data=b"image", width=1, height=1, metadata=metadata
        ),
    ],
    ids=["asset", "node", "part", "mesh", "material", "image"],
)
def test_nested_metadata_is_owned_by_constructors_and_copies(factory: Callable[..., Any]) -> None:
    metadata = {"nested": {"values": [1]}}
    original = factory(metadata)
    metadata["nested"]["values"].append(2)
    assert original.metadata == {"nested": {"values": [1]}}

    copied = original.copy()
    copied.metadata["nested"]["values"].append(3)
    assert original.metadata == {"nested": {"values": [1]}}
    original.metadata["nested"]["values"].append(4)
    assert copied.metadata == {"nested": {"values": [1, 3]}}


@pytest.mark.parametrize("copy_method", ["copy", "clone"])
def test_asset_copies_nested_graph_metadata_pmi_and_report(copy_method: str) -> None:
    source = {"nested": {"values": [1]}}
    annotation = PmiAnnotation(
        id="pmi", kind="dimension", text="Size", applies_to=["part"], plane=[[1, 0, 0]], source=source
    )
    source["nested"]["values"].append(2)
    assert annotation.source == {"nested": {"values": [1]}}
    options = {"nested": {"values": [1]}}
    report = Report(steps=[ReportStep("import", options=options)])
    mesh = Mesh(points=np.zeros((3, 3)), faces=np.array([[0, 1, 2]]), metadata=annotation.source)
    asset = Asset(
        root=Node(id="root", name="Root", children=[Node(id="child", name="Child", metadata=annotation.source)]),
        parts={"part": Part(id="part", name="Part", mesh=mesh, lod_meshes=[mesh])},
        pmi=[annotation],
        report=report,
    )
    annotation.applies_to.append("other")
    assert annotation.plane is not None
    annotation.plane[0][0] = 9
    cast(dict[str, list[int]], annotation.source["nested"])["values"].append(3)
    cast(dict[str, list[int]], report.steps[0].options["nested"])["values"].append(3)
    assert asset.pmi[0].applies_to == ["part"]
    assert asset.pmi[0].plane == [[1, 0, 0]]
    assert asset.pmi[0].source == {"nested": {"values": [1]}}
    assert asset.report.steps[0].options == {"nested": {"values": [1]}}

    copied = getattr(asset, copy_method)()
    copied.pmi[0].applies_to.append("copy")
    copied.pmi[0].plane[0][0] = 7
    copied.pmi[0].source["nested"]["values"].append(7)
    copied.report.steps[0].options["nested"]["values"].append(7)
    copied.root.children[0].metadata["nested"]["values"].append(7)
    copied.parts["part"].mesh.metadata["nested"]["values"].append(7)
    copied.parts["part"].lod_meshes[0].metadata["nested"]["values"].append(7)
    assert asset.pmi[0].applies_to == ["part"]
    assert asset.pmi[0].plane == [[1, 0, 0]]
    assert asset.pmi[0].source == {"nested": {"values": [1]}}
    assert asset.report.steps[0].options == {"nested": {"values": [1]}}
    assert asset.root.children[0].metadata == {"nested": {"values": [1]}}
    assert asset.parts["part"].mesh is not None
    assert asset.parts["part"].mesh.metadata == {"nested": {"values": [1]}}
    assert asset.parts["part"].lod_meshes[0].metadata == {"nested": {"values": [1]}}


def test_copy_preserves_native_source_handle_identity() -> None:
    primitive = pytest.importorskip("OCP.BRepPrimAPI")
    shape = primitive.BRepPrimAPI_MakeBox(1, 1, 1).Shape()
    part = Part(id="part", name="Part", source_shape=shape, metadata={"nested": {"values": [1]}})
    asset = Asset(root=Node(id="root", name="Root"), parts={part.id: part})

    assert part.source_shape is shape
    assert part.copy().source_shape is shape
    assert part.copy(keep_source=False).source_shape is None
    assert asset.parts[part.id].source_shape is shape
    assert asset.copy().parts[part.id].source_shape is shape
    assert asset.clone().parts[part.id].source_shape is shape
    assert asset.copy(keep_source=False).parts[part.id].source_shape is None
