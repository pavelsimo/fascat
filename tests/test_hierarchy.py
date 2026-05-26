from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import MergeOptions

runner = CliRunner()


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def _triangle(material_index: int = 0) -> Mesh:
    return Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        material_indices=np.array([material_index], dtype=int),
    )


def _asset() -> Asset:
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(
                    id="fasteners",
                    name="Fasteners",
                    children=[
                        Node(id="bolt_a", name="Bolt A", part_id="bolt", transform=_translation(0.0)),
                        Node(id="bolt_b", name="Bolt B", part_id="bolt", transform=_translation(2.0)),
                    ],
                ),
                Node(id="housing_node", name="Housing", part_id="housing", transform=_translation(10.0)),
            ],
        ),
        parts={
            "bolt": Part(
                id="bolt",
                name="Bolt",
                mesh=_triangle(),
                material_ids=["steel"],
                metadata={"kind": "fastener"},
            ),
            "housing": Part(
                id="housing",
                name="Housing",
                mesh=_triangle(),
                material_ids=["paint"],
                metadata={"kind": "casting"},
            ),
        },
        materials={
            "steel": Material(id="steel", name="Steel", base_color=(0.7, 0.7, 0.7, 1.0)),
            "paint": Material(id="paint", name="Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )


def test_merge_selected_geometry_bakes_transforms_and_keeps_parent() -> None:
    merged = _asset().merge(
        MergeOptions(mode="all", keep_parent=True, metadata="combine"),
        where=Filter.path("root/Fasteners/*"),
    )
    fasteners = next(node for node in merged.root.walk() if node.id == "fasteners")
    merged_nodes = [node for node in fasteners.children if node.part_id is not None]

    assert [node.name for node in fasteners.children] == ["Merged Geometry"]
    assert len(merged_nodes) == 1
    assert merged.part_count == 2
    assert merged.occurrence_count == 2
    assert merged.triangle_count == 3
    assert merged.draw_call_count == 2
    assert "bolt" not in merged.parts
    assert "housing" in merged.parts

    merged_part = merged.parts[merged_nodes[0].part_id or ""]
    merged_mesh = merged_part.mesh

    assert merged_mesh is not None
    assert merged_mesh.triangle_count == 2
    assert merged_mesh.bounds()[1][0] == 3.0
    assert merged_part.material_ids == ["steel"]
    assert merged_part.metadata["source_part_ids"] == "bolt"
    assert merged_part.metadata["source_node_ids"] == "bolt_a,bolt_b"
    assert merged.report.steps[-1].name == "merge"
    assert merged.report.steps[-1].before["draw_calls"] == 3
    assert merged.report.steps[-1].after["draw_calls"] == 2
    assert merged.report.steps[-1].options["matched"]["occurrences"] == 2


def test_merge_by_material_creates_one_part_per_material() -> None:
    asset = _asset()
    asset.parts["nut"] = Part(id="nut", name="Nut", mesh=_triangle(), material_ids=["paint"])
    fasteners = next(node for node in asset.root.walk() if node.id == "fasteners")
    fasteners.children.append(Node(id="nut_a", name="Nut A", part_id="nut", transform=_translation(4.0)))

    merged = asset.merge(MergeOptions(mode="by_material"), where=Filter.path("root/Fasteners/*"))
    merged_parts = [part for part in merged.parts.values() if part.id.startswith("merged_")]

    assert len(merged_parts) == 2
    assert sorted(part.material_ids for part in merged_parts) == [["paint"], ["steel"]]
    assert all(part.mesh is not None and part.mesh.triangle_count >= 1 for part in merged_parts)


def test_merge_respects_max_vertices_per_mesh() -> None:
    merged = _asset().merge(
        MergeOptions(mode="all", max_vertices_per_mesh=3),
        where=Filter.path("root/Fasteners/*"),
    )
    merged_parts = [part for part in merged.parts.values() if part.id.startswith("merged_")]

    assert len(merged_parts) == 2
    assert all(part.mesh is not None and part.mesh.vertex_count == 3 for part in merged_parts)


def test_cli_convert_accepts_merge_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--filter",
            "path=*/Fasteners/*",
            "--merge",
            "--merge-mode",
            "by-material",
            "--merge-metadata",
            "combine",
            "--max-vertices-per-mesh",
            "65535",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["merge"] is True
    assert payload["merge_mode"] == "by-material"
    assert payload["merge_metadata"] == "combine"


def test_cli_convert_requires_region_size_for_region_merge() -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "convert", "input.step", "output.usdc", "--merge", "--merge-mode", "regions"],
    )

    assert result.exit_code == 2
    assert "--merge-mode regions requires --region-size" in result.output
