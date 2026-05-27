from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import ExplodeOptions, MergeOptions, ReplaceOptions

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


def _two_material_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )


def _box_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 3, 0],
                [0, 0, 4],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=int),
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


def test_explode_by_material_replaces_selected_occurrence_with_child_parts() -> None:
    asset = _asset()
    asset.parts["panel"] = Part(
        id="panel",
        name="Panel",
        mesh=_two_material_mesh(),
        material_ids=["red", "blue"],
        metadata={"kind": "panel"},
    )
    asset.materials["red"] = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))
    asset.materials["blue"] = Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0))
    asset.root.children.append(Node(id="panel_node", name="Panel", part_id="panel"))

    exploded = asset.explode(ExplodeOptions(mode="by_material"), where=Filter.part("panel"))
    panel_node = next(node for node in exploded.root.walk() if node.id == "panel_node")
    child_parts = [exploded.parts[child.part_id or ""] for child in panel_node.children]

    assert panel_node.part_id is None
    assert len(child_parts) == 2
    assert sorted(part.material_ids for part in child_parts) == [["blue"], ["red"]]
    assert all(part.mesh is not None and part.mesh.triangle_count == 1 for part in child_parts)
    assert "panel" not in exploded.parts
    assert exploded.report.steps[-1].name == "explode"
    assert exploded.report.steps[-1].after["parts"] == exploded.part_count


def test_explode_connected_components_splits_disconnected_faces() -> None:
    asset = _asset()
    asset.parts["loose"] = Part(
        id="loose",
        name="Loose",
        mesh=Mesh(
            points=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        ),
    )
    asset.root.children.append(Node(id="loose_node", name="Loose", part_id="loose"))

    exploded = asset.explode(ExplodeOptions(mode="connected_components"), where=Filter.part("loose"))
    loose_node = next(node for node in exploded.root.walk() if node.id == "loose_node")

    assert len(loose_node.children) == 2
    assert all(exploded.parts[child.part_id or ""].mesh.triangle_count == 1 for child in loose_node.children)


def test_replace_selected_part_with_bounding_box_proxy() -> None:
    asset = _asset()
    asset.parts["block"] = Part(
        id="block",
        name="Block",
        mesh=_box_mesh(),
        material_ids=["steel"],
        metadata={"kind": "block"},
    )
    asset.root.children.append(Node(id="block_node", name="Block", part_id="block", transform=_translation(5.0)))

    replaced = asset.replace(ReplaceOptions(mode="bounding_box"), where=Filter.part("block"))
    node = next(node for node in replaced.root.walk() if node.id == "block_node")
    part = replaced.parts[node.part_id or ""]

    assert node.transform[0, 3] == 5.0
    assert part.id != "block"
    assert part.mesh is not None
    assert part.mesh.vertex_count == 8
    assert part.mesh.triangle_count == 12
    assert part.metadata["replacement_mode"] == "bounding_box"
    assert part.metadata["source_part_ids"] == "block"
    assert "block" not in replaced.parts
    assert replaced.report.steps[-1].name == "replace"


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


def test_cli_convert_accepts_explode_and_replace_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--filter",
            "material=rubber",
            "--explode",
            "connected-components",
            "--replace",
            "bounding-box",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["explode"] == "connected-components"
    assert payload["replace"] == "bounding-box"


def test_cli_convert_requires_region_size_for_region_merge() -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "convert", "input.step", "output.usdc", "--merge", "--merge-mode", "regions"],
    )

    assert result.exit_code == 2
    assert "--merge-mode regions requires --region-size" in result.output
