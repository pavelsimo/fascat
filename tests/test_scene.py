from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import SceneOptimizeOptions

runner = CliRunner()


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
                    id="group",
                    name="Group",
                    children=[
                        Node(id="bolt_a", name="Bolt A", part_id="bolt"),
                        Node(id="bolt_b", name="Bolt B", part_id="bolt"),
                    ],
                ),
                Node(id="housing", name="Housing", part_id="housing"),
            ],
        ),
        parts={
            "bolt": Part(id="bolt", name="Bolt", mesh=_triangle(), material_ids=["steel"]),
            "housing": Part(id="housing", name="Housing", mesh=_triangle(), material_ids=["paint"]),
        },
        materials={
            "steel": Material(id="steel", name="Steel", base_color=(0.7, 0.7, 0.7, 1.0)),
            "paint": Material(id="paint", name="Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )


def test_optimize_scene_batches_by_material_and_annotates_index_buffers() -> None:
    optimized = _asset().optimize_scene(
        SceneOptimizeOptions(batch_by_material=True, merge_compatible_meshes=True, index_buffer="auto")
    )

    assert optimized.draw_call_count == 2
    assert optimized.part_count == 2
    assert all(
        part.mesh is not None and part.mesh.metadata["index_buffer"] == "uint16" for part in optimized.parts.values()
    )
    assert optimized.report.steps[-1].name == "optimize_scene"
    assert optimized.report.steps[-1].before["draw_calls"] == 3
    assert optimized.report.steps[-1].after["draw_calls"] == 2


def test_optimize_scene_splits_large_merged_meshes() -> None:
    optimized = _asset().optimize_scene(
        SceneOptimizeOptions(merge_compatible_meshes=True, split_large_meshes=True, max_vertices_per_mesh=3)
    )

    assert optimized.part_count == 3
    assert all(part.mesh is not None and part.mesh.vertex_count <= 3 for part in optimized.parts.values())


def test_optimize_scene_splits_single_oversized_mesh() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh)},
    )

    optimized = asset.optimize_scene(SceneOptimizeOptions(split_large_meshes=True, max_vertices_per_mesh=3))

    assert optimized.part_count == 2
    assert optimized.root.children[0].part_id is None
    assert [child.part_id for child in optimized.root.children[0].children] == ["panel_split_1", "panel_split_2"]
    assert all(part.mesh is not None and part.mesh.vertex_count <= 3 for part in optimized.parts.values())


def test_optimize_scene_can_expand_instances() -> None:
    optimized = _asset().optimize_scene(SceneOptimizeOptions(instance_policy="expand"))
    occurrence_part_ids = [node.part_id for node in optimized.root.walk() if node.id in {"bolt_a", "bolt_b"}]

    assert occurrence_part_ids == ["bolt_bolt_a", "bolt_bolt_b"]
    assert optimized.part_count == 3
    assert optimized.metadata["scene_instanced_part_count"] == "0"


def test_cli_convert_accepts_scene_optimization_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--batch-by-material",
            "--merge-compatible-meshes",
            "--split-large-meshes",
            "--max-vertices-per-mesh",
            "65535",
            "--index-buffer",
            "auto",
            "--flatten",
            "safe",
            "--instance-policy",
            "auto",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["batch_by_material"] is True
    assert payload["merge_compatible_meshes"] is True
    assert payload["split_large_meshes"] is True
