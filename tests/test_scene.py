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


def test_optimize_scene_reconstructs_matching_separate_parts() -> None:
    mesh = _triangle()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="Bolt A", part_id="bolt_a"),
                Node(id="node_b", name="Bolt B", part_id="bolt_b"),
            ],
        ),
        parts={
            "bolt_a": Part(id="bolt_a", name="Bolt A", mesh=mesh.copy(), material_ids=["steel"]),
            "bolt_b": Part(id="bolt_b", name="Bolt B", mesh=mesh.copy(), material_ids=["steel"]),
        },
        materials={"steel": Material(id="steel", name="Steel", base_color=(0.7, 0.7, 0.7, 1.0))},
    )

    optimized = asset.optimize_scene(SceneOptimizeOptions(instance_policy="auto"))
    occurrence_part_ids = [node.part_id for node in optimized.root.walk() if node.part_id is not None]

    assert optimized.part_count == 1
    assert occurrence_part_ids == ["bolt_a", "bolt_a"]
    assert optimized.metadata["scene_reconstructed_part_count"] == "1"
    assert optimized.metadata["scene_reconstructed_occurrence_count"] == "1"
    assert optimized.metadata["scene_reconstructed_vertex_savings"] == "3"
    assert optimized.metadata["scene_reconstructed_triangle_savings"] == "1"
    assert optimized.metadata["scene_instanced_part_count"] == "1"


def test_optimize_scene_reports_instance_reconstruction_blockers() -> None:
    mesh = _triangle()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="Bolt A", part_id="bolt_a"),
                Node(id="node_b", name="Bolt B", part_id="bolt_b"),
                Node(id="node_c", name="Bolt C", part_id="bolt_c"),
            ],
        ),
        parts={
            "bolt_a": Part(id="bolt_a", name="Bolt A", mesh=mesh.copy(), material_ids=["steel"]),
            "bolt_b": Part(id="bolt_b", name="Bolt B", mesh=mesh.copy(), material_ids=["paint"]),
            "bolt_c": Part(
                id="bolt_c",
                name="Bolt C",
                mesh=mesh.copy(),
                material_ids=["steel"],
                metadata={"finish": "coated"},
            ),
        },
        materials={
            "steel": Material(id="steel", name="Steel", base_color=(0.7, 0.7, 0.7, 1.0)),
            "paint": Material(id="paint", name="Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )

    optimized = asset.optimize_scene(SceneOptimizeOptions(instance_policy="preserve"))

    assert optimized.part_count == 3
    assert optimized.metadata["scene_reconstructed_part_count"] == "0"
    assert any("material differences" in warning for warning in optimized.report.steps[-1].warnings)
    assert any("metadata differences" in warning for warning in optimized.report.steps[-1].warnings)


def test_optimize_scene_does_not_reconstruct_different_uv_payloads() -> None:
    mesh_a = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
    )
    mesh_b = Mesh(
        points=mesh_a.points,
        faces=mesh_a.faces,
        uvs={0: np.array([[0, 0], [0, 1], [1, 0]], dtype=float)},
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="Panel A", part_id="panel_a"),
                Node(id="node_b", name="Panel B", part_id="panel_b"),
            ],
        ),
        parts={
            "panel_a": Part(id="panel_a", name="Panel A", mesh=mesh_a, material_ids=["paint"]),
            "panel_b": Part(id="panel_b", name="Panel B", mesh=mesh_b, material_ids=["paint"]),
        },
        materials={"paint": Material(id="paint", name="Paint", base_color=(0.0, 0.0, 1.0, 1.0))},
    )

    optimized = asset.optimize_scene(SceneOptimizeOptions(instance_policy="auto"))

    assert optimized.part_count == 2
    assert optimized.metadata["scene_reconstructed_part_count"] == "0"
    assert any("vertex attribute differences" in warning for warning in optimized.report.steps[-1].warnings)


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
