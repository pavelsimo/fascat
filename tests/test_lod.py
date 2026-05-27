from __future__ import annotations

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import LODOptions


def test_lods_are_monotonic() -> None:
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    mesh = Mesh(points=points, faces=faces)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            (0.75, 0.5, 0.25),
            screen_coverage=(0.6, 0.3, 0.1),
            per_part_budget=True,
            validate=True,
        )
    )
    part = with_lods.parts["cube"]
    counts = [mesh.triangle_count, *[lod.triangle_count for lod in part.lod_meshes]]
    step = with_lods.report.steps[-1]

    assert counts == sorted(counts, reverse=True)
    assert part.metadata["lod_screen_coverage"] == "0.6,0.3,0.1"
    assert part.metadata["lod_per_part_budget"] == "true"
    assert part.lod_meshes[0].metadata["lod_screen_coverage"] == "0.6"
    assert step.before["lod_meshes"] == 0
    assert step.before["lod_triangles"] == 0
    assert step.after["lod_meshes"] == 3
    assert step.after["lod_vertices"] == sum(lod.vertex_count for lod in part.lod_meshes)
    assert step.after["lod_triangles"] == sum(counts[1:])
    assert part.metadata["lod_source_vertices"] == str(mesh.vertex_count)
    assert part.metadata["lod_source_triangles"] == str(mesh.triangle_count)
    assert part.metadata["lod_added_vertices"] == str(sum(lod.vertex_count for lod in part.lod_meshes))
    assert part.metadata["lod_added_triangles"] == str(sum(counts[1:]))
    assert part.metadata["lod_chain_triangles"] == str(sum(counts))
    assert part.metadata["lod_level_triangles"] == ",".join(str(count) for count in counts[1:])
    assert float(part.metadata["lod_triangle_multiplier"]) > 1.0
    assert int(part.metadata["lod_added_mesh_bytes"]) > 0
    assert with_lods.metadata["lod_source_triangles"] == str(mesh.triangle_count)
    assert with_lods.metadata["lod_added_triangles"] == str(sum(counts[1:]))
    assert with_lods.metadata["lod_chain_triangles"] == str(sum(counts))
    assert step.after["lod_source_triangles"] == mesh.triangle_count
    assert step.after["lod_added_triangles"] == sum(counts[1:])
    assert step.after["lod_chain_triangles"] == sum(counts)
    assert step.after["lod_added_mesh_bytes"] == int(with_lods.metadata["lod_added_mesh_bytes"])


def test_lods_can_omit_tiny_parts_at_lower_screen_coverage() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=int,
        ),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            ratios=(0.5, 0.25),
            screen_coverage=(1.0, 0.1),
            drop_tiny_parts=True,
            tiny_part_screen_size=1.0,
        )
    )

    assert with_lods.parts["cube"].lod_meshes[0].triangle_count > 0
    assert with_lods.parts["cube"].lod_meshes[1].triangle_count == 0
    assert with_lods.parts["cube"].lod_meshes[1].metadata["lod_omitted"] == "tiny_part"
    assert with_lods.parts["cube"].metadata["lod_omitted_tiny_part_meshes"] == "1"
    assert with_lods.metadata["lod_omitted_tiny_part_meshes"] == "1"


def test_lods_warn_when_selected_parts_have_no_mesh() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="mesh_node", name="Mesh", part_id="mesh"),
                Node(id="empty_node", name="Untessellated", part_id="empty"),
            ],
        ),
        parts={
            "mesh": Part(id="mesh", name="Mesh", mesh=mesh),
            "empty": Part(id="empty", name="Untessellated", mesh=None),
        },
    )

    with_lods = asset.lods(LODOptions((0.5,)))
    warnings = with_lods.report.steps[-1].warnings

    assert len(with_lods.parts["mesh"].lod_meshes) == 1
    assert with_lods.parts["empty"].lod_meshes == []
    assert with_lods.parts["empty"].metadata["lod_status"] == "skipped_no_mesh"
    assert with_lods.metadata["lod_generated_parts"] == "1"
    assert with_lods.metadata["lod_skipped_no_mesh_parts"] == "1"
    assert len(warnings) == 1
    assert "LOD generation skipped part without tessellated mesh: Untessellated" in warnings[0]


def test_lods_warn_when_no_tessellated_parts_match() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="empty_node", name="Untessellated", part_id="empty")]),
        parts={"empty": Part(id="empty", name="Untessellated", mesh=None)},
    )

    with_lods = asset.lods(LODOptions((0.5,)))
    warnings = with_lods.report.steps[-1].warnings

    assert with_lods.parts["empty"].lod_meshes == []
    assert with_lods.parts["empty"].metadata["lod_status"] == "skipped_no_mesh"
    assert with_lods.metadata["lod_generated_parts"] == "0"
    assert with_lods.metadata["lod_skipped_no_mesh_parts"] == "1"
    assert len(warnings) == 2
    assert "LOD generation skipped part without tessellated mesh: Untessellated" in warnings[0]
    assert warnings[1] == "LOD generation matched no tessellated mesh-bearing parts"
