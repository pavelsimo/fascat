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

    with_lods = asset.lods(LODOptions((0.75, 0.5, 0.25)))
    part = with_lods.parts["cube"]
    counts = [mesh.triangle_count, *[lod.triangle_count for lod in part.lod_meshes]]
    step = with_lods.report.steps[-1]

    assert counts == sorted(counts, reverse=True)
    assert step.before["lod_meshes"] == 0
    assert step.before["lod_triangles"] == 0
    assert step.after["lod_meshes"] == 3
    assert step.after["lod_vertices"] == sum(lod.vertex_count for lod in part.lod_meshes)
    assert step.after["lod_triangles"] == sum(counts[1:])
