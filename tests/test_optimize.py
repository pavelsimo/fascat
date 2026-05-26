from __future__ import annotations

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import OptimizeOptions


def test_optimize_can_duplicate_repeated_parts_per_occurrence() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part"),
            Node(id="node_b", name="B", part_id="part"),
        ],
    )
    asset = Asset(root=root, parts={"part": Part(id="part", name="Part", mesh=mesh)})

    optimized = asset.optimize(OptimizeOptions(simplify=False, optimize_buffers=False, preserve_instances=False))
    part_ids = [node.part_id for node in optimized.root.walk() if node.part_id is not None]

    assert optimized.part_count == 2
    assert len(set(part_ids)) == 2
    assert all(part_id is not None and part_id.startswith("part_") for part_id in part_ids)
    assert {part.metadata["source_part_id"] for part in optimized.parts.values()} == {"part"}
