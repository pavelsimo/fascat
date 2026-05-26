from __future__ import annotations

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import Tessellation


def test_tessellate_deduplicates_parts_by_mesh_fingerprint() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    fingerprint = mesh.fingerprint()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(id="part_a", name="Part A", mesh=mesh, fingerprint=fingerprint),
            "part_b": Part(id="part_b", name="Part B", mesh=mesh.copy(), fingerprint=fingerprint),
        },
    )

    tessellated = asset.tessellate(Tessellation())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert tessellated.part_count == 1
    assert part_ids == ["part_a", "part_a"]
