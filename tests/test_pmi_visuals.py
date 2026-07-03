from __future__ import annotations

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.metadata import PmiAnnotation
from fascat.pmi_visuals import build_pmi_visual_markers


def test_build_pmi_visual_markers_targets_current_part_ids() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Node", part_id="current")]),
        parts={
            "current": Part(
                id="current",
                name="Current",
                mesh=mesh,
                metadata={"source_part_id": "source"},
            )
        },
        pmi=[PmiAnnotation(id="dim1", kind="dimension", text="12.5", applies_to=["source"])],
    )

    markers = build_pmi_visual_markers(asset)

    assert len(markers) == 1
    marker = markers[0]
    assert marker.annotation_id == "dim1"
    assert marker.current_part_ids == ("current",)
    assert marker.points.shape[1] == 3
    assert marker.faces.shape[1] == 3
    assert marker.text_glyph_count == 4
    assert marker.anchor != (0.0, 0.0, 0.0)
