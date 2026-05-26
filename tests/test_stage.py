from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import StageOptions


@pytest.mark.requires_xatlas
def test_stage_unwrap_uv_uses_xatlas_backend() -> None:
    pytest.importorskip("xatlas")
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    staged = asset.stage(StageOptions(uv0="unwrap", uv1="unwrap"))
    staged_mesh = staged.parts["part"].mesh

    assert staged_mesh is not None
    assert staged_mesh.metadata["uv0"] == "xatlas"
    assert staged_mesh.metadata["uv1"] == "xatlas"
    assert sorted(staged_mesh.uvs) == [0, 1]
