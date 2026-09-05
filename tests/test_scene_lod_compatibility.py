from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import SceneOptimizeOptions


def _mesh() -> Mesh:
    return Mesh(
        points=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]),
        faces=np.array([[0, 1, 2]]),
        normals=np.array([[0.0, 0, 1]] * 3),
        tangents=np.array([[1.0, 0, 0, 1]] * 3),
        uvs={0: np.array([[0.0, 0], [1, 0], [0, 1]])},
        material_indices=np.array([0]),
        face_groups={"surface": np.array([0])},
    )


def _asset(lods: list[Mesh]) -> Asset:
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="a", name="a", part_id="a"),
                Node(id="b", name="b", part_id="b"),
            ],
        ),
        parts={
            "a": Part(id="a", name="same", mesh=_mesh(), lod_meshes=[_mesh()]),
            "b": Part(id="b", name="same", mesh=_mesh(), lod_meshes=lods),
        },
    )


@pytest.mark.parametrize("tolerance", [0.0, 0.01])
@pytest.mark.parametrize(
    "difference", ["points", "faces", "normals", "tangents", "uvs", "materials", "groups", "metadata", "count"]
)
def test_instance_reconstruction_preserves_different_lod_chains(difference: str, tolerance: float) -> None:
    mesh = _mesh()
    lods = [mesh]
    if difference == "points":
        mesh.points[0, 0] += 0.02
    elif difference == "faces":
        mesh.faces = mesh.faces[:, [0, 2, 1]]
    elif difference == "normals":
        assert mesh.normals is not None
        mesh.normals *= -1
    elif difference == "tangents":
        assert mesh.tangents is not None
        mesh.tangents[:, 3] *= -1
    elif difference == "uvs":
        mesh.uvs[0] *= 2
    elif difference == "materials":
        mesh.material_indices = None
    elif difference == "groups":
        mesh.face_groups = {"other": np.array([0])}
    elif difference == "metadata":
        mesh.metadata["lod_screen_coverage"] = "0.1"
    else:
        lods = []
    asset = _asset(lods)

    result = asset.optimize_scene(SceneOptimizeOptions(instance_similarity_tolerance=tolerance))

    assert result.part_count == 2
    assert {node.part_id for node in result.root.children} == {"a", "b"}
    assert len(result.parts["b"].lod_meshes) == len(lods)
    assert any("LOD differences" in warning for warning in result.report.warnings)


@pytest.mark.parametrize("tolerance", [0.0, 0.01])
def test_instance_reconstruction_reuses_identical_lod_chains(tolerance: float) -> None:
    asset = _asset([_mesh()])

    result = asset.optimize_scene(SceneOptimizeOptions(instance_similarity_tolerance=tolerance))

    assert result.part_count == 1
    assert len(result.parts["a"].lod_meshes) == 1
    assert asset.part_count == 2
    assert not any("LOD differences" in warning for warning in result.report.warnings)


@pytest.mark.parametrize("base_delta", [0.0, 0.001])
def test_similarity_tolerance_applies_to_lod_positions(base_delta: float) -> None:
    asset = _asset([_mesh()])
    assert asset.parts["b"].mesh is not None
    asset.parts["b"].mesh.points[0, 0] += base_delta
    asset.parts["b"].lod_meshes[0].points[0, 0] += 0.001
    result = asset.optimize_scene(SceneOptimizeOptions(instance_similarity_tolerance=0.01))
    assert result.part_count == 1
    assert not any("LOD differences" in warning for warning in result.report.warnings)
    assert asset.part_count == 2
