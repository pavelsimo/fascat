from __future__ import annotations

import numpy as np
import pytest

from fascat.mesh import Mesh, MeshValidationError
from fascat.options import RepairOptions


def test_mesh_removes_unreferenced_vertices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    repaired = mesh.remove_unreferenced_vertices()

    assert repaired.vertex_count == 3
    assert repaired.triangle_count == 1


def test_mesh_removes_degenerate_and_duplicate_faces() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 2, 2]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 0], [0, 0, 1]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions())

    assert repaired.triangle_count == 1
    repaired.validate()


def test_mesh_merges_close_vertices_with_tolerance() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [0.001, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 2, 3], [1, 2, 3]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(tolerance=0.01))

    assert repaired.vertex_count == 3
    assert repaired.triangle_count == 1


def test_mesh_computes_normals_without_nan() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    with_normals = mesh.compute_normals()

    assert with_normals.normals is not None
    assert np.isfinite(with_normals.normals).all()


def test_mesh_validation_rejects_out_of_range_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    with pytest.raises(MeshValidationError, match="out-of-range"):
        mesh.validate()


def test_box_uv_matches_vertex_count() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [2, 0, 0], [0, 4, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    staged = mesh.box_uv()

    assert staged.uvs[0].shape == (3, 2)
    assert staged.uvs[0].min() >= 0.0
    assert staged.uvs[0].max() <= 1.0
