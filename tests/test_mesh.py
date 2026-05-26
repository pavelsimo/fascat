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


def test_subdivide_long_edges_enforces_limit_and_preserves_materials() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [4, 0, 0], [0, 3, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        material_indices=np.array([1], dtype=int),
    )

    subdivided = mesh.subdivide_long_edges(1.0)
    edge_lengths = []
    for face in subdivided.faces:
        corners = subdivided.points[face]
        edge_lengths.extend(
            [
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[1]),
                np.linalg.norm(corners[0] - corners[2]),
            ]
        )

    assert max(edge_lengths) <= 1.0
    assert subdivided.triangle_count > mesh.triangle_count
    assert subdivided.material_indices is not None
    assert set(subdivided.material_indices.tolist()) == {1}


def test_optimize_buffers_preserves_uvs_and_material_indices() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)},
        material_indices=np.array([0, 1], dtype=int),
    )

    optimized = mesh.optimize_buffers()

    optimized.validate()
    assert optimized.uvs[0].shape == (optimized.vertex_count, 2)
    assert optimized.material_indices is not None
    assert sorted(optimized.material_indices.tolist()) == [0, 1]


def test_fill_holes_is_limited_to_small_non_planar_boundaries() -> None:
    open_sheet = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    open_tetrahedron = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3]], dtype=int),
    )

    assert open_sheet.fill_holes().triangle_count == open_sheet.triangle_count
    assert open_tetrahedron.fill_holes().triangle_count == 4


@pytest.mark.requires_xatlas
def test_unwrap_uv_uses_xatlas_backend() -> None:
    pytest.importorskip("xatlas")
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    unwrapped = mesh.unwrap_uv(0)

    assert unwrapped.metadata["uv0"] == "xatlas"
    assert unwrapped.uvs[0].shape == (unwrapped.vertex_count, 2)
    assert np.isfinite(unwrapped.uvs[0]).all()
