from __future__ import annotations

import math

import numpy as np
import pytest

import fascat.ops._visibility as visibility
from fascat.mesh import Mesh
from fascat.ops._visibility import BoolArray, FloatArray, _RayMeshIndex, face_ambient_occlusion, ray_hits_mesh_batch


@pytest.fixture(autouse=True)
def force_bvh_for_equivalence_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(visibility, "_AO_BVH_MIN_TRIANGLES", 0)


def _separated_triangles(count: int) -> FloatArray:
    triangles = np.tile(np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (count, 1, 1))
    triangles[:, :, 1] += 2 * np.arange(count)[:, None]
    return triangles


@pytest.mark.parametrize("max_t", [0.0, 1e-8, 0.5, 1.0, 2.0, 20.0, np.inf])
def test_index_matches_brute_force_random_rays(max_t: float) -> None:
    rng = np.random.default_rng(873)
    triangles = rng.uniform(-3, 3, (160, 3, 3))
    triangles[0] = 0.0  # Degenerate face, retained with its original index.
    index = _RayMeshIndex(triangles)
    for origin in rng.uniform(-4, 4, (12, 3)):
        directions = np.concatenate([rng.normal(size=(24, 3)), np.eye(3), -np.eye(3), np.zeros((1, 3))])
        for ignore_face in (-1, 0, 73, 159):
            expected = ray_hits_mesh_batch(origin, directions, triangles, ignore_face=ignore_face, max_t=max_t)
            actual = index.ray_hits(origin, directions, ignore_face=ignore_face, max_t=max_t)
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e6])
@pytest.mark.parametrize("offset", [0.0, 1e8])
def test_index_matches_brute_force_edges_parallel_and_distance_limits(scale: float, offset: float) -> None:
    triangles = _separated_triangles(100) * scale + offset
    index = _RayMeshIndex(triangles)
    directions = np.asarray([[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, -0.0, 2], [1e-16, 0, 1]], dtype=float)
    for face in (0, 49, 99):
        for point in (triangles[face, 0], triangles[face, :2].mean(axis=0), triangles[face].mean(axis=0)):
            origin = point - np.asarray([0, 0, scale])
            for max_t in (scale * 0.5, scale, np.nextafter(scale, np.inf), scale * 2):
                for ignore_face in (-1, face):
                    expected = ray_hits_mesh_batch(origin, directions, triangles, ignore_face=ignore_face, max_t=max_t)
                    actual = index.ray_hits(origin, directions, ignore_face=ignore_face, max_t=max_t)
                    np.testing.assert_array_equal(actual, expected)


def test_index_handles_empty_mesh_and_directions() -> None:
    origin = np.zeros(3)
    index = _RayMeshIndex(np.empty((0, 3, 3)))
    np.testing.assert_array_equal(index.ray_hits(origin, np.eye(3), ignore_face=-1, max_t=10), [False] * 3)
    index = _RayMeshIndex(_separated_triangles(1))
    assert index.ray_hits(origin, np.empty((0, 3)), ignore_face=-1, max_t=10).shape == (0,)


@pytest.mark.parametrize("strategy", ["conservative", "exterior", "advanced"])
def test_ambient_occlusion_matches_brute_force(strategy: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(91)
    triangles = rng.normal(size=(120, 3, 3))
    triangles[0] = 0.0
    mesh = Mesh(points=triangles.reshape(-1, 3), faces=np.arange(triangles.size // 3).reshape(-1, 3))
    actual = face_ambient_occlusion(mesh, strategy)

    class BruteForceIndex:
        def __init__(self, triangles: FloatArray) -> None:
            self.triangles = triangles

        def ray_hits(self, origin: FloatArray, directions: FloatArray, *, ignore_face: int, max_t: float) -> BoolArray:
            return ray_hits_mesh_batch(origin, directions, self.triangles, ignore_face=ignore_face, max_t=max_t)

    monkeypatch.setattr(visibility, "_RayMeshIndex", BruteForceIndex)
    np.testing.assert_array_equal(actual, face_ambient_occlusion(mesh, strategy))
    assert np.any(actual < 1.0)


def test_ambient_occlusion_builds_once_and_bounds_query_work(monkeypatch: pytest.MonkeyPatch) -> None:
    count = 1024
    triangles = _separated_triangles(count)
    # Half the faces see an occluder in the next layer; the upper layer sees sky.
    triangles[count // 2 :] = triangles[: count // 2] + [0, 0, 1]
    mesh = Mesh(points=triangles.reshape(-1, 3), faces=np.arange(count * 3).reshape(-1, 3))
    builds = 0
    triangle_tests = 0
    bound_tests = 0
    original_bounds = visibility._rays_intersect_bounds

    class CountingIndex(_RayMeshIndex):
        def __init__(self, triangles: FloatArray) -> None:
            nonlocal builds
            builds += 1
            super().__init__(triangles)

    def count_triangles(
        origin: FloatArray, directions: FloatArray, triangles: FloatArray, *, ignore_face: int, max_t: float
    ) -> BoolArray:
        nonlocal triangle_tests
        triangle_tests += len(directions) * len(triangles)
        assert len(triangles) <= visibility._AO_BVH_LEAF_SIZE
        return ray_hits_mesh_batch(origin, directions, triangles, ignore_face=ignore_face, max_t=max_t)

    def count_bounds(
        origin: FloatArray, directions: FloatArray, node: visibility._RayBvhNode, max_t: float
    ) -> BoolArray:
        nonlocal bound_tests
        bound_tests += len(directions)
        return original_bounds(origin, directions, node, max_t)

    monkeypatch.setattr(visibility, "_RayMeshIndex", CountingIndex)
    monkeypatch.setattr(visibility, "ray_hits_mesh_batch", count_triangles)
    monkeypatch.setattr(visibility, "_rays_intersect_bounds", count_bounds)
    values = face_ambient_occlusion(mesh)
    np.testing.assert_array_equal(values, np.repeat([0.0, 1.0], count // 2))
    assert builds == 1
    assert 0 < triangle_tests <= count * visibility._AO_BVH_LEAF_SIZE
    assert bound_tests <= count * (2 * math.ceil(math.log2(count)) + 1)


def test_index_stops_testing_hit_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    triangles = np.repeat(_separated_triangles(1), 1024, axis=0)
    index = _RayMeshIndex(triangles)
    triangle_tests = 0

    def count_triangles(
        origin: FloatArray, directions: FloatArray, triangles: FloatArray, *, ignore_face: int, max_t: float
    ) -> BoolArray:
        nonlocal triangle_tests
        triangle_tests += len(directions) * len(triangles)
        return ray_hits_mesh_batch(origin, directions, triangles, ignore_face=ignore_face, max_t=max_t)

    monkeypatch.setattr(visibility, "ray_hits_mesh_batch", count_triangles)
    hits = index.ray_hits(np.asarray([0.25, 0.25, -1.0]), np.asarray([[0.0, 0.0, 1.0]]), ignore_face=0, max_t=2)
    np.testing.assert_array_equal(hits, [True])
    assert triangle_tests <= visibility._AO_BVH_LEAF_SIZE


def test_small_mesh_uses_direct_predicate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(visibility, "_AO_BVH_MIN_TRIANGLES", 4096)
    triangles = _separated_triangles(100)
    index = _RayMeshIndex(triangles)
    assert index.root is None
    origin = np.array([0.25, 0.25, -1.0])
    directions = np.array([[0.0, 0.0, 1.0]])
    np.testing.assert_array_equal(
        index.ray_hits(origin, directions, ignore_face=-1, max_t=2),
        ray_hits_mesh_batch(origin, directions, triangles, ignore_face=-1, max_t=2),
    )
