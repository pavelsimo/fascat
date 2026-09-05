from itertools import combinations

import numpy as np
import pytest

from fascat import analysis
from fascat.mesh import Mesh


def _mesh_from_triangles(triangles: np.ndarray) -> Mesh:
    return Mesh(points=triangles.reshape(-1, 3), faces=np.arange(triangles.size // 3).reshape(-1, 3))


def _oracle(mesh: Mesh) -> tuple[set[tuple[int, int]], int]:
    triangles = mesh.points[mesh.faces]
    candidates = set()
    intersections = 0
    order = sorted(range(mesh.triangle_count), key=lambda index: (triangles[index, :, 0].min(), index))
    for left, right in combinations(order, 2):
        if set(mesh.faces[left]) & set(mesh.faces[right]):
            continue
        if any(
            triangles[left, :, axis].max() < triangles[right, :, axis].min()
            or triangles[right, :, axis].max() < triangles[left, :, axis].min()
            for axis in range(3)
        ):
            continue
        candidates.add((min(left, right), max(left, right)))
        intersections += analysis._triangles_intersect(triangles[left], triangles[right])
    return candidates, intersections


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("coplanar", [False, True])
def test_intersection_tree_matches_all_pairs_oracle(seed: int, coplanar: bool) -> None:
    rng = np.random.default_rng(seed)
    triangles = rng.uniform(-2.0, 2.0, size=(30, 3, 3))
    if coplanar:
        triangles[:, :, 2] = 0.0
    triangles[1] = triangles[0]  # Coincident geometry with separate indices counts.
    triangles[2] = triangles[2, 0]  # Degenerate geometry is handled by the exact test.
    mesh = _mesh_from_triangles(triangles)
    mesh.faces[3, 0] = mesh.faces[4, 1]  # A shared index excludes the pair.
    expected_pairs, expected_intersections = _oracle(mesh)
    geometry = mesh.points[mesh.faces]
    pairs = list(analysis._intersection_candidates(geometry.min(axis=1), geometry.max(axis=1), mesh.faces))
    normalized_pairs = {(min(left, right), max(left, right)) for left, right in pairs}

    assert normalized_pairs == expected_pairs
    assert len(pairs) == len(normalized_pairs)
    result = analysis._self_intersection_count(mesh, len(expected_pairs))
    assert result.intersections == expected_intersections
    assert result.pairs_checked == len(expected_pairs)
    assert result.truncated is False

    limited = analysis._self_intersection_count(mesh, 7)
    assert limited == analysis._self_intersection_count(mesh, 7)
    assert limited.pairs_checked == 7
    assert limited.truncated is True
    assert limited.intersections <= expected_intersections


@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("count", [2000, 4000, 8000])
def test_separated_triangles_have_linear_broad_phase_work(
    monkeypatch: pytest.MonkeyPatch, axis: int, count: int
) -> None:
    triangles = np.tile([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (count, 1, 1))
    triangles[:, :, axis] += 2 * np.arange(count)[:, None]
    mesh = _mesh_from_triangles(triangles)
    work = 0
    original = analysis._intersection_bounds_overlap

    def measure(left: analysis._IntersectionBoundsNode, right: analysis._IntersectionBoundsNode) -> bool:
        nonlocal work
        work += 1
        overlaps = original(left, right)
        if overlaps and left.indices is not None and right.indices is not None:
            assert left.size <= 8 and right.size <= 8
            work += left.size * right.size
        return overlaps

    monkeypatch.setattr(analysis, "_intersection_bounds_overlap", measure)
    result = analysis._self_intersection_count(mesh, 1)

    assert result.pairs_checked == 0
    assert result.intersections == 0
    assert result.truncated is False
    # Count node comparisons plus an upper bound on leaf box comparisons. This
    # catches quadratic broad-phase work without relying on machine timings.
    assert work <= 10 * count


def test_dense_candidates_stop_at_pair_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _mesh_from_triangles(np.tile([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], (4096, 1, 1)))
    visits = 0
    original = analysis._intersection_bounds_overlap

    def measure(left: analysis._IntersectionBoundsNode, right: analysis._IntersectionBoundsNode) -> bool:
        nonlocal visits
        visits += 1
        return original(left, right)

    monkeypatch.setattr(analysis, "_intersection_bounds_overlap", measure)
    result = analysis._self_intersection_count(mesh, 3)

    assert result.intersections == result.pairs_checked == result.pair_limit == 3
    assert result.truncated is True
    assert visits <= 16


@pytest.mark.parametrize("count", [0, 1, 2, 8, 9, 32])
def test_all_shared_indices_are_excluded_without_consuming_budget(count: int) -> None:
    mesh = Mesh(
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.tile([0, 1, 2], (count, 1)),
    )
    result = analysis._self_intersection_count(mesh, 1)

    assert result.intersections == result.pairs_checked == 0
    assert result.truncated is False


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_triangle_does_not_hide_other_intersections(invalid: float) -> None:
    triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangles = np.tile(triangle, (3, 1, 1))
    triangles[2, 0, 0] = invalid
    result = analysis._self_intersection_count(_mesh_from_triangles(triangles), 100)
    assert result.intersections == 1
    assert result.pairs_checked == 1
    assert not result.truncated
    triangles[:, 0, 0] = invalid
    result = analysis._self_intersection_count(_mesh_from_triangles(triangles), 100)
    assert result.pairs_checked == 0
