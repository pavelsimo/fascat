from __future__ import annotations

import logging
import math
import sys
from collections.abc import Iterator
from types import SimpleNamespace

import numpy as np
import pytest

import fascat.mesh as mesh_module
from fascat.mesh import Mesh, MeshValidationError
from fascat.options import DeleteDegeneratePolygonsOptions, MergeVerticesOptions, RepairOptions


def valid_triangle(**overrides: object) -> Mesh:
    values = {
        "points": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        "faces": np.array([[0, 1, 2]], dtype=int),
    }
    values.update(overrides)
    return Mesh(**values)


def mobius_strip_mesh(segments: int = 6) -> Mesh:
    points: list[np.ndarray] = []
    radius = 2.0
    half_width = 0.25
    for index in range(segments):
        theta = 2.0 * np.pi * index / segments
        radial = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
        vertical = np.array([0.0, 0.0, 1.0], dtype=float)
        twist = (np.cos(theta * 0.5) * radial) + (np.sin(theta * 0.5) * vertical)
        center = radius * radial
        points.append(center - (half_width * twist))
        points.append(center + (half_width * twist))

    faces: list[list[int]] = []
    for index in range(segments):
        left = index * 2
        right = left + 1
        if index == segments - 1:
            next_left = 1
            next_right = 0
        else:
            next_left = (index + 1) * 2
            next_right = next_left + 1
        faces.append([left, next_left, right])
        faces.append([right, next_left, next_right])
    return Mesh(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=int))


def flipped_tetrahedron_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 3, 1],
                [0, 2, 3],
                [1, 3, 2],
            ],
            dtype=int,
        ),
    )


def _install_fake_ckdtree(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, float]]:
    calls: list[tuple[int, float]] = []

    class FakeCKDTree:
        def __init__(self, points: np.ndarray) -> None:
            self.points = np.asarray(points, dtype=float)

        def query_pairs(self, radius: float) -> set[tuple[int, int]]:
            calls.append((int(self.points.shape[0]), float(radius)))
            pairs: set[tuple[int, int]] = set()
            for left in range(self.points.shape[0]):
                for right in range(left + 1, self.points.shape[0]):
                    if float(np.linalg.norm(self.points[left] - self.points[right])) <= radius:
                        pairs.add((left, right))
            return pairs

    monkeypatch.setattr(mesh_module, "_load_scipy_ckdtree", lambda: FakeCKDTree)
    return calls


def test_mesh_copies_mutable_inputs_on_construction() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
    uv0 = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    material_indices = np.array([0], dtype=int)
    group = np.array([0], dtype=int)
    metadata = {"source": "cad"}

    mesh = Mesh(
        points=points,
        faces=faces,
        normals=normals,
        uvs={0: uv0},
        material_indices=material_indices,
        face_groups={"panel": group},
        metadata=metadata,
    )
    points[0, 0] = 9.0
    faces[0, 0] = 2
    normals[0, 2] = -1.0
    uv0[0, 0] = 9.0
    material_indices[0] = 3
    group[0] = 2
    metadata["source"] = "changed"

    assert mesh.points[0, 0] == 0.0
    assert mesh.faces.tolist() == [[0, 1, 2]]
    assert mesh.normals is not None
    assert mesh.normals[0, 2] == 1.0
    assert mesh.uvs[0][0, 0] == 0.0
    assert mesh.material_indices is not None
    assert mesh.material_indices.tolist() == [0]
    assert mesh.face_groups["panel"].tolist() == [0]
    assert mesh.metadata == {"source": "cad"}


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

    repaired = mesh.repair(RepairOptions(quality_report=True))

    assert repaired.triangle_count == 1
    assert repaired.metadata["repair_duplicate_polygons_before"] == "1"
    assert repaired.metadata["repair_duplicate_polygons_after"] == "0"
    assert repaired.metadata["repair_degenerate_triangles_before"] == "1"
    assert repaired.metadata["repair_degenerate_triangles_after"] == "0"


def test_repair_skips_heavy_quality_diagnostics_by_default() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 0, 0], [1, -1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 3, 4]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions())

    assert repaired.metadata["repair_quality_report"] == "disabled"
    assert "repair_t_junctions_before" not in repaired.metadata
    assert "repair_boundary_gaps_before" not in repaired.metadata


def test_merge_vertices_preserves_attribute_seams_by_default() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [3, 2, 1]], dtype=int),
        normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=float),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]], dtype=float)},
    )

    merged = mesh.merge_vertices(MergeVerticesOptions())

    assert merged.vertex_count == 4
    assert merged.normals is not None
    assert sorted(merged.uvs) == [0]
    assert merged.metadata["merge_vertices_removed"] == "0"
    assert merged.metadata["merge_vertices_quality_report"] == "disabled"
    assert merged.metadata["merge_vertices_near_duplicate_check"] == "skipped"
    assert merged.metadata["merge_vertices_tolerance_advisory"] == "not_evaluated"
    assert "merge_vertices_candidate_position_buckets" not in merged.metadata


def test_merge_vertices_quality_report_records_candidate_diagnostics() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [3, 2, 1]], dtype=int),
        normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=float),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]], dtype=float)},
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(quality_report=True))

    assert merged.vertex_count == 4
    assert merged.metadata["merge_vertices_quality_report"] == "enabled"
    assert merged.metadata["merge_vertices_near_duplicate_check"] == "checked"
    assert merged.metadata["merge_vertices_candidate_position_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_vertices"] == "1"
    assert merged.metadata["merge_vertices_candidate_exact_duplicate_buckets"] == "0"
    assert merged.metadata["merge_vertices_candidate_boundary_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_non_manifold_buckets"] == "0"
    assert merged.metadata["merge_vertices_candidate_hard_edge_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_t_junctions"] == "0"
    assert merged.metadata["merge_vertices_candidate_boundary_gaps"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_protection"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_normals"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_tangents"] == "0"
    assert merged.metadata["merge_vertices_skipped_by_uvs"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_material_boundaries"] == "0"


def test_merge_vertices_reports_tangent_and_material_boundary_protection() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [3, 2, 1]], dtype=int),
        normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float),
        tangents=np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, -1]], dtype=float),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [0, 0]], dtype=float)},
        material_indices=np.array([0, 1], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(quality_report=True))

    assert merged.vertex_count == 4
    assert merged.metadata["merge_vertices_removed"] == "0"
    assert merged.metadata["merge_vertices_candidate_position_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_vertices"] == "1"
    assert merged.metadata["merge_vertices_candidate_exact_duplicate_buckets"] == "0"
    assert merged.metadata["merge_vertices_candidate_boundary_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_non_manifold_buckets"] == "0"
    assert merged.metadata["merge_vertices_candidate_hard_edge_buckets"] == "0"
    assert merged.metadata["merge_vertices_candidate_t_junctions"] == "0"
    assert merged.metadata["merge_vertices_candidate_boundary_gaps"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_protection"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_normals"] == "0"
    assert merged.metadata["merge_vertices_skipped_by_tangents"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_uvs"] == "0"
    assert merged.metadata["merge_vertices_skipped_by_material_boundaries"] == "1"


def test_merge_vertices_classifies_non_manifold_candidate_buckets() -> None:
    mesh = Mesh(
        points=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, 0]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1], [0, 1, 4], [5, 2, 3]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(quality_report=True))

    assert merged.metadata["merge_vertices_candidate_position_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_vertices"] == "1"
    assert merged.metadata["merge_vertices_candidate_exact_duplicate_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_boundary_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_non_manifold_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_hard_edge_buckets"] == "0"
    assert merged.metadata["merge_vertices_candidate_t_junctions"] == "2"
    assert merged.metadata["merge_vertices_candidate_boundary_gaps"] == "1"
    assert merged.metadata["merge_vertices_tolerance_risk"] == "exact_only"


def test_merge_vertices_reports_t_junction_and_boundary_gap_candidates() -> None:
    t_junction = Mesh(
        points=np.array(
            [[0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 0, 0], [1, -1, 0]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 4]], dtype=int),
    ).merge_vertices(MergeVerticesOptions(quality_report=True))
    boundary_gap = Mesh(
        points=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1.005, 0, 0], [2, 0, 0], [1.005, 1, 0]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    ).merge_vertices(MergeVerticesOptions(tolerance=0.01, quality_report=True))

    assert t_junction.metadata["merge_vertices_candidate_t_junctions"] == "1"
    assert t_junction.metadata["merge_vertices_candidate_boundary_gaps"] == "0"
    assert boundary_gap.metadata["merge_vertices_candidate_t_junctions"] == "0"
    assert boundary_gap.metadata["merge_vertices_candidate_boundary_gaps"] == "1"


def test_merge_vertices_reports_tolerance_scale_risk() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.3, quality_report=True))

    assert merged.metadata["merge_vertices_bbox_diagonal"] == "1.41421356"
    assert merged.metadata["merge_vertices_min_edge_length"] == "1"
    assert merged.metadata["merge_vertices_tolerance_bbox_ratio"] == "0.212132034"
    assert merged.metadata["merge_vertices_tolerance_min_edge_ratio"] == "0.3"
    assert merged.metadata["merge_vertices_tolerance_risk"] == "high_relative_to_min_edge"
    assert merged.metadata["merge_vertices_near_duplicate_pairs"] == "0"
    assert merged.metadata["merge_vertices_nearest_near_duplicate_distance"] == "0"
    assert merged.metadata["merge_vertices_tolerance_advisory"] == "none"


def test_merge_vertices_reports_tolerance_too_small_for_near_duplicates() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.001, 0, 0],
                [1.001, 2, 0],
                [0.001, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.0001, quality_report=True))

    assert merged.vertex_count == 6
    assert merged.metadata["merge_vertices_removed"] == "0"
    assert merged.metadata["merge_vertices_near_duplicate_pairs"] == "1"
    assert merged.metadata["merge_vertices_nearest_near_duplicate_distance"] == "0.001"
    assert merged.metadata["merge_vertices_tolerance_advisory"] == "near_duplicates_unmerged"


def test_merge_vertices_uses_distance_tolerance_across_position_buckets() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0.048, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.147, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.1, quality_report=True))

    assert merged.vertex_count == 5
    assert merged.metadata["merge_vertices_removed"] == "1"
    assert merged.metadata["merge_vertices_candidate_position_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_vertices"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_protection"] == "0"


def test_merge_vertices_uses_ckdtree_close_pair_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_ckdtree(monkeypatch)
    mesh = Mesh(
        points=np.array(
            [
                [0.048, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.147, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.1))

    assert merged.vertex_count == 5
    assert calls == [(6, 0.1)]


def test_near_duplicate_stats_use_ckdtree_close_pair_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_ckdtree(monkeypatch)
    mesh = Mesh(
        points=np.array([[0, 0, 0], [0.001, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.empty((0, 3), dtype=int),
    )

    near_pairs, nearest = mesh._near_duplicate_unmerged_stats(tolerance=0.0001, diagonal=1.0, min_edge=1.0)

    assert near_pairs == 1
    assert nearest == pytest.approx(0.001)
    assert calls == [(4, 0.01)]


def test_merge_vertices_close_pair_search_falls_back_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mesh_module, "_load_scipy_ckdtree", lambda: None)
    mesh = Mesh(
        points=np.array(
            [
                [0.048, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.147, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.1, quality_report=True))

    assert merged.vertex_count == 5
    assert merged.metadata["merge_vertices_candidate_position_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_vertices"] == "1"


def test_merge_vertices_preserves_cross_bucket_attribute_seams() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0.048, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.147, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        normals=np.array(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 1, 0]],
            dtype=float,
        ),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.1, quality_report=True))

    assert merged.vertex_count == 6
    assert merged.metadata["merge_vertices_removed"] == "0"
    assert merged.metadata["merge_vertices_candidate_position_buckets"] == "1"
    assert merged.metadata["merge_vertices_candidate_hard_edge_buckets"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_protection"] == "1"
    assert merged.metadata["merge_vertices_skipped_by_normals"] == "1"


def test_merge_vertices_can_ignore_attributes_and_remove_degenerates() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=int),
        normals=np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0]], dtype=float),
        tangents=np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, -1]], dtype=float),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]], dtype=float)},
    )

    merged = mesh.merge_vertices(
        MergeVerticesOptions(preserve_normals=False, preserve_tangents=False, preserve_uvs=False)
    )

    assert merged.vertex_count == 3
    assert merged.triangle_count == 1
    assert merged.normals is None
    assert merged.tangents is None
    assert merged.uvs == {}
    assert merged.metadata["merge_vertices_removed"] == "1"
    assert merged.metadata["merge_vertices_degenerate_triangles_removed"] == "1"
    merged.validate()


def test_delete_degenerate_polygons_reports_noop_and_removed_counts() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [0, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 0, 1], [0, 1, 3], [0, 4, 1]], dtype=int),
    )

    cleaned = mesh.delete_degenerate_polygons(DeleteDegeneratePolygonsOptions(area_epsilon=1e-12))

    assert cleaned.triangle_count == 1
    assert cleaned.vertex_count == 3
    assert cleaned.metadata["delete_degenerate_polygons_before"] == "3"
    assert cleaned.metadata["delete_degenerate_polygons_after"] == "0"
    assert cleaned.metadata["delete_degenerate_polygons_removed"] == "3"
    assert cleaned.metadata["delete_degenerate_polygons_vertices_removed"] == "2"
    assert cleaned.metadata["delete_degenerate_polygons_delete_duplicates"] == "true"
    assert cleaned.metadata["delete_degenerate_polygons_removed_duplicate_polygons"] == "0"
    assert cleaned.metadata["delete_degenerate_polygons_removed_duplicate_vertices"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_removed_collapsed_edges"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_removed_near_flat_area"] == "1"

    noop = cleaned.delete_degenerate_polygons(DeleteDegeneratePolygonsOptions())

    assert noop.triangle_count == 1
    assert noop.metadata["delete_degenerate_polygons_before"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_duplicate_polygons"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_duplicate_vertices"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_collapsed_edges"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_near_flat_area"] == "0"


def test_delete_degenerate_polygons_removes_duplicate_polygons_by_default() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 0], [0, 1, 3]], dtype=int),
        material_indices=np.array([4, 9, 7], dtype=int),
        face_groups={"duplicate": np.array([0, 1], dtype=int), "kept": np.array([2], dtype=int)},
    )

    cleaned = mesh.delete_degenerate_polygons(DeleteDegeneratePolygonsOptions())

    assert cleaned.triangle_count == 2
    assert cleaned.metadata["delete_degenerate_polygons_duplicate_polygons_before"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_duplicate_polygons_after"] == "0"
    assert cleaned.metadata["delete_degenerate_polygons_removed_duplicate_polygons"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_removed"] == "1"
    assert cleaned.material_indices is not None
    assert cleaned.material_indices.tolist() == [4, 7]
    assert cleaned.face_groups["duplicate"].tolist() == [0]
    assert cleaned.face_groups["kept"].tolist() == [1]


def test_delete_degenerate_polygons_can_preserve_duplicate_polygons() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 0]], dtype=int),
    )

    cleaned = mesh.delete_degenerate_polygons(DeleteDegeneratePolygonsOptions(delete_duplicates=False))

    assert cleaned.triangle_count == 2
    assert cleaned.metadata["delete_degenerate_polygons_delete_duplicates"] == "false"
    assert cleaned.metadata["delete_degenerate_polygons_duplicate_polygons_before"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_duplicate_polygons_after"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_removed_duplicate_polygons"] == "0"


def test_quality_metrics_counts_duplicate_polygons() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 0], [0, 1, 3]], dtype=int),
    )

    metrics = mesh.quality_metrics()

    assert metrics["duplicate_polygons"] == 1


def test_repair_records_t_junction_counts() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, -1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 4]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(quality_report=True))

    assert mesh.t_junction_count() == 1
    assert repaired.metadata["repair_t_junctions_before"] == "1"
    assert repaired.metadata["repair_t_junctions_after"] == "1"


def _t_junction_count_bruteforce(mesh: Mesh, *, tolerance: float = 1e-9) -> int:
    """Original O(edges x vertices) implementation, kept as a reference oracle for the
    spatial-hashed t_junction_count (the two must return identical counts)."""
    if mesh.triangle_count == 0 or mesh.vertex_count < 3:
        return 0
    distance_tolerance = max(float(tolerance), 1e-12)
    edges, _counts = mesh._undirected_edges_and_counts()
    conflicts: set[tuple[int, int, int]] = set()
    for start_index, end_index in edges.astype(int).tolist():
        start = mesh.points[start_index]
        end = mesh.points[end_index]
        vector = end - start
        length_squared = float(np.dot(vector, vector))
        if length_squared <= distance_tolerance * distance_tolerance:
            continue
        minimum = np.minimum(start, end) - distance_tolerance
        maximum = np.maximum(start, end) + distance_tolerance
        candidates = np.flatnonzero(np.all((mesh.points >= minimum) & (mesh.points <= maximum), axis=1))
        if candidates.size == 0:
            continue
        candidate_points = mesh.points[candidates]
        projection = ((candidate_points - start) @ vector) / length_squared
        length = math.sqrt(length_squared)
        endpoint_margin = distance_tolerance / length
        interior = (projection > endpoint_margin) & (projection < 1.0 - endpoint_margin)
        if not np.any(interior):
            continue
        projected = start + (projection[:, None] * vector)
        distances = np.linalg.norm(candidate_points - projected, axis=1)
        on_edge = interior & (distances <= distance_tolerance)
        for candidate in candidates[on_edge].astype(int).tolist():
            if candidate in {start_index, end_index}:
                continue
            conflicts.add((min(start_index, end_index), max(start_index, end_index), candidate))
    return len(conflicts)


def _random_mesh(rng: np.random.Generator, *, vertices: int, faces: int, scale: float = 1.0) -> Mesh:
    points = rng.uniform(-scale, scale, size=(vertices, 3))
    face_indices = rng.integers(0, vertices, size=(faces, 3))
    return Mesh(points=points, faces=face_indices)


def _grid_plane_mesh(n: int) -> Mesh:
    xs, ys = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
    points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(n * n)])
    idx = np.arange(n * n).reshape(n, n)
    faces: list[list[int]] = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
            faces.append([int(a), int(b), int(c)])
            faces.append([int(b), int(d), int(c)])
    return Mesh(points=points, faces=np.asarray(faces, dtype=int))


def _cap_fallback_mesh() -> Mesh:
    # A dense unit-scale cluster (median edge ~1) plus one very long diagonal edge, so that
    # the long edge's cell box exceeds the per-edge budget and exercises the full-scan fallback.
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
            [400.0, 400.0, 400.0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [0, 4, 5]], dtype=int)
    return Mesh(points=points, faces=faces)


def test_t_junction_count_matches_bruteforce() -> None:
    rng = np.random.default_rng(20240529)
    meshes: list[Mesh] = [
        valid_triangle(),
        Mesh(
            points=np.array([[0, 0, 0], [2, 0, 0], [0, 1, 0], [1, 0, 0], [1, -1, 0]], dtype=float),
            faces=np.array([[0, 1, 2], [0, 3, 4]], dtype=int),
        ),
        Mesh(points=np.zeros((4, 3), dtype=float), faces=np.array([[0, 1, 2], [1, 2, 3]], dtype=int)),
        _grid_plane_mesh(6),
        _cap_fallback_mesh(),
        _random_mesh(rng, vertices=80, faces=120),
        _random_mesh(rng, vertices=200, faces=400, scale=3.0),
        _random_mesh(rng, vertices=12, faces=400, scale=2.0),
    ]
    for index, mesh in enumerate(meshes):
        for tolerance in (1e-9, 1e-6, 1e-3, 0.1, 0.5, 1.0):
            assert mesh.t_junction_count(tolerance=tolerance) == _t_junction_count_bruteforce(
                mesh, tolerance=tolerance
            ), f"mesh #{index} disagreed at tolerance {tolerance}"


def test_repair_records_boundary_gap_counts() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1.005, 0, 0],
                [2, 0, 0],
                [1.005, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(tolerance=0.01, merge_vertices=False, quality_report=True))

    assert mesh.boundary_gap_count(tolerance=0.01) == 1
    assert repaired.metadata["repair_boundary_gaps_before"] == "1"
    assert repaired.metadata["repair_boundary_gaps_after"] == "1"


def test_boundary_gap_count_uses_ckdtree_close_pair_search(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _install_fake_ckdtree(monkeypatch)
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1.005, 0, 0],
                [2, 0, 0],
                [1.005, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    assert mesh.boundary_gap_count(tolerance=0.01) == 1
    assert calls == [(6, 0.01)]


def test_orientability_metrics_early_exits_for_non_manifold_edges(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=int),
    )

    def fail_deque(*_: object, **__: object) -> None:
        raise AssertionError("orientability bfs should not run for non-manifold meshes")

    monkeypatch.setattr(mesh_module, "deque", fail_deque)

    assert mesh.orientability_metrics() == {
        "orientation_components": 0,
        "non_orientable_edges": 0,
        "closed_orientation_components": 0,
        "flipped_orientation_components": 0,
    }


def test_orientability_metrics_detect_mobius_like_strip() -> None:
    mesh = mobius_strip_mesh()

    metrics = mesh.orientability_metrics()
    repaired = mesh.repair(RepairOptions(quality_report=True))

    assert metrics["orientation_components"] == 1
    assert metrics["non_orientable_edges"] == 1
    assert repaired.metadata["repair_orientation_components_before_orientation"] == "1"
    assert repaired.metadata["repair_non_orientable_edges_before_orientation"] == "1"


def test_repair_records_flipped_closed_orientation_components() -> None:
    mesh = flipped_tetrahedron_mesh()

    metrics = mesh.orientability_metrics()
    repaired = mesh.repair(RepairOptions(quality_report=True))
    not_fixed = mesh.repair(RepairOptions(fix_winding=False, quality_report=True))

    assert metrics["closed_orientation_components"] == 1
    assert metrics["flipped_orientation_components"] == 1
    assert repaired.metadata["repair_closed_orientation_components_before_orientation"] == "1"
    assert repaired.metadata["repair_closed_orientation_components_after_orientation"] == "1"
    assert repaired.metadata["repair_flipped_components_before_orientation"] == "1"
    assert repaired.metadata["repair_flipped_components_after_orientation"] == "0"
    assert not_fixed.metadata["repair_flipped_components_after_orientation"] == "1"
    assert repaired.metadata["repair_face_orientation_strategy"] == "exterior"
    assert repaired.metadata["repair_face_orientation_status"] == "closed_exterior"
    assert repaired.metadata["repair_normal_orientation_strategy"] == "from_faces"
    assert repaired.metadata["repair_normal_orientation_status"] == "generated_from_faces"


def test_repair_can_record_trusted_source_orientation_policy() -> None:
    mesh = flipped_tetrahedron_mesh()

    repaired = mesh.repair(RepairOptions(face_orientation="source_trusted", quality_report=True))

    assert repaired.metadata["repair_flipped_components_before_orientation"] == "1"
    assert repaired.metadata["repair_flipped_components_after_orientation"] == "1"
    assert repaired.metadata["repair_face_orientation_strategy"] == "source_trusted"
    assert repaired.metadata["repair_face_orientation_status"] == "trusted_source"


def test_repair_records_viewer_standpoint_orientation_intent() -> None:
    mesh = flipped_tetrahedron_mesh()

    repaired = mesh.repair(
        RepairOptions(
            quality_report=True,
            face_orientation="viewer_standpoint",
            normal_orientation="viewer_standpoint",
            viewer_position=(0.0, 0.0, 10.0),
        )
    )

    assert repaired.metadata["repair_flipped_components_after_orientation"] == "0"
    assert repaired.metadata["repair_face_orientation_status"] == "viewer_oriented"
    assert repaired.metadata["repair_normal_orientation_status"] == "oriented_to_viewer"
    assert repaired.metadata["repair_orientation_viewer_position"] == "0,0,10"


def test_repair_single_sided_open_shell_orients_each_component_consistently() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 0, 0],
                [3, 0, 0],
                [2, 1, 0],
                [3, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 3, 2],
                [4, 5, 6],
                [4, 7, 6],
            ],
            dtype=int,
        ),
    )

    def shared_edge_directions(candidate: Mesh) -> dict[tuple[int, int], list[int]]:
        directions: dict[tuple[int, int], list[int]] = {}
        for face in candidate.faces.astype(int).tolist():
            for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                key = (min(start, end), max(start, end))
                direction = 1 if (start, end) == key else -1
                directions.setdefault(key, []).append(direction)
        return {edge: values for edge, values in directions.items() if len(values) == 2}

    before = shared_edge_directions(mesh)
    repaired = mesh.repair(RepairOptions(face_orientation="single_sided_open_shell", quality_report=True))
    after = shared_edge_directions(repaired)

    assert before[(0, 2)] == [-1, -1]
    assert before[(4, 6)] == [-1, -1]
    assert after
    assert all(sum(values) == 0 for values in after.values())
    assert repaired.metadata["repair_face_orientation_status"] == "open_shell_component_consistent"
    assert repaired.metadata["repair_orientation_components_before_orientation"] == "2"


def test_repair_can_sew_t_junctions() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, -1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 4]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(fix_t_junctions=True, quality_report=True))

    assert repaired.metadata["repair_t_junctions_before"] == "1"
    assert repaired.metadata["repair_t_junctions_after"] == "0"
    assert repaired.metadata["repair_t_junction_sewing"] == "sewn"
    assert repaired.triangle_count > mesh.triangle_count


def test_repair_can_stitch_boundary_gaps() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1.005, 0, 0],
                [2, 0, 0],
                [1.005, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    repaired = mesh.repair(
        RepairOptions(tolerance=0.01, merge_vertices=False, stitch_boundary_gaps=True, quality_report=True)
    )

    assert repaired.metadata["repair_boundary_gaps_before"] == "1"
    assert repaired.metadata["repair_boundary_gaps_after"] == "0"
    assert repaired.metadata["repair_boundary_gap_stitching"] == "stitched"


def _gap_mesh_with_attributes(*, gap_uvs: bool) -> Mesh:
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1.005, 0, 0], [2, 0, 0], [1.005, 1, 0]],
        dtype=float,
    )
    uvs = points[:, :2].copy()
    if gap_uvs:
        uvs[3] = [5.0, 5.0]
    else:
        uvs[3] = uvs[1]
    normals = np.tile([0.0, 0.0, 1.0], (6, 1))
    tangents = np.tile([1.0, 0.0, 0.0, 1.0], (6, 1))
    return Mesh(
        points=points,
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        normals=normals,
        tangents=tangents,
        uvs={0: uvs},
    )


def test_stitch_boundary_gaps_preserves_vertex_attributes() -> None:
    mesh = _gap_mesh_with_attributes(gap_uvs=False)

    stitched = mesh.stitch_boundary_gaps(0.01)

    assert stitched.vertex_count == 5
    assert stitched.normals is not None and stitched.normals.shape == (5, 3)
    assert stitched.tangents is not None and stitched.tangents.shape == (5, 4)
    assert 0 in stitched.uvs and stitched.uvs[0].shape == (5, 2)
    original_uv_rows = {tuple(row) for row in mesh.uvs[0].tolist()}
    assert all(tuple(row) in original_uv_rows for row in stitched.uvs[0].tolist())
    assert np.allclose(stitched.normals, [0.0, 0.0, 1.0])
    assert stitched.metadata["boundary_gap_stitching_attributes"] == "representative_vertex"
    assert "boundary_gap_stitching_uv_conflicts" not in stitched.metadata


def test_stitch_boundary_gaps_records_uv_seam_conflicts() -> None:
    mesh = _gap_mesh_with_attributes(gap_uvs=True)

    stitched = mesh.stitch_boundary_gaps(0.01)

    assert stitched.uvs[0].shape == (5, 2)
    assert stitched.metadata["boundary_gap_stitching_uv_conflicts"] == "1"


def test_stitch_boundary_gaps_keeps_order_sensitive_bucket_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_ckdtree_load() -> None:
        raise AssertionError("stitching keeps bucket order to preserve representative attributes")

    monkeypatch.setattr(mesh_module, "_load_scipy_ckdtree", fail_ckdtree_load)
    mesh = _gap_mesh_with_attributes(gap_uvs=True)

    stitched = mesh.stitch_boundary_gaps(0.01)

    assert stitched.vertex_count == 5
    assert stitched.metadata["boundary_gap_stitching_attributes"] == "representative_vertex"
    assert stitched.metadata["boundary_gap_stitching_uv_conflicts"] == "1"


def test_repair_can_crack_non_manifold_edges() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(crack_non_manifold_edges=True, quality_report=True))

    assert repaired.metadata["repair_non_manifold_edges_before_cracking"] == "1"
    assert repaired.metadata["repair_non_manifold_edges_after_cracking"] == "0"
    assert repaired.metadata["repair_non_manifold_edge_cracking"] == "cracked"
    assert repaired.vertex_count > mesh.vertex_count


def test_repair_can_remove_sliver_faces() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [10, 0, 0],
                [0, 0.01, 0],
                [0, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 1, 3]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(remove_sliver_faces=True, sliver_aspect_ratio=100.0, quality_report=True))

    assert repaired.triangle_count == 1
    assert repaired.metadata["repair_sliver_face_removal"] == "removed"
    assert repaired.metadata["repair_sliver_faces_removed"] == "1"


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


def test_mesh_uses_angle_weighted_normals_by_default() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [4, 0, 0],
                [0, 1, 0],
                [0, 0, 2],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=int),
    )

    angle_weighted = mesh.compute_normals()
    area_weighted = mesh.compute_normals(angle_weighted=False)

    assert angle_weighted.normals is not None
    assert area_weighted.normals is not None
    assert not np.allclose(angle_weighted.normals[0], area_weighted.normals[0])
    assert np.linalg.norm(angle_weighted.normals[0]) == pytest.approx(1.0)


def test_hard_edge_normals_split_vertices_across_sharp_edges() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=int),
    )

    hard = mesh.compute_hard_edge_normals(hard_edge_angle=30.0)

    assert hard.vertex_count > mesh.vertex_count
    assert hard.normals is not None
    assert np.isfinite(hard.normals).all()
    assert np.allclose(np.linalg.norm(hard.normals, axis=1), 1.0)


def test_hard_edge_normals_reuse_smooth_components_and_split_sharp_edges() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [1, 0, 1],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [1, 3, 2], [1, 4, 3]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [1, 1], [1, 0.5]], dtype=float)},
        material_indices=np.array([0, 0, 1], dtype=int),
    )

    hard = mesh.compute_hard_edge_normals(hard_edge_angle=30.0)

    assert hard.faces.tolist() == [[0, 1, 2], [1, 3, 2], [4, 5, 6]]
    assert hard.vertex_count == 7
    assert hard.uvs[0].tolist() == [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
    ]
    assert hard.material_indices is not None
    assert hard.material_indices.tolist() == [0, 0, 1]
    assert hard.normals is not None
    assert np.allclose(hard.normals[1], [0.0, 0.0, 1.0])
    assert np.allclose(hard.normals[4], [-1.0, 0.0, 0.0])


def test_tangents_are_generated_from_uv0_and_normals() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
    ).compute_normals()

    tangent_mesh = mesh.compute_tangents()

    assert tangent_mesh.tangents is not None
    assert tangent_mesh.tangents.shape == (3, 4)
    assert np.isfinite(tangent_mesh.tangents).all()
    assert np.allclose(np.linalg.norm(tangent_mesh.tangents[:, :3], axis=1), 1.0)


def test_tangent_handedness_tracks_mirrored_uvs() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [0, 1], [1, 0]], dtype=float)},
    ).compute_normals()

    tangent_mesh = mesh.compute_tangents()

    assert tangent_mesh.tangents is not None
    assert np.all(tangent_mesh.tangents[:, 3] == -1.0)


def test_hard_edge_normals_and_repair_preserve_face_material_assignments() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 3, 1]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
        face_groups={"bottom": np.array([0], dtype=int), "side": np.array([1], dtype=int)},
    )

    repaired = mesh.compute_hard_edge_normals(hard_edge_angle=30.0).repair()

    assert repaired.material_indices is not None
    assert repaired.material_indices.tolist() == [0, 1]
    assert repaired.face_groups["bottom"].tolist() == [0]
    assert repaired.face_groups["side"].tolist() == [1]


def test_mesh_validation_rejects_out_of_range_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    with pytest.raises(MeshValidationError, match="out-of-range"):
        mesh.validate()


@pytest.mark.parametrize(
    ("mesh", "message"),
    [
        (valid_triangle(points=np.array([0, 0, 0], dtype=float)), "points must have shape"),
        (valid_triangle(faces=np.array([0, 1, 2], dtype=int)), "faces must have shape"),
        (valid_triangle(points=np.array([[0, 0, 0], [np.nan, 0, 0], [0, 1, 0]], dtype=float)), "NaN or Inf"),
        (valid_triangle(faces=np.array([[-1, 1, 2]], dtype=int)), "negative vertex indices"),
        (valid_triangle(faces=np.array([[0, 1, 9]], dtype=int)), "out-of-range vertex indices"),
    ],
)
def test_mesh_validation_rejects_invalid_core_arrays(mesh: Mesh, message: str) -> None:
    with pytest.raises(MeshValidationError, match=message):
        mesh.validate()


@pytest.mark.parametrize(
    ("mesh", "message"),
    [
        (valid_triangle(normals=np.array([[0, 0, 1]], dtype=float)), "normals must match points shape"),
        (
            valid_triangle(normals=np.array([[0, 0, 1], [0, np.inf, 1], [0, 0, 1]], dtype=float)),
            "normals must not contain NaN or Inf values",
        ),
        (valid_triangle(tangents=np.array([[1, 0, 0, 1]], dtype=float)), "tangents must have shape"),
        (
            valid_triangle(tangents=np.array([[1, 0, 0, 1], [np.nan, 0, 0, 1], [1, 0, 0, 1]], dtype=float)),
            "tangents must not contain NaN or Inf values",
        ),
        (valid_triangle(uvs={0: np.array([0, 0], dtype=float)}), "uv channel 0 must have shape"),
        (valid_triangle(uvs={0: np.array([[0, 0], [1, 0]], dtype=float)}), "uv channel 0 must match"),
        (
            valid_triangle(uvs={0: np.array([[0, 0], [np.nan, 0], [0, 1]], dtype=float)}),
            "uv channel 0 must not contain NaN or Inf values",
        ),
        (valid_triangle(material_indices=np.array([0, 1], dtype=int)), "material_indices must match"),
        (valid_triangle(material_indices=np.array([-1], dtype=int)), "material_indices must not contain negative"),
    ],
)
def test_mesh_validation_rejects_invalid_attribute_arrays(mesh: Mesh, message: str) -> None:
    with pytest.raises(MeshValidationError, match=message):
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


def _mesh_with_uv_triangles(uv_triangles: list[list[list[float]]]) -> Mesh:
    uv = np.asarray(uv_triangles, dtype=float).reshape(-1, 2)
    points = np.column_stack((uv, np.zeros(uv.shape[0], dtype=float)))
    faces = np.arange(uv.shape[0], dtype=int).reshape(-1, 3)
    return Mesh(points=points, faces=faces, uvs={0: uv})


def test_uv_layout_stats_detects_overlap_bounds_and_degenerate_faces() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
                [2, 0, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5], [0, 1, 6]], dtype=int),
        uvs={
            0: np.array(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [2, 0],
                ],
                dtype=float,
            )
        },
    )

    stats = mesh.uv_layout_stats(0)

    assert stats["out_of_unit_vertices"] == 1
    assert stats["degenerate_faces"] == 1
    assert stats["overlapping_face_pairs"] == 1

    skipped = mesh.uv_layout_stats(0, detect_overlaps=False)

    assert skipped["out_of_unit_vertices"] == 1
    assert skipped["degenerate_faces"] == 1
    assert skipped["overlapping_face_pairs"] == 0


def test_uv_layout_stats_counts_multi_cell_overlap_once(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _mesh_with_uv_triangles(
        [
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[10.0, 10.0], [10.02, 10.0], [10.0, 10.02]],
            [[11.0, 10.0], [11.02, 10.0], [11.0, 10.02]],
            [[12.0, 10.0], [12.02, 10.0], [12.0, 10.02]],
            [[13.0, 10.0], [13.02, 10.0], [13.0, 10.02]],
            [[14.0, 10.0], [14.02, 10.0], [14.0, 10.02]],
        ]
    )
    original_overlap = mesh_module._triangle_overlap_area_2d
    overlap_calls = 0

    def count_overlap(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> float:
        nonlocal overlap_calls
        overlap_calls += 1
        return original_overlap(left, right, tolerance=tolerance)

    monkeypatch.setattr(mesh_module, "_triangle_overlap_area_2d", count_overlap)

    stats = mesh.uv_layout_stats(0)

    assert stats["overlapping_face_pairs"] == 1
    assert overlap_calls == 1


def test_uv_layout_stats_skips_clipper_for_packed_non_overlapping_triangles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uv_triangles = [[[0.02, y], [0.98, y], [0.02, y + 0.02]] for y in [0.02, 0.14, 0.26, 0.38, 0.5, 0.62]]
    mesh = _mesh_with_uv_triangles(uv_triangles)
    original_overlap = mesh_module._triangle_overlap_area_2d
    overlap_calls = 0

    def count_overlap(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> float:
        nonlocal overlap_calls
        overlap_calls += 1
        return original_overlap(left, right, tolerance=tolerance)

    monkeypatch.setattr(mesh_module, "_triangle_overlap_area_2d", count_overlap)

    stats = mesh.uv_layout_stats(0)

    assert stats["overlapping_face_pairs"] == 0
    assert overlap_calls == 0


def test_uv_layout_stats_cache_is_per_channel_and_returns_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    uv = np.array(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [0, 0],
            [1, 0],
            [0, 1],
        ],
        dtype=float,
    )
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [0, 1, 1],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        uvs={0: uv.copy(), 1: uv.copy()},
    )
    original_overlap = mesh_module._triangle_overlap_area_2d
    overlap_calls = 0

    def count_overlap(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> float:
        nonlocal overlap_calls
        overlap_calls += 1
        return original_overlap(left, right, tolerance=tolerance)

    monkeypatch.setattr(mesh_module, "_triangle_overlap_area_2d", count_overlap)

    first = mesh.uv_layout_stats(0)
    mesh.uv_layout_stats(1)
    third = mesh.uv_layout_stats(0)

    assert first == third
    assert overlap_calls == 2
    first["overlapping_face_pairs"] = 99
    assert mesh.uv_layout_stats(0)["overlapping_face_pairs"] == 1
    assert overlap_calls == 2

    mesh.uvs[0] = mesh.uvs[0].copy()
    mesh.uvs[0][0, 0] = 0.25
    mesh.uv_layout_stats(0)

    assert overlap_calls == 3


def test_uv_seam_graph_stats_cache_invalidation_and_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    seamed_uv = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.1, 0.2],
            [0.9, 0.2],
            [1.0, -1.0],
        ],
        dtype=float,
    )
    unseamed_uv = seamed_uv.copy()
    unseamed_uv[3] = [0.0, 0.0]
    unseamed_uv[4] = [1.0, 0.0]
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0.04, 0, 0],
                [0.96, 0, 0],
                [1, -1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        uvs={0: seamed_uv.copy(), 1: unseamed_uv.copy()},
    )
    original_face_edges = mesh_module._face_major_edges_from_faces
    edge_build_calls = 0

    def count_face_edges(faces: np.ndarray) -> np.ndarray:
        nonlocal edge_build_calls
        edge_build_calls += 1
        return original_face_edges(faces)

    monkeypatch.setattr(mesh_module, "_face_major_edges_from_faces", count_face_edges)

    exact_stats = mesh.uv_seam_graph_stats(0)
    first = mesh.uv_seam_graph_stats(0, tolerance=0.1)
    calls_after_first = edge_build_calls
    channel_one = mesh.uv_seam_graph_stats(1, tolerance=0.1)
    calls_after_second_channel = edge_build_calls
    third = mesh.uv_seam_graph_stats(0, tolerance=0.1)

    assert exact_stats["edges"] == 0
    assert first == third
    assert first["mesh_vertices"] == 4
    assert first["position_vertices"] == 2
    assert first["edges"] == 1
    assert first["components"] == 1
    assert first["total_length"] == pytest.approx(1.0)
    assert first["longest_component_length"] == pytest.approx(1.0)
    assert channel_one["edges"] == 0
    assert calls_after_first == 2
    assert calls_after_second_channel == 3
    assert edge_build_calls == calls_after_second_channel
    first["edges"] = 99
    assert mesh.uv_seam_graph_stats(0, tolerance=0.1)["edges"] == 1
    assert edge_build_calls == calls_after_second_channel

    mesh.uvs[0] = mesh.uvs[0].copy()
    mesh.uvs[0][3] = [0.0, 0.0]
    mesh.uvs[0][4] = [1.0, 0.0]
    updated = mesh.uv_seam_graph_stats(0, tolerance=0.1)

    assert updated["edges"] == 0
    assert edge_build_calls == calls_after_second_channel + 1


def test_uv_distortion_metrics_record_islands_pack_and_stretch() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)},
    )
    distorted = Mesh(
        points=mesh.points,
        faces=mesh.faces,
        uvs={0: np.array([[0, 0], [1, 0], [0, 1], [1, 0.25]], dtype=float)},
    )

    metrics = mesh.uv_distortion_metrics(0)
    distorted_metrics = distorted.uv_distortion_metrics(0)

    assert metrics["island_count"] == 1
    assert metrics["pack_efficiency"] == pytest.approx(1.0)
    assert metrics["normalized_pack_efficiency"] == pytest.approx(1.0)
    assert metrics["max_angle_distortion_degrees"] == pytest.approx(0.0)
    assert metrics["max_edge_length_distortion"] == pytest.approx(0.0)
    assert distorted_metrics["island_count"] == 1
    assert distorted_metrics["pack_efficiency"] == pytest.approx(0.625)
    assert distorted_metrics["max_angle_distortion_degrees"] > 0.0
    assert distorted_metrics["max_edge_length_distortion"] > 0.0


def test_mesh_copy_does_not_reenter_constructor_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        uvs={0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)},
    )
    calls = 0

    def spy_post_init(self: Mesh) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(Mesh, "__post_init__", spy_post_init)

    copied = mesh.copy()
    copied.points[0, 0] = 9.0
    copied.uvs[0][0, 0] = 0.5

    assert calls == 0
    assert mesh.points[0, 0] == 0.0
    assert mesh.uvs[0][0, 0] == 0.0


def test_mesh_copy_reuses_shared_topology_without_sharing_mutable_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    edge_faces = mesh._edge_faces_map()
    copied = mesh.copy()
    edge_faces[(1, 2)].append(99)

    def fail_face_edges(self: Mesh) -> np.ndarray:
        pytest.fail("copied mesh should reuse the shared topology cache")

    monkeypatch.setattr(Mesh, "_face_edges", fail_face_edges)

    copied_edge_faces = copied._edge_faces_map()

    assert copied_edge_faces[(1, 2)] == [0, 1]


def test_equivalent_new_mesh_reuses_shared_topology_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh_module._clear_global_mesh_cache()
    try:
        mesh = Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
            faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        )
        expected = mesh._face_unit_normals()
        recreated = Mesh(points=mesh.points.copy(), faces=mesh.faces.copy())

        def fail_cross(*_args: object, **_kwargs: object) -> np.ndarray:
            pytest.fail("equivalent rebuilt mesh should reuse the shared topology cache")

        monkeypatch.setattr(mesh_module.np, "cross", fail_cross)

        reused = recreated._face_unit_normals()
        assert np.array_equal(reused, expected)
        reused[0, 0] = 99.0
        third = Mesh(points=mesh.points.copy(), faces=mesh.faces.copy())

        assert np.array_equal(third._face_unit_normals(), expected)
    finally:
        mesh_module._clear_global_mesh_cache()


def test_equivalent_new_mesh_reuses_shared_fingerprint_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh_module._clear_global_mesh_cache()
    try:
        mesh = valid_triangle()
        expected = mesh.fingerprint()
        recreated = Mesh(points=mesh.points.copy(), faces=mesh.faces.copy())

        def fail_sha1() -> object:
            pytest.fail("equivalent rebuilt mesh should reuse the shared fingerprint cache")

        monkeypatch.setattr(mesh_module.hashlib, "sha1", fail_sha1)

        assert recreated.fingerprint() == expected
    finally:
        mesh_module._clear_global_mesh_cache()


def test_mesh_copy_reuses_cached_orientability_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = flipped_tetrahedron_mesh()
    expected = mesh.orientability_metrics()
    expected["orientation_components"] = 99
    copied = mesh.copy()

    def fail_defaultdict(*_args: object, **_kwargs: object) -> object:
        pytest.fail("copied mesh should reuse cached orientability metrics")

    monkeypatch.setattr(mesh_module, "defaultdict", fail_defaultdict)

    assert mesh.orientability_metrics()["orientation_components"] == 1
    assert copied.orientability_metrics()["flipped_orientation_components"] == 1


def test_equivalent_new_mesh_reuses_shared_orientability_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh_module._clear_global_mesh_cache()
    try:
        mesh = flipped_tetrahedron_mesh()
        expected = mesh.orientability_metrics()
        recreated = Mesh(points=mesh.points.copy(), faces=mesh.faces.copy())

        def fail_defaultdict(*_args: object, **_kwargs: object) -> object:
            pytest.fail("equivalent rebuilt mesh should reuse the shared orientability cache")

        monkeypatch.setattr(mesh_module, "defaultdict", fail_defaultdict)

        assert recreated.orientability_metrics() == expected
    finally:
        mesh_module._clear_global_mesh_cache()


def test_mesh_orientability_cache_rebuilds_after_in_place_face_mutation() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [2, 1, 0],
                [1, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    assert mesh.orientability_metrics()["orientation_components"] == 2

    mesh.faces[1] = np.array([2, 1, 3], dtype=np.int64)

    assert mesh.orientability_metrics()["orientation_components"] == 1


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


def test_split_faces_at_edges_preserves_order_and_materials() -> None:
    points = np.array(
        [[0, 0, 0], [4, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    edge_corners = np.array([[0, 1], [1, 2]], dtype=int)
    material_indices = np.array([4, 9], dtype=int)

    next_points, next_faces, next_materials, changed = mesh_module._split_faces_at_edges(
        points,
        faces,
        edge_corners,
        np.array([True, False], dtype=np.bool_),
        material_indices,
    )

    assert changed is True
    assert next_points[-1].tolist() == [2.0, 0.0, 0.0]
    assert next_faces.tolist() == [[0, 6, 2], [6, 1, 2], [3, 4, 5]]
    assert next_materials is not None
    assert next_materials.tolist() == [4, 4, 9]


def test_collapse_short_edges_respects_boundary_preservation() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 3], [1, 2, 3]], dtype=int),
    )

    preserved = mesh.collapse_short_edges(0.1, preserve_boundaries=True)
    collapsed = mesh.collapse_short_edges(0.1, preserve_boundaries=False)

    assert preserved.vertex_count == mesh.vertex_count
    assert collapsed.vertex_count < mesh.vertex_count


def test_collapse_short_edges_averages_transitive_components() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4]], dtype=int),
    )

    collapsed = mesh.collapse_short_edges(0.05, preserve_boundaries=False)

    assert collapsed.triangle_count == 1
    assert any(np.allclose(point, [0.01, 0.0, 0.0]) for point in collapsed.points)


def test_improve_skinny_triangles_splits_long_internal_edges_and_reports_quality() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [10, 0, 0], [0.1, 1, 0], [9.9, -1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [1, 0, 3]], dtype=int),
        material_indices=np.array([2, 2], dtype=int),
    )

    before = mesh.quality_metrics(skinny_aspect_ratio=6.0)
    improved = mesh.improve_skinny_triangles(max_aspect_ratio=6.0, preserve_boundaries=True)
    after = improved.quality_metrics(skinny_aspect_ratio=6.0)

    assert before["skinny_triangles"] > after["skinny_triangles"]
    assert improved.triangle_count > mesh.triangle_count
    assert improved.material_indices is not None
    assert set(improved.material_indices.tolist()) == {2}


def test_quality_metrics_counts_edge_and_topology_risks() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [3, 0, 0], [0, 0.01, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
    )

    metrics = mesh.quality_metrics(min_edge_length=0.05, max_edge_length=2.0, skinny_aspect_ratio=20.0)

    assert metrics["short_edges"] > 0
    assert metrics["long_edges"] > 0
    assert metrics["skinny_triangles"] > 0
    assert metrics["boundary_edges"] > 0


def test_mesh_topology_cache_rebuilds_after_in_place_face_mutation() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
    )

    edge_faces = mesh._edge_faces_map()
    assert edge_faces[(0, 2)] == [0, 1]

    mesh.faces[1] = np.array([0, 1, 3], dtype=np.int64)

    updated_edge_faces = mesh._edge_faces_map()
    assert updated_edge_faces[(0, 1)] == [0, 1]
    assert updated_edge_faces[(0, 2)] == [0]


def test_shared_edge_incidence_feeds_topology_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([2, 4], dtype=int),
    )

    def fail_face_edges(self: Mesh) -> np.ndarray:
        pytest.fail("topology helpers should use shared edge incidence")

    monkeypatch.setattr(Mesh, "_face_edges", fail_face_edges)

    assert mesh._edge_faces_map()[(1, 2)] == [0, 1]
    assert mesh._boundary_loops() == [[0, 1, 3, 2]]
    assert mesh._vertex_material_signatures()[1] == (2, 4)
    assert mesh._vertex_material_signatures()[2] == (2, 4)


def test_mesh_fingerprint_cache_rebuilds_after_in_place_point_mutation() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    before = mesh.fingerprint()
    assert mesh.fingerprint() == before

    mesh.points[2] = np.array([0, 1, 1], dtype=np.float64)

    assert mesh.fingerprint() != before


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


def test_optimize_buffers_duplicate_face_groups_stay_in_range(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 0]], dtype=int),
        material_indices=np.array([3, 7], dtype=int),
        face_groups={
            "canonical": np.array([0], dtype=int),
            "duplicate": np.array([1], dtype=int),
        },
    )

    def optimize_vertex_cache(destination: np.ndarray, indices: np.ndarray, *, vertex_count: int) -> None:
        _ = vertex_count
        destination[:] = indices

    def generate_vertex_remap(remap: np.ndarray, _indices: np.ndarray, *, vertices: np.ndarray) -> int:
        remap[:] = np.arange(vertices.shape[0], dtype=np.uint32)
        return int(vertices.shape[0])

    def remap_index_buffer(destination: np.ndarray, indices: np.ndarray, *, remap: np.ndarray) -> None:
        destination[:] = remap[indices]

    monkeypatch.setitem(
        sys.modules,
        "meshoptimizer",
        SimpleNamespace(
            optimize_vertex_cache=optimize_vertex_cache,
            generate_vertex_remap=generate_vertex_remap,
            remap_index_buffer=remap_index_buffer,
        ),
    )

    optimized = mesh.optimize_buffers()

    optimized.validate()
    assert optimized.material_indices is not None
    assert optimized.material_indices.tolist() == [3, 3]
    assert "canonical" in optimized.face_groups
    assert "duplicate" not in optimized.face_groups
    for values in optimized.face_groups.values():
        assert values.size > 0
        assert np.all((values >= 0) & (values < optimized.triangle_count))


def test_optimize_buffers_logs_fallback_warning(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_optimize_vertex_cache(_destination: np.ndarray, _indices: np.ndarray, *, vertex_count: int) -> None:
        _ = vertex_count
        raise RuntimeError("cache optimizer failed")

    monkeypatch.setitem(
        sys.modules,
        "meshoptimizer",
        SimpleNamespace(optimize_vertex_cache=fail_optimize_vertex_cache),
    )
    mesh = valid_triangle()

    with caplog.at_level(logging.WARNING, logger="fascat"):
        optimized = mesh.optimize_buffers()

    assert optimized.to_dict() == mesh.to_dict()
    assert "optimize_buffers failed; returning unoptimized copy" in caplog.text


def test_optimize_buffers_warns_when_meshoptimizer_is_missing(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "meshoptimizer", None)
    mesh = valid_triangle()

    with caplog.at_level(logging.WARNING, logger="fascat"):
        optimized = mesh.optimize_buffers()

    assert optimized.to_dict() == mesh.to_dict()
    assert "meshoptimizer is not installed; returning unoptimized copy" in caplog.text
    assert 'pip install "fascat[meshopt]"' in caplog.text


def test_simplify_warns_when_meshoptimizer_is_missing(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "meshoptimizer", None)
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )

    with caplog.at_level(logging.WARNING, logger="fascat"):
        simplified = mesh.simplify(target_triangles=1)

    simplified.validate()
    assert "meshoptimizer is not installed; falling back to fast-simplification" in caplog.text
    assert 'pip install "fascat[meshopt]"' in caplog.text


def test_simplify_preserves_material_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )

    simplified = mesh.simplify(target_triangles=1)

    assert simplified.material_indices is not None
    assert simplified.material_indices.shape == (simplified.triangle_count,)
    assert set(simplified.material_indices.tolist()).issubset({0, 1})


def test_simplify_target_error_metadata_reports_hint_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )

    def fake_simplify(
        destination: np.ndarray,
        indices: np.ndarray,
        positions: np.ndarray,
        *,
        vertex_count: int,
        target_index_count: int,
        target_error: float,
        result_error: np.ndarray,
    ) -> int:
        _ = positions, vertex_count, target_index_count
        destination[:3] = indices[:3]
        result_error[0] = target_error * 2.0
        return 3

    monkeypatch.setitem(sys.modules, "meshoptimizer", SimpleNamespace(simplify=fake_simplify))

    simplified = mesh.simplify(target_triangles=1, target_error=0.125)

    assert simplified.metadata["simplify_error_bound"] == "0.125"
    assert simplified.metadata["simplify_error_bound_policy"] == "hint"
    assert simplified.metadata["simplify_result_error"] == "0.25"
    assert simplified.metadata["simplify_error_bound_status"] == "exceeded_hint"


def test_simplify_preserves_explicit_protected_faces() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [10, 0, 0],
                [11, 0, 0],
                [10, 1, 0],
                [20, 0, 0],
                [21, 0, 0],
                [20, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=int),
    )

    simplified = mesh.simplify(target_triangles=1, protected_faces=np.asarray([2], dtype=int))
    centroid = simplified.points[simplified.faces[0]].mean(axis=0)

    assert simplified.triangle_count == 1
    assert centroid[0] > 19.0
    assert simplified.metadata["simplification_preserved_feature_faces"] == "1"


def test_uv_seam_vertices_returns_empty_without_uvs_or_vertices() -> None:
    assert valid_triangle()._uv_seam_vertices() == set()

    empty = Mesh(
        points=np.empty((0, 3), dtype=float),
        faces=np.empty((0, 3), dtype=int),
        uvs={0: np.empty((0, 2), dtype=float)},
    )

    assert empty._uv_seam_vertices() == set()


def test_uv_seam_vertices_uses_rounded_position_groups_across_uv_channels() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0000000001, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 2, 4], [1, 3, 4]], dtype=int),
        uvs={
            0: np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [2.0, 0.0],
                ],
                dtype=float,
            ),
            1: np.array(
                [
                    [0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [9.0, 9.0],
                ],
                dtype=float,
            ),
        },
    )

    counts = mesh.feature_preservation_counts(preserve_uv_seams=True)

    assert mesh._uv_seam_vertices() == {0, 1}
    assert counts["uv_seam_faces"] == 2
    assert counts["total_feature_faces"] == 2


def test_merge_close_vertices_preserves_material_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [0.001, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 2, 3], [1, 2, 4]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )

    merged = mesh.merge_close_vertices(0.01)

    assert merged.material_indices is not None
    assert sorted(merged.material_indices.tolist()) == [0, 1]


def test_repair_drops_non_finite_faces_without_losing_material_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [np.nan, 0, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [0, 3, 2]], dtype=int),
        material_indices=np.array([1, 2], dtype=int),
    )

    repaired = mesh.repair(RepairOptions())

    assert repaired.triangle_count == 1
    assert repaired.material_indices is not None
    assert repaired.material_indices.tolist() == [1]


def test_repair_drops_invalid_face_indices_before_cleanup() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=float),
        faces=np.array([[0, 1, 2], [-1, 1, 2], [0, 1, 9]], dtype=int),
        material_indices=np.array([3, 4, 5], dtype=int),
        face_groups={"panel": np.array([0, 1, 2], dtype=int)},
    )

    repaired = mesh.repair()

    repaired.validate()
    assert repaired.vertex_count == 3
    assert repaired.triangle_count == 1
    assert repaired.faces.tolist() == [[0, 1, 2]]
    assert repaired.material_indices is not None
    assert repaired.material_indices.tolist() == [3]
    assert repaired.face_groups["panel"].tolist() == [0]


def test_fix_winding_preserves_face_linked_attributes(monkeypatch: pytest.MonkeyPatch) -> None:
    trimesh = pytest.importorskip("trimesh")

    def reverse_faces(tri: object, **_kwargs: object) -> None:
        tri.faces = tri.faces[::-1]  # type: ignore[attr-defined]

    monkeypatch.setattr(trimesh.repair, "fix_normals", reverse_faces)
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([4, 9], dtype=int),
        face_groups={"second": np.array([1], dtype=int)},
    )

    fixed = mesh.fix_winding()

    assert fixed.faces.tolist() == [[2, 1, 3], [0, 1, 2]]
    assert fixed.material_indices is not None
    assert fixed.material_indices.tolist() == [9, 4]
    assert fixed.face_groups["second"].tolist() == [0]


def test_filtering_faces_remaps_face_groups() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 0, 1], [0, 1, 2], [2, 1, 3]], dtype=int),
        face_groups={"panel": np.array([1, 2], dtype=int)},
    )

    repaired = mesh.remove_degenerate_faces()

    assert repaired.face_groups["panel"].tolist() == [0, 1]


def test_mesh_validation_rejects_invalid_face_group_indices() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        face_groups={"bad": np.array([1], dtype=int)},
    )

    with pytest.raises(MeshValidationError, match="face group bad"):
        mesh.validate()


def test_repair_computes_normals_around_optional_hole_fill(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    original_compute_normals = Mesh.compute_normals
    original_fill_holes = Mesh.fill_holes

    def tracked_fix_winding(self: Mesh) -> Mesh:
        order.append("fix_winding")
        return self.copy()

    def tracked_compute_normals(self: Mesh, *, angle_weighted: bool = True) -> Mesh:
        order.append("compute_normals")
        return original_compute_normals(self, angle_weighted=angle_weighted)

    def tracked_fill_holes(self: Mesh) -> Mesh:
        order.append("fill_holes")
        return original_fill_holes(self)

    monkeypatch.setattr(Mesh, "fix_winding", tracked_fix_winding)
    monkeypatch.setattr(Mesh, "compute_normals", tracked_compute_normals)
    monkeypatch.setattr(Mesh, "fill_holes", tracked_fill_holes)
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3]], dtype=int),
    )

    repaired = mesh.repair(RepairOptions(fill_small_holes=True))

    assert order == ["fix_winding", "compute_normals", "fill_holes", "compute_normals"]
    assert repaired.normals is not None
    assert repaired.triangle_count == 4


def test_mesh_to_dict_exposes_material_and_face_group_summary() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([2, 5], dtype=int),
        face_groups={"panel": np.array([1], dtype=int)},
    )

    payload = mesh.to_dict()

    assert payload["material_indices"] == {"count": 2, "unique": [2, 5]}
    assert payload["face_groups"] == {"panel": {"count": 1, "indices": [1]}}


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


def test_fill_holes_assigns_materials_from_nearest_faces() -> None:
    mesh = Mesh(
        points=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [10, 0, 0], [11, 0, 0], [10, 1, 0], [10, 0, 1]],
            dtype=float,
        ),
        faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [4, 6, 5], [4, 5, 7], [5, 6, 7]], dtype=int),
        material_indices=np.array([1, 1, 1, 0, 0, 0], dtype=int),
    )

    filled = mesh.fill_holes()

    assert filled.triangle_count == 8
    assert filled.material_indices is not None
    assert filled.material_indices[:6].tolist() == [1, 1, 1, 0, 0, 0]
    for face, material in zip(filled.faces[6:].tolist(), filled.material_indices[6:].tolist(), strict=True):
        expected = 1 if all(vertex < 4 for vertex in face) else 0
        assert material == expected
    assert filled.metadata["hole_fill_faces"] == "2"


def test_fill_holes_leaves_face_groups_unchanged() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3]], dtype=int),
        face_groups={"cap": np.array([0], dtype=int)},
    )

    filled = mesh.fill_holes()

    assert filled.triangle_count == 4
    assert filled.face_groups["cap"].tolist() == [0]
    assert all(indices.max() < 3 for indices in filled.face_groups.values())
    assert filled.metadata["hole_fill_faces"] == "1"


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


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e3])
def test_remove_degenerate_faces_default_epsilon_is_scale_invariant(scale: float) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float) * scale,
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )

    assert mesh.remove_degenerate_faces().triangle_count == 2


def test_remove_degenerate_faces_removes_relative_slivers_in_large_models() -> None:
    # The third face is a sliver whose area (~5e-10) is negligible relative to a
    # 1000-unit model but far above the historic fixed 1e-12 threshold.
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1000, 0, 0],
                [0, 1000, 0],
                [1000, 1000, 0],
                [500, 500, 0],
                [501, 500, 0],
                [500.5, 500 + 1e-9, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]], dtype=int),
    )

    cleaned = mesh.remove_degenerate_faces()

    assert cleaned.triangle_count == 2
    assert mesh.remove_degenerate_faces(area_epsilon=1e-12).triangle_count == 3


def test_remove_degenerate_faces_explicit_epsilon_is_authoritative() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    assert mesh.remove_degenerate_faces(area_epsilon=10.0).triangle_count == 0
    assert mesh.remove_degenerate_faces(area_epsilon=0.0).triangle_count == 1


def test_delete_degenerate_polygons_reports_resolved_epsilon_mode() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    auto = mesh.delete_degenerate_polygons()
    explicit = mesh.delete_degenerate_polygons(DeleteDegeneratePolygonsOptions(area_epsilon=1e-9))

    assert auto.metadata["delete_degenerate_polygons_area_epsilon_mode"] == "auto"
    assert float(auto.metadata["delete_degenerate_polygons_area_epsilon"]) == pytest.approx(2e-12)
    assert explicit.metadata["delete_degenerate_polygons_area_epsilon_mode"] == "explicit"
    assert explicit.metadata["delete_degenerate_polygons_area_epsilon"] == "1e-09"


def test_remap_face_attributes_invalidates_on_unmatched_faces() -> None:
    source = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
        face_groups={"panel": np.array([1], dtype=int)},
    )
    target = source.copy()
    target.faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=int)

    target._remap_face_attributes_from(source)

    assert target.material_indices is None
    assert target.face_groups == {}
    assert target.metadata["face_attribute_remap_dropped"] == "face_keys_unmatched"


def test_remap_face_attributes_invalidates_on_count_change() -> None:
    source = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )
    target = source.copy()
    target.faces = np.array([[0, 1, 2]], dtype=int)

    target._remap_face_attributes_from(source)

    assert target.material_indices is None
    assert target.metadata["face_attribute_remap_dropped"] == "triangle_count_changed"


def test_remap_face_attributes_follows_face_permutation() -> None:
    source = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
        face_groups={"panel": np.array([1], dtype=int)},
    )
    target = source.copy()
    target.faces = np.array([[1, 3, 2], [2, 1, 0]], dtype=int)

    target._remap_face_attributes_from(source)

    assert target.material_indices is not None
    assert target.material_indices.tolist() == [1, 0]
    assert target.face_groups["panel"].tolist() == [0]
    assert "face_attribute_remap_dropped" not in target.metadata


def _brute_force_nearest_materials(source: Mesh, points: np.ndarray, faces: np.ndarray) -> list[int]:
    assert source.material_indices is not None
    source_centroids = source.points[source.faces].mean(axis=1)
    target_centroids = points[faces].mean(axis=1)
    result: list[int] = []
    for centroid in target_centroids:
        distances = np.einsum("ij,ij->i", source_centroids - centroid, source_centroids - centroid)
        result.append(int(source.material_indices[int(np.argmin(distances))]))
    return result


def test_assign_materials_by_nearest_centroid_matches_brute_force() -> None:
    rng = np.random.default_rng(1234)
    source_points = rng.random((300, 3))
    source_faces = rng.integers(0, 300, size=(120, 3))
    material_indices = rng.integers(0, 5, size=120)
    source = Mesh(
        points=source_points,
        faces=source_faces,
        material_indices=material_indices.astype(int),
    )

    target_points = rng.random((50, 3))
    target_faces = rng.integers(0, 50, size=(30, 3))

    assigned = source._assign_materials_by_nearest_centroid(target_points, target_faces)

    assert assigned is not None
    expected = _brute_force_nearest_materials(source, target_points, target_faces)
    assert assigned.tolist() == expected


def test_assign_materials_by_nearest_centroid_kd_tree_matches_brute_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mesh_module, "_NEAREST_CENTROID_PAIR_LIMIT", 0)
    rng = np.random.default_rng(4321)
    source_points = rng.random((180, 3))
    source_faces = rng.integers(0, 180, size=(90, 3))
    material_indices = rng.integers(0, 7, size=90)
    source = Mesh(
        points=source_points,
        faces=source_faces,
        material_indices=material_indices.astype(int),
    )

    target_points = rng.random((60, 3))
    target_faces = rng.integers(0, 60, size=(40, 3))

    assigned = source._assign_materials_by_nearest_centroid(target_points, target_faces)

    assert assigned is not None
    expected = _brute_force_nearest_materials(source, target_points, target_faces)
    assert assigned.tolist() == expected


def test_assign_materials_by_nearest_centroid_reuses_cached_kd_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mesh_module, "_NEAREST_CENTROID_PAIR_LIMIT", 0)
    original_build = mesh_module._build_centroid_kd_tree
    top_level_builds = 0

    def count_kd_tree_build(
        centroids: np.ndarray,
        indices: np.ndarray,
    ) -> mesh_module._CentroidKdNode | None:
        nonlocal top_level_builds
        if indices.size == centroids.shape[0]:
            top_level_builds += 1
        return original_build(centroids, indices)

    monkeypatch.setattr(mesh_module, "_build_centroid_kd_tree", count_kd_tree_build)
    rng = np.random.default_rng(5678)
    source = Mesh(
        points=rng.random((180, 3)),
        faces=rng.integers(0, 180, size=(90, 3)),
        material_indices=rng.integers(0, 7, size=90).astype(int),
    )
    first_points = rng.random((60, 3))
    first_faces = rng.integers(0, 60, size=(40, 3))
    second_points = rng.random((45, 3))
    second_faces = rng.integers(0, 45, size=(35, 3))

    first = source._assign_materials_by_nearest_centroid(first_points, first_faces)
    second = source._assign_materials_by_nearest_centroid(second_points, second_faces)

    assert first is not None
    assert second is not None
    assert first.tolist() == _brute_force_nearest_materials(source, first_points, first_faces)
    assert second.tolist() == _brute_force_nearest_materials(source, second_points, second_faces)
    assert top_level_builds == 1

    source.points = source.points.copy()
    source.points[0, 0] += 1.0
    source._assign_materials_by_nearest_centroid(first_points, first_faces)

    assert top_level_builds == 2


def test_assign_materials_by_nearest_centroid_is_memory_bounded() -> None:
    # A large source mesh used to allocate a (target_chunk x n_source x 3) array,
    # which exhausted RAM and froze the terminal. The chunked implementation must
    # complete quickly with a tiny working set regardless of source size.
    rng = np.random.default_rng(7)
    n_source = 200_000
    source = Mesh(
        points=rng.random((n_source, 3)),
        faces=np.arange(n_source * 3).reshape((n_source, 3)) % n_source,
        material_indices=rng.integers(0, 8, size=n_source).astype(int),
    )
    target_points = rng.random((100, 3))
    target_faces = (np.arange(300).reshape((100, 3)) % 100).astype(int)

    assigned = source._assign_materials_by_nearest_centroid(target_points, target_faces)

    assert assigned is not None
    assert assigned.shape == (100,)
    assert assigned.min() >= 0 and assigned.max() < 8


def test_mesh_to_trimesh_copies_geometry_and_attributes() -> None:
    trimesh = pytest.importorskip("trimesh")
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    faces = np.array([[0, 1, 2]], dtype=int)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=float)
    uvs = {0: np.array([[0, 0], [1, 0], [0, 1]], dtype=float)}
    material_indices = np.array([2], dtype=int)
    mesh = Mesh(
        points=points,
        faces=faces,
        normals=normals,
        uvs=uvs,
        material_indices=material_indices,
        metadata={"source": "cad"},
    )

    converted = mesh.to_trimesh()
    mesh.points[0, 0] = 9.0
    mesh.material_indices[0] = 7

    assert isinstance(converted, trimesh.Trimesh)
    np.testing.assert_allclose(converted.vertices, points)
    np.testing.assert_array_equal(converted.faces, faces)
    np.testing.assert_allclose(converted.vertex_normals, normals)
    np.testing.assert_allclose(converted.vertex_attributes["uv0"], uvs[0])
    np.testing.assert_array_equal(converted.face_attributes["material_indices"], material_indices)
    assert converted.metadata == {"source": "cad"}


@pytest.fixture
def empty_shared_mesh_cache() -> Iterator[None]:
    mesh_module._clear_global_mesh_cache()
    yield
    mesh_module._clear_global_mesh_cache()


@pytest.mark.parametrize("nested", [False, True])
def test_shared_mesh_cache_byte_budget_evicts_least_recently_used(
    monkeypatch: pytest.MonkeyPatch, empty_shared_mesh_cache: object, nested: bool
) -> None:
    value = {(1, 2): [np.arange(128), 17]} if nested else np.arange(128)
    name = "edge_faces_map"
    # Zero occupies fewer bytes than positive integers on Python 3.10.
    # Use equally sized tokens so two entries exactly fill the byte budget.
    mesh_module._store_global_cache_value(name, (1,), value)
    entry_size = mesh_module._global_mesh_cache_bytes
    assert entry_size > 128 * 8
    mesh_module._clear_global_mesh_cache()
    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_BYTES", entry_size * 2)

    for index in range(1, 3):
        mesh_module._store_global_cache_value(name, (index,), value)
    assert mesh_module._global_cache_value(name, (1,)) is not mesh_module._MISSING_CACHE_VALUE
    mesh_module._store_global_cache_value(name, (3,), value)

    assert list(mesh_module._GLOBAL_MESH_CACHE) == [(name, (1,)), (name, (3,))]
    assert mesh_module._global_mesh_cache_bytes == entry_size * 2
    # Replacing an entry must subtract its old size before adding the new size.
    mesh_module._store_global_cache_value(name, (1,), 1)
    assert mesh_module._global_mesh_cache_bytes < entry_size * 2
    assert mesh_module._global_mesh_cache_bytes == sum(size for size, _ in mesh_module._GLOBAL_MESH_CACHE.values())


@pytest.mark.parametrize("nested", [False, True])
def test_shared_mesh_cache_skips_oversized_values_before_cloning(
    monkeypatch: pytest.MonkeyPatch, empty_shared_mesh_cache: object, nested: bool
) -> None:
    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_BYTES", 4096)
    value = {(index, index + 1): [index] for index in range(100)} if nested else np.arange(1024)

    def fail_clone(_value: object) -> object:
        pytest.fail("oversized entries must not be cloned")

    monkeypatch.setattr(mesh_module, "_clone_cache_value", fail_clone)
    mesh_module._store_global_cache_value("edge_faces_map", (0,), value)
    assert not mesh_module._GLOBAL_MESH_CACHE
    assert mesh_module._global_mesh_cache_bytes == 0


def test_shared_mesh_cache_accounts_for_copied_views_and_repeated_arrays(
    monkeypatch: pytest.MonkeyPatch, empty_shared_mesh_cache: object
) -> None:
    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_BYTES", 4096)
    view = np.arange(100_000)[::10_000]
    mesh_module._store_global_cache_value("undirected_edges_and_counts", (0,), (view, view))
    size, stored = next(iter(mesh_module._GLOBAL_MESH_CACHE.values()))
    assert isinstance(stored, tuple)
    assert all(array.base is None for array in stored)
    assert not np.shares_memory(*stored)
    assert size >= sum(sys.getsizeof(array) for array in stored)
    view[0] = -1
    assert stored[0][0] == 0


def test_mesh_copy_loads_derived_data_lazily_without_shared_cache(
    monkeypatch: pytest.MonkeyPatch, empty_shared_mesh_cache: object
) -> None:
    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_BYTES", 0)
    mesh = valid_triangle()
    expected_edges, expected_counts = mesh._undirected_edges_and_counts()
    assert mesh._undirected_edges_and_counts()[0] is expected_edges
    assert not mesh_module._GLOBAL_MESH_CACHE
    copied = mesh.copy()
    assert not copied._cache
    copied_edges, copied_counts = copied._undirected_edges_and_counts()
    assert np.array_equal(copied_edges, expected_edges)
    assert np.array_equal(copied_counts, expected_counts)
    assert not np.shares_memory(copied_edges, expected_edges)
    copied_edges[0, 0] = 99
    assert expected_edges[0, 0] == 0
    copied.faces[0] = [0, 2, 2]
    rebuilt_edges, _ = copied._undirected_edges_and_counts()
    assert 99 not in rebuilt_edges
    assert np.array_equal(mesh.faces, [[0, 1, 2]])


def test_shared_mesh_cache_bounds_retained_arrays_after_mesh_release(
    monkeypatch: pytest.MonkeyPatch, empty_shared_mesh_cache: object
) -> None:
    import gc
    import weakref

    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_BYTES", 16_384)
    random = np.random.default_rng(42)
    references = []
    for _ in range(20):
        mesh = Mesh(points=random.random((60, 3)), faces=random.integers(0, 60, (100, 3)))
        edges, counts = mesh._undirected_edges_and_counts()
        references.extend([weakref.ref(mesh), weakref.ref(edges), weakref.ref(counts)])
    del mesh, edges, counts
    gc.collect()

    assert all(reference() is None for reference in references)
    assert 0 < mesh_module._global_mesh_cache_bytes <= 16_384
    retained = [weakref.ref(array) for _, arrays in mesh_module._GLOBAL_MESH_CACHE.values() for array in arrays]
    assert sum(reference().nbytes for reference in retained) <= 16_384
    mesh_module._clear_global_mesh_cache()
    gc.collect()
    assert all(reference() is None for reference in retained)
    assert mesh_module._global_mesh_cache_bytes == 0


def test_shared_mesh_cache_threaded_hits_eviction_and_clear(
    monkeypatch: pytest.MonkeyPatch, empty_shared_mesh_cache: object
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_BYTES", 4096)
    monkeypatch.setattr(mesh_module, "_GLOBAL_MESH_CACHE_MAX_ENTRIES", 3)

    def exercise_cache(worker: int) -> None:
        for iteration in range(100):
            token = (iteration % 7,)
            mesh_module._store_global_cache_value("edge_faces_map", token, {token: [token[0]]})
            value = mesh_module._global_cache_value("edge_faces_map", token)
            if value is not mesh_module._MISSING_CACHE_VALUE:
                assert isinstance(value, dict)
                assert value[token] == [token[0]]
                value[token].append(-1)
            if (iteration + worker) % 31 == 0:
                mesh_module._clear_global_mesh_cache()
            with mesh_module._GLOBAL_MESH_CACHE_LOCK:
                assert len(mesh_module._GLOBAL_MESH_CACHE) <= 3
                assert 0 <= mesh_module._global_mesh_cache_bytes <= 4096
                assert mesh_module._global_mesh_cache_bytes == sum(
                    size for size, _ in mesh_module._GLOBAL_MESH_CACHE.values()
                )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(exercise_cache, range(8)))
