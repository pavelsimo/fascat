from __future__ import annotations

import numpy as np
import pytest

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

    repaired = mesh.repair(RepairOptions())

    assert repaired.triangle_count == 1
    assert repaired.metadata["repair_duplicate_polygons_before"] == "1"
    assert repaired.metadata["repair_duplicate_polygons_after"] == "0"
    assert repaired.metadata["repair_degenerate_triangles_before"] == "1"
    assert repaired.metadata["repair_degenerate_triangles_after"] == "0"


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

    merged = mesh.merge_vertices(MergeVerticesOptions())

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

    merged = mesh.merge_vertices(MergeVerticesOptions())

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
    ).merge_vertices(MergeVerticesOptions())
    boundary_gap = Mesh(
        points=np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1.005, 0, 0], [2, 0, 0], [1.005, 1, 0]],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
    ).merge_vertices(MergeVerticesOptions(tolerance=0.01))

    assert t_junction.metadata["merge_vertices_candidate_t_junctions"] == "1"
    assert t_junction.metadata["merge_vertices_candidate_boundary_gaps"] == "0"
    assert boundary_gap.metadata["merge_vertices_candidate_t_junctions"] == "0"
    assert boundary_gap.metadata["merge_vertices_candidate_boundary_gaps"] == "1"


def test_merge_vertices_reports_tolerance_scale_risk() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.3))

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

    merged = mesh.merge_vertices(MergeVerticesOptions(tolerance=0.0001))

    assert merged.vertex_count == 6
    assert merged.metadata["merge_vertices_removed"] == "0"
    assert merged.metadata["merge_vertices_near_duplicate_pairs"] == "1"
    assert merged.metadata["merge_vertices_nearest_near_duplicate_distance"] == "0.001"
    assert merged.metadata["merge_vertices_tolerance_advisory"] == "near_duplicates_unmerged"


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
    assert cleaned.metadata["delete_degenerate_polygons_removed_duplicate_vertices"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_removed_collapsed_edges"] == "1"
    assert cleaned.metadata["delete_degenerate_polygons_removed_near_flat_area"] == "1"

    noop = cleaned.delete_degenerate_polygons(DeleteDegeneratePolygonsOptions())

    assert noop.triangle_count == 1
    assert noop.metadata["delete_degenerate_polygons_before"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_duplicate_vertices"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_collapsed_edges"] == "0"
    assert noop.metadata["delete_degenerate_polygons_removed_near_flat_area"] == "0"


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

    repaired = mesh.repair(RepairOptions())

    assert mesh.t_junction_count() == 1
    assert repaired.metadata["repair_t_junctions_before"] == "1"
    assert repaired.metadata["repair_t_junctions_after"] == "1"


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

    repaired = mesh.repair(RepairOptions(tolerance=0.01, merge_vertices=False))

    assert mesh.boundary_gap_count(tolerance=0.01) == 1
    assert repaired.metadata["repair_boundary_gaps_before"] == "1"
    assert repaired.metadata["repair_boundary_gaps_after"] == "1"


def test_orientability_metrics_detect_mobius_like_strip() -> None:
    mesh = mobius_strip_mesh()

    metrics = mesh.orientability_metrics()
    repaired = mesh.repair(RepairOptions())

    assert metrics["orientation_components"] == 1
    assert metrics["non_orientable_edges"] == 1
    assert repaired.metadata["repair_orientation_components_before_orientation"] == "1"
    assert repaired.metadata["repair_non_orientable_edges_before_orientation"] == "1"


def test_repair_records_flipped_closed_orientation_components() -> None:
    mesh = flipped_tetrahedron_mesh()

    metrics = mesh.orientability_metrics()
    repaired = mesh.repair(RepairOptions())
    not_fixed = mesh.repair(RepairOptions(fix_winding=False))

    assert metrics["closed_orientation_components"] == 1
    assert metrics["flipped_orientation_components"] == 1
    assert repaired.metadata["repair_closed_orientation_components_before_orientation"] == "1"
    assert repaired.metadata["repair_closed_orientation_components_after_orientation"] == "1"
    assert repaired.metadata["repair_flipped_components_before_orientation"] == "1"
    assert repaired.metadata["repair_flipped_components_after_orientation"] == "0"
    assert not_fixed.metadata["repair_flipped_components_after_orientation"] == "1"


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


def test_collapse_short_edges_respects_boundary_preservation() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [0.01, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 3], [1, 2, 3]], dtype=int),
    )

    preserved = mesh.collapse_short_edges(0.1, preserve_boundaries=True)
    collapsed = mesh.collapse_short_edges(0.1, preserve_boundaries=False)

    assert preserved.vertex_count == mesh.vertex_count
    assert collapsed.vertex_count < mesh.vertex_count


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
