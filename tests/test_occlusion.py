from __future__ import annotations

import numpy as np
import pytest

import fascat.ops.occlusion as occlusion_module
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.ops.occlusion import (
    _build_occurrence_bvh,
    _build_triangle_bvh,
    _OccluderSet,
    _segment_blocked,
    _segment_intersects_mesh,
    _segment_intersects_occurrence,
    _segment_intersects_triangles,
    _segment_triangle_t,
    _WorldOccurrence,
)
from fascat.options import RemoveOccludedOptions

from ._actions_helpers import _cube_mesh, _merge_meshes, _translated_mesh, _triangle_strip


def test_segment_intersects_mesh_matches_scalar_triangle_hits() -> None:
    points = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 0],
            [2, 1, 0],
        ],
        dtype=float,
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=int)
    start = np.array([0.25, 0.25, 1.0], dtype=float)
    end = np.array([0.25, 0.25, -1.0], dtype=float)
    miss_start = np.array([1.5, 0.25, 1.0], dtype=float)
    miss_end = np.array([1.5, 0.25, -1.0], dtype=float)

    scalar_hits = [
        _segment_triangle_t(start, end, points[face]) is not None
        and 1e-8 < _segment_triangle_t(start, end, points[face]) < 1.0 - 1e-8
        for face in faces
    ]

    assert _segment_intersects_mesh(start, end, points, faces) is any(scalar_hits)
    assert _segment_intersects_mesh(miss_start, miss_end, points, faces) is False


def test_segment_intersects_occurrence_uses_triangle_bvh_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _triangle_strip(160)
    triangles = mesh.points[mesh.faces]
    triangle_bounds_min = triangles.min(axis=1)
    triangle_bounds_max = triangles.max(axis=1)
    occurrence = _WorldOccurrence(
        node=Node(id="node", name="node", part_id="part"),
        part_id="part",
        world_points=mesh.points,
        faces=mesh.faces,
        triangles=triangles,
        triangle_edges1=triangles[:, 1] - triangles[:, 0],
        triangle_edges2=triangles[:, 2] - triangles[:, 0],
        triangle_bounds_min=triangle_bounds_min,
        triangle_bounds_max=triangle_bounds_max,
        triangle_bvh=_build_triangle_bvh(triangle_bounds_min, triangle_bounds_max),
        bounds_min=mesh.points.min(axis=0),
        bounds_max=mesh.points.max(axis=0),
        volume=1.0,
    )
    batch_sizes: list[int] = []

    def record_triangle_batch(
        start_arg: np.ndarray,
        end_arg: np.ndarray,
        triangles_arg: np.ndarray,
        edge1: np.ndarray,
        edge2: np.ndarray,
        triangle_bounds_min_arg: np.ndarray,
        triangle_bounds_max_arg: np.ndarray,
    ) -> bool:
        batch_sizes.append(triangles_arg.shape[0])
        return _segment_intersects_triangles(
            start_arg,
            end_arg,
            triangles_arg,
            edge1,
            edge2,
            triangle_bounds_min_arg,
            triangle_bounds_max_arg,
        )

    monkeypatch.setattr("fascat.ops.occlusion._segment_intersects_triangles", record_triangle_batch)
    start = np.array([318.25, 0.25, 1.0], dtype=float)
    end = np.array([318.25, 0.25, -1.0], dtype=float)

    assert _segment_intersects_occurrence(start, end, occurrence) is True
    assert batch_sizes
    assert max(batch_sizes) < mesh.triangle_count


def test_segment_blocked_uses_occluder_bvh_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    occurrences: list[_WorldOccurrence] = []
    for index in range(128):
        base = float(index * 2)
        points = np.asarray([[base, 0.0, 0.0], [base + 1.0, 0.0, 0.0], [base, 1.0, 0.0]], dtype=float)
        faces = np.asarray([[0, 1, 2]], dtype=int)
        triangles = points[faces]
        triangle_bounds_min = triangles.min(axis=1)
        triangle_bounds_max = triangles.max(axis=1)
        occurrences.append(
            _WorldOccurrence(
                node=Node(id=f"node-{index}", name=f"node-{index}", part_id=f"part-{index}"),
                part_id=f"part-{index}",
                world_points=points,
                faces=faces,
                triangles=triangles,
                triangle_edges1=triangles[:, 1] - triangles[:, 0],
                triangle_edges2=triangles[:, 2] - triangles[:, 0],
                triangle_bounds_min=triangle_bounds_min,
                triangle_bounds_max=triangle_bounds_max,
                triangle_bvh=_build_triangle_bvh(triangle_bounds_min, triangle_bounds_max),
                bounds_min=points.min(axis=0),
                bounds_max=points.max(axis=0),
                volume=0.0,
            )
        )
    occluders = _OccluderSet(
        occurrences=tuple(occurrences),
        bvh=_build_occurrence_bvh(
            np.vstack([occurrence.bounds_min for occurrence in occurrences]),
            np.vstack([occurrence.bounds_max for occurrence in occurrences]),
        ),
    )
    checked_node_ids: list[str] = []
    original_segment_intersects_occurrence = _segment_intersects_occurrence

    def record_occurrence_check(start_arg: np.ndarray, end_arg: np.ndarray, occurrence: _WorldOccurrence) -> bool:
        checked_node_ids.append(occurrence.node.id)
        return original_segment_intersects_occurrence(start_arg, end_arg, occurrence)

    monkeypatch.setattr("fascat.ops.occlusion._segment_intersects_occurrence", record_occurrence_check)
    start = np.array([254.25, 0.25, 1.0], dtype=float)
    end = np.array([254.25, 0.25, -1.0], dtype=float)

    assert _segment_blocked(start, end, occluders) is True
    assert checked_node_ids == ["node-127"]


def test_segment_blocked_skips_excluded_occluder_in_shared_bvh(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    faces = np.asarray([[0, 1, 2]], dtype=int)
    triangles = points[faces]
    triangle_bounds_min = triangles.min(axis=1)
    triangle_bounds_max = triangles.max(axis=1)
    occurrence = _WorldOccurrence(
        node=Node(id="candidate", name="Candidate", part_id="part"),
        part_id="part",
        world_points=points,
        faces=faces,
        triangles=triangles,
        triangle_edges1=triangles[:, 1] - triangles[:, 0],
        triangle_edges2=triangles[:, 2] - triangles[:, 0],
        triangle_bounds_min=triangle_bounds_min,
        triangle_bounds_max=triangle_bounds_max,
        triangle_bvh=_build_triangle_bvh(triangle_bounds_min, triangle_bounds_max),
        bounds_min=points.min(axis=0),
        bounds_max=points.max(axis=0),
        volume=0.0,
    )
    occluders = _OccluderSet(
        occurrences=(occurrence,),
        bvh=_build_occurrence_bvh(occurrence.bounds_min.reshape(1, 3), occurrence.bounds_max.reshape(1, 3)),
        excluded_node_id="candidate",
    )

    def fail_occurrence_check(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("excluded candidate should not be tested as its own occluder")

    monkeypatch.setattr("fascat.ops.occlusion._segment_intersects_occurrence", fail_occurrence_check)

    assert not occluders
    assert (
        _segment_blocked(
            np.asarray([0.25, 0.25, 1.0], dtype=float),
            np.asarray([0.25, 0.25, -1.0], dtype=float),
            occluders,
        )
        is False
    )


def test_remove_occluded_builds_occluder_bvh_once_per_operation(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
                Node(id="side", name="Side", part_id="side"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
            "side": Part(id="side", name="Side", mesh=_translated_mesh(_cube_mesh(0.5), 4.0)),
        },
    )
    calls = 0
    original = occlusion_module._build_occurrence_bvh

    def count_occurrence_bvh(
        bounds_min_all: np.ndarray,
        bounds_max_all: np.ndarray,
        indices: np.ndarray | None = None,
        *,
        depth: int = 0,
    ) -> object | None:
        nonlocal calls
        if depth == 0:
            calls += 1
        return original(bounds_min_all, bounds_max_all, indices, depth=depth)

    monkeypatch.setattr(occlusion_module, "_build_occurrence_bvh", count_occurrence_bvh)

    asset.remove_occluded(RemoveOccludedOptions(level="parts", preserve_cavities=False))

    assert calls == 1


def test_compact_material_slots_remaps_sparse_slots() -> None:
    mesh = _triangle_strip(3)
    mesh.material_indices = np.asarray([1, 3, 1], dtype=np.int32)
    part = Part(id="part", name="Part", mesh=mesh, material_ids=["unused-0", "a", "unused-2", "b"])

    occlusion_module._compact_material_slots(part, mesh)

    assert part.material_ids == ["a", "b"]
    np.testing.assert_array_equal(mesh.material_indices, np.asarray([0, 1, 0], dtype=np.int64))


@pytest.mark.parametrize(
    "material_indices",
    [
        np.asarray([], dtype=np.int64),
        np.asarray([-1, 1], dtype=np.int64),
        np.asarray([0, 2], dtype=np.int64),
    ],
)
def test_compact_material_slots_keeps_invalid_inputs_unchanged(material_indices: np.ndarray) -> None:
    mesh = _triangle_strip(int(material_indices.size))
    mesh.material_indices = material_indices.copy()
    part = Part(id="part", name="Part", mesh=mesh, material_ids=["a", "b"])
    original_material_ids = list(part.material_ids)
    original_material_indices = mesh.material_indices.copy()

    occlusion_module._compact_material_slots(part, mesh)

    assert part.material_ids == original_material_ids
    np.testing.assert_array_equal(mesh.material_indices, original_material_indices)


def test_remove_occluded_removes_contained_part_nodes() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
        },
    )

    visible = asset.remove_occluded(RemoveOccludedOptions(level="parts", preserve_cavities=False))

    assert visible.occurrence_count == 1
    assert "inner" not in visible.parts
    assert visible.metadata["removed_occluded_nodes"] == "1"
    assert visible.metadata["removed_occluded_triangles"] == "12"
    assert any("sampled visibility" in item for item in visible.report.steps[-1].warnings)


def test_remove_occluded_records_visibility_sampling_metadata() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="part", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=_cube_mesh(1.0))},
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="triangles", hemi_evaluation=True))

    warnings = result.report.steps[-1].warnings
    assert any("sampled visibility" in item for item in warnings)
    assert result.metadata["occlusion_level"] == "triangles"
    assert result.metadata["occlusion_hemi_evaluation"] == "true"
    assert int(result.metadata["occlusion_direction_count"]) < 26
    assert result.metadata["occlusion_candidate_count"] == "1"
    assert result.metadata["occlusion_face_count"] == "12"
    assert result.metadata["occlusion_sample_count"] == "12"
    assert result.metadata["occlusion_visible_sample_count"] == "12"
    assert result.metadata["occlusion_hidden_sample_count"] == "0"
    assert result.metadata["occlusion_sample_coverage"] == "1"
    assert result.metadata["occlusion_direction_coverage"] == "1"
    assert result.metadata["occlusion_confidence"] == "1"


def test_remove_occluded_records_lower_confidence_for_sparse_sampling() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="strip", name="Strip", part_id="strip")]),
        parts={"strip": Part(id="strip", name="Strip", mesh=_triangle_strip(100))},
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="parts", precision=1, strategy="conservative"))

    assert result.metadata["occlusion_face_count"] == "100"
    assert result.metadata["occlusion_sample_count"] == "15"
    assert result.metadata["occlusion_sample_coverage"] == "0.15"
    assert result.metadata["occlusion_direction_coverage"] == "0.230769"
    assert result.metadata["occlusion_confidence"] == "0.15"


def test_remove_occluded_respects_transparent_occluders() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0), material_ids=["glass"]),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
        },
        materials={"glass": Material(id="glass", name="Glass", base_color=(0.8, 0.9, 1.0, 0.35), opacity=0.35)},
    )

    visible = asset.remove_occluded(
        RemoveOccludedOptions(level="parts", preserve_cavities=False, consider_transparency_opaque=False)
    )
    hidden = asset.remove_occluded(
        RemoveOccludedOptions(level="parts", preserve_cavities=False, consider_transparency_opaque=True)
    )

    assert visible.occurrence_count == 2
    assert hidden.occurrence_count == 1
    assert "inner" not in hidden.parts


def test_remove_occluded_keeps_side_by_side_parts() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="left", name="Left", part_id="left"),
                Node(id="right", name="Right", part_id="right"),
            ],
        ),
        parts={
            "left": Part(id="left", name="Left", mesh=_translated_mesh(_cube_mesh(0.5), -1.0)),
            "right": Part(id="right", name="Right", mesh=_translated_mesh(_cube_mesh(0.5), 1.0)),
        },
    )

    visible = asset.remove_occluded(RemoveOccludedOptions(level="parts", preserve_cavities=False))

    assert visible.occurrence_count == 2
    assert visible.metadata["removed_occluded_nodes"] == "0"


def test_remove_occluded_triangle_level_removes_hidden_occurrence() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="inner", name="Inner", part_id="inner"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "inner": Part(id="inner", name="Inner", mesh=_cube_mesh(0.5)),
        },
    )

    visible = asset.remove_occluded(
        RemoveOccludedOptions(level="triangles", preserve_cavities=False, neighbors_preservation=0)
    )

    assert visible.occurrence_count == 1
    assert "inner" not in visible.parts
    assert visible.metadata["removed_occluded_triangles"] == "12"


def test_remove_occluded_submesh_level_removes_hidden_material_group() -> None:
    candidate_mesh = _merge_meshes(
        [
            (_cube_mesh(0.5), 0),
            (_translated_mesh(_cube_mesh(0.5), 4.0), 1),
        ]
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="outer", name="Outer", part_id="outer"),
                Node(id="candidate", name="Candidate", part_id="candidate"),
            ],
        ),
        parts={
            "outer": Part(id="outer", name="Outer", mesh=_cube_mesh(2.0)),
            "candidate": Part(
                id="candidate",
                name="Candidate",
                mesh=candidate_mesh,
                material_ids=["hidden", "visible"],
            ),
        },
        materials={
            "hidden": Material(id="hidden", name="Hidden", base_color=(1.0, 0.0, 0.0, 1.0)),
            "visible": Material(id="visible", name="Visible", base_color=(0.0, 1.0, 0.0, 1.0)),
        },
    )

    result = asset.remove_occluded(RemoveOccludedOptions(level="submeshes", preserve_cavities=False))
    part = result.parts["candidate"]

    assert part.mesh is not None
    assert part.mesh.triangle_count == 12
    assert part.material_ids == ["visible"]
    assert result.metadata["removed_occluded_triangles"] == "12"
