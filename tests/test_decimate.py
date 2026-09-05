from __future__ import annotations

import numpy as np
import pytest

import fascat.ops.decimate as decimate_module
from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import DecimateOptions

from ._actions_helpers import _triangle_strip


def test_decimation_step_options_are_computed_before_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _triangle_strip(2)
    source = Asset(
        root=Node(id="root", name="Root", children=[Node(id="part_node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )
    options = DecimateOptions(target_ratio=0.5, jobs=1)

    def fake_decimate(
        asset: Asset,
        _options: DecimateOptions,
        *,
        selected_part_ids: set[str] | None = None,
    ) -> Asset:
        assert selected_part_ids is None
        result = asset.copy(keep_source=True)
        result.parts.clear()
        return result

    monkeypatch.setattr(decimate_module, "decimate_asset", fake_decimate)
    result = source.decimate(options)

    strategy = result.report.steps[-1].options["target_strategy"]
    assert isinstance(strategy, dict)
    assert strategy["source_triangles"] == 2


def _triangle_strip_with_uvs(count: int) -> Mesh:
    mesh = _triangle_strip(count)
    mesh.uvs[0] = np.column_stack(
        (
            np.linspace(0.0, 1.0, mesh.vertex_count),
            np.zeros(mesh.vertex_count),
        )
    )
    mesh.tangents = np.tile(np.asarray([1.0, 0.0, 0.0, 1.0], dtype=float), (mesh.vertex_count, 1))
    return mesh


def test_decimate_uses_selection_budget() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(6))},
    )

    decimated = asset.decimate(DecimateOptions(target_triangles=3, target_ratio=None))

    # The strip's boundary faces are protected, so the target is not safely reachable.
    assert decimated.triangle_count == 6
    assert decimated.metadata["decimate_target_status"] == "unmet"
    assert decimated.report.steps[-1].name == "decimate"
    assert decimated.report.steps[-1].options["target_triangles"] == 3
    target_strategy = decimated.report.steps[-1].options["target_strategy"]
    assert target_strategy["kind"] == "target_count"
    assert target_strategy["source"] == "explicit_target_triangles"
    assert target_strategy["workflow"] == "unity_target_polygon_count"
    assert target_strategy["backend_mode"] == "mesh_simplify"
    assert target_strategy["source_triangles"] == 6
    assert target_strategy["target_triangles"] == 3
    assert target_strategy["target_ratio"] is None
    assert target_strategy["effective_keep_ratio"] == 0.5
    assert decimated.metadata["decimate_source_triangles"] == "6"
    assert decimated.metadata["decimate_output_triangles"] == "6"
    assert decimated.metadata["decimate_target_strategy"] == "target_count"
    assert decimated.metadata["decimate_target_strategy_source"] == "explicit_target_triangles"
    assert decimated.metadata["decimate_target_strategy_workflow"] == "unity_target_polygon_count"
    assert decimated.metadata["decimate_target_triangles"] == "3"
    assert decimated.metadata["decimate_effective_keep_ratio"] == "0.5"
    assert decimated.metadata["decimate_budget_allocation"] == "global_selection"
    assert decimated.metadata["decimate_allocation_targets"] == "body:3"
    assert decimated.metadata["decimate_allocated_target_triangles"] == "3"
    assert decimated.metadata["decimate_allocation_part_count"] == "1"
    assert decimated.metadata["decimate_allocation_reduced_parts"] == "1"
    assert decimated.parts["body"].metadata["decimate_allocated_target_triangles"] == "3"
    assert decimated.parts["body"].metadata["decimate_allocation_target_reduction"] == "0.5"
    assert decimated.metadata["decimate_estimated_memory_bytes"] == "30000"
    assert decimated.metadata["decimate_estimated_memory_gb"] == "3e-05"
    assert decimated.metadata["decimate_memory_rule_gb_per_million_triangles"] == "5"
    assert decimated.metadata["decimate_iterative_threshold_triangles"] == "1000000"
    assert decimated.metadata["decimate_iterative_recommended"] == "false"
    assert decimated.metadata["decimate_simplification_passes"] == "1"
    assert decimated.metadata["decimate_iterative_passes"] == "0"
    assert decimated.metadata["decimate_max_part_simplification_passes"] == "1"
    assert decimated.parts["body"].metadata["decimate_target_strategy"] == "target_count"
    assert decimated.report.steps[-1].after["decimate_source_triangles"] == 6
    assert decimated.report.steps[-1].after["decimate_output_triangles"] == 6
    assert decimated.report.steps[-1].after["decimate_estimated_memory_bytes"] == 30_000
    assert decimated.report.steps[-1].after["decimate_iterative_threshold_triangles"] == 1_000_000
    assert decimated.report.steps[-1].after["decimate_iterative_recommended"] == 0
    assert decimated.report.steps[-1].after["decimate_simplification_passes"] == 1
    assert decimated.report.steps[-1].after["decimate_iterative_passes"] == 0
    assert decimated.report.steps[-1].after["decimate_allocated_target_triangles"] == 3
    assert decimated.report.steps[-1].after["decimate_allocation_reduced_parts"] == 1
    assert decimated.parts["body"].metadata["decimate_error_metric"] == "symmetric_vertex_nearest_distance"


def test_decimate_reports_ratio_target_strategy() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(8))},
    )

    decimated = asset.decimate(DecimateOptions(target_ratio=0.25))
    target_strategy = decimated.report.steps[-1].options["target_strategy"]

    assert target_strategy["kind"] == "target_ratio"
    assert target_strategy["source"] == "explicit_target_ratio"
    assert target_strategy["workflow"] == "unity_target_ratio"
    assert target_strategy["source_triangles"] == 8
    assert target_strategy["target_ratio"] == 0.25
    assert target_strategy["effective_keep_ratio"] == 0.25
    assert decimated.metadata["decimate_target_strategy"] == "target_ratio"
    assert decimated.metadata["decimate_target_strategy_source"] == "explicit_target_ratio"
    assert decimated.metadata["decimate_target_strategy_workflow"] == "unity_target_ratio"
    assert decimated.metadata["decimate_target_ratio"] == "0.25"
    assert decimated.metadata["decimate_effective_keep_ratio"] == "0.25"


def test_decimate_reports_selection_target_allocation_by_part() -> None:
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="small_node", name="Small", part_id="small"),
                Node(id="dense_node", name="Dense", part_id="dense"),
            ],
        ),
        parts={
            "small": Part(id="small", name="Small", mesh=_triangle_strip(2)),
            "dense": Part(id="dense", name="Dense", mesh=_triangle_strip(8)),
        },
    )

    decimated = asset.decimate(DecimateOptions(target_triangles=8, target_ratio=None))
    step = decimated.report.steps[-1]

    assert decimated.metadata["decimate_allocation_targets"] == "dense:6,small:2"
    assert decimated.metadata["decimate_allocated_target_triangles"] == "8"
    assert decimated.metadata["decimate_allocation_part_count"] == "2"
    assert decimated.metadata["decimate_allocation_preserved_parts"] == "1"
    assert decimated.metadata["decimate_allocation_reduced_parts"] == "1"
    assert decimated.metadata["decimate_allocation_min_target_triangles"] == "2"
    assert decimated.metadata["decimate_allocation_max_target_triangles"] == "6"
    assert decimated.parts["small"].metadata["decimate_allocated_target_triangles"] == "2"
    assert decimated.parts["small"].metadata["decimate_allocation_target_reduction"] == "0"
    assert decimated.parts["dense"].metadata["decimate_allocated_target_triangles"] == "6"
    assert decimated.parts["dense"].metadata["decimate_allocation_target_reduction"] == "0.25"
    assert step.after["decimate_allocated_target_triangles"] == 8
    assert step.after["decimate_allocation_preserved_parts"] == 1
    assert step.after["decimate_allocation_reduced_parts"] == 1


@pytest.mark.parametrize("budget_scope", ["selection", "part"])
@pytest.mark.parametrize("use_ratio", [False, True])
def test_decimate_unreachable_budget_preserves_closed_textured_surface(budget_scope: str, use_ratio: bool) -> None:
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=2)
    mesh = Mesh(
        points=np.asarray(sphere.vertices),
        faces=np.asarray(sphere.faces),
        uvs={0: np.asarray(sphere.vertices[:, :2])},
        material_indices=(sphere.triangles_center[:, 0] > 0).astype(np.int64),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=mesh)},
    )

    decimated = asset.decimate(
        DecimateOptions(
            target_triangles=None if use_ratio else 80,
            target_ratio=0.25 if use_ratio else None,
            budget_scope=budget_scope,
            jobs=1,
        )
    )

    result = decimated.parts["body"].mesh
    assert result is not None
    result.validate()
    assert result.triangle_count == mesh.triangle_count == 320
    assert result.quality_metrics()["boundary_edges"] == 0
    assert result.to_trimesh().volume == pytest.approx(sphere.volume)
    assert result.material_indices is not None
    np.testing.assert_array_equal(np.sort(result.material_indices), np.sort(mesh.material_indices))
    source_indices = {tuple(point): index for index, point in enumerate(mesh.points)}
    for index, point in enumerate(result.points):
        np.testing.assert_array_equal(result.uvs[0][index], mesh.uvs[0][source_indices[tuple(point)]])
    assert decimated.metadata["decimate_output_triangles"] == "320"
    assert decimated.metadata["decimate_target_status"] == "unmet"
    assert decimated.metadata["decimate_unmet_target_parts"] == "1"
    assert decimated.parts["body"].metadata["decimate_target_status"] == "unmet"
    assert decimated.parts["body"].metadata["decimate_allocated_target_triangles"] == "80"
    warning = (
        "part body: decimation produced 320 triangles from 320; target 80 was not reached; "
        "retaining the safe simplification result"
    )
    assert warning in decimated.report.warnings
    assert warning in decimated.report.steps[-1].warnings


def test_decimate_ratio_budget_does_not_repeat_successful_reduction(monkeypatch: pytest.MonkeyPatch) -> None:
    import trimesh

    source = trimesh.creation.icosphere(subdivisions=2)
    coarse = trimesh.creation.icosphere(subdivisions=1)
    mesh = Mesh(points=np.asarray(source.vertices), faces=np.asarray(source.faces))
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=mesh)},
    )

    def simplify_to_closed_surface(self: Mesh, *, target_triangles: int, **_kwargs: object) -> Mesh:
        assert self.triangle_count == 320
        assert target_triangles == 80
        return Mesh(points=np.asarray(coarse.vertices), faces=np.asarray(coarse.faces))

    monkeypatch.setattr(Mesh, "simplify", simplify_to_closed_surface)
    decimated = asset.decimate(DecimateOptions(target_ratio=0.25, jobs=1))

    result = decimated.parts["body"].mesh
    assert result is not None
    assert result.triangle_count == 80
    assert result.quality_metrics()["boundary_edges"] == 0
    assert decimated.metadata["decimate_target_status"] == "met"
    assert decimated.metadata["decimate_unmet_target_parts"] == "0"
    assert decimated.parts["body"].metadata["decimate_target_status"] == "met"


def test_decimate_iterative_threshold_controls_runtime_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    targets: list[int] = []

    def fake_simplify(self: Mesh, *, target_triangles: int | None = None, **_kwargs: object) -> Mesh:
        assert target_triangles is not None
        targets.append(target_triangles)
        return _triangle_strip(target_triangles)

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(6))},
    )

    decimated = asset.decimate(DecimateOptions(target_triangles=3, target_ratio=None, iterative_threshold=5))
    warnings = decimated.report.steps[-1].warnings

    assert targets == [5, 3]
    assert decimated.metadata["decimate_iterative_recommended"] == "true"
    assert decimated.metadata["decimate_iterative_threshold_triangles"] == "5"
    assert decimated.metadata["decimate_simplification_passes"] == "2"
    assert decimated.metadata["decimate_iterative_passes"] == "1"
    assert decimated.parts["body"].metadata["decimate_simplification_passes"] == "2"
    assert decimated.parts["body"].metadata["decimate_iterative_passes"] == "1"
    assert decimated.report.steps[-1].after["decimate_iterative_recommended"] == 1
    assert decimated.report.steps[-1].after["decimate_iterative_threshold_triangles"] == 5
    assert decimated.report.steps[-1].after["decimate_simplification_passes"] == 2
    assert decimated.report.steps[-1].after["decimate_iterative_passes"] == 1
    assert any("iterative decimation is recommended at or above 5 triangles" in warning for warning in warnings)


def test_decimate_warns_when_lod0_ratio_is_aggressive() -> None:
    for options in (
        DecimateOptions(target_ratio=0.1),
        DecimateOptions(target_triangles=1, target_ratio=None),
    ):
        asset = Asset(
            root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
            parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(10))},
        )

        decimated = asset.decimate(options)
        warnings = decimated.report.steps[-1].warnings

        assert decimated.metadata["decimate_requested_keep_ratio"] == "0.1"
        assert any("ratios below 20% can visibly distort close-view LOD0 assets" in warning for warning in warnings)


def test_quality_decimate_records_measured_error_metrics() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=_triangle_strip(8))},
    )

    decimated = asset.decimate(
        DecimateOptions(
            criterion="quality",
            target_ratio=None,
            surface_tolerance=0.25,
            line_tolerance=0.1,
            uv_tolerance=0.05,
            budget_scope="part",
        )
    )

    part = decimated.parts["body"]
    target_strategy = decimated.report.steps[-1].options["target_strategy"]
    assert target_strategy["kind"] == "quality_error"
    assert target_strategy["source"] == "meshoptimizer_target_error"
    assert target_strategy["workflow"] == "meshoptimizer_target_error_hint"
    assert target_strategy["surface_tolerance"] == 0.25
    assert target_strategy["line_tolerance"] == 0.1
    assert target_strategy["uv_tolerance"] == 0.05
    assert target_strategy["quality_error_bound"] == 0.25
    assert target_strategy["quality_bound_status"] == "hint"
    assert target_strategy["quality_bound_enforced"] is False
    assert target_strategy["quality_bound_policy"] == "hint"
    assert part.metadata["decimate_criterion"] == "quality"
    assert part.metadata["decimate_target_strategy"] == "quality_error"
    assert part.metadata["decimate_target_strategy_source"] == "meshoptimizer_target_error"
    assert part.metadata["decimate_target_strategy_workflow"] == "meshoptimizer_target_error_hint"
    assert part.metadata["decimate_quality_bound_policy"] == "hint"
    assert part.metadata["decimate_quality_bound_status"] == "hint"
    assert part.metadata["decimate_quality_bound_enforced"] == "false"
    assert part.metadata["decimate_quality_error_bound"] == "0.25"
    assert part.metadata["decimate_source_triangles"] == "8"
    assert int(part.metadata["decimate_output_triangles"]) <= 8
    assert float(part.metadata["decimate_triangle_reduction"]) >= 0.0
    assert decimated.metadata["decimate_budget_allocation"] == "per_part"
    assert decimated.metadata["decimate_target_strategy"] == "quality_error"
    assert decimated.metadata["decimate_quality_bound_policy"] == "hint"
    assert decimated.metadata["decimate_quality_bound_status"] == "hint"
    assert decimated.metadata["decimate_quality_bound_enforced"] == "false"
    assert decimated.metadata["decimate_quality_error_bound"] == "0.25"
    assert decimated.report.steps[-1].warnings == []


def test_decimate_uv_importance_controls_texture_coordinate_cleanup() -> None:
    cases = (
        ("preserve_islands", True, None),
        ("preserve_seams", False, "0"),
        ("ignore", False, "0"),
    )
    for mode, keeps_uvs, removed_channels in cases:
        asset = Asset(
            root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
            parts={"body": Part(id="body", name="Body", mesh=_triangle_strip_with_uvs(6))},
        )

        decimated = asset.decimate(DecimateOptions(target_ratio=0.5, uv_importance=mode))  # type: ignore[arg-type]
        mesh = decimated.parts["body"].mesh

        assert mesh is not None
        assert (0 in mesh.uvs) is keeps_uvs
        if not keeps_uvs:
            assert mesh.tangents is None
            assert decimated.parts["body"].metadata["decimate_removed_uv_channels"] == removed_channels
            assert decimated.metadata["decimate_removed_uv_channels"] == removed_channels
        assert decimated.parts["body"].metadata["decimate_uv_importance"] == mode
        assert decimated.metadata["decimate_uv_importance"] == mode


def test_decimate_cleanup_attributes_remove_unused_uvs_and_report_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_simplify(self: Mesh, *, target_triangles: int | None = None, **_kwargs: object) -> Mesh:
        assert target_triangles == 3
        assert sorted(self.uvs) == [0]
        assert self.tangents is None
        return self.copy()

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)
    mesh = _triangle_strip(6)
    mesh.uvs[0] = mesh.points[:, :2].copy()
    mesh.uvs[1] = np.zeros((mesh.vertex_count, 2), dtype=float)
    mesh.tangents = np.tile(np.asarray([1.0, 0.0, 0.0, 1.0], dtype=float), (mesh.vertex_count, 1))
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=mesh)},
    )

    decimated = asset.decimate(DecimateOptions(target_ratio=0.5, cleanup_attributes=("unused_uvs", "tangents")))
    part = decimated.parts["body"]
    output_mesh = part.mesh
    warnings = decimated.report.steps[-1].warnings

    assert output_mesh is not None
    assert sorted(output_mesh.uvs) == [0]
    assert output_mesh.tangents is None
    assert part.metadata["decimate_pre_cleanup_attributes"] == "unused_uvs,tangents"
    assert part.metadata["decimate_pre_cleanup_removed_uv_channels"] == "1"
    assert part.metadata["decimate_pre_cleanup_removed_tangents"] == "true"
    assert part.metadata["decimate_preserved_uv_channels"] == "0"
    assert part.metadata["decimate_uv_constraint_status"] == "preserved_for_simplification"
    assert decimated.metadata["decimate_pre_cleanup_removed_uv_channels"] == "1"
    assert decimated.metadata["decimate_pre_cleanup_removed_tangent_parts"] == "1"
    assert decimated.metadata["decimate_uv_constrained_parts"] == "1"
    assert decimated.report.steps[-1].after["decimate_pre_cleanup_removed_tangent_parts"] == 1
    assert any("preserved texture coordinates can reduce simplification efficiency" in warning for warning in warnings)


def test_decimate_reports_topology_protection_metrics() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.asarray([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["red", "blue"])},
    )

    decimated = asset.decimate(DecimateOptions(target_ratio=0.5))
    part = decimated.parts["panel"]
    step = decimated.report.steps[-1]

    assert part.metadata["decimate_protect_hole_boundary_faces"] == "2"
    assert part.metadata["decimate_protect_material_boundary_faces"] == "2"
    assert part.metadata["decimate_protect_silhouette_faces"] == "2"
    assert part.metadata["decimate_protect_total_feature_faces"] == "2"
    assert decimated.metadata["decimate_protected_feature_parts"] == "1"
    assert decimated.metadata["decimate_protect_total_feature_faces"] == "2"
    assert step.after["decimate_protected_feature_parts"] == 1
    assert step.after["decimate_protect_total_feature_faces"] == 2


def test_decimate_preserves_painted_and_ambient_occlusion_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[set[int]] = []

    def fake_simplify(
        self: Mesh,
        *,
        protected_faces: np.ndarray | None = None,
        **_kwargs: object,
    ) -> Mesh:
        captured.append(set() if protected_faces is None else set(protected_faces.astype(int).tolist()))
        return self.copy()

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)
    strategies: list[str] = []

    def fake_ambient_occlusion(_mesh: Mesh, strategy: str = "conservative") -> np.ndarray:
        strategies.append(strategy)
        return np.asarray([1.0, 0.9, 0.7, 0.2])

    monkeypatch.setattr(decimate_module, "face_ambient_occlusion", fake_ambient_occlusion)
    mesh = _triangle_strip(4)
    mesh.face_groups["painted_area"] = np.asarray([1], dtype=int)
    mesh.metadata["decimate_protected_faces"] = "2, 99, invalid"
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="body", name="Body", part_id="body")]),
        parts={"body": Part(id="body", name="Body", mesh=mesh)},
    )

    decimated = asset.decimate(
        DecimateOptions(
            target_ratio=0.5,
            protect_topology=False,
            preserve_painted_areas=True,
            preserve_ambient_occlusion=True,
            ambient_occlusion_strategy="advanced",
        )
    )
    part = decimated.parts["body"]
    step = decimated.report.steps[-1]

    assert captured == [{1, 2, 3}]
    assert strategies
    assert set(strategies) == {"advanced"}
    assert part.metadata["decimate_protect_painted_area_faces"] == "2"
    assert part.metadata["decimate_protect_ambient_occlusion_faces"] == "1"
    assert part.metadata["decimate_protect_importance_faces"] == "3"
    assert decimated.metadata["decimate_protect_painted_area_faces"] == "2"
    assert decimated.metadata["decimate_protect_ambient_occlusion_faces"] == "1"
    assert decimated.metadata["decimate_protect_importance_faces"] == "3"
    assert step.after["decimate_protect_painted_area_faces"] == 2
    assert step.after["decimate_protect_ambient_occlusion_faces"] == 1
    assert step.after["decimate_protect_importance_faces"] == 3
    assert step.options["preserve_painted_areas"] is True
    assert step.options["preserve_ambient_occlusion"] is True
    assert step.options["ambient_occlusion_strategy"] == "advanced"
