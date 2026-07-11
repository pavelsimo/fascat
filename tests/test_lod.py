from __future__ import annotations

import json

import numpy as np
import pytest

import fascat.ops.lod as lod_module
from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import LODGeneratorOptions, LODLevel, LODOptions

from ._actions_helpers import _cube_mesh


def test_lod_unique_ids_preserve_first_suffix_start() -> None:
    assert lod_module._unique_id("proxy", {}) == "proxy"
    assert lod_module._unique_id("proxy", {"proxy": object()}) == "proxy_1"
    assert lod_module._unique_id("proxy", {"proxy": object(), "proxy_1": object()}) == "proxy_2"


def _asset_with_imported_lod(*, valid: bool = True) -> Asset:
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    base = Mesh(points=points, faces=np.array([[0, 1, 2], [0, 1, 3], [1, 3, 4]], dtype=int))
    lod_faces = [[0, 1, 2], [0, 1, 3]] if valid else [[0, 1, 2], [0, 1, 3], [1, 3, 4]]
    lod = Mesh(points=points, faces=np.array(lod_faces, dtype=int), metadata={"lod_source": "imported"})
    part = Part(id="part", name="Part", mesh=base, lod_meshes=[lod])
    return Asset(root=Node(id="root", name="root", part_id="part"), parts={"part": part})


def test_lods_auto_retains_valid_imported_chain_and_applies_matching_coverage() -> None:
    result = _asset_with_imported_lod().lods(LODOptions((0.5,), source="auto", screen_coverage=(0.25,)))
    assert result.parts["part"].lod_meshes[0].triangle_count == 2
    assert result.parts["part"].lod_meshes[0].metadata["lod_screen_coverage"] == "0.25"
    assert result.parts["part"].metadata["lod_status"] == "retained_imported"


def test_lods_retained_imported_chain_uses_distinct_report_counters() -> None:
    result = _asset_with_imported_lod().lods(LODOptions((0.5,), source="auto"))
    part = result.parts["part"]
    retained = part.lod_meshes[0]
    step = result.report.steps[-1]

    assert result.metadata["lod_added_vertices"] == "0"
    assert result.metadata["lod_added_triangles"] == "0"
    assert result.metadata["lod_added_mesh_bytes"] == "0"
    assert result.metadata["lod_retained_vertices"] == str(retained.vertex_count)
    assert result.metadata["lod_retained_triangles"] == str(retained.triangle_count)
    assert int(result.metadata["lod_retained_mesh_bytes"]) > 0
    assert result.metadata["lod_chain_triangles"] == str(part.mesh.triangle_count + retained.triangle_count)
    assert part.metadata["lod_added_triangles"] == "0"
    assert part.metadata["lod_retained_triangles"] == str(retained.triangle_count)
    assert step.after["lod_added_triangles"] == 0
    assert step.after["lod_retained_triangles"] == retained.triangle_count


def test_lods_imported_coverage_mismatch_preserves_existing_metadata() -> None:
    asset = _asset_with_imported_lod()
    part = asset.parts["part"]
    part.metadata["lod_screen_coverage"] = "0.4"
    part.lod_meshes[0].metadata["lod_screen_coverage"] = "0.4"

    result = asset.lods(LODOptions((0.5, 0.25), source="auto", screen_coverage=(0.5, 0.25)))
    retained = result.parts["part"]

    assert retained.metadata["lod_screen_coverage"] == "0.4"
    assert retained.lod_meshes[0].metadata["lod_screen_coverage"] == "0.4"
    assert any("LOD screen coverage was not applied" in warning for warning in result.report.warnings)


def test_lods_auto_replaces_invalid_imported_chain_with_diagnostic() -> None:
    result = _asset_with_imported_lod(valid=False).lods(LODOptions((0.5,), source="auto"))
    assert result.parts["part"].metadata["lod_ratios"] == "0.5"
    assert any("Imported LOD chain is invalid" in warning for warning in result.report.warnings)


def test_lods_imported_retains_chain_without_generation() -> None:
    result = _asset_with_imported_lod(valid=False).lods(LODOptions((0.5,), source="imported"))
    assert result.parts["part"].lod_meshes[0].triangle_count == 3
    assert result.metadata["lod_generated_parts"] == "0"


def test_lods_are_monotonic() -> None:
    points = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=int,
    )
    mesh = Mesh(points=points, faces=faces)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            (0.75, 0.5, 0.25),
            screen_coverage=(0.6, 0.3, 0.12),
            per_part_budget=True,
            validate=True,
        )
    )
    part = with_lods.parts["cube"]
    counts = [mesh.triangle_count, *[lod.triangle_count for lod in part.lod_meshes]]
    step = with_lods.report.steps[-1]

    assert counts == sorted(counts, reverse=True)
    assert part.metadata["lod_screen_coverage"] == "0.6,0.3,0.12"
    assert part.metadata["lod_per_part_budget"] == "true"
    assert part.lod_meshes[0].metadata["lod_screen_coverage"] == "0.6"
    assert step.before["lod_meshes"] == 0
    assert step.before["lod_triangles"] == 0
    assert step.after["lod_meshes"] == 3
    assert step.after["lod_vertices"] == sum(lod.vertex_count for lod in part.lod_meshes)
    assert step.after["lod_triangles"] == sum(counts[1:])
    assert part.metadata["lod_source_vertices"] == str(mesh.vertex_count)
    assert part.metadata["lod_source_triangles"] == str(mesh.triangle_count)
    assert part.metadata["lod_added_vertices"] == str(sum(lod.vertex_count for lod in part.lod_meshes))
    assert part.metadata["lod_added_triangles"] == str(sum(counts[1:]))
    assert part.metadata["lod_chain_triangles"] == str(sum(counts))
    assert part.metadata["lod_level_triangles"] == ",".join(str(count) for count in counts[1:])
    assert float(part.metadata["lod_triangle_multiplier"]) > 1.0
    assert int(part.metadata["lod_added_mesh_bytes"]) > 0
    assert part.metadata["lod_level_instance_reuse"] == "not_applicable,not_applicable,not_applicable"
    assert part.metadata["lod_level_material_merge"] == "not_run,not_run,not_run"
    assert part.metadata["lod_level_texture_bake"] == "not_run,not_run,not_run"
    assert part.metadata["lod_level_culling_granularity"] == "part,part,part"
    assert part.metadata["lod_level_policy_advisory"] == (
        "conservative_geometry,conservative_geometry,progressive_geometry"
    )
    assert with_lods.metadata["lod_source_triangles"] == str(mesh.triangle_count)
    assert with_lods.metadata["lod_added_triangles"] == str(sum(counts[1:]))
    assert with_lods.metadata["lod_chain_triangles"] == str(sum(counts))
    assert with_lods.metadata["lod_reused_instance_levels"] == "0"
    assert with_lods.metadata["lod_material_merged_levels"] == "0"
    assert with_lods.metadata["lod_texture_baked_levels"] == "0"
    assert with_lods.metadata["lod_culling_changed_levels"] == "0"
    assert with_lods.metadata["lod_level_policy_advisory"] == (
        "conservative_geometry,conservative_geometry,progressive_geometry"
    )
    assert step.after["lod_source_triangles"] == mesh.triangle_count
    assert step.after["lod_added_triangles"] == sum(counts[1:])
    assert step.after["lod_chain_triangles"] == sum(counts)
    assert step.after["lod_added_mesh_bytes"] == int(with_lods.metadata["lod_added_mesh_bytes"])
    assert step.after["lod_reused_instance_levels"] == 0
    assert step.after["lod_advisory_count"] == 0


def test_lods_simplify_progressively_from_previous_level(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0]],
        dtype=float,
    )
    faces = np.array(
        [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3], [4, 6, 5], [6, 7, 5], [0, 2, 3], [4, 3, 5]],
        dtype=int,
    )
    mesh = Mesh(points=points, faces=faces)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="strip")]),
        parts={"strip": Part(id="strip", name="Strip", mesh=mesh)},
    )
    calls: list[tuple[int, int | None, float | None]] = []

    def fake_simplify(
        self: Mesh,
        *,
        target_triangles: int | None = None,
        ratio: float | None = None,
        **_kwargs: object,
    ) -> Mesh:
        calls.append((self.triangle_count, target_triangles, ratio))
        target = self.triangle_count if target_triangles is None else min(target_triangles, self.triangle_count)
        return self._filter_faces(np.arange(target, dtype=np.int64)).remove_unreferenced_vertices()

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)

    with_lods = asset.lods(LODOptions((0.5, 0.25, 0.125)))
    part = with_lods.parts["strip"]

    assert calls == [(8, 4, None), (4, 2, None), (2, 1, None)]
    assert [lod.triangle_count for lod in part.lod_meshes] == [4, 2, 1]
    assert part.metadata["lod_level_simplification_source"] == "source,previous,previous"
    assert [lod.metadata["lod_simplification_source"] for lod in part.lod_meshes] == [
        "source",
        "previous",
        "previous",
    ]


def test_lods_marks_simplification_source_when_retry_enforces_monotonicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    points = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0], [3, 0, 0], [3, 1, 0]],
        dtype=float,
    )
    faces = np.array(
        [[0, 1, 2], [1, 3, 2], [1, 4, 3], [4, 5, 3], [4, 6, 5], [6, 7, 5], [0, 2, 3], [4, 3, 5]],
        dtype=int,
    )
    mesh = Mesh(points=points, faces=faces)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="strip")]),
        parts={"strip": Part(id="strip", name="Strip", mesh=mesh)},
    )
    calls: list[tuple[int, int | None]] = []

    def fake_simplify(
        self: Mesh,
        *,
        target_triangles: int | None = None,
        **_kwargs: object,
    ) -> Mesh:
        calls.append((self.triangle_count, target_triangles))
        if self.triangle_count == 4 and target_triangles == 2:
            return mesh
        target = self.triangle_count if target_triangles is None else min(target_triangles, self.triangle_count)
        return self._filter_faces(np.arange(target, dtype=np.int64)).remove_unreferenced_vertices()

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)

    with_lods = asset.lods(LODOptions((0.5, 0.25)))
    part = with_lods.parts["strip"]

    assert calls == [(8, 4), (4, 2), (8, 4)]
    assert [lod.triangle_count for lod in part.lod_meshes] == [4, 4]
    assert part.metadata["lod_level_simplification_source"] == "source,previous_retry"
    assert [lod.metadata["lod_simplification_source"] for lod in part.lod_meshes] == [
        "source",
        "previous_retry",
    ]


def test_lods_report_level_policy_for_reused_instances() -> None:
    mesh = _triangle_mesh()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="Node A", part_id="shared"),
                Node(id="node_b", name="Node B", part_id="shared"),
            ],
        ),
        parts={"shared": Part(id="shared", name="Shared", mesh=mesh)},
    )

    with_lods = asset.lods(LODOptions((0.5, 0.25)))
    part = with_lods.parts["shared"]
    step = with_lods.report.steps[-1]

    assert len(part.lod_meshes) == 2
    assert part.metadata["lod_occurrences"] == "2"
    assert part.metadata["lod_level_instance_reuse"] == "preserved,preserved"
    assert part.metadata["lod_level_material_merge"] == "not_run,not_run"
    assert part.metadata["lod_level_texture_bake"] == "not_run,not_run"
    assert part.metadata["lod_level_culling_granularity"] == "part,part"
    assert part.lod_meshes[0].metadata["lod_instance_reuse"] == "preserved"
    assert with_lods.metadata["lod_reused_instance_levels"] == "2"
    assert with_lods.metadata["lod_material_merged_levels"] == "0"
    assert with_lods.metadata["lod_texture_baked_levels"] == "0"
    assert with_lods.metadata["lod_culling_changed_levels"] == "0"
    assert step.after["lod_reused_instance_levels"] == 2


def test_lods_can_omit_tiny_parts_at_lower_screen_coverage() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=int,
        ),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            ratios=(0.5, 0.25),
            screen_coverage=(1.0, 0.1),
            drop_tiny_parts=True,
            tiny_part_screen_size=1.0,
        )
    )

    assert with_lods.parts["cube"].lod_meshes[0].triangle_count > 0
    assert with_lods.parts["cube"].lod_meshes[1].triangle_count == 0
    assert with_lods.parts["cube"].lod_meshes[1].metadata["lod_omitted"] == "tiny_part"
    assert with_lods.parts["cube"].lod_meshes[1].metadata["lod_instance_reuse"] == "omitted"
    assert with_lods.parts["cube"].lod_meshes[1].metadata["lod_culling_granularity"] == "omitted_tiny_part"
    assert with_lods.parts["cube"].metadata["lod_level_instance_reuse"] == "not_applicable,omitted"
    assert with_lods.parts["cube"].metadata["lod_level_culling_granularity"] == "part,omitted_tiny_part"
    assert with_lods.parts["cube"].metadata["lod_omitted_tiny_part_meshes"] == "1"
    assert with_lods.parts["cube"].metadata["lod_culling_changed_levels"] == "1"
    assert with_lods.metadata["lod_omitted_tiny_part_meshes"] == "1"
    assert with_lods.metadata["lod_culling_changed_levels"] == "1"


def test_lods_report_chain_advisories_for_risky_levels() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=int,
        ),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            ratios=(0.3, 0.15, 0.1, 0.05, 0.025),
            screen_coverage=(0.7, 0.4, 0.2, 0.1, 0.05),
        )
    )
    part = with_lods.parts["cube"]
    advisories = json.loads(str(with_lods.metadata["lod_advisories"]))
    step = with_lods.report.steps[-1]

    assert with_lods.metadata["lod_advisory_count"] == "3"
    assert with_lods.metadata["lod_advisory_codes"] == (
        "excessive_lod_levels,aggressive_close_view_lods,far_lod_proxy_recommended"
    )
    assert [item["code"] for item in advisories] == [
        "excessive_lod_levels",
        "aggressive_close_view_lods",
        "far_lod_proxy_recommended",
    ]
    assert part.metadata["lod_level_policy_advisory"] == (
        "close_view_too_aggressive,mid_view_too_aggressive,"
        "progressive_geometry,progressive_geometry,far_proxy_recommended"
    )
    assert part.lod_meshes[-1].metadata["lod_policy_advisory"] == "far_proxy_recommended"
    assert step.after["lod_advisory_count"] == 3
    assert len(step.warnings) == 3
    assert "3-4 levels are usually enough" in step.warnings[0]
    assert "keep LOD1 and LOD2 visually conservative" in step.warnings[1]
    assert "enable far_lod_bake" in step.warnings[2]


def test_lods_warn_when_ratio_can_collapse_to_one_triangle() -> None:
    mesh = _triangle_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="tri")]),
        parts={"tri": Part(id="tri", name="Triangle", mesh=mesh)},
    )

    with_lods = asset.lods(LODOptions(ratios=(0.01,), screen_coverage=(0.02,)))
    advisories = json.loads(str(with_lods.metadata["lod_advisories"]))
    warnings = with_lods.report.steps[-1].warnings

    assert with_lods.parts["tri"].lod_meshes[0].triangle_count == 1
    assert "destructive_lod_ratio_floor" in with_lods.metadata["lod_advisory_codes"]
    floor_advisory = next(item for item in advisories if item["code"] == "destructive_lod_ratio_floor")
    assert floor_advisory["levels"] == [
        {
            "level": 1,
            "ratio": 0.01,
            "minimum_recommended_ratio": 0.01,
            "screen_coverage": 0.02,
        }
    ]
    assert any("commonly collapse to a one-triangle LOD" in warning for warning in warnings)


def test_lods_can_bake_far_level_material_policy() -> None:
    mesh = _triangle_mesh()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="Node A", part_id="shared"),
                Node(id="node_b", name="Node B", part_id="shared"),
            ],
        ),
        parts={"shared": Part(id="shared", name="Shared", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            ratios=(0.5, 0.1),
            screen_coverage=(0.5, 0.05),
            engine_profile="unity",
            far_lod_bake=True,
        )
    )
    part = with_lods.parts["shared"]
    far_lod = part.lod_meshes[-1]

    assert part.metadata["lod_engine_profile"] == "unity"
    assert part.metadata["lod_far_lod_bake"] == "true"
    assert part.metadata["lod_level_material_merge"] == "not_run,merged"
    assert part.metadata["lod_level_texture_bake"] == "not_run,baked"
    assert part.metadata["lod_switching_validation_status"] == "monotonic"
    assert far_lod.material_indices is not None
    assert far_lod.material_indices.tolist() == [0]
    assert far_lod.metadata["lod_far_bake"] == "one_material"
    assert with_lods.metadata["lod_material_merged_levels"] == "1"
    assert with_lods.metadata["lod_texture_baked_levels"] == "1"


def test_lods_apply_engine_specific_export_modes() -> None:
    mesh = _triangle_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    unity = asset.lods(LODOptions((0.5,), engine_profile="unity", mode="separate"))
    unreal = asset.lods(LODOptions((0.5,), engine_profile="unreal"))

    assert unity.metadata["lod_mode"] == "separate"
    assert unity.metadata["lod_export_mode"] == "variants"
    assert unity.parts["part"].lod_meshes[0].metadata["lod_export_mode"] == "variants"
    assert unreal.metadata["lod_engine_profile"] == "unreal"
    assert unreal.metadata["lod_export_mode"] == "separate"
    assert unreal.parts["part"].lod_meshes[0].metadata["lod_export_mode"] == "separate"


def test_lods_allow_per_level_switch_distance_overrides() -> None:
    mesh = _triangle_mesh()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    with_lods = asset.lods(
        LODOptions(
            ratios=(0.5, 0.25),
            screen_coverage=(0.5, 0.25),
            switch_distance_overrides=(12.0, None),
        )
    )
    part = with_lods.parts["part"]
    distances = [float(value) for value in part.metadata["lod_level_switch_distances"].split(",")]

    assert distances[0] == pytest.approx(12.0)
    assert distances[1] == pytest.approx(np.sqrt(2.0) / 0.25)
    assert part.metadata["lod_level_switch_distance_sources"] == "override,formula"
    assert with_lods.metadata["lod_level_switch_distance_sources"] == "override,formula"
    assert part.lod_meshes[0].metadata["lod_switch_distance"] == "12"
    assert part.lod_meshes[0].metadata["lod_switch_distance_source"] == "override"
    assert part.lod_meshes[1].metadata["lod_switch_distance_source"] == "formula"
    assert with_lods.report.steps[-1].options["switch_distance_overrides"] == [12.0, None]


def test_lods_can_build_scene_level_far_proxy() -> None:
    translate = np.eye(4, dtype=float)
    translate[0, 3] = 2.0
    mesh = _triangle_mesh()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="Node A", part_id="a"),
                Node(id="node_b", name="Node B", part_id="b", transform=translate),
            ],
        ),
        parts={
            "a": Part(id="a", name="A", mesh=mesh),
            "b": Part(id="b", name="B", mesh=mesh),
        },
    )

    with_lods = asset.lods(
        LODOptions(
            ratios=(0.5, 0.1),
            screen_coverage=(0.5, 0.05),
            far_lod_bake=True,
            scene_far_proxy=True,
        )
    )
    proxy = with_lods.parts[str(with_lods.metadata["lod_scene_far_proxy_part_id"])]
    step = with_lods.report.steps[-1]

    assert with_lods.metadata["lod_scene_far_proxy"] == "created"
    assert with_lods.metadata["lod_scene_far_proxy_draw_calls"] == "1"
    assert with_lods.metadata["lod_scene_far_proxy_source_parts"] == "2"
    assert with_lods.metadata["lod_scene_far_proxy_source_occurrences"] == "2"
    assert proxy.material_ids == [with_lods.metadata["lod_scene_far_proxy_material_id"]]
    assert proxy.mesh is not None
    assert proxy.mesh.triangle_count == 2
    assert proxy.mesh.material_indices is not None
    assert proxy.mesh.material_indices.tolist() == [0, 0]
    assert proxy.mesh.metadata["lod_material_merge"] == "scene_merged"
    assert proxy.mesh.metadata["lod_texture_bake"] == "scene_baked"
    assert proxy.mesh.points[:, 0].max() == pytest.approx(3.0)
    assert step.after["lod_scene_far_proxy_draw_calls"] == 1
    assert step.after["lod_scene_far_proxy_triangles"] == 2


def test_lods_warn_when_selected_parts_have_no_mesh() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="mesh_node", name="Mesh", part_id="mesh"),
                Node(id="empty_node", name="Untessellated", part_id="empty"),
            ],
        ),
        parts={
            "mesh": Part(id="mesh", name="Mesh", mesh=mesh),
            "empty": Part(id="empty", name="Untessellated", mesh=None),
        },
    )

    with_lods = asset.lods(LODOptions((0.5,)))
    warnings = with_lods.report.steps[-1].warnings

    assert len(with_lods.parts["mesh"].lod_meshes) == 1
    assert with_lods.parts["empty"].lod_meshes == []
    assert with_lods.parts["empty"].metadata["lod_status"] == "skipped_no_mesh"
    assert with_lods.metadata["lod_generated_parts"] == "1"
    assert with_lods.metadata["lod_skipped_no_mesh_parts"] == "1"
    assert len(warnings) == 1
    assert "LOD generation skipped part without tessellated mesh: Untessellated" in warnings[0]


def test_lods_warn_when_no_tessellated_parts_match() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="empty_node", name="Untessellated", part_id="empty")]),
        parts={"empty": Part(id="empty", name="Untessellated", mesh=None)},
    )

    with_lods = asset.lods(LODOptions((0.5,)))
    warnings = with_lods.report.steps[-1].warnings

    assert with_lods.parts["empty"].lod_meshes == []
    assert with_lods.parts["empty"].metadata["lod_status"] == "skipped_no_mesh"
    assert with_lods.metadata["lod_generated_parts"] == "0"
    assert with_lods.metadata["lod_skipped_no_mesh_parts"] == "1"
    assert len(warnings) == 2
    assert "LOD generation skipped part without tessellated mesh: Untessellated" in warnings[0]
    assert warnings[1] == "LOD generation matched no tessellated mesh-bearing parts"


def _triangle_mesh() -> Mesh:
    return Mesh(
        points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )


def test_run_lod_generators_records_screen_coverage_metadata() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="cube", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=_cube_mesh())},
    )

    with_lods = asset.run_lod_generators(
        LODGeneratorOptions(
            preset="vr",
            levels=(LODLevel(screen_coverage=0.5, target_ratio=0.5), LODLevel(0.2, 0.25)),
            validate=True,
        )
    )

    assert len(with_lods.parts["cube"].lod_meshes) == 2
    assert with_lods.parts["cube"].metadata["lod_screen_coverage"] == "0.5,0.2"
    assert with_lods.parts["cube"].lod_meshes[0].metadata["lod_screen_coverage"] == "0.5"
    assert with_lods.parts["cube"].lod_meshes[1].metadata["lod_screen_coverage"] == "0.2"
    assert with_lods.report.steps[-1].name == "run_lod_generators"


def test_run_lod_generators_propagates_switch_distance_override() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="cube", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=_cube_mesh())},
    )

    with_lods = asset.run_lod_generators(
        LODGeneratorOptions(levels=(LODLevel(screen_coverage=0.5, target_ratio=0.5, switch_distance_override=30.0),))
    )
    lod = with_lods.parts["cube"].lod_meshes[0]

    assert lod.metadata["lod_switch_distance"] == "30"
    assert lod.metadata["lod_switch_distance_source"] == "override"
    assert with_lods.parts["cube"].metadata["lod_level_switch_distance_sources"] == "override"
    assert with_lods.report.steps[-1].options["levels"][0]["switch_distance_override"] == 30.0


def test_lods_accepts_generator_options_as_primary_entry_point() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="cube", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=_cube_mesh())},
    )

    with_lods = asset.lods(
        LODGeneratorOptions(
            preset="vr",
            levels=(LODLevel(screen_coverage=0.5, target_ratio=0.5),),
            validate=True,
        )
    )

    assert len(with_lods.parts["cube"].lod_meshes) == 1
    assert with_lods.parts["cube"].metadata["lod_screen_coverage"] == "0.5"
    assert with_lods.report.steps[-1].name == "run_lod_generators"


def test_lods_rejects_generator_options_with_ratio_kwargs() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="cube", name="Cube", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=_cube_mesh())},
    )

    with pytest.raises(TypeError, match="LOD generator options"):
        asset.lods(LODGeneratorOptions(), ratios=(0.5,))


def test_run_lod_generators_warns_for_untessellated_parts() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="empty", name="Untessellated", part_id="empty")]),
        parts={"empty": Part(id="empty", name="Untessellated", mesh=None)},
    )

    with_lods = asset.run_lod_generators(LODGeneratorOptions(levels=(LODLevel(screen_coverage=0.5, target_ratio=0.5),)))
    warnings = with_lods.report.steps[-1].warnings

    assert with_lods.parts["empty"].metadata["lod_status"] == "skipped_no_mesh"
    assert with_lods.metadata["lod_skipped_no_mesh_parts"] == "1"
    assert with_lods.report.steps[-1].name == "run_lod_generators"
    assert any("LOD generation skipped part without tessellated mesh" in warning for warning in warnings)
    assert any("matched no tessellated mesh-bearing parts" in warning for warning in warnings)
