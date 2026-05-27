from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fascat.analysis import analyze_output
from fascat.asset import Asset, Node, Part
from fascat.filter import Filter
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import AnalyzeOptions


def _quality_asset() -> Asset:
    main_mesh = Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.1, 0.01, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
        faces=np.asarray(
            [
                [0, 1, 2],
                [1, 0, 3],
                [0, 1, 4],
                [0, 2, 2],
            ],
            dtype=int,
        ),
        material_indices=np.asarray([0, 1, 1, 0], dtype=int),
    )
    tiny_mesh = Mesh(
        points=np.asarray([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="main_node", name="Main", part_id="main"),
                Node(id="tiny_node", name="Tiny", part_id="tiny"),
            ],
        ),
        parts={
            "main": Part(id="main", name="Main", mesh=main_mesh, material_ids=["red", "blue"]),
            "tiny": Part(id="tiny", name="Tiny", mesh=tiny_mesh, material_ids=["red"]),
        },
        materials={
            "red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0)),
            "blue": Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )


def _asset_from_mesh(mesh: Mesh) -> Asset:
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="part_node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )


def test_asset_analyze_reports_geometry_quality_and_visual_risks(tmp_path: Path) -> None:
    asset = _quality_asset()

    report = asset.analyze(
        AnalyzeOptions(
            non_manifold_edges=True,
            open_boundaries=True,
            self_intersections=True,
            sliver_triangles=True,
            tiny_parts=True,
            draw_call_estimate=True,
            visual_risk=True,
            sliver_aspect_ratio=10.0,
            tiny_part_diagonal=0.02,
        )
    )
    output = tmp_path / "analysis.json"
    report.write_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["summary"]["non_manifold_edges"] >= 1
    assert payload["summary"]["open_boundaries"] > 0
    assert payload["summary"]["boundary_edges"] > 0
    assert payload["summary"]["degenerate_triangles"] == 1
    assert payload["summary"]["sliver_triangles"] >= 1
    assert payload["summary"]["tiny_parts"] == 1
    assert payload["summary"]["tiny_part_triangles"] == 1
    assert payload["summary"]["material_count"] == 2
    assert payload["summary"]["draw_call_estimate"] == 3
    assert payload["summary"]["draw_calls"] == 3
    assert payload["summary"]["draw_call_meshes"] == 2
    assert payload["summary"]["draw_call_materials"] == 2
    assert payload["summary"]["draw_call_submesh_slots"] == 3
    assert payload["summary"]["draw_call_material_slots"] == 3
    assert payload["summary"]["draw_call_mesh_instances"] == 2
    assert payload["summary"]["draw_call_reused_instances"] == 0
    assert payload["summary"]["draw_call_merged_batches"] == 0
    assert payload["summary"]["visual_risk_warnings"] > 0
    assert any("non-manifold edges" in warning for warning in payload["warnings"])
    assert payload["parts"][0]["part_id"] == "main"
    assert payload["parts"][1]["tiny"] is True


def test_asset_analyze_reports_actual_triangle_self_intersections() -> None:
    mesh = Mesh(
        points=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -0.5, -1.0],
                [0.0, -0.5, 1.0],
                [0.0, 0.5, 0.0],
                [2.0, 2.0, 0.0],
                [3.0, 2.0, 0.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=int),
    )
    asset = _asset_from_mesh(mesh)

    report = asset.analyze(AnalyzeOptions(self_intersections=True))

    assert report.summary["self_intersections"] == 1
    assert report.summary["self_intersection_warnings"] == 1
    assert report.parts[0]["self_intersections"] == 1


def test_asset_analyze_reports_coplanar_overlap_self_intersections() -> None:
    mesh = Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.5, 0.5, 0.0],
                [2.5, 0.5, 0.0],
                [0.5, 2.5, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    report = _asset_from_mesh(mesh).analyze(AnalyzeOptions(self_intersections=True))

    assert report.summary["self_intersections"] == 1
    assert report.summary["self_intersections_lower_bound"] is False
    assert report.parts[0]["self_intersections"] == 1


def test_asset_analyze_ignores_endpoint_contact_between_triangles() -> None:
    mesh = Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
    )

    report = _asset_from_mesh(mesh).analyze(AnalyzeOptions(self_intersections=True))

    assert report.summary["self_intersections"] == 0
    assert report.summary["self_intersection_pairs_checked"] == 1
    assert report.parts[0]["self_intersections"] == 0


def test_asset_analyze_ignores_adjacent_triangles_during_self_intersection_checks() -> None:
    mesh = Mesh(
        points=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
    )

    report = _asset_from_mesh(mesh).analyze(AnalyzeOptions(self_intersections=True))

    assert report.summary["self_intersections"] == 0
    assert report.summary["self_intersection_pairs_checked"] == 0
    assert report.parts[0]["self_intersections"] == 0


def test_asset_analyze_marks_truncated_self_intersections_as_lower_bound() -> None:
    points: list[list[float]] = []
    faces: list[list[int]] = []
    for index in range(4):
        base = len(points)
        offset = float(index * 10)
        points.extend([[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0], [offset, 1.0, 0.0]])
        faces.append([base, base + 1, base + 2])
    mesh = Mesh(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=int))

    report = _asset_from_mesh(mesh).analyze(AnalyzeOptions(self_intersections=True, max_self_intersection_pairs=2))

    assert report.summary["self_intersections"] == 0
    assert report.summary["self_intersections_lower_bound"] is True
    assert report.summary["self_intersection_pairs_checked"] == 2
    assert report.summary["self_intersection_pair_limit"] == 2
    assert report.parts[0]["self_intersections_lower_bound"] is True
    assert any("self_intersections=0 is a lower bound" in warning for warning in report.warnings)


def test_asset_analyze_does_not_count_aabb_only_self_intersection_candidates() -> None:
    mesh = Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.5, 1.8, -1.0],
                [1.5, 1.8, 1.0],
                [1.5, 3.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
    )
    asset = _asset_from_mesh(mesh)

    report = asset.analyze(AnalyzeOptions(self_intersections=True))

    assert report.summary["self_intersections"] == 0
    assert report.parts[0]["self_intersections"] == 0


def test_analyze_output_reconstructs_embedded_gltf_quality(tmp_path: Path) -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="triangle_node", name="Triangle", part_id="triangle")]),
        parts={
            "triangle": Part(
                id="triangle",
                name="Triangle",
                mesh=Mesh(
                    points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                    faces=np.asarray([[0, 1, 2]], dtype=int),
                    material_indices=np.asarray([0], dtype=int),
                ),
                material_ids=["mat"],
            )
        },
        materials={"mat": Material(id="mat", name="Mat", base_color=(0.2, 0.4, 0.6, 1.0))},
    )
    output = tmp_path / "triangle.gltf"
    asset.write_gltf(output)

    report = analyze_output(
        output,
        AnalyzeOptions(open_boundaries=True, draw_call_estimate=True),
    )

    assert report.stats["validated_meshes"] == 1
    assert report.stats["validated_triangles"] == 1
    assert report.summary["material_count"] == 1
    assert report.summary["draw_call_estimate"] == 1
    assert report.summary["open_boundaries"] == 1
    assert report.parts[0]["boundary_edges"] == 3


def test_analyze_output_can_scope_exported_gltf_with_filter(tmp_path: Path) -> None:
    asset = _quality_asset()
    output = tmp_path / "quality.gltf"
    asset.write_gltf(output)

    report = analyze_output(
        output,
        AnalyzeOptions(open_boundaries=True, draw_call_estimate=True),
        where=Filter.part("main"),
    )

    selection = report.summary["selection"]
    assert isinstance(selection, dict)
    assert selection["stats"]["parts"] == 1
    assert selection["stats"]["triangles"] == 4
    assert report.summary["parts"] == 1
    assert report.summary["material_count"] == 2
    assert [part["part_id"] for part in report.parts] == ["main"]
