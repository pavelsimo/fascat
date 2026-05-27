from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fascat.analysis import analyze_output
from fascat.asset import Asset, Node, Part
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
    assert payload["summary"]["visual_risk_warnings"] > 0
    assert any("non-manifold edges" in warning for warning in payload["warnings"])
    assert payload["parts"][0]["part_id"] == "main"
    assert payload["parts"][1]["tiny"] is True


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
