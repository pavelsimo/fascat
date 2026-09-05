from __future__ import annotations

import numpy as np
import pytest

from fascat.analysis import analyze_asset
from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import AnalyzeOptions


def _asset() -> Asset:
    mesh = Mesh(
        points=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0.001, 0]]),
        faces=np.array([[0, 1, 2], [0, 1, 3], [0, 0, 1]]),
    )
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="n", name="n", part_id="p")]),
        parts={"p": Part(id="p", name="p", mesh=mesh)},
    )


def _unrequested(*args: object, **kwargs: object) -> None:
    raise AssertionError("unrequested metric was evaluated")


@pytest.mark.parametrize(
    "options", [AnalyzeOptions(), AnalyzeOptions(draw_call_estimate=True), AnalyzeOptions(tiny_parts=True)]
)
def test_lightweight_analysis_does_not_compute_topology_or_shape(
    options: AnalyzeOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    asset = _asset()
    for name in ("quality_metrics", "_triangle_edge_lengths", "_triangle_areas", "_undirected_edges_and_counts"):
        monkeypatch.setattr(Mesh, name, _unrequested)

    report = analyze_asset(asset, options)

    assert report.summary["triangles"] == 3
    if options.draw_call_estimate:
        assert report.summary["draw_call_estimate"] == 1
    if options.tiny_parts:
        assert report.summary["tiny_parts"] == 0


def test_topology_analysis_does_not_compute_triangle_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = _asset()
    assert asset.parts["p"].mesh is not None
    expected = asset.parts["p"].mesh.quality_metrics()
    for name in ("quality_metrics", "_triangle_edge_lengths", "_triangle_areas"):
        monkeypatch.setattr(Mesh, name, _unrequested)

    report = analyze_asset(asset, AnalyzeOptions(non_manifold_edges=True, open_boundaries=True))

    assert report.summary["boundary_edges"] == expected["boundary_edges"]
    assert report.summary["non_manifold_edges"] == expected["non_manifold_edges"]


@pytest.mark.parametrize("epsilon", [None, 0.0, 0.001])
def test_sliver_analysis_does_not_compute_topology(epsilon: float | None, monkeypatch: pytest.MonkeyPatch) -> None:
    asset = _asset()
    assert asset.parts["p"].mesh is not None
    expected = asset.parts["p"].mesh.quality_metrics(area_epsilon=epsilon)
    for name in ("quality_metrics", "_undirected_edges_and_counts"):
        monkeypatch.setattr(Mesh, name, _unrequested)

    report = analyze_asset(asset, AnalyzeOptions(sliver_triangles=True, degenerate_area_epsilon=epsilon))

    assert report.summary["degenerate_triangles"] == expected["degenerate_triangles"]
    assert report.summary["sliver_triangles"] == expected["skinny_triangles"]
    assert report.summary["max_aspect_ratio"] == expected["max_aspect_ratio"]
