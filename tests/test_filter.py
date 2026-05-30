from __future__ import annotations

import json

import numpy as np
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter, parse_filter_expression
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import OptimizeOptions, StageOptions

runner = CliRunner()


def _mesh(size: float, triangles: int) -> Mesh:
    if triangles == 1:
        return Mesh(
            points=np.array([[0, 0, 0], [size, 0, 0], [0, size, 0]], dtype=float),
            faces=np.array([[0, 1, 2]], dtype=int),
            material_indices=np.array([0], dtype=int),
        )
    return Mesh(
        points=np.array([[0, 0, 0], [size, 0, 0], [0, size, 0], [size, size, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 0], dtype=int),
    )


def _asset() -> Asset:
    bolt = Part(
        id="bolt",
        name="Bolt",
        mesh=_mesh(1.0, 1),
        material_ids=["steel"],
        metadata={"kind": "fastener"},
    )
    housing = Part(
        id="housing",
        name="Housing",
        mesh=_mesh(10.0, 2),
        material_ids=["paint"],
        metadata={"kind": "casting"},
    )
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(
                    id="fasteners",
                    name="Fasteners",
                    children=[
                        Node(id="bolt_a", name="Bolt A", part_id="bolt"),
                        Node(id="nut_a", name="Nut A", part_id="bolt"),
                    ],
                ),
                Node(id="housing_node", name="Housing", part_id="housing"),
            ],
        ),
        parts={"bolt": bolt, "housing": housing},
        materials={
            "steel": Material(id="steel", name="Brushed Steel", base_color=(0.7, 0.7, 0.7, 1.0)),
            "paint": Material(id="paint", name="Blue Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )


def test_filter_selects_by_path_name_metadata_material_and_counts() -> None:
    asset = _asset()
    selector = Filter.all(
        Filter.path("root/Fasteners/*"),
        Filter.name("Bolt*"),
        Filter.metadata_value("kind", "fastener"),
        Filter.material("*Steel"),
        Filter.triangle_count(max=1),
        Filter.vertex_count(min=3),
    )

    selection = asset.select(selector)

    assert selection.stats() == {
        "nodes": 1,
        "parts": 1,
        "occurrences": 1,
        "materials": 1,
        "vertices": 3,
        "triangles": 1,
    }
    assert selection.matches[0].node_path == "root/Fasteners/Bolt A"
    assert selection.matches[0].part_id == "bolt"


def test_filter_selects_subtree_when_assembly_node_matches() -> None:
    selection = _asset().select(Filter.name("Fasteners"))

    assert [match.node_path for match in selection.matches] == [
        "root/Fasteners",
        "root/Fasteners/Bolt A",
        "root/Fasteners/Nut A",
    ]
    assert selection.stats()["occurrences"] == 2
    assert selection.stats()["parts"] == 1


def test_filter_logical_composition_and_cli_parser() -> None:
    selector = Filter.any(
        parse_filter_expression("triangles<=1"),
        Filter.not_(Filter.material("Blue*")),
    )

    selection = _asset().select(selector)

    assert {match.node_name for match in selection.matches if match.part_id is not None} == {"Bolt A", "Nut A"}


def test_scoped_stage_isolates_selected_occurrences_and_preserves_unmatched_parts() -> None:
    asset = _asset()

    staged = asset.stage(
        StageOptions(materials="display", uv0="none", uv1=None),
        where=Filter.path("root/Fasteners/Bolt A"),
    )
    bolt_node = next(node for node in staged.root.walk() if node.id == "bolt_a")
    nut_node = next(node for node in staged.root.walk() if node.id == "nut_a")

    assert bolt_node.part_id != "bolt"
    assert nut_node.part_id == "bolt"
    assert staged.parts["bolt"].material_ids == ["steel"]
    assert staged.parts["housing"].material_ids == ["paint"]
    assert bolt_node.part_id is not None
    assert staged.parts[bolt_node.part_id].material_ids == []
    assert staged.parts[bolt_node.part_id].metadata["display_color"] == "0.700000,0.700000,0.700000,1.000000"
    assert staged.report.steps[-1].options["matched"]["occurrences"] == 1


def test_scoped_optimize_only_changes_matching_parts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[int] = []

    def fake_simplify(self: Mesh, *, target_triangles: int | None = None, ratio: float | None = None) -> Mesh:
        calls.append(self.triangle_count)
        assert target_triangles == 1
        assert ratio is None
        return _mesh(1.0, 1)

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)

    optimized = _asset().optimize(
        OptimizeOptions(target_triangles=1, optimize_buffers=False),
        where=Filter.part("housing"),
    )

    assert optimized.parts["bolt"].mesh is not None
    assert optimized.parts["housing"].mesh is not None
    assert optimized.parts["bolt"].mesh.triangle_count == 1
    assert optimized.parts["housing"].mesh.triangle_count == 1
    assert calls == [2]
    assert optimized.report.steps[-1].options["matched"]["parts"] == 1


def test_cli_inspect_filter_reports_matches(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.cli as cli

    monkeypatch.setattr(cli, "_read_cad_for_cli", lambda _path, _ctx, _payload, **_kwargs: _asset())

    result = runner.invoke(
        app,
        [
            "--json",
            "inspect",
            "input.step",
            "--filter",
            "path=root/Fasteners/*",
            "--filter",
            "name=Bolt*",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["selection"]["stats"]["occurrences"] == 1
    assert payload["selection"]["matches"][0]["node_path"] == "root/Fasteners/Bolt A"


def test_cli_rejects_invalid_filter_expression() -> None:
    result = runner.invoke(app, ["--dry-run", "inspect", "input.step", "--filter", "foo=bar"])

    assert result.exit_code == 2
    assert "unsupported filter expression" in result.output
