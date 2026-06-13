from __future__ import annotations

import numpy as np
import pytest

import fascat as fc
from fascat._format import count_phrase, human_count


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (1, "1"),
        (999, "999"),
        (1000, "1K"),
        (1234, "1.2K"),
        (999_999, "1000K"),
        (1_200_000, "1.2M"),
        (2_100_000_000, "2.1G"),
    ],
)
def test_human_count(value: int, expected: str) -> None:
    assert human_count(value) == expected


def test_count_phrase_pluralizes() -> None:
    assert count_phrase(1, "material") == "1 material"
    assert count_phrase(38, "material") == "38 materials"
    assert count_phrase(2, "vertex") == "2 vertices"
    assert count_phrase(1_200_000, "triangle") == "1.2M triangles"


def _mesh() -> fc.Mesh:
    return fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )


def test_mesh_repr() -> None:
    assert repr(_mesh()) == "<Mesh: 3 vertices, 1 triangle>"


def test_node_repr() -> None:
    node = fc.Node(id="n1", name="Wheel", part_id="p7", children=[fc.Node(id="c", name="c")])

    assert repr(node) == "<Node n1 'Wheel' part=p7, 1 child>"
    assert repr(fc.Node(id="n2", name="Group")) == "<Node n2 'Group'>"


def test_part_repr_with_and_without_mesh() -> None:
    part = fc.Part(id="p7", name="Gear", mesh=_mesh(), material_ids=["m1", "m2"])

    assert repr(part) == "<Part p7 'Gear': 1 triangle, 2 materials>"
    assert repr(fc.Part(id="p8", name="Shell")) == "<Part p8 'Shell': no mesh>"


def test_asset_repr() -> None:
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="n", name="n", part_id="p")]),
        parts={"p": fc.Part(id="p", name="P", mesh=_mesh(), material_ids=["m"])},
        materials={"m": fc.Material(id="m", name="M", base_color=(1.0, 0.0, 0.0, 1.0))},
    )

    assert repr(asset) == "<Asset: 1 part, 1 triangle, 1 material>"


def test_options_repr_non_defaults_only() -> None:
    assert repr(fc.RepairOptions()) == "RepairOptions()"
    assert repr(fc.TessellationOptions(sag=0.2)) == "TessellationOptions(sag=0.2)"
    assert repr(fc.StageOptions(uv0="unwrap", jobs=4)) == "StageOptions(uv0='unwrap', jobs=4)"
    assert repr(fc.LODOptions(ratios=(0.5,))) == "LODOptions(ratios=(0.5,))"
