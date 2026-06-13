from __future__ import annotations

import numpy as np
import pytest

import fascat as fc
from fascat.options import UnwrapOptions


def _asset() -> fc.Asset:
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    return fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )


def test_repair_kwargs_match_options_object() -> None:
    via_kwargs = _asset().repair(tolerance=0.05, quality_report=True)
    via_options = _asset().repair(fc.RepairOptions(tolerance=0.05, quality_report=True))

    assert via_kwargs.report.steps[-1].options == via_options.report.steps[-1].options


def test_tessellate_kwargs_match_options_object() -> None:
    via_kwargs = _asset().tessellate(sag=0.2, angle=20.0)
    via_options = _asset().tessellate(fc.TessellationOptions(sag=0.2, angle=20.0))

    assert via_kwargs.report.steps[-1].options == via_options.report.steps[-1].options


def test_stage_kwargs_accept_nested_options() -> None:
    staged = _asset().stage(uv0="box", unwrap=UnwrapOptions(padding=4))

    step = staged.report.steps[-1]
    assert step.options["uv0"] == "box"
    assert step.options["unwrap"]["padding"] == 4


@pytest.mark.parametrize("method", ["repair", "tessellate", "stage", "optimize", "decimate"])
def test_options_and_kwargs_conflict_raises(method: str) -> None:
    options_by_method = {
        "repair": fc.RepairOptions(),
        "tessellate": fc.TessellationOptions(),
        "stage": fc.StageOptions(),
        "optimize": fc.OptimizeOptions(),
        "decimate": fc.DecimateOptions(),
    }
    asset = _asset()

    with pytest.raises(TypeError, match="not both"):
        getattr(asset, method)(options_by_method[method], quality_report=True)


def test_unknown_kwarg_raises_type_error() -> None:
    with pytest.raises(TypeError, match="toleranse"):
        _asset().repair(toleranse=0.1)  # type: ignore[call-arg]


def test_lods_accepts_ratio_sequence() -> None:
    with_lods = _asset().lods([0.5])

    assert len(with_lods.parts["part"].lod_meshes) == 1


def test_lods_rejects_duplicate_ratios() -> None:
    with pytest.raises(TypeError, match="not both"):
        _asset().lods([0.5], ratios=(0.25,))
