from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.pipeline_file import PipelineSpec


def _asset() -> Asset:
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
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="cube")]),
        parts={"cube": Part(id="cube", name="Cube", mesh=mesh)},
    )


def test_pipeline_lods_accepts_switch_distance_overrides() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {
                    "op": "lods",
                    "ratios": [0.5, 0.25],
                    "screen_coverage": [0.5, 0.25],
                    "switch_distance_overrides": [12.0, None],
                }
            ]
        }
    )

    result = spec.apply(_asset())
    part = result.parts["cube"]

    assert part.metadata["lod_level_switch_distance_sources"] == "override,formula"
    assert part.lod_meshes[0].metadata["lod_switch_distance"] == "12"
    assert result.report.steps[-1].options["switch_distance_overrides"] == [12.0, None]


def test_pipeline_lod_generator_levels_accept_switch_distance_override() -> None:
    spec = PipelineSpec.from_dict(
        {
            "steps": [
                {
                    "op": "run_lod_generators",
                    "levels": [
                        {
                            "screen_coverage": 0.5,
                            "target_ratio": 0.5,
                            "switch_distance_override": 30.0,
                        }
                    ],
                }
            ]
        }
    )

    result = spec.apply(_asset())
    lod = result.parts["cube"].lod_meshes[0]

    assert lod.metadata["lod_switch_distance"] == "30"
    assert lod.metadata["lod_switch_distance_source"] == "override"
    assert result.report.steps[-1].options["levels"][0]["switch_distance_override"] == 30.0


def test_pipeline_filter_without_criteria_is_rejected() -> None:
    # `name` inside [[filters]] is the filter identifier, not the node-name
    # criterion, so this used to parse into a filter matching every node.
    with pytest.raises(ValueError, match="declares no criteria"):
        PipelineSpec.from_dict(
            {
                "filters": [{"name": "Bolt*"}],
                "steps": [{"op": "decimate", "where": "Bolt*", "ratio": 0.5}],
            }
        )


def test_pipeline_filter_node_name_criterion_is_accepted() -> None:
    spec = PipelineSpec.from_dict(
        {
            "filters": [{"name": "bolts", "names": "Bolt*"}],
            "steps": [{"op": "decimate", "where": "bolts", "ratio": 0.5}],
        }
    )

    assert spec.filters["bolts"].to_dict()["criteria"] == {"name": ["Bolt*"]}
