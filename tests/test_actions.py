from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Node
from fascat.ops import actions, bake, decimate, holes, lod, occlusion
from fascat.ops._ids import unique_id


def test_actions_module_preserves_public_operation_imports() -> None:
    assert actions.bake_materials_asset is bake.bake_materials_asset
    assert actions.decimate_asset is decimate.decimate_asset
    assert actions.decimation_target_strategy is decimate.decimation_target_strategy
    assert actions.remove_holes_asset is holes.remove_holes_asset
    assert actions.remove_occluded_asset is occlusion.remove_occluded_asset
    assert actions.run_lod_generators_asset is lod.run_lod_generators_asset


def test_action_unique_ids_preserve_second_suffix_start() -> None:
    assert unique_id(set(), "resource") == "resource"
    assert unique_id({"resource"}, "resource") == "resource_2"
    assert unique_id({"resource", "resource_2"}, "resource") == "resource_3"


def test_node_world_walk_is_depth_first_composed_and_independent() -> None:
    root_transform = np.eye(4, dtype=np.float64)
    root_transform[0, 3] = 3.0
    rotation = np.asarray(
        [
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    mirror = np.diag(np.asarray([-1.0, 1.0, 1.0, 1.0], dtype=np.float64))
    root = Node(
        id="root",
        name="Root",
        transform=root_transform,
        children=[
            Node(
                id="branch",
                name="Branch",
                transform=rotation,
                children=[Node(id="mirrored", name="Mirrored", transform=mirror)],
            ),
            Node(id="sibling", name="Sibling"),
        ],
    )

    walked = list(root.walk_world())
    assert [node.id for node, _world in walked] == ["root", "branch", "mirrored", "sibling"]
    assert np.allclose(walked[0][1], root_transform)
    assert np.allclose(walked[1][1], root_transform @ rotation)
    assert np.allclose(walked[2][1], root_transform @ rotation @ mirror)
    assert np.linalg.det(walked[2][1][:3, :3]) < 0.0
    assert all(not np.shares_memory(walked[index][1], walked[index + 1][1]) for index in range(len(walked) - 1))
    walked[0][1][0, 3] = 99.0
    assert walked[1][1][0, 3] == pytest.approx(3.0)
    assert root.transform[0, 3] == pytest.approx(3.0)
