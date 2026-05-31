from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fascat.asset as asset_module
import fascat.ops.actions as actions_module
import fascat.ops.lod as lod_module
import fascat.ops.optimize as optimize_module
import fascat.ops.stage as stage_module
from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.ops.parallel import parallel_map, worker_count
from fascat.options import (
    DecimateOptions,
    LODOptions,
    MergeVerticesOptions,
    OptimizeOptions,
    RepairOptions,
    StageOptions,
)


def _strip_asset() -> Asset:
    parts: dict[str, Part] = {}
    children: list[Node] = []
    for index, part_id in enumerate(("left", "right")):
        offset = float(index * 3)
        mesh = Mesh(
            points=np.array(
                [
                    [offset, 0, 0],
                    [offset + 1, 0, 0],
                    [offset, 1, 0],
                    [offset + 1, 1, 0],
                ],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        )
        parts[part_id] = Part(id=part_id, name=part_id.title(), mesh=mesh)
        children.append(Node(id=f"{part_id}_node", name=part_id.title(), part_id=part_id))
    return Asset(root=Node(id="root", name="Root", children=children), parts=parts)


def test_parallel_map_preserves_input_order() -> None:
    assert worker_count(4, 2) == 2
    assert parallel_map([3, 1, 2], lambda value: value * 10, jobs=3) == [30, 10, 20]


def test_parallel_map_rejects_non_positive_jobs() -> None:
    with pytest.raises(ValueError, match="jobs"):
        worker_count(0, 2)


def test_part_operations_pass_requested_jobs(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_parallel_map(items: list[str], worker: Callable[[str], object], *, jobs: int) -> list[object]:
        calls.append((len(items), jobs))
        return [worker(item) for item in items]

    monkeypatch.setattr(asset_module, "parallel_map", fake_parallel_map)
    monkeypatch.setattr(actions_module, "parallel_map", fake_parallel_map)
    monkeypatch.setattr(lod_module, "parallel_map", fake_parallel_map)
    monkeypatch.setattr(optimize_module, "parallel_map", fake_parallel_map)
    monkeypatch.setattr(stage_module, "parallel_map", fake_parallel_map)

    source = _strip_asset()
    source.repair(RepairOptions(jobs=2))
    source.merge_vertices(MergeVerticesOptions(jobs=3))
    source.stage(StageOptions(uv0="box", uv1=None, jobs=4))
    source.optimize(OptimizeOptions(target_triangles=1, optimize_buffers=False, jobs=5))
    source.decimate(DecimateOptions(target_ratio=0.5, jobs=6))
    source.lods(LODOptions((0.5,), jobs=7))

    assert calls == [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7)]
