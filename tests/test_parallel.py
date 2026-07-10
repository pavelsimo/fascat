from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import fascat.asset as asset_module
import fascat.ops.decimate as decimate_module
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

    def fake_parallel_map(
        items: list[object],
        worker: Callable[[object], object],
        *,
        jobs: int,
        executor: str = "thread",
    ) -> list[object]:
        calls.append((len(items), jobs))
        return [worker(item) for item in items]

    monkeypatch.setattr(asset_module, "parallel_map", fake_parallel_map)
    monkeypatch.setattr(decimate_module, "parallel_map", fake_parallel_map)
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


def _square(value: int) -> int:
    return value * value


def test_default_jobs_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    from fascat import options

    monkeypatch.delenv("FASCAT_JOBS", raising=False)
    monkeypatch.setattr(options.os, "cpu_count", lambda: 16)
    assert options.default_jobs() == 4
    monkeypatch.setattr(options.os, "cpu_count", lambda: 2)
    assert options.default_jobs() == 2
    monkeypatch.setattr(options.os, "cpu_count", lambda: None)
    assert options.default_jobs() == 1


def test_default_jobs_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from fascat import options

    monkeypatch.setenv("FASCAT_JOBS", "3")
    assert options.default_jobs() == 3
    monkeypatch.setenv("FASCAT_JOBS", "0")
    assert options.default_jobs() == 1
    monkeypatch.setenv("FASCAT_JOBS", "garbage")
    assert options.default_jobs() >= 1


def test_options_jobs_default_uses_default_jobs(monkeypatch: pytest.MonkeyPatch) -> None:

    monkeypatch.setenv("FASCAT_JOBS", "2")
    assert RepairOptions().jobs == 2
    assert StageOptions().jobs == 2


def test_parallel_map_process_runs_module_worker() -> None:
    assert parallel_map([1, 2, 3], _square, jobs=2, executor="process") == [1, 4, 9]


def test_parallel_map_process_falls_back_for_closures() -> None:
    captured: list[int] = []

    def closure(value: int) -> int:
        captured.append(value)
        return value + 1

    assert parallel_map([1, 2], closure, jobs=2, executor="process") == [2, 3]
    assert sorted(captured) == [1, 2]


def test_mesh_getstate_drops_cache() -> None:
    import pickle

    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    mesh.fingerprint()
    restored = pickle.loads(pickle.dumps(mesh))

    assert restored._cache == {}
    assert restored.fingerprint() == mesh.fingerprint()


@pytest.mark.parametrize("operation", ["repair", "merge_vertices", "stage", "optimize", "decimate", "lods"])
def test_part_ops_payloads_exclude_source_shape(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    import pickle

    payload_batches: list[list[object]] = []

    def asserting_parallel_map(
        items: list[object],
        worker: Callable[[object], object],
        *,
        jobs: int,
        executor: str = "thread",
    ) -> list[object]:
        if executor == "process":
            payload_batches.append(list(items))
            for item in items:
                pickle.dumps(item)
        return [worker(item) for item in items]

    for module in (asset_module, decimate_module, lod_module, optimize_module, stage_module):
        monkeypatch.setattr(module, "parallel_map", asserting_parallel_map)

    source = _strip_asset()
    for part in source.parts.values():
        part.source_shape = object()  # unpicklable: must never cross the boundary

    calls = {
        "repair": lambda: source.repair(RepairOptions(jobs=2)),
        "merge_vertices": lambda: source.merge_vertices(MergeVerticesOptions(jobs=2)),
        "stage": lambda: source.stage(StageOptions(uv0="box", uv1=None, jobs=2)),
        "optimize": lambda: source.optimize(OptimizeOptions(target_triangles=1, optimize_buffers=False, jobs=2)),
        "decimate": lambda: source.decimate(DecimateOptions(target_ratio=0.5, jobs=2)),
        "lods": lambda: source.lods(LODOptions((0.5,), jobs=2)),
    }
    calls[operation]()

    assert payload_batches, f"{operation} did not request process execution"


def test_parallel_map_propagates_worker_type_errors() -> None:
    with pytest.raises(TypeError, match="worker boom"):
        parallel_map([1, 2], _raise_type_error, jobs=2, executor="process")


def _raise_type_error(value: int) -> int:
    raise TypeError("worker boom")


@pytest.mark.parametrize("operation", ["repair", "stage"])
def test_part_operations_run_through_real_process_pool(monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    # End-to-end: payloads and results must round-trip a real spawn-based pool.
    monkeypatch.setenv("FASCAT_JOBS", "2")
    source = _strip_asset()

    if operation == "repair":
        result = source.repair(RepairOptions(jobs=2))
    else:
        result = source.stage(StageOptions(uv0="box", uv1=None, jobs=2))

    for part in result.parts.values():
        assert part.mesh is not None
        assert part.mesh.triangle_count == 2
