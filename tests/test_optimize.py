from __future__ import annotations

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh, MeshValidationError
from fascat.options import OptimizeOptions


def mesh_with_triangles(count: int) -> Mesh:
    points = []
    faces = []
    for index in range(count):
        offset = len(points)
        base = float(index * 2)
        points.extend([[base, 0, 0], [base + 1, 0, 0], [base, 1, 0]])
        faces.append([offset, offset + 1, offset + 2])
    return Mesh(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=int))


def test_optimize_can_duplicate_repeated_parts_per_occurrence() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part"),
            Node(id="node_b", name="B", part_id="part"),
        ],
    )
    asset = Asset(root=root, parts={"part": Part(id="part", name="Part", mesh=mesh)})

    optimized = asset.optimize(OptimizeOptions(simplify=False, optimize_buffers=False, preserve_instances=False))
    part_ids = [node.part_id for node in optimized.root.walk() if node.part_id is not None]

    assert optimized.part_count == 2
    assert len(set(part_ids)) == 2
    assert all(part_id is not None and part_id.startswith("part_") for part_id in part_ids)
    assert {part.metadata["source_part_id"] for part in optimized.parts.values()} == {"part"}


def test_target_triangles_wins_over_ratio() -> None:
    mesh = Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )

    optimized = asset.optimize(OptimizeOptions(target_triangles=3, ratio=0.25, simplify=True, optimize_buffers=False))
    optimized_mesh = optimized.parts["part"].mesh

    assert optimized_mesh is not None
    assert optimized_mesh.triangle_count == 3


def test_optimize_validates_simplified_mesh_before_buffer_optimization(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    order: list[str] = []

    def fake_simplify(self: Mesh, *, target_triangles: int | None = None, ratio: float | None = None) -> Mesh:
        order.append("simplify")
        return Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
            faces=np.array([[0, 1, 9]], dtype=int),
        )

    def fail_optimize_buffers(self: Mesh) -> Mesh:
        order.append("optimize_buffers")
        raise AssertionError("buffer optimization must not run for invalid simplified meshes")

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)
    monkeypatch.setattr(Mesh, "optimize_buffers", fail_optimize_buffers)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh_with_triangles(2))},
    )

    with pytest.raises(MeshValidationError, match="out-of-range"):
        asset.optimize(OptimizeOptions(target_triangles=1, optimize_buffers=True))

    assert order == ["simplify"]


def test_optimize_runs_buffer_optimization_after_simplification(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    order: list[str] = []

    def fake_simplify(self: Mesh, *, target_triangles: int | None = None, ratio: float | None = None) -> Mesh:
        order.append("simplify")
        assert target_triangles == 1
        assert ratio is None
        return mesh_with_triangles(1)

    def fake_optimize_buffers(self: Mesh) -> Mesh:
        order.append("optimize_buffers")
        assert self.triangle_count == 1
        return self.copy()

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)
    monkeypatch.setattr(Mesh, "optimize_buffers", fake_optimize_buffers)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh_with_triangles(2))},
    )

    optimized = asset.optimize(OptimizeOptions(target_triangles=1, optimize_buffers=True))

    assert optimized.triangle_count == 1
    assert order == ["simplify", "optimize_buffers"]


def test_target_triangle_budget_is_allocated_across_unique_parts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: dict[int, int] = {}

    def fake_simplify(self: Mesh, *, target_triangles: int | None = None, ratio: float | None = None) -> Mesh:
        assert ratio is None
        assert target_triangles is not None
        calls[id(self)] = target_triangles
        return mesh_with_triangles(target_triangles)

    monkeypatch.setattr(Mesh, "simplify", fake_simplify)
    parts = {
        f"part_{index}": Part(id=f"part_{index}", name=f"Part {index}", mesh=mesh_with_triangles(4))
        for index in range(3)
    }
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[Node(id=f"node_{index}", name=f"Node {index}", part_id=f"part_{index}") for index in range(3)],
        ),
        parts=parts,
    )

    optimized = asset.optimize(OptimizeOptions(target_triangles=5, ratio=0.25, optimize_buffers=False))

    counts = sorted(part.mesh.triangle_count for part in optimized.parts.values() if part.mesh is not None)
    assert sum(counts) == 5
    assert counts == [1, 2, 2]
    assert sorted(calls.values()) == [1, 2, 2]


def test_target_triangle_budget_warns_when_unique_mesh_minimum_exceeds_target(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        Mesh,
        "simplify",
        lambda _self, *, target_triangles=None, ratio=None: mesh_with_triangles(int(target_triangles or 1)),
    )
    parts = {
        f"part_{index}": Part(id=f"part_{index}", name=f"Part {index}", mesh=mesh_with_triangles(2))
        for index in range(3)
    }
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[Node(id=f"node_{index}", name=f"Node {index}", part_id=f"part_{index}") for index in range(3)],
        ),
        parts=parts,
    )

    optimized = asset.optimize(OptimizeOptions(target_triangles=2, optimize_buffers=False))

    assert optimized.triangle_count == 3
    assert optimized.report.warnings == [
        "target_triangles is lower than the number of non-empty unique meshes; using one triangle per mesh"
    ]
    assert optimized.report.steps[-1].warnings == optimized.report.warnings
