from __future__ import annotations

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.options import Tessellation


def triangle_mesh() -> Mesh:
    return Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )


def test_tessellate_deduplicates_parts_by_mesh_fingerprint() -> None:
    mesh = triangle_mesh()
    fingerprint = mesh.fingerprint()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(id="part_a", name="Part A", mesh=mesh, fingerprint=fingerprint),
            "part_b": Part(id="part_b", name="Part B", mesh=mesh.copy(), fingerprint=fingerprint),
        },
    )

    tessellated = asset.tessellate(Tessellation())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert tessellated.part_count == 1
    assert part_ids == ["part_a", "part_a"]


def test_tessellate_keeps_distinct_per_face_material_assignments() -> None:
    mesh = triangle_mesh()
    red_face_mesh = Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces.copy(),
        material_indices=np.array([0], dtype=int),
    )
    blue_face_mesh = Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces.copy(),
        material_indices=np.array([1], dtype=int),
    )
    fingerprint = mesh.fingerprint()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(
                id="part_a",
                name="Part A",
                mesh=red_face_mesh,
                material_ids=["red", "blue"],
                fingerprint=fingerprint,
            ),
            "part_b": Part(
                id="part_b",
                name="Part B",
                mesh=blue_face_mesh,
                material_ids=["red", "blue"],
                fingerprint=fingerprint,
            ),
        },
    )

    tessellated = asset.tessellate(Tessellation())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert tessellated.part_count == 2
    assert part_ids == ["part_a", "part_b"]


def test_tessellate_reuses_source_shape_mesh_for_matching_parts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(id="part_a", name="Part A", source_shape=source_shape, material_ids=["red"]),
            "part_b": Part(id="part_b", name="Part B", source_shape=source_shape, material_ids=["red"]),
        },
    )
    calls: list[object] = []

    def fake_tessellate_shape(
        shape: object,
        _options: Tessellation,
        *,
        face_material_indices: list[int] | None = None,
    ) -> Mesh:
        calls.append(shape)
        assert face_material_indices is None
        return triangle_mesh()

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(Tessellation())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert calls == [source_shape]
    assert tessellated.part_count == 1
    assert part_ids == ["part_a", "part_a"]


def test_tessellate_source_shape_cache_respects_face_material_assignments(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(
                id="part_a",
                name="Part A",
                source_shape=source_shape,
                material_ids=["red", "blue"],
                metadata={"occt_face_material_indices": "0"},
            ),
            "part_b": Part(
                id="part_b",
                name="Part B",
                source_shape=source_shape,
                material_ids=["red", "blue"],
                metadata={"occt_face_material_indices": "1"},
            ),
        },
    )
    calls: list[list[int] | None] = []

    def fake_tessellate_shape(
        shape: object,
        _options: Tessellation,
        *,
        face_material_indices: list[int] | None = None,
    ) -> Mesh:
        assert shape is source_shape
        calls.append(face_material_indices)
        mesh = triangle_mesh()
        if face_material_indices is not None:
            mesh.material_indices = np.asarray(face_material_indices, dtype=int)
        return mesh

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(Tessellation())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert calls == [[0], [1]]
    assert tessellated.part_count == 2
    assert part_ids == ["part_a", "part_b"]


def test_tessellation_keep_brep_controls_source_shape_retention(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    root = Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")])
    asset = Asset(root=root, parts={"part": Part(id="part", name="Part", source_shape=source_shape)})
    calls: list[object] = []

    def fake_tessellate_shape(shape: object, _options: Tessellation, **_kwargs: object) -> Mesh:
        calls.append(shape)
        return triangle_mesh()

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    dropped = asset.tessellate(Tessellation(keep_brep=False))
    kept = asset.tessellate(Tessellation(keep_brep=True))

    assert calls == [source_shape, source_shape]
    assert dropped.parts["part"].source_shape is None
    assert kept.parts["part"].source_shape is source_shape
