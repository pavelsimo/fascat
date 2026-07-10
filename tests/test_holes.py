from __future__ import annotations

import numpy as np

import fascat.ops.holes as holes_module
import fascat.ops.occlusion as occlusion_module
from fascat.asset import Asset, Node, Part
from fascat.mesh import Mesh
from fascat.ops._mesh_utils import edge_faces
from fascat.options import RemoveHolesOptions


def _open_box_mesh() -> Mesh:
    points = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
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
    )
    return Mesh(points=points, faces=faces)


def _square_tube_mesh() -> Mesh:
    mesh = _open_box_mesh()
    mesh.faces = mesh.faces[2:].copy()
    return mesh


def test_remove_holes_fills_small_boundary_loop() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float),
        faces=np.asarray([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="shell", name="Shell", part_id="shell")]),
        parts={"shell": Part(id="shell", name="Shell", mesh=mesh)},
    )

    filled = asset.remove_holes(RemoveHolesOptions(max_diameter=2.0))

    assert filled.parts["shell"].mesh is not None
    assert filled.parts["shell"].mesh.triangle_count == 4
    assert filled.parts["shell"].metadata["removed_holes"] == "1"
    assert filled.metadata["removed_holes"] == "1"
    assert "BREP feature-level hole removal is not implemented" in filled.report.steps[-1].warnings[0]


def test_remove_holes_respects_surface_hole_filter() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="shell", name="Shell", part_id="shell")]),
        parts={
            "shell": Part(
                id="shell",
                name="Shell",
                mesh=Mesh(
                    points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                    faces=np.asarray([[0, 1, 2]], dtype=int),
                ),
            )
        },
    )

    skipped = asset.remove_holes(RemoveHolesOptions(through=True, blind=False, surface=False, prefer_brep=False))
    filled = asset.remove_holes(RemoveHolesOptions(through=False, blind=False, surface=True, prefer_brep=False))

    assert skipped.parts["shell"].mesh is not None
    assert skipped.parts["shell"].mesh.triangle_count == 1
    assert skipped.metadata["removed_holes"] == "0"
    assert filled.parts["shell"].mesh is not None
    assert filled.parts["shell"].mesh.triangle_count == 2
    assert filled.metadata["removed_surface_holes"] == "1"


def test_remove_holes_respects_blind_hole_filter_and_planar_span_diameter() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="box", name="Box", part_id="box")]),
        parts={"box": Part(id="box", name="Box", mesh=_open_box_mesh())},
    )

    skipped = asset.remove_holes(
        RemoveHolesOptions(through=False, blind=False, surface=True, max_diameter=1.1, prefer_brep=False)
    )
    filled = asset.remove_holes(
        RemoveHolesOptions(through=False, blind=True, surface=False, max_diameter=1.1, prefer_brep=False)
    )

    assert skipped.parts["box"].mesh is not None
    assert skipped.parts["box"].mesh.triangle_count == 10
    assert filled.parts["box"].mesh is not None
    assert filled.parts["box"].mesh.triangle_count == 12
    assert filled.parts["box"].metadata["removed_hole_types"] == "blind"
    assert filled.metadata["removed_blind_holes"] == "1"
    assert filled.metadata["removed_hole_max_diameter"] == "1"
    assert filled.metadata["removed_hole_diameter_method"] == "planar_span"


def test_remove_holes_respects_through_hole_filter() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="tube", name="Tube", part_id="tube")]),
        parts={"tube": Part(id="tube", name="Tube", mesh=_square_tube_mesh())},
    )

    skipped = asset.remove_holes(RemoveHolesOptions(through=False, blind=True, surface=True, prefer_brep=False))
    filled = asset.remove_holes(RemoveHolesOptions(through=True, blind=False, surface=False, prefer_brep=False))

    assert skipped.parts["tube"].mesh is not None
    assert skipped.parts["tube"].mesh.triangle_count == 8
    assert filled.parts["tube"].mesh is not None
    assert filled.parts["tube"].mesh.triangle_count == 12
    assert filled.parts["tube"].metadata["removed_hole_types"] == "through"
    assert filled.metadata["removed_through_holes"] == "2"


def test_action_edge_helpers_feed_boundary_loops_and_mask_expansion() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
    )

    assert edge_faces(mesh)[(1, 2)] == [0, 1]
    assert holes_module._boundary_loops(mesh) == [[0, 1, 3, 2]]
    assert occlusion_module._expand_face_mask(mesh, np.asarray([True, False], dtype=np.bool_), 1).tolist() == [
        True,
        True,
    ]
