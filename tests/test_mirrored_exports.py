import struct
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.io.obj import write_obj
from fascat.io.stl import write_stl
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import ObjExportOptions, StlExportOptions


def _asset(parent_sign: int, child_sign: int, vertex_normals: bool = False) -> tuple[Asset, NDArray[np.float64]]:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    normals = points - points.mean(axis=0)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    mesh = Mesh(
        points=points,
        faces=np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64),
        normals=normals if vertex_normals else None,
        material_indices=np.asarray([0, 1, 0, 1], dtype=np.int64),
    )
    parent = np.diag([parent_sign * 2.0, 2.25, 2.5, 1.0])
    parent[:3, 3] = [5, 6, 7]
    child = np.diag([1.0, float(child_sign), 1.0, 1.0])
    asset = Asset(
        root=Node(
            id="root",
            name="Root",
            transform=parent,
            children=[
                Node(id="transformed", name="Transformed", part_id="tetra", transform=child),
                Node(id="identity", name="Identity", part_id="tetra", transform=np.linalg.inv(parent)),
            ],
        ),
        parts={"tetra": Part(id="tetra", name="Tetra", mesh=mesh, material_ids=["red", "blue"])},
        materials={name: Material(id=name, name=name, base_color=(1.0, 1.0, 1.0, 1.0)) for name in ["red", "blue"]},
    )
    return asset, parent @ child


def _assert_outward(triangles: NDArray[np.float64]) -> None:
    # Both occurrences are closed tetrahedra: outward winding gives positive volume.
    for occurrence in triangles.reshape(2, 4, 3, 3):
        volume = np.einsum("ij,ij->i", occurrence[:, 0], np.cross(occurrence[:, 1], occurrence[:, 2])).sum() / 6
        assert volume > 0


@pytest.mark.parametrize("parent_sign,child_sign", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
@pytest.mark.parametrize("vertex_normals", [False, True])
@pytest.mark.parametrize("materials", [False, True])
def test_obj_preserves_world_orientation(
    tmp_path: Path, parent_sign: int, child_sign: int, vertex_normals: bool, materials: bool
) -> None:
    asset, world = _asset(parent_sign, child_sign, vertex_normals)
    mesh = asset.parts["tetra"].mesh
    assert mesh is not None
    original_faces = mesh.faces.copy()
    original_points = mesh.points.copy()
    original_normals = None if mesh.normals is None else mesh.normals.copy()
    output = tmp_path / "tetra.obj"

    write_obj(asset, output, options=ObjExportOptions(materials=materials))

    lines = output.read_text().splitlines()
    points = np.asarray([[float(value) for value in line.split()[1:]] for line in lines if line.startswith("v ")])
    normals = np.asarray([[float(value) for value in line.split()[1:]] for line in lines if line.startswith("vn ")])
    refs = np.asarray(
        [
            [[int(value) - 1 for value in corner.split("//")] for corner in line.split()[1:]]
            for line in lines
            if line.startswith("f ")
        ]
    )
    triangles = points[refs[:, :, 0]]
    _assert_outward(triangles)
    geometric_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    geometric_normals /= np.linalg.norm(geometric_normals, axis=1)[:, None]
    referenced_normals = normals[refs[:, :, 1]]
    assert np.all(np.einsum("ij,ikj->ik", geometric_normals, referenced_normals) > 0)
    if vertex_normals:
        np.testing.assert_array_equal(refs[:, :, 1], refs[:, :, 0])
        assert original_normals is not None
        expected_normals = original_normals @ np.linalg.inv(world[:3, :3])
        expected_normals /= np.linalg.norm(expected_normals, axis=1)[:, None]
        np.testing.assert_allclose(normals[:4], expected_normals)
        np.testing.assert_array_equal(mesh.normals, original_normals)
    else:
        np.testing.assert_array_equal(refs[:, :, 1], np.repeat(np.arange(8)[:, None], 3, axis=1))
        np.testing.assert_allclose(referenced_normals[:, 0], geometric_normals)
    np.testing.assert_array_equal(refs[4:, :, 0] - 4, original_faces)
    np.testing.assert_allclose(points[:4], original_points @ world[:3, :3].T + world[:3, 3])
    np.testing.assert_array_equal(mesh.faces, original_faces)
    np.testing.assert_array_equal(mesh.points, original_points)
    if materials:
        assert [line for line in lines if line.startswith("usemtl ")] == ["usemtl red", "usemtl blue"] * 4


@pytest.mark.parametrize("parent_sign,child_sign", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
@pytest.mark.parametrize("binary", [False, True])
def test_stl_preserves_world_orientation(tmp_path: Path, parent_sign: int, child_sign: int, binary: bool) -> None:
    asset, world = _asset(parent_sign, child_sign)
    mesh = asset.parts["tetra"].mesh
    assert mesh is not None
    original_faces = mesh.faces.copy()
    original_points = mesh.points.copy()
    output = tmp_path / "tetra.stl"

    write_stl(asset, output, options=StlExportOptions(binary=binary))

    if binary:
        payload = output.read_bytes()
        count = struct.unpack_from("<I", payload, 80)[0]
        records = np.asarray([struct.unpack_from("<12fH", payload, 84 + index * 50)[:12] for index in range(count)])
        normals = records[:, :3]
        triangles = records[:, 3:].reshape(-1, 3, 3)
    else:
        lines = [line.split() for line in output.read_text().splitlines()]
        normals = np.asarray(
            [[float(value) for value in line[2:]] for line in lines if line[:2] == ["facet", "normal"]]
        )
        triangles = np.asarray([[float(value) for value in line[1:]] for line in lines if line[0] == "vertex"]).reshape(
            -1, 3, 3
        )
    _assert_outward(triangles)
    geometric_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    geometric_normals /= np.linalg.norm(geometric_normals, axis=1)[:, None]
    np.testing.assert_allclose(normals, geometric_normals, atol=1e-7)
    np.testing.assert_allclose(
        triangles[:4].mean(axis=1), (original_points @ world[:3, :3].T + world[:3, 3])[original_faces].mean(axis=1)
    )
    np.testing.assert_allclose(triangles[4:], original_points[original_faces], atol=1e-12)
    np.testing.assert_array_equal(mesh.faces, original_faces)
    np.testing.assert_array_equal(mesh.points, original_points)


@pytest.mark.parametrize("parent_sign,child_sign", [(1, 1), (1, -1), (-1, 1), (-1, -1)])
def test_far_proxy_preserves_occurrence_winding(parent_sign: int, child_sign: int) -> None:
    from fascat.ops.lod import _build_scene_far_proxy
    from fascat.options import LODOptions

    asset, _ = _asset(parent_sign, child_sign)
    part = asset.parts["tetra"]
    assert part.mesh is not None
    original_faces = part.mesh.faces.copy()
    part.lod_meshes = [part.mesh.copy()]
    result = _build_scene_far_proxy(asset, LODOptions(), selected_part_ids=None)
    assert result is not None
    proxy = asset.parts[str(result["lod_scene_far_proxy_part_id"])].mesh
    assert proxy is not None
    _assert_outward(proxy.points[proxy.faces])
    np.testing.assert_array_equal(part.mesh.faces, original_faces)
    np.testing.assert_array_equal(part.lod_meshes[0].faces, original_faces)
