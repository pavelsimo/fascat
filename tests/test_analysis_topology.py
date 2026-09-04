from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fascat.analysis import _asset_from_output, analyze_output
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import AnalyzeOptions, StlExportOptions


def _asset(mesh: Mesh) -> Asset:
    return Asset(
        root=Node(id="root", name="Root", part_id="part"),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )


def _tetrahedron(*, split_vertices: bool = False) -> Mesh:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    if split_vertices:
        points = points[faces].reshape(-1, 3)
        faces = np.arange(12, dtype=np.int64).reshape(-1, 3)
    return Mesh(points=points, faces=faces)


@pytest.mark.parametrize("binary", [False, True])
def test_closed_stl_roundtrip_uses_geometric_topology(tmp_path: Path, binary: bool) -> None:
    asset = _asset(_tetrahedron())
    output = tmp_path / "closed.stl"
    asset.write_stl(output, options=StlExportOptions(binary=binary))
    options = AnalyzeOptions(open_boundaries=True, non_manifold_edges=True)

    source = asset.analyze(options)
    exported = analyze_output(output, options)

    for metric in ("open_boundaries", "boundary_edges", "non_manifold_edges"):
        assert source.summary[metric] == exported.summary[metric] == 0
    assert exported.summary["vertices"] == 12
    assert exported.stats["validated_points"] == 12
    assert exported.summary["triangles"] == 4


@pytest.mark.parametrize("seams", [False, True])
def test_gltf_material_and_attribute_seams_keep_closed_topology(tmp_path: Path, seams: bool) -> None:
    mesh = _tetrahedron(split_vertices=seams)
    mesh.material_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    if seams:
        triangles = mesh.points[mesh.faces]
        face_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
        mesh.normals = np.repeat(face_normals, 3, axis=0)
        mesh.uvs = {0: np.arange(24, dtype=float).reshape(12, 2) / 24}
    asset = _asset(mesh)
    asset.parts["part"].material_ids = ["red", "blue"]
    asset.materials = {
        name: Material(id=name, name=name, base_color=color)
        for name, color in (("red", (1.0, 0.0, 0.0, 1.0)), ("blue", (0.0, 0.0, 1.0, 1.0)))
    }
    output = tmp_path / "closed.gltf"
    asset.write_gltf(output)
    primitives = json.loads(output.read_text())["meshes"][0]["primitives"]
    assert len(primitives) == 2
    if seams:
        assert all("NORMAL" in item["attributes"] and "TEXCOORD_0" in item["attributes"] for item in primitives)

    loaded = _asset_from_output(output)
    loaded_mesh = next(iter(loaded.parts.values())).mesh
    assert loaded_mesh is not None
    original = loaded_mesh.copy()
    options = AnalyzeOptions(open_boundaries=True, non_manifold_edges=True, self_intersections=True)
    source = asset.analyze(options)
    exported = analyze_output(output, options)
    loaded_report = loaded.analyze(options)

    for metric in ("open_boundaries", "boundary_edges", "non_manifold_edges"):
        assert source.summary[metric] == exported.summary[metric] == loaded_report.summary[metric] == 0
    assert exported.summary["vertices"] == original.vertex_count
    assert original.vertex_count > 4
    np.testing.assert_array_equal(loaded_mesh.points, original.points)
    np.testing.assert_array_equal(loaded_mesh.faces, original.faces)
    np.testing.assert_array_equal(loaded_mesh.normals, original.normals)
    np.testing.assert_array_equal(loaded_mesh.material_indices, original.material_indices)
    for channel in original.uvs:
        np.testing.assert_array_equal(loaded_mesh.uvs[channel], original.uvs[channel])


def test_split_vertices_do_not_hide_an_actual_opening(tmp_path: Path) -> None:
    mesh = _tetrahedron()
    mesh.faces = mesh.faces[:3]
    output = tmp_path / "open.stl"
    _asset(mesh).write_stl(output)

    report = analyze_output(output, AnalyzeOptions(open_boundaries=True, non_manifold_edges=True))

    assert report.summary["open_boundaries"] == 1
    assert report.summary["boundary_edges"] == 3
    assert report.summary["non_manifold_edges"] == 0


def test_split_vertices_reveal_non_manifold_edge() -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, -1, 0]], dtype=float)
    faces = np.asarray([[0, 1, 2], [1, 0, 3], [0, 1, 4]], dtype=np.int64)
    mesh = Mesh(points=points[faces].reshape(-1, 3), faces=np.arange(9).reshape(-1, 3))

    report = _asset(mesh).analyze(AnalyzeOptions(open_boundaries=True, non_manifold_edges=True))

    assert report.summary["non_manifold_edges"] == 1
    assert report.summary["boundary_edges"] == 6


@pytest.mark.parametrize("gap", [1e-10, 1e-14])
def test_topology_preserves_even_tiny_nonzero_gaps(tmp_path: Path, gap: float) -> None:
    mesh = _tetrahedron(split_vertices=True)
    # Translate the bottom face away from the shell; rounding-based welding
    # would incorrectly fill this gap even though STL can represent it.
    mesh.points[:3, 2] -= gap
    output = tmp_path / "gap.stl"
    _asset(mesh).write_stl(output)

    report = analyze_output(output, AnalyzeOptions(open_boundaries=True))

    assert report.summary["open_boundaries"] == 2
    assert report.summary["boundary_edges"] == 6


def test_topology_does_not_join_disconnected_shells_at_one_position() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [-1, 0, 0], [0, -1, 0]], dtype=float),
        faces=np.arange(6).reshape(-1, 3),
    )

    report = _asset(mesh).analyze(AnalyzeOptions(open_boundaries=True))

    assert report.summary["open_boundaries"] == 2
    assert report.summary["boundary_edges"] == 6


def test_topology_retains_coincident_shell_multiplicity_and_self_intersections() -> None:
    tetra = _tetrahedron(split_vertices=True)
    mesh = Mesh(points=np.tile(tetra.points, (2, 1)), faces=np.arange(24).reshape(-1, 3))
    asset = _asset(mesh)
    before = mesh.copy()

    intersection_only = asset.analyze(AnalyzeOptions(self_intersections=True))
    combined = asset.analyze(AnalyzeOptions(open_boundaries=True, non_manifold_edges=True, self_intersections=True))

    # Duplicated shells are never silently reduced to one manifold tetrahedron.
    assert combined.summary["non_manifold_edges"] == 6
    assert combined.summary["triangles"] == 8
    assert combined.summary["vertices"] == 24
    assert combined.summary["self_intersections"] == intersection_only.summary["self_intersections"]
    assert intersection_only.summary["self_intersections"] >= 4
    assert (
        combined.summary["self_intersection_pairs_checked"]
        == intersection_only.summary["self_intersection_pairs_checked"]
    )
    np.testing.assert_array_equal(mesh.points, before.points)
    np.testing.assert_array_equal(mesh.faces, before.faces)


def test_topology_analysis_preserves_render_attributes() -> None:
    mesh = _tetrahedron(split_vertices=True)
    mesh.normals = np.arange(36, dtype=float).reshape(12, 3)
    mesh.tangents = np.arange(48, dtype=float).reshape(12, 4)
    mesh.uvs = {0: np.arange(24, dtype=float).reshape(12, 2)}
    mesh.material_indices = np.asarray([0, 0, 1, 1], dtype=np.int64)
    mesh.face_groups = {"shell": np.arange(4, dtype=np.int64)}
    original = mesh.copy()

    report = _asset(mesh).analyze(AnalyzeOptions(open_boundaries=True, non_manifold_edges=True))

    assert report.summary["open_boundaries"] == 0
    for attribute in ("points", "faces", "normals", "tangents", "material_indices"):
        np.testing.assert_array_equal(getattr(mesh, attribute), getattr(original, attribute))
    np.testing.assert_array_equal(mesh.uvs[0], original.uvs[0])
    np.testing.assert_array_equal(mesh.face_groups["shell"], original.face_groups["shell"])
