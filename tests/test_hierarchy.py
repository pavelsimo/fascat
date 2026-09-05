from __future__ import annotations

import json

import numpy as np
import pytest
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.ops import hierarchy as hierarchy_module
from fascat.options import ExplodeOptions, MergeOptions, ReplaceOptions, StageOptions

runner = CliRunner()


def _translation(x: float, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = [x, y, z]
    return transform


def _triangle(material_index: int = 0) -> Mesh:
    return Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
        material_indices=np.array([material_index], dtype=int),
    )


def _two_material_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.array([0, 1], dtype=int),
    )


def _box_mesh() -> Mesh:
    return Mesh(
        points=np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 3, 0],
                [0, 0, 4],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]], dtype=int),
    )


def test_mesh_subset_remaps_face_groups_and_drops_empty_groups() -> None:
    mesh = _box_mesh()
    mesh.face_groups["kept"] = np.asarray([1, 3], dtype=int)
    mesh.face_groups["dropped"] = np.asarray([0], dtype=int)

    subset = hierarchy_module._mesh_subset(mesh, np.asarray([1, 3], dtype=np.int64))

    assert subset is not None
    assert subset.face_groups["kept"].tolist() == [0, 1]
    assert "dropped" not in subset.face_groups


def _asset() -> Asset:
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(
                    id="fasteners",
                    name="Fasteners",
                    children=[
                        Node(id="bolt_a", name="Bolt A", part_id="bolt", transform=_translation(0.0)),
                        Node(id="bolt_b", name="Bolt B", part_id="bolt", transform=_translation(2.0)),
                    ],
                ),
                Node(id="housing_node", name="Housing", part_id="housing", transform=_translation(10.0)),
            ],
        ),
        parts={
            "bolt": Part(
                id="bolt",
                name="Bolt",
                mesh=_triangle(),
                material_ids=["steel"],
                metadata={"kind": "fastener"},
            ),
            "housing": Part(
                id="housing",
                name="Housing",
                mesh=_triangle(),
                material_ids=["paint"],
                metadata={"kind": "casting"},
            ),
        },
        materials={
            "steel": Material(id="steel", name="Steel", base_color=(0.7, 0.7, 0.7, 1.0)),
            "paint": Material(id="paint", name="Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )


def test_merge_selected_geometry_bakes_transforms_and_keeps_parent() -> None:
    merged = _asset().merge(
        MergeOptions(mode="all", keep_parent=True, metadata="combine"),
        where=Filter.path("root/Fasteners/*"),
    )
    fasteners = next(node for node in merged.root.walk() if node.id == "fasteners")
    merged_nodes = [node for node in fasteners.children if node.part_id is not None]

    assert [node.name for node in fasteners.children] == ["Merged Geometry"]
    assert len(merged_nodes) == 1
    assert merged.part_count == 2
    assert merged.occurrence_count == 2
    assert merged.triangle_count == 3
    assert merged.draw_call_count == 2
    assert "bolt" not in merged.parts
    assert "housing" in merged.parts

    merged_part = merged.parts[merged_nodes[0].part_id or ""]
    merged_mesh = merged_part.mesh

    assert merged_mesh is not None
    assert merged_mesh.triangle_count == 2
    assert merged_mesh.bounds()[1][0] == 3.0
    assert merged_part.material_ids == ["steel"]
    assert merged_part.metadata["source_part_ids"] == "bolt"
    assert merged_part.metadata["source_node_ids"] == "bolt_a,bolt_b"
    assert merged.report.steps[-1].name == "merge"
    assert merged.report.steps[-1].before["draw_calls"] == 3
    assert merged.report.steps[-1].after["draw_calls"] == 2
    assert merged.report.steps[-1].options["matched"]["occurrences"] == 2
    advisor = merged.report.steps[-1].options["export_advisor"]
    assert advisor["lost_reused_instances"] == 1
    assert advisor["draw_call_savings"] == 1
    assert advisor["added_merged_batches"] == 1
    assert any("merge flattens 2 instances of part Bolt" in warning for warning in merged.report.steps[-1].warnings)
    assert any("preserving or reconstructing instances" in warning for warning in merged.report.steps[-1].warnings)


def test_merge_by_material_creates_one_part_per_material() -> None:
    asset = _asset()
    asset.parts["nut"] = Part(id="nut", name="Nut", mesh=_triangle(), material_ids=["paint"])
    fasteners = next(node for node in asset.root.walk() if node.id == "fasteners")
    fasteners.children.append(Node(id="nut_a", name="Nut A", part_id="nut", transform=_translation(4.0)))

    merged = asset.merge(MergeOptions(mode="by_material"), where=Filter.path("root/Fasteners/*"))
    merged_parts = [part for part in merged.parts.values() if part.id.startswith("merged_")]

    assert len(merged_parts) == 2
    assert sorted(part.material_ids for part in merged_parts) == [["paint"], ["steel"]]
    assert all(part.mesh is not None and part.mesh.triangle_count >= 1 for part in merged_parts)


def test_merge_respects_max_vertices_per_mesh() -> None:
    merged = _asset().merge(
        MergeOptions(mode="all", max_vertices_per_mesh=3),
        where=Filter.path("root/Fasteners/*"),
    )
    merged_parts = [part for part in merged.parts.values() if part.id.startswith("merged_")]

    assert len(merged_parts) == 2
    assert all(part.mesh is not None and part.mesh.vertex_count == 3 for part in merged_parts)


def test_merge_uses_default_vertex_cap_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hierarchy_module, "DEFAULT_MAX_VERTICES_PER_MERGED_MESH", 3)

    merged = _asset().merge(
        MergeOptions(mode="all", max_vertices_per_mesh=None),
        where=Filter.path("root/Fasteners/*"),
    )
    merged_parts = [part for part in merged.parts.values() if part.id.startswith("merged_")]

    assert len(merged_parts) == 2
    assert all(part.mesh is not None and part.mesh.vertex_count == 3 for part in merged_parts)
    assert any("max_vertices_per_mesh was unset" in warning for warning in merged.report.steps[-1].warnings)


def test_explode_by_material_replaces_selected_occurrence_with_child_parts() -> None:
    asset = _asset()
    asset.parts["panel"] = Part(
        id="panel",
        name="Panel",
        mesh=_two_material_mesh(),
        material_ids=["red", "blue"],
        metadata={"kind": "panel"},
    )
    asset.materials["red"] = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))
    asset.materials["blue"] = Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0))
    asset.root.children.append(Node(id="panel_node", name="Panel", part_id="panel"))

    exploded = asset.explode(ExplodeOptions(mode="by_material"), where=Filter.part("panel"))
    panel_node = next(node for node in exploded.root.walk() if node.id == "panel_node")
    child_parts = [exploded.parts[child.part_id or ""] for child in panel_node.children]

    assert panel_node.part_id is None
    assert len(child_parts) == 2
    assert sorted(part.material_ids for part in child_parts) == [["blue"], ["red"]]
    assert all(part.mesh is not None and part.mesh.triangle_count == 1 for part in child_parts)
    assert "panel" not in exploded.parts
    assert exploded.report.steps[-1].name == "explode"
    assert exploded.report.steps[-1].after["parts"] == exploded.part_count


def test_explode_connected_components_splits_disconnected_faces() -> None:
    asset = _asset()
    asset.parts["loose"] = Part(
        id="loose",
        name="Loose",
        mesh=Mesh(
            points=np.array(
                [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0], [11, 0, 0], [10, 1, 0]],
                dtype=float,
            ),
            faces=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        ),
    )
    asset.root.children.append(Node(id="loose_node", name="Loose", part_id="loose"))

    exploded = asset.explode(ExplodeOptions(mode="connected_components"), where=Filter.part("loose"))
    loose_node = next(node for node in exploded.root.walk() if node.id == "loose_node")

    assert len(loose_node.children) == 2
    assert all(exploded.parts[child.part_id or ""].mesh.triangle_count == 1 for child in loose_node.children)


def test_replace_selected_part_with_bounding_box_proxy() -> None:
    asset = _asset()
    asset.parts["block"] = Part(
        id="block",
        name="Block",
        mesh=_box_mesh(),
        material_ids=["steel"],
        metadata={"kind": "block"},
    )
    asset.root.children.append(Node(id="block_node", name="Block", part_id="block", transform=_translation(5.0)))

    replaced = asset.replace(ReplaceOptions(mode="bounding_box"), where=Filter.part("block"))
    node = next(node for node in replaced.root.walk() if node.id == "block_node")
    part = replaced.parts[node.part_id or ""]

    assert node.transform[0, 3] == 5.0
    assert part.id != "block"
    assert part.mesh is not None
    assert part.mesh.vertex_count == 8
    assert part.mesh.triangle_count == 12
    assert part.metadata["replacement_mode"] == "bounding_box"
    assert part.metadata["source_part_ids"] == "block"
    assert "block" not in replaced.parts
    assert replaced.report.steps[-1].name == "replace"


def test_cli_convert_accepts_merge_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--filter",
            "path=*/Fasteners/*",
            "--merge",
            "--merge-mode",
            "by-material",
            "--merge-metadata",
            "combine",
            "--max-vertices-per-mesh",
            "65535",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["merge"] is True
    assert payload["merge_mode"] == "by-material"
    assert payload["merge_metadata"] == "combine"


def test_cli_convert_accepts_explode_and_replace_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--filter",
            "material=rubber",
            "--explode",
            "connected-components",
            "--replace",
            "bounding-box",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["explode"] == "connected-components"
    assert payload["replace"] == "bounding-box"


def test_cli_convert_requires_region_size_for_region_merge() -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "convert", "input.step", "output.usdc", "--merge", "--merge-mode", "regions"],
    )

    assert result.exit_code == 2
    assert "--merge-mode regions requires --region-size" in result.output


def _attributed_asset() -> Asset:
    mesh = _two_material_mesh()
    # Authored normals deliberately differ from geometric face normals.
    mesh.normals = np.tile(np.array([0.0, 1.0, 1.0]) / np.sqrt(2), (4, 1))
    mesh.tangents = np.tile([1.0, 0.0, 0.0, -1.0], (4, 1))
    mesh.uvs = {
        0: np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float),
        2: np.array([[0.1, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.9]]),
    }
    mesh.face_groups = {"shell": np.array([0, 1]), "detail": np.array([1])}
    lod = hierarchy_module._mesh_subset(mesh, np.array([1], dtype=np.int64))
    assert lod is not None
    lod.metadata["lod_screen_coverage"] = "0.25"
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="a", name="A", part_id="a")]),
        parts={"a": Part(id="a", name="A", mesh=mesh, lod_meshes=[lod], material_ids=["red", "blue"])},
        materials={
            "red": Material(id="red", name="Red", base_color=(1, 0, 0, 1), metadata={"base_color_texture": "red.png"}),
            "blue": Material(id="blue", name="Blue", base_color=(0, 0, 1, 1)),
        },
    )


def test_merge_one_part_preserves_attributes_and_lod_without_aliasing() -> None:
    source = _attributed_asset()
    result = source.merge(MergeOptions(mode="all"))
    part = next(iter(result.parts.values()))
    original = source.parts["a"]
    assert part.mesh is not None and original.mesh is not None
    assert len(part.lod_meshes) == 1
    assert part.material_ids == ["red", "blue"]
    assert result.materials["red"].metadata["base_color_texture"] == "red.png"
    for actual, expected in zip([part.mesh, *part.lod_meshes], [original.mesh, *original.lod_meshes], strict=True):
        np.testing.assert_allclose(actual.points, expected.points)
        np.testing.assert_array_equal(actual.faces, expected.faces)
        np.testing.assert_allclose(actual.normals, expected.normals)
        np.testing.assert_allclose(actual.tangents, expected.tangents)
        np.testing.assert_array_equal(actual.material_indices, expected.material_indices)
        assert set(actual.uvs) == {0, 2}
        for channel in expected.uvs:
            np.testing.assert_array_equal(actual.uvs[channel], expected.uvs[channel])
            assert not np.shares_memory(actual.uvs[channel], expected.uvs[channel])
        for name in expected.face_groups:
            np.testing.assert_array_equal(actual.face_groups[name], expected.face_groups[name])
        assert not np.shares_memory(actual.normals, expected.normals)
    assert part.lod_meshes[0].metadata["lod_screen_coverage"] == "0.25"


@pytest.mark.parametrize("mirrored", [False, True])
def test_merge_parts_transforms_attributes_and_remaps_each_lod(mirrored: bool) -> None:
    asset = _attributed_asset()
    other = asset.parts["a"].copy()
    other.id = "b"
    other.material_ids = ["blue", "red"]
    asset.parts["b"] = other
    transform = np.array([[0, -3, 0, 7], [2, 0, 0, 8], [0, 0, -4 if mirrored else 4, 9], [0, 0, 0, 1]], dtype=float)
    asset.root.children.append(Node(id="b", name="B", part_id="b", transform=transform))
    part = next(iter(asset.merge(MergeOptions(mode="all")).parts.values()))
    assert part.mesh is not None and other.mesh is not None
    for actual, source in zip([part.mesh, *part.lod_meshes], [other.mesh, *other.lod_meshes], strict=True):
        count = source.vertex_count
        expected_points = source.points @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(actual.points[count:], expected_points)
        expected_normal = np.array([-1 / 3, 0, -1 / 4 if mirrored else 1 / 4])
        expected_normal /= np.linalg.norm(expected_normal)
        np.testing.assert_allclose(actual.normals[count:], np.tile(expected_normal, (count, 1)))
        np.testing.assert_allclose(actual.tangents[count:], np.tile([0, 1, 0, 1 if mirrored else -1], (count, 1)))
        for channel in source.uvs:
            np.testing.assert_array_equal(actual.uvs[channel][count:], source.uvs[channel])
        expected_faces = source.faces[:, [0, 2, 1]] if mirrored else source.faces
        np.testing.assert_array_equal(actual.faces[source.triangle_count :], expected_faces + count)
        assert actual.face_groups["shell"].tolist() == list(range(source.triangle_count * 2))
    assert part.mesh.material_indices.tolist() == [0, 1, 1, 0]
    assert part.lod_meshes[0].material_indices.tolist() == [1, 0]
    assert part.mesh.face_groups["detail"].tolist() == [1, 3]


def test_merge_by_material_remaps_vertex_attributes_and_face_groups() -> None:
    asset = _attributed_asset()
    source = asset.parts["a"].mesh
    assert source is not None
    asset.parts["a"].lod_meshes = []
    parts = list(asset.merge(MergeOptions(mode="by_material")).parts.values())
    blue = next(part for part in parts if part.material_ids == ["blue"])
    assert blue.mesh is not None
    np.testing.assert_array_equal(blue.mesh.uvs[2], source.uvs[2][[1, 2, 3]])
    np.testing.assert_allclose(blue.mesh.normals, source.normals[[1, 2, 3]])
    np.testing.assert_array_equal(blue.mesh.tangents, source.tangents[[1, 2, 3]])
    assert blue.mesh.face_groups["detail"].tolist() == [0]
    assert blue.mesh.material_indices.tolist() == [0]


@pytest.mark.parametrize("difference", ["uv", "tangents", "lod_count", "lod_uv", "coverage"])
def test_merge_rejects_incompatible_attributes_and_lod_chains(difference: str) -> None:
    asset = _attributed_asset()
    other = asset.parts["a"].copy()
    other.id = "b"
    assert other.mesh is not None
    if difference == "uv":
        del other.mesh.uvs[2]
    elif difference == "tangents":
        other.mesh.tangents = None
    elif difference == "lod_count":
        other.lod_meshes = []
    elif difference == "lod_uv":
        del other.lod_meshes[0].uvs[0]
    else:
        other.lod_meshes[0].metadata["lod_screen_coverage"] = "0.5"
    asset.parts["b"] = other
    asset.root.children.append(Node(id="b", name="B", part_id="b"))
    with pytest.raises(ValueError, match="merge requires matching"):
        asset.merge(MergeOptions(mode="all"))
    assert list(asset.parts) == ["a", "b"]
    assert asset.parts["a"].mesh.uvs.keys() == {0, 2}
    assert len(asset.parts["a"].lod_meshes) == 1


def test_merge_rejects_splitting_stored_lods_by_material() -> None:
    with pytest.raises(ValueError, match="cannot split a stored LOD chain by material"):
        _attributed_asset().merge(MergeOptions(mode="by_material"))


def test_stage_then_merge_retains_hard_edges_uvs_and_tangents() -> None:
    asset = _attributed_asset()
    asset.parts["a"].lod_meshes = []
    asset.parts["a"].mesh = _box_mesh()
    staged = asset.stage(StageOptions(normal_mode="hard_edges", uv0="box", uv1="copy_uv0", tangents=True, jobs=1))
    source = staged.parts["a"].mesh
    assert source is not None and source.normals is not None and source.tangents is not None
    assert source.vertex_count > 4
    parent = Node(id="parent", name="Parent", transform=_translation(20), children=staged.root.children)
    staged.root.children = [parent]
    part = next(iter(staged.merge(MergeOptions(mode="all", keep_parent=True)).parts.values()))
    assert part.mesh is not None
    np.testing.assert_allclose(part.mesh.points, source.points)
    np.testing.assert_allclose(part.mesh.normals, source.normals)
    np.testing.assert_allclose(part.mesh.tangents, source.tangents)
    for channel in (0, 1):
        np.testing.assert_array_equal(part.mesh.uvs[channel], source.uvs[channel])


def test_merge_generates_missing_normals_without_overwriting_authored_normals() -> None:
    asset = _asset()
    asset.parts["bolt"].mesh.normals = np.tile([0.0, 1.0, 0.0], (3, 1))
    part = next(iter(asset.merge(MergeOptions(mode="all")).parts.values()))
    assert part.mesh is not None
    np.testing.assert_allclose(part.mesh.normals[:6], np.tile([0.0, 1.0, 0.0], (6, 1)))
    np.testing.assert_allclose(part.mesh.normals[6:], np.tile([0.0, 0.0, 1.0], (3, 1)))


def test_merge_rejects_singular_transform() -> None:
    asset = _attributed_asset()
    asset.root.children[0].transform[0, 0] = 0
    with pytest.raises(ValueError, match="singular transform"):
        asset.merge(MergeOptions(mode="all"))


def test_merge_rejects_mixed_material_assignments_unless_materials_are_dropped() -> None:
    asset = _asset()
    asset.parts["housing"].material_ids = []
    with pytest.raises(ValueError, match="assigned and unassigned face materials"):
        asset.merge(MergeOptions(mode="all"))
    part = next(iter(asset.merge(MergeOptions(mode="all", preserve_materials=False)).parts.values()))
    assert part.material_ids == []
    assert part.mesh is not None and part.mesh.material_indices is None


@pytest.mark.parametrize("same_distance", [False, True])
def test_merge_discards_source_lod_switch_distances(same_distance: bool) -> None:
    asset = _attributed_asset()
    other = asset.parts["a"].copy()
    other.id = "b"
    asset.parts["b"] = other
    asset.root.children.append(Node(id="b", name="B", part_id="b"))
    for part, distance in [(asset.parts["a"], "10"), (other, "10" if same_distance else "30")]:
        part.lod_meshes[0].metadata.update(lod_switch_distance=distance, lod_switch_distance_source="formula")
    merged = next(iter(asset.merge(MergeOptions(mode="all")).parts.values()))
    assert len(merged.lod_meshes) == 1
    assert "lod_switch_distance" not in merged.lod_meshes[0].metadata
    assert "lod_switch_distance_source" not in merged.lod_meshes[0].metadata
    assert merged.lod_meshes[0].metadata["lod_screen_coverage"] == "0.25"
