from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

import fascat.ops.bake as bake_module
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.ops._visibility import face_ambient_occlusion
from fascat.ops._visibility import ray_hits_mesh as _ray_hits_mesh
from fascat.ops._visibility import ray_hits_mesh_batch as _ray_hits_mesh_batch
from fascat.options import BakeMaterialOptions

from ._actions_helpers import _triangle_strip


def test_scoped_action_report_preserves_warning_order_and_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Asset(root=Node(id="root", name="Root"))
    source.report.add_warning("existing warning")
    options = BakeMaterialOptions(maps_resolution=8, lightmap_resolution=8)

    def fake_bake(
        asset: Asset,
        _options: BakeMaterialOptions,
        *,
        selected_part_ids: set[str] | None = None,
    ) -> Asset:
        assert selected_part_ids is None
        result = asset.copy(keep_source=True)
        result.report.add_warning("first action warning")
        result.report.add_warning("second action warning")
        return result

    monkeypatch.setattr(bake_module, "bake_materials_asset", fake_bake)
    result = source.bake_materials(options)
    step = result.report.steps[-1]

    assert step.name == "bake_materials"
    assert step.options == options.to_dict()
    assert step.before == step.after
    assert step.warnings == ["first action warning", "second action warning"]
    assert result.report.warnings == ["existing warning", "first action warning", "second action warning"]


def test_bake_materials_merges_selected_material_slots() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.asarray([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["red", "blue"])},
        materials={
            "red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0)),
            "blue": Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )

    baked = asset.bake_materials(
        BakeMaterialOptions(
            maps_resolution=64,
            lightmap_resolution=32,
            force_uv_generation=True,
            bake=("base_color", "opacity"),
        )
    )
    part = baked.parts["panel"]

    assert baked.material_count == 1
    assert part.material_ids == ["baked_material"]
    assert part.mesh is not None
    assert part.mesh.material_indices is None
    assert 0 in part.mesh.uvs
    assert (
        baked.materials["baked_material"].metadata["baked_texture_base_color_uri"].startswith("data:image/png;base64,")
    )
    assert baked.materials["baked_material"].metadata["baked_texture_base_color_image"] == "baked_base_color"
    assert baked.materials["baked_material"].metadata["baked_texture_kind"] == "raster_atlas"
    assert baked.materials["baked_material"].metadata["baked_texture_resolution"] == "64"
    assert baked.materials["baked_material"].metadata["baked_lightmap_resolution"] == "32"
    assert part.mesh.metadata["uv0_lightmap_resolution"] == "32"
    assert baked.images["baked_base_color"].width == 64
    assert baked.images["baked_base_color"].height == 64
    assert baked.metadata["baked_image_count"] == "1"
    assert baked.report.steps[-1].before["draw_calls"] == 2
    assert baked.report.steps[-1].after["draw_calls"] == 1
    if find_spec("xatlas") is None:
        assert baked.report.steps[-1].warnings == [
            "part panel could not use xatlas for bake UVs; falling back to AABB projection"
        ]
    else:
        assert baked.report.steps[-1].warnings == []


def test_bake_materials_records_emissive_material_vs_fallback_source() -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.asarray([0, 1], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["explicit", "fallback"])},
        materials={
            "explicit": Material(
                id="explicit",
                name="Explicit Black Emissive",
                base_color=(1.0, 1.0, 1.0, 1.0),
                metadata={"emissive_color": "0,0,0"},
            ),
            "fallback": Material(id="fallback", name="Fallback", base_color=(1.0, 1.0, 1.0, 1.0)),
        },
    )

    baked = asset.bake_materials(BakeMaterialOptions(maps_resolution=16, force_uv_generation=True, bake=("emissive",)))

    expected = {
        "baked_emissive_source": "mixed",
        "baked_emissive_material_faces": "1",
        "baked_emissive_fallback_faces": "1",
    }
    assert baked.images["baked_emissive"].metadata.items() >= expected.items()
    assert baked.materials["baked_material"].metadata.items() >= expected.items()


def test_face_bake_values_uses_material_lut_with_fallbacks() -> None:
    mesh = _triangle_strip(5)
    mesh.material_indices = np.asarray([0, 1, 2, 99], dtype=int)
    part = Part(id="panel", name="Panel", mesh=mesh, material_ids=["paint", "glow", "missing"])
    asset = Asset(
        root=Node(id="root", name="root"),
        parts={"panel": part},
        materials={
            "paint": Material(
                id="paint",
                name="Paint",
                base_color=(1.0, 0.25, 0.0, 0.9),
                metallic=0.25,
                roughness=0.75,
                opacity=0.5,
                metadata={"emissive_color": "1,0.5,0"},
            ),
            "glow": Material(
                id="glow",
                name="Glow",
                base_color=(0.0, 0.0, 1.0, 1.0),
                metallic=1.0,
                roughness=0.2,
                metadata={"emissive_color": "0,0,0"},
            ),
        },
    )

    assert bake_module._face_bake_values(asset, part, mesh, "base_color", "conservative").tolist() == [
        [255, 64, 0, 128],
        [0, 0, 255, 255],
        [255, 255, 255, 255],
        [255, 255, 255, 255],
        [255, 64, 0, 128],
    ]
    assert bake_module._face_bake_values(asset, part, mesh, "metallic_roughness", "conservative").tolist() == [
        [255, 191, 64, 255],
        [255, 51, 255, 255],
        [255, 128, 0, 255],
        [255, 128, 0, 255],
        [255, 191, 64, 255],
    ]
    assert bake_module._face_bake_values(asset, part, mesh, "emissive", "conservative").tolist() == [
        [255, 128, 0, 255],
        [0, 0, 0, 255],
        [0, 0, 0, 255],
        [0, 0, 0, 255],
        [255, 128, 0, 255],
    ]


def test_face_bake_values_uses_slot_zero_when_material_indices_are_absent() -> None:
    mesh = _triangle_strip(2)
    part = Part(id="panel", name="Panel", mesh=mesh, material_ids=["paint"])
    asset = Asset(
        root=Node(id="root", name="root"),
        parts={"panel": part},
        materials={"paint": Material(id="paint", name="Paint", base_color=(0.0, 1.0, 0.0, 1.0))},
    )
    empty_part = Part(id="empty", name="Empty", mesh=mesh)

    assert bake_module._face_bake_values(asset, part, mesh, "base_color", "conservative").tolist() == [
        [0, 255, 0, 255],
        [0, 255, 0, 255],
    ]
    assert bake_module._face_bake_values(asset, empty_part, mesh, "base_color", "conservative").tolist() == [
        [255, 255, 255, 255],
        [255, 255, 255, 255],
    ]


def test_emissive_provenance_counts_repeated_slots_once(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _triangle_strip(6)
    mesh.material_indices = np.asarray([0, 0, 0, 1, 2, 99], dtype=int)
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", mesh=mesh, material_ids=["explicit", "invalid", "missing"])},
        materials={
            "explicit": Material(
                id="explicit",
                name="Explicit Black Emissive",
                base_color=(1.0, 1.0, 1.0, 1.0),
                metadata={"emissive_color": "0,0,0"},
            ),
            "invalid": Material(
                id="invalid",
                name="Invalid Emissive",
                base_color=(1.0, 1.0, 1.0, 1.0),
                metadata={"emissive_color": "not,a,color"},
            ),
        },
    )
    calls: dict[str, int] = {}
    original = bake_module._emissive_color_with_source

    def count_material_classification(material: Material | None) -> tuple[tuple[float, float, float, float], str]:
        if material is not None:
            calls[material.id] = calls.get(material.id, 0) + 1
        return original(material)

    monkeypatch.setattr(bake_module, "_emissive_color_with_source", count_material_classification)

    assert bake_module._emissive_provenance_metadata(asset, ["panel"]) == {
        "baked_emissive_source": "mixed",
        "baked_emissive_material_faces": "3",
        "baked_emissive_fallback_faces": "3",
    }
    assert calls == {"explicit": 1, "invalid": 1}


def test_ambient_occlusion_detects_tiny_occluder_and_open_sky() -> None:
    mesh = Mesh(
        points=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.5],
                [1.0, 0.0, 0.5],
                [0.0, 1.0, 0.5],
            ],
            dtype=float,
        ),
        faces=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int),
    )
    open_mesh = Mesh(points=mesh.points[:3].copy(), faces=np.asarray([[0, 1, 2]], dtype=int))

    assert np.allclose(face_ambient_occlusion(mesh, "conservative"), np.asarray([0.0, 1.0]))
    assert np.allclose(face_ambient_occlusion(open_mesh, "conservative"), np.asarray([1.0]))


def test_ray_hits_mesh_vectorized_preserves_self_ignore_and_distance_window() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    triangles = points[np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int)]
    origin = np.asarray([0.25, 0.25, -1.0], dtype=float)
    direction = np.asarray([0.0, 0.0, 1.0], dtype=float)

    assert _ray_hits_mesh(origin, direction, triangles, ignore_face=99, max_t=1.5) is True
    assert _ray_hits_mesh(origin, direction, triangles, ignore_face=0, max_t=1.5) is False
    assert _ray_hits_mesh(origin, direction, triangles, ignore_face=0, max_t=3.5) is True
    assert _ray_hits_mesh(origin, direction, triangles, ignore_face=99, max_t=1.0) is False
    assert (
        _ray_hits_mesh(np.asarray([1.5, 0.25, -1.0], dtype=float), direction, triangles, ignore_face=99, max_t=3.5)
        is False
    )


def test_ray_hits_mesh_batch_matches_single_direction_wrapper() -> None:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ],
        dtype=float,
    )
    triangles = points[np.asarray([[0, 1, 2], [3, 4, 5]], dtype=int)]
    origin = np.asarray([0.25, 0.25, -1.0], dtype=float)
    directions = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    for ignore_face, max_t in ((99, 3.5), (0, 1.5), (0, 3.0), (0, 3.5)):
        batched = _ray_hits_mesh_batch(
            origin,
            directions,
            triangles,
            ignore_face=ignore_face,
            max_t=max_t,
            direction_chunk_size=2,
            triangle_chunk_size=1,
        )
        scalar = np.asarray(
            [
                _ray_hits_mesh(origin, direction, triangles, ignore_face=ignore_face, max_t=max_t)
                for direction in directions
            ],
            dtype=np.bool_,
        )

        assert np.array_equal(batched, scalar)
