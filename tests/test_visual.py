from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from fascat.analysis import analyze_output
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.visual import (
    VisualPreviewOptions,
    write_before_after_previews,
    write_lod_switch_previews,
    write_output_lod_switch_previews,
    write_preview,
)


def test_write_preview_creates_nonblank_png(tmp_path: Path) -> None:
    asset = _asset_with_lod()
    output = tmp_path / "preview.png"

    report = write_preview(asset, output, VisualPreviewOptions(width=128, height=96, supersample=1))

    assert report.path == str(output)
    assert report.triangles == 2
    assert report.occurrences == 1
    with Image.open(output) as image:
        assert image.size == (128, 96)
        pixels = np.asarray(image.convert("RGBA"))
    assert np.unique(pixels.reshape((-1, 4)), axis=0).shape[0] > 1


def test_write_before_after_previews_creates_contact_sheet(tmp_path: Path) -> None:
    before = _asset_with_lod()
    after = _asset_with_lod(scale=0.75)

    report = write_before_after_previews(
        before,
        after,
        tmp_path / "comparison",
        VisualPreviewOptions(width=96, height=96, supersample=1),
    )

    assert Path(report.before.path).exists()
    assert Path(report.after.path).exists()
    with Image.open(report.contact_sheet) as sheet:
        assert sheet.size == (192, 124)


def test_write_lod_switch_previews_reports_monotonic_triangle_counts(tmp_path: Path) -> None:
    asset = _asset_with_lod()

    report = write_lod_switch_previews(
        asset,
        tmp_path / "lods",
        VisualPreviewOptions(width=96, height=96, supersample=1),
    )

    assert [preview.lod_level for preview in report.previews] == [0, 1]
    assert [preview.triangles for preview in report.previews] == [2, 1]
    assert report.monotonic_triangles is True
    with Image.open(report.contact_sheet) as sheet:
        assert sheet.size == (192, 124)


def test_output_lod_preview_reads_fascat_gltf_lod_metadata(tmp_path: Path) -> None:
    asset = _asset_with_lod()
    output = tmp_path / "asset.glb"
    asset.write_gltf(output)

    analysis = analyze_output(output)
    report = write_output_lod_switch_previews(
        output,
        tmp_path / "output-lods",
        VisualPreviewOptions(width=96, height=96, supersample=1),
    )

    assert analysis.stats["lod_meshes"] == 1
    assert [preview.triangles for preview in report.previews] == [2, 1]
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["levels"] == 2


def _asset_with_lod(*, scale: float = 1.0) -> Asset:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [scale, 0.0, 0.0],
            [0.0, scale, 0.0],
            [scale, scale, 0.0],
        ],
        dtype=float,
    )
    mesh = Mesh(
        points=points,
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
        material_indices=np.asarray([0, 1], dtype=int),
    )
    lod_mesh = Mesh(
        points=points[:3].copy(),
        faces=np.asarray([[0, 1, 2]], dtype=int),
        metadata={"lod_ratio": "0.5", "lod_screen_coverage": "0.25"},
    )
    return Asset(
        root=Node(id="root", name="Root", children=[Node(id="part_node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh, material_ids=["red", "blue"], lod_meshes=[lod_mesh])},
        materials={
            "red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0)),
            "blue": Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )
