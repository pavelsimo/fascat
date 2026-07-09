from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fascat.analysis import analyze_output
from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.visual import (
    TurntableOptions,
    VisualDiffOptions,
    VisualPreviewOptions,
    compare_images,
    write_before_after_previews,
    write_lod_switch_previews,
    write_output_lod_switch_previews,
    write_preview,
    write_turntable_previews,
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


def test_compare_images_reports_threshold_pass_and_failure(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(baseline)
    Image.new("RGBA", (4, 4), (12, 20, 30, 255)).save(candidate)

    failing = compare_images(baseline, candidate, VisualDiffOptions(pixel_tolerance=0))
    passing = compare_images(
        baseline,
        candidate,
        VisualDiffOptions(pixel_tolerance=2, max_mean_absolute_error=1.0, max_changed_pixel_ratio=0.0),
    )

    assert failing.passed is False
    assert failing.changed_pixels == 16
    assert failing.max_absolute_error == 2
    assert passing.passed is True
    assert passing.changed_pixels == 0


def test_compare_images_fails_on_dimension_mismatch(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.png"
    candidate = tmp_path / "candidate.png"
    Image.new("RGBA", (4, 4), (10, 20, 30, 255)).save(baseline)
    Image.new("RGBA", (8, 4), (10, 20, 30, 255)).save(candidate)

    report = compare_images(baseline, candidate)

    assert report.passed is False
    assert report.changed_pixel_ratio == 1.0
    assert "different dimensions" in report.warnings[0]


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


def test_write_turntable_previews_writes_expected_views(tmp_path: Path) -> None:
    asset = _asset_with_lod()

    report = write_turntable_previews(
        asset,
        tmp_path / "turntable",
        VisualPreviewOptions(width=96, height=96, supersample=1),
        TurntableOptions(views=4, elevations=(30.0,)),
    )

    assert [view.name for view in report.views] == ["az000_el+30", "az090_el+30", "az180_el+30", "az270_el+30"]
    assert [view.azimuth for view in report.views] == [0.0, 90.0, 180.0, 270.0]
    assert all(view.elevation == 30.0 for view in report.views)
    assert report.diff_passed is None
    for view in report.views:
        assert Path(view.preview.path).exists()
    with Image.open(report.contact_sheet) as sheet:
        assert sheet.size == (384, 124)


def test_turntable_views_differ_across_azimuths(tmp_path: Path) -> None:
    asset = _asset_with_lod()

    report = write_turntable_previews(
        asset,
        tmp_path / "turntable",
        VisualPreviewOptions(width=96, height=96, supersample=1),
        TurntableOptions(views=4, elevations=(30.0,)),
    )

    images = []
    for view in report.views:
        with Image.open(view.preview.path) as image:
            images.append(np.asarray(image.convert("RGBA")))
    assert any(not np.array_equal(images[0], other) for other in images[1:])


def test_write_turntable_previews_diff_passes_for_identical_asset(tmp_path: Path) -> None:
    asset = _asset_with_lod()
    options = VisualPreviewOptions(width=96, height=96, supersample=1)
    turntable = TurntableOptions(views=2, elevations=(30.0,))
    write_turntable_previews(asset, tmp_path / "baseline", options, turntable)

    report = write_turntable_previews(
        asset,
        tmp_path / "candidate",
        options,
        turntable,
        baseline_dir=tmp_path / "baseline",
    )

    assert report.diff_passed is True
    assert all(view.diff is not None and view.diff.passed for view in report.views)
    payload = report.to_dict()
    diff_summary = payload["diff"]
    assert isinstance(diff_summary, dict)
    assert diff_summary["views_compared"] == 2
    assert diff_summary["views_failed"] == 0


def test_write_turntable_previews_diff_fails_for_changed_asset(tmp_path: Path) -> None:
    options = VisualPreviewOptions(width=96, height=96, supersample=1)
    turntable = TurntableOptions(views=2, elevations=(30.0,))
    write_turntable_previews(_asset_with_lod(), tmp_path / "baseline", options, turntable)

    report = write_turntable_previews(
        _asset_single_triangle(),
        tmp_path / "candidate",
        options,
        turntable,
        baseline_dir=tmp_path / "baseline",
    )

    assert report.diff_passed is False
    payload = report.to_dict()
    diff_summary = payload["diff"]
    assert isinstance(diff_summary, dict)
    assert diff_summary["views_failed"] > 0
    assert diff_summary["worst_view"] in [view.name for view in report.views]


def test_write_turntable_previews_reports_missing_baseline_view(tmp_path: Path) -> None:
    asset = _asset_with_lod()
    options = VisualPreviewOptions(width=96, height=96, supersample=1)
    turntable = TurntableOptions(views=2, elevations=(30.0,))
    baseline = write_turntable_previews(asset, tmp_path / "baseline", options, turntable)
    Path(baseline.views[0].preview.path).unlink()

    report = write_turntable_previews(
        asset,
        tmp_path / "candidate",
        options,
        turntable,
        baseline_dir=tmp_path / "baseline",
    )

    assert report.diff_passed is False
    missing = report.views[0].diff
    assert missing is not None
    assert missing.passed is False
    assert "baseline image missing" in missing.warnings[0]
    assert any("turntable baseline image missing" in warning for warning in report.warnings)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"views": 0},
        {"elevations": ()},
        {"elevations": (91.0,)},
        {"elevations": (-91.0,)},
    ],
)
def test_turntable_options_validation(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        TurntableOptions(**kwargs)  # type: ignore[arg-type]


def _asset_single_triangle() -> Asset:
    points = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    mesh = Mesh(points=points, faces=np.asarray([[0, 1, 2]], dtype=int))
    return Asset(
        root=Node(id="root", name="Root", children=[Node(id="part_node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
        materials={},
    )


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
