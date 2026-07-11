import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.runtime import RuntimeBrowserRenderReport, RuntimeBrowserReport
from fascat.visual import write_output_preview

from ._cli_test_helpers import (
    block_imports,
    compact,
    plain,
    runner,
)


def test_validate_help() -> None:
    result = runner.invoke(app, ["validate", "--help"])
    assert result.exit_code == 0
    assert "USD" in result.output
    assert "--geometry-quality" in plain(result.output)
    assert "--non-manifold-edges" in plain(result.output)
    assert "--draw-call-estimate" in plain(result.output)
    assert "--report" in plain(result.output)


def test_validate_dry_run() -> None:
    result = runner.invoke(app, ["--dry-run", "validate", "output.usdc"])
    assert result.exit_code == 0
    assert "Would validate output.usdc" in result.output


def test_validate_missing_file_fails() -> None:
    result = runner.invoke(app, ["validate", "missing.usdc"])
    assert result.exit_code == 1
    assert "Missing output file" in result.output


def test_validate_rejects_unknown_extension(tmp_path: Path) -> None:
    output_file = tmp_path / "output.txt"
    output_file.write_text("not usd", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(output_file)])
    assert result.exit_code == 2
    assert "Unsupported export extension" in result.output


def test_validate_can_scope_geometry_quality_with_filter(tmp_path: Path) -> None:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="kept_node", name="Kept", part_id="kept"),
                Node(id="skipped_node", name="Skipped", part_id="skipped"),
            ],
        ),
        parts={
            "kept": Part(id="kept", name="Kept", mesh=mesh, material_ids=["red"]),
            "skipped": Part(id="skipped", name="Skipped", mesh=mesh, material_ids=["blue"]),
        },
        materials={
            "red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0)),
            "blue": Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0)),
        },
    )
    output_file = tmp_path / "filtered.gltf"
    asset.write_gltf(output_file)

    result = runner.invoke(
        app,
        ["--json", "validate", str(output_file), "--geometry-quality", "--filter", "material=Red"],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    selection = payload["analysis"]["summary"]["selection"]
    assert selection["stats"]["parts"] == 1
    assert selection["matches"][0]["part_id"] == "kept"
    assert payload["analysis"]["summary"]["parts"] == 1


def test_validate_can_include_browser_runtime_measurement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "runtime.glb"
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    ).write_gltf(output_file)

    def fake_measure(path: str | Path, options: object = None) -> RuntimeBrowserReport:
        assert Path(path) == output_file
        return RuntimeBrowserReport(
            path=str(path),
            status="measured",
            browser="fake-browser",
            load_time_ms=10,
            measured_fps=60.0,
            frame_count=120,
            measurement_duration_ms=2000,
            memory_bytes=4096,
            meshes=1,
            triangles=1,
            workload_triangles=1,
            workload_scale=1.0,
        )

    monkeypatch.setattr("fascat.runtime.measure_browser_runtime", fake_measure)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--runtime-browser"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runtime_browser"]["status"] == "measured"
    assert payload["runtime_browser"]["measured_fps"] == 60.0


def test_validate_can_write_browser_render_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "runtime.glb"
    preview_file = tmp_path / "browser-preview.png"
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    ).write_gltf(output_file)

    def fake_preview(path: str | Path, preview_path: str | Path, options: object = None) -> RuntimeBrowserRenderReport:
        assert Path(path) == output_file
        assert Path(preview_path) == preview_file
        return RuntimeBrowserRenderReport(
            path=str(path),
            status="rendered",
            browser="fake-browser",
            preview_path=str(preview_path),
            width=800,
            height=600,
            meshes=1,
            triangles=1,
        )

    monkeypatch.setattr("fascat.runtime.write_browser_render_preview", fake_preview)

    result = runner.invoke(
        app, ["--json", "validate", str(output_file), "--runtime-browser-preview", str(preview_file)]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["runtime_browser_preview"]["status"] == "rendered"
    assert payload["runtime_browser_preview"]["preview_path"] == str(preview_file)


def test_validate_reports_unsupported_browser_render_preview(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_file = tmp_path / "runtime.glb"
    preview_file = tmp_path / "browser-preview.png"
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    ).write_gltf(output_file)

    def fake_preview(path: str | Path, preview_path: str | Path, options: object = None) -> RuntimeBrowserRenderReport:
        return RuntimeBrowserRenderReport(
            path=str(path),
            status="unsupported",
            browser=None,
            preview_path=str(preview_path),
            width=800,
            height=600,
            meshes=1,
            triangles=1,
            required_extensions=("KHR_draco_mesh_compression",),
            unsupported_extensions=("KHR_draco_mesh_compression",),
            preview_limitations=(
                "browser preview could not decode KHR_draco_mesh_compression: glTF Transform copy failed",
            ),
            error="browser preview could not decode KHR_draco_mesh_compression: glTF Transform copy failed",
        )

    monkeypatch.setattr("fascat.runtime.write_browser_render_preview", fake_preview)

    result = runner.invoke(
        app, ["--json", "validate", str(output_file), "--runtime-browser-preview", str(preview_file)]
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    preview = payload["runtime_browser_preview"]
    assert preview["status"] == "unsupported"
    assert preview["unsupported_extensions"] == ["KHR_draco_mesh_compression"]
    assert not preview_file.exists()


def test_validate_can_write_visual_preview_artifacts(tmp_path: Path) -> None:
    output_file = tmp_path / "visual.glb"
    preview_file = tmp_path / "preview.png"
    lod_dir = tmp_path / "lod-previews"
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    mesh = Mesh(
        points=points,
        faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    lod_mesh = Mesh(
        points=points[:3].copy(),
        faces=np.asarray([[0, 1, 2]], dtype=int),
        metadata={"lod_ratio": "0.5", "lod_screen_coverage": "0.25"},
    )
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh, lod_meshes=[lod_mesh])},
        up_axis="Y",
    ).write_gltf(output_file)

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output_file),
            "--visual-preview",
            str(preview_file),
            "--lod-preview-dir",
            str(lod_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["visual_preview"]["triangles"] == 2
    assert payload["lod_preview"]["levels"] == 2
    assert payload["lod_preview"]["monotonic_triangles"] is True
    assert preview_file.exists()
    assert (lod_dir / "lod0.png").exists()
    assert (lod_dir / "lod1.png").exists()
    assert (lod_dir / "lod-switching.png").exists()


def test_validate_can_compare_visual_preview_against_baseline(tmp_path: Path) -> None:
    output_file = tmp_path / "visual.glb"
    preview_file = tmp_path / "preview.png"
    baseline_file = tmp_path / "baseline.png"
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    ).write_gltf(output_file)
    write_output_preview(output_file, baseline_file)

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output_file),
            "--visual-preview",
            str(preview_file),
            "--visual-baseline",
            str(baseline_file),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["visual_diff"]["passed"] is True
    assert payload["visual_diff"]["changed_pixels"] == 0


def test_validate_fails_when_visual_diff_exceeds_threshold(tmp_path: Path) -> None:
    output_file = tmp_path / "visual.glb"
    preview_file = tmp_path / "preview.png"
    baseline_file = tmp_path / "baseline.png"
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
    )
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Triangle", part_id="part")]),
        parts={"part": Part(id="part", name="Triangle", mesh=mesh)},
        up_axis="Y",
    ).write_gltf(output_file)
    Image.new("RGBA", (512, 512), (0, 0, 0, 255)).save(baseline_file)

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output_file),
            "--visual-preview",
            str(preview_file),
            "--visual-baseline",
            str(baseline_file),
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["visual_diff"]["passed"] is False
    assert payload["visual_diff"]["changed_pixels"] > 0


def test_validate_rejects_visual_baseline_without_preview(tmp_path: Path) -> None:
    output_file = tmp_path / "visual.glb"
    baseline_file = tmp_path / "baseline.png"

    result = runner.invoke(app, ["validate", str(output_file), "--visual-baseline", str(baseline_file)])

    assert result.exit_code == 2
    assert "--visual-baseline requires --visual-preview" in result.output


def _write_turntable_glb(path: Path, *, triangle_only: bool = False) -> None:
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    if triangle_only:
        mesh = Mesh(points=points[:3].copy(), faces=np.asarray([[0, 1, 2]], dtype=int))
    else:
        mesh = Mesh(points=points, faces=np.asarray([[0, 1, 2], [2, 1, 3]], dtype=int))
    Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Shape", part_id="part")]),
        parts={"part": Part(id="part", name="Shape", mesh=mesh)},
        up_axis="Y",
    ).write_gltf(path)


def test_validate_can_write_turntable_previews(tmp_path: Path) -> None:
    output_file = tmp_path / "turntable.glb"
    turntable_dir = tmp_path / "views"
    _write_turntable_glb(output_file)

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output_file),
            "--turntable-dir",
            str(turntable_dir),
            "--turntable-views",
            "4",
            "--turntable-elevations",
            "0",
            "--turntable-width",
            "96",
            "--turntable-height",
            "96",
            "--turntable-supersample",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["turntable"]["view_count"] == 4
    assert payload["turntable"]["diff_passed"] is None
    names = [view["name"] for view in payload["turntable"]["views"]]
    assert names == ["az000_el+00", "az090_el+00", "az180_el+00", "az270_el+00"]
    for name in names:
        assert (turntable_dir / f"{name}.png").exists()
    assert (turntable_dir / "turntable.png").exists()


def test_validate_turntable_diff_passes_against_matching_baseline(tmp_path: Path) -> None:
    output_file = tmp_path / "turntable.glb"
    baseline_dir = tmp_path / "baseline-views"
    turntable_dir = tmp_path / "views"
    _write_turntable_glb(output_file)
    common = ["--turntable-views", "2", "--turntable-elevations", "30", "--turntable-width", "96"]
    common += ["--turntable-height", "96", "--turntable-supersample", "1"]
    baseline_run = runner.invoke(
        app, ["--json", "validate", str(output_file), "--turntable-dir", str(baseline_dir), *common]
    )
    assert baseline_run.exit_code == 0, baseline_run.output

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output_file),
            "--turntable-dir",
            str(turntable_dir),
            "--turntable-baseline-dir",
            str(baseline_dir),
            *common,
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["turntable"]["diff_passed"] is True
    assert payload["turntable"]["diff"]["views_failed"] == 0


def test_validate_fails_when_turntable_diff_exceeds_threshold(tmp_path: Path) -> None:
    baseline_file = tmp_path / "baseline.glb"
    output_file = tmp_path / "turntable.glb"
    baseline_dir = tmp_path / "baseline-views"
    turntable_dir = tmp_path / "views"
    _write_turntable_glb(baseline_file)
    _write_turntable_glb(output_file, triangle_only=True)
    common = ["--turntable-views", "2", "--turntable-elevations", "30", "--turntable-width", "96"]
    common += ["--turntable-height", "96", "--turntable-supersample", "1"]
    baseline_run = runner.invoke(
        app, ["--json", "validate", str(baseline_file), "--turntable-dir", str(baseline_dir), *common]
    )
    assert baseline_run.exit_code == 0, baseline_run.output

    result = runner.invoke(
        app,
        [
            "--json",
            "validate",
            str(output_file),
            "--turntable-dir",
            str(turntable_dir),
            "--turntable-baseline-dir",
            str(baseline_dir),
            *common,
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["turntable"]["diff_passed"] is False
    assert payload["turntable"]["diff"]["views_failed"] > 0
    assert payload["turntable"]["diff"]["worst_view"] in [view["name"] for view in payload["turntable"]["views"]]


def test_validate_rejects_turntable_baseline_without_turntable_dir(tmp_path: Path) -> None:
    output_file = tmp_path / "turntable.glb"
    baseline_dir = tmp_path / "baseline-views"

    result = runner.invoke(app, ["validate", str(output_file), "--turntable-baseline-dir", str(baseline_dir)])

    assert result.exit_code == 2
    assert "--turntable-baseline-dir requires --turntable-dir" in result.output


def test_validate_rejects_invalid_turntable_views(tmp_path: Path) -> None:
    output_file = tmp_path / "turntable.glb"
    turntable_dir = tmp_path / "views"

    result = runner.invoke(
        app,
        ["validate", str(output_file), "--turntable-dir", str(turntable_dir), "--turntable-views", "0"],
    )

    assert result.exit_code == 2
    assert "--turntable-views must be greater than 0" in result.output


@pytest.mark.parametrize("elevations", ["abc", "", "95", "-95,30"])
def test_validate_rejects_invalid_turntable_elevations(tmp_path: Path, elevations: str) -> None:
    output_file = tmp_path / "turntable.glb"
    turntable_dir = tmp_path / "views"

    result = runner.invoke(
        app,
        [
            "validate",
            str(output_file),
            "--turntable-dir",
            str(turntable_dir),
            "--turntable-elevations",
            elevations,
        ],
    )

    assert result.exit_code == 2
    assert "--turntable-elevations" in result.output


def test_validate_dry_run_includes_turntable_payload_fields(tmp_path: Path) -> None:
    output_file = tmp_path / "turntable.glb"
    turntable_dir = tmp_path / "views"

    result = runner.invoke(
        app,
        ["--json", "--dry-run", "validate", str(output_file), "--turntable-dir", str(turntable_dir)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["turntable_dir"] == str(turntable_dir)
    assert payload["turntable_views"] == 8
    assert payload["turntable_elevations"] == "-30,30"
    assert not turntable_dir.exists()


def test_validate_missing_usd_backend_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block_imports(monkeypatch, "pxr")
    output_file = tmp_path / "output.usda"
    output_file.write_text("#usda 1.0", encoding="utf-8")

    result = runner.invoke(app, ["validate", str(output_file)])

    assert result.exit_code == 1
    assert "USD validation requires usd-core" in result.output


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_validate_generated_usd(tmp_path: Path) -> None:
    output_file = tmp_path / "output.usda"
    convert_result = runner.invoke(
        app,
        [
            "convert",
            "tests/fixtures/spool-clamp-lid.step",
            str(output_file),
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
        ],
    )
    assert convert_result.exit_code == 0

    result = runner.invoke(app, ["validate", str(output_file)])
    assert result.exit_code == 0
    assert "valid USD" in compact(result.output)

    stdin_result = runner.invoke(app, ["validate", "-"], input=output_file.read_text(encoding="utf-8"))
    assert stdin_result.exit_code == 0
    assert "valid USD" in compact(stdin_result.output)
