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
    assert "--max-triangles" in plain(result.output)
    assert "--max-file-size-mb" in plain(result.output)
    assert "--strict-geometry" in plain(result.output)
    assert "--profile" in plain(result.output)


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


def _gate_results_by_name(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    gates = payload["gates"]
    assert isinstance(gates, dict)
    results = gates["results"]
    assert isinstance(results, list)
    return {result["gate"]: result for result in results}


def test_validate_gates_pass_with_thresholds(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(
        app,
        ["--json", "validate", str(output_file), "--max-non-manifold", "0", "--max-triangles", "10"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["gates"]["overall"] == "PASS"
    assert payload["gates"]["failed"] == 0
    results = _gate_results_by_name(payload)
    assert results["non_manifold_edges"]["status"] == "PASS"
    assert results["triangles"] == {"gate": "triangles", "status": "PASS", "actual": 2, "op": "<=", "limit": 10}
    assert payload["analysis"]["summary"]["non_manifold_edges"] == 0


def test_validate_gates_fail_sets_exit_code(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file, triangle_only=True)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--max-open-boundaries", "0"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["gates"]["overall"] == "FAIL"
    assert payload["gates"]["failed"] == 1
    results = _gate_results_by_name(payload)
    assert results["open_boundaries"]["status"] == "FAIL"
    assert results["open_boundaries"]["actual"] == 1
    assert results["open_boundaries"]["limit"] == 0


def test_validate_gates_human_output(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file, triangle_only=True)

    result = runner.invoke(app, ["validate", str(output_file), "--max-open-boundaries", "0", "--max-triangles", "10"])

    assert result.exit_code == 1
    output = plain(result.output)
    assert "PASS structural True == True" in output
    assert "FAIL open_boundaries 1 <= 0" in output
    assert "PASS triangles 1 <= 10" in output
    assert "SKIP turntable_diff None == 0" in output
    assert "OVERALL FAIL (1/3 evaluated gates failed)" in output


def test_validate_strict_geometry_shorthand(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file, triangle_only=True)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--strict-geometry"])

    assert result.exit_code == 1
    results = _gate_results_by_name(json.loads(result.output))
    assert results["non_manifold_edges"]["status"] == "PASS"
    assert results["self_intersections"]["status"] == "PASS"
    assert results["sliver_triangles"]["status"] == "PASS"
    assert results["open_boundaries"]["status"] == "FAIL"


def test_validate_strict_geometry_explicit_override(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file, triangle_only=True)

    result = runner.invoke(
        app,
        ["--json", "validate", str(output_file), "--strict-geometry", "--max-open-boundaries", "5"],
    )

    assert result.exit_code == 0, result.output
    results = _gate_results_by_name(json.loads(result.output))
    assert results["open_boundaries"] == {
        "gate": "open_boundaries",
        "status": "PASS",
        "actual": 1,
        "op": "<=",
        "limit": 5,
    }
    assert results["non_manifold_edges"]["limit"] == 0


def test_validate_gates_profile_budgets(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--profile", "realtime-web"])

    assert result.exit_code == 0, result.output
    results = _gate_results_by_name(json.loads(result.output))
    assert results["triangles"]["status"] == "PASS"
    assert isinstance(results["triangles"]["limit"], int)
    assert results["file_size_bytes"]["status"] == "PASS"
    assert results["file_size_bytes"]["actual"] == output_file.stat().st_size
    assert results["file_size_bytes"]["limit"] == 50 * 1024 * 1024


def test_validate_gates_inspect_only_skips_profile_budgets(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--profile", "inspect-only"])

    assert result.exit_code == 0, result.output
    results = _gate_results_by_name(json.loads(result.output))
    assert results["triangles"]["status"] == "SKIP"
    assert results["file_size_bytes"]["status"] == "SKIP"


def test_validate_gates_profile_explicit_triangle_override(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(
        app,
        ["--json", "validate", str(output_file), "--profile", "realtime-web", "--max-triangles", "1"],
    )

    assert result.exit_code == 1
    results = _gate_results_by_name(json.loads(result.output))
    assert results["triangles"] == {"gate": "triangles", "status": "FAIL", "actual": 2, "op": "<=", "limit": 1}


def test_validate_gates_profile_explicit_file_size_override(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(
        app,
        ["--json", "validate", str(output_file), "--profile", "realtime-web", "--max-file-size-mb", "0"],
    )

    assert result.exit_code == 1
    results = _gate_results_by_name(json.loads(result.output))
    assert results["file_size_bytes"]["status"] == "FAIL"
    assert results["file_size_bytes"]["limit"] == 0


@pytest.mark.parametrize(
    ("limit_mb", "expected_exit", "expected_status"),
    [("100", 0, "PASS"), ("0.000001", 1, "FAIL")],
)
def test_validate_gates_file_size(tmp_path: Path, limit_mb: str, expected_exit: int, expected_status: str) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--max-file-size-mb", limit_mb])

    assert result.exit_code == expected_exit
    results = _gate_results_by_name(json.loads(result.output))
    assert results["file_size_bytes"]["status"] == expected_status
    assert results["file_size_bytes"]["actual"] == output_file.stat().st_size


@pytest.mark.parametrize(
    "flag",
    [
        "--max-non-manifold",
        "--max-self-intersections",
        "--max-slivers",
        "--max-open-boundaries",
        "--max-triangles",
        "--max-file-size-mb",
    ],
)
def test_validate_rejects_negative_gate_thresholds(tmp_path: Path, flag: str) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(app, ["validate", str(output_file), flag, "-1"])

    assert result.exit_code == 2
    assert f"{flag} must be greater than or equal to 0." in result.output


def test_validate_rejects_unknown_profile(tmp_path: Path) -> None:
    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(app, ["validate", str(output_file), "--profile", "no-such-profile"])

    assert result.exit_code == 2


def test_validate_turntable_failure_appears_in_gates(tmp_path: Path) -> None:
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
            "--max-triangles",
            "10",
            "--turntable-dir",
            str(turntable_dir),
            "--turntable-baseline-dir",
            str(baseline_dir),
            *common,
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["gates"]["overall"] == "FAIL"
    results = _gate_results_by_name(payload)
    assert results["turntable_diff"]["status"] == "FAIL"
    assert results["turntable_diff"]["actual"] == payload["turntable"]["diff"]["views_failed"]
    assert results["triangles"]["status"] == "PASS"


def test_validate_no_gates_without_assertion_flags(tmp_path: Path) -> None:
    output_file = tmp_path / "plain.glb"
    _write_turntable_glb(output_file)

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--geometry-quality"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "gates" not in payload
    assert "OVERALL" not in plain(result.output)


def test_validate_structural_failure_reports_structural_gate(tmp_path: Path) -> None:
    output_file = tmp_path / "broken.glb"
    output_file.write_bytes(b"not a real glb")

    result = runner.invoke(app, ["--json", "validate", str(output_file), "--max-triangles", "10"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert "error" in payload
    assert payload["gates"]["overall"] == "FAIL"
    results = _gate_results_by_name(payload)
    assert results["structural"] == {
        "gate": "structural",
        "status": "FAIL",
        "actual": False,
        "op": "==",
        "limit": True,
    }


def test_validate_structural_failure_human_gate_lines(tmp_path: Path) -> None:
    output_file = tmp_path / "broken.glb"
    output_file.write_bytes(b"not a real glb")

    result = runner.invoke(app, ["validate", str(output_file), "--max-triangles", "10"])

    assert result.exit_code == 1
    output = plain(result.output)
    assert "FAIL structural False == True" in output
    assert "OVERALL FAIL (1/1 evaluated gates failed)" in output


@pytest.mark.parametrize("limit", [1, 2, 10])
@pytest.mark.parametrize("json_output", [False, True])
def test_validate_incomplete_intersection_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int, json_output: bool
) -> None:
    from dataclasses import replace

    from fascat.cli import _cmd_validate
    from fascat.options import AnalyzeOptions

    points = np.tile(np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), (4, 1))
    mesh = Mesh(points=points, faces=np.arange(12).reshape(4, 3))
    asset = Asset(
        root=Node(id="root", name="root", part_id="mesh"), parts={"mesh": Part(id="mesh", name="mesh", mesh=mesh)}
    )
    complete = asset.analyze(AnalyzeOptions(self_intersections=True))
    assert complete.summary["self_intersections"] == 6
    assert complete.summary["self_intersections_lower_bound"] is False
    output_file = tmp_path / "coincident.glb"
    asset.write_gltf(output_file)
    original_options = _cmd_validate._analyze_options

    def bounded_options(**flags: bool) -> AnalyzeOptions:
        return replace(original_options(**flags), max_self_intersection_pairs=2)

    monkeypatch.setattr(_cmd_validate, "_analyze_options", bounded_options)
    args = ["validate", str(output_file), "--max-self-intersections", str(limit)]
    result = runner.invoke(app, ["--json", *args] if json_output else args)

    assert result.exit_code == 1, result.output
    reason = "lower bound exceeds limit" if limit < 2 else "incomplete measurement cannot establish compliance"
    if json_output:
        payload = json.loads(result.output)
        assert payload["analysis"]["summary"]["self_intersections"] == 2
        assert payload["analysis"]["summary"]["self_intersections_lower_bound"] is True
        gate = _gate_results_by_name(payload)["self_intersections"]
        assert gate["status"] == "FAIL"
        assert gate["actual_lower_bound"] is True
        assert gate["reason"] == reason
        assert payload["gates"]["overall"] == "FAIL"
        assert payload["gates"]["failed"] == 1
        assert payload["gates"]["evaluated"] == 2
    else:
        output = compact(result.output)
        assert f"FAIL self_intersections >=2 <= {limit} ({reason})" in output
        assert "OVERALL FAIL (1/2 evaluated gates failed)" in output


@pytest.mark.requires_usd
@pytest.mark.parametrize("from_stdin", [False, True])
def test_validate_unavailable_requested_geometry_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, from_stdin: bool
) -> None:
    from fascat import analysis

    def unavailable_geometry(path: Path) -> Asset:
        raise RuntimeError("geometry unavailable")

    # Exercise the real validation-only fallback when mesh analysis cannot load.
    monkeypatch.setattr(analysis, "_asset_from_output", unavailable_geometry)
    output_file = tmp_path / "triangle.usda"
    output_file.write_text(
        """#usda 1.0
(defaultPrim = "Triangle")
def Mesh "Triangle" {
    uniform token subdivisionScheme = "none"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
""",
        encoding="utf-8",
    )
    args = ["--json", "validate", "-" if from_stdin else str(output_file), "--strict-geometry"]
    result = runner.invoke(app, args, input=output_file.read_bytes() if from_stdin else None)

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    gates = _gate_results_by_name(payload)
    for name in ["non_manifold_edges", "self_intersections", "sliver_triangles", "open_boundaries"]:
        assert gates[name]["status"] == "FAIL"
        assert gates[name]["actual"] is None
        assert gates[name]["reason"] == "measurement unavailable"
    assert gates["structural"]["status"] == "PASS"
    assert gates["visual_diff"]["status"] == "SKIP"
    assert payload["gates"]["failed"] == 4


@pytest.mark.requires_usd
@pytest.mark.parametrize("budget_args", [["--max-file-size-mb", "1"], ["--profile", "realtime-web"]])
def test_validate_stdin_file_size_gate_fails(budget_args: list[str]) -> None:
    result = runner.invoke(
        app,
        ["--json", "validate", "-", *budget_args],
        input="""#usda 1.0
(defaultPrim = "Triangle")
def Mesh "Triangle" {
    uniform token subdivisionScheme = "none"
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
}
""",
    )

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert "analysis" not in payload
    gate = _gate_results_by_name(payload)["file_size_bytes"]
    assert gate["status"] == "FAIL"
    assert gate["actual"] is None
    assert gate["reason"] == "measurement unavailable"
    assert payload["gates"]["overall"] == "FAIL"


@pytest.mark.parametrize("json_output", [False, True])
def test_validate_requested_limits_without_analysis_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, json_output: bool
) -> None:
    from fascat.cli import _cmd_validate

    def no_analysis(*args: object, **kwargs: object) -> tuple[dict[str, int], None]:
        return {"meshes": 1}, None

    output_file = tmp_path / "gated.glb"
    _write_turntable_glb(output_file)
    monkeypatch.setattr(_cmd_validate, "_validate_and_analyze_output_for_cli", no_analysis)
    args = ["validate", str(output_file), "--max-non-manifold", "0", "--max-triangles", "10"]
    result = runner.invoke(app, ["--json", *args] if json_output else args)

    assert result.exit_code == 1, result.output
    if json_output:
        payload = json.loads(result.output)
        assert "analysis" not in payload
        gates = _gate_results_by_name(payload)
        for name in ["non_manifold_edges", "triangles"]:
            assert gates[name]["status"] == "FAIL"
            assert gates[name]["reason"] == "measurement unavailable"
        assert payload["gates"]["failed"] == 2
        assert payload["gates"]["evaluated"] == 3
    else:
        assert "FAIL triangles None <= 10 (measurement unavailable)" in compact(result.output)
        assert "OVERALL FAIL (2/3 evaluated gates failed)" in compact(result.output)
