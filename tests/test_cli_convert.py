import io
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from fascat.asset import Asset, Node
from fascat.cli import app
from fascat.options import StepReadOptions
from fascat.report import Report

from ._cli_test_helpers import (
    block_imports,
    compact,
    invoke_run,
    plain,
    runner,
)


def test_convert_help() -> None:
    result = runner.invoke(app, ["convert", "--help"])
    assert result.exit_code == 0
    assert "--target-triangles" in plain(result.output)
    assert "--pipeline" in plain(result.output)
    assert "--min-edge-length" in plain(result.output)
    assert "--max-edge-length" in plain(result.output)
    assert "--max-polygon-length" in plain(result.output)
    assert "--quality-report" in plain(result.output)
    assert "--materials" in plain(result.output)
    assert "--material-mode" in plain(result.output)
    assert "--atlas-size" in plain(result.output)
    assert "--normals" in plain(result.output)
    assert "--tangents" in plain(result.output)
    assert "--uv1" in plain(result.output)
    assert "--normalize-uvs" in plain(result.output)
    assert "--no-preserve-instances" in plain(result.output)
    assert "--preserve-hard-edges" in plain(result.output)
    assert "material boundaries" in plain(result.output)
    assert "--batch-by-material" in plain(result.output)
    assert "--index-buffer" in plain(result.output)
    assert "--bake-materials" in plain(result.output)
    assert "--decimate" in plain(result.output)
    assert "iterative" in plain(result.output)
    assert "--preserve-painted-areas" in plain(result.output)
    assert "low-AO faces" in plain(result.output)
    assert "--uv-importance" in plain(result.output)
    assert "--sag-ratio" in plain(result.output)
    assert "--reuse-existing-meshes" in plain(result.output)
    assert "--free-edge-report" in plain(result.output)
    assert "--unwrap-method" in plain(result.output)
    assert "--unwrap-iterations" in plain(result.output)
    assert "--unwrap-tolerance" in plain(result.output)
    assert "--uv-sharp-to-seam" in plain(result.output)
    assert "--uv-forbid-overlapping" in plain(result.output)
    assert "--remove-holes" in plain(result.output)
    assert "--remove-occluded" in plain(result.output)
    assert "--explode" in plain(result.output)
    assert "--replace" in plain(result.output)
    assert "--run-lod-generators" in plain(result.output)
    assert "--lod-mode" in plain(result.output)
    assert "--lod-per-part-budget" in plain(result.output)
    assert "--lod-drop-tiny-parts" in plain(result.output)
    assert "--quantize" in plain(result.output)
    assert "--meshopt" in plain(result.output)
    assert "--file-size-budget-mb" in plain(result.output)
    assert "--obj-materials" in plain(result.output)
    assert "--stl-binary" in plain(result.output)


def test_convert_dry_run_json() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--sag-ratio",
            "0.01",
            "--lod-engine-profile",
            "unreal",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["command"] == "convert"
    assert payload["dry_run"] is True
    assert payload["sag_ratio"] == 0.01
    assert payload["max_polygon_length"] is None
    assert payload["free_edge_report"] is False
    assert payload["reuse_existing_meshes"] is True
    assert payload["lod_engine_profile"] == "unreal"
    assert payload["unwrap_method"] == "default"
    assert payload["unwrap_iterations"] is None
    assert payload["unwrap_tolerance"] is None
    assert payload["uv_sharp_to_seam"] is False
    assert payload["uv_forbid_overlapping"] is False
    diagnostics = {item["operation"]: item for item in payload["operation_diagnostics"]}
    assert diagnostics["import"]["level"] == "exact"
    assert diagnostics["tessellate"]["level"] == "exact"
    assert diagnostics["export"]["level"] == "exact"


def test_convert_dry_run_reports_approximate_and_metadata_only_operations() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--heal-brep",
            "--remove-sliver-faces",
            "--atlas",
            "--bake-materials",
            "--decimate",
            "--decimate-criterion",
            "quality",
            "--remove-holes",
            "--remove-occluded",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    diagnostics = {item["operation"]: item for item in payload["operation_diagnostics"]}
    assert diagnostics["heal_brep"]["level"] == "approximate"
    assert diagnostics["atlas"]["level"] == "metadata_only"
    assert diagnostics["bake_materials"]["level"] == "exact"
    assert diagnostics["decimate"]["level"] == "exact"
    assert diagnostics["remove_holes"]["level"] == "approximate"
    assert diagnostics["remove_occluded"]["level"] == "approximate"


def test_convert_dry_run_reports_merge_vertices_operation() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--merge-vertices",
            "--merge-vertex-tolerance",
            "0.001",
            "--drop-merge-vertex-attributes",
            "--ignore-merge-vertex-material-boundaries",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    diagnostics = {item["operation"]: item for item in payload["operation_diagnostics"]}
    assert payload["merge_vertices"] is True
    assert payload["merge_vertex_tolerance"] == 0.001
    assert payload["preserve_merge_vertex_attributes"] is False
    assert payload["preserve_merge_vertex_material_boundaries"] is False
    assert payload["merge_vertex_quality_report"] is False
    assert diagnostics["merge_vertices"]["level"] == "exact"


def test_convert_dry_run_reports_delete_degenerate_polygons_operation() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--delete-degenerate-polygons",
            "--degenerate-area-epsilon",
            "0.00001",
            "--keep-duplicate-polygons",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    diagnostics = {item["operation"]: item for item in payload["operation_diagnostics"]}
    assert payload["delete_degenerate_polygons"] is True
    assert payload["degenerate_area_epsilon"] == 0.00001
    assert payload["delete_duplicate_polygons"] is False
    assert diagnostics["delete_degenerate_polygons"]["level"] == "exact"


def test_convert_dry_run_defaults_output_to_usdc() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step"])
    assert result.exit_code == 0
    assert '"output": "input.usdc"' in result.output


def test_convert_dry_run_accepts_material_staging_mode() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "--materials", "display"])
    assert result.exit_code == 0
    assert '"materials": "display"' in result.output


def test_convert_dry_run_accepts_usd_layout() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "--usd-layout", "flat"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["usd_layout"] == "flat"


def test_convert_dry_run_accepts_uv1_copy_mode() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "--uv1", "copy-uv0"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["uv1"] == "copy-uv0"


def test_convert_dry_run_accepts_uv_normalization_channels() -> None:
    result = runner.invoke(app, ["--json", "--dry-run", "convert", "input.step", "--normalize-uvs", "1,0,1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["normalize_uvs"] == [1, 0]


def test_convert_dry_run_accepts_aabb_uv_projection_controls() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "--uv-aabb-scope",
            "shared",
            "--uv3d-size",
            "2.5",
            "--uv-preserve-existing",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["uv_aabb_scope"] == "shared"
    assert payload["uv3d_size"] == 2.5
    assert payload["uv_override_existing"] is False


def test_convert_dry_run_accepts_tangent_uv_channel() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "--tangents",
            "--tangent-uv-channel",
            "1",
            "--override-tangents",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["tangents"] is True
    assert payload["tangent_uv_channel"] == 1
    assert payload["override_tangents"] is True


def test_convert_dry_run_accepts_normal_generation_controls() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "--normals",
            "smooth",
            "--normal-weighting",
            "area",
            "--preserve-normals",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["normals"] == "smooth"
    assert payload["normal_weighting"] == "area"
    assert payload["override_normals"] is False


def test_convert_dry_run_accepts_gltf_output_and_mobile_profile() -> None:
    result = runner.invoke(
        app,
        ["--json", "--dry-run", "convert", "input.step", "output.glb", "--profile", "realtime-mobile"],
    )
    assert result.exit_code == 0
    assert '"output": "output.glb"' in result.output
    assert '"profile": "realtime-mobile"' in result.output


@pytest.mark.parametrize("profile", ["augmented-reality", "mixed-reality"])
def test_convert_dry_run_accepts_xr_device_profiles(profile: str) -> None:
    result = runner.invoke(
        app,
        ["--json", "--dry-run", "convert", "input.step", "output.glb", "--profile", profile],
    )
    assert result.exit_code == 0
    assert f'"profile": "{profile}"' in result.output


def test_convert_dry_run_accepts_target_device_profile_file(tmp_path: Path) -> None:
    profile_file = tmp_path / "phone-ar.toml"
    profile_file.write_text(
        """
name = "phone-ar-low"

[budget]
max_triangles = 42000
max_texture_resolution = 512
supported_compression = ["meshopt"]
supported_runtime_extensions = ["KHR_mesh_quantization", "EXT_meshopt_compression"]
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--profile",
            "realtime-mobile",
            "--target-device-profile",
            str(profile_file),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["profile"] == "phone-ar-low"
    assert payload["base_profile"] == "realtime-mobile"
    assert payload["target_device_profile"] == str(profile_file)
    assert payload["profile_options"]["budget"]["max_triangles"] == 42_000
    assert payload["profile_options"]["budget"]["max_vertices"] == 126_000
    assert payload["profile_options"]["budget"]["max_texture_resolution"] == 512
    assert payload["profile_options"]["budget"]["max_draw_calls"] == 250
    assert payload["profile_options"]["budget"]["supported_compression"] == ["meshopt"]
    assert payload["profile_options"]["budget"]["supported_runtime_extensions"] == [
        "KHR_mesh_quantization",
        "EXT_meshopt_compression",
    ]
    assert payload["profile_options"]["optimize"]["target_triangles"] == 42_000


def test_convert_dry_run_seeds_decimation_from_target_device_profile(tmp_path: Path) -> None:
    profile_file = tmp_path / "factory-tablet.toml"
    profile_file.write_text(
        """
name = "factory-tablet"

[budget]
max_triangles = 42000
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--profile",
            "realtime-mobile",
            "--target-device-profile",
            str(profile_file),
            "--decimate",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["target_triangles"] is None
    assert payload["decimate_target_triangles"] == 42_000
    assert payload["decimate_target_source"] == "profile_budget"


def test_convert_dry_run_accepts_pipeline_file(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "realtime.toml"
    pipeline_file.write_text(
        """
[import]
metadata = "none"
pmi = false
design_variants = true
design_variant_selection = ["left hand"]
existing_meshes = false
multi_file = true
delete_free_vertices = true
delete_lines = true
construction_curve_policy = "tessellate_tubes"
construction_curve_tube_radius = 0.025
source_units = "millimetre"
source_up_axis = "Z"
source_handedness = "right"
target_units = "metre"
target_up_axis = "Y"
target_handedness = "right"

[export]
metadata = "summary"
pmi = "none"

[[filters]]
name = "fasteners"
path = "*/Fasteners/*"
names = ["Bolt*"]

[[steps]]
op = "tessellate"
where = "fasteners"
sag = 0.2
sag-ratio = 0.01
max-polygon-length = 3.0
free-edge-report = true
reuse-existing-meshes = false

[[steps]]
op = "optimize"
where_not = "fasteners"
target_triangles = 80000
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["--json", "--dry-run", "convert", "input.step", "output.glb", "--pipeline", str(pipeline_file)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["pipeline"] == str(pipeline_file)
    assert payload["pipeline_import"]["metadata"] is False
    assert payload["pipeline_import"]["pmi"] is False
    assert payload["pipeline_import"]["design_variants"] is True
    assert payload["pipeline_import"]["design_variant_selection"] == ["left hand"]
    assert payload["pipeline_import"]["existing_meshes"] is False
    assert payload["pipeline_import"]["multi_file"] is True
    assert payload["pipeline_import"]["delete_free_vertices"] is True
    assert payload["pipeline_import"]["delete_lines"] is True
    assert payload["pipeline_import"]["construction_curve_policy"] == "tessellate_tubes"
    assert payload["pipeline_import"]["construction_curve_tube_radius"] == 0.025
    assert payload["pipeline_import"]["source_units"] == "millimetre"
    assert payload["pipeline_import"]["target_units"] == "metre"
    assert payload["pipeline_import"]["target_up_axis"] == "Y"
    assert payload["pipeline_import"]["target_handedness"] == "right"
    assert payload["pipeline_export"] == {"mode": "summary", "pmi": "none"}
    assert payload["pipeline_advisories"] == []
    assert payload["pipeline_filters"] == ["fasteners"]
    assert [step["op"] for step in payload["pipeline_steps"]] == ["tessellate", "optimize"]
    assert payload["pipeline_steps"][0]["sag_ratio"] == 0.01
    assert payload["pipeline_steps"][0]["max_polygon_length"] == 3.0
    assert payload["pipeline_steps"][0]["free_edge_report"] is True
    assert payload["pipeline_steps"][0]["reuse_existing_meshes"] is False


@pytest.mark.parametrize("input_name", ["legacy.igs", "legacy.iges", "native.brep", "model.jt"])
def test_convert_dry_run_accepts_non_step_cad_inputs(input_name: str) -> None:
    output_name = "output.glb"

    result = runner.invoke(app, ["--json", "--dry-run", "convert", input_name, output_name])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["input"] == input_name
    assert payload["output"] == output_name


def test_convert_jt_lod_selection_flag(tmp_path: Path) -> None:
    from tests._jt_builder import SyntheticPart, build_jt

    tetra_points = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    tetra_faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    coarser = ([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], tetra_faces)
    part = SyntheticPart(name="tetra.part", points=tetra_points, triangles=tetra_faces, lod_meshes=[coarser])
    source = tmp_path / "model.jt"
    source.write_bytes(build_jt([part]))
    output = tmp_path / "model.glb"
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["convert", str(source), str(output), "--jt-lod-selection", "all", "--report", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    import_step = next(step for step in report["steps"] if step["name"] == "import")
    assert import_step["options"]["format"] == "JT"
    assert import_step["options"]["lod_summary"]["lod_selection"] == "all"
    assert import_step["options"]["lod_summary"]["imported_lod_meshes"] == 1
    assert "--jt-lod-selection is deprecated; use --lod-selection instead" in result.stderr


def test_convert_generic_lod_selection_flag(tmp_path: Path) -> None:
    from tests._jt_builder import SyntheticPart, build_jt

    points = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    faces = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]
    source = tmp_path / "model.jt"
    source.write_bytes(build_jt([SyntheticPart(name="part", points=points, triangles=faces)]))
    report_path = tmp_path / "report.json"

    result = runner.invoke(
        app,
        ["convert", str(source), str(tmp_path / "model.glb"), "--lod-selection", "all", "--report", str(report_path)],
    )

    assert result.exit_code == 0, result.output
    report = json.loads(report_path.read_text())
    imported = next(step for step in report["steps"] if step["name"] == "import")
    assert imported["options"]["lod_summary"]["lod_selection"] == "all"


def test_convert_warns_when_lod_selection_is_ignored_for_non_jt_input() -> None:
    result = runner.invoke(app, ["--dry-run", "convert", "model.step", "out.glb", "--lod-selection", "all"])

    assert result.exit_code == 0, result.output
    assert "--lod-selection is ignored for non-JT inputs" in result.stderr


def test_convert_rejects_conflicting_lod_selection_flags() -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "convert", "model.jt", "out.glb", "--lod-selection", "all", "--jt-lod-selection", "finest"],
    )

    assert result.exit_code == 2
    assert "must not conflict" in result.output


def test_convert_dry_run_accepts_extra_step_inputs() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "root-a.step",
            "assembly.glb",
            "--input",
            "root-b.step",
            "--input",
            "root-c.stp",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["input"] == ["root-a.step", "root-b.step", "root-c.stp"]
    assert payload["extra_inputs"] == ["root-b.step", "root-c.stp"]


def test_convert_rejects_non_step_extra_inputs() -> None:
    result = runner.invoke(
        app,
        ["--dry-run", "convert", "root-a.step", "assembly.glb", "--input", "legacy.iges"],
    )

    assert result.exit_code == 2
    assert "multi-root CLI import currently supports only STEP" in result.output


def test_convert_passes_extra_step_inputs_to_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat as fc
    from fascat.cli import _io_helpers

    input_a = tmp_path / "root-a.step"
    input_b = tmp_path / "root-b.step"
    output = tmp_path / "assembly.glb"
    input_a.write_text("ISO-10303-21;", encoding="utf-8")
    input_b.write_text("ISO-10303-21;", encoding="utf-8")
    asset = fc.Asset(root=fc.Node(id="root", name="root"))
    captured: dict[str, object] = {}

    def fake_convert(input_path: object, *_args: object, **_kwargs: object) -> fc.Asset:
        captured["input_path"] = input_path
        return asset

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", fake_convert)

    result = runner.invoke(app, ["convert", str(input_a), str(output), "--input", str(input_b)])

    assert result.exit_code == 0, result.output
    assert captured["input_path"] == [input_a, input_b]


@pytest.mark.parametrize(
    ("suffix", "option_name"),
    [
        (".glb", "gltf_options"),
        (".usda", "usd_options"),
        (".obj", "obj_options"),
        (".stl", "stl_options"),
        (".fbx", "fbx_options"),
    ],
)
@pytest.mark.parametrize(("arguments", "expected_mb"), [([], 50.0), (["--file-size-budget-mb", "7.5"], 7.5)])
def test_convert_export_budget_inherits_profile_and_allows_explicit_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
    option_name: str,
    arguments: list[str],
    expected_mb: float,
) -> None:
    import fascat as fc
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    output_file = tmp_path / f"output{suffix}"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_convert(_input_path: object, *_args: object, **kwargs: object) -> fc.Asset:
        captured.update(kwargs)
        return fc.Asset(root=fc.Node(id="root", name="root"))

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", fake_convert)

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_file),
            str(output_file),
            "--profile",
            "realtime-web",
            *arguments,
        ],
    )

    assert result.exit_code == 0, result.output
    export_options = cast(Any, captured[option_name])
    assert export_options.file_size_budget_mb == expected_mb


def test_convert_passes_material_library_paths_to_import_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fascat as fc
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    output_file = tmp_path / "output.glb"
    library = tmp_path / "vendor-materials.json"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    library.write_text('{"materials":[]}', encoding="utf-8")
    asset = fc.Asset(root=fc.Node(id="root", name="root"))
    captured: dict[str, object] = {}

    def fake_convert(_input_path: object, *_args: object, **kwargs: object) -> fc.Asset:
        captured["import_options"] = kwargs["import_options"]
        return asset

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", fake_convert)

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_file),
            str(output_file),
            "--material-library",
            str(library),
            "--material-library-color-space",
            "linear",
        ],
    )

    assert result.exit_code == 0, result.output
    import_options = captured["import_options"]
    assert isinstance(import_options, StepReadOptions)
    assert import_options.material_library_paths == (str(library),)
    assert import_options.material_library_color_space == "linear"


def test_convert_dry_run_reports_pipeline_advisories(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "bad-order.toml"
    pipeline_file.write_text(
        """
[[steps]]
op = "decimate"

[[steps]]
op = "stage"
tangents = true
uv0 = "none"

[[steps]]
op = "bake_materials"
bake = ["ao"]

[[steps]]
op = "lods"
""",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        ["--json", "--dry-run", "convert", "input.step", "output.glb", "--pipeline", str(pipeline_file)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert [item["code"] for item in payload["pipeline_advisories"]] == [
        "decimate_before_repair",
        "tangents_without_uv0",
        "ao_bake_without_uv1",
        "lods_before_optimize",
    ]


def test_convert_rejects_invalid_pipeline_file(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "bad.toml"
    pipeline_file.write_text('[[steps]]\nwhere = "missing"\n', encoding="utf-8")

    result = runner.invoke(app, ["--dry-run", "convert", "input.step", "output.glb", "--pipeline", str(pipeline_file)])

    assert result.exit_code == 2
    assert "Invalid pipeline file" in result.output
    assert "line 1: pipeline step entries require an op" in result.output


def test_convert_missing_input_file_fails_before_processing(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["convert", "missing.step", "output.usdc"], capsys)

    assert result.exit_code == 1
    assert "Missing input file: missing.step" in result.stderr
    assert result.stdout == ""


def test_convert_rejects_inspect_only_profile_before_processing(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    input_file = tmp_path / "input.step"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")

    result = invoke_run(
        ["convert", str(input_file), str(tmp_path / "output.usdc"), "--profile", "inspect-only"], capsys
    )

    assert result.exit_code == 2
    assert "The inspect-only profile cannot be used for conversion." in result.stderr
    assert result.stdout == ""


def test_convert_missing_step_backend_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    block_imports(monkeypatch, "OCP")
    step_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(step_file), str(output_file)])

    assert result.exit_code == 1
    assert "STEP import requires cadquery-ocp" in result.output


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_defaults_to_binary_usdc_and_validates(tmp_path: Path) -> None:
    input_file = tmp_path / "spool.step"
    output_file = input_file.with_suffix(".usdc")
    shutil.copyfile("tests/fixtures/spool-clamp-lid.step", input_file)

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_file),
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert f"Converted {input_file} to {output_file}" in compact(result.output)

    validate_result = runner.invoke(app, ["validate", str(output_file)])
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_explicit_binary_usdc_and_validates(tmp_path: Path) -> None:
    output_file = tmp_path / "explicit.usdc"

    result = runner.invoke(
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

    assert result.exit_code == 0
    assert output_file.read_bytes().startswith(b"PXR-USDC")
    assert f"Converted {Path('tests/fixtures/spool-clamp-lid.step')} to" in compact(result.output)
    assert "1 parts" in compact(result.output)

    validate_result = runner.invoke(app, ["validate", str(output_file)])
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


@pytest.mark.requires_ocp
def test_convert_explicit_binary_glb_and_validates(tmp_path: Path) -> None:
    output_file = tmp_path / "explicit.glb"

    result = runner.invoke(
        app,
        [
            "convert",
            "tests/fixtures/spool-clamp-lid.step",
            str(output_file),
            "--profile",
            "virtual-reality",
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert output_file.read_bytes().startswith(b"glTF")
    assert f"Converted {Path('tests/fixtures/spool-clamp-lid.step')} to" in compact(result.output)

    validate_result = runner.invoke(app, ["validate", str(output_file)])
    assert validate_result.exit_code == 0
    assert "valid glTF" in compact(validate_result.output)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_fixture_writes_usd_and_report(tmp_path: Path) -> None:
    output_file = tmp_path / "output.usda"
    report_file = tmp_path / "report.json"

    result = runner.invoke(
        app,
        [
            "convert",
            "tests/fixtures/spool-clamp-lid.step",
            str(output_file),
            "--sag",
            "0.2",
            "--angle",
            "20",
            "--max-edge-length",
            "1000",
            "--target-triangles",
            "120",
            "--lods",
            "0.5",
            "--uv1",
            "box",
            "--materials",
            "display",
            "--report",
            str(report_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()
    assert report_file.exists()
    assert "Converted" in result.output
    report = json.loads(report_file.read_text(encoding="utf-8"))
    step_names = [step["name"] for step in report["steps"]]
    assert "write" in step_names
    assert "validate" in step_names
    assert report["source_path"].endswith("spool-clamp-lid.step")
    assert report["finished_at"] is not None
    assert any("uv1 violates lightmap/baking constraints" in warning for warning in report["warnings"])
    assert report["errors"] == []
    assert report["input_stats"]["parts"] == 1
    assert report["input_stats"]["materials"] == 1
    assert report["output_stats"]["parts"] == 1
    assert report["output_stats"]["materials"] == 0
    assert report["output_stats"]["triangles"] <= 120
    assert all(isinstance(step["warnings"], list) for step in report["steps"])
    assert all(step["duration"] >= 0.0 for step in report["steps"])
    assert all(isinstance(step["before"], dict) and isinstance(step["after"], dict) for step in report["steps"])


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_json_output_includes_stats_and_report(tmp_path: Path) -> None:
    output_file = tmp_path / "output.usda"

    result = runner.invoke(
        app,
        [
            "--json",
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

    assert result.exit_code == 0, result.output
    assert output_file.exists()
    payload = json.loads(result.output)
    step_names = [step["name"] for step in payload["report"]["steps"]]
    assert payload["command"] == "convert"
    assert payload["stats"]["parts"] == 1
    assert payload["stats"]["triangles"] > 0
    assert payload["report"]["input_stats"]["parts"] == 1
    assert payload["report"]["finished_at"] is not None
    assert step_names[-6:] == [
        "write",
        "validate",
        "workflow_recipe",
        "conversion_manifest",
        "workflow_summary",
        "profile_budget",
    ]


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_debug_usda_authors_debug_metadata(tmp_path: Path) -> None:
    from pxr import Usd

    output_file = tmp_path / "debug.usda"

    result = runner.invoke(
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
            "--debug",
        ],
    )

    assert result.exit_code == 0, result.output
    stage = Usd.Stage.Open(str(output_file))
    assert stage is not None
    assert stage.GetRootLayer().comment == "Generated by fascat debug mode"
    assert stage.GetDefaultPrim().GetCustomDataByKey("fascat:debug") is True

    validate_result = runner.invoke(app, ["validate", str(output_file)])
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_material_modes_write_cad_display_and_no_material_usd(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    fixture = "tests/fixtures/radial-fan-50x15.step"
    expected_display_color = pytest.approx(
        (0.009721217676997185, 0.009721217676997185, 0.009721217676997185),
        abs=1e-6,
    )
    cad_output = tmp_path / "cad.usda"
    display_output = tmp_path / "display.usda"
    none_output = tmp_path / "none.usda"

    for mode, output in (("cad", cad_output), ("display", display_output), ("none", none_output)):
        result = runner.invoke(
            app,
            [
                "convert",
                fixture,
                str(output),
                "--sag",
                "0.2",
                "--target-triangles",
                "80",
                "--lods",
                "0.5",
                "--materials",
                mode,
            ],
        )
        assert result.exit_code == 0

    cad_stage = Usd.Stage.Open(str(cad_output))
    display_stage = Usd.Stage.Open(str(display_output))
    none_stage = Usd.Stage.Open(str(none_output))
    assert cad_stage is not None
    assert display_stage is not None
    assert none_stage is not None
    cad_mesh = next(prim for prim in Usd.PrimRange(cad_stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    display_mesh = next(prim for prim in Usd.PrimRange(display_stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))
    none_mesh = next(prim for prim in Usd.PrimRange(none_stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh))

    assert cad_stage.GetPrimAtPath("/Materials")
    assert "MaterialBindingAPI" in cad_mesh.GetAppliedSchemas()

    assert not display_stage.GetPrimAtPath("/Materials")
    assert "MaterialBindingAPI" not in display_mesh.GetAppliedSchemas()
    assert tuple(UsdGeom.Mesh(display_mesh).GetDisplayColorAttr().Get()[0]) == expected_display_color

    assert not none_stage.GetPrimAtPath("/Materials")
    assert "MaterialBindingAPI" not in none_mesh.GetAppliedSchemas()
    assert tuple(UsdGeom.Mesh(none_mesh).GetDisplayColorAttr().Get()[0]) == (0.75, 0.75, 0.75)


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_reads_step_from_stdin_and_writes_usd_to_stdout() -> None:
    step_data = Path("tests/fixtures/spool-clamp-lid.step").read_text(encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "convert",
            "-",
            "-",
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
            "--debug",
        ],
        input=step_data,
    )

    assert result.exit_code == 0
    assert "#usda" in result.stdout
    assert 'def Xform "Scene"' in result.stdout
    assert "Converted" not in result.stdout

    validate_result = runner.invoke(app, ["validate", "-"], input=result.stdout)
    assert validate_result.exit_code == 0
    assert "valid USD" in compact(validate_result.output)


def test_convert_output_uses_requested_stdout_format(monkeypatch: pytest.MonkeyPatch) -> None:
    import click

    from fascat.cli import StdoutFormat, _io_helpers

    stdout = io.BytesIO()
    temp_paths: list[Path] = []

    def fake_convert(_input_path: object, output_path: str | Path, **_kwargs: object) -> Asset:
        output = Path(output_path)
        temp_paths.append(output)
        assert output.suffix == ".glb"
        output.write_bytes(b"glTF")
        return Asset(root=Node(id="root", name="Root"))

    monkeypatch.setattr(_io_helpers, "convert", fake_convert)
    monkeypatch.setattr(click, "get_binary_stream", lambda _name: stdout)

    _io_helpers._convert_output(
        input_path=Path("input.step"),
        output_path=Path("-"),
        profile="realtime-web",
        pipeline=None,
        tessellation=None,  # type: ignore[arg-type]
        stage=None,  # type: ignore[arg-type]
        import_options=None,  # type: ignore[arg-type]
        heal_brep=None,
        merge_vertices=None,
        delete_degenerate_polygons=None,
        merge=None,
        explode=None,
        replace=None,
        scene=None,
        bake_materials=None,
        remove_holes=None,
        remove_occluded=None,
        decimate=None,
        lod_generator=None,
        optimize=None,
        lods=None,
        where=None,
        progress=None,
        debug=False,
        gltf_options=None,
        usd_options=None,
        obj_options=None,
        stl_options=None,
        fbx_options=None,
        stdout_format=StdoutFormat.GLB,
    )

    assert stdout.getvalue() == b"glTF"
    assert temp_paths and not temp_paths[0].exists()


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_cli_stdio_paths_use_real_process_streams() -> None:
    step_data = Path("tests/fixtures/spool-clamp-lid.step").read_text(encoding="utf-8")

    inspect_result = subprocess.run(
        [sys.executable, "-m", "fascat", "--json", "inspect", "-"],
        input=step_data,
        capture_output=True,
        check=False,
        text=True,
    )
    assert inspect_result.returncode == 0
    inspect_payload = json.loads(inspect_result.stdout)
    assert inspect_payload["input"] == "-"
    assert inspect_payload["stats"]["parts"] == 1

    convert_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fascat",
            "convert",
            "-",
            "-",
            "--sag",
            "0.2",
            "--target-triangles",
            "80",
            "--lods",
            "0.5",
            "--debug",
        ],
        input=step_data,
        capture_output=True,
        check=False,
        text=True,
    )
    assert convert_result.returncode == 0
    assert "#usda" in convert_result.stdout
    assert all(line.startswith("warning:") for line in convert_result.stderr.splitlines())

    validate_result = subprocess.run(
        [sys.executable, "-m", "fascat", "validate", "-"],
        input=convert_result.stdout,
        capture_output=True,
        check=False,
        text=True,
    )
    assert validate_result.returncode == 0
    assert "valid USD" in compact(validate_result.stdout)


def test_convert_existing_output_requires_force(tmp_path: Path) -> None:
    step_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")
    output_file.write_text("#usdc", encoding="utf-8")

    result = runner.invoke(app, ["convert", str(step_file), str(output_file)])
    assert result.exit_code == 1
    assert "Pass --force" in compact(result.output)


def test_convert_writes_tessellation_quality_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat as fc
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    quality_file = tmp_path / "quality.json"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [3, 0, 0], [0, 0.01, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="Part", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )

    def fake_convert(*_args: object, **kwargs: object) -> fc.Asset:
        tessellation = kwargs["tessellation"]
        stage = kwargs["stage"]
        assert isinstance(tessellation, fc.TessellationOptions)
        assert isinstance(stage, fc.StageOptions)
        assert tessellation.sag_ratio == 0.02
        assert tessellation.min_edge_length == 0.01
        assert tessellation.max_polygon_length == 3.0
        assert tessellation.detail_adaptive is True
        assert tessellation.avoid_skinny_triangles is True
        assert tessellation.quality_report is True
        assert tessellation.free_edge_report is True
        assert tessellation.reuse_existing_meshes is False
        assert stage.unwrap.method == "isometric"
        assert stage.unwrap.iterations == 32
        assert stage.unwrap.tolerance == 0.001
        assert stage.unwrap.sharp_to_seam is True
        assert stage.unwrap.forbid_overlapping is True
        assert stage.aabb_projection.scope == "shared"
        assert stage.aabb_projection.uv3d_size == 2.5
        assert stage.aabb_projection.override_existing is False
        return asset

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", fake_convert)

    result = runner.invoke(
        app,
        [
            "convert",
            str(input_file),
            str(output_file),
            "--min-edge-length",
            "0.01",
            "--max-polygon-length",
            "3.0",
            "--sag-ratio",
            "0.02",
            "--retessellate-existing-meshes",
            "--free-edge-report",
            "--detail-adaptive",
            "--avoid-skinny-triangles",
            "--unwrap-method",
            "isometric",
            "--unwrap-iterations",
            "32",
            "--unwrap-tolerance",
            "0.001",
            "--uv-sharp-to-seam",
            "--uv-forbid-overlapping",
            "--uv-aabb-scope",
            "shared",
            "--uv3d-size",
            "2.5",
            "--uv-preserve-existing",
            "--quality-report",
            str(quality_file),
        ],
    )

    assert result.exit_code == 0
    quality = json.loads(quality_file.read_text(encoding="utf-8"))
    assert quality["summary"]["parts"] == 1
    assert quality["parts"][0]["part_id"] == "part"


def test_convert_writes_failure_report_sidecar(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    output_file = tmp_path / "output.usdc"
    report_file = tmp_path / "report.json"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    failure_report = Report(source_path=str(input_file))
    failure_report.add_error("invalid usd")
    failure_report.finish({"parts": 1, "triangles": 2})
    error = RuntimeError("invalid usd")
    error.report = failure_report

    def fail_convert(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", fail_convert)

    result = runner.invoke(app, ["convert", str(input_file), str(output_file), "--report", str(report_file)])

    assert result.exit_code == 1
    assert "invalid usd" in result.output
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    assert report["errors"] == ["invalid usd"]
    assert report["finished_at"] is not None


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_writes_failure_report_when_usd_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import fascat.io.usd as usd
    import fascat.pipeline as pipeline

    output_file = tmp_path / "output.usda"
    report_file = tmp_path / "report.json"

    def fail_validate(_path: str | Path, _output_format: object) -> dict[str, int]:
        raise RuntimeError("invalid generated USD")

    original_write_usd = usd.write_usd_with_validation_stats

    def write_usd_without_stats(
        asset: Asset,
        path: str | Path,
        *,
        debug: bool = False,
        options: object = None,
    ) -> None:
        original_write_usd(asset, path, debug=debug, options=options)

    monkeypatch.setattr(usd, "write_usd_with_validation_stats", write_usd_without_stats)
    monkeypatch.setattr(pipeline, "_validate_output", fail_validate)

    result = runner.invoke(
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
            "--report",
            str(report_file),
        ],
    )

    assert result.exit_code == 1
    assert "invalid generated USD" in result.output
    assert output_file.exists()
    assert report_file.exists()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    step_names = [step["name"] for step in report["steps"]]
    assert report["errors"] == ["invalid generated USD"]
    assert step_names[-2:] == ["write", "validate"]
    assert report["finished_at"] is not None
    assert report["output_stats"]["triangles"] > 0


def test_convert_rejects_invalid_lods() -> None:
    result = runner.invoke(app, ["--dry-run", "convert", "input.step", "output.usdc", "--lods", "1.5"])
    assert result.exit_code == 2
    assert "--lods ratios" in result.output


def test_convert_rejects_unsorted_lods_during_dry_run(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--lods", "0.25,0.5"], capsys)
    assert result.exit_code == 2
    assert "--lods ratios must be sorted from highest to lowest detail" in result.stderr


def test_convert_rejects_invalid_max_edge_length(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--max-edge-length", "0"], capsys)
    assert result.exit_code == 2
    assert "--max-edge-length must be greater than 0" in result.stderr


def test_convert_rejects_invalid_max_polygon_length(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--max-polygon-length", "0"], capsys)
    assert result.exit_code == 2
    assert "--max-polygon-length must be greater than 0" in result.stderr


def test_convert_rejects_invalid_min_edge_length(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--min-edge-length", "0"], capsys)
    assert result.exit_code == 2
    assert "--min-edge-length must be greater than 0" in result.stderr


def test_convert_rejects_invalid_sag_ratio(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--sag-ratio", "0"], capsys)
    assert result.exit_code == 2
    assert "--sag-ratio must be greater than 0" in result.stderr


def test_convert_rejects_min_edge_above_max_edge(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(
        [
            "--dry-run",
            "convert",
            "input.step",
            "output.usdc",
            "--min-edge-length",
            "2",
            "--max-edge-length",
            "1",
        ],
        capsys,
    )
    assert result.exit_code == 2
    assert "--min-edge-length must be less than or equal to --max-edge-length" in result.stderr


def test_convert_rejects_invalid_hard_edge_angle(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--hard-edge-angle", "0"], capsys)
    assert result.exit_code == 2
    assert "--hard-edge-angle must be greater than 0" in result.stderr


def test_convert_rejects_invalid_small_part_threshold(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(
        ["--dry-run", "convert", "input.step", "output.usdc", "--small-part-triangle-threshold", "-1"],
        capsys,
    )
    assert result.exit_code == 2
    assert "--small-part-triangle-threshold must be greater than or equal to 0" in result.stderr


def test_convert_rejects_invalid_uv_pipeline_values(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--texel-density", "0"], capsys)
    assert result.exit_code == 2
    assert "--texel-density must be greater than 0" in result.stderr

    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--uv-padding", "-1"], capsys)
    assert result.exit_code == 2
    assert "--uv-padding must be greater than or equal to 0" in result.stderr

    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--atlas-size", "0"], capsys)
    assert result.exit_code == 2
    assert "--atlas-size must be greater than 0" in result.stderr

    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--unwrap-iterations", "0"], capsys)
    assert result.exit_code == 2
    assert "--unwrap-iterations must be greater than 0" in result.stderr

    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--unwrap-tolerance", "-1"], capsys)
    assert result.exit_code == 2
    assert "--unwrap-tolerance must be greater than or equal to 0" in result.stderr

    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--uv3d-size", "0"], capsys)
    assert result.exit_code == 2
    assert "--uv3d-size must be greater than 0" in result.stderr


def test_convert_rejects_invalid_lods_as_json(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--json", "--dry-run", "convert", "input.step", "output.usdc", "--lods", "1.5"], capsys)
    assert result.exit_code == 2
    assert '"error": "--lods ratios must be greater than 0 and less than 1."' in result.stdout


def test_convert_rejects_bad_input_suffix(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.txt", "output.usdc"], capsys)
    assert result.exit_code == 2
    assert "Unsupported CAD extension" in result.stderr
    assert ".jt" in result.stderr


def test_convert_rejects_bad_output_suffix(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.txt"], capsys)
    assert result.exit_code == 2
    assert "Unsupported export extension" in result.stderr


def test_convert_rejects_zero_ratio(capsys) -> None:  # type: ignore[no-untyped-def]
    result = invoke_run(["--dry-run", "convert", "input.step", "output.usdc", "--ratio", "0"], capsys)
    assert result.exit_code == 2
    assert "--ratio must be greater than 0" in result.stderr


@pytest.mark.requires_ocp
@pytest.mark.requires_usd
def test_convert_reports_stage_progress_to_stderr(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    output_file = tmp_path / "output.usda"

    result = invoke_run(
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
        capsys,
    )

    assert result.exit_code == 0
    assert "Converted" in result.stdout
    progress_lines = [line for line in result.stderr.splitlines() if ":" in line and not line.startswith("warning:")]
    assert [line.split(":", 1)[0] for line in progress_lines] == [
        "source",
        "tessellate",
        "repair",
        "stage",
        "optimize",
        "lods",
        "write",
        "validate",
    ]
    assert all(re.search(r"\d+ parts", line) for line in progress_lines)
    assert all(re.search(r"\d+ triangles", line) for line in progress_lines)


def test_convert_interrupt_exits_130_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    output_file = tmp_path / "output.glb"

    def interrupt_convert(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", interrupt_convert)

    result = runner.invoke(app, ["convert", str(input_file), str(output_file)])

    assert result.exit_code == 130
    assert "Interrupted." in result.output
    assert "Traceback" not in result.output
    assert not output_file.exists()


def test_convert_interrupt_json_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")

    def interrupt_convert(*_args: object, **_kwargs: object) -> object:
        raise KeyboardInterrupt

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", interrupt_convert)

    result = runner.invoke(app, ["--json", "convert", str(input_file), str(tmp_path / "output.glb")])

    assert result.exit_code == 130
    payload = json.loads(result.stdout)
    assert payload["error"] == "interrupted"


def _convert_with_synthetic_warnings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    warning_count: int,
    arguments: list[str] | None = None,
) -> object:
    import fascat as fc
    from fascat.cli import _io_helpers

    input_file = tmp_path / "input.step"
    input_file.write_text("ISO-10303-21;", encoding="utf-8")
    output_file = tmp_path / "output.glb"

    def fake_convert(*_args: object, **_kwargs: object) -> fc.Asset:
        asset = fc.Asset(root=fc.Node(id="root", name="root"))
        for index in range(warning_count):
            asset.report.add_warning(f"profile budget exceeded for realtime-web: marker {index}")
        return asset

    monkeypatch.setattr(_io_helpers, "_convert_for_cli", fake_convert)
    return runner.invoke(app, [*(arguments or []), "convert", str(input_file), str(output_file)])


def test_convert_prints_report_warnings_on_stderr(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _convert_with_synthetic_warnings(monkeypatch, tmp_path, 2)

    assert result.exit_code == 0
    assert "warning: profile budget exceeded for realtime-web: marker 0" in result.stderr
    assert "warning:" not in result.stdout


def test_convert_warning_overflow_capped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _convert_with_synthetic_warnings(monkeypatch, tmp_path, 15)

    assert result.stderr.count("warning:") == 10
    assert "and 5 more warning(s)" in result.stderr


def test_convert_quiet_suppresses_warning_lines(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _convert_with_synthetic_warnings(monkeypatch, tmp_path, 2, ["--quiet"])

    assert "warning:" not in result.stderr


def test_convert_verbose_prints_all_report_warnings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _convert_with_synthetic_warnings(monkeypatch, tmp_path, 15, ["--verbose"])

    assert result.exit_code == 0
    assert result.stderr.count("warning:") == 15
    assert "more warning(s)" not in result.stderr


def test_convert_json_keeps_warnings_in_payload_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result = _convert_with_synthetic_warnings(monkeypatch, tmp_path, 2, ["--json"])

    assert result.exit_code == 0
    assert "warning:" not in result.stderr
    payload = json.loads(result.stdout)
    assert len(payload["report"]["warnings"]) == 2


def test_cli_convert_accepts_optimization_action_options_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--bake-materials",
            "--maps-resolution",
            "1024",
            "--force-uv-generation",
            "--bake",
            "base-color,opacity",
            "--ambient-occlusion-strategy",
            "advanced",
            "--decimate",
            "--decimate-criterion",
            "target",
            "--target-triangles",
            "1000",
            "--surface-tolerance",
            "0.1",
            "--line-tolerance",
            "0.02",
            "--normal-tolerance",
            "15",
            "--uv-tolerance",
            "0.01",
            "--decimate-iterative-threshold",
            "500",
            "--preserve-painted-areas",
            "--preserve-ambient-occlusion",
            "--budget-scope",
            "selection",
            "--uv-importance",
            "ignore",
            "--decimate-cleanup-attributes",
            "unused-uvs,tangents",
            "--remove-holes",
            "--hole-types",
            "through,blind,surface",
            "--max-hole-diameter",
            "3.0",
            "--remove-occluded",
            "--occlusion-strategy",
            "advanced",
            "--occlusion-level",
            "triangles",
            "--occlusion-precision",
            "2048",
            "--neighbors-preservation",
            "1",
            "--run-lod-generators",
            "--lod-mode",
            "variants",
            "--lod-per-part-budget",
            "--lod-drop-tiny-parts",
            "--lod-tiny-part-screen-size",
            "2",
            "--lod-preset",
            "vr",
            "--lod-screen-coverage",
            "0.5,0.2,0.05",
            "--lods",
            "0.5,0.25,0.1",
            "--validate-lods",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["bake_materials"] is True
    assert payload["bake"] == ["base_color", "opacity"]
    assert payload["ambient_occlusion_strategy"] == "advanced"
    assert payload["decimate"] is True
    assert payload["uv_importance"] == "ignore"
    assert payload["preserve_painted_areas"] is True
    assert payload["preserve_ambient_occlusion"] is True
    assert payload["decimate_cleanup_attributes"] == ["unused_uvs", "tangents"]
    assert payload["decimate_iterative_threshold"] == 500
    assert payload["remove_holes"] is True
    assert payload["remove_occluded"] is True
    assert payload["run_lod_generators"] is True
    assert payload["lod_per_part_budget"] is True
    assert payload["lod_drop_tiny_parts"] is True
    diagnostics = {item["operation"]: item for item in payload["operation_diagnostics"]}
    assert diagnostics["bake_materials"]["level"] == "exact"
    assert diagnostics["remove_holes"]["level"] == "approximate"
    assert diagnostics["remove_occluded"]["level"] == "approximate"
    assert diagnostics["decimate"]["level"] == "exact"
    assert diagnostics["run_lod_generators"]["level"] == "exact"
