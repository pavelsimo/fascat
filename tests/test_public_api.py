from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import fascat as fc
from fascat.options import AnalyzeOptions, MetadataExportOptions, ReplaceOptions
from fascat.validation import (
    VisualDiffOptions,
    VisualDiffReport,
    VisualPreviewOptions,
    VisualPreviewReport,
    compare_images,
    write_preview,
)


def test_fluent_asset_operations_chain() -> None:
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )

    with_lods = (
        asset.repair()
        .merge_vertices()
        .delete_degenerate_polygons()
        .stage(fc.StageOptions(uv0="box", uv1="box"))
        .optimize(fc.OptimizeOptions(target_triangles=1, preserve_instances=True))
        .lods(fc.LODOptions(ratios=(0.5,)))
    )
    lod_part = with_lods.parts["part"]
    assert lod_part.mesh is not None
    assert sorted(lod_part.mesh.uvs) == [0, 1]
    assert len(lod_part.lod_meshes) == 1

    replaced = with_lods.replace(ReplaceOptions(mode="bounding_box"))

    part = next(iter(replaced.parts.values()))
    assert part.mesh is not None
    assert part.mesh.triangle_count == 12


@pytest.mark.parametrize(
    "removed",
    [
        "tessellate",
        "repair",
        "merge_vertices",
        "delete_degenerate_polygons",
        "heal_brep",
        "stage",
        "optimize",
        "lods",
        "merge",
        "explode",
        "replace",
        "optimize_scene",
        "bake_materials",
        "process_textures",
        "decimate",
        "remove_holes",
        "remove_occluded",
        "run_lod_generators",
    ],
)
def test_module_level_operation_wrappers_are_removed(removed: str) -> None:
    assert not hasattr(fc, removed)
    assert removed not in fc.__all__


def test_exported_public_api_has_docstrings() -> None:
    missing: list[str] = []

    for name in fc.__all__:
        if name == "__version__":
            continue
        obj = getattr(fc, name)
        if inspect.ismodule(obj):
            continue
        if not inspect.getdoc(obj):
            missing.append(name)
        if not inspect.isclass(obj) or not obj.__module__.startswith("fascat"):
            continue

        for member_name, member in inspect.getmembers(obj):
            if member_name.startswith("_"):
                continue
            if isinstance(member, property):
                target = member.fget
            elif inspect.ismethod(member):
                target = member.__func__
            elif inspect.isfunction(member) or inspect.ismethoddescriptor(member):
                target = member
            else:
                continue
            if target is not None and not inspect.getdoc(target):
                missing.append(f"{name}.{member_name}")

    assert missing == []


def test_public_api_exposes_quality_analysis() -> None:
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )

    report = fc.analyze(asset, options=AnalyzeOptions(open_boundaries=True))

    assert isinstance(report, fc.AnalysisReport)
    assert report.summary["open_boundaries"] == 1


def test_public_api_exposes_visual_previews(tmp_path: Path) -> None:
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )

    report = write_preview(asset, tmp_path / "preview.png", VisualPreviewOptions(width=64, height=64, padding=12))
    diff = compare_images(tmp_path / "preview.png", tmp_path / "preview.png", VisualDiffOptions())

    assert isinstance(report, VisualPreviewReport)
    assert report.triangles == 1
    assert isinstance(diff, VisualDiffReport)
    assert diff.passed is True


def test_public_api_exposes_pipeline_spec(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "pipeline.toml"
    pipeline_file.write_text('[[steps]]\nop = "repair"\n', encoding="utf-8")

    spec = fc.PipelineSpec.from_file(pipeline_file)

    assert spec.steps[0].op == "repair"


def test_public_api_exposes_metadata_export_options() -> None:
    options = MetadataExportOptions(mode="summary", pmi="metadata-and-visuals")

    assert options.to_dict() == {"mode": "summary", "pmi": "metadata_and_visuals"}


def test_tessellate_records_full_options_in_report() -> None:
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part")},
    )

    tessellated = asset.tessellate(
        fc.TessellationOptions(
            sag=0.2,
            sag_ratio=0.01,
            angle=20.0,
            relative=False,
            min_edge_length=0.25,
            max_edge_length=5.0,
            max_polygon_length=4.0,
            preserve_boundaries=False,
            curvature_adaptive=True,
            detail_adaptive=True,
            avoid_skinny_triangles=True,
            quality_report=True,
            free_edge_report=True,
            create_normals=False,
            keep_brep=True,
            reuse_existing_meshes=False,
            part_settings={"Part": {"sag": 0.3}},
        )
    )
    step = tessellated.report.steps[-1]

    assert step.name == "tessellate"
    options = dict(step.options)
    tolerance_policy = options.pop("tolerance_policy")
    assert options == {
        "sag": 0.2,
        "sag_ratio": 0.01,
        "angle": 20.0,
        "relative": False,
        "min_edge_length": 0.25,
        "max_edge_length": 5.0,
        "max_polygon_length": 4.0,
        "preserve_boundaries": False,
        "curvature_adaptive": True,
        "detail_adaptive": True,
        "avoid_skinny_triangles": True,
        "quality_report": True,
        "free_edge_report": True,
        "cad_uvs": False,
        "tessellate_tangents": False,
        "free_edge_geometry": False,
        "create_normals": False,
        "keep_brep": True,
        "reuse_existing_meshes": False,
        "part_settings": {"Part": {"sag": 0.3}},
    }
    assert tolerance_policy["coordinate_space"] == "asset"
    assert tolerance_policy["active_deflection"] == 0.01
    assert tolerance_policy["active_deflection_kind"] == "sag_ratio"
    assert tolerance_policy["active_deflection_relative"] is True
    assert tolerance_policy["min_edge_length_meters"] == 0.00025
    assert tolerance_policy["max_edge_length_meters"] == 0.005
    assert tolerance_policy["max_polygon_length_meters"] == 0.004
    assert step.warnings == ["part has no source shape and cannot be tessellated: Part"]


def test_functional_write_usd_records_report_step(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.io.usd as usd

    asset = fc.Asset(root=fc.Node(id="root", name="root"))
    output = tmp_path / "output.usda"
    calls: dict[str, object] = {}

    def fake_write_usd(
        written_asset: fc.Asset,
        path: str | Path,
        *,
        debug: bool = False,
        options: fc.UsdExportOptions | None = None,
    ) -> None:
        calls["asset"] = written_asset
        calls["path"] = path
        calls["debug"] = debug
        calls["options"] = options

    monkeypatch.setattr(usd, "write_usd", fake_write_usd)

    fc.write_usd(asset, output, debug=True)
    step = asset.report.steps[-1]

    assert calls == {"asset": asset, "path": output, "debug": True, "options": fc.UsdExportOptions()}
    assert step.name == "write"
    assert step.options == {
        "format": "OpenUSD",
        "debug": True,
        "package": "default",
        "file_size_budget_mb": None,
        "metadata": {"mode": "full", "pmi": "metadata"},
    }
    assert asset.report.finished_at is not None


def test_functional_write_usd_attaches_failure_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.io.usd as usd

    asset = fc.Asset(root=fc.Node(id="root", name="root"))

    def fail_write_usd(
        _asset: fc.Asset,
        _path: str | Path,
        *,
        debug: bool = False,
        options: fc.UsdExportOptions | None = None,
    ) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(usd, "write_usd", fail_write_usd)

    with pytest.raises(RuntimeError, match="disk full") as error:
        fc.write_usd(asset, tmp_path / "output.usda")

    step = asset.report.steps[-1]
    assert error.value.report is asset.report
    assert asset.report.errors == ["disk full"]
    assert step.name == "write"
    assert step.after == asset.stats()
    assert asset.report.finished_at is not None


def test_functional_write_gltf_records_report_step(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.io.gltf as gltf

    asset = fc.Asset(root=fc.Node(id="root", name="root"))
    output = tmp_path / "output.glb"
    calls: dict[str, object] = {}

    def fake_write_gltf(
        written_asset: fc.Asset,
        path: str | Path,
        *,
        options: fc.GltfExportOptions | None = None,
    ) -> None:
        calls["asset"] = written_asset
        calls["path"] = path
        calls["options"] = options

    monkeypatch.setattr(gltf, "write_gltf", fake_write_gltf)

    fc.write_gltf(asset, output)
    step = asset.report.steps[-1]

    assert calls == {"asset": asset, "path": output, "options": fc.GltfExportOptions()}
    assert step.name == "write"
    options = dict(step.options)
    runtime_dependencies = options.pop("runtime_dependencies")
    assert options == {
        "format": "glTF",
        "preset": None,
        "quantize": False,
        "meshopt": False,
        "draco": False,
        "texture_compression": None,
        "texture_fallback_format": "auto",
        "png_compression": 6,
        "jpeg_quality": 85,
        "file_size_budget_mb": None,
        "size_ladder": False,
        "draco_compression_level": 5,
        "draco_quantize_position": 14,
        "draco_quantize_normal": 10,
        "draco_quantize_texcoord": 12,
        "draco_quantize_color": 8,
        "ktx2_quality": 128,
        "ktx2_effort": 2,
        "ktx2_uastc": None,
        "metadata": {"mode": "full", "pmi": "metadata"},
    }
    assert runtime_dependencies["extensions_used"] == []
    assert runtime_dependencies["extras"] == {"fascat": True, "metadata": "full", "pmi": "metadata"}
    assert asset.report.finished_at is not None


def test_functional_write_gltf_attaches_failure_report(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.io.gltf as gltf

    asset = fc.Asset(root=fc.Node(id="root", name="root"))

    def fail_write_gltf(
        _asset: fc.Asset,
        _path: str | Path,
        *,
        options: fc.GltfExportOptions | None = None,
    ) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(gltf, "write_gltf", fail_write_gltf)

    with pytest.raises(RuntimeError, match="disk full") as error:
        fc.write_gltf(asset, tmp_path / "output.glb")

    step = asset.report.steps[-1]
    assert error.value.report is asset.report
    assert asset.report.errors == ["disk full"]
    assert step.name == "write"
    assert asset.report.finished_at is not None


def test_node_to_dict_includes_transform() -> None:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = 2.5
    node = fc.Node(id="node", name="node", transform=transform)

    assert node.to_dict()["transform"] == transform.tolist()


_TOP_LEVEL_API = [
    "AnalysisReport",
    "Asset",
    "DecimateOptions",
    "Filter",
    "FilterExpressionError",
    "GltfExportOptions",
    "ImageResource",
    "LODOptions",
    "Material",
    "MergeOptions",
    "Mesh",
    "MeshValidationError",
    "Metadata",
    "Node",
    "OptimizeOptions",
    "Part",
    "PipelineSpec",
    "PipelineStep",
    "PlatformBudget",
    "PmiAnnotation",
    "RepairOptions",
    "SelectionMatch",
    "SelectionResult",
    "StageOptions",
    "TessellationOptions",
    "Tolerance",
    "UVOverlapError",
    "UsdExportOptions",
    "__version__",
    "analyze",
    "convert",
    "profiles",
    "read_brep",
    "read_iges",
    "read_step",
    "read_step_many",
    "validate_output",
    "write_fbx",
    "write_gltf",
    "write_obj",
    "write_stl",
    "write_usd",
]


def test_top_level_namespace_is_locked() -> None:
    assert sorted(fc.__all__) == _TOP_LEVEL_API


def test_top_level_names_resolve() -> None:
    for name in fc.__all__:
        assert getattr(fc, name) is not None


def test_validation_module_exposes_harness_surface() -> None:
    import fascat.validation as validation

    for name in validation.__all__:
        assert getattr(validation, name) is not None
    assert "measure_browser_runtime" in validation.__all__
    assert "compare_images" in validation.__all__
    assert "measure_gltf_size_ladder" in validation.__all__


def test_package_ships_py_typed_marker() -> None:
    assert (Path(fc.__file__).parent / "py.typed").is_file()
