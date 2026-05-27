from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fascat as fc


def test_functional_api_wraps_asset_operations() -> None:
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2], [2, 1, 3]], dtype=int),
    )
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )

    repaired = fc.repair(asset)
    staged = fc.stage(repaired, uv0="box", uv1="box")
    optimized = fc.optimize(staged, target_triangles=1, preserve_instances=True)
    with_lods = fc.lods(optimized, ratios=(0.5,))
    lod_part = with_lods.parts["part"]
    assert lod_part.mesh is not None
    assert sorted(lod_part.mesh.uvs) == [0, 1]
    assert len(lod_part.lod_meshes) == 1

    replaced = fc.replace(with_lods, options=fc.ReplaceOptions(mode="bounding_box"))

    part = next(iter(replaced.parts.values()))
    assert part.mesh is not None
    assert part.mesh.triangle_count == 12


def test_public_api_exposes_quality_analysis() -> None:
    mesh = fc.Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part", mesh=mesh)},
    )

    report = fc.analyze(asset, options=fc.AnalyzeOptions(open_boundaries=True))

    assert isinstance(report, fc.AnalysisReport)
    assert report.summary["open_boundaries"] == 1


def test_functional_api_wraps_tessellation_options() -> None:
    asset = fc.Asset(
        root=fc.Node(id="root", name="root", children=[fc.Node(id="node", name="node", part_id="part")]),
        parts={"part": fc.Part(id="part", name="Part")},
    )

    tessellated = fc.tessellate(
        asset,
        sag=0.2,
        angle=20.0,
        relative=False,
        min_edge_length=0.25,
        max_edge_length=5.0,
        preserve_boundaries=False,
        curvature_adaptive=True,
        avoid_skinny_triangles=True,
        quality_report=True,
        create_normals=False,
        keep_brep=True,
        part_settings={"Part": {"sag": 0.3}},
    )
    step = tessellated.report.steps[-1]

    assert step.name == "tessellate"
    assert step.options == {
        "sag": 0.2,
        "angle": 20.0,
        "relative": False,
        "min_edge_length": 0.25,
        "max_edge_length": 5.0,
        "preserve_boundaries": False,
        "curvature_adaptive": True,
        "avoid_skinny_triangles": True,
        "quality_report": True,
        "create_normals": False,
        "keep_brep": True,
        "part_settings": {"Part": {"sag": 0.3}},
    }
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
    assert step.options == {"format": "OpenUSD", "debug": True, "package": "default", "file_size_budget_mb": None}
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
    assert step.options == {
        "format": "glTF",
        "quantize": False,
        "meshopt": False,
        "draco": False,
        "texture_compression": None,
        "file_size_budget_mb": None,
    }
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
