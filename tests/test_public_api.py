from __future__ import annotations

from pathlib import Path

import numpy as np

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

    part = with_lods.parts["part"]
    assert part.mesh is not None
    assert sorted(part.mesh.uvs) == [0, 1]
    assert len(part.lod_meshes) == 1


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
        max_edge_length=5.0,
        create_normals=False,
        keep_brep=True,
    )
    step = tessellated.report.steps[-1]

    assert step.name == "tessellate"
    assert step.options == {
        "sag": 0.2,
        "angle": 20.0,
        "relative": False,
        "max_edge_length": 5.0,
        "create_normals": False,
        "keep_brep": True,
    }
    assert step.warnings == ["part has no source shape and cannot be tessellated: Part"]


def test_functional_write_usd_records_report_step(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.io.usd as usd

    asset = fc.Asset(root=fc.Node(id="root", name="root"))
    output = tmp_path / "output.usda"
    calls: dict[str, object] = {}

    def fake_write_usd(written_asset: fc.Asset, path: str | Path, *, debug: bool = False) -> None:
        calls["asset"] = written_asset
        calls["path"] = path
        calls["debug"] = debug

    monkeypatch.setattr(usd, "write_usd", fake_write_usd)

    fc.write_usd(asset, output, debug=True)
    step = asset.report.steps[-1]

    assert calls == {"asset": asset, "path": output, "debug": True}
    assert step.name == "write"
    assert step.options == {"format": "OpenUSD", "debug": True}
    assert asset.report.finished_at is not None


def test_node_to_dict_includes_transform() -> None:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = 2.5
    node = fc.Node(id="node", name="node", transform=transform)

    assert node.to_dict()["transform"] == transform.tolist()
