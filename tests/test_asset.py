from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part, _hierarchy_report_stats
from fascat.filter import Filter
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.report import Report, ReportStep


def test_material_copies_input_metadata() -> None:
    metadata = {"source": "cad"}

    material = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0), metadata=metadata)
    metadata["source"] = "changed"

    assert material.metadata == {"source": "cad"}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"base_color": (1.0, 0.0, 0.0)}, "base_color must contain RGBA values"),
        ({"base_color": (1.0, 0.0, float("nan"), 1.0)}, "base_color values must be finite"),
        ({"base_color": (1.0, -0.1, 0.0, 1.0)}, "base_color values must be between 0 and 1"),
        ({"metallic": float("inf")}, "metallic must be finite"),
        ({"metallic": 1.1}, "metallic must be between 0 and 1"),
        ({"roughness": float("nan")}, "roughness must be finite"),
        ({"roughness": -0.1}, "roughness must be between 0 and 1"),
        ({"opacity": float("inf")}, "opacity must be finite"),
        ({"opacity": -0.1}, "opacity must be between 0 and 1"),
    ],
)
def test_material_rejects_invalid_unit_values(overrides: dict[str, Any], match: str) -> None:
    kwargs: dict[str, Any] = {
        "id": "mat",
        "name": "Material",
        "base_color": (1.0, 0.0, 0.0, 1.0),
    }
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=match):
        Material(**kwargs)


def test_asset_copy_isolates_material_metadata() -> None:
    asset = Asset(
        root=Node(id="root", name="root"),
        materials={"red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0), metadata={"source": "cad"})},
    )

    copied = asset.copy()
    copied.materials["red"].metadata["source"] = "copy"

    assert asset.materials["red"].metadata == {"source": "cad"}
    assert copied.materials["red"].metadata == {"source": "copy"}


def test_node_copies_mutable_inputs() -> None:
    child = Node(id="child", name="child")
    children = [child]
    transform = np.eye(4, dtype=float)
    metadata = {"source": "cad"}

    node = Node(id="node", name="node", children=children, transform=transform, metadata=metadata)
    children.append(Node(id="other", name="other"))
    child.name = "changed"
    transform[0, 3] = 5.0
    metadata["source"] = "changed"

    assert [node_child.name for node_child in node.children] == ["child"]
    assert node.transform[0, 3] == 0.0
    assert node.metadata == {"source": "cad"}


def test_node_rejects_invalid_transform_shape() -> None:
    with pytest.raises(ValueError, match="transform"):
        Node(id="node", name="node", transform=np.eye(3, dtype=float))


def test_part_copies_owned_meshes_on_construction() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    lod_mesh = mesh.copy()

    part = Part(id="part", name="Part", mesh=mesh, lod_meshes=[lod_mesh])
    mesh.points[0, 0] = 9.0
    lod_mesh.faces[0, 0] = 2

    assert part.mesh is not None
    assert part.mesh.points[0, 0] == 0.0
    assert part.lod_meshes[0].faces.tolist() == [[0, 1, 2]]


def test_part_and_asset_copy_mutable_containers_on_construction() -> None:
    material_ids = ["red"]
    metadata = {"source": "cad"}
    root_metadata = {"source": "root"}
    child = Node(id="node", name="Node", part_id="part")
    lod_mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    lod_meshes = [lod_mesh]
    mesh = lod_mesh.copy()
    part = Part(id="part", name="Part", material_ids=material_ids, metadata=metadata, lod_meshes=lod_meshes)
    parts = {"part": part}
    material = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0), metadata={"source": "cad"})
    materials = {"red": material}
    root = Node(id="root", name="root", children=[child], metadata=root_metadata)

    asset = Asset(root=root, parts=parts, materials=materials)
    material_ids.append("blue")
    metadata["source"] = "changed"
    lod_meshes.append(lod_mesh.copy())
    lod_mesh.points[0, 0] = 9.0
    parts["other"] = Part(id="other", name="Other")
    part.name = "Changed"
    part.mesh = mesh
    material.metadata["source"] = "changed"
    materials["blue"] = Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0))
    root.children.append(Node(id="other", name="Other"))
    root_metadata["source"] = "changed"
    child.name = "Changed"

    assert part.material_ids == ["red"]
    assert part.metadata == {"source": "cad"}
    assert len(part.lod_meshes) == 1
    assert part.lod_meshes[0] is not lod_mesh
    assert part.lod_meshes[0].points[0, 0] == 0.0
    assert set(asset.parts) == {"part"}
    assert set(asset.materials) == {"red"}
    assert asset.parts["part"].name == "Part"
    assert asset.parts["part"].mesh is None
    assert asset.materials["red"].metadata == {"source": "cad"}
    assert asset.root.metadata == {"source": "root"}
    assert [node.name for node in asset.root.children] == ["Node"]


def test_report_models_copy_mutable_inputs_on_construction() -> None:
    options: dict[str, object] = {"mode": "cad"}
    before = {"parts": 1}
    after = {"parts": 2}
    step_warnings = ["step warning"]
    step = ReportStep("stage", options=options, before=before, after=after, warnings=step_warnings)
    steps = [step]
    warnings = ["report warning"]
    errors = ["report error"]
    input_stats = {"triangles": 10}
    output_stats = {"triangles": 5}

    report = Report(
        steps=steps,
        warnings=warnings,
        errors=errors,
        input_stats=input_stats,
        output_stats=output_stats,
    )
    options["mode"] = "changed"
    before["parts"] = 3
    after["parts"] = 4
    step_warnings.append("changed")
    step.options["mode"] = "local change"
    steps.append(ReportStep("write"))
    warnings.append("changed")
    errors.append("changed")
    input_stats["triangles"] = 99
    output_stats["triangles"] = 100

    assert report.steps[0].options == {"mode": "cad"}
    assert report.steps[0].before == {"parts": 1}
    assert report.steps[0].after == {"parts": 2}
    assert report.steps[0].warnings == ["step warning"]
    assert [step.name for step in report.steps] == ["stage"]
    assert report.warnings == ["report warning"]
    assert report.errors == ["report error"]
    assert report.input_stats == {"triangles": 10}
    assert report.output_stats == {"triangles": 5}


def test_asset_copies_report_on_construction() -> None:
    report = Report(warnings=["outside"], input_stats={"parts": 1})

    asset = Asset(root=Node(id="root", name="root"), report=report)
    report.add_warning("changed")
    report.input_stats["parts"] = 2

    assert asset.report.warnings == ["outside"]
    assert asset.report.input_stats == {"parts": 1}


def test_asset_copy_does_not_reenter_defensive_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh, lod_meshes=[mesh.copy()])},
        report=Report(steps=[ReportStep("stage", options={"mode": "cad"})]),
    )
    calls = {"asset": 0, "part": 0, "report": 0}

    def asset_post_init(self: Asset) -> None:
        calls["asset"] += 1

    def part_post_init(self: Part) -> None:
        calls["part"] += 1

    def report_post_init(self: Report) -> None:
        calls["report"] += 1

    monkeypatch.setattr(Asset, "__post_init__", asset_post_init)
    monkeypatch.setattr(Part, "__post_init__", part_post_init)
    monkeypatch.setattr(Report, "__post_init__", report_post_init)

    copied = asset.copy()
    copied.parts["part"].mesh.points[0, 0] = 9.0
    copied.parts["part"].lod_meshes[0].faces[0, 0] = 2
    copied.report.steps[0].options["mode"] = "changed"

    assert calls == {"asset": 0, "part": 0, "report": 0}
    assert asset.parts["part"].mesh is not None
    assert asset.parts["part"].mesh.points[0, 0] == 0.0
    assert asset.parts["part"].lod_meshes[0].faces.tolist() == [[0, 1, 2]]
    assert asset.report.steps[0].options == {"mode": "cad"}


def _scoped_asset() -> Asset:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="unique_node", name="Unique", part_id="unique"),
                Node(id="shared_a", name="Shared A", part_id="shared"),
                Node(id="shared_b", name="Shared B", part_id="shared"),
            ],
        ),
        parts={
            "unique": Part(id="unique", name="Unique", mesh=mesh),
            "shared": Part(id="shared", name="Shared", mesh=mesh),
        },
    )


def test_asset_stats_reuses_single_hierarchy_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = _scoped_asset()
    original_walk = asset.root.walk
    calls = 0

    def counting_walk() -> list[Node]:
        nonlocal calls
        calls += 1
        return original_walk()

    monkeypatch.setattr(asset.root, "walk", counting_walk)

    stats = asset.stats()

    assert calls == 1
    assert stats["nodes"] == 4
    assert stats["occurrences"] == 3


def test_hierarchy_report_stats_reuses_single_hierarchy_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    asset = _scoped_asset()
    original_walk = asset.root.walk
    calls = 0

    def counting_walk() -> list[Node]:
        nonlocal calls
        calls += 1
        return original_walk()

    monkeypatch.setattr(asset.root, "walk", counting_walk)

    stats = _hierarchy_report_stats(asset)

    assert calls == 1
    assert stats["nodes"] == 4
    assert stats["occurrences"] == 3
    assert stats["draw_calls"] == 3


def test_operation_scope_skips_asset_copy_when_selection_does_not_split_occurrences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = _scoped_asset()
    calls = 0
    original_copy = Asset.copy

    def spy_copy(target: Asset, *, keep_source: bool = True) -> Asset:
        nonlocal calls
        calls += 1
        return original_copy(target, keep_source=keep_source)

    monkeypatch.setattr(Asset, "copy", spy_copy)

    scope = asset._operation_scope(Filter.part("unique"))

    assert scope.asset is asset
    assert scope.selected_part_ids == {"unique"}
    assert calls == 0


def test_operation_scope_copies_only_when_selection_splits_shared_occurrences() -> None:
    asset = _scoped_asset()

    scope = asset._operation_scope(Filter.name("Shared A"))

    selected_node = next(node for node in scope.asset.root.walk() if node.id == "shared_a")
    unselected_node = next(node for node in scope.asset.root.walk() if node.id == "shared_b")

    assert scope.asset is not asset
    assert selected_node.part_id == "shared_selected"
    assert unselected_node.part_id == "shared"
    assert scope.selected_part_ids == {"shared_selected"}
    assert scope.asset.parts["shared_selected"].metadata["source_part_id"] == "shared"
    assert next(node for node in asset.root.walk() if node.id == "shared_a").part_id == "shared"


def test_report_summary_and_json_output(tmp_path: Path) -> None:
    report = Report(source_path="input.step", warnings=["fixed winding"], input_stats={"parts": 1, "triangles": 12})
    report.add_step(
        "repair",
        before={"parts": 2, "triangles": 12},
        after={"parts": 2, "triangles": 8},
        duration=0.5,
        warnings=["removed degenerate faces"],
    )
    report.finish({"parts": 2, "triangles": 8, "materials": 3})
    output = tmp_path / "report.json"

    report.write_json(output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert report.summary() == "2 parts, 8 triangles, 3 materials, 1 warnings"
    assert payload["source_path"] == "input.step"
    assert payload["output_stats"] == {"parts": 2, "triangles": 8, "materials": 3}
    assert payload["steps"][0]["name"] == "repair"
    assert payload["steps"][0]["warnings"] == ["removed degenerate faces"]


def test_asset_write_usd_records_report_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.usd as usd

    asset = Asset(root=Node(id="root", name="root"))
    output = tmp_path / "output.usda"
    calls: dict[str, object] = {}

    def fake_write_usd(
        written_asset: Asset,
        path: str | Path,
        *,
        debug: bool = False,
        options: object = None,
    ) -> None:
        calls["asset"] = written_asset
        calls["path"] = path
        calls["debug"] = debug
        calls["options"] = options

    monkeypatch.setattr(usd, "write_usd", fake_write_usd)

    asset.write_usd(output, debug=True)
    step = asset.report.steps[-1]

    assert calls["asset"] is asset
    assert calls["path"] == output
    assert calls["debug"] is True
    assert step.name == "write"
    assert step.options == {
        "format": "OpenUSD",
        "debug": True,
        "package": "default",
        "file_size_budget_mb": None,
        "metadata": {"mode": "full", "pmi": "metadata"},
    }
    assert step.before == asset.stats()
    assert step.after == asset.stats()
    assert step.duration >= 0.0
    assert asset.report.finished_at is not None
    assert asset.report.output_stats == asset.stats()


def test_asset_write_usd_records_failure_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.usd as usd

    asset = Asset(root=Node(id="root", name="root"))

    def fail_write_usd(_asset: Asset, _path: str | Path, *, debug: bool = False, options: object = None) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(usd, "write_usd", fail_write_usd)

    with pytest.raises(RuntimeError, match="disk full") as error:
        asset.write_usd(tmp_path / "output.usda")

    step = asset.report.steps[-1]
    assert error.value.report is asset.report
    assert asset.report.errors == ["disk full"]
    assert step.name == "write"
    assert step.after == asset.stats()
    assert asset.report.finished_at is not None
    assert asset.report.output_stats == asset.stats()


def test_asset_write_gltf_records_report_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.gltf as gltf

    asset = Asset(root=Node(id="root", name="root"))
    output = tmp_path / "output.glb"
    calls: dict[str, object] = {}

    def fake_write_gltf(written_asset: Asset, path: str | Path, *, options: object = None) -> None:
        calls["asset"] = written_asset
        calls["path"] = path
        calls["options"] = options

    monkeypatch.setattr(gltf, "write_gltf", fake_write_gltf)

    asset.write_gltf(output)
    step = asset.report.steps[-1]

    assert calls["asset"] is asset
    assert calls["path"] == output
    assert step.name == "write"
    options = dict(step.options)
    runtime_dependencies = options.pop("runtime_dependencies")
    assert options == {
        "format": "glTF",
        "quantize": False,
        "meshopt": False,
        "draco": False,
        "texture_compression": None,
        "texture_fallback_format": "auto",
        "png_compression": 6,
        "jpeg_quality": 85,
        "file_size_budget_mb": None,
        "metadata": {"mode": "full", "pmi": "metadata"},
    }
    assert runtime_dependencies["extensions_used"] == []
    assert runtime_dependencies["extras"] == {"fascat": True, "metadata": "full", "pmi": "metadata"}
    assert step.before == asset.stats()
    assert step.after == asset.stats()
    assert step.duration >= 0.0
    assert asset.report.finished_at is not None
    assert asset.report.output_stats == asset.stats()


def test_asset_write_gltf_records_failure_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.gltf as gltf

    asset = Asset(root=Node(id="root", name="root"))

    def fail_write_gltf(_asset: Asset, _path: str | Path, *, options: object = None) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(gltf, "write_gltf", fail_write_gltf)

    with pytest.raises(RuntimeError, match="disk full") as error:
        asset.write_gltf(tmp_path / "output.glb")

    step = asset.report.steps[-1]
    assert error.value.report is asset.report
    assert asset.report.errors == ["disk full"]
    assert step.name == "write"
    assert step.after == asset.stats()
    assert asset.report.finished_at is not None
    assert asset.report.output_stats == asset.stats()
