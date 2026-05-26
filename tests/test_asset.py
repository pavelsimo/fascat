from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh


def test_material_copies_input_metadata() -> None:
    metadata = {"source": "cad"}

    material = Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0), metadata=metadata)
    metadata["source"] = "changed"

    assert material.metadata == {"source": "cad"}


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
    transform[0, 3] = 5.0
    metadata["source"] = "changed"

    assert node.children == [child]
    assert node.transform[0, 3] == 0.0
    assert node.metadata == {"source": "cad"}


def test_node_rejects_invalid_transform_shape() -> None:
    with pytest.raises(ValueError, match="transform"):
        Node(id="node", name="node", transform=np.eye(3, dtype=float))


def test_part_and_asset_copy_mutable_containers_on_construction() -> None:
    material_ids = ["red"]
    metadata = {"source": "cad"}
    lod_mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    lod_meshes = [lod_mesh]
    part = Part(id="part", name="Part", material_ids=material_ids, metadata=metadata, lod_meshes=lod_meshes)
    parts = {"part": part}
    materials = {"red": Material(id="red", name="Red", base_color=(1.0, 0.0, 0.0, 1.0))}

    asset = Asset(root=Node(id="root", name="root"), parts=parts, materials=materials)
    material_ids.append("blue")
    metadata["source"] = "changed"
    lod_meshes.append(lod_mesh.copy())
    parts["other"] = Part(id="other", name="Other")
    materials["blue"] = Material(id="blue", name="Blue", base_color=(0.0, 0.0, 1.0, 1.0))

    assert part.material_ids == ["red"]
    assert part.metadata == {"source": "cad"}
    assert len(part.lod_meshes) == 1
    assert part.lod_meshes[0] is lod_mesh
    assert set(asset.parts) == {"part"}
    assert set(asset.materials) == {"red"}


def test_asset_write_usd_records_report_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.usd as usd

    asset = Asset(root=Node(id="root", name="root"))
    output = tmp_path / "output.usda"
    calls: dict[str, object] = {}

    def fake_write_usd(written_asset: Asset, path: str | Path, *, debug: bool = False) -> None:
        calls["asset"] = written_asset
        calls["path"] = path
        calls["debug"] = debug

    monkeypatch.setattr(usd, "write_usd", fake_write_usd)

    asset.write_usd(output, debug=True)
    step = asset.report.steps[-1]

    assert calls == {"asset": asset, "path": output, "debug": True}
    assert step.name == "write"
    assert step.options == {"format": "OpenUSD", "debug": True}
    assert step.before == asset.stats()
    assert step.after == asset.stats()
    assert step.duration >= 0.0
    assert asset.report.finished_at is not None
    assert asset.report.output_stats == asset.stats()


def test_asset_write_usd_records_failure_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.usd as usd

    asset = Asset(root=Node(id="root", name="root"))

    def fail_write_usd(_asset: Asset, _path: str | Path, *, debug: bool = False) -> None:
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
