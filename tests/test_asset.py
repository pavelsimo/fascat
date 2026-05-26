from __future__ import annotations

from fascat.asset import Asset, Node
from fascat.material import Material


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
