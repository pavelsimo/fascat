from __future__ import annotations

import pytest

from fascat.material import Material


def test_material_copies_metadata_and_reports_effective_opacity() -> None:
    metadata = {"source": "cad"}
    material = Material(
        id="paint",
        name="Paint",
        base_color=(0.1, 0.2, 0.3, 0.4),
        opacity=0.6,
        metadata=metadata,
    )
    metadata["source"] = "changed"

    copied = material.copy()

    assert material.metadata == {"source": "cad"}
    assert copied.metadata == material.metadata
    assert copied is not material
    assert material.effective_opacity == 0.4
    assert material.to_dict()["base_color"] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base_color": (1.0, 1.0, 1.0)}, "base_color"),
        ({"base_color": (1.0, 1.0, 1.0, 2.0)}, "between 0 and 1"),
        ({"metallic": -0.1}, "metallic"),
        ({"roughness": 1.1}, "roughness"),
        ({"opacity": -0.1}, "opacity"),
    ],
)
def test_material_validates_numeric_ranges(kwargs: dict[str, object], message: str) -> None:
    values = {
        "id": "paint",
        "name": "Paint",
        "base_color": (1.0, 1.0, 1.0, 1.0),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        Material(**values)  # type: ignore[arg-type]
