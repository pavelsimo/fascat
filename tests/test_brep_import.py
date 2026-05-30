from __future__ import annotations

from pathlib import Path

import pytest

from fascat.io.brep import read_brep
from fascat.options import BrepReadOptions

pytestmark = pytest.mark.requires_ocp


def _write_box_brep(path: Path) -> None:
    pytest.importorskip("OCP.BRepPrimAPI")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.BRepTools import BRepTools

    assert BRepTools.Write_s(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(), str(path))


def test_read_brep_imports_single_source_shape(tmp_path: Path) -> None:
    source = tmp_path / "box.brep"
    _write_box_brep(source)

    asset = read_brep(source, options=BrepReadOptions(target_units="metre", target_up_axis="Y"))

    assert asset.part_count == 1
    assert asset.occurrence_count == 1
    assert asset.material_count == 1
    assert asset.units == "metre"
    assert asset.up_axis == "Y"
    assert asset.report.steps[-1].options["format"] == "BREP"
    part = next(iter(asset.parts.values()))
    assert part.source_shape is not None
    assert part.material_ids
    assert part.metadata["loaded_representation"] == "brep"
    assert part.metadata["source_faces"] == "6"


def test_read_brep_rejects_non_brep_extension(tmp_path: Path) -> None:
    source = tmp_path / "box.step"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported BREP extension"):
        read_brep(source)
