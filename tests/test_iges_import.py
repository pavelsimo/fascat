from __future__ import annotations

from pathlib import Path

import pytest

from fascat.io.iges import read_iges
from fascat.options import IgesReadOptions

pytestmark = pytest.mark.requires_ocp


def _write_box_iges(path: Path) -> None:
    pytest.importorskip("OCP.BRepPrimAPI")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.IGESControl import IGESControl_Writer

    writer = IGESControl_Writer()
    assert writer.AddShape(BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape())
    assert writer.Write(str(path))


def test_read_iges_imports_xde_shape_tree(tmp_path: Path) -> None:
    source = tmp_path / "box.igs"
    _write_box_iges(source)

    asset = read_iges(source, options=IgesReadOptions(target_units="metre", target_up_axis="Y"))

    assert asset.part_count == 1
    assert asset.occurrence_count == 1
    assert asset.units == "metre"
    assert asset.up_axis == "Y"
    assert asset.report.steps[-1].options["format"] == "IGES"
    part = next(iter(asset.parts.values()))
    assert part.source_shape is not None
    assert part.metadata["loaded_representation"] == "brep"
    assert part.metadata["source_faces"] == "6"


def test_read_iges_rejects_non_iges_extension(tmp_path: Path) -> None:
    source = tmp_path / "box.step"
    source.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported IGES extension"):
        read_iges(source)
