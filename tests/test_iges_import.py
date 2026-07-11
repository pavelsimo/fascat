from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc
import fascat.io.iges as iges_io
from fascat.errors import FascatIOError
from fascat.io.iges import read_iges, read_iges_bytes
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

    with pytest.raises(FascatIOError, match="unsupported IGES extension") as error:
        read_iges(source)

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.parametrize(("name", "suffix"), [("input.iges", ".iges"), ("input.step", ".igs")])
def test_read_iges_bytes_uses_format_suffix(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    suffix: str,
) -> None:
    seen_paths: list[Path] = []

    def fake_read_iges_path(path: Path, *, source_identity: str, options: IgesReadOptions) -> fc.Asset:
        assert source_identity == name
        assert path.suffix == suffix
        seen_paths.append(path)
        return fc.Asset(root=fc.Node(id="root", name="Root"))

    monkeypatch.setattr(iges_io, "_read_iges_path", fake_read_iges_path)

    read_iges_bytes(b"IGES", name=name)

    assert seen_paths and not seen_paths[0].exists()
