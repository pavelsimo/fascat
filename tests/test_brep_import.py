from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc
import fascat.io.brep as brep_io
from fascat.errors import FascatIOError
from fascat.io.brep import read_brep, read_brep_bytes
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

    with pytest.raises(FascatIOError, match="unsupported BREP extension") as error:
        read_brep(source)

    assert isinstance(error.value.__cause__, ValueError)


def test_read_brep_bytes_always_uses_brep_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_paths: list[Path] = []

    def fake_read_brep_path(path: Path, *, source_identity: str, options: BrepReadOptions) -> fc.Asset:
        assert source_identity == "input.step"
        assert path.suffix == ".brep"
        seen_paths.append(path)
        return fc.Asset(root=fc.Node(id="root", name="Root"))

    monkeypatch.setattr(brep_io, "_read_brep_path", fake_read_brep_path)

    read_brep_bytes(b"BREP", name="input.step")

    assert seen_paths and not seen_paths[0].exists()
