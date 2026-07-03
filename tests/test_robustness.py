"""Malformed-input behavior: broken CAD input must produce clean errors, never tracebacks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

import fascat as fc
from fascat.cli import app

pytestmark = pytest.mark.requires_ocp

runner = CliRunner()

_FIXTURE = Path("tests/fixtures/spool-clamp-body.step")


def _write_empty_compound_step(path: Path) -> None:
    from OCP.BRep import BRep_Builder
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.STEPControl import STEPControl_AsIs
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDocStd import TDocStd_Document
    from OCP.TopoDS import TopoDS_Compound
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    app_handle = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("fascat-empty"))
    app_handle.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    compound = TopoDS_Compound()
    BRep_Builder().MakeCompound(compound)
    shape_tool.AddShape(compound, True, True)

    writer = STEPCAFControl_Writer()
    assert writer.Transfer(document, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone


def test_read_step_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(fc.FascatIOError, match="missing STEP file") as error:
        fc.read_step(tmp_path / "absent.step")

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_read_step_unsupported_extension_raises_value_error(tmp_path: Path) -> None:
    bogus = tmp_path / "model.xyz"
    bogus.write_text("not cad", encoding="utf-8")

    with pytest.raises(fc.FascatIOError, match="unsupported STEP extension") as error:
        fc.read_step(bogus)

    assert isinstance(error.value.__cause__, ValueError)


def test_read_step_garbage_bytes_raises_clean_runtime_error(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.step"
    garbage.write_bytes(os.urandom(4096))

    with pytest.raises(fc.FascatIOError, match="failed to read STEP file") as error:
        fc.read_step(garbage)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_read_step_empty_file_raises_clean_runtime_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.step"
    empty.write_bytes(b"")

    with pytest.raises(fc.FascatIOError, match="failed to read STEP file") as error:
        fc.read_step(empty)

    assert isinstance(error.value.__cause__, RuntimeError)


def test_read_step_truncated_fixture_fails_cleanly(tmp_path: Path) -> None:
    payload = _FIXTURE.read_bytes()
    truncated = tmp_path / "truncated.step"
    truncated.write_bytes(payload[: len(payload) * 2 // 5])

    with pytest.raises(fc.FascatIOError):
        fc.read_step(truncated)


def test_read_step_header_only_data_section_fails_cleanly_or_yields_empty_asset(tmp_path: Path) -> None:
    source = tmp_path / "header-only.step"
    source.write_text(
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_DESCRIPTION((''),'2;1');\n"
        "FILE_NAME('','2026-01-01T00:00:00',(''),(''),'','','');\n"
        "FILE_SCHEMA(('AUTOMOTIVE_DESIGN'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    try:
        asset = fc.read_step(source)
    except fc.FascatIOError as exc:
        assert "STEP" in str(exc)
    else:
        assert asset.part_count == 0


def test_read_step_empty_compound_produces_zero_triangle_asset(tmp_path: Path) -> None:
    source = tmp_path / "empty-compound.step"
    _write_empty_compound_step(source)

    asset = fc.read_step(source)
    tessellated = asset.tessellate()

    assert tessellated.triangle_count == 0


def test_cli_convert_garbage_step_exits_one_without_traceback(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.step"
    garbage.write_bytes(os.urandom(4096))

    result = runner.invoke(app, ["convert", str(garbage), str(tmp_path / "out.glb")])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert result.output.strip()
    assert not (tmp_path / "out.glb").exists()


def test_cli_convert_empty_assembly_exits_one_without_traceback(tmp_path: Path) -> None:
    source = tmp_path / "empty-compound.step"
    _write_empty_compound_step(source)

    result = runner.invoke(app, ["convert", str(source), str(tmp_path / "out.glb")])

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert not (tmp_path / "out.glb").exists()


def test_million_vertex_mesh_operations_complete() -> None:
    import numpy as np

    side = 1024
    ys, xs = np.mgrid[0:side, 0:side]
    points = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(side * side)]).astype(float)
    index = np.arange(side * side).reshape(side, side)
    quads = np.column_stack(
        [
            index[:-1, :-1].ravel(),
            index[:-1, 1:].ravel(),
            index[1:, :-1].ravel(),
            index[1:, 1:].ravel(),
        ]
    )
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [2, 1, 3]]]).astype(int)
    mesh = fc.Mesh(points=points, faces=faces)

    assert mesh.vertex_count > 1_000_000
    cleaned = mesh.remove_degenerate_faces()

    assert cleaned.triangle_count == mesh.triangle_count
    cleaned.validate()
