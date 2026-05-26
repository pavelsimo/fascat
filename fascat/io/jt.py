from __future__ import annotations

import tempfile
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from fascat.asset import Asset
from fascat.io.step import _read_xde_path, _reader_units

JT_SUFFIXES = {".jt"}


class _JTBackendUnavailable(RuntimeError):
    pass


def has_native_jt_backend() -> bool:
    try:
        return find_spec("OCP.JTCAFControl") is not None
    except ModuleNotFoundError:
        return False


def read_jt(path: str | Path) -> Asset:
    source = Path(path)
    return _read_jt_path(source, source_identity=str(source.resolve()))


def read_jt_bytes(data: bytes, *, name: str = "stdin.jt") -> Asset:
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".jt") as handle:
        handle.write(data)
        handle.flush()
        asset = _read_jt_path(Path(handle.name), source_identity=name, display_name=Path(name).stem)
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    return asset


def _read_jt_path(source: Path, *, source_identity: str, display_name: str | None = None) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing JT file: {source}")
    if source.suffix.lower() not in JT_SUFFIXES:
        raise ValueError(f"unsupported JT extension: {source.suffix or '<none>'}")

    try:
        return _read_jt_native_path(source, source_identity=source_identity, display_name=display_name)
    except _JTBackendUnavailable as exc:
        raise RuntimeError(_missing_backend_message()) from exc


def _read_jt_native_path(source: Path, *, source_identity: str, display_name: str | None = None) -> Asset:
    return _read_xde_path(
        source,
        source_identity=source_identity,
        display_name=display_name,
        import_format="JT",
        backend="OCP.JTCAFControl",
        reader=_read_jt_xde_document,
    )


def _read_jt_xde_document(path: Path) -> tuple[Any, Any, Any, str, float]:
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.JTCAFControl import JTCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except ImportError as exc:
        raise _JTBackendUnavailable("OCP.JTCAFControl is unavailable") from exc

    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("fascat"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)

    reader = JTCAFControl_Reader()
    _enable_reader_modes(reader)
    _read_and_transfer_jt(reader, path, document, done_status=IFSelect_RetDone)
    unit_name, meters_per_unit = _jt_reader_units(reader)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    return document, shape_tool, color_tool, unit_name, meters_per_unit


def _read_and_transfer_jt(reader: Any, path: Path, document: Any, *, done_status: Any) -> None:
    read_file = getattr(reader, "ReadFile", None)
    if callable(read_file):
        status = read_file(str(path))
        if status not in (done_status, True):
            raise RuntimeError(f"failed to read JT file: {path}")
        transfer = getattr(reader, "Transfer", None)
        if not callable(transfer):
            raise RuntimeError("installed JT backend does not expose Transfer()")
        if transfer(document) is False:
            raise RuntimeError(f"failed to transfer JT data into XDE document: {path}")
        return

    perform = getattr(reader, "Perform", None)
    if callable(perform):
        result = _perform_jt_import(perform, path, document)
        if result not in (done_status, True, None):
            raise RuntimeError(f"failed to read JT file: {path}")
        return

    raise RuntimeError("installed JT backend does not expose ReadFile() or Perform(file, document)")


def _perform_jt_import(perform: Any, path: Path, document: Any) -> Any:
    try:
        return perform(str(path), document)
    except TypeError as exc:
        raise RuntimeError("installed JT backend does not expose Perform(file, document)") from exc


def _enable_reader_modes(reader: Any) -> None:
    for name in ("SetNameMode", "SetColorMode", "SetMatMode", "SetLayerMode", "SetMetaMode", "SetProductMetaMode"):
        method = getattr(reader, name, None)
        if callable(method):
            try:
                method(True)
            except TypeError:
                continue


def _jt_reader_units(reader: Any) -> tuple[str, float]:
    try:
        return _reader_units(reader)
    except Exception:
        return "millimetre", 0.001


def _missing_backend_message() -> str:
    return "JT import requires Open Cascade JT Import-Export bindings exposed as OCP.JTCAFControl."
