from __future__ import annotations

from pathlib import Path

import pytest

from fascat.asset import Asset, Node
from fascat.io import importer
from fascat.io.jt import (
    _JTBackendUnavailable,
    _read_and_transfer_jt,
    has_native_jt_backend,
    read_jt,
    read_jt_bytes,
)
from fascat.report import Report


def _empty_imported_asset(source: Path, *, source_identity: str, display_name: str | None) -> Asset:
    stats = {"nodes": 1, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0}
    report = Report(source_path=str(source))
    report.input_stats = stats
    report.add_step("import", options={"format": "STEP", "backend": "OCP"}, before={}, after=stats)
    return Asset(
        root=Node(
            id="root",
            name=display_name or source.stem,
            metadata={"source": str(source), "source_identity": source_identity},
        ),
        source_path=source,
        report=report,
    )


def test_read_jt_requires_jt_suffix(tmp_path: Path) -> None:
    source = tmp_path / "input.step"
    source.write_text("ISO-10303-21;", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported JT extension"):
        read_jt(source)


def test_read_jt_reports_missing_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.jt as jt

    def missing_native_backend(*_args: object, **_kwargs: object) -> Asset:
        raise _JTBackendUnavailable("missing")

    monkeypatch.setattr(jt, "_read_jt_native_path", missing_native_backend)
    source = tmp_path / "input.jt"
    source.write_bytes(b"fake jt")

    with pytest.raises(RuntimeError, match="JT import requires"):
        read_jt(source)


def test_has_native_jt_backend_returns_boolean() -> None:
    assert isinstance(has_native_jt_backend(), bool)


def test_has_native_jt_backend_handles_missing_ocp_package(monkeypatch: pytest.MonkeyPatch) -> None:
    import fascat.io.jt as jt

    def missing_parent(_name: str) -> object:
        raise ModuleNotFoundError("OCP")

    monkeypatch.setattr(jt, "find_spec", missing_parent)

    assert has_native_jt_backend() is False


def test_read_jt_uses_native_reader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fascat.io.jt as jt

    source = tmp_path / "input.jt"
    source.write_bytes(b"fake jt")
    calls: dict[str, object] = {}

    def fake_read_jt_native_path(
        jt_path: Path,
        *,
        source_identity: str,
        display_name: str | None = None,
    ) -> Asset:
        calls["source"] = jt_path
        calls["source_identity"] = source_identity
        calls["display_name"] = display_name
        return _empty_imported_asset(jt_path, source_identity=source_identity, display_name=display_name)

    monkeypatch.setattr(jt, "_read_jt_native_path", fake_read_jt_native_path)

    asset = read_jt(source)

    assert calls == {
        "source": source,
        "source_identity": str(source.resolve()),
        "display_name": None,
    }
    assert asset.source_path == source
    assert asset.report.source_path == str(source)
    assert asset.root.name == "input"
    assert asset.root.metadata["source"] == str(source)


def test_read_jt_bytes_uses_name_as_source_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    import fascat.io.jt as jt

    def fake_read_jt_native_path(
        jt_path: Path,
        *,
        source_identity: str,
        display_name: str | None = None,
    ) -> Asset:
        assert jt_path.suffix == ".jt"
        return _empty_imported_asset(jt_path, source_identity=source_identity, display_name=display_name)

    monkeypatch.setattr(jt, "_read_jt_native_path", fake_read_jt_native_path)

    asset = read_jt_bytes(b"fake jt", name="stdin.jt")

    assert asset.source_path is None
    assert asset.report.source_path is None
    assert asset.root.metadata["source"] == "stdin.jt"
    assert asset.root.metadata["source_identity"] == "stdin.jt"


def test_jt_reader_transfers_read_file_result(tmp_path: Path) -> None:
    source = tmp_path / "input.jt"
    document = object()
    calls: list[tuple[str, object]] = []

    class Reader:
        def ReadFile(self, path: str) -> str:  # noqa: N802
            calls.append(("read", path))
            return "done"

        def Transfer(self, transferred_document: object) -> bool:  # noqa: N802
            calls.append(("transfer", transferred_document))
            return True

    _read_and_transfer_jt(Reader(), source, document, done_status="done")

    assert calls == [("read", str(source)), ("transfer", document)]


def test_jt_reader_rejects_failed_read_file(tmp_path: Path) -> None:
    class Reader:
        def ReadFile(self, _path: str) -> str:  # noqa: N802
            return "failed"

        def Transfer(self, _document: object) -> bool:  # noqa: N802
            return True

    with pytest.raises(RuntimeError, match="failed to read JT file"):
        _read_and_transfer_jt(Reader(), tmp_path / "input.jt", object(), done_status="done")


def test_jt_reader_rejects_failed_transfer(tmp_path: Path) -> None:
    class Reader:
        def ReadFile(self, _path: str) -> str:  # noqa: N802
            return "done"

        def Transfer(self, _document: object) -> bool:  # noqa: N802
            return False

    with pytest.raises(RuntimeError, match="failed to transfer JT data"):
        _read_and_transfer_jt(Reader(), tmp_path / "input.jt", object(), done_status="done")


def test_jt_reader_supports_perform_with_document(tmp_path: Path) -> None:
    source = tmp_path / "input.jt"
    document = object()
    calls: list[tuple[str, object]] = []

    class Reader:
        def Perform(self, path: str, transferred_document: object) -> bool:  # noqa: N802
            calls.append((path, transferred_document))
            return True

    _read_and_transfer_jt(Reader(), source, document, done_status="done")

    assert calls == [(str(source), document)]


def test_jt_reader_rejects_perform_without_document(tmp_path: Path) -> None:
    class Reader:
        def Perform(self, _path: str) -> bool:  # noqa: N802
            return True

    with pytest.raises(RuntimeError, match="Perform\\(file, document\\)"):
        _read_and_transfer_jt(Reader(), tmp_path / "input.jt", object(), done_status="done")


def test_read_cad_dispatches_step_and_jt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_read_step(_path: Path) -> Asset:
        calls.append("step")
        return _empty_imported_asset(tmp_path / "x.step", source_identity="step", display_name="step")

    def fake_read_jt(_path: Path) -> Asset:
        calls.append("jt")
        return _empty_imported_asset(tmp_path / "x.jt", source_identity="jt", display_name="jt")

    monkeypatch.setattr(importer, "read_step", fake_read_step)
    monkeypatch.setattr(importer, "read_jt", fake_read_jt)

    importer.read_cad(tmp_path / "input.step")
    importer.read_cad(tmp_path / "input.jt")

    assert calls == ["step", "jt"]
