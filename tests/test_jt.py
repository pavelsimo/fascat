from __future__ import annotations

from pathlib import Path

import pytest

from fascat.asset import Asset, Node
from fascat.io import importer
from fascat.io.jt import _JTBackendUnavailable, read_jt, read_jt_bytes
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
