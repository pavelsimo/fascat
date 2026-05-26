from __future__ import annotations

import builtins
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.io.jt import read_jt
from fascat.io.step import read_step
from fascat.io.usd import validate_usd, write_usd
from fascat.mesh import Mesh
from fascat.options import StageOptions, Tessellation


def _block_imports(monkeypatch: pytest.MonkeyPatch, *prefixes: str) -> None:
    original_import: Callable[..., Any] = builtins.__import__

    def guarded_import(name: str, *args: object, **kwargs: object) -> Any:
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def _asset_with_triangle() -> Asset:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=mesh)},
    )


def test_step_import_reports_missing_ocp_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _block_imports(monkeypatch, "OCP")
    step_file = tmp_path / "input.step"
    step_file.write_text("ISO-10303-21;", encoding="utf-8")

    with pytest.raises(RuntimeError, match="STEP import requires cadquery-ocp"):
        read_step(step_file)


def test_jt_import_reports_missing_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _block_imports(monkeypatch, "OCP")
    jt_file = tmp_path / "input.jt"
    jt_file.write_bytes(b"fake jt")

    with pytest.raises(RuntimeError, match="JT import requires"):
        read_jt(jt_file)


def test_step_tessellation_reports_missing_ocp_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from fascat.ops.tessellate import tessellate_shape

    _block_imports(monkeypatch, "OCP")

    with pytest.raises(RuntimeError, match="STEP tessellation requires cadquery-ocp"):
        tessellate_shape(object(), Tessellation())


def test_usd_export_reports_missing_usd_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _block_imports(monkeypatch, "pxr")

    with pytest.raises(RuntimeError, match="USD export requires usd-core"):
        write_usd(_asset_with_triangle(), tmp_path / "output.usda")


def test_usd_validation_reports_missing_usd_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _block_imports(monkeypatch, "pxr")

    with pytest.raises(RuntimeError, match="USD validation requires usd-core"):
        validate_usd(tmp_path / "output.usda")


def test_stage_unwrap_reports_missing_xatlas_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    _block_imports(monkeypatch, "xatlas")

    with pytest.raises(RuntimeError, match="UV unwrap requires the optional xatlas dependency"):
        _asset_with_triangle().stage(StageOptions(uv0="unwrap", uv1=None))
