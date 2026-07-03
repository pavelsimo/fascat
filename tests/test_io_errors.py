from __future__ import annotations

from pathlib import Path

import pytest

from fascat.asset import Asset, Node
from fascat.errors import FascatIOError
from fascat.io import gltf


def test_public_writer_wraps_stdlib_io_failure(tmp_path: Path) -> None:
    occupied_parent = tmp_path / "occupied"
    occupied_parent.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FascatIOError, match="write glTF failed") as error:
        gltf.write_gltf(Asset(root=Node(id="root", name="root")), occupied_parent / "model.glb")

    assert isinstance(error.value.__cause__, FileExistsError)


def test_public_writer_does_not_double_wrap_fascat_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sentinel = FascatIOError("already wrapped")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise sentinel

    monkeypatch.setattr(gltf, "_write_gltf", fail_write)

    with pytest.raises(FascatIOError) as error:
        gltf.write_gltf(Asset(root=Node(id="root", name="root")), tmp_path / "model.glb")

    assert error.value is sentinel
    assert error.value.__cause__ is None
