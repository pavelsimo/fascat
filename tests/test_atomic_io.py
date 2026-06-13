from __future__ import annotations

from pathlib import Path

import pytest

from fascat.io._atomic import atomic_output, atomic_outputs, publish_staged


def _hidden_temps(directory: Path) -> list[Path]:
    return [item for item in directory.iterdir() if item.name.startswith(".")]


def test_atomic_output_publishes_file_on_success(tmp_path: Path) -> None:
    target = tmp_path / "model.glb"

    with atomic_output(target) as temp:
        assert temp.parent == tmp_path
        temp.write_bytes(b"payload")
        assert not target.exists()

    assert target.read_bytes() == b"payload"
    assert _hidden_temps(tmp_path) == []


def test_atomic_output_keeps_suffix_for_temp_path(tmp_path: Path) -> None:
    with atomic_output(tmp_path / "scene.usda") as temp:
        assert temp.suffix == ".usda"
        temp.write_text("#usda 1.0\n")


def test_atomic_output_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "model.stl"
    target.write_bytes(b"old")

    with atomic_output(target) as temp:
        temp.write_bytes(b"new")

    assert target.read_bytes() == b"new"


def test_atomic_output_removes_temp_and_leaves_no_output_on_error(tmp_path: Path) -> None:
    target = tmp_path / "model.glb"

    with pytest.raises(RuntimeError, match="validation failed"), atomic_output(target) as temp:
        temp.write_bytes(b"partial")
        raise RuntimeError("validation failed")

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_output_cleans_up_on_keyboard_interrupt(tmp_path: Path) -> None:
    target = tmp_path / "model.glb"

    with pytest.raises(KeyboardInterrupt), atomic_output(target) as temp:
        temp.write_bytes(b"partial")
        raise KeyboardInterrupt

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_output_requires_the_block_to_write(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="atomic write produced no file"), atomic_output(tmp_path / "model.glb"):
        pass


def test_atomic_outputs_publishes_in_order_and_skips_unwritten(tmp_path: Path) -> None:
    mtl = tmp_path / "scene.mtl"
    obj = tmp_path / "scene.obj"

    with atomic_outputs((mtl, obj)) as (temp_mtl, temp_obj):
        temp_obj.write_text("o scene\n")

    assert obj.read_text() == "o scene\n"
    assert not mtl.exists()
    assert _hidden_temps(tmp_path) == []


def test_atomic_outputs_removes_all_temps_on_failure(tmp_path: Path) -> None:
    mtl = tmp_path / "scene.mtl"
    obj = tmp_path / "scene.obj"

    with pytest.raises(RuntimeError), atomic_outputs((mtl, obj)) as (temp_mtl, temp_obj):
        temp_mtl.write_text("newmtl m\n")
        temp_obj.write_text("o scene\n")
        raise RuntimeError("boom")

    assert list(tmp_path.iterdir()) == []


def test_publish_staged_replaces_targets(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "scene.bin").write_bytes(b"bin")
    (staging / "scene.gltf").write_text("{}")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "scene.gltf").write_text("old")

    publish_staged(
        [staging / "scene.bin", staging / "scene.gltf"],
        [out_dir / "scene.bin", out_dir / "scene.gltf"],
    )

    assert (out_dir / "scene.bin").read_bytes() == b"bin"
    assert (out_dir / "scene.gltf").read_text() == "{}"
    assert _hidden_temps(out_dir) == []
