from __future__ import annotations

import subprocess
import sys

import pytest

from fascat.io._registry import EXPORTERS, READERS, export_format_for_path, reader_for_suffix
from fascat.io._suffixes import CAD_SUFFIXES, EXPORT_SUFFIXES


def test_reader_registry_owns_every_cad_suffix_once() -> None:
    suffixes = [suffix for spec in READERS for suffix in spec.suffixes]

    assert set(suffixes) == CAD_SUFFIXES
    assert len(suffixes) == len(set(suffixes))


def test_exporter_registry_keys_and_suffixes_are_consistent() -> None:
    suffixes = [suffix for key, spec in EXPORTERS.items() for suffix in spec.suffixes if key == spec.format]

    assert set(EXPORTERS) == {spec.format for spec in EXPORTERS.values()}
    assert set(suffixes) == EXPORT_SUFFIXES
    assert len(suffixes) == len(set(suffixes))


@pytest.mark.parametrize(
    "path,expected",
    [
        ("-", "usd"),
        ("scene.USDC", "usd"),
        ("scene.glb", "gltf"),
        ("scene.obj", "obj"),
        ("scene.stl", "stl"),
        ("scene.fbx", "fbx"),
    ],
)
def test_export_format_for_path(path: str, expected: str) -> None:
    assert export_format_for_path(path) == expected


def test_registry_lookup_preserves_unsupported_extension_errors() -> None:
    with pytest.raises(ValueError, match=r"^unsupported CAD extension: \.cad$"):
        reader_for_suffix(".cad")
    with pytest.raises(ValueError, match=r"^unsupported export extension: \.zip$"):
        export_format_for_path("output.zip")


def test_registry_import_keeps_backends_lazy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, fascat.io._registry\n"
            "heavy = [name for name in ('fascat.io.step', 'fascat.io.usd', 'fascat.io.gltf') "
            "if name in sys.modules]\n"
            "assert not heavy, f'eagerly imported: {heavy}'\n",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
