from __future__ import annotations

import subprocess
import sys

import pytest

import fascat


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)


def test_import_fascat_is_lazy() -> None:
    completed = _run_python(
        "import sys, fascat\n"
        "heavy = [name for name in ('fascat.io.step', 'fascat.runtime', 'fascat.visual', 'PIL', 'numpy')"
        " if name in sys.modules]\n"
        "assert not heavy, f'eagerly imported: {heavy}'\n"
    )

    assert completed.returncode == 0, completed.stderr


def test_cli_import_does_not_load_harnesses() -> None:
    completed = _run_python(
        "import sys, fascat.cli\n"
        "heavy = [name for name in ('fascat.runtime', 'fascat.visual', 'PIL') if name in sys.modules]\n"
        "assert not heavy, f'eagerly imported: {heavy}'\n"
    )

    assert completed.returncode == 0, completed.stderr


def test_all_exports_resolve() -> None:
    for name in fascat.__all__:
        assert getattr(fascat, name) is not None


def test_submodules_resolve_lazily() -> None:
    assert fascat.options.UnwrapOptions is not None
    assert fascat.validation.compare_images is not None
    assert fascat.profiles.by_name is not None


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="no attribute 'tessellate'"):
        _ = fascat.tessellate  # type: ignore[attr-defined]


def test_dir_lists_public_surface() -> None:
    listing = dir(fascat)

    assert "Asset" in listing
    assert "convert" in listing
    assert "validation" in listing
    assert "options" in listing
