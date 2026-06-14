"""Transactional file publication for exporters.

Writers produce content at a hidden same-directory temp path and publish it
with an atomic ``os.replace`` only on success, so a failed export (validation
error, backend crash, Ctrl-C) never leaves a partial or corrupt file at the
destination. No fsync is performed — the guarantee is interrupt/exception
atomicity, not crash durability.
"""

from __future__ import annotations

import os
import secrets
import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path


def _temp_path_for(path: Path) -> Path:
    # The real suffix stays last: exporters and pxr dispatch file formats on it.
    return path.parent / f".{path.stem}-tmp-{secrets.token_hex(6)}{path.suffix}"


def _unlink_quietly(path: Path) -> None:
    with suppress(OSError):
        path.unlink()


def preflight_output_path(path: str | Path) -> None:
    """Raise early when an output path cannot be published."""
    if str(path) == "-":
        return
    target = Path(path)
    parent = target.parent if str(target.parent) else Path(".")
    if not parent.exists():
        raise FileNotFoundError(f"output directory does not exist: {parent}")
    if not parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {parent}")
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"output path is a directory: {target}")
    if not os.access(parent, os.W_OK):
        raise PermissionError(f"output directory is not writable: {parent}")
    if target.exists() and not os.access(target, os.W_OK):
        raise PermissionError(f"output file is not writable: {target}")


@contextmanager
def atomic_output(path: str | Path) -> Iterator[Path]:
    """Yield a temp path that replaces ``path`` atomically when the block succeeds."""
    target = Path(path)
    preflight_output_path(target)
    temp = _temp_path_for(target)
    try:
        yield temp
    except BaseException:
        _unlink_quietly(temp)
        raise
    if not temp.exists():
        raise FileNotFoundError(f"atomic write produced no file for: {target}")
    os.replace(temp, target)


@contextmanager
def atomic_outputs(paths: Sequence[str | Path]) -> Iterator[tuple[Path, ...]]:
    """Multi-file ``atomic_output``.

    Temps publish in the given order, so list the entry file last — a reader
    then never sees an entry referencing a missing sidecar. Temps the block
    chose not to write are skipped.
    """
    targets = [Path(path) for path in paths]
    for target in targets:
        preflight_output_path(target)
    temps = tuple(_temp_path_for(target) for target in targets)
    try:
        yield temps
    except BaseException:
        for temp in temps:
            _unlink_quietly(temp)
        raise
    for temp, target in zip(temps, targets, strict=True):
        if temp.exists():
            os.replace(temp, target)


def publish_staged(staged: Sequence[Path], targets: Sequence[Path]) -> None:
    """Publish files produced in a staging directory to their final targets.

    Each staged file is first copied next to its target under a temp name
    (``shutil.copyfile`` — safe across filesystems), then the temps are
    replaced into place in order; list the entry file last.
    """
    temps: list[Path] = []
    try:
        for source, target in zip(staged, targets, strict=True):
            preflight_output_path(target)
            temp = _temp_path_for(target)
            shutil.copyfile(source, temp)
            temps.append(temp)
    except BaseException:
        for temp in temps:
            _unlink_quietly(temp)
        raise
    for temp, target in zip(temps, targets, strict=True):
        os.replace(temp, target)
