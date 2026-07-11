from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from fascat.asset import Asset
from fascat.io._errors import wrap_io_errors
from fascat.io._reader_utils import patch_bytes_source, read_via_temporary_file
from fascat.io._suffixes import STEP_SUFFIXES
from fascat.io.step import multifile, single
from fascat.options import StepReadOptions


@wrap_io_errors("read STEP")
def read_step(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    """Read a STEP file into an asset with hierarchy and metadata."""
    source = Path(path)
    opts = options or StepReadOptions()
    if opts.multi_file:
        return multifile._read_step_with_external_references(source, opts)
    return single._read_step_path(source, source_identity=str(source.resolve()), options=opts)


@wrap_io_errors("read STEP files")
def read_step_many(
    paths: Iterable[str | Path],
    *,
    options: StepReadOptions | None = None,
    continue_on_error: bool = False,
) -> Asset:
    """Read multiple STEP roots into one namespaced asset."""
    return multifile._read_step_many(paths, options=options, continue_on_error=continue_on_error, reference_graph=None)


@wrap_io_errors("read STEP bytes")
def read_step_bytes(data: bytes, *, name: str = "stdin.step", options: StepReadOptions | None = None) -> Asset:
    suffix = Path(name).suffix.lower()
    asset = read_via_temporary_file(
        data,
        suffix=suffix if suffix in STEP_SUFFIXES else ".step",
        source_identity=name,
        options=options or StepReadOptions(),
        reader=single._read_step_path,
    )
    patch_bytes_source(asset, name)
    return asset
