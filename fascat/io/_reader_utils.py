from __future__ import annotations

import tempfile
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from fascat.asset import Asset


class _ReadOptions(Protocol):
    def to_dict(self) -> dict[str, object]: ...


_OptionsT = TypeVar("_OptionsT", bound=_ReadOptions)


def coerce_read_options(
    options: _ReadOptions | None,
    option_type: type[_OptionsT],
) -> _OptionsT:
    """Coerce compatible CAD read options without coupling readers together."""
    if options is None:
        return option_type()
    if isinstance(options, option_type):
        return options
    return option_type(**cast(Any, options.to_dict()))


def read_via_temporary_file(
    data: bytes,
    *,
    suffix: str,
    source_identity: str,
    options: _OptionsT,
    reader: Callable[..., Asset],
) -> Asset:
    """Run a path reader against bytes and always remove its temporary file."""
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(data)
            handle.flush()
        return reader(temp_path, source_identity=source_identity, options=options)
    finally:
        if temp_path is not None:
            with suppress(FileNotFoundError):
                temp_path.unlink()


def patch_bytes_source(asset: Asset, name: str) -> None:
    """Restore the caller-visible source identity after a temporary-file read."""
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    if asset.metadata:
        asset.metadata["source"] = name
        asset.metadata["source_identity"] = name
