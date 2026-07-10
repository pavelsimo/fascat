from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias

from fascat.io._suffixes import (
    BREP_SUFFIXES,
    FBX_SUFFIXES,
    GLTF_SUFFIXES,
    IGES_SUFFIXES,
    JT_SUFFIXES,
    OBJ_SUFFIXES,
    STEP_SUFFIXES,
    STL_SUFFIXES,
    USD_SUFFIXES,
)
from fascat.options import (
    FbxExportOptions,
    GltfExportOptions,
    ObjExportOptions,
    StepReadOptions,
    StlExportOptions,
    UsdExportOptions,
)

if TYPE_CHECKING:
    from fascat.asset import Asset

ExportFormat = Literal["usd", "gltf", "obj", "stl", "fbx"]
ExportOptions: TypeAlias = UsdExportOptions | GltfExportOptions | ObjExportOptions | StlExportOptions | FbxExportOptions
ReadAdapter: TypeAlias = Callable[..., "Asset"]


class ExportWriter(Protocol):
    def __call__(
        self,
        asset: Asset,
        path: str | Path,
        options: ExportOptions | None,
        *,
        debug: bool,
    ) -> dict[str, int] | None: ...


class ExportValidator(Protocol):
    def __call__(self, path: str | Path) -> dict[str, int]: ...


@dataclass(frozen=True)
class ReaderSpec:
    format: str
    suffixes: frozenset[str]
    read: ReadAdapter


@dataclass(frozen=True)
class ExporterSpec:
    format: ExportFormat
    label: str
    suffixes: frozenset[str]
    default_options: Callable[[], ExportOptions]
    validation_backend: str
    write_with_stats: ExportWriter
    validate: ExportValidator


def _read_step(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    from fascat.io.step import read_step

    return read_step(path, options=options) if options is not None else read_step(path)


def _read_iges(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    from fascat.io.iges import read_iges

    return read_iges(path, options=options)


def _read_brep(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    from fascat.io.brep import read_brep

    return read_brep(path, options=options)


def _read_jt(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    from fascat.io.jt import read_jt

    return read_jt(path, options=options)


def _usd_options_for_path(path: str | Path, options: UsdExportOptions | None) -> UsdExportOptions | None:
    if Path(path).suffix.lower() != ".usdz":
        return options
    if options is None:
        return UsdExportOptions(package="usdz")
    if options.package == "usdz":
        return options
    return replace(options, package="usdz")


def _write_usd(
    asset: Asset,
    path: str | Path,
    options: ExportOptions | None,
    *,
    debug: bool,
) -> dict[str, int] | None:
    from fascat.io.usd import write_usd_with_validation_stats

    assert options is None or isinstance(options, UsdExportOptions)
    return write_usd_with_validation_stats(asset, path, debug=debug, options=_usd_options_for_path(path, options))


def _write_gltf(
    asset: Asset,
    path: str | Path,
    options: ExportOptions | None,
    *,
    debug: bool,
) -> dict[str, int] | None:
    from fascat.io.gltf import write_gltf_with_validation

    _ = debug
    assert options is None or isinstance(options, GltfExportOptions)
    return write_gltf_with_validation(asset, path, options=options)


def _write_obj(
    asset: Asset,
    path: str | Path,
    options: ExportOptions | None,
    *,
    debug: bool,
) -> dict[str, int] | None:
    from fascat.io.obj import write_obj_with_validation_stats

    _ = debug
    assert options is None or isinstance(options, ObjExportOptions)
    return write_obj_with_validation_stats(asset, path, options=options)


def _write_stl(
    asset: Asset,
    path: str | Path,
    options: ExportOptions | None,
    *,
    debug: bool,
) -> dict[str, int] | None:
    from fascat.io.stl import write_stl_with_validation_stats

    _ = debug
    assert options is None or isinstance(options, StlExportOptions)
    return write_stl_with_validation_stats(asset, path, options=options)


def _write_fbx(
    asset: Asset,
    path: str | Path,
    options: ExportOptions | None,
    *,
    debug: bool,
) -> dict[str, int] | None:
    from fascat.io.fbx import write_fbx_with_validation_stats

    _ = debug
    assert options is None or isinstance(options, FbxExportOptions)
    return write_fbx_with_validation_stats(asset, path, options=options)


def _validate_usd(path: str | Path) -> dict[str, int]:
    from fascat.io.usd import validate_usd

    return validate_usd(path)


def _validate_gltf(path: str | Path) -> dict[str, int]:
    from fascat.io.gltf import validate_gltf

    return validate_gltf(path)


def _validate_obj(path: str | Path) -> dict[str, int]:
    from fascat.io.obj import validate_obj

    return validate_obj(path)


def _validate_stl(path: str | Path) -> dict[str, int]:
    from fascat.io.stl import validate_stl

    return validate_stl(path)


def _validate_fbx(path: str | Path) -> dict[str, int]:
    from fascat.io.fbx import validate_fbx

    return validate_fbx(path)


READERS: tuple[ReaderSpec, ...] = (
    ReaderSpec("step", STEP_SUFFIXES, _read_step),
    ReaderSpec("iges", IGES_SUFFIXES, _read_iges),
    ReaderSpec("brep", BREP_SUFFIXES, _read_brep),
    ReaderSpec("jt", JT_SUFFIXES, _read_jt),
)

EXPORTERS: dict[ExportFormat, ExporterSpec] = {
    "usd": ExporterSpec("usd", "OpenUSD", USD_SUFFIXES, UsdExportOptions, "usd-core", _write_usd, _validate_usd),
    "gltf": ExporterSpec("gltf", "glTF", GLTF_SUFFIXES, GltfExportOptions, "fascat-gltf", _write_gltf, _validate_gltf),
    "obj": ExporterSpec("obj", "OBJ", OBJ_SUFFIXES, ObjExportOptions, "fascat-obj", _write_obj, _validate_obj),
    "stl": ExporterSpec("stl", "STL", STL_SUFFIXES, StlExportOptions, "fascat-stl", _write_stl, _validate_stl),
    "fbx": ExporterSpec("fbx", "FBX", FBX_SUFFIXES, FbxExportOptions, "fascat-fbx", _write_fbx, _validate_fbx),
}


def reader_for_suffix(suffix: str) -> ReaderSpec:
    normalized = suffix.lower()
    for spec in READERS:
        if normalized in spec.suffixes:
            return spec
    raise ValueError(f"unsupported CAD extension: {normalized or '<none>'}")


def export_format_for_path(path: str | Path) -> ExportFormat:
    if str(path) == "-":
        return "usd"
    suffix = Path(path).suffix.lower()
    for spec in EXPORTERS.values():
        if suffix in spec.suffixes:
            return spec.format
    raise ValueError(f"unsupported export extension: {suffix or '<none>'}")
