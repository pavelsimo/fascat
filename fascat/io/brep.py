from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fascat._ocp import shape_fingerprint as _shape_fingerprint
from fascat.asset import Asset, Node, Part
from fascat.io import _import_base as _base
from fascat.io._errors import wrap_io_errors
from fascat.io._reader_utils import (
    coerce_read_options,
    patch_bytes_source,
    read_via_temporary_file,
)
from fascat.io._suffixes import BREP_SUFFIXES
from fascat.io.step import xde as _xde
from fascat.material import Material
from fascat.options import BrepReadOptions, StepReadOptions
from fascat.report import Report, timed_step

_DEFAULT_MATERIAL_COLOR = (0.75, 0.75, 0.75, 1.0)


@wrap_io_errors("read BREP")
def read_brep(path: str | Path, *, options: BrepReadOptions | StepReadOptions | None = None) -> Asset:
    """Read a native BREP file into an asset."""
    source = Path(path)
    return _read_brep_path(source, source_identity=str(source.resolve()), options=_coerce_options(options))


@wrap_io_errors("read BREP bytes")
def read_brep_bytes(
    data: bytes,
    *,
    name: str = "stdin.brep",
    options: BrepReadOptions | StepReadOptions | None = None,
) -> Asset:
    asset = read_via_temporary_file(
        data,
        suffix=".brep",
        source_identity=name,
        options=_coerce_options(options),
        reader=_read_brep_path,
    )
    patch_bytes_source(asset, name)
    return asset


def _read_brep_path(source: Path, *, source_identity: str, options: BrepReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing BREP file: {source}")
    if source.suffix.lower() not in BREP_SUFFIXES:
        raise ValueError(f"unsupported BREP extension: {source.suffix or '<none>'}")

    cleanup = _base._ImportCleanupStats()
    with timed_step() as timer:
        shape = _read_shape(source)
        topology = _xde._shape_topology_counts(shape)
        representation = _base._loaded_representation(topology)
        cleanup.record_loaded(representation)
        space = _base._space_normalization("millimetre", 0.001, options)
        material_id = _material_id(_DEFAULT_MATERIAL_COLOR)
        shape_hash = _shape_fingerprint(shape)
        part_id = _stable_id("part", f"{source_identity}:{shape_hash}")
        root = Node(
            id=_stable_id("node", f"{source_identity}:root"),
            name=source.stem,
            transform=space.transform,
            metadata={
                "source": str(source),
                "source_identity": source_identity,
                "space_normalization": space.metadata(),
            },
            children=[
                Node(
                    id=_stable_id("node", f"{source_identity}:root/1"),
                    name=source.stem,
                    part_id=part_id,
                    metadata={"loaded_representation": representation},
                )
            ],
        )
        parts = {
            part_id: Part(
                id=part_id,
                name=source.stem,
                source_shape=shape,
                material_ids=[material_id],
                metadata={
                    "source_identity": source_identity,
                    "source_name": source.stem,
                    "shape_fingerprint": shape_hash,
                    "loaded_representation": representation,
                    "source_vertices": str(topology.vertices),
                    "source_edges": str(topology.edges),
                    "source_faces": str(topology.faces),
                },
                fingerprint=shape_hash,
            )
        }
        materials = {
            material_id: Material(
                id=material_id,
                name="Default BREP material",
                base_color=_DEFAULT_MATERIAL_COLOR,
            )
        }

    report = Report(source_path=str(source))
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        units=space.target_units,
        meters_per_unit=space.target_meters_per_unit,
        up_axis=cast(Any, space.target_up_axis),
        source_path=source,
        metadata=_asset_metadata(source, source_identity, options, cleanup, space),
        pmi=[],
        report=report,
    )
    asset.report.input_stats = asset.stats()
    loaded_representations = _base._loaded_representation_report(asset)
    if asset.metadata:
        asset.metadata["import_representation_summary"] = loaded_representations["summary"]
    asset.report.add_step(
        "import",
        options={
            "format": "BREP",
            "backend": "OCP",
            "read_options": options.to_dict(),
            "metadata_count": _base._metadata_count(asset),
            "cleanup": cleanup.to_dict(),
            "space_normalization": space.metadata(),
            "loaded_representations": loaded_representations,
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
    )
    return asset


def _read_shape(path: Path) -> object:
    try:
        from OCP.BRep import BRep_Builder
        from OCP.BRepTools import BRepTools
        from OCP.TopoDS import TopoDS_Shape
    except ImportError as exc:
        raise RuntimeError("BREP import requires cadquery-ocp") from exc

    shape = TopoDS_Shape()
    builder = BRep_Builder()
    if not BRepTools.Read_s(shape, str(path), builder) or shape.IsNull():
        raise RuntimeError(f"failed to read BREP file: {path}")
    return shape


def _asset_metadata(
    source: Path,
    source_identity: str,
    options: BrepReadOptions,
    cleanup: _base._ImportCleanupStats,
    space: _base._SpaceNormalization,
) -> dict[str, object]:
    metadata = _base._asset_metadata(
        source,
        source_identity,
        options,
        _base._StepHeaderInfo(),
        cleanup,
        space,
    )
    if metadata:
        metadata["format"] = "BREP"
    return metadata


def _coerce_options(options: BrepReadOptions | StepReadOptions | None) -> BrepReadOptions:
    return coerce_read_options(options, BrepReadOptions)


def _material_id(color: tuple[float, float, float, float]) -> str:
    encoded = ",".join(f"{component:.6f}" for component in color)
    return _stable_id("mat", encoded)


def _stable_id(prefix: str, value: str) -> str:
    return _base._stable_id(prefix, value)
