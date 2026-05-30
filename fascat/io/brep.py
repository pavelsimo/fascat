from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any, cast

from fascat._ocp import shape_fingerprint as _shape_fingerprint
from fascat.asset import Asset, Node, Part
from fascat.io import step as _step
from fascat.material import Material
from fascat.options import BrepReadOptions, StepReadOptions
from fascat.report import Report, timed_step

BREP_SUFFIXES = {".brep"}
_DEFAULT_MATERIAL_COLOR = (0.75, 0.75, 0.75, 1.0)


def read_brep(path: str | Path, *, options: BrepReadOptions | StepReadOptions | None = None) -> Asset:
    source = Path(path)
    return _read_brep_path(source, source_identity=str(source.resolve()), options=_coerce_options(options))


def read_brep_bytes(
    data: bytes,
    *,
    name: str = "stdin.brep",
    options: BrepReadOptions | StepReadOptions | None = None,
) -> Asset:
    with tempfile.NamedTemporaryFile(suffix=".brep") as handle:
        handle.write(data)
        handle.flush()
        asset = _read_brep_path(Path(handle.name), source_identity=name, options=_coerce_options(options))
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    if asset.metadata:
        asset.metadata["source"] = name
        asset.metadata["source_identity"] = name
    return asset


def _read_brep_path(source: Path, *, source_identity: str, options: BrepReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing BREP file: {source}")
    if source.suffix.lower() not in BREP_SUFFIXES:
        raise ValueError(f"unsupported BREP extension: {source.suffix or '<none>'}")

    cleanup = _step._ImportCleanupStats()
    with timed_step() as timer:
        shape = _read_shape(source)
        topology = _step._shape_topology_counts(shape)
        representation = _step._loaded_representation(topology)
        cleanup.record_loaded(representation)
        space = _step._space_normalization("millimetre", 0.001, options)
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
    loaded_representations = _step._loaded_representation_report(asset)
    if asset.metadata:
        asset.metadata["import_representation_summary"] = loaded_representations["summary"]
    asset.report.add_step(
        "import",
        options={
            "format": "BREP",
            "backend": "OCP",
            "read_options": options.to_dict(),
            "metadata_count": _step._metadata_count(asset),
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
    cleanup: _step._ImportCleanupStats,
    space: _step._SpaceNormalization,
) -> dict[str, object]:
    metadata = _step._asset_metadata(
        source,
        source_identity,
        options,
        _step._StepHeaderInfo(),
        cleanup,
        space,
    )
    if metadata:
        metadata["format"] = "BREP"
    return metadata


def _coerce_options(options: BrepReadOptions | StepReadOptions | None) -> BrepReadOptions:
    if options is None:
        return BrepReadOptions()
    if isinstance(options, BrepReadOptions):
        return options
    return BrepReadOptions(**cast(Any, options.to_dict()))


def _material_id(color: tuple[float, float, float, float]) -> str:
    encoded = ",".join(f"{component:.6f}" for component in color)
    return _stable_id("mat", encoded)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
