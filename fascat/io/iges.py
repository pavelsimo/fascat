from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

from fascat.asset import Asset, Node, Part
from fascat.io import step as _step
from fascat.material import Material
from fascat.options import IgesReadOptions, StepReadOptions
from fascat.report import Report, timed_step

IGES_SUFFIXES = {".igs", ".iges"}


def read_iges(path: str | Path, *, options: IgesReadOptions | StepReadOptions | None = None) -> Asset:
    source = Path(path)
    return _read_iges_path(source, source_identity=str(source.resolve()), options=_coerce_options(options))


def read_iges_bytes(
    data: bytes,
    *,
    name: str = "stdin.igs",
    options: IgesReadOptions | StepReadOptions | None = None,
) -> Asset:
    suffix = Path(name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix if suffix in IGES_SUFFIXES else ".igs") as handle:
        handle.write(data)
        handle.flush()
        asset = _read_iges_path(Path(handle.name), source_identity=name, options=_coerce_options(options))
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    if asset.metadata:
        asset.metadata["source"] = name
        asset.metadata["source_identity"] = name
    return asset


def _read_iges_path(source: Path, *, source_identity: str, options: IgesReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing IGES file: {source}")
    if source.suffix.lower() not in IGES_SUFFIXES:
        raise ValueError(f"unsupported IGES extension: {source.suffix or '<none>'}")

    cleanup = _step._ImportCleanupStats()
    with timed_step() as timer:
        document, shape_tool, color_tool = _read_xde_document(source, options)
        space = _step._space_normalization("millimetre", 0.001, options)
        free_labels = _step._free_shape_labels(shape_tool)
        root = Node(
            id=_step._stable_id("node", f"{source_identity}:root"),
            name=source.stem,
            transform=space.transform,
            metadata={
                "source": str(source),
                "source_identity": source_identity,
                "space_normalization": space.metadata(),
            },
        )
        parts: dict[str, Part] = {}
        part_index: _step._PartIndex = {}
        materials: dict[str, Material] = {}
        for index, label in enumerate(free_labels, start=1):
            root.children.append(
                _step._build_node(
                    label,
                    f"root/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )

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
            "format": "IGES",
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
    _ = document
    return asset


def _read_xde_document(path: Path, options: IgesReadOptions) -> tuple[Any, Any, Any]:
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.IGESCAFControl import IGESCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except ImportError as exc:
        raise RuntimeError("IGES import requires cadquery-ocp") from exc

    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("fascat"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)

    reader = IGESCAFControl_Reader()
    reader.SetNameMode(options.metadata)
    reader.SetColorMode(True)
    reader.SetLayerMode(options.layers)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"failed to read IGES file: {path}")
    if not reader.Transfer(document):
        raise RuntimeError(f"failed to transfer IGES data into XDE document: {path}")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    return document, shape_tool, color_tool


def _asset_metadata(
    source: Path,
    source_identity: str,
    options: IgesReadOptions,
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
        metadata["format"] = "IGES"
    return metadata


def _coerce_options(options: IgesReadOptions | StepReadOptions | None) -> IgesReadOptions:
    if options is None:
        return IgesReadOptions()
    if isinstance(options, IgesReadOptions):
        return options
    return IgesReadOptions(**cast(Any, options.to_dict()))
