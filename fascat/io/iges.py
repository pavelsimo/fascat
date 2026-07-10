from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fascat.asset import Asset, Node, Part
from fascat.io import _import_base as _base
from fascat.io._errors import wrap_io_errors
from fascat.io._reader_utils import (
    coerce_read_options,
    patch_bytes_source,
    read_via_temporary_file,
)
from fascat.io._suffixes import IGES_SUFFIXES
from fascat.io.step import materials as _materials
from fascat.io.step import textures as _textures
from fascat.io.step import xde as _xde
from fascat.material import Material
from fascat.options import IgesReadOptions, StepReadOptions
from fascat.report import Report, timed_step


@wrap_io_errors("read IGES")
def read_iges(path: str | Path, *, options: IgesReadOptions | StepReadOptions | None = None) -> Asset:
    """Read an IGES file into an asset."""
    source = Path(path)
    return _read_iges_path(source, source_identity=str(source.resolve()), options=_coerce_options(options))


@wrap_io_errors("read IGES bytes")
def read_iges_bytes(
    data: bytes,
    *,
    name: str = "stdin.igs",
    options: IgesReadOptions | StepReadOptions | None = None,
) -> Asset:
    suffix = Path(name).suffix.lower()
    asset = read_via_temporary_file(
        data,
        suffix=suffix if suffix in IGES_SUFFIXES else ".igs",
        source_identity=name,
        options=_coerce_options(options),
        reader=_read_iges_path,
    )
    patch_bytes_source(asset, name)
    return asset


def _read_iges_path(source: Path, *, source_identity: str, options: IgesReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing IGES file: {source}")
    if source.suffix.lower() not in IGES_SUFFIXES:
        raise ValueError(f"unsupported IGES extension: {source.suffix or '<none>'}")

    cleanup = _base._ImportCleanupStats()
    with timed_step() as timer:
        document, shape_tool, color_tool, vis_material_tool = _read_xde_document(source, options)
        space = _base._space_normalization("millimetre", 0.001, options)
        free_labels = _xde._free_shape_labels(shape_tool)
        root = Node(
            id=_base._stable_id("node", f"{source_identity}:root"),
            name=source.stem,
            transform=space.transform,
            metadata={
                "source": str(source),
                "source_identity": source_identity,
                "space_normalization": space.metadata(),
            },
        )
        parts: dict[str, Part] = {}
        part_index: _base._PartIndex = {}
        materials: dict[str, Material] = {}
        for index, label in enumerate(free_labels, start=1):
            root.children.append(
                _xde._build_node(
                    label,
                    f"root/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    vis_material_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )
        source_textures = _textures._extract_source_textures(source, source_identity, options)
        texture_binding_summary = _textures._attach_source_textures_to_materials(materials, source_textures.images)
        material_libraries = _materials._extract_material_libraries(source, source_identity, options)
        material_library_binding_summary = _materials._apply_material_libraries_to_materials(
            materials, material_libraries
        )
        images = {**source_textures.images, **material_libraries.images}

    report = Report(source_path=str(source))
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        images=images,
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
        asset.metadata["source_texture_import"] = source_textures.summary
        asset.metadata["source_texture_bindings"] = texture_binding_summary
        asset.metadata["material_library_import"] = material_libraries.summary
        asset.metadata["material_library_bindings"] = material_library_binding_summary
    import_warnings = [*source_textures.warnings, *material_libraries.warnings]
    for warning in import_warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "IGES",
            "backend": "OCP",
            "read_options": options.to_dict(),
            "metadata_count": _base._metadata_count(asset),
            "cleanup": cleanup.to_dict(),
            "space_normalization": space.metadata(),
            "source_textures": source_textures.summary,
            "source_texture_bindings": texture_binding_summary,
            "material_libraries": material_libraries.summary,
            "material_library_bindings": material_library_binding_summary,
            "loaded_representations": loaded_representations,
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
        warnings=import_warnings,
    )
    _ = document
    return asset


def _read_xde_document(path: Path, options: IgesReadOptions) -> tuple[Any, Any, Any, Any]:
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
    vis_material_tool = XCAFDoc_DocumentTool.VisMaterialTool_s(document.Main())
    return document, shape_tool, color_tool, vis_material_tool


def _asset_metadata(
    source: Path,
    source_identity: str,
    options: IgesReadOptions,
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
        source_texture_summary=None,
        texture_binding_summary=None,
    )
    if metadata:
        metadata["format"] = "IGES"
    return metadata


def _coerce_options(options: IgesReadOptions | StepReadOptions | None) -> IgesReadOptions:
    return coerce_read_options(options, IgesReadOptions)
