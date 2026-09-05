from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from fascat.io._suffixes import (
    BREP_SUFFIXES,
    IGES_SUFFIXES,
    JT_SUFFIXES,
    STEP_SUFFIXES,
)
from fascat.options import (
    AnalyzeOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    ConversionProfile,
    DecimateOptions,
    DeleteDegeneratePolygonsOptions,
    ExplodeOptions,
    FbxExportOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    LODOptions,
    MergeOptions,
    MergeVerticesOptions,
    ObjExportOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    ReplaceOptions,
    SceneOptimizeOptions,
    StageOptions,
    StepReadOptions,
    StlExportOptions,
    TessellationOptions,
    UsdExportOptions,
)

if TYPE_CHECKING:
    from fascat.analysis import AnalysisReport
    from fascat.filter import Filter
    from fascat.pipeline_file import PipelineSpec

from ._enums import StdoutFormat
from ._output import _fail, _is_stdio, _require_existing_file


def by_name(name: str, **overrides: Any) -> ConversionProfile:
    from fascat.profiles import by_name as _by_name

    return _by_name(name, **overrides)


def profile_from_file(
    path: str | Path,
    *,
    base: str | ConversionProfile = "realtime-desktop",
) -> ConversionProfile:
    from fascat.profiles import from_file as _profile_from_file

    return _profile_from_file(path, base=base)


def convert(*args: Any, **kwargs: Any) -> Any:
    from fascat.pipeline import convert as _convert

    return _convert(*args, **kwargs)


def validate_export(path: str | Path) -> dict[str, int]:
    from fascat.pipeline import validate_output

    return validate_output(path)


def _write_tessellation_quality_report(asset: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asset.tessellation_quality_report(), indent=2, sort_keys=True), encoding="utf-8")


def _read_cad_for_cli(
    path: Path,
    ctx: typer.Context,
    payload: dict[str, Any],
    *,
    import_options: StepReadOptions | None = None,
) -> Any:
    if _is_stdio(path):
        from fascat.io.step import read_step_bytes

        data = sys.stdin.buffer.read()
        if not data:
            _fail(ctx, payload, "Missing input data on stdin.")
        return read_step_bytes(data, options=import_options)
    _require_existing_file(path, "input", ctx, payload)
    try:
        suffix = path.suffix.lower()
        if suffix in STEP_SUFFIXES:
            from fascat.io.step import read_step

            return read_step(path, options=import_options)
        if suffix in IGES_SUFFIXES:
            from fascat.io.iges import read_iges

            return read_iges(path, options=import_options)
        if suffix in BREP_SUFFIXES:
            from fascat.io.brep import read_brep

            return read_brep(path, options=import_options)
        if suffix in JT_SUFFIXES:
            from fascat.io.jt import read_jt

            return read_jt(path, options=import_options)
        raise ValueError(f"unsupported CAD extension: {path.suffix or '<none>'}")
    except Exception as exc:
        _fail(ctx, payload, str(exc))
        raise AssertionError("unreachable") from exc


def _convert_for_cli(
    input_path: Path | list[Path],
    output_path: Path,
    *,
    profile: ConversionProfile,
    pipeline: PipelineSpec | None,
    tessellation: TessellationOptions,
    stage: StageOptions,
    import_options: StepReadOptions,
    heal_brep: BrepHealOptions | None,
    merge_vertices: MergeVerticesOptions | None,
    delete_degenerate_polygons: DeleteDegeneratePolygonsOptions | None,
    merge: MergeOptions | None,
    explode: ExplodeOptions | None,
    replace: ReplaceOptions | None,
    scene: SceneOptimizeOptions | None,
    bake_materials: BakeMaterialOptions | None,
    remove_holes: RemoveHolesOptions | None,
    remove_occluded: RemoveOccludedOptions | None,
    decimate: DecimateOptions | None,
    lod_generator: LODGeneratorOptions | None,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    where: Filter | None,
    progress: Callable[[str, dict[str, int]], None] | None,
    debug: bool,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
    fbx_options: FbxExportOptions | None,
    stdout_format: StdoutFormat,
) -> Any:
    if isinstance(input_path, Path) and _is_stdio(input_path):
        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing input data on stdin.")
        with _temporary_step_file(data) as temp_input:
            return _convert_output(
                temp_input,
                output_path,
                profile,
                pipeline,
                tessellation,
                stage,
                import_options,
                heal_brep,
                merge_vertices,
                delete_degenerate_polygons,
                merge,
                explode,
                replace,
                scene,
                bake_materials,
                remove_holes,
                remove_occluded,
                decimate,
                lod_generator,
                optimize,
                lods,
                where,
                progress,
                debug,
                gltf_options,
                usd_options,
                obj_options,
                stl_options,
                fbx_options,
                stdout_format=stdout_format,
            )
    return _convert_output(
        input_path,
        output_path,
        profile,
        pipeline,
        tessellation,
        stage,
        import_options,
        heal_brep,
        merge_vertices,
        delete_degenerate_polygons,
        merge,
        explode,
        replace,
        scene,
        bake_materials,
        remove_holes,
        remove_occluded,
        decimate,
        lod_generator,
        optimize,
        lods,
        where,
        progress,
        debug,
        gltf_options,
        usd_options,
        obj_options,
        stl_options,
        fbx_options,
        stdout_format=stdout_format,
    )


def _convert_output(
    input_path: Path | list[Path],
    output_path: Path,
    profile: ConversionProfile,
    pipeline: PipelineSpec | None,
    tessellation: TessellationOptions,
    stage: StageOptions,
    import_options: StepReadOptions,
    heal_brep: BrepHealOptions | None,
    merge_vertices: MergeVerticesOptions | None,
    delete_degenerate_polygons: DeleteDegeneratePolygonsOptions | None,
    merge: MergeOptions | None,
    explode: ExplodeOptions | None,
    replace: ReplaceOptions | None,
    scene: SceneOptimizeOptions | None,
    bake_materials: BakeMaterialOptions | None,
    remove_holes: RemoveHolesOptions | None,
    remove_occluded: RemoveOccludedOptions | None,
    decimate: DecimateOptions | None,
    lod_generator: LODGeneratorOptions | None,
    optimize: OptimizeOptions | None,
    lods: LODOptions | None,
    where: Filter | None,
    progress: Callable[[str, dict[str, int]], None] | None,
    debug: bool,
    gltf_options: GltfExportOptions | None,
    usd_options: UsdExportOptions | None,
    obj_options: ObjExportOptions | None,
    stl_options: StlExportOptions | None,
    fbx_options: FbxExportOptions | None,
    *,
    stdout_format: StdoutFormat,
) -> Any:
    if _is_stdio(output_path):
        import tempfile

        import click

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=f".{stdout_format.value}", delete=False) as handle:
                temp_path = Path(handle.name)
            assert temp_path is not None
            asset = convert(
                input_path,
                temp_path,
                profile=profile,
                pipeline=pipeline,
                import_options=import_options,
                tessellation=tessellation,
                heal_brep=heal_brep,
                merge_vertices=merge_vertices,
                delete_degenerate_polygons=delete_degenerate_polygons,
                stage=stage,
                merge=merge,
                explode=explode,
                replace=replace,
                scene=scene,
                bake_materials=bake_materials,
                remove_holes=remove_holes,
                remove_occluded=remove_occluded,
                decimate=decimate,
                lod_generator=lod_generator,
                optimize=optimize,
                lods=lods,
                where=where,
                progress=progress,
                debug=debug,
                gltf_options=gltf_options,
                usd_options=usd_options,
                obj_options=obj_options,
                stl_options=stl_options,
                fbx_options=fbx_options,
            )
            stdout = click.get_binary_stream("stdout")
            stdout.write(temp_path.read_bytes())
            stdout.flush()
            return asset
        finally:
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()
    return convert(
        input_path,
        output_path,
        profile=profile,
        pipeline=pipeline,
        import_options=import_options,
        tessellation=tessellation,
        heal_brep=heal_brep,
        merge_vertices=merge_vertices,
        delete_degenerate_polygons=delete_degenerate_polygons,
        stage=stage,
        merge=merge,
        explode=explode,
        replace=replace,
        scene=scene,
        bake_materials=bake_materials,
        remove_holes=remove_holes,
        remove_occluded=remove_occluded,
        decimate=decimate,
        lod_generator=lod_generator,
        optimize=optimize,
        lods=lods,
        where=where,
        progress=progress,
        debug=debug,
        gltf_options=gltf_options,
        usd_options=usd_options,
        obj_options=obj_options,
        stl_options=stl_options,
        fbx_options=fbx_options,
    )


class _temporary_step_file:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.path: Path | None = None
        self._handle: Any = None

    def __enter__(self) -> Path:
        import tempfile

        self._handle = tempfile.NamedTemporaryFile(suffix=".step", delete=False)
        self._handle.write(self.data)
        self._handle.flush()
        self.path = Path(self._handle.name)
        self._handle.close()
        return self.path

    def __exit__(self, *_exc_info: object) -> None:
        if self.path is not None:
            with suppress(FileNotFoundError):
                self.path.unlink()


def _validate_and_analyze_output_for_cli(
    path: Path,
    options: AnalyzeOptions | None,
    *,
    where: Filter | None = None,
) -> tuple[dict[str, int], AnalysisReport | None]:
    if _is_stdio(path):
        import tempfile

        data = sys.stdin.buffer.read()
        if not data:
            raise RuntimeError("Missing USD data on stdin.")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
                handle.flush()
            assert temp_path is not None
            stats = validate_export(temp_path)
            stats["file_size_bytes"] = len(data)
            if options is not None:
                from fascat.analysis import analyze_output

                analysis = analyze_output(temp_path, options, where=where, validation_stats=stats, source_path="-")
            else:
                analysis = None
            return stats, analysis
        finally:
            if temp_path is not None:
                with suppress(FileNotFoundError):
                    temp_path.unlink()
    stats = validate_export(path)
    if options is not None:
        from fascat.analysis import analyze_output

        analysis = analyze_output(path, options, where=where, validation_stats=stats)
    else:
        analysis = None
    return stats, analysis
