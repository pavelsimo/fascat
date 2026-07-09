"""Convert CAD data into realtime-ready OpenUSD, glTF, OBJ, STL, and FBX assets.

The public surface loads lazily (PEP 562): ``import fascat`` is cheap, and each
name pulls in its home module on first attribute access.
"""

from typing import TYPE_CHECKING, Any

__version__ = "0.4.0"

if TYPE_CHECKING:  # static names for type checkers and IDEs; never executed at runtime
    from fascat import options, profiles, validation  # noqa: F401
    from fascat.analysis import AnalysisReport  # noqa: F401
    from fascat.asset import Asset, Node, Part  # noqa: F401
    from fascat.errors import Error, FascatError, FascatIOError  # noqa: F401
    from fascat.filter import Filter, FilterExpressionError, SelectionMatch, SelectionResult  # noqa: F401
    from fascat.image import ImageResource  # noqa: F401
    from fascat.io.brep import read_brep  # noqa: F401
    from fascat.io.iges import read_iges  # noqa: F401
    from fascat.io.jt import read_jt  # noqa: F401
    from fascat.io.step import read_step, read_step_many  # noqa: F401
    from fascat.material import Material  # noqa: F401
    from fascat.mesh import Mesh, MeshValidationError  # noqa: F401
    from fascat.metadata import Metadata, PmiAnnotation, Tolerance  # noqa: F401
    from fascat.ops.stage import UVOverlapError  # noqa: F401
    from fascat.options import (  # noqa: F401
        DecimateOptions,
        GltfExportOptions,
        LODOptions,
        MergeOptions,
        OptimizeOptions,
        PlatformBudget,
        RepairOptions,
        StageOptions,
        TessellationOptions,
        UsdExportOptions,
    )
    from fascat.pipeline import (  # noqa: F401
        analyze,
        convert,
        validate_output,
        write_fbx,
        write_gltf,
        write_obj,
        write_stl,
        write_usd,
    )
    from fascat.pipeline_file import PipelineSpec, PipelineStep  # noqa: F401

_SUBMODULES = frozenset({"options", "profiles", "validation"})
_EXPORTS: dict[str, str] = {
    "AnalysisReport": "fascat.analysis",
    "Asset": "fascat.asset",
    "Node": "fascat.asset",
    "Part": "fascat.asset",
    "Error": "fascat.errors",
    "FascatError": "fascat.errors",
    "FascatIOError": "fascat.errors",
    "Filter": "fascat.filter",
    "FilterExpressionError": "fascat.filter",
    "SelectionMatch": "fascat.filter",
    "SelectionResult": "fascat.filter",
    "ImageResource": "fascat.image",
    "read_brep": "fascat.io.brep",
    "read_iges": "fascat.io.iges",
    "read_jt": "fascat.io.jt",
    "read_step": "fascat.io.step",
    "read_step_many": "fascat.io.step",
    "Material": "fascat.material",
    "UVOverlapError": "fascat.ops.stage",
    "Mesh": "fascat.mesh",
    "MeshValidationError": "fascat.mesh",
    "Metadata": "fascat.metadata",
    "PmiAnnotation": "fascat.metadata",
    "Tolerance": "fascat.metadata",
    "DecimateOptions": "fascat.options",
    "GltfExportOptions": "fascat.options",
    "LODOptions": "fascat.options",
    "MergeOptions": "fascat.options",
    "OptimizeOptions": "fascat.options",
    "PlatformBudget": "fascat.options",
    "RepairOptions": "fascat.options",
    "StageOptions": "fascat.options",
    "TessellationOptions": "fascat.options",
    "UsdExportOptions": "fascat.options",
    "analyze": "fascat.pipeline",
    "convert": "fascat.pipeline",
    "validate_output": "fascat.pipeline",
    "write_fbx": "fascat.pipeline",
    "write_gltf": "fascat.pipeline",
    "write_obj": "fascat.pipeline",
    "write_stl": "fascat.pipeline",
    "write_usd": "fascat.pipeline",
    "PipelineSpec": "fascat.pipeline_file",
    "PipelineStep": "fascat.pipeline_file",
}

__all__ = [
    "AnalysisReport",
    "Asset",
    "DecimateOptions",
    "Error",
    "FascatError",
    "FascatIOError",
    "Filter",
    "FilterExpressionError",
    "GltfExportOptions",
    "ImageResource",
    "LODOptions",
    "Material",
    "MergeOptions",
    "Mesh",
    "MeshValidationError",
    "Metadata",
    "Node",
    "OptimizeOptions",
    "Part",
    "PipelineSpec",
    "PipelineStep",
    "PlatformBudget",
    "PmiAnnotation",
    "RepairOptions",
    "SelectionMatch",
    "SelectionResult",
    "StageOptions",
    "TessellationOptions",
    "Tolerance",
    "UVOverlapError",
    "UsdExportOptions",
    "__version__",
    "analyze",
    "convert",
    "profiles",
    "read_brep",
    "read_iges",
    "read_jt",
    "read_step",
    "read_step_many",
    "validate_output",
    "write_fbx",
    "write_gltf",
    "write_obj",
    "write_stl",
    "write_usd",
]


def __getattr__(name: str) -> Any:
    import importlib

    if name in _SUBMODULES:
        module = importlib.import_module(f"fascat.{name}")
        globals()[name] = module
        return module
    home = _EXPORTS.get(name)
    if home is None:
        raise AttributeError(f"module 'fascat' has no attribute {name!r}")
    value = getattr(importlib.import_module(home), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | _SUBMODULES)
