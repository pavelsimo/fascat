"""Convert CAD STEP data into realtime-ready OpenUSD and glTF assets."""

from fascat import profiles
from fascat.asset import Asset, Node, Part
from fascat.filter import Filter, FilterExpressionError, SelectionMatch, SelectionResult
from fascat.io.gltf import validate_gltf
from fascat.io.step import read_step
from fascat.io.usd import validate_usd
from fascat.material import Material
from fascat.mesh import Mesh, MeshValidationError
from fascat.metadata import Metadata, PmiAnnotation, Tolerance
from fascat.options import (
    AtlasOptions,
    BrepHealOptions,
    LODOptions,
    MergeOptions,
    OptimizeOptions,
    RepairOptions,
    StageOptions,
    StepReadOptions,
    Tessellation,
    UnwrapOptions,
)
from fascat.pipeline import (
    convert,
    heal_brep,
    lods,
    merge,
    optimize,
    repair,
    stage,
    tessellate,
    validate_output,
    write_gltf,
    write_usd,
)

__version__ = "0.1.0"

__all__ = [
    "Asset",
    "AtlasOptions",
    "BrepHealOptions",
    "Filter",
    "FilterExpressionError",
    "LODOptions",
    "Material",
    "Metadata",
    "MergeOptions",
    "Mesh",
    "MeshValidationError",
    "Node",
    "OptimizeOptions",
    "Part",
    "PmiAnnotation",
    "RepairOptions",
    "SelectionMatch",
    "SelectionResult",
    "StageOptions",
    "StepReadOptions",
    "Tessellation",
    "Tolerance",
    "UnwrapOptions",
    "__version__",
    "convert",
    "heal_brep",
    "lods",
    "merge",
    "optimize",
    "profiles",
    "read_step",
    "repair",
    "stage",
    "tessellate",
    "validate_gltf",
    "validate_output",
    "validate_usd",
    "write_gltf",
    "write_usd",
]
