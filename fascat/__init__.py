"""Convert CAD STEP data into realtime-ready OpenUSD assets."""

from fascat import profiles
from fascat.asset import Asset, Node, Part
from fascat.io.step import read_step
from fascat.io.usd import validate_usd, write_usd
from fascat.material import Material
from fascat.mesh import Mesh, MeshValidationError
from fascat.options import LODOptions, OptimizeOptions, RepairOptions, StageOptions, Tessellation
from fascat.pipeline import convert, optimize, repair, tessellate

__version__ = "0.1.0"

__all__ = [
    "Asset",
    "LODOptions",
    "Material",
    "Mesh",
    "MeshValidationError",
    "Node",
    "OptimizeOptions",
    "Part",
    "RepairOptions",
    "StageOptions",
    "Tessellation",
    "__version__",
    "convert",
    "optimize",
    "profiles",
    "read_step",
    "repair",
    "tessellate",
    "validate_usd",
    "write_usd",
]
