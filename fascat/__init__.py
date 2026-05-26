"""Convert CAD data into realtime-ready OpenUSD and glTF assets."""

from fascat import profiles
from fascat.asset import Asset, Node, Part
from fascat.io.gltf import validate_gltf
from fascat.io.importer import read_cad
from fascat.io.jt import has_native_jt_backend, read_jt
from fascat.io.step import read_step
from fascat.io.usd import validate_usd
from fascat.material import Material
from fascat.mesh import Mesh, MeshValidationError
from fascat.options import LODOptions, OptimizeOptions, RepairOptions, StageOptions, Tessellation
from fascat.pipeline import convert, lods, optimize, repair, stage, tessellate, validate_output, write_gltf, write_usd

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
    "has_native_jt_backend",
    "lods",
    "optimize",
    "profiles",
    "read_cad",
    "read_jt",
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
