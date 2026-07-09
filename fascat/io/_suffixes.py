from __future__ import annotations

STEP_SUFFIXES = frozenset({".step", ".stp"})
IGES_SUFFIXES = frozenset({".igs", ".iges"})
BREP_SUFFIXES = frozenset({".brep"})
JT_SUFFIXES = frozenset({".jt"})

GLTF_SUFFIXES = frozenset({".gltf", ".glb"})
OBJ_SUFFIXES = frozenset({".obj"})
STL_SUFFIXES = frozenset({".stl"})
FBX_SUFFIXES = frozenset({".fbx"})
USD_SUFFIXES = frozenset({".usd", ".usda", ".usdc", ".usdz"})

CAD_SUFFIXES = STEP_SUFFIXES | IGES_SUFFIXES | BREP_SUFFIXES | JT_SUFFIXES
EXPORT_SUFFIXES = USD_SUFFIXES | GLTF_SUFFIXES | OBJ_SUFFIXES | STL_SUFFIXES | FBX_SUFFIXES
