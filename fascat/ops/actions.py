"""Compatibility imports for the action operations."""

from fascat.ops.bake import bake_materials_asset
from fascat.ops.decimate import decimate_asset, decimation_target_strategy
from fascat.ops.holes import remove_holes_asset
from fascat.ops.lod import run_lod_generators_asset
from fascat.ops.occlusion import remove_occluded_asset

__all__ = [
    "bake_materials_asset",
    "decimate_asset",
    "decimation_target_strategy",
    "remove_holes_asset",
    "remove_occluded_asset",
    "run_lod_generators_asset",
]
