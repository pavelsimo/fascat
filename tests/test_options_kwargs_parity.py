"""The kwargs TypedDicts must mirror their backing Options dataclasses exactly."""

from __future__ import annotations

import dataclasses
import inspect
import typing

import fascat.options as options_module
from fascat.asset import Asset

REGISTRY = {
    "tessellate": (options_module.TessellateKwargs, options_module.TessellationOptions),
    "repair": (options_module.RepairKwargs, options_module.RepairOptions),
    "merge_vertices": (options_module.MergeVerticesKwargs, options_module.MergeVerticesOptions),
    "delete_degenerate_polygons": (
        options_module.DeleteDegeneratePolygonsKwargs,
        options_module.DeleteDegeneratePolygonsOptions,
    ),
    "heal_brep": (options_module.HealBrepKwargs, options_module.BrepHealOptions),
    "stage": (options_module.StageKwargs, options_module.StageOptions),
    "optimize": (options_module.OptimizeKwargs, options_module.OptimizeOptions),
    "lods": (options_module.LodsKwargs, options_module.LODOptions),
    "merge": (options_module.MergeKwargs, options_module.MergeOptions),
    "explode": (options_module.ExplodeKwargs, options_module.ExplodeOptions),
    "replace": (options_module.ReplaceKwargs, options_module.ReplaceOptions),
    "optimize_scene": (options_module.OptimizeSceneKwargs, options_module.SceneOptimizeOptions),
    "bake_materials": (options_module.BakeMaterialsKwargs, options_module.BakeMaterialOptions),
    "process_textures": (options_module.ProcessTexturesKwargs, options_module.TextureProcessOptions),
    "decimate": (options_module.DecimateKwargs, options_module.DecimateOptions),
    "remove_holes": (options_module.RemoveHolesKwargs, options_module.RemoveHolesOptions),
    "remove_occluded": (options_module.RemoveOccludedKwargs, options_module.RemoveOccludedOptions),
    "run_lod_generators": (options_module.RunLodGeneratorsKwargs, options_module.LODGeneratorOptions),
    "analyze": (options_module.AnalyzeKwargs, options_module.AnalyzeOptions),
}


def test_kwargs_typeddicts_mirror_options_fields() -> None:
    for method, (kwargs_cls, options_cls) in REGISTRY.items():
        kwarg_hints = typing.get_type_hints(kwargs_cls)
        option_hints = typing.get_type_hints(options_cls)
        field_names = {field.name for field in dataclasses.fields(options_cls)}
        assert set(kwarg_hints) == field_names, f"{method}: kwargs/fields drift"
        for name in field_names:
            assert kwarg_hints[name] == option_hints[name], f"{method}.{name}: type drift"


def test_every_kwargs_enabled_method_is_registered() -> None:
    for method in REGISTRY:
        signature = inspect.signature(getattr(Asset, method))
        kinds = {parameter.kind for parameter in signature.parameters.values()}
        assert inspect.Parameter.VAR_KEYWORD in kinds, f"Asset.{method} lost its **kwargs"
        assert "options" in signature.parameters, f"Asset.{method} lost its options escape hatch"
