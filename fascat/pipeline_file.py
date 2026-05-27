from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import tomli

from fascat.asset import Asset
from fascat.filter import Filter
from fascat.options import (
    AtlasOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    DecimateOptions,
    ExplodeOptions,
    LODGeneratorOptions,
    LODLevel,
    LODOptions,
    MergeOptions,
    MetadataExportOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    RepairOptions,
    ReplaceOptions,
    SceneOptimizeOptions,
    StageOptions,
    StepReadOptions,
    Tessellation,
    UnwrapOptions,
)


@dataclass(frozen=True)
class PipelineStep:
    op: str
    values: dict[str, object]

    def __post_init__(self) -> None:
        if not self.op:
            raise ValueError("pipeline step op must not be empty")
        object.__setattr__(self, "op", self.op.replace("-", "_"))
        object.__setattr__(self, "values", {_normalize_key(key): value for key, value in self.values.items()})

    def to_dict(self) -> dict[str, object]:
        return {"op": self.op, **dict(self.values)}


@dataclass(frozen=True)
class PipelineSpec:
    filters: dict[str, Filter]
    steps: tuple[PipelineStep, ...]
    import_options: StepReadOptions | None = None
    export_metadata: MetadataExportOptions | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("pipeline file must include at least one [[steps]] entry")

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineSpec:
        return cls.from_dict(_load_toml(path))

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> PipelineSpec:
        filters = _filters(values.get("filters", []))
        steps = _steps(values.get("steps", []))
        return cls(
            filters=filters,
            steps=tuple(steps),
            import_options=_import_options(values.get("import")),
            export_metadata=_export_metadata_options(values.get("export")),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "filters": sorted(self.filters),
            "steps": [step.to_dict() for step in self.steps],
            "import": None if self.import_options is None else self.import_options.to_dict(),
            "export": None if self.export_metadata is None else self.export_metadata.to_dict(),
        }

    def apply(
        self,
        asset: Asset,
        *,
        progress: Callable[[str, dict[str, int]], None] | None = None,
    ) -> Asset:
        result = asset
        for step in self.steps:
            where = self._where(step)
            result = _apply_step(result, step, where)
            if progress is not None:
                progress(step.op, result.stats())
        return result

    def _where(self, step: PipelineStep) -> Filter | None:
        where = step.values.get("where")
        where_not = step.values.get("where_not")
        if where is not None and where_not is not None:
            raise ValueError("pipeline step cannot set both where and where_not")
        if where_not is not None:
            return Filter.not_(self._named_filter(where_not, "where_not"))
        if where is not None:
            return self._named_filter(where, "where")
        return None

    def _named_filter(self, value: object, field: str) -> Filter:
        if not isinstance(value, str):
            raise ValueError(f"pipeline step {field} must name a filter")
        if value not in self.filters:
            raise ValueError(f"unknown pipeline filter: {value}")
        return self.filters[value]


def _load_toml(path: str | Path) -> dict[str, object]:
    with Path(path).open("rb") as handle:
        return cast(dict[str, object], tomli.load(handle))


def _filters(value: object) -> dict[str, Filter]:
    if value is None or value == []:
        return {}
    if not isinstance(value, list):
        raise ValueError("pipeline filters must be declared with [[filters]]")
    result: dict[str, Filter] = {}
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("pipeline filter entries must be tables")
        filter_id = item.get("name")
        if not isinstance(filter_id, str) or not filter_id:
            raise ValueError("pipeline filter entries require a non-empty name")
        if filter_id in result:
            raise ValueError(f"duplicate pipeline filter: {filter_id}")
        result[filter_id] = _filter_from_values(item, result)
    return result


def _filter_from_values(values: dict[object, object], named_filters: dict[str, Filter]) -> Filter:
    normalized = {_normalize_key(str(key)): item for key, item in values.items()}
    include = _filter_refs(normalized.get("include", []), named_filters, "include")
    exclude = _filter_refs(normalized.get("exclude", []), named_filters, "exclude")
    metadata = normalized.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("pipeline filter metadata must be a table")
    return Filter(
        path=cast(Any, normalized.get("path")),
        name=cast(Any, normalized.get("names", normalized.get("node_name"))),
        part_name=cast(Any, normalized.get("part_name")),
        part_id=cast(Any, normalized.get("part_id", normalized.get("part"))),
        material=cast(Any, normalized.get("material")),
        metadata=cast(dict[str, object] | None, metadata),
        min_bounds=cast(Any, normalized.get("min_bounds")),
        max_bounds=cast(Any, normalized.get("max_bounds")),
        min_diagonal=cast(float | None, normalized.get("min_diagonal")),
        max_diagonal=cast(float | None, normalized.get("max_diagonal")),
        min_triangles=cast(int | None, normalized.get("min_triangles")),
        max_triangles=cast(int | None, normalized.get("max_triangles")),
        min_vertices=cast(int | None, normalized.get("min_vertices")),
        max_vertices=cast(int | None, normalized.get("max_vertices")),
        include=include,
        exclude=exclude,
    )


def _filter_refs(value: object, named_filters: dict[str, Filter], label: str) -> tuple[Filter, ...]:
    if value is None or value == []:
        return ()
    names = [value] if isinstance(value, str) else value
    if not isinstance(names, list):
        raise ValueError(f"pipeline filter {label} must be a string or list of strings")
    refs: list[Filter] = []
    for name in names:
        if not isinstance(name, str) or name not in named_filters:
            raise ValueError(f"unknown pipeline filter referenced by {label}: {name}")
        refs.append(named_filters[name])
    return tuple(refs)


def _steps(value: object) -> list[PipelineStep]:
    if not isinstance(value, list):
        raise ValueError("pipeline steps must be declared with [[steps]]")
    steps: list[PipelineStep] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("pipeline step entries must be tables")
        op = item.get("op")
        if not isinstance(op, str):
            raise ValueError("pipeline step entries require an op")
        steps.append(PipelineStep(op=op, values={str(key): value for key, value in item.items() if key != "op"}))
    return steps


def _import_options(value: object) -> StepReadOptions | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("pipeline import settings must be a table")
    metadata_enabled = _metadata_import_enabled(value.get("metadata", True))
    return StepReadOptions(
        metadata=metadata_enabled,
        product_metadata=bool(value.get("product_metadata", metadata_enabled)),
        properties=bool(value.get("properties", metadata_enabled)),
        layers=bool(value.get("layers", metadata_enabled)),
        validation_properties=bool(value.get("validation_properties", metadata_enabled)),
        pmi=_pmi_import_enabled(value.get("pmi", True)),
    )


def _export_metadata_options(value: object) -> MetadataExportOptions | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("pipeline export settings must be a table")
    return MetadataExportOptions(
        mode=cast(Any, _literal(value.get("metadata", "full"))),
        pmi=cast(Any, _literal(value.get("pmi", "metadata"))),
    )


def _metadata_import_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.replace("-", "_").lower() != "none"
    raise ValueError("pipeline import metadata must be a bool or one of: none, summary, full")


def _pmi_import_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.replace("-", "_").lower() != "none"
    raise ValueError("pipeline import pmi must be a bool or string mode")


def _apply_step(asset: Asset, step: PipelineStep, where: Filter | None) -> Asset:
    values = step.values
    if step.op == "heal_brep":
        return asset.heal_brep(_brep_heal_options(values), where=where)
    if step.op == "tessellate":
        return asset.tessellate(_tessellation(values), where=where)
    if step.op == "repair":
        return asset.repair(_repair_options(values), where=where)
    if step.op == "stage":
        return asset.stage(_stage_options(values), where=where)
    if step.op == "merge":
        return asset.merge(_merge_options(values), where=where)
    if step.op == "explode":
        return asset.explode(_explode_options(values), where=where)
    if step.op == "replace":
        return asset.replace(_replace_options(values), where=where)
    if step.op in {"optimize_scene", "scene"}:
        return asset.optimize_scene(_scene_options(values), where=where)
    if step.op == "bake_materials":
        return asset.bake_materials(_bake_material_options(values), where=where)
    if step.op == "decimate":
        return asset.decimate(_decimate_options(values), where=where)
    if step.op == "remove_holes":
        return asset.remove_holes(_remove_holes_options(values), where=where)
    if step.op == "remove_occluded":
        return asset.remove_occluded(_remove_occluded_options(values), where=where)
    if step.op == "run_lod_generators":
        return asset.run_lod_generators(_lod_generator_options(values), where=where)
    if step.op == "optimize":
        return asset.optimize(_optimize_options(values), where=where)
    if step.op == "lods":
        return asset.lods(_lod_options(values), where=where)
    raise ValueError(f"unsupported pipeline step op: {step.op}")


def _tessellation(values: dict[str, object]) -> Tessellation:
    return Tessellation(
        sag=_as_float(values.get("sag", 0.1)),
        angle=_as_float(values.get("angle", 15.0)),
        relative=bool(values.get("relative", True)),
        min_edge_length=_as_optional_float(values.get("min_edge_length")),
        max_edge_length=_as_optional_float(values.get("max_edge_length")),
        preserve_boundaries=bool(values.get("preserve_boundaries", True)),
        curvature_adaptive=bool(values.get("curvature_adaptive", False)),
        avoid_skinny_triangles=bool(values.get("avoid_skinny_triangles", False)),
        quality_report=bool(values.get("quality_report", False)),
        create_normals=bool(values.get("create_normals", True)),
        keep_brep=bool(values.get("keep_brep", False)),
        part_settings=cast(dict[str, dict[str, object]], values.get("part_settings", {})),
    )


def _repair_options(values: dict[str, object]) -> RepairOptions:
    return RepairOptions(
        tolerance=_as_float(values.get("tolerance", 0.0)),
        merge_vertices=bool(values.get("merge_vertices", True)),
        delete_degenerate=bool(values.get("delete_degenerate", True)),
        fix_winding=bool(values.get("fix_winding", True)),
        fill_small_holes=bool(values.get("fill_small_holes", False)),
        area_epsilon=_as_float(values.get("area_epsilon", 1e-12)),
    )


def _brep_heal_options(values: dict[str, object]) -> BrepHealOptions:
    return BrepHealOptions(
        tolerance=_as_float(values.get("tolerance", 0.05)),
        sew_faces=bool(values.get("sew_faces", True)),
        fix_edges=bool(values.get("fix_edges", True)),
        remove_sliver_faces=bool(values.get("remove_sliver_faces", False)),
        max_sliver_area=_as_float(values.get("max_sliver_area", 1e-4)),
        unify_tolerances=bool(values.get("unify_tolerances", True)),
        fail_on_open_shells=bool(values.get("fail_on_open_shells", False)),
    )


def _stage_options(values: dict[str, object]) -> StageOptions:
    return StageOptions(
        materials=cast(Any, values.get("materials", "cad")),
        material_mode=cast(Any, values.get("material_mode", "cad")),
        merge_equivalent_materials=bool(values.get("merge_equivalent_materials", False)),
        normals=bool(values.get("normals", True)),
        normal_mode=cast(Any, _literal(values.get("normal_mode", "smooth"))),
        hard_edge_angle=_as_float(values.get("hard_edge_angle", 30.0)),
        preserve_face_boundaries=bool(values.get("preserve_face_boundaries", False)),
        tangents=bool(values.get("tangents", False)),
        validate_normals=bool(values.get("validate_normals", False)),
        unwrap=UnwrapOptions(
            texel_density=_as_optional_float(values.get("texel_density")),
            padding=_as_int(values.get("padding", values.get("uv_padding", 2))),
            max_stretch=_as_optional_float(values.get("max_stretch")),
        ),
        atlas=AtlasOptions(
            enabled=bool(values.get("atlas", False)),
            max_size=_as_int(values.get("atlas_size", values.get("max_size", 4096))),
        ),
        uv0=cast(Any, values.get("uv0", "box")),
        uv1=cast(Any, values.get("uv1")),
    )


def _merge_options(values: dict[str, object]) -> MergeOptions:
    return MergeOptions(
        mode=cast(Any, _literal(values.get("mode", "all"))),
        keep_parent=bool(values.get("keep_parent", True)),
        metadata=cast(Any, values.get("metadata", "preserve")),
        max_vertices_per_mesh=_as_optional_int(values.get("max_vertices_per_mesh", 65_535)),
        preserve_materials=bool(values.get("preserve_materials", True)),
        hierarchy_level=_as_int(values.get("hierarchy_level", 1)),
        region_size=_as_optional_float(values.get("region_size")),
        merge_strategy=cast(Any, _literal(values.get("merge_strategy", "all"))),
        remove_empty_nodes=bool(values.get("remove_empty_nodes", True)),
    )


def _explode_options(values: dict[str, object]) -> ExplodeOptions:
    return ExplodeOptions(
        mode=cast(Any, _literal(values.get("mode", "connected_components"))),
        metadata=cast(Any, values.get("metadata", "preserve")),
        remove_empty_nodes=bool(values.get("remove_empty_nodes", True)),
    )


def _replace_options(values: dict[str, object]) -> ReplaceOptions:
    return ReplaceOptions(
        mode=cast(Any, _literal(values.get("mode", "bounding_box"))),
        preserve_transform=bool(values.get("preserve_transform", True)),
        metadata=cast(Any, values.get("metadata", "preserve")),
        external_path=cast(str | None, values.get("external_path")),
    )


def _scene_options(values: dict[str, object]) -> SceneOptimizeOptions:
    return SceneOptimizeOptions(
        batch_by_material=bool(values.get("batch_by_material", False)),
        merge_compatible_meshes=bool(values.get("merge_compatible_meshes", False)),
        split_large_meshes=bool(values.get("split_large_meshes", False)),
        max_vertices_per_mesh=_as_optional_int(values.get("max_vertices_per_mesh", 65_535)),
        index_buffer=cast(Any, values.get("index_buffer", "auto")),
        flatten=cast(Any, values.get("flatten", "safe")),
        remove_empty_nodes=bool(values.get("remove_empty_nodes", True)),
        instance_policy=cast(Any, values.get("instance_policy", "auto")),
    )


def _bake_material_options(values: dict[str, object]) -> BakeMaterialOptions:
    return BakeMaterialOptions(
        maps_resolution=_as_int(values.get("maps_resolution", 2048)),
        force_uv_generation=bool(values.get("force_uv_generation", False)),
        uv_channel=_as_int(values.get("uv_channel", 0)),
        padding=_as_int(values.get("padding", 4)),
        bake=cast(Any, tuple(_literal(item) for item in _string_list(values.get("bake", ["base_color"])))),
        merge_output=bool(values.get("merge_output", True)),
    )


def _decimate_options(values: dict[str, object]) -> DecimateOptions:
    return DecimateOptions(
        criterion=cast(Any, values.get("criterion", "target")),
        target_triangles=_as_optional_int(values.get("target_triangles")),
        target_ratio=_as_optional_float(values.get("target_ratio", values.get("ratio", 0.5))),
        surface_tolerance=_as_optional_float(values.get("surface_tolerance")),
        line_tolerance=_as_optional_float(values.get("line_tolerance")),
        normal_tolerance=_as_float(values.get("normal_tolerance", 15.0)),
        uv_tolerance=_as_optional_float(values.get("uv_tolerance")),
        protect_topology=bool(values.get("protect_topology", True)),
        preserve_painted_areas=bool(values.get("preserve_painted_areas", False)),
        budget_scope=cast(Any, values.get("budget_scope", "selection")),
    )


def _remove_holes_options(values: dict[str, object]) -> RemoveHolesOptions:
    return RemoveHolesOptions(
        through=bool(values.get("through", True)),
        blind=bool(values.get("blind", True)),
        surface=bool(values.get("surface", True)),
        max_diameter=_as_optional_float(values.get("max_diameter", 3.0)),
        prefer_brep=bool(values.get("prefer_brep", True)),
    )


def _remove_occluded_options(values: dict[str, object]) -> RemoveOccludedOptions:
    return RemoveOccludedOptions(
        strategy=cast(Any, values.get("strategy", "advanced")),
        level=cast(Any, values.get("level", "triangles")),
        precision=_as_int(values.get("precision", 2048)),
        hemi_evaluation=bool(values.get("hemi_evaluation", False)),
        neighbors_preservation=_as_int(values.get("neighbors_preservation", 1)),
        consider_transparency_opaque=bool(values.get("consider_transparency_opaque", False)),
        preserve_cavities=bool(values.get("preserve_cavities", True)),
        minimum_cavity_volume_m3=_as_float(values.get("minimum_cavity_volume_m3", 0.5)),
    )


def _lod_generator_options(values: dict[str, object]) -> LODGeneratorOptions:
    levels = values.get("levels")
    return LODGeneratorOptions(
        preset=cast(Any, values.get("preset", "desktop")),
        levels=_lod_levels(levels),
        validate=bool(values.get("validate", True)),
        output=cast(Any, values.get("output", "variants")),
        allow_non_monotonic=bool(values.get("allow_non_monotonic", False)),
    )


def _lod_options(values: dict[str, object]) -> LODOptions:
    ratios = values.get("ratios", (0.5, 0.25, 0.1))
    return LODOptions(
        ratios=cast(Any, ratios),
        mode=cast(Any, values.get("mode", "variants")),
        screen_coverage=cast(Any, values.get("screen_coverage")),
        per_part_budget=bool(values.get("per_part_budget", False)),
        drop_tiny_parts=bool(values.get("drop_tiny_parts", False)),
        tiny_part_screen_size=_as_float(values.get("tiny_part_screen_size", 2.0)),
        validate=bool(values.get("validate", False)),
    )


def _optimize_options(values: dict[str, object]) -> OptimizeOptions:
    return OptimizeOptions(
        target_triangles=_as_optional_int(values.get("target_triangles")),
        ratio=_as_optional_float(values.get("ratio")),
        preserve_instances=bool(values.get("preserve_instances", True)),
        simplify=bool(values.get("simplify", True)),
        optimize_buffers=bool(values.get("optimize_buffers", True)),
        preserve_hard_edges=bool(values.get("preserve_hard_edges", False)),
        hard_edge_angle=_as_float(values.get("hard_edge_angle", 30.0)),
        preserve_holes=bool(values.get("preserve_holes", False)),
        preserve_material_boundaries=bool(values.get("preserve_material_boundaries", False)),
        preserve_uv_seams=bool(values.get("preserve_uv_seams", False)),
        preserve_small_parts=bool(values.get("preserve_small_parts", False)),
        small_part_triangle_threshold=_as_int(values.get("small_part_triangle_threshold", 64)),
        preserve_silhouette=bool(values.get("preserve_silhouette", False)),
    )


def _lod_levels(value: object) -> tuple[LODLevel, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("pipeline LOD generator levels must be a list of tables")
    levels: list[LODLevel] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("pipeline LOD generator levels must be tables")
        levels.append(
            LODLevel(
                screen_coverage=_as_float(item.get("screen_coverage")),
                target_ratio=_as_float(item.get("target_ratio")),
            )
        )
    return tuple(levels)


def _string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("pipeline value must be a string or list")


def _as_float(value: object) -> float:
    if isinstance(value, (int, float, str)):
        return float(value)
    raise ValueError(f"pipeline value must be numeric: {value!r}")


def _as_optional_float(value: object) -> float | None:
    return None if value is None else _as_float(value)


def _as_int(value: object) -> int:
    if isinstance(value, (int, str)):
        return int(value)
    raise ValueError(f"pipeline value must be an integer: {value!r}")


def _as_optional_int(value: object) -> int | None:
    return None if value is None else _as_int(value)


def _literal(value: object) -> object:
    return value.replace("-", "_") if isinstance(value, str) else value


def _normalize_key(value: str) -> str:
    return value.replace("-", "_")
