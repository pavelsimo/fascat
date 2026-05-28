from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import tomli

from fascat.asset import Asset
from fascat.filter import Filter
from fascat.options import (
    AabbProjectionOptions,
    AtlasOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    DecimateOptions,
    DeleteDegeneratePolygonsOptions,
    ExplodeOptions,
    LODGeneratorOptions,
    LODLevel,
    LODOptions,
    MergeOptions,
    MergeVerticesOptions,
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

_SUPPORTED_STEP_OPS = frozenset(
    {
        "heal_brep",
        "tessellate",
        "repair",
        "merge_vertices",
        "delete_degenerate_polygons",
        "stage",
        "merge",
        "explode",
        "replace",
        "optimize_scene",
        "scene",
        "bake_materials",
        "decimate",
        "remove_holes",
        "remove_occluded",
        "run_lod_generators",
        "optimize",
        "lods",
    }
)

_TOP_LEVEL_KEYS = frozenset({"import", "export", "filters", "steps"})
_IMPORT_KEYS = frozenset(
    {
        "metadata",
        "product_metadata",
        "properties",
        "layers",
        "validation_properties",
        "pmi",
        "design_variants",
        "existing_meshes",
        "multi_file",
        "delete_free_vertices",
        "delete_lines",
        "source_units",
        "source_meters_per_unit",
        "source_up_axis",
        "source_handedness",
        "target_units",
        "target_meters_per_unit",
        "target_up_axis",
        "target_handedness",
    }
)
_EXPORT_KEYS = frozenset({"metadata", "pmi"})
_FILTER_KEYS = frozenset(
    {
        "name",
        "include",
        "exclude",
        "metadata",
        "path",
        "names",
        "node_name",
        "part_name",
        "part_id",
        "part",
        "material",
        "min_bounds",
        "max_bounds",
        "min_diagonal",
        "max_diagonal",
        "min_triangles",
        "max_triangles",
        "min_vertices",
        "max_vertices",
    }
)
_COMMON_STEP_KEYS = frozenset({"where", "where_not"})
_TESSELLATION_KEYS = frozenset(
    {
        "sag",
        "sag_ratio",
        "angle",
        "relative",
        "min_edge_length",
        "max_edge_length",
        "max_polygon_length",
        "preserve_boundaries",
        "curvature_adaptive",
        "avoid_skinny_triangles",
        "quality_report",
        "free_edge_report",
        "create_normals",
        "keep_brep",
        "reuse_existing_meshes",
        "part_settings",
    }
)
_REPAIR_KEYS = frozenset(
    {
        "tolerance",
        "merge_vertices",
        "delete_degenerate",
        "fix_winding",
        "fill_small_holes",
        "area_epsilon",
    }
)
_MERGE_VERTICES_KEYS = frozenset(
    {
        "tolerance",
        "preserve_normals",
        "preserve_tangents",
        "preserve_uvs",
        "preserve_material_boundaries",
        "delete_degenerate",
        "area_epsilon",
    }
)
_DELETE_DEGENERATE_POLYGONS_KEYS = frozenset({"area_epsilon", "delete_duplicates"})
_BREP_HEAL_KEYS = frozenset(
    {
        "tolerance",
        "sew_faces",
        "fix_edges",
        "remove_sliver_faces",
        "max_sliver_area",
        "unify_tolerances",
        "fail_on_open_shells",
    }
)
_STAGE_KEYS = frozenset(
    {
        "materials",
        "material_mode",
        "merge_equivalent_materials",
        "normals",
        "normal_mode",
        "normal_weighting",
        "hard_edge_angle",
        "preserve_face_boundaries",
        "override_normals",
        "tangents",
        "tangent_uv_channel",
        "override_tangents",
        "validate_normals",
        "texel_density",
        "padding",
        "uv_padding",
        "max_stretch",
        "unwrap_method",
        "unwrap_iterations",
        "unwrap_tolerance",
        "sharp_to_seam",
        "forbid_overlapping",
        "atlas",
        "atlas_size",
        "max_size",
        "uv_aabb_scope",
        "uv3d_size",
        "uv_override_existing",
        "uv0",
        "uv1",
        "normalize_uvs",
    }
)
_MERGE_KEYS = frozenset(
    {
        "mode",
        "keep_parent",
        "metadata",
        "max_vertices_per_mesh",
        "preserve_materials",
        "hierarchy_level",
        "region_size",
        "merge_strategy",
        "remove_empty_nodes",
    }
)
_EXPLODE_KEYS = frozenset({"mode", "metadata", "remove_empty_nodes"})
_REPLACE_KEYS = frozenset({"mode", "preserve_transform", "metadata", "external_path"})
_SCENE_KEYS = frozenset(
    {
        "batch_by_material",
        "merge_compatible_meshes",
        "split_large_meshes",
        "max_vertices_per_mesh",
        "index_buffer",
        "flatten",
        "remove_empty_nodes",
        "instance_policy",
        "instance_similarity_tolerance",
    }
)
_BAKE_MATERIAL_KEYS = frozenset(
    {"maps_resolution", "force_uv_generation", "uv_channel", "padding", "bake", "merge_output"}
)
_DECIMATE_KEYS = frozenset(
    {
        "criterion",
        "target_triangles",
        "target_ratio",
        "ratio",
        "surface_tolerance",
        "line_tolerance",
        "normal_tolerance",
        "uv_tolerance",
        "iterative_threshold",
        "protect_topology",
        "preserve_painted_areas",
        "budget_scope",
        "uv_importance",
        "cleanup_attributes",
    }
)
_REMOVE_HOLES_KEYS = frozenset({"through", "blind", "surface", "max_diameter", "prefer_brep"})
_REMOVE_OCCLUDED_KEYS = frozenset(
    {
        "strategy",
        "level",
        "precision",
        "hemi_evaluation",
        "neighbors_preservation",
        "consider_transparency_opaque",
        "preserve_cavities",
        "minimum_cavity_volume_m3",
    }
)
_LOD_GENERATOR_KEYS = frozenset({"preset", "levels", "validate", "output", "allow_non_monotonic"})
_OPTIMIZE_KEYS = frozenset(
    {
        "target_triangles",
        "ratio",
        "preserve_instances",
        "simplify",
        "optimize_buffers",
        "preserve_hard_edges",
        "hard_edge_angle",
        "preserve_holes",
        "preserve_material_boundaries",
        "preserve_uv_seams",
        "preserve_small_parts",
        "small_part_triangle_threshold",
        "preserve_silhouette",
    }
)
_LOD_KEYS = frozenset(
    {
        "ratios",
        "mode",
        "screen_coverage",
        "per_part_budget",
        "drop_tiny_parts",
        "tiny_part_screen_size",
        "validate",
    }
)
_STEP_OPTION_KEYS = {
    "heal_brep": _BREP_HEAL_KEYS,
    "tessellate": _TESSELLATION_KEYS,
    "repair": _REPAIR_KEYS,
    "merge_vertices": _MERGE_VERTICES_KEYS,
    "delete_degenerate_polygons": _DELETE_DEGENERATE_POLYGONS_KEYS,
    "stage": _STAGE_KEYS,
    "merge": _MERGE_KEYS,
    "explode": _EXPLODE_KEYS,
    "replace": _REPLACE_KEYS,
    "optimize_scene": _SCENE_KEYS,
    "scene": _SCENE_KEYS,
    "bake_materials": _BAKE_MATERIAL_KEYS,
    "decimate": _DECIMATE_KEYS,
    "remove_holes": _REMOVE_HOLES_KEYS,
    "remove_occluded": _REMOVE_OCCLUDED_KEYS,
    "run_lod_generators": _LOD_GENERATOR_KEYS,
    "optimize": _OPTIMIZE_KEYS,
    "lods": _LOD_KEYS,
}
_METADATA_IMPORT_MODES = frozenset({"none", "summary", "full"})
_PMI_IMPORT_MODES = frozenset({"none", "summary", "metadata", "metadata_and_visuals", "full"})


@dataclass(frozen=True)
class _TomlTableLocation:
    line: int
    key_lines: dict[str, int]


@dataclass(frozen=True)
class _TomlLocation:
    top_keys: dict[str, int]
    import_section: _TomlTableLocation | None
    export_section: _TomlTableLocation | None
    filters: tuple[_TomlTableLocation, ...]
    steps: tuple[_TomlTableLocation, ...]


@dataclass(frozen=True)
class PipelineStep:
    op: str
    values: dict[str, object]

    def __post_init__(self) -> None:
        if not self.op:
            raise ValueError("pipeline step op must not be empty")
        normalized_op = self.op.replace("-", "_")
        if normalized_op not in _SUPPORTED_STEP_OPS:
            raise ValueError(f"unsupported pipeline step op: {normalized_op}")
        object.__setattr__(self, "op", normalized_op)
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
        values, location = _load_toml(path)
        return cls.from_dict(values, _location=location)

    @classmethod
    def from_dict(cls, values: dict[str, object], *, _location: _TomlLocation | None = None) -> PipelineSpec:
        _validate_top_level_keys(values, _location)
        filters = _filters(values.get("filters", []), _location)
        steps = _steps(values.get("steps", []), _location)
        if not steps:
            raise ValueError(
                _with_line("pipeline file must include at least one [[steps]] entry", _steps_line(_location))
            )
        _validate_step_filters(steps, filters, _location)
        return cls(
            filters=filters,
            steps=tuple(steps),
            import_options=_import_options(values.get("import"), _location),
            export_metadata=_export_metadata_options(values.get("export"), _location),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "filters": sorted(self.filters),
            "steps": [step.to_dict() for step in self.steps],
            "import": None if self.import_options is None else self.import_options.to_dict(),
            "export": None if self.export_metadata is None else self.export_metadata.to_dict(),
        }

    def advisories(self) -> list[dict[str, object]]:
        return _pipeline_advisories(self.steps)

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


def _pipeline_advisories(steps: tuple[PipelineStep, ...]) -> list[dict[str, object]]:
    advisories: list[dict[str, object]] = []
    saw_repair = False
    saw_uv0 = False
    saw_uv1 = False
    saw_optimize = False

    def add(index: int, step: PipelineStep, code: str, message: str) -> None:
        advisories.append(
            {
                "level": "warning",
                "code": code,
                "step": index,
                "operation": step.op,
                "message": message,
            }
        )

    for index, step in enumerate(steps, start=1):
        if step.op == "decimate" and not saw_repair:
            add(
                index,
                step,
                "decimate_before_repair",
                "decimation runs before mesh repair; repair should run before simplification",
            )
        if step.op == "stage":
            uv0 = _literal(step.values.get("uv0", "box"))
            uv1 = _literal(step.values.get("uv1", "none"))
            step_has_uv0 = uv0 != "none"
            step_has_uv1 = uv1 not in {None, "none"} and (uv1 != "copy_uv0" or saw_uv0 or step_has_uv0)
            if bool(step.values.get("tangents", False)):
                tangent_uv_channel = _as_int(step.values.get("tangent_uv_channel", 0))
                if tangent_uv_channel == 0 and uv0 == "none" and not saw_uv0:
                    add(
                        index,
                        step,
                        "tangents_without_uv0",
                        "tangents are requested before UV0 is available",
                    )
                elif tangent_uv_channel == 1 and not (saw_uv1 or step_has_uv1):
                    add(
                        index,
                        step,
                        "tangents_without_uv1",
                        "tangents are requested before UV1 is available",
                    )
            saw_uv0 = saw_uv0 or step_has_uv0
            saw_uv1 = saw_uv1 or step_has_uv1
        if step.op == "bake_materials":
            bake_maps = {_literal(item) for item in _string_list(step.values.get("bake", ["base_color"]))}
            bake_uv_channel = _as_int(step.values.get("uv_channel", 0))
            generates_uv1 = bool(step.values.get("force_uv_generation", False)) and bake_uv_channel == 1
            if "ao" in bake_maps and not (saw_uv1 or generates_uv1):
                add(
                    index,
                    step,
                    "ao_bake_without_uv1",
                    "ambient occlusion baking is requested before UV1 is available",
                )
        if step.op in {"run_lod_generators", "lods"} and not saw_optimize:
            add(
                index,
                step,
                "lods_before_optimize",
                "LOD generation runs before LOD0 optimization",
            )
        if step.op == "repair":
            saw_repair = True
        if step.op == "optimize":
            saw_optimize = True

    return advisories


def _load_toml(path: str | Path) -> tuple[dict[str, object], _TomlLocation]:
    text = Path(path).read_text(encoding="utf-8")
    return cast(dict[str, object], tomli.loads(text)), _scan_toml_locations(text)


def _filters(value: object, location: _TomlLocation | None = None) -> dict[str, Filter]:
    if value is None or value == []:
        return {}
    if not isinstance(value, list):
        raise ValueError(
            _with_line("pipeline filters must be declared with [[filters]]", _top_key_line(location, "filters"))
        )
    result: dict[str, Filter] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(_with_line("pipeline filter entries must be tables", _filter_line(location, index)))
        filter_location = _filter_location(location, index)
        _validate_unknown_keys(item, _FILTER_KEYS, "pipeline filter", filter_location)
        filter_id = item.get("name")
        if not isinstance(filter_id, str) or not filter_id:
            raise ValueError(
                _with_line(
                    "pipeline filter entries require a non-empty name", _table_or_key_line(filter_location, "name")
                )
            )
        if filter_id in result:
            raise ValueError(
                _with_line(f"duplicate pipeline filter: {filter_id}", _table_or_key_line(filter_location, "name"))
            )
        try:
            result[filter_id] = _filter_from_values(item, result)
        except ValueError as exc:
            message = str(exc)
            raise ValueError(_with_line(message, _message_line(filter_location, message, item))) from exc
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


def _steps(value: object, location: _TomlLocation | None = None) -> list[PipelineStep]:
    if not isinstance(value, list):
        raise ValueError(_with_line("pipeline steps must be declared with [[steps]]", _top_key_line(location, "steps")))
    steps: list[PipelineStep] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(_with_line("pipeline step entries must be tables", _step_line(location, index)))
        step_location = _step_location(location, index)
        op = item.get("op")
        if not isinstance(op, str):
            raise ValueError(_with_line("pipeline step entries require an op", _table_or_key_line(step_location, "op")))
        normalized_op = _normalize_key(op)
        if normalized_op not in _SUPPORTED_STEP_OPS:
            raise ValueError(
                _with_line(f"unsupported pipeline step op: {normalized_op}", _table_or_key_line(step_location, "op"))
            )
        values = {str(key): value for key, value in item.items() if _normalize_key(str(key)) != "op"}
        normalized_values = {_normalize_key(key): item_value for key, item_value in values.items()}
        _validate_unknown_keys(
            normalized_values,
            _COMMON_STEP_KEYS | _STEP_OPTION_KEYS[normalized_op],
            f"key for {normalized_op} pipeline step",
            step_location,
        )
        step = PipelineStep(op=op, values=values)
        _validate_step_options(step, index, step_location)
        steps.append(step)
    return steps


def _import_options(value: object, location: _TomlLocation | None = None) -> StepReadOptions | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(_with_line("pipeline import settings must be a table", _top_key_line(location, "import")))
    import_location = None if location is None else location.import_section
    _validate_unknown_keys(value, _IMPORT_KEYS, "pipeline import", import_location)
    try:
        metadata_enabled = _metadata_import_enabled(value.get("metadata", True))
        return StepReadOptions(
            metadata=metadata_enabled,
            product_metadata=bool(value.get("product_metadata", metadata_enabled)),
            properties=bool(value.get("properties", metadata_enabled)),
            layers=bool(value.get("layers", metadata_enabled)),
            validation_properties=bool(value.get("validation_properties", metadata_enabled)),
            pmi=_pmi_import_enabled(value.get("pmi", True)),
            design_variants=bool(value.get("design_variants", False)),
            existing_meshes=bool(value.get("existing_meshes", True)),
            multi_file=bool(value.get("multi_file", False)),
            delete_free_vertices=bool(value.get("delete_free_vertices", False)),
            delete_lines=bool(value.get("delete_lines", False)),
            source_units=cast(str | None, value.get("source_units")),
            source_meters_per_unit=_as_optional_float(value.get("source_meters_per_unit")),
            source_up_axis=cast(Any, _literal(value.get("source_up_axis", "Z"))),
            source_handedness=cast(Any, _literal(value.get("source_handedness", "right"))),
            target_units=cast(str | None, value.get("target_units")),
            target_meters_per_unit=_as_optional_float(value.get("target_meters_per_unit")),
            target_up_axis=cast(Any, _optional_literal(value.get("target_up_axis"))),
            target_handedness=cast(Any, _optional_literal(value.get("target_handedness"))),
        )
    except ValueError as exc:
        message = str(exc)
        raise ValueError(_with_line(message, _message_line(import_location, message, value))) from exc


def _export_metadata_options(value: object, location: _TomlLocation | None = None) -> MetadataExportOptions | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(_with_line("pipeline export settings must be a table", _top_key_line(location, "export")))
    export_location = None if location is None else location.export_section
    _validate_unknown_keys(value, _EXPORT_KEYS, "pipeline export", export_location)
    try:
        return MetadataExportOptions(
            mode=cast(Any, _literal(value.get("metadata", "full"))),
            pmi=cast(Any, _literal(value.get("pmi", "metadata"))),
        )
    except ValueError as exc:
        message = str(exc)
        raise ValueError(_with_line(message, _message_line(export_location, message, value))) from exc


def _metadata_import_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        mode = value.replace("-", "_").lower()
        if mode in _METADATA_IMPORT_MODES:
            return mode != "none"
    raise ValueError("pipeline import metadata must be a bool or one of: none, summary, full")


def _pmi_import_enabled(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        mode = value.replace("-", "_").lower()
        if mode in _PMI_IMPORT_MODES:
            return mode != "none"
    raise ValueError(
        "pipeline import pmi must be a bool or one of: none, summary, metadata, metadata_and_visuals, full"
    )


def _validate_top_level_keys(values: dict[str, object], location: _TomlLocation | None) -> None:
    _validate_unknown_keys(
        values, _TOP_LEVEL_KEYS, "top-level pipeline", None, key_lines=None if location is None else location.top_keys
    )


def _validate_step_filters(
    steps: list[PipelineStep],
    filters: dict[str, Filter],
    location: _TomlLocation | None,
) -> None:
    for index, step in enumerate(steps):
        step_location = _step_location(location, index)
        where = step.values.get("where")
        where_not = step.values.get("where_not")
        if where is not None and where_not is not None:
            raise ValueError(
                _with_line(
                    "pipeline step cannot set both where and where_not", _table_or_key_line(step_location, "where_not")
                )
            )
        if where_not is not None:
            _validate_filter_ref(where_not, filters, "where_not", step_location)
        if where is not None:
            _validate_filter_ref(where, filters, "where", step_location)


def _validate_filter_ref(
    value: object,
    filters: dict[str, Filter],
    field: str,
    location: _TomlTableLocation | None,
) -> None:
    line = _table_or_key_line(location, field)
    if not isinstance(value, str):
        raise ValueError(_with_line(f"pipeline step {field} must name a filter", line))
    if value not in filters:
        raise ValueError(_with_line(f"unknown pipeline filter: {value}", line))


def _validate_step_options(
    step: PipelineStep,
    index: int,
    location: _TomlTableLocation | None,
) -> None:
    try:
        if step.op == "heal_brep":
            _brep_heal_options(step.values)
        elif step.op == "tessellate":
            _tessellation(step.values)
        elif step.op == "repair":
            _repair_options(step.values)
        elif step.op == "merge_vertices":
            _merge_vertices_options(step.values)
        elif step.op == "delete_degenerate_polygons":
            _delete_degenerate_polygons_options(step.values)
        elif step.op == "stage":
            _stage_options(step.values)
        elif step.op == "merge":
            _merge_options(step.values)
        elif step.op == "explode":
            _explode_options(step.values)
        elif step.op == "replace":
            if _literal(step.values.get("mode", "bounding_box")) == "proxy_mesh":
                raise ValueError(
                    "replace mode proxy_mesh is not supported in pipeline files; use external_asset with external_path"
                )
            _replace_options(step.values)
        elif step.op in {"optimize_scene", "scene"}:
            _scene_options(step.values)
        elif step.op == "bake_materials":
            _bake_material_options(step.values)
        elif step.op == "decimate":
            _decimate_options(step.values)
        elif step.op == "remove_holes":
            _remove_holes_options(step.values)
        elif step.op == "remove_occluded":
            _remove_occluded_options(step.values)
        elif step.op == "run_lod_generators":
            _lod_generator_options(step.values)
        elif step.op == "optimize":
            _optimize_options(step.values)
        elif step.op == "lods":
            _lod_options(step.values)
        else:
            raise ValueError(f"unsupported pipeline step op: {step.op}")
    except ValueError as exc:
        message = f"pipeline step {index + 1} ({step.op}): {exc}"
        raise ValueError(_with_line(message, _message_line(location, str(exc), step.values))) from exc


def _apply_step(asset: Asset, step: PipelineStep, where: Filter | None) -> Asset:
    values = step.values
    if step.op == "heal_brep":
        return asset.heal_brep(_brep_heal_options(values), where=where)
    if step.op == "tessellate":
        return asset.tessellate(_tessellation(values), where=where)
    if step.op == "repair":
        return asset.repair(_repair_options(values), where=where)
    if step.op == "merge_vertices":
        return asset.merge_vertices(_merge_vertices_options(values), where=where)
    if step.op == "delete_degenerate_polygons":
        return asset.delete_degenerate_polygons(_delete_degenerate_polygons_options(values), where=where)
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
        sag_ratio=_as_optional_float(values.get("sag_ratio")),
        angle=_as_float(values.get("angle", 15.0)),
        relative=bool(values.get("relative", True)),
        min_edge_length=_as_optional_float(values.get("min_edge_length")),
        max_edge_length=_as_optional_float(values.get("max_edge_length")),
        max_polygon_length=_as_optional_float(values.get("max_polygon_length")),
        preserve_boundaries=bool(values.get("preserve_boundaries", True)),
        curvature_adaptive=bool(values.get("curvature_adaptive", False)),
        avoid_skinny_triangles=bool(values.get("avoid_skinny_triangles", False)),
        quality_report=bool(values.get("quality_report", False)),
        free_edge_report=bool(values.get("free_edge_report", False)),
        create_normals=bool(values.get("create_normals", True)),
        keep_brep=bool(values.get("keep_brep", False)),
        reuse_existing_meshes=bool(values.get("reuse_existing_meshes", True)),
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


def _merge_vertices_options(values: dict[str, object]) -> MergeVerticesOptions:
    return MergeVerticesOptions(
        tolerance=_as_float(values.get("tolerance", 0.0)),
        preserve_normals=bool(values.get("preserve_normals", True)),
        preserve_tangents=bool(values.get("preserve_tangents", True)),
        preserve_uvs=bool(values.get("preserve_uvs", True)),
        preserve_material_boundaries=bool(values.get("preserve_material_boundaries", True)),
        delete_degenerate=bool(values.get("delete_degenerate", True)),
        area_epsilon=_as_float(values.get("area_epsilon", 1e-12)),
    )


def _delete_degenerate_polygons_options(values: dict[str, object]) -> DeleteDegeneratePolygonsOptions:
    return DeleteDegeneratePolygonsOptions(
        area_epsilon=_as_float(values.get("area_epsilon", 1e-12)),
        delete_duplicates=bool(values.get("delete_duplicates", True)),
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
        normal_weighting=cast(Any, _literal(values.get("normal_weighting", "angle"))),
        hard_edge_angle=_as_float(values.get("hard_edge_angle", 30.0)),
        preserve_face_boundaries=bool(values.get("preserve_face_boundaries", False)),
        override_normals=bool(values.get("override_normals", True)),
        tangents=bool(values.get("tangents", False)),
        tangent_uv_channel=_as_int(values.get("tangent_uv_channel", 0)),
        override_tangents=bool(values.get("override_tangents", False)),
        validate_normals=bool(values.get("validate_normals", False)),
        unwrap=UnwrapOptions(
            texel_density=_as_optional_float(values.get("texel_density")),
            padding=_as_int(values.get("padding", values.get("uv_padding", 2))),
            max_stretch=_as_optional_float(values.get("max_stretch")),
            method=cast(Any, _literal(values.get("unwrap_method", "default"))),
            iterations=_as_optional_int(values.get("unwrap_iterations")),
            tolerance=_as_optional_float(values.get("unwrap_tolerance")),
            sharp_to_seam=bool(values.get("sharp_to_seam", False)),
            forbid_overlapping=bool(values.get("forbid_overlapping", False)),
        ),
        atlas=AtlasOptions(
            enabled=bool(values.get("atlas", False)),
            max_size=_as_int(values.get("atlas_size", values.get("max_size", 4096))),
        ),
        aabb_projection=AabbProjectionOptions(
            scope=cast(Any, _literal(values.get("uv_aabb_scope", "local"))),
            uv3d_size=_as_optional_float(values.get("uv3d_size")),
            override_existing=bool(values.get("uv_override_existing", True)),
        ),
        uv0=cast(Any, _literal(values.get("uv0", "box"))),
        uv1=cast(Any, _literal(values.get("uv1"))),
        normalize_uvs=_int_list(values.get("normalize_uvs", [])),
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
        instance_similarity_tolerance=_as_float(values.get("instance_similarity_tolerance", 0.0)),
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
        iterative_threshold=_as_int(values.get("iterative_threshold", 1_000_000)),
        protect_topology=bool(values.get("protect_topology", True)),
        preserve_painted_areas=bool(values.get("preserve_painted_areas", False)),
        budget_scope=cast(Any, values.get("budget_scope", "selection")),
        uv_importance=cast(Any, _literal(values.get("uv_importance", "preserve_islands"))),
        cleanup_attributes=cast(
            Any,
            tuple(_literal(item) for item in _string_list(values.get("cleanup_attributes", []))),
        ),
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


def _int_list(value: object) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("pipeline value must be a string or list")
    return tuple(dict.fromkeys(_as_int(item) for item in values))


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


def _optional_literal(value: object) -> object | None:
    return None if value is None else _literal(value)


def _normalize_key(value: str) -> str:
    return value.replace("-", "_")


def _scan_toml_locations(text: str) -> _TomlLocation:
    top_keys: dict[str, int] = {}
    filter_locations: list[_TomlTableLocation] = []
    step_locations: list[_TomlTableLocation] = []
    import_location: _TomlTableLocation | None = None
    export_location: _TomlTableLocation | None = None
    current: _TomlTableLocation | None = None

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        header = _toml_header(stripped)
        if header is not None:
            kind, name = header
            normalized = _normalize_key(name)
            if kind == "array" and normalized == "filters":
                current = _TomlTableLocation(line=line_number, key_lines={})
                filter_locations.append(current)
            elif kind == "array" and normalized == "steps":
                current = _TomlTableLocation(line=line_number, key_lines={})
                step_locations.append(current)
            elif kind == "table" and normalized == "import":
                current = _TomlTableLocation(line=line_number, key_lines={})
                import_location = current
            elif kind == "table" and normalized == "export":
                current = _TomlTableLocation(line=line_number, key_lines={})
                export_location = current
            else:
                current = None
                top_key = _normalize_key(name.split(".", 1)[0])
                if top_key not in _TOP_LEVEL_KEYS:
                    top_keys.setdefault(top_key, line_number)
            continue

        key = _toml_key(raw_line)
        if key is None:
            continue
        normalized_key = _normalize_key(key)
        if current is None:
            top_keys.setdefault(normalized_key, line_number)
        else:
            current.key_lines.setdefault(normalized_key, line_number)

    return _TomlLocation(
        top_keys=top_keys,
        import_section=import_location,
        export_section=export_location,
        filters=tuple(filter_locations),
        steps=tuple(step_locations),
    )


def _toml_header(stripped_line: str) -> tuple[str, str] | None:
    if stripped_line.startswith("[["):
        end = stripped_line.find("]]")
        if end != -1:
            return "array", stripped_line[2:end].strip()
    if stripped_line.startswith("["):
        end = stripped_line.find("]")
        if end != -1:
            return "table", stripped_line[1:end].strip()
    return None


def _toml_key(line: str) -> str | None:
    stripped = line.lstrip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key = stripped.split("=", 1)[0].strip()
    if not key or key[0] in {'"', "'"}:
        return None
    return key


def _validate_unknown_keys(
    values: Mapping[Any, Any],
    allowed: frozenset[str],
    label: str,
    location: _TomlTableLocation | None,
    *,
    key_lines: dict[str, int] | None = None,
) -> None:
    normalized_to_source = {_normalize_key(str(key)): str(key) for key in values}
    unknown = sorted(set(normalized_to_source) - allowed)
    if not unknown:
        return
    key = unknown[0]
    line = None
    if location is not None:
        line = location.key_lines.get(key, location.line)
    elif key_lines is not None:
        line = key_lines.get(key)
    if label.startswith("key for "):
        message = f"unsupported {label}: {normalized_to_source[key]}"
    else:
        message = f"unsupported {label} key: {normalized_to_source[key]}"
    raise ValueError(_with_line(message, line))


def _with_line(message: str, line: int | None) -> str:
    return message if line is None else f"line {line}: {message}"


def _top_key_line(location: _TomlLocation | None, key: str) -> int | None:
    if location is None:
        return None
    return location.top_keys.get(key)


def _steps_line(location: _TomlLocation | None) -> int | None:
    if location is None or not location.steps:
        return None
    return location.steps[0].line


def _filter_location(location: _TomlLocation | None, index: int) -> _TomlTableLocation | None:
    if location is None or index >= len(location.filters):
        return None
    return location.filters[index]


def _filter_line(location: _TomlLocation | None, index: int) -> int | None:
    filter_location = _filter_location(location, index)
    return None if filter_location is None else filter_location.line


def _step_location(location: _TomlLocation | None, index: int) -> _TomlTableLocation | None:
    if location is None or index >= len(location.steps):
        return None
    return location.steps[index]


def _step_line(location: _TomlLocation | None, index: int) -> int | None:
    step_location = _step_location(location, index)
    return None if step_location is None else step_location.line


def _table_or_key_line(location: _TomlTableLocation | None, key: str) -> int | None:
    if location is None:
        return None
    return location.key_lines.get(key, location.line)


def _message_line(
    location: _TomlTableLocation | None,
    message: str,
    values: Mapping[Any, Any],
) -> int | None:
    if location is None:
        return None
    normalized_message = message.replace("-", "_")
    for key in sorted((_normalize_key(str(key)) for key in values), key=len, reverse=True):
        if key in normalized_message:
            return location.key_lines.get(key, location.line)
    return location.line
