from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

UV0Mode = Literal["none", "box", "unwrap", "lightmap"]
UV1Mode = Literal["none", "box", "unwrap", "lightmap", "copy_uv0"]
UVMode = UV0Mode
UnwrapMethod = Literal["default", "conformal", "isometric"]
NormalMode = Literal["none", "smooth", "hard_edges", "flat"]
MaterialMode = Literal["cad", "display", "none"]
MaterialPipelineMode = Literal["cad", "pbr"]
LODMode = Literal["variants", "extras", "separate"]
MergeMode = Literal[
    "all",
    "by_material",
    "by_node_name",
    "by_part_name",
    "hierarchy_level",
    "parent_children",
    "final_level",
    "regions",
]
MergeMetadataPolicy = Literal["preserve", "combine", "summarize", "drop"]
MergeStrategy = Literal["all", "by_material"]
ExplodeMode = Literal["by_material", "connected_components"]
ReplaceMode = Literal["bounding_box", "proxy_mesh", "external_asset"]
IndexBufferMode = Literal["auto", "uint16", "uint32"]
FlattenMode = Literal["none", "safe", "all"]
InstancePolicy = Literal["auto", "preserve", "expand"]
BakeMaterialMap = Literal["base_color", "opacity", "normal", "roughness", "metallic", "ao", "emissive"]
DecimateCriterion = Literal["target", "quality"]
BudgetScope = Literal["part", "selection"]
DecimateUVImportance = Literal["preserve_islands", "preserve_seams", "ignore"]
HoleType = Literal["through", "blind", "surface"]
OcclusionStrategy = Literal["conservative", "exterior", "advanced"]
OcclusionLevel = Literal["parts", "submeshes", "triangles"]
LODPreset = Literal["desktop", "web", "mobile", "vr"]
LODOutput = Literal["variants", "extras", "separate"]
TextureCompression = Literal["ktx2", "basisu"]
UsdPackageMode = Literal["default", "usdz"]
MetadataExportMode = Literal["none", "summary", "full"]
PmiExportMode = Literal["none", "summary", "metadata", "metadata_and_visuals", "full"]
Axis = Literal["Y", "Z"]
Handedness = Literal["right", "left"]

_BAKE_MAPS = {"base_color", "opacity", "normal", "roughness", "metallic", "ao", "emissive"}
_HOLE_TYPES = {"through", "blind", "surface"}

_TESSELLATION_PART_SETTING_KEYS = {
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
}


def _validate_part_settings(part_settings: dict[str, dict[str, object]]) -> None:
    for selector, overrides in part_settings.items():
        if not selector:
            raise ValueError("part_settings keys must not be empty")
        if not isinstance(overrides, dict):
            raise ValueError("part_settings values must be dictionaries")
        unknown = set(overrides) - _TESSELLATION_PART_SETTING_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported part_settings keys: {names}")


@dataclass(frozen=True)
class Tessellation:
    sag: float = 0.1
    sag_ratio: float | None = None
    angle: float = 15.0
    relative: bool = True
    min_edge_length: float | None = None
    max_edge_length: float | None = None
    max_polygon_length: float | None = None
    preserve_boundaries: bool = True
    curvature_adaptive: bool = False
    avoid_skinny_triangles: bool = False
    quality_report: bool = False
    free_edge_report: bool = False
    create_normals: bool = True
    keep_brep: bool = False
    reuse_existing_meshes: bool = True
    part_settings: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sag <= 0.0:
            raise ValueError("tessellation sag must be greater than 0")
        if self.sag_ratio is not None and self.sag_ratio <= 0.0:
            raise ValueError("tessellation sag_ratio must be greater than 0 when set")
        if self.angle <= 0.0 or self.angle > 180.0:
            raise ValueError("tessellation angle must be greater than 0 and no more than 180")
        if self.min_edge_length is not None and self.min_edge_length <= 0.0:
            raise ValueError("min_edge_length must be greater than 0 when set")
        if self.max_edge_length is not None and self.max_edge_length <= 0.0:
            raise ValueError("max_edge_length must be greater than 0 when set")
        if self.max_polygon_length is not None and self.max_polygon_length <= 0.0:
            raise ValueError("max_polygon_length must be greater than 0 when set")
        if (
            self.min_edge_length is not None
            and self.max_edge_length is not None
            and self.min_edge_length > self.max_edge_length
        ):
            raise ValueError("min_edge_length must be less than or equal to max_edge_length")
        _validate_part_settings(self.part_settings)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RepairOptions:
    tolerance: float = 0.0
    merge_vertices: bool = True
    delete_degenerate: bool = True
    fix_winding: bool = True
    fill_small_holes: bool = False
    area_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.tolerance < 0.0:
            raise ValueError("repair tolerance must be greater than or equal to 0")
        if self.area_epsilon < 0.0:
            raise ValueError("area_epsilon must be greater than or equal to 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StepReadOptions:
    metadata: bool = True
    product_metadata: bool = True
    properties: bool = True
    layers: bool = True
    validation_properties: bool = True
    pmi: bool = True
    design_variants: bool = False
    existing_meshes: bool = True
    multi_file: bool = False
    delete_free_vertices: bool = False
    delete_lines: bool = False
    source_units: str | None = None
    source_meters_per_unit: float | None = None
    source_up_axis: Axis = "Z"
    source_handedness: Handedness = "right"
    target_units: str | None = None
    target_meters_per_unit: float | None = None
    target_up_axis: Axis | None = None
    target_handedness: Handedness | None = None

    def __post_init__(self) -> None:
        if self.source_meters_per_unit is not None and self.source_meters_per_unit <= 0.0:
            raise ValueError("source_meters_per_unit must be greater than 0 when set")
        if self.target_meters_per_unit is not None and self.target_meters_per_unit <= 0.0:
            raise ValueError("target_meters_per_unit must be greater than 0 when set")
        if self.source_up_axis not in {"Y", "Z"}:
            raise ValueError("source_up_axis must be one of: Y, Z")
        if self.target_up_axis not in {None, "Y", "Z"}:
            raise ValueError("target_up_axis must be one of: Y, Z")
        if self.source_handedness not in {"right", "left"}:
            raise ValueError("source_handedness must be one of: right, left")
        if self.target_handedness not in {None, "right", "left"}:
            raise ValueError("target_handedness must be one of: right, left")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BrepHealOptions:
    tolerance: float = 0.05
    sew_faces: bool = True
    fix_edges: bool = True
    remove_sliver_faces: bool = False
    max_sliver_area: float = 1e-4
    unify_tolerances: bool = True
    fail_on_open_shells: bool = False

    def __post_init__(self) -> None:
        if self.tolerance <= 0.0:
            raise ValueError("heal tolerance must be greater than 0")
        if self.max_sliver_area < 0.0:
            raise ValueError("max_sliver_area must be greater than or equal to 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UnwrapOptions:
    texel_density: float | None = None
    padding: int = 2
    max_stretch: float | None = None
    method: UnwrapMethod = "default"
    iterations: int | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        if self.texel_density is not None and self.texel_density <= 0.0:
            raise ValueError("texel_density must be greater than 0 when set")
        if self.padding < 0:
            raise ValueError("padding must be greater than or equal to 0")
        if self.max_stretch is not None and self.max_stretch < 0.0:
            raise ValueError("max_stretch must be greater than or equal to 0 when set")
        if self.method not in {"default", "conformal", "isometric"}:
            raise ValueError("unwrap method must be one of: default, conformal, isometric")
        if self.iterations is not None and self.iterations <= 0:
            raise ValueError("unwrap iterations must be greater than 0 when set")
        if self.tolerance is not None and self.tolerance < 0.0:
            raise ValueError("unwrap tolerance must be greater than or equal to 0 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class AtlasOptions:
    enabled: bool = False
    max_size: int = 4096

    def __post_init__(self) -> None:
        if self.max_size <= 0:
            raise ValueError("atlas max_size must be greater than 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StageOptions:
    materials: MaterialMode = "cad"
    material_mode: MaterialPipelineMode = "cad"
    merge_equivalent_materials: bool = False
    normals: bool = True
    normal_mode: NormalMode = "smooth"
    hard_edge_angle: float = 30.0
    preserve_face_boundaries: bool = False
    tangents: bool = False
    tangent_uv_channel: int = 0
    override_tangents: bool = False
    validate_normals: bool = False
    unwrap: UnwrapOptions = field(default_factory=UnwrapOptions)
    atlas: AtlasOptions = field(default_factory=AtlasOptions)
    uv0: UV0Mode | None = "box"
    uv1: UV1Mode | None = None
    normalize_uvs: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.uv0 is None:
            object.__setattr__(self, "uv0", "none")
        if isinstance(self.normalize_uvs, str):
            raise ValueError("normalize_uvs must be a sequence of UV channel indices")
        normalize_uvs = tuple(dict.fromkeys(int(channel) for channel in self.normalize_uvs))
        object.__setattr__(self, "normalize_uvs", normalize_uvs)
        if self.normal_mode == "none":
            object.__setattr__(self, "normals", False)
        if self.materials not in {"cad", "display", "none"}:
            raise ValueError("materials must be one of: cad, display, none")
        if self.material_mode not in {"cad", "pbr"}:
            raise ValueError("material_mode must be one of: cad, pbr")
        if self.normal_mode not in {"none", "smooth", "hard_edges", "flat"}:
            raise ValueError("normal_mode must be one of: none, smooth, hard_edges, flat")
        if self.hard_edge_angle <= 0.0 or self.hard_edge_angle > 180.0:
            raise ValueError("hard_edge_angle must be greater than 0 and no more than 180")
        if self.tangent_uv_channel < 0:
            raise ValueError("tangent_uv_channel must be greater than or equal to 0")
        if self.uv0 not in {"none", "box", "unwrap", "lightmap"}:
            raise ValueError("uv0 must be one of: none, box, unwrap, lightmap")
        if self.uv1 not in {None, "none", "box", "unwrap", "lightmap", "copy_uv0"}:
            raise ValueError("uv1 must be one of: none, box, unwrap, lightmap, copy_uv0")
        if any(channel < 0 for channel in self.normalize_uvs):
            raise ValueError("normalize_uvs values must be greater than or equal to 0")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "normalize_uvs": list(self.normalize_uvs)}


@dataclass(frozen=True)
class OptimizeOptions:
    target_triangles: int | None = None
    ratio: float | None = None
    preserve_instances: bool = True
    simplify: bool = True
    optimize_buffers: bool = True
    preserve_hard_edges: bool = False
    hard_edge_angle: float = 30.0
    preserve_holes: bool = False
    preserve_material_boundaries: bool = False
    preserve_uv_seams: bool = False
    preserve_small_parts: bool = False
    small_part_triangle_threshold: int = 64
    preserve_silhouette: bool = False

    def __post_init__(self) -> None:
        if self.target_triangles is not None and self.target_triangles <= 0:
            raise ValueError("target_triangles must be greater than 0 when set")
        if self.ratio is not None and (self.ratio <= 0.0 or self.ratio >= 1.0):
            raise ValueError("ratio must be greater than 0 and less than 1 when set")
        if self.hard_edge_angle <= 0.0 or self.hard_edge_angle > 180.0:
            raise ValueError("hard_edge_angle must be greater than 0 and no more than 180")
        if self.small_part_triangle_threshold < 0:
            raise ValueError("small_part_triangle_threshold must be greater than or equal to 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LODOptions:
    ratios: list[float] | tuple[float, ...] = (0.5, 0.25, 0.1)
    mode: LODMode = "variants"
    screen_coverage: list[float] | tuple[float, ...] | None = None
    per_part_budget: bool = False
    drop_tiny_parts: bool = False
    tiny_part_screen_size: float = 2.0
    validate: bool = False

    def __post_init__(self) -> None:
        ratios = tuple(float(ratio) for ratio in self.ratios)
        object.__setattr__(self, "ratios", ratios)
        if not ratios:
            raise ValueError("LOD ratios must include at least one value")
        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("LOD ratios must be greater than 0 and less than 1")
        if ratios != tuple(sorted(ratios, reverse=True)):
            raise ValueError("LOD ratios must be sorted from highest to lowest detail")
        if self.mode not in {"variants", "extras", "separate"}:
            raise ValueError("LOD mode must be one of: variants, extras, separate")
        if self.screen_coverage is not None:
            screen_coverage = tuple(float(value) for value in self.screen_coverage)
            object.__setattr__(self, "screen_coverage", screen_coverage)
            if len(screen_coverage) != len(ratios):
                raise ValueError("screen_coverage must contain one value per LOD ratio")
            if any(value <= 0.0 or value > 1.0 for value in screen_coverage):
                raise ValueError("screen_coverage values must be greater than 0 and no more than 1")
            if screen_coverage != tuple(sorted(screen_coverage, reverse=True)):
                raise ValueError("screen_coverage values must be sorted from highest to lowest")
        if self.tiny_part_screen_size < 0.0:
            raise ValueError("tiny_part_screen_size must be greater than or equal to 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "ratios": list(self.ratios),
            "mode": self.mode,
            "screen_coverage": None if self.screen_coverage is None else list(self.screen_coverage),
            "per_part_budget": self.per_part_budget,
            "drop_tiny_parts": self.drop_tiny_parts,
            "tiny_part_screen_size": self.tiny_part_screen_size,
            "validate": self.validate,
        }


@dataclass(frozen=True)
class MergeOptions:
    mode: MergeMode = "all"
    keep_parent: bool = True
    metadata: MergeMetadataPolicy = "preserve"
    max_vertices_per_mesh: int | None = 65_535
    preserve_materials: bool = True
    hierarchy_level: int = 1
    region_size: float | None = None
    merge_strategy: MergeStrategy = "all"
    remove_empty_nodes: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {
            "all",
            "by_material",
            "by_node_name",
            "by_part_name",
            "hierarchy_level",
            "parent_children",
            "final_level",
            "regions",
        }:
            raise ValueError("unsupported merge mode")
        if self.metadata not in {"preserve", "combine", "summarize", "drop"}:
            raise ValueError("merge metadata must be one of: preserve, combine, summarize, drop")
        if self.merge_strategy not in {"all", "by_material"}:
            raise ValueError("merge_strategy must be one of: all, by_material")
        if self.max_vertices_per_mesh is not None and self.max_vertices_per_mesh <= 0:
            raise ValueError("max_vertices_per_mesh must be greater than 0 when set")
        if self.hierarchy_level < 0:
            raise ValueError("hierarchy_level must be greater than or equal to 0")
        if self.mode == "regions" and (self.region_size is None or self.region_size <= 0.0):
            raise ValueError("region_size must be greater than 0 for regions merge mode")
        if self.region_size is not None and self.region_size <= 0.0:
            raise ValueError("region_size must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExplodeOptions:
    mode: ExplodeMode = "connected_components"
    metadata: MergeMetadataPolicy = "preserve"
    remove_empty_nodes: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"by_material", "connected_components"}:
            raise ValueError("explode mode must be one of: by_material, connected_components")
        if self.metadata not in {"preserve", "combine", "summarize", "drop"}:
            raise ValueError("explode metadata must be one of: preserve, combine, summarize, drop")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplaceOptions:
    mode: ReplaceMode = "bounding_box"
    preserve_transform: bool = True
    metadata: MergeMetadataPolicy = "preserve"
    proxy_mesh: object | None = None
    external_path: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"bounding_box", "proxy_mesh", "external_asset"}:
            raise ValueError("replace mode must be one of: bounding_box, proxy_mesh, external_asset")
        if self.metadata not in {"preserve", "combine", "summarize", "drop"}:
            raise ValueError("replace metadata must be one of: preserve, combine, summarize, drop")
        if self.mode == "proxy_mesh" and self.proxy_mesh is None:
            raise ValueError("proxy_mesh is required for proxy_mesh replacement")
        if self.mode == "external_asset" and not self.external_path:
            raise ValueError("external_path is required for external_asset replacement")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "preserve_transform": self.preserve_transform,
            "metadata": self.metadata,
            "proxy_mesh": self.proxy_mesh is not None,
            "external_path": self.external_path,
        }


@dataclass(frozen=True)
class SceneOptimizeOptions:
    batch_by_material: bool = False
    merge_compatible_meshes: bool = False
    split_large_meshes: bool = False
    max_vertices_per_mesh: int | None = 65_535
    index_buffer: IndexBufferMode = "auto"
    flatten: FlattenMode = "safe"
    remove_empty_nodes: bool = True
    instance_policy: InstancePolicy = "auto"

    def __post_init__(self) -> None:
        if self.max_vertices_per_mesh is not None and self.max_vertices_per_mesh <= 0:
            raise ValueError("max_vertices_per_mesh must be greater than 0 when set")
        if self.split_large_meshes and self.max_vertices_per_mesh is not None and self.max_vertices_per_mesh < 3:
            raise ValueError("max_vertices_per_mesh must be at least 3 when split_large_meshes is true")
        if self.index_buffer not in {"auto", "uint16", "uint32"}:
            raise ValueError("index_buffer must be one of: auto, uint16, uint32")
        if self.flatten not in {"none", "safe", "all"}:
            raise ValueError("flatten must be one of: none, safe, all")
        if self.instance_policy not in {"auto", "preserve", "expand"}:
            raise ValueError("instance_policy must be one of: auto, preserve, expand")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BakeMaterialOptions:
    maps_resolution: int = 2048
    force_uv_generation: bool = False
    uv_channel: int = 0
    padding: int = 4
    bake: tuple[BakeMaterialMap, ...] = ("base_color",)
    merge_output: bool = True

    def __post_init__(self) -> None:
        maps = tuple(str(item).replace("-", "_") for item in self.bake)
        object.__setattr__(self, "bake", maps)
        if self.maps_resolution <= 0:
            raise ValueError("maps_resolution must be greater than 0")
        if self.uv_channel < 0:
            raise ValueError("uv_channel must be greater than or equal to 0")
        if self.padding < 0:
            raise ValueError("padding must be greater than or equal to 0")
        if not maps:
            raise ValueError("bake must include at least one map")
        unknown = set(maps) - _BAKE_MAPS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unsupported bake maps: {names}")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "bake": list(self.bake)}


@dataclass(frozen=True)
class DecimateOptions:
    criterion: DecimateCriterion = "target"
    target_triangles: int | None = None
    target_ratio: float | None = 0.5
    surface_tolerance: float | None = None
    line_tolerance: float | None = None
    normal_tolerance: float = 15.0
    uv_tolerance: float | None = None
    protect_topology: bool = True
    preserve_painted_areas: bool = False
    budget_scope: BudgetScope = "selection"
    uv_importance: DecimateUVImportance = "preserve_islands"

    def __post_init__(self) -> None:
        if self.criterion not in {"target", "quality"}:
            raise ValueError("criterion must be one of: target, quality")
        if self.target_triangles is not None and self.target_triangles <= 0:
            raise ValueError("target_triangles must be greater than 0 when set")
        if self.target_ratio is not None and (self.target_ratio <= 0.0 or self.target_ratio >= 1.0):
            raise ValueError("target_ratio must be greater than 0 and less than 1 when set")
        for name, value in {
            "surface_tolerance": self.surface_tolerance,
            "line_tolerance": self.line_tolerance,
            "uv_tolerance": self.uv_tolerance,
        }.items():
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be greater than or equal to 0 when set")
        if self.normal_tolerance <= 0.0 or self.normal_tolerance > 180.0:
            raise ValueError("normal_tolerance must be greater than 0 and no more than 180")
        if self.budget_scope not in {"part", "selection"}:
            raise ValueError("budget_scope must be one of: part, selection")
        if self.uv_importance not in {"preserve_islands", "preserve_seams", "ignore"}:
            raise ValueError("uv_importance must be one of: preserve_islands, preserve_seams, ignore")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RemoveHolesOptions:
    through: bool = True
    blind: bool = True
    surface: bool = True
    max_diameter: float | None = 3.0
    prefer_brep: bool = True

    def __post_init__(self) -> None:
        if not (self.through or self.blind or self.surface):
            raise ValueError("at least one hole type must be enabled")
        if self.max_diameter is not None and self.max_diameter <= 0.0:
            raise ValueError("max_diameter must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RemoveOccludedOptions:
    strategy: OcclusionStrategy = "advanced"
    level: OcclusionLevel = "triangles"
    precision: int = 2048
    hemi_evaluation: bool = False
    neighbors_preservation: int = 1
    consider_transparency_opaque: bool = False
    preserve_cavities: bool = True
    minimum_cavity_volume_m3: float = 0.5

    def __post_init__(self) -> None:
        if self.strategy not in {"conservative", "exterior", "advanced"}:
            raise ValueError("strategy must be one of: conservative, exterior, advanced")
        if self.level not in {"parts", "submeshes", "triangles"}:
            raise ValueError("level must be one of: parts, submeshes, triangles")
        if self.precision <= 0:
            raise ValueError("precision must be greater than 0")
        if self.neighbors_preservation < 0:
            raise ValueError("neighbors_preservation must be greater than or equal to 0")
        if self.minimum_cavity_volume_m3 < 0.0:
            raise ValueError("minimum_cavity_volume_m3 must be greater than or equal to 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LODLevel:
    screen_coverage: float
    target_ratio: float

    def __post_init__(self) -> None:
        if self.screen_coverage <= 0.0 or self.screen_coverage > 1.0:
            raise ValueError("screen_coverage must be greater than 0 and no more than 1")
        if self.target_ratio <= 0.0 or self.target_ratio >= 1.0:
            raise ValueError("target_ratio must be greater than 0 and less than 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LODGeneratorOptions:
    preset: LODPreset = "desktop"
    levels: tuple[LODLevel, ...] = ()
    validate: bool = True
    output: LODOutput = "variants"
    allow_non_monotonic: bool = False

    def __post_init__(self) -> None:
        if self.preset not in {"desktop", "web", "mobile", "vr"}:
            raise ValueError("preset must be one of: desktop, web, mobile, vr")
        levels = self.levels or _lod_preset_levels(self.preset)
        object.__setattr__(self, "levels", tuple(levels))
        if not self.levels:
            raise ValueError("levels must include at least one LOD level")
        coverages = tuple(level.screen_coverage for level in self.levels)
        ratios = tuple(level.target_ratio for level in self.levels)
        if coverages != tuple(sorted(coverages, reverse=True)):
            raise ValueError("LOD screen coverage values must be sorted from highest to lowest")
        if ratios != tuple(sorted(ratios, reverse=True)):
            raise ValueError("LOD target ratios must be sorted from highest to lowest detail")
        if self.output not in {"variants", "extras", "separate"}:
            raise ValueError("output must be one of: variants, extras, separate")

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "levels": [level.to_dict() for level in self.levels],
            "validate": self.validate,
            "output": self.output,
            "allow_non_monotonic": self.allow_non_monotonic,
        }


def _lod_preset_levels(preset: LODPreset) -> tuple[LODLevel, ...]:
    if preset == "web":
        return (LODLevel(0.45, 0.5), LODLevel(0.15, 0.25))
    if preset == "mobile":
        return (LODLevel(0.4, 0.4), LODLevel(0.15, 0.2), LODLevel(0.05, 0.1))
    if preset == "vr":
        return (LODLevel(0.5, 0.5), LODLevel(0.2, 0.25), LODLevel(0.05, 0.1))
    return (LODLevel(0.5, 0.5), LODLevel(0.25, 0.25), LODLevel(0.1, 0.1))


@dataclass(frozen=True)
class AnalyzeOptions:
    non_manifold_edges: bool = False
    open_boundaries: bool = False
    self_intersections: bool = False
    sliver_triangles: bool = False
    tiny_parts: bool = False
    draw_call_estimate: bool = False
    visual_risk: bool = False
    sliver_aspect_ratio: float = 20.0
    degenerate_area_epsilon: float = 1e-12
    tiny_part_diagonal: float = 1.0
    max_self_intersection_pairs: int = 10_000

    def __post_init__(self) -> None:
        if self.sliver_aspect_ratio <= 1.0:
            raise ValueError("sliver_aspect_ratio must be greater than 1")
        if self.degenerate_area_epsilon < 0.0:
            raise ValueError("degenerate_area_epsilon must be greater than or equal to 0")
        if self.tiny_part_diagonal < 0.0:
            raise ValueError("tiny_part_diagonal must be greater than or equal to 0")
        if self.max_self_intersection_pairs <= 0:
            raise ValueError("max_self_intersection_pairs must be greater than 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MetadataExportOptions:
    mode: MetadataExportMode = "full"
    pmi: PmiExportMode = "metadata"

    def __post_init__(self) -> None:
        pmi = self.pmi.replace("-", "_") if isinstance(self.pmi, str) else self.pmi
        object.__setattr__(self, "pmi", pmi)
        if self.mode not in {"none", "summary", "full"}:
            raise ValueError("metadata export mode must be one of: none, summary, full")
        if self.pmi not in {"none", "summary", "metadata", "metadata_and_visuals", "full"}:
            raise ValueError("PMI export mode must be one of: none, summary, metadata, metadata_and_visuals, full")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GltfExportOptions:
    quantize: bool = False
    meshopt: bool = False
    draco: bool = False
    texture_compression: TextureCompression | None = None
    file_size_budget_mb: float | None = None
    metadata: MetadataExportOptions = field(default_factory=MetadataExportOptions)

    def __post_init__(self) -> None:
        if self.texture_compression not in {None, "ktx2", "basisu"}:
            raise ValueError("texture_compression must be one of: ktx2, basisu")
        if self.texture_compression is not None:
            raise ValueError("texture compression is not supported because no KTX2/Basis encoder backend is integrated")
        if self.draco:
            raise ValueError("draco compression is not supported because no Draco encoder backend is integrated")
        if self.file_size_budget_mb is not None and self.file_size_budget_mb <= 0.0:
            raise ValueError("file_size_budget_mb must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "metadata": self.metadata.to_dict()}


@dataclass(frozen=True)
class UsdExportOptions:
    package: UsdPackageMode = "default"
    file_size_budget_mb: float | None = None
    metadata: MetadataExportOptions = field(default_factory=MetadataExportOptions)

    def __post_init__(self) -> None:
        if self.package not in {"default", "usdz"}:
            raise ValueError("package must be one of: default, usdz")
        if self.file_size_budget_mb is not None and self.file_size_budget_mb <= 0.0:
            raise ValueError("file_size_budget_mb must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "metadata": self.metadata.to_dict()}


@dataclass(frozen=True)
class ObjExportOptions:
    materials: bool = True
    write_mtl: bool = True
    preserve_groups: bool = True
    file_size_budget_mb: float | None = None

    def __post_init__(self) -> None:
        if self.file_size_budget_mb is not None and self.file_size_budget_mb <= 0.0:
            raise ValueError("file_size_budget_mb must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StlExportOptions:
    binary: bool = True
    merge: bool = True
    file_size_budget_mb: float | None = None

    def __post_init__(self) -> None:
        if self.file_size_budget_mb is not None and self.file_size_budget_mb <= 0.0:
            raise ValueError("file_size_budget_mb must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlatformBudget:
    target_fps: int | None = None
    max_triangles: int | None = None
    max_vertices: int | None = None
    max_vertices_per_mesh: int | None = None
    max_texture_resolution: int | None = None
    max_texture_memory_mb: int | None = None
    max_load_time_ms: int | None = None
    max_draw_calls: int | None = None

    def __post_init__(self) -> None:
        if self.target_fps is not None and self.target_fps <= 0:
            raise ValueError("target_fps must be greater than 0 when set")
        if self.max_triangles is not None and self.max_triangles <= 0:
            raise ValueError("max_triangles must be greater than 0 when set")
        if self.max_vertices is not None and self.max_vertices <= 0:
            raise ValueError("max_vertices must be greater than 0 when set")
        if self.max_vertices_per_mesh is not None and self.max_vertices_per_mesh <= 0:
            raise ValueError("max_vertices_per_mesh must be greater than 0 when set")
        if self.max_texture_resolution is not None and self.max_texture_resolution <= 0:
            raise ValueError("max_texture_resolution must be greater than 0 when set")
        if self.max_texture_memory_mb is not None and self.max_texture_memory_mb <= 0:
            raise ValueError("max_texture_memory_mb must be greater than 0 when set")
        if self.max_load_time_ms is not None and self.max_load_time_ms <= 0:
            raise ValueError("max_load_time_ms must be greater than 0 when set")
        if self.max_draw_calls is not None and self.max_draw_calls <= 0:
            raise ValueError("max_draw_calls must be greater than 0 when set")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ConversionProfile:
    name: str
    tessellation: Tessellation | None
    repair: RepairOptions
    stage: StageOptions
    optimize: OptimizeOptions | None
    lods: LODOptions | None
    budget: PlatformBudget | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tessellation": self.tessellation.to_dict() if self.tessellation else None,
            "repair": self.repair.to_dict(),
            "stage": self.stage.to_dict(),
            "optimize": self.optimize.to_dict() if self.optimize else None,
            "lods": self.lods.to_dict() if self.lods else None,
            "budget": self.budget.to_dict() if self.budget else None,
        }
