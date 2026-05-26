from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

UVMode = Literal["none", "box", "unwrap", "lightmap"]
NormalMode = Literal["none", "smooth", "hard_edges", "flat"]
MaterialMode = Literal["cad", "display", "none"]
MaterialPipelineMode = Literal["cad", "pbr"]
LODMode = Literal["variants"]
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
IndexBufferMode = Literal["auto", "uint16", "uint32"]
FlattenMode = Literal["none", "safe", "all"]
InstancePolicy = Literal["auto", "preserve", "expand"]

_TESSELLATION_PART_SETTING_KEYS = {
    "sag",
    "angle",
    "relative",
    "min_edge_length",
    "max_edge_length",
    "preserve_boundaries",
    "curvature_adaptive",
    "avoid_skinny_triangles",
    "quality_report",
    "create_normals",
    "keep_brep",
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
    angle: float = 15.0
    relative: bool = True
    min_edge_length: float | None = None
    max_edge_length: float | None = None
    preserve_boundaries: bool = True
    curvature_adaptive: bool = False
    avoid_skinny_triangles: bool = False
    quality_report: bool = False
    create_normals: bool = True
    keep_brep: bool = False
    part_settings: dict[str, dict[str, object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sag <= 0.0:
            raise ValueError("tessellation sag must be greater than 0")
        if self.angle <= 0.0 or self.angle > 180.0:
            raise ValueError("tessellation angle must be greater than 0 and no more than 180")
        if self.min_edge_length is not None and self.min_edge_length <= 0.0:
            raise ValueError("min_edge_length must be greater than 0 when set")
        if self.max_edge_length is not None and self.max_edge_length <= 0.0:
            raise ValueError("max_edge_length must be greater than 0 when set")
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

    def __post_init__(self) -> None:
        if self.texel_density is not None and self.texel_density <= 0.0:
            raise ValueError("texel_density must be greater than 0 when set")
        if self.padding < 0:
            raise ValueError("padding must be greater than or equal to 0")
        if self.max_stretch is not None and self.max_stretch < 0.0:
            raise ValueError("max_stretch must be greater than or equal to 0 when set")

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
    validate_normals: bool = False
    unwrap: UnwrapOptions = field(default_factory=UnwrapOptions)
    atlas: AtlasOptions = field(default_factory=AtlasOptions)
    uv0: UVMode | None = "box"
    uv1: UVMode | None = None

    def __post_init__(self) -> None:
        if self.uv0 is None:
            object.__setattr__(self, "uv0", "none")
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
        if self.uv0 not in {"none", "box", "unwrap", "lightmap"}:
            raise ValueError("uv0 must be one of: none, box, unwrap, lightmap")
        if self.uv1 not in {None, "none", "box", "unwrap", "lightmap"}:
            raise ValueError("uv1 must be one of: none, box, unwrap, lightmap")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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

    def __post_init__(self) -> None:
        ratios = tuple(float(ratio) for ratio in self.ratios)
        object.__setattr__(self, "ratios", ratios)
        if not ratios:
            raise ValueError("LOD ratios must include at least one value")
        if any(ratio <= 0.0 or ratio >= 1.0 for ratio in ratios):
            raise ValueError("LOD ratios must be greater than 0 and less than 1")
        if ratios != tuple(sorted(ratios, reverse=True)):
            raise ValueError("LOD ratios must be sorted from highest to lowest detail")
        if self.mode != "variants":
            raise ValueError("only variant-based LODs are supported")

    def to_dict(self) -> dict[str, object]:
        return {"ratios": list(self.ratios), "mode": self.mode}


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
class ConversionProfile:
    name: str
    tessellation: Tessellation | None
    repair: RepairOptions
    stage: StageOptions
    optimize: OptimizeOptions | None
    lods: LODOptions | None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "tessellation": self.tessellation.to_dict() if self.tessellation else None,
            "repair": self.repair.to_dict(),
            "stage": self.stage.to_dict(),
            "optimize": self.optimize.to_dict() if self.optimize else None,
            "lods": self.lods.to_dict() if self.lods else None,
        }
