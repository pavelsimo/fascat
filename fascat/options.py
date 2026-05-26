from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

UVMode = Literal["none", "box", "unwrap"]
LODMode = Literal["variants"]


@dataclass(frozen=True)
class Tessellation:
    sag: float = 0.1
    angle: float = 15.0
    relative: bool = True
    max_edge_length: float | None = None
    create_normals: bool = True
    keep_brep: bool = False

    def __post_init__(self) -> None:
        if self.sag <= 0.0:
            raise ValueError("tessellation sag must be greater than 0")
        if self.angle <= 0.0 or self.angle > 180.0:
            raise ValueError("tessellation angle must be greater than 0 and no more than 180")
        if self.max_edge_length is not None and self.max_edge_length <= 0.0:
            raise ValueError("max_edge_length must be greater than 0 when set")

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
class StageOptions:
    materials: Literal["cad", "display", "none"] = "cad"
    normals: bool = True
    uv0: UVMode = "box"
    uv1: UVMode | None = None

    def __post_init__(self) -> None:
        if self.materials not in {"cad", "display", "none"}:
            raise ValueError("materials must be one of: cad, display, none")
        if self.uv0 not in {"none", "box", "unwrap"}:
            raise ValueError("uv0 must be one of: none, box, unwrap")
        if self.uv1 not in {None, "none", "box", "unwrap"}:
            raise ValueError("uv1 must be one of: none, box, unwrap")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OptimizeOptions:
    target_triangles: int | None = None
    ratio: float | None = None
    preserve_instances: bool = True
    simplify: bool = True
    optimize_buffers: bool = True

    def __post_init__(self) -> None:
        if self.target_triangles is not None and self.target_triangles <= 0:
            raise ValueError("target_triangles must be greater than 0 when set")
        if self.ratio is not None and (self.ratio <= 0.0 or self.ratio >= 1.0):
            raise ValueError("ratio must be greater than 0 and less than 1 when set")

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
