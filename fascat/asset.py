from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import LODOptions, OptimizeOptions, RepairOptions, StageOptions, Tessellation
from fascat.report import Report, timed_step

Transform = NDArray[np.float64]


def identity_transform() -> Transform:
    return np.eye(4, dtype=np.float64)


@dataclass
class Node:
    id: str
    name: str
    children: list[Node] = field(default_factory=list)
    part_id: str | None = None
    transform: Transform = field(default_factory=identity_transform)
    metadata: dict[str, str] = field(default_factory=dict)

    def copy(self) -> Node:
        return Node(
            id=self.id,
            name=self.name,
            children=[child.copy() for child in self.children],
            part_id=self.part_id,
            transform=self.transform.copy(),
            metadata=dict(self.metadata),
        )

    def walk(self) -> list[Node]:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "part_id": self.part_id,
            "children": [child.to_dict() for child in self.children],
            "metadata": dict(self.metadata),
        }


@dataclass
class Part:
    id: str
    name: str
    source_shape: object | None = None
    mesh: Mesh | None = None
    material_ids: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    fingerprint: str | None = None
    lod_meshes: list[Mesh] = field(default_factory=list)

    def copy(self, *, keep_source: bool = True) -> Part:
        return Part(
            id=self.id,
            name=self.name,
            source_shape=self.source_shape if keep_source else None,
            mesh=None if self.mesh is None else self.mesh.copy(),
            material_ids=list(self.material_ids),
            metadata=dict(self.metadata),
            fingerprint=self.fingerprint,
            lod_meshes=[mesh.copy() for mesh in self.lod_meshes],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "has_source_shape": self.source_shape is not None,
            "mesh": None if self.mesh is None else self.mesh.to_dict(),
            "material_ids": list(self.material_ids),
            "metadata": dict(self.metadata),
            "fingerprint": self.fingerprint,
            "lods": [mesh.to_dict() for mesh in self.lod_meshes],
        }


@dataclass
class Asset:
    root: Node
    parts: dict[str, Part] = field(default_factory=dict)
    materials: dict[str, Material] = field(default_factory=dict)
    units: str = "millimetre"
    meters_per_unit: float = 0.001
    up_axis: Literal["Y", "Z"] = "Z"
    source_path: Path | None = None
    report: Report = field(default_factory=Report)

    @property
    def part_count(self) -> int:
        return len(self.parts)

    @property
    def material_count(self) -> int:
        return len(self.materials)

    @property
    def triangle_count(self) -> int:
        return sum(part.mesh.triangle_count for part in self.parts.values() if part.mesh is not None)

    @property
    def vertex_count(self) -> int:
        return sum(part.mesh.vertex_count for part in self.parts.values() if part.mesh is not None)

    @property
    def occurrence_count(self) -> int:
        return sum(1 for node in self.root.walk() if node.part_id is not None)

    def copy(self, *, keep_source: bool = True) -> Asset:
        return Asset(
            root=self.root.copy(),
            parts={part_id: part.copy(keep_source=keep_source) for part_id, part in self.parts.items()},
            materials=dict(self.materials),
            units=self.units,
            meters_per_unit=self.meters_per_unit,
            up_axis=self.up_axis,
            source_path=self.source_path,
            report=self.report.copy(),
        )

    def stats(self) -> dict[str, int]:
        return {
            "nodes": len(self.root.walk()),
            "parts": self.part_count,
            "occurrences": self.occurrence_count,
            "materials": self.material_count,
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
        }

    def tessellate(self, options: Tessellation | None = None) -> Asset:
        from fascat.ops.tessellate import tessellate_asset

        opts = options or Tessellation()
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = tessellate_asset(self, opts)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "tessellate",
            options=opts.to_dict(),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def repair(self, options: RepairOptions | None = None) -> Asset:
        opts = options or RepairOptions()
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = self.copy(keep_source=True)
            for part in asset.parts.values():
                if part.mesh is not None:
                    part.mesh = part.mesh.repair(opts)
                    part.fingerprint = part.mesh.fingerprint()
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "repair",
            options=opts.to_dict(),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def stage(self, options: StageOptions | None = None) -> Asset:
        from fascat.ops.stage import stage_asset

        opts = options or StageOptions()
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = stage_asset(self, opts)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "stage",
            options=opts.to_dict(),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def optimize(self, options: OptimizeOptions | None = None) -> Asset:
        from fascat.ops.optimize import optimize_asset

        opts = options or OptimizeOptions()
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = optimize_asset(self, opts)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "optimize",
            options=opts.to_dict(),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def lods(self, options: LODOptions | None = None) -> Asset:
        from fascat.ops.lod import build_lods

        opts = options or LODOptions()
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = build_lods(self, opts)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "lods",
            options=opts.to_dict(),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def write_usd(self, path: str | Path, *, debug: bool = False) -> None:
        from fascat.io.usd import write_usd

        write_usd(self, path, debug=debug)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "units": self.units,
            "meters_per_unit": self.meters_per_unit,
            "up_axis": self.up_axis,
            "stats": self.stats(),
            "root": self.root.to_dict(),
            "parts": {part_id: part.to_dict() for part_id, part in self.parts.items()},
            "materials": {material_id: material.to_dict() for material_id, material in self.materials.items()},
            "report": self.report.to_dict(),
        }
