from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from fascat.material import Material
from fascat.mesh import Mesh
from fascat.metadata import Metadata, PmiAnnotation
from fascat.options import (
    BakeMaterialOptions,
    BrepHealOptions,
    DecimateOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    LODOptions,
    MergeOptions,
    ObjExportOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    RepairOptions,
    SceneOptimizeOptions,
    StageOptions,
    StlExportOptions,
    Tessellation,
    UsdExportOptions,
)
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
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.children = [child.copy() for child in self.children]
        self.transform = np.asarray(self.transform, dtype=np.float64).copy()
        if self.transform.shape != (4, 4):
            raise ValueError("node transform must have shape (4, 4)")
        self.metadata = dict(self.metadata)

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
            "transform": self.transform.tolist(),
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
    metadata: Metadata = field(default_factory=dict)
    fingerprint: str | None = None
    lod_meshes: list[Mesh] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mesh is not None:
            self.mesh = self.mesh.copy()
        self.material_ids = list(self.material_ids)
        self.metadata = dict(self.metadata)
        self.lod_meshes = [mesh.copy() for mesh in self.lod_meshes]

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
    metadata: Metadata = field(default_factory=dict)
    pmi: list[PmiAnnotation] = field(default_factory=list)
    report: Report = field(default_factory=Report)

    def __post_init__(self) -> None:
        self.root = self.root.copy()
        self.parts = {part_id: part.copy(keep_source=True) for part_id, part in self.parts.items()}
        self.materials = {material_id: material.copy() for material_id, material in self.materials.items()}
        self.metadata = dict(self.metadata)
        self.pmi = [annotation for annotation in self.pmi]
        self.report = self.report.copy()

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

    @property
    def draw_call_count(self) -> int:
        total = 0
        for node in self.root.walk():
            if node.part_id is None:
                continue
            part = self.parts.get(node.part_id)
            if part is None or part.mesh is None:
                continue
            if part.mesh.material_indices is None:
                total += 1
            else:
                total += max(1, len(set(part.mesh.material_indices.astype(int).tolist())))
        return total

    def copy(self, *, keep_source: bool = True) -> Asset:
        return Asset(
            root=self.root.copy(),
            parts={part_id: part.copy(keep_source=keep_source) for part_id, part in self.parts.items()},
            materials={material_id: material.copy() for material_id, material in self.materials.items()},
            units=self.units,
            meters_per_unit=self.meters_per_unit,
            up_axis=self.up_axis,
            source_path=self.source_path,
            metadata=dict(self.metadata),
            pmi=list(self.pmi),
            report=self.report.copy(),
        )

    def select(self, where: Any | None = None) -> Any:
        from fascat.filter import Filter

        selector = Filter.from_value(where) or Filter()
        return selector.select(self)

    def stats(self, *, include_lods: bool = False) -> dict[str, int]:
        stats = {
            "nodes": len(self.root.walk()),
            "parts": self.part_count,
            "occurrences": self.occurrence_count,
            "materials": self.material_count,
            "vertices": self.vertex_count,
            "triangles": self.triangle_count,
        }
        if include_lods:
            lod_meshes = [lod for part in self.parts.values() for lod in part.lod_meshes]
            stats["lod_meshes"] = len(lod_meshes)
            stats["lod_vertices"] = sum(mesh.vertex_count for mesh in lod_meshes)
            stats["lod_triangles"] = sum(mesh.triangle_count for mesh in lod_meshes)
        return stats

    def tessellation_quality_report(self) -> dict[str, object]:
        from fascat.ops.tessellate import build_tessellation_quality_report

        return build_tessellation_quality_report(self)

    def _report_stats(self) -> dict[str, int]:
        return self.stats(include_lods=any(part.lod_meshes for part in self.parts.values()))

    def tessellate(self, options: Tessellation | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.tessellate import tessellate_asset

        opts = options or Tessellation()
        scope = self._operation_scope(where)
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = tessellate_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "tessellate",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def repair(self, options: RepairOptions | None = None, *, where: Any | None = None) -> Asset:
        opts = options or RepairOptions()
        scope = self._operation_scope(where)
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = scope.asset.copy(keep_source=True)
            for part in asset.parts.values():
                if scope.selected_part_ids is not None and part.id not in scope.selected_part_ids:
                    continue
                if part.mesh is not None:
                    part.mesh = part.mesh.repair(opts)
                    part.fingerprint = part.mesh.fingerprint()
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "repair",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def stage(self, options: StageOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.stage import stage_asset

        opts = options or StageOptions()
        scope = self._operation_scope(where)
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = stage_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "stage",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def optimize(self, options: OptimizeOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.optimize import optimize_asset

        opts = options or OptimizeOptions()
        scope = self._operation_scope(where)
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = optimize_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "optimize",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def lods(self, options: LODOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.lod import build_lods

        opts = options or LODOptions()
        scope = self._operation_scope(where)
        before = self.stats(include_lods=True)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = build_lods(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "lods",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=asset.stats(include_lods=True),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def merge(self, options: MergeOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.hierarchy import merge_asset

        opts = options or MergeOptions()
        scope = self._operation_scope(where)
        selected_node_ids = (
            scope.selection.node_ids
            if scope.selection is not None
            else {node.id for node in scope.asset.root.walk() if node.part_id is not None}
        )
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = merge_asset(scope.asset, opts, selected_node_ids=selected_node_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "merge",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def optimize_scene(self, options: SceneOptimizeOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.scene import optimize_scene_asset

        opts = options or SceneOptimizeOptions()
        scope = self._operation_scope(where)
        selected_node_ids = (
            scope.selection.node_ids
            if scope.selection is not None
            else {node.id for node in scope.asset.root.walk() if node.part_id is not None}
        )
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = optimize_scene_asset(scope.asset, opts, selected_node_ids=selected_node_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "optimize_scene",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def bake_materials(self, options: BakeMaterialOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.actions import bake_materials_asset

        opts = options or BakeMaterialOptions()
        scope = self._operation_scope(where)
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = bake_materials_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "bake_materials",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def decimate(self, options: DecimateOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.actions import decimate_asset

        opts = options or DecimateOptions()
        scope = self._operation_scope(where)
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = decimate_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "decimate",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def remove_holes(self, options: RemoveHolesOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.actions import remove_holes_asset

        opts = options or RemoveHolesOptions()
        scope = self._operation_scope(where)
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = remove_holes_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "remove_holes",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def remove_occluded(self, options: RemoveOccludedOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.actions import remove_occluded_asset

        opts = options or RemoveOccludedOptions()
        scope = self._operation_scope(where)
        selected_node_ids = (
            scope.selection.node_ids
            if scope.selection is not None
            else {node.id for node in scope.asset.root.walk() if node.part_id is not None}
        )
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = remove_occluded_asset(scope.asset, opts, selected_node_ids=selected_node_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "remove_occluded",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def run_lod_generators(self, options: LODGeneratorOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.actions import run_lod_generators_asset

        opts = options or LODGeneratorOptions()
        scope = self._operation_scope(where)
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = run_lod_generators_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "run_lod_generators",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def heal_brep(self, options: BrepHealOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.heal import heal_brep_asset

        opts = options or BrepHealOptions()
        scope = self._operation_scope(where)
        before = self.stats()
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = heal_brep_asset(scope.asset, opts, selected_part_ids=scope.selected_part_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "heal_brep",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=asset.stats(),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def write_usd(
        self,
        path: str | Path,
        *,
        debug: bool = False,
        options: UsdExportOptions | None = None,
    ) -> None:
        from fascat.io.usd import write_usd

        if Path(path).suffix.lower() == ".usdz" and (options is None or options.package != "usdz"):
            opts = UsdExportOptions(
                package="usdz",
                file_size_budget_mb=None if options is None else options.file_size_budget_mb,
            )
        else:
            opts = options or UsdExportOptions()
        before = self._report_stats()
        step_options: dict[str, object] = {"format": "OpenUSD", "debug": debug, **opts.to_dict()}
        timer = timed_step()
        try:
            with timer:
                write_usd(self, path, debug=debug, options=opts)
        except Exception as exc:
            self.report.add_error(str(exc) or exc.__class__.__name__)
            self.report.add_step(
                "write",
                options=step_options,
                before=before,
                after=_stats_with_file_size(self._report_stats(), path, opts.file_size_budget_mb, self),
                duration=timer.duration,
            )
            self.report.finish(self._report_stats())
            cast(Any, exc).report = self.report
            raise
        self.report.add_step(
            "write",
            options=step_options,
            before=before,
            after=_stats_with_file_size(self._report_stats(), path, opts.file_size_budget_mb, self),
            duration=timer.duration,
        )
        self.report.finish(self._report_stats())

    def write_gltf(self, path: str | Path, *, options: GltfExportOptions | None = None) -> None:
        from fascat.io.gltf import write_gltf

        opts = options or GltfExportOptions()
        before = self._report_stats()
        step_options: dict[str, object] = {"format": "glTF", **opts.to_dict()}
        timer = timed_step()
        try:
            with timer:
                write_gltf(self, path, options=opts)
        except Exception as exc:
            self.report.add_error(str(exc) or exc.__class__.__name__)
            self.report.add_step(
                "write",
                options=step_options,
                before=before,
                after=_stats_with_file_size(self._report_stats(), path, opts.file_size_budget_mb, self),
                duration=timer.duration,
            )
            self.report.finish(self._report_stats())
            cast(Any, exc).report = self.report
            raise
        self.report.add_step(
            "write",
            options=step_options,
            before=before,
            after=_stats_with_file_size(self._report_stats(), path, opts.file_size_budget_mb, self),
            duration=timer.duration,
        )
        self.report.finish(self._report_stats())

    def write_obj(self, path: str | Path, *, options: ObjExportOptions | None = None) -> None:
        from fascat.io.obj import write_obj

        opts = options or ObjExportOptions()
        before = self._report_stats()
        step_options: dict[str, object] = {"format": "OBJ", **opts.to_dict()}
        timer = timed_step()
        with timer:
            write_obj(self, path, options=opts)
        self.report.add_step(
            "write",
            options=step_options,
            before=before,
            after=_stats_with_file_size(self._report_stats(), path, opts.file_size_budget_mb, self),
            duration=timer.duration,
        )
        self.report.finish(self._report_stats())

    def write_stl(self, path: str | Path, *, options: StlExportOptions | None = None) -> None:
        from fascat.io.stl import write_stl

        opts = options or StlExportOptions()
        before = self._report_stats()
        step_options: dict[str, object] = {"format": "STL", **opts.to_dict()}
        timer = timed_step()
        with timer:
            write_stl(self, path, options=opts)
        self.report.add_step(
            "write",
            options=step_options,
            before=before,
            after=_stats_with_file_size(self._report_stats(), path, opts.file_size_budget_mb, self),
            duration=timer.duration,
        )
        self.report.finish(self._report_stats())

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": str(self.source_path) if self.source_path else None,
            "units": self.units,
            "meters_per_unit": self.meters_per_unit,
            "up_axis": self.up_axis,
            "stats": self.stats(include_lods=True),
            "root": self.root.to_dict(),
            "parts": {part_id: part.to_dict() for part_id, part in self.parts.items()},
            "materials": {material_id: material.to_dict() for material_id, material in self.materials.items()},
            "metadata": dict(self.metadata),
            "pmi": [annotation.to_dict() for annotation in self.pmi],
            "report": self.report.to_dict(),
        }

    def _operation_scope(self, where: Any | None) -> _OperationScope:
        from fascat.filter import Filter

        selector = Filter.from_value(where)
        if selector is None:
            return _OperationScope(asset=self, selected_part_ids=None, selection=None)

        selection = selector.select(self)
        scoped_asset = self._isolate_selected_occurrences(selection.node_ids)
        selected_part_ids = {
            node.part_id
            for node in scoped_asset.root.walk()
            if node.id in selection.node_ids and node.part_id is not None
        }
        return _OperationScope(asset=scoped_asset, selected_part_ids=selected_part_ids, selection=selection)

    def _isolate_selected_occurrences(self, selected_node_ids: set[str]) -> Asset:
        asset = self.copy(keep_source=True)
        occurrences: dict[str, list[Node]] = {}
        for node in asset.root.walk():
            if node.part_id is not None and node.part_id in asset.parts:
                occurrences.setdefault(node.part_id, []).append(node)

        for part_id, nodes in occurrences.items():
            selected_nodes = [node for node in nodes if node.id in selected_node_ids]
            if not selected_nodes or len(selected_nodes) == len(nodes):
                continue
            new_part_id = _unique_part_id(asset.parts, part_id)
            part = asset.parts[part_id].copy(keep_source=True)
            part.id = new_part_id
            part.metadata = {**part.metadata, "source_part_id": part_id}
            asset.parts[new_part_id] = part
            for node in selected_nodes:
                node.part_id = new_part_id
        return asset


@dataclass(frozen=True)
class _OperationScope:
    asset: Asset
    selected_part_ids: set[str] | None
    selection: Any | None


def _options_with_scope(options: dict[str, object], scope: _OperationScope) -> dict[str, object]:
    if scope.selection is None:
        return options
    return {
        **options,
        "where": scope.selection.filter.to_dict(),
        "matched": scope.selection.stats(),
    }


def _hierarchy_report_stats(asset: Asset) -> dict[str, int]:
    return {**asset.stats(include_lods=True), "draw_calls": asset.draw_call_count}


def _stats_with_file_size(
    stats: dict[str, int],
    path: str | Path,
    budget_mb: float | None,
    asset: Asset,
) -> dict[str, int]:
    output_path = Path(path)
    if str(path) == "-" or not output_path.exists():
        return stats
    size = output_path.stat().st_size
    result = {**stats, "file_size_bytes": size}
    if budget_mb is not None:
        budget_bytes = int(budget_mb * 1_000_000)
        result["file_size_budget_bytes"] = budget_bytes
        if size > budget_bytes:
            asset.report.add_warning(f"file size budget exceeded: {size} bytes > {budget_bytes} bytes")
    return result


def _unique_part_id(parts: dict[str, Part], base: str) -> str:
    candidate = f"{base}_selected"
    suffix = 2
    while candidate in parts:
        candidate = f"{base}_selected_{suffix}"
        suffix += 1
    return candidate
