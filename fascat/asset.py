from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from fascat.material import Material
from fascat.mesh import Mesh
from fascat.metadata import Metadata, PmiAnnotation
from fascat.options import (
    AnalyzeOptions,
    BakeMaterialOptions,
    BrepHealOptions,
    DecimateOptions,
    DeleteDegeneratePolygonsOptions,
    ExplodeOptions,
    GltfExportOptions,
    LODGeneratorOptions,
    LODOptions,
    MergeOptions,
    MergeVerticesOptions,
    ObjExportOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
    RepairOptions,
    ReplaceOptions,
    SceneOptimizeOptions,
    StageOptions,
    StlExportOptions,
    Tessellation,
    UsdExportOptions,
)
from fascat.report import Report, timed_step

if TYPE_CHECKING:
    from fascat.analysis import AnalysisReport

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
        return self.draw_call_breakdown()["draw_calls"]

    def draw_call_breakdown(self) -> dict[str, int]:
        occurrence_counts: dict[str, int] = {}
        for node in self.root.walk():
            if node.part_id is None:
                continue
            part = self.parts.get(node.part_id)
            if part is None or part.mesh is None:
                continue
            occurrence_counts[node.part_id] = occurrence_counts.get(node.part_id, 0) + 1

        draw_calls = 0
        submesh_slots = 0
        material_slots = 0
        used_material_ids: set[str] = set()
        instanced_meshes = 0
        merged_batches = 0
        for part_id, occurrence_count in occurrence_counts.items():
            part = self.parts[part_id]
            slots = _part_draw_call_slots(part)
            draw_calls += slots * occurrence_count
            submesh_slots += slots
            material_slots += _part_material_slot_count(part, slots)
            used_material_ids.update(_part_used_material_ids(part))
            if occurrence_count > 1:
                instanced_meshes += 1
            if _part_is_merged_batch(part):
                merged_batches += 1

        mesh_instances = sum(occurrence_counts.values())
        mesh_count = len(occurrence_counts)
        return {
            "draw_calls": draw_calls,
            "draw_call_meshes": mesh_count,
            "draw_call_materials": len(used_material_ids),
            "draw_call_submesh_slots": submesh_slots,
            "draw_call_material_slots": material_slots,
            "draw_call_mesh_instances": mesh_instances,
            "draw_call_reused_instances": max(0, mesh_instances - mesh_count),
            "draw_call_instanced_meshes": instanced_meshes,
            "draw_call_merged_batches": merged_batches,
        }

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

    def analyze(self, options: AnalyzeOptions | None = None, *, where: Any | None = None) -> AnalysisReport:
        from fascat.analysis import analyze_asset

        opts = options or AnalyzeOptions()
        scope = self._operation_scope(where)
        return analyze_asset(scope.asset, opts, source_path=self.source_path)

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
        tolerance_policy = _tolerance_policy(
            scope.asset,
            length_tolerance=opts.tolerance,
            area_tolerance=opts.area_epsilon,
            length_key="vertex_merge_tolerance",
            area_key="degenerate_area_epsilon",
            operations={
                "vertex_merge": "enabled" if opts.merge_vertices else "disabled",
                "degenerate_polygon_cleanup": "enabled" if opts.delete_degenerate else "disabled",
                "t_junction_sewing": "not_implemented",
                "boundary_gap_stitching": "not_implemented",
                "non_manifold_edge_cracking": "not_implemented",
            },
        )
        repair_unit_metadata = _tolerance_policy_metadata("repair", tolerance_policy)
        with timed_step() as timer:
            asset = scope.asset.copy(keep_source=True)
            for part in asset.parts.values():
                if scope.selected_part_ids is not None and part.id not in scope.selected_part_ids:
                    continue
                if part.mesh is not None:
                    part.mesh = part.mesh.repair(opts)
                    part.mesh.metadata = {**part.mesh.metadata, **repair_unit_metadata}
                    non_orientable_edges = _metadata_int(
                        part.mesh.metadata.get("repair_non_orientable_edges_before_orientation"),
                        0,
                    )
                    if non_orientable_edges:
                        asset.report.add_warning(
                            f"part {part.id} has {non_orientable_edges} non-orientable shared edge(s) "
                            "before face orientation; Mobius-like topology cannot be fixed by winding normalization"
                        )
                    remaining_t_junctions = _metadata_int(part.mesh.metadata.get("repair_t_junctions_after"), 0)
                    if remaining_t_junctions:
                        asset.report.add_warning(
                            f"part {part.id} has {remaining_t_junctions} T-junction(s) after mesh repair; "
                            "T-junction sewing is not implemented"
                        )
                    remaining_boundary_gaps = _metadata_int(part.mesh.metadata.get("repair_boundary_gaps_after"), 0)
                    if remaining_boundary_gaps:
                        asset.report.add_warning(
                            f"part {part.id} has {remaining_boundary_gaps} boundary gap(s) after mesh repair; "
                            "boundary gap stitching is not implemented"
                        )
                    remaining_flipped_components = _metadata_int(
                        part.mesh.metadata.get("repair_flipped_components_after_orientation"),
                        0,
                    )
                    if remaining_flipped_components:
                        asset.report.add_warning(
                            f"part {part.id} has {remaining_flipped_components} flipped closed orientation "
                            "component(s) after mesh repair; outward face orientation was not produced"
                        )
                    part.fingerprint = part.mesh.fingerprint()
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "repair",
            options=_options_with_scope({**opts.to_dict(), "tolerance_policy": tolerance_policy}, scope),
            before=before,
            after=_repair_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def merge_vertices(self, options: MergeVerticesOptions | None = None, *, where: Any | None = None) -> Asset:
        opts = options or MergeVerticesOptions()
        scope = self._operation_scope(where)
        before = self.stats()
        warning_count = len(self.report.warnings)
        tolerance_policy = _tolerance_policy(
            scope.asset,
            length_tolerance=opts.tolerance,
            area_tolerance=opts.area_epsilon,
            length_key="merge_vertices_tolerance",
            area_key="merge_vertices_area_epsilon",
            operations={
                "attribute_protection": "enabled"
                if (opts.preserve_normals or opts.preserve_tangents or opts.preserve_uvs)
                else "disabled",
                "material_boundary_protection": "enabled" if opts.preserve_material_boundaries else "disabled",
                "degenerate_polygon_cleanup": "enabled" if opts.delete_degenerate else "disabled",
            },
        )
        merge_unit_metadata = _tolerance_policy_metadata("merge_vertices", tolerance_policy)
        with timed_step() as timer:
            asset = scope.asset.copy(keep_source=True)
            for part in asset.parts.values():
                if scope.selected_part_ids is not None and part.id not in scope.selected_part_ids:
                    continue
                if part.mesh is None:
                    continue
                part.mesh = part.mesh.merge_vertices(opts)
                part.mesh.metadata = {**part.mesh.metadata, **merge_unit_metadata}
                for warning in _merge_vertices_tolerance_warnings(part):
                    asset.report.add_warning(warning)
                part.fingerprint = part.mesh.fingerprint()
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "merge_vertices",
            options=_options_with_scope({**opts.to_dict(), "tolerance_policy": tolerance_policy}, scope),
            before=before,
            after=_merge_vertices_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def delete_degenerate_polygons(
        self,
        options: DeleteDegeneratePolygonsOptions | None = None,
        *,
        where: Any | None = None,
    ) -> Asset:
        opts = options or DeleteDegeneratePolygonsOptions()
        scope = self._operation_scope(where)
        before = self.stats()
        tolerance_policy = _tolerance_policy(
            scope.asset,
            length_tolerance=0.0,
            area_tolerance=opts.area_epsilon,
            length_key="delete_degenerate_polygons_tolerance",
            area_key="delete_degenerate_polygons_area_epsilon",
            operations={"degenerate_polygon_cleanup": "enabled"},
        )
        unit_metadata = _tolerance_policy_metadata("delete_degenerate_polygons", tolerance_policy)
        with timed_step() as timer:
            asset = scope.asset.copy(keep_source=True)
            for part in asset.parts.values():
                if scope.selected_part_ids is not None and part.id not in scope.selected_part_ids:
                    continue
                if part.mesh is None:
                    continue
                part.mesh = part.mesh.delete_degenerate_polygons(opts)
                part.mesh.metadata = {**part.mesh.metadata, **unit_metadata}
                part.fingerprint = part.mesh.fingerprint()
        asset.report.add_step(
            "delete_degenerate_polygons",
            options=_options_with_scope({**opts.to_dict(), "tolerance_policy": tolerance_policy}, scope),
            before=before,
            after=_delete_degenerate_polygons_report_stats(asset),
            duration=timer.duration,
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
            after=_stage_report_stats(asset),
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
            after=_lod_report_stats(asset),
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
        after = _hierarchy_report_stats(asset)
        step_options = _options_with_scope(opts.to_dict(), scope)
        _add_export_merge_advisor(asset, step_options, before, after)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "merge",
            options=step_options,
            before=before,
            after=after,
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def explode(self, options: ExplodeOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.hierarchy import explode_asset

        opts = options or ExplodeOptions()
        scope = self._operation_scope(where)
        selected_node_ids = (
            scope.selection.node_ids
            if scope.selection is not None
            else {node.id for node in scope.asset.root.walk() if node.part_id is not None}
        )
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = explode_asset(scope.asset, opts, selected_node_ids=selected_node_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "explode",
            options=_options_with_scope(opts.to_dict(), scope),
            before=before,
            after=_hierarchy_report_stats(asset),
            duration=timer.duration,
            warnings=step_warnings,
        )
        return asset

    def replace(self, options: ReplaceOptions | None = None, *, where: Any | None = None) -> Asset:
        from fascat.ops.hierarchy import replace_asset

        opts = options or ReplaceOptions()
        scope = self._operation_scope(where)
        selected_node_ids = (
            scope.selection.node_ids
            if scope.selection is not None
            else {node.id for node in scope.asset.root.walk() if node.part_id is not None}
        )
        before = _hierarchy_report_stats(self)
        warning_count = len(self.report.warnings)
        with timed_step() as timer:
            asset = replace_asset(scope.asset, opts, selected_node_ids=selected_node_ids)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "replace",
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
        after = _hierarchy_report_stats(asset)
        step_options = _options_with_scope(opts.to_dict(), scope)
        if opts.merge_compatible_meshes or opts.batch_by_material:
            _add_export_merge_advisor(asset, step_options, before, after)
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "optimize_scene",
            options=step_options,
            before=before,
            after=after,
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
            after=_decimation_report_stats(asset),
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
            after=_lod_report_stats(asset),
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
        tolerance_policy = _tolerance_policy(
            scope.asset,
            length_tolerance=opts.tolerance,
            area_tolerance=opts.max_sliver_area,
            length_key="heal_tolerance",
            area_key="max_sliver_area",
            operations={
                "sew_faces": "enabled" if opts.sew_faces else "disabled",
                "fix_edges": "enabled" if opts.fix_edges else "disabled",
                "unify_tolerances": "enabled" if opts.unify_tolerances else "disabled",
                "sliver_face_removal": "requested" if opts.remove_sliver_faces else "disabled",
                "t_junction_sewing": "not_implemented",
                "non_manifold_edge_cracking": "not_implemented",
            },
        )
        with timed_step() as timer:
            asset = heal_brep_asset(
                scope.asset,
                opts,
                selected_part_ids=scope.selected_part_ids,
                tolerance_policy=tolerance_policy,
            )
        step_warnings = asset.report.warnings[warning_count:]
        asset.report.add_step(
            "heal_brep",
            options=_options_with_scope({**opts.to_dict(), "tolerance_policy": tolerance_policy}, scope),
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
                metadata=UsdExportOptions().metadata if options is None else options.metadata,
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
        from fascat.io.gltf import runtime_dependency_report, write_gltf

        opts = options or GltfExportOptions()
        before = self._report_stats()
        step_options: dict[str, object] = {
            "format": "glTF",
            **opts.to_dict(),
            "runtime_dependencies": runtime_dependency_report(self, opts),
        }
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
        occurrences = _occurrences_by_part(self)
        if not _needs_occurrence_isolation(occurrences, selected_node_ids):
            return self

        asset = self.copy(keep_source=True)
        occurrences = _occurrences_by_part(asset)
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


def _add_export_merge_advisor(
    asset: Asset,
    options: dict[str, object],
    before: dict[str, int],
    after: dict[str, int],
) -> None:
    lost_reused_instances = max(
        0,
        before.get("draw_call_reused_instances", 0) - after.get("draw_call_reused_instances", 0),
    )
    if lost_reused_instances <= 0:
        return

    draw_call_savings = max(0, before.get("draw_calls", 0) - after.get("draw_calls", 0))
    added_merged_batches = max(
        0,
        after.get("draw_call_merged_batches", 0) - before.get("draw_call_merged_batches", 0),
    )
    advisory = {
        "lost_reused_instances": lost_reused_instances,
        "draw_call_savings": draw_call_savings,
        "added_merged_batches": added_merged_batches,
        "recommendation": (
            "preserve or reconstruct instances when GLB file size, memory use, or culling granularity "
            "matters more than reducing draw calls"
        ),
    }
    options["export_advisor"] = advisory
    asset.report.add_warning(
        "merge reduced reusable instances; preserving or reconstructing instances can reduce GLB file size, "
        "memory use, and culling loss when draw-call reduction is not the primary export goal"
    )


def _tolerance_policy(
    asset: Asset,
    *,
    length_tolerance: float,
    area_tolerance: float,
    length_key: str,
    area_key: str,
    operations: dict[str, str],
) -> dict[str, object]:
    source_units = _metadata_str(asset.metadata.get("source_units"), asset.units)
    source_meters_per_unit = _metadata_float(asset.metadata.get("source_meters_per_unit"), asset.meters_per_unit)
    coordinate_space = (
        "source_local"
        if source_units != asset.units or not np.isclose(source_meters_per_unit, asset.meters_per_unit)
        else "asset"
    )
    return {
        "coordinate_space": coordinate_space,
        "effective_units": source_units,
        "effective_meters_per_unit": source_meters_per_unit,
        "source_units": source_units,
        "source_meters_per_unit": source_meters_per_unit,
        "target_units": asset.units,
        "target_meters_per_unit": asset.meters_per_unit,
        length_key: length_tolerance,
        f"{length_key}_meters": length_tolerance * source_meters_per_unit,
        area_key: area_tolerance,
        f"{area_key}_square_meters": area_tolerance * source_meters_per_unit * source_meters_per_unit,
        "operations": dict(operations),
    }


def _tolerance_policy_metadata(prefix: str, policy: dict[str, object]) -> dict[str, object]:
    metadata: dict[str, object] = {
        f"{prefix}_coordinate_space": str(policy["coordinate_space"]),
        f"{prefix}_effective_units": str(policy["effective_units"]),
        f"{prefix}_effective_meters_per_unit": _format_metadata_float(policy["effective_meters_per_unit"]),
        f"{prefix}_source_units": str(policy["source_units"]),
        f"{prefix}_source_meters_per_unit": _format_metadata_float(policy["source_meters_per_unit"]),
        f"{prefix}_target_units": str(policy["target_units"]),
        f"{prefix}_target_meters_per_unit": _format_metadata_float(policy["target_meters_per_unit"]),
    }
    for key, value in policy.items():
        if key.endswith("_meters") or key.endswith("_square_meters"):
            metadata[f"{prefix}_{key}"] = _format_metadata_float(value)
    operations = policy.get("operations")
    if isinstance(operations, dict):
        for key, value in operations.items():
            metadata[f"{prefix}_{key}"] = str(value)
    return metadata


def _metadata_str(value: object, default: str) -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _metadata_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _metadata_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _format_metadata_float(value: object) -> str:
    numeric = _metadata_float(value, 0.0)
    return f"{numeric:.9g}"


def _part_draw_call_slots(part: Part) -> int:
    mesh = part.mesh
    if mesh is None:
        return 0
    if mesh.material_indices is None or mesh.material_indices.size == 0:
        return 1
    return max(1, len(set(mesh.material_indices.astype(int).tolist())))


def _part_material_slot_count(part: Part, draw_call_slots: int) -> int:
    if part.mesh is None:
        return 0
    return max(draw_call_slots, len(part.material_ids), 1)


def _part_used_material_ids(part: Part) -> set[str]:
    if part.mesh is None:
        return set()
    if part.mesh.material_indices is None or part.mesh.material_indices.size == 0:
        return set(part.material_ids)
    used: set[str] = set()
    for index in set(part.mesh.material_indices.astype(int).tolist()):
        if 0 <= index < len(part.material_ids):
            used.add(part.material_ids[index])
    return used


def _part_is_merged_batch(part: Part) -> bool:
    mesh_metadata = {} if part.mesh is None else part.mesh.metadata
    return _metadata_int(mesh_metadata.get("merged_occurrences"), 0) > 0


def _hierarchy_report_stats(asset: Asset) -> dict[str, int]:
    return {**asset.stats(include_lods=True), **asset.draw_call_breakdown()}


def _lod_report_stats(asset: Asset) -> dict[str, int]:
    stats = _hierarchy_report_stats(asset)
    for key in (
        "lod_generated_parts",
        "lod_skipped_no_mesh_parts",
        "lod_source_vertices",
        "lod_source_triangles",
        "lod_source_mesh_bytes",
        "lod_added_vertices",
        "lod_added_triangles",
        "lod_added_mesh_bytes",
        "lod_chain_vertices",
        "lod_chain_triangles",
        "lod_chain_mesh_bytes",
        "lod_omitted_tiny_part_meshes",
        "lod_reused_instance_levels",
        "lod_material_merged_levels",
        "lod_texture_baked_levels",
        "lod_culling_changed_levels",
    ):
        if key in asset.metadata:
            stats[key] = _metadata_int(asset.metadata[key], 0)
    return stats


def _repair_report_stats(asset: Asset) -> dict[str, int]:
    stats = asset.stats()
    non_orientable_edges = 0
    t_junctions_before = 0
    t_junctions_after = 0
    boundary_gaps_before = 0
    boundary_gaps_after = 0
    flipped_components_before = 0
    flipped_components_after = 0
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        non_orientable_edges += _metadata_int(
            part.mesh.metadata.get("repair_non_orientable_edges_before_orientation"),
            0,
        )
        t_junctions_before += _metadata_int(part.mesh.metadata.get("repair_t_junctions_before"), 0)
        t_junctions_after += _metadata_int(part.mesh.metadata.get("repair_t_junctions_after"), 0)
        boundary_gaps_before += _metadata_int(part.mesh.metadata.get("repair_boundary_gaps_before"), 0)
        boundary_gaps_after += _metadata_int(part.mesh.metadata.get("repair_boundary_gaps_after"), 0)
        flipped_components_before += _metadata_int(
            part.mesh.metadata.get("repair_flipped_components_before_orientation"),
            0,
        )
        flipped_components_after += _metadata_int(
            part.mesh.metadata.get("repair_flipped_components_after_orientation"),
            0,
        )
    if non_orientable_edges:
        stats["repair_non_orientable_edges_before_orientation"] = non_orientable_edges
    if t_junctions_before or t_junctions_after:
        stats["repair_t_junctions_before"] = t_junctions_before
        stats["repair_t_junctions_after"] = t_junctions_after
    if boundary_gaps_before or boundary_gaps_after:
        stats["repair_boundary_gaps_before"] = boundary_gaps_before
        stats["repair_boundary_gaps_after"] = boundary_gaps_after
    if flipped_components_before or flipped_components_after:
        stats["repair_flipped_components_before_orientation"] = flipped_components_before
        stats["repair_flipped_components_after_orientation"] = flipped_components_after
    return stats


def _merge_vertices_report_stats(asset: Asset) -> dict[str, int]:
    stats = asset.stats()
    merge_metadata_keys = (
        "merge_vertices_removed",
        "merge_vertices_degenerate_triangles_removed",
        "merge_vertices_candidate_position_buckets",
        "merge_vertices_candidate_vertices",
        "merge_vertices_candidate_exact_duplicate_buckets",
        "merge_vertices_candidate_boundary_buckets",
        "merge_vertices_candidate_non_manifold_buckets",
        "merge_vertices_candidate_hard_edge_buckets",
        "merge_vertices_candidate_t_junctions",
        "merge_vertices_candidate_boundary_gaps",
        "merge_vertices_near_duplicate_pairs",
        "merge_vertices_skipped_by_protection",
        "merge_vertices_skipped_by_normals",
        "merge_vertices_skipped_by_tangents",
        "merge_vertices_skipped_by_uvs",
        "merge_vertices_skipped_by_material_boundaries",
    )
    merge_metadata_totals = dict.fromkeys(merge_metadata_keys, 0)
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        for key in merge_metadata_keys:
            merge_metadata_totals[key] += _metadata_int(part.mesh.metadata.get(key), 0)
    stats.update(merge_metadata_totals)
    high_risk_parts = 0
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        if str(part.mesh.metadata.get("merge_vertices_tolerance_risk", "")) in {
            "high_relative_to_min_edge",
            "high_relative_to_bbox",
        }:
            high_risk_parts += 1
    stats["merge_vertices_tolerance_high_risk_parts"] = high_risk_parts
    too_small_parts = 0
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        if str(part.mesh.metadata.get("merge_vertices_tolerance_advisory", "")) == "near_duplicates_unmerged":
            too_small_parts += 1
    stats["merge_vertices_tolerance_too_small_parts"] = too_small_parts
    return stats


def _merge_vertices_tolerance_warnings(part: Part) -> list[str]:
    mesh = part.mesh
    if mesh is None:
        return []
    risk = str(mesh.metadata.get("merge_vertices_tolerance_risk", ""))
    if risk == "high_relative_to_min_edge":
        return [
            f"part {part.id} merge_vertices tolerance is high relative to its shortest mesh edge; "
            "nearby distinct features may collapse"
        ]
    if risk == "high_relative_to_bbox":
        return [
            f"part {part.id} merge_vertices tolerance is high relative to its bounding-box diagonal; "
            "verify broad tolerance merging is intended"
        ]
    if str(mesh.metadata.get("merge_vertices_tolerance_advisory", "")) == "near_duplicates_unmerged":
        near_pairs = _metadata_int(mesh.metadata.get("merge_vertices_near_duplicate_pairs"), 0)
        nearest = str(mesh.metadata.get("merge_vertices_nearest_near_duplicate_distance", "0"))
        return [
            f"part {part.id} merge_vertices tolerance is below {near_pairs} near-duplicate vertex pair(s); "
            f"closest remaining spacing is {nearest}"
        ]
    return []


def _delete_degenerate_polygons_report_stats(asset: Asset) -> dict[str, int]:
    stats = asset.stats()
    delete_metadata_keys = (
        "delete_degenerate_polygons_before",
        "delete_degenerate_polygons_after",
        "delete_degenerate_polygons_removed",
        "delete_degenerate_polygons_vertices_removed",
        "delete_degenerate_polygons_removed_duplicate_vertices",
        "delete_degenerate_polygons_removed_collapsed_edges",
        "delete_degenerate_polygons_removed_near_flat_area",
    )
    delete_metadata_totals = dict.fromkeys(delete_metadata_keys, 0)
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        for key in delete_metadata_keys:
            delete_metadata_totals[key] += _metadata_int(part.mesh.metadata.get(key), 0)
    stats.update(delete_metadata_totals)
    return stats


def _stage_report_stats(asset: Asset) -> dict[str, int]:
    stats = asset.stats()
    if "stage_bake_uv_channels_missing_repack" in asset.metadata:
        stats["stage_bake_uv_channels_missing_repack"] = _metadata_int(
            asset.metadata["stage_bake_uv_channels_missing_repack"],
            0,
        )
    if "stage_uv_policy_intent_channels" in asset.metadata:
        stats["stage_uv_policy_intent_channels"] = _metadata_int(
            asset.metadata["stage_uv_policy_intent_channels"],
            0,
        )
    if "stage_uv_forbid_overlapping_violations" in asset.metadata:
        stats["stage_uv_forbid_overlapping_violations"] = _metadata_int(
            asset.metadata["stage_uv_forbid_overlapping_violations"],
            0,
        )
    for key in (
        "stage_normals_generated_parts",
        "stage_normals_regenerated_parts",
        "stage_normals_preserved_parts",
        "stage_normals_disabled_parts",
    ):
        if key in asset.metadata:
            stats[key] = _metadata_int(asset.metadata[key], 0)
    return stats


def _decimation_report_stats(asset: Asset) -> dict[str, int]:
    stats = _hierarchy_report_stats(asset)
    for key in (
        "decimate_source_triangles",
        "decimate_output_triangles",
        "decimate_estimated_memory_bytes",
        "decimate_iterative_threshold_triangles",
        "decimate_simplification_passes",
        "decimate_iterative_passes",
        "decimate_max_part_simplification_passes",
        "decimate_allocated_target_triangles",
        "decimate_allocation_part_count",
        "decimate_allocation_preserved_parts",
        "decimate_allocation_reduced_parts",
        "decimate_allocation_min_target_triangles",
        "decimate_allocation_max_target_triangles",
        "decimate_pre_cleanup_removed_tangent_parts",
        "decimate_uv_constrained_parts",
        "decimate_uv_seam_constraint_vertices",
        "decimate_protected_feature_parts",
        "decimate_protect_hard_edge_faces",
        "decimate_protect_hole_boundary_faces",
        "decimate_protect_material_boundary_faces",
        "decimate_protect_uv_seam_faces",
        "decimate_protect_silhouette_faces",
        "decimate_protect_total_feature_faces",
    ):
        if key in asset.metadata:
            stats[key] = _metadata_int(asset.metadata[key], 0)
    if "decimate_iterative_recommended" in asset.metadata:
        stats["decimate_iterative_recommended"] = 1 if asset.metadata["decimate_iterative_recommended"] == "true" else 0
    return stats


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


def _occurrences_by_part(asset: Asset) -> dict[str, list[Node]]:
    occurrences: dict[str, list[Node]] = {}
    for node in asset.root.walk():
        if node.part_id is not None and node.part_id in asset.parts:
            occurrences.setdefault(node.part_id, []).append(node)
    return occurrences


def _needs_occurrence_isolation(occurrences: dict[str, list[Node]], selected_node_ids: set[str]) -> bool:
    for nodes in occurrences.values():
        selected_count = sum(1 for node in nodes if node.id in selected_node_ids)
        if selected_count and selected_count != len(nodes):
            return True
    return False
