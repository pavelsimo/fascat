from __future__ import annotations

import json
import math
from dataclasses import dataclass

from fascat.asset import Asset, Part
from fascat.mesh import Mesh
from fascat.ops.parallel import parallel_map
from fascat.options import OptimizeOptions


@dataclass(frozen=True)
class _OptimizedPart:
    part_id: str
    mesh: Mesh
    metadata: dict[str, object]
    fingerprint: str


def optimize_asset(asset: Asset, options: OptimizeOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    if not options.preserve_instances:
        result = _duplicate_parts_per_occurrence(result, selected_part_ids=selected_part_ids)
        if selected_part_ids is not None:
            selected_part_ids = {
                node.part_id
                for node in result.root.walk()
                if node.part_id is not None and "source_part_id" in result.parts[node.part_id].metadata
            }
    total_triangles = _selected_triangle_count(result.parts, selected_part_ids)
    targets = _targets_for_parts(
        result.parts,
        total_triangles,
        options.target_triangles,
        selected_part_ids=selected_part_ids,
        preserve_small_parts=options.preserve_small_parts,
        small_part_triangle_threshold=options.small_part_triangle_threshold,
    )
    if targets is not None and sum(targets.values()) > (options.target_triangles or 0):
        if options.preserve_small_parts:
            result.report.add_warning(
                "target_triangles is lower than the preserved feature minimum; preserving requested features"
            )
        else:
            result.report.add_warning(
                "target_triangles is lower than the number of non-empty unique meshes; using one triangle per mesh"
            )
    part_ids = [
        part.id
        for part in result.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    ]

    def optimize_part(part_id: str) -> _OptimizedPart:
        part = result.parts[part_id]
        if part.mesh is None:
            raise AssertionError("selected optimize part must have a mesh")
        return _optimize_part(part, options, target=targets.get(part.id) if targets is not None else None)

    for optimized in parallel_map(part_ids, optimize_part, jobs=options.jobs):
        part = result.parts[optimized.part_id]
        part.mesh = optimized.mesh
        part.metadata = optimized.metadata
        part.fingerprint = optimized.fingerprint
    return result


def _optimize_part(part: Part, options: OptimizeOptions, *, target: int | None) -> _OptimizedPart:
    if part.mesh is None:
        raise AssertionError("optimized part must have a mesh")
    metadata = dict(part.metadata)
    mesh = part.mesh
    if options.simplify:
        if _preserve_small_part(mesh, options):
            metadata["simplification_preserved"] = "small_part"
        else:
            feature_counts = _feature_counts(mesh, options)
            if _feature_preservation_enabled(options):
                mesh = mesh.simplify(
                    target_triangles=target,
                    ratio=None if target is not None else options.ratio,
                    preserve_hard_edges=options.preserve_hard_edges,
                    hard_edge_angle=options.hard_edge_angle,
                    preserve_holes=options.preserve_holes,
                    preserve_material_boundaries=options.preserve_material_boundaries,
                    preserve_uv_seams=options.preserve_uv_seams,
                    preserve_silhouette=options.preserve_silhouette,
                )
            else:
                mesh = mesh.simplify(target_triangles=target, ratio=None if target is not None else options.ratio)
            if feature_counts is not None:
                metadata["simplification_preserved_features"] = json.dumps(
                    feature_counts,
                    sort_keys=True,
                )
            mesh.validate()
    if options.optimize_buffers:
        mesh = mesh.optimize_buffers()
        mesh.validate()
    mesh = mesh.repair()
    return _OptimizedPart(part_id=part.id, mesh=mesh, metadata=metadata, fingerprint=mesh.fingerprint())


def _duplicate_parts_per_occurrence(asset: Asset, *, selected_part_ids: set[str] | None = None) -> Asset:
    parts: dict[str, Part] = {}
    counters: dict[str, int] = {}
    carried: set[str] = set()
    for node in asset.root.walk():
        if node.part_id is None or node.part_id not in asset.parts:
            continue
        source_id = node.part_id
        if selected_part_ids is not None and source_id not in selected_part_ids:
            if source_id not in carried:
                parts[source_id] = asset.parts[source_id].copy(keep_source=True)
                carried.add(source_id)
            continue
        occurrence_index = counters.get(source_id, 0) + 1
        counters[source_id] = occurrence_index
        source = asset.parts[source_id]
        part = source.copy(keep_source=True)
        part.id = f"{source_id}_{occurrence_index}"
        part.metadata = {
            **part.metadata,
            "source_part_id": source_id,
            "occurrence_node_id": node.id,
        }
        parts[part.id] = part
        node.part_id = part.id
    asset.parts = parts
    return asset


def _targets_for_parts(
    parts: dict[str, Part],
    total_triangles: int,
    target_triangles: int | None,
    selected_part_ids: set[str] | None = None,
    preserve_small_parts: bool = False,
    small_part_triangle_threshold: int = 64,
) -> dict[str, int] | None:
    if target_triangles is None or total_triangles <= target_triangles or total_triangles == 0:
        return None

    eligible = [
        (part_id, part.mesh.triangle_count)
        for part_id, part in parts.items()
        if (selected_part_ids is None or part_id in selected_part_ids)
        and part.mesh is not None
        and part.mesh.triangle_count > 0
    ]
    if not eligible:
        return None

    preserved = {
        part_id: triangle_count
        for part_id, triangle_count in eligible
        if preserve_small_parts and triangle_count <= small_part_triangle_threshold
    }
    remaining_eligible = [(part_id, count) for part_id, count in eligible if part_id not in preserved]
    remaining_target = target_triangles - sum(preserved.values())
    if not remaining_eligible:
        return preserved

    minimum_total = len(remaining_eligible)
    if remaining_target <= minimum_total:
        return {**preserved, **{part_id: 1 for part_id, _triangles in remaining_eligible}}

    targets: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []
    assigned = 0
    exact_targets: dict[str, float] = {}
    remaining_triangles = sum(triangle_count for _part_id, triangle_count in remaining_eligible)
    for part_id, triangle_count in remaining_eligible:
        exact = remaining_target * (triangle_count / remaining_triangles)
        exact_targets[part_id] = exact
        base = max(1, int(math.floor(exact)))
        targets[part_id] = base
        assigned += base
        remainders.append((exact - base, triangle_count, part_id))

    while assigned > remaining_target:
        removable = [
            (targets[part_id] - exact_targets[part_id], targets[part_id], part_id)
            for part_id in targets
            if targets[part_id] > 1
        ]
        if not removable:
            break
        _overage, _target, part_id = max(removable)
        targets[part_id] -= 1
        assigned -= 1

    remaining = max(0, remaining_target - assigned)
    for _remainder, _triangle_count, part_id in sorted(remainders, reverse=True)[:remaining]:
        targets[part_id] += 1
    return {**preserved, **targets}


def _selected_triangle_count(parts: dict[str, Part], selected_part_ids: set[str] | None) -> int:
    return sum(
        part.mesh.triangle_count
        for part_id, part in parts.items()
        if (selected_part_ids is None or part_id in selected_part_ids) and part.mesh is not None
    )


def _feature_preservation_enabled(options: OptimizeOptions) -> bool:
    return any(
        (
            options.preserve_hard_edges,
            options.preserve_holes,
            options.preserve_material_boundaries,
            options.preserve_uv_seams,
            options.preserve_silhouette,
        )
    )


def _preserve_small_part(mesh: Mesh, options: OptimizeOptions) -> bool:
    return options.preserve_small_parts and mesh.triangle_count <= options.small_part_triangle_threshold


def _feature_counts(mesh: Mesh, options: OptimizeOptions) -> dict[str, int] | None:
    if not _feature_preservation_enabled(options):
        return None
    return mesh.feature_preservation_counts(
        preserve_hard_edges=options.preserve_hard_edges,
        hard_edge_angle=options.hard_edge_angle,
        preserve_holes=options.preserve_holes,
        preserve_material_boundaries=options.preserve_material_boundaries,
        preserve_uv_seams=options.preserve_uv_seams,
        preserve_silhouette=options.preserve_silhouette,
    )
