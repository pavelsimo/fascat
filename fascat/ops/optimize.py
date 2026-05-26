from __future__ import annotations

import math

from fascat.asset import Asset, Part
from fascat.options import OptimizeOptions


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
    )
    if targets is not None and sum(targets.values()) > (options.target_triangles or 0):
        result.report.add_warning(
            "target_triangles is lower than the number of non-empty unique meshes; using one triangle per mesh"
        )
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.simplify:
            target = targets.get(part.id) if targets is not None else None
            mesh = mesh.simplify(target_triangles=target, ratio=None if target is not None else options.ratio)
            mesh.validate()
        if options.optimize_buffers:
            mesh = mesh.optimize_buffers()
            mesh.validate()
        mesh = mesh.repair()
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    return result


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

    minimum_total = len(eligible)
    if target_triangles <= minimum_total:
        return {part_id: 1 for part_id, _triangles in eligible}

    targets: dict[str, int] = {}
    remainders: list[tuple[float, int, str]] = []
    assigned = 0
    exact_targets: dict[str, float] = {}
    for part_id, triangle_count in eligible:
        exact = target_triangles * (triangle_count / total_triangles)
        exact_targets[part_id] = exact
        base = max(1, int(math.floor(exact)))
        targets[part_id] = base
        assigned += base
        remainders.append((exact - base, triangle_count, part_id))

    while assigned > target_triangles:
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

    remaining = max(0, target_triangles - assigned)
    for _remainder, _triangle_count, part_id in sorted(remainders, reverse=True)[:remaining]:
        targets[part_id] += 1
    return targets


def _selected_triangle_count(parts: dict[str, Part], selected_part_ids: set[str] | None) -> int:
    return sum(
        part.mesh.triangle_count
        for part_id, part in parts.items()
        if (selected_part_ids is None or part_id in selected_part_ids) and part.mesh is not None
    )
