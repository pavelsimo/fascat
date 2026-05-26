from __future__ import annotations

from fascat.asset import Asset, Part
from fascat.options import OptimizeOptions


def optimize_asset(asset: Asset, options: OptimizeOptions) -> Asset:
    result = asset.copy(keep_source=True)
    if not options.preserve_instances:
        result = _duplicate_parts_per_occurrence(result)
    total_triangles = result.triangle_count
    for part in result.parts.values():
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.simplify:
            target = _target_for_part(mesh.triangle_count, total_triangles, options.target_triangles)
            mesh = mesh.simplify(target_triangles=target, ratio=None if target is not None else options.ratio)
        if options.optimize_buffers:
            mesh = mesh.optimize_buffers()
        mesh = mesh.repair()
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    return result


def _duplicate_parts_per_occurrence(asset: Asset) -> Asset:
    parts: dict[str, Part] = {}
    counters: dict[str, int] = {}
    for node in asset.root.walk():
        if node.part_id is None or node.part_id not in asset.parts:
            continue
        source_id = node.part_id
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


def _target_for_part(part_triangles: int, total_triangles: int, target_triangles: int | None) -> int | None:
    if target_triangles is None or total_triangles <= target_triangles or total_triangles == 0:
        return None
    share = part_triangles / total_triangles
    return max(1, int(round(target_triangles * share)))
