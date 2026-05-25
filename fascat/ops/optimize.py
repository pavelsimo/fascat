from __future__ import annotations

from fascat.asset import Asset
from fascat.options import OptimizeOptions


def optimize_asset(asset: Asset, options: OptimizeOptions) -> Asset:
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.simplify:
            target = _target_for_part(mesh.triangle_count, asset.triangle_count, options.target_triangles)
            mesh = mesh.simplify(target_triangles=target, ratio=None if target is not None else options.ratio)
        if options.optimize_buffers:
            mesh = mesh.optimize_buffers()
        mesh = mesh.repair()
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    return result


def _target_for_part(part_triangles: int, total_triangles: int, target_triangles: int | None) -> int | None:
    if target_triangles is None or total_triangles <= target_triangles or total_triangles == 0:
        return None
    share = part_triangles / total_triangles
    return max(1, int(round(target_triangles * share)))
