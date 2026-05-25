from __future__ import annotations

from fascat.asset import Asset
from fascat.options import LODOptions


def build_lods(asset: Asset, options: LODOptions) -> Asset:
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        part.lod_meshes = []
        if part.mesh is None:
            continue
        previous_count = part.mesh.triangle_count
        for ratio in options.ratios:
            lod = part.mesh.simplify(ratio=ratio)
            if lod.triangle_count > previous_count:
                lod = lod.simplify(target_triangles=previous_count)
            previous_count = lod.triangle_count
            part.lod_meshes.append(lod)
    return result
