from __future__ import annotations

from fascat.asset import Asset
from fascat.options import LODOptions


def build_lods(asset: Asset, options: LODOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part.lod_meshes = []
        if part.mesh is None:
            continue
        previous_count = part.mesh.triangle_count
        for ratio in options.ratios:
            lod = part.mesh.simplify(ratio=ratio)
            if lod.triangle_count > previous_count:
                lod = lod.simplify(target_triangles=previous_count)
            lod.validate()
            previous_count = lod.triangle_count
            part.lod_meshes.append(lod)
    return result
