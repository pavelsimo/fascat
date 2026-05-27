from __future__ import annotations

import numpy as np

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.options import LODOptions


def build_lods(asset: Asset, options: LODOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    screen_coverage = _screen_coverage(options)
    generated_parts = 0
    skipped_parts = 0
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part.lod_meshes = []
        if part.mesh is None:
            skipped_parts += 1
            part.metadata["lod_status"] = "skipped_no_mesh"
            result.report.add_warning(f"LOD generation skipped part without tessellated mesh: {part.name}")
            continue
        generated_parts += 1
        previous_count = part.mesh.triangle_count
        diagonal = _mesh_diagonal(part.mesh)
        for index, ratio in enumerate(options.ratios):
            coverage = screen_coverage[index]
            if options.drop_tiny_parts and diagonal * coverage < options.tiny_part_screen_size:
                lod = _empty_lod(part.mesh)
                lod.metadata = {
                    **lod.metadata,
                    "lod_ratio": f"{ratio:.9g}",
                    "lod_screen_coverage": f"{coverage:.9g}",
                    "lod_omitted": "tiny_part",
                }
                part.lod_meshes.append(lod)
                previous_count = 0
                continue

            lod = part.mesh.simplify(ratio=ratio)
            if lod.triangle_count > previous_count:
                lod = lod.simplify(target_triangles=previous_count)
            lod.metadata = {
                **lod.metadata,
                "lod_ratio": f"{ratio:.9g}",
                "lod_screen_coverage": f"{coverage:.9g}",
                "lod_mode": options.mode,
                "lod_per_part_budget": str(options.per_part_budget).lower(),
            }
            lod.validate()
            previous_count = lod.triangle_count
            part.lod_meshes.append(lod)
        part.metadata = {
            **part.metadata,
            "lod_ratios": ",".join(f"{ratio:.9g}" for ratio in options.ratios),
            "lod_screen_coverage": ",".join(f"{value:.9g}" for value in screen_coverage),
            "lod_mode": options.mode,
            "lod_per_part_budget": str(options.per_part_budget).lower(),
            "lod_drop_tiny_parts": str(options.drop_tiny_parts).lower(),
        }
    result.metadata["lod_mode"] = options.mode
    result.metadata["lod_screen_coverage"] = ",".join(f"{value:.9g}" for value in screen_coverage)
    result.metadata["lod_generated_parts"] = str(generated_parts)
    result.metadata["lod_skipped_no_mesh_parts"] = str(skipped_parts)
    if generated_parts == 0 and skipped_parts:
        result.report.add_warning("LOD generation matched no tessellated mesh-bearing parts")
    if options.validate:
        _validate_lods(result, selected_part_ids=selected_part_ids)
    return result


def _screen_coverage(options: LODOptions) -> tuple[float, ...]:
    if options.screen_coverage is not None:
        return tuple(options.screen_coverage)
    return tuple(0.5 / (index + 1) for index, _ratio in enumerate(options.ratios))


def _mesh_diagonal(mesh: Mesh) -> float:
    mins, maxs = mesh.bounds()
    return float(np.linalg.norm(maxs - mins))


def _empty_lod(mesh: Mesh) -> Mesh:
    return Mesh(
        points=np.empty((0, 3), dtype=np.float64),
        faces=np.empty((0, 3), dtype=np.int64),
        metadata={**mesh.metadata, "lod_omitted": "tiny_part"},
    )


def _validate_lods(asset: Asset, *, selected_part_ids: set[str] | None) -> None:
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None or not part.lod_meshes:
            continue
        counts = [part.mesh.triangle_count, *[mesh.triangle_count for mesh in part.lod_meshes]]
        if counts != sorted(counts, reverse=True):
            raise ValueError(f"LOD triangles are not monotonic for part {part.id}")
        for lod in part.lod_meshes:
            metrics = lod.quality_metrics()
            if metrics["degenerate_triangles"]:
                asset.report.add_warning(f"LOD contains degenerate triangles for part {part.id}")
