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
    source_vertices = 0
    source_triangles = 0
    source_mesh_bytes = 0
    added_vertices = 0
    added_triangles = 0
    added_mesh_bytes = 0
    omitted_tiny_parts = 0
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
        part_source_vertices = part.mesh.vertex_count
        part_source_triangles = part.mesh.triangle_count
        part_source_bytes = _mesh_payload_bytes(part.mesh)
        part_lod_vertices = 0
        part_lod_triangles = 0
        part_lod_bytes = 0
        part_omitted_tiny = 0
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
                part_omitted_tiny += 1
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
            part_lod_vertices += lod.vertex_count
            part_lod_triangles += lod.triangle_count
            part_lod_bytes += _mesh_payload_bytes(lod)
        source_vertices += part_source_vertices
        source_triangles += part_source_triangles
        source_mesh_bytes += part_source_bytes
        added_vertices += part_lod_vertices
        added_triangles += part_lod_triangles
        added_mesh_bytes += part_lod_bytes
        omitted_tiny_parts += part_omitted_tiny
        level_vertices = ",".join(str(mesh.vertex_count) for mesh in part.lod_meshes)
        level_triangles = ",".join(str(mesh.triangle_count) for mesh in part.lod_meshes)
        part.metadata = {
            **part.metadata,
            "lod_ratios": ",".join(f"{ratio:.9g}" for ratio in options.ratios),
            "lod_screen_coverage": ",".join(f"{value:.9g}" for value in screen_coverage),
            "lod_mode": options.mode,
            "lod_per_part_budget": str(options.per_part_budget).lower(),
            "lod_drop_tiny_parts": str(options.drop_tiny_parts).lower(),
            "lod_source_vertices": str(part_source_vertices),
            "lod_source_triangles": str(part_source_triangles),
            "lod_source_mesh_bytes": str(part_source_bytes),
            "lod_added_vertices": str(part_lod_vertices),
            "lod_added_triangles": str(part_lod_triangles),
            "lod_added_mesh_bytes": str(part_lod_bytes),
            "lod_chain_vertices": str(part_source_vertices + part_lod_vertices),
            "lod_chain_triangles": str(part_source_triangles + part_lod_triangles),
            "lod_chain_mesh_bytes": str(part_source_bytes + part_lod_bytes),
            "lod_level_vertices": level_vertices,
            "lod_level_triangles": level_triangles,
            "lod_omitted_tiny_part_meshes": str(part_omitted_tiny),
            "lod_triangle_multiplier": _ratio_text(part_source_triangles + part_lod_triangles, part_source_triangles),
            "lod_mesh_byte_multiplier": _ratio_text(part_source_bytes + part_lod_bytes, part_source_bytes),
        }
    result.metadata["lod_mode"] = options.mode
    result.metadata["lod_screen_coverage"] = ",".join(f"{value:.9g}" for value in screen_coverage)
    result.metadata["lod_generated_parts"] = str(generated_parts)
    result.metadata["lod_skipped_no_mesh_parts"] = str(skipped_parts)
    result.metadata["lod_source_vertices"] = str(source_vertices)
    result.metadata["lod_source_triangles"] = str(source_triangles)
    result.metadata["lod_source_mesh_bytes"] = str(source_mesh_bytes)
    result.metadata["lod_added_vertices"] = str(added_vertices)
    result.metadata["lod_added_triangles"] = str(added_triangles)
    result.metadata["lod_added_mesh_bytes"] = str(added_mesh_bytes)
    result.metadata["lod_chain_vertices"] = str(source_vertices + added_vertices)
    result.metadata["lod_chain_triangles"] = str(source_triangles + added_triangles)
    result.metadata["lod_chain_mesh_bytes"] = str(source_mesh_bytes + added_mesh_bytes)
    result.metadata["lod_omitted_tiny_part_meshes"] = str(omitted_tiny_parts)
    result.metadata["lod_triangle_multiplier"] = _ratio_text(source_triangles + added_triangles, source_triangles)
    result.metadata["lod_mesh_byte_multiplier"] = _ratio_text(source_mesh_bytes + added_mesh_bytes, source_mesh_bytes)
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


def _mesh_payload_bytes(mesh: Mesh) -> int:
    total = mesh.points.nbytes + mesh.faces.nbytes
    if mesh.normals is not None:
        total += mesh.normals.nbytes
    if mesh.tangents is not None:
        total += mesh.tangents.nbytes
    if mesh.material_indices is not None:
        total += mesh.material_indices.nbytes
    for uv_values in mesh.uvs.values():
        total += uv_values.nbytes
    for face_group_values in mesh.face_groups.values():
        total += face_group_values.nbytes
    return int(total)


def _ratio_text(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0"
    return f"{(numerator / denominator):.9g}"


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
