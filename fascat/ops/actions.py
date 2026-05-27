from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import (
    BakeMaterialOptions,
    DecimateOptions,
    LODGeneratorOptions,
    LODOptions,
    OptimizeOptions,
    RemoveHolesOptions,
    RemoveOccludedOptions,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def bake_materials_asset(
    asset: Asset,
    options: BakeMaterialOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    result = asset.copy(keep_source=True)
    part_ids = _selected_mesh_part_ids(result, selected_part_ids)
    if not part_ids:
        result.report.add_warning("bake_materials matched no mesh-bearing parts")
        return result
    result.report.add_warning("bake_materials creates a flat merged material; texture image baking is not implemented")

    source_material_ids = sorted(
        {
            material_id
            for part_id in part_ids
            for material_id in result.parts[part_id].material_ids
            if material_id in result.materials
        }
    )
    baked_id = _unique_material_id(result.materials, "baked_material")
    baked = _baked_material(result, baked_id, source_material_ids, options)
    result.materials[baked.id] = baked

    for part_id in part_ids:
        part = result.parts[part_id]
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.force_uv_generation or options.uv_channel not in mesh.uvs:
            mesh = mesh.box_uv(options.uv_channel)
        mesh.metadata = {
            **mesh.metadata,
            "baked_material": baked.id,
            "baked_maps": ",".join(options.bake),
            "baked_maps_resolution": str(options.maps_resolution),
            "baked_uv_channel": str(options.uv_channel),
            "baked_padding": str(options.padding),
        }
        if options.merge_output:
            part.material_ids = [baked.id]
            mesh.material_indices = None
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "baked_material": baked.id,
            "source_material_ids": ",".join(source_material_ids),
        }
        part.fingerprint = mesh.fingerprint()

    if options.merge_output:
        _drop_unreferenced_materials(result)
    result.metadata["baked_material_count"] = "1"
    result.metadata["baked_source_material_count"] = str(len(source_material_ids))
    return result


def decimate_asset(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    if options.budget_scope == "selection":
        from fascat.ops.optimize import optimize_asset

        result = optimize_asset(
            asset,
            OptimizeOptions(
                target_triangles=options.target_triangles,
                ratio=_decimate_ratio(options),
                preserve_instances=True,
                simplify=True,
                optimize_buffers=True,
                preserve_hard_edges=True,
                hard_edge_angle=options.normal_tolerance,
                preserve_holes=options.protect_topology,
                preserve_material_boundaries=True,
                preserve_uv_seams=True,
                preserve_silhouette=options.protect_topology,
            ),
            selected_part_ids=selected_part_ids,
        )
        if options.criterion == "quality":
            result.report.add_warning(
                "decimate quality criterion uses a tolerance-derived target ratio; "
                "error-bounded simplification is not implemented"
            )
        _enforce_triangle_budget(result, options, selected_part_ids=selected_part_ids)
        return result

    result = asset.copy(keep_source=True)
    ratio = _decimate_ratio(options)
    if options.criterion == "quality":
        result.report.add_warning(
            "decimate quality criterion uses a tolerance-derived target ratio; "
            "error-bounded simplification is not implemented"
        )
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        target = options.target_triangles
        if target is not None:
            target = min(target, part.mesh.triangle_count)
        mesh = part.mesh.simplify(
            target_triangles=target,
            ratio=None if target is not None else ratio,
            preserve_hard_edges=True,
            hard_edge_angle=options.normal_tolerance,
            preserve_holes=options.protect_topology,
            preserve_material_boundaries=True,
            preserve_uv_seams=True,
            preserve_silhouette=options.protect_topology,
        )
        mesh = mesh.optimize_buffers().repair()
        target_budget = target if target is not None else _ratio_target(part.mesh, ratio)
        if target_budget is not None and mesh.triangle_count > target_budget:
            mesh = _sample_mesh_faces(mesh, target_budget).compute_normals()
        mesh.metadata = {
            **mesh.metadata,
            "decimate_criterion": options.criterion,
            "decimate_budget_scope": options.budget_scope,
        }
        mesh.validate()
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    return result


def remove_holes_asset(
    asset: Asset,
    options: RemoveHolesOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    result = asset.copy(keep_source=True)
    if options.prefer_brep:
        result.report.add_warning("BREP hole removal is not implemented; using mesh boundary-fill fallback")
    else:
        result.report.add_warning(
            "remove_holes uses mesh boundary-fill fallback; BREP hole classification is not implemented"
        )
    if not (options.through and options.blind and options.surface):
        result.report.add_warning(
            "mesh hole removal cannot classify through, blind, or surface holes; enabled hole types are recorded only"
        )
    removed_count = 0
    diameters: list[float] = []
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        mesh, stats = _fill_small_holes(part.mesh, max_diameter=options.max_diameter)
        if stats.count == 0:
            continue
        removed_count += stats.count
        diameters.extend(stats.diameters)
        mesh = mesh.compute_normals()
        mesh.validate()
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "removed_holes": str(stats.count),
            "removed_hole_types": _enabled_hole_types(options),
        }
        part.fingerprint = mesh.fingerprint()

    result.metadata["removed_holes"] = str(removed_count)
    if diameters:
        result.metadata["removed_hole_min_diameter"] = f"{min(diameters):.9g}"
        result.metadata["removed_hole_max_diameter"] = f"{max(diameters):.9g}"
    if options.prefer_brep and removed_count:
        result.metadata["remove_holes_backend"] = "mesh"
    return result


def remove_occluded_asset(
    asset: Asset,
    options: RemoveOccludedOptions,
    *,
    selected_node_ids: set[str],
) -> Asset:
    result = asset.copy(keep_source=True)
    result.report.add_warning(
        "remove_occluded uses part-level AABB containment fallback; true visibility sampling is not implemented"
    )
    if options.level != "parts":
        result.report.add_warning(
            f"occlusion level {options.level} uses part-level AABB containment fallback; "
            "submesh and triangle removal are not implemented"
        )
    if options.strategy != "conservative" or options.hemi_evaluation:
        result.report.add_warning(
            "remove_occluded uses conservative AABB containment; strategy and hemi_evaluation "
            "do not change visibility sampling"
        )
    occurrences = _world_occurrences(result)
    selected_occurrences = [item for item in occurrences if item.node.id in selected_node_ids]
    removed_node_ids: set[str] = set()
    for candidate in selected_occurrences:
        if _preserve_candidate_cavity(result, candidate, options):
            continue
        for occluder in occurrences:
            if occluder.node.id == candidate.node.id:
                continue
            if not options.consider_transparency_opaque and _part_is_transparent(result, occluder.part_id):
                continue
            if occluder.volume <= candidate.volume * 1.001:
                continue
            if _bbox_contains(occluder.bounds_min, occluder.bounds_max, candidate.bounds_min, candidate.bounds_max):
                removed_node_ids.add(candidate.node.id)
                break

    if removed_node_ids:
        _remove_part_nodes(result.root, removed_node_ids)
        _drop_unreferenced_parts(result)
    result.metadata["removed_occluded_nodes"] = str(len(removed_node_ids))
    result.metadata["occlusion_strategy"] = options.strategy
    result.metadata["occlusion_level"] = options.level
    return result


def run_lod_generators_asset(
    asset: Asset,
    options: LODGeneratorOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    from fascat.ops.lod import build_lods

    ratios = tuple(level.target_ratio for level in options.levels)
    result = build_lods(asset, LODOptions(ratios=ratios), selected_part_ids=selected_part_ids)
    coverage = ",".join(f"{level.screen_coverage:.9g}" for level in options.levels)
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.lod_meshes:
            part.metadata = {
                **part.metadata,
                "lod_generator_preset": options.preset,
                "lod_screen_coverage": coverage,
                "lod_output": options.output,
            }
    result.metadata["lod_generator_preset"] = options.preset
    result.metadata["lod_generator_output"] = options.output
    if options.validate:
        _validate_lod_monotonicity(result, selected_part_ids=selected_part_ids, allow=options.allow_non_monotonic)
    return result


@dataclass(frozen=True)
class _HoleFillStats:
    count: int
    diameters: tuple[float, ...]


@dataclass(frozen=True)
class _WorldOccurrence:
    node: Node
    part_id: str
    bounds_min: FloatArray
    bounds_max: FloatArray
    volume: float


def _selected_mesh_part_ids(asset: Asset, selected_part_ids: set[str] | None) -> set[str]:
    return {
        part.id
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    }


def _baked_material(
    asset: Asset,
    material_id: str,
    source_material_ids: list[str],
    options: BakeMaterialOptions,
) -> Material:
    materials = [asset.materials[material_id] for material_id in source_material_ids if material_id in asset.materials]
    if not materials:
        base_color = (1.0, 1.0, 1.0, 1.0)
        metallic = 0.0
        roughness = 0.5
        opacity = 1.0
    else:
        base_color_values = np.asarray([material.base_color for material in materials], dtype=np.float64)
        base_color = cast(tuple[float, float, float, float], tuple(base_color_values.mean(axis=0).tolist()))
        metallic = float(np.mean([material.metallic for material in materials]))
        roughness = float(np.mean([material.roughness for material in materials]))
        opacity = float(np.mean([material.opacity for material in materials]))
    return Material(
        id=material_id,
        name="Baked Material",
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        opacity=opacity,
        metadata={
            "baked": "true",
            "baked_maps": ",".join(options.bake),
            "maps_resolution": str(options.maps_resolution),
            "padding": str(options.padding),
            "source_material_ids": ",".join(source_material_ids),
        },
    )


def _decimate_ratio(options: DecimateOptions) -> float | None:
    if options.target_triangles is not None:
        return None
    if options.target_ratio is not None:
        return options.target_ratio
    if options.criterion == "quality":
        tolerance = max(options.surface_tolerance or 0.0, options.line_tolerance or 0.0, options.uv_tolerance or 0.0)
        return max(0.1, min(0.95, 1.0 - tolerance))
    return 0.5


def _enforce_triangle_budget(
    asset: Asset,
    options: DecimateOptions,
    *,
    selected_part_ids: set[str] | None,
) -> None:
    eligible = [
        part
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    ]
    if not eligible:
        return
    if options.target_triangles is not None:
        current_total = sum(cast(Mesh, part.mesh).triangle_count for part in eligible)
        if current_total <= options.target_triangles:
            return
        assigned = 0
        budgets: dict[str, int] = {}
        for part in eligible:
            mesh = cast(Mesh, part.mesh)
            exact = options.target_triangles * (mesh.triangle_count / current_total)
            budget = max(1, min(mesh.triangle_count, int(round(exact))))
            budgets[part.id] = budget
            assigned += budget
        while assigned > options.target_triangles:
            reducible = [part_id for part_id, budget in budgets.items() if budget > 1]
            if not reducible:
                break
            part_id = max(reducible, key=lambda item: budgets[item])
            budgets[part_id] -= 1
            assigned -= 1
        for part in eligible:
            mesh = cast(Mesh, part.mesh)
            if mesh.triangle_count > budgets[part.id]:
                part.mesh = _sample_mesh_faces(mesh, budgets[part.id]).compute_normals()
                part.fingerprint = part.mesh.fingerprint()
        return

    ratio = _decimate_ratio(options)
    for part in eligible:
        mesh = cast(Mesh, part.mesh)
        target = _ratio_target(mesh, ratio)
        if target is not None and mesh.triangle_count > target:
            part.mesh = _sample_mesh_faces(mesh, target).compute_normals()
            part.fingerprint = part.mesh.fingerprint()


def _ratio_target(mesh: Mesh | None, ratio: float | None) -> int | None:
    if mesh is None or ratio is None:
        return None
    return max(1, int(round(mesh.triangle_count * ratio)))


def _sample_mesh_faces(mesh: Mesh, target_triangles: int) -> Mesh:
    target = max(1, min(target_triangles, mesh.triangle_count))
    if target >= mesh.triangle_count:
        return mesh.copy()
    face_indices = np.unique(np.linspace(0, mesh.triangle_count - 1, target, dtype=np.int64))
    while face_indices.shape[0] < target:
        missing = [index for index in range(mesh.triangle_count) if index not in set(face_indices.astype(int).tolist())]
        face_indices = np.sort(np.concatenate([face_indices, np.asarray(missing[: target - face_indices.shape[0]])]))
    return _slice_faces(mesh, face_indices).remove_unreferenced_vertices()


def _slice_faces(mesh: Mesh, face_indices: IntArray) -> Mesh:
    face_lookup = {int(face_index): local_index for local_index, face_index in enumerate(face_indices.tolist())}
    return Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces[face_indices].copy(),
        normals=None if mesh.normals is None else mesh.normals.copy(),
        tangents=None if mesh.tangents is None else mesh.tangents.copy(),
        uvs={channel: values.copy() for channel, values in mesh.uvs.items()},
        material_indices=None if mesh.material_indices is None else mesh.material_indices[face_indices].copy(),
        face_groups={
            name: np.asarray(
                [face_lookup[int(face_index)] for face_index in group.tolist() if int(face_index) in face_lookup],
                dtype=np.int64,
            )
            for name, group in mesh.face_groups.items()
        },
        metadata=dict(mesh.metadata),
    )


def _enabled_hole_types(options: RemoveHolesOptions) -> str:
    values = []
    if options.through:
        values.append("through")
    if options.blind:
        values.append("blind")
    if options.surface:
        values.append("surface")
    return ",".join(values)


def _fill_small_holes(mesh: Mesh, *, max_diameter: float | None) -> tuple[Mesh, _HoleFillStats]:
    loops = _boundary_loops(mesh)
    fill_faces: list[list[int]] = []
    diameters: list[float] = []
    for loop in loops:
        if len(loop) < 3 or len(loop) > 8:
            continue
        diameter = _loop_diameter(mesh.points, loop)
        if max_diameter is not None and diameter > max_diameter:
            continue
        anchor = loop[0]
        for index in range(1, len(loop) - 1):
            fill_faces.append([anchor, loop[index], loop[index + 1]])
        diameters.append(diameter)

    if not fill_faces:
        return mesh.copy(), _HoleFillStats(count=0, diameters=())

    filled = mesh.copy()
    filled.faces = np.vstack([mesh.faces, np.asarray(fill_faces, dtype=np.int64)])
    if mesh.material_indices is not None:
        fill_material = int(mesh.material_indices[0]) if mesh.material_indices.size else 0
        filled.material_indices = np.concatenate(
            [mesh.material_indices.copy(), np.full(len(fill_faces), fill_material, dtype=np.int64)]
        )
    return filled, _HoleFillStats(count=len(diameters), diameters=tuple(diameters))


def _boundary_loops(mesh: Mesh) -> list[list[int]]:
    if mesh.triangle_count == 0:
        return []
    edge_counts: dict[tuple[int, int], int] = {}
    directed: list[tuple[int, int]] = []
    for face in mesh.faces.astype(int).tolist():
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(start, end)
            edge_counts[key] = edge_counts.get(key, 0) + 1
            directed.append((start, end))

    adjacency: dict[int, list[int]] = {}
    for start, end in directed:
        if edge_counts[_edge_key(start, end)] != 1:
            continue
        adjacency.setdefault(start, []).append(end)
        adjacency.setdefault(end, []).append(start)

    loops: list[list[int]] = []
    visited: set[tuple[int, int]] = set()
    for start, neighbors in adjacency.items():
        for neighbor in neighbors:
            edge = _edge_key(start, neighbor)
            if edge in visited:
                continue
            loop = [start, neighbor]
            visited.add(edge)
            previous = start
            current = neighbor
            while current != start:
                candidates = [item for item in adjacency.get(current, []) if item != previous]
                if not candidates:
                    break
                next_vertex = candidates[0]
                next_edge = _edge_key(current, next_vertex)
                if next_edge in visited and next_vertex != start:
                    break
                if next_vertex == start:
                    visited.add(next_edge)
                    break
                loop.append(next_vertex)
                visited.add(next_edge)
                previous, current = current, next_vertex
            if len(loop) >= 3:
                loops.append(loop)
    return loops


def _edge_key(start: int, end: int) -> tuple[int, int]:
    return (start, end) if start <= end else (end, start)


def _loop_diameter(points: FloatArray, loop: list[int]) -> float:
    loop_points = points[np.asarray(loop, dtype=np.int64)]
    if loop_points.shape[0] < 2:
        return 0.0
    delta = loop_points[:, None, :] - loop_points[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    return float(distances.max())


def _world_occurrences(asset: Asset) -> list[_WorldOccurrence]:
    occurrences: list[_WorldOccurrence] = []

    def walk(node: Node, world: FloatArray) -> None:
        current = world @ node.transform
        if node.part_id is not None and node.part_id in asset.parts:
            part = asset.parts[node.part_id]
            if part.mesh is not None:
                mins, maxs = part.mesh.bounds()
                world_min, world_max = _transform_bounds(mins, maxs, current)
                volume = float(np.prod(np.maximum(world_max - world_min, 0.0)))
                occurrences.append(
                    _WorldOccurrence(
                        node=node,
                        part_id=node.part_id,
                        bounds_min=world_min,
                        bounds_max=world_max,
                        volume=volume,
                    )
                )
        for child in node.children:
            walk(child, current)

    walk(asset.root, np.eye(4, dtype=np.float64))
    return occurrences


def _transform_bounds(mins: FloatArray, maxs: FloatArray, transform: FloatArray) -> tuple[FloatArray, FloatArray]:
    corners = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([corners, np.ones(corners.shape[0], dtype=np.float64)])
    transformed = (transform @ homogeneous.T).T[:, :3]
    return cast(FloatArray, transformed.min(axis=0)), cast(FloatArray, transformed.max(axis=0))


def _bbox_contains(outer_min: FloatArray, outer_max: FloatArray, inner_min: FloatArray, inner_max: FloatArray) -> bool:
    epsilon = 1e-9
    return bool(np.all(inner_min >= outer_min - epsilon) and np.all(inner_max <= outer_max + epsilon))


def _preserve_candidate_cavity(asset: Asset, candidate: _WorldOccurrence, options: RemoveOccludedOptions) -> bool:
    if not options.preserve_cavities:
        return False
    volume_m3 = candidate.volume * asset.meters_per_unit**3
    return volume_m3 >= options.minimum_cavity_volume_m3


def _part_is_transparent(asset: Asset, part_id: str) -> bool:
    part = asset.parts.get(part_id)
    if part is None:
        return False
    return any(
        asset.materials[material_id].opacity < 1.0
        for material_id in part.material_ids
        if material_id in asset.materials
    )


def _remove_part_nodes(node: Node, remove_node_ids: set[str]) -> bool:
    kept: list[Node] = []
    for child in node.children:
        keep_child = _remove_part_nodes(child, remove_node_ids)
        if child.id in remove_node_ids:
            child.part_id = None
            if child.children:
                kept.append(child)
            continue
        if keep_child:
            kept.append(child)
    node.children = kept
    return node.part_id is not None or bool(node.children)


def _drop_unreferenced_parts(asset: Asset) -> None:
    referenced = {node.part_id for node in asset.root.walk() if node.part_id is not None}
    asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id in referenced}


def _drop_unreferenced_materials(asset: Asset) -> None:
    referenced = {material_id for part in asset.parts.values() for material_id in part.material_ids}
    asset.materials = {
        material_id: material for material_id, material in asset.materials.items() if material_id in referenced
    }


def _validate_lod_monotonicity(
    asset: Asset,
    *,
    selected_part_ids: set[str] | None,
    allow: bool,
) -> None:
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None or not part.lod_meshes:
            continue
        counts = [part.mesh.triangle_count, *[mesh.triangle_count for mesh in part.lod_meshes]]
        if counts != sorted(counts, reverse=True):
            message = f"LOD triangles are not monotonic for part {part.id}"
            if allow:
                asset.report.add_warning(message)
            else:
                raise ValueError(message)


def _unique_material_id(materials: dict[str, Material], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in materials:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
