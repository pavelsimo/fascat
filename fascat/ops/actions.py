from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
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
    if options.level != "parts":
        _isolate_selected_occurrence_parts(result, selected_node_ids)
    result.report.add_warning(
        "remove_occluded uses deterministic sampled visibility; thin occluders may require higher precision"
    )
    occurrences = _world_occurrences(result)
    selected_occurrences = [item for item in occurrences if item.node.id in selected_node_ids]
    directions = _occlusion_directions(options)
    ray_distance = _occlusion_ray_distance(occurrences)
    removed_node_ids: set[str] = set()
    trims: list[_OcclusionTrim] = []
    for candidate in selected_occurrences:
        if _preserve_candidate_cavity(result, candidate, options):
            continue
        occluders = _candidate_occluders(result, candidate, occurrences, options)
        if options.level == "parts":
            if not _occurrence_has_visible_sample(candidate, occluders, directions, ray_distance, options):
                removed_node_ids.add(candidate.node.id)
            continue
        visible_faces = _visible_face_mask(candidate, occluders, directions, ray_distance)
        part = result.parts.get(candidate.part_id)
        mesh = None if part is None else part.mesh
        if mesh is None or mesh.triangle_count == 0 or bool(np.all(visible_faces)):
            continue
        if options.level == "submeshes":
            keep_mask = _submesh_keep_mask(mesh, visible_faces)
        else:
            keep_mask = _expand_face_mask(mesh, visible_faces, options.neighbors_preservation)
        if bool(np.all(keep_mask)):
            continue
        keep_faces = np.flatnonzero(keep_mask)
        removed_faces = int(mesh.triangle_count - keep_faces.shape[0])
        if keep_faces.size == 0:
            removed_node_ids.add(candidate.node.id)
        else:
            trims.append(
                _OcclusionTrim(
                    node_id=candidate.node.id,
                    part_id=candidate.part_id,
                    keep_faces=keep_faces.astype(np.int64),
                    removed_faces=removed_faces,
                )
            )

    if removed_node_ids:
        _remove_part_nodes(result.root, removed_node_ids)
    removed_faces_total = _removed_node_triangle_count(result, selected_occurrences, removed_node_ids)
    removed_faces_total += _apply_occlusion_trims(result, trims, removed_node_ids, options)
    if removed_node_ids or removed_faces_total:
        _drop_unreferenced_parts(result)
    result.metadata["removed_occluded_nodes"] = str(len(removed_node_ids))
    result.metadata["removed_occluded_triangles"] = str(removed_faces_total)
    result.metadata["occlusion_strategy"] = options.strategy
    result.metadata["occlusion_level"] = options.level
    result.metadata["occlusion_direction_count"] = str(len(directions))
    result.metadata["occlusion_hemi_evaluation"] = str(options.hemi_evaluation).lower()
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
    world_points: FloatArray
    faces: IntArray
    bounds_min: FloatArray
    bounds_max: FloatArray
    volume: float


@dataclass(frozen=True)
class _OcclusionTrim:
    node_id: str
    part_id: str
    keep_faces: IntArray
    removed_faces: int


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


def _isolate_selected_occurrence_parts(asset: Asset, selected_node_ids: set[str]) -> None:
    references: dict[str, list[Node]] = {}
    for node in asset.root.walk():
        if node.part_id is not None and node.part_id in asset.parts:
            references.setdefault(node.part_id, []).append(node)

    for part_id, nodes in list(references.items()):
        selected_nodes = [node for node in nodes if node.id in selected_node_ids]
        if not selected_nodes:
            continue
        for node in selected_nodes[:-1] if len(selected_nodes) == len(nodes) else selected_nodes:
            new_part_id = _unique_part_id(asset.parts, f"{part_id}_{node.id}")
            part = asset.parts[part_id].copy(keep_source=True)
            part.id = new_part_id
            part.metadata = {**part.metadata, "source_part_id": part_id}
            asset.parts[new_part_id] = part
            node.part_id = new_part_id


def _candidate_occluders(
    asset: Asset,
    candidate: _WorldOccurrence,
    occurrences: list[_WorldOccurrence],
    options: RemoveOccludedOptions,
) -> list[_WorldOccurrence]:
    return [
        occluder
        for occluder in occurrences
        if occluder.node.id != candidate.node.id
        and (options.consider_transparency_opaque or not _part_is_transparent(asset, occluder.part_id))
    ]


def _occlusion_directions(options: RemoveOccludedOptions) -> list[FloatArray]:
    if options.strategy == "conservative":
        vectors = [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
    elif options.strategy == "exterior":
        vectors = [(x, y, z) for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)] + [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ]
    else:
        vectors = [
            (x, y, z)
            for x in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for z in (-1.0, 0.0, 1.0)
            if (x, y, z) != (0.0, 0.0, 0.0)
        ]
    if options.hemi_evaluation:
        vectors = [vector for vector in vectors if vector[2] >= 0.0]
    directions: list[FloatArray] = []
    for vector in vectors:
        direction = np.asarray(vector, dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length > 0.0:
            directions.append(direction / length)
    return directions


def _occlusion_ray_distance(occurrences: list[_WorldOccurrence]) -> float:
    if not occurrences:
        return 1.0
    bounds_min = np.vstack([occurrence.bounds_min for occurrence in occurrences]).min(axis=0)
    bounds_max = np.vstack([occurrence.bounds_max for occurrence in occurrences]).max(axis=0)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    return max(diagonal * 2.0, 1.0)


def _occurrence_has_visible_sample(
    candidate: _WorldOccurrence,
    occluders: list[_WorldOccurrence],
    directions: list[FloatArray],
    ray_distance: float,
    options: RemoveOccludedOptions,
) -> bool:
    samples = _occurrence_visibility_samples(candidate, options.precision)
    if samples.size == 0 or not occluders:
        return True
    return any(_sample_is_visible(sample, occluders, directions, ray_distance) for sample in samples)


def _occurrence_visibility_samples(candidate: _WorldOccurrence, precision: int) -> FloatArray:
    face_samples = _face_centers(candidate)
    if face_samples.shape[0] > precision:
        indices = np.unique(np.linspace(0, face_samples.shape[0] - 1, precision, dtype=np.int64))
        face_samples = face_samples[indices]
    mins = candidate.bounds_min
    maxs = candidate.bounds_max
    box_samples = np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], maxs[1], maxs[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], mins[2]],
            [maxs[0], maxs[1], maxs[2]],
            [(mins[0] + maxs[0]) * 0.5, mins[1], (mins[2] + maxs[2]) * 0.5],
            [(mins[0] + maxs[0]) * 0.5, maxs[1], (mins[2] + maxs[2]) * 0.5],
            [mins[0], (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5],
            [maxs[0], (mins[1] + maxs[1]) * 0.5, (mins[2] + maxs[2]) * 0.5],
            [(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, mins[2]],
            [(mins[0] + maxs[0]) * 0.5, (mins[1] + maxs[1]) * 0.5, maxs[2]],
        ],
        dtype=np.float64,
    )
    if face_samples.size == 0:
        return box_samples
    return cast(FloatArray, np.vstack([face_samples, box_samples]))


def _visible_face_mask(
    candidate: _WorldOccurrence,
    occluders: list[_WorldOccurrence],
    directions: list[FloatArray],
    ray_distance: float,
) -> NDArray[np.bool_]:
    centers = _face_centers(candidate)
    if centers.size == 0 or not occluders:
        return np.ones(candidate.faces.shape[0], dtype=np.bool_)
    return np.asarray(
        [_sample_is_visible(center, occluders, directions, ray_distance) for center in centers],
        dtype=np.bool_,
    )


def _face_centers(occurrence: _WorldOccurrence) -> FloatArray:
    if occurrence.faces.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    triangles = occurrence.world_points[occurrence.faces]
    return cast(FloatArray, triangles.mean(axis=1))


def _sample_is_visible(
    sample: FloatArray,
    occluders: list[_WorldOccurrence],
    directions: list[FloatArray],
    ray_distance: float,
) -> bool:
    for direction in directions:
        origin = sample + direction * ray_distance
        if not _segment_blocked(origin, sample, occluders):
            return True
    return False


def _segment_blocked(start: FloatArray, end: FloatArray, occluders: list[_WorldOccurrence]) -> bool:
    for occluder in occluders:
        if not _segment_intersects_bounds(start, end, occluder.bounds_min, occluder.bounds_max):
            continue
        if _segment_intersects_mesh(start, end, occluder.world_points, occluder.faces):
            return True
    return False


def _segment_intersects_mesh(start: FloatArray, end: FloatArray, points: FloatArray, faces: IntArray) -> bool:
    for face in faces.astype(int).tolist():
        triangle = points[np.asarray(face, dtype=np.int64)]
        hit = _segment_triangle_t(start, end, triangle)
        if hit is not None and 1e-8 < hit < 1.0 - 1e-8:
            return True
    return False


def _segment_triangle_t(start: FloatArray, end: FloatArray, triangle: FloatArray) -> float | None:
    epsilon = 1e-12
    direction = end - start
    edge1 = triangle[1] - triangle[0]
    edge2 = triangle[2] - triangle[0]
    pvec = np.cross(direction, edge2)
    determinant = float(np.dot(edge1, pvec))
    if abs(determinant) <= epsilon:
        return None
    inverse = 1.0 / determinant
    tvec = start - triangle[0]
    u = float(np.dot(tvec, pvec)) * inverse
    if u < -epsilon or u > 1.0 + epsilon:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inverse
    if v < -epsilon or u + v > 1.0 + epsilon:
        return None
    t = float(np.dot(edge2, qvec)) * inverse
    if t < -epsilon or t > 1.0 + epsilon:
        return None
    return t


def _segment_intersects_bounds(
    start: FloatArray,
    end: FloatArray,
    bounds_min: FloatArray,
    bounds_max: FloatArray,
) -> bool:
    epsilon = 1e-12
    direction = end - start
    tmin = 0.0
    tmax = 1.0
    for axis in range(3):
        if abs(float(direction[axis])) <= epsilon:
            if start[axis] < bounds_min[axis] - epsilon or start[axis] > bounds_max[axis] + epsilon:
                return False
            continue
        inverse = 1.0 / float(direction[axis])
        near = (float(bounds_min[axis]) - float(start[axis])) * inverse
        far = (float(bounds_max[axis]) - float(start[axis])) * inverse
        if near > far:
            near, far = far, near
        tmin = max(tmin, near)
        tmax = min(tmax, far)
        if tmin > tmax:
            return False
    return True


def _submesh_keep_mask(mesh: Mesh, visible_faces: NDArray[np.bool_]) -> NDArray[np.bool_]:
    if mesh.material_indices is None:
        return cast(NDArray[np.bool_], visible_faces.copy())
    keep = np.zeros(mesh.triangle_count, dtype=np.bool_)
    for material_index in np.unique(mesh.material_indices):
        group = mesh.material_indices == material_index
        if bool(np.any(visible_faces[group])):
            keep[group] = True
    return keep


def _expand_face_mask(mesh: Mesh, visible_faces: NDArray[np.bool_], rings: int) -> NDArray[np.bool_]:
    keep = visible_faces.copy()
    if rings <= 0 or not bool(np.any(keep)):
        return cast(NDArray[np.bool_], keep)
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(mesh.faces.astype(int).tolist()):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces.setdefault(_edge_key(start, end), []).append(face_index)
    neighbors: list[set[int]] = [set() for _ in range(mesh.triangle_count)]
    for face_indices in edge_faces.values():
        if len(face_indices) < 2:
            continue
        for face_index in face_indices:
            neighbors[face_index].update(index for index in face_indices if index != face_index)
    frontier = set(np.flatnonzero(keep).astype(int).tolist())
    for _ in range(rings):
        next_frontier: set[int] = set()
        for face_index in frontier:
            next_frontier.update(neighbors[face_index])
        next_frontier.difference_update(np.flatnonzero(keep).astype(int).tolist())
        if not next_frontier:
            break
        keep[np.asarray(sorted(next_frontier), dtype=np.int64)] = True
        frontier = next_frontier
    return cast(NDArray[np.bool_], keep)


def _apply_occlusion_trims(
    asset: Asset,
    trims: list[_OcclusionTrim],
    removed_node_ids: set[str],
    options: RemoveOccludedOptions,
) -> int:
    removed_faces_total = 0
    for trim in trims:
        if trim.node_id in removed_node_ids:
            continue
        part = asset.parts.get(trim.part_id)
        if part is None or part.mesh is None:
            continue
        mesh = _slice_faces(part.mesh, trim.keep_faces).remove_unreferenced_vertices()
        if mesh.triangle_count:
            mesh = mesh.compute_normals()
        _compact_material_slots(part, mesh)
        mesh.metadata = {
            **mesh.metadata,
            "occlusion_level": options.level,
            "occlusion_removed_faces": str(trim.removed_faces),
        }
        mesh.validate()
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "occlusion_level": options.level,
            "occlusion_removed_faces": str(trim.removed_faces),
        }
        part.fingerprint = mesh.fingerprint()
        removed_faces_total += trim.removed_faces
    return removed_faces_total


def _removed_node_triangle_count(
    asset: Asset,
    selected_occurrences: list[_WorldOccurrence],
    removed_node_ids: set[str],
) -> int:
    total = 0
    for occurrence in selected_occurrences:
        if occurrence.node.id not in removed_node_ids:
            continue
        part = asset.parts.get(occurrence.part_id)
        if part is not None and part.mesh is not None:
            total += part.mesh.triangle_count
    return total


def _compact_material_slots(part: Part, mesh: Mesh) -> None:
    if mesh.material_indices is None:
        return
    material_ids = part.material_ids
    used = sorted({int(index) for index in mesh.material_indices.astype(int).tolist()})
    if not used or any(index < 0 or index >= len(material_ids) for index in used):
        return
    remap = {old: new for new, old in enumerate(used)}
    mesh.material_indices = np.asarray([remap[int(index)] for index in mesh.material_indices], dtype=np.int64)
    part.material_ids = [material_ids[index] for index in used]


def _world_occurrences(asset: Asset) -> list[_WorldOccurrence]:
    occurrences: list[_WorldOccurrence] = []

    def walk(node: Node, world: FloatArray) -> None:
        current = world @ node.transform
        if node.part_id is not None and node.part_id in asset.parts:
            part = asset.parts[node.part_id]
            if part.mesh is not None:
                world_points = _transform_points(part.mesh.points, current)
                if world_points.shape[0] == 0:
                    mins, maxs = part.mesh.bounds()
                    world_min, world_max = _transform_bounds(mins, maxs, current)
                else:
                    world_min = cast(FloatArray, world_points.min(axis=0))
                    world_max = cast(FloatArray, world_points.max(axis=0))
                volume = float(np.prod(np.maximum(world_max - world_min, 0.0)))
                occurrences.append(
                    _WorldOccurrence(
                        node=node,
                        part_id=node.part_id,
                        world_points=world_points,
                        faces=part.mesh.faces.copy(),
                        bounds_min=world_min,
                        bounds_max=world_max,
                        volume=volume,
                    )
                )
        for child in node.children:
            walk(child, current)

    walk(asset.root, np.eye(4, dtype=np.float64))
    return occurrences


def _transform_points(points: FloatArray, transform: FloatArray) -> FloatArray:
    if points.shape[0] == 0:
        return cast(FloatArray, points.copy())
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return cast(FloatArray, (transform @ homogeneous.T).T[:, :3])


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


def _unique_part_id(parts: dict[str, Part], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in parts:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate
