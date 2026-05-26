from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part, identity_transform
from fascat.mesh import Mesh
from fascat.options import MergeOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class _Occurrence:
    node: Node
    parent: Node
    node_path: str
    parent_path: str
    world_transform: FloatArray
    parent_world_transform: FloatArray
    part: Part


@dataclass(frozen=True)
class _MergeInput:
    occurrence: _Occurrence
    material_id: str | None
    face_indices: IntArray

    @property
    def vertex_count(self) -> int:
        return int(self.occurrence.part.mesh.vertex_count) if self.occurrence.part.mesh is not None else 0


@dataclass(frozen=True)
class _MergeGroup:
    key: tuple[object, ...]
    name: str
    parent: Node
    parent_world_transform: FloatArray
    inputs: tuple[_MergeInput, ...]


def merge_asset(asset: Asset, options: MergeOptions, *, selected_node_ids: set[str]) -> Asset:
    result = asset.copy(keep_source=True)
    occurrences = [
        occurrence
        for occurrence in _walk_occurrences(result)
        if occurrence.node.id in selected_node_ids and occurrence.part.mesh is not None
    ]
    if not occurrences:
        result.report.add_warning("merge matched no mesh-bearing part occurrences")
        return result

    groups = _build_groups(result, occurrences, options)
    merged_parts: dict[str, Part] = {}
    merged_nodes_by_parent: dict[str, list[Node]] = {}
    for group in groups:
        chunks = _split_group(group, options.max_vertices_per_mesh)
        for chunk_index, chunk in enumerate(chunks, start=1):
            part_id = _merged_part_id(result.parts, group.key, chunk_index)
            part_name = group.name if len(chunks) == 1 else f"{group.name} {chunk_index}"
            part = _merge_inputs(part_id, part_name, chunk, group.parent_world_transform, options)
            merged_parts[part.id] = part
            node_id = _merged_node_id(result.root, part.id)
            merged_nodes_by_parent.setdefault(group.parent.id, []).append(
                Node(id=node_id, name=part.name, part_id=part.id, transform=identity_transform())
            )

    _remove_selected_part_nodes(result.root, selected_node_ids)
    _append_merged_nodes(result.root, merged_nodes_by_parent)
    if options.remove_empty_nodes:
        _remove_empty_nodes(result.root)
    result.parts.update(merged_parts)
    _drop_unreferenced_parts(result)
    return result


def _walk_occurrences(asset: Asset) -> list[_Occurrence]:
    occurrences: list[_Occurrence] = []

    def walk(
        node: Node, parent: Node, node_path: str, parent_path: str, world: FloatArray, parent_world: FloatArray
    ) -> None:
        if node.part_id is not None and node.part_id in asset.parts:
            occurrences.append(
                _Occurrence(
                    node=node,
                    parent=parent,
                    node_path=node_path,
                    parent_path=parent_path,
                    world_transform=world.copy(),
                    parent_world_transform=parent_world.copy(),
                    part=asset.parts[node.part_id],
                )
            )
        for child in node.children:
            child_name = child.name or child.id
            child_world = world @ child.transform
            walk(
                child,
                node,
                f"{node_path}/{child_name}",
                node_path,
                child_world,
                world,
            )

    root_world = np.eye(4, dtype=np.float64)
    for child in asset.root.children:
        child_name = child.name or child.id
        walk(child, asset.root, f"{asset.root.name}/{child_name}", asset.root.name, child.transform, root_world)
    return occurrences


def _build_groups(asset: Asset, occurrences: list[_Occurrence], options: MergeOptions) -> list[_MergeGroup]:
    keyed: dict[tuple[object, ...], list[_MergeInput]] = {}
    for occurrence in occurrences:
        for merge_input in _inputs_for_occurrence(occurrence, split_by_material=options.mode == "by_material"):
            key = _group_key(merge_input, options)
            keyed.setdefault(key, []).append(merge_input)

    groups: list[_MergeGroup] = []
    for key, inputs in keyed.items():
        parent, parent_world = _group_parent(asset, inputs, options)
        groups.append(
            _MergeGroup(
                key=key,
                name=_group_name(key, inputs, options),
                parent=parent,
                parent_world_transform=parent_world,
                inputs=tuple(inputs),
            )
        )
    return groups


def _inputs_for_occurrence(occurrence: _Occurrence, *, split_by_material: bool) -> Iterable[_MergeInput]:
    mesh = occurrence.part.mesh
    if mesh is None:
        return ()
    all_faces = np.arange(mesh.triangle_count, dtype=np.int64)
    if not split_by_material:
        material_id = occurrence.part.material_ids[0] if occurrence.part.material_ids else None
        return (_MergeInput(occurrence=occurrence, material_id=material_id, face_indices=all_faces),)
    if mesh.material_indices is None:
        material_id = occurrence.part.material_ids[0] if occurrence.part.material_ids else None
        return (_MergeInput(occurrence=occurrence, material_id=material_id, face_indices=all_faces),)
    inputs: list[_MergeInput] = []
    for material_index in sorted(set(mesh.material_indices.astype(int).tolist())):
        material_id = (
            occurrence.part.material_ids[material_index] if material_index < len(occurrence.part.material_ids) else None
        )
        inputs.append(
            _MergeInput(
                occurrence=occurrence,
                material_id=material_id,
                face_indices=np.flatnonzero(mesh.material_indices == material_index).astype(np.int64),
            )
        )
    return tuple(inputs)


def _group_key(merge_input: _MergeInput, options: MergeOptions) -> tuple[object, ...]:
    occurrence = merge_input.occurrence
    if options.mode == "by_material":
        return ("material", merge_input.material_id or "none")
    if options.mode == "by_node_name":
        return ("node_name", occurrence.node.name)
    if options.mode == "by_part_name":
        return ("part_name", occurrence.part.name)
    if options.mode == "hierarchy_level":
        segments = occurrence.node_path.split("/")
        level = min(options.hierarchy_level, len(segments) - 1)
        return ("level", level, "/".join(segments[: level + 1]))
    if options.mode in {"parent_children", "final_level"}:
        return ("parent", occurrence.parent_path)
    if options.mode == "regions":
        center = _occurrence_center(merge_input)
        assert options.region_size is not None
        region = tuple(np.floor(center / float(options.region_size)).astype(int).tolist())
        if options.merge_strategy == "by_material":
            return ("region_material", region, merge_input.material_id or "none")
        return ("region", region)
    return ("all",)


def _group_parent(asset: Asset, inputs: list[_MergeInput], options: MergeOptions) -> tuple[Node, FloatArray]:
    if options.keep_parent:
        parents = {merge_input.occurrence.parent.id: merge_input.occurrence for merge_input in inputs}
        if len(parents) == 1:
            occurrence = next(iter(parents.values()))
            return occurrence.parent, occurrence.parent_world_transform
    return asset.root, np.eye(4, dtype=np.float64)


def _group_name(key: tuple[object, ...], inputs: list[_MergeInput], options: MergeOptions) -> str:
    if options.mode == "all":
        return "Merged Geometry"
    if options.mode == "by_material":
        material_id = str(key[1])
        return f"Merged {material_id}"
    if options.mode == "by_node_name":
        return f"Merged {inputs[0].occurrence.node.name}"
    if options.mode == "by_part_name":
        return f"Merged {inputs[0].occurrence.part.name}"
    if options.mode == "regions":
        return f"Merged Region {key[1]}"
    return "Merged " + str(key[-1]).replace("/", " ")


def _split_group(group: _MergeGroup, max_vertices: int | None) -> list[tuple[_MergeInput, ...]]:
    if max_vertices is None:
        return [group.inputs]
    chunks: list[tuple[_MergeInput, ...]] = []
    current: list[_MergeInput] = []
    current_vertices = 0
    for merge_input in group.inputs:
        if current and current_vertices + merge_input.vertex_count > max_vertices:
            chunks.append(tuple(current))
            current = []
            current_vertices = 0
        current.append(merge_input)
        current_vertices += merge_input.vertex_count
    if current:
        chunks.append(tuple(current))
    return chunks


def _merge_inputs(
    part_id: str,
    part_name: str,
    inputs: tuple[_MergeInput, ...],
    parent_world_transform: FloatArray,
    options: MergeOptions,
) -> Part:
    parent_inverse = np.asarray(np.linalg.inv(parent_world_transform), dtype=np.float64)
    points: list[FloatArray] = []
    faces: list[IntArray] = []
    face_material_ids: list[str | None] = []
    material_ids: list[str] = []
    material_index_by_id: dict[str, int] = {}
    offset = 0

    for merge_input in inputs:
        mesh = merge_input.occurrence.part.mesh
        if mesh is None or merge_input.face_indices.size == 0:
            continue
        used = np.unique(mesh.faces[merge_input.face_indices].reshape(-1))
        remap = np.full(mesh.vertex_count, -1, dtype=np.int64)
        remap[used] = np.arange(used.shape[0], dtype=np.int64)
        local_points = mesh.points[used]
        world_points = _transform_points(local_points, merge_input.occurrence.world_transform)
        parent_points = _transform_points(world_points, parent_inverse)
        points.append(parent_points)
        faces.append(remap[mesh.faces[merge_input.face_indices]] + offset)
        material_ids_for_faces = _face_material_ids(merge_input)
        face_material_ids.extend(material_ids_for_faces)
        if options.preserve_materials:
            for material_id in material_ids_for_faces:
                if material_id is not None and material_id not in material_index_by_id:
                    material_index_by_id[material_id] = len(material_ids)
                    material_ids.append(material_id)
        offset += used.shape[0]

    merged_mesh = Mesh(
        points=np.vstack(points) if points else np.empty((0, 3), dtype=np.float64),
        faces=np.vstack(faces) if faces else np.empty((0, 3), dtype=np.int64),
        material_indices=_material_indices(face_material_ids, material_index_by_id)
        if options.preserve_materials
        else None,
        metadata=_mesh_metadata(inputs),
    ).compute_normals()
    merged_mesh.validate()
    return Part(
        id=part_id,
        name=part_name,
        mesh=merged_mesh,
        material_ids=material_ids if options.preserve_materials else [],
        metadata=_part_metadata(inputs, options.metadata),
        fingerprint=merged_mesh.fingerprint(),
    )


def _face_material_ids(merge_input: _MergeInput) -> list[str | None]:
    mesh = merge_input.occurrence.part.mesh
    if mesh is None:
        return []
    if mesh.material_indices is None:
        return [merge_input.material_id] * int(merge_input.face_indices.shape[0])
    material_ids: list[str | None] = []
    for face_index in merge_input.face_indices.astype(int).tolist():
        material_index = int(mesh.material_indices[face_index])
        material_id = (
            merge_input.occurrence.part.material_ids[material_index]
            if material_index < len(merge_input.occurrence.part.material_ids)
            else None
        )
        material_ids.append(material_id)
    return material_ids


def _material_indices(face_material_ids: list[str | None], material_index_by_id: dict[str, int]) -> IntArray | None:
    if not material_index_by_id:
        return None
    material_ids = [material_id for material_id in face_material_ids if material_id is not None]
    if len(material_ids) != len(face_material_ids):
        return None
    return np.asarray([material_index_by_id[material_id] for material_id in material_ids], dtype=np.int64)


def _part_metadata(inputs: tuple[_MergeInput, ...], policy: str) -> dict[str, str]:
    if policy == "drop":
        return {}
    source_part_ids = sorted({merge_input.occurrence.part.id for merge_input in inputs})
    source_node_ids = sorted({merge_input.occurrence.node.id for merge_input in inputs})
    if policy == "summarize":
        return {
            "source_part_count": str(len(source_part_ids)),
            "source_node_count": str(len(source_node_ids)),
        }
    if policy == "preserve" and len(source_part_ids) == 1:
        return {
            **inputs[0].occurrence.part.metadata,
            "source_part_ids": ",".join(source_part_ids),
            "source_node_ids": ",".join(source_node_ids),
        }
    combined: dict[str, set[str]] = {}
    for merge_input in inputs:
        for key, value in merge_input.occurrence.part.metadata.items():
            combined.setdefault(key, set()).add(str(value))
    return {
        **{key: "|".join(sorted(values)) for key, values in combined.items()},
        "source_part_ids": ",".join(source_part_ids),
        "source_node_ids": ",".join(source_node_ids),
    }


def _mesh_metadata(inputs: tuple[_MergeInput, ...]) -> dict[str, str]:
    return {
        "merged_occurrences": str(len({merge_input.occurrence.node.id for merge_input in inputs})),
        "merged_parts": str(len({merge_input.occurrence.part.id for merge_input in inputs})),
    }


def _transform_points(points: FloatArray, transform: FloatArray) -> FloatArray:
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return cast(FloatArray, np.asarray((transform @ homogeneous.T).T[:, :3], dtype=np.float64))


def _occurrence_center(merge_input: _MergeInput) -> FloatArray:
    mesh = merge_input.occurrence.part.mesh
    if mesh is None:
        return np.zeros(3, dtype=np.float64)
    mins, maxs = mesh.bounds()
    center = _transform_points(((mins + maxs) * 0.5).reshape((1, 3)), merge_input.occurrence.world_transform)
    return cast(FloatArray, center[0])


def _remove_selected_part_nodes(node: Node, selected_node_ids: set[str]) -> None:
    kept: list[Node] = []
    for child in node.children:
        _remove_selected_part_nodes(child, selected_node_ids)
        if child.id in selected_node_ids and child.part_id is not None:
            continue
        kept.append(child)
    node.children = kept


def _append_merged_nodes(node: Node, merged_nodes_by_parent: dict[str, list[Node]]) -> None:
    node.children.extend(child.copy() for child in merged_nodes_by_parent.get(node.id, []))
    for child in node.children:
        _append_merged_nodes(child, merged_nodes_by_parent)


def _remove_empty_nodes(node: Node) -> bool:
    kept: list[Node] = []
    for child in node.children:
        if _remove_empty_nodes(child):
            kept.append(child)
    node.children = kept
    return node.part_id is not None or bool(node.children)


def _drop_unreferenced_parts(asset: Asset) -> None:
    referenced = {node.part_id for node in asset.root.walk() if node.part_id is not None}
    asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id in referenced}


def _merged_part_id(parts: dict[str, Part], key: tuple[object, ...], chunk_index: int) -> str:
    digest = hashlib.sha1(repr((key, chunk_index)).encode("utf-8")).hexdigest()[:12]
    candidate = f"merged_{digest}"
    suffix = 2
    while candidate in parts:
        candidate = f"merged_{digest}_{suffix}"
        suffix += 1
    return candidate


def _merged_node_id(root: Node, part_id: str) -> str:
    existing = {node.id for node in root.walk()}
    candidate = f"node_{part_id}"
    suffix = 2
    while candidate in existing:
        candidate = f"node_{part_id}_{suffix}"
        suffix += 1
    return candidate
