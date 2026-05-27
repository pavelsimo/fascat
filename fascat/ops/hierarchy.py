from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part, identity_transform
from fascat.mesh import Mesh
from fascat.metadata import Metadata
from fascat.options import ExplodeOptions, MergeOptions, ReplaceOptions

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


def explode_asset(asset: Asset, options: ExplodeOptions, *, selected_node_ids: set[str]) -> Asset:
    result = asset.copy(keep_source=True)
    occurrences = [
        occurrence
        for occurrence in _walk_occurrences(result)
        if occurrence.node.id in selected_node_ids and occurrence.part.mesh is not None
    ]
    if not occurrences:
        result.report.add_warning("explode matched no mesh-bearing part occurrences")
        return result

    exploded = 0
    for occurrence in occurrences:
        mesh = occurrence.part.mesh
        if mesh is None:
            continue
        groups = _explode_face_groups(occurrence.part, mesh, options.mode)
        if len(groups) <= 1:
            continue
        parts = _exploded_parts(result.parts, occurrence, groups, options)
        for part in parts:
            result.parts[part.id] = part
        occurrence.node.part_id = None
        occurrence.node.children = [
            Node(
                id=_exploded_node_id(result.root, occurrence.node.id, index),
                name=part.name,
                part_id=part.id,
                transform=identity_transform(),
                metadata=dict(part.metadata),
            )
            for index, part in enumerate(parts, start=1)
        ]
        exploded += 1

    if exploded == 0:
        result.report.add_warning("explode found no selected mesh that could be split")
        return result
    if options.remove_empty_nodes:
        _remove_empty_nodes(result.root)
    _drop_unreferenced_parts(result)
    return result


def replace_asset(asset: Asset, options: ReplaceOptions, *, selected_node_ids: set[str]) -> Asset:
    result = asset.copy(keep_source=True)
    occurrences = [
        occurrence
        for occurrence in _walk_occurrences(result)
        if occurrence.node.id in selected_node_ids and occurrence.part.mesh is not None
    ]
    if not occurrences:
        result.report.add_warning("replace matched no mesh-bearing part occurrences")
        return result

    for occurrence in occurrences:
        mesh = occurrence.part.mesh
        if mesh is None:
            continue
        replacement_mesh = _replacement_mesh(occurrence, options)
        part_id = _replacement_part_id(result.parts, occurrence.part.id, occurrence.node.id)
        metadata = _replacement_metadata(occurrence, options)
        replacement_part = Part(
            id=part_id,
            name=_replacement_name(occurrence.part, options),
            mesh=replacement_mesh,
            material_ids=list(occurrence.part.material_ids),
            metadata=metadata,
            fingerprint=replacement_mesh.fingerprint(),
        )
        result.parts[part_id] = replacement_part
        occurrence.node.part_id = part_id
        if not options.preserve_transform:
            occurrence.node.transform = identity_transform()

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


def _part_metadata(inputs: tuple[_MergeInput, ...], policy: str) -> Metadata:
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


def _mesh_metadata(inputs: tuple[_MergeInput, ...]) -> Metadata:
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


def _explode_face_groups(part: Part, mesh: Mesh, mode: str) -> list[tuple[str, IntArray, str | None]]:
    if mesh.triangle_count == 0:
        return []
    if mode == "by_material":
        if mesh.material_indices is None:
            material_id = part.material_ids[0] if part.material_ids else None
            return [("material_none", np.arange(mesh.triangle_count, dtype=np.int64), material_id)]
        groups: list[tuple[str, IntArray, str | None]] = []
        for material_index in sorted(set(mesh.material_indices.astype(int).tolist())):
            material_id = part.material_ids[material_index] if material_index < len(part.material_ids) else None
            groups.append(
                (
                    f"material_{material_index}",
                    np.flatnonzero(mesh.material_indices == material_index).astype(np.int64),
                    material_id,
                )
            )
        return groups
    return [
        (f"component_{index}", face_indices, None)
        for index, face_indices in enumerate(_connected_face_components(mesh), start=1)
    ]


def _connected_face_components(mesh: Mesh) -> list[IntArray]:
    faces_by_vertex: dict[int, list[int]] = {}
    for face_index, face in enumerate(mesh.faces.astype(int).tolist()):
        for vertex in face:
            faces_by_vertex.setdefault(vertex, []).append(face_index)
    components: list[IntArray] = []
    remaining = set(range(mesh.triangle_count))
    while remaining:
        start = remaining.pop()
        component = [start]
        stack = [start]
        while stack:
            face_index = stack.pop()
            for vertex in mesh.faces[face_index].astype(int).tolist():
                for neighbor in faces_by_vertex.get(vertex, []):
                    if neighbor not in remaining:
                        continue
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        components.append(np.asarray(sorted(component), dtype=np.int64))
    return components


def _exploded_parts(
    parts: dict[str, Part],
    occurrence: _Occurrence,
    groups: list[tuple[str, IntArray, str | None]],
    options: ExplodeOptions,
) -> list[Part]:
    result: list[Part] = []
    for index, (label, face_indices, material_id) in enumerate(groups, start=1):
        part_id = _exploded_part_id(parts, occurrence.part.id, occurrence.node.id, index)
        mesh = _mesh_subset(occurrence.part.mesh, face_indices)
        assert mesh is not None
        material_ids = list(occurrence.part.material_ids)
        if options.mode == "by_material" and material_id is not None:
            material_ids = [material_id]
            if mesh.material_indices is not None:
                mesh.material_indices = np.zeros(mesh.triangle_count, dtype=np.int64)
        metadata = _exploded_metadata(occurrence, label, options.metadata)
        mesh.metadata = {**mesh.metadata, "explode_label": label}
        mesh.validate()
        result.append(
            Part(
                id=part_id,
                name=f"{occurrence.part.name} {label.replace('_', ' ')}",
                mesh=mesh.compute_normals(),
                material_ids=material_ids,
                metadata=metadata,
                fingerprint=mesh.fingerprint(),
            )
        )
    return result


def _mesh_subset(mesh: Mesh | None, face_indices: IntArray) -> Mesh | None:
    if mesh is None or face_indices.size == 0:
        return None
    used = np.unique(mesh.faces[face_indices].reshape(-1))
    remap = np.full(mesh.vertex_count, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    result = Mesh(
        points=mesh.points[used].copy(),
        faces=remap[mesh.faces[face_indices]],
        normals=None if mesh.normals is None else mesh.normals[used].copy(),
        tangents=None if mesh.tangents is None else mesh.tangents[used].copy(),
        uvs={channel: values[used].copy() for channel, values in mesh.uvs.items()},
        material_indices=None if mesh.material_indices is None else mesh.material_indices[face_indices].copy(),
        face_groups={
            name: _remap_face_group(values, face_indices)
            for name, values in mesh.face_groups.items()
            if np.isin(values, face_indices).any()
        },
        metadata=dict(mesh.metadata),
    )
    return result


def _remap_face_group(values: IntArray, face_indices: IntArray) -> IntArray:
    face_position = {int(face_index): index for index, face_index in enumerate(face_indices.astype(int).tolist())}
    return np.asarray(
        [face_position[int(value)] for value in values.astype(int).tolist() if int(value) in face_position]
    )


def _exploded_metadata(occurrence: _Occurrence, label: str, policy: str) -> Metadata:
    if policy == "drop":
        return {}
    if policy == "summarize":
        return {
            "source_part_ids": occurrence.part.id,
            "source_node_ids": occurrence.node.id,
            "explode_label": label,
        }
    return {
        **occurrence.part.metadata,
        "source_part_ids": occurrence.part.id,
        "source_node_ids": occurrence.node.id,
        "explode_label": label,
    }


def _replacement_mesh(occurrence: _Occurrence, options: ReplaceOptions) -> Mesh:
    source = occurrence.part.mesh
    if source is None:
        return Mesh(points=np.empty((0, 3), dtype=np.float64), faces=np.empty((0, 3), dtype=np.int64))
    if options.mode == "proxy_mesh":
        if not isinstance(options.proxy_mesh, Mesh):
            raise TypeError("proxy_mesh replacement requires a Mesh")
        mesh = options.proxy_mesh.copy()
    else:
        mesh = _bounding_box_mesh(source)
    if not options.preserve_transform:
        parent_inverse = np.asarray(np.linalg.inv(occurrence.parent_world_transform), dtype=np.float64)
        mesh.points = _transform_points(mesh.points, parent_inverse @ occurrence.world_transform)
    mesh.metadata = {
        **mesh.metadata,
        "replacement_mode": options.mode,
        "source_part_id": occurrence.part.id,
        "source_node_id": occurrence.node.id,
    }
    if options.external_path is not None:
        mesh.metadata["external_path"] = options.external_path
    mesh.validate()
    return mesh.compute_normals()


def _bounding_box_mesh(mesh: Mesh) -> Mesh:
    mins, maxs = mesh.bounds()
    mins = mins.copy()
    maxs = maxs.copy()
    for axis in range(3):
        if np.isclose(mins[axis], maxs[axis]):
            mins[axis] -= 1e-6
            maxs[axis] += 1e-6
    x0, y0, z0 = mins.tolist()
    x1, y1, z1 = maxs.tolist()
    points = np.asarray(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [1, 5, 6],
            [1, 6, 2],
            [2, 6, 7],
            [2, 7, 3],
            [3, 7, 4],
            [3, 4, 0],
        ],
        dtype=np.int64,
    )
    return Mesh(points=points, faces=faces, metadata={"proxy": "bounding_box"})


def _replacement_metadata(occurrence: _Occurrence, options: ReplaceOptions) -> Metadata:
    if options.metadata == "drop":
        metadata: Metadata = {}
    elif options.metadata == "summarize":
        metadata = {"source_part_ids": occurrence.part.id, "source_node_ids": occurrence.node.id}
    else:
        metadata = {
            **occurrence.part.metadata,
            "source_part_ids": occurrence.part.id,
            "source_node_ids": occurrence.node.id,
        }
    metadata["replacement_mode"] = options.mode
    if options.external_path is not None:
        metadata["external_path"] = options.external_path
    return metadata


def _replacement_name(part: Part, options: ReplaceOptions) -> str:
    if options.mode == "bounding_box":
        return f"{part.name} Bounding Box"
    if options.mode == "external_asset":
        return f"{part.name} External Proxy"
    return f"{part.name} Proxy"


def _merged_part_id(parts: dict[str, Part], key: tuple[object, ...], chunk_index: int) -> str:
    digest = hashlib.sha1(repr((key, chunk_index)).encode("utf-8")).hexdigest()[:12]
    candidate = f"merged_{digest}"
    suffix = 2
    while candidate in parts:
        candidate = f"merged_{digest}_{suffix}"
        suffix += 1
    return candidate


def _exploded_part_id(parts: dict[str, Part], part_id: str, node_id: str, index: int) -> str:
    candidate = f"{part_id}_{node_id}_exploded_{index}"
    suffix = 2
    while candidate in parts:
        candidate = f"{part_id}_{node_id}_exploded_{index}_{suffix}"
        suffix += 1
    return candidate


def _exploded_node_id(root: Node, node_id: str, index: int) -> str:
    existing = {node.id for node in root.walk()}
    candidate = f"{node_id}_exploded_{index}"
    suffix = 2
    while candidate in existing:
        candidate = f"{node_id}_exploded_{index}_{suffix}"
        suffix += 1
    return candidate


def _replacement_part_id(parts: dict[str, Part], part_id: str, node_id: str) -> str:
    candidate = f"{part_id}_{node_id}_replacement"
    suffix = 2
    while candidate in parts:
        candidate = f"{part_id}_{node_id}_replacement_{suffix}"
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
