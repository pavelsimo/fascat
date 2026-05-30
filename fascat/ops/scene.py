from __future__ import annotations

import hashlib
import json

import numpy as np

from fascat.asset import Asset, Node, Part, identity_transform
from fascat.mesh import Mesh
from fascat.options import MergeOptions, SceneOptimizeOptions


def optimize_scene_asset(
    asset: Asset,
    options: SceneOptimizeOptions,
    *,
    selected_node_ids: set[str],
) -> Asset:
    result = asset.copy(keep_source=True)
    _apply_instance_policy(result, options, selected_node_ids)
    if options.merge_compatible_meshes or options.batch_by_material:
        from fascat.ops.hierarchy import merge_asset

        result = merge_asset(
            result,
            MergeOptions(
                mode="by_material" if options.batch_by_material else "all",
                keep_parent=options.flatten == "none",
                metadata="combine",
                max_vertices_per_mesh=options.max_vertices_per_mesh if options.split_large_meshes else None,
                preserve_materials=True,
                merge_strategy="by_material" if options.batch_by_material else "all",
                remove_empty_nodes=options.remove_empty_nodes,
            ),
            selected_node_ids=selected_node_ids,
        )
    if options.flatten == "all":
        _flatten_all(result)
    elif options.flatten == "safe":
        _flatten_safe(result.root)
    if options.remove_empty_nodes:
        _remove_empty_nodes(result.root)
    if options.split_large_meshes:
        _split_oversized_meshes(result, options, selected_node_ids)
    _annotate_index_buffers(result, options)
    _annotate_instance_policy(result, options)
    _drop_unreferenced_parts(result)
    return result


def _apply_instance_policy(asset: Asset, options: SceneOptimizeOptions, selected_node_ids: set[str]) -> None:
    if options.instance_policy in {"auto", "preserve"}:
        _reconstruct_instances(
            asset,
            selected_node_ids,
            similarity_tolerance=options.instance_similarity_tolerance,
        )
        return
    if options.instance_policy != "expand":
        return
    occurrence_counts = _part_occurrence_counts(asset)
    existing_part_ids = set(asset.parts)
    for node in asset.root.walk():
        if node.id not in selected_node_ids or node.part_id is None:
            continue
        source_id = node.part_id
        if occurrence_counts.get(source_id, 0) <= 1 or source_id not in asset.parts:
            continue
        source = asset.parts[source_id]
        part = source.copy(keep_source=True)
        part.id = _unique_part_id(existing_part_ids, f"{source_id}_{node.id}")
        part.metadata = {
            **part.metadata,
            "source_part_id": source_id,
            "occurrence_node_id": node.id,
            "scene_instance_policy": "expand",
        }
        asset.parts[part.id] = part
        existing_part_ids.add(part.id)
        node.part_id = part.id


def _reconstruct_instances(asset: Asset, selected_node_ids: set[str], *, similarity_tolerance: float) -> None:
    selected_part_ids = _selected_part_ids(asset, selected_node_ids)
    part_ids_by_fingerprint: dict[str, list[str]] = {}
    for part_id in sorted(selected_part_ids):
        part = asset.parts.get(part_id)
        if part is None or part.mesh is None:
            continue
        fingerprint = part.fingerprint or part.mesh.fingerprint()
        part.fingerprint = fingerprint
        part_ids_by_fingerprint.setdefault(fingerprint, []).append(part_id)

    replacements: dict[str, str] = {}
    material_blocked_groups = 0
    attribute_blocked_groups = 0
    metadata_blocked_groups = 0
    material_key_by_part: dict[str, tuple[tuple[str, ...], tuple[int, ...] | None]] = {}
    attribute_key_by_part: dict[
        str,
        tuple[str | None, str | None, tuple[tuple[int, str], ...], tuple[tuple[str, str], ...]],
    ] = {}
    metadata_key_by_part: dict[str, tuple[str, str]] = {}
    for part_ids in part_ids_by_fingerprint.values():
        if len(part_ids) <= 1:
            continue
        for part_id in part_ids:
            part = asset.parts[part_id]
            material_key_by_part.setdefault(part_id, _part_material_key(part))
            attribute_key_by_part.setdefault(part_id, _part_mesh_attribute_key(part.mesh))
            metadata_key_by_part.setdefault(part_id, _part_metadata_key(part))
        material_keys = {material_key_by_part[part_id] for part_id in part_ids}
        attribute_keys = {attribute_key_by_part[part_id] for part_id in part_ids}
        metadata_keys = {metadata_key_by_part[part_id] for part_id in part_ids}
        if len(material_keys) > 1:
            material_blocked_groups += 1
        if len(attribute_keys) > 1:
            attribute_blocked_groups += 1
        if len(metadata_keys) > 1:
            metadata_blocked_groups += 1

        canonical_by_key: dict[tuple[object, ...], str] = {}
        for part_id in part_ids:
            material_key = material_key_by_part[part_id]
            attribute_key = attribute_key_by_part[part_id]
            metadata_key = metadata_key_by_part[part_id]
            key = (material_key, attribute_key, metadata_key)
            canonical_id = canonical_by_key.get(key)
            if canonical_id is None:
                canonical_by_key[key] = part_id
                continue
            replacements[part_id] = canonical_id

    similarity_replacements, similarity_candidate_groups = _similar_instance_replacements(
        asset,
        selected_part_ids=selected_part_ids,
        replacements=replacements,
        tolerance=similarity_tolerance,
    )
    replacements.update(similarity_replacements)

    vertex_savings = 0
    triangle_savings = 0
    mesh_payload_savings = 0
    for part_id in replacements:
        mesh = asset.parts[part_id].mesh
        if mesh is None:
            continue
        vertex_savings += mesh.vertex_count
        triangle_savings += mesh.triangle_count
        mesh_payload_savings += _mesh_payload_bytes(mesh)

    remapped_occurrences = 0
    for node in asset.root.walk():
        if node.part_id in replacements:
            node.part_id = replacements[node.part_id]
            remapped_occurrences += 1
    if replacements:
        asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id not in replacements}

    asset.metadata["scene_reconstructed_part_count"] = str(len(replacements))
    asset.metadata["scene_reconstructed_occurrence_count"] = str(remapped_occurrences)
    asset.metadata["scene_reconstructed_vertex_savings"] = str(vertex_savings)
    asset.metadata["scene_reconstructed_triangle_savings"] = str(triangle_savings)
    asset.metadata["scene_reconstructed_mesh_payload_savings_bytes"] = str(mesh_payload_savings)
    asset.metadata["scene_similarity_tolerance"] = f"{similarity_tolerance:g}"
    asset.metadata["scene_similarity_candidate_group_count"] = str(similarity_candidate_groups)
    asset.metadata["scene_similarity_reconstructed_part_count"] = str(len(similarity_replacements))
    if material_blocked_groups:
        asset.report.add_warning(
            "instance reconstruction found "
            f"{material_blocked_groups} matching mesh group(s) with material differences that prevented full instancing"
        )
    if attribute_blocked_groups:
        asset.report.add_warning(
            "instance reconstruction found "
            f"{attribute_blocked_groups} matching mesh group(s) with vertex attribute differences that prevented full "
            "instancing"
        )
    if metadata_blocked_groups:
        asset.report.add_warning(
            "instance reconstruction found "
            f"{metadata_blocked_groups} matching mesh group(s) with metadata differences that prevented full instancing"
        )


def _similar_instance_replacements(
    asset: Asset,
    *,
    selected_part_ids: set[str],
    replacements: dict[str, str],
    tolerance: float,
) -> tuple[dict[str, str], int]:
    if tolerance <= 0.0:
        return {}, 0
    unresolved_part_ids = sorted(part_id for part_id in selected_part_ids if part_id not in replacements)
    part_ids_by_key: dict[tuple[object, ...], list[str]] = {}
    for part_id in unresolved_part_ids:
        part = asset.parts.get(part_id)
        if part is None or part.mesh is None:
            continue
        key = _part_similarity_key(part)
        if key is None:
            continue
        part_ids_by_key.setdefault(key, []).append(part_id)

    similar_replacements: dict[str, str] = {}
    candidate_groups = 0
    for part_ids in part_ids_by_key.values():
        if len(part_ids) <= 1:
            continue
        candidate_groups += 1
        clusters: list[str] = []
        for part_id in part_ids:
            mesh = asset.parts[part_id].mesh
            assert mesh is not None
            canonical_id = next(
                (
                    cluster_id
                    for cluster_id in clusters
                    if _mesh_positions_within_tolerance(asset.parts[cluster_id].mesh, mesh, tolerance)
                ),
                None,
            )
            if canonical_id is None:
                clusters.append(part_id)
            else:
                similar_replacements[part_id] = canonical_id
    return similar_replacements, candidate_groups


def _part_similarity_key(part: Part) -> tuple[object, ...] | None:
    mesh = part.mesh
    if mesh is None:
        return None
    return (
        _part_material_key(part),
        _part_mesh_attribute_key(mesh),
        _part_metadata_key(part),
        mesh.points.shape,
        _array_digest_required(mesh.faces),
    )


def _mesh_positions_within_tolerance(left: Mesh | None, right: Mesh | None, tolerance: float) -> bool:
    if left is None or right is None or left.points.shape != right.points.shape:
        return False
    distances = np.linalg.norm(left.points - right.points, axis=1)
    return bool(distances.size == 0 or float(distances.max()) <= tolerance)


def _mesh_payload_bytes(mesh: Mesh) -> int:
    total = int(mesh.points.nbytes + mesh.faces.nbytes)
    if mesh.normals is not None:
        total += int(mesh.normals.nbytes)
    if mesh.tangents is not None:
        total += int(mesh.tangents.nbytes)
    total += sum(int(values.nbytes) for values in mesh.uvs.values())
    if mesh.material_indices is not None:
        total += int(mesh.material_indices.nbytes)
    total += sum(int(values.nbytes) for values in mesh.face_groups.values())
    return total


def _part_material_key(part: Part) -> tuple[tuple[str, ...], tuple[int, ...] | None]:
    material_indices = None
    if part.mesh is not None and part.mesh.material_indices is not None:
        material_indices = tuple(int(value) for value in part.mesh.material_indices.tolist())
    return (tuple(part.material_ids), material_indices)


def _part_metadata_key(part: Part) -> tuple[str, str]:
    mesh_metadata = {} if part.mesh is None else part.mesh.metadata
    return (_metadata_key(part.metadata), _metadata_key(mesh_metadata))


def _part_mesh_attribute_key(
    mesh: Mesh | None,
) -> tuple[str | None, str | None, tuple[tuple[int, str], ...], tuple[tuple[str, str], ...]]:
    if mesh is None:
        return (None, None, (), ())
    uv_keys = tuple((channel, _array_digest_required(values)) for channel, values in sorted(mesh.uvs.items()))
    face_group_keys = tuple((name, _array_digest_required(values)) for name, values in sorted(mesh.face_groups.items()))
    return (_array_digest(mesh.normals), _array_digest(mesh.tangents), uv_keys, face_group_keys)


def _array_digest_required(values: np.ndarray) -> str:
    digest = _array_digest(values)
    assert digest is not None
    return digest


def _array_digest(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values)
    digest = hashlib.sha1()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _metadata_key(metadata: dict[str, object]) -> str:
    return json.dumps(metadata, sort_keys=True, default=str)


def _flatten_safe(node: Node) -> None:
    flattened: list[Node] = []
    for child in node.children:
        _flatten_safe(child)
        if child.part_id is None and not child.metadata and np.allclose(child.transform, np.eye(4)):
            flattened.extend(grandchild.copy() for grandchild in child.children)
        else:
            flattened.append(child)
    node.children = flattened


def _flatten_all(asset: Asset) -> None:
    leaves: list[Node] = []

    def walk(node: Node, world: np.ndarray) -> None:
        current = world @ node.transform
        if node.part_id is not None:
            leaves.append(
                Node(
                    id=node.id,
                    name=node.name,
                    part_id=node.part_id,
                    transform=current,
                    metadata=dict(node.metadata),
                )
            )
        for child in node.children:
            walk(child, current)

    for child in asset.root.children:
        walk(child, np.eye(4, dtype=np.float64))
    asset.root.children = leaves
    asset.root.transform = identity_transform()


def _remove_empty_nodes(node: Node) -> bool:
    kept: list[Node] = []
    for child in node.children:
        if _remove_empty_nodes(child):
            kept.append(child)
    node.children = kept
    return node.part_id is not None or bool(node.children)


def _split_oversized_meshes(asset: Asset, options: SceneOptimizeOptions, selected_node_ids: set[str]) -> None:
    max_vertices = options.max_vertices_per_mesh
    if max_vertices is None:
        return
    selected_part_ids = _selected_part_ids(asset, selected_node_ids)
    existing_part_ids = set(asset.parts)
    split_part_ids: dict[str, list[str]] = {}
    for part_id in sorted(selected_part_ids):
        part = asset.parts.get(part_id)
        if part is None or part.mesh is None or part.mesh.vertex_count <= max_vertices:
            continue
        chunks = _split_part(part, max_vertices=max_vertices, existing_part_ids=existing_part_ids)
        if len(chunks) <= 1:
            continue
        for chunk in chunks:
            asset.parts[chunk.id] = chunk
            existing_part_ids.add(chunk.id)
        split_part_ids[part_id] = [chunk.id for chunk in chunks]

    if not split_part_ids:
        return
    existing_node_ids = {node.id for node in asset.root.walk()}
    for node in asset.root.walk():
        if node.part_id not in split_part_ids:
            continue
        source_part_id = node.part_id
        node.part_id = None
        chunk_nodes = []
        for chunk_index, chunk_part_id in enumerate(split_part_ids[source_part_id], start=1):
            child_id = _unique_node_id(existing_node_ids, f"{node.id}_split_{chunk_index}")
            chunk_nodes.append(
                Node(
                    id=child_id,
                    name=f"{node.name} {chunk_index}",
                    part_id=chunk_part_id,
                    transform=identity_transform(),
                )
            )
        node.children = chunk_nodes + node.children


def _split_part(part: Part, *, max_vertices: int, existing_part_ids: set[str]) -> list[Part]:
    mesh = part.mesh
    if mesh is None or mesh.triangle_count == 0:
        return [part]
    face_chunks = _face_chunks(mesh, max_vertices=max_vertices)
    if len(face_chunks) <= 1:
        return [part]
    chunks: list[Part] = []
    for chunk_index, face_indices in enumerate(face_chunks, start=1):
        chunk_mesh = _slice_mesh(mesh, face_indices)
        chunk_id = _unique_part_id(existing_part_ids, f"{part.id}_split_{chunk_index}")
        existing_part_ids.add(chunk_id)
        chunk = Part(
            id=chunk_id,
            name=f"{part.name} {chunk_index}",
            mesh=chunk_mesh,
            material_ids=list(part.material_ids),
            metadata={
                **part.metadata,
                "split_source_part_id": part.id,
                "split_chunk": str(chunk_index),
                "split_chunks": str(len(face_chunks)),
            },
        )
        chunk.fingerprint = chunk_mesh.fingerprint()
        chunks.append(chunk)
    return chunks


def _face_chunks(mesh: Mesh, *, max_vertices: int) -> list[np.ndarray]:
    chunks: list[np.ndarray] = []
    current_faces: list[int] = []
    current_vertices: set[int] = set()
    for face_index, face in enumerate(mesh.faces.astype(int).tolist()):
        face_vertices = set(face)
        if current_faces and len(current_vertices | face_vertices) > max_vertices:
            chunks.append(np.asarray(current_faces, dtype=np.int64))
            current_faces = []
            current_vertices = set()
        current_faces.append(face_index)
        current_vertices.update(face_vertices)
    if current_faces:
        chunks.append(np.asarray(current_faces, dtype=np.int64))
    return chunks


def _slice_mesh(mesh: Mesh, face_indices: np.ndarray) -> Mesh:
    used = np.unique(mesh.faces[face_indices].reshape(-1))
    remap = np.full(mesh.vertex_count, -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    face_lookup = {int(face_index): local_index for local_index, face_index in enumerate(face_indices.tolist())}
    sliced = Mesh(
        points=mesh.points[used],
        faces=remap[mesh.faces[face_indices]],
        normals=None if mesh.normals is None else mesh.normals[used],
        tangents=None if mesh.tangents is None else mesh.tangents[used],
        uvs={channel: values[used] for channel, values in mesh.uvs.items()},
        material_indices=None if mesh.material_indices is None else mesh.material_indices[face_indices],
        face_groups={
            name: np.asarray(
                [face_lookup[int(face_index)] for face_index in group.tolist() if int(face_index) in face_lookup],
                dtype=np.int64,
            )
            for name, group in mesh.face_groups.items()
        },
        metadata=dict(mesh.metadata),
    )
    sliced.validate()
    return sliced


def _annotate_index_buffers(asset: Asset, options: SceneOptimizeOptions) -> None:
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        if options.index_buffer == "uint16" and part.mesh.vertex_count > 65_535:
            part.mesh.metadata["index_buffer"] = "uint32"
            asset.report.add_warning(f"part exceeds uint16 index range, using uint32: {part.name}")
        elif options.index_buffer == "uint32":
            part.mesh.metadata["index_buffer"] = "uint32"
        else:
            part.mesh.metadata["index_buffer"] = "uint16" if part.mesh.vertex_count <= 65_535 else "uint32"


def _annotate_instance_policy(asset: Asset, options: SceneOptimizeOptions) -> None:
    occurrence_counts = _part_occurrence_counts(asset)
    instanced = sum(1 for count in occurrence_counts.values() if count > 1)
    asset.metadata["scene_instance_policy"] = options.instance_policy
    asset.metadata["scene_instanced_part_count"] = str(instanced)


def _part_occurrence_counts(asset: Asset) -> dict[str, int]:
    occurrence_counts: dict[str, int] = {}
    for node in asset.root.walk():
        if node.part_id is not None:
            occurrence_counts[node.part_id] = occurrence_counts.get(node.part_id, 0) + 1
    return occurrence_counts


def _selected_part_ids(asset: Asset, selected_node_ids: set[str]) -> set[str]:
    part_ids = {node.part_id for node in asset.root.walk() if node.id in selected_node_ids and node.part_id is not None}
    for part_id, part in asset.parts.items():
        source_node_ids = set(str(part.metadata.get("source_node_ids", "")).split(","))
        if source_node_ids & selected_node_ids:
            part_ids.add(part_id)
    return {part_id for part_id in part_ids if part_id is not None}


def _drop_unreferenced_parts(asset: Asset) -> None:
    referenced = {node.part_id for node in asset.root.walk() if node.part_id is not None}
    asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id in referenced}


def _unique_part_id(existing: set[str], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _unique_node_id(existing: set[str], base: str) -> str:
    candidate = base
    suffix = 2
    while candidate in existing:
        candidate = f"{base}_{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate
