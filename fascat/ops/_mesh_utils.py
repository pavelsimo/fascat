from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.ops._arrays import remap_sliced_face_group, sliced_face_lookup

IntArray = NDArray[np.int64]


def selected_mesh_part_ids(asset: Asset, selected_part_ids: set[str] | None) -> set[str]:
    return {
        part.id
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    }


def selected_mesh_part_id_list(asset: Asset, selected_part_ids: set[str] | None) -> list[str]:
    return [
        part.id
        for part in asset.parts.values()
        if (selected_part_ids is None or part.id in selected_part_ids) and part.mesh is not None
    ]


def slice_faces(mesh: Mesh, face_indices: IntArray) -> Mesh:
    face_lookup = sliced_face_lookup(face_indices, mesh.triangle_count)
    return Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces[face_indices].copy(),
        normals=None if mesh.normals is None else mesh.normals.copy(),
        tangents=None if mesh.tangents is None else mesh.tangents.copy(),
        uvs={channel: values.copy() for channel, values in mesh.uvs.items()},
        material_indices=None if mesh.material_indices is None else mesh.material_indices[face_indices].copy(),
        face_groups={name: remap_sliced_face_group(face_lookup, group) for name, group in mesh.face_groups.items()},
        metadata=dict(mesh.metadata),
    )


def edge_faces(mesh: Mesh) -> dict[tuple[int, int], list[int]]:
    return grouped_edge_faces(face_major_edges(mesh), mesh.triangle_count)


def face_major_edges(mesh: Mesh) -> IntArray:
    if mesh.triangle_count == 0:
        return np.empty((0, 2), dtype=np.int64)
    return cast(
        IntArray,
        np.stack(
            [
                mesh.faces[:, [0, 1]],
                mesh.faces[:, [1, 2]],
                mesh.faces[:, [2, 0]],
            ],
            axis=1,
        ).reshape((-1, 2)),
    )


def grouped_edge_faces(directed_edges: IntArray, triangle_count: int) -> dict[tuple[int, int], list[int]]:
    if directed_edges.size == 0:
        return {}
    edge_keys = np.sort(directed_edges, axis=1)
    face_indices = np.repeat(np.arange(triangle_count, dtype=np.int64), 3)
    order = np.lexsort((np.arange(edge_keys.shape[0], dtype=np.int64), edge_keys[:, 1], edge_keys[:, 0]))
    sorted_keys = edge_keys[order]
    sorted_faces = face_indices[order]
    boundaries = np.flatnonzero(np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)) + 1
    starts = np.concatenate([np.zeros(1, dtype=np.int64), boundaries])
    ends = np.concatenate([boundaries, np.asarray([sorted_keys.shape[0]], dtype=np.int64)])
    return {
        (int(sorted_keys[int(start), 0]), int(sorted_keys[int(start), 1])): [
            int(face_index) for face_index in sorted_faces[int(start) : int(end)]
        ]
        for start, end in zip(starts, ends, strict=True)
    }
