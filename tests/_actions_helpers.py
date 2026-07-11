from __future__ import annotations

import numpy as np

from fascat.mesh import Mesh


def _triangle_strip(count: int) -> Mesh:
    points = []
    faces = []
    for index in range(count):
        offset = len(points)
        base = float(index * 2)
        points.extend([[base, 0, 0], [base + 1, 0, 0], [base, 1, 0]])
        faces.append([offset, offset + 1, offset + 2])
    return Mesh(points=np.asarray(points, dtype=float), faces=np.asarray(faces, dtype=int))


def _cube_mesh(scale: float = 1.0) -> Mesh:
    points = np.asarray(
        [
            [-scale, -scale, -scale],
            [scale, -scale, -scale],
            [scale, scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
            [scale, -scale, scale],
            [scale, scale, scale],
            [-scale, scale, scale],
        ],
        dtype=float,
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
        dtype=int,
    )
    return Mesh(points=points, faces=faces)


def _translated_mesh(mesh: Mesh, x: float, y: float = 0.0, z: float = 0.0) -> Mesh:
    translated = mesh.copy()
    translated.points = translated.points + np.asarray([x, y, z], dtype=float)
    return translated


def _merge_meshes(meshes: list[tuple[Mesh, int]]) -> Mesh:
    points: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    material_indices: list[int] = []
    offset = 0
    for mesh, material_index in meshes:
        points.append(mesh.points)
        faces.append(mesh.faces + offset)
        material_indices.extend([material_index] * mesh.triangle_count)
        offset += mesh.vertex_count
    return Mesh(
        points=np.vstack(points),
        faces=np.vstack(faces),
        material_indices=np.asarray(material_indices, dtype=int),
    )
