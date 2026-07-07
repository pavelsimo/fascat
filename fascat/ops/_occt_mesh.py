from __future__ import annotations

from typing import Any, cast

import numpy as np


def _transformed_occt_nodes(nodes: Any, node_lower: int, node_count: int, transform: Any) -> np.ndarray:
    raw_points = np.empty((node_count, 3), dtype=np.float64)
    for local_index in range(node_count):
        point = nodes.Value(node_lower + local_index)
        raw_points[local_index] = (float(point.X()), float(point.Y()), float(point.Z()))
    matrix = _occt_transform_matrix(transform)
    return cast(np.ndarray, raw_points @ matrix[:, :3].T + matrix[:, 3])


def _occt_transform_matrix(transform: Any) -> np.ndarray:
    return np.asarray(
        [
            [transform.Value(1, 1), transform.Value(1, 2), transform.Value(1, 3), transform.Value(1, 4)],
            [transform.Value(2, 1), transform.Value(2, 2), transform.Value(2, 3), transform.Value(2, 4)],
            [transform.Value(3, 1), transform.Value(3, 2), transform.Value(3, 3), transform.Value(3, 4)],
        ],
        dtype=np.float64,
    )


def _triangulation_faces(triangles: Any, triangle_lower: int, triangle_count: int) -> np.ndarray:
    faces = np.empty((triangle_count, 3), dtype=np.int64)
    for local_index in range(triangle_count):
        faces[local_index] = triangles.Value(triangle_lower + local_index).Get()
    return faces
