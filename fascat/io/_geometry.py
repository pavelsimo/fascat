from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def transform_points(points: FloatArray, transform: FloatArray) -> FloatArray:
    """Apply a homogeneous transform to 3D points."""
    if points.shape[0] == 0:
        return np.asarray(points.copy(), dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return np.asarray((transform @ homogeneous.T).T[:, :3], dtype=np.float64)


def normalize_rows(
    values: FloatArray,
    *,
    degenerate: Sequence[float] | None = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Normalize vectors row-wise and apply the requested degenerate fallback."""
    lengths = np.linalg.norm(values, axis=1)
    result = np.zeros_like(values, dtype=np.float64)
    nonzero = lengths > 0.0
    result[nonzero] = values[nonzero] / lengths[nonzero, None]
    if degenerate is not None:
        result[~nonzero] = np.asarray(degenerate, dtype=np.float64)
    return result


def face_normals(
    points: FloatArray,
    faces: IntArray,
    *,
    degenerate: Sequence[float] | None = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Calculate one normalized geometric normal per triangle."""
    if faces.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    triangles = points[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return normalize_rows(normals, degenerate=degenerate)


def transform_normals(
    normals: FloatArray,
    transform: FloatArray,
    *,
    degenerate: Sequence[float] | None = (0.0, 0.0, 1.0),
) -> FloatArray:
    """Apply the inverse-transpose normal matrix and renormalize."""
    linear = np.asarray(transform[:3, :3], dtype=np.float64)
    try:
        normal_matrix = np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        normal_matrix = linear
    transformed = np.asarray(normals @ normal_matrix.T, dtype=np.float64)
    return normalize_rows(transformed, degenerate=degenerate)
