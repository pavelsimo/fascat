from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from fascat.errors import FascatError


def invert_transform(transform: np.ndarray, *, label: str) -> NDArray[np.float64]:
    """Invert a world transform, naming the offending node when it is singular.

    Zero-scale nodes (suppressed components, degenerate mirrors, flattened
    subassemblies) are routine in real assemblies, so surface them as a
    ``FascatError`` that names the node instead of letting a bare
    ``LinAlgError: Singular matrix`` escape the library's error taxonomy.
    """
    try:
        return np.asarray(np.linalg.inv(transform), dtype=np.float64)
    except np.linalg.LinAlgError:
        raise FascatError(
            f"cannot invert the world transform of {label}: the transform is singular "
            "(zero scale or a collapsed node); fix or exclude the node before this operation"
        ) from None


def array_digest(values: np.ndarray | None) -> str | None:
    if values is None:
        return None
    array = np.ascontiguousarray(values)
    digest = hashlib.sha1()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def array_digest_required(values: np.ndarray) -> str:
    digest = array_digest(values)
    assert digest is not None
    return digest


def sliced_face_lookup(face_indices: np.ndarray, triangle_count: int) -> NDArray[np.int64]:
    indices = np.asarray(face_indices, dtype=np.int64)
    face_lookup = np.full(int(triangle_count), -1, dtype=np.int64)
    face_lookup[indices] = np.arange(indices.shape[0], dtype=np.int64)
    return face_lookup


def remap_sliced_face_group(face_lookup: np.ndarray, group: np.ndarray) -> NDArray[np.int64]:
    group_indices = np.asarray(group, dtype=np.int64)
    in_bounds = (group_indices >= 0) & (group_indices < face_lookup.shape[0])
    if not bool(np.any(in_bounds)):
        return np.asarray([], dtype=np.int64)
    remapped = face_lookup[group_indices[in_bounds]]
    return np.asarray(remapped[remapped >= 0], dtype=np.int64)
