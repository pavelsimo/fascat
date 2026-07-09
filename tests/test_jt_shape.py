from __future__ import annotations

import numpy as np
import pytest

from fascat.io.jt.shape import decode_shape_lod
from tests._jt_builder import build_tristrip_shape_lod_payload

_TETRA_POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
_TETRA_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])

_CUBE_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
_CUBE_FACES = np.array(
    [
        [0, 2, 1],
        [0, 3, 2],  # bottom (-z)
        [4, 5, 6],
        [4, 6, 7],  # top (+z)
        [0, 1, 5],
        [0, 5, 4],  # front (-y)
        [2, 3, 7],
        [2, 7, 6],  # back (+y)
        [1, 2, 6],
        [1, 6, 5],  # right (+x)
        [3, 0, 4],
        [3, 4, 7],  # left (-x)
    ]
)


def _grid_box(n: int) -> tuple[np.ndarray, np.ndarray]:
    """A closed box with each face subdivided into an n x n quad grid."""
    import trimesh

    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    for _ in range(n):
        mesh = mesh.subdivide()
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _canonical_per_face(
    points: np.ndarray, faces: np.ndarray, decimals: int = 4
) -> list[tuple[tuple[float, ...], ...]]:
    triangles = []
    for face in faces:
        coords = [tuple(np.round(points[index], decimals).tolist()) for index in face]
        pivot = min(range(3), key=lambda j: coords[j])
        triangles.append(tuple(coords[pivot:] + coords[:pivot]))
    return triangles


def _canonical(points: np.ndarray, faces: np.ndarray, decimals: int = 4) -> list[tuple[tuple[float, ...], ...]]:
    return sorted(_canonical_per_face(points, faces, decimals))


def _decode(payload: bytes, byte_order: str = "<", version: tuple[int, int] = (9, 5)):
    return decode_shape_lod(payload, version=version, byte_order=byte_order)


class TestTopologicalRoundTrip:
    def test_tetrahedron(self) -> None:
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES)
        shape = _decode(payload)
        assert shape.faces.shape == (4, 3)
        assert shape.points.shape == (4, 3)
        assert shape.normals is None
        assert _canonical(shape.points, shape.faces) == _canonical(_TETRA_POINTS, _TETRA_FACES)

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    def test_cube(self, byte_order: str) -> None:
        payload = build_tristrip_shape_lod_payload(_CUBE_POINTS, _CUBE_FACES, byte_order=byte_order)
        shape = _decode(payload, byte_order=byte_order)
        assert shape.faces.shape == (12, 3)
        assert _canonical(shape.points, shape.faces) == _canonical(_CUBE_POINTS, _CUBE_FACES)

    @pytest.mark.parametrize("codec", ["bitlength", "bitlength-fixed", "bitlength-variable", "arithmetic"])
    def test_cube_with_entropy_codecs(self, codec: str) -> None:
        payload = build_tristrip_shape_lod_payload(_CUBE_POINTS, _CUBE_FACES, cdp_codec=codec)
        shape = _decode(payload)
        assert _canonical(shape.points, shape.faces) == _canonical(_CUBE_POINTS, _CUBE_FACES)

    def test_face_groups_preserved(self) -> None:
        groups = [face // 2 for face in range(12)]
        payload = build_tristrip_shape_lod_payload(_CUBE_POINTS, _CUBE_FACES, face_groups=groups)
        shape = _decode(payload)

        def keyed(points: np.ndarray, faces: np.ndarray, values: list[int]) -> dict[object, int]:
            keys = _canonical_per_face(points, faces)
            return dict(zip(keys, [int(v) for v in values], strict=True))

        decoded = keyed(shape.points, shape.faces, shape.face_groups.tolist())
        expected = keyed(_CUBE_POINTS, _CUBE_FACES, groups)
        assert decoded == expected

    def test_subdivided_box_with_splits(self) -> None:
        pytest.importorskip("trimesh")
        points, faces = _grid_box(3)
        payload = build_tristrip_shape_lod_payload(points, faces)
        shape = _decode(payload)
        assert len(shape.faces) == len(faces)
        assert _canonical(shape.points, shape.faces) == _canonical(points, faces)

    def test_quantized_coordinates(self) -> None:
        payload = build_tristrip_shape_lod_payload(_CUBE_POINTS, _CUBE_FACES, quant_bits=12)
        shape = _decode(payload)
        step = 2.0 / ((1 << 12) - 1)
        decoded = _canonical(shape.points, shape.faces, decimals=2)
        expected = _canonical(_CUBE_POINTS, _CUBE_FACES, decimals=2)
        assert decoded == expected
        assert np.all(np.abs(shape.points - np.round(shape.points)) <= step + 1e-9)

    def test_per_vertex_normals(self) -> None:
        normals = _TETRA_POINTS - 0.25
        normals = normals / np.linalg.norm(normals + 1e-9, axis=1, keepdims=True)
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES, normals=normals)
        shape = _decode(payload)
        assert shape.normals is not None
        assert shape.points.shape == shape.normals.shape == (4, 3)
        # Normals must follow their vertices through the visit-order permutation.
        for index in range(4):
            source = np.where(np.all(np.isclose(_TETRA_POINTS, shape.points[index]), axis=1))[0][0]
            assert np.allclose(shape.normals[index], normals[source], atol=1e-6)

    def test_multiple_components(self) -> None:
        # Two disjoint tetrahedra in one shape.
        points = np.vstack([_TETRA_POINTS, _TETRA_POINTS + 10.0])
        faces = np.vstack([_TETRA_FACES, _TETRA_FACES + 4])
        payload = build_tristrip_shape_lod_payload(points, faces)
        shape = _decode(payload)
        assert len(shape.faces) == 8
        assert _canonical(shape.points, shape.faces) == _canonical(points, faces)


class TestJt10Decode:
    def test_round_trips_tetrahedron(self) -> None:
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES, version=(10, 0))
        shape = _decode(payload, version=(10, 0))
        assert len(shape.faces) == 4
        assert _canonical(shape.points, shape.faces) == _canonical(_TETRA_POINTS, _TETRA_FACES)

    def test_round_trips_quantized_coordinates(self) -> None:
        payload = build_tristrip_shape_lod_payload(_CUBE_POINTS, _CUBE_FACES, version=(10, 0), quant_bits=14)
        shape = _decode(payload, version=(10, 0))
        assert len(shape.faces) == 12
        assert _canonical(shape.points, shape.faces, decimals=3) == _canonical(_CUBE_POINTS, _CUBE_FACES, decimals=3)

    def test_round_trips_normals(self) -> None:
        normals = _TETRA_POINTS - _TETRA_POINTS.mean(axis=0)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES, normals=normals, version=(10, 0))
        shape = _decode(payload, version=(10, 0))
        assert shape.normals is not None
        assert np.allclose(np.linalg.norm(shape.normals, axis=1), 1.0, atol=1e-6)

    def test_corrupt_topology_hash_raises(self) -> None:
        payload = build_tristrip_shape_lod_payload(
            _TETRA_POINTS, _TETRA_FACES, version=(10, 0), corrupt_topology_hash=True
        )
        with pytest.raises(RuntimeError, match="topology hash mismatch"):
            _decode(payload, version=(10, 0))

    def test_v9_payload_under_v10_version_rejected(self) -> None:
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES)
        with pytest.raises(RuntimeError, match="unsupported JT 10 shape LOD payload element"):
            _decode(payload, version=(10, 5))


class TestShapeErrors:
    def test_topology_hash_mismatch(self) -> None:
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES, corrupt_topology_hash=True)
        with pytest.raises(RuntimeError, match="topology hash mismatch"):
            _decode(payload)

    def test_unsupported_element_guid(self) -> None:
        payload = bytearray(build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES))
        payload[4] ^= 0xFF  # clobber the element type GUID
        with pytest.raises(RuntimeError, match="unsupported JT shape LOD element"):
            _decode(bytes(payload))

    def test_open_mesh_rejected_by_builder(self) -> None:
        with pytest.raises(ValueError, match="not closed"):
            build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES[:3])

    def test_truncated_payload(self) -> None:
        payload = build_tristrip_shape_lod_payload(_TETRA_POINTS, _TETRA_FACES)
        with pytest.raises(RuntimeError, match="truncated JT"):
            _decode(payload[: len(payload) // 2])
