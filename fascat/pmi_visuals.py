from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node
from fascat.metadata import pmi_ids_by_part

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PmiVisualMarker:
    annotation_id: str
    kind: str
    text: str
    applies_to: tuple[str, ...]
    current_part_ids: tuple[str, ...]
    points: FloatArray
    faces: IntArray
    anchor: tuple[float, float, float]


@dataclass(frozen=True)
class _Bounds:
    minimum: FloatArray
    maximum: FloatArray

    @property
    def center(self) -> FloatArray:
        return (self.minimum + self.maximum) * 0.5

    @property
    def diagonal_length(self) -> float:
        return float(np.linalg.norm(self.maximum - self.minimum))


_MARKER_DIRECTIONS = (
    np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
    np.asarray([-1.0, 1.0, 1.0], dtype=np.float64),
    np.asarray([1.0, 1.0, -1.0], dtype=np.float64),
    np.asarray([-1.0, 1.0, -1.0], dtype=np.float64),
    np.asarray([1.0, -1.0, 1.0], dtype=np.float64),
    np.asarray([-1.0, -1.0, 1.0], dtype=np.float64),
    np.asarray([1.0, -1.0, -1.0], dtype=np.float64),
    np.asarray([-1.0, -1.0, -1.0], dtype=np.float64),
)


def build_pmi_visual_markers(
    asset: Asset,
    *,
    space: FloatArray | None = None,
    include_root_transform: bool = True,
) -> list[PmiVisualMarker]:
    """Build deterministic marker meshes for PMI records.

    These markers intentionally do not attempt to reconstruct AP242 text glyphs.
    They provide a renderable anchor, leader, and label plate that stays linked to
    the structured PMI metadata.
    """

    if not asset.pmi:
        return []
    space_matrix = np.eye(4, dtype=np.float64) if space is None else np.asarray(space, dtype=np.float64)
    if space_matrix.shape != (4, 4):
        raise ValueError("PMI visual export space must have shape (4, 4)")

    part_bounds, scene_bounds = _part_scene_bounds(
        asset,
        space_matrix,
        include_root_transform=include_root_transform,
    )
    annotation_parts = _annotation_current_parts(asset)
    markers: list[PmiVisualMarker] = []
    for index, annotation in enumerate(asset.pmi):
        current_part_ids = tuple(annotation_parts.get(annotation.id, ()))
        target_bounds = _merge_bounds(part_bounds[part_id] for part_id in current_part_ids if part_id in part_bounds)
        if target_bounds is None:
            target_bounds = scene_bounds
        points, faces, anchor = _marker_geometry(
            target_bounds,
            scene_bounds,
            index=index,
            text=annotation.text,
        )
        markers.append(
            PmiVisualMarker(
                annotation_id=annotation.id,
                kind=annotation.kind,
                text=annotation.text,
                applies_to=tuple(annotation.applies_to),
                current_part_ids=current_part_ids,
                points=points,
                faces=faces,
                anchor=(float(anchor[0]), float(anchor[1]), float(anchor[2])),
            )
        )
    return markers


def _annotation_current_parts(asset: Asset) -> dict[str, tuple[str, ...]]:
    by_part = pmi_ids_by_part(asset.parts, asset.pmi)
    result: dict[str, list[str]] = {}
    for part_id in sorted(by_part):
        for annotation_id in by_part[part_id]:
            items = result.setdefault(annotation_id, [])
            if part_id not in items:
                items.append(part_id)
    return {annotation_id: tuple(part_ids) for annotation_id, part_ids in result.items()}


def _part_scene_bounds(
    asset: Asset,
    space: FloatArray,
    *,
    include_root_transform: bool,
) -> tuple[dict[str, _Bounds], _Bounds | None]:
    occurrences = _occurrence_transforms(asset.root, include_root_transform=include_root_transform)
    part_bounds: dict[str, _Bounds] = {}
    scene_items: list[_Bounds] = []
    for part_id in sorted(asset.parts):
        part = asset.parts[part_id]
        if part.mesh is None or part.mesh.vertex_count == 0:
            continue
        transforms = occurrences.get(part_id)
        if not transforms:
            transforms = (np.eye(4, dtype=np.float64),)
        bounds_items = [
            _points_bounds(_transform_points(part.mesh.points, space @ transform)) for transform in transforms
        ]
        bounds = _merge_bounds(bounds_items)
        if bounds is None:
            continue
        part_bounds[part_id] = bounds
        scene_items.append(bounds)
    return part_bounds, _merge_bounds(scene_items)


def _occurrence_transforms(
    root: Node,
    *,
    include_root_transform: bool,
) -> dict[str, tuple[FloatArray, ...]]:
    result: dict[str, list[FloatArray]] = {}
    root_transform = root.transform if include_root_transform else np.eye(4, dtype=np.float64)

    def walk(node: Node, world: FloatArray) -> None:
        if node.part_id is not None:
            result.setdefault(node.part_id, []).append(world.copy())
        for child in node.children:
            walk(child, world @ child.transform)

    walk(root, root_transform)
    return {part_id: tuple(transforms) for part_id, transforms in result.items()}


def _marker_geometry(
    bounds: _Bounds | None,
    scene_bounds: _Bounds | None,
    *,
    index: int,
    text: str,
) -> tuple[FloatArray, IntArray, FloatArray]:
    if bounds is None:
        minimum = np.zeros(3, dtype=np.float64)
        maximum = np.zeros(3, dtype=np.float64)
        bounds = _Bounds(minimum=minimum, maximum=maximum)
    direction = _normalize(_MARKER_DIRECTIONS[index % len(_MARKER_DIRECTIONS)])
    size = _marker_size(bounds, scene_bounds)
    corner = np.where(direction >= 0.0, bounds.maximum, bounds.minimum)
    anchor = corner + direction * size * 2.5
    right, up = _marker_basis(direction)
    right = right * size
    up = up * size

    vertices: list[FloatArray] = []
    faces: list[tuple[int, int, int]] = []

    _append_leader(vertices, faces, bounds.center, anchor, right)
    _append_diamond(vertices, faces, anchor, right, up)
    _append_label_plate(vertices, faces, anchor, right, up, text)
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64), anchor


def _append_leader(
    vertices: list[FloatArray],
    faces: list[tuple[int, int, int]],
    target: FloatArray,
    anchor: FloatArray,
    right: FloatArray,
) -> None:
    half_width = right * 0.08
    start = len(vertices)
    vertices.extend(
        [
            target - half_width,
            target + half_width,
            anchor + half_width,
            target - half_width,
            anchor + half_width,
            anchor - half_width,
        ]
    )
    faces.extend([(start, start + 1, start + 2), (start + 3, start + 4, start + 5)])


def _append_diamond(
    vertices: list[FloatArray],
    faces: list[tuple[int, int, int]],
    center: FloatArray,
    right: FloatArray,
    up: FloatArray,
) -> None:
    start = len(vertices)
    vertices.extend(
        [
            center,
            center + right,
            center + up,
            center - right,
            center - up,
        ]
    )
    faces.extend(
        [
            (start, start + 1, start + 2),
            (start, start + 2, start + 3),
            (start, start + 3, start + 4),
            (start, start + 4, start + 1),
        ]
    )


def _append_label_plate(
    vertices: list[FloatArray],
    faces: list[tuple[int, int, int]],
    anchor: FloatArray,
    right: FloatArray,
    up: FloatArray,
    text: str,
) -> None:
    text_width = min(max(len(text), 1), 24)
    half_width = _normalize(right) * np.linalg.norm(right) * (0.9 + 0.025 * text_width)
    half_height = _normalize(up) * np.linalg.norm(up) * 0.35
    center = anchor + _normalize(right) * np.linalg.norm(right) * (1.35 + 0.025 * text_width)
    start = len(vertices)
    vertices.extend(
        [
            center - half_width - half_height,
            center + half_width - half_height,
            center + half_width + half_height,
            center - half_width + half_height,
        ]
    )
    faces.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])


def _marker_size(bounds: _Bounds, scene_bounds: _Bounds | None) -> float:
    target_size = bounds.diagonal_length * 0.08
    scene_size = 0.0 if scene_bounds is None else scene_bounds.diagonal_length * 0.035
    size = max(target_size, scene_size)
    return size if size > 0.0 else 1.0


def _marker_basis(direction: FloatArray) -> tuple[FloatArray, FloatArray]:
    world_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(world_up, direction)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.cross(np.asarray([1.0, 0.0, 0.0], dtype=np.float64), direction)
    right = _normalize(right)
    up = _normalize(np.cross(direction, right))
    return right, up


def _points_bounds(points: FloatArray) -> _Bounds:
    if points.size == 0:
        zeros = np.zeros(3, dtype=np.float64)
        return _Bounds(minimum=zeros, maximum=zeros)
    return _Bounds(
        minimum=points.min(axis=0),
        maximum=points.max(axis=0),
    )


def _merge_bounds(items: Iterable[_Bounds | None]) -> _Bounds | None:
    bounds = [item for item in items if item is not None]
    if not bounds:
        return None
    return _Bounds(
        minimum=np.vstack([item.minimum for item in bounds]).min(axis=0),
        maximum=np.vstack([item.maximum for item in bounds]).max(axis=0),
    )


def _transform_points(points: FloatArray, transform: FloatArray) -> FloatArray:
    if not len(points):
        return points
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return cast(FloatArray, np.asarray((transform @ homogeneous.T).T[:, :3], dtype=np.float64))


def _normalize(value: FloatArray) -> FloatArray:
    length = float(np.linalg.norm(value))
    if length <= 0.0:
        return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    return value / length
