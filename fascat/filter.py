from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

PatternValue = str | Sequence[str]
FilterMode = Literal["criteria", "all", "any", "not"]
FloatArray = NDArray[np.float64]


class FilterExpressionError(ValueError):
    """Raised when a CLI filter expression cannot be parsed."""


@dataclass(frozen=True)
class SelectionMatch:
    node_id: str
    node_path: str
    node_name: str
    part_id: str | None
    part_name: str | None
    material_ids: tuple[str, ...] = ()
    vertices: int = 0
    triangles: int = 0
    bounds_min: tuple[float, float, float] | None = None
    bounds_max: tuple[float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "node_id": self.node_id,
            "node_path": self.node_path,
            "node_name": self.node_name,
            "part_id": self.part_id,
            "part_name": self.part_name,
            "material_ids": list(self.material_ids),
            "vertices": self.vertices,
            "triangles": self.triangles,
        }
        if self.bounds_min is not None and self.bounds_max is not None:
            payload["bounds"] = {"min": list(self.bounds_min), "max": list(self.bounds_max)}
        return payload


@dataclass(frozen=True)
class SelectionResult:
    filter: Filter
    matches: tuple[SelectionMatch, ...] = ()

    @property
    def part_ids(self) -> set[str]:
        return {match.part_id for match in self.matches if match.part_id is not None}

    @property
    def node_ids(self) -> set[str]:
        return {match.node_id for match in self.matches}

    def stats(self) -> dict[str, int]:
        part_ids = self.part_ids
        material_ids = {material_id for match in self.matches for material_id in match.material_ids}
        vertices_by_part: dict[str, int] = {}
        triangles_by_part: dict[str, int] = {}
        for match in self.matches:
            if match.part_id is None:
                continue
            vertices_by_part.setdefault(match.part_id, match.vertices)
            triangles_by_part.setdefault(match.part_id, match.triangles)
        return {
            "nodes": len(self.matches),
            "parts": len(part_ids),
            "occurrences": sum(1 for match in self.matches if match.part_id is not None),
            "materials": len(material_ids),
            "vertices": sum(vertices_by_part.values()),
            "triangles": sum(triangles_by_part.values()),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "filter": self.filter.to_dict(),
            "stats": self.stats(),
            "matches": [match.to_dict() for match in self.matches],
        }


@dataclass(frozen=True)
class _FilterContext:
    node: Any
    node_path: str
    part: Any | None
    materials: tuple[Any, ...]
    bounds: tuple[FloatArray, FloatArray] | None


class Filter:
    """Reusable selector for applying operations to assembly subsets."""

    def __init__(
        self,
        *,
        path: PatternValue | None = None,
        name: PatternValue | None = None,
        part_name: PatternValue | None = None,
        part_id: PatternValue | None = None,
        material: PatternValue | None = None,
        metadata: dict[str, object] | None = None,
        min_bounds: Sequence[float] | None = None,
        max_bounds: Sequence[float] | None = None,
        min_diagonal: float | None = None,
        max_diagonal: float | None = None,
        min_triangles: int | None = None,
        max_triangles: int | None = None,
        min_vertices: int | None = None,
        max_vertices: int | None = None,
        include: Sequence[Filter] | None = None,
        exclude: Sequence[Filter] | None = None,
        _mode: FilterMode = "criteria",
        _children: Sequence[Filter] = (),
    ) -> None:
        self.path_patterns = _patterns(path)
        self.name_patterns = _patterns(name)
        self.part_name_patterns = _patterns(part_name)
        self.part_id_patterns = _patterns(part_id)
        self.material_patterns = _patterns(material)
        self.metadata = dict(metadata or {})
        self.min_bounds = _point(min_bounds, "min_bounds") if min_bounds is not None else None
        self.max_bounds = _point(max_bounds, "max_bounds") if max_bounds is not None else None
        self.min_diagonal = _non_negative_float(min_diagonal, "min_diagonal")
        self.max_diagonal = _non_negative_float(max_diagonal, "max_diagonal")
        self.min_triangles = _non_negative_int(min_triangles, "min_triangles")
        self.max_triangles = _non_negative_int(max_triangles, "max_triangles")
        self.min_vertices = _non_negative_int(min_vertices, "min_vertices")
        self.max_vertices = _non_negative_int(max_vertices, "max_vertices")
        self.include = tuple(include or ())
        self.exclude = tuple(exclude or ())
        self._mode = _mode
        self._children = tuple(_coerce_filter(child) for child in _children)
        if self._mode not in {"criteria", "all", "any", "not"}:
            raise ValueError("filter mode must be one of: criteria, all, any, not")
        if self._mode == "not" and len(self._children) != 1:
            raise ValueError("not filters must contain exactly one child filter")
        if self.min_diagonal is not None and self.max_diagonal is not None and self.min_diagonal > self.max_diagonal:
            raise ValueError("min_diagonal must be less than or equal to max_diagonal")
        if (
            self.min_triangles is not None
            and self.max_triangles is not None
            and self.min_triangles > self.max_triangles
        ):
            raise ValueError("min_triangles must be less than or equal to max_triangles")
        if self.min_vertices is not None and self.max_vertices is not None and self.min_vertices > self.max_vertices:
            raise ValueError("min_vertices must be less than or equal to max_vertices")

    @classmethod
    def path(cls, value: PatternValue) -> Filter:
        return cls(path=value)

    @classmethod
    def name(cls, value: PatternValue) -> Filter:
        return cls(name=value)

    @classmethod
    def part(cls, value: PatternValue) -> Filter:
        return cls(part_id=value)

    @classmethod
    def part_name(cls, value: PatternValue) -> Filter:
        return cls(part_name=value)

    @classmethod
    def material(cls, value: PatternValue) -> Filter:
        return cls(material=value)

    @classmethod
    def size(cls, *, min_diagonal: float | None = None, max_diagonal: float | None = None) -> Filter:
        return cls(min_diagonal=min_diagonal, max_diagonal=max_diagonal)

    @classmethod
    def triangle_count(cls, *, min: int | None = None, max: int | None = None) -> Filter:  # noqa: A002
        return cls(min_triangles=min, max_triangles=max)

    @classmethod
    def vertex_count(cls, *, min: int | None = None, max: int | None = None) -> Filter:  # noqa: A002
        return cls(min_vertices=min, max_vertices=max)

    @classmethod
    def bounds(cls, *, min: Sequence[float] | None = None, max: Sequence[float] | None = None) -> Filter:  # noqa: A002
        return cls(min_bounds=min, max_bounds=max)

    @classmethod
    def metadata_value(cls, key: str, value: object) -> Filter:
        return cls(metadata={key: value})

    @classmethod
    def all(cls, *filters: Filter) -> Filter:
        return cls(_mode="all", _children=filters)

    @classmethod
    def any(cls, *filters: Filter) -> Filter:
        return cls(_mode="any", _children=filters)

    @classmethod
    def not_(cls, filter: Filter) -> Filter:
        return cls(_mode="not", _children=(filter,))

    @classmethod
    def from_cli(cls, expressions: Sequence[str], *, exclude: Sequence[str] = ()) -> Filter | None:
        include_filters = tuple(parse_filter_expression(expression) for expression in expressions)
        exclude_filters = tuple(parse_filter_expression(expression) for expression in exclude)
        if not include_filters and not exclude_filters:
            return None
        base = (
            cls.all(*include_filters)
            if len(include_filters) > 1
            else (include_filters[0] if include_filters else cls())
        )
        if not exclude_filters:
            return base
        return cls(include=(base,), exclude=exclude_filters)

    @classmethod
    def from_value(cls, value: Filter | None) -> Filter | None:
        if value is None:
            return None
        if isinstance(value, Filter):
            return value
        raise TypeError("where must be a fascat.Filter or None")

    def select(self, asset: Any) -> SelectionResult:
        matches: list[SelectionMatch] = []
        self._collect_matches(
            asset, asset.root, _node_name(asset.root), ancestor_path="", ancestor_selected=False, matches=matches
        )
        return SelectionResult(filter=self, matches=tuple(matches))

    def matches(self, context: _FilterContext) -> bool:
        if self._requires_part_context() and context.part is None:
            return False
        if self._excluded(context):
            return False
        if self.include and not any(item.matches(context) for item in self.include):
            return False
        if self._mode == "all":
            return all(child.matches(context) for child in self._children)
        if self._mode == "any":
            return any(child.matches(context) for child in self._children)
        if self._mode == "not":
            return not self._children[0].matches(context)
        return self._criteria_matches(context)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"mode": self._mode}
        if self._children:
            payload["children"] = [child.to_dict() for child in self._children]
        if self.include:
            payload["include"] = [item.to_dict() for item in self.include]
        if self.exclude:
            payload["exclude"] = [item.to_dict() for item in self.exclude]
        criteria: dict[str, object] = {}
        for key, attribute in (
            ("path", "path_patterns"),
            ("name", "name_patterns"),
            ("part_name", "part_name_patterns"),
            ("part_id", "part_id_patterns"),
            ("material", "material_patterns"),
        ):
            value = getattr(self, attribute)
            if value:
                criteria[key] = list(value)
        if self.metadata:
            criteria["metadata"] = dict(self.metadata)
        if self.min_bounds is not None:
            criteria["min_bounds"] = list(self.min_bounds)
        if self.max_bounds is not None:
            criteria["max_bounds"] = list(self.max_bounds)
        for key in (
            "min_diagonal",
            "max_diagonal",
            "min_triangles",
            "max_triangles",
            "min_vertices",
            "max_vertices",
        ):
            value = getattr(self, key)
            if value is not None:
                criteria[key] = value
        if criteria:
            payload["criteria"] = criteria
        return payload

    def _collect_matches(
        self,
        asset: Any,
        node: Any,
        node_name: str,
        *,
        ancestor_path: str,
        ancestor_selected: bool,
        matches: list[SelectionMatch],
    ) -> None:
        node_path = f"{ancestor_path}/{node_name}" if ancestor_path else node_name
        context = _context_for(asset, node, node_path)
        direct = self.matches(context)
        selected = (ancestor_selected or direct) and not self._excluded(context)
        if selected:
            matches.append(_match_from_context(context))
        for child in node.children:
            self._collect_matches(
                asset,
                child,
                _node_name(child),
                ancestor_path=node_path,
                ancestor_selected=selected,
                matches=matches,
            )

    def _excluded(self, context: _FilterContext) -> bool:
        return any(item.matches(context) for item in self.exclude)

    def _criteria_matches(self, context: _FilterContext) -> bool:
        if self._requires_part_context() and context.part is None:
            return False
        if self.path_patterns and not _matches_any(context.node_path, self.path_patterns):
            return False
        if self.name_patterns and not _matches_any(_node_name(context.node), self.name_patterns):
            return False
        if self.part_id_patterns and (context.part is None or not _matches_any(context.part.id, self.part_id_patterns)):
            return False
        if self.part_name_patterns and (
            context.part is None or not _matches_any(context.part.name, self.part_name_patterns)
        ):
            return False
        if self.material_patterns and not _materials_match(context.materials, self.material_patterns):
            return False
        if self.metadata and not _metadata_matches(context, self.metadata):
            return False
        if not _bounds_match(context.bounds, self.min_bounds, self.max_bounds, self.min_diagonal, self.max_diagonal):
            return False
        triangles = _triangle_count(context.part)
        vertices = _vertex_count(context.part)
        if self.min_triangles is not None and triangles < self.min_triangles:
            return False
        if self.max_triangles is not None and triangles > self.max_triangles:
            return False
        if self.min_vertices is not None and vertices < self.min_vertices:
            return False
        return not (self.max_vertices is not None and vertices > self.max_vertices)

    def _requires_part_context(self) -> bool:
        if self._mode == "not":
            return self._children[0]._requires_part_context()
        if self._mode == "all":
            return any(child._requires_part_context() for child in self._children)
        if self._mode == "any":
            return bool(self._children) and all(child._requires_part_context() for child in self._children)
        return bool(
            self.part_id_patterns
            or self.part_name_patterns
            or self.material_patterns
            or self.min_bounds is not None
            or self.max_bounds is not None
            or self.min_diagonal is not None
            or self.max_diagonal is not None
            or self.min_triangles is not None
            or self.max_triangles is not None
            or self.min_vertices is not None
            or self.max_vertices is not None
        )


def parse_filter_expression(expression: str) -> Filter:
    expr = expression.strip()
    if not expr:
        raise FilterExpressionError("filter expression must not be empty")

    for operator in ("<=", ">=", "="):
        if operator not in expr:
            continue
        key, raw_value = expr.split(operator, 1)
        key = key.strip().replace("-", "_").lower()
        value = raw_value.strip()
        if not key or not value:
            raise FilterExpressionError(f"invalid filter expression: {expression}")
        return _filter_from_key_value(key, operator, value)
    raise FilterExpressionError(f"invalid filter expression: {expression}")


def _filter_from_key_value(key: str, operator: str, value: str) -> Filter:
    if key == "path" and operator == "=":
        return Filter.path(value)
    if key in {"name", "node_name"} and operator == "=":
        return Filter.name(value)
    if key in {"part", "part_id"} and operator == "=":
        return Filter.part(value)
    if key == "part_name" and operator == "=":
        return Filter.part_name(value)
    if key == "material" and operator == "=":
        return Filter.material(value)
    if key.startswith("metadata.") and operator == "=":
        return Filter.metadata_value(key.removeprefix("metadata."), value)
    if key.startswith("metadata:") and operator == "=":
        return Filter.metadata_value(key.removeprefix("metadata:"), value)
    if key in {"triangles", "triangle_count"}:
        count = _parse_int(value, key)
        return (
            Filter.triangle_count(min=count if operator == ">=" else None, max=count if operator == "<=" else None)
            if operator != "="
            else Filter.triangle_count(min=count, max=count)
        )
    if key in {"vertices", "vertex_count"}:
        count = _parse_int(value, key)
        return (
            Filter.vertex_count(min=count if operator == ">=" else None, max=count if operator == "<=" else None)
            if operator != "="
            else Filter.vertex_count(min=count, max=count)
        )
    if key in {"size", "diagonal"}:
        diagonal = _parse_float(value, key)
        return (
            Filter.size(
                min_diagonal=diagonal if operator == ">=" else None, max_diagonal=diagonal if operator == "<=" else None
            )
            if operator != "="
            else Filter.size(min_diagonal=diagonal, max_diagonal=diagonal)
        )
    raise FilterExpressionError(f"unsupported filter expression: {key}{operator}{value}")


def _context_for(asset: Any, node: Any, node_path: str) -> _FilterContext:
    part = asset.parts.get(node.part_id) if node.part_id is not None else None
    materials = (
        tuple(asset.materials[material_id] for material_id in part.material_ids if material_id in asset.materials)
        if part is not None
        else ()
    )
    bounds = _node_bounds(part, node.transform) if part is not None else None
    return _FilterContext(node=node, node_path=node_path, part=part, materials=materials, bounds=bounds)


def _match_from_context(context: _FilterContext) -> SelectionMatch:
    part = context.part
    bounds_min = None
    bounds_max = None
    if context.bounds is not None:
        bounds_min = _tuple3_from_array(context.bounds[0])
        bounds_max = _tuple3_from_array(context.bounds[1])
    return SelectionMatch(
        node_id=cast(str, context.node.id),
        node_path=context.node_path,
        node_name=_node_name(context.node),
        part_id=None if part is None else cast(str, part.id),
        part_name=None if part is None else cast(str, part.name),
        material_ids=() if part is None else tuple(cast(Sequence[str], part.material_ids)),
        vertices=_vertex_count(part),
        triangles=_triangle_count(part),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def _node_bounds(part: Any, transform: FloatArray) -> tuple[FloatArray, FloatArray] | None:
    if part.mesh is None:
        return None
    mins, maxs = part.mesh.bounds()
    corners = np.array(
        [
            [mins[0], mins[1], mins[2], 1.0],
            [mins[0], mins[1], maxs[2], 1.0],
            [mins[0], maxs[1], mins[2], 1.0],
            [mins[0], maxs[1], maxs[2], 1.0],
            [maxs[0], mins[1], mins[2], 1.0],
            [maxs[0], mins[1], maxs[2], 1.0],
            [maxs[0], maxs[1], mins[2], 1.0],
            [maxs[0], maxs[1], maxs[2], 1.0],
        ],
        dtype=np.float64,
    )
    transformed = (np.asarray(transform, dtype=np.float64) @ corners.T).T[:, :3]
    return transformed.min(axis=0), transformed.max(axis=0)


def _bounds_match(
    bounds: tuple[FloatArray, FloatArray] | None,
    min_bounds: tuple[float, float, float] | None,
    max_bounds: tuple[float, float, float] | None,
    min_diagonal: float | None,
    max_diagonal: float | None,
) -> bool:
    if min_bounds is None and max_bounds is None and min_diagonal is None and max_diagonal is None:
        return True
    if bounds is None:
        return False
    mins, maxs = bounds
    if min_bounds is not None and np.any(maxs < np.asarray(min_bounds, dtype=np.float64)):
        return False
    if max_bounds is not None and np.any(mins > np.asarray(max_bounds, dtype=np.float64)):
        return False
    diagonal = float(np.linalg.norm(maxs - mins))
    if min_diagonal is not None and diagonal < min_diagonal:
        return False
    return not (max_diagonal is not None and diagonal > max_diagonal)


def _materials_match(materials: tuple[Any, ...], patterns: tuple[str, ...]) -> bool:
    for material in materials:
        if _matches_any(cast(str, material.id), patterns) or _matches_any(cast(str, material.name), patterns):
            return True
    return False


def _metadata_matches(context: _FilterContext, expected: dict[str, object]) -> bool:
    metadata_sources: list[dict[str, object]] = [cast(dict[str, object], context.node.metadata)]
    if context.part is not None:
        metadata_sources.append(cast(dict[str, object], context.part.metadata))
        if context.part.mesh is not None:
            metadata_sources.append(cast(dict[str, object], context.part.mesh.metadata))
    metadata_sources.extend(cast(dict[str, object], material.metadata) for material in context.materials)
    for key, expected_value in expected.items():
        if not any(_metadata_value_matches(source.get(key), expected_value) for source in metadata_sources):
            return False
    return True


def _metadata_value_matches(actual: object, expected: object) -> bool:
    if actual is None:
        return False
    if isinstance(expected, str):
        return fnmatchcase(str(actual), expected)
    return actual == expected


def _matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(value, pattern) for pattern in patterns)


def _triangle_count(part: Any | None) -> int:
    if part is None or part.mesh is None:
        return 0
    return cast(int, part.mesh.triangle_count)


def _vertex_count(part: Any | None) -> int:
    if part is None or part.mesh is None:
        return 0
    return cast(int, part.mesh.vertex_count)


def _node_name(node: Any) -> str:
    return cast(str, node.name or node.id)


def _coerce_filter(value: Filter) -> Filter:
    if not isinstance(value, Filter):
        raise TypeError("logical filter children must be fascat.Filter instances")
    return value


def _patterns(value: PatternValue | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _point(value: Sequence[float], label: str) -> tuple[float, float, float]:
    point = tuple(float(item) for item in value)
    if len(point) != 3:
        raise ValueError(f"{label} must contain 3 values")
    if not np.isfinite(point).all():
        raise ValueError(f"{label} values must be finite")
    return (point[0], point[1], point[2])


def _tuple3_from_array(values: FloatArray) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _non_negative_float(value: float | None, label: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a non-negative finite number")
    return number


def _non_negative_int(value: int | None, label: str) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _parse_int(value: str, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise FilterExpressionError(f"{key} filter value must be an integer") from exc
    if parsed < 0:
        raise FilterExpressionError(f"{key} filter value must be non-negative")
    return parsed


def _parse_float(value: str, key: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise FilterExpressionError(f"{key} filter value must be a number") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise FilterExpressionError(f"{key} filter value must be a non-negative finite number")
    return parsed
