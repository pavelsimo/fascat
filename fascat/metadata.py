from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal

Metadata = dict[str, object]
PmiKind = Literal[
    "dimension",
    "tolerance",
    "datum",
    "datum_target",
    "feature_control_frame",
    "note",
    "saved_view",
    "annotation_plane",
]


@dataclass(frozen=True)
class Tolerance:
    upper: float | None = None
    lower: float | None = None
    kind: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable tolerance record."""
        return {
            "upper": self.upper,
            "lower": self.lower,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class PmiAnnotation:
    id: str
    kind: PmiKind
    text: str
    value: float | None = None
    unit: str | None = None
    tolerance: Tolerance | None = None
    applies_to: list[str] = field(default_factory=list)
    view: str | None = None
    plane: list[list[float]] | None = None
    source: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "applies_to", list(self.applies_to))
        object.__setattr__(self, "source", dict(self.source))
        if self.plane is not None:
            object.__setattr__(self, "plane", [list(row) for row in self.plane])

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable PMI annotation."""
        payload: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "type": self.kind,
            "text": self.text,
            "applies_to": list(self.applies_to),
            "appliesToPartIds": list(self.applies_to),
            "source": dict(self.source),
        }
        if self.value is not None:
            payload["value"] = self.value
        if self.unit is not None:
            payload["unit"] = self.unit
        if self.tolerance is not None:
            payload["tolerance"] = self.tolerance.to_dict()
        if self.view is not None:
            payload["view"] = self.view
        if self.plane is not None:
            payload["plane"] = [list(row) for row in self.plane]
        return payload


def pmi_ids_by_part(parts: Mapping[str, object], annotations: Iterable[PmiAnnotation]) -> dict[str, list[str]]:
    target_to_parts = _pmi_target_to_current_parts(parts)
    result: dict[str, list[str]] = {}
    for annotation in annotations:
        for target in annotation.applies_to:
            for part_id in target_to_parts.get(target, ()):
                ids = result.setdefault(part_id, [])
                if annotation.id not in ids:
                    ids.append(annotation.id)
    return result


def _pmi_target_to_current_parts(parts: Mapping[str, object]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for part_id, part in parts.items():
        current_id = str(part_id)
        _append_unique(result, current_id, current_id)
        metadata = getattr(part, "metadata", {})
        if isinstance(metadata, Mapping):
            for key in ("source_part_id", "source_part_ids"):
                for source_id in _metadata_id_list(metadata.get(key)):
                    _append_unique(result, source_id, current_id)
    return result


def _metadata_id_list(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.replace("|", ",").split(",") if item.strip())
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value),)


def _append_unique(values: dict[str, list[str]], key: str, item: str) -> None:
    items = values.setdefault(key, [])
    if item not in items:
        items.append(item)
