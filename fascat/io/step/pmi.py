from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from fascat.io.step.records import (
    _STEP_REFERENCE_RE,
    _iter_step_records,
    _read_step_scan_text,
    _step_number_values,
    _step_string_values,
    _StepRecord,
)
from fascat.metadata import PmiAnnotation, PmiKind, Tolerance
from fascat.options import StepReadOptions

_STEP_PMI_ENTITY_KINDS = {
    "DIMENSIONAL_SIZE": "dimension",
    "DIMENSIONAL_LOCATION": "dimension",
    "DIMENSIONAL_LOCATION_WITH_PATH": "dimension",
    "ANGULAR_SIZE": "dimension",
    "ANGULAR_LOCATION": "dimension",
    "DIMENSIONAL_CHARACTERISTIC_REPRESENTATION": "dimension",
    "SHAPE_DIMENSION_REPRESENTATION": "dimension",
    "LINEAR_DIMENSION": "dimension",
    "RADIAL_DIMENSION": "dimension",
    "DIAMETER_DIMENSION": "dimension",
    "PLUS_MINUS_TOLERANCE": "tolerance",
    "TOLERANCE_VALUE": "tolerance",
    "LIMITS_AND_FITS": "tolerance",
    "GEOMETRIC_TOLERANCE": "tolerance",
    "GEOMETRIC_TOLERANCE_RELATIONSHIP": "tolerance",
    "GEOMETRIC_TOLERANCE_WITH_DATUM_REFERENCE": "tolerance",
    "GEOMETRIC_TOLERANCE_WITH_DEFINED_AREA_UNIT": "tolerance",
    "GEOMETRIC_TOLERANCE_WITH_DEFINED_UNIT": "tolerance",
    "GEOMETRIC_TOLERANCE_WITH_MAXIMUM_TOLERANCE": "tolerance",
    "GEOMETRIC_TOLERANCE_WITH_MODIFIERS": "tolerance",
    "MODIFIED_GEOMETRIC_TOLERANCE": "tolerance",
    "UNEQUALLY_DISPOSED_GEOMETRIC_TOLERANCE": "tolerance",
    "GROUP_GEOMETRIC_TOLERANCE_WITH_DATUM_SYSTEM_REFERENCE": "tolerance",
    "ANGULARITY_TOLERANCE": "tolerance",
    "CIRCULAR_RUNOUT_TOLERANCE": "tolerance",
    "COAXIALITY_TOLERANCE": "tolerance",
    "CONCENTRICITY_TOLERANCE": "tolerance",
    "CYLINDRICITY_TOLERANCE": "tolerance",
    "FLATNESS_TOLERANCE": "tolerance",
    "LINE_PROFILE_TOLERANCE": "tolerance",
    "PARALLELISM_TOLERANCE": "tolerance",
    "PERPENDICULARITY_TOLERANCE": "tolerance",
    "POSITION_TOLERANCE": "tolerance",
    "ROUNDNESS_TOLERANCE": "tolerance",
    "STRAIGHTNESS_TOLERANCE": "tolerance",
    "SURFACE_PROFILE_TOLERANCE": "tolerance",
    "SYMMETRY_TOLERANCE": "tolerance",
    "TOTAL_RUNOUT_TOLERANCE": "tolerance",
    "DATUM": "datum",
    "DATUM_FEATURE": "datum",
    "DATUM_REFERENCE": "datum",
    "DATUM_REFERENCE_COMPARTMENT": "datum",
    "DATUM_REFERENCE_ELEMENT": "datum",
    "DATUM_SYSTEM": "datum",
    "DATUM_TARGET": "datum_target",
    "FEATURE_CONTROL_FRAME": "feature_control_frame",
    "ANNOTATION_TEXT": "note",
    "ANNOTATION_TEXT_OCCURRENCE": "note",
    "TEXT_LITERAL": "note",
    "TEXT_LITERAL_WITH_EXTENT": "note",
    "DRAUGHTING_CALLOUT": "note",
    "SAVED_VIEW": "saved_view",
    "ANNOTATION_PLANE": "annotation_plane",
}


_STEP_PMI_SEMANTIC_ENTITY_KINDS = {
    "ANNOTATION_CURVE_OCCURRENCE": "pmi_presentation",
    "ANNOTATION_FILL_AREA": "pmi_presentation_geometry",
    "ANNOTATION_FILL_AREA_OCCURRENCE": "pmi_presentation",
    "ANNOTATION_OCCURRENCE_ASSOCIATIVITY": "pmi_association",
    "ANNOTATION_PLACEHOLDER_OCCURRENCE": "pmi_presentation",
    "ANNOTATION_POINT_OCCURRENCE": "pmi_presentation",
    "ANNOTATION_SYMBOL": "pmi_presentation_geometry",
    "ANNOTATION_SYMBOL_OCCURRENCE": "pmi_presentation",
    "COLOUR_RGB": "pmi_presentation_style",
    "CURVE_STYLE": "pmi_presentation_style",
    "DRAUGHTING_CALLOUT_RELATIONSHIP": "pmi_relationship",
    "DRAUGHTING_MODEL": "pmi_presentation",
    "DRAUGHTING_MODEL_ITEM_ASSOCIATION": "pmi_presentation_association",
    "DRAUGHTING_PRE_DEFINED_CURVE_FONT": "pmi_presentation_style",
    "FILL_AREA_STYLE": "pmi_presentation_style",
    "GEOMETRIC_CURVE_SET": "pmi_presentation_geometry",
    "GEOMETRIC_ITEM_SPECIFIC_USAGE": "pmi_target_usage",
    "ID_ATTRIBUTE": "pmi_identifier",
    "MECHANICAL_DESIGN_GEOMETRIC_PRESENTATION_REPRESENTATION": "pmi_presentation",
    "PRE_DEFINED_COLOUR": "pmi_presentation_style",
    "PRESENTATION_LAYER_ASSIGNMENT": "pmi_presentation",
    "PRESENTATION_STYLE_ASSIGNMENT": "pmi_presentation",
    "PRODUCT_DEFINITION_SHAPE": "pmi_target",
    "PROPERTY_DEFINITION": "pmi_property",
    "PROPERTY_DEFINITION_REPRESENTATION": "pmi_property",
    "REPRESENTATION": "pmi_representation",
    "SHAPE_ASPECT": "pmi_target",
    "SHAPE_ASPECT_RELATIONSHIP": "pmi_target_relationship",
    "SHAPE_DEFINING_RELATIONSHIP": "pmi_target_relationship",
    "STYLED_ITEM": "pmi_presentation",
    "TESSELLATED_ANNOTATION_OCCURRENCE": "pmi_presentation",
    "TEXT_STYLE": "pmi_presentation_style",
    "DIRECTED_TOLERANCE_ZONE": "pmi_tolerance_zone",
    "NON_UNIFORM_ZONE_DEFINITION": "pmi_tolerance_zone_definition",
    "ORIENTED_TOLERANCE_ZONE": "pmi_tolerance_zone",
    "PROJECTED_ZONE_DEFINITION": "pmi_tolerance_zone_definition",
    "PROJECTED_ZONE_DEFINITION_WITH_OFFSET": "pmi_tolerance_zone_definition",
    "RUNOUT_ZONE_DEFINITION": "pmi_tolerance_zone_definition",
    "TOLERANCE_ZONE": "pmi_tolerance_zone",
    "TOLERANCE_ZONE_DEFINITION": "pmi_tolerance_zone_definition",
    "TOLERANCE_ZONE_FORM": "pmi_tolerance_zone_form",
    "TOLERANCE_ZONE_WITH_DATUM": "pmi_tolerance_zone",
}


_STEP_UNIT_RE = re.compile(r"\b(mm|millimet(?:er|re)|cm|centimet(?:er|re)|m|met(?:er|re)|in|inch|deg|degree)\b", re.I)


def _step_pmi_text(record: _StepRecord, strings: list[str], value: float | None) -> str:
    label = " / ".join(strings)
    if label:
        return label
    if value is not None:
        return f"{record.entity.lower().replace('_', ' ')} {value:g}"
    return record.entity.lower().replace("_", " ")


def _step_pmi_unit(strings: list[str]) -> str | None:
    for value in strings:
        match = _STEP_UNIT_RE.search(value)
        if match:
            token = match.group(1).lower()
            if token in {"mm", "millimeter", "millimetre"}:
                return "millimetre"
            if token in {"cm", "centimeter", "centimetre"}:
                return "centimetre"
            if token in {"m", "meter", "metre"}:
                return "metre"
            if token in {"in", "inch"}:
                return "inch"
            if token in {"deg", "degree"}:
                return "degree"
    return None


@dataclass(frozen=True)
class _StepPmiSemanticGraphNode:
    id: str
    entity: str
    kind: str
    label: str
    references: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "entity": self.entity,
            "kind": self.kind,
            "label": self.label,
            "references": list(self.references),
        }


@dataclass(frozen=True)
class _StepPmiSemanticGraphEdge:
    source: str
    target: str
    relationship: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
        }


@dataclass(frozen=True)
class _StepPmiSemanticGraphExtraction:
    nodes: tuple[_StepPmiSemanticGraphNode, ...]
    edges: tuple[_StepPmiSemanticGraphEdge, ...]
    summary: dict[str, int]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "warnings": list(self.warnings),
        }


def _extract_step_pmi_annotations(source: Path, options: StepReadOptions) -> list[PmiAnnotation]:
    if not options.pmi:
        return []
    text = _read_step_scan_text(source)
    if text is None:
        return []
    annotations: list[PmiAnnotation] = []
    for record in _iter_step_records(text):
        kind = _STEP_PMI_ENTITY_KINDS.get(record.entity)
        if kind is None:
            continue
        strings = _step_string_values(record.args)
        numbers = _step_number_values(record.args)
        value = numbers[0] if numbers else None
        text_value = _step_pmi_text(record, strings, value)
        references = tuple(f"#{item}" for item in _STEP_REFERENCE_RE.findall(record.args))
        tolerance = _step_pmi_tolerance(record.entity, kind, numbers)
        annotations.append(
            PmiAnnotation(
                id=f"step_pmi_{record.number}",
                kind=cast(PmiKind, kind),
                text=text_value,
                value=value,
                unit=_step_pmi_unit(strings),
                tolerance=tolerance,
                source={
                    "step_entity_id": f"#{record.number}",
                    "step_entity": record.entity,
                    "step_references": list(references),
                    "step_pmi_import": "textual_ap242_entity_scan",
                    "step_semantic_graph_node": f"#{record.number}",
                },
            )
        )
    return annotations


def _extract_step_pmi_semantic_graph(source: Path, options: StepReadOptions) -> _StepPmiSemanticGraphExtraction:
    if not options.pmi:
        return _StepPmiSemanticGraphExtraction(
            nodes=(),
            edges=(),
            summary=_empty_pmi_semantic_graph_summary(),
            warnings=(),
        )

    text = _read_step_scan_text(source)
    if text is None:
        return _StepPmiSemanticGraphExtraction(
            nodes=(),
            edges=(),
            summary=_empty_pmi_semantic_graph_summary(),
            warnings=(),
        )
    records = {f"#{record.number}": record for record in _iter_step_records(text)}
    pmi_ids = tuple(record_id for record_id, record in records.items() if record.entity in _STEP_PMI_ENTITY_KINDS)
    if not pmi_ids:
        return _StepPmiSemanticGraphExtraction(
            nodes=(),
            edges=(),
            summary=_empty_pmi_semantic_graph_summary(),
            warnings=(),
        )

    reverse_references: dict[str, list[str]] = {}
    for record_id, record in records.items():
        for reference in _step_record_references(record):
            reverse_references.setdefault(reference, []).append(record_id)

    included_ids: set[str] = set(pmi_ids)
    pending_ids: deque[str] = deque(pmi_ids)
    edges: list[_StepPmiSemanticGraphEdge] = []
    edge_keys: set[tuple[str, str, str]] = set()
    missing_references = 0

    def add_edge(source: str, target: str) -> None:
        edge_key = (source, target, "step_reference")
        if edge_key in edge_keys:
            return
        edge_keys.add(edge_key)
        edges.append(
            _StepPmiSemanticGraphEdge(
                source=source,
                target=target,
                relationship="step_reference",
            )
        )

    def include_record(record_id: str) -> None:
        if record_id in included_ids:
            return
        included_ids.add(record_id)
        pending_ids.append(record_id)

    while pending_ids:
        record_id = pending_ids.popleft()
        record = records[record_id]
        source_is_pmi = record_id in pmi_ids
        source_is_semantic = record.entity in _STEP_PMI_SEMANTIC_ENTITY_KINDS
        for reference in _step_record_references(record):
            target_record = records.get(reference)
            if source_is_pmi:
                add_edge(record_id, reference)
                if target_record is None:
                    missing_references += 1
                else:
                    include_record(reference)
            elif (
                source_is_semantic
                and target_record is not None
                and (
                    target_record.entity in _STEP_PMI_ENTITY_KINDS
                    or target_record.entity in _STEP_PMI_SEMANTIC_ENTITY_KINDS
                )
            ):
                add_edge(record_id, reference)
                include_record(reference)
        for source_id in reverse_references.get(record_id, ()):
            source_record = records[source_id]
            if source_record.entity not in _STEP_PMI_SEMANTIC_ENTITY_KINDS:
                continue
            add_edge(source_id, record_id)
            include_record(source_id)

    node_ids = sorted(included_ids, key=_step_entity_sort_key)
    nodes = tuple(
        _StepPmiSemanticGraphNode(
            id=record_id,
            entity=records[record_id].entity,
            kind=_step_pmi_graph_node_kind(records[record_id]),
            label=_step_record_label(records[record_id]),
            references=_step_record_references(records[record_id]),
        )
        for record_id in node_ids
    )
    cycle_count = _step_pmi_semantic_cycle_count(edges)
    summary = {
        "nodes": len(nodes),
        "pmi_nodes": len(pmi_ids),
        "referenced_nodes": len(nodes) - len(pmi_ids),
        "edges": len(edges),
        "missing_references": missing_references,
        "cycles": cycle_count,
    }
    warnings: list[str] = []
    if missing_references:
        warnings.append(f"STEP PMI semantic graph has {missing_references} reference(s) to records that were not found")
    if cycle_count:
        warnings.append(f"STEP PMI semantic graph contains {cycle_count} cycle(s)")
    return _StepPmiSemanticGraphExtraction(
        nodes=nodes,
        edges=tuple(edges),
        summary=summary,
        warnings=tuple(warnings),
    )


def _step_pmi_semantic_cycle_count(edges: list[_StepPmiSemanticGraphEdge]) -> int:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set())

    index = 0
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    cycle_count = 0

    def start_visit(start: str) -> None:
        nonlocal index, cycle_count

        indices[start] = index
        lowlinks[start] = index
        index += 1
        stack.append(start)
        on_stack.add(start)
        call_stack: list[tuple[str, Iterator[str]]] = [(start, iter(adjacency[start]))]

        while call_stack:
            node, targets = call_stack[-1]
            try:
                target = next(targets)
            except StopIteration:
                call_stack.pop()
                if lowlinks[node] == indices[node]:
                    component: list[str] = []
                    while True:
                        member = stack.pop()
                        on_stack.remove(member)
                        component.append(member)
                        if member == node:
                            break
                    if len(component) > 1 or any(member in adjacency[member] for member in component):
                        cycle_count += 1
                if call_stack:
                    parent = call_stack[-1][0]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[node])
                continue

            if target not in indices:
                indices[target] = index
                lowlinks[target] = index
                index += 1
                stack.append(target)
                on_stack.add(target)
                call_stack.append((target, iter(adjacency[target])))
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[target])

    for node in adjacency:
        if node not in indices:
            start_visit(node)
    return cycle_count


def _empty_pmi_semantic_graph_summary() -> dict[str, int]:
    return {
        "nodes": 0,
        "pmi_nodes": 0,
        "referenced_nodes": 0,
        "edges": 0,
        "missing_references": 0,
        "cycles": 0,
    }


def _step_record_references(record: _StepRecord) -> tuple[str, ...]:
    return tuple(dict.fromkeys(f"#{item}" for item in _STEP_REFERENCE_RE.findall(record.args)))


def _step_entity_sort_key(record_id: str) -> int:
    try:
        return int(record_id.removeprefix("#"))
    except ValueError:
        return 0


def _step_pmi_graph_node_kind(record: _StepRecord) -> str:
    kind = _STEP_PMI_ENTITY_KINDS.get(record.entity)
    if kind is not None:
        return f"pmi_{kind}"
    return _STEP_PMI_SEMANTIC_ENTITY_KINDS.get(record.entity, "referenced_step_entity")


def _step_record_label(record: _StepRecord) -> str:
    strings = _step_string_values(record.args)
    value = _step_number_values(record.args)
    if record.entity in _STEP_PMI_ENTITY_KINDS:
        return _step_pmi_text(record, strings, value[0] if value else None)
    if strings:
        return " / ".join(strings)
    return record.entity.lower().replace("_", " ")


def _step_pmi_tolerance(entity: str, kind: str, numbers: list[float]) -> Tolerance | None:
    if kind not in {"tolerance", "feature_control_frame"} or not numbers:
        return None
    if entity == "PLUS_MINUS_TOLERANCE" and len(numbers) >= 2:
        return Tolerance(upper=numbers[0], lower=numbers[1], kind=entity.lower())
    if len(numbers) >= 2 and numbers[0] >= 0.0 and numbers[1] <= 0.0:
        return Tolerance(upper=numbers[0], lower=numbers[1], kind=entity.lower())
    return Tolerance(upper=numbers[0], kind=entity.lower())
