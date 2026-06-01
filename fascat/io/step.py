from __future__ import annotations

import json
import re
import tempfile
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import date
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image

from fascat._ocp import shape_fingerprint as _shape_fingerprint
from fascat.asset import Asset, Node, Part
from fascat.image import ImageMimeType, ImageResource
from fascat.material import Material
from fascat.metadata import Metadata, PmiAnnotation, PmiKind, Tolerance
from fascat.options import StepReadOptions
from fascat.report import Report, timed_step

_PartIndex = dict[tuple[str, str, str, str], str]
_ArchiveTextureMap = dict[str, tuple[str, bytes]]
_UNIT_FACTORS = {
    "metre": 1.0,
    "meter": 1.0,
    "m": 1.0,
    "centimetre": 0.01,
    "centimeter": 0.01,
    "cm": 0.01,
    "millimetre": 0.001,
    "millimeter": 0.001,
    "mm": 0.001,
    "inch": 0.0254,
    "in": 0.0254,
    "foot": 0.3048,
    "feet": 0.3048,
    "ft": 0.3048,
}
_UNIT_NAMES = {
    "meter": "metre",
    "m": "metre",
    "centimeter": "centimetre",
    "cm": "centimetre",
    "millimeter": "millimetre",
    "mm": "millimetre",
    "in": "inch",
    "feet": "foot",
    "ft": "foot",
}
_SOURCE_TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".ktx2"}
_SOURCE_TEXTURE_REF_RE = re.compile(r"'([^']+\.(?:png|jpe?g|ktx2)(?:[#?][^']*)?)'", re.IGNORECASE)
_MATERIAL_RECORD_SUFFIXES = {".json", ".mtl"}
_MATERIAL_LIBRARY_CONTAINER_SUFFIXES = {".zip"}
_MATERIAL_LIBRARY_SUFFIXES = _MATERIAL_RECORD_SUFFIXES | _MATERIAL_LIBRARY_CONTAINER_SUFFIXES
_MATERIAL_LIBRARY_REF_RE = re.compile(r"'([^']+\.(?:json|mtl|zip)(?:[#?][^']*)?)'", re.IGNORECASE)
_STEP_SUFFIXES = {".step", ".stp"}
_STEP_EXTERNAL_REF_RE = re.compile(r"'([^']+\.(?:step|stp)(?:[#?][^']*)?)'", re.IGNORECASE)
_STEP_RECORD_START_RE = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)
_STEP_REFERENCE_RE = re.compile(r"#(\d+)")
_STEP_NUMBER_RE = re.compile(r"(?<![#A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")
_STEP_BOOLEAN_TOKEN_RE = re.compile(r"\.(TRUE|FALSE|T|F)\.", re.IGNORECASE)
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
    "GEOMETRIC_TOLERANCE_WITH_MODIFIERS": "tolerance",
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
_STEP_DESIGN_VARIANT_ENTITY_KINDS = {
    "APPLIED_EFFECTIVITY_ASSIGNMENT": "applied_effectivity_assignment",
    "APPLIEDEFFECTIVITYASSIGNMENT": "applied_effectivity_assignment",
    "APPLIED_EFFECTIVITY_CONTEXT_ASSIGNMENT": "applied_effectivity_context_assignment",
    "APPLIEDEFFECTIVITYCONTEXTASSIGNMENT": "applied_effectivity_context_assignment",
    "APPLIED_INEFFECTIVITY_ASSIGNMENT": "applied_ineffectivity_assignment",
    "APPLIEDINEFFECTIVITYASSIGNMENT": "applied_ineffectivity_assignment",
    "CLASS_USAGE_EFFECTIVITY_CONTEXT_ASSIGNMENT": "class_usage_effectivity_context_assignment",
    "CLASSUSAGEEFFECTIVITYCONTEXTASSIGNMENT": "class_usage_effectivity_context_assignment",
    "CONFIGURATION_DESIGN": "configuration_design",
    "CONFIGURATION_EFFECTIVITY": "configuration_effectivity",
    "CONFIGURATION_ITEM": "configuration_item",
    "COMPARISON_EQUAL": "comparison_equal",
    "COMPARISONEQUAL": "comparison_equal",
    "COMPARISON_GREATER": "comparison_greater",
    "COMPARISONGREATER": "comparison_greater",
    "COMPARISON_GREATER_EQUAL": "comparison_greater_equal",
    "COMPARISONGREATEREQUAL": "comparison_greater_equal",
    "COMPARISON_LESS": "comparison_less",
    "COMPARISONLESS": "comparison_less",
    "COMPARISON_LESS_EQUAL": "comparison_less_equal",
    "COMPARISONLESSEQUAL": "comparison_less_equal",
    "COMPARISON_NOT_EQUAL": "comparison_not_equal",
    "COMPARISONNOTEQUAL": "comparison_not_equal",
    "CONDITIONAL_CONCEPT_FEATURE": "conditional_concept_feature",
    "CONDITIONALCONCEPTFEATURE": "conditional_concept_feature",
    "CONDITIONAL_CONFIGURATION": "conditional_configuration",
    "CONDITIONALCONFIGURATION": "conditional_configuration",
    "CONDITIONAL_EFFECTIVITY": "conditional_effectivity",
    "CONFIGURED_EFFECTIVITY_ASSIGNMENT": "configured_effectivity_assignment",
    "CONFIGURED_EFFECTIVITY_CONTEXT_ASSIGNMENT": "configured_effectivity_context_assignment",
    "CONFIGUREDEFFECTIVITYCONTEXTASSIGNMENT": "configured_effectivity_context_assignment",
    "CONFIGURATIONEFFECTIVITY": "configuration_effectivity",
    "DATED_EFFECTIVITY": "dated_effectivity",
    "DATEDEFFECTIVITY": "dated_effectivity",
    "EFFECTIVITY": "effectivity",
    "EFFECTIVITYASSIGNMENT": "effectivity_assignment",
    "EFFECTIVITY_ASSIGNMENT": "effectivity_assignment",
    "EFFECTIVITYRELATIONSHIP": "effectivity_relationship",
    "EFFECTIVITY_RELATIONSHIP": "effectivity_relationship",
    "EQUALS_EXPRESSION": "equals_expression",
    "EQUALSEXPRESSION": "equals_expression",
    "INT_LITERAL": "int_literal",
    "INTLITERAL": "int_literal",
    "INT_NUMERIC_VARIABLE": "int_numeric_variable",
    "INTNUMERICVARIABLE": "int_numeric_variable",
    "INTEGER_REPRESENTATION_ITEM": "integer_representation_item",
    "INTEGERREPRESENTATIONITEM": "integer_representation_item",
    "LOT_EFFECTIVITY": "lot_effectivity",
    "LITERAL_NUMBER": "literal_number",
    "LITERALNUMBER": "literal_number",
    "MATHS_BOOLEAN_VARIABLE": "maths_boolean_variable",
    "MATHSBOOLEANVARIABLE": "maths_boolean_variable",
    "MATHS_INTEGER_VARIABLE": "maths_integer_variable",
    "MATHSINTEGERVARIABLE": "maths_integer_variable",
    "MATHS_REAL_VARIABLE": "maths_real_variable",
    "MATHSREALVARIABLE": "maths_real_variable",
    "NUMERIC_VARIABLE": "numeric_variable",
    "NUMERICVARIABLE": "numeric_variable",
    "PRODUCT_CONCEPT": "product_concept",
    "PRODUCT_CONCEPT_CONTEXT": "product_concept_context",
    "PRODUCT_CONCEPT_FEATURE": "product_concept_feature",
    "PRODUCT_CONCEPT_FEATURE_ASSOCIATION": "product_concept_feature_association",
    "PRODUCT_CONCEPT_FEATURE_CATEGORY": "product_concept_feature_category",
    "PRODUCT_CONCEPT_FEATURE_CATEGORY_USAGE": "product_concept_feature_category_usage",
    "PRODUCT_DEFINITION_EFFECTIVITY": "product_definition_effectivity",
    "PRODUCTDEFINITIONEFFECTIVITY": "product_definition_effectivity",
    "REAL_LITERAL": "real_literal",
    "REALLITERAL": "real_literal",
    "REAL_NUMERIC_VARIABLE": "real_numeric_variable",
    "REALNUMERICVARIABLE": "real_numeric_variable",
    "REAL_REPRESENTATION_ITEM": "real_representation_item",
    "REALREPRESENTATIONITEM": "real_representation_item",
    "SERIAL_NUMBERED_EFFECTIVITY": "serial_numbered_effectivity",
    "TIME_INTERVAL_BASED_EFFECTIVITY": "time_interval_based_effectivity",
    "AND_CONDITION": "and_condition",
    "AND_EXPRESSION": "and_expression",
    "ANDEXPRESSION": "and_expression",
    "ANDCONDITION": "and_condition",
    "BOOLEAN_LITERAL": "boolean_literal",
    "BOOLEAN_REPRESENTATION_ITEM": "boolean_representation_item",
    "BOOLEANREPRESENTATIONITEM": "boolean_representation_item",
    "BOOLEAN_VARIABLE": "boolean_variable",
    "NOT_CONDITION": "not_condition",
    "NOT_EXPRESSION": "not_expression",
    "NOTEXPRESSION": "not_expression",
    "NOTCONDITION": "not_condition",
    "OR_CONDITION": "or_condition",
    "OR_EXPRESSION": "or_expression",
    "OREXPRESSION": "or_expression",
    "ORCONDITION": "or_condition",
    "XOR_CONDITION": "xor_condition",
    "XOR_EXPRESSION": "xor_expression",
    "XOREXPRESSION": "xor_expression",
    "XORCONDITION": "xor_condition",
}
_STEP_UNIT_RE = re.compile(r"\b(mm|millimet(?:er|re)|cm|centimet(?:er|re)|m|met(?:er|re)|in|inch|deg|degree)\b", re.I)
_GENERIC_MATERIAL_TOKENS = {"cad", "color", "material", "mat", "texture", "map", "source"}
_TEXTURE_SLOT_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("metallic_roughness", ("metallicroughness", "metalrough", "metallic_roughness", "orm")),
    ("base_color", ("basecolor", "base_color", "albedo", "diffuse", "color", "colour")),
    ("normal", ("normal", "norm", "nrm")),
    ("occlusion", ("occlusion", "ambientocclusion", "ambient_occlusion", "ao")),
    ("roughness", ("roughness", "rough")),
    ("metallic", ("metallic", "metalness", "metal")),
    ("opacity", ("opacity", "alpha", "transparency")),
    ("emissive", ("emissive", "emission", "emit")),
)
_SOURCE_TEXTURE_EXPORT_SLOTS = {"base_color", "metallic_roughness", "normal", "occlusion", "emissive"}


@dataclass(frozen=True)
class _CadMaterialSpec:
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.5
    opacity: float = 1.0
    metadata: tuple[tuple[str, str], ...] = ()

    def metadata_dict(self) -> Metadata:
        return dict(self.metadata)


@dataclass(frozen=True)
class _MaterialLibraryRule:
    tokens: tuple[str, ...]
    metallic: float | None = None
    roughness: float | None = None
    opacity: float | None = None
    base_color: tuple[float, float, float, float] | None = None


@dataclass
class _SourceTextureExtraction:
    images: dict[str, ImageResource]
    summary: dict[str, int]
    warnings: list[str]


@dataclass(frozen=True)
class _MaterialLibrarySpec:
    name: str
    base_color: tuple[float, float, float, float] | None = None
    metallic: float | None = None
    roughness: float | None = None
    opacity: float | None = None
    texture_images: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def metadata_dict(self) -> Metadata:
        return dict(self.metadata)


@dataclass
class _MaterialLibraryExtraction:
    materials: list[_MaterialLibrarySpec]
    images: dict[str, ImageResource]
    summary: dict[str, int]
    warnings: list[str]


@dataclass(frozen=True)
class _StepMemberImport:
    index: int
    source: Path
    namespace: str
    asset: Asset


@dataclass(frozen=True)
class _StepExternalReferenceRecord:
    source: Path
    reference: str
    status: str
    resolved: Path | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "source": str(self.source),
            "reference": self.reference,
            "status": self.status,
        }
        if self.resolved is not None:
            data["resolved"] = str(self.resolved)
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class _StepExternalReferenceGraph:
    root: Path
    sources: list[Path]
    member_sources: list[Path]
    records: list[_StepExternalReferenceRecord]
    warnings: list[str]

    @property
    def has_references(self) -> bool:
        return bool(self.records)

    def summary(self) -> dict[str, int]:
        resolved_sources = {str(source.resolve()) for source in self.sources[1:]}
        return {
            "references": len(self.records),
            "resolved": sum(1 for record in self.records if record.status == "resolved"),
            "missing": sum(1 for record in self.records if record.status == "missing"),
            "unsupported": sum(1 for record in self.records if record.status == "unsupported"),
            "sources": len(self.sources),
            "resolved_sources": len(resolved_sources),
            "member_sources": len(self.member_sources),
            "resolved_occurrences": max(0, len(self.member_sources) - 1),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "sources": [str(source) for source in self.sources],
            "member_sources": [str(source) for source in self.member_sources],
            "summary": self.summary(),
            "references": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _StepDesignVariantRecord:
    id: str
    kind: str
    entity: str
    label: str
    references: tuple[str, ...]
    reference_labels: tuple[str, ...] = ()
    resolved_reference_labels: tuple[str, ...] = ()
    effectivity_kind: str | None = None
    effectivity_values: tuple[str, ...] = ()
    effectivity_range: tuple[str, ...] = ()
    condition_operator: str | None = None
    condition_value: bool | None = None
    condition_number: float | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "entity": self.entity,
            "label": self.label,
            "references": list(self.references),
            "reference_labels": list(self.reference_labels),
            "resolved_reference_labels": list(self.resolved_reference_labels),
        }
        if self.effectivity_kind is not None:
            data["effectivity_kind"] = self.effectivity_kind
        if self.effectivity_values:
            data["effectivity_values"] = list(self.effectivity_values)
        if self.effectivity_range:
            data["effectivity_range"] = list(self.effectivity_range)
        if self.condition_operator is not None:
            data["condition_operator"] = self.condition_operator
        if self.condition_value is not None:
            data["condition_value"] = self.condition_value
        if self.condition_number is not None:
            data["condition_number"] = self.condition_number
        return data


@dataclass(frozen=True)
class _StepConditionMatch:
    matched: bool
    positive: bool = False


@dataclass(frozen=True)
class _StepDesignVariantExtraction:
    records: tuple[_StepDesignVariantRecord, ...]
    summary: dict[str, int]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "records": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _StepDesignVariantSelectionResult:
    requested: tuple[str, ...]
    matched_records: tuple[str, ...]
    selector_terms: tuple[str, ...]
    status: str
    before_nodes: int
    after_nodes: int
    before_parts: int
    after_parts: int
    removed_nodes: int
    removed_parts: int
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": list(self.requested),
            "matched_records": list(self.matched_records),
            "selector_terms": list(self.selector_terms),
            "status": self.status,
            "before_nodes": self.before_nodes,
            "after_nodes": self.after_nodes,
            "before_parts": self.before_parts,
            "after_parts": self.after_parts,
            "removed_nodes": self.removed_nodes,
            "removed_parts": self.removed_parts,
            "warnings": list(self.warnings),
        }


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


@dataclass(frozen=True)
class _StepNamespaceMaps:
    nodes: dict[str, str]
    parts: dict[str, str]
    materials: dict[str, str]
    images: dict[str, str]


_MATERIAL_LIBRARY_RULES: tuple[_MaterialLibraryRule, ...] = (
    _MaterialLibraryRule(("stainless", "steel", "iron", "titanium"), metallic=1.0, roughness=0.32),
    _MaterialLibraryRule(("aluminum", "aluminium", "6061", "7075"), metallic=1.0, roughness=0.28),
    _MaterialLibraryRule(("chrome",), metallic=1.0, roughness=0.12),
    _MaterialLibraryRule(("brass",), metallic=1.0, roughness=0.24, base_color=(0.86, 0.66, 0.34, 1.0)),
    _MaterialLibraryRule(("copper",), metallic=1.0, roughness=0.22, base_color=(0.95, 0.55, 0.33, 1.0)),
    _MaterialLibraryRule(("bronze",), metallic=1.0, roughness=0.3, base_color=(0.72, 0.45, 0.24, 1.0)),
    _MaterialLibraryRule(("gold",), metallic=1.0, roughness=0.18, base_color=(1.0, 0.76, 0.34, 1.0)),
    _MaterialLibraryRule(("glass",), metallic=0.0, roughness=0.03, opacity=0.35),
    _MaterialLibraryRule(("acrylic", "polycarbonate", "transparent"), metallic=0.0, roughness=0.08, opacity=0.55),
    _MaterialLibraryRule(("rubber", "silicone"), metallic=0.0, roughness=0.78),
    _MaterialLibraryRule(("plastic", "abs", "pla", "nylon", "polymer"), metallic=0.0, roughness=0.58),
    _MaterialLibraryRule(("ceramic", "porcelain"), metallic=0.0, roughness=0.36),
    _MaterialLibraryRule(("paint", "coating", "powdercoat", "powder"), metallic=0.0, roughness=0.48),
    _MaterialLibraryRule(("wood", "timber"), metallic=0.0, roughness=0.64),
)


@dataclass(frozen=True)
class _StepHeaderInfo:
    schema: str = ""
    pmi_present: bool = False


@dataclass(frozen=True)
class _StepRecord:
    number: int
    entity: str
    args: str


@dataclass(frozen=True)
class _ShapeTopologyCounts:
    vertices: int = 0
    edges: int = 0
    faces: int = 0


@dataclass
class _ImportCleanupStats:
    brep_parts: int = 0
    construction_point_parts: int = 0
    construction_line_parts: int = 0
    empty_shape_parts: int = 0
    deleted_free_vertex_parts: int = 0
    deleted_free_vertices: int = 0
    deleted_line_parts: int = 0
    deleted_line_edges: int = 0
    deleted_line_vertices: int = 0

    def record_loaded(self, representation: str) -> None:
        if representation == "brep":
            self.brep_parts += 1
        elif representation == "construction_points":
            self.construction_point_parts += 1
        elif representation == "construction_lines":
            self.construction_line_parts += 1
        elif representation == "empty_shape":
            self.empty_shape_parts += 1

    def record_deleted(self, action: str, counts: _ShapeTopologyCounts) -> None:
        if action == "delete_free_vertices":
            self.deleted_free_vertex_parts += 1
            self.deleted_free_vertices += counts.vertices
        elif action == "delete_lines":
            self.deleted_line_parts += 1
            self.deleted_line_edges += counts.edges
            self.deleted_line_vertices += counts.vertices

    def to_dict(self) -> dict[str, int]:
        return {
            "brep_parts": self.brep_parts,
            "construction_point_parts": self.construction_point_parts,
            "construction_line_parts": self.construction_line_parts,
            "empty_shape_parts": self.empty_shape_parts,
            "deleted_free_vertex_parts": self.deleted_free_vertex_parts,
            "deleted_free_vertices": self.deleted_free_vertices,
            "deleted_line_parts": self.deleted_line_parts,
            "deleted_line_edges": self.deleted_line_edges,
            "deleted_line_vertices": self.deleted_line_vertices,
        }


@dataclass(frozen=True)
class _SpaceNormalization:
    source_units: str
    source_meters_per_unit: float
    source_up_axis: str
    source_handedness: str
    target_units: str
    target_meters_per_unit: float
    target_up_axis: str
    target_handedness: str
    transform: np.ndarray

    @property
    def changed(self) -> bool:
        return not np.allclose(self.transform, np.eye(4, dtype=np.float64))

    def metadata(self) -> dict[str, object]:
        return {
            "source_units": self.source_units,
            "source_meters_per_unit": self.source_meters_per_unit,
            "source_up_axis": self.source_up_axis,
            "source_handedness": self.source_handedness,
            "target_units": self.target_units,
            "target_meters_per_unit": self.target_meters_per_unit,
            "target_up_axis": self.target_up_axis,
            "target_handedness": self.target_handedness,
            "transform": self.transform.tolist(),
            "changed": self.changed,
        }


def read_step(path: str | Path, *, options: StepReadOptions | None = None) -> Asset:
    source = Path(path)
    opts = options or StepReadOptions()
    if opts.multi_file:
        return _read_step_with_external_references(source, opts)
    return _read_step_path(source, source_identity=str(source.resolve()), options=opts)


def read_step_many(
    paths: Iterable[str | Path],
    *,
    options: StepReadOptions | None = None,
    continue_on_error: bool = False,
) -> Asset:
    return _read_step_many(paths, options=options, continue_on_error=continue_on_error, reference_graph=None)


def _read_step_many(
    paths: Iterable[str | Path],
    *,
    options: StepReadOptions | None = None,
    continue_on_error: bool = False,
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> Asset:
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("read_step_many requires at least one STEP file")

    opts = options or StepReadOptions()
    member_options = replace(opts, multi_file=False)
    members: list[_StepMemberImport] = []
    failed_members: list[dict[str, object]] = []
    warnings: list[str] = []
    with timed_step() as timer:
        for index, source in enumerate(sources, start=1):
            namespace = _multi_file_namespace(index, source)
            try:
                member_asset = _read_step_path(
                    source,
                    source_identity=str(source.resolve()),
                    options=member_options,
                )
            except Exception as exc:
                warning = f"multi-file STEP member {index} ({source}) failed to import: {exc}"
                failed_members.append(
                    {
                        "index": index,
                        "source": str(source),
                        "namespace": namespace,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    raise
                warnings.append(warning)
                continue
            member_warnings = [
                f"multi-file STEP member {index} ({source}): {warning}" for warning in member_asset.report.warnings
            ]
            warnings.extend(member_warnings)
            members.append(_StepMemberImport(index=index, source=source, namespace=namespace, asset=member_asset))

        if not members:
            raise RuntimeError("multi-file STEP import did not import any members")

    return _merge_step_member_assets(
        members,
        failed_members=failed_members,
        options=opts,
        sources=sources,
        duration=timer.duration,
        warnings=warnings,
        reference_graph=reference_graph,
    )


def _read_step_with_external_references(source: Path, options: StepReadOptions) -> Asset:
    graph = _resolve_step_external_reference_graph(source)
    member_options = replace(options, multi_file=False)
    if len(graph.member_sources) > 1:
        return _read_step_many(graph.member_sources, options=options, reference_graph=graph)

    asset = _read_step_path(source, source_identity=str(source.resolve()), options=member_options)
    _attach_step_external_reference_graph(asset, graph, options)
    return asset


def _resolve_step_external_reference_graph(source: Path) -> _StepExternalReferenceGraph:
    root = source
    root_key = str(root.resolve())
    sources = [root]
    member_sources = [root]
    seen_sources = {root_key}
    records: list[_StepExternalReferenceRecord] = []
    warnings: list[str] = []
    queue = [root]

    while queue:
        current = queue.pop(0)
        for reference in _step_external_references(current):
            cleaned, unsupported_reason = _clean_step_external_reference(reference)
            if cleaned is None:
                records.append(
                    _StepExternalReferenceRecord(
                        source=current,
                        reference=reference,
                        status="unsupported",
                        reason=unsupported_reason or "unsupported external STEP reference",
                    )
                )
                warnings.append(
                    f"STEP external reference is unsupported: {reference} "
                    f"(referenced by {current}; {unsupported_reason})"
                )
                continue

            resolved = _resolve_step_external_reference(cleaned, current)
            if resolved is None:
                records.append(
                    _StepExternalReferenceRecord(
                        source=current,
                        reference=reference,
                        status="missing",
                        reason="not found relative to the referencing STEP file",
                    )
                )
                warnings.append(f"STEP external reference could not be resolved: {reference} (referenced by {current})")
                continue

            records.append(
                _StepExternalReferenceRecord(
                    source=current,
                    reference=reference,
                    status="resolved",
                    resolved=resolved,
                )
            )
            resolved_key = str(resolved.resolve())
            if resolved_key != root_key:
                member_sources.append(resolved)
            if resolved_key in seen_sources:
                continue
            seen_sources.add(resolved_key)
            sources.append(resolved)
            queue.append(resolved)

    return _StepExternalReferenceGraph(
        root=root,
        sources=sources,
        member_sources=member_sources,
        records=records,
        warnings=warnings,
    )


def _step_external_references(source: Path) -> list[str]:
    text = source.read_text(encoding="utf-8", errors="ignore")
    references: list[str] = []
    for match in _STEP_EXTERNAL_REF_RE.finditer(text):
        reference = match.group(1).replace("''", "'").strip()
        if not reference:
            continue
        references.append(reference)
    return references


def _clean_step_external_reference(reference: str) -> tuple[str | None, str | None]:
    value = reference.replace("''", "'").strip().strip('"<>')
    parsed = urlparse(value)
    if parsed.scheme and not _looks_like_windows_path(value):
        if parsed.scheme.lower() != "file":
            return None, f"unsupported URI scheme: {parsed.scheme}"
        value = unquote(parsed.path)
    else:
        value = unquote(value.split("#", 1)[0].split("?", 1)[0])
    value = value.strip()
    if not value:
        return None, "empty path"
    if Path(value).suffix.lower() not in _STEP_SUFFIXES:
        return None, "unsupported file extension"
    return value, None


def _looks_like_windows_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[0].isalpha() and value[2] in {"\\", "/"}


def _resolve_step_external_reference(reference: str, source: Path) -> Path | None:
    candidate = Path(reference)
    candidates = [candidate] if candidate.is_absolute() else [source.parent / candidate]
    if not candidate.is_absolute() and candidate.name != str(candidate):
        candidates.append(source.parent / candidate.name)
    for item in candidates:
        if item.exists() and item.is_file() and item.suffix.lower() in _STEP_SUFFIXES:
            return item.resolve()
    return None


def _read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing STEP file: {source}")
    if source.suffix.lower() not in _STEP_SUFFIXES:
        raise ValueError(f"unsupported STEP extension: {source.suffix or '<none>'}")

    header_info = _step_header_info(source)
    cleanup = _ImportCleanupStats()
    with timed_step() as timer:
        document, shape_tool, color_tool, vis_material_tool, unit_name, meters_per_unit = _read_xde_document(
            source, options
        )
        space = _space_normalization(unit_name, meters_per_unit, options)
        free_labels = _free_shape_labels(shape_tool)
        root = Node(
            id=_stable_id("node", f"{source_identity}:root"),
            name=source.stem,
            transform=space.transform,
            metadata={
                "source": str(source),
                "source_identity": source_identity,
                "space_normalization": space.metadata(),
            },
        )
        parts: dict[str, Part] = {}
        part_index: _PartIndex = {}
        materials: dict[str, Material] = {}
        for index, label in enumerate(free_labels, start=1):
            root.children.append(
                _build_node(
                    label,
                    f"root/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    vis_material_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )
        design_variants = _extract_step_design_variants(source, options)
        design_variant_selection = _apply_step_design_variant_selection(
            root, parts, materials, design_variants, options
        )
        source_textures = _extract_source_textures(source, source_identity, options)
        texture_binding_summary = _attach_source_textures_to_materials(materials, source_textures.images)
        material_libraries = _extract_material_libraries(source, source_identity, options)
        material_library_binding_summary = _apply_material_libraries_to_materials(materials, material_libraries)
        images = {**source_textures.images, **material_libraries.images}
        pmi = _extract_step_pmi_annotations(source, options)
        pmi_semantic_graph = _extract_step_pmi_semantic_graph(source, options)

    report = Report(source_path=str(source))
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        images=images,
        units=space.target_units,
        meters_per_unit=space.target_meters_per_unit,
        up_axis=cast(Any, space.target_up_axis),
        source_path=source,
        metadata=_asset_metadata(
            source,
            source_identity,
            options,
            header_info,
            cleanup,
            space,
            pmi_count=len(pmi),
            design_variant_summary=design_variants.summary,
        ),
        pmi=pmi,
        report=report,
    )
    asset.report.input_stats = asset.stats()
    metadata_count = _metadata_count(asset)
    unsupported_pmi_count = _unsupported_pmi_count(options, header_info, pmi_count=len(asset.pmi))
    import_decisions = _import_decisions(
        options,
        header_info,
        pmi_count=len(asset.pmi),
        unsupported_pmi_count=unsupported_pmi_count,
        cleanup=cleanup,
        space=space,
        source_texture_summary=source_textures.summary,
        texture_binding_summary=texture_binding_summary,
        material_library_summary=material_libraries.summary,
        material_library_binding_summary=material_library_binding_summary,
        pmi_semantic_graph_summary=pmi_semantic_graph.summary,
        design_variant_summary=design_variants.summary,
        design_variant_selection_summary=design_variant_selection.to_dict(),
    )
    loaded_representations = _loaded_representation_report(asset)
    if asset.metadata:
        asset.metadata["import_decisions"] = import_decisions
        asset.metadata["import_representation_summary"] = loaded_representations["summary"]
        asset.metadata["source_texture_import"] = source_textures.summary
        asset.metadata["source_texture_bindings"] = texture_binding_summary
        asset.metadata["material_library_import"] = material_libraries.summary
        asset.metadata["material_library_bindings"] = material_library_binding_summary
        asset.metadata["pmi_semantic_graph"] = pmi_semantic_graph.to_dict()
        asset.metadata["design_variant_import"] = design_variants.summary
        asset.metadata["design_variant_selection"] = design_variant_selection.to_dict()
        if design_variants.records:
            asset.metadata["design_variants"] = [record.to_dict() for record in design_variants.records]
    import_warnings = [
        *_import_warnings(
            options,
            header_info,
            unsupported_pmi_count,
            design_variant_count=design_variants.summary["records"],
        ),
        *source_textures.warnings,
        *material_libraries.warnings,
        *pmi_semantic_graph.warnings,
        *design_variants.warnings,
        *design_variant_selection.warnings,
    ]
    for warning in import_warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "STEP",
            "backend": "OCP",
            "read_options": options.to_dict(),
            "metadata_count": metadata_count,
            "pmi_count": len(asset.pmi),
            "unsupported_pmi_count": unsupported_pmi_count,
            "pmi_semantic_graph": pmi_semantic_graph.to_dict(),
            "design_variants": design_variants.to_dict(),
            "design_variant_selection": design_variant_selection.to_dict(),
            "pmi_schema": header_info.schema,
            "pmi_present": header_info.pmi_present,
            "cleanup": cleanup.to_dict(),
            "space_normalization": space.metadata(),
            "source_textures": source_textures.summary,
            "source_texture_bindings": texture_binding_summary,
            "material_libraries": material_libraries.summary,
            "material_library_bindings": material_library_binding_summary,
            "import_decisions": import_decisions,
            "loaded_representations": loaded_representations,
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
        warnings=import_warnings,
    )
    _ = document
    return asset


def read_step_bytes(data: bytes, *, name: str = "stdin.step", options: StepReadOptions | None = None) -> Asset:
    with tempfile.NamedTemporaryFile(suffix=Path(name).suffix or ".step") as handle:
        handle.write(data)
        handle.flush()
        asset = _read_step_path(Path(handle.name), source_identity=name, options=options or StepReadOptions())
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    if asset.metadata:
        asset.metadata["source"] = name
        asset.metadata["source_identity"] = name
    return asset


def _merge_step_member_assets(
    members: list[_StepMemberImport],
    *,
    failed_members: list[dict[str, object]],
    options: StepReadOptions,
    sources: list[Path],
    duration: float,
    warnings: list[str],
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> Asset:
    target_units = members[0].asset.units
    target_meters_per_unit = members[0].asset.meters_per_unit
    target_up_axis = members[0].asset.up_axis
    root_children: list[Node] = []
    parts: dict[str, Part] = {}
    materials: dict[str, Material] = {}
    images: dict[str, ImageResource] = {}
    pmi: list[PmiAnnotation] = []
    member_records: list[dict[str, object]] = []

    for member in members:
        maps = _step_namespace_maps(member.asset, member.namespace)
        root = _namespace_step_node(member.asset.root, maps)
        root.metadata.update(
            {
                "multi_file_member_index": member.index,
                "multi_file_member_source": str(member.source),
                "multi_file_member_namespace": member.namespace,
            }
        )
        unit_warning = _normalize_member_root_units(
            root,
            member.asset,
            target_units=target_units,
            target_meters_per_unit=target_meters_per_unit,
            target_up_axis=target_up_axis,
            source=member.source,
            index=member.index,
        )
        if unit_warning is not None:
            warnings.append(unit_warning)
        root_children.append(root)

        for image_id, image in member.asset.images.items():
            images[maps.images[image_id]] = _namespace_step_image(image, maps, member)
        for material_id, material in member.asset.materials.items():
            materials[maps.materials[material_id]] = _namespace_step_material(material, maps, member)
        for part_id, part in member.asset.parts.items():
            parts[maps.parts[part_id]] = _namespace_step_part(part, maps, member)
        pmi.extend(_namespace_step_pmi(annotation, maps, member) for annotation in member.asset.pmi)

        member_records.append(
            {
                "index": member.index,
                "source": str(member.source),
                "namespace": member.namespace,
                "status": "imported",
                "root_node_id": root.id,
                "nodes": len(member.asset.root.walk()),
                "parts": len(member.asset.parts),
                "materials": len(member.asset.materials),
                "images": len(member.asset.images),
                "warnings": len(member.asset.report.warnings),
                "units": member.asset.units,
                "meters_per_unit": member.asset.meters_per_unit,
                "up_axis": member.asset.up_axis,
            }
        )

    member_records = sorted([*member_records, *failed_members], key=lambda item: cast(int, item["index"]))
    if reference_graph is not None:
        warnings.extend(warning for warning in reference_graph.warnings if warning not in warnings)
    source_identity = _multi_file_source_identity(sources)
    import_decisions = _multi_file_import_decisions(
        options,
        len(members),
        len(failed_members),
        reference_graph=reference_graph,
    )
    root_metadata: Metadata = {
        "source": "multi-file STEP import",
        "source_identity": source_identity,
        "multi_file": "true",
        "multi_file_member_count": str(len(members)),
    }
    if reference_graph is not None:
        root_metadata.update(
            {
                "source": str(reference_graph.root),
                "external_reference_graph": "true",
                "external_reference_root": str(reference_graph.root),
            }
        )
    root = Node(
        id=_stable_id("node", f"{source_identity}:root"),
        name=(
            f"{reference_graph.root.stem} external-reference STEP assembly"
            if reference_graph is not None
            else "multi-file STEP assembly"
        ),
        children=root_children,
        metadata=root_metadata,
    )
    report = Report(source_path=None)
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        images=images,
        units=target_units,
        meters_per_unit=target_meters_per_unit,
        up_axis=target_up_axis,
        source_path=None,
        metadata=_multi_file_asset_metadata(
            options,
            source_identity=source_identity,
            members=member_records,
            target_units=target_units,
            target_meters_per_unit=target_meters_per_unit,
            target_up_axis=target_up_axis,
            reference_graph=reference_graph,
        ),
        pmi=pmi,
        report=report,
    )
    asset.report.input_stats = asset.stats()
    for warning in warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "STEP",
            "backend": "OCP",
            "read_options": {**options.to_dict(), "multi_file": True},
            "multi_file": True,
            "member_count": len(members),
            "failed_member_count": len(failed_members),
            "members": member_records,
            "import_decisions": import_decisions,
            **({"external_reference_graph": reference_graph.to_dict()} if reference_graph is not None else {}),
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=duration,
        warnings=warnings,
    )
    if asset.metadata:
        asset.metadata["import_decisions"] = import_decisions
    return asset


def _step_namespace_maps(asset: Asset, namespace: str) -> _StepNamespaceMaps:
    return _StepNamespaceMaps(
        nodes={node.id: f"{namespace}__{node.id}" for node in asset.root.walk()},
        parts={part_id: f"{namespace}__{part_id}" for part_id in asset.parts},
        materials={material_id: f"{namespace}__{material_id}" for material_id in asset.materials},
        images={image_id: f"{namespace}__{image_id}" for image_id in asset.images},
    )


def _namespace_step_node(node: Node, maps: _StepNamespaceMaps) -> Node:
    metadata = _namespace_metadata_ids(node.metadata, part_ids=maps.parts, node_ids=maps.nodes)
    return Node(
        id=maps.nodes[node.id],
        name=node.name,
        children=[_namespace_step_node(child, maps) for child in node.children],
        part_id=None if node.part_id is None else maps.parts.get(node.part_id, node.part_id),
        transform=node.transform,
        metadata=metadata,
    )


def _namespace_step_part(part: Part, maps: _StepNamespaceMaps, member: _StepMemberImport) -> Part:
    metadata = _namespace_metadata_ids(part.metadata, part_ids=maps.parts, node_ids=maps.nodes)
    metadata.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_part_id": part.id,
        }
    )
    return Part(
        id=maps.parts[part.id],
        name=part.name,
        source_shape=part.source_shape,
        mesh=None if part.mesh is None else part.mesh.copy(),
        material_ids=[maps.materials.get(material_id, material_id) for material_id in part.material_ids],
        metadata=metadata,
        fingerprint=part.fingerprint,
        lod_meshes=[mesh.copy() for mesh in part.lod_meshes],
    )


def _namespace_step_material(material: Material, maps: _StepNamespaceMaps, member: _StepMemberImport) -> Material:
    metadata = _namespace_metadata_ids(material.metadata, image_ids=maps.images)
    metadata.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_material_id": material.id,
        }
    )
    return Material(
        id=maps.materials[material.id],
        name=material.name,
        base_color=material.base_color,
        metallic=material.metallic,
        roughness=material.roughness,
        opacity=material.opacity,
        metadata=metadata,
    )


def _namespace_step_image(
    image: ImageResource,
    maps: _StepNamespaceMaps,
    member: _StepMemberImport,
) -> ImageResource:
    metadata = dict(image.metadata)
    metadata.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_image_id": image.id,
        }
    )
    return ImageResource(
        id=maps.images[image.id],
        name=image.name,
        mime_type=image.mime_type,
        data=image.data,
        width=image.width,
        height=image.height,
        metadata=metadata,
    )


def _namespace_step_pmi(
    annotation: PmiAnnotation,
    maps: _StepNamespaceMaps,
    member: _StepMemberImport,
) -> PmiAnnotation:
    source = dict(annotation.source)
    source.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_pmi_id": annotation.id,
        }
    )
    return PmiAnnotation(
        id=f"{member.namespace}__{annotation.id}",
        kind=annotation.kind,
        text=annotation.text,
        value=annotation.value,
        unit=annotation.unit,
        tolerance=annotation.tolerance,
        applies_to=[maps.parts.get(part_id, part_id) for part_id in annotation.applies_to],
        view=annotation.view,
        plane=None if annotation.plane is None else [list(row) for row in annotation.plane],
        source=source,
    )


def _namespace_metadata_ids(
    metadata: Metadata,
    *,
    part_ids: dict[str, str] | None = None,
    node_ids: dict[str, str] | None = None,
    image_ids: dict[str, str] | None = None,
) -> Metadata:
    result = dict(metadata)
    part_ids = part_ids or {}
    node_ids = node_ids or {}
    image_ids = image_ids or {}
    for key, value in list(result.items()):
        if key.endswith("_image") and isinstance(value, str) and value in image_ids:
            result[key] = image_ids[value]
        elif key in {"source_part_id", "source_part_ids", "split_source_part_id", "split_source_part_ids"}:
            result[key] = _namespace_metadata_value(value, part_ids)
        elif key in {"source_node_id", "source_node_ids", "split_source_node_id", "split_source_node_ids"}:
            result[key] = _namespace_metadata_value(value, node_ids)
    return result


def _namespace_metadata_value(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, str):
        separator = "|" if "|" in value and "," not in value else ","
        items = [item.strip() for item in value.replace("|", ",").split(",") if item.strip()]
        if not items:
            return value
        mapped = [mapping.get(item, item) for item in items]
        return separator.join(mapped)
    if isinstance(value, list):
        return [mapping.get(str(item), str(item)) for item in value]
    if isinstance(value, tuple):
        return tuple(mapping.get(str(item), str(item)) for item in value)
    return mapping.get(str(value), value)


def _normalize_member_root_units(
    root: Node,
    asset: Asset,
    *,
    target_units: str,
    target_meters_per_unit: float,
    target_up_axis: str,
    source: Path,
    index: int,
) -> str | None:
    scale = asset.meters_per_unit / target_meters_per_unit
    if not np.isclose(scale, 1.0):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] *= scale
        root.transform = transform @ root.transform
        root.metadata["multi_file_unit_conversion"] = {
            "source_units": asset.units,
            "source_meters_per_unit": asset.meters_per_unit,
            "target_units": target_units,
            "target_meters_per_unit": target_meters_per_unit,
            "scale": scale,
        }
    if asset.up_axis != target_up_axis:
        return (
            f"multi-file STEP member {index} ({source}) uses up axis {asset.up_axis}; "
            f"combined asset uses {target_up_axis}; set target_up_axis to normalize all members explicitly"
        )
    return None


def _multi_file_asset_metadata(
    options: StepReadOptions,
    *,
    source_identity: str,
    members: list[dict[str, object]],
    target_units: str,
    target_meters_per_unit: float,
    target_up_axis: str,
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> Metadata:
    if not options.metadata:
        return {}
    metadata: Metadata = {
        "source": "multi-file STEP import",
        "source_identity": source_identity,
        "units": target_units,
        "meters_per_unit": target_meters_per_unit,
        "up_axis": target_up_axis,
        "metadata_options": {**options.to_dict(), "multi_file": True},
        "multi_file_import": {
            "member_count": sum(1 for member in members if member.get("status") == "imported"),
            "failed_member_count": sum(1 for member in members if member.get("status") == "error"),
            "members": members,
        },
    }
    if reference_graph is not None:
        metadata["source"] = str(reference_graph.root)
        metadata["external_reference_graph"] = reference_graph.to_dict()
    return metadata


def _multi_file_import_decisions(
    options: StepReadOptions,
    member_count: int,
    failed_member_count: int,
    *,
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> dict[str, object]:
    if reference_graph is not None:
        return {
            "multi_file": _external_reference_import_decision(
                options,
                reference_graph,
                member_count=member_count,
                failed_member_count=failed_member_count,
            )
        }
    return {
        "multi_file": _import_decision(
            requested=True,
            effective=member_count > 0,
            state="approximated" if failed_member_count else "honored",
            detail=(
                "explicit STEP member paths were imported as separate deterministic namespaces; "
                "single-file external STEP reference resolution remains unsupported"
            ),
            counts={"members": member_count, "failed_members": failed_member_count},
        )
    }


def _external_reference_import_decision(
    options: StepReadOptions,
    graph: _StepExternalReferenceGraph,
    *,
    member_count: int,
    failed_member_count: int,
) -> dict[str, object]:
    summary = graph.summary()
    missing_or_unsupported = summary["missing"] + summary["unsupported"]
    if failed_member_count:
        state = "approximated"
    elif missing_or_unsupported:
        state = "missing_sources"
    else:
        state = "honored"
    detail = (
        "external STEP references were resolved from quoted STEP path records and imported as deterministic "
        "member occurrences"
        if summary["references"]
        else "no external STEP references were found in the master STEP file"
    )
    return _import_decision(
        requested=options.multi_file,
        effective=summary["resolved_occurrences"] > 0,
        state=state,
        detail=detail,
        counts={
            "members": member_count,
            "failed_members": failed_member_count,
            **summary,
        },
    )


def _attach_step_external_reference_graph(
    asset: Asset,
    graph: _StepExternalReferenceGraph,
    options: StepReadOptions,
) -> None:
    decision = _external_reference_import_decision(
        options,
        graph,
        member_count=1,
        failed_member_count=0,
    )
    graph_data = graph.to_dict()
    if asset.metadata:
        asset.metadata["external_reference_graph"] = graph_data
        metadata_options = asset.metadata.get("metadata_options")
        if isinstance(metadata_options, dict):
            metadata_options["multi_file"] = True
        import_decisions = asset.metadata.get("import_decisions")
        if isinstance(import_decisions, dict):
            import_decisions["multi_file"] = decision
    for step in asset.report.steps:
        if step.name != "import":
            continue
        read_options = step.options.get("read_options")
        if isinstance(read_options, dict):
            read_options["multi_file"] = True
        import_decisions = step.options.get("import_decisions")
        if isinstance(import_decisions, dict):
            import_decisions["multi_file"] = decision
        else:
            step.options["import_decisions"] = {"multi_file": decision}
        step.options["external_reference_graph"] = graph_data
        for warning in graph.warnings:
            if warning not in step.warnings:
                step.warnings.append(warning)
        break
    for warning in graph.warnings:
        if warning not in asset.report.warnings:
            asset.report.add_warning(warning)


def _multi_file_namespace(index: int, source: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", source.stem.lower()).strip("_") or "step"
    digest = _stable_id("ns", str(source.resolve())).split("_", 1)[1][:8]
    return f"member{index}_{stem}_{digest}"


def _multi_file_source_identity(sources: list[Path]) -> str:
    encoded = "|".join(str(source.resolve()) for source in sources)
    return _stable_id("step_multi", encoded)


def _read_xde_document(path: Path, options: StepReadOptions) -> tuple[Any, Any, Any, Any, str, float]:
    try:
        from OCP.IFSelect import IFSelect_RetDone
        from OCP.STEPCAFControl import STEPCAFControl_Reader
        from OCP.TCollection import TCollection_ExtendedString
        from OCP.TDocStd import TDocStd_Document
        from OCP.XCAFApp import XCAFApp_Application
        from OCP.XCAFDoc import XCAFDoc_DocumentTool
    except ImportError as exc:
        raise RuntimeError("STEP import requires cadquery-ocp") from exc

    app = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("fascat"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), document)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(options.metadata)
    reader.SetColorMode(True)
    reader.SetMatMode(True)
    reader.SetMetaMode(options.metadata or options.properties)
    reader.SetProductMetaMode(options.product_metadata)
    status = reader.ReadFile(str(path))
    if status != IFSelect_RetDone:
        raise RuntimeError(f"failed to read STEP file: {path}")

    unit_name, meters_per_unit = _reader_units(reader)
    if not reader.Transfer(document):
        raise RuntimeError(f"failed to transfer STEP data into XDE document: {path}")
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(document.Main())
    vis_material_tool = XCAFDoc_DocumentTool.VisMaterialTool_s(document.Main())
    return document, shape_tool, color_tool, vis_material_tool, unit_name, meters_per_unit


def _free_shape_labels(shape_tool: Any) -> list[Any]:
    from OCP.TDF import TDF_LabelSequence

    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)
    return [labels.Value(index) for index in range(labels.Lower(), labels.Upper() + 1)]


def _shape_topology_counts(shape: Any) -> _ShapeTopologyCounts:
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    return _ShapeTopologyCounts(
        vertices=_count_subshapes(shape, TopAbs_VERTEX, TopExp_Explorer),
        edges=_count_subshapes(shape, TopAbs_EDGE, TopExp_Explorer),
        faces=_count_subshapes(shape, TopAbs_FACE, TopExp_Explorer),
    )


def _count_subshapes(shape: Any, shape_type: Any, explorer_factory: Any) -> int:
    explorer = explorer_factory(shape, shape_type)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _space_normalization(unit_name: str, meters_per_unit: float, options: StepReadOptions) -> _SpaceNormalization:
    source_units, source_meters_per_unit = _space_units(
        unit_name,
        meters_per_unit,
        override_units=options.source_units,
        override_meters_per_unit=options.source_meters_per_unit,
    )
    target_units, target_meters_per_unit = _space_units(
        source_units,
        source_meters_per_unit,
        override_units=options.target_units,
        override_meters_per_unit=options.target_meters_per_unit,
    )
    target_up_axis = options.target_up_axis or options.source_up_axis
    target_handedness = options.target_handedness or options.source_handedness
    transform = _space_transform(
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=options.source_up_axis,
        source_handedness=options.source_handedness,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=target_up_axis,
        target_handedness=target_handedness,
    )
    return _SpaceNormalization(
        source_units=source_units,
        source_meters_per_unit=source_meters_per_unit,
        source_up_axis=options.source_up_axis,
        source_handedness=options.source_handedness,
        target_units=target_units,
        target_meters_per_unit=target_meters_per_unit,
        target_up_axis=target_up_axis,
        target_handedness=target_handedness,
        transform=transform,
    )


def _space_units(
    default_units: str,
    default_meters_per_unit: float,
    *,
    override_units: str | None,
    override_meters_per_unit: float | None,
) -> tuple[str, float]:
    unit_name = _canonical_unit_name(default_units)
    meters_per_unit = float(default_meters_per_unit)
    if override_units is not None:
        unit_name = _canonical_unit_name(override_units)
        meters_per_unit = _unit_factor(unit_name)
    if override_meters_per_unit is not None:
        meters_per_unit = float(override_meters_per_unit)
        if override_units is None:
            unit_name = "custom"
    return unit_name, meters_per_unit


def _canonical_unit_name(value: str) -> str:
    key = value.strip().lower()
    return _UNIT_NAMES.get(key, key or "unit")


def _unit_factor(unit_name: str) -> float:
    factor = _UNIT_FACTORS.get(unit_name)
    if factor is None:
        known = ", ".join(sorted({"metre", "centimetre", "millimetre", "inch", "foot"}))
        raise ValueError(f"unsupported unit name for space normalization: {unit_name}; known units: {known}")
    return factor


def _space_transform(
    *,
    source_meters_per_unit: float,
    source_up_axis: str,
    source_handedness: str,
    target_meters_per_unit: float,
    target_up_axis: str,
    target_handedness: str,
) -> np.ndarray:
    linear = (
        np.linalg.inv(_to_canonical_space(target_up_axis, target_handedness))
        @ _to_canonical_space(source_up_axis, source_handedness)
        * (source_meters_per_unit / target_meters_per_unit)
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = linear
    return transform


def _to_canonical_space(up_axis: str, handedness: str) -> np.ndarray:
    if up_axis == "Z":
        axis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
    else:
        axis = np.eye(3, dtype=np.float64)
    if handedness == "left":
        return np.diag([-1.0, 1.0, 1.0]) @ axis
    return axis


def _asset_metadata(
    source: Path,
    source_identity: str,
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    cleanup: _ImportCleanupStats,
    space: _SpaceNormalization,
    pmi_count: int = 0,
    design_variant_summary: dict[str, int] | None = None,
    source_texture_summary: dict[str, int] | None = None,
    texture_binding_summary: dict[str, int] | None = None,
    material_library_summary: dict[str, int] | None = None,
    material_library_binding_summary: dict[str, int] | None = None,
) -> Metadata:
    if not options.metadata:
        return {}
    metadata: Metadata = {
        "source": str(source),
        "source_identity": source_identity,
        "units": space.target_units,
        "meters_per_unit": space.target_meters_per_unit,
        "source_units": space.source_units,
        "source_meters_per_unit": space.source_meters_per_unit,
        "up_axis": space.target_up_axis,
        "source_up_axis": space.source_up_axis,
        "handedness": space.target_handedness,
        "source_handedness": space.source_handedness,
        "space_normalization": space.metadata(),
        "metadata_options": options.to_dict(),
        "import_cleanup": cleanup.to_dict(),
    }
    if source_texture_summary is not None:
        metadata["source_texture_import"] = source_texture_summary
    if design_variant_summary is not None:
        metadata["design_variant_import"] = design_variant_summary
    if texture_binding_summary is not None:
        metadata["source_texture_bindings"] = texture_binding_summary
    if material_library_summary is not None:
        metadata["material_library_import"] = material_library_summary
    if material_library_binding_summary is not None:
        metadata["material_library_bindings"] = material_library_binding_summary
    if header_info.schema:
        metadata["step_schema"] = header_info.schema
    if header_info.pmi_present or pmi_count:
        metadata["pmi_present"] = "true"
        metadata["pmi_import_status"] = "imported" if pmi_count else "unsupported" if options.pmi else "disabled"
        metadata["pmi_import_count"] = pmi_count
    return metadata


def _step_header_info(source: Path) -> _StepHeaderInfo:
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(131_072)
    header = text.split("ENDSEC;", 1)[0]
    schema_match = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", header, flags=re.IGNORECASE | re.DOTALL)
    schema = " ".join(schema_match.group(1).split()) if schema_match else ""
    upper_header = header.upper()
    pmi_present = "AP242" in schema.upper() and (
        "PRODUCT MANUFACTURING INFORMATION" in upper_header or "PMI" in upper_header
    )
    return _StepHeaderInfo(schema=schema, pmi_present=pmi_present)


def _extract_step_pmi_annotations(source: Path, options: StepReadOptions) -> list[PmiAnnotation]:
    if not options.pmi:
        return []
    text = source.read_text(encoding="utf-8", errors="ignore")
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

    text = source.read_text(encoding="utf-8", errors="ignore")
    records = {f"#{record.number}": record for record in _iter_step_records(text)}
    pmi_ids = tuple(record_id for record_id, record in records.items() if record.entity in _STEP_PMI_ENTITY_KINDS)
    if not pmi_ids:
        return _StepPmiSemanticGraphExtraction(
            nodes=(),
            edges=(),
            summary=_empty_pmi_semantic_graph_summary(),
            warnings=(),
        )

    referenced_ids: set[str] = set()
    edges: list[_StepPmiSemanticGraphEdge] = []
    missing_references = 0
    for record_id in pmi_ids:
        for reference in _step_record_references(records[record_id]):
            edges.append(
                _StepPmiSemanticGraphEdge(
                    source=record_id,
                    target=reference,
                    relationship="step_reference",
                )
            )
            if reference in records:
                referenced_ids.add(reference)
            else:
                missing_references += 1

    node_ids = sorted(set(pmi_ids) | referenced_ids, key=_step_entity_sort_key)
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
    summary = {
        "nodes": len(nodes),
        "pmi_nodes": len(pmi_ids),
        "referenced_nodes": len(nodes) - len(pmi_ids),
        "edges": len(edges),
        "missing_references": missing_references,
    }
    warnings = (
        (f"STEP PMI semantic graph has {missing_references} reference(s) to records that were not found",)
        if missing_references
        else ()
    )
    return _StepPmiSemanticGraphExtraction(
        nodes=nodes,
        edges=tuple(edges),
        summary=summary,
        warnings=warnings,
    )


def _empty_pmi_semantic_graph_summary() -> dict[str, int]:
    return {
        "nodes": 0,
        "pmi_nodes": 0,
        "referenced_nodes": 0,
        "edges": 0,
        "missing_references": 0,
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
    return f"pmi_{kind}" if kind is not None else "referenced_step_entity"


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


def _extract_step_design_variants(source: Path, options: StepReadOptions) -> _StepDesignVariantExtraction:
    if not options.design_variants and not options.design_variant_selection:
        return _StepDesignVariantExtraction(records=(), summary=_empty_design_variant_summary(), warnings=())

    text = source.read_text(encoding="utf-8", errors="ignore")
    step_records = {f"#{record.number}": record for record in _iter_step_records(text)}
    records: list[_StepDesignVariantRecord] = []
    for record in step_records.values():
        kind = _STEP_DESIGN_VARIANT_ENTITY_KINDS.get(record.entity)
        if kind is None:
            continue
        strings = _step_string_values(record.args)
        references = tuple(f"#{item}" for item in _STEP_REFERENCE_RE.findall(record.args))
        effectivity_values = _step_effectivity_values(record.entity, strings, record.args)
        records.append(
            _StepDesignVariantRecord(
                id=f"step_variant_{record.number}",
                kind=kind,
                entity=record.entity,
                label=" / ".join(strings) if strings else record.entity.lower().replace("_", " "),
                references=references,
                reference_labels=_step_reference_labels(references, step_records),
                resolved_reference_labels=_step_resolved_reference_labels(references, step_records),
                effectivity_kind=_step_effectivity_kind(record.entity),
                effectivity_values=effectivity_values,
                effectivity_range=_step_effectivity_range(record.entity, effectivity_values),
                condition_operator=_step_condition_operator(record.entity),
                condition_value=_step_condition_value(record.entity, record.args),
                condition_number=_step_condition_number(record.entity, record.args),
            )
        )

    warnings = (
        (
            "STEP design variant records were detected and reported as metadata; "
            "pass design_variant_selection to filter geometry by selected variant labels",
        )
        if records and not options.design_variant_selection
        else ()
    )
    return _StepDesignVariantExtraction(
        records=tuple(records),
        summary=_design_variant_summary(records),
        warnings=warnings,
    )


def _step_reference_labels(references: tuple[str, ...], records: dict[str, _StepRecord]) -> tuple[str, ...]:
    labels: list[str] = []
    for reference in references:
        record = records.get(reference)
        if record is None:
            continue
        label = _step_record_label(record)
        if label and label not in labels:
            labels.append(label)
    return tuple(labels)


def _step_resolved_reference_labels(references: tuple[str, ...], records: dict[str, _StepRecord]) -> tuple[str, ...]:
    labels: list[str] = []
    visited: set[str] = set()

    def visit(reference: str, depth: int) -> None:
        if depth > 4 or reference in visited:
            return
        visited.add(reference)
        record = records.get(reference)
        if record is None:
            return
        label = _step_record_label(record)
        if label and label not in labels:
            labels.append(label)
        for child in _step_record_references(record):
            visit(child, depth + 1)

    for reference in references:
        visit(reference, 0)
    return tuple(labels)


def _step_effectivity_kind(entity: str) -> str | None:
    if "EFFECTIVITY" not in entity:
        return None
    if "SERIAL" in entity:
        return "serial"
    if "LOT" in entity:
        return "lot"
    if "DATED" in entity:
        return "date"
    if "TIME_INTERVAL" in entity:
        return "time_interval"
    if "CONFIGURATION" in entity:
        return "configuration"
    if "PRODUCT_DEFINITION" in entity:
        return "product_definition"
    return "generic"


def _step_condition_operator(entity: str) -> str | None:
    normalized = entity.replace("_", "")
    if normalized in {"ANDEXPRESSION", "ANDCONDITION"}:
        return "and"
    if normalized in {"OREXPRESSION", "ORCONDITION"}:
        return "or"
    if normalized in {"XOREXPRESSION", "XORCONDITION"}:
        return "xor"
    if normalized in {"NOTEXPRESSION", "NOTCONDITION"}:
        return "not"
    if normalized in {"COMPARISONEQUAL", "EQUALSEXPRESSION"}:
        return "equals"
    if normalized == "COMPARISONGREATER":
        return "greater"
    if normalized == "COMPARISONGREATEREQUAL":
        return "greater_equal"
    if normalized == "COMPARISONLESS":
        return "less"
    if normalized == "COMPARISONLESSEQUAL":
        return "less_equal"
    if normalized == "COMPARISONNOTEQUAL":
        return "not_equals"
    if normalized in {"BOOLEANLITERAL", "BOOLEANREPRESENTATIONITEM"}:
        return "literal"
    if normalized in {
        "INTLITERAL",
        "INTEGERREPRESENTATIONITEM",
        "LITERALNUMBER",
        "REALLITERAL",
        "REALREPRESENTATIONITEM",
    }:
        return "numeric_literal"
    if normalized in {"CONDITIONALCONFIGURATION", "CONDITIONALCONCEPTFEATURE"}:
        return "conditional"
    if normalized in {"APPLIEDEFFECTIVITYASSIGNMENT", "CONFIGUREDEFFECTIVITYASSIGNMENT", "EFFECTIVITYASSIGNMENT"}:
        return "effectivity_assignment"
    if normalized in {
        "APPLIEDEFFECTIVITYCONTEXTASSIGNMENT",
        "CLASSUSAGEEFFECTIVITYCONTEXTASSIGNMENT",
        "CONFIGUREDEFFECTIVITYCONTEXTASSIGNMENT",
    }:
        return "effectivity_context_assignment"
    if normalized == "APPLIEDINEFFECTIVITYASSIGNMENT":
        return "ineffectivity_assignment"
    if normalized in {"CONFIGURATIONEFFECTIVITY", "PRODUCTDEFINITIONEFFECTIVITY"}:
        return "effectivity_usage"
    if normalized == "EFFECTIVITYRELATIONSHIP":
        return "effectivity_relationship"
    if normalized in {"BOOLEANVARIABLE", "MATHSBOOLEANVARIABLE"}:
        return "variable"
    if normalized in {
        "INTNUMERICVARIABLE",
        "MATHSINTEGERVARIABLE",
        "MATHSREALVARIABLE",
        "NUMERICVARIABLE",
        "REALNUMERICVARIABLE",
    }:
        return "numeric_variable"
    return None


def _step_condition_value(entity: str, args: str) -> bool | None:
    if _step_condition_operator(entity) != "literal":
        return None
    normalized = args.strip().upper()
    if match := _STEP_BOOLEAN_TOKEN_RE.search(normalized):
        return match.group(1) in {"T", "TRUE"}
    if normalized.startswith(".T.") or normalized.startswith(".TRUE.") or normalized in {"T", "TRUE"}:
        return True
    if normalized.startswith(".F.") or normalized.startswith(".FALSE.") or normalized in {"F", "FALSE"}:
        return False
    return None


def _step_condition_number(entity: str, args: str) -> float | None:
    if _step_condition_operator(entity) != "numeric_literal":
        return None
    values = _step_number_values(args)
    return values[-1] if values else None


def _step_effectivity_values(entity: str, strings: list[str], args: str) -> tuple[str, ...]:
    if "EFFECTIVITY" not in entity:
        return ()
    values: list[str] = [item for item in strings if item]
    if not values:
        values.extend(str(value) for value in _step_number_values(args))
    return tuple(dict.fromkeys(values))


def _step_effectivity_range(entity: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not values:
        return ()
    if entity == "SERIAL_NUMBERED_EFFECTIVITY":
        if len(values) >= 3:
            return values[1:3]
        return values[:2]
    if entity in {"DATED_EFFECTIVITY", "TIME_INTERVAL_BASED_EFFECTIVITY"}:
        date_values: list[tuple[date, str]] = []
        for value in values:
            parsed = _parse_effectivity_date(value)
            if parsed is not None:
                date_values.append((parsed, value))
        if date_values:
            date_values.sort(key=lambda item: item[0])
            return tuple(value for _, value in date_values[:2])
        if len(values) >= 2:
            return values[-2:]
    return ()


def _apply_step_design_variant_selection(
    root: Node,
    parts: dict[str, Part],
    materials: dict[str, Material],
    extraction: _StepDesignVariantExtraction,
    options: StepReadOptions,
) -> _StepDesignVariantSelectionResult:
    before_nodes = len(root.walk())
    before_parts = len(parts)
    requested = options.design_variant_selection
    if not requested:
        return _StepDesignVariantSelectionResult(
            requested=(),
            matched_records=(),
            selector_terms=(),
            status="not_requested",
            before_nodes=before_nodes,
            after_nodes=before_nodes,
            before_parts=before_parts,
            after_parts=before_parts,
            removed_nodes=0,
            removed_parts=0,
        )

    matched_records, selector_terms, condition_blocked = _design_variant_selector_terms(extraction.records, requested)
    warnings: list[str] = []
    if not extraction.records:
        warnings.append("STEP design variant selection was requested, but no supported variant records were detected")
        return _StepDesignVariantSelectionResult(
            requested=requested,
            matched_records=(),
            selector_terms=selector_terms,
            status="no_variant_records",
            before_nodes=before_nodes,
            after_nodes=before_nodes,
            before_parts=before_parts,
            after_parts=before_parts,
            removed_nodes=0,
            removed_parts=0,
            warnings=tuple(warnings),
        )
    if condition_blocked and not matched_records:
        warnings.append(
            "STEP design variant selection matched labels controlled by supported condition records, "
            "but the condition expression was not satisfied"
        )
    elif not matched_records:
        warnings.append(
            "STEP design variant selection did not match any supported variant record; "
            "using requested terms directly against imported geometry names"
        )

    original_children = root.children
    filtered_children = [
        selected
        for child in original_children
        if (selected := _filter_design_variant_node(child, parts, selector_terms)) is not None
    ]
    if not filtered_children:
        warnings.append(
            "STEP design variant selection matched no imported geometry by node, part, or source-name metadata"
        )
        return _StepDesignVariantSelectionResult(
            requested=requested,
            matched_records=matched_records,
            selector_terms=selector_terms,
            status="unmatched_geometry",
            before_nodes=before_nodes,
            after_nodes=before_nodes,
            before_parts=before_parts,
            after_parts=before_parts,
            removed_nodes=0,
            removed_parts=0,
            warnings=tuple(warnings),
        )

    root.children = filtered_children
    kept_part_ids = {node.part_id for node in root.walk() if node.part_id is not None}
    removed_part_ids = set(parts) - kept_part_ids
    for part_id in removed_part_ids:
        del parts[part_id]
    used_material_ids = {material_id for part in parts.values() for material_id in part.material_ids}
    for material_id in set(materials) - used_material_ids:
        del materials[material_id]
    for part in parts.values():
        part.metadata["design_variant_selected"] = "true"
        part.metadata["design_variant_selection"] = ",".join(requested)
    root.metadata["design_variant_selection"] = {
        "requested": list(requested),
        "matched_records": list(matched_records),
        "selector_terms": list(selector_terms),
    }

    after_nodes = len(root.walk())
    after_parts = len(parts)
    return _StepDesignVariantSelectionResult(
        requested=requested,
        matched_records=matched_records,
        selector_terms=selector_terms,
        status="applied",
        before_nodes=before_nodes,
        after_nodes=after_nodes,
        before_parts=before_parts,
        after_parts=after_parts,
        removed_nodes=max(0, before_nodes - after_nodes),
        removed_parts=max(0, before_parts - after_parts),
        warnings=tuple(warnings),
    )


def _design_variant_selector_terms(
    records: tuple[_StepDesignVariantRecord, ...],
    requested: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    normalized_requested = tuple(_normalize_variant_term(item) for item in requested)
    records_by_reference = {_design_variant_record_step_reference(record): record for record in records}
    conditional_dependency_refs = _conditional_dependency_references(records, records_by_reference)
    matched: list[str] = []
    terms: list[str] = list(requested)
    condition_blocked = False
    for record in records:
        direct_id_match = _design_variant_record_id_matches(record, normalized_requested)
        condition_match = _condition_record_matches_requested(
            record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(),
        )
        suppress_direct_match = (
            record.condition_operator is None
            and not direct_id_match
            and _design_variant_record_step_reference(record) in conditional_dependency_refs
        )
        condition_applies = (
            record.condition_operator is not None
            and record.condition_operator not in {"numeric_literal", "numeric_variable"}
            and condition_match.matched
            and condition_match.positive
        )
        if record.condition_operator is not None:
            direct_match = record.condition_operator not in {
                "conditional",
                "effectivity_assignment",
                "effectivity_context_assignment",
                "effectivity_relationship",
                "effectivity_usage",
                "ineffectivity_assignment",
                "numeric_literal",
                "numeric_variable",
            } and _design_variant_record_self_matches_requested(record, normalized_requested)
            condition_blocked = condition_blocked or (
                not direct_id_match
                and not direct_match
                and not condition_applies
                and _design_variant_record_matches_requested(record, requested, normalized_requested)
            )
        else:
            requested_record_match = _design_variant_record_matches_requested(record, requested, normalized_requested)
            direct_match = not suppress_direct_match and _design_variant_record_matches_requested(
                record, requested, normalized_requested
            )
            condition_blocked = condition_blocked or (suppress_direct_match and requested_record_match)
        condition_blocked = condition_blocked or (
            record.condition_operator in {"greater", "greater_equal", "less", "less_equal"}
            and not condition_applies
            and _condition_record_has_requested_numeric_operand(
                record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(),
            )
        )
        if not direct_id_match and not direct_match and not condition_applies:
            continue
        matched.append(record.id)
        terms.extend(_design_variant_label_terms(record.label))
        for label in record.reference_labels:
            terms.extend(_design_variant_label_terms(label))
        for label in record.resolved_reference_labels:
            terms.extend(_design_variant_label_terms(label))
        terms.extend(record.effectivity_values)
        terms.extend(record.references)
    selector_terms = tuple(
        item for item in dict.fromkeys(term.strip() for term in terms if term.strip()) if _normalize_variant_term(item)
    )
    if condition_blocked and not matched:
        selector_terms = ()
    return tuple(dict.fromkeys(matched)), selector_terms, condition_blocked


def _conditional_dependency_references(
    records: tuple[_StepDesignVariantRecord, ...],
    records_by_reference: dict[str, _StepDesignVariantRecord],
) -> set[str]:
    dependency_operators = {
        "and",
        "or",
        "xor",
        "not",
        "equals",
        "greater",
        "greater_equal",
        "less",
        "less_equal",
        "not_equals",
        "conditional",
        "effectivity_assignment",
        "effectivity_context_assignment",
        "effectivity_relationship",
        "effectivity_usage",
        "ineffectivity_assignment",
    }
    dependencies: set[str] = {
        reference
        for record in records
        if record.condition_operator in dependency_operators
        for reference in record.references
    }
    changed = True
    while changed:
        changed = False
        for record in records:
            record_reference = _design_variant_record_step_reference(record)
            if (
                record.condition_operator is None
                and record_reference not in dependencies
                and any(reference in dependencies for reference in record.references)
            ):
                dependencies.add(record_reference)
                changed = True
    return {reference for reference in dependencies if reference in records_by_reference}


def _condition_record_matches_requested(
    record: _StepDesignVariantRecord,
    records_by_reference: dict[str, _StepDesignVariantRecord],
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
    *,
    visited: set[str],
) -> _StepConditionMatch:
    record_reference = _design_variant_record_step_reference(record)
    if record_reference in visited:
        return _StepConditionMatch(matched=False)
    visited.add(record_reference)
    operator = record.condition_operator
    if operator is None:
        matched = _design_variant_record_matches_requested(record, requested, normalized_requested)
        return _StepConditionMatch(matched=matched, positive=matched)
    if operator == "literal":
        return _StepConditionMatch(matched=bool(record.condition_value))
    if operator == "variable":
        matched = _design_variant_record_self_matches_requested(record, normalized_requested)
        return _StepConditionMatch(matched=matched, positive=matched)
    if operator == "numeric_literal":
        return _StepConditionMatch(matched=record.condition_number is not None)
    if operator == "numeric_variable":
        matched = _design_variant_record_requested_number(record, requested, normalized_requested) is not None
        return _StepConditionMatch(matched=matched, positive=matched)
    if operator == "effectivity_usage":
        matched = _effectivity_condition_record_matches_requested(
            record,
            requested,
            normalized_requested,
        )
        return _StepConditionMatch(matched=matched, positive=matched)

    child_records = [
        child for reference in record.references if (child := records_by_reference.get(reference)) is not None
    ]
    children = [
        _condition_record_matches_requested(
            child_record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        for child_record in child_records
    ]
    if not children:
        matched = _design_variant_record_matches_requested(record, requested, normalized_requested)
        return _StepConditionMatch(matched=matched, positive=matched)
    if operator == "not":
        child_match = children[0]
        return _StepConditionMatch(matched=not child_match.matched, positive=False)
    if operator == "conditional":
        condition_children = [
            child_match
            for child_record, child_match in zip(child_records, children, strict=False)
            if child_record.condition_operator is not None
        ]
        target_children = [
            child_match
            for child_record, child_match in zip(child_records, children, strict=False)
            if child_record.condition_operator is None
        ]
        if condition_children:
            matched = all(child.matched for child in condition_children)
            return _StepConditionMatch(
                matched=matched,
                positive=matched
                and (
                    any(child.positive for child in condition_children)
                    or any(child.positive for child in target_children)
                ),
            )
    if operator in {"effectivity_assignment", "ineffectivity_assignment"}:
        effectivity_condition_children: list[_StepConditionMatch] = []
        effectivity_target_children: list[_StepConditionMatch] = []
        for child_record, child_match in zip(child_records, children, strict=False):
            if child_record.effectivity_kind is not None:
                matched = _effectivity_condition_record_matches_requested(
                    child_record,
                    requested,
                    normalized_requested,
                )
                effectivity_condition_children.append(_StepConditionMatch(matched=matched, positive=matched))
            elif child_record.condition_operator is not None:
                effectivity_condition_children.append(child_match)
            else:
                effectivity_target_children.append(child_match)
        if effectivity_condition_children:
            matched = all(child.matched for child in effectivity_condition_children)
            if operator == "ineffectivity_assignment":
                return _StepConditionMatch(matched=matched, positive=False)
            return _StepConditionMatch(
                matched=matched,
                positive=matched
                and (
                    any(child.positive for child in effectivity_condition_children)
                    or any(child.positive for child in effectivity_target_children)
                ),
            )
    if operator == "effectivity_context_assignment":
        assignment_operator_kinds = ("effectivity_assignment", "ineffectivity_assignment")
        assignment_children = [
            child_match
            for child_record, child_match in zip(child_records, children, strict=False)
            if child_record.condition_operator in assignment_operator_kinds
        ]
        if assignment_children:
            matched = all(child.matched for child in assignment_children)
            return _StepConditionMatch(
                matched=matched,
                positive=matched and any(child.positive for child in assignment_children),
            )
    if operator == "effectivity_relationship":
        effectivity_children = [
            child_match
            for child_record, child_match in zip(child_records, children, strict=False)
            if child_record.effectivity_kind is not None
        ]
        if effectivity_children:
            matched_children = [child for child in effectivity_children if child.matched]
            return _StepConditionMatch(
                matched=bool(matched_children),
                positive=any(child.positive for child in matched_children),
            )
    if operator == "and":
        matched = all(child.matched for child in children)
        return _StepConditionMatch(matched=matched, positive=matched and any(child.positive for child in children))
    if operator == "or":
        matched_children = [child for child in children if child.matched]
        return _StepConditionMatch(
            matched=bool(matched_children),
            positive=any(child.positive for child in matched_children),
        )
    if operator == "xor":
        matched_children = [child for child in children if child.matched]
        return _StepConditionMatch(
            matched=len(matched_children) == 1,
            positive=any(child.positive for child in matched_children),
        )
    if operator in {"greater", "greater_equal", "less", "less_equal"}:
        numeric_children = [
            _condition_record_numeric_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for child_record in child_records
        ]
        values = [item[0] for item in numeric_children if item[0] is not None]
        matched = (
            len(values) == len(numeric_children)
            and len(values) >= 2
            and _numeric_comparison_matches(
                operator,
                values,
            )
        )
        return _StepConditionMatch(matched=matched, positive=matched and any(item[1] for item in numeric_children))
    if operator == "equals":
        matched = (
            len(children) >= 2
            and any(child.matched for child in children)
            and all(child.matched == children[0].matched for child in children)
        )
        return _StepConditionMatch(matched=matched, positive=matched and any(child.positive for child in children))
    if operator == "not_equals":
        matched = (
            len(children) >= 2
            and any(child.matched for child in children)
            and any(child.matched != children[0].matched for child in children[1:])
        )
        return _StepConditionMatch(matched=matched, positive=matched and any(child.positive for child in children))

    matched_children = [child for child in children if child.matched]
    return _StepConditionMatch(
        matched=bool(matched_children),
        positive=any(child.positive for child in matched_children),
    )


def _condition_record_numeric_value(
    record: _StepDesignVariantRecord,
    records_by_reference: dict[str, _StepDesignVariantRecord],
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
    *,
    visited: set[str],
) -> tuple[float | None, bool]:
    record_reference = _design_variant_record_step_reference(record)
    if record_reference in visited:
        return None, False
    visited.add(record_reference)
    if record.condition_operator == "numeric_literal":
        return record.condition_number, False
    if record.condition_operator == "numeric_variable":
        value = _design_variant_record_requested_number(record, requested, normalized_requested)
        return value, value is not None
    return None, False


def _condition_record_has_requested_numeric_operand(
    record: _StepDesignVariantRecord,
    records_by_reference: dict[str, _StepDesignVariantRecord],
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
    *,
    visited: set[str],
) -> bool:
    record_reference = _design_variant_record_step_reference(record)
    if record_reference in visited:
        return False
    visited.add(record_reference)
    if record.condition_operator == "numeric_variable":
        return _design_variant_record_requested_number(record, requested, normalized_requested) is not None
    return any(
        _condition_record_has_requested_numeric_operand(
            child_record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        for reference in record.references
        if (child_record := records_by_reference.get(reference)) is not None
    )


def _numeric_comparison_matches(operator: str, values: list[float]) -> bool:
    pairs = zip(values, values[1:], strict=False)
    if operator == "greater":
        return all(left > right for left, right in pairs)
    if operator == "greater_equal":
        return all(left >= right for left, right in pairs)
    if operator == "less":
        return all(left < right for left, right in pairs)
    if operator == "less_equal":
        return all(left <= right for left, right in pairs)
    return False


def _design_variant_record_matches_requested(
    record: _StepDesignVariantRecord,
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
) -> bool:
    haystack = _normalize_variant_term(_design_variant_record_haystack(record))
    return any(query and query in haystack for query in normalized_requested) or _effectivity_record_matches_requested(
        record, requested
    )


def _design_variant_record_self_matches_requested(
    record: _StepDesignVariantRecord,
    normalized_requested: tuple[str, ...],
) -> bool:
    haystack = _normalize_variant_term(
        " ".join(
            (
                record.id,
                _design_variant_record_step_reference(record),
                record.entity,
                record.kind,
                record.label,
                *(record.effectivity_values or ()),
            )
        )
    )
    return any(query and query in haystack for query in normalized_requested)


def _design_variant_record_requested_number(
    record: _StepDesignVariantRecord,
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
) -> float | None:
    selectors = tuple(
        dict.fromkeys(
            selector
            for value in (record.label, record.id, _design_variant_record_step_reference(record))
            if (selector := _normalize_variant_term(value))
        )
    )
    for raw, normalized in zip(requested, normalized_requested, strict=False):
        if not any(selector and selector in normalized for selector in selectors):
            continue
        value_text = raw
        for separator in ("=", ":"):
            if separator not in raw:
                continue
            left, right = raw.split(separator, 1)
            normalized_left = _normalize_variant_term(left)
            if any(selector and selector in normalized_left for selector in selectors):
                value_text = right
                break
        numbers = _step_number_values(value_text)
        if numbers:
            return numbers[-1]
    return None


def _effectivity_condition_record_matches_requested(
    record: _StepDesignVariantRecord,
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
) -> bool:
    return _design_variant_record_self_matches_requested(
        record, normalized_requested
    ) or _effectivity_record_matches_requested(
        record,
        requested,
    )


def _design_variant_record_id_matches(
    record: _StepDesignVariantRecord,
    normalized_requested: tuple[str, ...],
) -> bool:
    normalized_ids = {
        _normalize_variant_term(record.id),
        _normalize_variant_term(_design_variant_record_step_reference(record)),
    }
    return any(query and query in normalized_ids for query in normalized_requested)


def _design_variant_record_haystack(record: _StepDesignVariantRecord) -> str:
    return " ".join(
        (
            record.id,
            _design_variant_record_step_reference(record),
            record.entity,
            record.kind,
            record.label,
            *(record.effectivity_values or ()),
            *record.references,
            *record.reference_labels,
            *record.resolved_reference_labels,
        )
    )


def _design_variant_record_step_reference(record: _StepDesignVariantRecord) -> str:
    suffix = record.id.removeprefix("step_variant_")
    return f"#{suffix}" if suffix.isdigit() else record.id


def _effectivity_record_matches_requested(record: _StepDesignVariantRecord, requested: tuple[str, ...]) -> bool:
    if not record.effectivity_range:
        return False
    return any(
        _effectivity_range_contains(record.effectivity_kind, record.effectivity_range, item) for item in requested
    )


def _effectivity_range_contains(kind: str | None, bounds: tuple[str, ...], value: str) -> bool:
    if not bounds or not value.strip():
        return False
    start = bounds[0] if len(bounds) >= 1 else ""
    end = bounds[1] if len(bounds) >= 2 else ""
    if kind in {"date", "time_interval"}:
        target_date = _parse_effectivity_date(value)
        start_date = _parse_effectivity_date(start)
        end_date = _parse_effectivity_date(end)
        if target_date is not None and (start_date is not None or end_date is not None):
            return (start_date is None or target_date >= start_date) and (end_date is None or target_date <= end_date)
    return (not start or _compare_ordered_identifier(value, start) >= 0) and (
        not end or _compare_ordered_identifier(value, end) <= 0
    )


def _parse_effectivity_date(value: str) -> date | None:
    stripped = value.strip()
    if not stripped:
        return None
    iso_match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", stripped)
    if iso_match is not None:
        try:
            return date.fromisoformat(iso_match.group(1))
        except ValueError:
            return None
    compact_match = re.fullmatch(r"\D*(\d{4})(\d{2})(\d{2})\D*", stripped)
    if compact_match is None:
        return None
    try:
        return date(int(compact_match.group(1)), int(compact_match.group(2)), int(compact_match.group(3)))
    except ValueError:
        return None


def _compare_ordered_identifier(left: str, right: str) -> int:
    left_key = _ordered_identifier_key(left)
    right_key = _ordered_identifier_key(right)
    if left_key is not None and right_key is not None and left_key[:2] == right_key[:2]:
        left_number = left_key[2]
        right_number = right_key[2]
        return (left_number > right_number) - (left_number < right_number)
    normalized_left = left.strip().lower()
    normalized_right = right.strip().lower()
    return (normalized_left > normalized_right) - (normalized_left < normalized_right)


def _ordered_identifier_key(value: str) -> tuple[str, str, int] | None:
    match = re.fullmatch(r"(\D*?)(\d+)(\D*)", value.strip().lower())
    if match is None:
        return None
    return (match.group(1), match.group(3), int(match.group(2)))


def _design_variant_label_terms(label: str) -> tuple[str, ...]:
    terms: list[str] = []
    for item in (part.strip() for part in label.split(" / ")):
        if not item:
            continue
        terms.append(item)
        words = item.split()
        if len(words) > 1 and words[0].lower() in {"select", "selected", "choose", "chosen"}:
            terms.append(" ".join(words[1:]))
    return tuple(dict.fromkeys(terms))


def _filter_design_variant_node(
    node: Node,
    parts: dict[str, Part],
    selector_terms: tuple[str, ...],
) -> Node | None:
    filtered_children = [
        selected
        for child in node.children
        if (selected := _filter_design_variant_node(child, parts, selector_terms)) is not None
    ]
    if _design_variant_node_matches(node, parts, selector_terms):
        kept = node.copy()
        return kept
    if filtered_children:
        kept = node.copy()
        kept.children = filtered_children
        return kept
    return None


def _design_variant_node_matches(node: Node, parts: dict[str, Part], selector_terms: tuple[str, ...]) -> bool:
    fields = [node.name, *[str(value) for value in node.metadata.values()]]
    if node.part_id is not None and node.part_id in parts:
        part = parts[node.part_id]
        fields.extend([part.id, part.name, *[str(value) for value in part.metadata.values()]])
    haystack = _normalize_variant_term(" ".join(fields))
    return any(term and term in haystack for term in (_normalize_variant_term(item) for item in selector_terms))


def _normalize_variant_term(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9#]+", value.lower()))


def _design_variant_summary(records: Iterable[_StepDesignVariantRecord]) -> dict[str, int]:
    items = list(records)
    return {
        "records": len(items),
        "configuration_items": sum(1 for item in items if item.entity == "CONFIGURATION_ITEM"),
        "product_concept_features": sum(
            1 for item in items if item.entity in {"PRODUCT_CONCEPT_FEATURE", "CONDITIONAL_CONCEPT_FEATURE"}
        ),
        "effectivity_records": sum(
            1
            for item in items
            if "EFFECTIVITY" in item.entity
            or item.entity in {"CONFIGURATION_DESIGN", "CONFIGURED_EFFECTIVITY_ASSIGNMENT"}
        ),
        "conditional_records": sum(1 for item in items if item.condition_operator is not None),
    }


def _empty_design_variant_summary() -> dict[str, int]:
    return {
        "records": 0,
        "configuration_items": 0,
        "product_concept_features": 0,
        "effectivity_records": 0,
        "conditional_records": 0,
    }


def _iter_step_records(text: str) -> list[_StepRecord]:
    records: list[_StepRecord] = []
    position = 0
    while match := _STEP_RECORD_START_RE.search(text, position):
        args_start = match.end()
        args_end = _find_step_record_args_end(text, args_start - 1)
        if args_end is None:
            position = match.end()
            continue
        records.append(
            _StepRecord(
                number=int(match.group(1)),
                entity=match.group(2).upper(),
                args=text[args_start:args_end],
            )
        )
        position = args_end + 1
    return records


def _find_step_record_args_end(text: str, open_paren_index: int) -> int | None:
    depth = 0
    in_string = False
    index = open_paren_index
    while index < len(text):
        char = text[index]
        if in_string:
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                in_string = False
            index += 1
            continue
        if char == "'":
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
        index += 1
    return None


def _step_string_values(text: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "'":
            index += 1
            continue
        index += 1
        value: list[str] = []
        while index < len(text):
            char = text[index]
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    value.append("'")
                    index += 2
                    continue
                index += 1
                break
            value.append(char)
            index += 1
        cleaned = " ".join("".join(value).split())
        if cleaned:
            values.append(cleaned)
    return values


def _step_number_values(text: str) -> list[float]:
    unquoted = _strip_step_strings(text)
    return [float(match.group(0)) for match in _STEP_NUMBER_RE.finditer(unquoted)]


def _strip_step_strings(text: str) -> str:
    chars = list(text)
    index = 0
    while index < len(chars):
        if chars[index] != "'":
            index += 1
            continue
        chars[index] = " "
        index += 1
        while index < len(chars):
            char = chars[index]
            chars[index] = " "
            if char == "'":
                if index + 1 < len(chars) and chars[index + 1] == "'":
                    chars[index + 1] = " "
                    index += 2
                    continue
                index += 1
                break
            index += 1
    return "".join(chars)


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


def _unsupported_pmi_count(options: StepReadOptions, header_info: _StepHeaderInfo, *, pmi_count: int) -> int:
    if not options.pmi or not header_info.pmi_present or pmi_count:
        return 0
    return 1


def _import_warnings(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    unsupported_pmi_count: int,
    *,
    design_variant_count: int = 0,
) -> list[str]:
    warnings: list[str] = []
    if options.pmi and unsupported_pmi_count:
        warnings.append(
            "STEP file advertises AP242 PMI, but no supported typed PMI entities were extracted; annotations are omitted"
        )
    if (options.design_variants or options.design_variant_selection) and design_variant_count == 0:
        warnings.append(
            "STEP design variant import was requested, but no supported design variant records were detected"
        )
    if options.multi_file:
        warnings.append("multi-file STEP assembly import is not implemented; external references are not loaded")
    return warnings


def _import_decisions(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    *,
    pmi_count: int,
    unsupported_pmi_count: int,
    cleanup: _ImportCleanupStats,
    space: _SpaceNormalization,
    source_texture_summary: dict[str, int] | None = None,
    texture_binding_summary: dict[str, int] | None = None,
    material_library_summary: dict[str, int] | None = None,
    material_library_binding_summary: dict[str, int] | None = None,
    pmi_semantic_graph_summary: dict[str, int] | None = None,
    design_variant_summary: dict[str, int] | None = None,
    design_variant_selection_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    cleanup_counts = cleanup.to_dict()
    texture_summary = source_texture_summary or {
        "references": 0,
        "resolved": 0,
        "missing": 0,
        "unsupported": 0,
        "unreadable": 0,
    }
    binding_summary = texture_binding_summary or {"bound_images": 0, "bound_materials": 0, "unbound_images": 0}
    library_summary = material_library_summary or _empty_material_library_summary()
    library_binding_summary = material_library_binding_summary or _empty_material_library_binding_summary()
    pmi_graph_summary = {**_empty_pmi_semantic_graph_summary(), **(pmi_semantic_graph_summary or {})}
    variant_summary = {**_empty_design_variant_summary(), **(design_variant_summary or {})}
    return {
        "metadata": _import_decision(
            requested=options.metadata,
            effective=options.metadata,
            state="honored" if options.metadata else "disabled",
        ),
        "product_metadata": _import_decision(
            requested=options.product_metadata,
            effective=options.product_metadata,
            state="honored" if options.product_metadata else "disabled",
        ),
        "properties": _import_decision(
            requested=options.properties,
            effective=options.metadata or options.properties,
            state="honored" if options.properties else "disabled",
            detail="OCP metadata transfer is enabled when general metadata or properties are requested",
        ),
        "layers": _import_decision(
            requested=options.layers,
            effective=False,
            state="unsupported" if options.layers else "disabled",
            detail="normalized STEP layer records are not exposed by the current importer",
        ),
        "validation_properties": _import_decision(
            requested=options.validation_properties,
            effective=options.validation_properties,
            state="approximated" if options.validation_properties else "disabled",
            detail="source topology counts are derived after transfer; typed STEP validation properties are not extracted",
        ),
        "pmi": _pmi_import_decision(options, header_info, pmi_count, unsupported_pmi_count, pmi_graph_summary),
        "design_variants": _design_variant_import_decision(options, variant_summary, design_variant_selection_summary),
        "existing_meshes": _import_decision(
            requested=options.existing_meshes,
            effective=options.existing_meshes,
            state="backend_default" if options.existing_meshes else "disabled",
            detail="OCP transfer keeps BREP source shapes; reusable source tessellation payloads are not detected separately yet",
        ),
        "multi_file": _import_decision(
            requested=options.multi_file,
            effective=False,
            state="unsupported" if options.multi_file else "disabled",
            detail="multi-file STEP external-reference resolution is not implemented",
        ),
        "source_textures": _import_decision(
            requested=options.source_textures,
            effective=options.source_textures and texture_summary["resolved"] > 0,
            state=(
                "disabled"
                if not options.source_textures
                else "honored"
                if texture_summary["resolved"] > 0 or texture_summary["references"] == 0
                else "missing_sources"
            ),
            detail=(
                "external source texture references are scanned from STEP string records and resolved "
                "against the source directory plus source_texture_search_paths"
            ),
            counts={
                "references": texture_summary["references"],
                "resolved": texture_summary["resolved"],
                "missing": texture_summary["missing"],
                "unsupported": texture_summary["unsupported"],
                "unreadable": texture_summary["unreadable"],
                "bound_images": binding_summary["bound_images"],
                "bound_materials": binding_summary["bound_materials"],
                "unbound_images": binding_summary["unbound_images"],
            },
        ),
        "material_library_mapping": _import_decision(
            requested=options.material_library_mapping,
            effective=options.material_library_mapping
            and (library_binding_summary["applied_materials"] > 0 or library_summary["references"] == 0),
            state=_material_library_import_state(options, library_summary, library_binding_summary),
            detail=(
                "known CAD material names and supported JSON/MTL sidecar material libraries are mapped "
                "to PBR factors and texture slots when present"
            ),
            counts={**library_summary, **library_binding_summary},
        ),
        "delete_free_vertices": _import_decision(
            requested=options.delete_free_vertices,
            effective=options.delete_free_vertices,
            state="honored" if options.delete_free_vertices else "disabled",
            counts={
                "deleted_parts": cleanup_counts["deleted_free_vertex_parts"],
                "deleted_vertices": cleanup_counts["deleted_free_vertices"],
            },
        ),
        "delete_lines": _import_decision(
            requested=options.delete_lines,
            effective=options.delete_lines,
            state="honored" if options.delete_lines else "disabled",
            counts={
                "deleted_parts": cleanup_counts["deleted_line_parts"],
                "deleted_edges": cleanup_counts["deleted_line_edges"],
                "deleted_vertices": cleanup_counts["deleted_line_vertices"],
            },
        ),
        "construction_curves": _construction_curve_import_decision(options, cleanup_counts),
        "space_normalization": _import_decision(
            requested={
                "source_units": options.source_units,
                "source_meters_per_unit": options.source_meters_per_unit,
                "source_up_axis": options.source_up_axis,
                "source_handedness": options.source_handedness,
                "target_units": options.target_units,
                "target_meters_per_unit": options.target_meters_per_unit,
                "target_up_axis": options.target_up_axis,
                "target_handedness": options.target_handedness,
            },
            effective=space.metadata(),
            state="honored" if space.changed else "backend_default",
        ),
    }


def _import_decision(
    *,
    requested: object,
    effective: object,
    state: str,
    detail: str | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    decision: dict[str, object] = {
        "requested": requested,
        "effective": effective,
        "state": state,
    }
    if detail:
        decision["detail"] = detail
    if counts is not None:
        decision["counts"] = counts
    return decision


def _pmi_import_decision(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    pmi_count: int,
    unsupported_pmi_count: int,
    semantic_graph_summary: dict[str, int],
) -> dict[str, object]:
    counts = {
        "imported": pmi_count,
        "unsupported": unsupported_pmi_count,
        "semantic_graph_nodes": semantic_graph_summary["nodes"],
        "semantic_graph_edges": semantic_graph_summary["edges"],
        "semantic_graph_missing_references": semantic_graph_summary["missing_references"],
    }
    if not options.pmi:
        return _import_decision(requested=False, effective=False, state="disabled")
    if pmi_count:
        return _import_decision(
            requested=True,
            effective=True,
            state="honored",
            detail="common STEP AP242 PMI entities were extracted into typed metadata annotations and a semantic reference graph",
            counts=counts,
        )
    if unsupported_pmi_count:
        return _import_decision(
            requested=True,
            effective=False,
            state="unsupported",
            detail="STEP AP242 PMI markers were detected, but typed PMI entity extraction is not implemented",
            counts=counts,
        )
    if not header_info.pmi_present:
        return _import_decision(
            requested=True,
            effective=False,
            state="not_present",
            detail="PMI import was requested, but the STEP header did not advertise PMI content",
            counts=counts,
        )
    return _import_decision(
        requested=True,
        effective=True,
        state="honored",
        counts=counts,
    )


def _design_variant_import_decision(
    options: StepReadOptions,
    summary: dict[str, int],
    selection_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    counts = {**_empty_design_variant_summary(), **summary}
    if selection_summary:
        for key in ("before_nodes", "after_nodes", "before_parts", "after_parts", "removed_nodes", "removed_parts"):
            value = selection_summary.get(key)
            if isinstance(value, int):
                counts[f"selection_{key}"] = value
        matched = selection_summary.get("matched_records")
        requested = selection_summary.get("requested")
        if isinstance(matched, list):
            counts["selection_matched_records"] = len(matched)
        if isinstance(requested, list):
            counts["selection_requested"] = len(requested)
    if not options.design_variants and not options.design_variant_selection:
        return _import_decision(requested=False, effective=False, state="disabled")
    if counts["records"] == 0:
        return _import_decision(
            requested=True if options.design_variants else list(options.design_variant_selection),
            effective=False,
            state="not_present",
            detail="design variant import was requested, but no supported STEP configuration records were detected",
            counts=counts,
        )
    if options.design_variant_selection and selection_summary:
        status = str(selection_summary.get("status", "not_requested"))
        if status == "applied":
            detail = (
                "supported STEP configuration/design-variant records were scanned and the imported geometry tree "
                "was filtered using selected variant record labels, effectivity values/ranges, simple condition "
                "records, and referenced STEP labels"
            )
        else:
            detail = (
                "supported STEP configuration/design-variant records were scanned, but selected variant geometry "
                "could not be matched by the current name/reference-based selector"
            )
        return _import_decision(
            requested=list(options.design_variant_selection),
            effective=status == "applied",
            state="approximated" if status == "applied" else "unmatched",
            detail=detail,
            counts=counts,
        )
    return _import_decision(
        requested=True,
        effective=True,
        state="approximated",
        detail=(
            "supported STEP configuration/design-variant records are reported as metadata; "
            "pass design_variant_selection to apply name/reference/condition-based geometry filtering"
        ),
        counts=counts,
    )


def _construction_curve_import_decision(options: StepReadOptions, cleanup_counts: dict[str, int]) -> dict[str, object]:
    policy = _construction_curve_policy(options)
    counts = {
        "preserved_parts": cleanup_counts["construction_line_parts"],
        "deleted_parts": cleanup_counts["deleted_line_parts"],
        "deleted_edges": cleanup_counts["deleted_line_edges"],
        "deleted_vertices": cleanup_counts["deleted_line_vertices"],
    }
    if policy == "delete":
        return _import_decision(
            requested=policy,
            effective=counts["deleted_parts"] > 0,
            state="honored" if counts["deleted_parts"] > 0 else "not_present",
            detail="construction-only line shapes are deleted during import",
            counts=counts,
        )
    if policy == "tessellate_tubes":
        return _import_decision(
            requested=policy,
            effective=counts["preserved_parts"] > 0,
            state="approximated" if counts["preserved_parts"] > 0 else "not_present",
            detail=(
                "construction-only line shapes are preserved with tube tessellation metadata; "
                "the tessellation step converts them to triangle tubes"
            ),
            counts=counts,
        )
    return _import_decision(
        requested=policy,
        effective=counts["preserved_parts"] > 0,
        state="honored" if counts["preserved_parts"] > 0 else "not_present",
        detail="construction-only line shapes are preserved as source-shape metadata without mesh geometry",
        counts=counts,
    )


def _loaded_representation_report(asset: Asset) -> dict[str, object]:
    parts = [_part_representation_record(part) for part in sorted(asset.parts.values(), key=lambda item: item.id)]
    deleted_nodes = [
        _deleted_node_representation_record(node)
        for node in asset.root.walk()
        if "import_cleanup" in node.metadata and node.part_id is None
    ]
    return {
        "summary": _representation_summary(parts, deleted_nodes),
        "parts": parts,
        "deleted_nodes": deleted_nodes,
    }


def _part_representation_record(part: Part) -> dict[str, object]:
    record: dict[str, object] = {
        "part_id": part.id,
        "name": part.name,
        "loaded_representation": str(part.metadata.get("loaded_representation", "unknown")),
        "cleanup_action": str(part.metadata.get("import_cleanup", "preserved")),
        "source_vertices": _metadata_int(part.metadata.get("source_vertices")),
        "source_edges": _metadata_int(part.metadata.get("source_edges")),
        "source_faces": _metadata_int(part.metadata.get("source_faces")),
        "source_name": str(part.metadata.get("source_name", "")),
    }
    if "construction_curve_policy" in part.metadata:
        record["construction_curve_policy"] = str(part.metadata["construction_curve_policy"])
    if "construction_curve_tube_radius" in part.metadata:
        record["construction_curve_tube_radius"] = _metadata_float(part.metadata["construction_curve_tube_radius"])
    if "mixed_construction_curve_action" in part.metadata:
        record["mixed_construction_curve_action"] = str(part.metadata["mixed_construction_curve_action"])
        record["mixed_construction_curve_edges"] = _metadata_int(part.metadata.get("mixed_construction_curve_edges"))
    return record


def _deleted_node_representation_record(node: Node) -> dict[str, object]:
    record: dict[str, object] = {
        "node_id": node.id,
        "name": node.name,
        "loaded_representation": str(node.metadata.get("loaded_representation", "unknown")),
        "cleanup_action": str(node.metadata.get("import_cleanup", "preserved")),
        "source_vertices": _metadata_int(node.metadata.get("source_vertices")),
        "source_edges": _metadata_int(node.metadata.get("source_edges")),
        "source_faces": _metadata_int(node.metadata.get("source_faces")),
    }
    if "construction_curve_policy" in node.metadata:
        record["construction_curve_policy"] = str(node.metadata["construction_curve_policy"])
    if "mixed_construction_curve_split" in node.metadata:
        record["mixed_construction_curve_split"] = str(node.metadata["mixed_construction_curve_split"])
    return record


def _representation_summary(
    parts: list[dict[str, object]],
    deleted_nodes: list[dict[str, object]],
) -> dict[str, int]:
    summary = {
        "brep_parts": 0,
        "construction_point_parts": 0,
        "construction_line_parts": 0,
        "empty_shape_parts": 0,
        "unknown_parts": 0,
        "deleted_nodes": len(deleted_nodes),
        "deleted_free_vertex_nodes": 0,
        "deleted_line_nodes": 0,
    }
    for part in parts:
        representation = part.get("loaded_representation")
        if representation == "brep":
            summary["brep_parts"] += 1
        elif representation == "construction_points":
            summary["construction_point_parts"] += 1
        elif representation == "construction_lines":
            summary["construction_line_parts"] += 1
        elif representation == "empty_shape":
            summary["empty_shape_parts"] += 1
        else:
            summary["unknown_parts"] += 1
    for node in deleted_nodes:
        cleanup_action = node.get("cleanup_action")
        if cleanup_action == "delete_free_vertices":
            summary["deleted_free_vertex_nodes"] += 1
        elif cleanup_action == "delete_lines":
            summary["deleted_line_nodes"] += 1
    return summary


def _metadata_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _metadata_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _loaded_representation(counts: _ShapeTopologyCounts) -> str:
    if counts.faces > 0:
        return "brep"
    if counts.edges > 0:
        return "construction_lines"
    if counts.vertices > 0:
        return "construction_points"
    return "empty_shape"


def _cleanup_action(counts: _ShapeTopologyCounts, options: StepReadOptions) -> str | None:
    if counts.faces > 0:
        return None
    if counts.edges > 0 and _construction_curve_policy(options) == "delete":
        return "delete_lines"
    if counts.edges == 0 and counts.vertices > 0 and options.delete_free_vertices:
        return "delete_free_vertices"
    return None


def _construction_curve_policy(options: StepReadOptions) -> str:
    return "delete" if options.delete_lines else options.construction_curve_policy


def _construction_curve_metadata(options: StepReadOptions, representation: str) -> dict[str, str]:
    if representation != "construction_lines":
        return {}
    metadata = {"construction_curve_policy": _construction_curve_policy(options)}
    if metadata["construction_curve_policy"] == "tessellate_tubes":
        metadata["construction_curve_tube_radius"] = str(options.construction_curve_tube_radius)
    return metadata


def _metadata_count(asset: Asset) -> int:
    return (
        len(asset.metadata)
        + sum(len(node.metadata) for node in asset.root.walk())
        + sum(len(part.metadata) for part in asset.parts.values())
        + sum(len(material.metadata) for material in asset.materials.values())
    )


def _build_node(
    label: Any,
    occurrence_path: str,
    source_identity: str,
    shape_tool: Any,
    color_tool: Any,
    vis_material_tool: Any,
    parts: dict[str, Part],
    part_index: _PartIndex,
    materials: dict[str, Material],
    options: StepReadOptions,
    cleanup: _ImportCleanupStats,
) -> Node:
    from OCP.TDF import TDF_LabelSequence
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    label_entry = _label_entry(label)
    node = Node(
        id=_stable_id("node", f"{source_identity}:{occurrence_path}"),
        name=_label_name(label) or f"Node {label_entry}",
        transform=_label_transform(label),
        metadata={"step_label": label_entry},
    )

    if XCAFDoc_ShapeTool.IsAssembly_s(label):
        children = TDF_LabelSequence()
        XCAFDoc_ShapeTool.GetComponents_s(label, children, False)
        for index in range(children.Lower(), children.Upper() + 1):
            child = children.Value(index)
            node.children.append(
                _build_node(
                    child,
                    f"{occurrence_path}/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    vis_material_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )
        return node

    shape_label = _shape_definition_label(label)
    shape = XCAFDoc_ShapeTool.GetShape_s(shape_label)
    if shape.IsNull():
        return node
    topology = _shape_topology_counts(shape)
    representation = _loaded_representation(topology)
    mixed_construction_shape = _mixed_construction_curve_shape(shape, topology)
    mixed_construction_counts = (
        _shape_topology_counts(mixed_construction_shape) if mixed_construction_shape is not None else None
    )
    cleanup_action = _cleanup_action(topology, options)
    if cleanup_action is not None:
        cleanup.record_deleted(cleanup_action, topology)
        node.metadata.update(
            {
                "loaded_representation": representation,
                "import_cleanup": cleanup_action,
                "source_vertices": str(topology.vertices),
                "source_edges": str(topology.edges),
                "source_faces": str(topology.faces),
                **_construction_curve_metadata(options, representation),
            }
        )
        return node
    cleanup.record_loaded(representation)

    part_entry = _label_entry(shape_label)
    color = _label_color(label) or _label_color(shape_label) or (0.75, 0.75, 0.75, 1.0)
    base_spec = (
        _label_visual_material_spec(vis_material_tool, label, options)
        or _label_visual_material_spec(vis_material_tool, shape_label, options)
        or _color_material_spec(color)
    )
    material_id = _material_id_from_spec(base_spec)
    face_material_ids, face_material_specs = _face_material_ids(
        shape_tool,
        color_tool,
        vis_material_tool,
        shape_label,
        shape,
        base_material_id=material_id,
        options=options,
    )
    material_ids, face_material_indices = _material_binding_plan(material_id, face_material_ids)
    material_signature = "|".join(material_ids)
    if any(index != 0 for index in face_material_indices):
        material_signature = f"{material_signature}:{','.join(str(index) for index in face_material_indices)}"
    shape_hash = _shape_fingerprint(shape)
    part_id, is_new_part = _canonical_part_id(
        source_identity=source_identity,
        part_entry=part_entry,
        shape_hash=shape_hash,
        material_signature=material_signature,
        part_index=part_index,
    )
    node.part_id = part_id
    if is_new_part:
        _ensure_material(materials, material_id, base_spec)
        for face_material_id, face_spec in face_material_specs.items():
            _ensure_material(materials, face_material_id, face_spec)
        metadata: Metadata = {
            "step_label": part_entry,
            "occurrence_label": label_entry,
            "source_identity": source_identity,
            "source_name": _label_name(shape_label) or "",
            "shape_fingerprint": shape_hash,
            "loaded_representation": representation,
            "source_vertices": str(topology.vertices),
            "source_edges": str(topology.edges),
            "source_faces": str(topology.faces),
            **_construction_curve_metadata(options, representation),
        }
        if mixed_construction_counts is not None:
            metadata.update(
                _mixed_construction_curve_metadata(
                    options,
                    "deleted" if _construction_curve_policy(options) == "delete" else "split",
                    mixed_construction_counts,
                )
            )
        if any(index != 0 for index in face_material_indices):
            metadata["occt_face_material_indices"] = ",".join(str(index) for index in face_material_indices)
        parts[part_id] = Part(
            id=part_id,
            name=_label_name(shape_label) or _label_name(label) or f"Part {part_entry}",
            source_shape=shape,
            material_ids=material_ids,
            metadata=metadata,
            fingerprint=shape_hash,
        )
    if mixed_construction_shape is not None and mixed_construction_counts is not None:
        if _construction_curve_policy(options) == "delete":
            cleanup.record_deleted("delete_lines", mixed_construction_counts)
        else:
            curve_node = _build_mixed_construction_curve_node(
                source_identity=source_identity,
                occurrence_path=occurrence_path,
                label_entry=label_entry,
                part_entry=part_entry,
                source_name=_label_name(shape_label) or _label_name(label) or f"Part {part_entry}",
                shape=mixed_construction_shape,
                counts=mixed_construction_counts,
                material_ids=material_ids,
                part_index=part_index,
                parts=parts,
                options=options,
                cleanup=cleanup,
            )
            node.children.append(curve_node)
    return node


def _mixed_construction_curve_shape(shape: Any, counts: _ShapeTopologyCounts) -> Any | None:
    if counts.faces == 0 or counts.edges == 0:
        return None
    try:
        from OCP.BRep import BRep_Builder
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS, TopoDS_Compound
    except ImportError:
        return None

    face_edges: list[Any] = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = face_explorer.Current()
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            face_edges.append(TopoDS.Edge_s(edge_explorer.Current()))
            edge_explorer.Next()
        face_explorer.Next()
    if not face_edges:
        return None

    free_edges: list[Any] = []
    edge_explorer = TopExp_Explorer(shape, TopAbs_EDGE)
    while edge_explorer.More():
        edge = TopoDS.Edge_s(edge_explorer.Current())
        if not any(edge.IsSame(face_edge) for face_edge in face_edges) and not any(
            edge.IsSame(existing) for existing in free_edges
        ):
            free_edges.append(edge)
        edge_explorer.Next()
    if not free_edges:
        return None

    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    for edge in free_edges:
        builder.Add(compound, edge)
    return compound


def _build_mixed_construction_curve_node(
    *,
    source_identity: str,
    occurrence_path: str,
    label_entry: str,
    part_entry: str,
    source_name: str,
    shape: Any,
    counts: _ShapeTopologyCounts,
    material_ids: list[str],
    part_index: _PartIndex,
    parts: dict[str, Part],
    options: StepReadOptions,
    cleanup: _ImportCleanupStats,
) -> Node:
    cleanup.record_loaded("construction_lines")
    curve_entry = f"{part_entry}:construction_curves"
    shape_hash = _shape_fingerprint(shape)
    material_signature = "|".join(material_ids)
    part_id, is_new_part = _canonical_part_id(
        source_identity=source_identity,
        part_entry=curve_entry,
        shape_hash=shape_hash,
        material_signature=f"{material_signature}:construction_curves",
        part_index=part_index,
    )
    if is_new_part:
        metadata: Metadata = {
            "step_label": curve_entry,
            "occurrence_label": label_entry,
            "source_identity": source_identity,
            "source_name": f"{source_name} construction curves",
            "shape_fingerprint": shape_hash,
            "loaded_representation": "construction_lines",
            "source_vertices": str(counts.vertices),
            "source_edges": str(counts.edges),
            "source_faces": str(counts.faces),
            "mixed_construction_curve_split": "true",
            **_construction_curve_metadata(options, "construction_lines"),
        }
        parts[part_id] = Part(
            id=part_id,
            name=f"{source_name} Construction Curves",
            source_shape=shape,
            material_ids=list(material_ids),
            metadata=metadata,
            fingerprint=shape_hash,
        )
    return Node(
        id=_stable_id("node", f"{source_identity}:{occurrence_path}:construction_curves"),
        name=f"{source_name} Construction Curves",
        part_id=part_id,
        metadata={
            "step_label": curve_entry,
            "occurrence_label": label_entry,
            "loaded_representation": "construction_lines",
            "mixed_construction_curve_split": "true",
            "source_vertices": str(counts.vertices),
            "source_edges": str(counts.edges),
            "source_faces": str(counts.faces),
            **_construction_curve_metadata(options, "construction_lines"),
        },
    )


def _mixed_construction_curve_metadata(
    options: StepReadOptions,
    action: str,
    counts: _ShapeTopologyCounts,
) -> dict[str, str]:
    metadata = {
        "mixed_construction_curve_policy": _construction_curve_policy(options),
        "mixed_construction_curve_action": action,
        "mixed_construction_curve_vertices": str(counts.vertices),
        "mixed_construction_curve_edges": str(counts.edges),
    }
    if action == "split":
        metadata["mixed_construction_curve_split"] = "true"
    return metadata


def _canonical_part_id(
    *,
    source_identity: str,
    part_entry: str,
    shape_hash: str,
    material_signature: str,
    part_index: _PartIndex,
) -> tuple[str, bool]:
    label_key = ("label", source_identity, part_entry, material_signature)
    existing = part_index.get(label_key)
    if existing is not None:
        return existing, False

    shape_key = ("shape", source_identity, shape_hash, material_signature)
    existing = part_index.get(shape_key)
    if existing is not None:
        part_index[label_key] = existing
        return existing, False

    part_id = _stable_id("part", f"{source_identity}:{part_entry}")
    part_index[label_key] = part_id
    part_index[shape_key] = part_id
    return part_id, True


def _material_binding_plan(base_material_id: str, face_material_ids: list[str]) -> tuple[list[str], list[int]]:
    material_ids = [base_material_id]
    material_indices: list[int] = []
    for face_material_id in face_material_ids:
        if face_material_id not in material_ids:
            material_ids.append(face_material_id)
        material_indices.append(material_ids.index(face_material_id))
    return material_ids, material_indices


def _ensure_material(
    materials: dict[str, Material],
    material_id: str,
    spec: _CadMaterialSpec,
) -> None:
    if material_id not in materials:
        materials[material_id] = Material(
            id=material_id,
            name=spec.name or f"CAD material {material_id[-8:]}",
            base_color=spec.base_color,
            metallic=spec.metallic,
            roughness=spec.roughness,
            opacity=spec.opacity,
            metadata=spec.metadata_dict(),
        )


def _color_material_spec(color: tuple[float, float, float, float]) -> _CadMaterialSpec:
    color = _clamp_color(color)
    return _CadMaterialSpec(
        name=f"CAD color {_material_id(color)[-8:]}",
        base_color=color,
        opacity=color[3],
        metadata=(("cad_material_source", "color"), ("pbr_mapping_status", "color_only")),
    )


def _material_id_from_spec(spec: _CadMaterialSpec) -> str:
    metadata = spec.metadata_dict()
    if metadata.get("cad_material_source") == "color":
        return _material_id(spec.base_color)
    encoded = ",".join(
        [
            spec.name,
            *(f"{component:.6f}" for component in spec.base_color),
            f"{spec.metallic:.6f}",
            f"{spec.roughness:.6f}",
            f"{spec.opacity:.6f}",
            str(metadata.get("cad_material_source", "")),
            str(metadata.get("cad_material_mapping_rule", "")),
        ]
    )
    return _stable_id("mat", encoded)


def _label_visual_material_spec(
    vis_material_tool: Any,
    label: Any,
    options: StepReadOptions,
) -> _CadMaterialSpec | None:
    if vis_material_tool is None:
        return None
    try:
        if not vis_material_tool.IsSetShapeMaterial(label):
            return None
        material = vis_material_tool.GetShapeMaterial_s(label)
    except Exception:
        return None
    spec = _visual_material_spec(material)
    return _apply_material_library_mapping(spec, options) if spec is not None else None


def _shape_visual_material_spec(
    vis_material_tool: Any,
    shape: Any,
    options: StepReadOptions,
) -> _CadMaterialSpec | None:
    if vis_material_tool is None:
        return None
    try:
        if not vis_material_tool.IsSetShapeMaterial(shape):
            return None
        material = vis_material_tool.GetShapeMaterial(shape)
    except Exception:
        return None
    spec = _visual_material_spec(material)
    return _apply_material_library_mapping(spec, options) if spec is not None else None


def _visual_material_spec(material: Any) -> _CadMaterialSpec | None:
    if material is None:
        return None
    try:
        if material.IsEmpty():
            return None
    except Exception:
        return None

    name = _ocp_string(material.RawName()) or "CAD visual material"
    metadata: Metadata = {"cad_material_source": "xde_visual_material", "cad_material_name": name}
    try:
        if material.HasPbrMaterial() and material.PbrMaterial().IsDefined():
            pbr = material.PbrMaterial()
            color = _quantity_rgba_tuple(pbr.BaseColor())
            metallic = _clamp01(float(pbr.Metallic()))
            roughness = max(0.04, _clamp01(float(pbr.Roughness())))
            metadata.update(
                {
                    "cad_visual_material_model": "pbr",
                    "pbr_mapping_status": "source_pbr",
                    "cad_material_metallic": f"{metallic:g}",
                    "cad_material_roughness": f"{roughness:g}",
                }
            )
            return _CadMaterialSpec(
                name=name,
                base_color=color,
                metallic=metallic,
                roughness=roughness,
                opacity=color[3],
                metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
            )
    except Exception:
        pass

    try:
        if material.HasCommonMaterial() and material.CommonMaterial().IsDefined():
            common = material.CommonMaterial()
            color3 = _quantity_color_tuple(common.DiffuseColor())
            opacity = 1.0 - _clamp01(float(common.Transparency()))
            shininess = _clamp01(float(common.Shininess()))
            roughness = max(0.04, 1.0 - shininess**0.5)
            color = (color3[0], color3[1], color3[2], opacity)
            metadata.update(
                {
                    "cad_visual_material_model": "common",
                    "pbr_mapping_status": "common_to_pbr",
                    "cad_material_shininess": f"{shininess:g}",
                    "cad_material_transparency": f"{1.0 - opacity:g}",
                }
            )
            return _CadMaterialSpec(
                name=name,
                base_color=color,
                metallic=0.0,
                roughness=roughness,
                opacity=opacity,
                metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
            )
    except Exception:
        pass

    try:
        color = _quantity_rgba_tuple(material.BaseColor())
    except Exception:
        return None
    metadata.update({"cad_visual_material_model": "base_color", "pbr_mapping_status": "base_color_to_pbr"})
    return _CadMaterialSpec(
        name=name,
        base_color=color,
        opacity=color[3],
        metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
    )


def _apply_material_library_mapping(spec: _CadMaterialSpec, options: StepReadOptions) -> _CadMaterialSpec:
    if not options.material_library_mapping:
        metadata = {**spec.metadata_dict(), "pbr_mapping_status": "source_only"}
        return _replace_material_spec(spec, metadata=metadata)
    tokens = set(_name_tokens(spec.name))
    if not tokens:
        return spec
    for rule in _MATERIAL_LIBRARY_RULES:
        matched = sorted(tokens.intersection(rule.tokens))
        if not matched:
            continue
        color = spec.base_color
        if rule.base_color is not None and _is_default_material_color(spec.base_color):
            color = rule.base_color
        opacity = spec.opacity if rule.opacity is None else min(spec.opacity, rule.opacity)
        color = (color[0], color[1], color[2], min(color[3], opacity))
        metadata = {
            **spec.metadata_dict(),
            "pbr_mapping_status": "library_rule",
            "cad_material_mapping_rule": matched[0],
        }
        return _CadMaterialSpec(
            name=spec.name,
            base_color=color,
            metallic=spec.metallic if rule.metallic is None else rule.metallic,
            roughness=spec.roughness if rule.roughness is None else rule.roughness,
            opacity=opacity,
            metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
        )
    metadata = {**spec.metadata_dict(), "pbr_mapping_status": spec.metadata_dict().get("pbr_mapping_status", "no_rule")}
    return _replace_material_spec(spec, metadata=metadata)


def _replace_material_spec(spec: _CadMaterialSpec, *, metadata: Metadata) -> _CadMaterialSpec:
    return _CadMaterialSpec(
        name=spec.name,
        base_color=spec.base_color,
        metallic=spec.metallic,
        roughness=spec.roughness,
        opacity=spec.opacity,
        metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
    )


def _shape_definition_label(label: Any) -> Any:
    from OCP.TDF import TDF_Label
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    if XCAFDoc_ShapeTool.IsReference_s(label):
        referred = TDF_Label()
        if XCAFDoc_ShapeTool.GetReferredShape_s(label, referred):
            return referred
    return label


def _label_entry(label: Any) -> str:
    from OCP.TCollection import TCollection_AsciiString
    from OCP.TDF import TDF_Tool

    value = TCollection_AsciiString()
    TDF_Tool.Entry_s(label, value)
    return str(value.ToCString())


def _label_name(label: Any) -> str | None:
    from OCP.TDataStd import TDataStd_Name

    attribute = TDataStd_Name()
    if not label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
        return None
    value = str(attribute.Get().ToExtString()).strip()
    return value or None


def _label_color(label: Any) -> tuple[float, float, float, float] | None:
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf, XCAFDoc_ColorTool

    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        color = Quantity_Color()
        if XCAFDoc_ColorTool.GetColor_s(label, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
    return None


def _face_material_ids(
    shape_tool: Any,
    color_tool: Any,
    vis_material_tool: Any,
    shape_label: Any,
    shape: Any,
    *,
    base_material_id: str,
    options: StepReadOptions,
) -> tuple[list[str], dict[str, _CadMaterialSpec]]:
    from OCP.TDF import TDF_Label
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    material_ids: list[str] = []
    specs: dict[str, _CadMaterialSpec] = {}
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        spec = _shape_visual_material_spec(vis_material_tool, face, options)
        sub_label = TDF_Label()
        found_sub_label = shape_tool.FindSubShape(shape_label, face, sub_label)
        if spec is None and found_sub_label:
            spec = _label_visual_material_spec(vis_material_tool, sub_label, options)
        color = _shape_color(color_tool, face)
        if spec is None and color is None and found_sub_label:
            color = _label_color(sub_label)
        if spec is None and color is not None:
            spec = _color_material_spec(color)
        if spec is None:
            material_ids.append(base_material_id)
        else:
            material_id = _material_id_from_spec(spec)
            material_ids.append(material_id)
            specs[material_id] = spec
        explorer.Next()
    return material_ids, specs


def _shape_color(color_tool: Any, shape: Any) -> tuple[float, float, float, float] | None:
    from OCP.Quantity import Quantity_Color
    from OCP.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_ColorSurf

    for color_type in (XCAFDoc_ColorSurf, XCAFDoc_ColorGen):
        color = Quantity_Color()
        if color_tool.GetColor(shape, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
        if color_tool.GetInstanceColor(shape, color_type, color):
            return (float(color.Red()), float(color.Green()), float(color.Blue()), 1.0)
    return None


def _empty_material_library_summary() -> dict[str, int]:
    return {
        "references": 0,
        "resolved": 0,
        "missing": 0,
        "unsupported": 0,
        "unreadable": 0,
        "materials": 0,
        "textures": 0,
        "texture_missing": 0,
        "texture_unreadable": 0,
    }


def _empty_material_library_binding_summary() -> dict[str, int]:
    return {
        "library_materials": 0,
        "matched_library_materials": 0,
        "unmatched_library_materials": 0,
        "applied_materials": 0,
        "bound_textures": 0,
    }


def _material_library_import_state(
    options: StepReadOptions,
    summary: dict[str, int],
    binding_summary: dict[str, int],
) -> str:
    if not options.material_library_mapping:
        return "disabled"
    if summary["references"] and summary["resolved"] == 0:
        return "missing_sources"
    if summary["unreadable"] and summary["materials"] == 0:
        return "unsupported"
    if summary["materials"] and binding_summary["applied_materials"] == 0:
        return "approximated"
    return "honored"


def _extract_material_libraries(
    source: Path,
    source_identity: str,
    options: StepReadOptions,
) -> _MaterialLibraryExtraction:
    if not options.material_library_mapping:
        return _MaterialLibraryExtraction(
            materials=[],
            images={},
            summary=_empty_material_library_summary(),
            warnings=[],
        )

    references = _material_library_references(source)
    configured_paths = [Path(path) for path in options.material_library_paths]
    search_roots = [
        source.parent,
        *(Path(path) for path in options.source_texture_search_paths),
        *(path for path in configured_paths if path.is_dir()),
    ]
    candidates: list[tuple[str, Path | None]] = []
    missing = 0
    unsupported = 0
    unreadable = 0
    texture_missing = 0
    texture_unreadable = 0
    warnings: list[str] = []

    for reference in references:
        library_path = _resolve_material_library_reference(reference, search_roots)
        if library_path is None:
            missing += 1
            warnings.append(f"material library reference could not be resolved: {reference}")
            candidates.append((reference, None))
        else:
            candidates.append((reference, library_path))

    for configured in configured_paths:
        explicit_candidates = _material_library_path_candidates(configured)
        if not explicit_candidates:
            missing += 1
            warnings.append(f"material library path could not be resolved: {configured}")
            candidates.append((str(configured), None))
            continue
        candidates.extend((str(configured), path) for path in explicit_candidates)

    materials: list[_MaterialLibrarySpec] = []
    images: dict[str, ImageResource] = {}
    seen_libraries: set[Path] = set()
    seen_texture_paths: set[str] = set()
    resolved_libraries = 0
    for reference, library_path in candidates:
        if library_path is None:
            continue
        suffix = library_path.suffix.lower()
        if suffix not in _MATERIAL_LIBRARY_SUFFIXES:
            unsupported += 1
            warnings.append(f"material library format is unsupported: {library_path}")
            continue
        resolved = library_path.resolve()
        if resolved in seen_libraries:
            continue
        seen_libraries.add(resolved)
        try:
            specs, texture_stats = _load_material_library(
                library_path,
                source_identity=source_identity,
                reference=reference,
                images=images,
                seen_texture_paths=seen_texture_paths,
                search_roots=[library_path.parent, *search_roots],
            )
        except ValueError as exc:
            unreadable += 1
            warnings.append(str(exc))
            continue
        resolved_libraries += 1
        materials.extend(specs)
        texture_missing += texture_stats["missing"]
        texture_unreadable += texture_stats["unreadable"]

    summary = _empty_material_library_summary()
    summary.update(
        {
            "references": len(references) + len(configured_paths),
            "resolved": resolved_libraries,
            "missing": missing,
            "unsupported": unsupported,
            "unreadable": unreadable,
            "materials": len(materials),
            "textures": len(images),
            "texture_missing": texture_missing,
            "texture_unreadable": texture_unreadable,
        }
    )
    return _MaterialLibraryExtraction(materials=materials, images=images, summary=summary, warnings=warnings)


def _material_library_references(source: Path) -> list[str]:
    try:
        text = source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    references: list[str] = []
    seen: set[str] = set()
    for match in _MATERIAL_LIBRARY_REF_RE.finditer(text):
        reference = _clean_source_texture_reference(match.group(1))
        if not reference or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    return references


def _resolve_material_library_reference(reference: str, search_roots: list[Path]) -> Path | None:
    candidate = Path(reference)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        for root in search_roots:
            candidates.append(root / candidate)
            if candidate.name != str(candidate):
                candidates.append(root / candidate.name)
    for item in candidates:
        try:
            if item.is_file() and item.suffix.lower() in _MATERIAL_LIBRARY_SUFFIXES:
                return item
        except OSError:
            continue
    return None


def _material_library_path_candidates(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _MATERIAL_LIBRARY_SUFFIXES:
        return [path]
    if path.is_dir():
        try:
            return sorted(
                item for item in path.iterdir() if item.is_file() and item.suffix.lower() in _MATERIAL_LIBRARY_SUFFIXES
            )
        except OSError:
            return []
    return []


def _load_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    if path.suffix.lower() == ".json":
        return _load_json_material_library(
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
        )
    if path.suffix.lower() == ".mtl":
        return _load_mtl_material_library(
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
        )
    if path.suffix.lower() == ".zip":
        return _load_zipped_material_library(
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
        )
    raise ValueError(f"material library format is unsupported: {path}")


def _load_json_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"material library could not be read: {path}") from exc
    entries = _json_material_entries(payload)
    if not entries:
        raise ValueError(f"material library did not contain supported material records: {path}")
    specs: list[_MaterialLibrarySpec] = []
    texture_stats = {"missing": 0, "unreadable": 0}
    for entry in entries:
        spec, stats = _json_material_spec(
            entry,
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            library_label=str(path),
        )
        texture_stats["missing"] += stats["missing"]
        texture_stats["unreadable"] += stats["unreadable"]
        if spec is not None:
            specs.append(spec)
    if not specs:
        raise ValueError(f"material library did not contain supported material records: {path}")
    return specs, texture_stats


def _load_zipped_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    _ = search_roots
    try:
        with zipfile.ZipFile(path) as archive:
            material_members = _archive_material_members(archive)
            if not material_members:
                raise ValueError(f"material library archive did not contain JSON or MTL records: {path}")
            archive_textures = _archive_texture_members(archive)
            specs: list[_MaterialLibrarySpec] = []
            texture_stats = {"missing": 0, "unreadable": 0}
            member_errors: list[str] = []
            for member_name in material_members:
                member_path = Path(member_name)
                library_label = f"{path}!/{member_name}"
                try:
                    member_bytes = archive.read(member_name)
                except (KeyError, OSError, zipfile.BadZipFile):
                    member_errors.append(f"material library archive member could not be read: {library_label}")
                    continue
                try:
                    if member_path.suffix.lower() == ".json":
                        try:
                            payload = json.loads(member_bytes.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ValueError(f"material library could not be read: {library_label}") from exc
                        entries = _json_material_entries(payload)
                        if not entries:
                            raise ValueError(
                                f"material library did not contain supported material records: {library_label}"
                            )
                    else:
                        lines = member_bytes.decode("utf-8", errors="ignore").splitlines()
                        entries = _mtl_material_entries(lines, library_label)
                    member_specs, member_texture_stats = _material_specs_from_entries(
                        entries,
                        member_path,
                        source_identity=source_identity,
                        reference=reference,
                        images=images,
                        seen_texture_paths=seen_texture_paths,
                        search_roots=[],
                        library_label=library_label,
                        archive_textures=archive_textures,
                        archive_container=path,
                    )
                except ValueError as exc:
                    member_errors.append(str(exc))
                    continue
                specs.extend(member_specs)
                texture_stats["missing"] += member_texture_stats["missing"]
                texture_stats["unreadable"] += member_texture_stats["unreadable"]
            if not specs:
                detail = f": {member_errors[0]}" if member_errors else ""
                raise ValueError(f"material library archive did not contain supported material records: {path}{detail}")
            return specs, texture_stats
    except zipfile.BadZipFile as exc:
        raise ValueError(f"material library archive could not be read: {path}") from exc
    except OSError as exc:
        raise ValueError(f"material library archive could not be read: {path}") from exc


def _archive_material_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if _safe_archive_member_name(name) and PurePosixPath(name).suffix.lower() in _MATERIAL_RECORD_SUFFIXES
    )


def _archive_texture_members(archive: zipfile.ZipFile) -> _ArchiveTextureMap:
    textures: _ArchiveTextureMap = {}
    for name in archive.namelist():
        if not _safe_archive_member_name(name) or PurePosixPath(name).suffix.lower() not in _SOURCE_TEXTURE_SUFFIXES:
            continue
        try:
            textures[_archive_member_key(name)] = (_archive_member_name(name), archive.read(name))
        except (KeyError, OSError, zipfile.BadZipFile):
            continue
    return textures


def _safe_archive_member_name(name: str) -> bool:
    cleaned = name.replace("\\", "/")
    member = PurePosixPath(cleaned)
    return bool(cleaned and member.name and not member.is_absolute() and ".." not in member.parts)


def _archive_member_name(name: str) -> str:
    return str(PurePosixPath(name.replace("\\", "/"))).lstrip("./")


def _archive_member_key(name: str) -> str:
    member = PurePosixPath(name.replace("\\", "/"))
    parts: list[str] = []
    for part in member.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts).lower()


def _json_material_entries(payload: object) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [cast(dict[str, object], item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("materials", "materialLibrary", "material_library", "library", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return _json_material_entries(value)
    entries: list[dict[str, object]] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        entry = dict(cast(dict[str, object], value))
        entry.setdefault("name", key)
        entries.append(entry)
    return entries


def _json_material_spec(
    entry: dict[str, object],
    library_path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
) -> tuple[_MaterialLibrarySpec | None, dict[str, int]]:
    name = _json_material_name(entry)
    if not name:
        return None, {"missing": 0, "unreadable": 0}
    pbr = _json_mapping(entry.get("pbrMetallicRoughness")) or _json_mapping(entry.get("pbr_metallic_roughness")) or {}
    base_color = (
        _json_color(entry.get("base_color"))
        or _json_color(entry.get("baseColor"))
        or _json_color(entry.get("baseColorFactor"))
        or _json_color(entry.get("base_color_factor"))
        or _json_color(entry.get("diffuseColor"))
        or _json_color(entry.get("diffuse"))
        or _json_color(entry.get("albedo"))
        or _json_color(entry.get("color"))
        or _json_color(pbr.get("baseColorFactor"))
    )
    metallic = _optional_material_float(
        entry.get("metallic"),
        entry.get("metallicFactor"),
        entry.get("metalness"),
        entry.get("metalnessFactor"),
        pbr.get("metallicFactor"),
    )
    roughness = _optional_material_float(
        entry.get("roughness"), entry.get("roughnessFactor"), pbr.get("roughnessFactor")
    )
    opacity = _json_opacity(entry)
    texture_images, texture_stats = _json_material_texture_images(
        entry,
        pbr,
        library_path,
        source_identity=source_identity,
        images=images,
        seen_texture_paths=seen_texture_paths,
        search_roots=search_roots,
        material_name=name,
        library_label=library_label,
        archive_textures=archive_textures,
        archive_container=archive_container,
    )
    metadata: Metadata = {
        "cad_material_source": "material_library",
        "material_library_name": name,
        "material_library_reference": reference,
        "material_library_path": library_label,
        "pbr_mapping_status": "material_library",
    }
    if archive_container is not None:
        metadata["material_library_container"] = str(archive_container)
    return (
        _MaterialLibrarySpec(
            name=name,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            opacity=opacity,
            texture_images=tuple(texture_images),
            metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
        ),
        texture_stats,
    )


def _json_material_name(entry: dict[str, object]) -> str:
    for key in ("name", "materialName", "material_name", "displayName", "display_name", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _json_mapping(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _json_color(value: object) -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("#"):
            return _hex_color(stripped)
        parts = [part for part in re.split(r"[\s,]+", stripped) if part]
        if len(parts) in {3, 4}:
            return _numeric_color(parts)
        return None
    if isinstance(value, dict):
        red = _optional_material_float(value.get("r"), value.get("red"))
        green = _optional_material_float(value.get("g"), value.get("green"))
        blue = _optional_material_float(value.get("b"), value.get("blue"))
        alpha = _optional_material_float(value.get("a"), value.get("alpha"))
        if red is None or green is None or blue is None:
            return None
        color = (red, green, blue, 1.0 if alpha is None else alpha)
        return _normalize_color_range(color)
    if isinstance(value, list | tuple) and len(value) in {3, 4}:
        return _numeric_color(list(value))
    return None


def _hex_color(value: str) -> tuple[float, float, float, float] | None:
    encoded = value.strip().removeprefix("#")
    if len(encoded) not in {6, 8}:
        return None
    try:
        red = int(encoded[0:2], 16) / 255.0
        green = int(encoded[2:4], 16) / 255.0
        blue = int(encoded[4:6], 16) / 255.0
        alpha = int(encoded[6:8], 16) / 255.0 if len(encoded) == 8 else 1.0
    except ValueError:
        return None
    return (_clamp01(red), _clamp01(green), _clamp01(blue), _clamp01(alpha))


def _numeric_color(values: list[object]) -> tuple[float, float, float, float] | None:
    parsed = [_optional_material_float(value) for value in values]
    if any(value is None for value in parsed):
        return None
    numbers = [cast(float, value) for value in parsed]
    red, green, blue = numbers[:3]
    alpha = numbers[3] if len(numbers) == 4 else 1.0
    return _normalize_color_range((red, green, blue, alpha))


def _normalize_color_range(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if any(component > 1.0 for component in color):
        return (
            _clamp01(color[0] / 255.0),
            _clamp01(color[1] / 255.0),
            _clamp01(color[2] / 255.0),
            _clamp01(color[3] / 255.0),
        )
    return _clamp_color(color)


def _optional_material_float(*values: object) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, str | int | float):
            continue
        try:
            return _clamp01(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _json_opacity(entry: dict[str, object]) -> float | None:
    opacity = _optional_material_float(entry.get("opacity"), entry.get("alpha"))
    if opacity is not None:
        return opacity
    transparency = _optional_material_float(entry.get("transparency"))
    if transparency is not None:
        return 1.0 - transparency
    base_color = _json_color(entry.get("baseColorFactor")) or _json_color(entry.get("base_color"))
    if base_color is not None:
        return base_color[3]
    return None


def _json_material_texture_images(
    entry: dict[str, object],
    pbr: dict[str, object],
    library_path: Path,
    *,
    source_identity: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    material_name: str,
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    references: list[tuple[str, str]] = []
    texture_map = _json_mapping(entry.get("textures"))
    if texture_map is not None:
        for key, value in texture_map.items():
            slot = _material_library_texture_slot(key)
            reference = _texture_reference_from_value(value)
            if slot is not None and reference:
                references.append((slot, reference))
    flat_keys = {
        "baseColorTexture": "base_color",
        "base_color_texture": "base_color",
        "diffuseTexture": "base_color",
        "albedoTexture": "base_color",
        "metallicRoughnessTexture": "metallic_roughness",
        "metallic_roughness_texture": "metallic_roughness",
        "normalTexture": "normal",
        "normal_texture": "normal",
        "occlusionTexture": "occlusion",
        "aoTexture": "occlusion",
        "emissiveTexture": "emissive",
        "opacityTexture": "opacity",
        "alphaTexture": "opacity",
        "roughnessTexture": "roughness",
        "metallicTexture": "metallic",
    }
    for key, slot in flat_keys.items():
        reference = _texture_reference_from_value(entry.get(key))
        if reference:
            references.append((slot, reference))
    for key, slot in {"baseColorTexture": "base_color", "metallicRoughnessTexture": "metallic_roughness"}.items():
        reference = _texture_reference_from_value(pbr.get(key))
        if reference:
            references.append((slot, reference))
    return _load_material_library_texture_references(
        references,
        library_path,
        source_identity=source_identity,
        images=images,
        seen_texture_paths=seen_texture_paths,
        search_roots=search_roots,
        material_name=material_name,
        library_label=library_label,
        archive_textures=archive_textures,
        archive_container=archive_container,
    )


def _texture_reference_from_value(value: object) -> str:
    if isinstance(value, str):
        return _clean_source_texture_reference(value)
    if isinstance(value, dict):
        for key in ("uri", "path", "file", "filename", "source", "name"):
            reference = value.get(key)
            if isinstance(reference, str) and reference.strip():
                return _clean_source_texture_reference(reference)
    return ""


def _load_mtl_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise ValueError(f"material library could not be read: {path}") from exc
    entries = _mtl_material_entries(lines, str(path))
    specs, texture_stats = _material_specs_from_entries(
        entries,
        path,
        source_identity=source_identity,
        reference=reference,
        images=images,
        seen_texture_paths=seen_texture_paths,
        search_roots=search_roots,
        library_label=str(path),
    )
    return specs, texture_stats


def _mtl_material_entries(lines: list[str], library_label: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        keyword, _, rest = line.partition(" ")
        key = keyword.lower()
        value = rest.strip()
        if key == "newmtl":
            if current is not None:
                entries.append(current)
            current = {"name": value}
            continue
        if current is None:
            continue
        if key == "kd":
            current["base_color"] = value
        elif key == "d":
            current["opacity"] = value
        elif key == "tr":
            transparency = _optional_material_float(value)
            if transparency is not None:
                current["opacity"] = 1.0 - transparency
        elif key == "pm":
            current["metallic"] = value
        elif key == "pr":
            current["roughness"] = value
        elif key == "ns":
            try:
                shininess = _clamp01(float(value) / 1000.0)
            except ValueError:
                shininess = None
            if shininess is not None:
                current["roughness"] = max(0.04, 1.0 - shininess**0.5)
        elif key.startswith("map_") or key in {"bump", "norm"}:
            textures = cast(dict[str, object], current.setdefault("textures", {}))
            slot = _material_library_texture_slot(key)
            if slot is not None:
                textures[slot] = _mtl_texture_reference(value)
    if current is not None:
        entries.append(current)
    if not entries:
        raise ValueError(f"material library did not contain supported material records: {library_label}")
    return entries


def _material_specs_from_entries(
    entries: list[dict[str, object]],
    library_path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    specs: list[_MaterialLibrarySpec] = []
    texture_stats = {"missing": 0, "unreadable": 0}
    for entry in entries:
        spec, stats = _json_material_spec(
            entry,
            library_path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            library_label=library_label,
            archive_textures=archive_textures,
            archive_container=archive_container,
        )
        texture_stats["missing"] += stats["missing"]
        texture_stats["unreadable"] += stats["unreadable"]
        if spec is not None:
            specs.append(spec)
    return specs, texture_stats


def _mtl_texture_reference(value: str) -> str:
    tokens = [token for token in value.split() if token]
    if not tokens:
        return ""
    return _clean_source_texture_reference(tokens[-1])


def _material_library_texture_slot(key: str) -> str | None:
    normalized = key.lower().replace("-", "_")
    aliases = {
        "map_kd": "base_color",
        "map_ka": "base_color",
        "map_ke": "emissive",
        "map_d": "opacity",
        "map_bump": "normal",
        "bump": "normal",
        "norm": "normal",
        "map_pr": "roughness",
        "map_pm": "metallic",
        "map_orm": "metallic_roughness",
    }
    if normalized in aliases:
        return aliases[normalized]
    return _texture_slot(normalized)


def _resolve_archive_texture_reference(
    reference: str,
    archive_textures: _ArchiveTextureMap,
    library_path: Path,
) -> tuple[str, bytes] | None:
    library_parent = PurePosixPath(library_path.as_posix()).parent
    reference_path = PurePosixPath(reference.replace("\\", "/"))
    candidate_keys = [
        _archive_member_key(str(library_parent / reference_path)),
        _archive_member_key(str(reference_path)),
        _archive_member_key(reference_path.name),
    ]
    for key in candidate_keys:
        if key in archive_textures:
            return archive_textures[key]
    basename = reference_path.name.lower()
    basename_matches = [item for key, item in archive_textures.items() if PurePosixPath(key).name == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def _load_material_library_texture_references(
    references: list[tuple[str, str]],
    library_path: Path,
    *,
    source_identity: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    material_name: str,
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    texture_images: list[tuple[str, str]] = []
    missing = 0
    unreadable = 0
    seen_references: set[tuple[str, str]] = set()
    for slot, reference in references:
        key = (slot, reference)
        if key in seen_references:
            continue
        seen_references.add(key)
        if archive_textures is not None and archive_container is not None:
            resolved_archive = _resolve_archive_texture_reference(reference, archive_textures, library_path)
            if resolved_archive is None:
                missing += 1
                continue
            member_name, data = resolved_archive
            identity = f"{archive_container.resolve()}!/{member_name}"
            path_label = f"{archive_container}!/{member_name}"
            try:
                image = _load_source_texture_data(
                    data,
                    suffix=Path(member_name).suffix.lower(),
                    name=Path(member_name).name,
                    source_identity=source_identity,
                    reference=reference,
                    path_label=path_label,
                    stable_identity=identity,
                )
            except ValueError:
                unreadable += 1
                continue
        else:
            texture_path = _resolve_source_texture(reference, search_roots)
            if texture_path is None:
                missing += 1
                continue
            identity = str(texture_path.resolve())
            try:
                image = _load_source_texture(texture_path, source_identity=source_identity, reference=reference)
            except ValueError:
                unreadable += 1
                continue
        if identity in seen_texture_paths:
            existing = next(
                (
                    image_id
                    for image_id, existing_image in images.items()
                    if str(existing_image.metadata.get("source_texture_identity", "")) == identity
                    or str(existing_image.metadata.get("source_texture_path", "")) == identity
                ),
                None,
            )
            if existing is not None:
                texture_images.append((slot, existing))
            continue
        seen_texture_paths.add(identity)
        image_id = _stable_id("img", f"{source_identity}:{library_label}:{identity}:{slot}:{material_name}")
        metadata = dict(image.metadata)
        metadata.update(
            {
                "source_texture_slot": slot,
                "source_texture_material_name": material_name,
                "material_library_path": library_label,
                "source_texture_identity": identity,
            }
        )
        if archive_container is not None:
            metadata["material_library_container"] = str(archive_container)
        images[image_id] = ImageResource(
            id=image_id,
            name=image.name,
            mime_type=image.mime_type,
            data=image.data,
            width=image.width,
            height=image.height,
            metadata=metadata,
        )
        texture_images.append((slot, image_id))
    return texture_images, {"missing": missing, "unreadable": unreadable}


def _apply_material_libraries_to_materials(
    materials: dict[str, Material],
    extraction: _MaterialLibraryExtraction,
) -> dict[str, int]:
    summary = _empty_material_library_binding_summary()
    summary["library_materials"] = len(extraction.materials)
    applied_materials: set[str] = set()
    for spec in extraction.materials:
        targets = _material_library_targets(materials, spec)
        if not targets:
            summary["unmatched_library_materials"] += 1
            continue
        summary["matched_library_materials"] += 1
        for material in targets:
            materials[material.id] = _material_with_library_spec(material, spec)
            applied_materials.add(material.id)
            summary["bound_textures"] += len(spec.texture_images)
    summary["applied_materials"] = len(applied_materials)
    return summary


def _material_library_targets(
    materials: dict[str, Material],
    spec: _MaterialLibrarySpec,
) -> list[Material]:
    if not materials:
        return []
    spec_key = _material_match_key(spec.name)
    exact: list[Material] = []
    fuzzy: list[Material] = []
    for material in materials.values():
        names = [
            material.name,
            str(material.metadata.get("cad_material_name", "")),
            str(material.metadata.get("material_library_name", "")),
        ]
        if spec_key and any(_material_match_key(name) == spec_key for name in names if name):
            exact.append(material)
            continue
        material_tokens = set().union(*(_name_tokens(name) for name in names if name))
        spec_tokens = set(_name_tokens(spec.name)) - _GENERIC_MATERIAL_TOKENS
        if spec_tokens and material_tokens.intersection(spec_tokens):
            fuzzy.append(material)
    if exact:
        return exact
    if fuzzy:
        return fuzzy
    return []


def _material_match_key(value: str) -> str:
    return "_".join(token for token in _name_tokens(value) if token not in _GENERIC_MATERIAL_TOKENS)


def _material_with_library_spec(material: Material, spec: _MaterialLibrarySpec) -> Material:
    opacity = material.opacity if spec.opacity is None else spec.opacity
    base_color = material.base_color if spec.base_color is None else spec.base_color
    if spec.opacity is not None:
        base_color = (base_color[0], base_color[1], base_color[2], min(base_color[3], opacity))
    metadata = {
        **material.metadata,
        **spec.metadata_dict(),
        "material_library_matched": "true",
        "material_library_material_name": spec.name,
    }
    for slot, image_id in spec.texture_images:
        metadata[f"source_texture_{slot}_image"] = image_id
        if slot in _SOURCE_TEXTURE_EXPORT_SLOTS:
            metadata.setdefault(f"source_texture_{slot}_image", image_id)
        existing = metadata.get("source_texture_slots")
        slots = set(str(existing).split(",")) if isinstance(existing, str) and existing else set()
        slots.add(slot)
        metadata["source_texture_slots"] = ",".join(sorted(slots))
    return Material(
        id=material.id,
        name=material.name,
        base_color=base_color,
        metallic=material.metallic if spec.metallic is None else spec.metallic,
        roughness=material.roughness if spec.roughness is None else spec.roughness,
        opacity=opacity,
        metadata=metadata,
    )


def _extract_source_textures(source: Path, source_identity: str, options: StepReadOptions) -> _SourceTextureExtraction:
    if not options.source_textures:
        return _SourceTextureExtraction(
            images={},
            summary={"references": 0, "resolved": 0, "missing": 0, "unsupported": 0, "unreadable": 0},
            warnings=[],
        )

    references = _source_texture_references(source)
    search_roots = [source.parent, *(Path(path) for path in options.source_texture_search_paths)]
    images: dict[str, ImageResource] = {}
    seen_paths: set[Path] = set()
    missing = 0
    unsupported = 0
    unreadable = 0
    warnings: list[str] = []
    for reference in references:
        texture_path = _resolve_source_texture(reference, search_roots)
        if texture_path is None:
            missing += 1
            warnings.append(f"source texture reference could not be resolved: {reference}")
            continue
        suffix = texture_path.suffix.lower()
        if suffix not in _SOURCE_TEXTURE_SUFFIXES:
            unsupported += 1
            continue
        resolved = texture_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            image = _load_source_texture(texture_path, source_identity=source_identity, reference=reference)
        except ValueError as exc:
            unreadable += 1
            warnings.append(str(exc))
            continue
        images[image.id] = image
    return _SourceTextureExtraction(
        images=images,
        summary={
            "references": len(references),
            "resolved": len(images),
            "missing": missing,
            "unsupported": unsupported,
            "unreadable": unreadable,
        },
        warnings=warnings,
    )


def _source_texture_references(source: Path) -> list[str]:
    try:
        text = source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    references: list[str] = []
    seen: set[str] = set()
    for match in _SOURCE_TEXTURE_REF_RE.finditer(text):
        reference = _clean_source_texture_reference(match.group(1))
        if not reference or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    return references


def _clean_source_texture_reference(reference: str) -> str:
    value = reference.replace("''", "'").strip().strip('"<>')
    if not value:
        return ""
    if value.lower().startswith("file://"):
        parsed = urlparse(value)
        value = unquote(parsed.path)
    value = value.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
    return value.strip()


def _resolve_source_texture(reference: str, search_roots: list[Path]) -> Path | None:
    candidate = Path(reference)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        for root in search_roots:
            candidates.append(root / candidate)
            if candidate.name != str(candidate):
                candidates.append(root / candidate.name)
    for item in candidates:
        try:
            if item.is_file():
                return item
        except OSError:
            continue
    return None


def _load_source_texture(path: Path, *, source_identity: str, reference: str) -> ImageResource:
    data = path.read_bytes()
    return _load_source_texture_data(
        data,
        suffix=path.suffix.lower(),
        name=path.name,
        source_identity=source_identity,
        reference=reference,
        path_label=str(path),
        stable_identity=str(path.resolve()),
    )


def _load_source_texture_data(
    data: bytes,
    *,
    suffix: str,
    name: str,
    source_identity: str,
    reference: str,
    path_label: str,
    stable_identity: str,
) -> ImageResource:
    if suffix == ".ktx2":
        size = _ktx2_dimensions(data)
        if size is None:
            raise ValueError(f"source texture could not be read as KTX2: {path_label}")
        width, height = size
        mime_type: ImageMimeType = "image/ktx2"
    else:
        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
                mime_type = "image/png" if image.format == "PNG" else "image/jpeg"
        except Exception as exc:
            raise ValueError(f"source texture could not be read: {path_label}") from exc
    digest = _stable_id("img", f"{source_identity}:{stable_identity}:{len(data)}")
    slot = _texture_slot(Path(name).stem)
    metadata: Metadata = {
        "source_texture": "true",
        "source_texture_reference": reference,
        "source_texture_path": path_label,
        "source_texture_identity": stable_identity,
        "source_texture_slot": slot or "unknown",
    }
    return ImageResource(
        id=digest,
        name=name,
        mime_type=mime_type,
        data=data,
        width=width,
        height=height,
        metadata=metadata,
    )


def _ktx2_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 28 or data[:12] != b"\xabKTX 20\xbb\r\n\x1a\n":
        return None
    width = int.from_bytes(data[20:24], "little")
    height = int.from_bytes(data[24:28], "little")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _attach_source_textures_to_materials(
    materials: dict[str, Material],
    images: dict[str, ImageResource],
) -> dict[str, int]:
    bound_images = 0
    bound_materials: set[str] = set()
    unbound_images = 0
    for image in images.values():
        slot = str(image.metadata.get("source_texture_slot", "unknown"))
        if slot == "unknown":
            unbound_images += 1
            continue
        targets = _source_texture_material_targets(materials, image)
        if not targets:
            unbound_images += 1
            continue
        export_slot = slot if slot in _SOURCE_TEXTURE_EXPORT_SLOTS else None
        for material in targets:
            material.metadata[f"source_texture_{slot}_image"] = image.id
            material.metadata[f"source_texture_{slot}_name"] = image.name
            if export_slot is not None:
                material.metadata.setdefault(f"source_texture_{export_slot}_image", image.id)
            existing = material.metadata.get("source_texture_slots")
            slots = set(str(existing).split(",")) if isinstance(existing, str) and existing else set()
            slots.add(slot)
            material.metadata["source_texture_slots"] = ",".join(sorted(slots))
            bound_materials.add(material.id)
        bound_images += 1
    return {
        "bound_images": bound_images,
        "bound_materials": len(bound_materials),
        "unbound_images": unbound_images,
    }


def _source_texture_material_targets(materials: dict[str, Material], image: ImageResource) -> list[Material]:
    if len(materials) == 1:
        return list(materials.values())
    image_tokens = set(_name_tokens(Path(image.name).stem))
    targets: list[Material] = []
    for material in materials.values():
        material_tokens = set(_name_tokens(material.name))
        material_tokens.update(_name_tokens(str(material.metadata.get("cad_material_name", ""))))
        material_tokens.update(token for token in _name_tokens(material.id) if token not in _GENERIC_MATERIAL_TOKENS)
        if image_tokens.intersection(material_tokens - _GENERIC_MATERIAL_TOKENS):
            targets.append(material)
    return targets


def _texture_slot(name: str) -> str | None:
    normalized = "".join(_name_tokens(name))
    tokens = set(_name_tokens(name))
    for slot, aliases in _TEXTURE_SLOT_TOKENS:
        for alias in aliases:
            if alias in normalized or alias in tokens:
                return slot
    return None


def _label_transform(label: Any) -> np.ndarray:
    from OCP.XCAFDoc import XCAFDoc_ShapeTool

    location = XCAFDoc_ShapeTool.GetLocation_s(label)
    transform = location.Transformation()
    matrix = np.eye(4, dtype=np.float64)
    for row in range(1, 4):
        for column in range(1, 5):
            matrix[row - 1, column - 1] = float(transform.Value(row, column))
    return matrix


def _reader_units(reader: Any) -> tuple[str, float]:
    from OCP.TColStd import TColStd_SequenceOfAsciiString

    length_units = TColStd_SequenceOfAsciiString()
    angle_units = TColStd_SequenceOfAsciiString()
    solid_angle_units = TColStd_SequenceOfAsciiString()
    reader.Reader().FileUnits(length_units, angle_units, solid_angle_units)
    if length_units.Length() == 0:
        return "millimetre", 0.001
    unit = str(length_units.Value(length_units.Lower()).ToCString()).lower()
    return unit, _meters_per_unit(unit)


def _meters_per_unit(unit: str) -> float:
    normalized = unit.lower().replace("meter", "metre")
    if "inch" in normalized:
        return 0.0254
    if "foot" in normalized or "feet" in normalized:
        return 0.3048
    if "centimetre" in normalized:
        return 0.01
    if "millimetre" in normalized:
        return 0.001
    if "metre" in normalized:
        return 1.0
    return 0.001


def _material_id(color: tuple[float, float, float, float]) -> str:
    encoded = ",".join(f"{component:.6f}" for component in color)
    return _stable_id("mat", encoded)


def _clamp_color(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        _clamp01(float(color[0])),
        _clamp01(float(color[1])),
        _clamp01(float(color[2])),
        _clamp01(float(color[3])),
    )


def _quantity_rgba_tuple(color: Any) -> tuple[float, float, float, float]:
    rgb = color.GetRGB()
    return (
        _clamp01(float(rgb.Red())),
        _clamp01(float(rgb.Green())),
        _clamp01(float(rgb.Blue())),
        _clamp01(float(color.Alpha())),
    )


def _quantity_color_tuple(color: Any) -> tuple[float, float, float]:
    return (_clamp01(float(color.Red())), _clamp01(float(color.Green())), _clamp01(float(color.Blue())))


def _is_default_material_color(color: tuple[float, float, float, float]) -> bool:
    return np.allclose(np.asarray(color[:3], dtype=float), np.asarray((0.75, 0.75, 0.75), dtype=float), atol=1e-6)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def _ocp_string(value: Any) -> str:
    if value is None:
        return ""
    for method in ("ToExtString", "ToCString"):
        if hasattr(value, method):
            try:
                return str(getattr(value, method)()).strip()
            except Exception:
                continue
    return str(value).strip()


def _stable_id(prefix: str, value: str) -> str:
    import hashlib

    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"
