from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from fascat.asset import Node, Part
from fascat.io.step.pmi import _step_record_label, _step_record_references
from fascat.io.step.records import (
    _STEP_REFERENCE_RE,
    _iter_step_records,
    _read_step_scan_text,
    _step_number_values,
    _step_string_values,
    _StepRecord,
)
from fascat.material import Material
from fascat.options import StepReadOptions

_STEP_STRICT_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


_STEP_STRICT_INTEGER_RE = re.compile(r"[-+]?\d+")


_STEP_BOOLEAN_TOKEN_RE = re.compile(r"\.(TRUE|FALSE|T|F)\.", re.IGNORECASE)


_STEP_DESIGN_VARIANT_ENTITY_KINDS = {
    "ABS_FUNCTION": "abs_function",
    "ABSFUNCTION": "abs_function",
    "ACOS_FUNCTION": "acos_function",
    "ACOSFUNCTION": "acos_function",
    "APPLIED_EFFECTIVITY_ASSIGNMENT": "applied_effectivity_assignment",
    "APPLIEDEFFECTIVITYASSIGNMENT": "applied_effectivity_assignment",
    "APPLIED_EFFECTIVITY_CONTEXT_ASSIGNMENT": "applied_effectivity_context_assignment",
    "APPLIEDEFFECTIVITYCONTEXTASSIGNMENT": "applied_effectivity_context_assignment",
    "ASIN_FUNCTION": "asin_function",
    "ASINFUNCTION": "asin_function",
    "ATAN_FUNCTION": "atan_function",
    "ATANFUNCTION": "atan_function",
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
    "CONCAT_EXPRESSION": "concat_expression",
    "CONCATEXPRESSION": "concat_expression",
    "CONDITIONAL_CONCEPT_FEATURE": "conditional_concept_feature",
    "CONDITIONALCONCEPTFEATURE": "conditional_concept_feature",
    "CONDITIONAL_CONFIGURATION": "conditional_configuration",
    "CONDITIONALCONFIGURATION": "conditional_configuration",
    "CONDITIONAL_EFFECTIVITY": "conditional_effectivity",
    "CONFIGURED_EFFECTIVITY_ASSIGNMENT": "configured_effectivity_assignment",
    "CONFIGURED_EFFECTIVITY_CONTEXT_ASSIGNMENT": "configured_effectivity_context_assignment",
    "CONFIGUREDEFFECTIVITYCONTEXTASSIGNMENT": "configured_effectivity_context_assignment",
    "CONFIGURATIONEFFECTIVITY": "configuration_effectivity",
    "COS_FUNCTION": "cos_function",
    "COSFUNCTION": "cos_function",
    "DATED_EFFECTIVITY": "dated_effectivity",
    "DATEDEFFECTIVITY": "dated_effectivity",
    "DIV_EXPRESSION": "div_expression",
    "DIVEXPRESSION": "div_expression",
    "EFFECTIVITY": "effectivity",
    "EFFECTIVITYASSIGNMENT": "effectivity_assignment",
    "EFFECTIVITY_ASSIGNMENT": "effectivity_assignment",
    "EFFECTIVITYRELATIONSHIP": "effectivity_relationship",
    "EFFECTIVITY_RELATIONSHIP": "effectivity_relationship",
    "EQUALS_EXPRESSION": "equals_expression",
    "EQUALSEXPRESSION": "equals_expression",
    "EXP_FUNCTION": "exp_function",
    "EXPFUNCTION": "exp_function",
    "EXPRESSION_EXTENSION_NUMERIC": "expression_extension_numeric",
    "EXPRESSIONEXTENSIONNUMERIC": "expression_extension_numeric",
    "EXPRESSION_EXTENSION_STRING": "expression_extension_string",
    "EXPRESSIONEXTENSIONSTRING": "expression_extension_string",
    "FORMAT_FUNCTION": "format_function",
    "FORMATFUNCTION": "format_function",
    "INDEX_EXPRESSION": "index_expression",
    "INDEXEXPRESSION": "index_expression",
    "INT_LITERAL": "int_literal",
    "INTLITERAL": "int_literal",
    "INT_NUMERIC_VARIABLE": "int_numeric_variable",
    "INTNUMERICVARIABLE": "int_numeric_variable",
    "INT_VALUE_FUNCTION": "int_value_function",
    "INTVALUEFUNCTION": "int_value_function",
    "INTEGER_REPRESENTATION_ITEM": "integer_representation_item",
    "INTEGERREPRESENTATIONITEM": "integer_representation_item",
    "INTERVAL_EXPRESSION": "interval_expression",
    "INTERVALEXPRESSION": "interval_expression",
    "LIKE_EXPRESSION": "like_expression",
    "LIKEEXPRESSION": "like_expression",
    "LENGTH_FUNCTION": "length_function",
    "LENGTHFUNCTION": "length_function",
    "LOG10_FUNCTION": "log10_function",
    "LOG10FUNCTION": "log10_function",
    "LOG2_FUNCTION": "log2_function",
    "LOG2FUNCTION": "log2_function",
    "LOG_FUNCTION": "log_function",
    "LOGFUNCTION": "log_function",
    "LOGICAL_LITERAL": "logical_literal",
    "LOGICALLITERAL": "logical_literal",
    "LOGICAL_REPRESENTATION_ITEM": "logical_representation_item",
    "LOGICALREPRESENTATIONITEM": "logical_representation_item",
    "LOT_EFFECTIVITY": "lot_effectivity",
    "LITERAL_NUMBER": "literal_number",
    "LITERALNUMBER": "literal_number",
    "MINUS_EXPRESSION": "minus_expression",
    "MINUSEXPRESSION": "minus_expression",
    "MOD_EXPRESSION": "mod_expression",
    "MODEXPRESSION": "mod_expression",
    "MULT_EXPRESSION": "mult_expression",
    "MULTEXPRESSION": "mult_expression",
    "MATHS_BOOLEAN_VARIABLE": "maths_boolean_variable",
    "MATHSBOOLEANVARIABLE": "maths_boolean_variable",
    "MATHS_INTEGER_VARIABLE": "maths_integer_variable",
    "MATHSINTEGERVARIABLE": "maths_integer_variable",
    "MATHS_REAL_VARIABLE": "maths_real_variable",
    "MATHSREALVARIABLE": "maths_real_variable",
    "MATHS_STRING_VARIABLE": "maths_string_variable",
    "MATHSSTRINGVARIABLE": "maths_string_variable",
    "MAXIMUM_FUNCTION": "maximum_function",
    "MAXIMUMFUNCTION": "maximum_function",
    "MINIMUM_FUNCTION": "minimum_function",
    "MINIMUMFUNCTION": "minimum_function",
    "MINUS_FUNCTION": "minus_function",
    "MINUSFUNCTION": "minus_function",
    "NUMERIC_VARIABLE": "numeric_variable",
    "NUMERICVARIABLE": "numeric_variable",
    "ODD_FUNCTION": "odd_function",
    "ODDFUNCTION": "odd_function",
    "PLUS_EXPRESSION": "plus_expression",
    "PLUSEXPRESSION": "plus_expression",
    "POWER_EXPRESSION": "power_expression",
    "POWEREXPRESSION": "power_expression",
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
    "RATIONAL_REPRESENTATION_ITEM": "rational_representation_item",
    "RATIONALREPRESENTATIONITEM": "rational_representation_item",
    "SERIAL_NUMBERED_EFFECTIVITY": "serial_numbered_effectivity",
    "SIN_FUNCTION": "sin_function",
    "SINFUNCTION": "sin_function",
    "SLASH_EXPRESSION": "slash_expression",
    "SLASHEXPRESSION": "slash_expression",
    "SQUARE_ROOT_FUNCTION": "square_root_function",
    "SQUAREROOTFUNCTION": "square_root_function",
    "STRING_LITERAL": "string_literal",
    "STRINGLITERAL": "string_literal",
    "STRING_VARIABLE": "string_variable",
    "STRINGVARIABLE": "string_variable",
    "SUBSTRING_EXPRESSION": "substring_expression",
    "SUBSTRINGEXPRESSION": "substring_expression",
    "TAN_FUNCTION": "tan_function",
    "TANFUNCTION": "tan_function",
    "TIME_INTERVAL_BASED_EFFECTIVITY": "time_interval_based_effectivity",
    "VALUE_FUNCTION": "value_function",
    "VALUEFUNCTION": "value_function",
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


_STEP_NUMERIC_ARITHMETIC_OPERATORS = {
    "numeric_add",
    "numeric_divide",
    "numeric_mod",
    "numeric_multiply",
    "numeric_power",
    "numeric_subtract",
}


_STEP_NUMERIC_FUNCTION_OPERATORS = {
    "numeric_abs",
    "numeric_acos",
    "numeric_asin",
    "numeric_atan",
    "numeric_cos",
    "numeric_exp",
    "numeric_log",
    "numeric_log10",
    "numeric_log2",
    "numeric_max",
    "numeric_min",
    "numeric_negate",
    "numeric_sin",
    "numeric_sqrt",
    "numeric_tan",
}


_STEP_STRING_EXPRESSION_OPERATORS = {"string_concat", "string_format", "string_index", "string_substring"}


_STEP_NUMERIC_STRING_FUNCTION_OPERATORS = {"string_integer_value", "string_length", "string_value"}


_STEP_NUMERIC_FORMAT_RE = re.compile(
    r"^(?:[^%]|%%)*%"
    r"(?P<flags>[-+ #0]{0,5})"
    r"(?P<width>\d{0,3})"
    r"(?:\.(?P<precision>\d{1,3}))?"
    r"(?P<type>[diouxXeEfFgG])"
    r"(?:[^%]|%%)*$"
)


_STEP_CONDITION_OPERAND_OPERATORS = {
    "numeric_literal",
    "numeric_variable",
    "string_literal",
    "string_variable",
    *_STEP_NUMERIC_ARITHMETIC_OPERATORS,
    *_STEP_NUMERIC_FUNCTION_OPERATORS,
    *_STEP_NUMERIC_STRING_FUNCTION_OPERATORS,
    *_STEP_STRING_EXPRESSION_OPERATORS,
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
    condition_text: str | None = None

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
        if self.condition_text is not None:
            data["condition_text"] = self.condition_text
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


def _extract_step_design_variants(source: Path, options: StepReadOptions) -> _StepDesignVariantExtraction:
    if not options.design_variants and not options.design_variant_selection:
        return _StepDesignVariantExtraction(records=(), summary=_empty_design_variant_summary(), warnings=())

    text = _read_step_scan_text(source)
    if text is None:
        return _StepDesignVariantExtraction(records=(), summary=_empty_design_variant_summary(), warnings=())
    step_records = {f"#{record.number}": record for record in _iter_step_records(text)}
    records: list[_StepDesignVariantRecord] = []
    for record in step_records.values():
        kind = _STEP_DESIGN_VARIANT_ENTITY_KINDS.get(record.entity)
        if kind is None:
            continue
        strings = _step_string_values(record.args)
        references = tuple(f"#{item}" for item in _STEP_REFERENCE_RE.findall(record.args))
        effectivity_values = _step_effectivity_values(record.entity, strings, record.args, step_records)
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
                condition_text=_step_condition_text(record.entity, record.args),
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
    if normalized == "INTERVALEXPRESSION":
        return "interval"
    if normalized == "LIKEEXPRESSION":
        return "like"
    if normalized == "EXPRESSIONEXTENSIONSTRING":
        return "string_literal"
    if normalized == "INDEXEXPRESSION":
        return "string_index"
    if normalized == "FORMATFUNCTION":
        return "string_format"
    if normalized == "EXPRESSIONEXTENSIONNUMERIC":
        return "numeric_literal"
    if normalized == "PLUSEXPRESSION":
        return "numeric_add"
    if normalized == "MINUSEXPRESSION":
        return "numeric_subtract"
    if normalized == "MULTEXPRESSION":
        return "numeric_multiply"
    if normalized in {"DIVEXPRESSION", "RATIONALREPRESENTATIONITEM", "SLASHEXPRESSION"}:
        return "numeric_divide"
    if normalized == "MODEXPRESSION":
        return "numeric_mod"
    if normalized == "POWEREXPRESSION":
        return "numeric_power"
    if normalized == "ABSFUNCTION":
        return "numeric_abs"
    if normalized == "ACOSFUNCTION":
        return "numeric_acos"
    if normalized == "ASINFUNCTION":
        return "numeric_asin"
    if normalized == "ATANFUNCTION":
        return "numeric_atan"
    if normalized == "COSFUNCTION":
        return "numeric_cos"
    if normalized == "EXPFUNCTION":
        return "numeric_exp"
    if normalized == "LOGFUNCTION":
        return "numeric_log"
    if normalized == "LOG2FUNCTION":
        return "numeric_log2"
    if normalized == "LOG10FUNCTION":
        return "numeric_log10"
    if normalized == "MINUSFUNCTION":
        return "numeric_negate"
    if normalized == "SQUAREROOTFUNCTION":
        return "numeric_sqrt"
    if normalized == "MAXIMUMFUNCTION":
        return "numeric_max"
    if normalized == "MINIMUMFUNCTION":
        return "numeric_min"
    if normalized == "SINFUNCTION":
        return "numeric_sin"
    if normalized == "TANFUNCTION":
        return "numeric_tan"
    if normalized == "ODDFUNCTION":
        return "numeric_odd"
    if normalized == "LENGTHFUNCTION":
        return "string_length"
    if normalized == "VALUEFUNCTION":
        return "string_value"
    if normalized == "INTVALUEFUNCTION":
        return "string_integer_value"
    if normalized == "CONCATEXPRESSION":
        return "string_concat"
    if normalized == "SUBSTRINGEXPRESSION":
        return "string_substring"
    if normalized in {"BOOLEANLITERAL", "BOOLEANREPRESENTATIONITEM", "LOGICALLITERAL", "LOGICALREPRESENTATIONITEM"}:
        return "literal"
    if normalized in {
        "INTLITERAL",
        "INTEGERREPRESENTATIONITEM",
        "LITERALNUMBER",
        "REALLITERAL",
        "REALREPRESENTATIONITEM",
    }:
        return "numeric_literal"
    if normalized == "STRINGLITERAL":
        return "string_literal"
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
    if normalized in {"MATHSSTRINGVARIABLE", "STRINGVARIABLE"}:
        return "string_variable"
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


def _step_condition_text(entity: str, args: str) -> str | None:
    if _step_condition_operator(entity) != "string_literal":
        return None
    values = _step_string_values(args)
    return values[-1] if values else None


def _step_effectivity_values(
    entity: str,
    strings: list[str],
    args: str,
    records: dict[str, _StepRecord],
) -> tuple[str, ...]:
    if "EFFECTIVITY" not in entity:
        return ()
    values: list[str] = [item for item in strings if item]
    normalized_entity = entity.replace("_", "")
    if normalized_entity in {"DATEDEFFECTIVITY", "TIMEINTERVALBASEDEFFECTIVITY"}:
        references = tuple(f"#{item}" for item in _STEP_REFERENCE_RE.findall(args))
        values.extend(_step_referenced_date_values(references, records))
    if not values:
        values.extend(str(value) for value in _step_number_values(args))
    return tuple(dict.fromkeys(values))


def _step_referenced_date_values(references: tuple[str, ...], records: dict[str, _StepRecord]) -> list[str]:
    values: list[str] = []
    visited: set[str] = set()

    def visit(reference: str, depth: int) -> None:
        if depth > 6 or reference in visited:
            return
        visited.add(reference)
        record = records.get(reference)
        if record is None:
            return
        for value in _step_record_date_values(record):
            if value not in values:
                values.append(value)
        for child_reference in _step_record_references(record):
            visit(child_reference, depth + 1)

    for reference in references:
        visit(reference, 0)
    return values


def _step_record_date_values(record: _StepRecord) -> tuple[str, ...]:
    values: list[str] = []
    for value in _step_string_values(record.args):
        parsed = _parse_effectivity_date(value)
        if parsed is not None:
            values.append(parsed.isoformat())
    if parsed := _step_record_calendar_date(record):
        values.append(parsed.isoformat())
    return tuple(dict.fromkeys(values))


def _step_record_calendar_date(record: _StepRecord) -> date | None:
    normalized_entity = record.entity.replace("_", "")
    if normalized_entity == "CALENDARDATE":
        components = _step_integer_components(record.args, 3)
        if components is None:
            return None
        year, day, month = components
        try:
            return date(year, month, day)
        except ValueError:
            return None
    if normalized_entity == "ORDINALDATE":
        components = _step_integer_components(record.args, 2)
        if components is None:
            return None
        year, day_of_year = components
        if day_of_year < 1:
            return None
        try:
            parsed = date(year, 1, 1) + timedelta(days=day_of_year - 1)
        except (OverflowError, ValueError):
            return None
        return parsed if parsed.year == year else None
    if normalized_entity == "WEEKOFYEARANDDAYDATE":
        components = _step_integer_components(record.args, 3)
        if components is None:
            return None
        year, week, day = components
        try:
            return date.fromisocalendar(year, week, day)
        except ValueError:
            return None
    return None


def _step_integer_components(args: str, count: int) -> tuple[int, ...] | None:
    numbers = _step_number_values(args)
    if len(numbers) < count or not all(value.is_integer() for value in numbers[:count]):
        return None
    return tuple(int(value) for value in numbers[:count])


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
    selector_term_cache: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}
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
            and record.condition_operator not in _STEP_CONDITION_OPERAND_OPERATORS
            and condition_match.matched
            and condition_match.positive
        )
        if record.condition_operator is not None:
            if record.condition_operator == "variable":
                direct_match = _design_variant_record_requested_boolean(record, requested, normalized_requested) is True
            else:
                direct_match = record.condition_operator not in {
                    "conditional",
                    "effectivity_assignment",
                    "effectivity_context_assignment",
                    "effectivity_relationship",
                    "effectivity_usage",
                    "ineffectivity_assignment",
                    *_STEP_CONDITION_OPERAND_OPERATORS,
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
            record.condition_operator in {"greater", "greater_equal", "less", "less_equal", "interval", "numeric_odd"}
            and not condition_applies
            and _condition_record_has_requested_numeric_operand(
                record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(),
            )
        )
        condition_blocked = condition_blocked or (
            record.condition_operator in {"equals", "not_equals"}
            and not condition_applies
            and (
                _condition_record_has_requested_numeric_operand(
                    record,
                    records_by_reference,
                    requested,
                    normalized_requested,
                    visited=set(),
                )
                or _condition_record_has_requested_string_operand(
                    record,
                    records_by_reference,
                    requested,
                    normalized_requested,
                    visited=set(),
                )
            )
        )
        condition_blocked = condition_blocked or (
            record.condition_operator == "like"
            and not condition_applies
            and (
                _condition_record_has_requested_string_operand(
                    record,
                    records_by_reference,
                    requested,
                    normalized_requested,
                    visited=set(),
                )
                or _condition_record_has_requested_numeric_operand(
                    record,
                    records_by_reference,
                    requested,
                    normalized_requested,
                    visited=set(),
                )
            )
        )
        if not direct_id_match and not direct_match and not condition_applies:
            continue
        matched.append(record.id)
        terms.extend(
            _design_variant_geometry_selector_terms(
                record,
                records_by_reference,
                visited=set(),
                selector_term_cache=selector_term_cache,
            )
        )
    selector_terms = tuple(
        item for item in dict.fromkeys(term.strip() for term in terms if term.strip()) if _normalize_variant_term(item)
    )
    if condition_blocked and not matched:
        selector_terms = ()
    return tuple(dict.fromkeys(matched)), selector_terms, condition_blocked


def _design_variant_geometry_selector_terms(
    record: _StepDesignVariantRecord,
    records_by_reference: dict[str, _StepDesignVariantRecord],
    *,
    visited: set[str],
    selector_term_cache: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    record_reference = _design_variant_record_step_reference(record)
    if record_reference in visited or record.condition_operator in _STEP_CONDITION_OPERAND_OPERATORS:
        return ()
    cache_key = (record_reference, tuple(sorted(visited)))
    if selector_term_cache is not None and cache_key in selector_term_cache:
        return selector_term_cache[cache_key]
    visited.add(record_reference)
    if record.condition_operator is None:
        selector_terms = _design_variant_record_selector_terms(record)
        if selector_term_cache is not None:
            selector_term_cache[cache_key] = selector_terms
        return selector_terms

    terms: list[str] = []
    for reference in record.references:
        child_record = records_by_reference.get(reference)
        if child_record is None:
            continue
        terms.extend(
            _design_variant_geometry_selector_terms(
                child_record,
                records_by_reference,
                visited=set(visited),
                selector_term_cache=selector_term_cache,
            )
        )
    if record.condition_operator in {
        "effectivity_assignment",
        "effectivity_context_assignment",
        "effectivity_relationship",
        "effectivity_usage",
        "ineffectivity_assignment",
    }:
        terms.extend(_design_variant_record_selector_terms(record))
    if record.condition_operator == "conditional" and record.kind == "conditional_concept_feature":
        terms.extend(_design_variant_record_selector_terms(record))
    selector_terms = tuple(dict.fromkeys(terms))
    if selector_term_cache is not None:
        selector_term_cache[cache_key] = selector_terms
    return selector_terms


def _design_variant_record_selector_terms(record: _StepDesignVariantRecord) -> tuple[str, ...]:
    terms: list[str] = []
    terms.extend(_design_variant_label_terms(record.label))
    for label in record.reference_labels:
        terms.extend(_design_variant_label_terms(label))
    for label in record.resolved_reference_labels:
        terms.extend(_design_variant_label_terms(label))
    terms.extend(record.effectivity_values)
    terms.extend(record.references)
    return tuple(dict.fromkeys(terms))


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
        "interval",
        "like",
        *_STEP_NUMERIC_ARITHMETIC_OPERATORS,
        *_STEP_NUMERIC_FUNCTION_OPERATORS,
        *_STEP_NUMERIC_STRING_FUNCTION_OPERATORS,
        *_STEP_STRING_EXPRESSION_OPERATORS,
        "numeric_odd",
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
        bool_value = _design_variant_record_requested_boolean(record, requested, normalized_requested)
        return _StepConditionMatch(matched=bool_value is True, positive=bool_value is not None)
    if operator == "numeric_literal":
        return _StepConditionMatch(matched=record.condition_number is not None)
    if operator == "numeric_variable":
        matched = _design_variant_record_requested_number(record, requested, normalized_requested) is not None
        return _StepConditionMatch(matched=matched, positive=matched)
    if operator == "string_literal":
        return _StepConditionMatch(matched=record.condition_text is not None)
    if operator == "string_variable":
        matched = _design_variant_record_requested_text(record, requested, normalized_requested) is not None
        return _StepConditionMatch(matched=matched, positive=matched)
    if operator in _STEP_NUMERIC_ARITHMETIC_OPERATORS:
        value, positive = _condition_record_numeric_value(
            record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(),
        )
        matched = value is not None
        return _StepConditionMatch(matched=matched, positive=matched and positive)
    if operator in _STEP_NUMERIC_FUNCTION_OPERATORS:
        value, positive = _condition_record_numeric_value(
            record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(),
        )
        matched = value is not None
        return _StepConditionMatch(matched=matched, positive=matched and positive)
    if operator == "numeric_odd":
        numeric_children = [
            _condition_record_numeric_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        values = [item[0] for item in numeric_children if item[0] is not None]
        matched = len(values) == len(numeric_children) and len(values) == 1 and _numeric_odd_matches(values[0])
        return _StepConditionMatch(matched=matched, positive=matched and any(item[1] for item in numeric_children))
    if operator in _STEP_NUMERIC_STRING_FUNCTION_OPERATORS:
        value, positive = _condition_record_numeric_value(
            record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(),
        )
        matched = value is not None
        return _StepConditionMatch(matched=matched, positive=matched and positive)
    if operator in _STEP_STRING_EXPRESSION_OPERATORS:
        text_value, positive = _condition_record_string_value(
            record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(),
        )
        matched = text_value is not None
        return _StepConditionMatch(matched=matched, positive=matched and positive)
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
        return _StepConditionMatch(matched=not child_match.matched, positive=child_match.positive)
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
    if operator == "interval":
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
        matched = len(values) == len(numeric_children) and len(values) >= 3 and _numeric_interval_matches(values)
        return _StepConditionMatch(matched=matched, positive=matched and any(item[1] for item in numeric_children))
    if operator == "like":
        string_children = [
            _condition_record_string_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for child_record in child_records
        ]
        string_values = [item[0] for item in string_children if item[0] is not None]
        matched = (
            len(string_values) == len(string_children)
            and len(string_values) >= 2
            and _string_like_matches(string_values[0], string_values[1])
        )
        return _StepConditionMatch(matched=matched, positive=matched and any(item[1] for item in string_children))
    if operator == "equals":
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
        numeric_values = [item[0] for item in numeric_children if item[0] is not None]
        if any(_condition_record_produces_numeric_value(child_record) for child_record in child_records):
            if len(numeric_values) != len(numeric_children) or len(numeric_values) < 2:
                return _StepConditionMatch(matched=False)
            matched = _numeric_equality_matches(numeric_values)
            return _StepConditionMatch(
                matched=matched,
                positive=matched and any(item[1] for item in numeric_children),
            )
        string_children = [
            _condition_record_string_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for child_record in child_records
        ]
        string_values = [item[0] for item in string_children if item[0] is not None]
        if any(_condition_record_produces_string_value(child_record) for child_record in child_records):
            if len(string_values) != len(string_children) or len(string_values) < 2:
                return _StepConditionMatch(matched=False)
            matched = _string_equality_matches(string_values)
            return _StepConditionMatch(
                matched=matched,
                positive=matched and any(item[1] for item in string_children),
            )
        matched = (
            len(children) >= 2
            and any(child.matched for child in children)
            and all(child.matched == children[0].matched for child in children)
        )
        return _StepConditionMatch(matched=matched, positive=matched and any(child.positive for child in children))
    if operator == "not_equals":
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
        numeric_values = [item[0] for item in numeric_children if item[0] is not None]
        if any(_condition_record_produces_numeric_value(child_record) for child_record in child_records):
            if len(numeric_values) != len(numeric_children) or len(numeric_values) < 2:
                return _StepConditionMatch(matched=False)
            matched = not _numeric_equality_matches(numeric_values)
            return _StepConditionMatch(
                matched=matched,
                positive=matched and any(item[1] for item in numeric_children),
            )
        string_children = [
            _condition_record_string_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for child_record in child_records
        ]
        string_values = [item[0] for item in string_children if item[0] is not None]
        if any(_condition_record_produces_string_value(child_record) for child_record in child_records):
            if len(string_values) != len(string_children) or len(string_values) < 2:
                return _StepConditionMatch(matched=False)
            matched = not _string_equality_matches(string_values)
            return _StepConditionMatch(
                matched=matched,
                positive=matched and any(item[1] for item in string_children),
            )
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
    if record.condition_operator in _STEP_NUMERIC_ARITHMETIC_OPERATORS:
        numeric_children = [
            _condition_record_numeric_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        values = [item[0] for item in numeric_children if item[0] is not None]
        if len(values) != len(numeric_children) or not values:
            return None, False
        return _numeric_arithmetic_value(record.condition_operator, values), any(item[1] for item in numeric_children)
    if record.condition_operator in _STEP_NUMERIC_FUNCTION_OPERATORS:
        numeric_children = [
            _condition_record_numeric_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        values = [item[0] for item in numeric_children if item[0] is not None]
        if len(values) != len(numeric_children) or not values:
            return None, False
        return _numeric_function_value(record.condition_operator, values), any(item[1] for item in numeric_children)
    if record.condition_operator == "string_length":
        string_children = [
            _condition_record_string_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        string_values = [item[0] for item in string_children if item[0] is not None]
        if len(string_values) != len(string_children) or len(string_values) != 1:
            return None, False
        return float(len(string_values[0])), any(item[1] for item in string_children)
    if record.condition_operator in {"string_integer_value", "string_value"}:
        string_children = [
            _condition_record_string_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        string_values = [item[0] for item in string_children if item[0] is not None]
        if len(string_values) != len(string_children) or len(string_values) != 1:
            return None, False
        value = _numeric_value_from_condition_text(
            string_values[0],
            integer=record.condition_operator == "string_integer_value",
        )
        return value, value is not None and any(item[1] for item in string_children)
    return None, False


def _condition_record_string_value(
    record: _StepDesignVariantRecord,
    records_by_reference: dict[str, _StepDesignVariantRecord],
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
    *,
    visited: set[str],
) -> tuple[str | None, bool]:
    record_reference = _design_variant_record_step_reference(record)
    if record_reference in visited:
        return None, False
    visited.add(record_reference)
    if record.condition_operator == "string_literal":
        return record.condition_text, False
    if record.condition_operator == "string_variable":
        value = _design_variant_record_requested_text(record, requested, normalized_requested)
        return value, value is not None
    if record.condition_operator == "string_concat":
        string_children = [
            _condition_record_string_value(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        string_values = [item[0] for item in string_children if item[0] is not None]
        if len(string_values) != len(string_children) or len(string_values) < 2:
            return None, False
        return "".join(string_values), any(item[1] for item in string_children)
    if record.condition_operator == "string_format":
        child_records = [
            child_record
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        if len(child_records) != 2:
            return None, False
        number, number_positive = _condition_record_numeric_value(
            child_records[0],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        format_string, format_positive = _condition_record_string_value(
            child_records[1],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        if number is None or format_string is None:
            return None, False
        formatted = _string_format_numeric_value(number, format_string)
        return formatted, formatted is not None and (number_positive or format_positive)
    if record.condition_operator == "string_index":
        child_records = [
            child_record
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        if len(child_records) != 2:
            return None, False
        text, text_positive = _condition_record_string_value(
            child_records[0],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        index, index_positive = _condition_record_numeric_value(
            child_records[1],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        if text is None or index is None:
            return None, False
        indexed = _string_index_value(text, index)
        return indexed, indexed is not None and (text_positive or index_positive)
    if record.condition_operator == "string_substring":
        child_records = [
            child_record
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        ]
        if len(child_records) != 3:
            return None, False
        text, text_positive = _condition_record_string_value(
            child_records[0],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        start, start_positive = _condition_record_numeric_value(
            child_records[1],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        end, end_positive = _condition_record_numeric_value(
            child_records[2],
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        if text is None or start is None or end is None:
            return None, False
        substring = _string_substring_value(text, start, end)
        return substring, substring is not None and (text_positive or start_positive or end_positive)
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
    if record.condition_operator in _STEP_NUMERIC_STRING_FUNCTION_OPERATORS:
        return any(
            _condition_record_has_requested_string_operand(
                child_record,
                records_by_reference,
                requested,
                normalized_requested,
                visited=set(visited),
            )
            for reference in record.references
            if (child_record := records_by_reference.get(reference)) is not None
        )
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


def _condition_record_produces_numeric_value(record: _StepDesignVariantRecord) -> bool:
    return record.condition_operator in {
        "numeric_literal",
        "numeric_variable",
        *_STEP_NUMERIC_ARITHMETIC_OPERATORS,
        *_STEP_NUMERIC_FUNCTION_OPERATORS,
        *_STEP_NUMERIC_STRING_FUNCTION_OPERATORS,
    }


def _condition_record_has_requested_string_operand(
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
    if record.condition_operator == "string_variable":
        return _design_variant_record_requested_text(record, requested, normalized_requested) is not None
    return any(
        _condition_record_has_requested_string_operand(
            child_record,
            records_by_reference,
            requested,
            normalized_requested,
            visited=set(visited),
        )
        for reference in record.references
        if (child_record := records_by_reference.get(reference)) is not None
    )


def _condition_record_produces_string_value(record: _StepDesignVariantRecord) -> bool:
    return record.condition_operator in {
        "string_literal",
        "string_variable",
        *_STEP_STRING_EXPRESSION_OPERATORS,
    }


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


def _numeric_interval_matches(values: list[float]) -> bool:
    low, item, high = values[:3]
    return low <= item <= high


def _numeric_equality_matches(values: list[float]) -> bool:
    if len(values) < 2 or not all(np.isfinite(value) for value in values):
        return False
    first = values[0]
    return all(np.isclose(first, value, rtol=1e-12, atol=1e-12) for value in values[1:])


def _numeric_arithmetic_value(operator: str, values: list[float]) -> float | None:
    if operator == "numeric_add":
        return sum(values)
    if operator == "numeric_subtract" and len(values) >= 2:
        result = values[0]
        for value in values[1:]:
            result -= value
        return result
    if operator == "numeric_multiply":
        result = 1.0
        for value in values:
            result *= value
        return result
    if operator == "numeric_divide" and len(values) >= 2:
        result = values[0]
        for value in values[1:]:
            if value == 0:
                return None
            result /= value
        return result
    if operator == "numeric_mod" and len(values) >= 2:
        result = values[0]
        for value in values[1:]:
            if value == 0:
                return None
            result %= value
        return result
    if operator == "numeric_power" and len(values) == 2:
        try:
            result = values[0] ** values[1]
        except (OverflowError, TypeError, ValueError):
            return None
        if isinstance(result, (int, float)) and np.isfinite(result):
            return float(result)
    return None


def _numeric_function_value(operator: str, values: list[float]) -> float | None:
    if not all(np.isfinite(value) for value in values):
        return None
    if operator == "numeric_abs" and len(values) == 1:
        return abs(values[0])
    if operator == "numeric_acos" and len(values) == 1:
        if not -1.0 <= values[0] <= 1.0:
            return None
        return _finite_numeric_result(float(np.arccos(values[0])))
    if operator == "numeric_asin" and len(values) == 1:
        if not -1.0 <= values[0] <= 1.0:
            return None
        return _finite_numeric_result(float(np.arcsin(values[0])))
    if operator == "numeric_atan" and len(values) == 1:
        return _finite_numeric_result(float(np.arctan(values[0])))
    if operator == "numeric_atan" and len(values) == 2:
        return _finite_numeric_result(float(np.arctan2(values[0], values[1])))
    if operator == "numeric_cos" and len(values) == 1:
        return _finite_numeric_result(float(np.cos(values[0])))
    if operator == "numeric_exp" and len(values) == 1:
        return _finite_numeric_result(float(np.exp(values[0])))
    if operator == "numeric_log" and len(values) == 1:
        if values[0] <= 0:
            return None
        return _finite_numeric_result(float(np.log(values[0])))
    if operator == "numeric_log2" and len(values) == 1:
        if values[0] <= 0:
            return None
        return _finite_numeric_result(float(np.log2(values[0])))
    if operator == "numeric_log10" and len(values) == 1:
        if values[0] <= 0:
            return None
        return _finite_numeric_result(float(np.log10(values[0])))
    if operator == "numeric_negate" and len(values) == 1:
        return -values[0]
    if operator == "numeric_sin" and len(values) == 1:
        return _finite_numeric_result(float(np.sin(values[0])))
    if operator == "numeric_sqrt" and len(values) == 1:
        if values[0] < 0:
            return None
        return _finite_numeric_result(float(values[0] ** 0.5))
    if operator == "numeric_tan" and len(values) == 1:
        return _finite_numeric_result(float(np.tan(values[0])))
    if operator == "numeric_max":
        return max(values)
    if operator == "numeric_min":
        return min(values)
    return None


def _finite_numeric_result(value: float) -> float | None:
    return value if np.isfinite(value) else None


def _numeric_odd_matches(value: float) -> bool:
    return bool(np.isfinite(value) and value.is_integer() and int(value) % 2 != 0)


def _string_like_matches(value: str, pattern: str) -> bool:
    normalized_value = _normalize_condition_text(value)
    normalized_pattern = _normalize_condition_text(pattern)
    if not normalized_value or not normalized_pattern:
        return False
    wildcard = object()
    single = object()
    tokens: list[str | object] = []
    for char in normalized_pattern:
        if char in {"*", "%"}:
            tokens.append(wildcard)
        elif char in {"?", "_"}:
            tokens.append(single)
        else:
            tokens.append(re.escape(char))
    pattern_re = (
        "^" + "".join(".*" if token is wildcard else "." if token is single else str(token) for token in tokens) + "$"
    )
    return re.fullmatch(pattern_re, normalized_value) is not None


def _string_equality_matches(values: list[str]) -> bool:
    if len(values) < 2:
        return False
    normalized = [_normalize_condition_text(value) for value in values]
    first = normalized[0]
    return bool(first) and all(value == first for value in normalized[1:])


def _string_substring_value(value: str, start: float, end: float) -> str | None:
    if not (np.isfinite(start) and np.isfinite(end)):
        return None
    if not (float(start).is_integer() and float(end).is_integer()):
        return None
    start_index = int(start)
    end_index = int(end)
    if start_index < 1 or end_index < start_index or end_index > len(value):
        return None
    return value[start_index - 1 : end_index]


def _string_index_value(value: str, index: float) -> str | None:
    if not np.isfinite(index) or not float(index).is_integer():
        return None
    index_value = int(index)
    if index_value < 1 or index_value > len(value):
        return None
    return value[index_value - 1]


def _string_format_numeric_value(value: float, format_string: str) -> str | None:
    if not np.isfinite(value) or len(format_string) > 128:
        return None
    if not format_string:
        return _condition_number_to_string(value)
    match = _STEP_NUMERIC_FORMAT_RE.fullmatch(format_string)
    if match is None:
        return None
    width = int(match.group("width") or "0")
    precision = int(match.group("precision") or "0")
    if width > 128 or precision > 16:
        return None
    format_type = match.group("type")
    format_value: float | int = value
    if format_type in "diouxX":
        if not float(value).is_integer():
            return None
        format_value = int(value)
    try:
        result = format_string % format_value
    except (TypeError, ValueError, OverflowError):
        return None
    if len(result) > 256:
        return None
    return result


def _condition_number_to_string(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.12g}"


def _numeric_value_from_condition_text(value: str, *, integer: bool) -> float | None:
    stripped = value.strip()
    if integer:
        if _STEP_STRICT_INTEGER_RE.fullmatch(stripped) is None:
            return None
        return float(int(stripped))
    if _STEP_STRICT_NUMBER_RE.fullmatch(stripped) is None:
        return None
    parsed = float(stripped)
    return parsed if np.isfinite(parsed) else None


def _boolean_value_from_condition_text(value: str) -> bool | None:
    stripped = value.strip()
    if not stripped:
        return None
    if match := _STEP_BOOLEAN_TOKEN_RE.search(stripped.upper()):
        return match.group(1) in {"T", "TRUE"}
    normalized = _normalize_condition_text(stripped)
    if normalized in {"1", "on", "t", "true", "yes"}:
        return True
    if normalized in {"0", "f", "false", "no", "off"}:
        return False
    return None


def _normalize_condition_text(value: str) -> str:
    return " ".join(value.lower().split())


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


def _design_variant_record_requested_text(
    record: _StepDesignVariantRecord,
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
) -> str | None:
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
        for separator in ("=", ":"):
            if separator not in raw:
                continue
            left, right = raw.split(separator, 1)
            normalized_left = _normalize_variant_term(left)
            if any(selector and selector in normalized_left for selector in selectors):
                value = " ".join(right.split())
                return value if value else None
    return None


def _design_variant_record_requested_boolean(
    record: _StepDesignVariantRecord,
    requested: tuple[str, ...],
    normalized_requested: tuple[str, ...],
) -> bool | None:
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
        for separator in ("=", ":"):
            if separator not in raw:
                continue
            left, right = raw.split(separator, 1)
            normalized_left = _normalize_variant_term(left)
            if any(selector and selector in normalized_left for selector in selectors):
                return _boolean_value_from_condition_text(right)
        return True
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
