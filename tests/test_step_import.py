from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import fascat as fc
import fascat.io.step as step_io
from fascat.io.step import (
    _apply_material_libraries_to_materials,
    _apply_material_library_mapping,
    _apply_step_design_variant_selection,
    _attach_source_textures_to_materials,
    _build_mixed_construction_curve_node,
    _CadMaterialSpec,
    _canonical_part_id,
    _cleanup_action,
    _color_material_spec,
    _extract_material_libraries,
    _extract_source_textures,
    _extract_step_design_variants,
    _extract_step_pmi_annotations,
    _extract_step_pmi_semantic_graph,
    _import_decisions,
    _import_warnings,
    _ImportCleanupStats,
    _loaded_representation,
    _loaded_representation_report,
    _material_binding_plan,
    _mixed_construction_curve_metadata,
    _mixed_construction_curve_shape,
    _resolve_step_external_reference_graph,
    _shape_fingerprint,
    _shape_topology_counts,
    _ShapeTopologyCounts,
    _space_normalization,
    _StepHeaderInfo,
    read_step_many,
)
from fascat.options import StepReadOptions
from fascat.report import Report


def test_canonical_part_id_reuses_matching_shape_and_material() -> None:
    part_index: dict[tuple[str, str, str, str], str] = {}

    first_id, first_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:1",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )
    second_id, second_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:2",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )
    different_material_id, different_material_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:3",
        shape_hash="shape-a",
        material_signature="mat-blue",
        part_index=part_index,
    )

    assert first_is_new is True
    assert second_is_new is False
    assert first_id == second_id
    assert different_material_is_new is True
    assert different_material_id != first_id


def test_canonical_part_id_prefers_source_label_before_shape_hash() -> None:
    part_index: dict[tuple[str, str, str, str], str] = {}

    first_id, first_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:1",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )
    repeated_label_id, repeated_label_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:1",
        shape_hash="unstable-shape-hash",
        material_signature="mat-red",
        part_index=part_index,
    )
    repeated_shape_id, repeated_shape_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:2",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )

    assert first_is_new is True
    assert repeated_label_is_new is False
    assert repeated_shape_is_new is False
    assert repeated_label_id == first_id
    assert repeated_shape_id == first_id


def test_material_binding_plan_maps_step_face_colors_to_indices() -> None:
    material_ids, material_indices = _material_binding_plan(
        "mat-red",
        ["mat-red", "mat-blue", "mat-red", "mat-green"],
    )

    assert material_ids == ["mat-red", "mat-blue", "mat-green"]
    assert material_indices == [0, 1, 0, 2]


def test_shape_fingerprint_falls_back_to_python_hash() -> None:
    class ShapeWithoutHashCode:
        def __hash__(self) -> int:
            return 123

    assert _shape_fingerprint(ShapeWithoutHashCode()) == "123"


def test_step_import_warnings_report_unsupported_import_intent() -> None:
    warnings = _import_warnings(
        StepReadOptions(design_variants=True, multi_file=True),
        _StepHeaderInfo(schema="AP242", pmi_present=True),
        unsupported_pmi_count=1,
    )

    assert warnings == [
        "STEP file advertises AP242 PMI, but no supported typed PMI entities were extracted; annotations are omitted",
        "STEP design variant import was requested, but no supported design variant records were detected",
        "multi-file STEP assembly import is not implemented; external references are not loaded",
    ]


def test_step_text_pmi_extraction_reads_common_ap242_records(tmp_path: Path) -> None:
    source = tmp_path / "pmi.step"
    source.write_text(
        "ISO-10303-21;\n"
        "HEADER;\n"
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF "
        "{ 1 0 10303 442 1 1 4 }'));\n"
        "ENDSEC;\n"
        "DATA;\n"
        "#20=PRODUCT_DEFINITION_SHAPE('bracket','',#19);\n"
        "#30=DIMENSIONAL_SIZE(#20,'hole diameter 12.5 mm',12.5);\n"
        "#31=GEOMETRIC_TOLERANCE('position tolerance',0.2,#20);\n"
        "#32=DATUM('A',#20);\n"
        "#33=ANNOTATION_TEXT_OCCURRENCE('inspect after plating',#20);\n"
        "#34=DIMENSIONAL_LOCATION(#20,#21,'slot offset 8 mm',8.0);\n"
        "#35=PLUS_MINUS_TOLERANCE('profile plus minus',0.1,-0.05,#31);\n"
        "#36=DATUM_REFERENCE_COMPARTMENT('datum A primary',#32);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    annotations = _extract_step_pmi_annotations(source, StepReadOptions(pmi=True))

    assert [annotation.kind for annotation in annotations] == [
        "dimension",
        "tolerance",
        "datum",
        "note",
        "dimension",
        "tolerance",
        "datum",
    ]
    assert [annotation.id for annotation in annotations] == [
        "step_pmi_30",
        "step_pmi_31",
        "step_pmi_32",
        "step_pmi_33",
        "step_pmi_34",
        "step_pmi_35",
        "step_pmi_36",
    ]
    assert annotations[0].text == "hole diameter 12.5 mm"
    assert annotations[0].value == 12.5
    assert annotations[0].unit == "millimetre"
    assert annotations[1].tolerance is not None
    assert annotations[1].tolerance.upper == 0.2
    assert annotations[1].source["step_references"] == ["#20"]
    assert annotations[3].text == "inspect after plating"
    assert annotations[4].text == "slot offset 8 mm"
    assert annotations[4].unit == "millimetre"
    assert annotations[5].tolerance is not None
    assert annotations[5].tolerance.upper == 0.1
    assert annotations[5].tolerance.lower == -0.05
    assert annotations[6].text == "datum A primary"


def test_step_pmi_semantic_graph_records_referenced_entities(tmp_path: Path) -> None:
    source = tmp_path / "pmi-graph.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#20=PRODUCT_DEFINITION_SHAPE('bracket','',#19);\n"
        "#21=SHAPE_ASPECT('hole face','',#20,.T.);\n"
        "#30=DIMENSIONAL_SIZE(#21,'hole diameter 12.5 mm',12.5);\n"
        "#31=GEOMETRIC_TOLERANCE('position tolerance',0.2,#20,#999);\n"
        "#32=PLUS_MINUS_TOLERANCE('profile plus minus',0.1,-0.05,#30);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=True))
    payload = graph.to_dict()

    assert graph.summary == {
        "nodes": 5,
        "pmi_nodes": 3,
        "referenced_nodes": 2,
        "edges": 4,
        "missing_references": 1,
    }
    assert [node["id"] for node in payload["nodes"]] == ["#20", "#21", "#30", "#31", "#32"]
    assert payload["nodes"][2]["kind"] == "pmi_dimension"
    assert payload["nodes"][2]["references"] == ["#21"]
    assert payload["nodes"][3]["kind"] == "pmi_tolerance"
    assert payload["nodes"][4]["kind"] == "pmi_tolerance"
    assert {"source": "#31", "target": "#999", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#32", "target": "#30", "relationship": "step_reference"} in payload["edges"]
    assert graph.warnings == ("STEP PMI semantic graph has 1 reference(s) to records that were not found",)


def test_step_text_pmi_extraction_respects_disabled_pmi(tmp_path: Path) -> None:
    source = tmp_path / "pmi.step"
    source.write_text("#1=GEOMETRIC_TOLERANCE('flatness',0.05,#2);\n", encoding="utf-8")

    assert _extract_step_pmi_annotations(source, StepReadOptions(pmi=False)) == []
    assert _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=False)).summary["nodes"] == 0


def test_step_design_variant_extraction_reads_configuration_records(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONFIGURATION_ITEM('mounting side','left/right handed option',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('left hand','select left mounting side',#2);\n"
        "#12=CONFIGURATION_DESIGN(#10,#11);\n"
        "#13=CONFIGURATION_EFFECTIVITY('serial range A',#12);\n"
        "#14=SERIAL_NUMBERED_EFFECTIVITY('SN-A-001','SN-A-099',#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_step_design_variants(source, StepReadOptions(design_variants=True))

    assert extraction.summary == {
        "records": 5,
        "configuration_items": 1,
        "product_concept_features": 1,
        "effectivity_records": 3,
        "conditional_records": 0,
    }
    assert [record.kind for record in extraction.records] == [
        "configuration_item",
        "product_concept_feature",
        "configuration_design",
        "configuration_effectivity",
        "serial_numbered_effectivity",
    ]
    assert extraction.records[0].label == "mounting side / left/right handed option"
    assert extraction.records[2].reference_labels == (
        "mounting side / left/right handed option",
        "left hand / select left mounting side",
    )
    assert extraction.records[2].references == ("#10", "#11")
    assert extraction.records[3].resolved_reference_labels == (
        "configuration design",
        "mounting side / left/right handed option",
        "left hand / select left mounting side",
    )
    assert extraction.records[4].effectivity_kind == "serial"
    assert extraction.records[4].effectivity_values == ("SN-A-001", "SN-A-099")
    assert extraction.records[4].effectivity_range == ("SN-A-001", "SN-A-099")
    assert extraction.warnings == (
        "STEP design variant records were detected and reported as metadata; "
        "pass design_variant_selection to filter geometry by selected variant labels",
    )


def test_step_design_variant_extraction_respects_disabled_option(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text("#10=CONFIGURATION_ITEM('mounting side','left/right handed option',#1);\n", encoding="utf-8")

    extraction = _extract_step_design_variants(source, StepReadOptions(design_variants=False))

    assert extraction.records == ()
    assert extraction.summary == {
        "records": 0,
        "configuration_items": 0,
        "product_concept_features": 0,
        "effectivity_records": 0,
        "conditional_records": 0,
    }
    assert extraction.warnings == ()


def test_step_design_variant_selection_filters_matching_geometry(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONFIGURATION_ITEM('mounting side','left/right handed option',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('left hand','select left housing',#2);\n"
        "#12=CONFIGURATION_DESIGN(#10,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    options = StepReadOptions(design_variant_selection=("left hand",))
    extraction = _extract_step_design_variants(source, options)
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="left-node", name="Left Housing", part_id="left"),
            fc.Node(id="right-node", name="Right Housing", part_id="right"),
        ],
    )
    parts = {
        "left": fc.Part(
            id="left",
            name="Left Housing",
            material_ids=["left-mat"],
            metadata={"source_name": "left housing"},
        ),
        "right": fc.Part(
            id="right",
            name="Right Housing",
            material_ids=["right-mat"],
            metadata={"source_name": "right housing"},
        ),
    }
    materials = {
        "left-mat": fc.Material(id="left-mat", name="Left Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "right-mat": fc.Material(id="right-mat", name="Right Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    selection = _apply_step_design_variant_selection(root, parts, materials, extraction, options)

    assert selection.status == "applied"
    assert selection.matched_records == ("step_variant_11", "step_variant_12")
    assert selection.before_nodes == 3
    assert selection.after_nodes == 2
    assert selection.removed_parts == 1
    assert [child.name for child in root.children] == ["Left Housing"]
    assert set(parts) == {"left"}
    assert set(materials) == {"left-mat"}
    assert parts["left"].metadata["design_variant_selected"] == "true"


def test_step_design_variant_selection_resolves_serial_effectivity_range(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONFIGURATION_ITEM('mounting side','left/right handed option',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('left hand','select left housing',#2);\n"
        "#12=CONFIGURATION_DESIGN(#10,#11);\n"
        "#13=SERIAL_NUMBERED_EFFECTIVITY('SN-A-001','SN-A-099',#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    options = StepReadOptions(design_variant_selection=("SN-A-050",))
    extraction = _extract_step_design_variants(source, options)
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="left-node", name="Left Housing", part_id="left"),
            fc.Node(id="right-node", name="Right Housing", part_id="right"),
        ],
    )
    parts = {
        "left": fc.Part(
            id="left",
            name="Left Housing",
            material_ids=["left-mat"],
            metadata={"source_name": "left housing"},
        ),
        "right": fc.Part(
            id="right",
            name="Right Housing",
            material_ids=["right-mat"],
            metadata={"source_name": "right housing"},
        ),
    }
    materials = {
        "left-mat": fc.Material(id="left-mat", name="Left Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "right-mat": fc.Material(id="right-mat", name="Right Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    selection = _apply_step_design_variant_selection(root, parts, materials, extraction, options)

    assert selection.status == "applied"
    assert selection.matched_records == ("step_variant_13",)
    assert "left housing" in [term.lower() for term in selection.selector_terms]
    assert [child.name for child in root.children] == ["Left Housing"]
    assert set(parts) == {"left"}


def test_step_design_variant_selection_resolves_dated_effectivity_range(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONFIGURATION_ITEM('release window','date gated option',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('inspection cover','select inspection cover',#2);\n"
        "#12=CONFIGURATION_DESIGN(#10,#11);\n"
        "#13=DATED_EFFECTIVITY('release A','2026-12-31','2026-01-01',#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    options = StepReadOptions(design_variant_selection=("2026-06-01",))
    extraction = _extract_step_design_variants(source, options)
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="cover-node", name="Inspection Cover", part_id="cover"),
            fc.Node(id="plug-node", name="Blanking Plug", part_id="plug"),
        ],
    )
    parts = {
        "cover": fc.Part(
            id="cover",
            name="Inspection Cover",
            material_ids=["cover-mat"],
            metadata={"source_name": "inspection cover"},
        ),
        "plug": fc.Part(
            id="plug",
            name="Blanking Plug",
            material_ids=["plug-mat"],
            metadata={"source_name": "blanking plug"},
        ),
    }
    materials = {
        "cover-mat": fc.Material(id="cover-mat", name="Cover Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "plug-mat": fc.Material(id="plug-mat", name="Plug Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    selection = _apply_step_design_variant_selection(root, parts, materials, extraction, options)

    assert extraction.records[3].effectivity_kind == "date"
    assert extraction.records[3].effectivity_range == ("2026-01-01", "2026-12-31")
    assert selection.status == "applied"
    assert selection.matched_records == ("step_variant_13",)
    assert [child.name for child in root.children] == ["Inspection Cover"]
    assert set(parts) == {"cover"}


def test_step_design_variant_selection_gates_effectivity_assignment_targets(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('inspection cover','select inspection cover',#1);\n"
        "#11=SERIAL_NUMBERED_EFFECTIVITY('SN-A-001','SN-A-099',#10);\n"
        "#20=EFFECTIVITY_ASSIGNMENT(#11,#10);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="cover-node", name="Inspection Cover", part_id="cover"),
            fc.Node(id="plug-node", name="Blanking Plug", part_id="plug"),
        ],
    )
    parts = {
        "cover": fc.Part(
            id="cover",
            name="Inspection Cover",
            material_ids=["cover-mat"],
            metadata={"source_name": "inspection cover"},
        ),
        "plug": fc.Part(
            id="plug",
            name="Blanking Plug",
            material_ids=["plug-mat"],
            metadata={"source_name": "blanking plug"},
        ),
    }
    materials = {
        "cover-mat": fc.Material(id="cover-mat", name="Cover Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "plug-mat": fc.Material(id="plug-mat", name="Plug Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    serial_options = StepReadOptions(design_variant_selection=("SN-A-050",))
    serial_extraction = _extract_step_design_variants(source, serial_options)
    serial_root = root.copy()
    serial_parts = {key: part.copy() for key, part in parts.items()}
    serial_materials = dict(materials)

    serial_selection = _apply_step_design_variant_selection(
        serial_root,
        serial_parts,
        serial_materials,
        serial_extraction,
        serial_options,
    )

    assert serial_extraction.summary["effectivity_records"] == 2
    assert serial_extraction.summary["conditional_records"] == 1
    assert serial_extraction.records[2].condition_operator == "effectivity_assignment"
    assert serial_selection.status == "applied"
    assert serial_selection.matched_records == ("step_variant_20",)
    assert [child.name for child in serial_root.children] == ["Inspection Cover"]
    assert set(serial_parts) == {"cover"}

    target_options = StepReadOptions(design_variant_selection=("inspection cover",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]


def test_step_design_variant_selection_requires_supported_boolean_conditions(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('left hand','left condition',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('premium trim','premium condition',#1);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('left premium package','select left housing',#1);\n"
        "#13=CONFIGURATION_DESIGN(#12,#10);\n"
        "#20=AND_EXPRESSION((#10,#11));\n"
        "#21=CONDITIONAL_CONFIGURATION('left premium condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="left-node", name="Left Housing", part_id="left"),
            fc.Node(id="right-node", name="Right Housing", part_id="right"),
        ],
    )
    parts = {
        "left": fc.Part(
            id="left",
            name="Left Housing",
            material_ids=["left-mat"],
            metadata={"source_name": "left housing"},
        ),
        "right": fc.Part(
            id="right",
            name="Right Housing",
            material_ids=["right-mat"],
            metadata={"source_name": "right housing"},
        ),
    }
    materials = {
        "left-mat": fc.Material(id="left-mat", name="Left Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "right-mat": fc.Material(id="right-mat", name="Right Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    partial_options = StepReadOptions(design_variant_selection=("left hand",))
    partial_extraction = _extract_step_design_variants(source, partial_options)
    partial_selection = _apply_step_design_variant_selection(
        root.copy(),
        dict(parts),
        dict(materials),
        partial_extraction,
        partial_options,
    )

    assert partial_extraction.summary["conditional_records"] == 2
    assert partial_extraction.records[4].condition_operator == "and"
    assert partial_selection.status == "unmatched_geometry"
    assert partial_selection.matched_records == ()
    assert "condition expression was not satisfied" in partial_selection.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("left premium package",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]

    full_options = StepReadOptions(design_variant_selection=("left hand", "premium trim"))
    full_extraction = _extract_step_design_variants(source, full_options)
    full_root = root.copy()
    full_parts = dict(parts)
    full_materials = dict(materials)

    full_selection = _apply_step_design_variant_selection(
        full_root,
        full_parts,
        full_materials,
        full_extraction,
        full_options,
    )

    assert full_selection.status == "applied"
    assert full_selection.matched_records == ("step_variant_20", "step_variant_21")
    assert "left housing" in [term.lower() for term in full_selection.selector_terms]
    assert [child.name for child in full_root.children] == ["Left Housing"]
    assert set(full_parts) == {"left"}


def test_step_design_variant_selection_supports_not_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('left hand','left condition',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('premium trim','premium condition',#1);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('left standard package','select standard housing',#1);\n"
        "#20=NOT_EXPRESSION(#11);\n"
        "#21=AND_EXPRESSION((#10,#20));\n"
        "#22=CONDITIONAL_CONFIGURATION('left standard condition',#21,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="standard-node", name="Standard Housing", part_id="standard"),
            fc.Node(id="premium-node", name="Premium Housing", part_id="premium"),
        ],
    )
    parts = {
        "standard": fc.Part(
            id="standard",
            name="Standard Housing",
            material_ids=["standard-mat"],
            metadata={"source_name": "standard housing"},
        ),
        "premium": fc.Part(
            id="premium",
            name="Premium Housing",
            material_ids=["premium-mat"],
            metadata={"source_name": "premium housing"},
        ),
    }
    materials = {
        "standard-mat": fc.Material(id="standard-mat", name="Standard Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "premium-mat": fc.Material(id="premium-mat", name="Premium Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }
    standard_options = StepReadOptions(design_variant_selection=("left hand",))
    standard_extraction = _extract_step_design_variants(source, standard_options)
    standard_root = root.copy()
    standard_parts = {key: part.copy() for key, part in parts.items()}
    standard_materials = dict(materials)

    standard_selection = _apply_step_design_variant_selection(
        standard_root,
        standard_parts,
        standard_materials,
        standard_extraction,
        standard_options,
    )

    assert standard_extraction.summary["conditional_records"] == 3
    assert standard_selection.status == "applied"
    assert standard_selection.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in standard_root.children] == ["Standard Housing"]
    assert set(standard_parts) == {"standard"}

    premium_options = StepReadOptions(design_variant_selection=("left hand", "premium trim"))
    premium_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, premium_options),
        premium_options,
    )

    assert premium_selection.status == "unmatched_geometry"
    assert premium_selection.matched_records == ()


def test_step_design_variant_selection_evaluates_boolean_literals(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('service package','select service panel',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('blocked package','select blocked panel',#1);\n"
        "#20=BOOLEAN_LITERAL(.T.);\n"
        "#21=BOOLEAN_LITERAL(.F.);\n"
        "#30=CONDITIONAL_CONFIGURATION('service condition',#20,#10);\n"
        "#31=CONDITIONAL_CONFIGURATION('blocked condition',#21,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="service-node", name="Service Panel", part_id="service"),
            fc.Node(id="blocked-node", name="Blocked Panel", part_id="blocked"),
        ],
    )
    parts = {
        "service": fc.Part(
            id="service",
            name="Service Panel",
            material_ids=["service-mat"],
            metadata={"source_name": "service panel"},
        ),
        "blocked": fc.Part(
            id="blocked",
            name="Blocked Panel",
            material_ids=["blocked-mat"],
            metadata={"source_name": "blocked panel"},
        ),
    }
    materials = {
        "service-mat": fc.Material(id="service-mat", name="Service Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "blocked-mat": fc.Material(id="blocked-mat", name="Blocked Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    service_options = StepReadOptions(design_variant_selection=("service package",))
    service_extraction = _extract_step_design_variants(source, service_options)
    service_root = root.copy()
    service_parts = {key: part.copy() for key, part in parts.items()}
    service_materials = dict(materials)

    service_selection = _apply_step_design_variant_selection(
        service_root,
        service_parts,
        service_materials,
        service_extraction,
        service_options,
    )

    assert service_extraction.summary["conditional_records"] == 4
    assert service_extraction.records[2].condition_operator == "literal"
    assert service_extraction.records[2].condition_value is True
    assert service_extraction.records[3].condition_value is False
    assert service_selection.status == "applied"
    assert service_selection.matched_records == ("step_variant_30",)
    assert [child.name for child in service_root.children] == ["Service Panel"]
    assert set(service_parts) == {"service"}

    blocked_options = StepReadOptions(design_variant_selection=("blocked package",))
    blocked_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection.status == "unmatched_geometry"
    assert blocked_selection.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection.warnings[0]


@pytest.mark.parametrize(
    ("variable_record", "expected_kind"),
    (
        ("#10=BOOLEAN_VARIABLE('service enabled');\n", "boolean_variable"),
        ("#10=MATHS_BOOLEAN_VARIABLE(#1,'service enabled');\n", "maths_boolean_variable"),
    ),
)
def test_step_design_variant_selection_evaluates_boolean_variables(
    tmp_path: Path,
    variable_record: str,
    expected_kind: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        f"{variable_record}"
        "#11=PRODUCT_CONCEPT_FEATURE('service package','select service panel',#1);\n"
        "#20=CONDITIONAL_CONFIGURATION('service variable condition',#10,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="service-node", name="Service Panel", part_id="service"),
            fc.Node(id="blocked-node", name="Blocked Panel", part_id="blocked"),
        ],
    )
    parts = {
        "service": fc.Part(
            id="service",
            name="Service Panel",
            material_ids=["service-mat"],
            metadata={"source_name": "service panel"},
        ),
        "blocked": fc.Part(
            id="blocked",
            name="Blocked Panel",
            material_ids=["blocked-mat"],
            metadata={"source_name": "blocked panel"},
        ),
    }
    materials = {
        "service-mat": fc.Material(id="service-mat", name="Service Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "blocked-mat": fc.Material(id="blocked-mat", name="Blocked Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    variable_options = StepReadOptions(design_variant_selection=("service enabled",))
    variable_extraction = _extract_step_design_variants(source, variable_options)
    variable_root = root.copy()
    variable_parts = {key: part.copy() for key, part in parts.items()}
    variable_materials = dict(materials)

    variable_selection = _apply_step_design_variant_selection(
        variable_root,
        variable_parts,
        variable_materials,
        variable_extraction,
        variable_options,
    )

    assert variable_extraction.summary["conditional_records"] == 2
    assert variable_extraction.records[0].kind == expected_kind
    assert variable_extraction.records[0].condition_operator == "variable"
    assert variable_selection.status == "applied"
    assert variable_selection.matched_records == ("step_variant_10", "step_variant_20")
    assert [child.name for child in variable_root.children] == ["Service Panel"]
    assert set(variable_parts) == {"service"}

    target_options = StepReadOptions(design_variant_selection=("service package",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]


def test_step_design_variant_selection_evaluates_equals_expression(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('left hand','left condition',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('premium trim','premium condition',#1);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('matched package','select matched panel',#1);\n"
        "#20=EQUALS_EXPRESSION(#10,#11);\n"
        "#21=CONDITIONAL_CONFIGURATION('matched condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="matched-node", name="Matched Panel", part_id="matched"),
            fc.Node(id="fallback-node", name="Fallback Panel", part_id="fallback"),
        ],
    )
    parts = {
        "matched": fc.Part(
            id="matched",
            name="Matched Panel",
            material_ids=["matched-mat"],
            metadata={"source_name": "matched panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback panel"},
        ),
    }
    materials = {
        "matched-mat": fc.Material(id="matched-mat", name="Matched Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    partial_options = StepReadOptions(design_variant_selection=("left hand",))
    partial_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, partial_options),
        partial_options,
    )

    assert partial_selection.status == "unmatched_geometry"
    assert partial_selection.matched_records == ()
    assert "condition expression was not satisfied" in partial_selection.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("matched package",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]

    full_options = StepReadOptions(design_variant_selection=("left hand", "premium trim"))
    full_extraction = _extract_step_design_variants(source, full_options)
    full_root = root.copy()
    full_parts = {key: part.copy() for key, part in parts.items()}
    full_materials = dict(materials)

    full_selection = _apply_step_design_variant_selection(
        full_root,
        full_parts,
        full_materials,
        full_extraction,
        full_options,
    )

    assert full_extraction.records[3].kind == "equals_expression"
    assert full_extraction.records[3].condition_operator == "equals"
    assert full_extraction.summary["conditional_records"] == 2
    assert full_selection.status == "applied"
    assert full_selection.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in full_root.children] == ["Matched Panel"]
    assert set(full_parts) == {"matched"}


def test_step_design_variant_selection_evaluates_comparison_not_equal(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('left hand','left condition',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('premium trim','premium condition',#1);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('mismatch package','select mismatch panel',#1);\n"
        "#20=COMPARISON_NOT_EQUAL(#10,#11);\n"
        "#21=CONDITIONAL_CONFIGURATION('mismatch condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="mismatch-node", name="Mismatch Panel", part_id="mismatch"),
            fc.Node(id="fallback-node", name="Fallback Panel", part_id="fallback"),
        ],
    )
    parts = {
        "mismatch": fc.Part(
            id="mismatch",
            name="Mismatch Panel",
            material_ids=["mismatch-mat"],
            metadata={"source_name": "mismatch panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback panel"},
        ),
    }
    materials = {
        "mismatch-mat": fc.Material(id="mismatch-mat", name="Mismatch Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    mismatch_options = StepReadOptions(design_variant_selection=("left hand",))
    mismatch_extraction = _extract_step_design_variants(source, mismatch_options)
    mismatch_root = root.copy()
    mismatch_parts = {key: part.copy() for key, part in parts.items()}
    mismatch_materials = dict(materials)

    mismatch_selection = _apply_step_design_variant_selection(
        mismatch_root,
        mismatch_parts,
        mismatch_materials,
        mismatch_extraction,
        mismatch_options,
    )

    assert mismatch_extraction.records[3].kind == "comparison_not_equal"
    assert mismatch_extraction.records[3].condition_operator == "not_equals"
    assert mismatch_selection.status == "applied"
    assert mismatch_selection.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in mismatch_root.children] == ["Mismatch Panel"]
    assert set(mismatch_parts) == {"mismatch"}

    equal_options = StepReadOptions(design_variant_selection=("left hand", "premium trim"))
    equal_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, equal_options),
        equal_options,
    )

    assert equal_selection.status == "unmatched_geometry"
    assert equal_selection.matched_records == ()
    assert "condition expression was not satisfied" in equal_selection.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("mismatch package",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]


def test_step_design_variant_selection_gates_conditional_concept_feature_label(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('left hand','left condition',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('premium trim','premium condition',#1);\n"
        "#20=AND_EXPRESSION((#10,#11));\n"
        "#30=CONDITIONAL_CONCEPT_FEATURE('left premium package','select premium housing',#20);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="premium-node", name="Premium Housing", part_id="premium"),
            fc.Node(id="standard-node", name="Standard Housing", part_id="standard"),
        ],
    )
    parts = {
        "premium": fc.Part(
            id="premium",
            name="Premium Housing",
            material_ids=["premium-mat"],
            metadata={"source_name": "premium housing"},
        ),
        "standard": fc.Part(
            id="standard",
            name="Standard Housing",
            material_ids=["standard-mat"],
            metadata={"source_name": "standard housing"},
        ),
    }
    materials = {
        "premium-mat": fc.Material(id="premium-mat", name="Premium Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "standard-mat": fc.Material(id="standard-mat", name="Standard Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    target_options = StepReadOptions(design_variant_selection=("left premium package",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]

    full_options = StepReadOptions(design_variant_selection=("left hand", "premium trim"))
    full_extraction = _extract_step_design_variants(source, full_options)
    full_root = root.copy()
    full_parts = {key: part.copy() for key, part in parts.items()}
    full_materials = dict(materials)

    full_selection = _apply_step_design_variant_selection(
        full_root,
        full_parts,
        full_materials,
        full_extraction,
        full_options,
    )

    assert full_extraction.records[3].kind == "conditional_concept_feature"
    assert full_extraction.records[3].condition_operator == "conditional"
    assert full_extraction.summary["product_concept_features"] == 3
    assert full_extraction.summary["conditional_records"] == 2
    assert full_selection.status == "applied"
    assert full_selection.matched_records == ("step_variant_20", "step_variant_30")
    assert [child.name for child in full_root.children] == ["Premium Housing"]
    assert set(full_parts) == {"premium"}


def test_step_design_variant_selection_supports_conditional_effectivity_assignment(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONDITIONAL_EFFECTIVITY('service window','enabled by release condition','apply','qualify');\n"
        "#11=PRODUCT_CONCEPT_FEATURE('service package','select service panel',#1);\n"
        "#20=CONFIGURED_EFFECTIVITY_ASSIGNMENT(#10,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="service-node", name="Service Panel", part_id="service"),
            fc.Node(id="blank-node", name="Blank Panel", part_id="blank"),
        ],
    )
    parts = {
        "service": fc.Part(
            id="service",
            name="Service Panel",
            material_ids=["service-mat"],
            metadata={"source_name": "service panel"},
        ),
        "blank": fc.Part(
            id="blank",
            name="Blank Panel",
            material_ids=["blank-mat"],
            metadata={"source_name": "blank panel"},
        ),
    }
    materials = {
        "service-mat": fc.Material(id="service-mat", name="Service Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "blank-mat": fc.Material(id="blank-mat", name="Blank Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    effectivity_options = StepReadOptions(design_variant_selection=("service window",))
    effectivity_extraction = _extract_step_design_variants(source, effectivity_options)
    effectivity_root = root.copy()
    effectivity_parts = {key: part.copy() for key, part in parts.items()}
    effectivity_materials = dict(materials)

    effectivity_selection = _apply_step_design_variant_selection(
        effectivity_root,
        effectivity_parts,
        effectivity_materials,
        effectivity_extraction,
        effectivity_options,
    )

    assert effectivity_extraction.records[0].kind == "conditional_effectivity"
    assert effectivity_extraction.records[0].effectivity_kind == "generic"
    assert effectivity_extraction.records[2].condition_operator == "effectivity_assignment"
    assert effectivity_extraction.summary["effectivity_records"] == 2
    assert effectivity_extraction.summary["conditional_records"] == 1
    assert effectivity_selection.status == "applied"
    assert effectivity_selection.matched_records == ("step_variant_20",)
    assert [child.name for child in effectivity_root.children] == ["Service Panel"]
    assert set(effectivity_parts) == {"service"}

    target_options = StepReadOptions(design_variant_selection=("service package",))
    target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, target_options),
        target_options,
    )

    assert target_selection.status == "unmatched_geometry"
    assert target_selection.matched_records == ()
    assert "condition expression was not satisfied" in target_selection.warnings[0]


def test_step_import_cleanup_actions_cover_construction_only_shapes() -> None:
    point_counts = _ShapeTopologyCounts(vertices=3)
    line_counts = _ShapeTopologyCounts(vertices=4, edges=2)
    brep_counts = _ShapeTopologyCounts(vertices=8, edges=12, faces=6)

    assert _loaded_representation(point_counts) == "construction_points"
    assert _loaded_representation(line_counts) == "construction_lines"
    assert _loaded_representation(brep_counts) == "brep"
    assert _cleanup_action(point_counts, StepReadOptions(delete_free_vertices=True)) == "delete_free_vertices"
    assert _cleanup_action(line_counts, StepReadOptions(delete_lines=True)) == "delete_lines"
    assert _cleanup_action(line_counts, StepReadOptions(construction_curve_policy="delete")) == "delete_lines"
    assert _cleanup_action(line_counts, StepReadOptions(construction_curve_policy="tessellate_tubes")) is None
    assert _cleanup_action(brep_counts, StepReadOptions(delete_free_vertices=True, delete_lines=True)) is None


def test_mixed_construction_curve_shape_extracts_edges_not_used_by_faces() -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    construction_edge = BRepBuilderAPI_MakeEdge(gp_Pnt(2.0, 0.0, 0.0), gp_Pnt(3.0, 0.0, 0.0)).Edge()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, box)
    builder.Add(compound, construction_edge)

    curve_shape = _mixed_construction_curve_shape(compound, _shape_topology_counts(compound))

    assert curve_shape is not None
    assert _shape_topology_counts(curve_shape) == _ShapeTopologyCounts(vertices=2, edges=1, faces=0)
    assert _mixed_construction_curve_shape(box, _shape_topology_counts(box)) is None


def test_mixed_construction_curve_metadata_records_policy_and_action() -> None:
    metadata = _mixed_construction_curve_metadata(
        StepReadOptions(construction_curve_policy="tessellate_tubes"),
        "split",
        _ShapeTopologyCounts(vertices=2, edges=1),
    )

    assert metadata == {
        "mixed_construction_curve_policy": "tessellate_tubes",
        "mixed_construction_curve_action": "split",
        "mixed_construction_curve_vertices": "2",
        "mixed_construction_curve_edges": "1",
        "mixed_construction_curve_split": "true",
    }


def test_mixed_construction_curve_node_preserves_policy_metadata() -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.gp import gp_Pnt

    curve_shape = BRepBuilderAPI_MakeEdge(gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 0.0, 0.0)).Edge()
    counts = _shape_topology_counts(curve_shape)
    parts: dict[str, fc.Part] = {}
    cleanup = _ImportCleanupStats()
    options = StepReadOptions(construction_curve_policy="tessellate_tubes", construction_curve_tube_radius=0.025)

    node = _build_mixed_construction_curve_node(
        source_identity="panel.step",
        occurrence_path="root/1",
        label_entry="0:1",
        part_entry="0:1:1",
        source_name="Panel",
        shape=curve_shape,
        counts=counts,
        material_ids=["mat-default"],
        part_index={},
        parts=parts,
        options=options,
        cleanup=cleanup,
    )

    assert cleanup.to_dict()["construction_line_parts"] == 1
    assert node.part_id in parts
    assert node.metadata["mixed_construction_curve_split"] == "true"
    part = parts[str(node.part_id)]
    assert part.name == "Panel Construction Curves"
    assert part.metadata["loaded_representation"] == "construction_lines"
    assert part.metadata["mixed_construction_curve_split"] == "true"
    assert part.metadata["construction_curve_policy"] == "tessellate_tubes"
    assert part.metadata["construction_curve_tube_radius"] == "0.025"


def test_step_read_options_normalize_construction_curve_policy() -> None:
    options = StepReadOptions(construction_curve_policy="tessellate-tubes")

    assert options.construction_curve_policy == "tessellate_tubes"

    with pytest.raises(ValueError, match="construction_curve_tube_radius"):
        StepReadOptions(construction_curve_policy="tessellate_tubes", construction_curve_tube_radius=0.0)


def test_step_import_decisions_report_requested_effective_states() -> None:
    cleanup = _ImportCleanupStats()
    cleanup.record_deleted("delete_lines", _ShapeTopologyCounts(vertices=4, edges=2))
    space = _space_normalization(
        "millimetre",
        0.001,
        StepReadOptions(target_units="metre", target_up_axis="Y", target_handedness="right"),
    )

    decisions = _import_decisions(
        StepReadOptions(
            design_variants=True,
            multi_file=True,
            delete_free_vertices=True,
            delete_lines=True,
        ),
        _StepHeaderInfo(schema="AP242", pmi_present=True),
        pmi_count=0,
        unsupported_pmi_count=1,
        cleanup=cleanup,
        space=space,
    )

    assert decisions["pmi"]["state"] == "unsupported"
    assert decisions["design_variants"]["state"] == "not_present"
    assert decisions["design_variants"]["counts"]["records"] == 0
    assert decisions["multi_file"]["state"] == "unsupported"
    assert decisions["source_textures"]["state"] == "honored"
    assert decisions["material_library_mapping"]["state"] == "honored"
    assert decisions["delete_free_vertices"]["state"] == "honored"
    assert decisions["delete_free_vertices"]["counts"] == {"deleted_parts": 0, "deleted_vertices": 0}
    assert decisions["delete_lines"]["counts"] == {
        "deleted_parts": 1,
        "deleted_edges": 2,
        "deleted_vertices": 4,
    }
    assert decisions["construction_curves"]["requested"] == "delete"
    assert decisions["construction_curves"]["state"] == "honored"
    assert decisions["construction_curves"]["counts"]["deleted_parts"] == 1
    assert decisions["space_normalization"]["state"] == "honored"


def test_step_import_decisions_report_extracted_typed_pmi() -> None:
    space = _space_normalization("millimetre", 0.001, StepReadOptions())

    decisions = _import_decisions(
        StepReadOptions(pmi=True),
        _StepHeaderInfo(schema="AP242", pmi_present=True),
        pmi_count=2,
        unsupported_pmi_count=0,
        cleanup=_ImportCleanupStats(),
        space=space,
        pmi_semantic_graph_summary={
            "nodes": 4,
            "pmi_nodes": 2,
            "referenced_nodes": 2,
            "edges": 3,
            "missing_references": 0,
        },
    )

    assert decisions["pmi"]["state"] == "honored"
    assert decisions["pmi"]["effective"] is True
    assert decisions["pmi"]["counts"] == {
        "imported": 2,
        "unsupported": 0,
        "semantic_graph_nodes": 4,
        "semantic_graph_edges": 3,
        "semantic_graph_missing_references": 0,
    }


def test_step_import_decisions_report_detected_design_variants() -> None:
    space = _space_normalization("millimetre", 0.001, StepReadOptions())

    decisions = _import_decisions(
        StepReadOptions(design_variants=True),
        _StepHeaderInfo(schema="AP242", pmi_present=False),
        pmi_count=0,
        unsupported_pmi_count=0,
        cleanup=_ImportCleanupStats(),
        space=space,
        design_variant_summary={
            "records": 3,
            "configuration_items": 1,
            "product_concept_features": 1,
            "effectivity_records": 1,
            "conditional_records": 0,
        },
    )

    assert decisions["design_variants"]["state"] == "approximated"
    assert decisions["design_variants"]["effective"] is True
    assert decisions["design_variants"]["counts"] == {
        "records": 3,
        "configuration_items": 1,
        "product_concept_features": 1,
        "effectivity_records": 1,
        "conditional_records": 0,
    }


def test_step_source_texture_extraction_loads_sidecar_images_and_binds_single_material(tmp_path: Path) -> None:
    texture = tmp_path / "panel_baseColor.png"
    Image.new("RGBA", (4, 2), (128, 64, 32, 255)).save(texture)
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('panel_baseColor.png');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())
    material = fc.Material(id="paint", name="Paint", base_color=(1.0, 1.0, 1.0, 1.0))
    summary = _attach_source_textures_to_materials({"paint": material}, extraction.images)

    image = next(iter(extraction.images.values()))
    assert extraction.summary == {"references": 1, "resolved": 1, "missing": 0, "unsupported": 0, "unreadable": 0}
    assert image.mime_type == "image/png"
    assert (image.width, image.height) == (4, 2)
    assert image.metadata["source_texture_slot"] == "base_color"
    assert material.metadata["source_texture_base_color_image"] == image.id
    assert material.metadata["source_texture_slots"] == "base_color"
    assert summary == {"bound_images": 1, "bound_materials": 1, "unbound_images": 0}


def test_step_source_texture_extraction_reports_missing_references(tmp_path: Path) -> None:
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('missing_normal.jpg');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())

    assert extraction.images == {}
    assert extraction.summary == {"references": 1, "resolved": 0, "missing": 1, "unsupported": 0, "unreadable": 0}
    assert extraction.warnings == ["source texture reference could not be resolved: missing_normal.jpg"]


def test_step_material_library_json_maps_pbr_factors_and_textures(tmp_path: Path) -> None:
    texture = tmp_path / "steel_baseColor.png"
    Image.new("RGBA", (2, 2), (200, 210, 220, 255)).save(texture)
    library = tmp_path / "vendor-materials.json"
    library.write_text(
        """
{
  "materials": [
    {
      "materialName": "Brushed Steel",
      "baseColorFactor": [0.78, 0.8, 0.82, 1.0],
      "metallicFactor": 1.0,
      "roughnessFactor": 0.22,
      "textures": {"baseColor": "steel_baseColor.png"}
    }
  ]
}
""",
        encoding="utf-8",
    )
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.json');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    materials = {
        "steel": fc.Material(id="steel", name="Brushed Steel", base_color=(0.75, 0.75, 0.75, 1.0)),
    }

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())
    summary = _apply_material_libraries_to_materials(materials, extraction)
    material = materials["steel"]

    assert extraction.summary == {
        "references": 1,
        "resolved": 1,
        "missing": 0,
        "unsupported": 0,
        "unreadable": 0,
        "materials": 1,
        "textures": 1,
        "texture_missing": 0,
        "texture_unreadable": 0,
    }
    assert summary == {
        "library_materials": 1,
        "matched_library_materials": 1,
        "unmatched_library_materials": 0,
        "applied_materials": 1,
        "bound_textures": 1,
    }
    assert material.base_color == pytest.approx((0.78, 0.8, 0.82, 1.0))
    assert material.metallic == pytest.approx(1.0)
    assert material.roughness == pytest.approx(0.22)
    assert material.metadata["material_library_matched"] == "true"
    assert material.metadata["source_texture_base_color_image"] in extraction.images


def test_step_material_library_mtl_can_be_supplied_explicitly(tmp_path: Path) -> None:
    texture = tmp_path / "aluminum.png"
    Image.new("RGB", (1, 1), (160, 170, 180)).save(texture)
    library = tmp_path / "vendor.mtl"
    library.write_text(
        """
newmtl Anodized Aluminum
Kd 0.55 0.6 0.65
Pm 1
Pr 0.18
map_Kd aluminum.png
""",
        encoding="utf-8",
    )
    source = tmp_path / "panel.step"
    source.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")
    materials = {
        "aluminum": fc.Material(id="aluminum", name="Anodized Aluminum", base_color=(0.75, 0.75, 0.75, 1.0)),
    }

    extraction = _extract_material_libraries(
        source,
        "panel.step",
        StepReadOptions(material_library_paths=(str(library),)),
    )
    summary = _apply_material_libraries_to_materials(materials, extraction)
    material = materials["aluminum"]

    assert extraction.summary["references"] == 1
    assert extraction.summary["resolved"] == 1
    assert extraction.summary["materials"] == 1
    assert extraction.summary["textures"] == 1
    assert summary["applied_materials"] == 1
    assert material.base_color == pytest.approx((0.55, 0.6, 0.65, 1.0))
    assert material.metallic == pytest.approx(1.0)
    assert material.roughness == pytest.approx(0.18)
    assert material.metadata["source_texture_base_color_image"] in extraction.images


def test_step_material_library_zip_container_maps_pbr_and_textures(tmp_path: Path) -> None:
    image_buffer = BytesIO()
    Image.new("RGBA", (2, 1), (180, 90, 40, 255)).save(image_buffer, format="PNG")
    library = tmp_path / "vendor-materials.zip"
    payload = {
        "materials": [
            {
                "materialName": "Burnt Copper",
                "baseColorFactor": [0.7, 0.35, 0.16, 1.0],
                "metallicFactor": 1.0,
                "roughnessFactor": 0.24,
                "textures": {"baseColor": "../textures/copper_baseColor.png"},
            }
        ]
    }
    with zipfile.ZipFile(library, "w") as archive:
        archive.writestr("metadata/manifest.json", '{"name":"not a material library"}')
        archive.writestr("materials/vendor.json", json.dumps(payload))
        archive.writestr("textures/copper_baseColor.png", image_buffer.getvalue())
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.zip');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    materials = {
        "copper": fc.Material(id="copper", name="Burnt Copper", base_color=(0.6, 0.3, 0.2, 1.0)),
    }

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())
    summary = _apply_material_libraries_to_materials(materials, extraction)
    material = materials["copper"]
    image_id = material.metadata["source_texture_base_color_image"]
    image = extraction.images[image_id]

    assert extraction.summary == {
        "references": 1,
        "resolved": 1,
        "missing": 0,
        "unsupported": 0,
        "unreadable": 0,
        "materials": 1,
        "textures": 1,
        "texture_missing": 0,
        "texture_unreadable": 0,
    }
    assert summary["applied_materials"] == 1
    assert summary["bound_textures"] == 1
    assert material.base_color == pytest.approx((0.7, 0.35, 0.16, 1.0))
    assert material.metallic == pytest.approx(1.0)
    assert material.roughness == pytest.approx(0.24)
    assert material.metadata["material_library_path"].endswith("vendor-materials.zip!/materials/vendor.json")
    assert material.metadata["material_library_container"] == str(library)
    assert image.mime_type == "image/png"
    assert image.metadata["source_texture_path"].endswith("vendor-materials.zip!/textures/copper_baseColor.png")
    assert "vendor-materials.zip!/" in image.metadata["source_texture_identity"]


def test_read_step_many_namespaces_members_and_prefixes_member_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, bool]] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        calls.append((source, source_identity, options.multi_file))
        material = fc.Material(
            id="mat",
            name="Paint",
            base_color=(0.8, 0.2, 0.1, 1.0),
            metadata={"source_texture_base_color_image": "img"},
        )
        image = fc.ImageResource(
            id="img",
            name=f"{source.stem}.png",
            mime_type="image/png",
            data=b"png",
            width=1,
            height=1,
        )
        report = Report(source_path=str(source))
        report.input_stats = {
            "nodes": 2,
            "parts": 1,
            "occurrences": 1,
            "materials": 1,
            "images": 1,
            "vertices": 0,
            "triangles": 0,
        }
        report.add_warning(f"{source.stem} member warning")
        report.add_step("import", options={"format": "STEP"}, after=report.input_stats)
        return fc.Asset(
            root=fc.Node(
                id="root",
                name=source.stem,
                children=[fc.Node(id="occurrence", name="occurrence", part_id="part")],
            ),
            parts={"part": fc.Part(id="part", name="Part", material_ids=["mat"])},
            materials={"mat": material},
            images={"img": image},
            metadata={"source": str(source), "source_identity": source_identity},
            report=report,
        )

    monkeypatch.setattr(step_io, "_read_step_path", fake_read_step_path)

    first = tmp_path / "assembly-a.step"
    second = tmp_path / "assembly-b.step"
    asset = read_step_many([first, second], options=StepReadOptions(multi_file=True))
    repeated = read_step_many([first, second], options=StepReadOptions(multi_file=True))

    assert [call[0] for call in calls[:2]] == [first, second]
    assert [call[2] for call in calls[:2]] == [False, False]
    assert asset.source_path is None
    assert asset.report.source_path is None
    assert asset.stats()["parts"] == 2
    assert asset.stats()["occurrences"] == 2
    assert "part" not in asset.parts
    assert "mat" not in asset.materials
    assert "img" not in asset.images
    assert sorted(asset.parts) == sorted(repeated.parts)
    assert sorted(asset.materials) == sorted(repeated.materials)
    assert sorted(asset.images) == sorted(repeated.images)
    for part in asset.parts.values():
        assert part.material_ids
        assert part.material_ids[0] in asset.materials
        assert part.metadata["multi_file_member_part_id"] == "part"
    for material in asset.materials.values():
        image_id = material.metadata["source_texture_base_color_image"]
        assert image_id in asset.images
        assert material.metadata["multi_file_member_material_id"] == "mat"
    assert [child.metadata["multi_file_member_index"] for child in asset.root.children] == [1, 2]
    assert all("member warning" in warning for warning in asset.report.warnings)

    import_step = asset.report.steps[0]
    assert import_step.name == "import"
    assert import_step.options["multi_file"] is True
    assert import_step.options["member_count"] == 2
    assert import_step.options["failed_member_count"] == 0
    assert import_step.options["import_decisions"]["multi_file"]["state"] == "honored"
    assert asset.metadata["multi_file_import"]["member_count"] == 2


def test_read_step_many_can_continue_after_member_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity, options
        if source.name == "bad.step":
            raise RuntimeError("boom")
        report = Report(source_path=str(source))
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem, children=[fc.Node(id="occurrence", name="occurrence")]),
            report=report,
        )

    monkeypatch.setattr(step_io, "_read_step_path", fake_read_step_path)

    asset = read_step_many(
        [tmp_path / "good.step", tmp_path / "bad.step"],
        options=StepReadOptions(),
        continue_on_error=True,
    )

    import_step = asset.report.steps[0]
    assert import_step.options["member_count"] == 1
    assert import_step.options["failed_member_count"] == 1
    assert import_step.options["import_decisions"]["multi_file"]["state"] == "approximated"
    assert "bad.step" in asset.report.warnings[0]
    assert "boom" in asset.report.warnings[0]

    with pytest.raises(RuntimeError, match="boom"):
        read_step_many([tmp_path / "good.step", tmp_path / "bad.step"])


def test_read_step_multi_file_resolves_master_external_reference_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "parts" / "child.stp"
    child.parent.mkdir()
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('parts/child.stp');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")
    calls: list[tuple[Path, bool]] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        calls.append((source, options.multi_file))
        report = Report(source_path=str(source))
        report.add_step(
            "import",
            options={
                "format": "STEP",
                "read_options": options.to_dict(),
                "import_decisions": {"multi_file": {"state": "disabled"}},
            },
        )
        return fc.Asset(
            root=fc.Node(
                id="root",
                name=source.stem,
                children=[fc.Node(id="occurrence", name=source.stem, part_id="part")],
            ),
            parts={"part": fc.Part(id="part", name=source.stem.title())},
            metadata={"source": str(source), "import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_io, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert calls == [(master, False), (child.resolve(), False)]
    assert asset.stats()["parts"] == 2
    assert asset.root.metadata["external_reference_graph"] == "true"
    assert asset.metadata["external_reference_graph"]["summary"] == {
        "references": 1,
        "resolved": 1,
        "missing": 0,
        "unsupported": 0,
        "sources": 2,
        "resolved_sources": 1,
        "member_sources": 2,
        "resolved_occurrences": 1,
    }
    import_options = asset.report.steps[0].options
    assert import_options["external_reference_graph"]["summary"]["resolved_sources"] == 1
    assert import_options["import_decisions"]["multi_file"]["state"] == "honored"
    assert import_options["import_decisions"]["multi_file"]["effective"] is True


def test_read_step_multi_file_reports_missing_master_external_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = tmp_path / "master.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('missing.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    calls: list[tuple[Path, bool]] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity
        calls.append((source, options.multi_file))
        report = Report(source_path=str(source))
        report.add_step("import", options={"read_options": options.to_dict(), "import_decisions": {}})
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem),
            metadata={"import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_io, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert calls == [(master, False)]
    assert asset.metadata["external_reference_graph"]["summary"]["missing"] == 1
    assert asset.metadata["import_decisions"]["multi_file"]["state"] == "missing_sources"
    assert asset.report.steps[0].options["read_options"]["multi_file"] is True
    assert "missing.step" in asset.report.warnings[0]


def test_external_step_reference_graph_resolves_nested_references_once(tmp_path: Path) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "child.step"
    leaf = tmp_path / "leaf.stp"
    master.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('child.step');\n"
        "#2=DOCUMENT_FILE('missing.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('leaf.stp');\n"
        "#2=DOCUMENT_FILE('master.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    leaf.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")

    graph = _resolve_step_external_reference_graph(master)

    assert graph.sources == [master, child.resolve(), leaf.resolve()]
    assert graph.summary() == {
        "references": 4,
        "resolved": 3,
        "missing": 1,
        "unsupported": 0,
        "sources": 3,
        "resolved_sources": 2,
        "member_sources": 3,
        "resolved_occurrences": 2,
    }
    assert any("missing.step" in warning for warning in graph.warnings)


def test_external_step_reference_graph_preserves_duplicate_occurrences(tmp_path: Path) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "bolt.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('bolt.step');\n"
        "#2=DOCUMENT_FILE('bolt.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")

    graph = _resolve_step_external_reference_graph(master)

    assert graph.sources == [master, child.resolve()]
    assert graph.member_sources == [master, child.resolve(), child.resolve()]
    assert graph.summary()["resolved_sources"] == 1
    assert graph.summary()["resolved_occurrences"] == 2


def test_external_step_reference_import_preserves_duplicate_occurrences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "bolt.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('bolt.step');\n"
        "#2=DOCUMENT_FILE('bolt.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity
        assert options.multi_file is False
        calls.append(source)
        report = Report(source_path=str(source))
        report.add_step("import", options={"read_options": options.to_dict(), "import_decisions": {}})
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem),
            metadata={"import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_io, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert calls == [master, child.resolve(), child.resolve()]
    assert len(asset.root.children) == 3
    assert asset.metadata["external_reference_graph"]["summary"]["resolved_sources"] == 1
    assert asset.metadata["external_reference_graph"]["summary"]["resolved_occurrences"] == 2


def test_material_library_mapping_applies_known_cad_material_rules() -> None:
    spec = _color_material_spec((0.75, 0.75, 0.75, 1.0))
    steel = _apply_material_library_mapping(
        _CadMaterialSpec(
            name="Stainless Steel 304",
            base_color=spec.base_color,
            metadata=(("cad_material_source", "xde_visual_material"),),
        ),
        StepReadOptions(),
    )

    assert steel.metallic == pytest.approx(1.0)
    assert steel.roughness == pytest.approx(0.32)
    assert steel.metadata_dict()["pbr_mapping_status"] == "library_rule"
    assert steel.metadata_dict()["cad_material_mapping_rule"] == "stainless"


def test_loaded_representation_report_lists_parts_and_deleted_nodes() -> None:
    asset = fc.Asset(
        root=fc.Node(
            id="root",
            name="root",
            children=[
                fc.Node(id="node-a", name="Part A", part_id="part-a"),
                fc.Node(
                    id="node-deleted",
                    name="construction line",
                    metadata={
                        "loaded_representation": "construction_lines",
                        "import_cleanup": "delete_lines",
                        "source_vertices": "2",
                        "source_edges": "1",
                        "source_faces": "0",
                    },
                ),
            ],
        ),
        parts={
            "part-a": fc.Part(
                id="part-a",
                name="Part A",
                metadata={
                    "loaded_representation": "brep",
                    "source_vertices": "8",
                    "source_edges": "12",
                    "source_faces": "6",
                    "source_name": "Source Part A",
                },
            )
        },
    )

    report = _loaded_representation_report(asset)

    assert report["summary"] == {
        "brep_parts": 1,
        "construction_point_parts": 0,
        "construction_line_parts": 0,
        "empty_shape_parts": 0,
        "unknown_parts": 0,
        "deleted_nodes": 1,
        "deleted_free_vertex_nodes": 0,
        "deleted_line_nodes": 1,
    }
    assert report["parts"] == [
        {
            "part_id": "part-a",
            "name": "Part A",
            "loaded_representation": "brep",
            "cleanup_action": "preserved",
            "source_vertices": 8,
            "source_edges": 12,
            "source_faces": 6,
            "source_name": "Source Part A",
        }
    ]
    assert report["deleted_nodes"] == [
        {
            "node_id": "node-deleted",
            "name": "construction line",
            "loaded_representation": "construction_lines",
            "cleanup_action": "delete_lines",
            "source_vertices": 2,
            "source_edges": 1,
            "source_faces": 0,
        }
    ]


def test_step_space_normalization_builds_reported_root_transform() -> None:
    space = _space_normalization(
        "millimetre",
        0.001,
        StepReadOptions(target_units="metre", target_up_axis="Y", target_handedness="right"),
    )

    assert space.source_units == "millimetre"
    assert space.target_units == "metre"
    assert space.source_up_axis == "Z"
    assert space.target_up_axis == "Y"
    assert space.changed is True
    assert np.allclose(
        space.transform,
        np.array(
            [
                [0.001, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.001, 0.0],
                [0.0, -0.001, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    assert space.metadata()["changed"] is True


@pytest.mark.requires_ocp
def test_step_ap242_pmi_fixture_reports_unsupported_pmi_import() -> None:
    fixture = Path("tests/fixtures/raspberry-pi-camera-3-mount.step")

    asset = fc.read_step(fixture)
    import_step = asset.report.steps[0]

    assert import_step.options["pmi_present"] is True
    assert str(import_step.options["pmi_schema"]).startswith("AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF")
    assert import_step.options["pmi_count"] == 0
    assert import_step.options["unsupported_pmi_count"] == 1
    assert asset.metadata["pmi_present"] == "true"
    assert asset.metadata["pmi_import_status"] == "unsupported"
    assert import_step.warnings == asset.report.warnings
    assert "AP242 PMI" in import_step.warnings[0]


@pytest.mark.requires_ocp
def test_step_shape_fingerprints_are_stable_across_imports() -> None:
    fixture = Path("tests/fixtures/spool-clamp-lid.step")

    first = fc.read_step(fixture)
    second = fc.read_step(fixture)

    assert [part.metadata["shape_fingerprint"] for part in first.parts.values()] == [
        part.metadata["shape_fingerprint"] for part in second.parts.values()
    ]
