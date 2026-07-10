from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc
from fascat.io._import_base import (
    _ImportCleanupStats,
    _space_normalization,
    _StepHeaderInfo,
)
from fascat.io.step import variants as step_variants
from fascat.io.step.single import (
    _import_decisions,
)
from fascat.io.step.variants import (
    _apply_step_design_variant_selection,
    _extract_step_design_variants,
)
from fascat.options import StepReadOptions


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
        "conditional_records": 1,
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
    assert extraction.records[3].condition_operator == "effectivity_usage"
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


def test_step_design_variant_selector_terms_memoizes_duplicate_geometry_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = step_variants._StepDesignVariantRecord(
        id="step_variant_11",
        kind="product_concept_feature",
        entity="PRODUCT_CONCEPT_FEATURE",
        label="left housing",
        references=(),
    )
    conditional = step_variants._StepDesignVariantRecord(
        id="step_variant_20",
        kind="conditional_concept_feature",
        entity="CONDITIONAL_CONCEPT_FEATURE",
        label="left package",
        references=("#11", "#11"),
        condition_operator="conditional",
    )
    original_selector_terms = step_variants._design_variant_record_selector_terms
    target_selector_calls = 0

    def count_target_selector_terms(record: step_variants._StepDesignVariantRecord) -> tuple[str, ...]:
        nonlocal target_selector_calls
        if record.id == target.id:
            target_selector_calls += 1
        return original_selector_terms(record)

    monkeypatch.setattr(step_variants, "_design_variant_record_selector_terms", count_target_selector_terms)

    matched_records, selector_terms, condition_blocked = step_variants._design_variant_selector_terms(
        (target, conditional),
        ("left housing",),
    )

    assert matched_records == ("step_variant_20",)
    assert "left housing" in [term.lower() for term in selector_terms]
    assert condition_blocked is True
    assert target_selector_calls == 1


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


def test_step_design_variant_selection_resolves_referenced_time_interval_effectivity_range(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONFIGURATION_ITEM('service interval','time gated option',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('interval cover','select interval cover',#2);\n"
        "#12=CONFIGURATION_DESIGN(#10,#11);\n"
        "#13=TIME_INTERVAL_BASED_EFFECTIVITY('service interval',#14);\n"
        "#14=TIME_INTERVAL_WITH_BOUNDS('service bounds',#15,#16);\n"
        "#15=CALENDAR_DATE(2026,1,1);\n"
        "#16=CALENDAR_DATE(2026,31,12);\n"
        "#20=EFFECTIVITY_ASSIGNMENT(#13,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="cover-node", name="Interval Cover", part_id="cover"),
            fc.Node(id="plug-node", name="Blanking Plug", part_id="plug"),
        ],
    )
    parts = {
        "cover": fc.Part(
            id="cover",
            name="Interval Cover",
            material_ids=["cover-mat"],
            metadata={"source_name": "interval cover"},
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

    matching_options = StepReadOptions(design_variant_selection=("2026-06-01",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_selection = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    time_effectivity = next(
        record for record in matching_extraction.records if record.kind == "time_interval_based_effectivity"
    )
    assert time_effectivity.effectivity_kind == "time_interval"
    assert time_effectivity.effectivity_values == ("service interval", "2026-01-01", "2026-12-31")
    assert time_effectivity.effectivity_range == ("2026-01-01", "2026-12-31")
    assert matching_selection.status == "applied"
    assert matching_selection.matched_records == ("step_variant_20",)
    assert [child.name for child in matching_root.children] == ["Interval Cover"]
    assert set(matching_parts) == {"cover"}

    out_of_range_options = StepReadOptions(design_variant_selection=("2027-01-01",))
    out_of_range_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, out_of_range_options),
        out_of_range_options,
    )

    assert out_of_range_selection.status == "unmatched_geometry"
    assert out_of_range_selection.matched_records == ()

    target_options = StepReadOptions(design_variant_selection=("interval cover",))
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


def test_step_design_variant_selection_resolves_referenced_step_date_subtypes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=CONFIGURATION_ITEM('maintenance interval','alternate STEP date types',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('maintenance cover','select maintenance cover',#2);\n"
        "#12=CONFIGURATION_DESIGN(#10,#11);\n"
        "#13=TIME_INTERVAL_BASED_EFFECTIVITY('maintenance interval',#14);\n"
        "#14=TIME_INTERVAL_WITH_BOUNDS('maintenance bounds',#15,#16);\n"
        "#15=ORDINAL_DATE(2020,1);\n"
        "#16=WEEK_OF_YEAR_AND_DAY_DATE(2020,53,4);\n"
        "#20=EFFECTIVITY_ASSIGNMENT(#13,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="cover-node", name="Maintenance Cover", part_id="cover"),
            fc.Node(id="plug-node", name="Blanking Plug", part_id="plug"),
        ],
    )
    parts = {
        "cover": fc.Part(
            id="cover",
            name="Maintenance Cover",
            material_ids=["cover-mat"],
            metadata={"source_name": "maintenance cover"},
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
    options = StepReadOptions(design_variant_selection=("2020-06-01",))
    extraction = _extract_step_design_variants(source, options)

    selection = _apply_step_design_variant_selection(root, parts, materials, extraction, options)

    time_effectivity = next(record for record in extraction.records if record.kind == "time_interval_based_effectivity")
    assert time_effectivity.effectivity_values == ("maintenance interval", "2020-01-01", "2020-12-31")
    assert time_effectivity.effectivity_range == ("2020-01-01", "2020-12-31")
    assert selection.status == "applied"
    assert selection.matched_records == ("step_variant_20",)
    assert [child.name for child in root.children] == ["Maintenance Cover"]
    assert set(parts) == {"cover"}


def test_step_design_variant_selection_gates_product_definition_effectivity_usage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_DEFINITION('assembly','source assembly',#1,#2);\n"
        "#11=PRODUCT_DEFINITION('service panel','selected occurrence',#1,#2);\n"
        "#20=PRODUCT_DEFINITION_RELATIONSHIP('service option','effectivity usage',#10,#11);\n"
        "#30=PRODUCT_DEFINITION_EFFECTIVITY('release 2',#20);\n"
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

    release_options = StepReadOptions(design_variant_selection=("release 2",))
    release_extraction = _extract_step_design_variants(source, release_options)
    release_root = root.copy()
    release_parts = {key: part.copy() for key, part in parts.items()}
    release_materials = dict(materials)

    release_selection = _apply_step_design_variant_selection(
        release_root,
        release_parts,
        release_materials,
        release_extraction,
        release_options,
    )

    assert release_extraction.records[0].kind == "product_definition_effectivity"
    assert release_extraction.records[0].condition_operator == "effectivity_usage"
    assert release_extraction.records[0].reference_labels == ("service option / effectivity usage",)
    assert "service panel / selected occurrence" in release_extraction.records[0].resolved_reference_labels
    assert release_selection.status == "applied"
    assert release_selection.matched_records == ("step_variant_30",)
    assert "service panel" in [term.lower() for term in release_selection.selector_terms]
    assert [child.name for child in release_root.children] == ["Service Panel"]
    assert set(release_parts) == {"service"}

    target_options = StepReadOptions(design_variant_selection=("service panel",))
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


def test_step_design_variant_selection_resolves_effectivity_relationship_usage(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=SERIAL_NUMBERED_EFFECTIVITY('SN-A-001','SN-A-099',#1);\n"
        "#20=PRODUCT_DEFINITION('assembly','source assembly',#1,#2);\n"
        "#21=PRODUCT_DEFINITION('relationship panel','selected occurrence',#1,#2);\n"
        "#30=PRODUCT_DEFINITION_RELATIONSHIP('relationship option','effectivity usage',#20,#21);\n"
        "#40=PRODUCT_DEFINITION_EFFECTIVITY('release 2',#30);\n"
        "#50=EFFECTIVITY_RELATIONSHIP('serial release link','serial gates release',#40,#10);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="relationship-node", name="Relationship Panel", part_id="relationship"),
            fc.Node(id="blank-node", name="Blank Panel", part_id="blank"),
        ],
    )
    parts = {
        "relationship": fc.Part(
            id="relationship",
            name="Relationship Panel",
            material_ids=["relationship-mat"],
            metadata={"source_name": "relationship panel"},
        ),
        "blank": fc.Part(
            id="blank",
            name="Blank Panel",
            material_ids=["blank-mat"],
            metadata={"source_name": "blank panel"},
        ),
    }
    materials = {
        "relationship-mat": fc.Material(
            id="relationship-mat",
            name="Relationship Paint",
            base_color=(1.0, 0.0, 0.0, 1.0),
        ),
        "blank-mat": fc.Material(id="blank-mat", name="Blank Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
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

    assert [record.kind for record in serial_extraction.records] == [
        "serial_numbered_effectivity",
        "product_definition_effectivity",
        "effectivity_relationship",
    ]
    assert serial_extraction.records[1].condition_operator == "effectivity_usage"
    assert serial_extraction.records[2].condition_operator == "effectivity_relationship"
    assert serial_extraction.records[2].reference_labels == (
        "release 2",
        "SN-A-001 / SN-A-099",
    )
    assert "relationship panel / selected occurrence" in serial_extraction.records[2].resolved_reference_labels
    assert serial_selection.status == "applied"
    assert serial_selection.matched_records == ("step_variant_50",)
    assert "relationship panel" in [term.lower() for term in serial_selection.selector_terms]
    assert [child.name for child in serial_root.children] == ["Relationship Panel"]
    assert set(serial_parts) == {"relationship"}

    target_options = StepReadOptions(design_variant_selection=("relationship panel",))
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


def test_step_design_variant_selection_resolves_applied_effectivity_assignment_targets(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=SERIAL_NUMBERED_EFFECTIVITY('SN-A-001','SN-A-099',#1);\n"
        "#20=PRODUCT_DEFINITION('cover panel','selected cover occurrence',#1,#2);\n"
        "#30=APPLIED_EFFECTIVITY_ASSIGNMENT(#10,(#20));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="cover-node", name="Cover Panel", part_id="cover"),
            fc.Node(id="plug-node", name="Blanking Plug", part_id="plug"),
        ],
    )
    parts = {
        "cover": fc.Part(
            id="cover",
            name="Cover Panel",
            material_ids=["cover-mat"],
            metadata={"source_name": "cover panel"},
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

    assert [record.kind for record in serial_extraction.records] == [
        "serial_numbered_effectivity",
        "applied_effectivity_assignment",
    ]
    assert serial_extraction.records[1].condition_operator == "effectivity_assignment"
    assert serial_extraction.records[1].reference_labels == (
        "SN-A-001 / SN-A-099",
        "cover panel / selected cover occurrence",
    )
    assert serial_selection.status == "applied"
    assert serial_selection.matched_records == ("step_variant_30",)
    assert "cover panel" in [term.lower() for term in serial_selection.selector_terms]
    assert [child.name for child in serial_root.children] == ["Cover Panel"]
    assert set(serial_parts) == {"cover"}

    target_options = StepReadOptions(design_variant_selection=("cover panel",))
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


def test_step_design_variant_selection_resolves_effectivity_context_assignment_targets(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=EFFECTIVITY('service window');\n"
        "#20=PRODUCT_CONCEPT_FEATURE('service package','service condition',#1);\n"
        "#30=PRODUCT_DEFINITION('context panel','selected context occurrence',#1,#2);\n"
        "#40=CONFIGURED_EFFECTIVITY_ASSIGNMENT(#10,#20);\n"
        "#41=CONFIGURED_EFFECTIVITY_CONTEXT_ASSIGNMENT(#40,(#30));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="context-node", name="Context Panel", part_id="context"),
            fc.Node(id="other-node", name="Other Panel", part_id="other"),
        ],
    )
    parts = {
        "context": fc.Part(
            id="context",
            name="Context Panel",
            material_ids=["context-mat"],
            metadata={"source_name": "context panel"},
        ),
        "other": fc.Part(
            id="other",
            name="Other Panel",
            material_ids=["other-mat"],
            metadata={"source_name": "other panel"},
        ),
    }
    materials = {
        "context-mat": fc.Material(
            id="context-mat",
            name="Context Paint",
            base_color=(1.0, 0.0, 0.0, 1.0),
        ),
        "other-mat": fc.Material(id="other-mat", name="Other Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
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

    assert [record.kind for record in effectivity_extraction.records] == [
        "effectivity",
        "product_concept_feature",
        "configured_effectivity_assignment",
        "configured_effectivity_context_assignment",
    ]
    assert effectivity_extraction.records[3].condition_operator == "effectivity_context_assignment"
    assert effectivity_extraction.records[3].reference_labels == (
        "configured effectivity assignment",
        "context panel / selected context occurrence",
    )
    assert "context panel / selected context occurrence" in effectivity_extraction.records[3].resolved_reference_labels
    assert effectivity_selection.status == "applied"
    assert effectivity_selection.matched_records == ("step_variant_40", "step_variant_41")
    assert "context panel" in [term.lower() for term in effectivity_selection.selector_terms]
    assert [child.name for child in effectivity_root.children] == ["Context Panel"]
    assert set(effectivity_parts) == {"context"}

    target_options = StepReadOptions(design_variant_selection=("context panel",))
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


def test_step_design_variant_selection_suppresses_applied_ineffectivity_assignment_targets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=EFFECTIVITY('service window');\n"
        "#20=PRODUCT_DEFINITION('current panel','selected active occurrence',#1,#2);\n"
        "#21=PRODUCT_DEFINITION('obsolete panel','assigned inactive occurrence',#1,#2);\n"
        "#30=APPLIED_EFFECTIVITY_ASSIGNMENT(#10,(#20));\n"
        "#31=APPLIED_INEFFECTIVITY_ASSIGNMENT(#10,(#21));\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="current-node", name="Current Panel", part_id="current"),
            fc.Node(id="obsolete-node", name="Obsolete Panel", part_id="obsolete"),
        ],
    )
    parts = {
        "current": fc.Part(
            id="current",
            name="Current Panel",
            material_ids=["current-mat"],
            metadata={"source_name": "current panel"},
        ),
        "obsolete": fc.Part(
            id="obsolete",
            name="Obsolete Panel",
            material_ids=["obsolete-mat"],
            metadata={"source_name": "obsolete panel"},
        ),
    }
    materials = {
        "current-mat": fc.Material(id="current-mat", name="Current Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "obsolete-mat": fc.Material(id="obsolete-mat", name="Obsolete Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
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

    assert [record.kind for record in effectivity_extraction.records] == [
        "effectivity",
        "applied_effectivity_assignment",
        "applied_ineffectivity_assignment",
    ]
    assert effectivity_extraction.records[2].condition_operator == "ineffectivity_assignment"
    assert effectivity_extraction.records[2].reference_labels == (
        "service window",
        "obsolete panel / assigned inactive occurrence",
    )
    assert effectivity_selection.status == "applied"
    assert effectivity_selection.matched_records == ("step_variant_30",)
    assert "current panel" in [term.lower() for term in effectivity_selection.selector_terms]
    assert "obsolete panel" not in [term.lower() for term in effectivity_selection.selector_terms]
    assert [child.name for child in effectivity_root.children] == ["Current Panel"]
    assert set(effectivity_parts) == {"current"}

    inactive_target_options = StepReadOptions(design_variant_selection=("obsolete panel",))
    inactive_target_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, inactive_target_options),
        inactive_target_options,
    )

    assert inactive_target_selection.status == "unmatched_geometry"
    assert inactive_target_selection.matched_records == ()
    assert "condition expression was not satisfied" in inactive_target_selection.warnings[0]


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


@pytest.mark.parametrize(
    ("true_literal", "false_literal", "expected_kind"),
    (
        ("#20=BOOLEAN_LITERAL(.T.);\n", "#21=BOOLEAN_LITERAL(.F.);\n", "boolean_literal"),
        (
            "#20=BOOLEAN_REPRESENTATION_ITEM('service literal',.T.);\n",
            "#21=BOOLEAN_REPRESENTATION_ITEM('blocked literal',.F.);\n",
            "boolean_representation_item",
        ),
        ("#20=LOGICAL_LITERAL(.T.);\n", "#21=LOGICAL_LITERAL(.F.);\n", "logical_literal"),
        (
            "#20=LOGICAL_REPRESENTATION_ITEM('service literal',.T.);\n",
            "#21=LOGICAL_REPRESENTATION_ITEM('blocked literal',.F.);\n",
            "logical_representation_item",
        ),
    ),
)
def test_step_design_variant_selection_evaluates_boolean_literals(
    tmp_path: Path,
    true_literal: str,
    false_literal: str,
    expected_kind: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=PRODUCT_CONCEPT_FEATURE('service package','select service panel',#1);\n"
        "#11=PRODUCT_CONCEPT_FEATURE('blocked package','select blocked panel',#1);\n"
        f"{true_literal}"
        f"{false_literal}"
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
    assert service_extraction.records[2].kind == expected_kind
    assert service_extraction.records[2].condition_operator == "literal"
    assert service_extraction.records[2].condition_value is True
    assert service_extraction.records[3].kind == expected_kind
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


def test_step_design_variant_selection_evaluates_boolean_variable_assignments(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_BOOLEAN_VARIABLE(#1,'service enabled');\n"
        "#11=PRODUCT_CONCEPT_FEATURE('enabled package','select enabled panel',#1);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('disabled package','select disabled panel',#1);\n"
        "#20=CONDITIONAL_CONFIGURATION('enabled condition',#10,#11);\n"
        "#21=NOT_EXPRESSION(#10);\n"
        "#22=CONDITIONAL_CONFIGURATION('disabled condition',#21,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="enabled-node", name="Enabled Panel", part_id="enabled"),
            fc.Node(id="disabled-node", name="Disabled Panel", part_id="disabled"),
        ],
    )
    parts = {
        "enabled": fc.Part(
            id="enabled",
            name="Enabled Panel",
            material_ids=["enabled-mat"],
            metadata={"source_name": "enabled panel"},
        ),
        "disabled": fc.Part(
            id="disabled",
            name="Disabled Panel",
            material_ids=["disabled-mat"],
            metadata={"source_name": "disabled panel"},
        ),
    }
    materials = {
        "enabled-mat": fc.Material(id="enabled-mat", name="Enabled Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "disabled-mat": fc.Material(id="disabled-mat", name="Disabled Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    true_options = StepReadOptions(design_variant_selection=("service enabled=true",))
    true_extraction = _extract_step_design_variants(source, true_options)
    true_root = root.copy()
    true_parts = {key: part.copy() for key, part in parts.items()}
    true_materials = dict(materials)

    true_selection = _apply_step_design_variant_selection(
        true_root,
        true_parts,
        true_materials,
        true_extraction,
        true_options,
    )

    assert true_extraction.records[0].kind == "maths_boolean_variable"
    assert true_extraction.records[0].condition_operator == "variable"
    assert true_selection.status == "applied"
    assert true_selection.matched_records == ("step_variant_10", "step_variant_20")
    assert [child.name for child in true_root.children] == ["Enabled Panel"]
    assert set(true_parts) == {"enabled"}

    false_options = StepReadOptions(design_variant_selection=("service enabled=false",))
    false_extraction = _extract_step_design_variants(source, false_options)
    false_root = root.copy()
    false_parts = {key: part.copy() for key, part in parts.items()}
    false_materials = dict(materials)

    false_selection = _apply_step_design_variant_selection(
        false_root,
        false_parts,
        false_materials,
        false_extraction,
        false_options,
    )

    assert false_selection.status == "applied"
    assert false_selection.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in false_root.children] == ["Disabled Panel"]
    assert set(false_parts) == {"disabled"}

    target_options = StepReadOptions(design_variant_selection=("enabled package",))
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


@pytest.mark.parametrize(
    (
        "variable_record",
        "literal_record",
        "comparison_record",
        "matching_selection",
        "blocked_selection",
        "expected_variable_kind",
        "expected_operator",
    ),
    [
        (
            "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n",
            "#11=REAL_LITERAL(15.0);\n",
            "#20=COMPARISON_EQUAL(#10,#11);\n",
            "load rating=15",
            "load rating=12",
            "maths_real_variable",
            "equals",
        ),
        (
            "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n",
            "#11=REAL_LITERAL(15.0);\n",
            "#20=COMPARISON_NOT_EQUAL(#10,#11);\n",
            "load rating=12",
            "load rating=15",
            "maths_real_variable",
            "not_equals",
        ),
        (
            "#10=MATHS_STRING_VARIABLE(#1,'finish');\n",
            "#11=STRING_LITERAL('black anodized');\n",
            "#20=EQUALS_EXPRESSION(#10,#11);\n",
            "finish=black anodized",
            "finish=silver anodized",
            "maths_string_variable",
            "equals",
        ),
        (
            "#10=MATHS_STRING_VARIABLE(#1,'finish');\n",
            "#11=STRING_LITERAL('black anodized');\n",
            "#20=COMPARISON_NOT_EQUAL(#10,#11);\n",
            "finish=silver anodized",
            "finish=black anodized",
            "maths_string_variable",
            "not_equals",
        ),
    ],
)
def test_step_design_variant_selection_evaluates_value_equality_conditions(
    tmp_path: Path,
    variable_record: str,
    literal_record: str,
    comparison_record: str,
    matching_selection: str,
    blocked_selection: str,
    expected_variable_kind: str,
    expected_operator: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        f"{variable_record}"
        f"{literal_record}"
        "#12=PRODUCT_CONCEPT_FEATURE('value package','select value panel',#1);\n"
        f"{comparison_record}"
        "#21=CONDITIONAL_CONFIGURATION('value equality condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="value-node", name="Value Panel", part_id="value"),
            fc.Node(id="fallback-node", name="Fallback Panel", part_id="fallback"),
        ],
    )
    parts = {
        "value": fc.Part(
            id="value",
            name="Value Panel",
            material_ids=["value-mat"],
            metadata={"source_name": "value panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback panel"},
        ),
    }
    materials = {
        "value-mat": fc.Material(id="value-mat", name="Value Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=(matching_selection,))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[0].kind == expected_variable_kind
    assert matching_extraction.records[3].condition_operator == expected_operator
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Value Panel"]
    assert set(matching_parts) == {"value"}

    blocked_options = StepReadOptions(design_variant_selection=(blocked_selection,))
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

    target_options = StepReadOptions(design_variant_selection=("value package",))
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


@pytest.mark.parametrize(
    ("comparison_record", "matching_selection", "blocked_selection", "expected_operator"),
    (
        ("#20=COMPARISON_GREATER(#10,#11);\n", "load rating=15", "load rating=12", "greater"),
        ("#20=COMPARISON_GREATER_EQUAL(#10,#11);\n", "load rating=12.5", "load rating=12", "greater_equal"),
        ("#20=COMPARISON_LESS(#10,#11);\n", "load rating=10", "load rating=15", "less"),
        ("#20=COMPARISON_LESS_EQUAL(#10,#11);\n", "load rating=12.5", "load rating=15", "less_equal"),
    ),
)
def test_step_design_variant_selection_evaluates_numeric_comparisons(
    tmp_path: Path,
    comparison_record: str,
    matching_selection: str,
    blocked_selection: str,
    expected_operator: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n"
        "#11=REAL_LITERAL(12.5);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('heavy package','select heavy panel',#1);\n"
        f"{comparison_record}"
        "#21=CONDITIONAL_CONFIGURATION('heavy condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="heavy-node", name="Heavy Panel", part_id="heavy"),
            fc.Node(id="light-node", name="Light Panel", part_id="light"),
        ],
    )
    parts = {
        "heavy": fc.Part(
            id="heavy",
            name="Heavy Panel",
            material_ids=["heavy-mat"],
            metadata={"source_name": "heavy panel"},
        ),
        "light": fc.Part(
            id="light",
            name="Light Panel",
            material_ids=["light-mat"],
            metadata={"source_name": "light panel"},
        ),
    }
    materials = {
        "heavy-mat": fc.Material(id="heavy-mat", name="Heavy Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "light-mat": fc.Material(id="light-mat", name="Light Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=(matching_selection,))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[0].kind == "maths_real_variable"
    assert matching_extraction.records[0].condition_operator == "numeric_variable"
    assert matching_extraction.records[1].kind == "real_literal"
    assert matching_extraction.records[1].condition_operator == "numeric_literal"
    assert matching_extraction.records[1].condition_number == 12.5
    assert matching_extraction.records[3].condition_operator == expected_operator
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Heavy Panel"]
    assert set(matching_parts) == {"heavy"}

    blocked_options = StepReadOptions(design_variant_selection=(blocked_selection,))
    blocked_selection_result = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection_result.status == "unmatched_geometry"
    assert blocked_selection_result.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection_result.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("heavy package",))
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


def test_step_design_variant_selection_evaluates_numeric_interval_expression(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n"
        "#11=REAL_LITERAL(10.0);\n"
        "#12=REAL_LITERAL(20.0);\n"
        "#13=PRODUCT_CONCEPT_FEATURE('heavy package','select heavy panel',#1);\n"
        "#20=INTERVAL_EXPRESSION((#11,#10,#12));\n"
        "#21=CONDITIONAL_CONFIGURATION('heavy range condition',#20,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="heavy-node", name="Heavy Panel", part_id="heavy"),
            fc.Node(id="light-node", name="Light Panel", part_id="light"),
        ],
    )
    parts = {
        "heavy": fc.Part(
            id="heavy",
            name="Heavy Panel",
            material_ids=["heavy-mat"],
            metadata={"source_name": "heavy panel"},
        ),
        "light": fc.Part(
            id="light",
            name="Light Panel",
            material_ids=["light-mat"],
            metadata={"source_name": "light panel"},
        ),
    }
    materials = {
        "heavy-mat": fc.Material(id="heavy-mat", name="Heavy Paint", base_color=(1.0, 0.0, 0.0, 1.0)),
        "light-mat": fc.Material(id="light-mat", name="Light Paint", base_color=(0.0, 0.0, 1.0, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("load rating=15",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[4].kind == "interval_expression"
    assert matching_extraction.records[4].condition_operator == "interval"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Heavy Panel"]
    assert set(matching_parts) == {"heavy"}

    blocked_options = StepReadOptions(design_variant_selection=("load rating=25",))
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

    target_options = StepReadOptions(design_variant_selection=("heavy package",))
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


def test_step_design_variant_selection_evaluates_numeric_expression_extension(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n"
        "#11=EXPRESSION_EXTENSION_NUMERIC(12.5,#90);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('extended load package','select extended load panel',#1);\n"
        "#20=COMPARISON_EQUAL(#10,#11);\n"
        "#90=SI_UNIT(.MILLI.,.METRE.);\n"
        "#21=CONDITIONAL_CONFIGURATION('extended load condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="extended-node", name="Extended Load Panel", part_id="extended"),
            fc.Node(id="fallback-node", name="Fallback Load Panel", part_id="fallback"),
        ],
    )
    parts = {
        "extended": fc.Part(
            id="extended",
            name="Extended Load Panel",
            material_ids=["extended-mat"],
            metadata={"source_name": "extended load panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Load Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback load panel"},
        ),
    }
    materials = {
        "extended-mat": fc.Material(id="extended-mat", name="Extended Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("load rating=12.5",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[1].kind == "expression_extension_numeric"
    assert matching_extraction.records[1].condition_operator == "numeric_literal"
    assert matching_extraction.records[1].condition_number == 12.5
    assert matching_extraction.records[3].condition_operator == "equals"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Extended Load Panel"]
    assert set(matching_parts) == {"extended"}

    blocked_options = StepReadOptions(design_variant_selection=("load rating=8.0",))
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

    target_options = StepReadOptions(design_variant_selection=("extended load package",))
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


def test_step_design_variant_selection_evaluates_rational_representation_item(
    tmp_path: Path,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load factor');\n"
        "#11=INT_LITERAL(1);\n"
        "#12=INT_LITERAL(2);\n"
        "#13=PRODUCT_CONCEPT_FEATURE('half load package','select half load panel',#1);\n"
        "#20=RATIONAL_REPRESENTATION_ITEM('half',(#11,#12));\n"
        "#21=COMPARISON_EQUAL(#10,#20);\n"
        "#22=CONDITIONAL_CONFIGURATION('half load condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="half-node", name="Half Load Panel", part_id="half"),
            fc.Node(id="full-node", name="Full Load Panel", part_id="full"),
        ],
    )
    parts = {
        "half": fc.Part(
            id="half",
            name="Half Load Panel",
            material_ids=["half-mat"],
            metadata={"source_name": "half load panel"},
        ),
        "full": fc.Part(
            id="full",
            name="Full Load Panel",
            material_ids=["full-mat"],
            metadata={"source_name": "full load panel"},
        ),
    }
    materials = {
        "half-mat": fc.Material(id="half-mat", name="Half Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "full-mat": fc.Material(id="full-mat", name="Full Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("load factor=0.5",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[4].kind == "rational_representation_item"
    assert matching_extraction.records[4].condition_operator == "numeric_divide"
    assert matching_extraction.records[5].condition_operator == "equals"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Half Load Panel"]
    assert set(matching_parts) == {"half"}

    blocked_options = StepReadOptions(design_variant_selection=("load factor=0.75",))
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

    target_options = StepReadOptions(design_variant_selection=("half load package",))
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


@pytest.mark.parametrize(
    (
        "expression_record",
        "literal_record",
        "threshold_record",
        "matching_selection",
        "blocked_selection",
        "expected_kind",
        "expected_operator",
    ),
    (
        (
            "#20=PLUS_EXPRESSION((#10,#11));\n",
            "#11=REAL_LITERAL(5.0);\n",
            "#12=REAL_LITERAL(20.0);\n",
            "load rating=15",
            "load rating=10",
            "plus_expression",
            "numeric_add",
        ),
        (
            "#20=MINUS_EXPRESSION(#10,#11);\n",
            "#11=REAL_LITERAL(5.0);\n",
            "#12=REAL_LITERAL(10.0);\n",
            "load rating=15",
            "load rating=12",
            "minus_expression",
            "numeric_subtract",
        ),
        (
            "#20=MULT_EXPRESSION((#10,#11));\n",
            "#11=REAL_LITERAL(2.0);\n",
            "#12=REAL_LITERAL(30.0);\n",
            "load rating=15",
            "load rating=10",
            "mult_expression",
            "numeric_multiply",
        ),
        (
            "#20=DIV_EXPRESSION(#10,#11);\n",
            "#11=REAL_LITERAL(3.0);\n",
            "#12=REAL_LITERAL(5.0);\n",
            "load rating=15",
            "load rating=12",
            "div_expression",
            "numeric_divide",
        ),
        (
            "#20=SLASH_EXPRESSION(#10,#11);\n",
            "#11=REAL_LITERAL(3.0);\n",
            "#12=REAL_LITERAL(5.0);\n",
            "load rating=15",
            "load rating=12",
            "slash_expression",
            "numeric_divide",
        ),
        (
            "#20=MOD_EXPRESSION(#10,#11);\n",
            "#11=REAL_LITERAL(4.0);\n",
            "#12=REAL_LITERAL(3.0);\n",
            "load rating=15",
            "load rating=12",
            "mod_expression",
            "numeric_mod",
        ),
        (
            "#20=POWER_EXPRESSION(#10,#11);\n",
            "#11=REAL_LITERAL(2.0);\n",
            "#12=REAL_LITERAL(16.0);\n",
            "load rating=4",
            "load rating=3",
            "power_expression",
            "numeric_power",
        ),
    ),
)
def test_step_design_variant_selection_evaluates_numeric_arithmetic_expression(
    tmp_path: Path,
    expression_record: str,
    literal_record: str,
    threshold_record: str,
    matching_selection: str,
    blocked_selection: str,
    expected_kind: str,
    expected_operator: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n"
        f"{literal_record}"
        f"{threshold_record}"
        "#13=PRODUCT_CONCEPT_FEATURE('calculated package','select calculated panel',#1);\n"
        f"{expression_record}"
        "#21=COMPARISON_GREATER_EQUAL(#20,#12);\n"
        "#22=CONDITIONAL_CONFIGURATION('calculated load condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="calculated-node", name="Calculated Panel", part_id="calculated"),
            fc.Node(id="baseline-node", name="Baseline Panel", part_id="baseline"),
        ],
    )
    parts = {
        "calculated": fc.Part(
            id="calculated",
            name="Calculated Panel",
            material_ids=["calculated-mat"],
            metadata={"source_name": "calculated panel"},
        ),
        "baseline": fc.Part(
            id="baseline",
            name="Baseline Panel",
            material_ids=["baseline-mat"],
            metadata={"source_name": "baseline panel"},
        ),
    }
    materials = {
        "calculated-mat": fc.Material(id="calculated-mat", name="Calculated Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "baseline-mat": fc.Material(id="baseline-mat", name="Baseline Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=(matching_selection,))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[4].kind == expected_kind
    assert matching_extraction.records[4].condition_operator == expected_operator
    assert matching_extraction.records[5].condition_operator == "greater_equal"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Calculated Panel"]
    assert set(matching_parts) == {"calculated"}

    blocked_options = StepReadOptions(design_variant_selection=(blocked_selection,))
    blocked_selection_result = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection_result.status == "unmatched_geometry"
    assert blocked_selection_result.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection_result.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("calculated package",))
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


@pytest.mark.parametrize(
    ("variable_record", "expected_kind"),
    (
        ("#10=STRING_VARIABLE('finish');\n", "string_variable"),
        ("#10=MATHS_STRING_VARIABLE(#1,'finish');\n", "maths_string_variable"),
    ),
)
def test_step_design_variant_selection_evaluates_like_expression_string_assignment(
    tmp_path: Path,
    variable_record: str,
    expected_kind: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        f"{variable_record}"
        "#11=STRING_LITERAL('black*');\n"
        "#12=PRODUCT_CONCEPT_FEATURE('black package','select black panel',#1);\n"
        "#20=LIKE_EXPRESSION(#10,#11);\n"
        "#21=CONDITIONAL_CONFIGURATION('finish condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="black-node", name="Black Panel", part_id="black"),
            fc.Node(id="silver-node", name="Silver Panel", part_id="silver"),
        ],
    )
    parts = {
        "black": fc.Part(
            id="black",
            name="Black Panel",
            material_ids=["black-mat"],
            metadata={"source_name": "black panel"},
        ),
        "silver": fc.Part(
            id="silver",
            name="Silver Panel",
            material_ids=["silver-mat"],
            metadata={"source_name": "silver panel"},
        ),
    }
    materials = {
        "black-mat": fc.Material(id="black-mat", name="Black Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "silver-mat": fc.Material(id="silver-mat", name="Silver Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("finish=black anodized",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[0].kind == expected_kind
    assert matching_extraction.records[0].condition_operator == "string_variable"
    assert matching_extraction.records[1].kind == "string_literal"
    assert matching_extraction.records[1].condition_operator == "string_literal"
    assert matching_extraction.records[1].condition_text == "black*"
    assert matching_extraction.records[3].kind == "like_expression"
    assert matching_extraction.records[3].condition_operator == "like"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Black Panel"]
    assert set(matching_parts) == {"black"}

    blocked_options = StepReadOptions(design_variant_selection=("finish=silver anodized",))
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

    target_options = StepReadOptions(design_variant_selection=("black package",))
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


def test_step_design_variant_selection_evaluates_concat_expression_in_like_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_STRING_VARIABLE(#1,'finish');\n"
        "#11=STRING_LITERAL('anodized');\n"
        "#12=STRING_LITERAL('blackanodized');\n"
        "#13=PRODUCT_CONCEPT_FEATURE('black finish package','select black finish panel',#1);\n"
        "#20=CONCAT_EXPRESSION((#10,#11));\n"
        "#21=LIKE_EXPRESSION(#20,#12);\n"
        "#22=CONDITIONAL_CONFIGURATION('concatenated finish condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="black-node", name="Black Finish Panel", part_id="black"),
            fc.Node(id="silver-node", name="Silver Finish Panel", part_id="silver"),
        ],
    )
    parts = {
        "black": fc.Part(
            id="black",
            name="Black Finish Panel",
            material_ids=["black-mat"],
            metadata={"source_name": "black finish panel"},
        ),
        "silver": fc.Part(
            id="silver",
            name="Silver Finish Panel",
            material_ids=["silver-mat"],
            metadata={"source_name": "silver finish panel"},
        ),
    }
    materials = {
        "black-mat": fc.Material(id="black-mat", name="Black Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "silver-mat": fc.Material(id="silver-mat", name="Silver Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("finish=black",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[4].kind == "concat_expression"
    assert matching_extraction.records[4].condition_operator == "string_concat"
    assert matching_extraction.records[5].kind == "like_expression"
    assert matching_extraction.records[5].condition_operator == "like"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Black Finish Panel"]
    assert set(matching_parts) == {"black"}

    blocked_options = StepReadOptions(design_variant_selection=("finish=silver",))
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

    target_options = StepReadOptions(design_variant_selection=("black finish package",))
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


def test_step_design_variant_selection_evaluates_substring_expression_in_like_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_STRING_VARIABLE(#1,'finish');\n"
        "#11=INT_LITERAL(1);\n"
        "#12=INT_LITERAL(5);\n"
        "#13=STRING_LITERAL('black');\n"
        "#14=PRODUCT_CONCEPT_FEATURE('black finish package','select black finish panel',#1);\n"
        "#20=SUBSTRING_EXPRESSION((#10,#11,#12));\n"
        "#21=LIKE_EXPRESSION(#20,#13);\n"
        "#22=CONDITIONAL_CONFIGURATION('substring finish condition',#21,#14);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="black-node", name="Black Finish Panel", part_id="black"),
            fc.Node(id="silver-node", name="Silver Finish Panel", part_id="silver"),
        ],
    )
    parts = {
        "black": fc.Part(
            id="black",
            name="Black Finish Panel",
            material_ids=["black-mat"],
            metadata={"source_name": "black finish panel"},
        ),
        "silver": fc.Part(
            id="silver",
            name="Silver Finish Panel",
            material_ids=["silver-mat"],
            metadata={"source_name": "silver finish panel"},
        ),
    }
    materials = {
        "black-mat": fc.Material(id="black-mat", name="Black Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "silver-mat": fc.Material(id="silver-mat", name="Silver Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("finish=black anodized",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[5].kind == "substring_expression"
    assert matching_extraction.records[5].condition_operator == "string_substring"
    assert matching_extraction.records[6].condition_operator == "like"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Black Finish Panel"]
    assert set(matching_parts) == {"black"}

    blocked_options = StepReadOptions(design_variant_selection=("finish=silver anodized",))
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

    target_options = StepReadOptions(design_variant_selection=("black finish package",))
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


def test_step_design_variant_selection_evaluates_index_expression_in_like_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_STRING_VARIABLE(#1,'finish');\n"
        "#11=INT_LITERAL(1);\n"
        "#12=STRING_LITERAL('b');\n"
        "#13=PRODUCT_CONCEPT_FEATURE('black finish package','select black finish panel',#1);\n"
        "#20=INDEX_EXPRESSION((#10,#11));\n"
        "#21=LIKE_EXPRESSION(#20,#12);\n"
        "#22=CONDITIONAL_CONFIGURATION('indexed finish condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="black-node", name="Black Finish Panel", part_id="black"),
            fc.Node(id="silver-node", name="Silver Finish Panel", part_id="silver"),
        ],
    )
    parts = {
        "black": fc.Part(
            id="black",
            name="Black Finish Panel",
            material_ids=["black-mat"],
            metadata={"source_name": "black finish panel"},
        ),
        "silver": fc.Part(
            id="silver",
            name="Silver Finish Panel",
            material_ids=["silver-mat"],
            metadata={"source_name": "silver finish panel"},
        ),
    }
    materials = {
        "black-mat": fc.Material(id="black-mat", name="Black Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "silver-mat": fc.Material(id="silver-mat", name="Silver Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("finish=black anodized",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[4].kind == "index_expression"
    assert matching_extraction.records[4].condition_operator == "string_index"
    assert matching_extraction.records[5].condition_operator == "like"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Black Finish Panel"]
    assert set(matching_parts) == {"black"}

    blocked_options = StepReadOptions(design_variant_selection=("finish=silver anodized",))
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

    target_options = StepReadOptions(design_variant_selection=("black finish package",))
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


def test_step_design_variant_selection_evaluates_format_function_in_like_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load rating');\n"
        "#11=STRING_LITERAL('%.1f');\n"
        "#12=STRING_LITERAL('12.5');\n"
        "#13=PRODUCT_CONCEPT_FEATURE('formatted load package','select formatted load panel',#1);\n"
        "#20=FORMAT_FUNCTION((#10,#11));\n"
        "#21=LIKE_EXPRESSION(#20,#12);\n"
        "#22=CONDITIONAL_CONFIGURATION('formatted load condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="formatted-node", name="Formatted Load Panel", part_id="formatted"),
            fc.Node(id="fallback-node", name="Fallback Load Panel", part_id="fallback"),
        ],
    )
    parts = {
        "formatted": fc.Part(
            id="formatted",
            name="Formatted Load Panel",
            material_ids=["formatted-mat"],
            metadata={"source_name": "formatted load panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Load Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback load panel"},
        ),
    }
    materials = {
        "formatted-mat": fc.Material(id="formatted-mat", name="Formatted Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("load rating=12.5",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[4].kind == "format_function"
    assert matching_extraction.records[4].condition_operator == "string_format"
    assert matching_extraction.records[5].condition_operator == "like"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Formatted Load Panel"]
    assert set(matching_parts) == {"formatted"}

    blocked_options = StepReadOptions(design_variant_selection=("load rating=8.0",))
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

    target_options = StepReadOptions(design_variant_selection=("formatted load package",))
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


def test_step_design_variant_selection_evaluates_string_expression_extension(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_STRING_VARIABLE(#1,'finish');\n"
        "#11=EXPRESSION_EXTENSION_STRING('black anodized',#90);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('extended finish package','select extended finish panel',#1);\n"
        "#20=LIKE_EXPRESSION(#10,#11);\n"
        "#90=SI_UNIT(.MILLI.,.METRE.);\n"
        "#21=CONDITIONAL_CONFIGURATION('extended finish condition',#20,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="extended-node", name="Extended Finish Panel", part_id="extended"),
            fc.Node(id="fallback-node", name="Fallback Finish Panel", part_id="fallback"),
        ],
    )
    parts = {
        "extended": fc.Part(
            id="extended",
            name="Extended Finish Panel",
            material_ids=["extended-mat"],
            metadata={"source_name": "extended finish panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Finish Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback finish panel"},
        ),
    }
    materials = {
        "extended-mat": fc.Material(id="extended-mat", name="Extended Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("finish=black anodized",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[1].kind == "expression_extension_string"
    assert matching_extraction.records[1].condition_operator == "string_literal"
    assert matching_extraction.records[1].condition_text == "black anodized"
    assert matching_extraction.records[3].condition_operator == "like"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Extended Finish Panel"]
    assert set(matching_parts) == {"extended"}

    blocked_options = StepReadOptions(design_variant_selection=("finish=silver anodized",))
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

    target_options = StepReadOptions(design_variant_selection=("extended finish package",))
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


def test_step_design_variant_selection_evaluates_length_function_in_numeric_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_STRING_VARIABLE(#1,'finish code');\n"
        "#11=REAL_LITERAL(5.0);\n"
        "#12=PRODUCT_CONCEPT_FEATURE('long finish package','select long finish panel',#1);\n"
        "#20=LENGTH_FUNCTION(#10);\n"
        "#21=COMPARISON_GREATER_EQUAL(#20,#11);\n"
        "#22=CONDITIONAL_CONFIGURATION('finish length condition',#21,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="long-node", name="Long Finish Panel", part_id="long"),
            fc.Node(id="short-node", name="Short Finish Panel", part_id="short"),
        ],
    )
    parts = {
        "long": fc.Part(
            id="long",
            name="Long Finish Panel",
            material_ids=["long-mat"],
            metadata={"source_name": "long finish panel"},
        ),
        "short": fc.Part(
            id="short",
            name="Short Finish Panel",
            material_ids=["short-mat"],
            metadata={"source_name": "short finish panel"},
        ),
    }
    materials = {
        "long-mat": fc.Material(id="long-mat", name="Long Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "short-mat": fc.Material(id="short-mat", name="Short Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("finish code=black",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[3].kind == "length_function"
    assert matching_extraction.records[3].condition_operator == "string_length"
    assert matching_extraction.records[4].condition_operator == "greater_equal"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Long Finish Panel"]
    assert set(matching_parts) == {"long"}

    blocked_options = StepReadOptions(design_variant_selection=("finish code=blue",))
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

    target_options = StepReadOptions(design_variant_selection=("long finish package",))
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


@pytest.mark.parametrize(
    (
        "function_record",
        "expected_kind",
        "expected_operator",
        "threshold_record",
        "matching_selection",
        "blocked_selection",
    ),
    [
        (
            "#20=VALUE_FUNCTION(#10);\n",
            "value_function",
            "string_value",
            "#11=REAL_LITERAL(12.5);\n",
            "load code=15.5",
            "load code=9.5",
        ),
        (
            "#20=INT_VALUE_FUNCTION(#10);\n",
            "int_value_function",
            "string_integer_value",
            "#11=INT_LITERAL(15);\n",
            "load code=15",
            "load code=14",
        ),
    ],
)
def test_step_design_variant_selection_evaluates_value_functions_in_numeric_condition(
    tmp_path: Path,
    function_record: str,
    expected_kind: str,
    expected_operator: str,
    threshold_record: str,
    matching_selection: str,
    blocked_selection: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_STRING_VARIABLE(#1,'load code');\n"
        f"{threshold_record}"
        "#12=PRODUCT_CONCEPT_FEATURE('heavy load package','select heavy load panel',#1);\n"
        f"{function_record}"
        "#21=COMPARISON_GREATER_EQUAL(#20,#11);\n"
        "#22=CONDITIONAL_CONFIGURATION('load code condition',#21,#12);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="heavy-node", name="Heavy Load Panel", part_id="heavy"),
            fc.Node(id="light-node", name="Light Load Panel", part_id="light"),
        ],
    )
    parts = {
        "heavy": fc.Part(
            id="heavy",
            name="Heavy Load Panel",
            material_ids=["heavy-mat"],
            metadata={"source_name": "heavy load panel"},
        ),
        "light": fc.Part(
            id="light",
            name="Light Load Panel",
            material_ids=["light-mat"],
            metadata={"source_name": "light load panel"},
        ),
    }
    materials = {
        "heavy-mat": fc.Material(id="heavy-mat", name="Heavy Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "light-mat": fc.Material(id="light-mat", name="Light Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=(matching_selection,))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[3].kind == expected_kind
    assert matching_extraction.records[3].condition_operator == expected_operator
    assert matching_extraction.records[4].condition_operator == "greater_equal"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Heavy Load Panel"]
    assert set(matching_parts) == {"heavy"}

    blocked_options = StepReadOptions(design_variant_selection=(blocked_selection,))
    blocked_selection_result = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection_result.status == "unmatched_geometry"
    assert blocked_selection_result.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection_result.warnings[0]

    invalid_options = StepReadOptions(design_variant_selection=("load code=15A",))
    invalid_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, invalid_options),
        invalid_options,
    )

    assert invalid_selection.status == "unmatched_geometry"
    assert invalid_selection.matched_records == ()
    assert "condition expression was not satisfied" in invalid_selection.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("heavy load package",))
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


@pytest.mark.parametrize(
    (
        "function_record",
        "extra_record",
        "threshold_record",
        "comparison_record",
        "matching_selection",
        "blocked_selection",
        "expected_kind",
        "expected_operator",
    ),
    [
        (
            "#20=ABS_FUNCTION(#10);\n",
            "",
            "#11=REAL_LITERAL(5.0);\n",
            "#21=COMPARISON_GREATER_EQUAL(#20,#11);\n",
            "load delta=-7",
            "load delta=-3",
            "abs_function",
            "numeric_abs",
        ),
        (
            "#20=MINUS_FUNCTION(#10);\n",
            "",
            "#11=REAL_LITERAL(5.0);\n",
            "#21=COMPARISON_GREATER_EQUAL(#20,#11);\n",
            "load delta=-7",
            "load delta=-3",
            "minus_function",
            "numeric_negate",
        ),
        (
            "#20=SQUARE_ROOT_FUNCTION(#10);\n",
            "",
            "#11=REAL_LITERAL(4.0);\n",
            "#21=COMPARISON_GREATER_EQUAL(#20,#11);\n",
            "load delta=25",
            "load delta=9",
            "square_root_function",
            "numeric_sqrt",
        ),
        (
            "#20=MAXIMUM_FUNCTION((#10,#11));\n",
            "#11=REAL_LITERAL(5.0);\n",
            "#12=REAL_LITERAL(10.0);\n",
            "#21=COMPARISON_GREATER_EQUAL(#20,#12);\n",
            "load delta=12",
            "load delta=4",
            "maximum_function",
            "numeric_max",
        ),
        (
            "#20=MINIMUM_FUNCTION((#10,#11));\n",
            "#11=REAL_LITERAL(12.0);\n",
            "#12=REAL_LITERAL(10.0);\n",
            "#21=COMPARISON_GREATER_EQUAL(#20,#12);\n",
            "load delta=11",
            "load delta=8",
            "minimum_function",
            "numeric_min",
        ),
    ],
)
def test_step_design_variant_selection_evaluates_numeric_functions_in_numeric_condition(
    tmp_path: Path,
    function_record: str,
    extra_record: str,
    threshold_record: str,
    comparison_record: str,
    matching_selection: str,
    blocked_selection: str,
    expected_kind: str,
    expected_operator: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'load delta');\n"
        f"{extra_record}"
        f"{threshold_record}"
        "#13=PRODUCT_CONCEPT_FEATURE('calculated function package','select calculated function panel',#1);\n"
        f"{function_record}"
        f"{comparison_record}"
        "#22=CONDITIONAL_CONFIGURATION('calculated function condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="calculated-node", name="Calculated Function Panel", part_id="calculated"),
            fc.Node(id="baseline-node", name="Baseline Function Panel", part_id="baseline"),
        ],
    )
    parts = {
        "calculated": fc.Part(
            id="calculated",
            name="Calculated Function Panel",
            material_ids=["calculated-mat"],
            metadata={"source_name": "calculated function panel"},
        ),
        "baseline": fc.Part(
            id="baseline",
            name="Baseline Function Panel",
            material_ids=["baseline-mat"],
            metadata={"source_name": "baseline function panel"},
        ),
    }
    materials = {
        "calculated-mat": fc.Material(id="calculated-mat", name="Calculated Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "baseline-mat": fc.Material(id="baseline-mat", name="Baseline Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=(matching_selection,))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    function = next(record for record in matching_extraction.records if record.kind == expected_kind)
    assert function.condition_operator == expected_operator
    assert any(record.condition_operator == "greater_equal" for record in matching_extraction.records)
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Calculated Function Panel"]
    assert set(matching_parts) == {"calculated"}

    blocked_options = StepReadOptions(design_variant_selection=(blocked_selection,))
    blocked_selection_result = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection_result.status == "unmatched_geometry"
    assert blocked_selection_result.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection_result.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("calculated function package",))
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


@pytest.mark.parametrize(
    (
        "function_record",
        "threshold_record",
        "matching_selection",
        "blocked_selection",
        "expected_kind",
        "expected_operator",
    ),
    [
        (
            "#20=SIN_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(0.5);\n",
            "function input=1.57079632679",
            "function input=0",
            "sin_function",
            "numeric_sin",
        ),
        (
            "#20=COS_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(0.5);\n",
            "function input=0",
            "function input=1.57079632679",
            "cos_function",
            "numeric_cos",
        ),
        (
            "#20=TAN_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(0.5);\n",
            "function input=0.78539816339",
            "function input=0.25",
            "tan_function",
            "numeric_tan",
        ),
        (
            "#20=ASIN_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(0.5);\n",
            "function input=1",
            "function input=0.25",
            "asin_function",
            "numeric_asin",
        ),
        (
            "#20=ACOS_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(1.0);\n",
            "function input=0",
            "function input=1",
            "acos_function",
            "numeric_acos",
        ),
        (
            "#20=ATAN_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(0.5);\n",
            "function input=1",
            "function input=0.25",
            "atan_function",
            "numeric_atan",
        ),
        (
            "#20=EXP_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(2.0);\n",
            "function input=1",
            "function input=0",
            "exp_function",
            "numeric_exp",
        ),
        (
            "#20=LOG_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(1.0);\n",
            "function input=3",
            "function input=1",
            "log_function",
            "numeric_log",
        ),
        (
            "#20=LOG2_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(3.0);\n",
            "function input=8",
            "function input=4",
            "log2_function",
            "numeric_log2",
        ),
        (
            "#20=LOG10_FUNCTION(#10);\n",
            "#11=REAL_LITERAL(2.0);\n",
            "function input=100",
            "function input=10",
            "log10_function",
            "numeric_log10",
        ),
    ],
)
def test_step_design_variant_selection_evaluates_elementary_numeric_functions(
    tmp_path: Path,
    function_record: str,
    threshold_record: str,
    matching_selection: str,
    blocked_selection: str,
    expected_kind: str,
    expected_operator: str,
) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'function input');\n"
        f"{threshold_record}"
        "#13=PRODUCT_CONCEPT_FEATURE('elementary function package','select elementary function panel',#1);\n"
        f"{function_record}"
        "#21=COMPARISON_GREATER_EQUAL(#20,#11);\n"
        "#22=CONDITIONAL_CONFIGURATION('elementary function condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="function-node", name="Elementary Function Panel", part_id="function"),
            fc.Node(id="fallback-node", name="Fallback Function Panel", part_id="fallback"),
        ],
    )
    parts = {
        "function": fc.Part(
            id="function",
            name="Elementary Function Panel",
            material_ids=["function-mat"],
            metadata={"source_name": "elementary function panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Fallback Function Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "fallback function panel"},
        ),
    }
    materials = {
        "function-mat": fc.Material(id="function-mat", name="Function Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=(matching_selection,))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    function = next(record for record in matching_extraction.records if record.kind == expected_kind)
    assert function.condition_operator == expected_operator
    assert any(record.condition_operator == "greater_equal" for record in matching_extraction.records)
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Elementary Function Panel"]
    assert set(matching_parts) == {"function"}

    blocked_options = StepReadOptions(design_variant_selection=(blocked_selection,))
    blocked_selection_result = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection_result.status == "unmatched_geometry"
    assert blocked_selection_result.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection_result.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("elementary function package",))
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


@pytest.mark.parametrize(
    ("operator", "values"),
    [
        ("numeric_log", [0.0]),
        ("numeric_log", [-1.0]),
        ("numeric_log2", [0.0]),
        ("numeric_log10", [-10.0]),
        ("numeric_sqrt", [-1.0]),
    ],
)
def test_numeric_function_value_rejects_invalid_log_and_sqrt_domains(operator: str, values: list[float]) -> None:
    assert step_variants._numeric_function_value(operator, values) is None


def test_step_design_variant_selection_evaluates_binary_atan_function(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_REAL_VARIABLE(#1,'rise');\n"
        "#11=MATHS_REAL_VARIABLE(#1,'run');\n"
        "#12=REAL_LITERAL(0.7);\n"
        "#13=PRODUCT_CONCEPT_FEATURE('angle package','select angle panel',#1);\n"
        "#20=ATAN_FUNCTION((#10,#11));\n"
        "#21=COMPARISON_GREATER_EQUAL(#20,#12);\n"
        "#22=CONDITIONAL_CONFIGURATION('angle condition',#21,#13);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="angle-node", name="Angle Panel", part_id="angle"),
            fc.Node(id="fallback-node", name="Reserve Panel", part_id="fallback"),
        ],
    )
    parts = {
        "angle": fc.Part(
            id="angle",
            name="Angle Panel",
            material_ids=["angle-mat"],
            metadata={"source_name": "angle panel"},
        ),
        "fallback": fc.Part(
            id="fallback",
            name="Reserve Panel",
            material_ids=["fallback-mat"],
            metadata={"source_name": "reserve panel"},
        ),
    }
    materials = {
        "angle-mat": fc.Material(id="angle-mat", name="Angle Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "fallback-mat": fc.Material(id="fallback-mat", name="Fallback Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("rise=1", "run=1"))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    function = next(record for record in matching_extraction.records if record.kind == "atan_function")
    assert function.condition_operator == "numeric_atan"
    assert any(record.condition_operator == "greater_equal" for record in matching_extraction.records)
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_21", "step_variant_22")
    assert [child.name for child in matching_root.children] == ["Angle Panel"]
    assert set(matching_parts) == {"angle"}

    blocked_options = StepReadOptions(design_variant_selection=("rise=0.5", "run=2"))
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

    target_options = StepReadOptions(design_variant_selection=("angle package",))
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


def test_step_design_variant_selection_evaluates_odd_function_condition(tmp_path: Path) -> None:
    source = tmp_path / "variants.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#10=MATHS_INTEGER_VARIABLE(#1,'load count');\n"
        "#11=PRODUCT_CONCEPT_FEATURE('odd load package','select odd load panel',#1);\n"
        "#20=ODD_FUNCTION(#10);\n"
        "#21=CONDITIONAL_CONFIGURATION('odd load condition',#20,#11);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    root = fc.Node(
        id="root",
        name="Assembly",
        children=[
            fc.Node(id="odd-node", name="Odd Load Panel", part_id="odd"),
            fc.Node(id="even-node", name="Even Load Panel", part_id="even"),
        ],
    )
    parts = {
        "odd": fc.Part(
            id="odd",
            name="Odd Load Panel",
            material_ids=["odd-mat"],
            metadata={"source_name": "odd load panel"},
        ),
        "even": fc.Part(
            id="even",
            name="Even Load Panel",
            material_ids=["even-mat"],
            metadata={"source_name": "even load panel"},
        ),
    }
    materials = {
        "odd-mat": fc.Material(id="odd-mat", name="Odd Paint", base_color=(0.0, 0.0, 0.0, 1.0)),
        "even-mat": fc.Material(id="even-mat", name="Even Paint", base_color=(0.7, 0.7, 0.7, 1.0)),
    }

    matching_options = StepReadOptions(design_variant_selection=("load count=5",))
    matching_extraction = _extract_step_design_variants(source, matching_options)
    matching_root = root.copy()
    matching_parts = {key: part.copy() for key, part in parts.items()}
    matching_materials = dict(materials)

    matching_result = _apply_step_design_variant_selection(
        matching_root,
        matching_parts,
        matching_materials,
        matching_extraction,
        matching_options,
    )

    assert matching_extraction.records[0].kind == "maths_integer_variable"
    assert matching_extraction.records[0].condition_operator == "numeric_variable"
    assert matching_extraction.records[2].kind == "odd_function"
    assert matching_extraction.records[2].condition_operator == "numeric_odd"
    assert matching_result.status == "applied"
    assert matching_result.matched_records == ("step_variant_20", "step_variant_21")
    assert [child.name for child in matching_root.children] == ["Odd Load Panel"]
    assert set(matching_parts) == {"odd"}

    blocked_options = StepReadOptions(design_variant_selection=("load count=4",))
    blocked_selection_result = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, blocked_options),
        blocked_options,
    )

    assert blocked_selection_result.status == "unmatched_geometry"
    assert blocked_selection_result.matched_records == ()
    assert "condition expression was not satisfied" in blocked_selection_result.warnings[0]

    invalid_options = StepReadOptions(design_variant_selection=("load count=5.5",))
    invalid_selection = _apply_step_design_variant_selection(
        root.copy(),
        {key: part.copy() for key, part in parts.items()},
        dict(materials),
        _extract_step_design_variants(source, invalid_options),
        invalid_options,
    )

    assert invalid_selection.status == "unmatched_geometry"
    assert invalid_selection.matched_records == ()
    assert "condition expression was not satisfied" in invalid_selection.warnings[0]

    target_options = StepReadOptions(design_variant_selection=("odd load package",))
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
