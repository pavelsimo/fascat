from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc
from fascat.io._import_base import (
    _ImportCleanupStats,
    _space_normalization,
    _StepHeaderInfo,
)
from fascat.io.step import pmi as step_pmi
from fascat.io.step.pmi import (
    _extract_step_pmi_annotations,
    _extract_step_pmi_semantic_graph,
    _step_pmi_semantic_cycle_count,
    _StepPmiSemanticGraphEdge,
)
from fascat.io.step.single import (
    _import_decisions,
)
from fascat.options import StepReadOptions


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


def test_step_text_pmi_extraction_reads_named_geometric_tolerance_subtypes(tmp_path: Path) -> None:
    source = tmp_path / "pmi.step"
    tolerance_entities = (
        "ANGULARITY_TOLERANCE",
        "CIRCULAR_RUNOUT_TOLERANCE",
        "COAXIALITY_TOLERANCE",
        "CONCENTRICITY_TOLERANCE",
        "CYLINDRICITY_TOLERANCE",
        "FLATNESS_TOLERANCE",
        "LINE_PROFILE_TOLERANCE",
        "PARALLELISM_TOLERANCE",
        "PERPENDICULARITY_TOLERANCE",
        "POSITION_TOLERANCE",
        "ROUNDNESS_TOLERANCE",
        "STRAIGHTNESS_TOLERANCE",
        "SURFACE_PROFILE_TOLERANCE",
        "SYMMETRY_TOLERANCE",
        "TOTAL_RUNOUT_TOLERANCE",
    )
    records = "".join(
        f"#{30 + index}={entity}('{entity.lower().replace('_', ' ')}',0.{index + 1},#20);\n"
        for index, entity in enumerate(tolerance_entities)
    )
    source.write_text(
        f"ISO-10303-21;\nDATA;\n#20=PRODUCT_DEFINITION_SHAPE('bracket','',#19);\n{records}ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    annotations = _extract_step_pmi_annotations(source, StepReadOptions(pmi=True))
    graph = _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=True))

    assert len(annotations) == len(tolerance_entities)
    assert {annotation.kind for annotation in annotations} == {"tolerance"}
    assert [annotation.source["step_entity"] for annotation in annotations] == list(tolerance_entities)
    assert annotations[5].text == "flatness tolerance"
    assert annotations[5].tolerance is not None
    assert annotations[5].tolerance.upper == 0.6
    assert graph.summary["pmi_nodes"] == len(tolerance_entities)


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
        "edges": 5,
        "missing_references": 1,
        "cycles": 0,
    }
    assert [node["id"] for node in payload["nodes"]] == ["#20", "#21", "#30", "#31", "#32"]
    assert payload["nodes"][0]["kind"] == "pmi_target"
    assert payload["nodes"][1]["kind"] == "pmi_target"
    assert payload["nodes"][2]["kind"] == "pmi_dimension"
    assert payload["nodes"][2]["references"] == ["#21"]
    assert payload["nodes"][3]["kind"] == "pmi_tolerance"
    assert payload["nodes"][4]["kind"] == "pmi_tolerance"
    assert {"source": "#21", "target": "#20", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#31", "target": "#999", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#32", "target": "#30", "relationship": "step_reference"} in payload["edges"]
    assert graph.warnings == ("STEP PMI semantic graph has 1 reference(s) to records that were not found",)


def test_step_pmi_semantic_graph_reports_cycles(tmp_path: Path) -> None:
    source = tmp_path / "pmi-cycle.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#21=SHAPE_ASPECT('hole face','',#40,.T.);\n"
        "#30=DIMENSIONAL_SIZE(#21,'hole diameter 12.5 mm',12.5);\n"
        "#40=SHAPE_ASPECT_RELATIONSHIP('cyclic target','',#21,#30);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=True))
    payload = graph.to_dict()

    assert graph.summary["cycles"] == 1
    assert graph.summary["missing_references"] == 0
    assert {node["id"] for node in payload["nodes"]} == {"#21", "#30", "#40"}
    assert {"source": "#30", "target": "#21", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#21", "target": "#40", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#40", "target": "#30", "relationship": "step_reference"} in payload["edges"]
    assert graph.warnings == ("STEP PMI semantic graph contains 1 cycle(s)",)


def test_step_pmi_semantic_cycle_count_handles_long_chains_without_recursion() -> None:
    edges = [
        _StepPmiSemanticGraphEdge(
            source=f"#{index}",
            target=f"#{index + 1}",
            relationship="step_reference",
        )
        for index in range(1_500)
    ]

    assert _step_pmi_semantic_cycle_count(edges) == 0


def test_step_pmi_semantic_graph_includes_callout_and_associativity_records(tmp_path: Path) -> None:
    source = tmp_path / "pmi-callout.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#20=PRODUCT_DEFINITION_SHAPE('bracket','',#19);\n"
        "#21=SHAPE_ASPECT('hole face','',#20,.T.);\n"
        "#30=DIMENSIONAL_SIZE(#21,'hole diameter 12.5 mm',12.5);\n"
        "#31=ANNOTATION_TEXT_OCCURRENCE('hole callout text',#21);\n"
        "#40=DRAUGHTING_CALLOUT('hole callout',(#30,#31));\n"
        "#41=DRAUGHTING_CALLOUT_RELATIONSHIP('callout relationship','',#40,#30);\n"
        "#42=ANNOTATION_OCCURRENCE_ASSOCIATIVITY('annotation target',#31,#21);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=True))
    payload = graph.to_dict()

    assert graph.summary == {
        "nodes": 7,
        "pmi_nodes": 3,
        "referenced_nodes": 4,
        "edges": 9,
        "missing_references": 0,
        "cycles": 0,
    }
    assert [node["id"] for node in payload["nodes"]] == ["#20", "#21", "#30", "#31", "#40", "#41", "#42"]
    assert payload["nodes"][5]["kind"] == "pmi_relationship"
    assert payload["nodes"][6]["kind"] == "pmi_association"
    assert {"source": "#41", "target": "#40", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#41", "target": "#30", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#42", "target": "#31", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#42", "target": "#21", "relationship": "step_reference"} in payload["edges"]


def test_step_pmi_semantic_graph_includes_tolerance_zone_records(tmp_path: Path) -> None:
    source = tmp_path / "pmi-tolerance-zone.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#20=PRODUCT_DEFINITION_SHAPE('bracket','',#19);\n"
        "#21=SHAPE_ASPECT('hole axis','',#20,.T.);\n"
        "#30=GEOMETRIC_TOLERANCE('position tolerance',0.2,#21);\n"
        "#40=TOLERANCE_ZONE_FORM('cylindrical');\n"
        "#42=TOLERANCE_ZONE('position zone','',#20,.T.,(#30),#40);\n"
        "#43=PROJECTED_ZONE_DEFINITION(#42,(#21),#21,#50);\n"
        "#50=LENGTH_MEASURE_WITH_UNIT(LENGTH_MEASURE(12.0),#51);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=True))
    payload = graph.to_dict()

    assert graph.summary == {
        "nodes": 6,
        "pmi_nodes": 1,
        "referenced_nodes": 5,
        "edges": 7,
        "missing_references": 0,
        "cycles": 0,
    }
    assert [node["id"] for node in payload["nodes"]] == ["#20", "#21", "#30", "#40", "#42", "#43"]
    assert payload["nodes"][3]["kind"] == "pmi_tolerance_zone_form"
    assert payload["nodes"][4]["kind"] == "pmi_tolerance_zone"
    assert payload["nodes"][5]["kind"] == "pmi_tolerance_zone_definition"
    assert {"source": "#42", "target": "#30", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#42", "target": "#40", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#43", "target": "#42", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#43", "target": "#21", "relationship": "step_reference"} in payload["edges"]


def test_step_pmi_semantic_graph_includes_annotation_presentation_records(tmp_path: Path) -> None:
    source = tmp_path / "pmi-presentation.step"
    source.write_text(
        "ISO-10303-21;\n"
        "DATA;\n"
        "#20=PRODUCT_DEFINITION_SHAPE('bracket','',#19);\n"
        "#21=SHAPE_ASPECT('hole axis','',#20,.T.);\n"
        "#30=GEOMETRIC_TOLERANCE('position tolerance',0.2,#21);\n"
        "#40=DRAUGHTING_CALLOUT('position callout',(#41,#30));\n"
        "#41=ANNOTATION_CURVE_OCCURRENCE('position callout',(#42),#43);\n"
        "#42=GEOMETRIC_CURVE_SET('position polyline',(#44));\n"
        "#43=PRESENTATION_STYLE_ASSIGNMENT((#45));\n"
        "#45=CURVE_STYLE('solid',#46,#47);\n"
        "#46=DRAUGHTING_PRE_DEFINED_CURVE_FONT('continuous');\n"
        "#47=COLOUR_RGB('red',1.0,0.0,0.0);\n"
        "#50=DRAUGHTING_MODEL('pmi presentation',(#40,#41),#60);\n"
        "#51=DRAUGHTING_MODEL_ITEM_ASSOCIATION('semantic link','',#50,#41,#30);\n"
        "ENDSEC;\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=True))
    payload = graph.to_dict()

    assert graph.summary == {
        "nodes": 12,
        "pmi_nodes": 2,
        "referenced_nodes": 10,
        "edges": 14,
        "missing_references": 0,
        "cycles": 0,
    }
    assert [node["id"] for node in payload["nodes"]] == [
        "#20",
        "#21",
        "#30",
        "#40",
        "#41",
        "#42",
        "#43",
        "#45",
        "#46",
        "#47",
        "#50",
        "#51",
    ]
    node_kinds = {node["id"]: node["kind"] for node in payload["nodes"]}
    assert node_kinds["#41"] == "pmi_presentation"
    assert node_kinds["#42"] == "pmi_presentation_geometry"
    assert node_kinds["#45"] == "pmi_presentation_style"
    assert node_kinds["#50"] == "pmi_presentation"
    assert node_kinds["#51"] == "pmi_presentation_association"
    assert {"source": "#41", "target": "#42", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#43", "target": "#45", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#45", "target": "#46", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#50", "target": "#40", "relationship": "step_reference"} in payload["edges"]
    assert {"source": "#51", "target": "#30", "relationship": "step_reference"} in payload["edges"]


def test_step_text_pmi_extraction_respects_disabled_pmi(tmp_path: Path) -> None:
    source = tmp_path / "pmi.step"
    source.write_text("#1=GEOMETRIC_TOLERANCE('flatness',0.05,#2);\n", encoding="utf-8")

    assert _extract_step_pmi_annotations(source, StepReadOptions(pmi=False)) == []
    assert _extract_step_pmi_semantic_graph(source, StepReadOptions(pmi=False)).summary["nodes"] == 0


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
            "cycles": 0,
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
        "semantic_graph_cycles": 0,
    }


def test_step_pmi_annotation_text_decodes_unicode_escapes(tmp_path: Path) -> None:
    source = tmp_path / "pmi.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=ANNOTATION_TEXT_OCCURRENCE('len \\X2\\00B5\\X0\\m');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    annotations = step_pmi._extract_step_pmi_annotations(source, StepReadOptions(pmi=True))

    assert any("µm" in annotation.text for annotation in annotations)


def test_pmi_extraction_completes_on_unterminated_string(tmp_path: Path) -> None:
    source = tmp_path / "pmi.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=BROKEN_ENTITY('unterminated\n"
        "#2=DIMENSIONAL_SIZE(#9,'diameter 5.0');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    annotations = step_pmi._extract_step_pmi_annotations(source, StepReadOptions(pmi=True))

    assert any("diameter" in annotation.text for annotation in annotations)


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
