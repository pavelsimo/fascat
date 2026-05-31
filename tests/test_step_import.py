from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import fascat as fc
from fascat.io.step import (
    _apply_material_library_mapping,
    _attach_source_textures_to_materials,
    _CadMaterialSpec,
    _canonical_part_id,
    _cleanup_action,
    _color_material_spec,
    _extract_source_textures,
    _import_decisions,
    _import_warnings,
    _ImportCleanupStats,
    _loaded_representation,
    _loaded_representation_report,
    _material_binding_plan,
    _shape_fingerprint,
    _ShapeTopologyCounts,
    _space_normalization,
    _StepHeaderInfo,
)
from fascat.options import StepReadOptions


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
        "STEP file advertises AP242 PMI, but PMI entity import is not implemented; annotations are omitted",
        "STEP design variant import is not implemented; variants are omitted",
        "multi-file STEP assembly import is not implemented; external references are not loaded",
    ]


def test_step_import_cleanup_actions_cover_construction_only_shapes() -> None:
    point_counts = _ShapeTopologyCounts(vertices=3)
    line_counts = _ShapeTopologyCounts(vertices=4, edges=2)
    brep_counts = _ShapeTopologyCounts(vertices=8, edges=12, faces=6)

    assert _loaded_representation(point_counts) == "construction_points"
    assert _loaded_representation(line_counts) == "construction_lines"
    assert _loaded_representation(brep_counts) == "brep"
    assert _cleanup_action(point_counts, StepReadOptions(delete_free_vertices=True)) == "delete_free_vertices"
    assert _cleanup_action(line_counts, StepReadOptions(delete_lines=True)) == "delete_lines"
    assert _cleanup_action(brep_counts, StepReadOptions(delete_free_vertices=True, delete_lines=True)) is None


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
    assert decisions["design_variants"]["state"] == "unsupported"
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
    assert decisions["space_normalization"]["state"] == "honored"


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
