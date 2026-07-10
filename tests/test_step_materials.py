from __future__ import annotations

import json
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

import fascat as fc
from fascat.io.step import materials as step_materials
from fascat.io.step import textures as step_textures
from fascat.io.step.materials import (
    _apply_material_libraries_to_materials,
    _apply_material_library_mapping,
    _CadMaterialSpec,
    _color_material_spec,
    _extract_material_libraries,
    _material_binding_plan,
)
from fascat.io.step.textures import (
    _attach_source_textures_to_materials,
    _extract_source_textures,
)
from fascat.options import StepReadOptions


def test_material_binding_plan_maps_step_face_colors_to_indices() -> None:
    material_ids, material_indices = _material_binding_plan(
        "mat-red",
        ["mat-blue", "mat-red", "mat-green", "mat-blue", "mat-yellow", "mat-green"],
    )

    assert material_ids == ["mat-red", "mat-blue", "mat-green", "mat-yellow"]
    assert material_indices == [1, 0, 2, 1, 3, 2]


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


def test_step_source_texture_extraction_loads_ktx2_dimensions(tmp_path: Path) -> None:
    texture = tmp_path / "panel_baseColor.ktx2"
    data = bytearray(b"\xabKTX 20\xbb\r\n\x1a\n" + b"\0" * 68)
    data[20:24] = (4).to_bytes(4, "little")
    data[24:28] = (2).to_bytes(4, "little")
    data[36:40] = (1).to_bytes(4, "little")
    data[40:44] = (1).to_bytes(4, "little")
    texture.write_bytes(bytes(data))
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('panel_baseColor.ktx2');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())

    image = next(iter(extraction.images.values()))
    assert extraction.summary == {"references": 1, "resolved": 1, "missing": 0, "unsupported": 0, "unreadable": 0}
    assert image.mime_type == "image/ktx2"
    assert (image.width, image.height) == (4, 2)


def test_truncated_ktx2_source_texture_is_reported_unreadable(tmp_path: Path) -> None:
    texture = tmp_path / "panel_baseColor.ktx2"
    data = bytearray(b"\xabKTX 20\xbb\r\n\x1a\n" + b"\0" * 16)
    data[20:24] = (4).to_bytes(4, "little")
    data[24:28] = (2).to_bytes(4, "little")
    texture.write_bytes(bytes(data))
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('panel_baseColor.ktx2');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())

    assert extraction.images == {}
    assert extraction.summary == {"references": 1, "resolved": 0, "missing": 0, "unsupported": 0, "unreadable": 1}
    assert any("could not be read as KTX2" in warning for warning in extraction.warnings)


def test_resolve_source_texture_rejects_parent_directory_traversal(tmp_path: Path) -> None:
    from fascat.io.step.textures import _resolve_source_texture

    root = tmp_path / "cad"
    root.mkdir()
    outside = tmp_path / "leak.png"
    outside.write_bytes(b"png")

    assert _resolve_source_texture("../leak.png", [root]) is None


def test_resolve_source_texture_rejects_absolute_reference_outside_roots(tmp_path: Path) -> None:
    from fascat.io.step.textures import _resolve_source_texture

    root = tmp_path / "cad"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"png")

    assert _resolve_source_texture(str(outside), [root]) is None


def test_resolve_source_texture_allows_absolute_reference_inside_search_root(tmp_path: Path) -> None:
    from fascat.io.step.textures import _resolve_source_texture

    root = tmp_path / "cad"
    root.mkdir()
    inside = root / "panel.png"
    inside.write_bytes(b"png")

    assert _resolve_source_texture(str(inside), [root]) == inside


def test_resolve_source_texture_allows_dotdot_that_stays_within_root(tmp_path: Path) -> None:
    from fascat.io.step.textures import _resolve_source_texture

    root = tmp_path / "cad"
    (root / "sub").mkdir(parents=True)
    inside = root / "panel.png"
    inside.write_bytes(b"png")

    resolved = _resolve_source_texture("sub/../panel.png", [root])

    assert resolved is not None
    assert resolved.resolve() == inside.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlinks")
def test_resolve_source_texture_rejects_symlink_escape(tmp_path: Path) -> None:
    from fascat.io.step.textures import _resolve_source_texture

    root = tmp_path / "cad"
    root.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(b"png")
    (root / "alias.png").symlink_to(outside)

    assert _resolve_source_texture("alias.png", [root]) is None


def test_resolve_material_library_reference_rejects_traversal(tmp_path: Path) -> None:
    from fascat.io.step.materials import _resolve_material_library_reference

    root = tmp_path / "cad"
    root.mkdir()
    outside = tmp_path / "library.json"
    outside.write_text("{}", encoding="utf-8")

    assert _resolve_material_library_reference("../library.json", [root]) is None
    assert _resolve_material_library_reference(str(outside), [root]) is None


def test_step_source_texture_extraction_does_not_resolve_traversal_reference(tmp_path: Path) -> None:
    root = tmp_path / "cad"
    root.mkdir()
    leak = tmp_path / "leak.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(leak)
    source = root / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('../leak.png');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())

    assert extraction.images == {}
    assert extraction.summary["missing"] == 1
    assert extraction.summary["resolved"] == 0


def test_step_source_texture_reference_decodes_unicode_filename(tmp_path: Path) -> None:
    texture = tmp_path / "panél.png"
    Image.new("RGBA", (2, 2), (10, 20, 30, 255)).save(texture)
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('pan\\X2\\00E9\\X0\\l.png');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())

    assert extraction.summary["resolved"] == 1


def test_oversized_material_library_is_reported_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_materials, "_MAX_MATERIAL_LIBRARY_BYTES", 4)
    library = tmp_path / "materials.json"
    library.write_text('{"materials": [{"name": "steel"}]}', encoding="utf-8")
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('materials.json');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())

    assert extraction.summary["unreadable"] == 1
    assert any("is too large" in warning for warning in extraction.warnings)


def test_oversized_source_texture_is_reported_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_textures, "_MAX_SOURCE_TEXTURE_BYTES", 4)
    texture = tmp_path / "panel.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(texture)
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('panel.png');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_source_textures(source, "panel.step", StepReadOptions())

    assert extraction.summary["unreadable"] == 1
    assert any("is too large" in warning for warning in extraction.warnings)


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


@pytest.mark.parametrize(
    ("color_space", "color", "expected"),
    (
        ("auto", [128, 64, 32], (128 / 255, 64 / 255, 32 / 255, 1.0)),
        ("srgb255", [128, 64, 32, 128], (128 / 255, 64 / 255, 32 / 255, 128 / 255)),
        ("linear", [1.5, 0.5, 0.25, 1.0], (1.0, 0.5, 0.25, 1.0)),
    ),
)
def test_step_material_library_color_space_is_explicit(
    tmp_path: Path, color_space: str, color: list[float], expected: tuple[float, float, float, float]
) -> None:
    library = tmp_path / "vendor-materials.json"
    library.write_text(
        json.dumps({"materials": [{"materialName": "Paint", "baseColorFactor": color}]}),
        encoding="utf-8",
    )
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.json');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    materials = {"paint": fc.Material(id="paint", name="Paint", base_color=(0.75, 0.75, 0.75, 1.0))}

    extraction = _extract_material_libraries(
        source, "panel.step", StepReadOptions(material_library_color_space=color_space)
    )
    _apply_material_libraries_to_materials(materials, extraction)
    material = materials["paint"]

    assert material.base_color == pytest.approx(expected)
    assert material.metadata["material_library_color_space"] == color_space


def test_step_material_library_json_dedupes_duplicate_texture_slots(tmp_path: Path) -> None:
    first_base = tmp_path / "first_base.png"
    second_base = tmp_path / "second_base.png"
    normal = tmp_path / "normal.png"
    Image.new("RGBA", (1, 1), (200, 10, 10, 255)).save(first_base)
    Image.new("RGBA", (1, 1), (10, 200, 10, 255)).save(second_base)
    Image.new("RGBA", (1, 1), (128, 128, 255, 255)).save(normal)
    library = tmp_path / "vendor-materials.json"
    library.write_text(
        json.dumps(
            {
                "materials": [
                    {
                        "materialName": "Paint",
                        "textures": {"baseColor": first_base.name},
                        "baseColorTexture": second_base.name,
                        "normalTexture": normal.name,
                        "pbrMetallicRoughness": {"baseColorTexture": second_base.name},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.json');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    materials = {"paint": fc.Material(id="paint", name="Paint", base_color=(0.75, 0.75, 0.75, 1.0))}

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())
    summary = _apply_material_libraries_to_materials(materials, extraction)
    material = materials["paint"]
    base_image = extraction.images[str(material.metadata["source_texture_base_color_image"])]

    assert extraction.summary["textures"] == 2
    assert summary["bound_textures"] == 2
    assert base_image.metadata["source_texture_path"].endswith("first_base.png")
    assert material.metadata["source_texture_normal_image"] in extraction.images
    assert material.metadata["source_texture_slots"] == "base_color,normal"


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


def test_material_library_zip_entry_cap_is_reported_unreadable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(step_materials, "_MAX_MATERIAL_LIBRARY_ARCHIVE_ENTRIES", 1)
    library = tmp_path / "vendor-materials.zip"
    with zipfile.ZipFile(library, "w") as archive:
        archive.writestr("materials/a.json", '{"materials":[{"name":"a"}]}')
        archive.writestr("materials/b.json", '{"materials":[{"name":"b"}]}')
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.zip');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())

    assert extraction.summary["unreadable"] == 1
    assert any("too many entries" in warning for warning in extraction.warnings)


def test_material_library_zip_uncompressed_size_cap_is_reported_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(step_materials, "_MAX_MATERIAL_LIBRARY_ARCHIVE_UNCOMPRESSED_BYTES", 8)
    library = tmp_path / "vendor-materials.zip"
    with zipfile.ZipFile(library, "w") as archive:
        archive.writestr("materials/vendor.json", '{"materials":[{"name":"steel"}]}')

    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.zip');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())

    assert extraction.summary["unreadable"] == 1
    assert any("uncompressed payload is too large" in warning for warning in extraction.warnings)


def test_material_library_json_depth_cap_is_reported_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(step_materials, "_MAX_MATERIAL_LIBRARY_JSON_DEPTH", 1)
    library = tmp_path / "vendor-materials.json"
    library.write_text(
        json.dumps({"library": {"library": {"materials": [{"name": "steel"}]}}}),
        encoding="utf-8",
    )
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.json');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())

    assert extraction.summary["unreadable"] == 1
    assert any("JSON nesting is too deep" in warning for warning in extraction.warnings)


def test_material_library_zip_json_depth_cap_is_reported_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(step_materials, "_MAX_MATERIAL_LIBRARY_JSON_DEPTH", 1)
    library = tmp_path / "vendor-materials.zip"
    payload = {"library": {"library": {"materials": [{"name": "steel"}]}}}
    with zipfile.ZipFile(library, "w") as archive:
        archive.writestr("materials/vendor.json", json.dumps(payload))
    source = tmp_path / "panel.step"
    source.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('vendor-materials.zip');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    extraction = _extract_material_libraries(source, "panel.step", StepReadOptions())

    assert extraction.summary["unreadable"] == 1
    assert any("JSON nesting is too deep" in warning for warning in extraction.warnings)


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
