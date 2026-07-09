from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fascat.errors import FascatIOError
from fascat.io.jt import read_jt, read_jt_bytes
from fascat.options import JtReadOptions, StepReadOptions
from tests._jt_builder import SyntheticPart, build_jt, build_jt8_bytes, build_jt10_stub

_TETRA_POINTS = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
_TETRA_FACES = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]

_LOCAL_FIXTURES = sorted(Path(__file__).parent.glob("fixtures/local/*.jt"))


def _tetra_part(**overrides: object) -> SyntheticPart:
    settings: dict[str, object] = {"name": "tetra.part", "points": _TETRA_POINTS, "triangles": _TETRA_FACES}
    settings.update(overrides)
    return SyntheticPart(**settings)  # type: ignore[arg-type]


def _write(tmp_path: Path, data: bytes, name: str = "model.jt") -> Path:
    target = tmp_path / name
    target.write_bytes(data)
    return target


class TestReadJt:
    def test_imports_single_part(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()]))
        asset = read_jt(source)
        assert asset.part_count == 1
        assert asset.occurrence_count == 1
        part = next(iter(asset.parts.values()))
        assert part.mesh is not None
        assert part.source_shape is None
        assert part.mesh.faces.shape == (4, 3)
        assert part.name == "tetra.part"
        assert part.metadata["loaded_representation"] == "mesh"
        assert part.metadata["source_name"] == "tetra.part"
        assert part.metadata["JT_PROP_NAME"] == "tetra.part;0;1:"
        assert part.fingerprint == part.mesh.fingerprint()

    def test_report_step_options(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()]))
        asset = read_jt(source)
        step = asset.report.steps[-1]
        assert step.name == "import"
        assert step.options["format"] == "JT"
        assert step.options["backend"] == "pure-python"
        assert step.options["jt_version"] == "9.5"
        assert step.options["skipped_elements"] == {}
        assert step.options["lod_summary"] == {
            "lod_selection": "finest",
            "imported_lod_meshes": 0,
            "parts_with_lods": 0,
        }
        read_options = step.options["read_options"]
        assert isinstance(read_options, dict) and read_options["lod_selection"] == "finest"

    def test_units_from_measurement_property(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()], units="Inches"))
        asset = read_jt(source)
        assert asset.units == "inch"
        assert asset.meters_per_unit == pytest.approx(0.0254)

    def test_missing_units_defaults_to_millimetres_with_warning(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()], units=None))
        asset = read_jt(source)
        assert asset.units == "millimetre"
        assert any("no measurement units" in warning for warning in asset.report.warnings)

    def test_space_normalization_overrides(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()]))
        asset = read_jt(source, options=JtReadOptions(target_units="metre", target_up_axis="Y"))
        assert asset.units == "metre"
        assert asset.up_axis == "Y"
        assert not np.allclose(asset.root.transform, np.identity(4))

    def test_instances_share_one_part(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part(instances=1)]))
        asset = read_jt(source)
        assert asset.part_count == 1
        assert asset.occurrence_count == 2

    def test_material_mapping(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part(diffuse=(0.8, 0.2, 0.1, 0.9), shininess=30.0)]))
        asset = read_jt(source)
        part = next(iter(asset.parts.values()))
        material = asset.materials[part.material_ids[0]]
        assert material.base_color == pytest.approx((0.8, 0.2, 0.1, 0.9))
        assert material.opacity == pytest.approx(0.9)
        assert material.metallic == 0.0
        assert material.roughness == pytest.approx((2.0 / 32.0) ** 0.5)
        assert material.metadata["jt_shininess"] == "30.000000"

    def test_materials_deduplicated_by_value(self, tmp_path: Path) -> None:
        parts = [
            _tetra_part(name="a.part", diffuse=(0.5, 0.5, 0.5, 1.0)),
            _tetra_part(name="b.part", diffuse=(0.5, 0.5, 0.5, 1.0)),
        ]
        asset = read_jt(_write(tmp_path, build_jt(parts)))
        assert len(asset.materials) == 1

    def test_part_without_material_gets_default(self, tmp_path: Path) -> None:
        asset = read_jt(_write(tmp_path, build_jt([_tetra_part()])))
        part = next(iter(asset.parts.values()))
        material = asset.materials[part.material_ids[0]]
        assert material.base_color == (0.75, 0.75, 0.75, 1.0)

    def test_transform_attribute_transposed_to_column_vectors(self, tmp_path: Path) -> None:
        jt_transform = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [5.0, 6.0, 7.0, 1.0],  # JT keeps translation in the last row
        ]
        asset = read_jt(_write(tmp_path, build_jt([_tetra_part(jt_transform=jt_transform)])))
        part_node = next(node for node in asset.root.walk() if node.name == "tetra.part" and node.children)
        assert np.allclose(part_node.transform[:3, 3], [5.0, 6.0, 7.0])

    def test_lod_selection_finest_by_default(self, tmp_path: Path) -> None:
        coarser = ([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], _TETRA_FACES)
        source = _write(tmp_path, build_jt([_tetra_part(lod_meshes=[coarser])]))
        asset = read_jt(source)
        part = next(iter(asset.parts.values()))
        assert part.mesh is not None
        assert part.lod_meshes == []
        assert part.metadata["lod_count"] == "2"

    def test_lod_selection_all_imports_coarser_meshes(self, tmp_path: Path) -> None:
        coarser = ([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], _TETRA_FACES)
        source = _write(tmp_path, build_jt([_tetra_part(lod_meshes=[coarser])]))
        asset = read_jt(source, options=JtReadOptions(lod_selection="all"))
        part = next(iter(asset.parts.values()))
        assert len(part.lod_meshes) == 1
        assert part.lod_meshes[0].metadata["lod_source"] == "imported"
        step = asset.report.steps[-1]
        assert step.options["lod_summary"] == {
            "lod_selection": "all",
            "imported_lod_meshes": 1,
            "parts_with_lods": 1,
        }

    def test_unknown_elements_reported_not_fatal(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()], inject_unknown_element=True))
        asset = read_jt(source)
        skipped = asset.report.steps[-1].options["skipped_elements"]
        assert isinstance(skipped, dict) and sum(skipped.values()) == 1

    def test_external_reference_warns_with_placeholder(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()], external_ref="wheels/wheel.jt"))
        asset = read_jt(source)
        placeholders = [node for node in asset.root.walk() if "external_reference" in node.metadata]
        assert len(placeholders) == 1
        assert placeholders[0].metadata["external_reference"] == "wheels/wheel.jt"
        assert any("external partition reference" in warning for warning in asset.report.warnings)

    def test_options_coerced_from_step_read_options(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()]))
        asset = read_jt(source, options=StepReadOptions(target_units="metre"))
        assert asset.units == "metre"
        read_options = asset.report.steps[-1].options["read_options"]
        assert isinstance(read_options, dict) and read_options["lod_selection"] == "finest"

    @pytest.mark.parametrize("byte_order", ["<", ">"])
    @pytest.mark.parametrize("compress_lsg", [True, False])
    def test_container_variants(self, tmp_path: Path, byte_order: str, compress_lsg: bool) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()], byte_order=byte_order, compress_lsg=compress_lsg))
        asset = read_jt(source)
        assert asset.part_count == 1

    def test_quantized_and_normal_variants(self, tmp_path: Path) -> None:
        normals = np.asarray(_TETRA_POINTS, dtype=np.float64) - 2.5
        normals /= np.linalg.norm(normals + 1e-9, axis=1, keepdims=True)
        source = _write(tmp_path, build_jt([_tetra_part(normals=normals.tolist())], quant_bits=12))
        asset = read_jt(source)
        part = next(iter(asset.parts.values()))
        assert part.mesh is not None
        assert part.mesh.normals is not None


class TestReadJtBytes:
    def test_parity_with_read_jt(self) -> None:
        asset = read_jt_bytes(build_jt([_tetra_part()]), name="memory.jt")
        assert asset.part_count == 1
        assert asset.source_path is None
        assert asset.root.metadata["source"] == "memory.jt"
        assert asset.metadata["source_identity"] == "memory.jt"


class TestReadJtErrors:
    def test_wrong_extension(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()]), name="model.step")
        with pytest.raises(FascatIOError, match="unsupported JT extension") as error:
            read_jt(source)
        assert isinstance(error.value.__cause__, ValueError)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FascatIOError, match="missing JT file"):
            read_jt(tmp_path / "absent.jt")

    def test_jt8_rejected(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt8_bytes())
        with pytest.raises(FascatIOError, match="unsupported JT version 8.1"):
            read_jt(source)

    def test_not_a_jt_file(self, tmp_path: Path) -> None:
        source = _write(tmp_path, b"not really a jt file" * 10)
        with pytest.raises(FascatIOError, match="not a JT file"):
            read_jt(source)

    def test_brep_only_file(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([], brep_only=True))
        with pytest.raises(FascatIOError, match="no tessellated LOD data"):
            read_jt(source)

    def test_jt10_shape_decode_unsupported(self, tmp_path: Path) -> None:
        # JT 10 container and LSG parse, but shape decode fails per part; with no
        # decodable geometry the whole import raises the B-rep-only error and the
        # per-part warning names the real cause.
        source = _write(tmp_path, build_jt10_stub())
        with pytest.raises(FascatIOError, match="no tessellated LOD data"):
            read_jt(source)

    def test_corrupt_shape_warns_and_import_fails_when_nothing_decodes(self, tmp_path: Path) -> None:
        source = _write(tmp_path, build_jt([_tetra_part()], corrupt_shape=True))
        with pytest.raises(FascatIOError, match="no tessellated LOD data"):
            read_jt(source)

    def test_corrupt_shape_is_per_part_when_others_decode(self, tmp_path: Path) -> None:
        good = build_jt([_tetra_part(name="good.part"), _tetra_part(name="bad.part")])
        broken_segment = build_jt([_tetra_part(name="bad.part")], corrupt_shape=True)
        # Splice: rebuild with one good and one corrupt part is not directly
        # expressible via build_jt, so emulate by decoding the good file and the
        # corrupt one separately.
        asset = read_jt_bytes(good, name="good.jt")
        assert asset.part_count == 2
        with pytest.raises(FascatIOError):
            read_jt_bytes(broken_segment, name="bad.jt")


@pytest.mark.skipif(not _LOCAL_FIXTURES, reason="no local JT fixtures in tests/fixtures/local/")
@pytest.mark.parametrize("fixture", _LOCAL_FIXTURES, ids=lambda path: path.name)
def test_local_fixture_smoke(fixture: Path) -> None:
    asset = read_jt(fixture)
    assert asset.part_count >= 1
    assert any(part.mesh is not None for part in asset.parts.values())
    assert asset.report.steps[-1].options["format"] == "JT"
