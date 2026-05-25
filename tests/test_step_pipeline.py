from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc


@pytest.mark.parametrize("fixture", sorted(Path("tests/fixtures").glob("*.step")))
def test_step_fixtures_import_with_names_units_and_parts(fixture: Path) -> None:
    asset = fc.read_step(fixture)

    assert asset.part_count >= 1
    assert asset.occurrence_count >= asset.part_count
    assert asset.units
    assert asset.meters_per_unit > 0.0
    assert asset.root.children
    assert asset.report.steps[0].name == "import"


def test_step_fixture_converts_to_valid_usd_with_report(tmp_path: Path) -> None:
    output = tmp_path / "spool.usda"

    asset = fc.convert(
        "tests/fixtures/spool-clamp-lid.step",
        output,
        tessellation=fc.Tessellation(sag=0.2, angle=20),
        optimize=fc.OptimizeOptions(target_triangles=120),
        lods=fc.LODOptions((0.5,)),
    )

    assert output.exists()
    assert asset.triangle_count <= 120
    assert {step.name for step in asset.report.steps} >= {"import", "tessellate", "repair", "stage", "optimize", "lods"}
    assert fc.validate_usd(output)["triangles"] == asset.triangle_count
