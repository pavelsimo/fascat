from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import fascat as fc

pytestmark = [pytest.mark.requires_ocp, pytest.mark.requires_usd]


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


def test_tessellation_max_edge_length_limits_fixture_edges() -> None:
    asset = fc.read_step("tests/fixtures/spool-clamp-lid.step").tessellate(
        fc.Tessellation(sag=0.2, angle=20, max_edge_length=10.0, create_normals=False)
    )
    mesh = next(part.mesh for part in asset.parts.values() if part.mesh is not None)
    edge_lengths = []
    for face in mesh.faces:
        corners = mesh.points[face]
        edge_lengths.extend(
            [
                np.linalg.norm(corners[1] - corners[0]),
                np.linalg.norm(corners[2] - corners[1]),
                np.linalg.norm(corners[0] - corners[2]),
            ]
        )

    assert max(edge_lengths) <= 10.0
    assert mesh.normals is None


def test_convert_progress_callback_receives_stage_stats(tmp_path: Path) -> None:
    output = tmp_path / "spool.usda"
    progress: list[tuple[str, dict[str, int]]] = []

    fc.convert(
        "tests/fixtures/spool-clamp-lid.step",
        output,
        tessellation=fc.Tessellation(sag=0.2, angle=20),
        optimize=fc.OptimizeOptions(target_triangles=120),
        lods=fc.LODOptions((0.5,)),
        progress=lambda step, stats: progress.append((step, stats)),
    )

    assert [step for step, _stats in progress] == [
        "source",
        "tessellate",
        "repair",
        "stage",
        "optimize",
        "lods",
        "write",
    ]
    assert all("triangles" in stats for _step, stats in progress)
