from __future__ import annotations

import pytest

import fascat as fc
from fascat import profiles


def test_profiles_produce_deterministic_option_sets() -> None:
    desktop = profiles.realtime_desktop()
    web = profiles.realtime_web()

    assert desktop.to_dict()["name"] == "realtime-desktop"
    assert web.to_dict()["name"] == "realtime-web"
    assert desktop.tessellation is not None
    assert desktop.tessellation.sag == 0.1
    assert desktop.optimize is not None
    assert desktop.optimize.target_triangles == 1_000_000
    assert desktop.lods is not None
    assert desktop.lods.ratios == (0.5, 0.25, 0.1)


def test_lod_options_normalize_list_ratios() -> None:
    options = fc.LODOptions(ratios=[0.5, 0.25, 0.1])

    assert options.ratios == (0.5, 0.25, 0.1)
    assert options.to_dict()["ratios"] == [0.5, 0.25, 0.1]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: fc.Tessellation(sag=0), "sag"),
        (lambda: fc.Tessellation(angle=181), "angle"),
        (lambda: fc.RepairOptions(tolerance=-1), "tolerance"),
        (lambda: fc.StageOptions(materials="bad"), "materials"),
        (lambda: fc.OptimizeOptions(ratio=1), "ratio"),
        (lambda: fc.LODOptions((0.25, 0.5)), "sorted"),
    ],
)
def test_options_validate_bad_inputs(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
