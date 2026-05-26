from __future__ import annotations

import pytest

import fascat as fc
from fascat import profiles
from fascat.options import ConversionProfile


@pytest.mark.parametrize(
    ("profile", "name", "sag", "angle", "target_triangles", "uv0", "lods"),
    [
        (profiles.inspect_only(), "inspect-only", None, None, None, "none", None),
        (profiles.realtime_desktop(), "realtime-desktop", 0.1, 15.0, 1_000_000, "box", (0.5, 0.25, 0.1)),
        (profiles.realtime_web(), "realtime-web", 0.2, 20.0, 250_000, "box", (0.5, 0.25)),
        (profiles.virtual_reality(), "virtual-reality", 0.15, 15.0, 500_000, "box", (0.5, 0.25, 0.125)),
    ],
)
def test_profiles_match_documented_default_table(
    profile: ConversionProfile,
    name: str,
    sag: float | None,
    angle: float | None,
    target_triangles: int | None,
    uv0: str,
    lods: tuple[float, ...] | None,
) -> None:
    assert profile.to_dict()["name"] == name
    assert profiles.by_name(name).to_dict() == profile.to_dict()
    assert profile.stage.uv0 == uv0
    assert profile.stage.uv1 is None

    if sag is None:
        assert profile.tessellation is None
    else:
        assert profile.tessellation is not None
        assert profile.tessellation.sag == sag
        assert profile.tessellation.angle == angle

    if target_triangles is None:
        assert profile.optimize is None
    else:
        assert profile.optimize is not None
        assert profile.optimize.target_triangles == target_triangles

    if lods is None:
        assert profile.lods is None
    else:
        assert profile.lods is not None
        assert profile.lods.ratios == lods


def test_lod_options_normalize_list_ratios() -> None:
    options = fc.LODOptions(ratios=[0.5, 0.25, 0.1])

    assert options.ratios == (0.5, 0.25, 0.1)
    assert options.to_dict()["ratios"] == [0.5, 0.25, 0.1]


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: fc.Tessellation(sag=0), "sag"),
        (lambda: fc.Tessellation(angle=0), "angle"),
        (lambda: fc.Tessellation(angle=181), "angle"),
        (lambda: fc.Tessellation(min_edge_length=0), "min_edge_length"),
        (lambda: fc.Tessellation(max_edge_length=0), "max_edge_length"),
        (lambda: fc.Tessellation(min_edge_length=2, max_edge_length=1), "min_edge_length"),
        (lambda: fc.Tessellation(part_settings={"part": {"bad": True}}), "unsupported part_settings"),
        (lambda: fc.RepairOptions(tolerance=-1), "tolerance"),
        (lambda: fc.RepairOptions(area_epsilon=-1), "area_epsilon"),
        (lambda: fc.StageOptions(materials="bad"), "materials"),
        (lambda: fc.StageOptions(material_mode="bad"), "material_mode"),
        (lambda: fc.StageOptions(normal_mode="bad"), "normal_mode"),
        (lambda: fc.StageOptions(hard_edge_angle=0), "hard_edge_angle"),
        (lambda: fc.StageOptions(uv0="bad"), "uv0"),
        (lambda: fc.StageOptions(uv1="bad"), "uv1"),
        (lambda: fc.UnwrapOptions(texel_density=0), "texel_density"),
        (lambda: fc.UnwrapOptions(padding=-1), "padding"),
        (lambda: fc.UnwrapOptions(max_stretch=-1), "max_stretch"),
        (lambda: fc.AtlasOptions(max_size=0), "atlas max_size"),
        (lambda: fc.OptimizeOptions(target_triangles=0), "target_triangles"),
        (lambda: fc.OptimizeOptions(ratio=0), "ratio"),
        (lambda: fc.OptimizeOptions(ratio=1), "ratio"),
        (lambda: fc.OptimizeOptions(hard_edge_angle=0), "hard_edge_angle"),
        (lambda: fc.OptimizeOptions(hard_edge_angle=181), "hard_edge_angle"),
        (lambda: fc.OptimizeOptions(small_part_triangle_threshold=-1), "small_part_triangle_threshold"),
        (lambda: fc.LODOptions(()), "at least one"),
        (lambda: fc.LODOptions((1.0,)), "greater than 0"),
        (lambda: fc.LODOptions((0.25, 0.5)), "sorted"),
        (lambda: fc.LODOptions((0.5,), mode="payloads"), "variant-based"),
    ],
)
def test_options_validate_bad_inputs(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
