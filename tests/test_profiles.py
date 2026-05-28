from __future__ import annotations

import json
from pathlib import Path

import pytest

import fascat as fc
from fascat import profiles
from fascat.mesh import Mesh
from fascat.options import ConversionProfile


def _sized_mesh(size: float) -> Mesh:
    return Mesh(
        points=[
            [0.0, 0.0, 0.0],
            [size, 0.0, 0.0],
            [0.0, size, 0.0],
        ],
        faces=[[0, 1, 2]],
    )


@pytest.mark.parametrize(
    (
        "profile",
        "name",
        "sag",
        "angle",
        "target_triangles",
        "uv0",
        "lods",
        "target_fps",
        "max_vertices_per_mesh",
        "max_texture_resolution",
        "max_texture_memory_mb",
        "max_load_time_ms",
        "max_draw_calls",
        "unity_reference_profile",
        "unity_reference_triangles",
        "unity_reference_draw_calls",
    ),
    [
        (
            profiles.inspect_only(),
            "inspect-only",
            None,
            None,
            None,
            "none",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        (
            profiles.realtime_desktop(),
            "realtime-desktop",
            0.1,
            15.0,
            1_000_000,
            "box",
            (0.5, 0.25, 0.1),
            60,
            65_535,
            4_096,
            512,
            2_000,
            2_000,
            "desktop",
            (10_000_000, 100_000_000),
            10_000,
        ),
        (
            profiles.realtime_web(),
            "realtime-web",
            0.2,
            20.0,
            250_000,
            "box",
            (0.5, 0.25),
            60,
            65_535,
            2_048,
            128,
            3_000,
            500,
            "webgl",
            (100_000, 1_000_000),
            200,
        ),
        (
            profiles.realtime_mobile(),
            "realtime-mobile",
            0.25,
            20.0,
            150_000,
            "box",
            (0.5, 0.25),
            60,
            65_535,
            2_048,
            128,
            2_500,
            250,
            "mobile",
            (100_000, 500_000),
            1_000,
        ),
        (
            profiles.virtual_reality(),
            "virtual-reality",
            0.15,
            15.0,
            500_000,
            "box",
            (0.5, 0.25, 0.125),
            90,
            65_535,
            2_048,
            256,
            1_500,
            250,
            "vr",
            (500_000, 2_000_000),
            1_000,
        ),
        (
            profiles.augmented_reality(),
            "augmented-reality",
            0.3,
            22.5,
            100_000,
            "box",
            (0.5, 0.25),
            60,
            65_535,
            1_024,
            64,
            1_500,
            150,
            "ar",
            (50_000, 250_000),
            500,
        ),
        (
            profiles.mixed_reality(),
            "mixed-reality",
            0.35,
            25.0,
            75_000,
            "box",
            (0.5, 0.25),
            60,
            65_535,
            1_024,
            64,
            1_200,
            100,
            "mixed-reality",
            (50_000, 200_000),
            500,
        ),
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
    target_fps: int | None,
    max_vertices_per_mesh: int | None,
    max_texture_resolution: int | None,
    max_texture_memory_mb: int | None,
    max_load_time_ms: int | None,
    max_draw_calls: int | None,
    unity_reference_profile: str | None,
    unity_reference_triangles: tuple[int, int] | None,
    unity_reference_draw_calls: int | None,
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

    if target_fps is None:
        assert profile.budget is None
    else:
        assert profile.budget is not None
        assert profile.budget.target_fps == target_fps
        assert profile.budget.max_triangles == target_triangles
        assert profile.budget.max_vertices == target_triangles * 3
        assert profile.budget.max_vertices_per_mesh == max_vertices_per_mesh
        assert profile.budget.max_texture_resolution == max_texture_resolution
        assert profile.budget.max_texture_memory_mb == max_texture_memory_mb
        assert profile.budget.max_load_time_ms == max_load_time_ms
        assert profile.budget.max_draw_calls == max_draw_calls
        assert profile.budget.unity_reference_profile == unity_reference_profile
        assert profile.budget.unity_reference_triangles == unity_reference_triangles
        assert profile.budget.unity_reference_draw_calls == unity_reference_draw_calls
        assert profile.budget.to_dict()["unity_reference_triangles"] == list(unity_reference_triangles)


def test_lod_options_normalize_list_ratios() -> None:
    options = fc.LODOptions(ratios=[0.5, 0.25, 0.1])

    assert options.ratios == (0.5, 0.25, 0.1)
    assert options.to_dict()["ratios"] == [0.5, 0.25, 0.1]


def test_builtin_profiles_expose_unity_workflow_recipes() -> None:
    recipes = {
        "inspect-only": "inspectable-cad",
        "realtime-desktop": "high-fidelity-desktop",
        "realtime-web": "web-glb",
        "realtime-mobile": "mobile-glb",
        "virtual-reality": "vr-glb",
        "augmented-reality": "ar-glb",
        "mixed-reality": "mixed-reality-glb",
    }

    for profile_name, recipe_name in recipes.items():
        profile = profiles.by_name(profile_name)
        assert profile.recipe is not None
        assert profile.recipe.name == recipe_name
        assert profile.to_dict()["recipe"]["name"] == recipe_name
        assert profile.to_dict()["recipe"]["choices"]

    web_recipe = profiles.realtime_web().recipe
    assert web_recipe is not None
    choices = {choice.setting: choice for choice in web_recipe.choices}
    assert choices["sag_and_angle"].value == {"sag": 0.2, "angle": 20.0}
    assert choices["texture_compression"].status == "unsupported"
    assert choices["gltf_geometry_compression"].status == "metadata_only"


def test_target_device_profile_from_toml_overlays_base_budget(tmp_path: Path) -> None:
    profile_file = tmp_path / "factory-tablet.toml"
    profile_file.write_text(
        """
name = "factory-tablet-ar"

[budget]
target_fps = 60
max_triangles = 42000
max_texture_resolution = 512
supported_compression = ["meshopt"]
supported_runtime_extensions = ["KHR_mesh_quantization", "EXT_meshopt_compression"]
unity_reference_profile = "tablet-ar"
unity_reference_triangles = [30000, 60000]
""",
        encoding="utf-8",
    )

    profile = profiles.from_file(profile_file, base="realtime-mobile")

    assert profile.name == "factory-tablet-ar"
    assert profile.tessellation == profiles.realtime_mobile().tessellation
    assert profile.optimize is not None
    assert profile.optimize.target_triangles == 42_000
    assert profile.budget is not None
    assert profile.budget.max_triangles == 42_000
    assert profile.budget.max_vertices == 126_000
    assert profile.budget.max_texture_resolution == 512
    assert profile.budget.max_draw_calls == 250
    assert profile.budget.supported_compression == ("meshopt",)
    assert profile.budget.supported_runtime_extensions == ("KHR_mesh_quantization", "EXT_meshopt_compression")
    assert profile.budget.unity_reference_profile == "tablet-ar"
    assert profile.budget.unity_reference_triangles == (30_000, 60_000)


def test_target_device_profile_from_json_overlays_base_budget(tmp_path: Path) -> None:
    profile_file = tmp_path / "headset.json"
    profile_file.write_text(
        json.dumps(
            {
                "name": "warehouse-headset",
                "budget": {
                    "max_draw_calls": 80,
                    "max_load_time_ms": 900,
                    "supported_compression": ["quantization"],
                    "supported_runtime_extensions": ["KHR_mesh_quantization"],
                },
            }
        ),
        encoding="utf-8",
    )

    profile = profiles.from_file(profile_file, base=profiles.mixed_reality())

    assert profile.name == "warehouse-headset"
    assert profile.optimize is not None
    assert profile.optimize.target_triangles == 75_000
    assert profile.budget is not None
    assert profile.budget.max_triangles == 75_000
    assert profile.budget.max_vertices == 225_000
    assert profile.budget.max_draw_calls == 80
    assert profile.budget.max_load_time_ms == 900
    assert profile.budget.supported_compression == ("quantization",)
    assert profile.budget.supported_runtime_extensions == ("KHR_mesh_quantization",)


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"name": "bad"}, "must include a budget table"),
        ({"name": "bad", "budget": {"max_triangles": 10}, "bad": True}, "unsupported key"),
        ({"name": "bad", "budget": {"bad": True}}, "unsupported key"),
    ],
)
def test_target_device_profile_rejects_invalid_mappings(values: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        profiles.from_mapping(values)


def test_size_adaptive_tessellation_builds_part_settings_from_bounds() -> None:
    asset = fc.Asset(
        root=fc.Node(
            id="root",
            name="root",
            children=[
                fc.Node(id="small_node", name="Small", part_id="small"),
                fc.Node(id="large_node", name="Large", part_id="large"),
                fc.Node(id="bulk_node", name="Bulk", part_id="bulk"),
                fc.Node(id="empty_node", name="Empty", part_id="empty"),
            ],
        ),
        parts={
            "small": fc.Part(id="small", name="Small", mesh=_sized_mesh(1.0)),
            "large": fc.Part(id="large", name="Large", mesh=_sized_mesh(10.0)),
            "bulk": fc.Part(id="bulk", name="Bulk", mesh=_sized_mesh(10.0)),
            "empty": fc.Part(id="empty", name="Empty"),
        },
    )

    adaptive = profiles.size_adaptive_tessellation(
        asset,
        base=fc.TessellationOptions(sag=0.5, angle=30.0, part_settings={"Large": {"angle": 12.0}}),
        bands=(
            profiles.TessellationSizeBand(max_diagonal=2.0, sag=0.02, angle=8.0, max_polygon_length=0.5),
            profiles.TessellationSizeBand(max_diagonal=None, sag=0.2, sag_ratio=0.01, angle=18.0),
        ),
    )

    assert adaptive.sag == 0.5
    assert adaptive.angle == 30.0
    assert adaptive.part_settings["small"] == {
        "sag": 0.02,
        "angle": 8.0,
        "max_polygon_length": 0.5,
    }
    assert adaptive.part_settings["Large"] == {"angle": 12.0}
    assert "large" not in adaptive.part_settings
    assert adaptive.part_settings["bulk"] == {
        "sag": 0.2,
        "sag_ratio": 0.01,
        "angle": 18.0,
    }
    assert "empty" not in adaptive.part_settings


def test_size_adaptive_tessellation_requires_bands() -> None:
    asset = fc.Asset(root=fc.Node(id="root", name="root"))

    with pytest.raises(ValueError, match="size band"):
        profiles.size_adaptive_tessellation(asset, bands=())


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: fc.TessellationOptions(sag=0), "sag"),
        (lambda: fc.TessellationOptions(sag_ratio=0), "sag_ratio"),
        (lambda: fc.TessellationOptions(angle=0), "angle"),
        (lambda: fc.TessellationOptions(angle=181), "angle"),
        (lambda: fc.TessellationOptions(min_edge_length=0), "min_edge_length"),
        (lambda: fc.TessellationOptions(max_edge_length=0), "max_edge_length"),
        (lambda: fc.TessellationOptions(max_polygon_length=0), "max_polygon_length"),
        (lambda: fc.TessellationOptions(min_edge_length=2, max_edge_length=1), "min_edge_length"),
        (lambda: fc.TessellationOptions(part_settings={"part": {"bad": True}}), "unsupported part_settings"),
        (lambda: fc.PlatformBudget(target_fps=0), "target_fps"),
        (lambda: fc.PlatformBudget(max_triangles=0), "max_triangles"),
        (lambda: fc.PlatformBudget(max_vertices=0), "max_vertices"),
        (lambda: fc.PlatformBudget(max_vertices_per_mesh=0), "max_vertices_per_mesh"),
        (lambda: fc.PlatformBudget(max_texture_resolution=0), "max_texture_resolution"),
        (lambda: fc.PlatformBudget(max_texture_memory_mb=0), "max_texture_memory_mb"),
        (lambda: fc.PlatformBudget(max_load_time_ms=0), "max_load_time_ms"),
        (lambda: fc.PlatformBudget(max_draw_calls=0), "max_draw_calls"),
        (lambda: fc.PlatformBudget(unity_reference_profile=""), "unity_reference_profile"),
        (lambda: fc.PlatformBudget(unity_reference_triangles=(1,)), "unity_reference_triangles"),
        (lambda: fc.PlatformBudget(unity_reference_triangles=(0, 1)), "unity_reference_triangles"),
        (lambda: fc.PlatformBudget(unity_reference_triangles=(2, 1)), "unity_reference_triangles"),
        (lambda: fc.PlatformBudget(unity_reference_draw_calls=0), "unity_reference_draw_calls"),
        (lambda: fc.PlatformBudget(supported_compression="meshopt"), "supported_compression"),
        (lambda: fc.PlatformBudget(supported_compression=("meshopt", "")), "supported_compression"),
        (
            lambda: fc.PlatformBudget(supported_runtime_extensions="KHR_mesh_quantization"),
            "supported_runtime_extensions",
        ),
        (
            lambda: fc.PlatformBudget(supported_runtime_extensions=("KHR_mesh_quantization", "")),
            "supported_runtime_extensions",
        ),
        (lambda: fc.WorkflowRecipeChoice(stage="", setting="uv0", value="box"), "stage"),
        (lambda: fc.WorkflowRecipeChoice(stage="stage", setting="", value="box"), "setting"),
        (lambda: fc.WorkflowRecipeChoice(stage="stage", setting="uv0", value="box", status="bad"), "status"),
        (lambda: fc.WorkflowRecipe(name="", target="web", description="web recipe"), "name"),
        (lambda: fc.MergeVerticesOptions(tolerance=-1), "merge vertices tolerance"),
        (lambda: fc.MergeVerticesOptions(area_epsilon=-1), "area_epsilon"),
        (lambda: fc.DeleteDegeneratePolygonsOptions(area_epsilon=-1), "area_epsilon"),
        (lambda: fc.RepairOptions(tolerance=-1), "tolerance"),
        (lambda: fc.RepairOptions(area_epsilon=-1), "area_epsilon"),
        (lambda: fc.RepairOptions(face_orientation="bad"), "face_orientation"),
        (lambda: fc.RepairOptions(normal_orientation="bad"), "normal_orientation"),
        (lambda: fc.RepairOptions(face_orientation="viewer_standpoint"), "viewer_position"),
        (lambda: fc.RepairOptions(viewer_position=(0.0, 0.0)), "viewer_position"),
        (lambda: fc.SceneOptimizeOptions(instance_similarity_tolerance=-1), "instance_similarity_tolerance"),
        (lambda: fc.StageOptions(materials="bad"), "materials"),
        (lambda: fc.StageOptions(material_mode="bad"), "material_mode"),
        (lambda: fc.StageOptions(normal_mode="bad"), "normal_mode"),
        (lambda: fc.StageOptions(normal_weighting="bad"), "normal_weighting"),
        (lambda: fc.StageOptions(hard_edge_angle=0), "hard_edge_angle"),
        (lambda: fc.StageOptions(uv0="bad"), "uv0"),
        (lambda: fc.StageOptions(uv1="bad"), "uv1"),
        (lambda: fc.StageOptions(uv0="copy_uv0"), "uv0"),
        (lambda: fc.StageOptions(tangent_uv_channel=-1), "tangent_uv_channel"),
        (lambda: fc.StageOptions(normalize_uvs=(-1,)), "normalize_uvs"),
        (lambda: fc.UnwrapOptions(texel_density=0), "texel_density"),
        (lambda: fc.UnwrapOptions(padding=-1), "padding"),
        (lambda: fc.UnwrapOptions(max_stretch=-1), "max_stretch"),
        (lambda: fc.UnwrapOptions(method="bad"), "unwrap method"),
        (lambda: fc.UnwrapOptions(iterations=0), "unwrap iterations"),
        (lambda: fc.UnwrapOptions(tolerance=-1), "unwrap tolerance"),
        (lambda: fc.AtlasOptions(max_size=0), "atlas max_size"),
        (lambda: fc.OptimizeOptions(target_triangles=0), "target_triangles"),
        (lambda: fc.OptimizeOptions(ratio=0), "ratio"),
        (lambda: fc.OptimizeOptions(ratio=1), "ratio"),
        (lambda: fc.OptimizeOptions(hard_edge_angle=0), "hard_edge_angle"),
        (lambda: fc.OptimizeOptions(hard_edge_angle=181), "hard_edge_angle"),
        (lambda: fc.OptimizeOptions(small_part_triangle_threshold=-1), "small_part_triangle_threshold"),
        (lambda: fc.SceneOptimizeOptions(max_vertices_per_mesh=0), "max_vertices_per_mesh"),
        (lambda: fc.SceneOptimizeOptions(split_large_meshes=True, max_vertices_per_mesh=2), "max_vertices_per_mesh"),
        (lambda: fc.SceneOptimizeOptions(index_buffer="bad"), "index_buffer"),
        (lambda: fc.SceneOptimizeOptions(flatten="bad"), "flatten"),
        (lambda: fc.SceneOptimizeOptions(instance_policy="bad"), "instance_policy"),
        (lambda: fc.BakeMaterialOptions(maps_resolution=0), "maps_resolution"),
        (lambda: fc.BakeMaterialOptions(bake=("bad",)), "unsupported bake maps"),
        (lambda: fc.DecimateOptions(target_ratio=1.0), "target_ratio"),
        (lambda: fc.DecimateOptions(normal_tolerance=0.0), "normal_tolerance"),
        (lambda: fc.DecimateOptions(uv_importance="bad"), "uv_importance"),
        (lambda: fc.DecimateOptions(cleanup_attributes=("bad",)), "cleanup_attributes"),
        (lambda: fc.DecimateOptions(iterative_threshold=0), "iterative_threshold"),
        (lambda: fc.RemoveHolesOptions(through=False, blind=False, surface=False), "hole type"),
        (lambda: fc.RemoveOccludedOptions(precision=0), "precision"),
        (lambda: fc.LODLevel(screen_coverage=0.0, target_ratio=0.5), "screen_coverage"),
        (lambda: fc.LODGeneratorOptions(preset="bad"), "preset"),
        (lambda: fc.LODOptions(()), "at least one"),
        (lambda: fc.LODOptions((1.0,)), "greater than 0"),
        (lambda: fc.LODOptions((0.25, 0.5)), "sorted"),
        (lambda: fc.LODOptions((0.5,), mode="payloads"), "LOD mode"),
        (lambda: fc.LODOptions((0.5,), screen_coverage=(0.5, 0.25)), "one value per LOD"),
        (lambda: fc.LODOptions((0.5,), tiny_part_screen_size=-1.0), "tiny_part_screen_size"),
        (lambda: fc.GltfExportOptions(texture_compression="zip"), "texture_compression"),
        (lambda: fc.GltfExportOptions(texture_compression="ktx2"), "texture compression is not supported"),
        (lambda: fc.GltfExportOptions(texture_fallback_format="webp"), "texture_fallback_format"),
        (lambda: fc.GltfExportOptions(png_compression=-1), "png_compression"),
        (lambda: fc.GltfExportOptions(png_compression=10), "png_compression"),
        (lambda: fc.GltfExportOptions(jpeg_quality=-1), "jpeg_quality"),
        (lambda: fc.GltfExportOptions(jpeg_quality=101), "jpeg_quality"),
        (lambda: fc.GltfExportOptions(draco=True), "draco compression is not supported"),
        (lambda: fc.GltfExportOptions(file_size_budget_mb=0), "file_size_budget_mb"),
        (lambda: fc.UsdExportOptions(package="zip"), "package"),
        (lambda: fc.ObjExportOptions(file_size_budget_mb=0), "file_size_budget_mb"),
        (lambda: fc.StlExportOptions(file_size_budget_mb=0), "file_size_budget_mb"),
    ],
)
def test_options_validate_bad_inputs(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]
