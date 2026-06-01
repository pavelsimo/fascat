from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from fascat.asset import Asset, Node, Part
from fascat.image import ImageResource
from fascat.io.gltf import validate_gltf
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import GltfExportOptions
from fascat.visual import VisualDiffOptions, VisualPreviewOptions, write_preview

RuntimeParityTarget = Literal["browser", "unity", "unreal"]

_SUITE_SCHEMA = "fascat.runtime-parity-suite.v1"
_DEFAULT_TARGETS: tuple[RuntimeParityTarget, ...] = ("browser", "unity", "unreal")


@dataclass(frozen=True)
class RuntimeParityFixture:
    name: str
    purpose: str
    asset_path: str
    software_baseline_path: str
    checks: tuple[str, ...]
    materials: int
    textures: int
    triangles: int

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "asset_path": self.asset_path,
            "software_baseline_path": self.software_baseline_path,
            "checks": list(self.checks),
            "materials": self.materials,
            "textures": self.textures,
            "triangles": self.triangles,
        }


@dataclass(frozen=True)
class RuntimeParitySuiteReport:
    directory: str
    manifest_path: str
    targets: tuple[RuntimeParityTarget, ...]
    recommended_diff: VisualDiffOptions
    fixtures: tuple[RuntimeParityFixture, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "manifest_path": self.manifest_path,
            "targets": list(self.targets),
            "recommended_diff": self.recommended_diff.to_dict(),
            "fixtures": [fixture.to_dict() for fixture in self.fixtures],
        }


@dataclass(frozen=True)
class _FixtureSpec:
    name: str
    purpose: str
    checks: tuple[str, ...]
    build: Callable[[], Asset]


def write_runtime_parity_suite(
    directory: str | Path,
    *,
    preview_options: VisualPreviewOptions | None = None,
    diff_options: VisualDiffOptions | None = None,
) -> RuntimeParitySuiteReport:
    """Write bundled GLB fixtures and baseline PNGs for runtime preview parity checks."""

    output_dir = Path(directory)
    assets_dir = output_dir / "assets"
    baselines_dir = output_dir / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    baselines_dir.mkdir(parents=True, exist_ok=True)

    preview_opts = preview_options or VisualPreviewOptions(width=512, height=512, padding=40)
    diff_opts = diff_options or VisualDiffOptions(
        pixel_tolerance=8,
        max_mean_absolute_error=18.0,
        max_changed_pixel_ratio=0.35,
    )

    fixtures: list[RuntimeParityFixture] = []
    manifest_fixtures: list[dict[str, object]] = []
    for spec in _runtime_parity_specs():
        asset = spec.build()
        asset_path = assets_dir / f"{spec.name}.glb"
        baseline_path = baselines_dir / f"{spec.name}.png"
        asset.write_gltf(asset_path, options=GltfExportOptions())
        stats = validate_gltf(asset_path)
        preview = write_preview(asset, baseline_path, preview_opts)
        fixture = RuntimeParityFixture(
            name=spec.name,
            purpose=spec.purpose,
            asset_path=str(asset_path),
            software_baseline_path=str(baseline_path),
            checks=spec.checks,
            materials=len(asset.materials),
            textures=len(asset.images),
            triangles=stats["triangles"],
        )
        fixtures.append(fixture)
        preview_payload = preview.to_dict()
        preview_payload["path"] = _relative_posix(baseline_path, output_dir)
        manifest_fixtures.append(
            {
                "name": spec.name,
                "purpose": spec.purpose,
                "asset": _relative_posix(asset_path, output_dir),
                "software_baseline": _relative_posix(baseline_path, output_dir),
                "checks": list(spec.checks),
                "materials": len(asset.materials),
                "textures": len(asset.images),
                "triangles": stats["triangles"],
                "preview": preview_payload,
                "commands": _fixture_commands(spec.name, diff_opts),
            }
        )

    manifest_path = output_dir / "runtime-parity-suite.json"
    manifest = {
        "schema": _SUITE_SCHEMA,
        "targets": list(_DEFAULT_TARGETS),
        "recommended_diff": diff_opts.to_dict(),
        "layout": {
            "assets": "assets",
            "baselines": "baselines",
            "engine_previews": "previews/{fixture}-{engine}.png",
            "browser_previews": "previews/{fixture}-browser.png",
        },
        "fixtures": manifest_fixtures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return RuntimeParitySuiteReport(
        directory=str(output_dir),
        manifest_path=str(manifest_path),
        targets=_DEFAULT_TARGETS,
        recommended_diff=diff_opts,
        fixtures=tuple(fixtures),
    )


def _runtime_parity_specs() -> tuple[_FixtureSpec, ...]:
    return (
        _FixtureSpec(
            name="pbr-material-grid",
            purpose="exercise scalar PBR base color, metallic, roughness, and alpha handling across previews",
            checks=("base_color_factor", "metallic_factor", "roughness_factor", "alpha_blend"),
            build=_pbr_material_grid_asset,
        ),
        _FixtureSpec(
            name="texture-map-grid",
            purpose="exercise embedded base-color, metallic-roughness, normal, occlusion, and emissive texture slots",
            checks=(
                "base_color_texture",
                "metallic_roughness_texture",
                "normal_texture",
                "occlusion_texture",
                "emissive_texture",
                "uv0_sampling",
            ),
            build=_texture_map_grid_asset,
        ),
        _FixtureSpec(
            name="normal-lighting-wedges",
            purpose="exercise surface normal orientation and fixed preview-light response on angled panels",
            checks=("normal_orientation", "directional_lighting", "backface_consistency"),
            build=_normal_lighting_wedges_asset,
        ),
    )


def _pbr_material_grid_asset() -> Asset:
    material_ids = ["matte-red", "metal-blue", "clear-green", "warm-plastic"]
    mesh = _panel_grid_mesh(material_count=len(material_ids), spacing=1.35).compute_normals()
    return Asset(
        root=Node(
            id="root", name="Runtime Parity PBR Materials", children=[Node(id="grid", name="PBR Grid", part_id="grid")]
        ),
        parts={
            "grid": Part(
                id="grid",
                name="PBR Material Grid",
                mesh=mesh,
                material_ids=material_ids,
                metadata={"runtime_parity_fixture": "pbr-material-grid"},
            )
        },
        materials={
            "matte-red": Material("matte-red", "Matte Red", (0.86, 0.08, 0.04, 1.0), roughness=0.92),
            "metal-blue": Material(
                "metal-blue",
                "Polished Blue Metal",
                (0.08, 0.28, 0.95, 1.0),
                metallic=1.0,
                roughness=0.18,
            ),
            "clear-green": Material(
                "clear-green",
                "Transparent Green",
                (0.10, 0.72, 0.28, 0.55),
                roughness=0.35,
                opacity=0.70,
            ),
            "warm-plastic": Material(
                "warm-plastic",
                "Warm Plastic",
                (0.95, 0.62, 0.12, 1.0),
                roughness=0.62,
            ),
        },
        up_axis="Y",
        metadata={"runtime_parity_suite": _SUITE_SCHEMA, "runtime_parity_fixture": "pbr-material-grid"},
    )


def _texture_map_grid_asset() -> Asset:
    material_id = "textured-panel"
    image_ids = {
        "base": "checker-base",
        "metallic": "checker-metallic-roughness",
        "normal": "checker-normal",
        "occlusion": "checker-occlusion",
        "emissive": "checker-emissive",
    }
    mesh = _single_panel_mesh(width=2.2, height=1.55).compute_normals()
    return Asset(
        root=Node(
            id="root",
            name="Runtime Parity Texture Maps",
            children=[Node(id="textured-panel", name="Textured Panel", part_id="textured-panel")],
        ),
        parts={
            "textured-panel": Part(
                id="textured-panel",
                name="Texture Map Panel",
                mesh=mesh,
                material_ids=[material_id],
                metadata={"runtime_parity_fixture": "texture-map-grid"},
            )
        },
        materials={
            material_id: Material(
                material_id,
                "Texture Map Panel",
                (1.0, 1.0, 1.0, 1.0),
                metallic=0.2,
                roughness=0.55,
                metadata={
                    "baked_maps": "base_color,metallic_roughness,normal,occlusion,emissive",
                    "baked_texture_base_color_image": image_ids["base"],
                    "baked_texture_metallic_roughness_image": image_ids["metallic"],
                    "baked_texture_normal_image": image_ids["normal"],
                    "baked_texture_occlusion_image": image_ids["occlusion"],
                    "baked_texture_emissive_image": image_ids["emissive"],
                },
            )
        },
        images={
            image_ids["base"]: _image_resource(image_ids["base"], "Base Checker", _checker_png()),
            image_ids["metallic"]: _image_resource(
                image_ids["metallic"], "Metallic Roughness Checker", _metallic_roughness_png()
            ),
            image_ids["normal"]: _image_resource(image_ids["normal"], "Normal Checker", _normal_png()),
            image_ids["occlusion"]: _image_resource(image_ids["occlusion"], "Occlusion Checker", _occlusion_png()),
            image_ids["emissive"]: _image_resource(image_ids["emissive"], "Emissive Checker", _emissive_png()),
        },
        up_axis="Y",
        metadata={"runtime_parity_suite": _SUITE_SCHEMA, "runtime_parity_fixture": "texture-map-grid"},
    )


def _normal_lighting_wedges_asset() -> Asset:
    material_ids = ["neutral-matte", "cool-matte", "warm-matte"]
    mesh = _wedge_panel_mesh().compute_normals()
    return Asset(
        root=Node(
            id="root",
            name="Runtime Parity Normal Lighting",
            children=[Node(id="wedges", name="Normal Lighting Wedges", part_id="wedges")],
        ),
        parts={
            "wedges": Part(
                id="wedges",
                name="Normal Lighting Wedges",
                mesh=mesh,
                material_ids=material_ids,
                metadata={"runtime_parity_fixture": "normal-lighting-wedges"},
            )
        },
        materials={
            "neutral-matte": Material("neutral-matte", "Neutral Matte", (0.58, 0.62, 0.66, 1.0), roughness=0.85),
            "cool-matte": Material("cool-matte", "Cool Matte", (0.34, 0.58, 0.88, 1.0), roughness=0.78),
            "warm-matte": Material("warm-matte", "Warm Matte", (0.90, 0.54, 0.28, 1.0), roughness=0.78),
        },
        up_axis="Y",
        metadata={"runtime_parity_suite": _SUITE_SCHEMA, "runtime_parity_fixture": "normal-lighting-wedges"},
    )


def _panel_grid_mesh(*, material_count: int, spacing: float) -> Mesh:
    centers = [
        (-0.5 * spacing, -0.5 * spacing),
        (0.5 * spacing, -0.5 * spacing),
        (-0.5 * spacing, 0.5 * spacing),
        (0.5 * spacing, 0.5 * spacing),
    ]
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    material_indices: list[int] = []
    for material_index in range(material_count):
        cx, cz = centers[material_index]
        _append_panel(
            points,
            faces,
            uvs,
            cx=cx,
            cz=cz,
            width=1.0,
            height=1.0,
            material_indices=material_indices,
            material_index=material_index,
        )
    return Mesh(
        points=np.asarray(points, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs={0: np.asarray(uvs, dtype=np.float64)},
        material_indices=np.asarray(material_indices, dtype=np.int64),
    )


def _single_panel_mesh(*, width: float, height: float) -> Mesh:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    material_indices: list[int] = []
    _append_panel(
        points,
        faces,
        uvs,
        cx=0.0,
        cz=0.0,
        width=width,
        height=height,
        material_indices=material_indices,
        material_index=0,
    )
    return Mesh(
        points=np.asarray(points, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs={0: np.asarray(uvs, dtype=np.float64)},
        material_indices=np.asarray(material_indices, dtype=np.int64),
    )


def _wedge_panel_mesh() -> Mesh:
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = []
    material_indices: list[int] = []
    panels = (
        (-1.15, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.42, 0.0),
        (1.15, 0.0, -0.18, 0.48),
    )
    for material_index, (cx, cz, tilt_x, tilt_z) in enumerate(panels):
        _append_panel(
            points,
            faces,
            uvs,
            cx=cx,
            cz=cz,
            width=0.95,
            height=1.25,
            material_indices=material_indices,
            material_index=material_index,
            tilt_x=tilt_x,
            tilt_z=tilt_z,
        )
    return Mesh(
        points=np.asarray(points, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        uvs={0: np.asarray(uvs, dtype=np.float64)},
        material_indices=np.asarray(material_indices, dtype=np.int64),
    )


def _append_panel(
    points: list[tuple[float, float, float]],
    faces: list[tuple[int, int, int]],
    uvs: list[tuple[float, float]],
    *,
    cx: float,
    cz: float,
    width: float,
    height: float,
    material_indices: list[int],
    material_index: int,
    tilt_x: float = 0.0,
    tilt_z: float = 0.0,
) -> None:
    base = len(points)
    corners = (
        (-0.5 * width, -0.5 * height, 0.0, 0.0),
        (0.5 * width, -0.5 * height, 1.0, 0.0),
        (0.5 * width, 0.5 * height, 1.0, 1.0),
        (-0.5 * width, 0.5 * height, 0.0, 1.0),
    )
    for local_x, local_z, u, v in corners:
        y = local_x * tilt_x + local_z * tilt_z
        points.append((cx + local_x, y, cz + local_z))
        uvs.append((u, v))
    faces.extend(((base, base + 2, base + 1), (base, base + 3, base + 2)))
    material_indices.extend((material_index, material_index))


def _image_resource(image_id: str, name: str, data: bytes) -> ImageResource:
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
    return ImageResource(id=image_id, name=name, mime_type="image/png", data=data, width=width, height=height)


def _checker_png() -> bytes:
    colors = ((220, 62, 44, 255), (246, 190, 66, 255), (44, 120, 220, 255), (240, 244, 248, 255))
    return _grid_png(colors)


def _metallic_roughness_png() -> bytes:
    colors = ((0, 230, 255, 255), (0, 96, 180, 255), (0, 180, 72, 255), (0, 42, 230, 255))
    return _grid_png(colors)


def _normal_png() -> bytes:
    colors = ((128, 128, 255, 255), (156, 128, 247, 255), (128, 156, 247, 255), (104, 128, 240, 255))
    return _grid_png(colors)


def _occlusion_png() -> bytes:
    colors = ((255, 255, 255, 255), (190, 190, 190, 255), (128, 128, 128, 255), (224, 224, 224, 255))
    return _grid_png(colors)


def _emissive_png() -> bytes:
    colors = ((0, 0, 0, 255), (30, 90, 180, 255), (180, 70, 20, 255), (40, 160, 110, 255))
    return _grid_png(colors)


def _grid_png(colors: tuple[tuple[int, int, int, int], ...]) -> bytes:
    image = Image.new("RGBA", (4, 4), colors[0])
    pixels = image.load()
    assert pixels is not None
    for y in range(4):
        for x in range(4):
            pixels[x, y] = colors[(x // 2) + (y // 2) * 2]
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _fixture_commands(name: str, diff_options: VisualDiffOptions) -> dict[str, object]:
    diff_args = (
        f"--visual-diff-pixel-tolerance {diff_options.pixel_tolerance} "
        f"--visual-diff-mean-threshold {diff_options.max_mean_absolute_error:g} "
        f"--visual-diff-changed-pixel-ratio {diff_options.max_changed_pixel_ratio:g}"
    )
    return {
        "browser": (f"fascat validate assets/{name}.glb --runtime-browser-preview previews/{name}-browser.png"),
        "unity": (
            f"fascat validate assets/{name}.glb --runtime-engine unity "
            f"--runtime-engine-preview previews/{name}-unity.png "
            f"--runtime-engine-baseline baselines/{name}.png {diff_args}"
        ),
        "unreal": (
            f"fascat validate assets/{name}.glb --runtime-engine unreal "
            f"--runtime-engine-preview previews/{name}-unreal.png "
            f"--runtime-engine-baseline baselines/{name}.png {diff_args}"
        ),
    }


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
