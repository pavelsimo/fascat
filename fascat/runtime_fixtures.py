from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from fascat.asset import Asset, Node, Part
from fascat.image import ImageResource
from fascat.io.gltf import validate_gltf
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import GltfExportOptions
from fascat.runtime import (
    RuntimeBrowserRenderOptions,
    RuntimeEngineOptions,
    measure_engine_runtime,
    write_browser_render_preview,
)
from fascat.visual import VisualDiffOptions, VisualPreviewOptions, compare_images, write_preview

RuntimeParityTarget = Literal["browser", "unity", "unreal"]

_SUITE_SCHEMA = "fascat.runtime-parity-suite.v1"
_CAPTURE_SCHEMA = "fascat.runtime-parity-captures.v1"
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
class RuntimeParityCapture:
    fixture: str
    target: RuntimeParityTarget
    asset_path: str
    baseline_path: str
    preview_path: str
    status: str
    render_status: str
    passed: bool | None
    diff: dict[str, object] | None = None
    runtime_report: dict[str, object] | None = None
    golden_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "fixture": self.fixture,
            "target": self.target,
            "asset_path": self.asset_path,
            "baseline_path": self.baseline_path,
            "preview_path": self.preview_path,
            "status": self.status,
            "render_status": self.render_status,
            "passed": self.passed,
            "diff": self.diff,
            "runtime_report": self.runtime_report,
            "golden_path": self.golden_path,
            "error": self.error,
        }


@dataclass(frozen=True)
class RuntimeParityCaptureReport:
    directory: str
    manifest_path: str
    results_path: str
    targets: tuple[RuntimeParityTarget, ...]
    promoted_goldens: bool
    captures: tuple[RuntimeParityCapture, ...]

    @property
    def passed(self) -> bool:
        return all(capture.passed is True for capture in self.captures)

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "manifest_path": self.manifest_path,
            "results_path": self.results_path,
            "targets": list(self.targets),
            "promoted_goldens": self.promoted_goldens,
            "passed": self.passed,
            "captures": [capture.to_dict() for capture in self.captures],
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


def capture_runtime_parity_suite(
    directory: str | Path,
    *,
    targets: tuple[RuntimeParityTarget, ...] = _DEFAULT_TARGETS,
    browser_command: str | None = None,
    unity_command: str | None = None,
    unreal_command: str | None = None,
    unity_project: str | Path | None = None,
    unreal_project: str | Path | None = None,
    engine_timeout_seconds: float = 120.0,
    diff_options: VisualDiffOptions | None = None,
    promote_goldens: bool = False,
) -> RuntimeParityCaptureReport:
    """Capture browser/engine previews for an existing runtime parity fixture suite."""

    if not targets:
        raise ValueError("runtime parity capture requires at least one target")
    if any(target not in _DEFAULT_TARGETS for target in targets):
        raise ValueError("runtime parity capture targets must be one of: browser, unity, unreal")
    if engine_timeout_seconds <= 0.0:
        raise ValueError("runtime parity engine timeout must be greater than 0")

    output_dir = Path(directory)
    manifest_path = output_dir / "runtime-parity-suite.json"
    if not manifest_path.exists():
        write_runtime_parity_suite(output_dir)
    manifest = _load_runtime_parity_manifest(manifest_path)
    diff_opts = diff_options or _manifest_diff_options(manifest)

    previews_dir = output_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    captures: list[RuntimeParityCapture] = []
    for fixture in _manifest_fixtures(manifest):
        name = _manifest_string(fixture, "name")
        asset_path = output_dir / _manifest_string(fixture, "asset")
        baseline_path = output_dir / _manifest_string(fixture, "software_baseline")
        if not asset_path.exists():
            raise FileNotFoundError(f"runtime parity fixture asset is missing: {asset_path}")
        if not baseline_path.exists():
            raise FileNotFoundError(f"runtime parity software baseline is missing: {baseline_path}")
        for target in targets:
            captures.append(
                _capture_runtime_parity_target(
                    name,
                    target,
                    asset_path,
                    baseline_path,
                    previews_dir / f"{name}-{target}.png",
                    diff_opts,
                    browser_command=browser_command,
                    unity_command=unity_command,
                    unreal_command=unreal_command,
                    unity_project=unity_project,
                    unreal_project=unreal_project,
                    engine_timeout_seconds=engine_timeout_seconds,
                    promote_goldens=promote_goldens,
                    output_dir=output_dir,
                )
            )

    results_path = output_dir / "runtime-parity-captures.json"
    capture_report = RuntimeParityCaptureReport(
        directory=str(output_dir),
        manifest_path=str(manifest_path),
        results_path=str(results_path),
        targets=targets,
        promoted_goldens=promote_goldens,
        captures=tuple(captures),
    )
    results_payload = {"schema": _CAPTURE_SCHEMA, **capture_report.to_dict()}
    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return capture_report


def _capture_runtime_parity_target(
    fixture: str,
    target: RuntimeParityTarget,
    asset_path: Path,
    baseline_path: Path,
    preview_path: Path,
    diff_options: VisualDiffOptions,
    *,
    browser_command: str | None,
    unity_command: str | None,
    unreal_command: str | None,
    unity_project: str | Path | None,
    unreal_project: str | Path | None,
    engine_timeout_seconds: float,
    promote_goldens: bool,
    output_dir: Path,
) -> RuntimeParityCapture:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if target == "browser":
        browser_report = write_browser_render_preview(
            asset_path,
            preview_path,
            RuntimeBrowserRenderOptions(browser=browser_command),
        )
        status = browser_report.status
        render_status = browser_report.status
        error = browser_report.error
        runtime_report = browser_report.to_dict()
        rendered = browser_report.status in {"rendered", "rendered_partial"} and preview_path.exists()
    else:
        engine_target: Literal["unity", "unreal"] = "unity" if target == "unity" else "unreal"
        engine_report = measure_engine_runtime(
            asset_path,
            RuntimeEngineOptions(
                engine=engine_target,
                executable=unity_command if engine_target == "unity" else unreal_command,
                project=unity_project if engine_target == "unity" else unreal_project,
                preview_path=preview_path,
                timeout_seconds=engine_timeout_seconds,
            ),
        )
        status = engine_report.status
        render_status = engine_report.render_status
        error = engine_report.render_error or engine_report.error
        runtime_report = engine_report.to_dict()
        rendered = engine_report.render_status == "rendered" and preview_path.exists()

    diff_payload: dict[str, object] | None = None
    passed: bool | None = None
    golden_path: Path | None = None
    if rendered:
        diff_report = compare_images(baseline_path, preview_path, diff_options)
        diff_payload = diff_report.to_dict()
        passed = diff_report.passed
        if promote_goldens:
            golden_path = output_dir / "goldens" / target / f"{fixture}.png"
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(preview_path, golden_path)
    elif error is None:
        error = "runtime parity target did not write a preview"

    return RuntimeParityCapture(
        fixture=fixture,
        target=target,
        asset_path=str(asset_path),
        baseline_path=str(baseline_path),
        preview_path=str(preview_path),
        status=status,
        render_status=render_status,
        passed=passed,
        diff=diff_payload,
        runtime_report=runtime_report,
        golden_path=None if golden_path is None else str(golden_path),
        error=error,
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


def _load_runtime_parity_manifest(manifest_path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"runtime parity manifest is not valid JSON: {manifest_path}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("runtime parity manifest must be a JSON object")
    if loaded.get("schema") != _SUITE_SCHEMA:
        raise RuntimeError(f"unsupported runtime parity manifest schema: {loaded.get('schema')!r}")
    return loaded


def _manifest_diff_options(manifest: dict[str, Any]) -> VisualDiffOptions:
    value = manifest.get("recommended_diff")
    if not isinstance(value, dict):
        return VisualDiffOptions(pixel_tolerance=8, max_mean_absolute_error=18.0, max_changed_pixel_ratio=0.35)
    return VisualDiffOptions(
        pixel_tolerance=_manifest_int(value, "pixel_tolerance", 8),
        max_mean_absolute_error=_manifest_float(value, "max_mean_absolute_error", 18.0),
        max_changed_pixel_ratio=_manifest_float(value, "max_changed_pixel_ratio", 0.35),
    )


def _manifest_fixtures(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    fixtures = manifest.get("fixtures")
    if not isinstance(fixtures, list):
        raise RuntimeError("runtime parity manifest must contain a fixtures array")
    result: list[dict[str, Any]] = []
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise RuntimeError("runtime parity fixture entries must be objects")
        result.append(fixture)
    return result


def _manifest_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"runtime parity manifest field {key!r} must be a non-empty string")
    return value


def _manifest_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    return int(value) if isinstance(value, int | float | str) and not isinstance(value, bool) else default


def _manifest_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    return float(value) if isinstance(value, int | float | str) and not isinstance(value, bool) else default


def _relative_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()
