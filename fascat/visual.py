from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh

FloatArray = NDArray[np.float64]
Rgba = tuple[int, int, int, int]


@dataclass(frozen=True)
class VisualPreviewOptions:
    width: int = 512
    height: int = 512
    padding: int = 32
    background_color: Rgba = (248, 249, 250, 255)
    edge_color: Rgba = (38, 43, 51, 180)
    draw_edges: bool = True
    supersample: int = 2

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("visual preview dimensions must be greater than 0")
        if self.padding < 0:
            raise ValueError("visual preview padding must be greater than or equal to 0")
        if self.width <= self.padding * 2 or self.height <= self.padding * 2:
            raise ValueError("visual preview padding must leave drawable space")
        if self.supersample <= 0:
            raise ValueError("visual preview supersample must be greater than 0")
        _validate_color(self.background_color, "background_color")
        _validate_color(self.edge_color, "edge_color")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VisualPreviewReport:
    path: str
    width: int
    height: int
    lod_level: int
    occurrences: int
    meshes: int
    triangles: int
    fallback_source_occurrences: int = 0
    omitted_occurrences: int = 0
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "lod_level": self.lod_level,
            "occurrences": self.occurrences,
            "meshes": self.meshes,
            "triangles": self.triangles,
            "fallback_source_occurrences": self.fallback_source_occurrences,
            "omitted_occurrences": self.omitted_occurrences,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class VisualComparisonReport:
    directory: str
    before: VisualPreviewReport
    after: VisualPreviewReport
    contact_sheet: str

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "before": self.before.to_dict(),
            "after": self.after.to_dict(),
            "contact_sheet": self.contact_sheet,
        }


@dataclass(frozen=True)
class VisualDiffOptions:
    pixel_tolerance: int = 0
    max_mean_absolute_error: float = 0.0
    max_changed_pixel_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.pixel_tolerance < 0 or self.pixel_tolerance > 255:
            raise ValueError("visual diff pixel_tolerance must be between 0 and 255")
        if self.max_mean_absolute_error < 0.0 or self.max_mean_absolute_error > 255.0:
            raise ValueError("visual diff max_mean_absolute_error must be between 0 and 255")
        if self.max_changed_pixel_ratio < 0.0 or self.max_changed_pixel_ratio > 1.0:
            raise ValueError("visual diff max_changed_pixel_ratio must be between 0 and 1")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VisualDiffReport:
    baseline_path: str
    candidate_path: str
    passed: bool
    baseline_width: int
    baseline_height: int
    candidate_width: int
    candidate_height: int
    total_pixels: int
    changed_pixels: int
    changed_pixel_ratio: float
    mean_absolute_error: float
    max_absolute_error: int
    options: VisualDiffOptions
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_path": self.baseline_path,
            "candidate_path": self.candidate_path,
            "passed": self.passed,
            "baseline_width": self.baseline_width,
            "baseline_height": self.baseline_height,
            "candidate_width": self.candidate_width,
            "candidate_height": self.candidate_height,
            "total_pixels": self.total_pixels,
            "changed_pixels": self.changed_pixels,
            "changed_pixel_ratio": self.changed_pixel_ratio,
            "mean_absolute_error": self.mean_absolute_error,
            "max_absolute_error": self.max_absolute_error,
            "options": self.options.to_dict(),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class LodSwitchPreviewReport:
    directory: str
    contact_sheet: str
    previews: list[VisualPreviewReport] = field(default_factory=list)
    monotonic_triangles: bool = True
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "previews", list(self.previews))
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "directory": self.directory,
            "contact_sheet": self.contact_sheet,
            "previews": [preview.to_dict() for preview in self.previews],
            "levels": len(self.previews),
            "monotonic_triangles": self.monotonic_triangles,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _RenderOccurrence:
    part: Part
    mesh: Mesh
    points: FloatArray
    fallback_source: bool = False


@dataclass(frozen=True)
class _CameraBasis:
    right: FloatArray
    up: FloatArray
    view: FloatArray


@dataclass(frozen=True)
class _ProjectedTriangle:
    depth: float
    polygon: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    color: Rgba


def write_preview(
    asset: Asset,
    path: str | Path,
    options: VisualPreviewOptions | None = None,
    *,
    lod_level: int = 0,
) -> VisualPreviewReport:
    opts = options or VisualPreviewOptions()
    if lod_level < 0:
        raise ValueError("lod_level must be greater than or equal to 0")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    occurrences, omitted_occurrences = _render_occurrences(asset, lod_level=lod_level)
    warnings: list[str] = []
    if not occurrences:
        warnings.append("preview contains no renderable mesh triangles")
        _write_empty_preview(output_path, opts)
        return VisualPreviewReport(
            path=str(output_path),
            width=opts.width,
            height=opts.height,
            lod_level=lod_level,
            occurrences=0,
            meshes=0,
            triangles=0,
            omitted_occurrences=omitted_occurrences,
            warnings=tuple(warnings),
        )

    image = _render_asset(asset, occurrences, opts)
    image.save(output_path)
    fallback_source_occurrences = sum(1 for occurrence in occurrences if occurrence.fallback_source)
    if fallback_source_occurrences:
        warnings.append(f"LOD{lod_level} preview used source meshes for {fallback_source_occurrences} occurrence(s)")
    if omitted_occurrences:
        warnings.append(f"LOD{lod_level} preview omitted {omitted_occurrences} occurrence(s)")
    return VisualPreviewReport(
        path=str(output_path),
        width=opts.width,
        height=opts.height,
        lod_level=lod_level,
        occurrences=len(occurrences),
        meshes=len({occurrence.part.id for occurrence in occurrences}),
        triangles=sum(occurrence.mesh.triangle_count for occurrence in occurrences),
        fallback_source_occurrences=fallback_source_occurrences,
        omitted_occurrences=omitted_occurrences,
        warnings=tuple(warnings),
    )


def write_output_preview(
    output_path: str | Path,
    preview_path: str | Path,
    options: VisualPreviewOptions | None = None,
    *,
    lod_level: int = 0,
) -> VisualPreviewReport:
    asset = _asset_from_output_path(output_path)
    return write_preview(asset, preview_path, options, lod_level=lod_level)


def write_before_after_previews(
    before: Asset,
    after: Asset,
    directory: str | Path,
    options: VisualPreviewOptions | None = None,
) -> VisualComparisonReport:
    opts = options or VisualPreviewOptions()
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    before_report = write_preview(before, output_dir / "before.png", opts)
    after_report = write_preview(after, output_dir / "after.png", opts)
    contact_sheet = output_dir / "before-after.png"
    _write_contact_sheet(
        [(Path(before_report.path), "before"), (Path(after_report.path), "after")],
        contact_sheet,
        opts,
    )
    return VisualComparisonReport(
        directory=str(output_dir),
        before=before_report,
        after=after_report,
        contact_sheet=str(contact_sheet),
    )


def compare_images(
    baseline_path: str | Path,
    candidate_path: str | Path,
    options: VisualDiffOptions | None = None,
) -> VisualDiffReport:
    opts = options or VisualDiffOptions()
    baseline = Path(baseline_path)
    candidate = Path(candidate_path)
    with Image.open(baseline) as baseline_image:
        baseline_rgba = baseline_image.convert("RGBA")
        baseline_values = np.asarray(baseline_rgba, dtype=np.int16)
        baseline_width, baseline_height = baseline_rgba.size
    with Image.open(candidate) as candidate_image:
        candidate_rgba = candidate_image.convert("RGBA")
        candidate_values = np.asarray(candidate_rgba, dtype=np.int16)
        candidate_width, candidate_height = candidate_rgba.size

    warnings: list[str] = []
    if baseline_values.shape != candidate_values.shape:
        warnings.append("visual diff images have different dimensions")
        return VisualDiffReport(
            baseline_path=str(baseline),
            candidate_path=str(candidate),
            passed=False,
            baseline_width=baseline_width,
            baseline_height=baseline_height,
            candidate_width=candidate_width,
            candidate_height=candidate_height,
            total_pixels=0,
            changed_pixels=0,
            changed_pixel_ratio=1.0,
            mean_absolute_error=255.0,
            max_absolute_error=255,
            options=opts,
            warnings=tuple(warnings),
        )

    diff = np.abs(baseline_values - candidate_values)
    changed = np.any(diff > opts.pixel_tolerance, axis=2)
    total_pixels = int(changed.size)
    changed_pixels = int(np.count_nonzero(changed))
    mean_absolute_error = float(np.mean(diff)) if diff.size else 0.0
    max_absolute_error = int(np.max(diff)) if diff.size else 0
    changed_pixel_ratio = changed_pixels / total_pixels if total_pixels else 0.0
    passed = mean_absolute_error <= opts.max_mean_absolute_error and changed_pixel_ratio <= opts.max_changed_pixel_ratio
    return VisualDiffReport(
        baseline_path=str(baseline),
        candidate_path=str(candidate),
        passed=passed,
        baseline_width=baseline_width,
        baseline_height=baseline_height,
        candidate_width=candidate_width,
        candidate_height=candidate_height,
        total_pixels=total_pixels,
        changed_pixels=changed_pixels,
        changed_pixel_ratio=changed_pixel_ratio,
        mean_absolute_error=mean_absolute_error,
        max_absolute_error=max_absolute_error,
        options=opts,
        warnings=tuple(warnings),
    )


def write_lod_switch_previews(
    asset: Asset,
    directory: str | Path,
    options: VisualPreviewOptions | None = None,
) -> LodSwitchPreviewReport:
    opts = options or VisualPreviewOptions()
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_lods = max((len(part.lod_meshes) for part in asset.parts.values()), default=0)
    previews = [
        write_preview(asset, output_dir / f"lod{level}.png", opts, lod_level=level) for level in range(max_lods + 1)
    ]
    contact_sheet = output_dir / "lod-switching.png"
    _write_contact_sheet(
        [(Path(preview.path), f"LOD{preview.lod_level} - {preview.triangles} tris") for preview in previews],
        contact_sheet,
        opts,
    )
    triangle_counts = [preview.triangles for preview in previews]
    monotonic = all(after <= before for before, after in zip(triangle_counts, triangle_counts[1:], strict=False))
    warnings: list[str] = []
    if max_lods == 0:
        warnings.append("asset has no LOD meshes to preview")
    if not monotonic:
        warnings.append("LOD preview triangle counts are not monotonic")
    for preview in previews:
        warnings.extend(preview.warnings)
    return LodSwitchPreviewReport(
        directory=str(output_dir),
        contact_sheet=str(contact_sheet),
        previews=previews,
        monotonic_triangles=monotonic,
        warnings=tuple(_dedupe(warnings)),
    )


def write_output_lod_switch_previews(
    output_path: str | Path,
    directory: str | Path,
    options: VisualPreviewOptions | None = None,
) -> LodSwitchPreviewReport:
    asset = _asset_from_output_path(output_path)
    return write_lod_switch_previews(asset, directory, options)


def _render_asset(asset: Asset, occurrences: list[_RenderOccurrence], options: VisualPreviewOptions) -> Image.Image:
    supersample = options.supersample
    width = options.width * supersample
    height = options.height * supersample
    padding = options.padding * supersample
    image = Image.new("RGBA", (width, height), options.background_color)
    all_points = np.vstack([occurrence.points for occurrence in occurrences if len(occurrence.points)])
    basis = _camera_basis(asset.up_axis)
    projected = _project(all_points, basis)
    minimum = projected.min(axis=0)
    maximum = projected.max(axis=0)
    extent = maximum - minimum
    scale_x = (width - padding * 2) / extent[0] if extent[0] > 0.0 else width - padding * 2
    scale_y = (height - padding * 2) / extent[1] if extent[1] > 0.0 else height - padding * 2
    scale = max(1.0, min(scale_x, scale_y))
    center = (minimum + maximum) * 0.5
    triangles = _projected_triangles(asset, occurrences, basis, center, scale, width, height)
    draw = ImageDraw.Draw(image, "RGBA")
    edge_color = _scale_color(options.edge_color, 1.0)
    for triangle in sorted(triangles, key=lambda item: item.depth):
        draw.polygon(triangle.polygon, fill=triangle.color)
        if options.draw_edges:
            draw.line((*triangle.polygon, triangle.polygon[0]), fill=edge_color, width=max(1, supersample))
    if supersample == 1:
        return image
    return image.resize((options.width, options.height), Image.Resampling.LANCZOS)


def _projected_triangles(
    asset: Asset,
    occurrences: list[_RenderOccurrence],
    basis: _CameraBasis,
    center: FloatArray,
    scale: float,
    width: int,
    height: int,
) -> list[_ProjectedTriangle]:
    result: list[_ProjectedTriangle] = []
    for occurrence in occurrences:
        projected_points = _project(occurrence.points, basis)
        depths = occurrence.points @ basis.view
        for face_index, face in enumerate(occurrence.mesh.faces):
            face_points = occurrence.points[face]
            normal = np.cross(face_points[1] - face_points[0], face_points[2] - face_points[0])
            normal_length = float(np.linalg.norm(normal))
            if normal_length <= 0.0:
                continue
            brightness = 0.52 + 0.48 * abs(float(np.dot(normal / normal_length, basis.view)))
            color = _face_color(asset, occurrence.part, occurrence.mesh, face_index, brightness)
            polygon = tuple(
                (
                    ((projected_points[index, 0] - center[0]) * scale) + (width * 0.5),
                    (height * 0.5) - ((projected_points[index, 1] - center[1]) * scale),
                )
                for index in face
            )
            result.append(
                _ProjectedTriangle(
                    depth=float(np.mean(depths[face])),
                    polygon=cast(tuple[tuple[float, float], tuple[float, float], tuple[float, float]], polygon),
                    color=color,
                )
            )
    return result


def _render_occurrences(asset: Asset, *, lod_level: int) -> tuple[list[_RenderOccurrence], int]:
    occurrences: list[_RenderOccurrence] = []
    omitted_occurrences = 0

    def walk(node: Node, world: FloatArray) -> None:
        nonlocal omitted_occurrences
        current = world @ node.transform
        if node.part_id is not None:
            part = asset.parts.get(node.part_id)
            if part is not None:
                mesh, fallback_source, omitted = _mesh_for_lod(part, lod_level)
                if omitted:
                    omitted_occurrences += 1
                elif mesh is not None and mesh.triangle_count > 0:
                    occurrences.append(
                        _RenderOccurrence(
                            part=part,
                            mesh=mesh,
                            points=_transform_points(mesh.points, current),
                            fallback_source=fallback_source,
                        )
                    )
        for child in node.children:
            walk(child, current)

    walk(asset.root, np.eye(4, dtype=np.float64))
    return occurrences, omitted_occurrences


def _mesh_for_lod(part: Part, lod_level: int) -> tuple[Mesh | None, bool, bool]:
    if lod_level == 0:
        return part.mesh, False, part.mesh is None
    index = lod_level - 1
    if index < len(part.lod_meshes):
        lod = part.lod_meshes[index]
        omitted = lod.vertex_count == 0 or lod.triangle_count == 0 or lod.metadata.get("lod_omitted") is not None
        return lod if not omitted else None, False, omitted
    return part.mesh, True, part.mesh is None


def _transform_points(points: FloatArray, transform: FloatArray) -> FloatArray:
    if not len(points):
        return points
    homogeneous = np.column_stack([points, np.ones(points.shape[0], dtype=np.float64)])
    return cast(FloatArray, np.asarray((transform @ homogeneous.T).T[:, :3], dtype=np.float64))


def _camera_basis(up_axis: str) -> _CameraBasis:
    world_up = np.asarray([0.0, 1.0, 0.0] if up_axis == "Y" else [0.0, 0.0, 1.0], dtype=np.float64)
    view = _normalize(np.asarray([1.0, -1.2, 0.85], dtype=np.float64))
    right = np.cross(world_up, view)
    if float(np.linalg.norm(right)) < 1e-9:
        right = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    right = _normalize(right)
    up = _normalize(np.cross(view, right))
    return _CameraBasis(right=right, up=up, view=view)


def _project(points: FloatArray, basis: _CameraBasis) -> FloatArray:
    values = np.column_stack([points @ basis.right, points @ basis.up])
    return cast(FloatArray, values.astype(np.float64))


def _face_color(asset: Asset, part: Part, mesh: Mesh, face_index: int, brightness: float) -> Rgba:
    material = _face_material(asset, part, mesh, face_index)
    if material is not None:
        base = material.base_color
        opacity = material.opacity * base[3]
        color = (
            int(round(base[0] * 255.0)),
            int(round(base[1] * 255.0)),
            int(round(base[2] * 255.0)),
            int(round(opacity * 255.0)),
        )
    else:
        color = _fallback_color(part.id)
    return _scale_color(color, brightness)


def _face_material(asset: Asset, part: Part, mesh: Mesh, face_index: int) -> Material | None:
    material_id: str | None = None
    if mesh.material_indices is not None and face_index < len(mesh.material_indices):
        material_slot = int(mesh.material_indices[face_index])
        if 0 <= material_slot < len(part.material_ids):
            material_id = part.material_ids[material_slot]
    elif part.material_ids:
        material_id = part.material_ids[0]
    return asset.materials.get(material_id) if material_id is not None else None


def _fallback_color(key: str) -> Rgba:
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    return (96 + digest[0] % 112, 96 + digest[1] % 112, 96 + digest[2] % 112, 255)


def _scale_color(color: Rgba, brightness: float) -> Rgba:
    return (
        _byte(color[0] * brightness),
        _byte(color[1] * brightness),
        _byte(color[2] * brightness),
        color[3],
    )


def _byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _normalize(values: FloatArray) -> FloatArray:
    length = float(np.linalg.norm(values))
    if length == 0.0:
        return values
    return values / length


def _write_empty_preview(path: Path, options: VisualPreviewOptions) -> None:
    Image.new("RGBA", (options.width, options.height), options.background_color).save(path)


def _write_contact_sheet(items: list[tuple[Path, str]], path: Path, options: VisualPreviewOptions) -> None:
    label_height = 28
    width = options.width * len(items)
    height = options.height + label_height
    sheet = Image.new("RGBA", (width, height), options.background_color)
    draw = ImageDraw.Draw(sheet, "RGBA")
    for index, (item_path, label) in enumerate(items):
        with Image.open(item_path) as source:
            image = source.convert("RGBA")
        sheet.paste(image, (index * options.width, label_height))
        draw.text((index * options.width + 8, 8), label, fill=(24, 27, 32, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def _asset_from_output_path(output_path: str | Path) -> Asset:
    path = Path(output_path)
    if str(output_path) == "-":
        raise ValueError("visual previews require a file path, not stdin")
    from fascat.analysis import _asset_from_output

    return _asset_from_output(path)


def _validate_color(color: Rgba, label: str) -> None:
    if len(color) != 4:
        raise ValueError(f"{label} must contain RGBA byte values")
    if any(value < 0 or value > 255 for value in color):
        raise ValueError(f"{label} values must be between 0 and 255")


def _dedupe(values: list[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)
