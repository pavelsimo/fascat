from __future__ import annotations

import base64
import binascii
import io as pyio
import math
import struct
import zlib
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from fascat.asset import Asset, Part
from fascat.image import ImageResource
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.ops._ids import unique_id
from fascat.ops._mesh_utils import selected_mesh_part_ids
from fascat.ops._visibility import face_ambient_occlusion
from fascat.options import BakeMaterialOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def bake_materials_asset(
    asset: Asset,
    options: BakeMaterialOptions,
    *,
    selected_part_ids: set[str] | None = None,
) -> Asset:
    result = asset.copy(keep_source=True)
    part_ids = sorted(selected_mesh_part_ids(result, selected_part_ids))
    if not part_ids:
        result.report.add_warning("bake_materials matched no mesh-bearing parts")
        return result

    source_material_ids = sorted(
        {
            material_id
            for part_id in part_ids
            for material_id in result.parts[part_id].material_ids
            if material_id in result.materials
        }
    )
    _prepare_bake_atlas(result, part_ids, options)
    image_ids = _bake_texture_images(result, part_ids, options)
    baked_id = unique_id(result.materials, "baked_material")
    baked = _baked_material(result, baked_id, source_material_ids, options, image_ids)
    result.materials[baked.id] = baked

    for part_id in part_ids:
        part = result.parts[part_id]
        if part.mesh is None:
            continue
        mesh = part.mesh
        mesh.metadata = {
            **mesh.metadata,
            "baked_material": baked.id,
            "baked_maps": ",".join(options.bake),
            "baked_maps_resolution": str(options.maps_resolution),
            "baked_lightmap_resolution": str(options.lightmap_resolution),
            "baked_uv_channel": str(options.uv_channel),
            "baked_padding": str(options.padding),
        }
        if options.merge_output:
            part.material_ids = [baked.id]
            mesh.material_indices = None
        part.mesh = mesh
        part.metadata = {
            **part.metadata,
            "baked_material": baked.id,
            "source_material_ids": ",".join(source_material_ids),
        }
        part.fingerprint = mesh.fingerprint()

    if options.merge_output:
        _drop_unreferenced_materials(result)
    result.metadata["baked_image_count"] = str(len(image_ids))
    result.metadata["baked_material_count"] = "1"
    result.metadata["baked_source_material_count"] = str(len(source_material_ids))
    return result


def _prepare_bake_atlas(asset: Asset, part_ids: list[str], options: BakeMaterialOptions) -> None:
    rects = _atlas_rects(part_ids, options)
    for part_id in part_ids:
        part = asset.parts[part_id]
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.force_uv_generation or options.uv_channel not in mesh.uvs:
            try:
                mesh = mesh.unwrap_uv(
                    options.uv_channel,
                    padding=options.padding,
                    resolution=options.lightmap_resolution,
                )
            except RuntimeError:
                asset.report.add_warning(
                    f"part {part_id} could not use xatlas for bake UVs; falling back to AABB projection"
                )
                mesh = mesh.box_uv(options.uv_channel)
        else:
            mesh = mesh.copy()
        mesh = _pack_mesh_uvs_into_rect(mesh, options.uv_channel, rects[part_id], options)
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()


def _atlas_rects(part_ids: list[str], options: BakeMaterialOptions) -> dict[str, tuple[float, float, float, float]]:
    count = max(1, len(part_ids))
    columns = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / columns))
    cell_width = 1.0 / columns
    cell_height = 1.0 / rows
    margin = min(0.45 * min(cell_width, cell_height), options.padding / max(1.0, float(options.lightmap_resolution)))
    rects: dict[str, tuple[float, float, float, float]] = {}
    for index, part_id in enumerate(part_ids):
        column = index % columns
        row = index // columns
        u0 = column * cell_width + margin
        v0 = row * cell_height + margin
        u1 = (column + 1) * cell_width - margin
        v1 = (row + 1) * cell_height - margin
        if u1 <= u0:
            center = (column + 0.5) * cell_width
            u0 = u1 = center
        if v1 <= v0:
            center = (row + 0.5) * cell_height
            v0 = v1 = center
        rects[part_id] = (u0, v0, u1, v1)
    return rects


def _pack_mesh_uvs_into_rect(
    mesh: Mesh,
    channel: int,
    rect: tuple[float, float, float, float],
    options: BakeMaterialOptions,
) -> Mesh:
    if channel not in mesh.uvs:
        return mesh
    result = mesh.copy()
    uv = result.uvs[channel]
    if uv.size:
        mins = uv.min(axis=0)
        maxs = uv.max(axis=0)
        span = maxs - mins
        span[span == 0.0] = 1.0
        normalized = (uv - mins) / span
        u0, v0, u1, v1 = rect
        result.uvs[channel] = np.column_stack(
            (
                u0 + normalized[:, 0] * (u1 - u0),
                v0 + normalized[:, 1] * (v1 - v0),
            )
        ).astype(np.float64)
    prefix = f"uv{channel}"
    result.metadata[f"{prefix}_atlas_pack_status"] = "packed"
    result.metadata[f"{prefix}_atlas_rect"] = ",".join(f"{value:.9g}" for value in rect)
    result.metadata[f"{prefix}_atlas_padding_pixels"] = str(options.padding)
    result.metadata[f"{prefix}_atlas_resolution"] = str(options.maps_resolution)
    result.metadata[f"{prefix}_lightmap_resolution"] = str(options.lightmap_resolution)
    result.metadata[f"{prefix}_padding_status"] = "applied" if options.padding else "not_requested"
    result.tangents = None
    return result


def _bake_texture_images(asset: Asset, part_ids: list[str], options: BakeMaterialOptions) -> dict[str, str]:
    maps = {str(item) for item in options.bake}
    image_ids: dict[str, str] = {}
    requests: list[tuple[str, str]] = []
    if {"base_color", "opacity"} & maps:
        requests.append(("base_color", "base_color"))
    if {"metallic", "roughness"} & maps:
        requests.append(("metallic_roughness", "metallic_roughness"))
    if "normal" in maps:
        requests.append(("normal", "normal"))
    if "ao" in maps:
        requests.append(("occlusion", "ao"))
    if "emissive" in maps:
        requests.append(("emissive", "emissive"))

    emissive_provenance = _emissive_provenance_metadata(asset, part_ids) if "emissive" in maps else {}
    for texture_name, kind in requests:
        pixels = _raster_bake_map(asset, part_ids, options, kind)
        data = _png_bytes(pixels)
        image_id = unique_id(asset.images, f"baked_{texture_name}")
        metadata: dict[str, object] = {
            "baked": "true",
            "baked_map": texture_name,
            "baked_uv_channel": str(options.uv_channel),
            "baked_padding": str(options.padding),
        }
        if texture_name == "emissive":
            metadata.update(emissive_provenance)
        asset.images[image_id] = ImageResource(
            id=image_id,
            name=f"Baked {texture_name.replace('_', ' ').title()}",
            mime_type="image/png",
            data=data,
            width=options.maps_resolution,
            height=options.maps_resolution,
            metadata=metadata,
        )
        image_ids[texture_name] = image_id
    return image_ids


def _raster_bake_map(asset: Asset, part_ids: list[str], options: BakeMaterialOptions, kind: str) -> NDArray[np.uint8]:
    size = int(options.maps_resolution)
    pixels = np.empty((size, size, 4), dtype=np.uint8)
    pixels[:, :] = _default_bake_pixel(kind)
    filled = np.zeros((size, size), dtype=bool)
    for part_id in part_ids:
        part = asset.parts[part_id]
        mesh = part.mesh
        if mesh is None or options.uv_channel not in mesh.uvs:
            continue
        face_values = _face_bake_values(asset, part, mesh, kind, options.ambient_occlusion_strategy)
        uv = np.clip(mesh.uvs[options.uv_channel], 0.0, 1.0)
        for face_index, face in enumerate(mesh.faces.astype(int)):
            _rasterize_triangle(pixels, filled, uv[face], face_values[face_index])
    if options.padding:
        _dilate_padding(pixels, filled, options.padding)
    return pixels


def _default_bake_pixel(kind: str) -> NDArray[np.uint8]:
    defaults = {
        "base_color": (255, 255, 255, 0),
        "metallic_roughness": (255, 128, 0, 255),
        "normal": (128, 128, 255, 255),
        "ao": (255, 255, 255, 255),
        "emissive": (0, 0, 0, 255),
    }
    return np.asarray(defaults[kind], dtype=np.uint8)


def _face_bake_values(
    asset: Asset,
    part: Part,
    mesh: Mesh,
    kind: str,
    ambient_occlusion_strategy: str,
) -> NDArray[np.uint8]:
    if kind == "normal":
        normals = _bake_face_normals(mesh)
        encoded = np.clip((normals * 0.5 + 0.5) * 255.0, 0, 255).round().astype(np.uint8)
        alpha = np.full((mesh.triangle_count, 1), 255, dtype=np.uint8)
        return cast(NDArray[np.uint8], np.hstack((encoded, alpha)))
    if kind == "ao":
        ao = face_ambient_occlusion(mesh, ambient_occlusion_strategy)
        values = np.clip(ao[:, None] * 255.0, 0, 255).round().astype(np.uint8)
        alpha = np.full((mesh.triangle_count, 1), 255, dtype=np.uint8)
        return cast(NDArray[np.uint8], np.hstack((np.repeat(values, 3, axis=1), alpha)))

    lut = _material_bake_lut(asset, part, kind)
    return lut[_face_material_slots(part, mesh)]


def _material_bake_lut(asset: Asset, part: Part, kind: str) -> NDArray[np.uint8]:
    fallback = _material_bake_row(None, kind)
    lut = np.empty((len(part.material_ids) + 1, 4), dtype=np.uint8)
    for slot, material_id in enumerate(part.material_ids):
        material = asset.materials.get(material_id)
        lut[slot] = fallback if material is None else _material_bake_row(material, kind)
    lut[-1] = fallback
    return lut


def _material_bake_row(material: Material | None, kind: str) -> NDArray[np.uint8]:
    if kind == "base_color":
        color = material.base_color if material is not None else (1.0, 1.0, 1.0, 1.0)
        opacity = material.opacity if material is not None else color[3]
        values = (color[0], color[1], color[2], min(color[3], opacity))
    elif kind == "metallic_roughness":
        metallic = 0.0 if material is None else material.metallic
        roughness = 0.5 if material is None else material.roughness
        values = (1.0, roughness, metallic, 1.0)
    else:
        values, _source = _emissive_color_with_source(material)
    return np.asarray([_color_byte(value) for value in values], dtype=np.uint8)


def _face_material_slots(part: Part, mesh: Mesh) -> IntArray:
    fallback_slot = len(part.material_ids)
    if not part.material_ids or mesh.triangle_count == 0:
        return np.full(mesh.triangle_count, fallback_slot, dtype=np.int64)

    slots = np.zeros(mesh.triangle_count, dtype=np.int64)
    if mesh.material_indices is None:
        return slots

    available = min(mesh.triangle_count, mesh.material_indices.shape[0])
    if available <= 0:
        return slots
    candidates = mesh.material_indices[:available].astype(np.int64, copy=False)
    valid = (candidates >= 0) & (candidates < len(part.material_ids))
    invalid_positions = np.flatnonzero(~valid)
    slots[invalid_positions] = fallback_slot
    valid_positions = np.flatnonzero(valid)
    slots[valid_positions] = candidates[valid_positions]
    return slots


def _face_material(asset: Asset, part: Part, mesh: Mesh, face_index: int) -> Material | None:
    if not part.material_ids:
        return None
    material_index = 0
    if mesh.material_indices is not None and face_index < mesh.material_indices.shape[0]:
        material_index = int(mesh.material_indices[face_index])
    if material_index < 0 or material_index >= len(part.material_ids):
        return None
    return asset.materials.get(part.material_ids[material_index])


def _emissive_color(material: Material | None) -> tuple[float, float, float, float]:
    color, _source = _emissive_color_with_source(material)
    return color


def _emissive_color_with_source(material: Material | None) -> tuple[tuple[float, float, float, float], str]:
    if material is None:
        return (0.0, 0.0, 0.0, 1.0), "fallback"
    value = material.metadata.get("emissive_color")
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",")]
        if len(pieces) >= 3:
            try:
                return (float(pieces[0]), float(pieces[1]), float(pieces[2]), 1.0), "material"
            except ValueError:
                pass
    return (0.0, 0.0, 0.0, 1.0), "fallback"


def _emissive_provenance_metadata(asset: Asset, part_ids: list[str]) -> dict[str, str]:
    material_faces = 0
    fallback_faces = 0
    for part_id in part_ids:
        part = asset.parts[part_id]
        mesh = part.mesh
        if mesh is None:
            continue
        slots = _face_material_slots(part, mesh)
        unique_slots, counts = np.unique(slots, return_counts=True)
        material_sources = np.asarray(
            [_emissive_slot_is_material(asset, part, int(slot)) for slot in unique_slots],
            dtype=bool,
        )
        material_faces += int(counts[material_sources].sum())
        fallback_faces += int(counts[~material_sources].sum())
    if material_faces and fallback_faces:
        source = "mixed"
    elif material_faces:
        source = "material"
    elif fallback_faces:
        source = "fallback"
    else:
        source = "none"
    return {
        "baked_emissive_source": source,
        "baked_emissive_material_faces": str(material_faces),
        "baked_emissive_fallback_faces": str(fallback_faces),
    }


def _emissive_slot_is_material(asset: Asset, part: Part, slot: int) -> bool:
    if slot < 0 or slot >= len(part.material_ids):
        return False
    material = asset.materials.get(part.material_ids[slot])
    if material is None:
        return False
    _color, source = _emissive_color_with_source(material)
    return source == "material"


def _bake_face_normals(mesh: Mesh) -> FloatArray:
    if mesh.triangle_count == 0:
        return np.empty((0, 3), dtype=np.float64)
    triangles = mesh.points[mesh.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0.0
    normals[valid] = normals[valid] / lengths[valid, None]
    normals[~valid] = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return normals


def _rasterize_triangle(
    pixels: NDArray[np.uint8],
    filled: NDArray[np.bool_],
    triangle_uv: FloatArray,
    color: NDArray[np.uint8],
) -> None:
    size = pixels.shape[0]
    x = triangle_uv[:, 0] * (size - 1)
    y = (1.0 - triangle_uv[:, 1]) * (size - 1)
    min_x = max(0, int(math.floor(float(np.min(x)))))
    max_x = min(size - 1, int(math.ceil(float(np.max(x)))))
    min_y = max(0, int(math.floor(float(np.min(y)))))
    max_y = min(size - 1, int(math.ceil(float(np.max(y)))))
    if max_x < min_x or max_y < min_y:
        return
    p0 = np.asarray([x[0], y[0]], dtype=np.float64)
    p1 = np.asarray([x[1], y[1]], dtype=np.float64)
    p2 = np.asarray([x[2], y[2]], dtype=np.float64)
    denom = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(float(denom)) <= 1e-12:
        return
    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    w0 = ((p1[1] - p2[1]) * (grid_x - p2[0]) + (p2[0] - p1[0]) * (grid_y - p2[1])) / denom
    w1 = ((p2[1] - p0[1]) * (grid_x - p2[0]) + (p0[0] - p2[0]) * (grid_y - p2[1])) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    if not np.any(mask):
        return
    region = pixels[min_y : max_y + 1, min_x : max_x + 1]
    region[mask] = color
    filled_region = filled[min_y : max_y + 1, min_x : max_x + 1]
    filled_region[mask] = True


def _dilate_padding(pixels: NDArray[np.uint8], filled: NDArray[np.bool_], iterations: int) -> None:
    for _index in range(iterations):
        candidates = np.zeros_like(filled)
        candidates[1:, :] |= filled[:-1, :]
        candidates[:-1, :] |= filled[1:, :]
        candidates[:, 1:] |= filled[:, :-1]
        candidates[:, :-1] |= filled[:, 1:]
        candidates &= ~filled
        if not np.any(candidates):
            return
        new_pixels = pixels.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            source_y = slice(max(0, -dy), pixels.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), pixels.shape[1] - max(0, dx))
            target_y = slice(max(0, dy), pixels.shape[0] - max(0, -dy))
            target_x = slice(max(0, dx), pixels.shape[1] - max(0, -dx))
            target_mask = candidates[target_y, target_x] & filled[source_y, source_x]
            new_pixels[target_y, target_x][target_mask] = pixels[source_y, source_x][target_mask]
        pixels[:] = new_pixels
        filled |= candidates


def _png_bytes(pixels: NDArray[np.uint8]) -> bytes:
    buffer = pyio.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels), mode="RGBA").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def _baked_material(
    asset: Asset,
    material_id: str,
    source_material_ids: list[str],
    options: BakeMaterialOptions,
    image_ids: dict[str, str],
) -> Material:
    materials = [asset.materials[material_id] for material_id in source_material_ids if material_id in asset.materials]
    if not materials:
        base_color = (1.0, 1.0, 1.0, 1.0)
        metallic = 0.0
        roughness = 0.5
        opacity = 1.0
    else:
        base_color_values = np.asarray([material.base_color for material in materials], dtype=np.float64)
        base_color = cast(tuple[float, float, float, float], tuple(base_color_values.mean(axis=0).tolist()))
        metallic = float(np.mean([material.metallic for material in materials]))
        roughness = float(np.mean([material.roughness for material in materials]))
        opacity = float(np.mean([material.opacity for material in materials]))
    metadata: dict[str, object] = {
        "baked": "true",
        "baked_maps": ",".join(options.bake),
        "maps_resolution": str(options.maps_resolution),
        "lightmap_resolution": str(options.lightmap_resolution),
        "padding": str(options.padding),
        "source_material_ids": ",".join(source_material_ids),
        "baked_texture_kind": "raster_atlas",
        "baked_texture_resolution": str(options.maps_resolution),
        "baked_lightmap_resolution": str(options.lightmap_resolution),
    }
    metadata.update(_baked_texture_metadata(asset, image_ids))
    return Material(
        id=material_id,
        name="Baked Material",
        base_color=base_color,
        metallic=metallic,
        roughness=roughness,
        opacity=opacity,
        metadata=metadata,
    )


def _baked_texture_metadata(asset: Asset, image_ids: dict[str, str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for texture_name, image_id in image_ids.items():
        image = asset.images[image_id]
        metadata[f"baked_texture_{texture_name}_image"] = image_id
        metadata[f"baked_texture_{texture_name}_uri"] = image.data_uri()
        if texture_name == "emissive":
            for key in (
                "baked_emissive_source",
                "baked_emissive_material_faces",
                "baked_emissive_fallback_faces",
            ):
                value = image.metadata.get(key)
                if value is not None:
                    metadata[key] = str(value)
    return metadata


def _solid_png_data_uri(color: tuple[float, float, float, float]) -> str:
    pixel = bytes(_color_byte(value) for value in color)
    raw = b"\x00" + pixel
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw))
        + _png_chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii")


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    checksum = binascii.crc32(chunk_type)
    checksum = binascii.crc32(data, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", checksum)


def _color_byte(value: float) -> int:
    return max(0, min(255, int(round(value * 255.0))))


def _drop_unreferenced_materials(asset: Asset) -> None:
    referenced = {material_id for part in asset.parts.values() for material_id in part.material_ids}
    asset.materials = {
        material_id: material for material_id, material in asset.materials.items() if material_id in referenced
    }
