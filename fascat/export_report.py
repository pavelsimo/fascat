from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from fascat.mesh import Mesh

_TEXTURE_URI_METADATA_KEYS = (
    "baked_texture_base_color_uri",
    "baked_texture_metallic_roughness_uri",
    "baked_texture_normal_uri",
    "baked_texture_occlusion_uri",
    "baked_texture_emissive_uri",
)
_TEXTURE_IMAGE_METADATA_KEYS = (
    "baked_texture_base_color_image",
    "baked_texture_metallic_roughness_image",
    "baked_texture_normal_image",
    "baked_texture_occlusion_image",
    "baked_texture_emissive_image",
)
_TEXTURE_METADATA_KEY_PAIRS = tuple(zip(_TEXTURE_IMAGE_METADATA_KEYS, _TEXTURE_URI_METADATA_KEYS, strict=True))


def stats_with_file_size(
    stats: dict[str, int],
    path: str | Path,
    budget_mb: float | None,
    asset: Any,
) -> dict[str, int]:
    output_path = Path(path)
    if str(path) == "-" or not output_path.exists():
        return stats
    size = output_path.stat().st_size
    referenced_ids = referenced_material_ids(asset)
    referenced_materials_by_id = referenced_materials(asset, referenced_ids=referenced_ids)
    estimates = export_payload_estimates(asset, referenced_materials_by_id=referenced_materials_by_id)
    result = {
        **stats,
        "file_size_bytes": size,
        **estimates,
        "export_estimated_payload_bytes": sum(estimates.values()),
        **export_material_counts(asset, referenced_ids=referenced_ids),
        **export_image_counts(asset, referenced_materials_by_id=referenced_materials_by_id),
    }
    if budget_mb is not None:
        budget_bytes = int(budget_mb * 1_000_000)
        result["file_size_budget_bytes"] = budget_bytes
        if size > budget_bytes:
            asset.report.add_warning(f"file size budget exceeded: {size} bytes > {budget_bytes} bytes")
    return result


def export_payload_estimates(
    asset: Any,
    *,
    referenced_materials_by_id: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    return {
        "export_estimated_geometry_bytes": _geometry_bytes(asset),
        "export_estimated_texture_bytes": _texture_bytes(asset, referenced_materials_by_id=referenced_materials_by_id),
        "export_estimated_metadata_bytes": _metadata_bytes(asset),
    }


def export_material_counts(
    asset: Any,
    *,
    referenced_ids: Collection[str] | None = None,
) -> dict[str, int]:
    referenced = (
        referenced_material_ids(asset)
        if referenced_ids is None
        else {material_id for material_id in referenced_ids if material_id in asset.materials}
    )
    return {
        "export_source_material_count": len(asset.materials),
        "export_referenced_material_count": len(referenced),
        "export_unused_material_count": max(0, len(asset.materials) - len(referenced)),
        "export_written_material_count": len(referenced),
    }


def export_image_counts(
    asset: Any,
    *,
    referenced_materials_by_id: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    referenced = referenced_materials(asset) if referenced_materials_by_id is None else referenced_materials_by_id
    source_refs = _texture_refs(asset, asset.materials.values())
    referenced_refs = _texture_refs(asset, referenced.values())
    source_image_ids = _source_image_ids(asset)
    source_unique = source_image_ids | set(source_refs)
    referenced_unique = set(referenced_refs)
    return {
        "export_source_image_count": len(source_unique),
        "export_source_image_reference_count": len(source_refs),
        "export_referenced_image_count": len(referenced_unique),
        "export_referenced_image_reference_count": len(referenced_refs),
        "export_unused_image_count": len(source_unique - referenced_unique),
        "export_duplicate_image_reference_count": max(0, len(referenced_refs) - len(referenced_unique)),
        "export_written_image_count": len(referenced_unique),
    }


def referenced_material_ids(asset: Any) -> set[str]:
    referenced: set[str] = set()
    for part in asset.parts.values():
        if part.mesh is not None:
            referenced.update(_mesh_material_ids(part, part.mesh))
        for lod_mesh in part.lod_meshes:
            referenced.update(_mesh_material_ids(part, lod_mesh))
    return {material_id for material_id in referenced if material_id in asset.materials}


def referenced_materials(
    asset: Any,
    *,
    referenced_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    referenced = referenced_material_ids(asset) if referenced_ids is None else referenced_ids
    return {material_id: material for material_id, material in asset.materials.items() if material_id in referenced}


def _mesh_material_ids(part: Any, mesh: Mesh) -> set[str]:
    if mesh.material_indices is None or mesh.material_indices.size == 0:
        return set(part.material_ids)
    used: set[str] = set()
    for index in np.unique(mesh.material_indices.astype(np.int64, copy=False)).tolist():
        if 0 <= index < len(part.material_ids):
            used.add(part.material_ids[index])
    return used


def _geometry_bytes(asset: Any) -> int:
    total = 0
    for part in asset.parts.values():
        if part.mesh is not None:
            total += _mesh_payload_bytes(part.mesh)
        for lod_mesh in part.lod_meshes:
            total += _mesh_payload_bytes(lod_mesh)
    return total


def _mesh_payload_bytes(mesh: Mesh) -> int:
    total = int(mesh.points.nbytes + mesh.faces.nbytes)
    if mesh.normals is not None:
        total += int(mesh.normals.nbytes)
    if mesh.tangents is not None:
        total += int(mesh.tangents.nbytes)
    total += sum(int(values.nbytes) for values in mesh.uvs.values())
    if mesh.material_indices is not None:
        total += int(mesh.material_indices.nbytes)
    total += sum(int(values.nbytes) for values in mesh.face_groups.values())
    return total


def _texture_bytes(
    asset: Any,
    *,
    referenced_materials_by_id: Mapping[str, Any] | None = None,
) -> int:
    referenced = referenced_materials(asset) if referenced_materials_by_id is None else referenced_materials_by_id
    return sum(_texture_ref_payload_bytes(asset, ref) for ref in set(_texture_refs(asset, referenced.values())))


def _texture_refs(asset: Any, materials: Iterable[Any]) -> list[str]:
    refs: list[str] = []
    for material in materials:
        for image_key, uri_key in _TEXTURE_METADATA_KEY_PAIRS:
            value = material.metadata.get(image_key)
            if isinstance(value, str) and value in getattr(asset, "images", {}):
                refs.append(f"image:{value}")
                continue
            uri = material.metadata.get(uri_key)
            if isinstance(uri, str) and uri.startswith("data:image/"):
                refs.append(f"uri:{uri}")
    return refs


def _source_image_ids(asset: Any) -> set[str]:
    return {f"image:{image_id}" for image_id in getattr(asset, "images", {})}


def _texture_ref_payload_bytes(asset: Any, ref: str) -> int:
    if ref.startswith("image:"):
        image = getattr(asset, "images", {}).get(ref.removeprefix("image:"))
        return 0 if image is None else len(image.data)
    return _data_uri_payload_bytes(ref.removeprefix("uri:"))


def _data_uri_payload_bytes(value: str) -> int:
    if not value.startswith("data:image/") or "," not in value:
        return len(value.encode("utf-8"))
    _header, encoded = value.split(",", 1)
    try:
        return len(base64.b64decode(encoded, validate=True))
    except (binascii.Error, ValueError):
        return len(encoded.encode("utf-8"))


def _metadata_bytes(asset: Any) -> int:
    payload: dict[str, object] = {}
    asset_metadata = _export_metadata(asset.metadata)
    if asset_metadata:
        payload["asset"] = asset_metadata

    node_metadata = {}
    for node in asset.root.walk():
        metadata = _export_metadata(node.metadata)
        if metadata:
            node_metadata[node.id] = metadata
    if node_metadata:
        payload["nodes"] = node_metadata

    part_metadata: dict[str, object] = {}
    for part_id, part in asset.parts.items():
        part_payload: dict[str, object] = {}
        metadata = _export_metadata(part.metadata)
        if metadata:
            part_payload["metadata"] = metadata
        if part.mesh is not None:
            mesh_metadata = _export_metadata(part.mesh.metadata)
            if mesh_metadata:
                part_payload["mesh"] = mesh_metadata
        lod_metadata = [_export_metadata(mesh.metadata) for mesh in part.lod_meshes]
        lod_metadata = [metadata for metadata in lod_metadata if metadata]
        if lod_metadata:
            part_payload["lods"] = lod_metadata
        if part_payload:
            part_metadata[part_id] = part_payload
    if part_metadata:
        payload["parts"] = part_metadata

    material_metadata = {
        material_id: metadata
        for material_id, material in asset.materials.items()
        if (metadata := _export_metadata(material.metadata))
    }
    if material_metadata:
        payload["materials"] = material_metadata

    if asset.pmi:
        payload["pmi"] = [annotation.to_dict() for annotation in asset.pmi]

    if not payload:
        return 0
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))


def _export_metadata(metadata: dict[str, object]) -> dict[str, object]:
    hidden = {*_TEXTURE_URI_METADATA_KEYS, *_TEXTURE_IMAGE_METADATA_KEYS}
    return {key: value for key, value in metadata.items() if key not in hidden}
