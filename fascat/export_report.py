from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path
from typing import Any

from fascat.mesh import Mesh

_TEXTURE_URI_METADATA_KEYS = (
    "baked_texture_base_color_uri",
    "baked_texture_metallic_roughness_uri",
    "baked_texture_normal_uri",
    "baked_texture_occlusion_uri",
    "baked_texture_emissive_uri",
)


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
    estimates = export_payload_estimates(asset)
    result = {
        **stats,
        "file_size_bytes": size,
        **estimates,
        "export_estimated_payload_bytes": sum(estimates.values()),
    }
    if budget_mb is not None:
        budget_bytes = int(budget_mb * 1_000_000)
        result["file_size_budget_bytes"] = budget_bytes
        if size > budget_bytes:
            asset.report.add_warning(f"file size budget exceeded: {size} bytes > {budget_bytes} bytes")
    return result


def export_payload_estimates(asset: Any) -> dict[str, int]:
    return {
        "export_estimated_geometry_bytes": _geometry_bytes(asset),
        "export_estimated_texture_bytes": _texture_bytes(asset),
        "export_estimated_metadata_bytes": _metadata_bytes(asset),
    }


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


def _texture_bytes(asset: Any) -> int:
    total = 0
    for material in asset.materials.values():
        for key in _TEXTURE_URI_METADATA_KEYS:
            value = material.metadata.get(key)
            if isinstance(value, str):
                total += _data_uri_payload_bytes(value)
    return total


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
    return {key: value for key, value in metadata.items() if key not in _TEXTURE_URI_METADATA_KEYS}
