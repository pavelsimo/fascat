from __future__ import annotations

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.options import StageOptions


def stage_asset(asset: Asset, options: StageOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    if options.uv1 == "unwrap":
        _require_xatlas()
    if options.uv0 == "unwrap":
        _require_xatlas()
    _stage_materials(result, options, selected_part_ids=selected_part_ids)

    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.normals:
            mesh = mesh.compute_normals()
        if options.uv0 == "box":
            mesh = mesh.box_uv(0)
        elif options.uv0 == "unwrap":
            mesh = _unwrap_uv(mesh, 0)
        if options.uv1 == "box":
            mesh = mesh.box_uv(1)
        elif options.uv1 == "unwrap":
            mesh = _unwrap_uv(mesh, 1)
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    return result


def _stage_materials(asset: Asset, options: StageOptions, *, selected_part_ids: set[str] | None) -> None:
    if options.materials == "cad":
        return
    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        part.metadata.pop("display_color", None)
        if options.materials == "display" and part.material_ids:
            material = asset.materials.get(part.material_ids[0])
            if material is not None:
                part.metadata["display_color"] = ",".join(f"{value:.6f}" for value in material.base_color)
        part.material_ids = []
        if part.mesh is not None:
            part.mesh.material_indices = None
    if selected_part_ids is None:
        asset.materials = {}


def _require_xatlas() -> None:
    try:
        import xatlas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("UV unwrap requires the optional xatlas dependency") from exc


def _unwrap_uv(mesh: Mesh, channel: int) -> Mesh:
    return mesh.unwrap_uv(channel)
