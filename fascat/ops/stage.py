from __future__ import annotations

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.options import StageOptions


def stage_asset(asset: Asset, options: StageOptions) -> Asset:
    result = asset.copy(keep_source=True)
    if options.uv1 == "unwrap":
        _require_xatlas()
    if options.uv0 == "unwrap":
        _require_xatlas()

    for part in result.parts.values():
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.normals or mesh.normals is None:
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


def _require_xatlas() -> None:
    try:
        import xatlas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("UV unwrap requires the optional xatlas dependency") from exc


def _unwrap_uv(mesh: Mesh, channel: int) -> Mesh:
    return mesh.unwrap_uv(channel)
