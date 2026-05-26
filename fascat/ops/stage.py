from __future__ import annotations

from fascat.asset import Asset
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import StageOptions


def stage_asset(asset: Asset, options: StageOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    if options.uv1 in {"unwrap", "lightmap"}:
        _require_xatlas()
    if options.uv0 in {"unwrap", "lightmap"}:
        _require_xatlas()
    if options.merge_equivalent_materials:
        _merge_equivalent_materials(result, selected_part_ids=selected_part_ids)
    if options.material_mode == "pbr":
        _normalize_materials_to_pbr(result)
    if options.atlas.enabled:
        _tag_material_atlas(result, options)
    _stage_materials(result, options, selected_part_ids=selected_part_ids)

    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.mesh is None:
            continue
        mesh = part.mesh
        if options.normals and options.normal_mode == "hard_edges":
            mesh = mesh.compute_hard_edge_normals(
                hard_edge_angle=options.hard_edge_angle,
                preserve_face_boundaries=options.preserve_face_boundaries,
            )
        elif options.normals and options.normal_mode == "flat":
            mesh = mesh.compute_flat_normals()
        elif options.normals:
            mesh = mesh.compute_normals()
        else:
            mesh.tangents = None
        if options.uv0 == "box":
            mesh = mesh.box_uv(0)
            _tag_uv_metadata(mesh, 0, "box", options)
        elif options.uv0 in {"unwrap", "lightmap"}:
            mesh = _unwrap_uv(mesh, 0)
            _tag_uv_metadata(mesh, 0, options.uv0, options)
        if options.uv1 == "box":
            mesh = mesh.box_uv(1)
            _tag_uv_metadata(mesh, 1, "box", options)
        elif options.uv1 in {"unwrap", "lightmap"}:
            mesh = _unwrap_uv(mesh, 1)
            _tag_uv_metadata(mesh, 1, options.uv1, options)
        if options.tangents:
            mesh = mesh.compute_tangents()
        if options.validate_normals:
            mesh.validate_normals(require_tangents=options.tangents)
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


def _merge_equivalent_materials(asset: Asset, *, selected_part_ids: set[str] | None) -> None:
    canonical_by_key: dict[tuple[object, ...], str] = {}
    replacement: dict[str, str] = {}
    for material_id, material in asset.materials.items():
        key = _material_key(material)
        canonical_id = canonical_by_key.get(key)
        if canonical_id is None:
            canonical_by_key[key] = material_id
            continue
        replacement[material_id] = canonical_id
    if not replacement:
        return

    for part in asset.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        old_ids = list(part.material_ids)
        remapped_ids: list[str] = []
        index_by_id: dict[str, int] = {}
        old_index_to_new: dict[int, int] = {}
        for old_index, material_id in enumerate(old_ids):
            new_id = replacement.get(material_id, material_id)
            if new_id not in index_by_id:
                index_by_id[new_id] = len(remapped_ids)
                remapped_ids.append(new_id)
            old_index_to_new[old_index] = index_by_id[new_id]
        part.material_ids = remapped_ids
        if part.mesh is not None and part.mesh.material_indices is not None:
            part.mesh.material_indices = part.mesh.material_indices.copy()
            for old_index, new_index in old_index_to_new.items():
                part.mesh.material_indices[part.mesh.material_indices == old_index] = new_index
    if selected_part_ids is None:
        asset.materials = {
            material_id: material for material_id, material in asset.materials.items() if material_id not in replacement
        }


def _normalize_materials_to_pbr(asset: Asset) -> None:
    normalized: dict[str, Material] = {}
    for material_id, material in asset.materials.items():
        alpha = min(material.base_color[3], material.opacity)
        metadata = {**material.metadata, "material_mode": "pbr", "pbr_normalized": "true"}
        normalized[material_id] = Material(
            id=material.id,
            name=material.name,
            base_color=(material.base_color[0], material.base_color[1], material.base_color[2], alpha),
            metallic=max(0.0, min(1.0, material.metallic)),
            roughness=max(0.04, min(1.0, material.roughness)),
            opacity=alpha,
            metadata=metadata,
        )
    asset.materials = normalized


def _tag_material_atlas(asset: Asset, options: StageOptions) -> None:
    source_ids = sorted(asset.materials)
    for material in asset.materials.values():
        material.metadata["atlas"] = "atlas_0"
        material.metadata["atlas_size"] = str(options.atlas.max_size)
        material.metadata["atlas_source_count"] = str(len(source_ids))
        material.metadata["texture_bake_hooks"] = "base_color,opacity"
        if options.unwrap.texel_density is not None:
            material.metadata["texel_density"] = str(options.unwrap.texel_density)
    for part in asset.parts.values():
        if part.material_ids:
            part.metadata["atlas"] = "atlas_0"
            part.metadata["atlas_material_ids"] = ",".join(part.material_ids)


def _tag_uv_metadata(mesh: Mesh, channel: int, mode: str, options: StageOptions) -> None:
    prefix = f"uv{channel}"
    mesh.metadata.setdefault(prefix, mode)
    mesh.metadata[f"{prefix}_mode"] = mode
    mesh.metadata[f"{prefix}_padding"] = str(options.unwrap.padding)
    if options.unwrap.texel_density is not None:
        mesh.metadata[f"{prefix}_texel_density"] = str(options.unwrap.texel_density)
    if options.unwrap.max_stretch is not None:
        mesh.metadata[f"{prefix}_max_stretch"] = str(options.unwrap.max_stretch)
    if options.atlas.enabled:
        mesh.metadata[f"{prefix}_atlas"] = "atlas_0"
        mesh.metadata[f"{prefix}_atlas_size"] = str(options.atlas.max_size)


def _material_key(material: Material) -> tuple[object, ...]:
    return (
        tuple(round(value, 6) for value in material.base_color),
        round(material.metallic, 6),
        round(material.roughness, 6),
        round(material.opacity, 6),
    )


def _require_xatlas() -> None:
    try:
        import xatlas  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("UV unwrap requires the optional xatlas dependency") from exc


def _unwrap_uv(mesh: Mesh, channel: int) -> Mesh:
    return mesh.unwrap_uv(channel)
