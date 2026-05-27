from __future__ import annotations

import numpy as np

from fascat.asset import Asset
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import StageOptions


def stage_asset(asset: Asset, options: StageOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    tangent_summary = {
        "generated": 0,
        "regenerated": 0,
        "dropped": 0,
        "invalidated": 0,
        "missing_uv0": 0,
    }
    if options.uv1 in {"unwrap", "lightmap"}:
        _require_xatlas()
    if options.uv0 in {"unwrap", "lightmap"}:
        _require_xatlas()
    if options.uv0 in {"unwrap", "lightmap"} or options.uv1 in {"unwrap", "lightmap"}:
        _warn_unwrap_solver_limits(result, options)
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
        had_tangents = mesh.tangents is not None
        uv_modes: dict[int, str] = {}
        edited_uv_channels: set[int] = set()
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
            uv_modes[0] = "box"
            edited_uv_channels.add(0)
        elif options.uv0 in {"unwrap", "lightmap"}:
            mesh = _unwrap_uv(mesh, 0)
            _tag_uv_metadata(mesh, 0, options.uv0, options)
            uv_modes[0] = options.uv0
            edited_uv_channels.add(0)
        if options.uv1 == "box":
            mesh = mesh.box_uv(1)
            _tag_uv_metadata(mesh, 1, "box", options)
            uv_modes[1] = "box"
            edited_uv_channels.add(1)
        elif options.uv1 in {"unwrap", "lightmap"}:
            mesh = _unwrap_uv(mesh, 1)
            _tag_uv_metadata(mesh, 1, options.uv1, options)
            uv_modes[1] = options.uv1
            edited_uv_channels.add(1)
        elif options.uv1 == "copy_uv0":
            mesh = _copy_uv_channel(result, part.id, mesh, source=0, target=1)
            if 1 in mesh.uvs:
                _tag_uv_metadata(mesh, 1, "copy_uv0", options)
                uv_modes[1] = "copy_uv0"
                edited_uv_channels.add(1)
        for channel in options.normalize_uvs:
            mesh = _normalize_uv_channel(result, part.id, mesh, channel)
            if channel in mesh.uvs:
                edited_uv_channels.add(channel)
        mesh = _stage_tangents(
            result,
            part.id,
            mesh,
            options,
            had_tangents=had_tangents,
            edited_uv_channels=edited_uv_channels,
            tangent_summary=tangent_summary,
        )
        if options.validate_normals:
            mesh.validate_normals(require_tangents=options.tangents)
        _tag_uv_layout_quality(result, part.id, mesh, uv_modes)
        part.mesh = mesh
        part.fingerprint = mesh.fingerprint()
    _tag_tangent_summary(result, tangent_summary)
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
    if mode.startswith("copy_uv"):
        mesh.metadata[f"{prefix}_source_channel"] = mode.removeprefix("copy_uv")
        mesh.metadata[f"{prefix}_copy_status"] = "copied"
    mesh.metadata[f"{prefix}_padding"] = str(options.unwrap.padding)
    if options.unwrap.texel_density is not None:
        mesh.metadata[f"{prefix}_texel_density"] = str(options.unwrap.texel_density)
    if options.unwrap.max_stretch is not None:
        mesh.metadata[f"{prefix}_max_stretch"] = str(options.unwrap.max_stretch)
    if mode in {"unwrap", "lightmap"}:
        mesh.metadata[f"{prefix}_unwrap_backend"] = "xatlas"
        mesh.metadata[f"{prefix}_unwrap_method"] = options.unwrap.method
        if options.unwrap.method != "default":
            mesh.metadata[f"{prefix}_unwrap_method_status"] = "intent"
        if options.unwrap.iterations is not None:
            mesh.metadata[f"{prefix}_unwrap_iterations"] = str(options.unwrap.iterations)
            mesh.metadata[f"{prefix}_unwrap_iterations_status"] = "intent"
        if options.unwrap.tolerance is not None:
            mesh.metadata[f"{prefix}_unwrap_tolerance"] = str(options.unwrap.tolerance)
            mesh.metadata[f"{prefix}_unwrap_tolerance_status"] = "intent"
    if options.atlas.enabled:
        mesh.metadata[f"{prefix}_atlas"] = "atlas_0"
        mesh.metadata[f"{prefix}_atlas_size"] = str(options.atlas.max_size)


def _copy_uv_channel(asset: Asset, part_id: str, mesh: Mesh, *, source: int, target: int) -> Mesh:
    result = mesh.copy()
    if source not in result.uvs:
        result.metadata[f"uv{target}_copy_status"] = "missing_source"
        asset.report.add_warning(f"part {part_id} requested UV{target} copy from UV{source}, but UV{source} is missing")
        return result
    result.uvs[target] = result.uvs[source].copy()
    return result


def _normalize_uv_channel(asset: Asset, part_id: str, mesh: Mesh, channel: int) -> Mesh:
    result = mesh.copy()
    prefix = f"uv{channel}"
    if channel not in result.uvs:
        result.metadata[f"{prefix}_normalize_status"] = "missing_channel"
        asset.report.add_warning(f"part {part_id} requested UV{channel} normalization, but UV{channel} is missing")
        return result
    uv = result.uvs[channel]
    if uv.shape[0] == 0:
        result.metadata[f"{prefix}_normalize_status"] = "empty"
        return result
    mins = uv.min(axis=0)
    maxs = uv.max(axis=0)
    span = maxs - mins
    normalized = np.zeros_like(uv)
    active = span > 0.0
    normalized[:, active] = (uv[:, active] - mins[active]) / span[active]
    result.uvs[channel] = normalized
    if channel == 0:
        result.tangents = None
    result.metadata[f"{prefix}_normalize_status"] = "normalized"
    result.metadata[f"{prefix}_normalize_bounds_before"] = ",".join(f"{value:.9g}" for value in (*mins, *maxs))
    if not bool(np.all(active)):
        zero_axes = [axis for axis, enabled in zip(("u", "v"), active.tolist(), strict=True) if not enabled]
        result.metadata[f"{prefix}_normalize_zero_axes"] = ",".join(zero_axes)
    return result


def _stage_tangents(
    asset: Asset,
    part_id: str,
    mesh: Mesh,
    options: StageOptions,
    *,
    had_tangents: bool,
    edited_uv_channels: set[int],
    tangent_summary: dict[str, int],
) -> Mesh:
    invalidated_by_uv_edit = bool(had_tangents and edited_uv_channels and mesh.tangents is None)
    if invalidated_by_uv_edit:
        channels = ",".join(str(channel) for channel in sorted(edited_uv_channels))
        mesh.metadata["tangents_invalidated_by_uv_edit"] = channels
        tangent_summary["invalidated"] += 1

    if options.tangents:
        if 0 not in mesh.uvs:
            mesh = mesh.copy()
            mesh.tangents = None
            mesh.metadata["tangents_status"] = "missing_uv0"
            tangent_summary["missing_uv0"] += 1
            asset.report.add_warning(
                f"part {part_id} requested tangents but UV0 is missing; generate or preserve UV0 before tangents"
            )
            return mesh
        mesh = mesh.compute_tangents()
        mesh.metadata["tangents_status"] = "regenerated" if invalidated_by_uv_edit else "generated"
        mesh.metadata["tangents_uv_channel"] = "0"
        tangent_summary["generated"] += 1
        if invalidated_by_uv_edit:
            tangent_summary["regenerated"] += 1
        return mesh

    if had_tangents and mesh.tangents is None:
        mesh.metadata["tangents_status"] = "dropped"
        if edited_uv_channels:
            mesh.metadata["tangents_drop_reason"] = "uv_edit"
            channels = ",".join(str(channel) for channel in sorted(edited_uv_channels))
            asset.report.add_warning(
                f"part {part_id} existing tangents were dropped because UVs were regenerated on channel(s): {channels}"
            )
        elif not options.normals:
            mesh.metadata["tangents_drop_reason"] = "normals_disabled"
            asset.report.add_warning(f"part {part_id} existing tangents were dropped because normals are disabled")
        else:
            mesh.metadata["tangents_drop_reason"] = "not_requested"
        tangent_summary["dropped"] += 1
    elif mesh.tangents is not None:
        mesh.metadata["tangents_status"] = "preserved"
    return mesh


def _tag_tangent_summary(asset: Asset, tangent_summary: dict[str, int]) -> None:
    for name, count in tangent_summary.items():
        if count:
            asset.metadata[f"stage_tangents_{name}_parts"] = str(count)


def _warn_unwrap_solver_limits(asset: Asset, options: StageOptions) -> None:
    if options.unwrap.method == "default" and options.unwrap.iterations is None and options.unwrap.tolerance is None:
        return
    asset.report.add_warning(
        "xatlas unwrap backend records method, iteration, and tolerance controls as intent; "
        "the current backend does not expose those solver controls"
    )


def _tag_uv_layout_quality(asset: Asset, part_id: str, mesh: Mesh, uv_modes: dict[int, str]) -> None:
    for channel in sorted(mesh.uvs):
        prefix = f"uv{channel}"
        mode = uv_modes.get(channel, str(mesh.metadata.get(f"{prefix}_mode", mesh.metadata.get(prefix, "existing"))))
        stats = mesh.uv_layout_stats(channel)
        mesh.metadata[f"{prefix}_out_of_unit_vertices"] = str(stats["out_of_unit_vertices"])
        mesh.metadata[f"{prefix}_degenerate_faces"] = str(stats["degenerate_faces"])
        mesh.metadata[f"{prefix}_overlap_pairs"] = str(stats["overlapping_face_pairs"])
        if channel != 1 and mode != "lightmap":
            continue
        problems: list[str] = []
        if stats["out_of_unit_vertices"]:
            problems.append(f"{stats['out_of_unit_vertices']} UV vertices outside 0..1")
        if stats["degenerate_faces"]:
            problems.append(f"{stats['degenerate_faces']} degenerate UV faces")
        if stats["overlapping_face_pairs"]:
            problems.append(f"{stats['overlapping_face_pairs']} overlapping UV face pairs")
        if problems:
            asset.report.add_warning(
                f"part {part_id} {prefix} violates lightmap/baking constraints: {', '.join(problems)}"
            )


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
