from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.material import Material


def write_usd(asset: Asset, path: str | Path) -> None:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("USD export requires usd-core") from exc

    output_path = Path(path)
    if output_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise ValueError(f"unsupported USD extension: {output_path.suffix or '<none>'}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.CreateNew(str(output_path))
    if stage is None:
        raise RuntimeError(f"failed to create USD stage: {output_path}")

    UsdGeom.SetStageMetersPerUnit(stage, asset.meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z if asset.up_axis == "Z" else UsdGeom.Tokens.y)

    scene_path = "/Scene"
    scene = UsdGeom.Xform.Define(stage, scene_path)
    stage.SetDefaultPrim(scene.GetPrim())

    material_paths = _write_materials(stage, asset.materials)
    prototype_paths = _write_prototypes(stage, asset.parts, material_paths)
    _write_node(stage, asset.root, scene_path, asset.parts, prototype_paths)

    if not stage.GetRootLayer().Save():
        raise RuntimeError(f"failed to save USD stage: {output_path}")


def validate_usd(path: str | Path) -> dict[str, int]:
    try:
        from pxr import Usd, UsdGeom
    except ImportError as exc:
        raise RuntimeError("USD validation requires usd-core") from exc

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"failed to open USD stage: {path}")
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError("USD stage has no defaultPrim")

    mesh_count = 0
    point_count = 0
    face_count = 0
    for prim in Usd.PrimRange(default_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        scheme = mesh.GetSubdivisionSchemeAttr().Get()
        if scheme != "none":
            raise RuntimeError(f"mesh {prim.GetPath()} has subdivisionScheme={scheme!r}, expected 'none'")
        points = mesh.GetPointsAttr().Get() or []
        counts = mesh.GetFaceVertexCountsAttr().Get() or []
        indices = mesh.GetFaceVertexIndicesAttr().Get() or []
        if any(count != 3 for count in counts):
            raise RuntimeError(f"mesh {prim.GetPath()} contains non-triangle faces")
        if len(indices) != len(counts) * 3:
            raise RuntimeError(f"mesh {prim.GetPath()} has invalid face index count")
        if indices and (min(indices) < 0 or max(indices) >= len(points)):
            raise RuntimeError(f"mesh {prim.GetPath()} has out-of-range face indices")
        mesh_count += 1
        point_count += len(points)
        face_count += len(counts)

    if mesh_count == 0:
        raise RuntimeError("USD stage contains no meshes under defaultPrim")
    return {"meshes": mesh_count, "points": point_count, "triangles": face_count}


def _write_materials(stage: Any, materials: dict[str, Material]) -> dict[str, str]:
    from pxr import Gf, Sdf, UsdShade

    paths: dict[str, str] = {}
    for material in materials.values():
        material_path = f"/Materials/{_usd_name(material.id)}"
        usd_material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*material.base_color[:3]))
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(material.opacity))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(material.metallic))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(material.roughness))
        usd_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        paths[material.id] = material_path
    return paths


def _write_prototypes(stage: Any, parts: dict[str, Part], material_paths: dict[str, str]) -> dict[tuple[str, int], str]:
    from pxr import UsdGeom

    prototype_paths: dict[tuple[str, int], str] = {}
    UsdGeom.Scope.Define(stage, "/__Prototypes")
    for part in parts.values():
        meshes = [part.mesh, *part.lod_meshes]
        for lod_index, mesh in enumerate(meshes):
            if mesh is None:
                continue
            part_path = f"/__Prototypes/{_usd_name(part.id)}_lod{lod_index}"
            UsdGeom.Xform.Define(stage, part_path)
            _write_mesh(stage, f"{part_path}/Mesh", part, mesh, material_paths)
            prototype_paths[(part.id, lod_index)] = part_path
    return prototype_paths


def _write_node(
    stage: Any,
    node: Node,
    parent_path: str,
    parts: dict[str, Part],
    prototype_paths: dict[tuple[str, int], str],
) -> None:
    from pxr import Gf, Sdf, UsdGeom

    counters: dict[str, int] = {}
    for child in node.children:
        base_name = _usd_name(child.name or child.id)
        index = counters.get(base_name, 0)
        counters[base_name] = index + 1
        name = base_name if index == 0 else f"{base_name}_{index + 1}"
        path = f"{parent_path}/{name}"
        xform = UsdGeom.Xform.Define(stage, path)
        if not np.allclose(child.transform, np.eye(4)):
            xform.AddTransformOp().Set(Gf.Matrix4d(child.transform.tolist()))
        xform.GetPrim().SetCustomDataByKey("fascat:originalName", child.name)
        if child.part_id is not None and (child.part_id, 0) in prototype_paths:
            part = parts[child.part_id]
            prim = xform.GetPrim()
            if part.lod_meshes:
                variant_set = prim.GetVariantSets().AddVariantSet("lod")
                for lod_index in range(len(part.lod_meshes) + 1):
                    variant_name = f"lod{lod_index}"
                    variant_set.AddVariant(variant_name)
                    variant_set.SetVariantSelection(variant_name)
                    with variant_set.GetVariantEditContext():
                        references = prim.GetReferences()
                        references.ClearReferences()
                        references.AddInternalReference(Sdf.Path(prototype_paths[(child.part_id, lod_index)]))
                variant_set.SetVariantSelection("lod0")
            else:
                prim.GetReferences().AddInternalReference(Sdf.Path(prototype_paths[(child.part_id, 0)]))
            if _part_occurrence_count(node, child.part_id) > 1:
                prim.SetInstanceable(True)
        _write_node(stage, child, path, parts, prototype_paths)


def _write_mesh(stage: Any, path: str, part: Part, mesh: Any, material_paths: dict[str, str]) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    mesh.validate()
    usd_mesh = UsdGeom.Mesh.Define(stage, path)
    usd_mesh.CreateSubdivisionSchemeAttr("none")
    usd_mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in mesh.points.tolist()])
    usd_mesh.CreateFaceVertexCountsAttr([3] * mesh.triangle_count)
    usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).astype(int).tolist())
    mins, maxs = mesh.bounds()
    usd_mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*mins.tolist()), Gf.Vec3f(*maxs.tolist())]))
    if mesh.normals is not None:
        usd_mesh.CreateNormalsAttr([Gf.Vec3f(*normal) for normal in mesh.normals.tolist()])
        usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if 0 in mesh.uvs:
        primvar = UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        )
        primvar.Set([Gf.Vec2f(*uv) for uv in mesh.uvs[0].tolist()])
    if part.material_ids:
        material_path = material_paths.get(part.material_ids[0])
        if material_path:
            material = UsdShade.Material.Get(stage, material_path)
            UsdShade.MaterialBindingAPI(usd_mesh).Bind(material)
    usd_mesh.GetPrim().SetCustomDataByKey("fascat:partId", part.id)
    usd_mesh.GetPrim().SetCustomDataByKey("fascat:originalName", part.name)


def _part_occurrence_count(node: Node, part_id: str) -> int:
    return sum(1 for item in node.walk() if item.part_id == part_id)


def _usd_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "item"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned
