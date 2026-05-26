from __future__ import annotations

import math
from typing import Any

import numpy as np

from fascat.asset import Asset
from fascat.mesh import Mesh
from fascat.options import Tessellation


def tessellate_asset(asset: Asset, options: Tessellation) -> Asset:
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        if part.mesh is not None:
            continue
        if part.source_shape is None:
            result.report.add_warning(f"part has no source shape and cannot be tessellated: {part.name}")
            continue
        part.mesh = tessellate_shape(
            part.source_shape,
            options,
            face_material_indices=_face_material_indices_from_metadata(part.metadata),
        )
        if part.material_ids and part.mesh.material_indices is None:
            part.mesh.material_indices = np.zeros(part.mesh.triangle_count, dtype=np.int64)
        part.fingerprint = part.mesh.fingerprint()
        if not options.keep_brep:
            part.source_shape = None
    return _deduplicate_parts_by_fingerprint(result)


def tessellate_shape(
    shape: object,
    options: Tessellation,
    *,
    face_material_indices: list[int] | None = None,
) -> Mesh:
    try:
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopLoc import TopLoc_Location
        from OCP.TopoDS import TopoDS
    except ImportError as exc:
        raise RuntimeError("STEP tessellation requires cadquery-ocp") from exc

    brep_shape = shape
    mesher = BRepMesh_IncrementalMesh(
        brep_shape,
        float(options.sag),
        bool(options.relative),
        math.radians(float(options.angle)),
        True,
    )
    mesher.Perform()

    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    material_indices: list[int] = []
    face_index = 0
    explorer = TopExp_Explorer(brep_shape, TopAbs_FACE)
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        material_index = None
        if face_material_indices is not None and face_index < len(face_material_indices):
            material_index = face_material_indices[face_index]
        if triangulation is not None:
            offset = len(points)
            transform = location.Transformation()
            for node_index in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(node_index).Transformed(transform)
                points.append((point.X(), point.Y(), point.Z()))
            reversed_face = face.Orientation() == TopAbs_REVERSED
            for triangle_index in range(1, triangulation.NbTriangles() + 1):
                a, b, c = triangulation.Triangle(triangle_index).Get()
                if reversed_face:
                    faces.append((offset + c - 1, offset + b - 1, offset + a - 1))
                else:
                    faces.append((offset + a - 1, offset + b - 1, offset + c - 1))
                if material_index is not None:
                    material_indices.append(material_index)
        face_index += 1
        explorer.Next()

    mesh = Mesh(
        points=np.asarray(points, dtype=np.float64).reshape((-1, 3)),
        faces=np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        material_indices=(
            np.asarray(material_indices, dtype=np.int64)
            if material_indices and len(material_indices) == len(faces)
            else None
        ),
        metadata={"occt_faces": str(face_index)},
    )
    if options.max_edge_length is not None:
        mesh = mesh.subdivide_long_edges(options.max_edge_length)
    if options.create_normals:
        mesh = mesh.compute_normals()
    else:
        mesh.normals = None
    mesh.validate()
    return mesh


def _face_material_indices_from_metadata(metadata: dict[str, str]) -> list[int] | None:
    value = metadata.get("occt_face_material_indices")
    if not value:
        return None
    return [int(item) for item in value.split(",") if item]


def _deduplicate_parts_by_fingerprint(asset: Asset) -> Asset:
    canonical_by_key: dict[tuple[str, tuple[str, ...]], str] = {}
    replacements: dict[str, str] = {}
    for part_id, part in asset.parts.items():
        if part.fingerprint is None:
            continue
        key = (part.fingerprint, tuple(part.material_ids))
        canonical_id = canonical_by_key.get(key)
        if canonical_id is None:
            canonical_by_key[key] = part_id
            continue
        replacements[part_id] = canonical_id
    if not replacements:
        return asset

    for node in asset.root.walk():
        if node.part_id in replacements:
            node.part_id = replacements[node.part_id]
    asset.parts = {part_id: part for part_id, part in asset.parts.items() if part_id not in replacements}
    return asset


def shape_fingerprint(shape: Any) -> str:
    try:
        from OCP.TopoDS import TopoDS_Shape
    except ImportError:
        return str(id(shape))
    if isinstance(shape, TopoDS_Shape):
        return str(shape.HashCode(2_147_483_647))
    return str(id(shape))
