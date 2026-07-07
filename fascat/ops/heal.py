from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset
from fascat.mesh import _triangle_overlap_area_2d
from fascat.ops._occt_mesh import _transformed_occt_nodes, _triangulation_faces
from fascat.options import BrepHealOptions

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class BrepStatus:
    kind: str
    solids: int = 0
    shells: int = 0
    wires: int = 0
    edges: int = 0
    faces: int = 0
    open_shells: int = 0
    free_edges: int = 0
    small_edges: int = 0
    sliver_faces: int = 0
    overlapping_face_pairs: int = 0
    z_fighting_faces: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "solids": self.solids,
            "shells": self.shells,
            "wires": self.wires,
            "edges": self.edges,
            "faces": self.faces,
            "open_shells": self.open_shells,
            "free_edges": self.free_edges,
            "small_edges": self.small_edges,
            "sliver_faces": self.sliver_faces,
            "overlapping_face_pairs": self.overlapping_face_pairs,
            "z_fighting_faces": self.z_fighting_faces,
        }


@dataclass(frozen=True)
class _FaceOverlapDescriptor:
    index: int
    face: object
    points: FloatArray
    triangles: IntArray
    triangle_areas: FloatArray
    area: float
    normal: FloatArray
    plane_offset: float
    bbox_min: FloatArray
    bbox_max: FloatArray


@dataclass(frozen=True)
class _FaceOverlapReport:
    pair_count: int = 0
    face_count: int = 0
    faces_to_remove: tuple[object, ...] = ()


@dataclass(frozen=True)
class BrepHealDiagnostics:
    faces_removed: int = 0
    edges_removed: int = 0
    same_domain_faces_removed: int = 0
    same_domain_edges_removed: int = 0
    overlapping_faces_removed: int = 0
    open_shell_groups: int = 0
    open_shell_grouped_shells: int = 0
    open_shell_grouped_faces: int = 0


@dataclass(frozen=True)
class _OpenShellGroupPlan:
    groups: tuple[object, ...] = ()
    shells: int = 0
    faces: int = 0


def heal_brep_asset(
    asset: Asset,
    options: BrepHealOptions,
    *,
    selected_part_ids: set[str] | None = None,
    tolerance_policy: Mapping[str, object] | None = None,
) -> Asset:
    result = asset.copy(keep_source=True)
    if options.remove_sliver_faces:
        result.report.add_warning(
            "sliver face removal is not supported by the current BREP backend; "
            "sliver faces are reported but source shapes are left unchanged"
        )
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.source_shape is None:
            result.report.add_warning(f"part has no source shape and cannot be BREP healed: {part.name}")
            continue
        healed_shape, before, after, warnings, diagnostics = heal_shape(part.source_shape, options)
        for warning in warnings:
            result.report.add_warning(f"{part.name}: {warning}")
        if options.fail_on_open_shells and after.open_shells > 0:
            raise RuntimeError(f"BREP healing left open shells in part: {part.name}")
        _add_topology_warnings(result, part.name, after, options)
        part.source_shape = healed_shape
        part.metadata = {
            **part.metadata,
            "brep_kind": after.kind,
            "brep_solids": str(after.solids),
            "brep_shells": str(after.shells),
            "brep_wires": str(after.wires),
            "brep_edges": str(after.edges),
            "brep_faces": str(after.faces),
            "brep_open_shells": str(after.open_shells),
            "brep_free_edges": str(after.free_edges),
            "brep_unstitched_edges": str(after.free_edges),
            "brep_small_edges": str(after.small_edges),
            "brep_sliver_faces": str(after.sliver_faces),
            "brep_overlapping_face_pairs": str(after.overlapping_face_pairs),
            "brep_z_fighting_faces": str(after.z_fighting_faces),
            "brep_overlapping_face_pairs_resolved": str(
                max(0, before.overlapping_face_pairs - after.overlapping_face_pairs)
            ),
            "brep_overlapping_faces_removed": str(diagnostics.overlapping_faces_removed),
            "brep_faces_removed": str(diagnostics.faces_removed),
            "brep_edges_removed": str(diagnostics.edges_removed),
            "brep_same_domain_faces_removed": str(diagnostics.same_domain_faces_removed),
            "brep_same_domain_edges_removed": str(diagnostics.same_domain_edges_removed),
            "brep_open_shell_grouping": _open_shell_grouping_status(options, diagnostics),
            "brep_open_shell_groups": str(diagnostics.open_shell_groups),
            "brep_open_shell_grouped_shells": str(diagnostics.open_shell_grouped_shells),
            "brep_open_shell_grouped_faces": str(diagnostics.open_shell_grouped_faces),
            "brep_heal_operations": _operation_summary(options),
            "brep_before": str(before.to_dict()),
            "brep_after": str(after.to_dict()),
            **_tolerance_policy_metadata("brep_heal", tolerance_policy),
        }
    return result


def heal_shape(
    shape: object,
    options: BrepHealOptions,
) -> tuple[object, BrepStatus, BrepStatus, list[str], BrepHealDiagnostics]:
    before = brep_status(
        shape,
        max_sliver_area=options.max_sliver_area,
        small_edge_length=options.tolerance,
        detect_overlaps=options.remove_overlapping_faces,
        overlap_tolerance=options.tolerance,
        overlap_area_ratio=options.overlap_area_ratio,
    )
    healed = shape
    grouped_diagnostics = BrepHealDiagnostics()
    warnings: list[str] = []
    if options.group_open_shells:
        group_plan = _open_shell_group_plan(healed)
        if group_plan.groups:
            healed, grouped_warnings, grouped_diagnostics = _heal_open_shell_groups(group_plan, options)
            warnings.extend(grouped_warnings)
        else:
            healed, operation_warnings, grouped_diagnostics = _run_heal_operations(healed, options, before=before)
            warnings.extend(operation_warnings)
    else:
        healed, operation_warnings, grouped_diagnostics = _run_heal_operations(healed, options, before=before)
        warnings.extend(operation_warnings)
    after = brep_status(
        healed,
        max_sliver_area=options.max_sliver_area,
        small_edge_length=options.tolerance,
        detect_overlaps=options.remove_overlapping_faces,
        overlap_tolerance=options.tolerance,
        overlap_area_ratio=options.overlap_area_ratio,
    )
    diagnostics = BrepHealDiagnostics(
        faces_removed=max(0, before.faces - after.faces),
        edges_removed=max(0, before.edges - after.edges),
        same_domain_faces_removed=grouped_diagnostics.same_domain_faces_removed,
        same_domain_edges_removed=grouped_diagnostics.same_domain_edges_removed,
        overlapping_faces_removed=grouped_diagnostics.overlapping_faces_removed,
        open_shell_groups=grouped_diagnostics.open_shell_groups,
        open_shell_grouped_shells=grouped_diagnostics.open_shell_grouped_shells,
        open_shell_grouped_faces=grouped_diagnostics.open_shell_grouped_faces,
    )
    return healed, before, after, warnings, diagnostics


def _run_heal_operations(
    shape: object,
    options: BrepHealOptions,
    *,
    before: BrepStatus,
) -> tuple[object, list[str], BrepHealDiagnostics]:
    healed = shape
    warnings: list[str] = []
    same_domain_faces_removed = 0
    same_domain_edges_removed = 0
    overlapping_faces_removed = 0
    try:
        if options.fix_edges or options.unify_tolerances:
            healed = _fix_shape(healed, options)
        if options.sew_faces:
            healed = _sew_shape(healed, options)
        if options.unify_same_domain:
            before_unify = brep_status(
                healed,
                max_sliver_area=options.max_sliver_area,
                small_edge_length=options.tolerance,
            )
            healed = _unify_same_domain_shape(healed, options)
            after_unify = brep_status(
                healed,
                max_sliver_area=options.max_sliver_area,
                small_edge_length=options.tolerance,
            )
            same_domain_faces_removed = max(0, before_unify.faces - after_unify.faces)
            same_domain_edges_removed = max(0, before_unify.edges - after_unify.edges)
        if options.remove_overlapping_faces:
            healed, overlap_report = _remove_overlapping_faces_shape(healed, options)
            overlapping_faces_removed = len(overlap_report.faces_to_remove)
        if options.remove_sliver_faces and before.sliver_faces:
            warnings.append("sliver face removal is not supported by the current BREP backend")
    except Exception as exc:
        warnings.append(f"BREP healer skipped unsupported operation: {exc}")
        healed = shape
        same_domain_faces_removed = 0
        same_domain_edges_removed = 0
        overlapping_faces_removed = 0
    return (
        healed,
        warnings,
        BrepHealDiagnostics(
            same_domain_faces_removed=same_domain_faces_removed,
            same_domain_edges_removed=same_domain_edges_removed,
            overlapping_faces_removed=overlapping_faces_removed,
        ),
    )


def _open_shell_group_plan(shape: object) -> _OpenShellGroupPlan:
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp, TopExp_Explorer
        from OCP.TopoDS import TopoDS
    except ImportError:
        return _OpenShellGroupPlan()

    try:
        if _count_subshapes(shape, TopAbs_SOLID, TopExp_Explorer) > 0:
            return _OpenShellGroupPlan()
        shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    except Exception:
        return _OpenShellGroupPlan()

    shells: list[object] = []
    open_shells = 0
    face_count = 0
    edge_count = 0
    while shell_explorer.More():
        shell = TopoDS.Shell_s(shell_explorer.Current())
        shells.append(shell)
        if _shell_is_open(shell, BRepCheck_Analyzer, TopAbs_EDGE, TopAbs_FACE, TopExp):
            open_shells += 1
        face_count += _count_subshapes(shell, TopAbs_FACE, TopExp_Explorer)
        edge_count += _count_subshapes(shell, TopAbs_EDGE, TopExp_Explorer)
        shell_explorer.Next()

    if len(shells) <= 1 or open_shells == 0:
        return _OpenShellGroupPlan()
    if face_count != _count_subshapes(shape, TopAbs_FACE, TopExp_Explorer):
        return _OpenShellGroupPlan()
    if edge_count != _count_subshapes(shape, TopAbs_EDGE, TopExp_Explorer):
        return _OpenShellGroupPlan()
    return _OpenShellGroupPlan(groups=tuple(shells), shells=len(shells), faces=face_count)


def _heal_open_shell_groups(
    group_plan: _OpenShellGroupPlan,
    options: BrepHealOptions,
) -> tuple[object, list[str], BrepHealDiagnostics]:
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    warnings: list[str] = []
    same_domain_faces_removed = 0
    same_domain_edges_removed = 0
    overlapping_faces_removed = 0
    for index, group in enumerate(group_plan.groups, start=1):
        before = brep_status(
            group,
            max_sliver_area=options.max_sliver_area,
            small_edge_length=options.tolerance,
            detect_overlaps=options.remove_overlapping_faces,
            overlap_tolerance=options.tolerance,
            overlap_area_ratio=options.overlap_area_ratio,
        )
        healed_group, group_warnings, diagnostics = _run_heal_operations(group, options, before=before)
        warnings.extend(f"open-shell group {index}: {warning}" for warning in group_warnings)
        same_domain_faces_removed += diagnostics.same_domain_faces_removed
        same_domain_edges_removed += diagnostics.same_domain_edges_removed
        overlapping_faces_removed += diagnostics.overlapping_faces_removed
        builder.Add(compound, _ensure_shell_container(healed_group))
    return (
        compound,
        warnings,
        BrepHealDiagnostics(
            same_domain_faces_removed=same_domain_faces_removed,
            same_domain_edges_removed=same_domain_edges_removed,
            overlapping_faces_removed=overlapping_faces_removed,
            open_shell_groups=len(group_plan.groups),
            open_shell_grouped_shells=group_plan.shells,
            open_shell_grouped_faces=group_plan.faces,
        ),
    )


def _ensure_shell_container(shape: object) -> object:
    try:
        from OCP.BRep import BRep_Builder
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopoDS import TopoDS_Shell
    except ImportError:
        return shape

    if _count_subshapes(shape, TopAbs_SOLID, TopExp_Explorer) > 0:
        return shape
    if _count_subshapes(shape, TopAbs_SHELL, TopExp_Explorer) > 0:
        return shape
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    if not face_explorer.More():
        return shape
    shell = TopoDS_Shell()
    builder = BRep_Builder()
    builder.MakeShell(shell)
    while face_explorer.More():
        builder.Add(shell, face_explorer.Current())
        face_explorer.Next()
    return shell


def brep_status(
    shape: object,
    *,
    max_sliver_area: float = 0.0,
    small_edge_length: float = 0.0,
    detect_overlaps: bool = False,
    overlap_tolerance: float = 0.0,
    overlap_area_ratio: float = 0.995,
) -> BrepStatus:
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
        from OCP.TopExp import TopExp, TopExp_Explorer
    except ImportError:
        return BrepStatus(kind="unknown")

    solids = _count_subshapes(shape, TopAbs_SOLID, TopExp_Explorer)
    shells = _count_subshapes(shape, TopAbs_SHELL, TopExp_Explorer)
    wires = _count_subshapes(shape, TopAbs_WIRE, TopExp_Explorer)
    edges = _count_subshapes(shape, TopAbs_EDGE, TopExp_Explorer)
    faces = _count_subshapes(shape, TopAbs_FACE, TopExp_Explorer)
    free_edges = _count_free_edges(shape, TopAbs_EDGE, TopAbs_FACE, TopExp)
    small_edges = _count_small_edges(
        shape,
        TopAbs_EDGE,
        TopExp_Explorer,
        BRepGProp,
        GProp_GProps,
        max_length=small_edge_length,
    )
    open_shells = 0
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = shell_explorer.Current()
        if _shell_is_open(shell, BRepCheck_Analyzer, TopAbs_EDGE, TopAbs_FACE, TopExp):
            open_shells += 1
        shell_explorer.Next()
    sliver_faces = 0
    if max_sliver_area > 0.0:
        face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while face_explorer.More():
            face = face_explorer.Current()
            props = GProp_GProps()
            BRepGProp.SurfaceProperties_s(face, props)
            if float(props.Mass()) <= max_sliver_area:
                sliver_faces += 1
            face_explorer.Next()
    if solids:
        kind = "solid"
    elif shells and open_shells:
        kind = "open_surface"
    elif shells:
        kind = "shell"
    elif faces:
        kind = "surface"
    else:
        kind = "unknown"
    overlap_report = _FaceOverlapReport()
    if detect_overlaps and faces > 1:
        try:
            overlap_report = _detect_overlapping_faces(
                shape,
                tolerance=overlap_tolerance if overlap_tolerance > 0.0 else small_edge_length,
                overlap_area_ratio=overlap_area_ratio,
            )
        except Exception:
            overlap_report = _FaceOverlapReport()
    return BrepStatus(
        kind=kind,
        solids=solids,
        shells=shells,
        wires=wires,
        edges=edges,
        faces=faces,
        open_shells=open_shells,
        free_edges=free_edges,
        small_edges=small_edges,
        sliver_faces=sliver_faces,
        overlapping_face_pairs=overlap_report.pair_count,
        z_fighting_faces=overlap_report.face_count,
    )


def _shell_is_open(shell: object, analyzer_type: Any, edge_type: Any, face_type: Any, top_exp: Any) -> bool:
    return bool(not analyzer_type(shell).IsValid() or _count_free_edges(shell, edge_type, face_type, top_exp) > 0)


def _fix_shape(shape: object, options: BrepHealOptions) -> object:
    from OCP.ShapeFix import ShapeFix_Shape

    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(float(options.tolerance))
    fixer.Perform()
    return fixer.Shape()


def _sew_shape(shape: object, options: BrepHealOptions) -> object:
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing

    sewing = BRepBuilderAPI_Sewing(float(options.tolerance))
    sewing.Add(shape)
    sewing.Perform()
    return sewing.SewedShape()


def _unify_same_domain_shape(shape: object, options: BrepHealOptions) -> object:
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    unifier = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    unifier.SetLinearTolerance(float(options.tolerance))
    unifier.Build()
    return unifier.Shape()


def _remove_overlapping_faces_shape(shape: object, options: BrepHealOptions) -> tuple[object, _FaceOverlapReport]:
    from OCP.BRepTools import BRepTools_ReShape

    overlap_report = _detect_overlapping_faces(
        shape,
        tolerance=options.tolerance,
        overlap_area_ratio=options.overlap_area_ratio,
    )
    if not overlap_report.faces_to_remove:
        return shape, overlap_report
    reshaper = BRepTools_ReShape()
    for face in overlap_report.faces_to_remove:
        reshaper.Remove(face)
    return reshaper.Apply(shape), overlap_report


def _detect_overlapping_faces(
    shape: object,
    *,
    tolerance: float,
    overlap_area_ratio: float,
) -> _FaceOverlapReport:
    descriptors = _face_overlap_descriptors(shape, tolerance=tolerance)
    if len(descriptors) < 2:
        return _FaceOverlapReport()
    bbox_min = np.min(np.vstack([descriptor.bbox_min for descriptor in descriptors]), axis=0)
    bbox_max = np.max(np.vstack([descriptor.bbox_max for descriptor in descriptors]), axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    plane_tolerance = _overlap_plane_tolerance(tolerance, diagonal)
    overlapping_pairs = 0
    overlapping_face_ids: set[int] = set()
    faces_to_remove: dict[int, object] = {}
    for left_index, left in enumerate(descriptors[:-1]):
        for right in descriptors[left_index + 1 :]:
            if not _face_bboxes_overlap(left, right, tolerance=plane_tolerance):
                continue
            if not _faces_share_plane(left, right, tolerance=plane_tolerance):
                continue
            overlap_area = _projected_overlap_area(left, right, tolerance=plane_tolerance)
            if overlap_area <= 0.0:
                continue
            if overlap_area / max(min(left.area, right.area), 1e-15) < overlap_area_ratio:
                continue
            overlapping_pairs += 1
            overlapping_face_ids.add(left.index)
            overlapping_face_ids.add(right.index)
            remove = left if left.area < right.area else right
            faces_to_remove.setdefault(remove.index, remove.face)
    return _FaceOverlapReport(
        pair_count=overlapping_pairs,
        face_count=len(overlapping_face_ids),
        faces_to_remove=tuple(faces_to_remove.values()),
    )


def _face_overlap_descriptors(shape: object, *, tolerance: float) -> list[_FaceOverlapDescriptor]:
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IMeshTools import IMeshTools_Parameters
    from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    parameters = IMeshTools_Parameters()
    parameters.Deflection = max(float(tolerance), 1e-6)
    parameters.Angle = 0.5
    mesher = BRepMesh_IncrementalMesh(shape, parameters)
    mesher.Perform()

    descriptors: list[_FaceOverlapDescriptor] = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is not None:
            descriptor = _face_overlap_descriptor(
                face_index,
                face,
                triangulation,
                location.Transformation(),
                reversed_face=face.Orientation() == TopAbs_REVERSED,
            )
            if descriptor is not None:
                descriptors.append(descriptor)
        face_index += 1
        explorer.Next()
    return descriptors


def _face_overlap_descriptor(
    face_index: int,
    face: object,
    triangulation: Any,
    transform: Any,
    *,
    reversed_face: bool,
) -> _FaceOverlapDescriptor | None:
    node_count = int(triangulation.NbNodes())
    triangle_count = int(triangulation.NbTriangles())
    if node_count < 3 or triangle_count < 1:
        return None

    nodes = triangulation.MapNodeArray()
    node_lower = int(nodes.Lower())
    points = _transformed_occt_nodes(nodes, node_lower, node_count, transform)

    triangles = triangulation.MapTriangleArray()
    triangle_lower = int(triangles.Lower())
    face_triangles = _triangulation_faces(triangles, triangle_lower, triangle_count)
    if reversed_face:
        face_triangles = face_triangles[:, ::-1]
    face_triangles -= 1

    triangle_points = points[face_triangles]
    crosses = np.cross(triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0])
    double_areas = np.linalg.norm(crosses, axis=1)
    triangle_areas = double_areas * 0.5
    area = float(np.sum(triangle_areas))
    if area <= 1e-15:
        return None
    weighted_normal = np.sum(crosses, axis=0)
    normal_length = float(np.linalg.norm(weighted_normal))
    if normal_length <= 1e-15:
        return None
    normal = weighted_normal / normal_length
    plane_offset = float(np.dot(normal, points[face_triangles[0, 0]]))
    return _FaceOverlapDescriptor(
        index=face_index,
        face=face,
        points=points,
        triangles=face_triangles,
        triangle_areas=triangle_areas,
        area=area,
        normal=normal,
        plane_offset=plane_offset,
        bbox_min=np.min(points, axis=0),
        bbox_max=np.max(points, axis=0),
    )


def _overlap_plane_tolerance(tolerance: float, diagonal: float) -> float:
    if diagonal <= 0.0:
        return max(float(tolerance), 1e-9)
    return max(min(float(tolerance), diagonal * 1e-5), 1e-9)


def _face_bboxes_overlap(left: _FaceOverlapDescriptor, right: _FaceOverlapDescriptor, *, tolerance: float) -> bool:
    return bool(
        np.all(left.bbox_min <= right.bbox_max + tolerance) and np.all(right.bbox_min <= left.bbox_max + tolerance)
    )


def _faces_share_plane(left: _FaceOverlapDescriptor, right: _FaceOverlapDescriptor, *, tolerance: float) -> bool:
    dot = float(np.dot(left.normal, right.normal))
    if abs(dot) < math.cos(math.radians(0.25)):
        return False
    right_offset = right.plane_offset if dot >= 0.0 else -right.plane_offset
    return abs(left.plane_offset - right_offset) <= tolerance


def _projected_overlap_area(
    left: _FaceOverlapDescriptor,
    right: _FaceOverlapDescriptor,
    *,
    tolerance: float,
) -> float:
    u_axis, v_axis = _projection_axes(left.normal)
    left_projected = _project_points(left.points, u_axis, v_axis)
    right_projected = _project_points(right.points, u_axis, v_axis)
    left_triangles = left_projected[left.triangles]
    right_triangles = right_projected[right.triangles]
    left_mins = np.min(left_triangles, axis=1)
    left_maxs = np.max(left_triangles, axis=1)
    right_mins = np.min(right_triangles, axis=1)
    right_maxs = np.max(right_triangles, axis=1)
    area = 0.0
    area_tolerance = max(tolerance * tolerance, 1e-14)
    for left_index, left_triangle in enumerate(left_triangles):
        candidates = np.where(
            np.all(left_mins[left_index] <= right_maxs + tolerance, axis=1)
            & np.all(right_mins <= left_maxs[left_index] + tolerance, axis=1)
        )[0]
        for right_index in candidates:
            area += _triangle_overlap_area_2d(
                left_triangle,
                right_triangles[right_index],
                tolerance=area_tolerance,
            )
    return area


def _projection_axes(normal: FloatArray) -> tuple[FloatArray, FloatArray]:
    axis = np.zeros(3, dtype=np.float64)
    axis[int(np.argmin(np.abs(normal)))] = 1.0
    u_axis = np.cross(normal, axis)
    u_length = float(np.linalg.norm(u_axis))
    if u_length <= 1e-15:
        u_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u_axis /= u_length
    v_axis = np.cross(normal, u_axis)
    v_axis /= max(float(np.linalg.norm(v_axis)), 1e-15)
    return u_axis, v_axis


def _project_points(points: FloatArray, u_axis: FloatArray, v_axis: FloatArray) -> FloatArray:
    return np.column_stack((points @ u_axis, points @ v_axis)).astype(np.float64)


def _count_subshapes(shape: object, shape_type: Any, explorer_type: Any) -> int:
    count = 0
    explorer = explorer_type(shape, shape_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def _count_free_edges(shape: object, edge_type: Any, face_type: Any, top_exp: Any) -> int:
    try:
        from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

        edge_faces = TopTools_IndexedDataMapOfShapeListOfShape()
        top_exp.MapShapesAndAncestors_s(shape, edge_type, face_type, edge_faces)
        count = 0
        for index in range(1, edge_faces.Extent() + 1):
            if edge_faces.FindFromIndex(index).Extent() <= 1:
                count += 1
        return count
    except Exception:
        return 0


def _count_small_edges(
    shape: object,
    edge_type: Any,
    explorer_type: Any,
    brep_gprop: Any,
    props_type: Any,
    *,
    max_length: float,
) -> int:
    if max_length <= 0.0:
        return 0
    count = 0
    explorer = explorer_type(shape, edge_type)
    while explorer.More():
        edge = explorer.Current()
        props = props_type()
        try:
            brep_gprop.LinearProperties_s(edge, props)
            if float(props.Mass()) <= max_length:
                count += 1
        except Exception:
            pass
        explorer.Next()
    return count


def _add_topology_warnings(asset: Asset, part_name: str, status: BrepStatus, options: BrepHealOptions) -> None:
    if status.open_shells > 0:
        asset.report.add_warning(f"{part_name}: BREP healing left {status.open_shells} open shell(s)")
    if status.free_edges > 0:
        asset.report.add_warning(f"{part_name}: BREP healing left {status.free_edges} free/unstitched edge(s)")
    if status.small_edges > 0:
        asset.report.add_warning(
            f"{part_name}: BREP healing left {status.small_edges} edge(s) at or below tolerance {options.tolerance:g}"
        )
    if status.overlapping_face_pairs > 0:
        asset.report.add_warning(
            f"{part_name}: BREP healing left {status.overlapping_face_pairs} overlapping/z-fighting face pair(s)"
        )


def _operation_summary(options: BrepHealOptions) -> str:
    operations: list[str] = []
    if options.group_open_shells:
        operations.append("group_open_shells")
    if options.fix_edges:
        operations.append("fix_edges")
    if options.unify_tolerances:
        operations.append("unify_tolerances")
    if options.sew_faces:
        operations.append("sew_faces")
    if options.unify_same_domain:
        operations.append("unify_same_domain")
    if options.remove_overlapping_faces:
        operations.append("remove_overlapping_faces")
    if options.remove_sliver_faces:
        operations.append("remove_sliver_faces")
    return ",".join(operations)


def _open_shell_grouping_status(options: BrepHealOptions, diagnostics: BrepHealDiagnostics) -> str:
    if not options.group_open_shells:
        return "disabled"
    return "grouped" if diagnostics.open_shell_groups else "not_applicable"


def _tolerance_policy_metadata(prefix: str, policy: Mapping[str, object] | None) -> dict[str, object]:
    if policy is None:
        return {}
    metadata: dict[str, object] = {
        f"{prefix}_coordinate_space": str(policy["coordinate_space"]),
        f"{prefix}_effective_units": str(policy["effective_units"]),
        f"{prefix}_effective_meters_per_unit": _format_metadata_float(policy["effective_meters_per_unit"]),
        f"{prefix}_source_units": str(policy["source_units"]),
        f"{prefix}_source_meters_per_unit": _format_metadata_float(policy["source_meters_per_unit"]),
        f"{prefix}_target_units": str(policy["target_units"]),
        f"{prefix}_target_meters_per_unit": _format_metadata_float(policy["target_meters_per_unit"]),
    }
    for key, value in policy.items():
        if key.endswith("_meters") or key.endswith("_square_meters"):
            metadata[f"{prefix}_{key}"] = _format_metadata_float(value)
    operations = policy.get("operations")
    if isinstance(operations, Mapping):
        for key, value in operations.items():
            metadata[f"{prefix}_{key}"] = str(value)
    return metadata


def _format_metadata_float(value: object) -> str:
    if isinstance(value, bool):
        numeric = 0.0
    elif isinstance(value, int | float):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            numeric = 0.0
    else:
        numeric = 0.0
    return f"{numeric:.9g}"
