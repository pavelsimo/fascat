from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fascat.asset import Asset
from fascat.options import BrepHealOptions


@dataclass(frozen=True)
class BrepStatus:
    kind: str
    solids: int = 0
    shells: int = 0
    faces: int = 0
    open_shells: int = 0
    sliver_faces: int = 0

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": self.kind,
            "solids": self.solids,
            "shells": self.shells,
            "faces": self.faces,
            "open_shells": self.open_shells,
            "sliver_faces": self.sliver_faces,
        }


def heal_brep_asset(asset: Asset, options: BrepHealOptions, *, selected_part_ids: set[str] | None = None) -> Asset:
    result = asset.copy(keep_source=True)
    for part in result.parts.values():
        if selected_part_ids is not None and part.id not in selected_part_ids:
            continue
        if part.source_shape is None:
            result.report.add_warning(f"part has no source shape and cannot be BREP healed: {part.name}")
            continue
        healed_shape, before, after, warnings = heal_shape(part.source_shape, options)
        for warning in warnings:
            result.report.add_warning(f"{part.name}: {warning}")
        if options.fail_on_open_shells and after.open_shells > 0:
            raise RuntimeError(f"BREP healing left open shells in part: {part.name}")
        part.source_shape = healed_shape
        part.metadata = {
            **part.metadata,
            "brep_kind": after.kind,
            "brep_solids": str(after.solids),
            "brep_shells": str(after.shells),
            "brep_faces": str(after.faces),
            "brep_open_shells": str(after.open_shells),
            "brep_sliver_faces": str(after.sliver_faces),
            "brep_before": str(before.to_dict()),
            "brep_after": str(after.to_dict()),
        }
    return result


def heal_shape(shape: object, options: BrepHealOptions) -> tuple[object, BrepStatus, BrepStatus, list[str]]:
    before = brep_status(shape, max_sliver_area=options.max_sliver_area)
    healed = shape
    warnings: list[str] = []
    try:
        if options.fix_edges or options.unify_tolerances:
            healed = _fix_shape(healed, options)
        if options.sew_faces:
            healed = _sew_shape(healed, options)
        if options.remove_sliver_faces and before.sliver_faces:
            warnings.append("sliver face removal is not supported by the current BREP backend")
    except Exception as exc:
        warnings.append(f"BREP healer skipped unsupported operation: {exc}")
        healed = shape
    after = brep_status(healed, max_sliver_area=options.max_sliver_area)
    return healed, before, after, warnings


def brep_status(shape: object, *, max_sliver_area: float = 0.0) -> BrepStatus:
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.BRepGProp import BRepGProp
        from OCP.GProp import GProp_GProps
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL, TopAbs_SOLID
        from OCP.TopExp import TopExp_Explorer
    except ImportError:
        return BrepStatus(kind="unknown")

    solids = _count_subshapes(shape, TopAbs_SOLID, TopExp_Explorer)
    shells = _count_subshapes(shape, TopAbs_SHELL, TopExp_Explorer)
    faces = _count_subshapes(shape, TopAbs_FACE, TopExp_Explorer)
    open_shells = 0
    shell_explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    while shell_explorer.More():
        shell = shell_explorer.Current()
        if not BRepCheck_Analyzer(shell).IsValid():
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
    return BrepStatus(
        kind=kind, solids=solids, shells=shells, faces=faces, open_shells=open_shells, sliver_faces=sliver_faces
    )


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


def _count_subshapes(shape: object, shape_type: Any, explorer_type: Any) -> int:
    count = 0
    explorer = explorer_type(shape, shape_type)
    while explorer.More():
        count += 1
        explorer.Next()
    return count
