from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fascat.asset import Asset, Node, Part
from fascat.io._import_base import (
    _annotate_mirrored_transforms,
    _asset_metadata,
    _construction_curve_policy,
    _empty_mirrored_transform_summary,
    _ImportCleanupStats,
    _loaded_representation_report,
    _metadata_count,
    _mirrored_transform_warnings,
    _PartIndex,
    _space_normalization,
    _SpaceNormalization,
    _stable_id,
    _StepHeaderInfo,
)
from fascat.io._suffixes import STEP_SUFFIXES as _STEP_SUFFIXES
from fascat.io.step.materials import (
    _apply_material_libraries_to_materials,
    _empty_material_library_binding_summary,
    _empty_material_library_summary,
    _extract_material_libraries,
    _material_library_import_state,
)
from fascat.io.step.pmi import (
    _empty_pmi_semantic_graph_summary,
    _extract_step_pmi_annotations,
    _extract_step_pmi_semantic_graph,
)
from fascat.io.step.records import _MAX_STEP_SCAN_BYTES, _step_header_info, _step_scan_capped
from fascat.io.step.textures import _attach_source_textures_to_materials, _extract_source_textures
from fascat.io.step.variants import (
    _apply_step_design_variant_selection,
    _empty_design_variant_summary,
    _extract_step_design_variants,
)
from fascat.io.step.xde import _build_node, _free_shape_labels, _read_xde_document
from fascat.material import Material
from fascat.options import StepReadOptions
from fascat.report import Report, timed_step


def _read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> Asset:
    if not source.exists():
        raise FileNotFoundError(f"missing STEP file: {source}")
    if source.suffix.lower() not in _STEP_SUFFIXES:
        raise ValueError(f"unsupported STEP extension: {source.suffix or '<none>'}")

    header_info = _step_header_info(source)
    cleanup = _ImportCleanupStats()
    with timed_step() as timer:
        document, shape_tool, color_tool, vis_material_tool, unit_name, meters_per_unit = _read_xde_document(
            source, options
        )
        space = _space_normalization(unit_name, meters_per_unit, options)
        free_labels = _free_shape_labels(shape_tool)
        root = Node(
            id=_stable_id("node", f"{source_identity}:root"),
            name=source.stem,
            transform=space.transform,
            metadata={
                "source": str(source),
                "source_identity": source_identity,
                "space_normalization": space.metadata(),
            },
        )
        parts: dict[str, Part] = {}
        part_index: _PartIndex = {}
        materials: dict[str, Material] = {}
        for index, label in enumerate(free_labels, start=1):
            root.children.append(
                _build_node(
                    label,
                    f"root/{index}",
                    source_identity,
                    shape_tool,
                    color_tool,
                    vis_material_tool,
                    parts,
                    part_index,
                    materials,
                    options,
                    cleanup,
                )
            )
        design_variants = _extract_step_design_variants(source, options)
        design_variant_selection = _apply_step_design_variant_selection(
            root, parts, materials, design_variants, options
        )
        source_textures = _extract_source_textures(source, source_identity, options)
        texture_binding_summary = _attach_source_textures_to_materials(materials, source_textures.images)
        material_libraries = _extract_material_libraries(source, source_identity, options)
        material_library_binding_summary = _apply_material_libraries_to_materials(materials, material_libraries)
        images = {**source_textures.images, **material_libraries.images}
        pmi = _extract_step_pmi_annotations(source, options)
        pmi_semantic_graph = _extract_step_pmi_semantic_graph(source, options)
        mirrored_transforms = _annotate_mirrored_transforms(root)

    report = Report(source_path=str(source))
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        images=images,
        units=space.target_units,
        meters_per_unit=space.target_meters_per_unit,
        up_axis=cast(Any, space.target_up_axis),
        source_path=source,
        metadata=_asset_metadata(
            source,
            source_identity,
            options,
            header_info,
            cleanup,
            space,
            pmi_count=len(pmi),
            design_variant_summary=design_variants.summary,
            mirrored_transform_summary=mirrored_transforms,
        ),
        pmi=pmi,
        report=report,
    )
    asset.report.input_stats = asset.stats()
    metadata_count = _metadata_count(asset)
    unsupported_pmi_count = _unsupported_pmi_count(options, header_info, pmi_count=len(asset.pmi))
    import_decisions = _import_decisions(
        options,
        header_info,
        pmi_count=len(asset.pmi),
        unsupported_pmi_count=unsupported_pmi_count,
        cleanup=cleanup,
        space=space,
        source_texture_summary=source_textures.summary,
        texture_binding_summary=texture_binding_summary,
        material_library_summary=material_libraries.summary,
        material_library_binding_summary=material_library_binding_summary,
        pmi_semantic_graph_summary=pmi_semantic_graph.summary,
        design_variant_summary=design_variants.summary,
        design_variant_selection_summary=design_variant_selection.to_dict(),
        mirrored_transform_summary=mirrored_transforms,
    )
    loaded_representations = _loaded_representation_report(asset)
    if asset.metadata:
        asset.metadata["import_decisions"] = import_decisions
        asset.metadata["import_representation_summary"] = loaded_representations["summary"]
        asset.metadata["source_texture_import"] = source_textures.summary
        asset.metadata["source_texture_bindings"] = texture_binding_summary
        asset.metadata["material_library_import"] = material_libraries.summary
        asset.metadata["material_library_bindings"] = material_library_binding_summary
        asset.metadata["pmi_semantic_graph"] = pmi_semantic_graph.to_dict()
        asset.metadata["design_variant_import"] = design_variants.summary
        asset.metadata["design_variant_selection"] = design_variant_selection.to_dict()
        asset.metadata["mirrored_transforms"] = mirrored_transforms
        if design_variants.records:
            asset.metadata["design_variants"] = [record.to_dict() for record in design_variants.records]
    import_warnings = [
        *_import_warnings(
            options,
            header_info,
            unsupported_pmi_count,
            design_variant_count=design_variants.summary["records"],
            scan_capped=_step_scan_capped(source),
        ),
        *source_textures.warnings,
        *material_libraries.warnings,
        *pmi_semantic_graph.warnings,
        *design_variants.warnings,
        *design_variant_selection.warnings,
        *_mirrored_transform_warnings(mirrored_transforms),
    ]
    for warning in import_warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "STEP",
            "backend": "OCP",
            "read_options": options.to_dict(),
            "metadata_count": metadata_count,
            "pmi_count": len(asset.pmi),
            "unsupported_pmi_count": unsupported_pmi_count,
            "pmi_semantic_graph": pmi_semantic_graph.to_dict(),
            "design_variants": design_variants.to_dict(),
            "design_variant_selection": design_variant_selection.to_dict(),
            "pmi_schema": header_info.schema,
            "pmi_present": header_info.pmi_present,
            "cleanup": cleanup.to_dict(),
            "space_normalization": space.metadata(),
            "source_textures": source_textures.summary,
            "source_texture_bindings": texture_binding_summary,
            "material_libraries": material_libraries.summary,
            "material_library_bindings": material_library_binding_summary,
            "import_decisions": import_decisions,
            "loaded_representations": loaded_representations,
            "mirrored_transforms": mirrored_transforms,
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
        warnings=import_warnings,
    )
    _ = document
    return asset


def _unsupported_pmi_count(options: StepReadOptions, header_info: _StepHeaderInfo, *, pmi_count: int) -> int:
    if not options.pmi or not header_info.pmi_present or pmi_count:
        return 0
    return 1


def _import_warnings(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    unsupported_pmi_count: int,
    *,
    design_variant_count: int = 0,
    scan_capped: bool = False,
) -> list[str]:
    warnings: list[str] = []
    if scan_capped:
        warnings.append(
            "auxiliary STEP text scans skipped: file exceeds "
            f"{_MAX_STEP_SCAN_BYTES} bytes (textual PMI, design variants, "
            "source textures, material libraries)"
        )
    if options.pmi and unsupported_pmi_count:
        warnings.append(
            "STEP file advertises AP242 PMI, but no supported typed PMI entities were extracted; annotations are omitted"
        )
    if (options.design_variants or options.design_variant_selection) and design_variant_count == 0:
        warnings.append(
            "STEP design variant import was requested, but no supported design variant records were detected"
        )
    return warnings


def _import_decisions(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    *,
    pmi_count: int,
    unsupported_pmi_count: int,
    cleanup: _ImportCleanupStats,
    space: _SpaceNormalization,
    source_texture_summary: dict[str, int] | None = None,
    texture_binding_summary: dict[str, int] | None = None,
    material_library_summary: dict[str, int] | None = None,
    material_library_binding_summary: dict[str, int] | None = None,
    pmi_semantic_graph_summary: dict[str, int] | None = None,
    design_variant_summary: dict[str, int] | None = None,
    design_variant_selection_summary: dict[str, object] | None = None,
    mirrored_transform_summary: dict[str, int] | None = None,
) -> dict[str, object]:
    cleanup_counts = cleanup.to_dict()
    texture_summary = source_texture_summary or {
        "references": 0,
        "resolved": 0,
        "missing": 0,
        "unsupported": 0,
        "unreadable": 0,
    }
    binding_summary = texture_binding_summary or {"bound_images": 0, "bound_materials": 0, "unbound_images": 0}
    library_summary = material_library_summary or _empty_material_library_summary()
    library_binding_summary = material_library_binding_summary or _empty_material_library_binding_summary()
    pmi_graph_summary = {**_empty_pmi_semantic_graph_summary(), **(pmi_semantic_graph_summary or {})}
    variant_summary = {**_empty_design_variant_summary(), **(design_variant_summary or {})}
    mirrored_summary = {**_empty_mirrored_transform_summary(), **(mirrored_transform_summary or {})}
    mirrored_detected = mirrored_summary["local_mirrored_nodes"] > 0 or mirrored_summary["world_mirrored_nodes"] > 0
    return {
        "metadata": _import_decision(
            requested=options.metadata,
            effective=options.metadata,
            state="honored" if options.metadata else "disabled",
        ),
        "product_metadata": _import_decision(
            requested=options.product_metadata,
            effective=options.product_metadata,
            state="honored" if options.product_metadata else "disabled",
        ),
        "properties": _import_decision(
            requested=options.properties,
            effective=options.metadata or options.properties,
            state="honored" if options.properties else "disabled",
            detail="OCP metadata transfer is enabled when general metadata or properties are requested",
        ),
        "layers": _import_decision(
            requested=options.layers,
            effective=False,
            state="unsupported" if options.layers else "disabled",
            detail="normalized STEP layer records are not exposed by the current importer",
        ),
        "validation_properties": _import_decision(
            requested=options.validation_properties,
            effective=options.validation_properties,
            state="approximated" if options.validation_properties else "disabled",
            detail="source topology counts are derived after transfer; typed STEP validation properties are not extracted",
        ),
        "pmi": _pmi_import_decision(options, header_info, pmi_count, unsupported_pmi_count, pmi_graph_summary),
        "design_variants": _design_variant_import_decision(options, variant_summary, design_variant_selection_summary),
        "existing_meshes": _import_decision(
            requested=options.existing_meshes,
            effective=options.existing_meshes,
            state="backend_default" if options.existing_meshes else "disabled",
            detail="OCP transfer keeps BREP source shapes; reusable source tessellation payloads are not detected separately yet",
        ),
        "multi_file": _import_decision(
            requested=options.multi_file,
            effective=False,
            state="unsupported" if options.multi_file else "disabled",
            detail=(
                "single-file STEP import does not resolve external-reference graphs; "
                "use read_step(..., multi_file=True) or read_step_many(...) for multi-file imports"
            ),
        ),
        "source_textures": _import_decision(
            requested=options.source_textures,
            effective=options.source_textures and texture_summary["resolved"] > 0,
            state=(
                "disabled"
                if not options.source_textures
                else "honored"
                if texture_summary["resolved"] > 0 or texture_summary["references"] == 0
                else "missing_sources"
            ),
            detail=(
                "external source texture references are scanned from STEP string records and resolved "
                "against the source directory plus source_texture_search_paths"
            ),
            counts={
                "references": texture_summary["references"],
                "resolved": texture_summary["resolved"],
                "missing": texture_summary["missing"],
                "unsupported": texture_summary["unsupported"],
                "unreadable": texture_summary["unreadable"],
                "bound_images": binding_summary["bound_images"],
                "bound_materials": binding_summary["bound_materials"],
                "unbound_images": binding_summary["unbound_images"],
            },
        ),
        "material_library_mapping": _import_decision(
            requested=options.material_library_mapping,
            effective=options.material_library_mapping
            and (library_binding_summary["applied_materials"] > 0 or library_summary["references"] == 0),
            state=_material_library_import_state(options, library_summary, library_binding_summary),
            detail=(
                "known CAD material names and supported JSON/MTL sidecar material libraries are mapped "
                "to PBR factors and texture slots when present"
            ),
            counts={**library_summary, **library_binding_summary},
        ),
        "delete_free_vertices": _import_decision(
            requested=options.delete_free_vertices,
            effective=options.delete_free_vertices,
            state="honored" if options.delete_free_vertices else "disabled",
            counts={
                "deleted_parts": cleanup_counts["deleted_free_vertex_parts"],
                "deleted_vertices": cleanup_counts["deleted_free_vertices"],
            },
        ),
        "delete_lines": _import_decision(
            requested=options.delete_lines,
            effective=options.delete_lines,
            state="honored" if options.delete_lines else "disabled",
            counts={
                "deleted_parts": cleanup_counts["deleted_line_parts"],
                "deleted_edges": cleanup_counts["deleted_line_edges"],
                "deleted_vertices": cleanup_counts["deleted_line_vertices"],
            },
        ),
        "construction_curves": _construction_curve_import_decision(options, cleanup_counts),
        "mirrored_transforms": _import_decision(
            requested=True,
            effective=mirrored_detected,
            state="detected" if mirrored_detected else "not_present",
            detail=(
                "negative-determinant transforms are preserved and reported; downstream normal/winding "
                "compensation may be required"
            ),
            counts=mirrored_summary,
        ),
        "space_normalization": _import_decision(
            requested={
                "source_units": options.source_units,
                "source_meters_per_unit": options.source_meters_per_unit,
                "source_up_axis": options.source_up_axis,
                "source_handedness": options.source_handedness,
                "target_units": options.target_units,
                "target_meters_per_unit": options.target_meters_per_unit,
                "target_up_axis": options.target_up_axis,
                "target_handedness": options.target_handedness,
            },
            effective=space.metadata(),
            state="honored" if space.changed else "backend_default",
        ),
    }


def _import_decision(
    *,
    requested: object,
    effective: object,
    state: str,
    detail: str | None = None,
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    decision: dict[str, object] = {
        "requested": requested,
        "effective": effective,
        "state": state,
    }
    if detail:
        decision["detail"] = detail
    if counts is not None:
        decision["counts"] = counts
    return decision


def _pmi_import_decision(
    options: StepReadOptions,
    header_info: _StepHeaderInfo,
    pmi_count: int,
    unsupported_pmi_count: int,
    semantic_graph_summary: dict[str, int],
) -> dict[str, object]:
    counts = {
        "imported": pmi_count,
        "unsupported": unsupported_pmi_count,
        "semantic_graph_nodes": semantic_graph_summary["nodes"],
        "semantic_graph_edges": semantic_graph_summary["edges"],
        "semantic_graph_missing_references": semantic_graph_summary["missing_references"],
        "semantic_graph_cycles": semantic_graph_summary.get("cycles", 0),
    }
    if not options.pmi:
        return _import_decision(requested=False, effective=False, state="disabled")
    if pmi_count:
        return _import_decision(
            requested=True,
            effective=True,
            state="honored",
            detail="common STEP AP242 PMI entities were extracted into typed metadata annotations and a semantic reference graph",
            counts=counts,
        )
    if unsupported_pmi_count:
        return _import_decision(
            requested=True,
            effective=False,
            state="unsupported",
            detail="STEP AP242 PMI markers were detected, but typed PMI entity extraction is not implemented",
            counts=counts,
        )
    if not header_info.pmi_present:
        return _import_decision(
            requested=True,
            effective=False,
            state="not_present",
            detail="PMI import was requested, but the STEP header did not advertise PMI content",
            counts=counts,
        )
    return _import_decision(
        requested=True,
        effective=True,
        state="honored",
        counts=counts,
    )


def _design_variant_import_decision(
    options: StepReadOptions,
    summary: dict[str, int],
    selection_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    counts = {**_empty_design_variant_summary(), **summary}
    if selection_summary:
        for key in ("before_nodes", "after_nodes", "before_parts", "after_parts", "removed_nodes", "removed_parts"):
            value = selection_summary.get(key)
            if isinstance(value, int):
                counts[f"selection_{key}"] = value
        matched = selection_summary.get("matched_records")
        requested = selection_summary.get("requested")
        if isinstance(matched, list):
            counts["selection_matched_records"] = len(matched)
        if isinstance(requested, list):
            counts["selection_requested"] = len(requested)
    if not options.design_variants and not options.design_variant_selection:
        return _import_decision(requested=False, effective=False, state="disabled")
    if counts["records"] == 0:
        return _import_decision(
            requested=True if options.design_variants else list(options.design_variant_selection),
            effective=False,
            state="not_present",
            detail="design variant import was requested, but no supported STEP configuration records were detected",
            counts=counts,
        )
    if options.design_variant_selection and selection_summary:
        status = str(selection_summary.get("status", "not_requested"))
        if status == "applied":
            detail = (
                "supported STEP configuration/design-variant records were scanned and the imported geometry tree "
                "was filtered using selected variant record labels, effectivity values/ranges, simple condition "
                "records, and referenced STEP labels"
            )
        else:
            detail = (
                "supported STEP configuration/design-variant records were scanned, but selected variant geometry "
                "could not be matched by the current name/reference-based selector"
            )
        return _import_decision(
            requested=list(options.design_variant_selection),
            effective=status == "applied",
            state="approximated" if status == "applied" else "unmatched",
            detail=detail,
            counts=counts,
        )
    return _import_decision(
        requested=True,
        effective=True,
        state="approximated",
        detail=(
            "supported STEP configuration/design-variant records are reported as metadata; "
            "pass design_variant_selection to apply name/reference/condition-based geometry filtering"
        ),
        counts=counts,
    )


def _construction_curve_import_decision(options: StepReadOptions, cleanup_counts: dict[str, int]) -> dict[str, object]:
    policy = _construction_curve_policy(options)
    counts = {
        "preserved_parts": cleanup_counts["construction_line_parts"],
        "deleted_parts": cleanup_counts["deleted_line_parts"],
        "deleted_edges": cleanup_counts["deleted_line_edges"],
        "deleted_vertices": cleanup_counts["deleted_line_vertices"],
    }
    if policy == "delete":
        return _import_decision(
            requested=policy,
            effective=counts["deleted_parts"] > 0,
            state="honored" if counts["deleted_parts"] > 0 else "not_present",
            detail="construction-only line shapes are deleted during import",
            counts=counts,
        )
    if policy == "tessellate_tubes":
        return _import_decision(
            requested=policy,
            effective=counts["preserved_parts"] > 0,
            state="approximated" if counts["preserved_parts"] > 0 else "not_present",
            detail=(
                "construction-only line shapes are preserved with tube tessellation metadata; "
                "the tessellation step converts them to triangle tubes"
            ),
            counts=counts,
        )
    return _import_decision(
        requested=policy,
        effective=counts["preserved_parts"] > 0,
        state="honored" if counts["preserved_parts"] > 0 else "not_present",
        detail="construction-only line shapes are preserved as source-shape metadata without mesh geometry",
        counts=counts,
    )
