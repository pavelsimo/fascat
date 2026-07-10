from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast
from urllib.parse import unquote, urlparse

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.image import ImageResource
from fascat.io._import_base import _stable_id
from fascat.io._suffixes import STEP_SUFFIXES as _STEP_SUFFIXES
from fascat.io.step import single
from fascat.io.step.records import (
    _MAX_STEP_SCAN_BYTES,
    _decode_step_string,
    _read_step_scan_text,
    _step_scan_capped,
)
from fascat.io.step.single import _import_decision
from fascat.material import Material
from fascat.metadata import Metadata, PmiAnnotation
from fascat.options import StepReadOptions
from fascat.report import Report, timed_step

_STEP_EXTERNAL_REF_RE = re.compile(r"'([^']+\.(?:step|stp)(?:[#?][^']*)?)'", re.IGNORECASE)


@dataclass(frozen=True)
class _StepMemberImport:
    index: int
    source: Path
    namespace: str
    asset: Asset


@dataclass(frozen=True)
class _StepExternalReferenceRecord:
    source: Path
    reference: str
    status: str
    resolved: Path | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "source": str(self.source),
            "reference": self.reference,
            "status": self.status,
        }
        if self.resolved is not None:
            data["resolved"] = str(self.resolved)
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class _StepExternalReferenceGraph:
    root: Path
    sources: list[Path]
    member_sources: list[Path]
    records: list[_StepExternalReferenceRecord]
    warnings: list[str]

    @property
    def has_references(self) -> bool:
        return bool(self.records)

    def summary(self) -> dict[str, int]:
        resolved_sources = {str(source.resolve()) for source in self.sources[1:]}
        return {
            "references": len(self.records),
            "resolved": sum(1 for record in self.records if record.status == "resolved"),
            "missing": sum(1 for record in self.records if record.status == "missing"),
            "unsupported": sum(1 for record in self.records if record.status == "unsupported"),
            "cycles": sum(1 for record in self.records if record.status == "cycle"),
            "sources": len(self.sources),
            "resolved_sources": len(resolved_sources),
            "member_sources": len(self.member_sources),
            "resolved_occurrences": max(0, len(self.member_sources) - 1),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "sources": [str(source) for source in self.sources],
            "member_sources": [str(source) for source in self.member_sources],
            "summary": self.summary(),
            "references": [record.to_dict() for record in self.records],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class _StepNamespaceMaps:
    nodes: dict[str, str]
    parts: dict[str, str]
    materials: dict[str, str]
    images: dict[str, str]


def _read_step_many(
    paths: Iterable[str | Path],
    *,
    options: StepReadOptions | None = None,
    continue_on_error: bool = False,
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> Asset:
    sources = [Path(path) for path in paths]
    if not sources:
        raise ValueError("read_step_many requires at least one STEP file")

    opts = options or StepReadOptions()
    member_options = replace(opts, multi_file=False)
    members: list[_StepMemberImport] = []
    failed_members: list[dict[str, object]] = []
    warnings: list[str] = []
    with timed_step() as timer:
        for index, source in enumerate(sources, start=1):
            namespace = _multi_file_namespace(index, source)
            try:
                member_asset = single._read_step_path(
                    source,
                    source_identity=str(source.resolve()),
                    options=member_options,
                )
            except Exception as exc:
                warning = f"multi-file STEP member {index} ({source}) failed to import: {exc}"
                failed_members.append(
                    {
                        "index": index,
                        "source": str(source),
                        "namespace": namespace,
                        "status": "error",
                        "error": str(exc),
                    }
                )
                if not continue_on_error:
                    raise
                warnings.append(warning)
                continue
            member_warnings = [
                f"multi-file STEP member {index} ({source}): {warning}" for warning in member_asset.report.warnings
            ]
            warnings.extend(member_warnings)
            members.append(_StepMemberImport(index=index, source=source, namespace=namespace, asset=member_asset))

        if not members:
            raise RuntimeError("multi-file STEP import did not import any members")

    return _merge_step_member_assets(
        members,
        failed_members=failed_members,
        options=opts,
        sources=sources,
        duration=timer.duration,
        warnings=warnings,
        reference_graph=reference_graph,
    )


def _read_step_with_external_references(source: Path, options: StepReadOptions) -> Asset:
    graph = _resolve_step_external_reference_graph(source)
    member_options = replace(options, multi_file=False)
    if len(graph.member_sources) > 1:
        return _read_step_many(graph.member_sources, options=options, reference_graph=graph)

    asset = single._read_step_path(source, source_identity=str(source.resolve()), options=member_options)
    _attach_step_external_reference_graph(asset, graph, options)
    return asset


def _resolve_step_external_reference_graph(source: Path) -> _StepExternalReferenceGraph:
    root = source
    root_key = str(root.resolve())
    sources = [root]
    member_sources = [root]
    seen_sources = {root_key}
    reference_cache: dict[str, list[str]] = {}
    records: list[_StepExternalReferenceRecord] = []
    warnings: list[str] = []
    queue: deque[tuple[Path, tuple[str, ...]]] = deque([(root, (root_key,))])

    while queue:
        current, path_keys = queue.popleft()
        if _step_scan_capped(current):
            warnings.append(f"external reference scan skipped: {current} exceeds {_MAX_STEP_SCAN_BYTES} bytes")
            continue
        current_key = str(current.resolve())
        references = reference_cache.get(current_key)
        if references is None:
            references = _step_external_references(current)
            reference_cache[current_key] = references
        for reference in references:
            cleaned, unsupported_reason = _clean_step_external_reference(reference)
            if cleaned is None:
                records.append(
                    _StepExternalReferenceRecord(
                        source=current,
                        reference=reference,
                        status="unsupported",
                        reason=unsupported_reason or "unsupported external STEP reference",
                    )
                )
                warnings.append(
                    f"STEP external reference is unsupported: {reference} "
                    f"(referenced by {current}; {unsupported_reason})"
                )
                continue

            resolved = _resolve_step_external_reference(cleaned, current)
            if resolved is None:
                records.append(
                    _StepExternalReferenceRecord(
                        source=current,
                        reference=reference,
                        status="missing",
                        reason="not found relative to the referencing STEP file",
                    )
                )
                warnings.append(f"STEP external reference could not be resolved: {reference} (referenced by {current})")
                continue

            resolved_key = str(resolved.resolve())
            if resolved_key in path_keys:
                records.append(
                    _StepExternalReferenceRecord(
                        source=current,
                        reference=reference,
                        status="cycle",
                        resolved=resolved,
                        reason="external STEP reference cycle detected",
                    )
                )
                warnings.append(
                    f"STEP external reference cycle detected: {reference} "
                    f"(referenced by {current}; resolves to {resolved})"
                )
                continue

            records.append(
                _StepExternalReferenceRecord(
                    source=current,
                    reference=reference,
                    status="resolved",
                    resolved=resolved,
                )
            )
            if resolved_key != root_key:
                member_sources.append(resolved)
            if resolved_key not in seen_sources:
                seen_sources.add(resolved_key)
                sources.append(resolved)
            queue.append((resolved, (*path_keys, resolved_key)))

    return _StepExternalReferenceGraph(
        root=root,
        sources=sources,
        member_sources=member_sources,
        records=records,
        warnings=warnings,
    )


def _step_external_references(source: Path) -> list[str]:
    text = _read_step_scan_text(source)
    if text is None:
        return []
    references: list[str] = []
    for match in _STEP_EXTERNAL_REF_RE.finditer(text):
        reference = match.group(1).replace("''", "'").strip()
        if not reference:
            continue
        references.append(reference)
    return references


def _clean_step_external_reference(reference: str) -> tuple[str | None, str | None]:
    value = _decode_step_string(reference.replace("''", "'")).strip().strip('"<>')
    parsed = urlparse(value)
    if parsed.scheme and not _looks_like_windows_path(value):
        if parsed.scheme.lower() != "file":
            return None, f"unsupported URI scheme: {parsed.scheme}"
        value = unquote(parsed.path)
    else:
        value = unquote(value.split("#", 1)[0].split("?", 1)[0])
    value = value.strip()
    if not value:
        return None, "empty path"
    if Path(value).suffix.lower() not in _STEP_SUFFIXES:
        return None, "unsupported file extension"
    return value, None


def _looks_like_windows_path(value: str) -> bool:
    return len(value) >= 3 and value[1] == ":" and value[0].isalpha() and value[2] in {"\\", "/"}


def _resolve_step_external_reference(reference: str, source: Path) -> Path | None:
    candidate = Path(reference)
    candidates = [candidate] if candidate.is_absolute() else [source.parent / candidate]
    if not candidate.is_absolute() and candidate.name != str(candidate):
        candidates.append(source.parent / candidate.name)
    for item in candidates:
        if item.exists() and item.is_file() and item.suffix.lower() in _STEP_SUFFIXES:
            return item.resolve()
    return None


def _merge_step_member_assets(
    members: list[_StepMemberImport],
    *,
    failed_members: list[dict[str, object]],
    options: StepReadOptions,
    sources: list[Path],
    duration: float,
    warnings: list[str],
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> Asset:
    target_units = members[0].asset.units
    target_meters_per_unit = members[0].asset.meters_per_unit
    target_up_axis = members[0].asset.up_axis
    root_children: list[Node] = []
    parts: dict[str, Part] = {}
    materials: dict[str, Material] = {}
    images: dict[str, ImageResource] = {}
    pmi: list[PmiAnnotation] = []
    member_records: list[dict[str, object]] = []
    part_dedupe_index: dict[tuple[object, ...], str] = {}

    for member in members:
        maps = _step_namespace_maps(member.asset, member.namespace)
        deduplicated_parts = _dedupe_step_member_parts(member.asset, maps, part_dedupe_index)
        root = _namespace_step_node(member.asset.root, maps)
        root.metadata.update(
            {
                "multi_file_member_index": member.index,
                "multi_file_member_source": str(member.source),
                "multi_file_member_namespace": member.namespace,
            }
        )
        unit_warning = _normalize_member_root_units(
            root,
            member.asset,
            target_units=target_units,
            target_meters_per_unit=target_meters_per_unit,
            target_up_axis=target_up_axis,
            source=member.source,
            index=member.index,
        )
        if unit_warning is not None:
            warnings.append(unit_warning)
        root_children.append(root)

        for image_id, image in member.asset.images.items():
            images[maps.images[image_id]] = _namespace_step_image(image, maps, member)
        for material_id, material in member.asset.materials.items():
            materials[maps.materials[material_id]] = _namespace_step_material(material, maps, member)
        for part_id, part in member.asset.parts.items():
            namespaced_part_id = maps.parts[part_id]
            if namespaced_part_id in parts:
                continue
            parts[namespaced_part_id] = _namespace_step_part(part, maps, member)
        pmi.extend(_namespace_step_pmi(annotation, maps, member) for annotation in member.asset.pmi)

        member_records.append(
            {
                "index": member.index,
                "source": str(member.source),
                "namespace": member.namespace,
                "status": "imported",
                "root_node_id": root.id,
                "nodes": len(member.asset.root.walk()),
                "parts": len(member.asset.parts),
                "deduplicated_parts": deduplicated_parts,
                "materials": len(member.asset.materials),
                "images": len(member.asset.images),
                "warnings": len(member.asset.report.warnings),
                "units": member.asset.units,
                "meters_per_unit": member.asset.meters_per_unit,
                "up_axis": member.asset.up_axis,
            }
        )

    member_records = sorted([*member_records, *failed_members], key=lambda item: cast(int, item["index"]))
    if reference_graph is not None:
        warnings.extend(warning for warning in reference_graph.warnings if warning not in warnings)
    source_identity = _multi_file_source_identity(sources)
    import_decisions = _multi_file_import_decisions(
        options,
        len(members),
        len(failed_members),
        reference_graph=reference_graph,
    )
    root_metadata: Metadata = {
        "source": "multi-file STEP import",
        "source_identity": source_identity,
        "multi_file": "true",
        "multi_file_member_count": str(len(members)),
    }
    if reference_graph is not None:
        root_metadata.update(
            {
                "source": str(reference_graph.root),
                "external_reference_graph": "true",
                "external_reference_root": str(reference_graph.root),
            }
        )
    root = Node(
        id=_stable_id("node", f"{source_identity}:root"),
        name=(
            f"{reference_graph.root.stem} external-reference STEP assembly"
            if reference_graph is not None
            else "multi-file STEP assembly"
        ),
        children=root_children,
        metadata=root_metadata,
    )
    report = Report(source_path=None)
    asset = Asset(
        root=root,
        parts=parts,
        materials=materials,
        images=images,
        units=target_units,
        meters_per_unit=target_meters_per_unit,
        up_axis=target_up_axis,
        source_path=None,
        metadata=_multi_file_asset_metadata(
            options,
            source_identity=source_identity,
            members=member_records,
            target_units=target_units,
            target_meters_per_unit=target_meters_per_unit,
            target_up_axis=target_up_axis,
            reference_graph=reference_graph,
        ),
        pmi=pmi,
        report=report,
    )
    asset.report.input_stats = asset.stats()
    for warning in warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "STEP",
            "backend": "OCP",
            "read_options": {**options.to_dict(), "multi_file": True},
            "multi_file": True,
            "member_count": len(members),
            "failed_member_count": len(failed_members),
            "members": member_records,
            "import_decisions": import_decisions,
            **({"external_reference_graph": reference_graph.to_dict()} if reference_graph is not None else {}),
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=duration,
        warnings=warnings,
    )
    if asset.metadata:
        asset.metadata["import_decisions"] = import_decisions
    return asset


def _step_namespace_maps(asset: Asset, namespace: str) -> _StepNamespaceMaps:
    return _StepNamespaceMaps(
        nodes={node.id: f"{namespace}__{node.id}" for node in asset.root.walk()},
        parts={part_id: f"{namespace}__{part_id}" for part_id in asset.parts},
        materials={material_id: f"{namespace}__{material_id}" for material_id in asset.materials},
        images={image_id: f"{namespace}__{image_id}" for image_id in asset.images},
    )


def _dedupe_step_member_parts(
    asset: Asset,
    maps: _StepNamespaceMaps,
    part_dedupe_index: dict[tuple[object, ...], str],
) -> int:
    deduplicated = 0
    for part_id, part in asset.parts.items():
        key = _step_member_part_dedupe_key(part, asset)
        if key is None:
            continue
        canonical_part_id = part_dedupe_index.get(key)
        if canonical_part_id is None:
            part_dedupe_index[key] = maps.parts[part_id]
            continue
        maps.parts[part_id] = canonical_part_id
        deduplicated += 1
    return deduplicated


def _step_member_part_dedupe_key(part: Part, asset: Asset) -> tuple[object, ...] | None:
    fingerprint = part.fingerprint
    if fingerprint is None and part.mesh is not None:
        fingerprint = part.mesh.fingerprint()
    if fingerprint is None:
        return None
    material_keys = tuple(_step_member_material_dedupe_key(material_id, asset) for material_id in part.material_ids)
    lod_keys = tuple(mesh.fingerprint() for mesh in part.lod_meshes)
    return (
        fingerprint,
        part.metadata.get("loaded_representation", ""),
        part.metadata.get("occt_face_material_indices", ""),
        material_keys,
        lod_keys,
    )


def _step_member_material_dedupe_key(material_id: str, asset: Asset) -> tuple[object, ...]:
    material = asset.materials.get(material_id)
    if material is None:
        return ("missing", material_id)
    payload_metadata = {
        key: value
        for key, value in material.metadata.items()
        if key
        not in {
            "material_library_path",
            "material_library_reference",
            "material_library_container",
        }
        and not key.endswith("_name")
    }
    return (
        material.name,
        material.base_color,
        material.metallic,
        material.roughness,
        material.opacity,
        _step_member_metadata_dedupe_value(payload_metadata, asset.images),
    )


def _step_member_metadata_dedupe_value(value: object, images: dict[str, ImageResource]) -> object:
    if isinstance(value, str):
        image = images.get(value)
        if image is not None:
            return ("image", _step_member_image_dedupe_key(image))
        return value
    if isinstance(value, dict):
        return tuple(
            sorted((str(key), _step_member_metadata_dedupe_value(item, images)) for key, item in value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_step_member_metadata_dedupe_value(item, images) for item in value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)


def _step_member_image_dedupe_key(image: ImageResource) -> tuple[object, ...]:
    return (image.mime_type, image.width, image.height, image.data)


def _namespace_step_node(node: Node, maps: _StepNamespaceMaps) -> Node:
    metadata = _namespace_metadata_ids(node.metadata, part_ids=maps.parts, node_ids=maps.nodes)
    return Node(
        id=maps.nodes[node.id],
        name=node.name,
        children=[_namespace_step_node(child, maps) for child in node.children],
        part_id=None if node.part_id is None else maps.parts.get(node.part_id, node.part_id),
        transform=node.transform,
        metadata=metadata,
    )


def _namespace_step_part(part: Part, maps: _StepNamespaceMaps, member: _StepMemberImport) -> Part:
    metadata = _namespace_metadata_ids(part.metadata, part_ids=maps.parts, node_ids=maps.nodes)
    metadata.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_part_id": part.id,
        }
    )
    return Part(
        id=maps.parts[part.id],
        name=part.name,
        source_shape=part.source_shape,
        mesh=None if part.mesh is None else part.mesh.copy(),
        material_ids=[maps.materials.get(material_id, material_id) for material_id in part.material_ids],
        metadata=metadata,
        fingerprint=part.fingerprint,
        lod_meshes=[mesh.copy() for mesh in part.lod_meshes],
    )


def _namespace_step_material(material: Material, maps: _StepNamespaceMaps, member: _StepMemberImport) -> Material:
    metadata = _namespace_metadata_ids(material.metadata, image_ids=maps.images)
    metadata.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_material_id": material.id,
        }
    )
    return Material(
        id=maps.materials[material.id],
        name=material.name,
        base_color=material.base_color,
        metallic=material.metallic,
        roughness=material.roughness,
        opacity=material.opacity,
        metadata=metadata,
    )


def _namespace_step_image(
    image: ImageResource,
    maps: _StepNamespaceMaps,
    member: _StepMemberImport,
) -> ImageResource:
    metadata = dict(image.metadata)
    metadata.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_image_id": image.id,
        }
    )
    return ImageResource(
        id=maps.images[image.id],
        name=image.name,
        mime_type=image.mime_type,
        data=image.data,
        width=image.width,
        height=image.height,
        metadata=metadata,
    )


def _namespace_step_pmi(
    annotation: PmiAnnotation,
    maps: _StepNamespaceMaps,
    member: _StepMemberImport,
) -> PmiAnnotation:
    source = dict(annotation.source)
    source.update(
        {
            "multi_file_member_index": member.index,
            "multi_file_member_source": str(member.source),
            "multi_file_member_namespace": member.namespace,
            "multi_file_member_pmi_id": annotation.id,
        }
    )
    return PmiAnnotation(
        id=f"{member.namespace}__{annotation.id}",
        kind=annotation.kind,
        text=annotation.text,
        value=annotation.value,
        unit=annotation.unit,
        tolerance=annotation.tolerance,
        applies_to=[maps.parts.get(part_id, part_id) for part_id in annotation.applies_to],
        view=annotation.view,
        plane=None if annotation.plane is None else [list(row) for row in annotation.plane],
        source=source,
    )


def _namespace_metadata_ids(
    metadata: Metadata,
    *,
    part_ids: dict[str, str] | None = None,
    node_ids: dict[str, str] | None = None,
    image_ids: dict[str, str] | None = None,
) -> Metadata:
    result = dict(metadata)
    part_ids = part_ids or {}
    node_ids = node_ids or {}
    image_ids = image_ids or {}
    for key, value in result.items():
        if key.endswith("_image") and isinstance(value, str) and value in image_ids:
            result[key] = image_ids[value]
        elif key in {"source_part_id", "source_part_ids", "split_source_part_id", "split_source_part_ids"}:
            result[key] = _namespace_metadata_value(value, part_ids)
        elif key in {"source_node_id", "source_node_ids", "split_source_node_id", "split_source_node_ids"}:
            result[key] = _namespace_metadata_value(value, node_ids)
    return result


def _namespace_metadata_value(value: object, mapping: dict[str, str]) -> object:
    if isinstance(value, str):
        separator = "|" if "|" in value and "," not in value else ","
        items = [item.strip() for item in value.replace("|", ",").split(",") if item.strip()]
        if not items:
            return value
        mapped = [mapping.get(item, item) for item in items]
        return separator.join(mapped)
    if isinstance(value, list):
        return [mapping.get(str(item), str(item)) for item in value]
    if isinstance(value, tuple):
        return tuple(mapping.get(str(item), str(item)) for item in value)
    return mapping.get(str(value), value)


def _normalize_member_root_units(
    root: Node,
    asset: Asset,
    *,
    target_units: str,
    target_meters_per_unit: float,
    target_up_axis: str,
    source: Path,
    index: int,
) -> str | None:
    scale = asset.meters_per_unit / target_meters_per_unit
    if not np.isclose(scale, 1.0):
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] *= scale
        root.transform = transform @ root.transform
        root.metadata["multi_file_unit_conversion"] = {
            "source_units": asset.units,
            "source_meters_per_unit": asset.meters_per_unit,
            "target_units": target_units,
            "target_meters_per_unit": target_meters_per_unit,
            "scale": scale,
        }
    if asset.up_axis != target_up_axis:
        return (
            f"multi-file STEP member {index} ({source}) uses up axis {asset.up_axis}; "
            f"combined asset uses {target_up_axis}; set target_up_axis to normalize all members explicitly"
        )
    return None


def _multi_file_asset_metadata(
    options: StepReadOptions,
    *,
    source_identity: str,
    members: list[dict[str, object]],
    target_units: str,
    target_meters_per_unit: float,
    target_up_axis: str,
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> Metadata:
    if not options.metadata:
        return {}
    metadata: Metadata = {
        "source": "multi-file STEP import",
        "source_identity": source_identity,
        "units": target_units,
        "meters_per_unit": target_meters_per_unit,
        "up_axis": target_up_axis,
        "metadata_options": {**options.to_dict(), "multi_file": True},
        "multi_file_import": {
            "member_count": sum(1 for member in members if member.get("status") == "imported"),
            "failed_member_count": sum(1 for member in members if member.get("status") == "error"),
            "members": members,
        },
    }
    if reference_graph is not None:
        metadata["source"] = str(reference_graph.root)
        metadata["external_reference_graph"] = reference_graph.to_dict()
    return metadata


def _multi_file_import_decisions(
    options: StepReadOptions,
    member_count: int,
    failed_member_count: int,
    *,
    reference_graph: _StepExternalReferenceGraph | None = None,
) -> dict[str, object]:
    if reference_graph is not None:
        return {
            "multi_file": _external_reference_import_decision(
                options,
                reference_graph,
                member_count=member_count,
                failed_member_count=failed_member_count,
            )
        }
    return {
        "multi_file": _import_decision(
            requested=True,
            effective=member_count > 0,
            state="approximated" if failed_member_count else "honored",
            detail=(
                "explicit STEP member paths were imported as separate deterministic namespaces; "
                "quoted external STEP references are resolved by read_step(..., multi_file=True)"
            ),
            counts={"members": member_count, "failed_members": failed_member_count},
        )
    }


def _external_reference_import_decision(
    options: StepReadOptions,
    graph: _StepExternalReferenceGraph,
    *,
    member_count: int,
    failed_member_count: int,
) -> dict[str, object]:
    summary = graph.summary()
    missing_or_unsupported = summary["missing"] + summary["unsupported"]
    cycle_count = summary.get("cycles", 0)
    if failed_member_count:
        state = "approximated"
    elif missing_or_unsupported:
        state = "missing_sources"
    elif cycle_count:
        state = "approximated"
    else:
        state = "honored"
    detail = (
        "external STEP references were resolved from quoted STEP path records and imported as deterministic "
        "member occurrences"
        if summary["references"]
        else "no external STEP references were found in the master STEP file"
    )
    return _import_decision(
        requested=options.multi_file,
        effective=summary["resolved_occurrences"] > 0,
        state=state,
        detail=detail,
        counts={
            "members": member_count,
            "failed_members": failed_member_count,
            **summary,
        },
    )


def _attach_step_external_reference_graph(
    asset: Asset,
    graph: _StepExternalReferenceGraph,
    options: StepReadOptions,
) -> None:
    decision = _external_reference_import_decision(
        options,
        graph,
        member_count=1,
        failed_member_count=0,
    )
    graph_data = graph.to_dict()
    if asset.metadata:
        asset.metadata["external_reference_graph"] = graph_data
        metadata_options = asset.metadata.get("metadata_options")
        if isinstance(metadata_options, dict):
            metadata_options["multi_file"] = True
        import_decisions = asset.metadata.get("import_decisions")
        if isinstance(import_decisions, dict):
            import_decisions["multi_file"] = decision
    for step in asset.report.steps:
        if step.name != "import":
            continue
        read_options = step.options.get("read_options")
        if isinstance(read_options, dict):
            read_options["multi_file"] = True
        import_decisions = step.options.get("import_decisions")
        if isinstance(import_decisions, dict):
            import_decisions["multi_file"] = decision
        else:
            step.options["import_decisions"] = {"multi_file": decision}
        step.options["external_reference_graph"] = graph_data
        for warning in graph.warnings:
            if warning not in step.warnings:
                step.warnings.append(warning)
        break
    for warning in graph.warnings:
        if warning not in asset.report.warnings:
            asset.report.add_warning(warning)


def _multi_file_namespace(index: int, source: Path) -> str:
    stem = re.sub(r"[^a-z0-9]+", "_", source.stem.lower()).strip("_") or "step"
    digest = _stable_id("ns", str(source.resolve())).split("_", 1)[1][:8]
    return f"member{index}_{stem}_{digest}"


def _multi_file_source_identity(sources: list[Path]) -> str:
    encoded = "|".join(str(source.resolve()) for source in sources)
    return _stable_id("step_multi", encoded)
