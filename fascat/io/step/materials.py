from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from fascat.image import ImageResource
from fascat.io._import_base import _stable_id
from fascat.io.step.records import _ensure_loadable_file_size, _name_tokens, _step_scan_capped
from fascat.io.step.textures import (
    _GENERIC_MATERIAL_TOKENS,
    _MAX_SOURCE_TEXTURE_BYTES,
    _SOURCE_TEXTURE_EXPORT_SLOTS,
    _SOURCE_TEXTURE_SUFFIXES,
    _ArchiveTextureMap,
    _clean_source_texture_reference,
    _confine_to_search_roots,
    _load_source_texture,
    _load_source_texture_data,
    _resolve_source_texture,
    _texture_slot,
)
from fascat.material import Material
from fascat.metadata import Metadata
from fascat.options import StepReadOptions

_MATERIAL_RECORD_SUFFIXES = {".json", ".mtl"}


_MATERIAL_LIBRARY_CONTAINER_SUFFIXES = {".zip"}


_MATERIAL_LIBRARY_SUFFIXES = _MATERIAL_RECORD_SUFFIXES | _MATERIAL_LIBRARY_CONTAINER_SUFFIXES


_MATERIAL_LIBRARY_REF_RE = re.compile(r"'([^']+\.(?:json|mtl|zip)(?:[#?][^']*)?)'", re.IGNORECASE)


_MAX_MATERIAL_LIBRARY_BYTES = 16 * 1024 * 1024


_MAX_MATERIAL_LIBRARY_ARCHIVE_ENTRIES = 512


_MAX_MATERIAL_LIBRARY_ARCHIVE_UNCOMPRESSED_BYTES = 128 * 1024 * 1024


_MAX_MATERIAL_LIBRARY_JSON_DEPTH = 64


@dataclass(frozen=True)
class _CadMaterialSpec:
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.5
    opacity: float = 1.0
    metadata: tuple[tuple[str, str], ...] = ()

    def metadata_dict(self) -> Metadata:
        return dict(self.metadata)


@dataclass(frozen=True)
class _MaterialLibraryRule:
    tokens: tuple[str, ...]
    metallic: float | None = None
    roughness: float | None = None
    opacity: float | None = None
    base_color: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class _MaterialLibrarySpec:
    name: str
    base_color: tuple[float, float, float, float] | None = None
    metallic: float | None = None
    roughness: float | None = None
    opacity: float | None = None
    texture_images: tuple[tuple[str, str], ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def metadata_dict(self) -> Metadata:
        return dict(self.metadata)


@dataclass
class _MaterialLibraryExtraction:
    materials: list[_MaterialLibrarySpec]
    images: dict[str, ImageResource]
    summary: dict[str, int]
    warnings: list[str]


_MATERIAL_LIBRARY_RULES: tuple[_MaterialLibraryRule, ...] = (
    _MaterialLibraryRule(("stainless", "steel", "iron", "titanium"), metallic=1.0, roughness=0.32),
    _MaterialLibraryRule(("aluminum", "aluminium", "6061", "7075"), metallic=1.0, roughness=0.28),
    _MaterialLibraryRule(("chrome",), metallic=1.0, roughness=0.12),
    _MaterialLibraryRule(("brass",), metallic=1.0, roughness=0.24, base_color=(0.86, 0.66, 0.34, 1.0)),
    _MaterialLibraryRule(("copper",), metallic=1.0, roughness=0.22, base_color=(0.95, 0.55, 0.33, 1.0)),
    _MaterialLibraryRule(("bronze",), metallic=1.0, roughness=0.3, base_color=(0.72, 0.45, 0.24, 1.0)),
    _MaterialLibraryRule(("gold",), metallic=1.0, roughness=0.18, base_color=(1.0, 0.76, 0.34, 1.0)),
    _MaterialLibraryRule(("glass",), metallic=0.0, roughness=0.03, opacity=0.35),
    _MaterialLibraryRule(("acrylic", "polycarbonate", "transparent"), metallic=0.0, roughness=0.08, opacity=0.55),
    _MaterialLibraryRule(("rubber", "silicone"), metallic=0.0, roughness=0.78),
    _MaterialLibraryRule(("plastic", "abs", "pla", "nylon", "polymer"), metallic=0.0, roughness=0.58),
    _MaterialLibraryRule(("ceramic", "porcelain"), metallic=0.0, roughness=0.36),
    _MaterialLibraryRule(("paint", "coating", "powdercoat", "powder"), metallic=0.0, roughness=0.48),
    _MaterialLibraryRule(("wood", "timber"), metallic=0.0, roughness=0.64),
)


def _material_binding_plan(base_material_id: str, face_material_ids: list[str]) -> tuple[list[str], list[int]]:
    material_ids = [base_material_id]
    material_index_by_id = {base_material_id: 0}
    material_indices: list[int] = []
    for face_material_id in face_material_ids:
        material_index = material_index_by_id.get(face_material_id)
        if material_index is None:
            material_index = len(material_ids)
            material_ids.append(face_material_id)
            material_index_by_id[face_material_id] = material_index
        material_indices.append(material_index)
    return material_ids, material_indices


def _ensure_material(
    materials: dict[str, Material],
    material_id: str,
    spec: _CadMaterialSpec,
) -> None:
    if material_id not in materials:
        materials[material_id] = Material(
            id=material_id,
            name=spec.name or f"CAD material {material_id[-8:]}",
            base_color=spec.base_color,
            metallic=spec.metallic,
            roughness=spec.roughness,
            opacity=spec.opacity,
            metadata=spec.metadata_dict(),
        )


def _color_material_spec(color: tuple[float, float, float, float]) -> _CadMaterialSpec:
    color = _clamp_color(color)
    return _CadMaterialSpec(
        name=f"CAD color {_material_id(color)[-8:]}",
        base_color=color,
        opacity=color[3],
        metadata=(("cad_material_source", "color"), ("pbr_mapping_status", "color_only")),
    )


def _material_id_from_spec(spec: _CadMaterialSpec) -> str:
    metadata = spec.metadata_dict()
    if metadata.get("cad_material_source") == "color":
        return _material_id(spec.base_color)
    encoded = ",".join(
        [
            spec.name,
            *(f"{component:.6f}" for component in spec.base_color),
            f"{spec.metallic:.6f}",
            f"{spec.roughness:.6f}",
            f"{spec.opacity:.6f}",
            str(metadata.get("cad_material_source", "")),
            str(metadata.get("cad_material_mapping_rule", "")),
        ]
    )
    return _stable_id("mat", encoded)


def _label_visual_material_spec(
    vis_material_tool: Any,
    label: Any,
    options: StepReadOptions,
) -> _CadMaterialSpec | None:
    if vis_material_tool is None:
        return None
    try:
        if not vis_material_tool.IsSetShapeMaterial(label):
            return None
        material = vis_material_tool.GetShapeMaterial_s(label)
    except Exception:
        return None
    spec = _visual_material_spec(material)
    return _apply_material_library_mapping(spec, options) if spec is not None else None


def _shape_visual_material_spec(
    vis_material_tool: Any,
    shape: Any,
    options: StepReadOptions,
) -> _CadMaterialSpec | None:
    if vis_material_tool is None:
        return None
    try:
        if not vis_material_tool.IsSetShapeMaterial(shape):
            return None
        material = vis_material_tool.GetShapeMaterial(shape)
    except Exception:
        return None
    spec = _visual_material_spec(material)
    return _apply_material_library_mapping(spec, options) if spec is not None else None


def _visual_material_spec(material: Any) -> _CadMaterialSpec | None:
    if material is None:
        return None
    try:
        if material.IsEmpty():
            return None
    except Exception:
        return None

    name = _ocp_string(material.RawName()) or "CAD visual material"
    metadata: Metadata = {"cad_material_source": "xde_visual_material", "cad_material_name": name}
    try:
        if material.HasPbrMaterial() and material.PbrMaterial().IsDefined():
            pbr = material.PbrMaterial()
            color = _quantity_rgba_tuple(pbr.BaseColor())
            metallic = _clamp01(float(pbr.Metallic()))
            roughness = max(0.04, _clamp01(float(pbr.Roughness())))
            metadata.update(
                {
                    "cad_visual_material_model": "pbr",
                    "pbr_mapping_status": "source_pbr",
                    "cad_material_metallic": f"{metallic:g}",
                    "cad_material_roughness": f"{roughness:g}",
                }
            )
            return _CadMaterialSpec(
                name=name,
                base_color=color,
                metallic=metallic,
                roughness=roughness,
                opacity=color[3],
                metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
            )
    except Exception:
        pass

    try:
        if material.HasCommonMaterial() and material.CommonMaterial().IsDefined():
            common = material.CommonMaterial()
            color3 = _quantity_color_tuple(common.DiffuseColor())
            opacity = 1.0 - _clamp01(float(common.Transparency()))
            shininess = _clamp01(float(common.Shininess()))
            roughness = max(0.04, 1.0 - shininess**0.5)
            color = (color3[0], color3[1], color3[2], opacity)
            metadata.update(
                {
                    "cad_visual_material_model": "common",
                    "pbr_mapping_status": "common_to_pbr",
                    "cad_material_shininess": f"{shininess:g}",
                    "cad_material_transparency": f"{1.0 - opacity:g}",
                }
            )
            return _CadMaterialSpec(
                name=name,
                base_color=color,
                metallic=0.0,
                roughness=roughness,
                opacity=opacity,
                metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
            )
    except Exception:
        pass

    try:
        color = _quantity_rgba_tuple(material.BaseColor())
    except Exception:
        return None
    metadata.update({"cad_visual_material_model": "base_color", "pbr_mapping_status": "base_color_to_pbr"})
    return _CadMaterialSpec(
        name=name,
        base_color=color,
        opacity=color[3],
        metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
    )


def _apply_material_library_mapping(spec: _CadMaterialSpec, options: StepReadOptions) -> _CadMaterialSpec:
    if not options.material_library_mapping:
        metadata = {**spec.metadata_dict(), "pbr_mapping_status": "source_only"}
        return _replace_material_spec(spec, metadata=metadata)
    tokens = set(_name_tokens(spec.name))
    if not tokens:
        return spec
    for rule in _MATERIAL_LIBRARY_RULES:
        matched = sorted(tokens.intersection(rule.tokens))
        if not matched:
            continue
        color = spec.base_color
        if rule.base_color is not None and _is_default_material_color(spec.base_color):
            color = rule.base_color
        opacity = spec.opacity if rule.opacity is None else min(spec.opacity, rule.opacity)
        color = (color[0], color[1], color[2], min(color[3], opacity))
        metadata = {
            **spec.metadata_dict(),
            "pbr_mapping_status": "library_rule",
            "cad_material_mapping_rule": matched[0],
        }
        return _CadMaterialSpec(
            name=spec.name,
            base_color=color,
            metallic=spec.metallic if rule.metallic is None else rule.metallic,
            roughness=spec.roughness if rule.roughness is None else rule.roughness,
            opacity=opacity,
            metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
        )
    metadata = {**spec.metadata_dict(), "pbr_mapping_status": spec.metadata_dict().get("pbr_mapping_status", "no_rule")}
    return _replace_material_spec(spec, metadata=metadata)


def _replace_material_spec(spec: _CadMaterialSpec, *, metadata: Metadata) -> _CadMaterialSpec:
    return _CadMaterialSpec(
        name=spec.name,
        base_color=spec.base_color,
        metallic=spec.metallic,
        roughness=spec.roughness,
        opacity=spec.opacity,
        metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
    )


def _empty_material_library_summary() -> dict[str, int]:
    return {
        "references": 0,
        "resolved": 0,
        "missing": 0,
        "unsupported": 0,
        "unreadable": 0,
        "materials": 0,
        "textures": 0,
        "texture_missing": 0,
        "texture_unreadable": 0,
    }


def _empty_material_library_binding_summary() -> dict[str, int]:
    return {
        "library_materials": 0,
        "matched_library_materials": 0,
        "unmatched_library_materials": 0,
        "applied_materials": 0,
        "bound_textures": 0,
    }


def _material_library_import_state(
    options: StepReadOptions,
    summary: dict[str, int],
    binding_summary: dict[str, int],
) -> str:
    if not options.material_library_mapping:
        return "disabled"
    if summary["references"] and summary["resolved"] == 0:
        return "missing_sources"
    if summary["unreadable"] and summary["materials"] == 0:
        return "unsupported"
    if summary["materials"] and binding_summary["applied_materials"] == 0:
        return "approximated"
    return "honored"


def _extract_material_libraries(
    source: Path,
    source_identity: str,
    options: StepReadOptions,
) -> _MaterialLibraryExtraction:
    if not options.material_library_mapping:
        return _MaterialLibraryExtraction(
            materials=[],
            images={},
            summary=_empty_material_library_summary(),
            warnings=[],
        )

    references = _material_library_references(source)
    configured_paths = [Path(path) for path in options.material_library_paths]
    search_roots = [
        source.parent,
        *(Path(path) for path in options.source_texture_search_paths),
        *(path for path in configured_paths if path.is_dir()),
    ]
    candidates: list[tuple[str, Path | None]] = []
    missing = 0
    unsupported = 0
    unreadable = 0
    texture_missing = 0
    texture_unreadable = 0
    warnings: list[str] = []

    for reference in references:
        library_path = _resolve_material_library_reference(reference, search_roots)
        if library_path is None:
            missing += 1
            warnings.append(f"material library reference could not be resolved: {reference}")
            candidates.append((reference, None))
        else:
            candidates.append((reference, library_path))

    for configured in configured_paths:
        explicit_candidates = _material_library_path_candidates(configured)
        if not explicit_candidates:
            missing += 1
            warnings.append(f"material library path could not be resolved: {configured}")
            candidates.append((str(configured), None))
            continue
        candidates.extend((str(configured), path) for path in explicit_candidates)

    materials: list[_MaterialLibrarySpec] = []
    images: dict[str, ImageResource] = {}
    seen_libraries: set[Path] = set()
    seen_texture_paths: set[str] = set()
    resolved_libraries = 0
    for reference, library_path in candidates:
        if library_path is None:
            continue
        suffix = library_path.suffix.lower()
        if suffix not in _MATERIAL_LIBRARY_SUFFIXES:
            unsupported += 1
            warnings.append(f"material library format is unsupported: {library_path}")
            continue
        resolved = library_path.resolve()
        if resolved in seen_libraries:
            continue
        seen_libraries.add(resolved)
        try:
            specs, texture_stats = _load_material_library(
                library_path,
                source_identity=source_identity,
                reference=reference,
                images=images,
                seen_texture_paths=seen_texture_paths,
                search_roots=[library_path.parent, *search_roots],
                color_space=options.material_library_color_space,
            )
        except ValueError as exc:
            unreadable += 1
            warnings.append(str(exc))
            continue
        resolved_libraries += 1
        materials.extend(specs)
        texture_missing += texture_stats["missing"]
        texture_unreadable += texture_stats["unreadable"]

    summary = _empty_material_library_summary()
    summary.update(
        {
            "references": len(references) + len(configured_paths),
            "resolved": resolved_libraries,
            "missing": missing,
            "unsupported": unsupported,
            "unreadable": unreadable,
            "materials": len(materials),
            "textures": len(images),
            "texture_missing": texture_missing,
            "texture_unreadable": texture_unreadable,
        }
    )
    return _MaterialLibraryExtraction(materials=materials, images=images, summary=summary, warnings=warnings)


def _material_library_references(source: Path) -> list[str]:
    if _step_scan_capped(source):
        return []
    try:
        text = source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    references: list[str] = []
    seen: set[str] = set()
    for match in _MATERIAL_LIBRARY_REF_RE.finditer(text):
        reference = _clean_source_texture_reference(match.group(1))
        if not reference or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    return references


def _resolve_material_library_reference(reference: str, search_roots: list[Path]) -> Path | None:
    candidate = Path(reference)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        for root in search_roots:
            candidates.append(root / candidate)
            if candidate.name != str(candidate):
                candidates.append(root / candidate.name)
    for item in candidates:
        if not _confine_to_search_roots(item, search_roots):
            continue
        try:
            if item.is_file() and item.suffix.lower() in _MATERIAL_LIBRARY_SUFFIXES:
                return item
        except OSError:
            continue
    return None


def _material_library_path_candidates(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in _MATERIAL_LIBRARY_SUFFIXES:
        return [path]
    if path.is_dir():
        try:
            return sorted(
                item for item in path.iterdir() if item.is_file() and item.suffix.lower() in _MATERIAL_LIBRARY_SUFFIXES
            )
        except OSError:
            return []
    return []


def _load_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    color_space: str,
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    if path.suffix.lower() == ".json":
        return _load_json_material_library(
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            color_space=color_space,
        )
    if path.suffix.lower() == ".mtl":
        return _load_mtl_material_library(
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            color_space=color_space,
        )
    if path.suffix.lower() == ".zip":
        return _load_zipped_material_library(
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            color_space=color_space,
        )
    raise ValueError(f"material library format is unsupported: {path}")


def _load_json_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    color_space: str,
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    _ensure_loadable_file_size(path, _MAX_MATERIAL_LIBRARY_BYTES, "material library")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"material library could not be read: {path}") from exc
    entries = _json_material_entries(payload)
    if not entries:
        raise ValueError(f"material library did not contain supported material records: {path}")
    specs: list[_MaterialLibrarySpec] = []
    texture_stats = {"missing": 0, "unreadable": 0}
    for entry in entries:
        spec, stats = _json_material_spec(
            entry,
            path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            library_label=str(path),
            color_space=color_space,
        )
        texture_stats["missing"] += stats["missing"]
        texture_stats["unreadable"] += stats["unreadable"]
        if spec is not None:
            specs.append(spec)
    if not specs:
        raise ValueError(f"material library did not contain supported material records: {path}")
    return specs, texture_stats


def _load_zipped_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    color_space: str,
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    _ = search_roots
    _ensure_loadable_file_size(path, _MAX_MATERIAL_LIBRARY_BYTES, "material library archive")
    try:
        with zipfile.ZipFile(path) as archive:
            _validate_material_library_archive_limits(path, archive)
            material_members = _archive_material_members(archive)
            if not material_members:
                raise ValueError(f"material library archive did not contain JSON or MTL records: {path}")
            archive_textures = _archive_texture_members(archive)
            specs: list[_MaterialLibrarySpec] = []
            texture_stats = {"missing": 0, "unreadable": 0}
            member_errors: list[str] = []
            for member_name in material_members:
                member_path = Path(member_name)
                library_label = f"{path}!/{member_name}"
                try:
                    member_bytes = archive.read(member_name)
                except (KeyError, OSError, zipfile.BadZipFile):
                    member_errors.append(f"material library archive member could not be read: {library_label}")
                    continue
                try:
                    if member_path.suffix.lower() == ".json":
                        try:
                            payload = json.loads(member_bytes.decode("utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                            raise ValueError(f"material library could not be read: {library_label}") from exc
                        entries = _json_material_entries(payload)
                        if not entries:
                            raise ValueError(
                                f"material library did not contain supported material records: {library_label}"
                            )
                    else:
                        lines = member_bytes.decode("utf-8", errors="ignore").splitlines()
                        entries = _mtl_material_entries(lines, library_label)
                    member_specs, member_texture_stats = _material_specs_from_entries(
                        entries,
                        member_path,
                        source_identity=source_identity,
                        reference=reference,
                        images=images,
                        seen_texture_paths=seen_texture_paths,
                        search_roots=[],
                        library_label=library_label,
                        archive_textures=archive_textures,
                        archive_container=path,
                        color_space=color_space,
                    )
                except ValueError as exc:
                    member_errors.append(str(exc))
                    continue
                specs.extend(member_specs)
                texture_stats["missing"] += member_texture_stats["missing"]
                texture_stats["unreadable"] += member_texture_stats["unreadable"]
            if not specs:
                detail = f": {member_errors[0]}" if member_errors else ""
                raise ValueError(f"material library archive did not contain supported material records: {path}{detail}")
            return specs, texture_stats
    except zipfile.BadZipFile as exc:
        raise ValueError(f"material library archive could not be read: {path}") from exc
    except OSError as exc:
        raise ValueError(f"material library archive could not be read: {path}") from exc


def _validate_material_library_archive_limits(path: Path, archive: zipfile.ZipFile) -> None:
    entries = [info for info in archive.infolist() if not info.is_dir()]
    if len(entries) > _MAX_MATERIAL_LIBRARY_ARCHIVE_ENTRIES:
        raise ValueError(
            f"material library archive has too many entries "
            f"({len(entries)} > {_MAX_MATERIAL_LIBRARY_ARCHIVE_ENTRIES}): {path}"
        )
    total_uncompressed = 0
    for info in entries:
        total_uncompressed += int(info.file_size)
        suffix = PurePosixPath(info.filename.replace("\\", "/")).suffix.lower()
        label = f"{path}!/{_archive_member_name(info.filename)}"
        if suffix in _MATERIAL_RECORD_SUFFIXES and info.file_size > _MAX_MATERIAL_LIBRARY_BYTES:
            raise ValueError(f"material library archive member is too large: {label}")
        if suffix in _SOURCE_TEXTURE_SUFFIXES and info.file_size > _MAX_SOURCE_TEXTURE_BYTES:
            raise ValueError(f"material library archive texture is too large: {label}")
    if total_uncompressed > _MAX_MATERIAL_LIBRARY_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"material library archive uncompressed payload is too large "
            f"({total_uncompressed} > {_MAX_MATERIAL_LIBRARY_ARCHIVE_UNCOMPRESSED_BYTES} bytes): {path}"
        )


def _archive_material_members(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        name
        for name in archive.namelist()
        if _safe_archive_member_name(name) and PurePosixPath(name).suffix.lower() in _MATERIAL_RECORD_SUFFIXES
    )


def _archive_texture_members(archive: zipfile.ZipFile) -> _ArchiveTextureMap:
    textures: _ArchiveTextureMap = {}
    for name in archive.namelist():
        if not _safe_archive_member_name(name) or PurePosixPath(name).suffix.lower() not in _SOURCE_TEXTURE_SUFFIXES:
            continue
        try:
            textures[_archive_member_key(name)] = (_archive_member_name(name), archive.read(name))
        except (KeyError, OSError, zipfile.BadZipFile):
            continue
    return textures


def _safe_archive_member_name(name: str) -> bool:
    cleaned = name.replace("\\", "/")
    member = PurePosixPath(cleaned)
    return bool(cleaned and member.name and not member.is_absolute() and ".." not in member.parts)


def _archive_member_name(name: str) -> str:
    return str(PurePosixPath(name.replace("\\", "/"))).lstrip("./")


def _archive_member_key(name: str) -> str:
    member = PurePosixPath(name.replace("\\", "/"))
    parts: list[str] = []
    for part in member.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts).lower()


def _json_material_entries(payload: object, *, depth: int = 0) -> list[dict[str, object]]:
    if isinstance(payload, list):
        return [cast(dict[str, object], item) for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("materials", "materialLibrary", "material_library", "library", "items"):
        value = payload.get(key)
        if isinstance(value, list):
            return [cast(dict[str, object], item) for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            if depth >= _MAX_MATERIAL_LIBRARY_JSON_DEPTH:
                raise ValueError("material library JSON nesting is too deep")
            return _json_material_entries(value, depth=depth + 1)
    entries: list[dict[str, object]] = []
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        entry = dict(cast(dict[str, object], value))
        entry.setdefault("name", key)
        entries.append(entry)
    return entries


def _json_material_spec(
    entry: dict[str, object],
    library_path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
    color_space: str = "auto",
) -> tuple[_MaterialLibrarySpec | None, dict[str, int]]:
    name = _json_material_name(entry)
    if not name:
        return None, {"missing": 0, "unreadable": 0}
    pbr = _json_mapping(entry.get("pbrMetallicRoughness")) or _json_mapping(entry.get("pbr_metallic_roughness")) or {}
    base_color = (
        _json_color(entry.get("base_color"), color_space)
        or _json_color(entry.get("baseColor"), color_space)
        or _json_color(entry.get("baseColorFactor"), color_space)
        or _json_color(entry.get("base_color_factor"), color_space)
        or _json_color(entry.get("diffuseColor"), color_space)
        or _json_color(entry.get("diffuse"), color_space)
        or _json_color(entry.get("albedo"), color_space)
        or _json_color(entry.get("color"), color_space)
        or _json_color(pbr.get("baseColorFactor"), color_space)
    )
    metallic = _optional_material_float(
        entry.get("metallic"),
        entry.get("metallicFactor"),
        entry.get("metalness"),
        entry.get("metalnessFactor"),
        pbr.get("metallicFactor"),
    )
    roughness = _optional_material_float(
        entry.get("roughness"), entry.get("roughnessFactor"), pbr.get("roughnessFactor")
    )
    opacity = _json_opacity(entry, color_space)
    texture_images, texture_stats = _json_material_texture_images(
        entry,
        pbr,
        library_path,
        source_identity=source_identity,
        images=images,
        seen_texture_paths=seen_texture_paths,
        search_roots=search_roots,
        material_name=name,
        library_label=library_label,
        archive_textures=archive_textures,
        archive_container=archive_container,
    )
    metadata: Metadata = {
        "cad_material_source": "material_library",
        "material_library_name": name,
        "material_library_reference": reference,
        "material_library_path": library_label,
        "pbr_mapping_status": "material_library",
        "material_library_color_space": color_space,
    }
    if archive_container is not None:
        metadata["material_library_container"] = str(archive_container)
    return (
        _MaterialLibrarySpec(
            name=name,
            base_color=base_color,
            metallic=metallic,
            roughness=roughness,
            opacity=opacity,
            texture_images=tuple(texture_images),
            metadata=tuple(sorted((key, str(value)) for key, value in metadata.items())),
        ),
        texture_stats,
    )


def _json_material_name(entry: dict[str, object]) -> str:
    for key in ("name", "materialName", "material_name", "displayName", "display_name", "id"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _json_mapping(value: object) -> dict[str, object] | None:
    return cast(dict[str, object], value) if isinstance(value, dict) else None


def _json_color(value: object, color_space: str = "auto") -> tuple[float, float, float, float] | None:
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.startswith("#"):
            return _hex_color(stripped)
        parts = [part for part in re.split(r"[\s,]+", stripped) if part]
        if len(parts) in {3, 4}:
            return _numeric_color(parts, color_space)
        return None
    if isinstance(value, dict):
        red = _optional_material_number(value.get("r"), value.get("red"))
        green = _optional_material_number(value.get("g"), value.get("green"))
        blue = _optional_material_number(value.get("b"), value.get("blue"))
        alpha = _optional_material_number(value.get("a"), value.get("alpha"))
        if red is None or green is None or blue is None:
            return None
        return _numeric_color_components(red, green, blue, alpha, color_space)
    if isinstance(value, list | tuple) and len(value) in {3, 4}:
        return _numeric_color(list(value), color_space)
    return None


def _hex_color(value: str) -> tuple[float, float, float, float] | None:
    encoded = value.strip().removeprefix("#")
    if len(encoded) not in {6, 8}:
        return None
    try:
        red = int(encoded[0:2], 16) / 255.0
        green = int(encoded[2:4], 16) / 255.0
        blue = int(encoded[4:6], 16) / 255.0
        alpha = int(encoded[6:8], 16) / 255.0 if len(encoded) == 8 else 1.0
    except ValueError:
        return None
    return (_clamp01(red), _clamp01(green), _clamp01(blue), _clamp01(alpha))


def _numeric_color(values: list[object], color_space: str = "auto") -> tuple[float, float, float, float] | None:
    parsed = [_optional_material_number(value) for value in values]
    if any(value is None for value in parsed):
        return None
    numbers = [cast(float, value) for value in parsed]
    red, green, blue = numbers[:3]
    alpha = numbers[3] if len(numbers) == 4 else None
    return _numeric_color_components(red, green, blue, alpha, color_space)


def _numeric_color_components(
    red: float, green: float, blue: float, alpha: float | None, color_space: str
) -> tuple[float, float, float, float]:
    scale_255 = color_space == "srgb255" or (color_space == "auto" and max(red, green, blue) > 1.0)
    alpha_value = 255.0 if alpha is None and scale_255 else 1.0 if alpha is None else alpha
    return _normalize_color_range((red, green, blue, alpha_value), color_space)


def _normalize_color_range(
    color: tuple[float, float, float, float], color_space: str = "auto"
) -> tuple[float, float, float, float]:
    if color_space == "srgb255" or (color_space == "auto" and any(component > 1.0 for component in color)):
        return (
            _clamp01(color[0] / 255.0),
            _clamp01(color[1] / 255.0),
            _clamp01(color[2] / 255.0),
            _clamp01(color[3] / 255.0),
        )
    return _clamp_color(color)


def _optional_material_number(*values: object) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, str | int | float):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _optional_material_float(*values: object) -> float | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if not isinstance(value, str | int | float):
            continue
        try:
            return _clamp01(float(value))
        except (TypeError, ValueError):
            continue
    return None


def _json_opacity(entry: dict[str, object], color_space: str = "auto") -> float | None:
    opacity = _optional_material_float(entry.get("opacity"), entry.get("alpha"))
    if opacity is not None:
        return opacity
    transparency = _optional_material_float(entry.get("transparency"))
    if transparency is not None:
        return 1.0 - transparency
    base_color = _json_color(entry.get("baseColorFactor"), color_space) or _json_color(
        entry.get("base_color"), color_space
    )
    if base_color is not None:
        return base_color[3]
    return None


def _json_material_texture_images(
    entry: dict[str, object],
    pbr: dict[str, object],
    library_path: Path,
    *,
    source_identity: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    material_name: str,
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    references: list[tuple[str, str]] = []
    texture_map = _json_mapping(entry.get("textures"))
    if texture_map is not None:
        for key, value in texture_map.items():
            slot = _material_library_texture_slot(key)
            reference = _texture_reference_from_value(value)
            if slot is not None and reference:
                references.append((slot, reference))
    flat_keys = {
        "baseColorTexture": "base_color",
        "base_color_texture": "base_color",
        "diffuseTexture": "base_color",
        "albedoTexture": "base_color",
        "metallicRoughnessTexture": "metallic_roughness",
        "metallic_roughness_texture": "metallic_roughness",
        "normalTexture": "normal",
        "normal_texture": "normal",
        "occlusionTexture": "occlusion",
        "aoTexture": "occlusion",
        "emissiveTexture": "emissive",
        "opacityTexture": "opacity",
        "alphaTexture": "opacity",
        "roughnessTexture": "roughness",
        "metallicTexture": "metallic",
    }
    for key, slot in flat_keys.items():
        reference = _texture_reference_from_value(entry.get(key))
        if reference:
            references.append((slot, reference))
    for key, slot in {"baseColorTexture": "base_color", "metallicRoughnessTexture": "metallic_roughness"}.items():
        reference = _texture_reference_from_value(pbr.get(key))
        if reference:
            references.append((slot, reference))
    return _load_material_library_texture_references(
        _dedupe_texture_references(references),
        library_path,
        source_identity=source_identity,
        images=images,
        seen_texture_paths=seen_texture_paths,
        search_roots=search_roots,
        material_name=material_name,
        library_label=library_label,
        archive_textures=archive_textures,
        archive_container=archive_container,
    )


def _dedupe_texture_references(references: list[tuple[str, str]]) -> list[tuple[str, str]]:
    deduped: list[tuple[str, str]] = []
    seen_slots: set[str] = set()
    for slot, reference in references:
        if slot in seen_slots:
            continue
        seen_slots.add(slot)
        deduped.append((slot, reference))
    return deduped


def _texture_reference_from_value(value: object) -> str:
    if isinstance(value, str):
        return _clean_source_texture_reference(value)
    if isinstance(value, dict):
        for key in ("uri", "path", "file", "filename", "source", "name"):
            reference = value.get(key)
            if isinstance(reference, str) and reference.strip():
                return _clean_source_texture_reference(reference)
    return ""


def _load_mtl_material_library(
    path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    color_space: str,
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    _ensure_loadable_file_size(path, _MAX_MATERIAL_LIBRARY_BYTES, "material library")
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError as exc:
        raise ValueError(f"material library could not be read: {path}") from exc
    entries = _mtl_material_entries(lines, str(path))
    specs, texture_stats = _material_specs_from_entries(
        entries,
        path,
        source_identity=source_identity,
        reference=reference,
        images=images,
        seen_texture_paths=seen_texture_paths,
        search_roots=search_roots,
        library_label=str(path),
        color_space=color_space,
    )
    return specs, texture_stats


def _mtl_material_entries(lines: list[str], library_label: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        keyword, _, rest = line.partition(" ")
        key = keyword.lower()
        value = rest.strip()
        if key == "newmtl":
            if current is not None:
                entries.append(current)
            current = {"name": value}
            continue
        if current is None:
            continue
        if key == "kd":
            current["base_color"] = value
        elif key == "d":
            current["opacity"] = value
        elif key == "tr":
            transparency = _optional_material_float(value)
            if transparency is not None:
                current["opacity"] = 1.0 - transparency
        elif key == "pm":
            current["metallic"] = value
        elif key == "pr":
            current["roughness"] = value
        elif key == "ns":
            try:
                shininess = _clamp01(float(value) / 1000.0)
            except ValueError:
                shininess = None
            if shininess is not None:
                current["roughness"] = max(0.04, 1.0 - shininess**0.5)
        elif key.startswith("map_") or key in {"bump", "norm"}:
            textures = cast(dict[str, object], current.setdefault("textures", {}))
            slot = _material_library_texture_slot(key)
            if slot is not None:
                textures[slot] = _mtl_texture_reference(value)
    if current is not None:
        entries.append(current)
    if not entries:
        raise ValueError(f"material library did not contain supported material records: {library_label}")
    return entries


def _material_specs_from_entries(
    entries: list[dict[str, object]],
    library_path: Path,
    *,
    source_identity: str,
    reference: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
    color_space: str = "auto",
) -> tuple[list[_MaterialLibrarySpec], dict[str, int]]:
    specs: list[_MaterialLibrarySpec] = []
    texture_stats = {"missing": 0, "unreadable": 0}
    for entry in entries:
        spec, stats = _json_material_spec(
            entry,
            library_path,
            source_identity=source_identity,
            reference=reference,
            images=images,
            seen_texture_paths=seen_texture_paths,
            search_roots=search_roots,
            library_label=library_label,
            archive_textures=archive_textures,
            archive_container=archive_container,
            color_space=color_space,
        )
        texture_stats["missing"] += stats["missing"]
        texture_stats["unreadable"] += stats["unreadable"]
        if spec is not None:
            specs.append(spec)
    return specs, texture_stats


def _mtl_texture_reference(value: str) -> str:
    tokens = [token for token in value.split() if token]
    if not tokens:
        return ""
    return _clean_source_texture_reference(tokens[-1])


def _material_library_texture_slot(key: str) -> str | None:
    normalized = key.lower().replace("-", "_")
    aliases = {
        "map_kd": "base_color",
        "map_ka": "base_color",
        "map_ke": "emissive",
        "map_d": "opacity",
        "map_bump": "normal",
        "bump": "normal",
        "norm": "normal",
        "map_pr": "roughness",
        "map_pm": "metallic",
        "map_orm": "metallic_roughness",
    }
    if normalized in aliases:
        return aliases[normalized]
    return _texture_slot(normalized)


def _resolve_archive_texture_reference(
    reference: str,
    archive_textures: _ArchiveTextureMap,
    library_path: Path,
) -> tuple[str, bytes] | None:
    library_parent = PurePosixPath(library_path.as_posix()).parent
    reference_path = PurePosixPath(reference.replace("\\", "/"))
    candidate_keys = [
        _archive_member_key(str(library_parent / reference_path)),
        _archive_member_key(str(reference_path)),
        _archive_member_key(reference_path.name),
    ]
    for key in candidate_keys:
        if key in archive_textures:
            return archive_textures[key]
    basename = reference_path.name.lower()
    basename_matches = [item for key, item in archive_textures.items() if PurePosixPath(key).name == basename]
    if len(basename_matches) == 1:
        return basename_matches[0]
    return None


def _load_material_library_texture_references(
    references: list[tuple[str, str]],
    library_path: Path,
    *,
    source_identity: str,
    images: dict[str, ImageResource],
    seen_texture_paths: set[str],
    search_roots: list[Path],
    material_name: str,
    library_label: str,
    archive_textures: _ArchiveTextureMap | None = None,
    archive_container: Path | None = None,
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    texture_images: list[tuple[str, str]] = []
    missing = 0
    unreadable = 0
    seen_references: set[tuple[str, str]] = set()
    for slot, reference in references:
        key = (slot, reference)
        if key in seen_references:
            continue
        seen_references.add(key)
        if archive_textures is not None and archive_container is not None:
            resolved_archive = _resolve_archive_texture_reference(reference, archive_textures, library_path)
            if resolved_archive is None:
                missing += 1
                continue
            member_name, data = resolved_archive
            identity = f"{archive_container.resolve()}!/{member_name}"
            path_label = f"{archive_container}!/{member_name}"
            try:
                image = _load_source_texture_data(
                    data,
                    suffix=Path(member_name).suffix.lower(),
                    name=Path(member_name).name,
                    source_identity=source_identity,
                    reference=reference,
                    path_label=path_label,
                    stable_identity=identity,
                )
            except ValueError:
                unreadable += 1
                continue
        else:
            texture_path = _resolve_source_texture(reference, search_roots)
            if texture_path is None:
                missing += 1
                continue
            identity = str(texture_path.resolve())
            try:
                image = _load_source_texture(texture_path, source_identity=source_identity, reference=reference)
            except ValueError:
                unreadable += 1
                continue
        if identity in seen_texture_paths:
            existing = next(
                (
                    image_id
                    for image_id, existing_image in images.items()
                    if str(existing_image.metadata.get("source_texture_identity", "")) == identity
                    or str(existing_image.metadata.get("source_texture_path", "")) == identity
                ),
                None,
            )
            if existing is not None:
                texture_images.append((slot, existing))
            continue
        seen_texture_paths.add(identity)
        image_id = _stable_id("img", f"{source_identity}:{library_label}:{identity}:{slot}:{material_name}")
        metadata = dict(image.metadata)
        metadata.update(
            {
                "source_texture_slot": slot,
                "source_texture_material_name": material_name,
                "material_library_path": library_label,
                "source_texture_identity": identity,
            }
        )
        if archive_container is not None:
            metadata["material_library_container"] = str(archive_container)
        images[image_id] = ImageResource(
            id=image_id,
            name=image.name,
            mime_type=image.mime_type,
            data=image.data,
            width=image.width,
            height=image.height,
            metadata=metadata,
        )
        texture_images.append((slot, image_id))
    return texture_images, {"missing": missing, "unreadable": unreadable}


def _apply_material_libraries_to_materials(
    materials: dict[str, Material],
    extraction: _MaterialLibraryExtraction,
) -> dict[str, int]:
    summary = _empty_material_library_binding_summary()
    summary["library_materials"] = len(extraction.materials)
    applied_materials: set[str] = set()
    for spec in extraction.materials:
        targets = _material_library_targets(materials, spec)
        if not targets:
            summary["unmatched_library_materials"] += 1
            continue
        summary["matched_library_materials"] += 1
        for material in targets:
            materials[material.id] = _material_with_library_spec(material, spec)
            applied_materials.add(material.id)
            summary["bound_textures"] += len(spec.texture_images)
    summary["applied_materials"] = len(applied_materials)
    return summary


def _material_library_targets(
    materials: dict[str, Material],
    spec: _MaterialLibrarySpec,
) -> list[Material]:
    if not materials:
        return []
    spec_key = _material_match_key(spec.name)
    exact: list[Material] = []
    fuzzy: list[Material] = []
    for material in materials.values():
        names = [
            material.name,
            str(material.metadata.get("cad_material_name", "")),
            str(material.metadata.get("material_library_name", "")),
        ]
        if spec_key and any(_material_match_key(name) == spec_key for name in names if name):
            exact.append(material)
            continue
        material_tokens = set().union(*(_name_tokens(name) for name in names if name))
        spec_tokens = set(_name_tokens(spec.name)) - _GENERIC_MATERIAL_TOKENS
        if spec_tokens and material_tokens.intersection(spec_tokens):
            fuzzy.append(material)
    if exact:
        return exact
    if fuzzy:
        return fuzzy
    return []


def _material_match_key(value: str) -> str:
    return "_".join(token for token in _name_tokens(value) if token not in _GENERIC_MATERIAL_TOKENS)


def _material_with_library_spec(material: Material, spec: _MaterialLibrarySpec) -> Material:
    opacity = material.opacity if spec.opacity is None else spec.opacity
    base_color = material.base_color if spec.base_color is None else spec.base_color
    if spec.opacity is not None:
        base_color = (base_color[0], base_color[1], base_color[2], min(base_color[3], opacity))
    metadata = {
        **material.metadata,
        **spec.metadata_dict(),
        "material_library_matched": "true",
        "material_library_material_name": spec.name,
    }
    for slot, image_id in spec.texture_images:
        metadata[f"source_texture_{slot}_image"] = image_id
        if slot in _SOURCE_TEXTURE_EXPORT_SLOTS:
            metadata.setdefault(f"source_texture_{slot}_image", image_id)
        existing = metadata.get("source_texture_slots")
        slots = set(str(existing).split(",")) if isinstance(existing, str) and existing else set()
        slots.add(slot)
        metadata["source_texture_slots"] = ",".join(sorted(slots))
    return Material(
        id=material.id,
        name=material.name,
        base_color=base_color,
        metallic=material.metallic if spec.metallic is None else spec.metallic,
        roughness=material.roughness if spec.roughness is None else spec.roughness,
        opacity=opacity,
        metadata=metadata,
    )


def _material_id(color: tuple[float, float, float, float]) -> str:
    encoded = ",".join(f"{component:.6f}" for component in color)
    return _stable_id("mat", encoded)


def _clamp_color(color: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        _clamp01(float(color[0])),
        _clamp01(float(color[1])),
        _clamp01(float(color[2])),
        _clamp01(float(color[3])),
    )


def _quantity_rgba_tuple(color: Any) -> tuple[float, float, float, float]:
    rgb = color.GetRGB()
    return (
        _clamp01(float(rgb.Red())),
        _clamp01(float(rgb.Green())),
        _clamp01(float(rgb.Blue())),
        _clamp01(float(color.Alpha())),
    )


def _quantity_color_tuple(color: Any) -> tuple[float, float, float]:
    return (_clamp01(float(color.Red())), _clamp01(float(color.Green())), _clamp01(float(color.Blue())))


def _is_default_material_color(color: tuple[float, float, float, float]) -> bool:
    return np.allclose(np.asarray(color[:3], dtype=float), np.asarray((0.75, 0.75, 0.75), dtype=float), atol=1e-6)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _ocp_string(value: Any) -> str:
    if value is None:
        return ""
    for method in ("ToExtString", "ToCString"):
        if hasattr(value, method):
            try:
                return str(getattr(value, method)()).strip()
            except Exception:
                continue
    return str(value).strip()
