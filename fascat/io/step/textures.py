from __future__ import annotations

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote, urlparse

from fascat.image import ImageMimeType, ImageResource
from fascat.io._import_base import _stable_id
from fascat.io.step.records import _decode_step_string, _ensure_loadable_file_size, _name_tokens, _step_scan_capped
from fascat.material import Material
from fascat.metadata import Metadata
from fascat.options import StepReadOptions

_ArchiveTextureMap = dict[str, tuple[str, bytes]]


_SOURCE_TEXTURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".ktx2"}


_SOURCE_TEXTURE_REF_RE = re.compile(r"'([^']+\.(?:png|jpe?g|ktx2)(?:[#?][^']*)?)'", re.IGNORECASE)


_KTX2_IDENTIFIER = b"\xabKTX 20\xbb\r\n\x1a\n"


_KTX2_HEADER_BYTES = 80


_MAX_SOURCE_TEXTURE_BYTES = 64 * 1024 * 1024


_GENERIC_MATERIAL_TOKENS = {"cad", "color", "material", "mat", "texture", "map", "source"}


_TEXTURE_SLOT_TOKENS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("metallic_roughness", ("metallicroughness", "metalrough", "metallic_roughness", "orm")),
    ("base_color", ("basecolor", "base_color", "albedo", "diffuse", "color", "colour")),
    ("normal", ("normal", "norm", "nrm")),
    ("occlusion", ("occlusion", "ambientocclusion", "ambient_occlusion", "ao")),
    ("roughness", ("roughness", "rough")),
    ("metallic", ("metallic", "metalness", "metal")),
    ("opacity", ("opacity", "alpha", "transparency")),
    ("emissive", ("emissive", "emission", "emit")),
)


_SOURCE_TEXTURE_EXPORT_SLOTS = {"base_color", "metallic_roughness", "normal", "occlusion", "emissive"}


@dataclass
class _SourceTextureExtraction:
    images: dict[str, ImageResource]
    summary: dict[str, int]
    warnings: list[str]


def _confine_to_search_roots(candidate: Path, search_roots: list[Path]) -> bool:
    """True when the candidate's resolved path stays inside at least one search root.

    References come from untrusted STEP file content; resolving both sides
    rejects ``..`` traversal and symlink escapes while still allowing
    references that re-enter a configured root.
    """
    try:
        resolved = candidate.resolve()
    except OSError:
        return False
    for root in search_roots:
        try:
            if resolved.is_relative_to(root.resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def _extract_source_textures(source: Path, source_identity: str, options: StepReadOptions) -> _SourceTextureExtraction:
    if not options.source_textures:
        return _SourceTextureExtraction(
            images={},
            summary={"references": 0, "resolved": 0, "missing": 0, "unsupported": 0, "unreadable": 0},
            warnings=[],
        )

    references = _source_texture_references(source)
    search_roots = [source.parent, *(Path(path) for path in options.source_texture_search_paths)]
    images: dict[str, ImageResource] = {}
    seen_paths: set[Path] = set()
    missing = 0
    unsupported = 0
    unreadable = 0
    warnings: list[str] = []
    for reference in references:
        texture_path = _resolve_source_texture(reference, search_roots)
        if texture_path is None:
            missing += 1
            warnings.append(f"source texture reference could not be resolved: {reference}")
            continue
        suffix = texture_path.suffix.lower()
        if suffix not in _SOURCE_TEXTURE_SUFFIXES:
            unsupported += 1
            continue
        resolved = texture_path.resolve()
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)
        try:
            image = _load_source_texture(texture_path, source_identity=source_identity, reference=reference)
        except ValueError as exc:
            unreadable += 1
            warnings.append(str(exc))
            continue
        images[image.id] = image
    return _SourceTextureExtraction(
        images=images,
        summary={
            "references": len(references),
            "resolved": len(images),
            "missing": missing,
            "unsupported": unsupported,
            "unreadable": unreadable,
        },
        warnings=warnings,
    )


def _source_texture_references(source: Path) -> list[str]:
    if _step_scan_capped(source):
        return []
    try:
        text = source.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    references: list[str] = []
    seen: set[str] = set()
    for match in _SOURCE_TEXTURE_REF_RE.finditer(text):
        reference = _clean_source_texture_reference(match.group(1))
        if not reference or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    return references


def _clean_source_texture_reference(reference: str) -> str:
    value = _decode_step_string(reference.replace("''", "'")).strip().strip('"<>')
    if not value:
        return ""
    if value.lower().startswith("file://"):
        parsed = urlparse(value)
        value = unquote(parsed.path)
    value = value.split("#", 1)[0].split("?", 1)[0].replace("\\", "/")
    return value.strip()


def _resolve_source_texture(reference: str, search_roots: list[Path]) -> Path | None:
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
            if item.is_file():
                return item
        except OSError:
            continue
    return None


def _load_source_texture(path: Path, *, source_identity: str, reference: str) -> ImageResource:
    _ensure_loadable_file_size(path, _MAX_SOURCE_TEXTURE_BYTES, "source texture")
    data = path.read_bytes()
    return _load_source_texture_data(
        data,
        suffix=path.suffix.lower(),
        name=path.name,
        source_identity=source_identity,
        reference=reference,
        path_label=str(path),
        stable_identity=str(path.resolve()),
    )


def _load_source_texture_data(
    data: bytes,
    *,
    suffix: str,
    name: str,
    source_identity: str,
    reference: str,
    path_label: str,
    stable_identity: str,
) -> ImageResource:
    if suffix == ".ktx2":
        size = _ktx2_dimensions(data)
        if size is None:
            raise ValueError(f"source texture could not be read as KTX2: {path_label}")
        width, height = size
        mime_type: ImageMimeType = "image/ktx2"
    else:
        try:
            from PIL import Image

            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
                mime_type = "image/png" if image.format == "PNG" else "image/jpeg"
        except Exception as exc:
            raise ValueError(f"source texture could not be read: {path_label}") from exc
    digest = _stable_id("img", f"{source_identity}:{stable_identity}:{len(data)}")
    slot = _texture_slot(Path(name).stem)
    metadata: Metadata = {
        "source_texture": "true",
        "source_texture_reference": reference,
        "source_texture_path": path_label,
        "source_texture_identity": stable_identity,
        "source_texture_slot": slot or "unknown",
    }
    return ImageResource(
        id=digest,
        name=name,
        mime_type=mime_type,
        data=data,
        width=width,
        height=height,
        metadata=metadata,
    )


def _ktx2_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < _KTX2_HEADER_BYTES or data[: len(_KTX2_IDENTIFIER)] != _KTX2_IDENTIFIER:
        return None
    width = int.from_bytes(data[20:24], "little")
    height = int.from_bytes(data[24:28], "little")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _attach_source_textures_to_materials(
    materials: dict[str, Material],
    images: dict[str, ImageResource],
) -> dict[str, int]:
    bound_images = 0
    bound_materials: set[str] = set()
    unbound_images = 0
    for image in images.values():
        slot = str(image.metadata.get("source_texture_slot", "unknown"))
        if slot == "unknown":
            unbound_images += 1
            continue
        targets = _source_texture_material_targets(materials, image)
        if not targets:
            unbound_images += 1
            continue
        export_slot = slot if slot in _SOURCE_TEXTURE_EXPORT_SLOTS else None
        for material in targets:
            material.metadata[f"source_texture_{slot}_image"] = image.id
            material.metadata[f"source_texture_{slot}_name"] = image.name
            if export_slot is not None:
                material.metadata.setdefault(f"source_texture_{export_slot}_image", image.id)
            existing = material.metadata.get("source_texture_slots")
            slots = set(str(existing).split(",")) if isinstance(existing, str) and existing else set()
            slots.add(slot)
            material.metadata["source_texture_slots"] = ",".join(sorted(slots))
            bound_materials.add(material.id)
        bound_images += 1
    return {
        "bound_images": bound_images,
        "bound_materials": len(bound_materials),
        "unbound_images": unbound_images,
    }


def _source_texture_material_targets(materials: dict[str, Material], image: ImageResource) -> list[Material]:
    if len(materials) == 1:
        return list(materials.values())
    image_tokens = set(_name_tokens(Path(image.name).stem))
    targets: list[Material] = []
    for material in materials.values():
        material_tokens = set(_name_tokens(material.name))
        material_tokens.update(_name_tokens(str(material.metadata.get("cad_material_name", ""))))
        material_tokens.update(token for token in _name_tokens(material.id) if token not in _GENERIC_MATERIAL_TOKENS)
        if image_tokens.intersection(material_tokens - _GENERIC_MATERIAL_TOKENS):
            targets.append(material)
    return targets


def _texture_slot(name: str) -> str | None:
    normalized = "".join(_name_tokens(name))
    tokens = set(_name_tokens(name))
    for slot, aliases in _TEXTURE_SLOT_TOKENS:
        for alias in aliases:
            if alias in normalized or alias in tokens:
                return slot
    return None
