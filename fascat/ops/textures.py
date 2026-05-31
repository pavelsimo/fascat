from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import cast

from PIL import Image

from fascat.asset import Asset
from fascat.image import ImageMimeType, ImageResource
from fascat.options import TextureProcessOptions


@dataclass(frozen=True)
class _ProcessedImage:
    image: ImageResource
    resized: bool = False
    converted: bool = False
    alpha_flattened: bool = False
    skipped: bool = False


def process_textures_asset(asset: Asset, options: TextureProcessOptions) -> Asset:
    result = asset.copy(keep_source=True)
    processed: dict[str, ImageResource] = {}
    resized = 0
    converted = 0
    skipped = 0
    alpha_flattened = 0
    for image_id, image in result.images.items():
        item = _process_image(image, options)
        processed[image_id] = item.image
        resized += int(item.resized)
        converted += int(item.converted)
        skipped += int(item.skipped)
        alpha_flattened += int(item.alpha_flattened)

    dedupe_map: dict[str, str] = {}
    if options.dedupe:
        processed, dedupe_map = _dedupe_images(processed)
    if dedupe_map:
        _rewrite_material_image_references(result, dedupe_map)

    result.images = processed
    result.metadata.update(
        {
            "texture_process_source_images": str(len(asset.images)),
            "texture_process_output_images": str(len(result.images)),
            "texture_process_resized_images": str(resized),
            "texture_process_converted_images": str(converted),
            "texture_process_deduped_images": str(len(dedupe_map)),
            "texture_process_skipped_images": str(skipped),
            "texture_process_alpha_flattened_images": str(alpha_flattened),
            "texture_process_fallback_format": options.fallback_format,
            "texture_process_max_resolution": "none" if options.max_resolution is None else str(options.max_resolution),
        }
    )
    if alpha_flattened:
        result.report.add_warning(
            f"texture processing flattened alpha for {alpha_flattened} image(s) because JPEG fallback was requested"
        )
    return result


def _process_image(image: ImageResource, options: TextureProcessOptions) -> _ProcessedImage:
    if image.mime_type == "image/ktx2":
        return _ProcessedImage(
            image.copy(),
            skipped=True,
        )
    with Image.open(BytesIO(image.data)) as opened:
        opened.load()
        has_alpha = _has_alpha(opened)
        target_format = _target_format(opened, options, has_alpha)
        working = opened.convert("RGBA" if has_alpha else "RGB")
        original_size = working.size
        resized = False
        if options.max_resolution is not None and max(working.size) > options.max_resolution:
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            working.thumbnail((options.max_resolution, options.max_resolution), resampling)
            resized = working.size != original_size
        alpha_flattened = False
        if target_format == "JPEG" and _has_alpha(working):
            background = Image.new("RGB", working.size, (255, 255, 255))
            background.paste(working, mask=working.getchannel("A"))
            working = background
            alpha_flattened = True
        elif target_format == "JPEG":
            working = working.convert("RGB")
        else:
            working = working.convert("RGBA" if _has_alpha(working) else "RGB")

        output = BytesIO()
        if target_format == "PNG":
            working.save(output, format="PNG", compress_level=options.png_compression)
            mime_type: ImageMimeType = "image/png"
        else:
            working.save(output, format="JPEG", quality=options.jpeg_quality, optimize=True)
            mime_type = "image/jpeg"
        data = output.getvalue()
        converted = mime_type != image.mime_type
        metadata = {
            **image.metadata,
            "texture_process_original_mime_type": image.mime_type,
            "texture_process_original_width": str(image.width),
            "texture_process_original_height": str(image.height),
            "texture_process_fallback_format": options.fallback_format,
            "texture_process_resized": str(resized).lower(),
            "texture_process_converted": str(converted).lower(),
        }
        if alpha_flattened:
            metadata["texture_process_alpha_flattened"] = "true"
        return _ProcessedImage(
            ImageResource(
                id=image.id,
                name=image.name,
                mime_type=mime_type,
                data=data,
                width=int(working.width),
                height=int(working.height),
                metadata=metadata,
            ),
            resized=resized,
            converted=converted,
            alpha_flattened=alpha_flattened,
        )


def _target_format(image: Image.Image, options: TextureProcessOptions, has_alpha: bool) -> str:
    if options.fallback_format == "png":
        return "PNG"
    if options.fallback_format == "jpeg":
        return "JPEG"
    if has_alpha:
        return "PNG"
    if image.format == "PNG" and image.mode in {"P", "L", "LA"}:
        return "PNG"
    return "JPEG"


def _has_alpha(image: Image.Image) -> bool:
    if image.mode in {"RGBA", "LA"}:
        extrema = cast(tuple[int, int], image.getchannel("A").getextrema())
        return extrema[0] < 255
    return image.mode == "P" and "transparency" in image.info


def _dedupe_images(images: dict[str, ImageResource]) -> tuple[dict[str, ImageResource], dict[str, str]]:
    kept: dict[str, ImageResource] = {}
    first_by_hash: dict[str, str] = {}
    dedupe_map: dict[str, str] = {}
    for image_id, image in images.items():
        key = _image_content_key(image)
        existing = first_by_hash.get(key)
        if existing is None:
            first_by_hash[key] = image_id
            kept[image_id] = image
            continue
        dedupe_map[image_id] = existing
    return kept, dedupe_map


def _image_content_key(image: ImageResource) -> str:
    if image.mime_type == "image/ktx2":
        return "ktx2:" + hashlib.sha1(image.data).hexdigest()
    try:
        with Image.open(BytesIO(image.data)) as opened:
            rgba = opened.convert("RGBA")
            digest = hashlib.sha1()
            digest.update(str(rgba.size).encode("ascii"))
            digest.update(rgba.tobytes())
            return digest.hexdigest()
    except Exception:
        return "bytes:" + hashlib.sha1(image.data).hexdigest()


def _rewrite_material_image_references(asset: Asset, dedupe_map: dict[str, str]) -> None:
    for material in asset.materials.values():
        for key, value in list(material.metadata.items()):
            if isinstance(value, str) and value in dedupe_map:
                material.metadata[key] = dedupe_map[value]
