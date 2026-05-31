from __future__ import annotations

from io import BytesIO

from PIL import Image

from fascat.asset import Asset, Node
from fascat.image import ImageResource
from fascat.material import Material
from fascat.options import TextureProcessOptions


def _png_bytes(mode: str, color: tuple[int, ...], size: tuple[int, int] = (4, 4)) -> bytes:
    image = Image.new(mode, size, color)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_process_textures_resizes_converts_dedupes_and_rewrites_material_refs() -> None:
    image_data = _png_bytes("RGB", (200, 20, 10))
    asset = Asset(
        root=Node(id="root", name="root"),
        images={
            "paint_a": ImageResource(
                id="paint_a",
                name="Paint A",
                mime_type="image/png",
                data=image_data,
                width=4,
                height=4,
            ),
            "paint_b": ImageResource(
                id="paint_b",
                name="Paint B",
                mime_type="image/png",
                data=image_data,
                width=4,
                height=4,
            ),
        },
        materials={
            "mat": Material(
                id="mat",
                name="Mat",
                base_color=(1.0, 1.0, 1.0, 1.0),
                metadata={"baked_texture_base_color_image": "paint_b"},
            )
        },
    )

    processed = asset.process_textures(TextureProcessOptions(max_resolution=2, fallback_format="jpeg"))
    image = processed.images["paint_a"]

    assert sorted(processed.images) == ["paint_a"]
    assert image.mime_type == "image/jpeg"
    assert image.width == 2
    assert image.height == 2
    assert image.metadata["texture_process_resized"] == "true"
    assert image.metadata["texture_process_converted"] == "true"
    assert processed.materials["mat"].metadata["baked_texture_base_color_image"] == "paint_a"
    assert processed.metadata["texture_process_resized_images"] == "2"
    assert processed.metadata["texture_process_deduped_images"] == "1"
    assert processed.report.steps[-1].name == "process_textures"


def test_process_textures_auto_keeps_alpha_safe_png() -> None:
    image_data = _png_bytes("RGBA", (255, 0, 0, 128))
    asset = Asset(
        root=Node(id="root", name="root"),
        images={
            "alpha": ImageResource(
                id="alpha",
                name="Alpha",
                mime_type="image/png",
                data=image_data,
                width=4,
                height=4,
            )
        },
    )

    processed = asset.process_textures(TextureProcessOptions(fallback_format="auto"))

    assert processed.images["alpha"].mime_type == "image/png"
    assert processed.metadata["texture_process_alpha_flattened_images"] == "0"
    assert processed.report.steps[-1].warnings == []
