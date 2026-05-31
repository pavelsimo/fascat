from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Literal

from fascat.metadata import Metadata

ImageMimeType = Literal["image/png", "image/jpeg", "image/ktx2"]


@dataclass(frozen=True)
class ImageResource:
    id: str
    name: str
    mime_type: ImageMimeType
    data: bytes
    width: int
    height: int
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("image id must not be empty")
        if self.mime_type not in {"image/png", "image/jpeg", "image/ktx2"}:
            raise ValueError("unsupported image mime type")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be greater than 0")
        object.__setattr__(self, "data", bytes(self.data))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def copy(self) -> ImageResource:
        return ImageResource(
            id=self.id,
            name=self.name,
            mime_type=self.mime_type,
            data=self.data,
            width=self.width,
            height=self.height,
            metadata=dict(self.metadata),
        )

    def data_uri(self) -> str:
        encoded = base64.b64encode(self.data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "mime_type": self.mime_type,
            "byte_length": len(self.data),
            "width": self.width,
            "height": self.height,
            "metadata": dict(self.metadata),
        }
