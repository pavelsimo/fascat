from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float = 0.0
    roughness: float = 0.5
    opacity: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))
        if len(self.base_color) != 4:
            raise ValueError("base_color must contain RGBA values")
        if any(value < 0.0 or value > 1.0 for value in self.base_color):
            raise ValueError("base_color values must be between 0 and 1")
        if self.metallic < 0.0 or self.metallic > 1.0:
            raise ValueError("metallic must be between 0 and 1")
        if self.roughness < 0.0 or self.roughness > 1.0:
            raise ValueError("roughness must be between 0 and 1")
        if self.opacity < 0.0 or self.opacity > 1.0:
            raise ValueError("opacity must be between 0 and 1")

    def copy(self) -> Material:
        return Material(
            id=self.id,
            name=self.name,
            base_color=self.base_color,
            metallic=self.metallic,
            roughness=self.roughness,
            opacity=self.opacity,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "base_color": list(self.base_color),
            "metallic": self.metallic,
            "roughness": self.roughness,
            "opacity": self.opacity,
            "metadata": dict(self.metadata),
        }
