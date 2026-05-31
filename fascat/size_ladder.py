from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from fascat.asset import Asset
from fascat.options import GltfExportOptions, resolve_gltf_export_options


@dataclass(frozen=True)
class GltfSizeLadderVariant:
    name: str
    options: GltfExportOptions
    status: str
    file_size_bytes: int = 0
    error: str | None = None

    def to_dict(self, *, baseline_bytes: int) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "status": self.status,
            "file_size_bytes": self.file_size_bytes,
            "options": self.options.to_dict(),
        }
        if baseline_bytes > 0 and self.file_size_bytes > 0:
            payload["ratio_to_baseline"] = self.file_size_bytes / baseline_bytes
            payload["savings_bytes"] = max(0, baseline_bytes - self.file_size_bytes)
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class GltfSizeLadderReport:
    variants: tuple[GltfSizeLadderVariant, ...]
    warnings: tuple[str, ...] = ()

    def to_step_options(self) -> dict[str, object]:
        baseline_bytes = self.baseline_bytes
        return {
            "format": "glTF",
            "artifact": "compressed_glb",
            "variants": [variant.to_dict(baseline_bytes=baseline_bytes) for variant in self.variants],
        }

    def to_step_after(self, base_stats: dict[str, int]) -> dict[str, int]:
        measured = [
            variant for variant in self.variants if variant.status == "measured" and variant.file_size_bytes > 0
        ]
        sizes = [variant.file_size_bytes for variant in measured]
        baseline_bytes = self.baseline_bytes
        requested_bytes = self.requested_bytes
        smallest_bytes = min(sizes, default=0)
        return {
            **base_stats,
            "size_ladder_variants": len(self.variants),
            "size_ladder_measured_variants": len(measured),
            "size_ladder_unavailable_variants": sum(1 for variant in self.variants if variant.status != "measured"),
            "size_ladder_baseline_bytes": baseline_bytes,
            "size_ladder_requested_bytes": requested_bytes,
            "size_ladder_smallest_bytes": smallest_bytes,
            "size_ladder_best_savings_bytes": max(0, baseline_bytes - smallest_bytes) if smallest_bytes else 0,
        }

    @property
    def baseline_bytes(self) -> int:
        return _variant_size(self.variants, "baseline")

    @property
    def requested_bytes(self) -> int:
        return _variant_size(self.variants, "requested")


def measure_gltf_size_ladder(
    asset: Asset,
    *,
    options: GltfExportOptions | None = None,
) -> GltfSizeLadderReport:
    opts = resolve_gltf_export_options(options)
    warnings: list[str] = []
    variants: list[GltfSizeLadderVariant] = []
    with tempfile.TemporaryDirectory(prefix="fascat-size-ladder-") as directory:
        workdir = Path(directory)
        for name, variant_options in _size_ladder_variants(opts):
            output = workdir / f"{name}.glb"
            try:
                _write_gltf_variant(asset, output, variant_options)
            except Exception as exc:
                message = f"glTF size ladder variant {name} could not be measured: {exc}"
                warnings.append(message)
                variants.append(
                    GltfSizeLadderVariant(
                        name=name,
                        options=variant_options,
                        status="unavailable",
                        error=str(exc) or exc.__class__.__name__,
                    )
                )
                continue
            variants.append(
                GltfSizeLadderVariant(
                    name=name,
                    options=variant_options,
                    status="measured",
                    file_size_bytes=output.stat().st_size,
                )
            )
    return GltfSizeLadderReport(variants=tuple(variants), warnings=tuple(warnings))


def _size_ladder_variants(options: GltfExportOptions) -> list[tuple[str, GltfExportOptions]]:
    metadata = options.metadata
    variants = [
        ("baseline", GltfExportOptions(metadata=metadata)),
        ("quantized", GltfExportOptions(quantize=True, metadata=metadata)),
        ("meshopt", GltfExportOptions(quantize=True, meshopt=True, metadata=metadata)),
        ("draco", GltfExportOptions(draco=True, metadata=metadata)),
    ]
    if options.texture_compression is not None:
        variants.append(
            (
                "texture_compressed",
                GltfExportOptions(
                    quantize=True,
                    meshopt=True,
                    texture_compression=options.texture_compression,
                    texture_fallback_format=options.texture_fallback_format,
                    png_compression=options.png_compression,
                    jpeg_quality=options.jpeg_quality,
                    metadata=metadata,
                ),
            )
        )
    variants.append(("requested", replace(options, size_ladder=False)))
    return variants


def _write_gltf_variant(asset: Asset, output: Path, options: GltfExportOptions) -> None:
    from fascat.io.gltf import write_gltf

    write_gltf(asset, output, options=options)


def _variant_size(variants: tuple[GltfSizeLadderVariant, ...], name: str) -> int:
    for variant in variants:
        if variant.name == name:
            return variant.file_size_bytes
    return 0
