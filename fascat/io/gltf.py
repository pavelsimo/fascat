from __future__ import annotations

import base64
import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset, Node, Part
from fascat.export_report import referenced_materials
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.metadata import pmi_ids_by_part
from fascat.options import GltfExportOptions, MetadataExportOptions

GLTF_SUFFIXES = {".gltf", ".glb"}
BinaryPayload = bytes | bytearray

_GLB_MAGIC = b"glTF"
_GLB_VERSION = 2
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_BYTE = 5120
_UNSIGNED_BYTE = 5121
_SHORT = 5122
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125
_KHR_MESH_QUANTIZATION = "KHR_mesh_quantization"
_EXT_MESHOPT_COMPRESSION = "EXT_meshopt_compression"
_KHR_DRACO_MESH_COMPRESSION = "KHR_draco_mesh_compression"
_KHR_TEXTURE_BASISU = "KHR_texture_basisu"
_MSFT_LOD = "MSFT_lod"
_FASCAT_EXTRAS = "extras.fascat"
_RUNTIME_MATRIX_EXTENSIONS = (
    _KHR_MESH_QUANTIZATION,
    _EXT_MESHOPT_COMPRESSION,
    _MSFT_LOD,
    _KHR_DRACO_MESH_COMPRESSION,
    _KHR_TEXTURE_BASISU,
    _FASCAT_EXTRAS,
)
_BAKED_TEXTURE_METADATA_KEYS = (
    "baked_texture_base_color_uri",
    "baked_texture_metallic_roughness_uri",
    "baked_texture_normal_uri",
    "baked_texture_occlusion_uri",
    "baked_texture_emissive_uri",
)

_COMPONENT_SIZES = {
    _BYTE: 1,
    _UNSIGNED_BYTE: 1,
    _UNSIGNED_SHORT: 2,
    _SHORT: 2,
    _UNSIGNED_INT: 4,
    _FLOAT: 4,
}
_ACCESSOR_WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


@dataclass(frozen=True)
class _RuntimeTarget:
    label: str
    notes: tuple[str, ...]
    support: dict[str, str]


_RUNTIME_TARGETS = {
    "unity_gltfast": _RuntimeTarget(
        label="Unity glTFast",
        notes=(
            "prefer GLB for Unity runtime import",
            "validate installed package support for optional extensions before shipping",
        ),
        support={
            _KHR_MESH_QUANTIZATION: "loader must support quantized accessors when this required extension is emitted",
            _EXT_MESHOPT_COMPRESSION: "optional; decoder support enables compressed bufferView payloads",
            _MSFT_LOD: "optional handoff metadata; map to engine LOD groups during import if the loader ignores it",
            _KHR_DRACO_MESH_COMPRESSION: "future encoder output must be matched with a Draco-capable loader",
            _KHR_TEXTURE_BASISU: "future KTX2/Basis output must be matched with a texture-transcoding loader",
            _FASCAT_EXTRAS: "safe metadata channel for custom Unity import scripts",
        },
    ),
    "web": _RuntimeTarget(
        label="Web",
        notes=(
            "prefer GLB for browser delivery",
            "test exact loader extension support before relying on compressed payloads",
        ),
        support={
            _KHR_MESH_QUANTIZATION: "requires a quantization-capable glTF loader",
            _EXT_MESHOPT_COMPRESSION: "optional; fallback buffer data remains available to generic loaders",
            _MSFT_LOD: "optional; app code may need to interpret LOD metadata",
            _KHR_DRACO_MESH_COMPRESSION: "future encoder output must be paired with a Draco decoder",
            _KHR_TEXTURE_BASISU: "future texture output should use KTX2/Basis with PNG/JPEG fallbacks when needed",
            _FASCAT_EXTRAS: "generic web loaders may ignore Fascat extras",
        },
    ),
    "mobile": _RuntimeTarget(
        label="Mobile",
        notes=(
            "prioritize small geometry and texture payloads",
            "keep conservative fallbacks for older embedded loaders",
        ),
        support={
            _KHR_MESH_QUANTIZATION: "use when loader support and precision loss are validated on target devices",
            _EXT_MESHOPT_COMPRESSION: "optional; fallback buffers trade larger files for broader compatibility",
            _MSFT_LOD: "optional; app or engine integration should author native runtime LOD groups",
            _KHR_DRACO_MESH_COMPRESSION: "future encoder output must account for mobile decode cost",
            _KHR_TEXTURE_BASISU: "future KTX2/Basis output should honor mobile texture budget presets",
            _FASCAT_EXTRAS: "safe to preserve for diagnostics; ignored by normal rendering paths",
        },
    ),
    "xr": _RuntimeTarget(
        label="XR",
        notes=(
            "validate visual quality and decode cost on headset-class hardware",
            "prefer predictable fallbacks for performance-critical scenes",
        ),
        support={
            _KHR_MESH_QUANTIZATION: "requires loader support and visual validation for close-view CAD edges",
            _EXT_MESHOPT_COMPRESSION: "optional; fallback buffers avoid hard dependency on decoder setup",
            _MSFT_LOD: "optional; convert to engine-native LOD groups for reliable runtime switching",
            _KHR_DRACO_MESH_COMPRESSION: "future encoder output must be tested for headset decode latency",
            _KHR_TEXTURE_BASISU: "future KTX2/Basis output should be profiled against XR texture-memory budgets",
            _FASCAT_EXTRAS: "metadata can drive custom XR import or validation tools",
        },
    ),
}


def write_gltf(asset: Asset, path: str | Path, *, options: GltfExportOptions | None = None) -> None:
    _write_gltf(asset, path, options=options, validate=False)


def write_gltf_with_validation(
    asset: Asset,
    path: str | Path,
    *,
    options: GltfExportOptions | None = None,
) -> dict[str, int]:
    validation_stats = _write_gltf(asset, path, options=options, validate=True)
    assert validation_stats is not None
    return validation_stats


def _write_gltf(
    asset: Asset,
    path: str | Path,
    *,
    options: GltfExportOptions | None,
    validate: bool,
) -> dict[str, int] | None:
    opts = options or GltfExportOptions()
    output_path = Path(path)
    suffix = output_path.suffix.lower()
    if suffix not in GLTF_SUFFIXES:
        raise ValueError(f"unsupported glTF extension: {output_path.suffix or '<none>'}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document, binary = _build_document(asset, metadata_options=opts.metadata, quantize=opts.quantize)
    _apply_export_options(document, opts)
    if opts.meshopt:
        binary = _apply_meshopt_compression(document, binary)
    validation_stats = validate_gltf_document(document, binary) if validate else None
    if suffix == ".gltf":
        _embed_binary_uri(document, binary)
    if suffix == ".glb":
        output_path.write_bytes(_pack_glb(document, binary))
        return validation_stats
    output_path.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return validation_stats


def runtime_dependency_report(asset: Asset, options: GltfExportOptions | None = None) -> dict[str, object]:
    opts = options or GltfExportOptions()
    extensions_used: list[str] = []
    extensions_required: list[str] = []
    expected_support: dict[str, str] = {
        _FASCAT_EXTRAS: "optional glTF extras; generic loaders may ignore Fascat metadata",
    }
    has_meshes = _has_exportable_meshes(asset)
    if has_meshes and opts.quantize:
        extensions_used.append(_KHR_MESH_QUANTIZATION)
        extensions_required.append(_KHR_MESH_QUANTIZATION)
        expected_support[_KHR_MESH_QUANTIZATION] = "required extension; runtime loader must support quantized accessors"
    if has_meshes and opts.meshopt:
        extensions_used.append(_EXT_MESHOPT_COMPRESSION)
        expected_support[_EXT_MESHOPT_COMPRESSION] = (
            "optional extension with fallback buffer data; decoder support enables compressed payload use"
        )
    if _has_exportable_lods(asset):
        extensions_used.append(_MSFT_LOD)
        expected_support[_MSFT_LOD] = "optional extension; runtimes without support can load LOD0 only"
    return {
        "extensions_used": extensions_used,
        "extensions_required": extensions_required,
        "extras": {
            "fascat": True,
            "metadata": opts.metadata.mode,
            "pmi": opts.metadata.pmi,
        },
        "expected_runtime_support": expected_support,
        "runtime_compatibility": _runtime_compatibility_matrix(
            extensions_used=extensions_used,
            extensions_required=extensions_required,
        ),
        "runtime_decision_matrix": _runtime_decision_matrix(
            asset,
            opts,
            extensions_used=extensions_used,
        ),
        "not_written": {
            _KHR_DRACO_MESH_COMPRESSION: "unsupported; draco=True is rejected before export",
            _KHR_TEXTURE_BASISU: "unsupported; texture_compression is rejected before export",
        },
    }


def _runtime_compatibility_matrix(
    *,
    extensions_used: Sequence[str],
    extensions_required: Sequence[str],
) -> dict[str, object]:
    states = {
        extension: _runtime_extension_state(extension, extensions_used, extensions_required)
        for extension in _RUNTIME_MATRIX_EXTENSIONS
    }
    return {
        target_id: {
            "label": target.label,
            "notes": list(target.notes),
            "extensions": {
                extension: {
                    "state": states[extension],
                    "support": target.support[extension],
                    "fallback": _runtime_extension_fallback(extension),
                }
                for extension in _RUNTIME_MATRIX_EXTENSIONS
            },
        }
        for target_id, target in _RUNTIME_TARGETS.items()
    }


def _runtime_extension_state(
    extension: str,
    extensions_used: Sequence[str],
    extensions_required: Sequence[str],
) -> str:
    if extension == _FASCAT_EXTRAS:
        return "metadata"
    if extension in extensions_required:
        return "required"
    if extension in extensions_used:
        return "optional"
    if extension in {_KHR_DRACO_MESH_COMPRESSION, _KHR_TEXTURE_BASISU}:
        return "not_written"
    return "not_used"


def _runtime_extension_fallback(extension: str) -> str:
    if extension == _KHR_MESH_QUANTIZATION:
        return "no fallback when emitted; the extension is marked required"
    if extension == _EXT_MESHOPT_COMPRESSION:
        return "fallback buffer data is included for loaders that ignore meshopt"
    if extension == _MSFT_LOD:
        return "LOD0 remains loadable when the runtime ignores MSFT_lod"
    if extension in {_KHR_DRACO_MESH_COMPRESSION, _KHR_TEXTURE_BASISU}:
        return "not written by Fascat yet; requests are rejected before export"
    return "generic loaders may ignore Fascat extras without breaking geometry"


def _runtime_decision_matrix(
    asset: Asset,
    options: GltfExportOptions,
    *,
    extensions_used: Sequence[str],
) -> dict[str, object]:
    has_meshes = _has_exportable_meshes(asset)
    has_textures = _has_exportable_textures(asset)
    fallback_policy = _runtime_texture_fallback_policy(asset, options)
    return {
        "geometry": {
            "quantization": {
                "extension": _KHR_MESH_QUANTIZATION,
                "state": _method_state(
                    enabled=_KHR_MESH_QUANTIZATION in extensions_used,
                    available=has_meshes,
                    enabled_state="enabled_required",
                ),
                "best_for": "smaller vertex payloads when loader support and CAD-edge precision are validated",
                "tradeoff": "required extension when emitted; generic loaders without support cannot load the asset",
                "recommendation": (
                    "enable for web, mobile, or XR only after validating visual tolerance on the target loader"
                    if not options.quantize
                    else "validate quantized CAD edges on close-view target devices"
                ),
            },
            "meshopt": {
                "extension": _EXT_MESHOPT_COMPRESSION,
                "state": _method_state(
                    enabled=_EXT_MESHOPT_COMPRESSION in extensions_used,
                    available=has_meshes,
                    enabled_state="enabled_optional",
                ),
                "best_for": "web and mobile delivery where fast decode and broad fallback compatibility matter",
                "tradeoff": "fallback buffer data increases written file size but keeps generic loaders usable",
                "recommendation": (
                    "prefer meshopt over future Draco when decode speed and fallback compatibility matter"
                    if not options.meshopt
                    else "keep fallback validation enabled for runtimes that ignore meshopt payloads"
                ),
            },
            "draco": {
                "extension": _KHR_DRACO_MESH_COMPRESSION,
                "state": "unsupported",
                "best_for": "smallest geometry payloads when a Draco-capable loader and decode budget are proven",
                "tradeoff": "requires decoder setup and can add noticeable startup/decode cost on mobile or XR",
                "recommendation": "not selectable until Fascat integrates a reliable Draco encoder backend",
            },
        },
        "textures": {
            "ktx2_basisu": {
                "extension": _KHR_TEXTURE_BASISU,
                "state": "unsupported",
                "best_for": "GPU texture delivery on web, mobile, and XR after real image assets exist",
                "tradeoff": "requires texture transcoding support and target-specific quality settings",
                "recommendation": "not selectable until Fascat has a first-class image graph and KTX2/Basis encoder",
            },
            "png_jpeg_fallbacks": {
                "extension": None,
                "state": "source_textures_present" if has_textures else "no_texture_payload",
                "best_for": "broad compatibility when KTX2/Basis is unavailable or unsupported by the runtime",
                "tradeoff": "larger downloads and texture memory than GPU-native compressed textures",
                "fallback_format": options.texture_fallback_format,
                "resolved_format": fallback_policy["resolved_format"],
                "png_compression": options.png_compression,
                "jpeg_quality": options.jpeg_quality,
                "alpha_texture_sets": fallback_policy["alpha_texture_sets"],
                "color_only_texture_sets": fallback_policy["color_only_texture_sets"],
                "jpeg_alpha_risk_sets": fallback_policy["jpeg_alpha_risk_sets"],
                "recommendation": (
                    fallback_policy["recommendation"]
                    if has_textures
                    else "no fallback texture files are needed because the asset has no texture payload"
                ),
            },
        },
        "targets": {
            "unity_gltfast": {
                "preferred_container": "glb",
                "geometry": "use quantization or meshopt only after confirming installed glTFast extension support",
                "textures": "prefer KTX2/Basis later, with PNG/JPEG fallbacks for unsupported projects",
            },
            "web": {
                "preferred_container": "glb",
                "geometry": "prefer meshopt for fast browser decode; use Draco only when smallest download matters more",
                "textures": "prefer KTX2/Basis later, while keeping PNG/JPEG fallbacks for loader coverage",
            },
            "mobile": {
                "preferred_container": "glb",
                "geometry": "prefer meshopt or quantization after device tests; treat future Draco decode cost carefully",
                "textures": "use KTX2/Basis later with profile texture limits and PNG/JPEG fallbacks as needed",
            },
            "xr": {
                "preferred_container": "glb",
                "geometry": "prefer predictable meshopt/quantization paths over decode-heavy future Draco payloads",
                "textures": "profile KTX2/Basis transcoding and memory before removing PNG/JPEG fallbacks",
            },
        },
    }


def _method_state(*, enabled: bool, available: bool, enabled_state: str) -> str:
    if enabled:
        return enabled_state
    return "available_not_requested" if available else "not_applicable"


def _has_exportable_meshes(asset: Asset) -> bool:
    for part in asset.parts.values():
        meshes = (part.mesh, *part.lod_meshes)
        if any(mesh is not None and mesh.vertex_count > 0 for mesh in meshes):
            return True
    return False


def _has_exportable_textures(asset: Asset) -> bool:
    return any(
        any(material.metadata.get(key) for key in _BAKED_TEXTURE_METADATA_KEYS) for material in asset.materials.values()
    )


def _runtime_texture_fallback_policy(asset: Asset, options: GltfExportOptions) -> dict[str, object]:
    texture_materials = [
        material
        for material in referenced_materials(asset).values()
        if any(material.metadata.get(key) for key in _BAKED_TEXTURE_METADATA_KEYS)
    ]
    alpha_sets = sum(1 for material in texture_materials if _material_needs_alpha_safe_fallback(material))
    color_sets = max(0, len(texture_materials) - alpha_sets)
    label = _texture_fallback_label(options.texture_fallback_format)
    jpeg_alpha_risk_sets = alpha_sets if options.texture_fallback_format == "jpeg" else 0
    if jpeg_alpha_risk_sets:
        recommendation = (
            f"avoid JPEG fallback for {jpeg_alpha_risk_sets} alpha-bearing texture set(s); "
            "use auto or PNG until image conversion can split alpha maps"
        )
    else:
        recommendation = f"keep {label} fallbacks while texture compression support is unavailable"
    return {
        "resolved_format": label,
        "alpha_texture_sets": alpha_sets,
        "color_only_texture_sets": color_sets,
        "jpeg_alpha_risk_sets": jpeg_alpha_risk_sets,
        "recommendation": recommendation,
    }


def _texture_fallback_label(texture_fallback_format: str) -> str:
    if texture_fallback_format == "png":
        return "PNG"
    if texture_fallback_format == "jpeg":
        return "JPEG"
    return "PNG/JPEG"


def _material_needs_alpha_safe_fallback(material: Material) -> bool:
    maps = _baked_texture_maps(material.metadata.get("baked_maps"))
    return material.opacity < 1.0 or material.base_color[3] < 1.0 or "opacity" in maps


def _baked_texture_maps(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def _has_exportable_lods(asset: Asset) -> bool:
    for part in asset.parts.values():
        if part.mesh is None:
            continue
        if any(mesh.vertex_count > 0 and "lod_omitted" not in mesh.metadata for mesh in part.lod_meshes):
            return True
    return False


def validate_gltf(path: str | Path) -> dict[str, int]:
    document, buffers = _read_document(Path(path))
    return validate_gltf_payload(document, buffers)


def validate_gltf_document(document: dict[str, Any], binary: BinaryPayload) -> dict[str, int]:
    return validate_gltf_payload(document, [binary])


def validate_gltf_payload(document: dict[str, Any], buffers: Sequence[BinaryPayload]) -> dict[str, int]:
    asset_info = _object(document.get("asset"), "glTF asset metadata")
    if asset_info.get("version") != "2.0":
        raise RuntimeError("glTF asset version must be 2.0")

    _validate_buffers(document, buffers)
    context = _GltfValidationContext.from_document(document)
    stats = _validate_default_scene(context)
    if stats["meshes"] == 0:
        raise RuntimeError("glTF asset contains no meshes in default scene")
    return stats


@dataclass(frozen=True)
class _GltfValidationContext:
    document: dict[str, Any]
    scenes: list[Any]
    nodes: list[Any]
    meshes: list[Any]
    accessors: list[Any]

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> _GltfValidationContext:
        return cls(
            document=document,
            scenes=_array(document.get("scenes"), "scenes"),
            nodes=_array(document.get("nodes"), "nodes"),
            meshes=_array(document.get("meshes"), "meshes"),
            accessors=_array(document.get("accessors"), "accessors"),
        )


@dataclass
class _BufferBuilder:
    data: bytearray = field(default_factory=bytearray)
    buffer_views: list[dict[str, object]] = field(default_factory=list)
    accessors: list[dict[str, object]] = field(default_factory=list)

    def add_accessor(
        self,
        values: NDArray[Any],
        *,
        component_type: int,
        accessor_type: Literal["SCALAR", "VEC2", "VEC3", "VEC4"],
        target: int,
        minimum: list[float | int] | None = None,
        maximum: list[float | int] | None = None,
        normalized: bool = False,
        byte_stride: int | None = None,
    ) -> int:
        self._align()
        contiguous = np.ascontiguousarray(values)
        byte_offset = len(self.data)
        payload = _accessor_payload(contiguous, accessor_type, component_type, byte_stride)
        self.data.extend(payload)
        buffer_view: dict[str, object] = {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(payload),
            "target": target,
        }
        if byte_stride is not None:
            buffer_view["byteStride"] = byte_stride
        self.buffer_views.append(buffer_view)
        accessor: dict[str, object] = {
            "bufferView": len(self.buffer_views) - 1,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(contiguous.shape[0]),
            "type": accessor_type,
        }
        if minimum is not None:
            accessor["min"] = minimum
        if maximum is not None:
            accessor["max"] = maximum
        if normalized:
            accessor["normalized"] = True
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def _align(self) -> None:
        padding = (-len(self.data)) % 4
        if padding:
            self.data.extend(b"\x00" * padding)


def _accessor_payload(
    values: NDArray[Any],
    accessor_type: Literal["SCALAR", "VEC2", "VEC3", "VEC4"],
    component_type: int,
    byte_stride: int | None,
) -> bytes:
    width = _ACCESSOR_WIDTHS[accessor_type]
    component_size = _COMPONENT_SIZES[component_type]
    item_size = width * component_size
    if byte_stride is None:
        return values.tobytes()
    if byte_stride < item_size or byte_stride % 4:
        raise ValueError("glTF vertex byte stride must be a 4-byte aligned size that fits the accessor item")
    rows = np.ascontiguousarray(values.reshape((values.shape[0], width)))
    payload = np.zeros((rows.shape[0], byte_stride), dtype=np.uint8)
    payload[:, :item_size] = rows.view(np.uint8).reshape((rows.shape[0], item_size))
    return payload.tobytes()


@dataclass(frozen=True)
class _ExportSpace:
    linear: NDArray[np.float64]
    normal_linear: NDArray[np.float64]
    matrix: NDArray[np.float64]
    inverse_matrix: NDArray[np.float64]


@dataclass(frozen=True)
class _PartQuantization:
    offset: NDArray[np.float64]
    scale: float

    @property
    def matrix(self) -> NDArray[np.float64]:
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 0] = self.scale
        matrix[1, 1] = self.scale
        matrix[2, 2] = self.scale
        matrix[:3, 3] = self.offset
        return matrix


def _build_document(
    asset: Asset,
    *,
    metadata_options: MetadataExportOptions,
    quantize: bool,
) -> tuple[dict[str, Any], bytearray]:
    export_space = _export_space(asset)
    quantization = _part_quantization(asset, export_space) if quantize else {}
    builder = _BufferBuilder()
    images: list[dict[str, object]] = []
    textures: list[dict[str, object]] = []
    texture_indices_by_uri: dict[str, int] = {}
    material_indices = _write_materials(
        referenced_materials(asset),
        metadata_options,
        images,
        textures,
        texture_indices_by_uri,
    )
    meshes: list[dict[str, Any]] = []
    part_meshes: dict[str, int] = {}
    part_lods: dict[str, list[dict[str, object]]] = {}
    pmi_by_part = _pmi_by_part(asset)

    for part in asset.parts.values():
        if part.mesh is None:
            continue
        part_meshes[part.id] = _append_mesh(
            builder,
            meshes,
            part,
            part.mesh,
            material_indices,
            export_space,
            metadata_options,
            quantization.get(part.id),
            lod=0,
            pmi_ids=pmi_by_part.get(part.id, []),
        )
        lod_entries: list[dict[str, object]] = []
        for lod, lod_mesh in enumerate(part.lod_meshes, start=1):
            mesh_index = _append_mesh(
                builder,
                meshes,
                part,
                lod_mesh,
                material_indices,
                export_space,
                metadata_options,
                quantization.get(part.id),
                lod=lod,
                pmi_ids=pmi_by_part.get(part.id, []),
            )
            lod_entries.append(_lod_entry(mesh_index, lod, lod_mesh))
        if lod_entries:
            part_lods[part.id] = lod_entries

    nodes: list[dict[str, Any]] = []
    root_node = _append_node(nodes, asset.root, part_meshes, part_lods, export_space, metadata_options, quantization)
    binary = builder.data
    buffers: list[dict[str, object]] = [{"byteLength": len(binary)}]

    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "fascat"},
        "scene": 0,
        "scenes": [{"name": asset.root.name or "Scene", "nodes": [root_node]}],
        "nodes": nodes,
        "buffers": buffers,
        "bufferViews": builder.buffer_views,
        "accessors": builder.accessors,
        "meshes": meshes,
        "extras": {
            "fascat": {
                "units": asset.units,
                "metersPerUnit": asset.meters_per_unit,
                "sourceUpAxis": asset.up_axis,
                "exportUnits": "metre",
                "exportUpAxis": "Y",
                **_asset_metadata_extras(asset, metadata_options),
            }
        },
    }
    if material_indices:
        document["materials"] = [
            {key: value for key, value in material.items() if key != "_fascat_index"}
            for material in material_indices.values()
        ]
    if images:
        document["images"] = images
        document["textures"] = textures
    if quantization:
        _add_extension_used(document, _KHR_MESH_QUANTIZATION)
        _add_extension_required(document, _KHR_MESH_QUANTIZATION)
    if _uses_msft_lod(nodes):
        _add_extension_used(document, _MSFT_LOD)
    return document, binary


def _write_materials(
    materials: dict[str, Material],
    metadata_options: MetadataExportOptions,
    images: list[dict[str, object]],
    textures: list[dict[str, object]],
    texture_indices_by_uri: dict[str, int],
) -> dict[str, dict[str, Any]]:
    written: dict[str, dict[str, Any]] = {}
    for index, material in enumerate(materials.values()):
        fascat_extras: dict[str, object] = {"materialId": material.id}
        _add_metadata_extras(fascat_extras, material.metadata, metadata_options)
        gltf_material: dict[str, Any] = {
            "name": material.name or material.id,
            "pbrMetallicRoughness": {
                "baseColorFactor": list(material.base_color),
                "metallicFactor": material.metallic,
                "roughnessFactor": material.roughness,
            },
            "extras": {"fascat": fascat_extras},
        }
        _add_baked_textures(gltf_material, material, images, textures, texture_indices_by_uri)
        if material.opacity < 1.0 or material.base_color[3] < 1.0:
            gltf_material["alphaMode"] = "BLEND"
        gltf_material["_fascat_index"] = index
        written[material.id] = gltf_material
    return written


def _add_baked_textures(
    gltf_material: dict[str, Any],
    material: Material,
    images: list[dict[str, object]],
    textures: list[dict[str, object]],
    texture_indices_by_uri: dict[str, int],
) -> None:
    pbr = cast(dict[str, Any], gltf_material["pbrMetallicRoughness"])
    base_color_uri = _metadata_uri(material, "baked_texture_base_color_uri")
    if base_color_uri is not None:
        pbr["baseColorTexture"] = {
            "index": _append_image_texture(
                images,
                textures,
                texture_indices_by_uri,
                f"{material.id}_base_color",
                base_color_uri,
            )
        }
    metallic_roughness_uri = _metadata_uri(material, "baked_texture_metallic_roughness_uri")
    if metallic_roughness_uri is not None:
        pbr["metallicRoughnessTexture"] = {
            "index": _append_image_texture(
                images,
                textures,
                texture_indices_by_uri,
                f"{material.id}_metallic_roughness",
                metallic_roughness_uri,
            )
        }
    normal_uri = _metadata_uri(material, "baked_texture_normal_uri")
    if normal_uri is not None:
        gltf_material["normalTexture"] = {
            "index": _append_image_texture(
                images,
                textures,
                texture_indices_by_uri,
                f"{material.id}_normal",
                normal_uri,
            )
        }
    occlusion_uri = _metadata_uri(material, "baked_texture_occlusion_uri")
    if occlusion_uri is not None:
        gltf_material["occlusionTexture"] = {
            "index": _append_image_texture(
                images,
                textures,
                texture_indices_by_uri,
                f"{material.id}_occlusion",
                occlusion_uri,
            )
        }
    emissive_uri = _metadata_uri(material, "baked_texture_emissive_uri")
    if emissive_uri is not None:
        gltf_material["emissiveTexture"] = {
            "index": _append_image_texture(
                images,
                textures,
                texture_indices_by_uri,
                f"{material.id}_emissive",
                emissive_uri,
            )
        }


def _append_image_texture(
    images: list[dict[str, object]],
    textures: list[dict[str, object]],
    texture_indices_by_uri: dict[str, int],
    name: str,
    uri: str,
) -> int:
    existing = texture_indices_by_uri.get(uri)
    if existing is not None:
        return existing
    images.append({"name": name, "uri": uri})
    textures.append({"source": len(images) - 1})
    texture_index = len(textures) - 1
    texture_indices_by_uri[uri] = texture_index
    return texture_index


def _metadata_uri(material: Material, key: str) -> str | None:
    value = material.metadata.get(key)
    return value if isinstance(value, str) and value.startswith("data:image/") else None


def _asset_metadata_extras(asset: Asset, options: MetadataExportOptions) -> dict[str, object]:
    extras: dict[str, object] = {}
    if options.mode != "none":
        extras["metadataSummary"] = _metadata_summary(asset)
    if options.mode == "full":
        extras["metadata"] = dict(asset.metadata)
    elif options.mode == "summary":
        extras["metadata"] = dict(cast(dict[str, object], extras["metadataSummary"]))
    if options.pmi != "none":
        pmi = [annotation.to_dict() for annotation in asset.pmi]
        extras["pmi"] = pmi if options.pmi not in {"summary"} else {"count": len(pmi)}
    return extras


def _metadata_summary(asset: Asset) -> dict[str, object]:
    return {
        "asset": len(asset.metadata),
        "nodes": sum(len(node.metadata) for node in asset.root.walk()),
        "parts": sum(len(part.metadata) for part in asset.parts.values()),
        "materials": sum(len(material.metadata) for material in asset.materials.values()),
        "pmi": len(asset.pmi),
    }


def _part_quantization(asset: Asset, export_space: _ExportSpace) -> dict[str, _PartQuantization]:
    result: dict[str, _PartQuantization] = {}
    for part in asset.parts.values():
        meshes = [mesh for mesh in (part.mesh, *part.lod_meshes) if mesh is not None and mesh.vertex_count]
        if not meshes:
            continue
        points = np.vstack([_points_to_export_space(mesh.points, export_space.linear) for mesh in meshes])
        minimum = points.min(axis=0)
        maximum = points.max(axis=0)
        extent = maximum - minimum
        max_extent = float(extent.max())
        scale = max_extent / float(np.iinfo(np.uint16).max) if max_extent > 0.0 else 1.0
        result[part.id] = _PartQuantization(offset=minimum.astype(np.float64), scale=scale)
    return result


def _quantize_positions(points: NDArray[np.float64], quantization: _PartQuantization) -> NDArray[np.uint16]:
    if not len(points):
        return np.empty((0, 3), dtype=np.uint16)
    scaled = (points - quantization.offset) / quantization.scale
    return cast(NDArray[np.uint16], np.clip(np.rint(scaled), 0, np.iinfo(np.uint16).max).astype(np.uint16))


def _quantize_snorm8(values: NDArray[np.float64]) -> NDArray[np.int8]:
    quantized = np.clip(np.rint(np.clip(values, -1.0, 1.0) * 127.0), -127, 127)
    return cast(NDArray[np.int8], quantized.astype(np.int8))


def _can_quantize_unorm(values: NDArray[np.float64]) -> bool:
    return bool(values.size == 0 or (np.all(values >= 0.0) and np.all(values <= 1.0)))


def _quantize_unorm16(values: NDArray[np.float64]) -> NDArray[np.uint16]:
    quantized = np.clip(np.rint(np.clip(values, 0.0, 1.0) * 65535.0), 0, np.iinfo(np.uint16).max)
    return cast(NDArray[np.uint16], quantized.astype(np.uint16))


def _append_mesh(
    builder: _BufferBuilder,
    meshes: list[dict[str, Any]],
    part: Part,
    mesh: Mesh,
    material_indices: dict[str, dict[str, Any]],
    export_space: _ExportSpace,
    metadata_options: MetadataExportOptions,
    quantization: _PartQuantization | None,
    *,
    lod: int,
    pmi_ids: list[str],
) -> int:
    mesh.validate()
    points = _points_to_export_space(mesh.points, export_space.linear)
    if quantization is None:
        float_points = points.astype(np.float32)
        position_accessor = builder.add_accessor(
            float_points,
            component_type=_FLOAT,
            accessor_type="VEC3",
            target=_ARRAY_BUFFER,
            minimum=float_points.min(axis=0).astype(float).tolist() if len(float_points) else [0.0, 0.0, 0.0],
            maximum=float_points.max(axis=0).astype(float).tolist() if len(float_points) else [0.0, 0.0, 0.0],
        )
    else:
        quantized_points = _quantize_positions(points, quantization)
        position_accessor = builder.add_accessor(
            quantized_points,
            component_type=_UNSIGNED_SHORT,
            accessor_type="VEC3",
            target=_ARRAY_BUFFER,
            minimum=quantized_points.min(axis=0).astype(int).tolist() if len(quantized_points) else [0, 0, 0],
            maximum=quantized_points.max(axis=0).astype(int).tolist() if len(quantized_points) else [0, 0, 0],
            byte_stride=8,
        )
    attributes: dict[str, int] = {"POSITION": position_accessor}
    if mesh.normals is not None:
        normals = _normals_to_export_space(mesh.normals, export_space.normal_linear)
        if quantization is None:
            attributes["NORMAL"] = builder.add_accessor(
                normals.astype(np.float32),
                component_type=_FLOAT,
                accessor_type="VEC3",
                target=_ARRAY_BUFFER,
            )
        else:
            attributes["NORMAL"] = builder.add_accessor(
                _quantize_snorm8(normals),
                component_type=_BYTE,
                accessor_type="VEC3",
                target=_ARRAY_BUFFER,
                normalized=True,
                byte_stride=4,
            )
    if mesh.tangents is not None:
        tangents = mesh.tangents.copy()
        tangents[:, :3] = _normals_to_export_space(mesh.tangents[:, :3], export_space.normal_linear)
        if quantization is None:
            attributes["TANGENT"] = builder.add_accessor(
                tangents.astype(np.float32),
                component_type=_FLOAT,
                accessor_type="VEC4",
                target=_ARRAY_BUFFER,
            )
        else:
            attributes["TANGENT"] = builder.add_accessor(
                _quantize_snorm8(tangents),
                component_type=_BYTE,
                accessor_type="VEC4",
                target=_ARRAY_BUFFER,
                normalized=True,
            )
    if 0 in mesh.uvs:
        uv0 = mesh.uvs[0]
        if quantization is not None and _can_quantize_unorm(uv0):
            attributes["TEXCOORD_0"] = builder.add_accessor(
                _quantize_unorm16(uv0),
                component_type=_UNSIGNED_SHORT,
                accessor_type="VEC2",
                target=_ARRAY_BUFFER,
                normalized=True,
            )
        else:
            attributes["TEXCOORD_0"] = builder.add_accessor(
                uv0.astype(np.float32),
                component_type=_FLOAT,
                accessor_type="VEC2",
                target=_ARRAY_BUFFER,
            )
    if 1 in mesh.uvs:
        uv1 = mesh.uvs[1]
        if quantization is not None and _can_quantize_unorm(uv1):
            attributes["TEXCOORD_1"] = builder.add_accessor(
                _quantize_unorm16(uv1),
                component_type=_UNSIGNED_SHORT,
                accessor_type="VEC2",
                target=_ARRAY_BUFFER,
                normalized=True,
            )
        else:
            attributes["TEXCOORD_1"] = builder.add_accessor(
                uv1.astype(np.float32),
                component_type=_FLOAT,
                accessor_type="VEC2",
                target=_ARRAY_BUFFER,
            )

    primitives: list[dict[str, Any]] = []
    for material_id, faces in _face_groups(part, mesh):
        if faces.size == 0:
            continue
        indices = mesh.faces[faces].reshape(-1)
        if mesh.vertex_count <= np.iinfo(np.uint16).max:
            index_values = indices.astype(np.uint16)
            component_type = _UNSIGNED_SHORT
        else:
            index_values = indices.astype(np.uint32)
            component_type = _UNSIGNED_INT
        primitive: dict[str, Any] = {
            "attributes": attributes,
            "indices": builder.add_accessor(
                index_values.reshape(-1, 1),
                component_type=component_type,
                accessor_type="SCALAR",
                target=_ELEMENT_ARRAY_BUFFER,
            ),
            "mode": 4,
        }
        if material_id is not None and material_id in material_indices:
            primitive["material"] = int(material_indices[material_id]["_fascat_index"])
        primitives.append(primitive)

    fascat_extras: dict[str, object] = {
        "partId": part.id,
        "originalName": part.name,
        "lod": lod,
    }
    _add_metadata_extras(fascat_extras, part.metadata, metadata_options)
    _add_pmi_link_extras(fascat_extras, pmi_ids, metadata_options)
    gltf_mesh: dict[str, Any] = {
        "name": f"{part.name or part.id}_lod{lod}",
        "primitives": primitives,
        "extras": {"fascat": fascat_extras},
    }
    meshes.append(gltf_mesh)
    return len(meshes) - 1


def _apply_export_options(document: dict[str, Any], options: GltfExportOptions) -> None:
    fascat_extras = document.setdefault("extras", {}).setdefault("fascat", {})
    fascat_extras["exportOptions"] = options.to_dict()
    compression: dict[str, object] = {}
    if options.quantize:
        compression["quantize"] = True
    if options.meshopt:
        compression["meshopt"] = True
    if options.draco:
        compression["draco"] = True
    if compression:
        fascat_extras["compression"] = compression


def _apply_meshopt_compression(document: dict[str, Any], binary: BinaryPayload) -> bytearray:
    try:
        import meshoptimizer
    except ImportError as exc:
        raise RuntimeError("glTF meshopt compression requires meshoptimizer") from exc

    payload = bytearray(binary)
    accessors_by_view = _accessors_by_buffer_view(document)
    compressed_views = 0
    for view_index, view_value in enumerate(_array(document.get("bufferViews"), "bufferViews")):
        view = _object(view_value, f"bufferView {view_index}")
        if int(view.get("buffer", 0)) != 0:
            continue
        accessor = accessors_by_view.get(view_index)
        if accessor is None:
            continue
        byte_offset = _int(view.get("byteOffset", 0), f"bufferView {view_index} byteOffset")
        byte_length = _int(view.get("byteLength"), f"bufferView {view_index} byteLength")
        source = binary[byte_offset : byte_offset + byte_length]
        spec = _meshopt_view_spec(view, accessor, source)
        if spec is None:
            continue
        encoded = _meshopt_encode(source, spec, meshoptimizer)
        if not encoded:
            continue
        compressed_offset = _append_aligned(payload, encoded)
        view.setdefault("extensions", {})[_EXT_MESHOPT_COMPRESSION] = {
            "buffer": 0,
            "byteOffset": compressed_offset,
            "byteLength": len(encoded),
            "byteStride": spec.byte_stride,
            "count": spec.count,
            "mode": spec.mode,
        }
        compressed_views += 1
    if compressed_views:
        _add_extension_used(document, _EXT_MESHOPT_COMPRESSION)
        buffers = _array(document.get("buffers"), "buffers")
        _object(buffers[0], "buffer 0")["byteLength"] = len(payload)
    return payload


@dataclass(frozen=True)
class _MeshoptViewSpec:
    mode: str
    count: int
    byte_stride: int
    component_type: int


def _accessors_by_buffer_view(document: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for accessor_index, accessor_value in enumerate(_array(document.get("accessors"), "accessors")):
        accessor = _object(accessor_value, f"accessor {accessor_index}")
        view_index = accessor.get("bufferView")
        if isinstance(view_index, int):
            result[view_index] = accessor
    return result


def _meshopt_view_spec(
    view: dict[str, Any],
    accessor: dict[str, Any],
    source: BinaryPayload,
) -> _MeshoptViewSpec | None:
    component_type = _int(accessor.get("componentType"), "meshopt accessor componentType")
    accessor_type = accessor.get("type")
    if not isinstance(accessor_type, str) or accessor_type not in _ACCESSOR_WIDTHS:
        return None
    component_size = _COMPONENT_SIZES.get(component_type)
    if component_size is None:
        return None
    count = _int(accessor.get("count"), "meshopt accessor count")
    byte_stride = _int(
        view.get("byteStride", component_size * _ACCESSOR_WIDTHS[accessor_type]),
        "meshopt bufferView byteStride",
    )
    if count <= 0 or byte_stride <= 0 or len(source) != count * byte_stride:
        return None
    if view.get("target") == _ELEMENT_ARRAY_BUFFER:
        mode = "TRIANGLES" if count % 3 == 0 else "INDICES"
        if byte_stride not in {2, 4}:
            return None
        return _MeshoptViewSpec(mode=mode, count=count, byte_stride=byte_stride, component_type=component_type)
    if view.get("target") == _ARRAY_BUFFER and byte_stride % 4 == 0 and byte_stride <= 256:
        return _MeshoptViewSpec(mode="ATTRIBUTES", count=count, byte_stride=byte_stride, component_type=component_type)
    return None


def _meshopt_encode(source: BinaryPayload, spec: _MeshoptViewSpec, meshoptimizer: Any) -> bytes:
    if spec.mode in {"TRIANGLES", "INDICES"}:
        dtype = np.dtype("<u2") if spec.byte_stride == 2 else np.dtype("<u4")
        indices = np.frombuffer(source, dtype=dtype, count=spec.count).copy()
        if spec.mode == "TRIANGLES":
            return cast(bytes, meshoptimizer.encode_index_buffer(indices, index_count=spec.count))
        return cast(bytes, meshoptimizer.encode_index_sequence(indices, index_count=spec.count))
    vertices = np.frombuffer(source, dtype=np.uint8).reshape((spec.count, spec.byte_stride)).copy()
    return cast(
        bytes,
        meshoptimizer.encode_vertex_buffer(vertices, vertex_count=spec.count, vertex_size=spec.byte_stride),
    )


def _append_aligned(payload: bytearray, data: bytes) -> int:
    padding = (-len(payload)) % 4
    if padding:
        payload.extend(b"\x00" * padding)
    offset = len(payload)
    payload.extend(data)
    return offset


def _add_extension_used(document: dict[str, Any], extension: str) -> None:
    extensions = document.setdefault("extensionsUsed", [])
    if isinstance(extensions, list) and extension not in extensions:
        extensions.append(extension)


def _add_extension_required(document: dict[str, Any], extension: str) -> None:
    extensions = document.setdefault("extensionsRequired", [])
    if isinstance(extensions, list) and extension not in extensions:
        extensions.append(extension)
    _add_extension_used(document, extension)


def _embed_binary_uri(document: dict[str, Any], binary: BinaryPayload) -> None:
    buffers = _array(document.get("buffers"), "buffers")
    buffer = _object(buffers[0], "buffer 0")
    buffer["byteLength"] = len(binary)
    buffer["uri"] = "data:application/octet-stream;base64," + base64.b64encode(binary).decode("ascii")


def _lod_entry(mesh_index: int, lod: int, mesh: Mesh) -> dict[str, object]:
    entry: dict[str, object] = {"level": lod, "mesh": mesh_index}
    if "lod_ratio" in mesh.metadata:
        entry["ratio"] = _metadata_float(mesh.metadata["lod_ratio"])
    if "lod_screen_coverage" in mesh.metadata:
        entry["screenCoverage"] = _metadata_float(mesh.metadata["lod_screen_coverage"])
    if mesh.metadata.get("lod_omitted"):
        entry["omitted"] = mesh.metadata["lod_omitted"]
    return entry


def _metadata_float(value: object) -> object:
    try:
        return float(str(value))
    except ValueError:
        return str(value)


def _face_groups(part: Part, mesh: Mesh) -> list[tuple[str | None, NDArray[np.int64]]]:
    all_faces = np.arange(mesh.triangle_count, dtype=np.int64)
    if mesh.material_indices is None:
        material_id = part.material_ids[0] if part.material_ids else None
        return [(material_id, all_faces)]
    groups: list[tuple[str | None, NDArray[np.int64]]] = []
    for material_index in sorted(set(mesh.material_indices.astype(int).tolist())):
        material_id = part.material_ids[material_index] if material_index < len(part.material_ids) else None
        groups.append((material_id, np.flatnonzero(mesh.material_indices == material_index).astype(np.int64)))
    return groups


def _append_node(
    nodes: list[dict[str, Any]],
    node: Node,
    part_meshes: dict[str, int],
    part_lods: dict[str, list[dict[str, object]]],
    export_space: _ExportSpace,
    metadata_options: MetadataExportOptions,
    quantization: dict[str, _PartQuantization],
) -> int:
    fascat_extras: dict[str, object] = {"nodeId": node.id}
    if metadata_options.mode == "full":
        fascat_extras.update(node.metadata)
    _add_metadata_extras(fascat_extras, node.metadata, metadata_options)
    gltf_node: dict[str, Any] = {
        "name": node.name or node.id,
        "extras": {"fascat": fascat_extras},
    }
    transform = _node_transform(node, export_space, quantization)
    if not np.allclose(transform, np.eye(4)):
        gltf_node["matrix"] = transform.T.reshape(-1).astype(float).tolist()
    if node.part_id is not None and node.part_id in part_meshes:
        gltf_node["mesh"] = part_meshes[node.part_id]
        lods = part_lods.get(node.part_id)
        if lods:
            gltf_node["extras"]["fascat"]["lodMeshIndices"] = [entry["mesh"] for entry in lods]
            gltf_node["extras"]["fascat"]["lods"] = lods
            msft_lods = [entry for entry in lods if "omitted" not in entry]
            msft_lod_node_indices = [
                _append_lod_node(nodes, node, entry, export_space, metadata_options, quantization)
                for entry in msft_lods
            ]
            if msft_lod_node_indices:
                gltf_node.setdefault("extensions", {})[_MSFT_LOD] = {"ids": msft_lod_node_indices}
                screen_coverage = [entry["screenCoverage"] for entry in msft_lods if "screenCoverage" in entry]
                if screen_coverage:
                    gltf_node["extras"]["MSFT_screencoverage"] = screen_coverage
    index = len(nodes)
    nodes.append(gltf_node)
    children = [
        _append_node(nodes, child, part_meshes, part_lods, export_space, metadata_options, quantization)
        for child in node.children
    ]
    if children:
        gltf_node["children"] = children
    return index


def _append_lod_node(
    nodes: list[dict[str, Any]],
    node: Node,
    lod_entry: dict[str, object],
    export_space: _ExportSpace,
    metadata_options: MetadataExportOptions,
    quantization: dict[str, _PartQuantization],
) -> int:
    level = _int(lod_entry["level"], "LOD level")
    mesh_index = _int(lod_entry["mesh"], "LOD mesh index")
    gltf_node: dict[str, Any] = {
        "name": f"{node.name or node.id}_lod{level}",
        "mesh": mesh_index,
        "extras": {
            "fascat": {
                "nodeId": f"{node.id}_lod{level}",
                "sourceNodeId": node.id,
                "lod": level,
            }
        },
    }
    if metadata_options.mode == "full" and node.metadata:
        gltf_node["extras"]["fascat"]["metadata"] = dict(node.metadata)
    transform = _node_transform(node, export_space, quantization)
    if not np.allclose(transform, np.eye(4)):
        gltf_node["matrix"] = transform.T.reshape(-1).astype(float).tolist()
    nodes.append(gltf_node)
    return len(nodes) - 1


def _node_transform(
    node: Node,
    export_space: _ExportSpace,
    quantization: dict[str, _PartQuantization],
) -> NDArray[np.float64]:
    transform = _matrix_to_export_space(node.transform, export_space)
    if node.part_id is not None and node.part_id in quantization:
        transform = transform @ quantization[node.part_id].matrix
    return transform


def _uses_msft_lod(nodes: list[dict[str, Any]]) -> bool:
    return any(_MSFT_LOD in node.get("extensions", {}) for node in nodes)


def _pmi_by_part(asset: Asset) -> dict[str, list[str]]:
    return pmi_ids_by_part(asset.parts, asset.pmi)


def _add_metadata_extras(
    payload: dict[str, object],
    metadata: dict[str, object],
    options: MetadataExportOptions,
) -> None:
    if options.mode == "full":
        payload["metadata"] = dict(metadata)
    elif options.mode == "summary" and metadata:
        payload["metadataSummary"] = {"count": len(metadata)}


def _add_pmi_link_extras(
    payload: dict[str, object],
    pmi_ids: list[str],
    options: MetadataExportOptions,
) -> None:
    if not pmi_ids or options.pmi == "none":
        return
    if options.pmi == "summary":
        payload["pmiCount"] = len(pmi_ids)
    else:
        payload["pmiIds"] = list(pmi_ids)


def _export_space(asset: Asset) -> _ExportSpace:
    axis = np.eye(3, dtype=np.float64)
    if asset.up_axis == "Z":
        axis = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float64,
        )
    linear = axis * float(asset.meters_per_unit)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = linear
    inverse = np.asarray(np.linalg.inv(matrix), dtype=np.float64)
    return _ExportSpace(
        linear=linear,
        normal_linear=axis,
        matrix=matrix,
        inverse_matrix=inverse,
    )


def _points_to_export_space(points: NDArray[np.float64], linear: NDArray[np.float64]) -> NDArray[np.float64]:
    return cast(NDArray[np.float64], points @ linear.T)


def _normals_to_export_space(normals: NDArray[np.float64], linear: NDArray[np.float64]) -> NDArray[np.float64]:
    transformed = cast(NDArray[np.float64], normals @ linear.T)
    lengths = np.linalg.norm(transformed, axis=1)
    valid = lengths > 0.0
    transformed[valid] = transformed[valid] / lengths[valid, None]
    transformed[~valid] = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return transformed


def _matrix_to_export_space(matrix: NDArray[np.float64], export_space: _ExportSpace) -> NDArray[np.float64]:
    return export_space.matrix @ matrix @ export_space.inverse_matrix


def _pack_glb(document: dict[str, Any], binary: BinaryPayload) -> bytearray:
    json_payload = json.dumps(document, separators=(",", ":"), sort_keys=False).encode("utf-8")
    json_padding = (-len(json_payload)) % 4
    bin_padding = (-len(binary)) % 4
    json_length = len(json_payload) + json_padding
    bin_length = len(binary) + bin_padding
    length = 12 + 8 + json_length + 8 + bin_length
    payload = bytearray(length)
    offset = 0
    struct.pack_into("<4sII", payload, offset, _GLB_MAGIC, _GLB_VERSION, length)
    offset += 12
    struct.pack_into("<II", payload, offset, json_length, _JSON_CHUNK)
    offset += 8
    payload[offset : offset + len(json_payload)] = json_payload
    if json_padding:
        payload[offset + len(json_payload) : offset + json_length] = b" " * json_padding
    offset += json_length
    struct.pack_into("<II", payload, offset, bin_length, _BIN_CHUNK)
    offset += 8
    payload[offset : offset + len(binary)] = binary
    return payload


def _read_document(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    suffix = path.suffix.lower()
    if suffix == ".glb":
        return _read_glb(path)
    if suffix == ".gltf":
        return _read_gltf(path)
    raise RuntimeError(f"unsupported glTF extension: {path.suffix or '<none>'}")


def _read_glb(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    payload = path.read_bytes()
    if len(payload) < 20:
        raise RuntimeError("invalid GLB header")
    magic, version, length = struct.unpack_from("<4sII", payload, 0)
    if magic != _GLB_MAGIC or version != _GLB_VERSION:
        raise RuntimeError("invalid GLB header")
    if length != len(payload):
        raise RuntimeError("invalid GLB length")
    offset = 12
    document: dict[str, Any] | None = None
    binary = b""
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise RuntimeError("invalid GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise RuntimeError("invalid GLB chunk length")
        offset += chunk_length
        if chunk_type == _JSON_CHUNK:
            document = cast(dict[str, Any], json.loads(chunk.decode("utf-8").rstrip(" \x00")))
        elif chunk_type == _BIN_CHUNK:
            binary = chunk
    if document is None:
        raise RuntimeError("GLB contains no JSON chunk")
    return document, [binary]


def _read_gltf(path: Path) -> tuple[dict[str, Any], list[bytes]]:
    document = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    buffers: list[bytes] = []
    for index, buffer in enumerate(_array(document.get("buffers"), "buffers")):
        buffer_object = _object(buffer, f"buffer {index}")
        uri = buffer_object.get("uri")
        if not isinstance(uri, str) or not uri.startswith("data:"):
            raise RuntimeError("glTF validation only supports embedded data URI buffers")
        try:
            _, encoded = uri.split(",", 1)
            buffers.append(base64.b64decode(encoded))
        except ValueError as exc:
            raise RuntimeError("invalid glTF data URI buffer") from exc
    return document, buffers


def _validate_buffers(document: dict[str, Any], buffers: Sequence[BinaryPayload]) -> None:
    buffer_objects = _array(document.get("buffers"), "buffers")
    if len(buffer_objects) != len(buffers):
        raise RuntimeError("glTF buffer count does not match payload count")
    for index, buffer in enumerate(buffer_objects):
        buffer_length = _int(_object(buffer, f"buffer {index}").get("byteLength"), f"buffer {index} byteLength")
        if buffer_length > len(buffers[index]):
            raise RuntimeError(f"glTF buffer {index} is shorter than its byteLength")
    for index, buffer_view in enumerate(_array(document.get("bufferViews"), "bufferViews")):
        view = _object(buffer_view, f"bufferView {index}")
        buffer_index = _int(view.get("buffer", 0), f"bufferView {index} buffer")
        if buffer_index < 0 or buffer_index >= len(buffers):
            raise RuntimeError(f"glTF bufferView {index} references an invalid buffer")
        offset = _int(view.get("byteOffset", 0), f"bufferView {index} byteOffset")
        length = _int(view.get("byteLength"), f"bufferView {index} byteLength")
        if offset < 0 or length < 0 or offset + length > len(buffers[buffer_index]):
            raise RuntimeError(f"glTF bufferView {index} is out of range")
    for index, accessor in enumerate(_array(document.get("accessors"), "accessors")):
        _validate_accessor_storage(document, index, _object(accessor, f"accessor {index}"))


def _validate_accessor_storage(document: dict[str, Any], index: int, accessor: dict[str, Any]) -> None:
    buffer_views = _array(document.get("bufferViews"), "bufferViews")
    view_index = _int(accessor.get("bufferView"), f"accessor {index} bufferView")
    if view_index < 0 or view_index >= len(buffer_views):
        raise RuntimeError(f"glTF accessor {index} references an invalid bufferView")
    view = _object(buffer_views[view_index], f"bufferView {view_index}")
    count = _int(accessor.get("count"), f"accessor {index} count")
    component_type = _int(accessor.get("componentType"), f"accessor {index} componentType")
    accessor_type = accessor.get("type")
    if component_type not in _COMPONENT_SIZES:
        raise RuntimeError(f"glTF accessor {index} uses unsupported componentType")
    if accessor_type not in _ACCESSOR_WIDTHS:
        raise RuntimeError(f"glTF accessor {index} uses unsupported type")
    byte_offset = _int(accessor.get("byteOffset", 0), f"accessor {index} byteOffset")
    stride = _int(view.get("byteStride", 0), f"bufferView {view_index} byteStride") if "byteStride" in view else 0
    item_size = _COMPONENT_SIZES[component_type] * _ACCESSOR_WIDTHS[cast(str, accessor_type)]
    needed = byte_offset
    if count > 0:
        needed += (count - 1) * stride + item_size if stride else count * item_size
    if count < 0 or byte_offset < 0 or needed > _int(view.get("byteLength"), f"bufferView {view_index} byteLength"):
        raise RuntimeError(f"glTF accessor {index} is out of range")


def _validate_default_scene(context: _GltfValidationContext) -> dict[str, int]:
    scene_index = _int(context.document.get("scene", 0), "default scene")
    if scene_index < 0 or scene_index >= len(context.scenes):
        raise RuntimeError("glTF default scene index is invalid")
    stats = {"meshes": 0, "points": 0, "triangles": 0}
    for node_index in _array(
        _object(context.scenes[scene_index], f"scene {scene_index}").get("nodes", []), "scene nodes"
    ):
        _walk_node(context, _int(node_index, "scene node"), stats, stack=set())
    if len(context.nodes) == 0:
        raise RuntimeError("glTF asset contains no nodes")
    return stats


def _walk_node(context: _GltfValidationContext, node_index: int, stats: dict[str, int], *, stack: set[int]) -> None:
    if node_index < 0 or node_index >= len(context.nodes):
        raise RuntimeError("glTF node index is invalid")
    if node_index in stack:
        raise RuntimeError("glTF node hierarchy contains a cycle")
    node = _object(context.nodes[node_index], f"node {node_index}")
    mesh_index = node.get("mesh")
    if mesh_index is not None:
        _validate_mesh(context, _int(mesh_index, f"node {node_index} mesh"), stats)
    for child_index in _array(node.get("children", []), f"node {node_index} children"):
        _walk_node(context, _int(child_index, f"node {node_index} child"), stats, stack=stack | {node_index})


def _validate_mesh(context: _GltfValidationContext, mesh_index: int, stats: dict[str, int]) -> None:
    if mesh_index < 0 or mesh_index >= len(context.meshes):
        raise RuntimeError("glTF mesh index is invalid")
    mesh = _object(context.meshes[mesh_index], f"mesh {mesh_index}")
    position_accessors: set[int] = set()
    triangles = 0
    for primitive_index, primitive_value in enumerate(_array(mesh.get("primitives"), f"mesh {mesh_index} primitives")):
        primitive = _object(primitive_value, f"mesh {mesh_index} primitive {primitive_index}")
        if _int(primitive.get("mode", 4), f"mesh {mesh_index} primitive mode") != 4:
            raise RuntimeError("glTF validation only supports triangle primitives")
        attributes = _object(primitive.get("attributes"), f"mesh {mesh_index} primitive attributes")
        position_index = _int(attributes.get("POSITION"), f"mesh {mesh_index} POSITION accessor")
        position_accessor = _require_accessor(context, position_index, accessor_type="VEC3")
        if not _position_accessor_allowed(context.document, position_accessor):
            raise RuntimeError("glTF POSITION accessor must use FLOAT or KHR_mesh_quantization component types")
        position_accessors.add(position_index)
        indices = primitive.get("indices")
        if indices is None:
            position_count = _accessor_count(context, position_index)
            if position_count % 3:
                raise RuntimeError("glTF non-indexed triangle primitive has invalid vertex count")
            triangles += position_count // 3
        else:
            index = _int(indices, f"mesh {mesh_index} indices accessor")
            accessor = _require_accessor(context, index, accessor_type="SCALAR")
            component_type = _int(accessor.get("componentType"), f"accessor {index} componentType")
            if component_type not in {_UNSIGNED_SHORT, _UNSIGNED_INT}:
                raise RuntimeError("glTF index accessor must use unsigned integer components")
            index_count = _int(accessor.get("count"), f"accessor {index} count")
            if index_count % 3:
                raise RuntimeError("glTF indexed triangle primitive has invalid index count")
            triangles += index_count // 3
    stats["meshes"] += 1
    stats["points"] += sum(_accessor_count(context, accessor) for accessor in position_accessors)
    stats["triangles"] += triangles


def _require_accessor(
    context: _GltfValidationContext | dict[str, Any],
    accessor_index: int,
    *,
    component_type: int | None = None,
    accessor_type: str | None = None,
) -> dict[str, Any]:
    accessors = (
        context.accessors
        if isinstance(context, _GltfValidationContext)
        else _array(context.get("accessors"), "accessors")
    )
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise RuntimeError("glTF accessor index is invalid")
    accessor = _object(accessors[accessor_index], f"accessor {accessor_index}")
    if (
        component_type is not None
        and _int(accessor.get("componentType"), f"accessor {accessor_index} componentType") != component_type
    ):
        raise RuntimeError("glTF accessor has the wrong component type")
    if accessor_type is not None and accessor.get("type") != accessor_type:
        raise RuntimeError("glTF accessor has the wrong type")
    return accessor


def _position_accessor_allowed(document: dict[str, Any], accessor: dict[str, Any]) -> bool:
    component_type = _int(accessor.get("componentType"), "POSITION accessor componentType")
    if component_type == _FLOAT:
        return True
    return _has_extension(document, _KHR_MESH_QUANTIZATION) and component_type in {
        _BYTE,
        _UNSIGNED_BYTE,
        _SHORT,
        _UNSIGNED_SHORT,
    }


def _has_extension(document: dict[str, Any], extension: str) -> bool:
    used = document.get("extensionsUsed", [])
    required = document.get("extensionsRequired", [])
    return (isinstance(used, list) and extension in used) or (isinstance(required, list) and extension in required)


def _accessor_count(context: _GltfValidationContext, accessor_index: int) -> int:
    accessor = _require_accessor(context, accessor_index)
    return _int(accessor.get("count"), f"accessor {accessor_index} count")


def _array(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise RuntimeError(f"glTF {label} must be an array")
    return value


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"glTF {label} must be an object")
    return value


def _int(value: Any, label: str) -> int:
    if not isinstance(value, int):
        raise RuntimeError(f"glTF {label} must be an integer")
    return value
