from __future__ import annotations

from enum import Enum


class Profile(str, Enum):
    INSPECT_ONLY = "inspect-only"
    REALTIME_DESKTOP = "realtime-desktop"
    REALTIME_WEB = "realtime-web"
    REALTIME_MOBILE = "realtime-mobile"
    VIRTUAL_REALITY = "virtual-reality"
    AUGMENTED_REALITY = "augmented-reality"
    MIXED_REALITY = "mixed-reality"


class ExportPreset(str, Enum):
    DESKTOP = "desktop"
    WEB = "web"
    MOBILE = "mobile"
    VR = "vr"
    AR = "ar"


class StdoutFormat(str, Enum):
    USDA = "usda"
    USDC = "usdc"
    USDZ = "usdz"
    GLTF = "gltf"
    GLB = "glb"
    OBJ = "obj"
    STL = "stl"
    FBX = "fbx"


class RuntimeEngineMode(str, Enum):
    UNITY = "unity"
    UNREAL = "unreal"


class RuntimeParityCaptureMode(str, Enum):
    BROWSER = "browser"
    UNITY = "unity"
    UNREAL = "unreal"


class AxisMode(str, Enum):
    Y = "Y"
    Z = "Z"


class HandednessMode(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class ConstructionCurvePolicyMode(str, Enum):
    PRESERVE_METADATA = "preserve-metadata"
    DELETE = "delete"
    TESSELLATE_TUBES = "tessellate-tubes"


class MaterialLibraryColorSpaceMode(str, Enum):
    AUTO = "auto"
    LINEAR = "linear"
    SRGB255 = "srgb255"


class JtLodSelectionMode(str, Enum):
    FINEST = "finest"
    ALL = "all"


class UV0Mode(str, Enum):
    NONE = "none"
    BOX = "box"
    UNWRAP = "unwrap"
    LIGHTMAP = "lightmap"


class UV1Mode(str, Enum):
    NONE = "none"
    BOX = "box"
    UNWRAP = "unwrap"
    LIGHTMAP = "lightmap"
    COPY_UV0 = "copy-uv0"


class UnwrapMethod(str, Enum):
    DEFAULT = "default"
    CONFORMAL = "conformal"
    ISOMETRIC = "isometric"


class AabbProjectionScope(str, Enum):
    LOCAL = "local"
    SHARED = "shared"


class MaterialMode(str, Enum):
    CAD = "cad"
    DISPLAY = "display"
    NONE = "none"


class MaterialPipelineMode(str, Enum):
    CAD = "cad"
    PBR = "pbr"


class NormalMode(str, Enum):
    NONE = "none"
    SMOOTH = "smooth"
    HARD_EDGES = "hard-edges"
    FLAT = "flat"


class NormalWeighting(str, Enum):
    ANGLE = "angle"
    AREA = "area"


class MergeMode(str, Enum):
    ALL = "all"
    BY_MATERIAL = "by-material"
    BY_NODE_NAME = "by-node-name"
    BY_PART_NAME = "by-part-name"
    HIERARCHY_LEVEL = "hierarchy-level"
    PARENT_CHILDREN = "parent-children"
    FINAL_LEVEL = "final-level"
    REGIONS = "regions"


class MergeMetadata(str, Enum):
    PRESERVE = "preserve"
    COMBINE = "combine"
    SUMMARIZE = "summarize"
    DROP = "drop"


class MergeStrategy(str, Enum):
    ALL = "all"
    BY_MATERIAL = "by-material"


class ExplodeMode(str, Enum):
    BY_MATERIAL = "by-material"
    CONNECTED_COMPONENTS = "connected-components"


class ReplaceMode(str, Enum):
    BOUNDING_BOX = "bounding-box"
    EXTERNAL_ASSET = "external-asset"


class IndexBufferMode(str, Enum):
    AUTO = "auto"
    UINT16 = "uint16"
    UINT32 = "uint32"


class FlattenMode(str, Enum):
    NONE = "none"
    SAFE = "safe"
    ALL = "all"


class InstancePolicy(str, Enum):
    AUTO = "auto"
    PRESERVE = "preserve"
    EXPAND = "expand"


class DecimateCriterion(str, Enum):
    TARGET = "target"
    QUALITY = "quality"


class BudgetScope(str, Enum):
    PART = "part"
    SELECTION = "selection"


class UVImportance(str, Enum):
    PRESERVE_ISLANDS = "preserve-islands"
    PRESERVE_SEAMS = "preserve-seams"
    IGNORE = "ignore"


class OcclusionStrategy(str, Enum):
    CONSERVATIVE = "conservative"
    EXTERIOR = "exterior"
    ADVANCED = "advanced"


class OcclusionLevel(str, Enum):
    PARTS = "parts"
    SUBMESHES = "submeshes"
    TRIANGLES = "triangles"


class LODPreset(str, Enum):
    DESKTOP = "desktop"
    WEB = "web"
    MOBILE = "mobile"
    VR = "vr"


class LODMode(str, Enum):
    VARIANTS = "variants"
    EXTRAS = "extras"
    SEPARATE = "separate"


class LODEngineProfile(str, Enum):
    GENERIC = "generic"
    UNITY = "unity"
    UNREAL = "unreal"


class UsdPackage(str, Enum):
    DEFAULT = "default"
    USDZ = "usdz"


class UsdLayout(str, Enum):
    AUTO = "auto"
    INSTANCED = "instanced"
    FLAT = "flat"


class MetadataMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"


class PmiMode(str, Enum):
    NONE = "none"
    SUMMARY = "summary"
    FULL = "full"
    METADATA = "metadata"
    METADATA_AND_VISUALS = "metadata-and-visuals"
