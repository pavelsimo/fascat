"""Validation and measurement harnesses.

One import surface for the runtime, parity, visual-diff, and size-ladder
machinery that used to live at the top level. These stacks pull heavy
dependencies (Pillow, subprocess harnesses), so they stay out of the core
``fascat`` namespace.
"""

from __future__ import annotations

from fascat.runtime import (
    RuntimeBrowserOptions,
    RuntimeBrowserRenderOptions,
    RuntimeBrowserRenderReport,
    RuntimeBrowserReport,
    measure_browser_runtime,
    write_browser_render_preview,
)
from fascat.size_ladder import GltfSizeLadderReport, GltfSizeLadderVariant, measure_gltf_size_ladder
from fascat.visual import (
    LodSwitchPreviewReport,
    TurntableOptions,
    TurntableReport,
    TurntableViewReport,
    VisualComparisonReport,
    VisualDiffOptions,
    VisualDiffReport,
    VisualPreviewOptions,
    VisualPreviewReport,
    compare_images,
    write_before_after_previews,
    write_lod_switch_previews,
    write_output_lod_switch_previews,
    write_output_preview,
    write_output_turntable_previews,
    write_preview,
    write_turntable_previews,
)

__all__ = [
    "GltfSizeLadderReport",
    "GltfSizeLadderVariant",
    "LodSwitchPreviewReport",
    "RuntimeBrowserOptions",
    "RuntimeBrowserRenderOptions",
    "RuntimeBrowserRenderReport",
    "RuntimeBrowserReport",
    "TurntableOptions",
    "TurntableReport",
    "TurntableViewReport",
    "VisualComparisonReport",
    "VisualDiffOptions",
    "VisualDiffReport",
    "VisualPreviewOptions",
    "VisualPreviewReport",
    "compare_images",
    "measure_browser_runtime",
    "measure_gltf_size_ladder",
    "write_before_after_previews",
    "write_browser_render_preview",
    "write_lod_switch_previews",
    "write_output_lod_switch_previews",
    "write_output_preview",
    "write_output_turntable_previews",
    "write_preview",
    "write_turntable_previews",
]
