from __future__ import annotations

import fascat.validation as validation


def test_validation_module_reexports_runtime_visual_and_size_ladder_surface() -> None:
    exported = set(validation.__all__)

    assert "GltfSizeLadderReport" in exported
    assert "RuntimeBrowserReport" in exported
    assert "VisualDiffReport" in exported
    assert validation.GltfSizeLadderReport.__name__ == "GltfSizeLadderReport"
    assert callable(validation.measure_gltf_size_ladder)
