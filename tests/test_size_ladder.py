from __future__ import annotations

from fascat.options import GltfExportOptions
from fascat.size_ladder import GltfSizeLadderReport, GltfSizeLadderVariant


def test_gltf_size_ladder_report_summarizes_measured_variants() -> None:
    report = GltfSizeLadderReport(
        variants=(
            GltfSizeLadderVariant("baseline", GltfExportOptions(), "measured", file_size_bytes=100),
            GltfSizeLadderVariant(
                "requested",
                GltfExportOptions(quantize=True),
                "measured",
                file_size_bytes=70,
            ),
            GltfSizeLadderVariant("draco", GltfExportOptions(draco=True), "unavailable", error="missing encoder"),
        ),
        warnings=("glTF size ladder variant draco could not be measured: missing encoder",),
    )

    after = report.to_step_after({"triangles": 1})
    options = report.to_step_options()

    assert report.baseline_bytes == 100
    assert report.requested_bytes == 70
    assert after["size_ladder_variants"] == 3
    assert after["size_ladder_measured_variants"] == 2
    assert after["size_ladder_unavailable_variants"] == 1
    assert after["size_ladder_best_savings_bytes"] == 30
    assert options["variants"][1]["ratio_to_baseline"] == 0.7
    assert options["variants"][2]["error"] == "missing encoder"
