from __future__ import annotations

from dataclasses import replace

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


def test_size_ladder_rungs_inherit_non_compression_options() -> None:
    from fascat.options import MetadataExportOptions
    from fascat.size_ladder import _size_ladder_variants

    requested = GltfExportOptions(
        preset="web",
        draco=True,
        draco_compression_level=10,
        draco_quantize_position=16,
        jpeg_quality=70,
        file_size_budget_mb=12.0,
        metadata=MetadataExportOptions(),
    )

    rungs = dict(_size_ladder_variants(requested))

    for name, options in rungs.items():
        assert options.draco_compression_level == 10, name
        assert options.draco_quantize_position == 16, name
        assert options.jpeg_quality == 70, name
        assert options.file_size_budget_mb == 12.0, name
    assert rungs["baseline"].quantize is False
    assert rungs["baseline"].draco is False
    assert rungs["baseline"].texture_compression is None
    assert rungs["draco"].draco is True
    assert rungs["draco"].quantize is False
    assert rungs["requested"] == replace(requested, size_ladder=False)


def test_size_ladder_smallest_excludes_the_uncompressed_baseline() -> None:
    report = GltfSizeLadderReport(
        variants=(
            GltfSizeLadderVariant("baseline", GltfExportOptions(), "measured", file_size_bytes=100),
            GltfSizeLadderVariant("draco", GltfExportOptions(draco=True), "measured", file_size_bytes=140),
        )
    )

    after = report.to_step_after({})

    assert after["size_ladder_smallest_bytes"] == 140
    assert after["size_ladder_best_savings_bytes"] == 0
