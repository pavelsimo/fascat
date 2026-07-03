from __future__ import annotations

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.export_report import export_image_counts, export_material_counts, stats_with_file_size
from fascat.image import ImageResource
from fascat.material import Material
from fascat.mesh import Mesh


def _asset() -> Asset:
    mesh = Mesh(
        points=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.asarray([[0, 1, 2]], dtype=int),
        material_indices=np.asarray([0], dtype=int),
    )
    return Asset(
        root=Node(id="root", name="root", children=[Node(id="tri", name="Triangle", part_id="tri")]),
        parts={"tri": Part(id="tri", name="Triangle", mesh=mesh, material_ids=["paint"])},
        materials={
            "paint": Material(
                id="paint",
                name="Paint",
                base_color=(0.2, 0.4, 0.6, 1.0),
                metadata={"baked_texture_base_color_image": "albedo"},
            ),
            "unused": Material(id="unused", name="Unused", base_color=(1.0, 0.0, 0.0, 1.0)),
        },
        images={
            "albedo": ImageResource(id="albedo", name="Albedo", mime_type="image/png", data=b"abc", width=1, height=1),
            "unused": ImageResource(id="unused", name="Unused", mime_type="image/png", data=b"z", width=1, height=1),
        },
    )


def test_export_report_counts_referenced_materials_and_images() -> None:
    asset = _asset()

    assert export_material_counts(asset) == {
        "export_source_material_count": 2,
        "export_referenced_material_count": 1,
        "export_unused_material_count": 1,
        "export_written_material_count": 1,
    }
    assert export_image_counts(asset) == {
        "export_source_image_count": 2,
        "export_source_image_reference_count": 1,
        "export_referenced_image_count": 1,
        "export_referenced_image_reference_count": 1,
        "export_unused_image_count": 1,
        "export_duplicate_image_reference_count": 0,
        "export_written_image_count": 1,
    }


def test_stats_with_file_size_records_budget_warning(tmp_path) -> None:  # type: ignore[no-untyped-def]
    asset = _asset()
    output = tmp_path / "out.glb"
    output.write_bytes(b"x" * 10)

    stats = stats_with_file_size({"triangles": 1}, output, 0.000001, asset)

    assert stats["file_size_bytes"] == 10
    assert stats["file_size_budget_bytes"] == 1
    assert stats["export_referenced_material_count"] == 1
    assert stats["export_estimated_payload_bytes"] >= stats["export_estimated_geometry_bytes"]
    assert asset.report.warnings == ["file size budget exceeded: 10 bytes > 1 bytes"]
