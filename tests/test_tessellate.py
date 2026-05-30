from __future__ import annotations

import json

import numpy as np
import pytest

from fascat.asset import Asset, Node, Part
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.ops.tessellate import _apply_mesh_tessellation_controls
from fascat.options import TessellationOptions


def triangle_mesh() -> Mesh:
    return Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )


def test_tessellate_deduplicates_parts_by_mesh_fingerprint() -> None:
    mesh = triangle_mesh()
    fingerprint = mesh.fingerprint()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(id="part_a", name="Part A", mesh=mesh, fingerprint=fingerprint),
            "part_b": Part(id="part_b", name="Part B", mesh=mesh.copy(), fingerprint=fingerprint),
        },
    )

    tessellated = asset.tessellate(TessellationOptions())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert tessellated.part_count == 1
    assert part_ids == ["part_a", "part_a"]


def test_tessellate_keeps_distinct_per_face_material_assignments() -> None:
    mesh = triangle_mesh()
    red_face_mesh = Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces.copy(),
        material_indices=np.array([0], dtype=int),
    )
    blue_face_mesh = Mesh(
        points=mesh.points.copy(),
        faces=mesh.faces.copy(),
        material_indices=np.array([1], dtype=int),
    )
    fingerprint = mesh.fingerprint()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(
                id="part_a",
                name="Part A",
                mesh=red_face_mesh,
                material_ids=["red", "blue"],
                fingerprint=fingerprint,
            ),
            "part_b": Part(
                id="part_b",
                name="Part B",
                mesh=blue_face_mesh,
                material_ids=["red", "blue"],
                fingerprint=fingerprint,
            ),
        },
    )

    tessellated = asset.tessellate(TessellationOptions())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert tessellated.part_count == 2
    assert part_ids == ["part_a", "part_b"]


def test_tessellation_edge_controls_skip_second_pass_when_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_subdivide(self: Mesh, _max_edge_length: float) -> Mesh:
        nonlocal calls
        calls += 1
        return self.copy()

    monkeypatch.setattr(Mesh, "subdivide_long_edges", fake_subdivide)

    result = _apply_mesh_tessellation_controls(triangle_mesh(), TessellationOptions(max_edge_length=10.0))

    assert calls == 1
    assert result.metadata["tessellation_edge_control_passes"] == "1"


def test_tessellation_edge_controls_rerun_when_first_pass_changes_mesh(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_subdivide(self: Mesh, _max_edge_length: float) -> Mesh:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Mesh(
                points=np.vstack([self.points, np.array([[0.5, 0.5, 0.0]], dtype=float)]),
                faces=np.array([[0, 1, 3], [1, 2, 3]], dtype=int),
                metadata=dict(self.metadata),
            )
        return self.copy()

    monkeypatch.setattr(Mesh, "subdivide_long_edges", fake_subdivide)

    result = _apply_mesh_tessellation_controls(triangle_mesh(), TessellationOptions(max_edge_length=0.5))

    assert calls == 2
    assert result.triangle_count == 2
    assert result.metadata["tessellation_edge_control_passes"] == "2"


def test_tessellate_reuses_source_shape_mesh_for_matching_parts(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(id="part_a", name="Part A", source_shape=source_shape, material_ids=["red"]),
            "part_b": Part(id="part_b", name="Part B", source_shape=source_shape, material_ids=["red"]),
        },
    )
    calls: list[object] = []

    def fake_tessellate_shape(
        shape: object,
        _options: TessellationOptions,
        *,
        face_material_indices: list[int] | None = None,
    ) -> Mesh:
        calls.append(shape)
        assert face_material_indices is None
        return triangle_mesh()

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(TessellationOptions())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert calls == [source_shape]
    assert tessellated.part_count == 1
    assert part_ids == ["part_a", "part_a"]


def test_tessellate_source_shape_cache_respects_face_material_assignments(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    root = Node(
        id="root",
        name="root",
        children=[
            Node(id="node_a", name="A", part_id="part_a"),
            Node(id="node_b", name="B", part_id="part_b"),
        ],
    )
    asset = Asset(
        root=root,
        parts={
            "part_a": Part(
                id="part_a",
                name="Part A",
                source_shape=source_shape,
                material_ids=["red", "blue"],
                metadata={"occt_face_material_indices": "0"},
            ),
            "part_b": Part(
                id="part_b",
                name="Part B",
                source_shape=source_shape,
                material_ids=["red", "blue"],
                metadata={"occt_face_material_indices": "1"},
            ),
        },
    )
    calls: list[list[int] | None] = []

    def fake_tessellate_shape(
        shape: object,
        _options: TessellationOptions,
        *,
        face_material_indices: list[int] | None = None,
    ) -> Mesh:
        assert shape is source_shape
        calls.append(face_material_indices)
        mesh = triangle_mesh()
        if face_material_indices is not None:
            mesh.material_indices = np.asarray(face_material_indices, dtype=int)
        return mesh

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(TessellationOptions())
    part_ids = [node.part_id for node in tessellated.root.walk() if node.part_id is not None]

    assert calls == [[0], [1]]
    assert tessellated.part_count == 2
    assert part_ids == ["part_a", "part_b"]


def test_tessellation_keep_brep_controls_source_shape_retention(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    root = Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")])
    asset = Asset(root=root, parts={"part": Part(id="part", name="Part", source_shape=source_shape)})
    calls: list[object] = []

    def fake_tessellate_shape(shape: object, _options: TessellationOptions, **_kwargs: object) -> Mesh:
        calls.append(shape)
        return triangle_mesh().compute_normals()

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    dropped = asset.tessellate(TessellationOptions(keep_brep=False))
    kept = asset.tessellate(TessellationOptions(keep_brep=True))

    assert calls == [source_shape, source_shape]
    assert dropped.parts["part"].source_shape is None
    assert kept.parts["part"].source_shape is source_shape
    assert dropped.parts["part"].metadata["brep_patch_cleanup"] == "deleted"
    assert dropped.parts["part"].metadata["source_shape_retained"] == "false"
    assert kept.parts["part"].metadata["brep_patch_cleanup"] == "retained"
    assert kept.parts["part"].metadata["source_shape_retained"] == "true"
    dropped_sources = json.loads(str(dropped.parts["part"].metadata["tessellation_attribute_sources"]))
    kept_sources = json.loads(str(kept.parts["part"].metadata["tessellation_attribute_sources"]))
    assert dropped_sources["positions"] == "tessellation"
    assert dropped_sources["normals"] == "tessellation"
    assert dropped_sources["uvs"] == {"status": "not_generated_by_tessellation"}
    assert dropped_sources["brep_patches"] == "deleted"
    assert kept_sources["brep_patches"] == "retained"


def test_tessellation_attribute_sources_record_reused_meshes() -> None:
    mesh = triangle_mesh().compute_normals()
    mesh.uvs[0] = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    mesh.face_groups = {"imported_face": np.array([0], dtype=int)}
    root = Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")])
    asset = Asset(
        root=root,
        parts={"part": Part(id="part", name="Part", mesh=mesh, source_shape=object())},
    )

    tessellated = asset.tessellate(TessellationOptions(free_edge_report=True))
    part = tessellated.parts["part"]
    sources = json.loads(str(part.metadata["tessellation_attribute_sources"]))

    assert sources == {
        "brep_patches": "unchanged_existing_mesh_reuse",
        "face_groups": "imported_mesh",
        "free_edges": "diagnostic_only",
        "normals": "imported_mesh",
        "positions": "imported_mesh",
        "tangents": "missing",
        "triangles": "imported_mesh",
        "uvs": {"0": "imported_mesh"},
    }
    assert part.mesh is not None
    assert json.loads(str(part.mesh.metadata["tessellation_attribute_sources"])) == sources


def test_tessellation_warns_about_retained_patch_and_submesh_risk(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    material_ids = [f"mat_{index}" for index in range(16)]
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Panel", part_id="part")]),
        parts={
            "part": Part(
                id="part",
                name="Panel",
                source_shape=source_shape,
                material_ids=material_ids,
                metadata={"source_faces": "65"},
            )
        },
    )

    def fake_tessellate_shape(shape: object, _options: TessellationOptions, **_kwargs: object) -> Mesh:
        assert shape is source_shape
        face_count = 65
        points = np.asarray(
            [
                point
                for face_index in range(face_count)
                for point in ((face_index, 0, 0), (face_index, 1, 0), (face_index, 0, 1))
            ],
            dtype=float,
        )
        faces = np.asarray([[index * 3, index * 3 + 1, index * 3 + 2] for index in range(face_count)], dtype=int)
        return Mesh(
            points=points,
            faces=faces,
            material_indices=np.asarray([index % len(material_ids) for index in range(face_count)], dtype=int),
            face_groups={f"occt_face_{index}": np.asarray([index], dtype=int) for index in range(face_count)},
            metadata={"occt_faces": str(face_count)},
        )

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(TessellationOptions(keep_brep=True))
    part = tessellated.parts["part"]
    assert part.mesh is not None
    assert part.metadata["tessellation_face_groups"] == "65"
    assert part.metadata["tessellation_estimated_draw_calls"] == "16"
    assert part.metadata["tessellation_face_group_export_risk"] == "high"
    assert part.metadata["tessellation_draw_call_export_risk"] == "high"
    assert part.metadata["brep_retained_patch_count"] == "65"
    assert part.metadata["brep_patch_export_risk"] == "high"
    assert part.mesh.metadata["tessellation_face_groups"] == "65"
    assert part.mesh.metadata["brep_retained_patch_count"] == "65"
    assert tessellated.report.steps[-1].warnings == [
        "part has 65 CAD face group(s) after tessellation; "
        "per-face grouping can increase submesh or draw-call pressure: Panel",
        "part is estimated to emit 16 material draw call(s) after tessellation: Panel",
        "part retains 65 BREP patch(es) after tessellation; "
        "review draw-call and export-size risk before runtime export: Panel",
    ]


def test_tessellate_reuses_existing_meshes_by_default(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=triangle_mesh(), source_shape=source_shape)},
    )
    calls: list[object] = []

    def fake_tessellate_shape(shape: object, _options: TessellationOptions, **_kwargs: object) -> Mesh:
        calls.append(shape)
        return Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
        )

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(TessellationOptions())

    assert calls == []
    assert tessellated.parts["part"].mesh is not None
    assert tessellated.parts["part"].mesh.triangle_count == 1
    assert tessellated.parts["part"].source_shape is source_shape


def test_tessellate_replaces_existing_meshes_when_requested(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=triangle_mesh(), source_shape=source_shape)},
    )
    calls: list[object] = []

    def fake_tessellate_shape(shape: object, _options: TessellationOptions, **_kwargs: object) -> Mesh:
        calls.append(shape)
        return Mesh(
            points=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float),
            faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=int),
        )

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(TessellationOptions(reuse_existing_meshes=False))

    assert calls == [source_shape]
    assert tessellated.parts["part"].mesh is not None
    assert tessellated.parts["part"].mesh.triangle_count == 2
    assert tessellated.parts["part"].source_shape is None


def test_tessellate_warns_when_existing_mesh_cannot_be_retessellated() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=triangle_mesh())},
    )

    tessellated = asset.tessellate(TessellationOptions(reuse_existing_meshes=False))

    assert tessellated.parts["part"].mesh is not None
    assert tessellated.report.steps[-1].warnings == [
        "part has existing mesh but no source shape and cannot be retessellated: Part"
    ]


def test_occt_mesh_parameters_use_sag_ratio_as_relative_deflection() -> None:
    import fascat.ops.tessellate as tessellate_module

    class Parameters:
        pass

    parameters = tessellate_module._occt_mesh_parameters(
        TessellationOptions(sag=0.25, sag_ratio=0.01, angle=20.0, relative=False, curvature_adaptive=True),
        Parameters,
    )

    assert parameters.Deflection == 0.01
    assert parameters.Relative is True
    assert parameters.DeflectionInterior == 0.005
    assert parameters.ControlSurfaceDeflection is True
    assert parameters.ForceFaceDeflection is True


def test_tessellation_report_includes_unit_aware_tolerance_policy() -> None:
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1_000_000, 0, 0], [0, 1_000_000, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Panel", part_id="part")]),
        parts={"part": Part(id="part", name="Panel", mesh=mesh, source_shape=object())},
        units="metre",
        meters_per_unit=1.0,
        metadata={"source_units": "millimetre", "source_meters_per_unit": "0.001"},
    )

    tessellated = asset.tessellate(
        TessellationOptions(
            sag=5000.0,
            relative=False,
            max_polygon_length=2_000_000.0,
        )
    )

    step_policy = tessellated.report.steps[-1].options["tolerance_policy"]
    assert step_policy["coordinate_space"] == "source_local"
    assert step_policy["effective_units"] == "millimetre"
    assert step_policy["target_units"] == "metre"
    assert step_policy["active_deflection_kind"] == "absolute_sag"
    assert step_policy["sag_meters"] == 5.0
    assert step_policy["sag_target_units"] == 5.0
    assert step_policy["max_polygon_length_meters"] == 2000.0
    assert step_policy["max_polygon_length_target_units"] == 2000.0

    part = tessellated.parts["part"]
    assert part.mesh is not None
    policy = json.loads(str(part.metadata["tessellation_tolerance_policy"]))
    assert [advisory["code"] for advisory in policy["advisories"]] == [
        "coarse_normalized_sag",
        "coarse_normalized_max_polygon_length",
    ]
    assert part.metadata["tessellation_effective_units"] == "millimetre"
    assert part.metadata["tessellation_target_units"] == "metre"
    assert part.metadata["tessellation_sag_meters"] == "5"
    assert part.metadata["tessellation_sag_target_units"] == "5"
    assert part.metadata["tessellation_max_polygon_length_meters"] == "2000"
    assert part.metadata["tessellation_tolerance_advisory_codes"] == (
        "coarse_normalized_sag,coarse_normalized_max_polygon_length"
    )
    assert part.mesh.metadata["tessellation_tolerance_policy"] == part.metadata["tessellation_tolerance_policy"]
    assert part.mesh.metadata["tessellation_max_polygon_length_target_units"] == "2000"
    assert tessellated.report.steps[-1].warnings == [
        "tessellation sag converts to a very large target-space tolerance after unit normalization; "
        "verify sag is specified in source/local units: Panel",
        "tessellation max_polygon_length converts to a very large target-space length after unit normalization; "
        "verify it is specified in source/local units: Panel",
    ]


def test_relative_tessellation_policy_keeps_sag_unitless() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=triangle_mesh(), source_shape=object())},
        units="metre",
        meters_per_unit=1.0,
        metadata={"source_units": "millimetre", "source_meters_per_unit": "0.001"},
    )

    tessellated = asset.tessellate(TessellationOptions(sag=0.05, relative=True))

    policy = json.loads(str(tessellated.parts["part"].metadata["tessellation_tolerance_policy"]))
    assert policy["active_deflection_kind"] == "relative_sag"
    assert policy["active_deflection_relative"] is True
    assert "sag_meters" not in policy
    assert "tessellation_sag_meters" not in tessellated.parts["part"].metadata
    assert tessellated.report.steps[-1].warnings == []


def test_tessellate_cache_respects_per_part_settings_and_records_quality(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="node_a", name="A", part_id="part_a"),
                Node(id="node_b", name="B", part_id="part_b"),
            ],
        ),
        parts={
            "part_a": Part(id="part_a", name="Part A", source_shape=source_shape),
            "part_b": Part(id="part_b", name="Part B", source_shape=source_shape),
        },
    )
    calls: list[tuple[float, float | None]] = []

    def fake_tessellate_shape(
        shape: object,
        options: TessellationOptions,
        *,
        face_material_indices: list[int] | None = None,
    ) -> Mesh:
        assert shape is source_shape
        assert face_material_indices is None
        calls.append((options.sag, options.sag_ratio))
        return triangle_mesh()

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(
        TessellationOptions(
            sag=0.1,
            quality_report=True,
            part_settings={"part_b": {"sag": 0.25, "sag_ratio": 0.01, "max_edge_length": 0.75}},
        )
    )

    assert calls == [(0.1, None), (0.25, 0.01)]
    report = tessellated.tessellation_quality_report()
    assert report["summary"]["parts"] == 2
    part_b_payload = json.loads(str(tessellated.parts["part_b"].metadata["tessellation_quality"]))
    assert part_b_payload["options"]["sag"] == 0.25
    assert part_b_payload["options"]["sag_ratio"] == 0.01
    assert part_b_payload["options"]["max_edge_length"] == 0.75


def test_tessellation_quality_report_uses_max_polygon_length(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", source_shape=source_shape)},
    )

    def fake_tessellate_shape(shape: object, _options: TessellationOptions, **_kwargs: object) -> Mesh:
        assert shape is source_shape
        return triangle_mesh()

    monkeypatch.setattr(tessellate_module, "tessellate_shape", fake_tessellate_shape)

    tessellated = asset.tessellate(
        TessellationOptions(quality_report=True, max_edge_length=5.0, max_polygon_length=0.5)
    )
    payload = json.loads(str(tessellated.parts["part"].metadata["tessellation_quality"]))

    assert payload["options"]["max_edge_length"] == 5.0
    assert payload["options"]["max_polygon_length"] == 0.5
    assert payload["metrics"]["long_edges"] == 3
    assert tessellated.parts["part"].metadata["tessellation_long_polygon_edges"] == "3"
    assert tessellated.report.steps[-1].warnings == [
        "part has 3 tessellated edges longer than max_polygon_length: Part"
    ]


def test_tessellation_max_polygon_length_warns_without_quality_report(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", source_shape=source_shape)},
    )

    monkeypatch.setattr(tessellate_module, "tessellate_shape", lambda *_args, **_kwargs: triangle_mesh())

    tessellated = asset.tessellate(TessellationOptions(max_polygon_length=0.5))

    assert "tessellation_quality" not in tessellated.parts["part"].metadata
    assert tessellated.parts["part"].metadata["tessellation_long_polygon_edges"] == "3"
    assert tessellated.report.steps[-1].warnings == [
        "part has 3 tessellated edges longer than max_polygon_length: Part"
    ]


def test_tessellation_quality_advisor_warns_on_coarse_absolute_sag(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Small", part_id="part")]),
        parts={"part": Part(id="part", name="Small", source_shape=source_shape)},
    )

    monkeypatch.setattr(tessellate_module, "tessellate_shape", lambda *_args, **_kwargs: triangle_mesh())

    tessellated = asset.tessellate(TessellationOptions(sag=0.1, relative=False, quality_report=True))
    part = tessellated.parts["part"]
    payload = json.loads(str(part.metadata["tessellation_quality"]))
    advisories = json.loads(str(part.metadata["tessellation_quality_advisories"]))

    assert part.metadata["tessellation_quality_advisory_count"] == "1"
    assert part.metadata["tessellation_quality_advisory_codes"] == "coarse_absolute_sag"
    assert advisories[0]["code"] == "coarse_absolute_sag"
    assert payload["advisories"][0]["code"] == "coarse_absolute_sag"
    assert tessellated.report.steps[-1].warnings == [
        "tessellation sag is 7.1% of the part bounding-box diagonal; "
        "small or high-detail features may be undersampled: Small"
    ]


def test_tessellation_quality_advisor_warns_on_aggressive_max_length(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.tessellate as tessellate_module

    source_shape = object()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Panel", part_id="part")]),
        parts={"part": Part(id="part", name="Panel", source_shape=source_shape)},
    )

    monkeypatch.setattr(tessellate_module, "tessellate_shape", lambda *_args, **_kwargs: triangle_mesh())

    tessellated = asset.tessellate(TessellationOptions(max_edge_length=0.01))
    part = tessellated.parts["part"]
    advisories = json.loads(str(part.metadata["tessellation_quality_advisories"]))

    assert part.metadata["tessellation_quality_advisory_count"] == "1"
    assert part.metadata["tessellation_quality_advisory_codes"] == "aggressive_max_length"
    assert advisories[0]["length_kind"] == "max_edge_length"
    assert "reserve aggressive polygon-length limits" in tessellated.report.steps[-1].warnings[0]


def test_tessellation_quality_advisor_flags_shiny_high_detail_parts() -> None:
    material = Material(
        id="chrome",
        name="Polished chrome",
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=1.0,
        roughness=0.2,
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Inspection", part_id="part")]),
        parts={
            "part": Part(
                id="part",
                name="Inspection",
                mesh=triangle_mesh(),
                material_ids=[material.id],
                metadata={"surface_detail": "high"},
            )
        },
        materials={material.id: material},
    )

    tessellated = asset.tessellate(TessellationOptions(quality_report=True))
    part = tessellated.parts["part"]
    advisories = json.loads(str(part.metadata["tessellation_quality_advisories"]))
    payload = json.loads(str(part.metadata["tessellation_quality"]))

    assert part.metadata["tessellation_quality_advisory_count"] == "1"
    assert part.metadata["tessellation_quality_advisory_codes"] == "detail_sensitive_tessellation"
    assert advisories[0]["detail_contexts"] == ["high_detail_metadata", "shiny_material"]
    assert advisories[0]["recommendation"] == "set per-part sag_ratio or enable curvature_adaptive for this part"
    assert payload["advisories"][0]["code"] == "detail_sensitive_tessellation"
    assert tessellated.report.steps[-1].warnings == [
        "part has shiny or high-detail material/metadata but tessellation uses bulk criteria "
        "without sag_ratio or curvature_adaptive; consider finer per-part tessellation: Inspection"
    ]


def test_tessellation_quality_advisor_accepts_detail_tuned_settings() -> None:
    material = Material(
        id="chrome",
        name="Polished chrome",
        base_color=(0.8, 0.8, 0.8, 1.0),
        metallic=1.0,
        roughness=0.2,
    )
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Inspection", part_id="part")]),
        parts={
            "part": Part(
                id="part",
                name="Inspection",
                mesh=triangle_mesh(),
                material_ids=[material.id],
                metadata={"surface_detail": "high"},
            )
        },
        materials={material.id: material},
    )

    tessellated = asset.tessellate(TessellationOptions(sag_ratio=0.01, quality_report=True))
    payload = json.loads(str(tessellated.parts["part"].metadata["tessellation_quality"]))

    assert "tessellation_quality_advisories" not in tessellated.parts["part"].metadata
    assert payload["advisories"] == []
    assert tessellated.report.steps[-1].warnings == []


def test_tessellation_free_edge_report_records_reused_meshes() -> None:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="Part", part_id="part")]),
        parts={"part": Part(id="part", name="Part", mesh=triangle_mesh())},
    )

    tessellated = asset.tessellate(TessellationOptions(free_edge_report=True))
    part = tessellated.parts["part"]

    assert part.mesh is not None
    assert part.metadata["tessellation_free_edges"] == "3"
    assert part.metadata["tessellation_non_manifold_edges"] == "0"
    assert part.mesh.metadata["tessellation_free_edges"] == "3"
    assert part.mesh.metadata["tessellation_non_manifold_edges"] == "0"
    assert tessellated.report.steps[-1].warnings == ["part has 3 free tessellation edges: Part"]
