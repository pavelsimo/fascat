from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.mesh import Mesh
from fascat.ops.heal import BrepHealDiagnostics, BrepStatus, _face_overlap_descriptor, brep_status, heal_shape
from fascat.options import BrepHealOptions, ConversionProfile, RepairOptions, StageOptions
from fascat.pipeline import convert

runner = CliRunner()


def _asset_with_brep() -> Asset:
    return Asset(
        root=Node(
            id="root",
            name="root",
            children=[
                Node(id="selected", name="Selected", part_id="selected"),
                Node(id="other", name="Other", part_id="other"),
            ],
        ),
        parts={
            "selected": Part(id="selected", name="Selected", source_shape={"shape": "selected"}),
            "other": Part(id="other", name="Other", source_shape={"shape": "other"}),
        },
    )


def test_heal_brep_scopes_to_selected_parts_and_records_status(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    def fake_heal_shape(
        shape: object, _options: BrepHealOptions
    ) -> tuple[object, BrepStatus, BrepStatus, list[str], BrepHealDiagnostics]:
        return (
            {"healed": shape},
            BrepStatus(kind="open_surface", shells=1, faces=3, open_shells=1),
            BrepStatus(kind="solid", solids=1, faces=3, open_shells=0),
            ["fixed trims"],
            BrepHealDiagnostics(),
        )

    monkeypatch.setattr(heal, "heal_shape", fake_heal_shape)

    healed = _asset_with_brep().heal_brep(BrepHealOptions(tolerance=0.1), where=Filter.part("selected"))

    assert healed.parts["selected"].source_shape == {"healed": {"shape": "selected"}}
    assert healed.parts["other"].source_shape == {"shape": "other"}
    assert healed.parts["selected"].metadata["brep_kind"] == "solid"
    assert healed.parts["selected"].metadata["brep_open_shells"] == "0"
    assert healed.parts["selected"].metadata["brep_free_edges"] == "0"
    assert healed.parts["selected"].metadata["brep_small_edges"] == "0"
    assert (
        healed.parts["selected"].metadata["brep_heal_operations"]
        == "group_open_shells,fix_edges,unify_tolerances,sew_faces,unify_same_domain,remove_overlapping_faces"
    )
    assert healed.report.warnings == ["Selected: fixed trims"]
    assert healed.report.steps[-1].name == "heal_brep"
    assert healed.report.steps[-1].options["matched"]["parts"] == 1


def test_heal_brep_report_includes_unit_aware_tolerance_policy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    def fake_heal_shape(
        shape: object, _options: BrepHealOptions
    ) -> tuple[object, BrepStatus, BrepStatus, list[str], BrepHealDiagnostics]:
        return (
            shape,
            BrepStatus(kind="surface", faces=1),
            BrepStatus(kind="surface", faces=1),
            [],
            BrepHealDiagnostics(),
        )

    monkeypatch.setattr(heal, "heal_shape", fake_heal_shape)
    asset = _asset_with_brep()
    asset.units = "metre"
    asset.meters_per_unit = 1.0
    asset.metadata = {"source_units": "millimetre", "source_meters_per_unit": 0.001}

    healed = asset.heal_brep(BrepHealOptions(tolerance=2.0, max_sliver_area=3.0))

    step = healed.report.steps[-1]
    policy = step.options["tolerance_policy"]
    assert isinstance(policy, dict)
    assert policy["coordinate_space"] == "source_local"
    assert policy["effective_units"] == "millimetre"
    assert policy["target_units"] == "metre"
    assert policy["heal_tolerance_meters"] == pytest.approx(0.002)
    assert policy["max_sliver_area_square_meters"] == pytest.approx(0.000003)
    assert policy["operations"]["open_shell_grouping"] == "enabled"
    assert policy["operations"]["t_junction_sewing"] == "not_implemented"
    assert healed.parts["selected"].metadata["brep_heal_effective_units"] == "millimetre"
    assert healed.parts["selected"].metadata["brep_heal_target_units"] == "metre"
    assert healed.parts["selected"].metadata["brep_heal_heal_tolerance_meters"] == "0.002"
    assert healed.parts["selected"].metadata["brep_heal_max_sliver_area_square_meters"] == "3e-06"


def test_heal_brep_records_same_domain_cleanup_metadata(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    def fake_heal_shape(
        shape: object, _options: BrepHealOptions
    ) -> tuple[object, BrepStatus, BrepStatus, list[str], BrepHealDiagnostics]:
        assert _options.unify_same_domain is True
        return (
            shape,
            BrepStatus(kind="shell", shells=1, edges=12, faces=6),
            BrepStatus(kind="shell", shells=1, edges=10, faces=5),
            [],
            BrepHealDiagnostics(
                faces_removed=1,
                edges_removed=2,
                same_domain_faces_removed=1,
                same_domain_edges_removed=2,
            ),
        )

    monkeypatch.setattr(heal, "heal_shape", fake_heal_shape)

    healed = _asset_with_brep().heal_brep(BrepHealOptions())
    selected = healed.parts["selected"]
    policy = healed.report.steps[-1].options["tolerance_policy"]

    assert selected.metadata["brep_same_domain_faces_removed"] == "1"
    assert selected.metadata["brep_same_domain_edges_removed"] == "2"
    assert selected.metadata["brep_heal_same_domain_cleanup"] == "enabled"
    assert isinstance(policy, dict)
    assert policy["operations"]["same_domain_cleanup"] == "enabled"


def test_brep_status_dict_includes_topology_risk_counts() -> None:
    status = BrepStatus(
        kind="open_surface",
        shells=1,
        wires=2,
        edges=7,
        faces=3,
        open_shells=1,
        free_edges=4,
        small_edges=2,
        sliver_faces=1,
    )

    assert status.to_dict() == {
        "kind": "open_surface",
        "solids": 0,
        "shells": 1,
        "wires": 2,
        "edges": 7,
        "faces": 3,
        "open_shells": 1,
        "free_edges": 4,
        "small_edges": 2,
        "sliver_faces": 1,
        "overlapping_face_pairs": 0,
        "z_fighting_faces": 0,
    }


def test_brep_status_reports_closed_box_topology() -> None:
    pytest.importorskip("OCP.BRepPrimAPI")
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox

    status = brep_status(BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape(), small_edge_length=0.5)

    assert status.kind == "solid"
    assert status.solids == 1
    assert status.faces == 6
    assert status.edges >= 12
    assert status.free_edges == 0
    assert status.small_edges == 0


def _two_open_shell_compound() -> object:
    pytest.importorskip("OCP.BRep")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Shell

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for offset in (0.0, 3.0):
        plane = gp_Pln(gp_Pnt(offset, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
        face = BRepBuilderAPI_MakeFace(plane, 0.0, 1.0, 0.0, 1.0).Face()
        shell = TopoDS_Shell()
        builder.MakeShell(shell)
        builder.Add(shell, face)
        builder.Add(compound, shell)
    return compound


def test_brep_status_counts_open_shells_from_free_edges() -> None:
    status = brep_status(_two_open_shell_compound())

    assert status.kind == "open_surface"
    assert status.shells == 2
    assert status.open_shells == 2
    assert status.free_edges == 8


def test_heal_shape_groups_open_shells_before_cleanup() -> None:
    shape = _two_open_shell_compound()

    healed_shape, before, after, warnings, diagnostics = heal_shape(shape, BrepHealOptions(tolerance=0.05))

    assert healed_shape is not shape
    assert warnings == []
    assert before.shells == 2
    assert before.open_shells == 2
    assert after.shells == 2
    assert after.open_shells == 2
    assert diagnostics.open_shell_groups == 2
    assert diagnostics.open_shell_grouped_shells == 2
    assert diagnostics.open_shell_grouped_faces == 2


def test_heal_brep_records_open_shell_grouping_metadata() -> None:
    shape = _two_open_shell_compound()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="skins", name="Skins", part_id="skins")]),
        parts={"skins": Part(id="skins", name="Skins", source_shape=shape)},
    )

    healed = asset.heal_brep(BrepHealOptions(tolerance=0.05))

    skins = healed.parts["skins"]
    policy = healed.report.steps[-1].options["tolerance_policy"]
    assert skins.metadata["brep_open_shell_grouping"] == "grouped"
    assert skins.metadata["brep_open_shell_groups"] == "2"
    assert skins.metadata["brep_open_shell_grouped_shells"] == "2"
    assert skins.metadata["brep_open_shell_grouped_faces"] == "2"
    assert skins.metadata["brep_heal_open_shell_grouping"] == "enabled"
    assert isinstance(policy, dict)
    assert policy["operations"]["open_shell_grouping"] == "enabled"


def _duplicate_coplanar_face_compound() -> object:
    pytest.importorskip("OCP.BRep")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    plane = gp_Pln(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(0.0, 0.0, 1.0))
    face_a = BRepBuilderAPI_MakeFace(plane, 0.0, 1.0, 0.0, 1.0).Face()
    face_b = BRepBuilderAPI_MakeFace(plane, 0.0, 1.0, 0.0, 1.0).Face()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, face_a)
    builder.Add(compound, face_b)
    return compound


def test_brep_status_detects_coplanar_z_fighting_faces() -> None:
    shape = _duplicate_coplanar_face_compound()

    status = brep_status(
        shape,
        detect_overlaps=True,
        overlap_tolerance=0.05,
        overlap_area_ratio=0.995,
    )

    assert status.faces == 2
    assert status.overlapping_face_pairs == 1
    assert status.z_fighting_faces == 2


def test_heal_shape_removes_redundant_overlapping_faces() -> None:
    shape = _duplicate_coplanar_face_compound()

    healed_shape, before, after, warnings, diagnostics = heal_shape(shape, BrepHealOptions(tolerance=0.05))

    assert healed_shape is not shape
    assert warnings == []
    assert before.faces == 2
    assert before.overlapping_face_pairs == 1
    assert after.faces == 1
    assert after.overlapping_face_pairs == 0
    assert after.z_fighting_faces == 0
    assert diagnostics.faces_removed == 1
    assert diagnostics.overlapping_faces_removed == 1


def test_heal_brep_records_overlap_cleanup_metadata() -> None:
    shape = _duplicate_coplanar_face_compound()
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="panel", name="Panel", part_id="panel")]),
        parts={"panel": Part(id="panel", name="Panel", source_shape=shape)},
    )

    healed = asset.heal_brep(BrepHealOptions(tolerance=0.05))

    panel = healed.parts["panel"]
    assert panel.metadata["brep_faces"] == "1"
    assert panel.metadata["brep_overlapping_face_pairs"] == "0"
    assert panel.metadata["brep_z_fighting_faces"] == "0"
    assert panel.metadata["brep_overlapping_face_pairs_resolved"] == "1"
    assert panel.metadata["brep_overlapping_faces_removed"] == "1"
    assert panel.metadata["brep_faces_removed"] == "1"
    assert panel.metadata["brep_heal_overlap_z_fighting_cleanup"] == "enabled"


def test_face_overlap_descriptor_uses_shared_occt_mesh_helpers() -> None:
    class FakePoint:
        def __init__(self, x: float, y: float, z: float = 0.0) -> None:
            self.x = x
            self.y = y
            self.z = z

        def X(self) -> float:
            return self.x

        def Y(self) -> float:
            return self.y

        def Z(self) -> float:
            return self.z

        def Transformed(self, _transform: object) -> object:
            raise AssertionError("heal overlap descriptors should transform nodes in bulk")

    class FakeArray:
        def __init__(self, values: list[object], lower: int) -> None:
            self.values = values
            self.lower = lower

        def Lower(self) -> int:
            return self.lower

        def Value(self, index: int) -> object:
            return self.values[index - self.lower]

    class FakeTriangle:
        def __init__(self, a: int, b: int, c: int) -> None:
            self.values = (a, b, c)

        def Get(self) -> tuple[int, int, int]:
            return self.values

    class FakeTriangulation:
        def __init__(self) -> None:
            self.nodes = [
                FakePoint(0.0, 0.0),
                FakePoint(1.0, 0.0),
                FakePoint(0.0, 1.0),
                FakePoint(1.0, 1.0),
            ]
            self.triangles = [FakeTriangle(1, 2, 3), FakeTriangle(2, 4, 3)]

        def NbNodes(self) -> int:
            return len(self.nodes)

        def NbTriangles(self) -> int:
            return len(self.triangles)

        def MapNodeArray(self) -> FakeArray:
            return FakeArray(self.nodes, lower=5)

        def MapTriangleArray(self) -> FakeArray:
            return FakeArray(self.triangles, lower=9)

    class FakeTransform:
        def __init__(self) -> None:
            self.matrix = (
                (-2.0, 0.0, 0.0, 10.0),
                (0.0, 3.0, 0.0, 20.0),
                (0.0, 0.0, 1.0, 30.0),
            )

        def Value(self, row: int, column: int) -> float:
            return self.matrix[row - 1][column - 1]

    descriptor = _face_overlap_descriptor(
        7,
        object(),
        FakeTriangulation(),
        FakeTransform(),
        reversed_face=True,
    )

    assert descriptor is not None
    np.testing.assert_allclose(
        descriptor.points,
        [
            [10.0, 20.0, 30.0],
            [8.0, 20.0, 30.0],
            [10.0, 23.0, 30.0],
            [8.0, 23.0, 30.0],
        ],
    )
    np.testing.assert_array_equal(descriptor.triangles, [[2, 1, 0], [2, 3, 1]])
    np.testing.assert_allclose(descriptor.triangle_areas, [3.0, 3.0])
    assert descriptor.area == pytest.approx(6.0)
    np.testing.assert_allclose(descriptor.normal, [0.0, 0.0, 1.0])
    assert descriptor.plane_offset == pytest.approx(30.0)


def test_heal_brep_reports_remaining_topology_risks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    def fake_heal_shape(
        shape: object, _options: BrepHealOptions
    ) -> tuple[object, BrepStatus, BrepStatus, list[str], BrepHealDiagnostics]:
        return (
            shape,
            BrepStatus(kind="open_surface", shells=1, wires=1, edges=5, faces=2, open_shells=1, free_edges=4),
            BrepStatus(
                kind="open_surface",
                shells=1,
                wires=1,
                edges=5,
                faces=2,
                open_shells=1,
                free_edges=2,
                small_edges=1,
            ),
            [],
            BrepHealDiagnostics(),
        )

    monkeypatch.setattr(heal, "heal_shape", fake_heal_shape)

    healed = _asset_with_brep().heal_brep(BrepHealOptions(tolerance=0.25), where=Filter.part("selected"))

    selected = healed.parts["selected"]
    assert selected.metadata["brep_edges"] == "5"
    assert selected.metadata["brep_free_edges"] == "2"
    assert selected.metadata["brep_unstitched_edges"] == "2"
    assert selected.metadata["brep_small_edges"] == "1"
    assert "free_edges': 2" in selected.metadata["brep_after"]
    assert healed.report.warnings == [
        "Selected: BREP healing left 1 open shell(s)",
        "Selected: BREP healing left 2 free/unstitched edge(s)",
        "Selected: BREP healing left 1 edge(s) at or below tolerance 0.25",
    ]


def test_heal_brep_can_fail_on_remaining_open_shells(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    monkeypatch.setattr(
        heal,
        "heal_shape",
        lambda shape, _options: (
            shape,
            BrepStatus(kind="open_surface", open_shells=1),
            BrepStatus(kind="open_surface", open_shells=1),
            [],
            BrepHealDiagnostics(),
        ),
    )

    with pytest.raises(RuntimeError, match="open shells"):
        _asset_with_brep().heal_brep(BrepHealOptions(fail_on_open_shells=True))


def test_heal_brep_reports_unsupported_sliver_face_removal(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    monkeypatch.setattr(
        heal,
        "heal_shape",
        lambda shape, _options: (
            shape,
            BrepStatus(kind="solid", sliver_faces=1),
            BrepStatus(kind="solid", sliver_faces=1),
            [],
            BrepHealDiagnostics(),
        ),
    )

    healed = _asset_with_brep().heal_brep(BrepHealOptions(remove_sliver_faces=True))

    warning = healed.report.steps[-1].warnings[0]
    assert "sliver face removal is not supported" in warning
    assert "left unchanged" in warning


def test_convert_runs_heal_brep_before_tessellation(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal
    import fascat.ops.tessellate as tessellate
    import fascat.pipeline as pipeline

    calls: list[str] = []
    mesh = Mesh(
        points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
        faces=np.array([[0, 1, 2]], dtype=int),
    )
    asset = _asset_with_brep()
    asset.parts = {"selected": Part(id="selected", name="Selected", source_shape={"shape": "selected"})}
    asset.root.children = [Node(id="selected", name="Selected", part_id="selected")]
    profile = ConversionProfile(
        name="test",
        tessellation=None,
        repair=RepairOptions(),
        stage=StageOptions(uv0="none", uv1=None),
        optimize=None,
        lods=None,
    )

    def fake_heal_shape(
        shape: object, _options: BrepHealOptions
    ) -> tuple[object, BrepStatus, BrepStatus, list[str], BrepHealDiagnostics]:
        calls.append("heal")
        return (shape, BrepStatus(kind="shell"), BrepStatus(kind="solid", solids=1), [], BrepHealDiagnostics())

    def fake_tessellate_asset(written_asset: Asset, _options: object, *, selected_part_ids=None) -> Asset:  # type: ignore[no-untyped-def]
        calls.append("tessellate")
        result = written_asset.copy(keep_source=True)
        result.parts["selected"].mesh = mesh
        return result

    monkeypatch.setattr(pipeline, "read_step", lambda _path: asset)
    monkeypatch.setattr(heal, "heal_shape", fake_heal_shape)
    monkeypatch.setattr(tessellate, "tessellate_asset", fake_tessellate_asset)
    monkeypatch.setattr(pipeline, "_write_gltf", lambda _asset, _path, *, options=None: None)

    converted = convert(
        "input.step",
        tmp_path / "output.glb",
        profile=profile,
        tessellation=__import__("fascat").TessellationOptions(),
        heal_brep=BrepHealOptions(),
        validate_output=False,
    )

    assert calls == ["heal", "tessellate"]
    assert [step.name for step in converted.report.steps if step.name in {"heal_brep", "tessellate"}] == [
        "heal_brep",
        "tessellate",
    ]


def test_cli_convert_accepts_heal_brep_during_dry_run() -> None:
    result = runner.invoke(
        app,
        [
            "--json",
            "--dry-run",
            "convert",
            "input.step",
            "output.glb",
            "--heal-brep",
            "--heal-tolerance",
            "0.1",
            "--overlap-area-ratio",
            "0.9",
            "--remove-sliver-faces",
            "--max-sliver-area",
            "0.001",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"heal_brep": true' in result.output
    assert '"heal_tolerance": 0.1' in result.output
    assert '"group_open_shells": true' in result.output
    assert '"cleanup_overlapping_faces": true' in result.output
    assert '"overlap_area_ratio": 0.9' in result.output


def test_cli_rejects_invalid_heal_tolerance() -> None:
    result = runner.invoke(
        app, ["--dry-run", "convert", "input.step", "output.glb", "--heal-brep", "--heal-tolerance", "0"]
    )

    assert result.exit_code == 2
    assert "--heal-tolerance must be greater than 0" in result.output


def test_cli_rejects_invalid_overlap_area_ratio() -> None:
    result = runner.invoke(
        app, ["--dry-run", "convert", "input.step", "output.glb", "--heal-brep", "--overlap-area-ratio", "1.1"]
    )

    assert result.exit_code == 2
    assert "--overlap-area-ratio must be greater than 0 and no more than 1" in result.output
