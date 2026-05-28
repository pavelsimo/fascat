from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from fascat.asset import Asset, Node, Part
from fascat.cli import app
from fascat.filter import Filter
from fascat.mesh import Mesh
from fascat.ops.heal import BrepStatus, brep_status
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

    def fake_heal_shape(shape: object, _options: BrepHealOptions) -> tuple[object, BrepStatus, BrepStatus, list[str]]:
        return (
            {"healed": shape},
            BrepStatus(kind="open_surface", shells=1, faces=3, open_shells=1),
            BrepStatus(kind="solid", solids=1, faces=3, open_shells=0),
            ["fixed trims"],
        )

    monkeypatch.setattr(heal, "heal_shape", fake_heal_shape)

    healed = _asset_with_brep().heal_brep(BrepHealOptions(tolerance=0.1), where=Filter.part("selected"))

    assert healed.parts["selected"].source_shape == {"healed": {"shape": "selected"}}
    assert healed.parts["other"].source_shape == {"shape": "other"}
    assert healed.parts["selected"].metadata["brep_kind"] == "solid"
    assert healed.parts["selected"].metadata["brep_open_shells"] == "0"
    assert healed.parts["selected"].metadata["brep_free_edges"] == "0"
    assert healed.parts["selected"].metadata["brep_small_edges"] == "0"
    assert healed.parts["selected"].metadata["brep_heal_operations"] == "fix_edges,unify_tolerances,sew_faces"
    assert healed.report.warnings == ["Selected: fixed trims"]
    assert healed.report.steps[-1].name == "heal_brep"
    assert healed.report.steps[-1].options["matched"]["parts"] == 1


def test_heal_brep_report_includes_unit_aware_tolerance_policy(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    def fake_heal_shape(shape: object, _options: BrepHealOptions) -> tuple[object, BrepStatus, BrepStatus, list[str]]:
        return (
            shape,
            BrepStatus(kind="surface", faces=1),
            BrepStatus(kind="surface", faces=1),
            [],
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
    assert policy["operations"]["t_junction_sewing"] == "not_implemented"
    assert healed.parts["selected"].metadata["brep_heal_effective_units"] == "millimetre"
    assert healed.parts["selected"].metadata["brep_heal_target_units"] == "metre"
    assert healed.parts["selected"].metadata["brep_heal_heal_tolerance_meters"] == "0.002"
    assert healed.parts["selected"].metadata["brep_heal_max_sliver_area_square_meters"] == "3e-06"


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


def test_heal_brep_reports_remaining_topology_risks(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import fascat.ops.heal as heal

    def fake_heal_shape(shape: object, _options: BrepHealOptions) -> tuple[object, BrepStatus, BrepStatus, list[str]]:
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

    def fake_heal_shape(shape: object, _options: BrepHealOptions) -> tuple[object, BrepStatus, BrepStatus, list[str]]:
        calls.append("heal")
        return (shape, BrepStatus(kind="shell"), BrepStatus(kind="solid", solids=1), [])

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
            "--remove-sliver-faces",
            "--max-sliver-area",
            "0.001",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"heal_brep": true' in result.output
    assert '"heal_tolerance": 0.1' in result.output


def test_cli_rejects_invalid_heal_tolerance() -> None:
    result = runner.invoke(
        app, ["--dry-run", "convert", "input.step", "output.glb", "--heal-brep", "--heal-tolerance", "0"]
    )

    assert result.exit_code == 2
    assert "--heal-tolerance must be greater than 0" in result.output
