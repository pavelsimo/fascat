from __future__ import annotations

from pathlib import Path

import pytest

import fascat as fc
from fascat.io.step import read_step_many
from fascat.io.step import single as step_single
from fascat.io.step.multifile import (
    _multi_file_import_decisions,
    _namespace_metadata_ids,
    _resolve_step_external_reference_graph,
)
from fascat.options import StepReadOptions
from fascat.report import Report


def test_read_step_many_decision_detail_references_supported_external_graph() -> None:
    decisions = _multi_file_import_decisions(StepReadOptions(multi_file=True), 2, 0)
    detail = decisions["multi_file"]["detail"]

    assert decisions["multi_file"]["state"] == "honored"
    assert "read_step(..., multi_file=True)" in detail
    assert "unsupported" not in detail


def test_read_step_many_namespaces_members_and_prefixes_member_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[Path, str, bool]] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        calls.append((source, source_identity, options.multi_file))
        material = fc.Material(
            id="mat",
            name="Paint",
            base_color=(0.8, 0.2, 0.1, 1.0),
            metadata={"source_texture_base_color_image": "img"},
        )
        image = fc.ImageResource(
            id="img",
            name=f"{source.stem}.png",
            mime_type="image/png",
            data=b"png",
            width=1,
            height=1,
        )
        report = Report(source_path=str(source))
        report.input_stats = {
            "nodes": 2,
            "parts": 1,
            "occurrences": 1,
            "materials": 1,
            "images": 1,
            "vertices": 0,
            "triangles": 0,
        }
        report.add_warning(f"{source.stem} member warning")
        report.add_step("import", options={"format": "STEP"}, after=report.input_stats)
        return fc.Asset(
            root=fc.Node(
                id="root",
                name=source.stem,
                children=[fc.Node(id="occurrence", name="occurrence", part_id="part")],
            ),
            parts={"part": fc.Part(id="part", name="Part", material_ids=["mat"])},
            materials={"mat": material},
            images={"img": image},
            metadata={"source": str(source), "source_identity": source_identity},
            report=report,
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    first = tmp_path / "assembly-a.step"
    second = tmp_path / "assembly-b.step"
    asset = read_step_many([first, second], options=StepReadOptions(multi_file=True))
    repeated = read_step_many([first, second], options=StepReadOptions(multi_file=True))

    assert [call[0] for call in calls[:2]] == [first, second]
    assert [call[2] for call in calls[:2]] == [False, False]
    assert asset.source_path is None
    assert asset.report.source_path is None
    assert asset.stats()["parts"] == 2
    assert asset.stats()["occurrences"] == 2
    assert "part" not in asset.parts
    assert "mat" not in asset.materials
    assert "img" not in asset.images
    assert sorted(asset.parts) == sorted(repeated.parts)
    assert sorted(asset.materials) == sorted(repeated.materials)
    assert sorted(asset.images) == sorted(repeated.images)
    for part in asset.parts.values():
        assert part.material_ids
        assert part.material_ids[0] in asset.materials
        assert part.metadata["multi_file_member_part_id"] == "part"
    for material in asset.materials.values():
        image_id = material.metadata["source_texture_base_color_image"]
        assert image_id in asset.images
        assert material.metadata["multi_file_member_material_id"] == "mat"
    assert [child.metadata["multi_file_member_index"] for child in asset.root.children] == [1, 2]
    assert all("member warning" in warning for warning in asset.report.warnings)

    import_step = asset.report.steps[0]
    assert import_step.name == "import"
    assert import_step.options["multi_file"] is True
    assert import_step.options["member_count"] == 2
    assert import_step.options["failed_member_count"] == 0
    assert import_step.options["import_decisions"]["multi_file"]["state"] == "honored"
    assert asset.metadata["multi_file_import"]["member_count"] == 2


def test_namespace_metadata_ids_remaps_values_without_key_mutation() -> None:
    metadata = {
        "source_part_id": "part",
        "source_part_ids": "part|other",
        "source_node_ids": ["node", "other"],
        "source_texture_base_color_image": "img",
        "label": "part",
    }

    result = _namespace_metadata_ids(
        metadata,
        part_ids={"part": "member__part"},
        node_ids={"node": "member__node"},
        image_ids={"img": "member__img"},
    )

    assert set(result) == set(metadata)
    assert result == {
        "source_part_id": "member__part",
        "source_part_ids": "member__part|other",
        "source_node_ids": ["member__node", "other"],
        "source_texture_base_color_image": "member__img",
        "label": "part",
    }
    assert metadata == {
        "source_part_id": "part",
        "source_part_ids": "part|other",
        "source_node_ids": ["node", "other"],
        "source_texture_base_color_image": "img",
        "label": "part",
    }


def test_read_step_many_reuses_identical_member_parts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity, options
        image_id = f"img-{source.stem}"
        material = fc.Material(
            id="mat",
            name="Paint",
            base_color=(0.8, 0.2, 0.1, 1.0),
            metadata={
                "source_texture_base_color_image": image_id,
                "source_texture_base_color_name": f"{source.stem}.png",
                "material_library_path": f"/vendor/{source.stem}/materials.json",
                "material_library_reference": f"{source.stem}/materials.json",
                "material_library_container": f"{source.stem}.zip",
            },
        )
        image = fc.ImageResource(
            id=image_id,
            name=f"{source.stem}.png",
            mime_type="image/png",
            data=b"same-png",
            width=2,
            height=2,
        )
        return fc.Asset(
            root=fc.Node(
                id="root",
                name=source.stem,
                children=[fc.Node(id="occurrence", name="occurrence", part_id="part")],
            ),
            parts={"part": fc.Part(id="part", name="Part", material_ids=["mat"], fingerprint="shared-shape")},
            materials={"mat": material},
            images={image_id: image},
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    first = tmp_path / "assembly-a.step"
    second = tmp_path / "assembly-b.step"
    asset = read_step_many([first, second], options=StepReadOptions(multi_file=True))

    member_part_ids = [child.children[0].part_id for child in asset.root.children]

    assert asset.stats()["parts"] == 1
    assert asset.stats()["occurrences"] == 2
    assert len(set(member_part_ids)) == 1
    assert member_part_ids[0] in asset.parts
    assert asset.report.steps[0].options["members"][0]["deduplicated_parts"] == 0
    assert asset.report.steps[0].options["members"][1]["deduplicated_parts"] == 1
    assert asset.metadata["multi_file_import"]["members"][1]["deduplicated_parts"] == 1


def test_read_step_many_keeps_matching_shape_with_different_materials_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity, options
        color = (0.8, 0.2, 0.1, 1.0) if source.stem.endswith("a") else (0.1, 0.2, 0.8, 1.0)
        return fc.Asset(
            root=fc.Node(
                id="root",
                name=source.stem,
                children=[fc.Node(id="occurrence", name="occurrence", part_id="part")],
            ),
            parts={"part": fc.Part(id="part", name="Part", material_ids=["mat"], fingerprint="shared-shape")},
            materials={"mat": fc.Material(id="mat", name="Paint", base_color=color)},
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    first = tmp_path / "assembly-a.step"
    second = tmp_path / "assembly-b.step"
    asset = read_step_many([first, second], options=StepReadOptions(multi_file=True))

    member_part_ids = [child.children[0].part_id for child in asset.root.children]

    assert asset.stats()["parts"] == 2
    assert len(set(member_part_ids)) == 2
    assert asset.report.steps[0].options["members"][1]["deduplicated_parts"] == 0


def test_read_step_many_can_continue_after_member_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity, options
        if source.name == "bad.step":
            raise RuntimeError("boom")
        report = Report(source_path=str(source))
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem, children=[fc.Node(id="occurrence", name="occurrence")]),
            report=report,
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    asset = read_step_many(
        [tmp_path / "good.step", tmp_path / "bad.step"],
        options=StepReadOptions(),
        continue_on_error=True,
    )

    import_step = asset.report.steps[0]
    assert import_step.options["member_count"] == 1
    assert import_step.options["failed_member_count"] == 1
    assert import_step.options["import_decisions"]["multi_file"]["state"] == "approximated"
    assert "bad.step" in asset.report.warnings[0]
    assert "boom" in asset.report.warnings[0]

    with pytest.raises(fc.FascatIOError, match="boom") as error:
        read_step_many([tmp_path / "good.step", tmp_path / "bad.step"])

    assert isinstance(error.value.__cause__, RuntimeError)


def test_read_step_multi_file_resolves_master_external_reference_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "parts" / "child.stp"
    child.parent.mkdir()
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('parts/child.stp');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")
    calls: list[tuple[Path, bool]] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        calls.append((source, options.multi_file))
        report = Report(source_path=str(source))
        report.add_step(
            "import",
            options={
                "format": "STEP",
                "read_options": options.to_dict(),
                "import_decisions": {"multi_file": {"state": "disabled"}},
            },
        )
        return fc.Asset(
            root=fc.Node(
                id="root",
                name=source.stem,
                children=[fc.Node(id="occurrence", name=source.stem, part_id="part")],
            ),
            parts={"part": fc.Part(id="part", name=source.stem.title())},
            metadata={"source": str(source), "import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert calls == [(master, False), (child.resolve(), False)]
    assert asset.stats()["parts"] == 2
    assert asset.root.metadata["external_reference_graph"] == "true"
    assert asset.metadata["external_reference_graph"]["summary"] == {
        "references": 1,
        "resolved": 1,
        "missing": 0,
        "unsupported": 0,
        "cycles": 0,
        "sources": 2,
        "resolved_sources": 1,
        "member_sources": 2,
        "resolved_occurrences": 1,
    }
    import_options = asset.report.steps[0].options
    assert import_options["external_reference_graph"]["summary"]["resolved_sources"] == 1
    assert import_options["import_decisions"]["multi_file"]["state"] == "honored"
    assert import_options["import_decisions"]["multi_file"]["effective"] is True


def test_read_step_multi_file_reports_missing_master_external_references(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    master = tmp_path / "master.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=EXTERNAL_REFERENCE('missing.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    calls: list[tuple[Path, bool]] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity
        calls.append((source, options.multi_file))
        report = Report(source_path=str(source))
        report.add_step("import", options={"read_options": options.to_dict(), "import_decisions": {}})
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem),
            metadata={"import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert calls == [(master, False)]
    assert asset.metadata["external_reference_graph"]["summary"]["missing"] == 1
    assert asset.metadata["import_decisions"]["multi_file"]["state"] == "missing_sources"
    assert asset.report.steps[0].options["read_options"]["multi_file"] is True
    assert "missing.step" in asset.report.warnings[0]


def test_external_step_reference_graph_reports_reference_cycles(tmp_path: Path) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "child.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('child.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('master.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _resolve_step_external_reference_graph(master)

    cycle = next(record for record in graph.records if record.status == "cycle")
    assert cycle.source == child.resolve()
    assert cycle.resolved == master.resolve()
    assert graph.summary()["cycles"] == 1
    assert graph.summary()["resolved"] == 1
    assert any("cycle detected" in warning for warning in graph.warnings)


def test_external_step_reference_graph_resolves_nested_references_once(tmp_path: Path) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "child.step"
    leaf = tmp_path / "leaf.stp"
    master.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('child.step');\n"
        "#2=DOCUMENT_FILE('missing.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('leaf.stp');\n"
        "#2=DOCUMENT_FILE('master.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    leaf.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")

    graph = _resolve_step_external_reference_graph(master)

    assert graph.sources == [master, child.resolve(), leaf.resolve()]
    assert graph.summary() == {
        "references": 4,
        "resolved": 2,
        "missing": 1,
        "unsupported": 0,
        "cycles": 1,
        "sources": 3,
        "resolved_sources": 2,
        "member_sources": 3,
        "resolved_occurrences": 2,
    }
    assert any("missing.step" in warning for warning in graph.warnings)
    assert any("cycle detected" in warning for warning in graph.warnings)


def test_external_step_reference_graph_detects_cycles_from_alternate_paths(tmp_path: Path) -> None:
    master = tmp_path / "master.step"
    a = tmp_path / "a.step"
    b = tmp_path / "b.step"
    c = tmp_path / "c.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('a.step');\n#2=DOCUMENT_FILE('b.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    a.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('c.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    b.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('c.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    c.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('b.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    graph = _resolve_step_external_reference_graph(master)

    cycle = next(record for record in graph.records if record.status == "cycle")
    assert cycle.source == c.resolve()
    assert cycle.resolved == b.resolve()
    assert graph.sources == [master, a.resolve(), b.resolve(), c.resolve()]
    assert graph.summary()["cycles"] >= 1


def test_external_step_reference_graph_preserves_duplicate_occurrences(tmp_path: Path) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "bolt.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('bolt.step');\n"
        "#2=DOCUMENT_FILE('bolt.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")

    graph = _resolve_step_external_reference_graph(master)

    assert graph.sources == [master, child.resolve()]
    assert graph.member_sources == [master, child.resolve(), child.resolve()]
    assert graph.summary()["resolved_sources"] == 1
    assert graph.summary()["resolved_occurrences"] == 2


def test_external_step_reference_import_preserves_duplicate_occurrences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "bolt.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n"
        "#1=DOCUMENT_FILE('bolt.step');\n"
        "#2=DOCUMENT_FILE('bolt.step');\n"
        "ENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text("ISO-10303-21;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n", encoding="utf-8")
    calls: list[Path] = []

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity
        assert options.multi_file is False
        calls.append(source)
        report = Report(source_path=str(source))
        report.add_step("import", options={"read_options": options.to_dict(), "import_decisions": {}})
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem),
            metadata={"import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert calls == [master, child.resolve(), child.resolve()]
    assert len(asset.root.children) == 3
    assert asset.metadata["external_reference_graph"]["summary"]["resolved_sources"] == 1
    assert asset.metadata["external_reference_graph"]["summary"]["resolved_occurrences"] == 2


def test_read_step_multi_file_reports_external_reference_cycles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    master = tmp_path / "master.step"
    child = tmp_path / "child.step"
    master.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('child.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )
    child.write_text(
        "ISO-10303-21;\nDATA;\n#1=DOCUMENT_FILE('master.step');\nENDSEC;\nEND-ISO-10303-21;\n",
        encoding="utf-8",
    )

    def fake_read_step_path(source: Path, *, source_identity: str, options: StepReadOptions) -> fc.Asset:
        _ = source_identity
        assert options.multi_file is False
        report = Report(source_path=str(source))
        report.add_step("import", options={"read_options": options.to_dict(), "import_decisions": {}})
        return fc.Asset(
            root=fc.Node(id="root", name=source.stem),
            metadata={"import_decisions": {}},
            report=report,
        )

    monkeypatch.setattr(step_single, "_read_step_path", fake_read_step_path)

    asset = fc.read_step(master, options=StepReadOptions(multi_file=True))

    assert asset.metadata["external_reference_graph"]["summary"]["cycles"] == 1
    assert asset.metadata["import_decisions"]["multi_file"]["state"] == "approximated"
    assert any("cycle detected" in warning for warning in asset.report.warnings)
