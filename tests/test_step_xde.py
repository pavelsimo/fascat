from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

import fascat as fc
from fascat.io._import_base import (
    _annotate_mirrored_transforms,
    _cleanup_action,
    _ImportCleanupStats,
    _loaded_representation,
    _loaded_representation_report,
    _mirrored_transform_warnings,
    _ShapeTopologyCounts,
    _space_normalization,
    _StepHeaderInfo,
)
from fascat.io.step.materials import (
    _color_material_spec,
    _material_id_from_spec,
)
from fascat.io.step.single import (
    _import_decisions,
)
from fascat.io.step.xde import (
    _build_mixed_construction_curve_node,
    _canonical_part_id,
    _face_material_ids,
    _mixed_construction_curve_metadata,
    _mixed_construction_curve_shape,
    _shape_fingerprint,
    _shape_topology_counts,
)
from fascat.options import StepReadOptions


def test_canonical_part_id_reuses_matching_shape_and_material() -> None:
    part_index: dict[tuple[str, str, str, str], str] = {}

    first_id, first_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:1",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )
    second_id, second_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:2",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )
    different_material_id, different_material_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:3",
        shape_hash="shape-a",
        material_signature="mat-blue",
        part_index=part_index,
    )

    assert first_is_new is True
    assert second_is_new is False
    assert first_id == second_id
    assert different_material_is_new is True
    assert different_material_id != first_id


def test_canonical_part_id_prefers_source_label_before_shape_hash() -> None:
    part_index: dict[tuple[str, str, str, str], str] = {}

    first_id, first_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:1",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )
    repeated_label_id, repeated_label_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:1",
        shape_hash="unstable-shape-hash",
        material_signature="mat-red",
        part_index=part_index,
    )
    repeated_shape_id, repeated_shape_is_new = _canonical_part_id(
        source_identity="model.step",
        part_entry="0:1:2",
        shape_hash="shape-a",
        material_signature="mat-red",
        part_index=part_index,
    )

    assert first_is_new is True
    assert repeated_label_is_new is False
    assert repeated_shape_is_new is False
    assert repeated_label_id == first_id
    assert repeated_shape_id == first_id


class _FakeStepQuantityColor:
    created_count: ClassVar[int] = 0

    def __init__(self) -> None:
        type(self).created_count += 1
        self._red = 0.0
        self._green = 0.0
        self._blue = 0.0
        self._alpha = 1.0

    def set_rgba(self, rgba: tuple[float, float, float, float]) -> None:
        self._red, self._green, self._blue, self._alpha = rgba

    def Red(self) -> float:
        return self._red

    def Green(self) -> float:
        return self._green

    def Blue(self) -> float:
        return self._blue

    def Alpha(self) -> float:
        return self._alpha

    def GetRGB(self) -> _FakeStepQuantityColor:
        return self


class _FakeStepShape:
    def __init__(self, name: str, faces: list[_FakeStepShape] | None = None) -> None:
        self.name = name
        self.faces = faces or []


class _FakeStepLabel:
    def __init__(
        self,
        name: str = "",
        colors: dict[str, tuple[float, float, float, float]] | None = None,
    ) -> None:
        self.name = name
        self.colors = colors or {}

    def copy_from(self, other: _FakeStepLabel) -> None:
        self.name = other.name
        self.colors = dict(other.colors)


class _FakeStepVisualMaterial:
    def __init__(self, name: str, rgba: tuple[float, float, float, float]) -> None:
        self._name = name
        self._color = _FakeStepQuantityColor()
        self._color.set_rgba(rgba)

    def IsEmpty(self) -> bool:
        return False

    def RawName(self) -> str:
        return self._name

    def HasPbrMaterial(self) -> bool:
        return False

    def HasCommonMaterial(self) -> bool:
        return False

    def BaseColor(self) -> _FakeStepQuantityColor:
        return self._color


class _FakeStepVisualMaterialTool:
    def __init__(
        self,
        *,
        shape_materials: dict[str, _FakeStepVisualMaterial] | None = None,
        label_materials: dict[str, _FakeStepVisualMaterial] | None = None,
    ) -> None:
        self.shape_materials = shape_materials or {}
        self.label_materials = label_materials or {}
        self.shape_calls: list[str] = []
        self.label_calls: list[str] = []

    def IsSetShapeMaterial(self, target: _FakeStepShape | _FakeStepLabel) -> bool:
        if isinstance(target, _FakeStepLabel):
            self.label_calls.append(target.name)
            return target.name in self.label_materials
        self.shape_calls.append(target.name)
        return target.name in self.shape_materials

    def GetShapeMaterial(self, shape: _FakeStepShape) -> _FakeStepVisualMaterial:
        return self.shape_materials[shape.name]

    def GetShapeMaterial_s(self, label: _FakeStepLabel) -> _FakeStepVisualMaterial:
        return self.label_materials[label.name]


class _FakeStepShapeTool:
    def __init__(
        self,
        sub_labels: dict[str, _FakeStepLabel] | None = None,
        *,
        fail_on_find: bool = False,
    ) -> None:
        self.sub_labels = sub_labels or {}
        self.fail_on_find = fail_on_find
        self.find_calls: list[tuple[str, str]] = []

    def FindSubShape(self, shape_label: _FakeStepLabel, face: _FakeStepShape, sub_label: _FakeStepLabel) -> bool:
        if self.fail_on_find:
            raise AssertionError("FindSubShape should not be called")
        self.find_calls.append((shape_label.name, face.name))
        label = self.sub_labels.get(face.name)
        if label is None:
            return False
        sub_label.copy_from(label)
        return True


class _FakeStepColorTool:
    def __init__(
        self,
        shape_colors: dict[str, tuple[float, float, float, float]] | None = None,
        *,
        fail_on_lookup: bool = False,
    ) -> None:
        self.shape_colors = shape_colors or {}
        self.fail_on_lookup = fail_on_lookup
        self.calls: list[tuple[str, str, int]] = []

    def GetColor(self, shape: _FakeStepShape, color_type: str, color: _FakeStepQuantityColor) -> bool:
        return self._set_shape_color("shape", shape, color_type, color)

    def GetInstanceColor(self, shape: _FakeStepShape, color_type: str, color: _FakeStepQuantityColor) -> bool:
        return self._set_shape_color("instance", shape, color_type, color)

    def _set_shape_color(
        self,
        source: str,
        shape: _FakeStepShape,
        color_type: str,
        color: _FakeStepQuantityColor,
    ) -> bool:
        if self.fail_on_lookup:
            raise AssertionError("shape color should not be called")
        self.calls.append((source, color_type, id(color)))
        if source != "shape":
            return False
        rgba = self.shape_colors.get(shape.name)
        if rgba is None:
            return False
        color.set_rgba(rgba)
        return True


class _FakeStepColorToolStatic:
    label_calls: ClassVar[list[tuple[str, str, int]]] = []

    @staticmethod
    def GetColor_s(label: _FakeStepLabel, color_type: str, color: _FakeStepQuantityColor) -> bool:
        _FakeStepColorToolStatic.label_calls.append((label.name, color_type, id(color)))
        rgba = label.colors.get(color_type)
        if rgba is None:
            return False
        color.set_rgba(rgba)
        return True


class _FakeStepExplorer:
    def __init__(self, shape: _FakeStepShape, shape_type: str) -> None:
        assert shape_type == "face"
        self._faces = shape.faces
        self._index = 0

    def More(self) -> bool:
        return self._index < len(self._faces)

    def Current(self) -> _FakeStepShape:
        return self._faces[self._index]

    def Next(self) -> None:
        self._index += 1


class _FakeStepTopoDS:
    @staticmethod
    def Face_s(shape: _FakeStepShape) -> _FakeStepShape:
        return shape


def _install_fake_step_face_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeStepQuantityColor.created_count = 0
    _FakeStepColorToolStatic.label_calls = []

    fake_ocp = types.ModuleType("OCP")
    fake_ocp.__path__ = []  # type: ignore[attr-defined]
    fake_quantity = types.ModuleType("OCP.Quantity")
    fake_quantity.Quantity_Color = _FakeStepQuantityColor
    fake_tdf = types.ModuleType("OCP.TDF")
    fake_tdf.TDF_Label = _FakeStepLabel
    fake_top_abs = types.ModuleType("OCP.TopAbs")
    fake_top_abs.TopAbs_FACE = "face"
    fake_top_exp = types.ModuleType("OCP.TopExp")
    fake_top_exp.TopExp_Explorer = _FakeStepExplorer
    fake_topods = types.ModuleType("OCP.TopoDS")
    fake_topods.TopoDS = _FakeStepTopoDS
    fake_xcafdoc = types.ModuleType("OCP.XCAFDoc")
    fake_xcafdoc.XCAFDoc_ColorSurf = "surf"
    fake_xcafdoc.XCAFDoc_ColorGen = "gen"
    fake_xcafdoc.XCAFDoc_ColorTool = _FakeStepColorToolStatic

    for name, module in {
        "OCP": fake_ocp,
        "OCP.Quantity": fake_quantity,
        "OCP.TDF": fake_tdf,
        "OCP.TopAbs": fake_top_abs,
        "OCP.TopExp": fake_top_exp,
        "OCP.TopoDS": fake_topods,
        "OCP.XCAFDoc": fake_xcafdoc,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_face_material_ids_direct_shape_material_skips_subshape_and_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_step_face_modules(monkeypatch)
    face = _FakeStepShape("face-a")
    shape = _FakeStepShape("solid", faces=[face])
    visual_tool = _FakeStepVisualMaterialTool(
        shape_materials={"face-a": _FakeStepVisualMaterial("Direct Face Paint", (0.2, 0.4, 0.6, 0.8))}
    )
    shape_tool = _FakeStepShapeTool(fail_on_find=True)
    color_tool = _FakeStepColorTool(fail_on_lookup=True)
    _FakeStepQuantityColor.created_count = 0

    material_ids, specs = _face_material_ids(
        shape_tool,
        color_tool,
        visual_tool,
        _FakeStepLabel("solid-label"),
        shape,
        base_material_id="base-mat",
        options=StepReadOptions(),
    )

    spec = next(iter(specs.values()))
    assert material_ids == [_material_id_from_spec(spec)]
    assert spec.name == "Direct Face Paint"
    assert color_tool.calls == []
    assert _FakeStepColorToolStatic.label_calls == []
    assert _FakeStepQuantityColor.created_count == 0


def test_face_material_ids_label_visual_material_wins_over_shape_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_step_face_modules(monkeypatch)
    face = _FakeStepShape("face-a")
    shape = _FakeStepShape("solid", faces=[face])
    sub_label = _FakeStepLabel("face-label")
    visual_tool = _FakeStepVisualMaterialTool(
        label_materials={"face-label": _FakeStepVisualMaterial("Label Face Paint", (0.1, 0.2, 0.3, 1.0))}
    )
    shape_tool = _FakeStepShapeTool({"face-a": sub_label})
    color_tool = _FakeStepColorTool({"face-a": (0.9, 0.0, 0.0, 1.0)}, fail_on_lookup=True)
    shape_label = _FakeStepLabel("solid-label")
    _FakeStepQuantityColor.created_count = 0

    material_ids, specs = _face_material_ids(
        shape_tool,
        color_tool,
        visual_tool,
        shape_label,
        shape,
        base_material_id="base-mat",
        options=StepReadOptions(),
    )

    spec = next(iter(specs.values()))
    assert material_ids == [_material_id_from_spec(spec)]
    assert spec.name == "Label Face Paint"
    assert shape_tool.find_calls == [("solid-label", "face-a")]
    assert color_tool.calls == []
    assert _FakeStepQuantityColor.created_count == 0


def test_face_material_ids_shape_and_label_color_fallbacks_are_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_step_face_modules(monkeypatch)
    shape_color_face = _FakeStepShape("shape-color-face")
    label_color_face = _FakeStepShape("label-color-face")
    shape = _FakeStepShape("solid", faces=[shape_color_face, label_color_face])
    label_color = (0.0, 0.1, 0.8, 1.0)
    shape_color = (0.8, 0.1, 0.0, 1.0)
    shape_tool = _FakeStepShapeTool({"label-color-face": _FakeStepLabel("label-color", colors={"surf": label_color})})
    color_tool = _FakeStepColorTool({"shape-color-face": shape_color})
    visual_tool = _FakeStepVisualMaterialTool()
    expected_ids = [
        _material_id_from_spec(_color_material_spec(shape_color)),
        _material_id_from_spec(_color_material_spec(label_color)),
    ]

    first_ids, first_specs = _face_material_ids(
        shape_tool,
        color_tool,
        visual_tool,
        _FakeStepLabel("solid-label"),
        shape,
        base_material_id="base-mat",
        options=StepReadOptions(),
    )
    second_ids, second_specs = _face_material_ids(
        shape_tool,
        color_tool,
        visual_tool,
        _FakeStepLabel("solid-label"),
        shape,
        base_material_id="base-mat",
        options=StepReadOptions(),
    )

    assert first_ids == second_ids == expected_ids
    assert first_specs[expected_ids[0]].base_color == shape_color
    assert first_specs[expected_ids[1]].base_color == label_color
    assert first_specs[expected_ids[0]].metadata_dict()["cad_material_source"] == "color"
    assert second_specs[expected_ids[1]].base_color == label_color
    assert _FakeStepQuantityColor.created_count == 2
    assert {call[2] for call in _FakeStepColorToolStatic.label_calls[:1]} == {call[2] for call in color_tool.calls[:3]}


def test_shape_fingerprint_falls_back_to_python_hash() -> None:
    class ShapeWithoutHashCode:
        def __hash__(self) -> int:
            return 123

    assert _shape_fingerprint(ShapeWithoutHashCode()) == "123"


def test_step_import_cleanup_actions_cover_construction_only_shapes() -> None:
    point_counts = _ShapeTopologyCounts(vertices=3)
    line_counts = _ShapeTopologyCounts(vertices=4, edges=2)
    brep_counts = _ShapeTopologyCounts(vertices=8, edges=12, faces=6)

    assert _loaded_representation(point_counts) == "construction_points"
    assert _loaded_representation(line_counts) == "construction_lines"
    assert _loaded_representation(brep_counts) == "brep"
    assert _cleanup_action(point_counts, StepReadOptions(delete_free_vertices=True)) == "delete_free_vertices"
    assert _cleanup_action(line_counts, StepReadOptions(delete_lines=True)) == "delete_lines"
    assert _cleanup_action(line_counts, StepReadOptions(construction_curve_policy="delete")) == "delete_lines"
    assert _cleanup_action(line_counts, StepReadOptions(construction_curve_policy="tessellate_tubes")) is None
    assert _cleanup_action(brep_counts, StepReadOptions(delete_free_vertices=True, delete_lines=True)) is None


def test_mixed_construction_curve_shape_extracts_edges_not_used_by_faces() -> None:
    pytest.importorskip("OCP")
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt
    from OCP.TopoDS import TopoDS_Compound

    box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    construction_edge = BRepBuilderAPI_MakeEdge(gp_Pnt(2.0, 0.0, 0.0), gp_Pnt(3.0, 0.0, 0.0)).Edge()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, box)
    builder.Add(compound, construction_edge)

    curve_shape = _mixed_construction_curve_shape(compound, _shape_topology_counts(compound))

    assert curve_shape is not None
    assert _shape_topology_counts(curve_shape) == _ShapeTopologyCounts(vertices=2, edges=1, faces=0)
    assert _mixed_construction_curve_shape(box, _shape_topology_counts(box)) is None


def test_mixed_construction_curve_shape_uses_occt_shape_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEdge:
        def __init__(self, key: str) -> None:
            self.key = key

        def IsSame(self, other: object) -> bool:
            raise AssertionError("free-edge detection should use shape maps, not IsSame scans")

    class FakeShape:
        def __init__(self, *, faces: list[FakeShape] | None = None, edges: list[FakeEdge] | None = None) -> None:
            self.faces = faces or []
            self.edges = edges or []

    class FakeIndexedMapOfShape:
        def __init__(self) -> None:
            self._keys: set[str] = set()

        def Add(self, shape: FakeEdge) -> None:
            self._keys.add(shape.key)

        def Contains(self, shape: FakeEdge) -> bool:
            return shape.key in self._keys

        def IsEmpty(self) -> bool:
            return not self._keys

    class FakeExplorer:
        def __init__(self, shape: FakeShape, shape_type: str) -> None:
            self._items = shape.faces if shape_type == "face" else shape.edges
            self._index = 0

        def More(self) -> bool:
            return self._index < len(self._items)

        def Current(self) -> FakeShape | FakeEdge:
            return self._items[self._index]

        def Next(self) -> None:
            self._index += 1

    class FakeTopExp:
        @staticmethod
        def MapShapes_s(shape: FakeShape, shape_type: str, shape_map: FakeIndexedMapOfShape) -> None:
            assert shape_type == "edge"
            for edge in shape.edges:
                shape_map.Add(edge)

    class FakeTopoDS:
        @staticmethod
        def Edge_s(shape: FakeEdge) -> FakeEdge:
            return shape

    class FakeCompound:
        def __init__(self) -> None:
            self.edges: list[FakeEdge] = []

    class FakeBuilder:
        def MakeCompound(self, compound: FakeCompound) -> None:
            compound.edges.clear()

        def Add(self, compound: FakeCompound, edge: FakeEdge) -> None:
            compound.edges.append(edge)

    face_edge = FakeEdge("face")
    free_edge = FakeEdge("free")
    duplicate_free_edge = FakeEdge("free")
    face = FakeShape(edges=[face_edge, FakeEdge("face")])
    shape = FakeShape(faces=[face], edges=[face_edge, free_edge, duplicate_free_edge])

    fake_ocp = types.ModuleType("OCP")
    fake_ocp.__path__ = []  # type: ignore[attr-defined]
    fake_brep = types.ModuleType("OCP.BRep")
    fake_brep.BRep_Builder = FakeBuilder
    fake_top_abs = types.ModuleType("OCP.TopAbs")
    fake_top_abs.TopAbs_EDGE = "edge"
    fake_top_abs.TopAbs_FACE = "face"
    fake_top_exp = types.ModuleType("OCP.TopExp")
    fake_top_exp.TopExp = FakeTopExp
    fake_top_exp.TopExp_Explorer = FakeExplorer
    fake_top_tools = types.ModuleType("OCP.TopTools")
    fake_top_tools.TopTools_IndexedMapOfShape = FakeIndexedMapOfShape
    fake_topods = types.ModuleType("OCP.TopoDS")
    fake_topods.TopoDS = FakeTopoDS
    fake_topods.TopoDS_Compound = FakeCompound

    for name, module in {
        "OCP": fake_ocp,
        "OCP.BRep": fake_brep,
        "OCP.TopAbs": fake_top_abs,
        "OCP.TopExp": fake_top_exp,
        "OCP.TopTools": fake_top_tools,
        "OCP.TopoDS": fake_topods,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    curve_shape = _mixed_construction_curve_shape(shape, _ShapeTopologyCounts(vertices=4, edges=3, faces=1))

    assert curve_shape is not None
    assert [edge.key for edge in curve_shape.edges] == ["free"]


def test_mixed_construction_curve_metadata_records_policy_and_action() -> None:
    metadata = _mixed_construction_curve_metadata(
        StepReadOptions(construction_curve_policy="tessellate_tubes"),
        "split",
        _ShapeTopologyCounts(vertices=2, edges=1),
    )

    assert metadata == {
        "mixed_construction_curve_policy": "tessellate_tubes",
        "mixed_construction_curve_action": "split",
        "mixed_construction_curve_vertices": "2",
        "mixed_construction_curve_edges": "1",
        "mixed_construction_curve_split": "true",
    }


def test_mixed_construction_curve_node_preserves_policy_metadata() -> None:
    pytest.importorskip("OCP")
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
    from OCP.gp import gp_Pnt

    curve_shape = BRepBuilderAPI_MakeEdge(gp_Pnt(0.0, 0.0, 0.0), gp_Pnt(1.0, 0.0, 0.0)).Edge()
    counts = _shape_topology_counts(curve_shape)
    parts: dict[str, fc.Part] = {}
    cleanup = _ImportCleanupStats()
    options = StepReadOptions(construction_curve_policy="tessellate_tubes", construction_curve_tube_radius=0.025)

    node = _build_mixed_construction_curve_node(
        source_identity="panel.step",
        occurrence_path="root/1",
        label_entry="0:1",
        part_entry="0:1:1",
        source_name="Panel",
        shape=curve_shape,
        counts=counts,
        material_ids=["mat-default"],
        part_index={},
        parts=parts,
        options=options,
        cleanup=cleanup,
    )

    assert cleanup.to_dict()["construction_line_parts"] == 1
    assert node.part_id in parts
    assert node.metadata["mixed_construction_curve_split"] == "true"
    part = parts[str(node.part_id)]
    assert part.name == "Panel Construction Curves"
    assert part.metadata["loaded_representation"] == "construction_lines"
    assert part.metadata["mixed_construction_curve_split"] == "true"
    assert part.metadata["construction_curve_policy"] == "tessellate_tubes"
    assert part.metadata["construction_curve_tube_radius"] == "0.025"


def test_loaded_representation_report_lists_parts_and_deleted_nodes() -> None:
    asset = fc.Asset(
        root=fc.Node(
            id="root",
            name="root",
            children=[
                fc.Node(id="node-a", name="Part A", part_id="part-a"),
                fc.Node(
                    id="node-deleted",
                    name="construction line",
                    metadata={
                        "loaded_representation": "construction_lines",
                        "import_cleanup": "delete_lines",
                        "source_vertices": "2",
                        "source_edges": "1",
                        "source_faces": "0",
                    },
                ),
            ],
        ),
        parts={
            "part-a": fc.Part(
                id="part-a",
                name="Part A",
                metadata={
                    "loaded_representation": "brep",
                    "source_vertices": "8",
                    "source_edges": "12",
                    "source_faces": "6",
                    "source_name": "Source Part A",
                },
            )
        },
    )

    report = _loaded_representation_report(asset)

    assert report["summary"] == {
        "brep_parts": 1,
        "construction_point_parts": 0,
        "construction_line_parts": 0,
        "empty_shape_parts": 0,
        "unknown_parts": 0,
        "deleted_nodes": 1,
        "deleted_free_vertex_nodes": 0,
        "deleted_line_nodes": 1,
    }
    assert report["parts"] == [
        {
            "part_id": "part-a",
            "name": "Part A",
            "loaded_representation": "brep",
            "cleanup_action": "preserved",
            "source_vertices": 8,
            "source_edges": 12,
            "source_faces": 6,
            "source_name": "Source Part A",
        }
    ]
    assert report["deleted_nodes"] == [
        {
            "node_id": "node-deleted",
            "name": "construction line",
            "loaded_representation": "construction_lines",
            "cleanup_action": "delete_lines",
            "source_vertices": 2,
            "source_edges": 1,
            "source_faces": 0,
        }
    ]


def test_step_space_normalization_builds_reported_root_transform() -> None:
    space = _space_normalization(
        "millimetre",
        0.001,
        StepReadOptions(target_units="metre", target_up_axis="Y", target_handedness="right"),
    )

    assert space.source_units == "millimetre"
    assert space.target_units == "metre"
    assert space.source_up_axis == "Z"
    assert space.target_up_axis == "Y"
    assert space.changed is True
    assert np.allclose(
        space.transform,
        np.array(
            [
                [0.001, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.001, 0.0],
                [0.0, -0.001, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    assert space.metadata()["changed"] is True
    assert space.metadata()["mirrored"] is False


def test_step_space_normalization_reports_handedness_mirror() -> None:
    space = _space_normalization("millimetre", 0.001, StepReadOptions(target_handedness="left"))

    assert space.determinant < 0.0
    assert space.mirrored is True
    assert space.metadata()["mirrored"] is True


def test_step_mirrored_transform_summary_annotates_negative_determinants() -> None:
    mirrored = np.eye(4, dtype=np.float64)
    mirrored[0, 0] = -1.0
    root = fc.Node(
        id="root",
        name="Root",
        children=[fc.Node(id="occurrence", name="Occurrence", part_id="part", transform=mirrored)],
    )

    summary = _annotate_mirrored_transforms(root)
    child = root.children[0]

    assert summary == {"local_mirrored_nodes": 1, "world_mirrored_nodes": 1, "mirrored_part_occurrences": 1}
    assert child.metadata["local_transform_mirrored"] == "true"
    assert child.metadata["world_transform_mirrored"] == "true"
    assert child.metadata["local_transform_determinant"] == pytest.approx(-1.0)
    assert child.metadata["world_transform_determinant"] == pytest.approx(-1.0)


def test_step_mirrored_transform_decision_and_warning_report_risk() -> None:
    summary = {"local_mirrored_nodes": 1, "world_mirrored_nodes": 1, "mirrored_part_occurrences": 1}
    decisions = _import_decisions(
        StepReadOptions(),
        _StepHeaderInfo(schema=None, pmi_present=False),
        pmi_count=0,
        unsupported_pmi_count=0,
        cleanup=_ImportCleanupStats(),
        space=_space_normalization("millimetre", 0.001, StepReadOptions()),
        mirrored_transform_summary=summary,
    )
    warnings = _mirrored_transform_warnings(summary)

    assert decisions["mirrored_transforms"]["state"] == "detected"
    assert decisions["mirrored_transforms"]["effective"] is True
    assert decisions["mirrored_transforms"]["counts"] == summary
    assert "negative determinants" in warnings[0]
    assert "normal/winding compensation" in warnings[0]


@pytest.mark.requires_ocp
def test_step_shape_fingerprints_are_stable_across_imports() -> None:
    fixture = Path("tests/fixtures/spool-clamp-lid.step")

    first = fc.read_step(fixture)
    second = fc.read_step(fixture)

    assert [part.metadata["shape_fingerprint"] for part in first.parts.values()] == [
        part.metadata["shape_fingerprint"] for part in second.parts.values()
    ]
