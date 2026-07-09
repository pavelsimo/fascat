from __future__ import annotations

import numpy as np
import pytest

from fascat.io.jt.lsg import LateLoadedRef, display_name, parse_lsg
from tests._jt_builder import LsgBuilder, guid_bytes, make_guid


def _two_part_scene(byte_order: str = "<", version: tuple[int, int] = (9, 5)) -> tuple[LsgBuilder, dict[str, int]]:
    builder = LsgBuilder(byte_order=byte_order, version=version)
    material = builder.add_material((0.8, 0.2, 0.1, 1.0), shininess=64.0)
    transform = builder.add_transform(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [10.0, 20.0, 30.0, 1.0],
        ]
    )
    shape_a = builder.add_tristrip_shape_node()
    shape_b = builder.add_tristrip_shape_node(attribute_ids=[material])
    part_a = builder.add_part([shape_a], attribute_ids=[material])
    part_b = builder.add_part([shape_b])
    instance = builder.add_instance(part_b, attribute_ids=[transform])
    assembly = builder.add_group([part_a, part_b, instance])
    root = builder.add_partition([assembly], object_id=builder.new_id())
    builder.set_string_property(part_a, "JT_PROP_NAME", "widget.part;0;1:")
    builder.set_string_property(root, "JT_PROP_MEASUREMENT_UNITS", "Millimeters")
    ids = {
        "material": material,
        "transform": transform,
        "shape_a": shape_a,
        "shape_b": shape_b,
        "part_a": part_a,
        "part_b": part_b,
        "instance": instance,
        "assembly": assembly,
        "root": root,
    }
    return builder, ids


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_parses_two_part_assembly_graph(byte_order: str) -> None:
    builder, ids = _two_part_scene(byte_order)
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order=byte_order)

    root = lsg.nodes[ids["root"]]
    assert root.kind == "partition"
    assert root.child_ids == (ids["assembly"],)
    assembly = lsg.nodes[ids["assembly"]]
    assert assembly.kind == "group"
    assert assembly.child_ids == (ids["part_a"], ids["part_b"], ids["instance"])
    assert lsg.nodes[ids["part_a"]].kind == "part"
    assert lsg.nodes[ids["part_a"]].child_ids == (ids["shape_a"],)
    assert lsg.nodes[ids["instance"]].kind == "instance"
    assert lsg.nodes[ids["instance"]].child_ids == (ids["part_b"],)
    assert lsg.nodes[ids["shape_a"]].kind == "tristrip_shape"
    assert not lsg.skipped_elements


def test_first_node_is_root() -> None:
    builder, ids = _two_part_scene()
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    # Attributes are not nodes; the first node element seen becomes the root candidate.
    assert lsg.root_id == ids["shape_a"]


def test_material_attribute_values() -> None:
    builder, ids = _two_part_scene()
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    material = lsg.materials[ids["material"]]
    assert material.diffuse == pytest.approx((0.8, 0.2, 0.1, 1.0))
    assert material.shininess == 64.0
    assert material.reflectivity == 0.5
    assert lsg.nodes[ids["part_a"]].attribute_ids == (ids["material"],)


def test_transform_attribute_expands_stored_mask() -> None:
    builder, ids = _two_part_scene()
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    matrix = lsg.transforms[ids["transform"]].matrix
    expected = np.identity(4)
    expected[3, :3] = [10.0, 20.0, 30.0]
    assert np.allclose(matrix, expected)


def test_dense_transform_round_trips() -> None:
    builder = LsgBuilder()
    rotation = [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [5.0, -3.0, 0.25, 1.0],
    ]
    transform = builder.add_transform(rotation)
    builder.add_partition([])
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    assert np.allclose(lsg.transforms[transform].matrix, np.array(rotation))


def test_properties_resolved_from_table() -> None:
    builder, ids = _two_part_scene()
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    assert lsg.properties[ids["part_a"]]["JT_PROP_NAME"] == "widget.part;0;1:"
    assert lsg.properties[ids["root"]]["JT_PROP_MEASUREMENT_UNITS"] == "Millimeters"


def test_integer_and_float_atoms() -> None:
    builder = LsgBuilder()
    part = builder.add_part([])
    builder.add_partition([part])
    builder.set_properties(
        part,
        [
            (builder.add_string_atom("COUNT"), builder.add_integer_atom(42)),
            (builder.add_string_atom("MASS"), builder.add_float_atom(1.5)),
        ],
    )
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    assert lsg.properties[part]["COUNT"] == 42
    assert lsg.properties[part]["MASS"] == 1.5


def test_late_loaded_refs_collected_separately() -> None:
    builder = LsgBuilder()
    shape = builder.add_tristrip_shape_node()
    builder.add_partition([shape])
    segment_guid = make_guid(77)
    builder.attach_late_loaded(shape, "JT_LLPROP_SHAPEDATA", segment_guid, 7)
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    refs = lsg.late_loaded[shape]
    assert len(refs) == 1
    assert isinstance(refs[0], LateLoadedRef)
    assert refs[0].segment_id == guid_bytes(segment_guid)
    assert refs[0].segment_type == 7
    assert "JT_LLPROP_SHAPEDATA" not in lsg.properties.get(shape, {})


def test_unknown_elements_skipped_and_counted() -> None:
    builder = LsgBuilder()
    part = builder.add_part([])
    builder.add_unknown_element(seed=0xABC)
    builder.add_unknown_element(seed=0xABC)
    builder.add_unknown_element(seed=0xDEF)
    builder.add_partition([part])
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    assert lsg.nodes[part].kind == "part"
    assert sum(lsg.skipped_elements.values()) == 3
    assert len(lsg.skipped_elements) == 2


def test_lod_and_range_lod_children() -> None:
    builder = LsgBuilder()
    fine = builder.add_tristrip_shape_node()
    coarse = builder.add_tristrip_shape_node()
    range_lod = builder.add_range_lod([fine, coarse], range_limits=[100.0, 500.0])
    lod = builder.add_lod([fine, coarse])
    builder.add_partition([range_lod, lod])
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    assert lsg.nodes[range_lod].kind == "range_lod"
    assert lsg.nodes[range_lod].child_ids == (fine, coarse)
    assert lsg.nodes[range_lod].range_limits == (100.0, 500.0)
    assert lsg.nodes[lod].kind == "lod"
    assert lsg.nodes[lod].child_ids == (fine, coarse)


def test_partition_external_file_reference() -> None:
    builder = LsgBuilder()
    external = builder.add_partition([], file_name="wheels/wheel.jt")
    builder.add_partition([external])
    lsg = parse_lsg(builder.payload(), version=(9, 5), byte_order="<")
    assert lsg.nodes[external].file_name == "wheels/wheel.jt"


def test_jt10_u8_version_fields() -> None:
    builder, ids = _two_part_scene(version=(10, 0))
    lsg = parse_lsg(builder.payload(), version=(10, 0), byte_order="<")
    assert lsg.nodes[ids["root"]].kind == "partition"
    assert lsg.properties[ids["part_a"]]["JT_PROP_NAME"] == "widget.part;0;1:"


def test_element_overrun_raises() -> None:
    builder = LsgBuilder()
    builder.add_part([])
    builder.add_partition([])
    payload = bytearray(builder.payload())
    payload[0] -= 8  # shrink the first element's declared length below what its parser reads
    with pytest.raises(RuntimeError, match="overran its length"):
        parse_lsg(bytes(payload), version=(9, 5), byte_order="<")


def test_display_name_strips_encoding() -> None:
    assert display_name("AlignmentPin.part;0;1:") == "AlignmentPin.part"
    assert display_name("Chrome material") == "Chrome material"
