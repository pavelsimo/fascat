from __future__ import annotations

import math
import struct
from io import StringIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fascat.asset import Asset
from fascat.io._atomic import atomic_output
from fascat.io._errors import wrap_io_errors
from fascat.io._geometry import normalize_rows, transform_points
from fascat.io._suffixes import STL_SUFFIXES
from fascat.options import StlExportOptions

ASCII_STL_TRIANGLE_WARNING_THRESHOLD = 10_000
FloatArray = NDArray[np.float64]


@wrap_io_errors("write STL")
def write_stl(asset: Asset, path: str | Path, *, options: StlExportOptions | None = None) -> None:
    _write_stl(asset, path, options=options, validate=False)


@wrap_io_errors("write STL with validation")
def write_stl_with_validation_stats(
    asset: Asset,
    path: str | Path,
    *,
    options: StlExportOptions | None = None,
) -> dict[str, int] | None:
    return _write_stl(asset, path, options=options, validate=True)


def _write_stl(
    asset: Asset,
    path: str | Path,
    *,
    options: StlExportOptions | None,
    validate: bool,
) -> dict[str, int] | None:
    opts = options or StlExportOptions()
    output_path = Path(path)
    if output_path.suffix.lower() not in STL_SUFFIXES:
        raise ValueError(f"unsupported STL extension: {output_path.suffix or '<none>'}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    triangles = _triangles(asset)
    if not opts.binary and triangles.shape[0] > ASCII_STL_TRIANGLE_WARNING_THRESHOLD:
        asset.report.add_warning(
            "ASCII STL export selected for "
            f"{triangles.shape[0]:,} triangles; binary STL is recommended above "
            f"{ASCII_STL_TRIANGLE_WARNING_THRESHOLD:,} triangles"
        )
    with atomic_output(output_path) as temp:
        if opts.binary:
            temp.write_bytes(_binary_stl(triangles))
        else:
            temp.write_text(_ascii_stl(triangles), encoding="utf-8")
        stats = validate_stl(temp) if validate else None
    return stats


def validate_stl(path: str | Path) -> dict[str, int]:
    payload = Path(path).read_bytes()
    if len(payload) >= 84:
        triangle_count = struct.unpack_from("<I", payload, 80)[0]
        expected = 84 + triangle_count * 50
        if expected == len(payload):
            if triangle_count == 0:
                raise RuntimeError("STL asset contains no triangles")
            # Each packed record contains a normal, three vertices, and a uint16 attribute.
            coordinates = np.ndarray((triangle_count, 12), dtype="<f4", buffer=payload, offset=84, strides=(50, 4))
            if not np.isfinite(coordinates).all():
                raise RuntimeError("STL asset contains non-finite coordinates or normals")
            return {"meshes": 1, "points": triangle_count * 3, "triangles": triangle_count}
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise RuntimeError("invalid STL binary length or ASCII encoding") from exc
    triangle_count = _validate_ascii_stl(text)
    if triangle_count == 0:
        raise RuntimeError("STL asset contains no triangles")
    return {"meshes": 1, "points": triangle_count * 3, "triangles": triangle_count}


def _validate_ascii_stl(text: str) -> int:
    state = "solid"
    vertices = 0
    triangles = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        label = f"STL line {line_number}"
        if state == "solid" and fields[0] == "solid":
            state = "facet"
        elif state == "facet" and fields[0] == "endsolid":
            state = "solid"
        elif state == "facet" and fields[:2] == ["facet", "normal"]:
            _validate_stl_vector(fields[2:], label)
            state = "loop"
        elif state == "loop" and fields == ["outer", "loop"]:
            vertices = 0
            state = "vertex"
        elif state == "vertex" and fields[0] == "vertex":
            _validate_stl_vector(fields[1:], label)
            vertices += 1
            if vertices == 3:
                state = "endloop"
        elif state == "endloop" and fields == ["endloop"]:
            state = "endfacet"
        elif state == "endfacet" and fields == ["endfacet"]:
            triangles += 1
            state = "facet"
        else:
            raise RuntimeError(f"{label} has invalid facet structure (expected {state})")
    if state != "solid":
        raise RuntimeError("STL contains an incomplete solid or facet")
    return triangles


def _validate_stl_vector(fields: list[str], label: str) -> None:
    if len(fields) != 3:
        raise RuntimeError(f"{label} requires three coordinates")
    try:
        finite = all(math.isfinite(float(value)) for value in fields)
    except ValueError as exc:
        raise RuntimeError(f"{label} contains invalid coordinates") from exc
    if not finite:
        raise RuntimeError(f"{label} contains non-finite coordinates")


def _triangles(asset: Asset) -> FloatArray:
    chunks: list[FloatArray] = []
    for node, current in asset.root.walk_world(np.eye(4, dtype=np.float64)):
        if node.part_id is not None and node.part_id in asset.parts:
            mesh = asset.parts[node.part_id].mesh
            if mesh is not None:
                points = transform_points(mesh.points, current)
                chunks.append(points[mesh.faces].astype(np.float64))
    if not chunks:
        return np.empty((0, 3, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def _binary_stl(triangles: FloatArray) -> bytes:
    data = bytearray(b"Generated by fascat".ljust(80, b"\x00"))
    data.extend(struct.pack("<I", triangles.shape[0]))
    records = np.zeros(
        triangles.shape[0],
        dtype=[("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")],
    )
    records["normal"] = _normals(triangles).astype(np.float32, copy=False)
    records["vertices"] = triangles.astype(np.float32, copy=False)
    data.extend(records.tobytes())
    return bytes(data)


def _ascii_stl(triangles: FloatArray) -> str:
    if triangles.shape[0] == 0:
        return "solid fascat\nendsolid fascat\n"
    rows = np.column_stack([_normals(triangles), triangles.reshape((triangles.shape[0], 9))])
    output = StringIO()
    output.write("solid fascat\n")
    np.savetxt(
        output,
        rows,
        fmt=(
            "  facet normal %.9g %.9g %.9g\n"
            "    outer loop\n"
            "      vertex %.9g %.9g %.9g\n"
            "      vertex %.9g %.9g %.9g\n"
            "      vertex %.9g %.9g %.9g\n"
            "    endloop\n"
            "  endfacet"
        ),
    )
    output.write("endsolid fascat\n")
    return output.getvalue()


def _normals(triangles: FloatArray) -> FloatArray:
    if triangles.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    return normalize_rows(normals, degenerate=None)
