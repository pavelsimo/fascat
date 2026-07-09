"""JT import orchestration: container -> LSG -> shape decode -> Asset.

Mirrors the reader contract of ``fascat/io/iges.py`` / ``fascat/io/brep.py``:
options coercion, ``wrap_io_errors``, report steps, and stable sha1 ids. JT
import is tessellation-only — parts carry ``mesh`` with ``source_shape=None``
and flow through the mesh-reuse pipeline path without OCP.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from fascat.asset import Asset, Node, Part
from fascat.io import step as _step
from fascat.io._errors import wrap_io_errors
from fascat.io._suffixes import JT_SUFFIXES
from fascat.io.jt import lsg as _lsg
from fascat.io.jt.container import (
    BREP_SEGMENT_TYPES,
    SHAPE_SEGMENT_TYPES,
    FileHeader,
    Toc,
    load_segment,
    read_file_header,
    read_toc,
)
from fascat.io.jt.shape import DecodedShape, decode_shape_lod
from fascat.material import Material
from fascat.mesh import Mesh
from fascat.options import JtReadOptions, StepReadOptions
from fascat.report import Report, timed_step

_DEFAULT_MATERIAL_COLOR = (0.75, 0.75, 0.75, 1.0)

_UNIT_NAMES = {
    "micrometers": ("micrometre", 1e-6),
    "millimeters": ("millimetre", 0.001),
    "centimeters": ("centimetre", 0.01),
    "decimeters": ("decimetre", 0.1),
    "meters": ("metre", 1.0),
    "kilometers": ("kilometre", 1000.0),
    "mils": ("mil", 0.0000254),
    "inches": ("inch", 0.0254),
    "feet": ("foot", 0.3048),
    "yards": ("yard", 0.9144),
}


@wrap_io_errors("read JT")
def read_jt(path: str | Path, *, options: JtReadOptions | StepReadOptions | None = None) -> Asset:
    """Read a JT file into an asset."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"missing JT file: {source}")
    if source.suffix.lower() not in JT_SUFFIXES:
        raise ValueError(f"unsupported JT extension: {source.suffix or '<none>'}")
    data = source.read_bytes()
    return _read_jt_data(data, source=source, source_identity=str(source.resolve()), options=_coerce_options(options))


@wrap_io_errors("read JT bytes")
def read_jt_bytes(
    data: bytes,
    *,
    name: str = "stdin.jt",
    options: JtReadOptions | StepReadOptions | None = None,
) -> Asset:
    asset = _read_jt_data(data, source=Path(name), source_identity=name, options=_coerce_options(options))
    asset.source_path = None
    asset.report.source_path = None
    asset.root.metadata["source"] = name
    if asset.metadata:
        asset.metadata["source"] = name
        asset.metadata["source_identity"] = name
    return asset


def _coerce_options(options: JtReadOptions | StepReadOptions | None) -> JtReadOptions:
    if options is None:
        return JtReadOptions()
    if isinstance(options, JtReadOptions):
        return options
    return JtReadOptions(**cast(Any, options.to_dict()))


def _read_jt_data(data: bytes, *, source: Path, source_identity: str, options: JtReadOptions) -> Asset:
    cleanup = _step._ImportCleanupStats()
    warnings: list[str] = []
    with timed_step() as timer:
        header = read_file_header(data)
        toc = read_toc(data, header)
        lsg_entry = toc.find(header.lsg_segment_id)
        if lsg_entry is None:
            lsg_entry = next((entry for entry in toc.entries if entry.segment_type == 1), None)
        if lsg_entry is None:
            raise RuntimeError("corrupt JT file: no Logical Scene Graph segment")
        graph = _lsg.parse_lsg(
            load_segment(data, lsg_entry, header=header),
            version=header.version,
            byte_order=header.byte_order,
        )
        unit_name, meters_per_unit = _resolve_units(graph, warnings)
        space = _step._space_normalization(unit_name, meters_per_unit, options)
        traverser = _Traverser(data, header, toc, graph, options, source_identity, warnings)
        root = traverser.build_root(source.stem, space)
        if not traverser.parts:
            raise RuntimeError("JT file contains no tessellated LOD data (B-rep only); re-export with tessellation")

    report = Report(source_path=str(source))
    asset = Asset(
        root=root,
        parts=traverser.parts,
        materials=traverser.materials,
        units=space.target_units,
        meters_per_unit=space.target_meters_per_unit,
        up_axis=cast(Any, space.target_up_axis),
        source_path=source,
        metadata=_asset_metadata(source, source_identity, options, cleanup, space),
        pmi=[],
        report=report,
    )
    asset.report.input_stats = asset.stats()
    loaded_representations = _step._loaded_representation_report(asset)
    if asset.metadata:
        asset.metadata["import_representation_summary"] = loaded_representations["summary"]
    for warning in warnings:
        asset.report.add_warning(warning)
    asset.report.add_step(
        "import",
        options={
            "format": "JT",
            "backend": "pure-python",
            "jt_version": f"{header.version[0]}.{header.version[1]}",
            "read_options": options.to_dict(),
            "metadata_count": _step._metadata_count(asset),
            "cleanup": cleanup.to_dict(),
            "space_normalization": space.metadata(),
            "lod_summary": traverser.lod_summary(),
            "skipped_elements": dict(graph.skipped_elements),
            "loaded_representations": loaded_representations,
        },
        before={"nodes": 0, "parts": 0, "occurrences": 0, "materials": 0, "vertices": 0, "triangles": 0},
        after=asset.stats(),
        duration=timer.duration,
        warnings=warnings,
    )
    return asset


def _asset_metadata(
    source: Path,
    source_identity: str,
    options: JtReadOptions,
    cleanup: _step._ImportCleanupStats,
    space: _step._SpaceNormalization,
) -> dict[str, object]:
    metadata = _step._asset_metadata(source, source_identity, options, _step._StepHeaderInfo(), cleanup, space)
    if metadata:
        metadata["format"] = "JT"
    return metadata


def _resolve_units(graph: _lsg.Lsg, warnings: list[str]) -> tuple[str, float]:
    """Map the conventional JT_PROP_MEASUREMENT_UNITS property to fascat units."""
    declared: str | None = None
    if graph.root_id is not None:
        for properties in [graph.properties.get(graph.root_id, {})] + [
            graph.properties.get(oid, {}) for oid in graph.nodes
        ]:
            value = properties.get("JT_PROP_MEASUREMENT_UNITS")
            if isinstance(value, str) and value:
                declared = value
                break
    if declared is None:
        warnings.append("JT file declares no measurement units; assuming millimetres")
        return "millimetre", 0.001
    mapped = _UNIT_NAMES.get(declared.strip().lower())
    if mapped is None:
        warnings.append(f"unrecognized JT measurement units {declared!r}; assuming millimetres")
        return "millimetre", 0.001
    return mapped


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _material_id(color: tuple[float, float, float, float]) -> str:
    encoded = ",".join(f"{component:.6f}" for component in color)
    return _stable_id("mat", encoded)


class _Traverser:
    """Builds the fascat node tree, parts, and materials from the parsed LSG."""

    def __init__(
        self,
        data: bytes,
        header: FileHeader,
        toc: Toc,
        graph: _lsg.Lsg,
        options: JtReadOptions,
        source_identity: str,
        warnings: list[str],
    ) -> None:
        self._data = data
        self._header = header
        self._toc = toc
        self._graph = graph
        self._options = options
        self._source_identity = source_identity
        self._warnings = warnings
        self.parts: dict[str, Part] = {}
        self.materials: dict[str, Material] = {}
        self._part_by_object: dict[int, str | None] = {}
        self._shape_cache: dict[bytes, DecodedShape] = {}
        self._saw_brep_refs = False
        self._imported_lod_meshes = 0

    def lod_summary(self) -> dict[str, object]:
        return {
            "lod_selection": self._options.lod_selection,
            "imported_lod_meshes": self._imported_lod_meshes,
            "parts_with_lods": sum(1 for part in self.parts.values() if part.lod_meshes),
        }

    def build_root(self, default_name: str, space: _step._SpaceNormalization) -> Node:
        root_id = self._find_root_id()
        metadata: dict[str, object] = {
            "source": self._source_identity,
            "source_identity": self._source_identity,
            "space_normalization": space.metadata(),
        }
        root = Node(
            id=_stable_id("node", f"{self._source_identity}:root"),
            name=default_name,
            transform=space.transform,
            metadata=metadata,
        )
        if root_id is not None:
            child = self._build_node(root_id, "root/0", None, default_name, set())
            if child is not None:
                root.children.append(child)
        return root

    def _find_root_id(self) -> int | None:
        referenced = {child for node in self._graph.nodes.values() for child in node.child_ids}
        for object_id, node in self._graph.nodes.items():
            if node.kind == "partition" and object_id not in referenced:
                return object_id
        for object_id in self._graph.nodes:
            if object_id not in referenced:
                return object_id
        return self._graph.root_id

    def _node_name(self, object_id: int, fallback: str) -> str:
        raw = self._graph.properties.get(object_id, {}).get("JT_PROP_NAME")
        if isinstance(raw, str) and raw:
            return _lsg.display_name(raw)
        return fallback

    def _build_node(
        self,
        object_id: int,
        path: str,
        material_id: str | None,
        inherited_name: str,
        visiting: frozenset[int] | set[int],
    ) -> Node | None:
        lsg_node = self._graph.nodes.get(object_id)
        if lsg_node is None:
            self._warnings.append(f"JT node {object_id} is referenced but missing or unsupported; skipped")
            return None
        if object_id in visiting:
            self._warnings.append(f"JT node {object_id} forms a reference cycle; pruned")
            return None
        visiting = set(visiting) | {object_id}
        for attribute_id in lsg_node.attribute_ids:
            jt_material = self._graph.materials.get(attribute_id)
            if jt_material is not None:
                material_id = self._register_material(jt_material)
        name = self._node_name(object_id, inherited_name)
        node = Node(id=_stable_id("node", f"{self._source_identity}:{path}"), name=name)
        for attribute_id in lsg_node.attribute_ids:
            jt_transform = self._graph.transforms.get(attribute_id)
            if jt_transform is not None:
                # JT stores row-vector matrices (p' = pAM); fascat uses column vectors.
                node.transform = np.ascontiguousarray(jt_transform.matrix.T)
        if lsg_node.kind == "partition" and lsg_node.file_name:
            node.metadata["external_reference"] = lsg_node.file_name
            self._warnings.append(
                f"JT external partition reference '{lsg_node.file_name}' is not resolved; placeholder node emitted"
            )
            return node
        shape_refs = self._shape_refs(object_id)
        if shape_refs:
            part_id = self._part_for_shape(object_id, shape_refs, name)
            if part_id is not None:
                node.part_id = part_id
                node.metadata["loaded_representation"] = "mesh"
                part = self.parts[part_id]
                if not part.material_ids:
                    part.material_ids = [material_id if material_id is not None else self._default_material_id()]
        child_ids = list(lsg_node.child_ids)
        if lsg_node.kind in ("lod", "range_lod") and child_ids:
            representative = child_ids[0]
            child = self._build_node(representative, f"{path}/0", material_id, name, visiting)
            if child is not None:
                node.children.append(child)
                if len(child_ids) > 1:
                    self._attach_coarser_lods(child, child_ids[1:])
            return node
        for index, child_id in enumerate(child_ids):
            child = self._build_node(child_id, f"{path}/{index}", material_id, name, visiting)
            if child is not None:
                node.children.append(child)
        properties = self._flattened_properties(object_id)
        if properties:
            part_id = node.part_id
            if part_id is None and lsg_node.kind == "part":
                part_id = self._first_part_id(node)
            if part_id is not None:
                part_metadata = self.parts[part_id].metadata
                for key, value in properties.items():
                    part_metadata.setdefault(key, value)
        return node

    def _flattened_properties(self, object_id: int) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in self._graph.properties.get(object_id, {}).items():
            if isinstance(value, (str, int, float)):
                out[key] = value
        return out

    def _shape_refs(self, object_id: int) -> list[_lsg.LateLoadedRef]:
        refs = []
        for ref in self._graph.late_loaded.get(object_id, []):
            if ref.segment_type in SHAPE_SEGMENT_TYPES:
                refs.append(ref)
            elif ref.segment_type in BREP_SEGMENT_TYPES:
                self._saw_brep_refs = True
        return sorted(refs, key=lambda ref: ref.segment_type)

    def _part_for_shape(self, object_id: int, refs: list[_lsg.LateLoadedRef], name: str) -> str | None:
        if object_id in self._part_by_object:
            return self._part_by_object[object_id]
        shape = self._decode_ref(refs[0], name)
        if shape is None:
            self._part_by_object[object_id] = None
            return None
        mesh = Mesh(points=shape.points, faces=shape.faces, normals=shape.normals)
        part_id = _stable_id("part", f"{self._source_identity}:object:{object_id}")
        part = Part(
            id=part_id,
            name=name,
            source_shape=None,
            mesh=mesh,
            fingerprint=mesh.fingerprint(),
            metadata={
                "source_identity": self._source_identity,
                "source_name": name,
                "loaded_representation": "mesh",
                "lod_count": str(len(refs)),
            },
        )
        if self._options.lod_selection == "all":
            for ref in refs[1:]:
                coarser = self._decode_ref(ref, name)
                if coarser is None:
                    continue
                lod_mesh = Mesh(points=coarser.points, faces=coarser.faces, normals=coarser.normals)
                lod_mesh.metadata["lod_source"] = "imported"
                part.lod_meshes.append(lod_mesh)
                self._imported_lod_meshes += 1
        self.parts[part_id] = part
        self._part_by_object[object_id] = part_id
        return part_id

    def _decode_ref(self, ref: _lsg.LateLoadedRef, name: str) -> DecodedShape | None:
        cached = self._shape_cache.get(ref.segment_id)
        if cached is not None:
            return cached
        entry = self._toc.find(ref.segment_id)
        if entry is None:
            self._warnings.append(f"JT shape segment for part '{name}' is missing from the TOC; part skipped")
            return None
        try:
            payload = load_segment(self._data, entry, header=self._header)
            shape = decode_shape_lod(payload, version=self._header.version, byte_order=self._header.byte_order)
        except RuntimeError as exc:
            self._warnings.append(f"JT shape decode failed for part '{name}': {exc}")
            return None
        self._shape_cache[ref.segment_id] = shape
        return shape

    def _attach_coarser_lods(self, representative: Node, sibling_ids: list[int]) -> None:
        part_id = self._first_part_id(representative)
        if part_id is None:
            return
        part = self.parts[part_id]
        part.metadata["lod_count"] = str(1 + len(sibling_ids))
        if self._options.lod_selection != "all":
            return
        for sibling_id in sibling_ids:
            for ref in self._collect_shape_refs(sibling_id, set()):
                shape = self._decode_ref(ref, part.name)
                if shape is None:
                    continue
                lod_mesh = Mesh(points=shape.points, faces=shape.faces, normals=shape.normals)
                lod_mesh.metadata["lod_source"] = "imported"
                part.lod_meshes.append(lod_mesh)
                self._imported_lod_meshes += 1
                break  # finest representation of this coarser LOD level

    def _first_part_id(self, node: Node) -> str | None:
        if node.part_id is not None:
            return node.part_id
        for child in node.children:
            found = self._first_part_id(child)
            if found is not None:
                return found
        return None

    def _collect_shape_refs(self, object_id: int, seen: set[int]) -> list[_lsg.LateLoadedRef]:
        if object_id in seen:
            return []
        seen.add(object_id)
        refs = self._shape_refs(object_id)
        if refs:
            return refs
        lsg_node = self._graph.nodes.get(object_id)
        if lsg_node is None:
            return []
        for child_id in lsg_node.child_ids:
            refs = self._collect_shape_refs(child_id, seen)
            if refs:
                return refs
        return []

    def _default_material_id(self) -> str:
        material_id = _material_id(_DEFAULT_MATERIAL_COLOR)
        if material_id not in self.materials:
            self.materials[material_id] = Material(
                id=material_id,
                name="Default JT material",
                base_color=_DEFAULT_MATERIAL_COLOR,
            )
        return material_id

    def _register_material(self, jt_material: _lsg.JtMaterial) -> str:
        diffuse = tuple(float(component) for component in jt_material.diffuse)
        material_id = _material_id(cast("tuple[float, float, float, float]", diffuse))
        if material_id not in self.materials:
            roughness = min(max(math.sqrt(2.0 / (jt_material.shininess + 2.0)), 0.0), 1.0)
            self.materials[material_id] = Material(
                id=material_id,
                name="JT material",
                base_color=cast("tuple[float, float, float, float]", diffuse),
                metallic=0.0,
                roughness=roughness,
                opacity=float(jt_material.diffuse[3]),
                metadata={
                    "jt_ambient": ",".join(f"{value:.6f}" for value in jt_material.ambient),
                    "jt_specular": ",".join(f"{value:.6f}" for value in jt_material.specular),
                    "jt_emission": ",".join(f"{value:.6f}" for value in jt_material.emission),
                    "jt_shininess": f"{jt_material.shininess:.6f}",
                },
            )
        return material_id
