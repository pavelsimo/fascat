from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

GateStatus = Literal["PASS", "FAIL", "SKIP"]

GateValue = int | float | bool | None

_GEOMETRY_SUMMARY_KEYS = {
    "non_manifold_edges": "non_manifold_edges",
    "self_intersections": "self_intersections",
    "sliver_triangles": "sliver_triangles",
    "open_boundaries": "open_boundaries",
}


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: GateStatus
    actual: GateValue
    op: str
    limit: GateValue

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "status": self.status,
            "actual": self.actual,
            "op": self.op,
            "limit": self.limit,
        }


@dataclass(frozen=True)
class GateThresholds:
    max_non_manifold: int | None = None
    max_self_intersections: int | None = None
    max_slivers: int | None = None
    max_open_boundaries: int | None = None
    max_triangles: int | None = None
    max_file_size_mb: float | None = None
    triangles_requested: bool = False
    file_size_requested: bool = False


def resolve_thresholds(
    *,
    max_non_manifold: int | None,
    max_self_intersections: int | None,
    max_slivers: int | None,
    max_open_boundaries: int | None,
    max_triangles: int | None,
    max_file_size_mb: float | None,
    strict_geometry: bool,
    profile_max_triangles: int | None,
    profile_max_file_size_mb: float | None,
    profile_requested: bool,
) -> GateThresholds:
    geometry_default = 0 if strict_geometry else None
    return GateThresholds(
        max_non_manifold=max_non_manifold if max_non_manifold is not None else geometry_default,
        max_self_intersections=max_self_intersections if max_self_intersections is not None else geometry_default,
        max_slivers=max_slivers if max_slivers is not None else geometry_default,
        max_open_boundaries=max_open_boundaries if max_open_boundaries is not None else geometry_default,
        max_triangles=max_triangles if max_triangles is not None else profile_max_triangles,
        max_file_size_mb=(max_file_size_mb if max_file_size_mb is not None else profile_max_file_size_mb),
        triangles_requested=max_triangles is not None or profile_requested,
        file_size_requested=max_file_size_mb is not None or profile_requested,
    )


def _threshold_gate(gate: str, actual: GateValue, limit: GateValue, *, op: str = "<=") -> GateResult:
    if actual is None or limit is None:
        return GateResult(gate=gate, status="SKIP", actual=actual, op=op, limit=limit)
    passed = actual <= limit if op == "<=" else actual == limit
    return GateResult(gate=gate, status="PASS" if passed else "FAIL", actual=actual, op=op, limit=limit)


def structural_gate(*, ok: bool) -> GateResult:
    return GateResult(gate="structural", status="PASS" if ok else "FAIL", actual=ok, op="==", limit=True)


def evaluate_gates(
    thresholds: GateThresholds,
    *,
    summary: Mapping[str, object] | None,
    triangles: int | None,
    file_size_bytes: int | None,
    visual_diff_passed: bool | None,
    turntable_views_failed: int | None,
    lod_monotonic: bool | None,
    include_report_gates: bool,
) -> list[GateResult]:
    results: list[GateResult] = []
    if include_report_gates:
        results.append(structural_gate(ok=True))
    geometry_limits = {
        "non_manifold_edges": thresholds.max_non_manifold,
        "self_intersections": thresholds.max_self_intersections,
        "sliver_triangles": thresholds.max_slivers,
        "open_boundaries": thresholds.max_open_boundaries,
    }
    for gate, limit in geometry_limits.items():
        if limit is None:
            continue
        raw = summary.get(_GEOMETRY_SUMMARY_KEYS[gate]) if summary is not None else None
        actual = raw if isinstance(raw, int | float) and not isinstance(raw, bool) else None
        results.append(_threshold_gate(gate, actual, limit))
    if thresholds.max_triangles is not None or thresholds.triangles_requested:
        results.append(_threshold_gate("triangles", triangles, thresholds.max_triangles))
    if thresholds.max_file_size_mb is not None or thresholds.file_size_requested:
        limit_bytes = None if thresholds.max_file_size_mb is None else thresholds.max_file_size_mb * 1024 * 1024
        if isinstance(limit_bytes, float) and limit_bytes.is_integer():
            limit_bytes = int(limit_bytes)
        results.append(_threshold_gate("file_size_bytes", file_size_bytes, limit_bytes))
    if visual_diff_passed is not None or include_report_gates:
        results.append(_threshold_gate("visual_diff", visual_diff_passed, True, op="=="))
    if turntable_views_failed is not None or include_report_gates:
        results.append(_threshold_gate("turntable_diff", turntable_views_failed, 0, op="=="))
    if include_report_gates:
        results.append(_threshold_gate("lod_monotonic_triangles", lod_monotonic, True, op="=="))
    return results


def gates_to_dict(results: Sequence[GateResult]) -> dict[str, object]:
    failed = sum(1 for result in results if result.status == "FAIL")
    evaluated = sum(1 for result in results if result.status != "SKIP")
    return {
        "overall": "FAIL" if failed else "PASS",
        "failed": failed,
        "evaluated": evaluated,
        "results": [result.to_dict() for result in results],
    }


def format_gate_lines(results: Sequence[GateResult]) -> list[str]:
    lines = [f"{result.status} {result.gate} {result.actual} {result.op} {result.limit}" for result in results]
    failed = sum(1 for result in results if result.status == "FAIL")
    evaluated = sum(1 for result in results if result.status != "SKIP")
    overall = "FAIL" if failed else "PASS"
    lines.append(f"OVERALL {overall} ({failed}/{evaluated} evaluated gates failed)")
    return lines


def any_gate_failed(results: Sequence[GateResult]) -> bool:
    return any(result.status == "FAIL" for result in results)
