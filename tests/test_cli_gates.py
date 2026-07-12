from __future__ import annotations

import pytest

from fascat.cli._gates import (
    GateResult,
    GateThresholds,
    any_gate_failed,
    evaluate_gates,
    format_gate_lines,
    gates_to_dict,
    resolve_thresholds,
    structural_gate,
)


def _statuses(results: list[GateResult]) -> dict[str, str]:
    return {result.gate: result.status for result in results}


def test_resolve_thresholds_defaults_to_no_gates() -> None:
    thresholds = resolve_thresholds(
        max_non_manifold=None,
        max_self_intersections=None,
        max_slivers=None,
        max_open_boundaries=None,
        max_triangles=None,
        max_file_size_mb=None,
        strict_geometry=False,
        profile_max_triangles=None,
        profile_max_file_size_mb=None,
        profile_requested=False,
    )
    assert thresholds == GateThresholds()


def test_resolve_thresholds_strict_geometry_sets_zero_limits() -> None:
    thresholds = resolve_thresholds(
        max_non_manifold=None,
        max_self_intersections=None,
        max_slivers=None,
        max_open_boundaries=None,
        max_triangles=None,
        max_file_size_mb=None,
        strict_geometry=True,
        profile_max_triangles=None,
        profile_max_file_size_mb=None,
        profile_requested=False,
    )
    assert thresholds.max_non_manifold == 0
    assert thresholds.max_self_intersections == 0
    assert thresholds.max_slivers == 0
    assert thresholds.max_open_boundaries == 0


def test_resolve_thresholds_explicit_overrides_strict_geometry() -> None:
    thresholds = resolve_thresholds(
        max_non_manifold=None,
        max_self_intersections=None,
        max_slivers=5,
        max_open_boundaries=None,
        max_triangles=None,
        max_file_size_mb=None,
        strict_geometry=True,
        profile_max_triangles=None,
        profile_max_file_size_mb=None,
        profile_requested=False,
    )
    assert thresholds.max_slivers == 5
    assert thresholds.max_non_manifold == 0


def test_resolve_thresholds_profile_supplies_budget() -> None:
    thresholds = resolve_thresholds(
        max_non_manifold=None,
        max_self_intersections=None,
        max_slivers=None,
        max_open_boundaries=None,
        max_triangles=None,
        max_file_size_mb=None,
        strict_geometry=False,
        profile_max_triangles=250_000,
        profile_max_file_size_mb=50.0,
        profile_requested=True,
    )
    assert thresholds.max_triangles == 250_000
    assert thresholds.triangles_requested is True
    assert thresholds.max_file_size_mb == 50.0
    assert thresholds.file_size_requested is True


def test_resolve_thresholds_explicit_overrides_profile_budget() -> None:
    thresholds = resolve_thresholds(
        max_non_manifold=None,
        max_self_intersections=None,
        max_slivers=None,
        max_open_boundaries=None,
        max_triangles=100,
        max_file_size_mb=10.0,
        strict_geometry=False,
        profile_max_triangles=250_000,
        profile_max_file_size_mb=50.0,
        profile_requested=True,
    )
    assert thresholds.max_triangles == 100
    assert thresholds.max_file_size_mb == 10.0


@pytest.mark.parametrize(
    ("summary_value", "limit", "expected"),
    [
        (0, 0, "PASS"),
        (1, 0, "FAIL"),
        (3, 3, "PASS"),
        (4, 3, "FAIL"),
        (None, 0, "SKIP"),
    ],
)
def test_evaluate_gates_geometry_statuses(summary_value: int | None, limit: int, expected: str) -> None:
    summary = {} if summary_value is None else {"non_manifold_edges": summary_value}
    results = evaluate_gates(
        GateThresholds(max_non_manifold=limit),
        summary=summary,
        triangles=None,
        file_size_bytes=None,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    assert _statuses(results)["non_manifold_edges"] == expected


def test_evaluate_gates_skips_geometry_without_summary() -> None:
    results = evaluate_gates(
        GateThresholds(max_open_boundaries=0),
        summary=None,
        triangles=None,
        file_size_bytes=None,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    assert _statuses(results) == {"open_boundaries": "SKIP"}


def test_evaluate_gates_omits_unrequested_gates() -> None:
    results = evaluate_gates(
        GateThresholds(),
        summary={"non_manifold_edges": 5},
        triangles=100,
        file_size_bytes=100,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    assert results == []


def test_evaluate_gates_triangles_and_file_size() -> None:
    results = evaluate_gates(
        GateThresholds(max_triangles=10, max_file_size_mb=1.0),
        summary=None,
        triangles=11,
        file_size_bytes=512,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    statuses = _statuses(results)
    assert statuses["triangles"] == "FAIL"
    assert statuses["file_size_bytes"] == "PASS"
    file_size = next(result for result in results if result.gate == "file_size_bytes")
    assert file_size.limit == 1024 * 1024


def test_evaluate_gates_requested_but_unresolved_budgets_skip() -> None:
    results = evaluate_gates(
        GateThresholds(triangles_requested=True, file_size_requested=True),
        summary=None,
        triangles=100,
        file_size_bytes=100,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    assert _statuses(results) == {"triangles": "SKIP", "file_size_bytes": "SKIP"}


def test_evaluate_gates_report_gates_follow_inputs_without_gating() -> None:
    results = evaluate_gates(
        GateThresholds(),
        summary=None,
        triangles=None,
        file_size_bytes=None,
        visual_diff_passed=False,
        turntable_views_failed=2,
        lod_monotonic=False,
        include_report_gates=False,
    )
    statuses = _statuses(results)
    assert statuses == {"visual_diff": "FAIL", "turntable_diff": "FAIL"}


def test_evaluate_gates_report_gates_render_as_skip_when_gating_active() -> None:
    results = evaluate_gates(
        GateThresholds(max_triangles=10),
        summary=None,
        triangles=1,
        file_size_bytes=None,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=True,
    )
    statuses = _statuses(results)
    assert statuses["structural"] == "PASS"
    assert statuses["visual_diff"] == "SKIP"
    assert statuses["turntable_diff"] == "SKIP"
    assert statuses["lod_monotonic_triangles"] == "SKIP"


def test_evaluate_gates_turntable_and_lod_failures() -> None:
    results = evaluate_gates(
        GateThresholds(),
        summary=None,
        triangles=None,
        file_size_bytes=None,
        visual_diff_passed=True,
        turntable_views_failed=0,
        lod_monotonic=False,
        include_report_gates=True,
    )
    statuses = _statuses(results)
    assert statuses["visual_diff"] == "PASS"
    assert statuses["turntable_diff"] == "PASS"
    assert statuses["lod_monotonic_triangles"] == "FAIL"


def test_structural_gate_failure() -> None:
    result = structural_gate(ok=False)
    assert result.status == "FAIL"
    assert result.actual is False
    assert result.limit is True


def test_gates_to_dict_counts_exclude_skips() -> None:
    results = evaluate_gates(
        GateThresholds(max_non_manifold=0, max_triangles=10, file_size_requested=True),
        summary={"non_manifold_edges": 1},
        triangles=5,
        file_size_bytes=100,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    payload = gates_to_dict(results)
    assert payload["overall"] == "FAIL"
    assert payload["failed"] == 1
    assert payload["evaluated"] == 2
    assert isinstance(payload["results"], list)
    assert len(payload["results"]) == 3


def test_format_gate_lines_layout() -> None:
    results = evaluate_gates(
        GateThresholds(max_non_manifold=0, max_triangles=10),
        summary={"non_manifold_edges": 1},
        triangles=5,
        file_size_bytes=None,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    lines = format_gate_lines(results)
    assert lines == [
        "FAIL non_manifold_edges 1 <= 0",
        "PASS triangles 5 <= 10",
        "OVERALL FAIL (1/2 evaluated gates failed)",
    ]


def test_any_gate_failed_ignores_skips() -> None:
    passing = evaluate_gates(
        GateThresholds(max_non_manifold=0, triangles_requested=True),
        summary={"non_manifold_edges": 0},
        triangles=5,
        file_size_bytes=None,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    assert any_gate_failed(passing) is False
    failing = evaluate_gates(
        GateThresholds(max_non_manifold=0),
        summary={"non_manifold_edges": 2},
        triangles=None,
        file_size_bytes=None,
        visual_diff_passed=None,
        turntable_views_failed=None,
        lod_monotonic=None,
        include_report_gates=False,
    )
    assert any_gate_failed(failing) is True
