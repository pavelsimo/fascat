#!/usr/bin/env python3
"""Deterministic gate checker for the cad-to-rt3d skill.

Parses saved ``fascat --json validate`` output (and optionally the convert
``--report`` JSON) and prints one PASS/FAIL line per gate plus an OVERALL
verdict. Exits 0 when every evaluated gate passes, 1 otherwise. Gates whose
inputs are absent are reported as SKIP and do not affect the verdict.

Stdlib-only on purpose: it must run in any environment, independent of the
fascat install.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _find_key(payload: Any, key: str) -> Any:
    """Return the first value for *key* found by depth-first walk, else None."""
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_key(value, key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_key(item, key)
            if found is not None:
                return found
    return None


def _gate(name: str, actual: float | None, limit: float | None, *, op: str = "<=") -> dict[str, Any]:
    if actual is None or limit is None:
        return {"gate": name, "status": "SKIP", "actual": actual, "op": op, "limit": limit}
    passed = actual <= limit if op == "<=" else actual == limit
    return {"gate": name, "status": "PASS" if passed else "FAIL", "actual": actual, "op": op, "limit": limit}


def _as_number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def evaluate(
    validate_payload: dict[str, Any],
    convert_payload: dict[str, Any] | None,
    output_path: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    summary = _find_key(validate_payload.get("analysis"), "summary") or {}
    stats = validate_payload.get("stats") or {}
    gates: list[dict[str, Any]] = []

    gates.append(_gate("non_manifold_edges", _as_number(summary.get("non_manifold_edges")), args.max_non_manifold))
    gates.append(
        _gate("self_intersections", _as_number(summary.get("self_intersections")), args.max_self_intersections)
    )
    gates.append(_gate("sliver_triangles", _as_number(summary.get("sliver_triangles")), args.max_slivers))
    gates.append(_gate("open_boundaries", _as_number(summary.get("open_boundaries")), args.max_open_boundaries))

    triangles = _as_number(stats.get("triangles"))
    if triangles is None:
        triangles = _as_number(summary.get("triangles"))
    triangle_budget = args.max_triangles
    if triangle_budget is None and convert_payload is not None:
        triangle_budget = _as_number(_find_key(convert_payload, "profile_triangle_budget"))
    gates.append(_gate("triangles_within_budget", triangles, triangle_budget))
    if convert_payload is not None:
        over_budget = _as_number(_find_key(convert_payload, "profile_triangles_over_budget"))
        gates.append(_gate("profile_triangles_over_budget", over_budget, 0.0))

    file_size: float | None = None
    if output_path is not None and Path(output_path).exists():
        file_size = float(Path(output_path).stat().st_size)
    elif convert_payload is not None:
        file_size = _as_number(_find_key(convert_payload, "file_size_bytes"))
    size_budget = args.max_file_size_mb * 1024 * 1024 if args.max_file_size_mb is not None else None
    if size_budget is None and convert_payload is not None:
        size_budget = _as_number(_find_key(convert_payload, "file_size_budget_bytes"))
    gates.append(_gate("file_size_bytes", file_size, size_budget))

    turntable = validate_payload.get("turntable")
    if isinstance(turntable, dict) and turntable.get("diff_passed") is not None:
        gates.append(
            {
                "gate": "turntable_diff",
                "status": "PASS" if turntable["diff_passed"] else "FAIL",
                "actual": turntable.get("diff", {}).get("views_failed"),
                "op": "==",
                "limit": 0,
            }
        )
    else:
        gates.append({"gate": "turntable_diff", "status": "SKIP", "actual": None, "op": "==", "limit": 0})

    lod_preview = validate_payload.get("lod_preview")
    if isinstance(lod_preview, dict) and "monotonic_triangles" in lod_preview:
        gates.append(
            {
                "gate": "lod_monotonic_triangles",
                "status": "PASS" if lod_preview["monotonic_triangles"] else "FAIL",
                "actual": lod_preview["monotonic_triangles"],
                "op": "==",
                "limit": True,
            }
        )
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validate-json", required=True, help="Saved `fascat --json validate` stdout")
    parser.add_argument("--convert-json", help="Saved `fascat convert --report` JSON (budget fields)")
    parser.add_argument("--output", help="Converted output file, for on-disk size checking")
    parser.add_argument("--max-triangles", type=float, default=None, help="Triangle budget override")
    parser.add_argument("--max-file-size-mb", type=float, default=None, help="File size budget override in MiB")
    parser.add_argument("--max-non-manifold", type=float, default=0.0)
    parser.add_argument("--max-self-intersections", type=float, default=0.0)
    parser.add_argument("--max-slivers", type=float, default=0.0)
    parser.add_argument("--max-open-boundaries", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text")
    args = parser.parse_args(argv)

    validate_payload = _load_json(args.validate_json)
    convert_payload = _load_json(args.convert_json) if args.convert_json else None
    gates = evaluate(validate_payload, convert_payload, args.output, args)

    failed = [gate for gate in gates if gate["status"] == "FAIL"]
    evaluated = [gate for gate in gates if gate["status"] != "SKIP"]
    overall = "PASS" if not failed else "FAIL"
    if args.json:
        print(json.dumps({"overall": overall, "failed": len(failed), "evaluated": len(evaluated), "gates": gates}))
    else:
        for gate in gates:
            print(f"{gate['status']} {gate['gate']} {gate['actual']} {gate['op']} {gate['limit']}")
        print(f"OVERALL {overall} ({len(failed)}/{len(evaluated)} evaluated gates failed)")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
