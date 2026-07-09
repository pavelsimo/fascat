from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GATES_SCRIPT = Path(__file__).resolve().parent.parent / "skills" / "cad-to-rt3d" / "scripts" / "gates.py"


def _run_gates(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATES_SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _write_validate_payload(path: Path, **overrides: object) -> Path:
    payload: dict[str, object] = {
        "command": "validate",
        "stats": {"triangles": 5000},
        "analysis": {
            "summary": {
                "non_manifold_edges": 0,
                "self_intersections": 0,
                "open_boundaries": 0,
                "sliver_triangles": 0,
            }
        },
        "turntable": {"diff_passed": True, "diff": {"views_failed": 0}},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_gates_pass_for_clean_payload(tmp_path: Path) -> None:
    validate_json = _write_validate_payload(tmp_path / "validate.json")

    result = _run_gates(["--validate-json", str(validate_json), "--max-triangles", "10000"])

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OVERALL PASS" in result.stdout
    assert "PASS non_manifold_edges" in result.stdout
    assert "PASS turntable_diff" in result.stdout


def test_gates_fail_on_geometry_defects(tmp_path: Path) -> None:
    validate_json = _write_validate_payload(
        tmp_path / "validate.json",
        analysis={
            "summary": {
                "non_manifold_edges": 3,
                "self_intersections": 0,
                "open_boundaries": 1,
                "sliver_triangles": 0,
            }
        },
    )

    result = _run_gates(["--validate-json", str(validate_json)])

    assert result.returncode == 1
    assert "FAIL non_manifold_edges 3.0 <= 0.0" in result.stdout
    assert "FAIL open_boundaries 1.0 <= 0.0" in result.stdout
    assert "OVERALL FAIL" in result.stdout


def test_gates_fail_on_turntable_diff(tmp_path: Path) -> None:
    validate_json = _write_validate_payload(
        tmp_path / "validate.json",
        turntable={"diff_passed": False, "diff": {"views_failed": 2}},
    )

    result = _run_gates(["--validate-json", str(validate_json)])

    assert result.returncode == 1
    assert "FAIL turntable_diff 2" in result.stdout


def test_gates_use_convert_report_budgets(tmp_path: Path) -> None:
    validate_json = _write_validate_payload(tmp_path / "validate.json")
    convert_json = tmp_path / "report.json"
    convert_json.write_text(
        json.dumps(
            {
                "steps": [
                    {"after": {"profile_triangle_budget": 4000, "profile_triangles_over_budget": 1000}},
                    {"after": {"file_size_bytes": 100, "file_size_budget_bytes": 500}},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = _run_gates(["--validate-json", str(validate_json), "--convert-json", str(convert_json)])

    assert result.returncode == 1
    assert "FAIL triangles_within_budget 5000.0 <= 4000.0" in result.stdout
    assert "FAIL profile_triangles_over_budget 1000.0 <= 0.0" in result.stdout
    assert "PASS file_size_bytes 100.0 <= 500.0" in result.stdout


def test_gates_skip_missing_inputs_and_emit_json(tmp_path: Path) -> None:
    validate_json = tmp_path / "validate.json"
    validate_json.write_text(json.dumps({"command": "validate", "stats": {}}), encoding="utf-8")

    result = _run_gates(["--validate-json", str(validate_json), "--json"])

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["overall"] == "PASS"
    assert payload["evaluated"] == 0
    assert all(gate["status"] == "SKIP" for gate in payload["gates"])


def test_gates_check_output_file_size(tmp_path: Path) -> None:
    validate_json = _write_validate_payload(tmp_path / "validate.json")
    output_file = tmp_path / "output.glb"
    output_file.write_bytes(b"\0" * 2048)

    result = _run_gates(
        [
            "--validate-json",
            str(validate_json),
            "--output",
            str(output_file),
            "--max-file-size-mb",
            "0.001",
        ]
    )

    assert result.returncode == 1
    assert "FAIL file_size_bytes 2048.0" in result.stdout
