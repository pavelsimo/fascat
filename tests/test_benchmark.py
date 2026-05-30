from __future__ import annotations

from pathlib import Path

import numpy as np

import fascat.benchmark as benchmark
from fascat.asset import Asset, Node, Part
from fascat.benchmark import BenchmarkOptions, run_benchmarks
from fascat.mesh import Mesh


def _asset_with_timed_report() -> Asset:
    asset = Asset(
        root=Node(id="root", name="root", children=[Node(id="node", name="node", part_id="part")]),
        parts={
            "part": Part(
                id="part",
                name="Part",
                mesh=Mesh(
                    points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float),
                    faces=np.array([[0, 1, 2]], dtype=int),
                ),
            )
        },
    )
    asset.report.add_step("import", options={}, before={}, after={}, duration=0.25)
    asset.report.add_step("write", options={}, before={}, after={}, duration=0.5)
    return asset


def test_run_benchmarks_records_stage_durations(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    calls: list[tuple[Path, Path, str, bool]] = []

    def fake_convert(
        input_path: Path,
        output_path: Path,
        *,
        profile: str,
        validate_output: bool,
    ) -> Asset:
        calls.append((input_path, output_path, profile, validate_output))
        return _asset_with_timed_report()

    monkeypatch.setattr(benchmark, "convert", fake_convert)

    report = run_benchmarks(
        BenchmarkOptions(
            inputs=(Path("input.step"),),
            output_dir=tmp_path,
            output_suffix=".usdc",
            profile="inspect-only",
            validate_output=True,
        )
    )

    payload = report.to_dict()
    assert calls == [(Path("input.step"), tmp_path / "input.usdc", "inspect-only", True)]
    assert payload["runs"][0]["stages"] == {"import": 0.25, "write": 0.5}
    assert payload["summary"]["stages"] == {"import": 0.25, "write": 0.5}


def test_benchmark_options_validate_inputs(tmp_path: Path) -> None:
    try:
        BenchmarkOptions(inputs=(), output_dir=tmp_path)
    except ValueError as exc:
        assert "inputs" in str(exc)
    else:
        raise AssertionError("empty benchmark inputs should fail")
