from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fascat.pipeline import convert


@dataclass(frozen=True)
class BenchmarkOptions:
    inputs: tuple[Path, ...]
    output_dir: Path
    output_suffix: str = ".glb"
    profile: str = "realtime-desktop"
    repeat: int = 1
    validate_output: bool = False

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("benchmark inputs must not be empty")
        if self.repeat <= 0:
            raise ValueError("benchmark repeat must be greater than 0")
        if not self.output_suffix.startswith("."):
            raise ValueError("benchmark output_suffix must start with '.'")


@dataclass(frozen=True)
class BenchmarkRun:
    input_path: Path
    output_path: Path
    iteration: int
    wall_seconds: float
    peak_rss_mb: float | None
    stages: dict[str, float]
    stats: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "input": str(self.input_path),
            "output": str(self.output_path),
            "iteration": self.iteration,
            "wall_seconds": self.wall_seconds,
            "peak_rss_mb": self.peak_rss_mb,
            "stages": dict(self.stages),
            "stats": dict(self.stats),
        }


@dataclass(frozen=True)
class BenchmarkReport:
    profile: str
    output_suffix: str
    repeat: int
    validate_output: bool
    runs: tuple[BenchmarkRun, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "output_suffix": self.output_suffix,
            "repeat": self.repeat,
            "validate_output": self.validate_output,
            "runs": [run.to_dict() for run in self.runs],
            "summary": _summary(self.runs),
        }


def run_benchmarks(options: BenchmarkOptions) -> BenchmarkReport:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[BenchmarkRun] = []
    for input_path in options.inputs:
        for iteration in range(1, options.repeat + 1):
            output_path = _output_path(input_path, options.output_dir, options.output_suffix, iteration, options.repeat)
            rss_before = _peak_rss_mb()
            started = time.perf_counter()
            asset = convert(
                input_path,
                output_path,
                profile=options.profile,
                validate_output=options.validate_output,
            )
            wall_seconds = time.perf_counter() - started
            rss_after = _peak_rss_mb()
            runs.append(
                BenchmarkRun(
                    input_path=input_path,
                    output_path=output_path,
                    iteration=iteration,
                    wall_seconds=wall_seconds,
                    peak_rss_mb=_max_optional(rss_before, rss_after),
                    stages=_stage_durations(asset.report.to_dict()),
                    stats=asset.stats(include_lods=any(part.lod_meshes for part in asset.parts.values())),
                )
            )
    return BenchmarkReport(
        profile=options.profile,
        output_suffix=options.output_suffix,
        repeat=options.repeat,
        validate_output=options.validate_output,
        runs=tuple(runs),
    )


def _stage_durations(report: dict[str, Any]) -> dict[str, float]:
    stages: dict[str, float] = {}
    for step in report.get("steps", []):
        if not isinstance(step, dict):
            continue
        name = step.get("name")
        duration = step.get("duration")
        if isinstance(name, str) and isinstance(duration, int | float):
            stages[name] = stages.get(name, 0.0) + float(duration)
    return stages


def _summary(runs: tuple[BenchmarkRun, ...]) -> dict[str, object]:
    if not runs:
        return {"total_wall_seconds": 0.0, "max_peak_rss_mb": None, "stages": {}}
    stage_totals: dict[str, float] = {}
    for run in runs:
        for name, duration in run.stages.items():
            stage_totals[name] = stage_totals.get(name, 0.0) + duration
    peak_values = [run.peak_rss_mb for run in runs if run.peak_rss_mb is not None]
    return {
        "total_wall_seconds": sum(run.wall_seconds for run in runs),
        "max_peak_rss_mb": max(peak_values) if peak_values else None,
        "stages": stage_totals,
    }


def _output_path(input_path: Path, output_dir: Path, suffix: str, iteration: int, repeat: int) -> Path:
    stem = input_path.stem if repeat == 1 else f"{input_path.stem}-{iteration}"
    return output_dir / f"{stem}{suffix}"


def _peak_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = float(usage.ru_maxrss)
    if value <= 0.0:
        return None
    # Linux reports KiB, macOS reports bytes.
    return value / (1024.0 if value < 10_000_000 else 1024.0 * 1024.0)


def _max_optional(left: float | None, right: float | None) -> float | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)
