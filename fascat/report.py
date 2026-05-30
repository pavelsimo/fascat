from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReportStep:
    name: str
    options: dict[str, object] = field(default_factory=dict)
    duration: float = 0.0
    before: dict[str, int] = field(default_factory=dict)
    after: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.options = dict(self.options)
        self.before = dict(self.before)
        self.after = dict(self.after)
        self.warnings = list(self.warnings)

    @classmethod
    def _adopt(
        cls,
        *,
        name: str,
        options: dict[str, object],
        duration: float,
        before: dict[str, int],
        after: dict[str, int],
        warnings: list[str],
    ) -> ReportStep:
        step = object.__new__(cls)
        step.name = name
        step.options = options
        step.duration = duration
        step.before = before
        step.after = after
        step.warnings = warnings
        return step

    def copy(self) -> ReportStep:
        return ReportStep._adopt(
            name=self.name,
            options=dict(self.options),
            duration=self.duration,
            before=dict(self.before),
            after=dict(self.after),
            warnings=list(self.warnings),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "options": dict(self.options),
            "duration": self.duration,
            "before": dict(self.before),
            "after": dict(self.after),
            "warnings": list(self.warnings),
        }


@dataclass
class Report:
    source_path: str | None = None
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None
    steps: list[ReportStep] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    input_stats: dict[str, int] = field(default_factory=dict)
    output_stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = [step.copy() for step in self.steps]
        self.warnings = list(self.warnings)
        self.errors = list(self.errors)
        self.input_stats = dict(self.input_stats)
        self.output_stats = dict(self.output_stats)

    @classmethod
    def _adopt(
        cls,
        *,
        source_path: str | None,
        started_at: str,
        finished_at: str | None,
        steps: list[ReportStep],
        warnings: list[str],
        errors: list[str],
        input_stats: dict[str, int],
        output_stats: dict[str, int],
    ) -> Report:
        report = object.__new__(cls)
        report.source_path = source_path
        report.started_at = started_at
        report.finished_at = finished_at
        report.steps = steps
        report.warnings = warnings
        report.errors = errors
        report.input_stats = input_stats
        report.output_stats = output_stats
        return report

    def copy(self) -> Report:
        return Report._adopt(
            source_path=self.source_path,
            started_at=self.started_at,
            finished_at=self.finished_at,
            steps=[step.copy() for step in self.steps],
            warnings=list(self.warnings),
            errors=list(self.errors),
            input_stats=dict(self.input_stats),
            output_stats=dict(self.output_stats),
        )

    def finish(self, output_stats: dict[str, int] | None = None) -> None:
        self.finished_at = _now()
        if output_stats is not None:
            self.output_stats = dict(output_stats)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_step(
        self,
        name: str,
        *,
        options: dict[str, object] | None = None,
        before: dict[str, int] | None = None,
        after: dict[str, int] | None = None,
        duration: float = 0.0,
        warnings: list[str] | None = None,
    ) -> None:
        self.steps.append(
            ReportStep(
                name=name,
                options=options or {},
                before=before or {},
                after=after or {},
                duration=duration,
                warnings=warnings or [],
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [step.to_dict() for step in self.steps],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "input_stats": dict(self.input_stats),
            "output_stats": dict(self.output_stats),
        }

    def summary(self) -> str:
        stats = self.output_stats or self.input_stats
        parts = stats.get("parts", 0)
        triangles = stats.get("triangles", 0)
        materials = stats.get("materials", 0)
        return f"{parts} parts, {triangles} triangles, {materials} materials, {len(self.warnings)} warnings"

    def write_json(self, path: str | Path) -> None:
        import json

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


class timed_step:
    def __enter__(self) -> timed_step:
        self.started = perf_counter()
        self.duration = 0.0
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.duration = perf_counter() - self.started
