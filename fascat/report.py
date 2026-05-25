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

    def copy(self) -> Report:
        return Report(
            source_path=self.source_path,
            started_at=self.started_at,
            finished_at=self.finished_at,
            steps=[
                ReportStep(
                    name=step.name,
                    options=dict(step.options),
                    duration=step.duration,
                    before=dict(step.before),
                    after=dict(step.after),
                    warnings=list(step.warnings),
                )
                for step in self.steps
            ],
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
