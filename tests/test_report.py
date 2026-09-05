from __future__ import annotations

import logging
from typing import cast

from fascat.report import Report, ReportStep


def test_report_warnings_emit_library_log_records(caplog) -> None:  # type: ignore[no-untyped-def]
    report = Report()

    with caplog.at_level(logging.WARNING, logger="fascat"):
        report.add_warning("careful now")

    assert report.warnings == ["careful now"]
    assert [record.message for record in caplog.records] == ["careful now"]


def test_report_options_are_owned_by_construction_and_copy() -> None:
    options = {"nested": {"values": [1]}}
    step = ReportStep("stage", options=options)
    report = Report(steps=[step])
    options["nested"]["values"].append(2)
    assert step.options == {"nested": {"values": [1]}}
    cast(dict[str, list[int]], step.options["nested"])["values"].append(3)
    assert report.steps[0].options == {"nested": {"values": [1]}}

    copied = report.copy()
    cast(dict[str, list[int]], copied.steps[0].options["nested"])["values"].append(4)
    assert report.steps[0].options == {"nested": {"values": [1]}}
    assert copied.steps[0].options == {"nested": {"values": [1, 4]}}
