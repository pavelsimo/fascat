from __future__ import annotations

import logging

from fascat.report import Report


def test_report_warnings_emit_library_log_records(caplog) -> None:  # type: ignore[no-untyped-def]
    report = Report()

    with caplog.at_level(logging.WARNING, logger="fascat"):
        report.add_warning("careful now")

    assert report.warnings == ["careful now"]
    assert [record.message for record in caplog.records] == ["careful now"]
