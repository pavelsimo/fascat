import json
from pathlib import Path

import pytest

from fascat.cli import app

from ._cli_test_helpers import (
    runner,
)


def test_runtime_fixtures_command_writes_suite(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    result = runner.invoke(app, ["--json", "runtime-fixtures", str(suite_dir)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "runtime-fixtures"
    assert payload["suite"]["targets"] == ["browser", "unity", "unreal"]
    assert len(payload["suite"]["fixtures"]) == 6
    assert (suite_dir / "runtime-parity-suite.json").is_file()
    assert (suite_dir / "assets" / "pbr-material-grid.glb").is_file()
    assert (suite_dir / "assets" / "ktx2-basis-fallback.glb").is_file()
    assert (suite_dir / "assets" / "lod-profile-unity.glb").is_file()
    assert (suite_dir / "assets" / "lod-profile-unreal.glb").is_file()
    assert (suite_dir / "baselines" / "pbr-material-grid.png").is_file()


def test_runtime_fixtures_dry_run_does_not_write(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    result = runner.invoke(app, ["--json", "--dry-run", "runtime-fixtures", str(suite_dir)])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["command"] == "runtime-fixtures"
    assert payload["dry_run"] is True
    assert not suite_dir.exists()


def test_runtime_fixtures_can_capture_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "runtime-parity"

    class FakeCaptureReport:
        captures = (object(), object())
        passed = True

        def to_dict(self) -> dict[str, object]:
            return {"passed": True, "captures": [{"target": "browser"}, {"target": "unity"}]}

    def fake_capture(directory: str | Path, **kwargs: object) -> FakeCaptureReport:
        assert Path(directory) == suite_dir
        assert kwargs["targets"] == ("browser", "unity")
        assert kwargs["browser_command"] == "Chrome"
        assert kwargs["unity_command"] == "Unity"
        assert kwargs["promote_goldens"] is True
        assert kwargs["require_goldens"] is True
        return FakeCaptureReport()

    monkeypatch.setattr("fascat.runtime_fixtures.capture_runtime_parity_suite", fake_capture)

    result = runner.invoke(
        app,
        [
            "--json",
            "runtime-fixtures",
            str(suite_dir),
            "--capture",
            "browser",
            "--capture",
            "unity",
            "--runtime-browser-command",
            "Chrome",
            "--unity-command",
            "Unity",
            "--promote-goldens",
            "--require-goldens",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["capture"] == ["browser", "unity"]
    assert payload["captures"]["passed"] is True
    assert len(payload["captures"]["captures"]) == 2


def test_runtime_fixtures_can_require_capture_goldens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    suite_dir = tmp_path / "runtime-parity"

    class FakeCaptureReport:
        captures = (object(),)
        passed = False

        def to_dict(self) -> dict[str, object]:
            return {
                "passed": False,
                "captures": [{"target": "browser", "status": "missing_golden"}],
            }

    def fake_capture(directory: str | Path, **kwargs: object) -> FakeCaptureReport:
        assert Path(directory) == suite_dir
        assert kwargs["targets"] == ("browser",)
        assert kwargs["require_goldens"] is True
        return FakeCaptureReport()

    monkeypatch.setattr("fascat.runtime_fixtures.capture_runtime_parity_suite", fake_capture)

    result = runner.invoke(
        app,
        ["--json", "runtime-fixtures", str(suite_dir), "--capture", "browser", "--require-goldens"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "runtime parity target captures failed required golden validation."
    assert payload["captures"]["passed"] is False


def test_runtime_fixtures_can_check_golden_coverage(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    result = runner.invoke(app, ["--json", "runtime-fixtures", str(suite_dir), "--check-goldens"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["check_goldens"] is True
    assert payload["golden_coverage"]["passed"] is False
    assert payload["golden_coverage"]["missing_count"] == 18
    assert (suite_dir / "runtime-parity-golden-coverage.json").is_file()


def test_runtime_fixtures_can_require_checked_goldens(tmp_path: Path) -> None:
    suite_dir = tmp_path / "runtime-parity"

    result = runner.invoke(
        app,
        ["--json", "runtime-fixtures", str(suite_dir), "--check-goldens", "--require-goldens"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "runtime parity target goldens are missing or invalid."
    assert payload["golden_coverage"]["missing_count"] == 18
