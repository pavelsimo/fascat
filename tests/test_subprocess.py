from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from fascat._subprocess import run_guarded


def test_run_guarded_returns_completed_process() -> None:
    completed = run_guarded([sys.executable, "-c", "print('ok')"], timeout=30)

    assert completed.returncode == 0
    assert completed.stdout.strip() == "ok"
    assert completed.stderr == ""


def test_run_guarded_captures_failure_output() -> None:
    completed = run_guarded(
        [sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"],
        timeout=30,
    )

    assert completed.returncode == 3
    assert "boom" in completed.stderr


def test_run_guarded_raises_timeout_expired() -> None:
    with pytest.raises(subprocess.TimeoutExpired):
        run_guarded([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)


def test_run_guarded_passes_cwd(tmp_path: Path) -> None:
    completed = run_guarded(
        [sys.executable, "-c", "import os; print(os.getcwd())"],
        timeout=30,
        cwd=tmp_path,
    )

    assert Path(completed.stdout.strip()).resolve() == tmp_path.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process-group semantics")
def test_run_guarded_kills_whole_process_group_on_timeout(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    child_script = (
        "import subprocess, sys, time\n"
        "grandchild = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(grandchild.pid))\n"
        "time.sleep(60)\n"
    )

    with pytest.raises(subprocess.TimeoutExpired):
        run_guarded([sys.executable, "-c", child_script], timeout=2.0)

    assert pid_file.exists(), "child never started its grandchild"
    grandchild_pid = int(pid_file.read_text())
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    pytest.fail(f"grandchild {grandchild_pid} survived the process-group kill")
