"""Hardened subprocess execution for external tools and runtime harnesses.

``subprocess.run(..., timeout=...)`` kills only the direct child on timeout;
browsers, engines, and node tools spawn process trees whose children survive
and can keep reading from temporary directories after cleanup. ``run_guarded``
launches every child in its own process group (session on POSIX) and kills the
whole group before the timeout propagates, so by the time callers see
``subprocess.TimeoutExpired`` no descendant is still running.

Residual limitation: a grandchild that creates its own session escapes the
group kill. Chromium, Unity/Unreal harnesses, and node do not do this.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

GLTF_TRANSFORM_TIMEOUT_SECONDS = 300.0
KTX2_ENCODE_TIMEOUT_SECONDS = 600.0


def run_guarded(
    command: Sequence[str],
    *,
    timeout: float | None,
    cwd: str | Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Drop-in for ``subprocess.run(..., check=False, capture_output=True, text=True, timeout=...)``.

    On timeout the child's entire process group is killed and reaped before
    ``subprocess.TimeoutExpired`` is raised.
    """
    if sys.platform == "win32":
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            start_new_session=True,
        )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_group(process)
        # Drain pipes and reap the child so no zombie or open handle outlives
        # the caller's temporary directories.
        drained_stdout: str | None = None
        drained_stderr: str | None = None
        with suppress(Exception):
            drained_stdout, drained_stderr = process.communicate()
        # Whatever the tool managed to print before the timeout is usually the
        # only diagnostic the caller gets, so carry it onto the re-raise.
        raise subprocess.TimeoutExpired(
            list(command),
            timeout or 0.0,
            output=_captured(drained_stdout, exc.stdout),
            stderr=_captured(drained_stderr, exc.stderr),
        ) from None
    except BaseException:
        _kill_process_group(process)
        with suppress(Exception):
            process.communicate()
        raise
    return subprocess.CompletedProcess(list(command), process.returncode, stdout, stderr)


def _captured(drained: str | None, partial: object) -> str | None:
    """Prefer the fully drained stream, falling back to the timeout's partial capture."""
    if drained:
        return drained
    if isinstance(partial, bytes):
        return partial.decode("utf-8", "replace")
    if isinstance(partial, str):
        return partial
    return drained


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    if sys.platform == "win32":
        # taskkill /T walks and terminates the whole tree.
        with suppress(Exception):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                timeout=10,
                check=False,
            )
        with suppress(Exception):
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        with suppress(Exception):
            process.kill()
