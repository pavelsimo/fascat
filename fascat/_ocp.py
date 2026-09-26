"""OCP/OCCT helpers: shape fingerprints and native message plumbing."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from io import BytesIO
from typing import Any

_MESSAGE_LOCK = threading.Lock()
_MESSAGE_PRINTER: Any | None = None


def configure_occt_messages() -> None:
    """Route OCCT's own diagnostics to stderr instead of stdout.

    The OCCT native layer prints parser errors and status lines through its
    default ``Message_PrinterOStream``, which writes to ``std::cout`` with
    hardcoded ANSI color. That bypasses fascat's ``out``/``err`` consoles and
    breaks the CLI contract: it corrupts ``--json`` payloads and binary
    ``-`` (stdout) streams, and ignores ``NO_COLOR``. Registering our own
    printer bound to ``std::cerr`` puts those messages on the diagnostic
    channel, where every other fascat error already goes.

    Safe to call repeatedly and from any OCCT entry point; the printer is
    installed once and only its color setting is refreshed afterwards.
    """
    global _MESSAGE_PRINTER
    try:
        from OCP.Message import Message, Message_Gravity, Message_PrinterOStream
    except Exception:  # pragma: no cover - OCP is an optional native dependency
        return
    colorize = occt_color_enabled()
    with _MESSAGE_LOCK:
        if _MESSAGE_PRINTER is not None:
            _MESSAGE_PRINTER.SetToColorize(colorize)
            return
        try:
            messenger = Message.DefaultMessenger_s()
            # OCCT maps the pseudo-filename "cerr" onto std::cerr.
            printer = Message_PrinterOStream("cerr", False, Message_Gravity.Message_Info)
            printer.SetToColorize(colorize)
            messenger.RemovePrinters(Message_PrinterOStream.get_type_descriptor_s())
            messenger.AddPrinter(printer)
        except Exception:  # pragma: no cover - never let plumbing break a conversion
            return
        # Keep a Python reference so the printer outlives this call.
        _MESSAGE_PRINTER = printer


def occt_color_enabled() -> bool:
    """Return whether OCCT diagnostics may use ANSI color on stderr."""
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        return False


def shape_fingerprint(shape: Any) -> str:
    brep_fingerprint = _brep_fingerprint(shape)
    if brep_fingerprint is not None:
        return brep_fingerprint

    hash_code = getattr(shape, "HashCode", None)
    if callable(hash_code):
        try:
            return str(hash_code(2_147_483_647))
        except Exception:
            pass
    try:
        return str(hash(shape))
    except Exception:
        return str(id(shape))


def _brep_fingerprint(shape: Any) -> str | None:
    try:
        from OCP.BRepTools import BRepTools
        from OCP.TopTools import TopTools_FormatVersion

        stream = BytesIO()
        BRepTools.Write_s(
            shape,
            stream,
            False,
            False,
            TopTools_FormatVersion.TopTools_FormatVersion_VERSION_1,
        )
    except Exception:
        return None
    data = stream.getvalue()
    if not data:
        return None
    return hashlib.sha1(data).hexdigest()
