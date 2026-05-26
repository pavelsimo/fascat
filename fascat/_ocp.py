from __future__ import annotations

import hashlib
from io import BytesIO
from typing import Any


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
