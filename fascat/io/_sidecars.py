"""Sidecar accounting for exported entry files.

A ``.gltf`` written through the gltf-transform post-processing path keeps its
payload in an external ``.bin`` (plus texture files) published beside the entry
file, so size budgets and gates that stat only the entry would pass on the
strength of a few-KB JSON document.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote


def sidecar_paths(path: str | Path) -> list[Path]:
    """Return the external files an exported entry file references."""
    entry = Path(path)
    if entry.suffix.lower() != ".gltf":
        return []
    try:
        document = json.loads(entry.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(document, dict):
        return []
    parent = entry.parent
    found: dict[Path, None] = {}
    for section in ("buffers", "images"):
        entries = document.get(section)
        if not isinstance(entries, list):
            continue
        for item in entries:
            if not isinstance(item, dict):
                continue
            uri = item.get("uri")
            if not isinstance(uri, str) or not uri or uri.startswith("data:") or "://" in uri:
                continue
            candidate = parent / unquote(uri)
            if candidate.is_file() and candidate != entry:
                found[candidate] = None
    return list(found)


def sidecar_bytes(path: str | Path) -> int:
    """Return the total on-disk size of an entry file's published sidecars."""
    total = 0
    for sidecar in sidecar_paths(path):
        try:
            total += sidecar.stat().st_size
        except OSError:
            continue
    return total
