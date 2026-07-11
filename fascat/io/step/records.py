from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from fascat.io._import_base import CadHeaderInfo

_MAX_STEP_SCAN_BYTES = 64 * 1024 * 1024


_STEP_RECORD_START_RE = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\(", re.IGNORECASE)


_STEP_REFERENCE_RE = re.compile(r"#(\d+)")


_STEP_NUMBER_RE = re.compile(r"(?<![#A-Za-z0-9_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


@dataclass(frozen=True)
class _StepRecord:
    number: int
    entity: str
    args: str


def _step_scan_capped(source: Path) -> bool:
    try:
        return source.stat().st_size > _MAX_STEP_SCAN_BYTES
    except OSError:
        return False


def _read_step_scan_text(source: Path) -> str | None:
    """Read a STEP file for an auxiliary text pass; None when over the scan cap."""
    if _step_scan_capped(source):
        return None
    return source.read_text(encoding="utf-8", errors="ignore")


def _ensure_loadable_file_size(path: Path, limit: int, label: str) -> None:
    try:
        size = path.stat().st_size
    except OSError:
        return
    if size > limit:
        raise ValueError(f"{label} is too large: {path} ({size} bytes exceeds {limit} byte limit)")


def _step_header_info(source: Path) -> CadHeaderInfo:
    with source.open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(131_072)
    header = text.split("ENDSEC;", 1)[0]
    schema_match = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'", header, flags=re.IGNORECASE | re.DOTALL)
    schema = " ".join(schema_match.group(1).split()) if schema_match else ""
    upper_header = header.upper()
    pmi_present = "AP242" in schema.upper() and (
        "PRODUCT MANUFACTURING INFORMATION" in upper_header or "PMI" in upper_header
    )
    return CadHeaderInfo(schema=schema, pmi_present=pmi_present)


def _iter_step_records(text: str) -> list[_StepRecord]:
    records: list[_StepRecord] = []
    position = 0
    while match := _STEP_RECORD_START_RE.search(text, position):
        args_start = match.end()
        args_end = _find_step_record_args_end(text, args_start - 1)
        if args_end is None:
            position = match.end()
            continue
        records.append(
            _StepRecord(
                number=int(match.group(1)),
                entity=match.group(2).upper(),
                args=text[args_start:args_end],
            )
        )
        position = args_end + 1
    return records


_MAX_STEP_RECORD_ARGS_BYTES = 1_048_576


def _find_step_record_args_end(
    text: str, open_paren_index: int, *, max_scan: int = _MAX_STEP_RECORD_ARGS_BYTES
) -> int | None:
    depth = 0
    in_string = False
    index = open_paren_index
    limit = min(len(text), open_paren_index + max_scan)
    while index < limit:
        char = text[index]
        if in_string:
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    index += 2
                    continue
                in_string = False
            index += 1
            continue
        if char == "'":
            in_string = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
        index += 1
    return None


_ASCII_WHITESPACE_RE = re.compile(r"[ \t\r\n\f\v]+")


_STEP_CODEPAGES = {
    "A": "iso8859-1",
    "B": "iso8859-2",
    "C": "iso8859-3",
    "D": "iso8859-4",
    "E": "iso8859-5",
    "F": "iso8859-6",
    "G": "iso8859-7",
    "H": "iso8859-8",
    "I": "iso8859-9",
}


def _decode_step_hex_groups(digits: str, group_size: int) -> str | None:
    if not digits or len(digits) % group_size != 0:
        return None
    try:
        if group_size == 4:
            return bytes.fromhex(digits).decode("utf-16-be")
        characters: list[str] = []
        for start in range(0, len(digits), 8):
            code_point = int(digits[start : start + 8], 16)
            if code_point > 0x10FFFF or 0xD800 <= code_point <= 0xDFFF:
                return None
            characters.append(chr(code_point))
        return "".join(characters)
    except ValueError:
        return None


def _decode_step_string(value: str) -> str:
    """Decode ISO 10303-21 string control directives.

    Handles ``\\\\``, ``\\S\\c`` (codepage high half), ``\\P{A-I}\\`` (codepage
    selection), ``\\X\\HH`` (Latin-1), ``\\X2\\…\\X0\\`` (UTF-16BE groups), and
    ``\\X4\\…\\X0\\`` (UCS-4 groups). Malformed or incomplete directives stay
    literal — untrusted input must never raise here.
    """
    if "\\" not in value:
        return value
    result: list[str] = []
    codepage = _STEP_CODEPAGES["A"]
    index = 0
    length = len(value)
    while index < length:
        char = value[index]
        if char != "\\":
            result.append(char)
            index += 1
            continue
        if value.startswith("\\\\", index):
            result.append("\\")
            index += 2
            continue
        if value.startswith("\\S\\", index) and index + 3 < length:
            target = value[index + 3]
            if ord(target) < 0x80:
                try:
                    result.append(bytes([ord(target) + 0x80]).decode(codepage))
                    index += 4
                    continue
                except UnicodeDecodeError:
                    pass
        if value.startswith("\\P", index) and index + 3 < length and value[index + 3] == "\\":
            mapped = _STEP_CODEPAGES.get(value[index + 2].upper())
            if mapped is not None:
                codepage = mapped
                index += 4
                continue
        if value.startswith(("\\X2\\", "\\X4\\"), index):
            group_size = 4 if value[index + 2] == "2" else 8
            terminator = value.find("\\X0\\", index + 4)
            if terminator != -1:
                decoded = _decode_step_hex_groups(value[index + 4 : terminator], group_size)
                if decoded is not None:
                    result.append(decoded)
                    index = terminator + 4
                    continue
        if value.startswith("\\X\\", index) and index + 5 <= length:
            try:
                result.append(chr(int(value[index + 3 : index + 5], 16)))
                index += 5
                continue
            except ValueError:
                pass
        result.append(char)
        index += 1
    return "".join(result)


def _step_string_values(text: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != "'":
            index += 1
            continue
        index += 1
        value: list[str] = []
        while index < len(text):
            char = text[index]
            if char == "'":
                if index + 1 < len(text) and text[index + 1] == "'":
                    value.append("'")
                    index += 2
                    continue
                index += 1
                break
            value.append(char)
            index += 1
        # Collapse only ASCII whitespace: decoded directives can produce
        # meaningful Unicode whitespace (e.g. \S\<space> -> NBSP) that must survive.
        cleaned = _ASCII_WHITESPACE_RE.sub(" ", _decode_step_string("".join(value))).strip()
        if cleaned:
            values.append(cleaned)
    return values


def _step_number_values(text: str) -> list[float]:
    unquoted = _strip_step_strings(text)
    return [float(match.group(0)) for match in _STEP_NUMBER_RE.finditer(unquoted)]


def _strip_step_strings(text: str) -> str:
    chars = list(text)
    index = 0
    while index < len(chars):
        if chars[index] != "'":
            index += 1
            continue
        chars[index] = " "
        index += 1
        while index < len(chars):
            char = chars[index]
            chars[index] = " "
            if char == "'":
                if index + 1 < len(chars) and chars[index + 1] == "'":
                    chars[index + 1] = " "
                    index += 2
                    continue
                index += 1
                break
            index += 1
    return "".join(chars)


def _name_tokens(value: str) -> list[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]
