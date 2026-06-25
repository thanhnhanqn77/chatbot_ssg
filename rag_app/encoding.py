from __future__ import annotations

import re
from typing import Any


_CP1252_REVERSE: dict[str, int] = {}
for _byte in range(256):
    try:
        _CP1252_REVERSE[bytes([_byte]).decode("cp1252")] = _byte
    except UnicodeDecodeError:
        _CP1252_REVERSE[chr(_byte)] = _byte

_MOJIBAKE_MARKERS = set("ÃÂÄÅÆâ€œ€")
_VIETNAMESE_RE = re.compile(
    r"[ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúý"
    r"ĂăĐđĨĩŨũƠơƯư"
    r"Ạ-ỹ]"
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def _badness(text: str) -> int:
    marker_count = sum(1 for ch in text if ch in _MOJIBAKE_MARKERS)
    control_count = len(_CONTROL_RE.findall(text))
    replacement_count = text.count("�")
    return marker_count * 2 + control_count * 4 + replacement_count * 8


def _goodness(text: str) -> int:
    return len(_VIETNAMESE_RE.findall(text)) * 3 - _badness(text)


def _to_original_utf8_bytes(text: str) -> bytes | None:
    out = bytearray()
    for ch in text:
        codepoint = ord(ch)
        if codepoint <= 0xFF:
            out.append(codepoint)
            continue
        byte = _CP1252_REVERSE.get(ch)
        if byte is None:
            return None
        out.append(byte)
    return bytes(out)


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text that was decoded as latin-1/cp1252.

    Clean Vietnamese strings are left untouched because they cannot be encoded
    back into the mojibake byte range.
    """

    if not text or not any(ch in _MOJIBAKE_MARKERS or "\x7f" <= ch <= "\x9f" for ch in text):
        return text

    best = text
    best_score = _goodness(text)

    current = text
    for _ in range(2):
        raw = _to_original_utf8_bytes(current)
        if raw is None:
            break
        try:
            candidate = raw.decode("utf-8")
        except UnicodeDecodeError:
            break
        candidate_score = _goodness(candidate)
        if candidate_score <= best_score:
            break
        best = candidate
        best_score = candidate_score
        current = candidate

    return best


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = repair_mojibake(str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    lines = [re.sub(r"[ ]{2,}", " ", line).strip() for line in text.split("\n")]

    cleaned: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned.append("")
            previous_blank = True
            continue
        cleaned.append(line)
        previous_blank = False

    return "\n".join(cleaned).strip()


def clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    return value
