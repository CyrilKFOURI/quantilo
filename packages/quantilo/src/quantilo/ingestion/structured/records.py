"""Where the records are in a document (R-SST-01 to 07) and what surrounds them (`_meta`)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from quantilo.ingestion.structured.errors import StructuredError
from quantilo.ingestion.structured.model import Format, RecordsCandidate
from quantilo.ingestion.structured.profile import is_value_map

OBJECT_SHARE = 0.9  # R-SST-02: a list of objects
JSON_DEPTH = 3
XML_DEPTH = 4  # the root element adds one level


@dataclass
class Records:
    keys: tuple[str, ...]  # path of the records from the root ("" = the root)
    items: list[Any]  # the records, each a dict
    rule: str
    reason: str
    is_list: bool  # the records come from a list (the locator ends with [*])
    candidates: list[RecordsCandidate] = field(default_factory=list)
    meta: dict[str, Any] | None = None  # R-SST-05
    locator: str = "$"  # JSONPath of one record ("$.results[*]")


def as_record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {"value": value}


def dotted(keys: tuple[str, ...]) -> str:
    return ".".join(keys)


def objects_share(items: list[Any]) -> float:
    present = [item for item in items if item is not None]
    if not present:
        return 0.0
    return sum(isinstance(item, dict) for item in present) / len(present)


def find_candidates(root: dict[str, Any], depth: int) -> list[tuple[tuple[str, ...], int]]:
    """Lists of objects reached through objects only, at most `depth` keys from the root."""
    found: list[tuple[tuple[str, ...], int]] = []

    def visit(node: dict[str, Any], path: tuple[str, ...]) -> None:
        for key, value in node.items():
            child = (*path, key)
            if isinstance(value, list):
                if value and objects_share(value) >= OBJECT_SHARE:
                    found.append((child, len(value)))
            elif isinstance(value, dict) and len(child) < depth:
                visit(value, child)

    visit(root, ())
    # the longest first; at equal length the shallowest, then the first seen (stable sort)
    return sorted(found, key=lambda item: (-item[1], len(item[0])))


def resolve(root: Any, path: str) -> tuple[tuple[str, ...], Any]:
    """A dotted path whose keys may themselves hold dots ("1. open"): the longest key wins."""
    keys: list[str] = []
    node = root
    rest = path
    while rest:
        if not isinstance(node, dict):
            break
        match = None
        for key in sorted(node, key=len, reverse=True):
            if rest == key or rest.startswith(key + "."):
                match = key
                break
        if match is None:
            break
        keys.append(match)
        node = node[match]
        rest = rest[len(match) + 1 :]
    if rest:
        raise StructuredError(f"No records at {path!r}: this path is not in the document.")
    return tuple(keys), node


def without(node: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """The document with the records list taken out (what surrounds it: R-SST-05)."""
    copy = dict(node)
    head, rest = keys[0], keys[1:]
    if rest and isinstance(copy.get(head), dict):
        inner = without(copy[head], rest)
        if inner:
            copy[head] = inner
        else:
            del copy[head]
    else:
        copy.pop(head, None)
    return copy


SIMPLE_KEY = re.compile(r"[A-Za-z_@#][\w@#-]*")


def locator_key(key: str) -> str:
    """One step of a JSONPath: `.name`, or `['odd name']`."""
    if SIMPLE_KEY.fullmatch(key):
        return "." + key
    return "['" + key.replace("\\", "\\\\").replace("'", "\\'") + "']"


def from_value(keys: tuple[str, ...], value: Any, rule: str, reason: str) -> Records:
    base = "$" + "".join(locator_key(key) for key in keys)
    if isinstance(value, list):
        items = [as_record(item) for item in value]
        return Records(keys, items, rule, reason, True, locator=base + "[*]")
    return Records(keys, [as_record(value)], rule, reason, False, locator=base)


def select_records(root: Any, fmt: Format, records_path: str | None) -> Records:
    if records_path is not None:
        keys, value = resolve(root, records_path)
        records = from_value(keys, value, "R-OVR", "records path chosen by you")
    elif isinstance(root, list):
        records = from_value((), root, "R-SST-01", "the document is a list: one row per element")
    elif isinstance(root, dict):
        records = select_in_object(root, fmt)
    else:
        records = Records(
            (), [{"value": root}], "R-SST-07", "the document is a single value", False
        )
    if records.keys and isinstance(root, dict):
        meta = without(root, records.keys)
        records.meta = meta or None
    return records


def select_in_object(root: dict[str, Any], fmt: Format) -> Records:
    found = find_candidates(root, XML_DEPTH if fmt == "xml" else JSON_DEPTH)
    candidates = [RecordsCandidate(dotted(path), count) for path, count in found]
    if found:
        keys, count = found[0]
        value = root
        for key in keys:
            value = value[key]
        if fmt == "xml":
            rule = "R-SST-03"
            reason = f"<{keys[-1]}> is the element repeated most under one parent ({count:,} times)"
        else:
            rule = "R-SST-02"
            reason = f"{dotted(keys)} is the longest list of objects ({count:,} elements)"
        records = from_value(keys, value, rule, reason)
        records.candidates = candidates
        return records
    # no list of objects: a configuration (one record), or a dictionary of values
    if fmt != "xml":
        why = is_value_map(root)
        if why:
            items = [{"key": key, "value": value} for key, value in root.items()]
            reason = f"the document is a dictionary ({why}): one row per key"
            return Records((), items, "R-SST-07", reason, True, locator="$.*")
    return Records((), [root], "R-SST-07", "no list of objects: the document is one record", False)
