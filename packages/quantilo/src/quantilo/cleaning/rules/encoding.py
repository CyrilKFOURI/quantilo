"""N28 (audit 2026-11-12, § 4.3): wider encodings in B-05.

B-05 (rules/basic.py) maps each mis-encoded value to the one `text.repair_encoding` reads back;
that function now also undoes a value read wrong two or three times, Latin-1's C1 controls and
UTF-8 read as Mac Roman. This module, run right after B-05:

- says in B-05's reason how the values were misread when it is not the usual once as
  Windows-1252 ("read wrong twice: 3 values"), so the user sees which repair is proposed;
- adds B-05:lost, an information: values holding '�' (U+FFFD), a character replaced before the
  file was saved. No reading brings it back, so nothing is changed; the rows are counted.

Every reading is a byte round trip checked value by value (no guess, no model).
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from quantilo.cleaning import text as tx
from quantilo.cleaning.compile import is_text
from quantilo.ingestion.sql import ident, lit

if TYPE_CHECKING:
    from quantilo.cleaning.rules.basic import Context

SHOWN = 3
# How B-05 says each way a value was misread (text.repair_encoding).
WAYS = {
    "windows-1252": "UTF-8 read as Windows-1252",
    "latin-1": "UTF-8 read as Latin-1",
    "double": "read wrong twice or more (double encoding)",
    "mac-roman": "UTF-8 read as Mac Roman",
}
USUAL = "windows-1252"


def _n(count: int) -> str:
    return f"{count:,}"


def ways(mapping: dict[str, str]) -> Counter[str]:
    """How many values of a B-05 map were misread each way."""
    found: Counter[str] = Counter()
    for value in mapping:
        repaired = tx.repair_encoding(value)
        if repaired is not None:
            found[repaired[1]] += 1
    return found


def explain(context: Context) -> None:
    """B-05's reason names the readings it undoes when one is not the usual one."""
    for suggestion in context.found:
        if suggestion.rule != "B-05" or suggestion.step is None:
            continue
        mapping = getattr(suggestion.step, "map", {}) or {}
        counted = ways({str(k): str(v) for k, v in mapping.items()})
        if not counted or set(counted) == {USUAL}:
            continue
        parts = [
            f"{WAYS[way]} ({_n(count)} value{'s' if count > 1 else ''})"
            for way, count in counted.most_common()
        ]
        suggestion.reason = f"{suggestion.reason.rstrip()} Read back from: {'; '.join(parts)}."
        suggestion.reason_code, suggestion.reason_vars = None, {}  # M02: said in English only


def lost(context: Context) -> None:
    """B-05:lost: rows whose value holds '�', counted on the distinct values when they are read,
    else by one SQL count."""
    for column in context.columns:
        if not is_text(column.dtype) or not column.nonnull:
            continue
        stages = context.stages.get(column.name)
        if stages is not None:
            hit = [(raw, count) for raw, count in stages.raw if tx.lost_characters(raw)]
            cells = sum(count for _, count in hit)
            examples = [value for value, _ in hit[:SHOWN]]
        else:
            name = ident(column.name)
            cells = int(
                context.one(
                    f"SELECT count(*) FROM {context.source} WHERE contains({name}, {lit(tx.LOST)})"
                )[0]
            )
            examples = [
                str(row[0])
                for row in context.connection.execute(
                    f"SELECT DISTINCT {name} FROM {context.source} "
                    f"WHERE contains({name}, {lit(tx.LOST)}) ORDER BY 1 LIMIT {SHOWN}"
                ).fetchall()
            ]
        if not cells:
            continue
        noun = "value holds" if cells == 1 else "values hold"
        context.add(
            "B-05",
            "I",
            column,
            f"Characters lost in {column.name}",
            f"{_n(cells)} {noun} '{tx.LOST}', a character lost before the file was saved: it "
            "cannot be repaired. Retype it, or ask for the file again in UTF-8. Check your data.",
            cells=cells,
            examples=[(value, None) for value in examples],
            key="lost",
        )


def b05_more(context: Context) -> None:
    explain(context)
    lost(context)
