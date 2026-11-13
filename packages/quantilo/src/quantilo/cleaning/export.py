"""N45: a cleaning written as code an auditor reads without the app (Power Query shows its M).

- `to_sql`: one DuckDB script, the very query `compile.py` builds and `apply.py` writes (one CTE
  per step, the same columns, the rows in the original's order), each step commented with its rule,
  where it came from and what it does. Run with `duckdb -c ".read cleaning.sql"`.
- `to_python`: a script that replays the steps with the `quantilo` library (`dx.open(…).clean(steps)
  .apply()`), as the recipes' scripts do (030), then copies the cleaned Parquet.

Both read the dataset the cleaning was made on (the app's Parquet copy, never written) and write
`cleaned.parquet`; run on that dataset, they give the saved version's rows exactly. Every name and
value reaches the code as a literal (SQL `lit`, Python `repr`), never pasted; comments lose their
line breaks.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import duckdb

from quantilo.cleaning.compile import ROW, compile_steps
from quantilo.cleaning.replay import step_columns
from quantilo.ingestion.sql import ident, lit
from quantilo.recipes.script import comment as one_line
from quantilo.recipes.script import file_stem, literal

# PEP 263: a declaration in the first two lines changes how Python decodes the whole file. The
# script states UTF-8 itself, first, before any text that comes from the data.
ENCODING_LINE = "# -*- coding: utf-8 -*-"
CODING = re.compile(r"(coding)([:=])", re.IGNORECASE)


def comment(text: str) -> str:
    """One comment line (recipes' `comment`) that can never read as an encoding declaration."""
    return CODING.sub(r"\1 \2", one_line(text))


ORIGINS = {
    "auto": "applied automatically, a reliable fix",
    "suggested": "a suggestion ticked by the user",
    "manual": "added by the user",
}
TARGET = "cleaned.parquet"


@dataclass
class ExportInfo:
    """What the header of an exported cleaning says."""

    name: str  # "sales · Cleaned v2"
    made_on: str | None = None  # the dataset it was made on, by name
    # the step's sentence, by step id (a suggestion's title); else its op and columns
    titles: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # extra header lines (out of date…)


def _line(text: str) -> str:
    return comment(text).replace("# ", "-- ", 1)


def step_sentence(step: Any, titles: Mapping[str, str]) -> str:
    """'s3 · B-04 · applied automatically, a reliable fix · Read 'n/a' as missing in status'."""
    columns = step_columns(step)
    what = titles.get(step.id) or (f"{step.op} on {', '.join(columns)}" if columns else step.op)
    parts = [step.id, step.rule or "", ORIGINS.get(step.origin, step.origin), what]
    return " · ".join(part for part in parts if part)


def _counts(steps: Sequence[Any]) -> str:
    on = [s for s in steps if s.enabled]
    by = {key: sum(1 for s in on if s.origin == key) for key in ORIGINS}
    text = (
        f"{len(steps)} steps: {by['auto']} applied automatically, {by['suggested']} ticked "
        f"suggestions, {by['manual']} added by hand"
    )
    off = len(steps) - len(on)
    return f"{text}; {off} turned off (not run)." if off else f"{text}."


def to_sql(
    connection: duckdb.DuckDBPyConnection,
    parquet_path: str,
    steps: Sequence[Any],
    info: ExportInfo,
    sources: Mapping[str, str] | None = None,
    target: str = TARGET,
) -> str:
    """The cleaning as one DuckDB script: `COPY (WITH t0 …, t1 …) TO target`."""
    compiled = compile_steps(connection, parquet_path, list(steps), sources=sources)
    by_step = {stage.step.id: index for index, stage in enumerate(compiled.stages) if stage.step}
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        _line(f"Cleaning “{info.name}”, exported from Quantilo on {stamp}."),
        _line(f"Made on: {info.made_on or 'a dataset'}; it reads the app's copy of it below,"),
        _line("which it never writes, and writes the cleaned rows, in the original's order, to"),
        _line(f"'{target}'. Run it with DuckDB 1.1 or later:"),
        _line(f'    duckdb -c ".read {file_stem(info.name)}.sql"'),
        _line("Change the paths to clean another file with the same columns."),
        _line(_counts(steps)),
        *(_line(note) for note in info.notes),
        "COPY (",
        "WITH",
    ]
    ctes: list[str] = []
    pending = [_line("t0: the original, each row with its number in the file (kept to the end).")]
    ctes.append(f"{chr(10).join(pending)}\n{compiled.ctes[0]}")
    pending = []
    for step in steps:
        sentence = step_sentence(step, info.titles)
        if not step.enabled:
            pending.append(_line(f"{sentence}: turned off, not run."))
            continue
        index = by_step.get(step.id)
        if index is None:
            problem = compiled.problems.get(step.id, "it cannot run")
            pending.append(_line(f"{sentence}: skipped, {problem}"))
            continue
        pending.append(_line(f"t{index}: {sentence}"))
        ctes.append(f"{chr(10).join(pending)}\n{compiled.ctes[index]}")
        pending = []
    lines.append(",\n".join(ctes))
    lines += pending
    final = compiled.final.name
    lines.append(f"SELECT {compiled.output_columns()} FROM {final} ORDER BY {ident(ROW)}")
    lines.append(f") TO {lit(target)} (FORMAT parquet, COMPRESSION zstd);")
    return "\n".join(lines) + "\n"


def to_python(
    parquet_path: str,
    steps: Sequence[Any],
    info: ExportInfo,
    sources: Mapping[str, str] | None = None,
    target: str = TARGET,
) -> str:
    """The cleaning as a Python script on the `quantilo` library (030's way)."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    stem = file_stem(info.name)
    lines = [
        ENCODING_LINE,  # first: no text from the data may declare another encoding
        comment(f"Cleaning “{info.name}”, exported from Quantilo on {stamp}."),
        comment(f"Made on: {info.made_on or 'a dataset'}. It replays the steps with the"),
        comment("quantilo library on the app's copy of that dataset (never written), then"),
        comment(f"writes the cleaned rows to '{target}':"),
        comment(f"    python {stem}.py [path/to/data] [cleaned.parquet]"),
        comment("Another file is cleaned the same way when it has the same columns."),
        comment(_counts(steps)),
        *(comment(note) for note in info.notes),
        "",
        "import shutil",
        "import sys",
        "",
        "import quantilo as dx",
        "",
        f"PATH = sys.argv[1] if len(sys.argv) > 1 else {literal(parquet_path)}",
        f"TARGET = sys.argv[2] if len(sys.argv) > 2 else {literal(target)}",
        "STEPS = [",
    ]
    for step in steps:
        lines.append("    " + comment(step_sentence(step, info.titles)))
        body = literal(step.model_dump(mode="json")).replace("\n", "\n    ")
        lines.append(f"    {body},")
    lines += [
        "]",
        "",
        "cleaning = dx.open(PATH).clean(STEPS)",
    ]
    for name, path in (sources or {}).items():  # C-57, C-58: the datasets appended or merged
        lines.append(f"cleaning.attach({literal(name)}, dx.open({literal(path)}))")
    lines += [
        f"cleaned = cleaning.apply(name={literal(info.name)})",
        "shutil.copyfile(cleaned.path, TARGET)",
        "print(cleaning.report.summary() if cleaning.report else 'cleaned')",
        "",
    ]
    return "\n".join(lines)
