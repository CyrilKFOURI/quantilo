"""N46: the cleaning journal. Who did what, when: each change of the draft (a step added, removed,
changed, turned off, moved; the reliable fixes applied for the user), each saved version, each
suggestion dismissed and each change of the checks; and, per version, a report that can be read
without the app (Markdown, HTML): the rules applied (automatically, ticked, added by hand), cells
changed per column, the rows removed and why, examples before and after, the types changed.

The entries come from coded comparisons of two lists of steps (`diff`), never from a guess.
"""

import html
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from quantilo.cleaning.replay import step_columns
from quantilo.cleaning.report import CleanReport

Actor = Literal["user", "auto", "agent", "replay"]
Action = Literal[
    "applied",  # a reliable fix applied for the user (the draft Clean opens with)
    "added",
    "removed",
    "changed",
    "turned_off",
    "turned_on",
    "moved",
    "reset",
    "saved",
    "dismissed",
    "restored",
    "checks",
]
ACTORS: dict[str, str] = {
    "user": "you",
    "auto": "applied automatically",
    "agent": "the assistant",
    "replay": "a replay",
}
ORIGINS: dict[str, str] = {
    "auto": "applied automatically (a reliable fix)",
    "suggested": "a suggestion you ticked",
    "manual": "added by hand",
}


@dataclass
class JournalEntry:
    at: str
    actor: Actor
    action: Action
    step_id: str | None = None
    op: str | None = None
    rule: str | None = None
    origin: str | None = None
    columns: list[str] = field(default_factory=list)
    cells: int | None = None  # the cells its suggestion changes (when it came from one)
    rows: int | None = None
    detail: str | None = None  # a sentence (the suggestion's title, a version's summary…)
    version_id: str | None = None


@dataclass
class Known:
    """What the suggestions say about a step (its title, cells, rows, an example)."""

    title: str
    cells: int = 0
    rows: int = 0
    example: tuple[str | None, str | None] | None = None


def _entry(
    at: str, actor: Actor, action: Action, step: Any, known: Mapping[str, Known]
) -> JournalEntry:
    about = known.get(step.suggestion or "")
    return JournalEntry(
        at=at,
        actor=actor,
        action=action,
        step_id=step.id,
        op=step.op,
        rule=step.rule,
        origin=step.origin,
        columns=step_columns(step),
        cells=about.cells if about else None,
        rows=about.rows if about else None,
        detail=about.title if about else None,
    )


def _body(step: Any) -> str:
    return json.dumps(step.model_dump(mode="json", exclude={"enabled"}), sort_keys=True)


def diff(
    before: Sequence[Any],
    after: Sequence[Any],
    at: str,
    actor: Actor,
    known: Mapping[str, Known] | None = None,
) -> list[JournalEntry]:
    """What turned `before` into `after`, step by step (by step id): added, removed, changed,
    turned off or on, and one `moved` when the order of the steps kept changed."""
    known = known or {}
    old = {s.id: s for s in before}
    new = {s.id: s for s in after}
    entries: list[JournalEntry] = []
    for step in after:
        previous = old.get(step.id)
        if previous is None:
            entries.append(_entry(at, actor, "added", step, known))
            continue
        if _body(previous) != _body(step):
            entries.append(_entry(at, actor, "changed", step, known))
        if previous.enabled != step.enabled:
            action: Action = "turned_on" if step.enabled else "turned_off"
            entries.append(_entry(at, actor, action, step, known))
    entries += [_entry(at, actor, "removed", s, known) for s in before if s.id not in new]
    kept_before = [s.id for s in before if s.id in new]
    kept_after = [s.id for s in after if s.id in old]
    if kept_before != kept_after:
        entries.append(JournalEntry(at, actor, "moved", detail=" → ".join(kept_after[:30])))
    return entries


# --- A version's journal -------------------------------------------------------------------------


@dataclass
class JournalStep:
    id: str
    op: str
    rule: str | None
    origin: str
    enabled: bool
    title: str
    columns: list[str]
    cells: int | None = None
    rows_removed: int = 0
    example: list[str | None] | None = None  # [before, after]


@dataclass
class CleanJournal:
    """A cleaned version's journal: what it is, its steps with where each came from, what they
    changed (the version's report) and the changes that led to it."""

    name: str
    dataset_id: str
    made_from: str
    made_at: str
    origin: str  # user, or auto (a version made after an import)
    steps: list[JournalStep]
    report: CleanReport | None
    entries: list[JournalEntry]
    applied: dict[str, int] = field(default_factory=dict)  # steps by origin (enabled only)


def journal_steps(
    steps: Sequence[Any], report: CleanReport | None, known: Mapping[str, Known]
) -> list[JournalStep]:
    removed = {s.id: s.rows_removed for s in report.steps} if report else {}
    found: list[JournalStep] = []
    for step in steps:
        about = known.get(step.suggestion or "")
        columns = step_columns(step)
        found.append(
            JournalStep(
                id=step.id,
                op=step.op,
                rule=step.rule,
                origin=step.origin,
                enabled=step.enabled,
                title=about.title if about else f"{step.op} on {', '.join(columns)}".strip(),
                columns=columns,
                cells=about.cells if about else None,
                rows_removed=removed.get(step.id, 0),
                example=list(about.example) if about and about.example else None,
            )
        )
    return found


def counts(steps: Sequence[JournalStep]) -> dict[str, int]:
    on = [s for s in steps if s.enabled]
    return {origin: sum(1 for s in on if s.origin == origin) for origin in ORIGINS}


def sentence(entry: JournalEntry) -> str:
    """'you removed s3 (B-04, null_values on status): Read 'n/a' as missing in status'."""
    who = ACTORS.get(entry.actor, entry.actor)
    words = {
        "applied": "applied",
        "added": "added",
        "removed": "removed",
        "changed": "changed",
        "turned_off": "turned off",
        "turned_on": "turned on",
        "moved": "reordered the steps",
        "reset": "reset the draft",
        "saved": "saved",
        "dismissed": "dismissed",
        "restored": "suggested again",
        "checks": "changed the checks",
    }
    text = f"{who} {words.get(entry.action, entry.action)}"
    if entry.step_id:
        about = ", ".join(p for p in (entry.rule, entry.op) if p)
        on = f" on {', '.join(entry.columns)}" if entry.columns else ""
        text += f" {entry.step_id} ({about}{on})"
    if entry.detail:
        text += f": {entry.detail}"
    if entry.cells:
        text += f" · {entry.cells:,} cells"
    if entry.rows:
        text += f" · {entry.rows:,} rows"
    return text


def _value(value: str | None) -> str:
    return "empty" if value is None else value


def _sections(journal: CleanJournal) -> list[tuple[str, list[str], list[list[str]] | None]]:
    """The journal as titled sections: (title, lines, table rows with a header first)."""
    report = journal.report
    applied = journal.applied or counts(journal.steps)
    head = [
        f"Made {journal.made_at} from {journal.made_from}"
        + (" (automatically, after an import)" if journal.origin == "auto" else "")
        + ".",
        f"{applied.get('auto', 0)} steps applied automatically (reliable fixes), "
        f"{applied.get('suggested', 0)} suggestions ticked, {applied.get('manual', 0)} steps "
        "added by hand.",
    ]
    if report:
        head.append(report.summary() + f"; {report.rows_before:,} rows → {report.rows_after:,}.")
    sections: list[tuple[str, list[str], list[list[str]] | None]] = [("Summary", head, None)]
    table = [["Step", "Rule", "Where it came from", "What it does", "Cells", "Rows removed"]]
    for step in journal.steps:
        table.append(
            [
                step.id + ("" if step.enabled else " (off)"),
                step.rule or "",
                ORIGINS.get(step.origin, step.origin),
                step.title,
                "" if step.cells is None else f"{step.cells:,}",
                f"{step.rows_removed:,}" if step.rows_removed else "",
            ]
        )
    sections.append(("Steps", [], table))
    if report and report.cells_by_column:
        cells = [["Column", "Cells changed"]] + [
            [name, f"{count:,}"] for name, count in sorted(report.cells_by_column.items())
        ]
        sections.append(("Cells changed per column", [], cells))
    if report and report.rows_removed:
        why = {s.id: s.title for s in journal.steps}
        lines = [
            f"{s.rows_removed:,} rows by {s.id} ({why.get(s.id, s.op)})"
            for s in report.steps
            if s.rows_removed
        ]
        sections.append(("Rows removed", lines, None))
    examples = [s for s in journal.steps if s.example and s.enabled]
    if examples:
        rows = [["Step", "Before", "After"]] + [
            [s.id, _value(s.example[0]), _value(s.example[1])] for s in examples if s.example
        ]
        sections.append(("Examples", [], rows))
    if report and report.type_changes:
        types = [["Column", "Before", "After"]] + [
            [c.column, c.before, c.after] for c in report.type_changes
        ]
        sections.append(("Types changed", [], types))
    if report and (report.added or report.dropped):
        lines = []
        if report.added:
            lines.append(f"Columns added: {', '.join(report.added)}.")
        if report.dropped:
            lines.append(f"Columns removed: {', '.join(report.dropped)}.")
        sections.append(("Columns", lines, None))
    log = [f"{entry.at[:19].replace('T', ' ')} · {sentence(entry)}" for entry in journal.entries]
    sections.append(("Journal", log or ["No change of the draft was recorded."], None))
    return sections


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(journal: CleanJournal) -> str:
    lines = [f"# Cleaning log: {_md_cell(journal.name)}", ""]
    for title, texts, table in _sections(journal):
        lines += [f"## {title}", ""]
        lines += [f"- {_md_cell(text)}" for text in texts]
        if table:
            lines.append("| " + " | ".join(_md_cell(c) for c in table[0]) + " |")
            lines.append("|" + "---|" * len(table[0]))
            lines += ["| " + " | ".join(_md_cell(c) for c in row) + " |" for row in table[1:]]
        lines.append("")
    lines.append("Exported from Quantilo. Check your data.")
    return "\n".join(lines) + "\n"


def to_html(journal: CleanJournal, escape: Callable[[str], str] = html.escape) -> str:
    """A page of its own: no script, no outside file, every text escaped."""
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>Cleaning log: {escape(journal.name)}</title>",
        "<style>body{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;"
        "padding:0 16px;color:#1b1b1f;background:#fff}table{border-collapse:collapse;width:100%}"
        "th,td{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:top}"
        "th{color:#555;font-weight:500}td{font-variant-numeric:tabular-nums}</style>",
        "</head><body>",
        f"<h1>Cleaning log: {escape(journal.name)}</h1>",
    ]
    for title, texts, table in _sections(journal):
        parts.append(f"<h2>{escape(title)}</h2>")
        if texts:
            parts.append("<ul>" + "".join(f"<li>{escape(t)}</li>" for t in texts) + "</ul>")
        if table:
            header = "".join(f"<th>{escape(c)}</th>" for c in table[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{escape(c)}</td>" for c in row) + "</tr>"
                for row in table[1:]
            )
            parts.append(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>")
    parts.append("<p>Exported from Quantilo. Check your data.</p></body></html>")
    return "\n".join(parts) + "\n"
