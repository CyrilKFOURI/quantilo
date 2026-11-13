"""How the open dataset was read from its file (080, J2 of the table-detection study): the table's
place, header, rows set aside, rules and confidence, the restructurings proposed, the keys to the
other tables of its source. The agent reads the app's own detection; it never guesses."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel

from app.modules.agent.engine.tools import AppLink, Tool, ToolContext, ToolError, ToolOutput
from app.modules.agent.tools import AppServices

MAX_TABLES = 10  # a workbook read as a whole sheet: the tables its detection sees, at most


class NoInput(BaseModel):
    pass


def _rules(choice: dict[str, Any]) -> list[str]:
    return [f"{r['rule']}: {r['message']}" for r in choice.get("reasons") or [] if r.get("message")]


def _table(choice: dict[str, Any], reshaped: bool) -> dict[str, Any]:
    """A detected table (`TableChoice` as stored in `datasets.layout`) as the model reads it."""
    area = choice.get("area") or {}
    out: dict[str, Any] = {
        "location": choice.get("location"),
        "level": choice.get("level"),
        "confidence": choice.get("confidence"),
        "sentence": choice.get("summary"),
        "rows": choice.get("rows"),
        "columns": choice.get("columns"),
        "kind": choice.get("kind"),
        "rules": _rules(choice),
    }
    if area.get("header_rows"):
        out["header_rows"] = area["header_rows"]
    if area.get("dropped_rows"):
        out["rows_set_aside"] = len(area["dropped_rows"])
    if choice.get("warnings"):
        out["warnings"] = choice["warnings"]
    proposals = choice.get("proposals") or []
    if proposals:
        out["restructurings"] = {"applied": reshaped, "proposed": proposals}
    return out


class DescribeLayout(Tool[NoInput]):
    name: ClassVar[str] = "describe_layout"
    description: ClassVar[str] = (
        "How the dataset was read from its file: for a table found in a workbook or a JSON / XML "
        "document, its place (sheet and range), header rows, rows set aside (totals, notes), the "
        "rules that found it with their confidence, the restructurings proposed (applied or not) "
        "and the keys to the other tables of the same file. Use it when the user doubts the "
        "import (missing rows, odd header, a total counted twice) or asks where the data comes "
        "from. Not for the columns' types (use list_columns)."
    )
    Input: ClassVar[type[BaseModel]] = NoInput

    def __init__(self, services: AppServices) -> None:
        self.services = services

    def label(self, args: NoInput) -> str:
        return "Reading how the file was imported"

    def run(self, ctx: ToolContext, args: NoInput) -> ToolOutput:
        datasets = self.services.datasets
        if datasets is None:
            raise ToolError("The import details are not available here.")
        source, siblings, record, links = datasets.origin(ctx.dataset_id)
        options = json.loads(record.options or "{}")
        path = Path(record.source_path)
        content: dict[str, Any] = {"file": path.name, "format": record.format}
        reshaped = bool(options.get("reshape"))
        tokens = {key: options[key] for key in ("null_tokens", "text_tokens") if options.get(key)}
        if tokens:
            content["placeholders_chosen_by_the_user"] = tokens  # T2u
        link = AppLink("columns", {}, "Check the columns")
        if record.layout:
            choice = json.loads(record.layout)
            table = _table(choice, reshaped)
            content["table"] = table
            if source is not None:
                content["source"] = {
                    "name": source.name,
                    "kind": source.kind,
                    "tables": len(siblings),
                }
            if links:
                content["keys"] = [
                    {"parent": lk.parent_id, "child": lk.child_id, "parent_key": lk.parent_key,
                     "child_key": lk.child_key, "rule": lk.rule}
                    for lk in links
                ]  # fmt: skip
            summary = (
                f"{table['location']}: {table['level']} confidence "
                f"({float(table['confidence'] or 0):.2f}). {table['sentence']}"
            )
            return ToolOutput(summary, content, link)
        if record.format == "excel" and path.is_file():
            from quantilo.ingestion.preview import sheet_choices

            sheets = options.get("sheets") or None
            choices = sheet_choices(path, sheets[:1] if sheets else None)
            header = options.get("header_row") or {}
            content["read"] = "whole sheet" + (
                f", header row {next(iter(header.values()))} (W1)" if header else ""
            )
            content["tables_the_detection_sees"] = [
                _table(asdict(c), reshaped) for c in choices[:MAX_TABLES]
            ]
            several = len(choices) > 1 or any(c.level != "high" for c in choices)
            if several:
                content["advice"] = (
                    "Several tables, or a doubtful one: re-import the file with 'Choose tables…' "
                    "to import each table alone (080b, R-TBL-01)."
                )
            summary = (
                f"{path.name}: read as a whole sheet; the detection sees "
                f"{len(choices)} table{'s' if len(choices) != 1 else ''} there."
            )
            return ToolOutput(summary, content, link)
        content["read"] = (
            "as a plain table: no layout detection applies to this format"
            if record.format != "excel"
            else "as a whole sheet (the file is no longer there to look again)"
        )
        csv = options.get("csv")
        if csv:
            content["csv"] = csv
        return ToolOutput(f"{path.name}: read {content['read']}.", content, link)
