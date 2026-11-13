"""A recipe written as a readable Python script that uses the library, to run it without the app
(Windows Task Scheduler, cron, a notebook). It prints the recipe's key figures as JSON.

Every column name and value is written as a Python literal (`repr`), never pasted into code;
text that ends up in comments has its line breaks removed.
"""

import math
import pprint
import re
import unicodedata
from typing import Any

from quantilo.recipes.model import (
    ColumnStep,
    FilterStep,
    KpiStep,
    MlStep,
    OutputStep,
    Recipe,
    TimeSeriesStep,
    ordered,
)
from quantilo.recipes.steps import sentence


def literal(value: Any) -> str:
    """A Python literal for plain JSON-like data (refuses anything else)."""

    def check(item: Any) -> Any:
        if item is None or isinstance(item, bool | int | str):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("A recipe value is not a finite number.")
            return item
        if isinstance(item, list | tuple):
            return [check(v) for v in item]
        if isinstance(item, dict):
            return {str(k): check(v) for k, v in item.items()}
        raise TypeError(f"Not plain data: {type(item).__name__}")

    return pprint.pformat(check(value), width=88, sort_dicts=False)


# PEP 263: "coding:" or "coding=" in one of a file's first two comment lines sets how Python decodes
# the whole file; a dataset named "x coding=utf-7" would turn "+AAo-" into a line break and its
# next words into code. Generated scripts declare UTF-8 on their first line (ENCODING_LINE) and a
# comment never carries the pattern.
ENCODING_LINE = "# -*- coding: utf-8 -*-"
_CODING = re.compile(r"coding([:=])")


def comment(text: str) -> str:
    """One comment line: line breaks and other control characters become spaces, and an encoding
    declaration (PEP 263) is broken up."""
    clean = "".join(" " if unicodedata.category(c) in ("Cc", "Cf", "Zl", "Zp") else c for c in text)
    return "# " + _CODING.sub(lambda match: "coding " + match.group(1), clean)


def file_stem(name: str) -> str:
    return (
        "".join(c if c.isascii() and (c.isalnum() or c in "-_") else "_" for c in name) or "recipe"
    )


def _indent(text: str, by: int) -> str:
    pad = " " * by
    return text.replace("\n", "\n" + pad)


def _call(name: str, *args: Any, **kwargs: Any) -> str:
    """`name(args…, key=value…)` on one line when it fits, else one argument per line."""
    parts = [literal(a) for a in args]
    parts += [f"{key}={literal(value)}" for key, value in kwargs.items()]
    one_line = f"{name}({', '.join(parts)})"
    if len(one_line) <= 88 and "\n" not in one_line:
        return one_line
    inner = "".join(f"\n    {_indent(part, 4)}," for part in parts)
    return f"{name}({inner}\n)"


def _rows(conditions: list[Any], base: str = "ds") -> str:
    text = base
    for condition in conditions:
        dumped = condition.model_dump(mode="json", exclude_defaults=True)
        text += f".where(dx.Condition.model_validate({_indent(literal(dumped), 4)}))"
    return text


def to_python(recipe: Recipe, path: str = "data.csv") -> str:
    names = {s.id: s.name for s in recipe.steps if isinstance(s, FilterStep)}
    steps = [s for s in ordered(recipe.steps) if s.enabled]
    stem = file_stem(recipe.name)
    lines = [
        ENCODING_LINE,
        comment(f"Recipe “{recipe.name}”, version {recipe.version}, exported from Quantilo."),
        comment(f"Made on {recipe.source or 'a dataset'}; replays it with the quantilo"),
        comment("library, then prints its key figures as JSON:"),
        comment(f"    python {stem}.py path/to/new_file.csv"),
        comment("The file's columns are expected under the recipe's names."),
        "",
        "import json",
        "import sys",
        "",
        "import quantilo as dx",
        "",
        f"PATH = sys.argv[1] if len(sys.argv) > 1 else {literal(path)}",
        "ds = dx.open(PATH)",
        "figures = {}",
        "rows = {}  # filter step → its filtered rows",
    ]
    section = None
    titles = {
        "column": "Columns",
        "filter": "Filters",
        "kpi": "KPIs",
        "timeseries": "Time series",
        "ml": "Machine learning",
        "output": "Output",
    }
    for step in steps:
        if step.kind != section:
            section = step.kind
            lines += ["", comment(f"--- {titles[step.kind]} ---")]
        lines.append(comment(sentence(step, names)))
        base = f"rows[{literal(step.filter)}]" if getattr(step, "filter", None) else "ds"
        if isinstance(step, ColumnStep):
            if step.type is not None:
                lines.append(_call("ds.retype", step.column, step.type))
            if step.role is not None:
                lines.append(_call("ds.set_role", step.column, step.role or None))
            if step.order:
                lines.append(_call("ds.set_order", step.column, step.order))
            if step.excluded is True:
                lines.append(_call("ds.exclude", step.column))
            elif step.excluded is False:
                lines.append(_call("ds.include", step.column))
        elif isinstance(step, FilterStep):
            lines.append(f"rows[{literal(step.id)}] = {_rows(step.conditions)}")
        elif isinstance(step, KpiStep):
            definition = step.definition.model_dump(
                mode="json", by_alias=True, exclude_defaults=True
            )
            lines.append(f"kpi = {base}.kpi({_indent(literal(definition), 4)})")
            key, label = literal(step.id), literal(step.name)
            lines.append(f"figures[{key}] = {{'label': {label}, 'value': kpi.value}}")
        elif isinstance(step, TimeSeriesStep):
            a = step.analysis
            options: dict[str, Any] = {}
            for key in ("group", "series", "window", "season"):
                if getattr(a, key) is not None:
                    options[key] = getattr(a, key)
            for key, default in (
                ("period", "auto"),
                ("aggregate", "auto"),
                ("align", "trailing"),
                ("gaps", "leave"),
            ):
                if getattr(a, key) != default:
                    options[key] = getattr(a, key)
            lines.append(f"series = {_call(f'{base}.timeseries', a.time, a.value, **options)}")
            if step.forecast:
                forecast_options: dict[str, Any] = {"horizon": a.horizon, "models": a.models}
                if a.drivers:  # dicts, as `series.forecast(drivers=...)` takes them
                    forecast_options["drivers"] = [d.model_dump() for d in a.drivers]
                if a.models and "sarima" in a.models:
                    forecast_options |= {
                        "order": list(a.order),
                        "seasonal_order": list(a.seasonal_order),
                    }
                lines.append(f"forecast = {_call('series.forecast', **forecast_options)}")
                lines.append(
                    "best = next((r for r in forecast.results if r.key == forecast.best), None)"
                )
                label = f"{a.value or 'Rows'}: next value"
                lines.append(
                    f"figures[{literal(step.id)}] = {{'label': {literal(label)}, "
                    "'value': best.forecast[0] if best and best.forecast else None}"
                )
        elif isinstance(step, MlStep):
            options = {"validation": step.validation.model_dump(mode="json", exclude_defaults=True)}
            if step.classes is not None:
                options["classes"] = step.classes.model_dump(mode="json", exclude_defaults=True)
            learner = _call(f"{_rows(step.conditions)}.ml", step.target, **options)
            preparation = (
                step.preparation.model_dump(mode="json", exclude_defaults=True)
                if step.preparation is not None
                else None
            )
            lines.append(f"learner = {learner}")
            train: dict[str, Any] = {"settings": step.settings} if step.settings else {}
            if preparation is not None:
                train["preparation"] = preparation
            lines.append(f"model = {_call('learner.train', step.model, **train)}")
            key, label = literal(step.id), literal(f"{step.model} → {step.target}")
            lines.append(
                f"figures[{key}] = {{'label': {label}, "
                "'metric': model.primary, 'value': model.score}"
            )
        elif isinstance(step, OutputStep):
            if step.report:
                lines.append(f"{base}.report({literal(f'{stem}_report.html')})")
            if step.csv:
                lines.append(f"{base}.export({literal(f'{stem}_rows.csv')})")
    lines += ["", "print(json.dumps(figures, indent=2))", ""]
    return "\n".join(lines)
