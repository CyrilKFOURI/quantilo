"""Write the engine's closed lists as a JSON fixture (M01): the model catalogue (names, families,
descriptions, settings, options, why the defaults, badges, rules, limits, recommendations) and the
KPI templates (kinds of datasets, fixed KPI names, name patterns), and the reasons' templates (M02).
The engine stays in English; the interface translates these by id, and a frontend test checks that
every one has its texts. It also writes the reasons shown on two samples (`reason-examples.json`:
columns, Basic cleaning, KPI suggestions and warnings), which the interface's test reads back.

    python -m app.engine_texts_dump ../frontend/src/api/engine-texts.json
"""

import ast
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import quantilo
from quantilo.kpi import archetypes, rules, suggest
from quantilo.ml import auto, browser
from quantilo.ml import catalog as ml
from quantilo.reasons import placeholders

TARGET = Path(__file__).resolve().parents[2] / "frontend" / "src" / "api" / "engine-texts.json"
EXAMPLES = TARGET.with_name("reason-examples.json")
SAMPLES = Path(__file__).resolve().parents[2] / "samples"


def _contexts() -> list[ml.Context]:
    """Enough kinds of data for every data-dependent default, badge and limit to show."""
    found = []
    for task in ("regression", "binary", "multiclass"):
        for rows in (200, 5_000, 30_000, 200_000):
            for minority in (None, 0.05, 0.4) if task != "regression" else (None,):
                for features in (3, 50):
                    found.append(ml.Context(rows, task, minority, features, 0.5, 0.1))
    return found


def _add(bucket: list[str], text: str | None) -> None:
    if text and text not in bucket:
        bucket.append(text)


def _models(contexts: list[ml.Context]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    badges: list[str] = []
    limits: list[str] = []
    for key, spec in ml.CATALOGUE.items():
        model = models.setdefault(
            key,
            {
                "label": spec.label,
                "family": spec.family,
                "description": spec.description,
                "settings": {},
                "rules": [],
            },
        )
        for rule in spec.rules:
            _add(model["rules"], rule.reason)
        for context in contexts:
            if context.task not in spec.tasks:
                continue
            for badge in spec.badges(context):
                _add(badges, badge)
            _add(limits, spec.limit(context))
            for setting in spec.settings(context):
                found = model["settings"].setdefault(
                    setting.key,
                    {"label": setting.label, "group": setting.group, "options": {}, "whys": []},
                )
                for option in setting.options:
                    found["options"][str(option)] = setting.option_labels.get(option, str(option))
                _add(found["whys"], setting.why)
    return {"models": models, "badges": badges, "limits": limits}


def _ml() -> dict[str, Any]:
    contexts = _contexts()
    desktop = _models(contexts)
    browser.IN_BROWSER = True  # the browser version (102) says some defaults differently
    try:
        lite = _models(contexts)
    finally:
        browser.IN_BROWSER = sys.platform == "emscripten"
    for key, model in lite["models"].items():
        for name, setting in model["settings"].items():
            for why in setting["whys"]:
                _add(desktop["models"][key]["settings"][name]["whys"], why)
    for name in ("badges", "limits"):
        for text in lite[name]:
            _add(desktop[name], text)
    recommended: list[str] = []
    for context in contexts:
        for _, why in ml.recommend(context):
            _add(recommended, why)
    groups: list[str] = []
    for model in desktop["models"].values():
        for setting in model["settings"].values():
            _add(groups, setting["group"])
    return {
        "families": ml.FAMILIES,
        "groups": groups,
        "recommended": recommended,
        **desktop,
        # the Auto's entries that are not models of the catalogue, and its tuned suffix
        "auto": {
            "labels": {auto.CONSTANT: "Constant", auto.ENSEMBLE: auto.ENSEMBLE_LABEL},
            "families": ["Baseline", "Ensemble"],
            "tuned": " · tuned",
        },
    }


def _pattern(node: ast.expr) -> str | None:
    """A string literal as written, an f-string with `{}` for each value; None otherwise."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
        )
    return None


def _kpi_names() -> dict[str, str]:
    """The name each rule of `suggest.py` gives its KPI: fixed ("Average basket"), a pattern
    ("{} per customer"), or "" when it is built from the measure (`measure_label`)."""
    names: dict[str, str] = {}
    tree = ast.parse(inspect.getsource(suggest))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_Candidate"):
            continue
        if len(node.args) < 2:
            continue
        rule, name = _pattern(node.args[0]), _pattern(node.args[1])
        if rule and name is not None:
            names.setdefault(rule, name)
        elif rule:
            names.setdefault(rule, "{}")  # a name made beforehand (R-SUG-32: price per unit)
    return dict(sorted(names.items()))


def _archetype_message() -> str:
    tree = ast.parse(inspect.getsource(archetypes))
    found = [_pattern(n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr)]
    return next(p for p in found if p and p.startswith("Looks like"))


def _kpi() -> dict[str, Any]:
    return {
        "archetypes": {t.id: t.label for t in archetypes.templates()},
        "message": _archetype_message(),
        "names": _kpi_names(),
        "aggs": rules.AGG_WORDS,
        "delays": rules.DELAY_WORDS,
    }


def _literal(node: ast.expr | None) -> list[str] | None:
    """A string literal, or the two of `a if test else b`; None for anything else."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.IfExp):
        body, orelse = _literal(node.body), _literal(node.orelse)
        if body and orelse:
            return body + orelse
    return None


def _reason_calls(tree: ast.AST, path: str) -> list[tuple[str, str, list[str]]]:
    """(code, English template, vars) of each `say`, `say_count` and engine `rule` call."""
    found: list[tuple[str, str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        name = node.func.id
        if name not in ("say", "say_count", "rule") or (
            name == "rule" and not path.endswith("engine.py")
        ):
            continue
        if node.args and isinstance(node.args[0], ast.Name):
            continue  # a helper passing its own code on (`rule`, `say_count`)
        names = sorted(k.arg for k in node.keywords if k.arg)
        codes = _literal(node.args[0]) if node.args else None
        if name == "say_count":
            one, many = _literal(node.args[2]), _literal(node.args[3])
            if not (codes and one and many) or len(codes) != 1:
                raise ValueError(f"{path}:{node.lineno}: say_count needs literal texts")
            found += [
                (codes[0], many[0], ["count", *names]),
                (f"{codes[0]}#one", one[0], ["count", *names]),
            ]
            continue
        templates = _literal(node.args[1]) if len(node.args) > 1 else None
        if not codes or not templates or len(codes) not in (1, len(templates)):
            raise ValueError(f"{path}:{node.lineno}: {name}() needs a literal code and template")
        for index, template in enumerate(templates):
            code = codes[index] if len(codes) > 1 else codes[0]
            if name == "rule":
                template = f"{template} ({code.split('.')[0]})"
            found.append((code, template, names))
    return found


def _reasons() -> dict[str, Any]:
    """M02: every reason template the engine says (`quantilo.reasons.say`), by code, with the
    vars it cites; the interface's English copy (reasons.json) is checked against it."""
    root = Path(inspect.getfile(quantilo)).parent
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "say(" not in source and "say_count(" not in source:
            continue
        for code, template, names in _reason_calls(ast.parse(source), path.as_posix()):
            cited = sorted(set(placeholders(template)))
            if cited != sorted(set(names)):
                raise ValueError(f"{code}: the template cites {cited}, the call gives {names}")
            entry = {"template": template, "vars": cited}
            if found.setdefault(code, entry) != entry:
                raise ValueError(
                    f"{code}: two templates ({found[code]['template']!r}, {template!r})"
                )
    return dict(sorted(found.items()))


def reason_examples() -> list[dict[str, Any]]:
    """M02: the reasons the pages show on dirty_traps and messy_orders, once each: where, the
    English text, its code and vars (none: no template), and for a suggestion its `measured`."""
    import tempfile

    import quantilo as dx
    from quantilo.reasons import details

    found: dict[str, dict[str, Any]] = {}

    def add(place: str, text: str, code: str | None, values: dict[str, str], **more: Any) -> None:
        found.setdefault(text, {"place": place, "text": text, "code": code, "vars": values, **more})

    with tempfile.TemporaryDirectory() as cache:
        for name in ("dirty_traps.csv", "messy_orders.csv"):
            ds = dx.open(SAMPLES / name, cache_dir=cache)
            for column in ds.types:
                for item in details(column.reasons):
                    add("columns", item.text, item.code, item.vars)
            for s in ds.clean().suggestions:
                add("clean", s.reason, s.reason_code, s.reason_vars, measured=s.measured)
            kpis = ds.kpi_suggestions().kpis
            for kpi in kpis:
                for item in kpi.reason_details:
                    add("kpi", item.text, item.code, item.vars)
            for kpi in kpis[:3]:
                for item in ds.kpi(kpi.definition).warning_details:
                    add("kpi", item.text, item.code, item.vars)
    return list(found.values())


def dump() -> dict[str, Any]:
    return {"ml": _ml(), "kpi": _kpi(), "reasons": _reasons()}


def text() -> str:
    return json.dumps(dump(), indent=1, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else TARGET
    target.write_text(text(), encoding="utf-8")
    examples = json.dumps(reason_examples(), indent=1, ensure_ascii=False) + "\n"
    target.with_name(EXAMPLES.name).write_text(examples, encoding="utf-8")
