"""What each step reads (to skip it when a column is missing), the same step on renamed columns,
and the sentence the recipe page shows ("Treat code_postal as categorical")."""

from collections.abc import Callable

from quantilo.filters.model import Condition
from quantilo.kpi.definition import Definition
from quantilo.recipes.model import (
    ColumnStep,
    FilterStep,
    KpiStep,
    MlStep,
    OutputStep,
    Step,
    TimeSeriesStep,
)

TYPE_LABELS = {
    "numeric_continuous": "a continuous number",
    "numeric_discrete": "a discrete number",
    "categorical_nominal": "categorical",
    "categorical_ordinal": "ordered categorical",
    "boolean": "true / false",
    "datetime": "a date",
    "text": "text",
    "identifier": "an identifier",
    "ignored": "ignored",
}
FORECAST_LABELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive",
    "ets_auto": "ETS",
    "arima_auto": "ARIMA",
    "drift": "Drift",
    "mean": "Mean",
    "ses": "Simple exponential smoothing",
    "holt": "Holt",
    "holt_winters": "Holt-Winters",
    "theta": "Theta",
    "sarima": "SARIMA",
    "tslm": "Linear regression",
    "structural": "Structural model (Kalman)",
    "gbm_lags": "Gradient boosting",
    "mlp_lags": "Neural network",
    "croston": "Croston",
    "sba": "SBA",
    "tsb": "TSB",
    "adida": "ADIDA",
    "combination": "Combination",
}
DEFAULT_FORECAST = ("naive", "seasonal_naive", "ets_auto", "arima_auto")
AGG_LABELS = {
    "count": "number of rows",
    "sum": "sum",
    "mean": "mean",
    "median": "median",
    "min": "minimum",
    "max": "maximum",
    "distinct": "distinct values",
    "true_share": "share of true",
    "share": "share of values",
    "weighted_mean": "weighted mean",
}
VALIDATIONS = {
    "split": "a train / test split",
    "kfold": "{folds}-fold cross-validation",
    "repeated_kfold": "repeated {folds}-fold cross-validation",
    "group_kfold": "group {folds}-fold cross-validation",
    "time_split": "a time-based split",
    "loo": "leave-one-out",
    "panel_walk_forward": "a walk-forward by period",
}


def _condition_columns(conditions: list[Condition]) -> list[str]:
    found: list[str] = []
    for condition in conditions:
        found += [condition.column, *condition.columns]
        if condition.order_by:
            found.append(condition.order_by)
    return found


def _definition_columns(definition: Definition) -> list[str]:
    found = [name for m in definition.measures for name in m.columns()]
    found += [g.column for g in definition.groups]
    if definition.time is not None:
        found.append(definition.time.column)
    return found


def _validation_columns(step: MlStep) -> list[str]:
    v = step.validation
    return [c for c in (v.group, v.time_column, v.period_column, v.entity_column) if c]


def columns_read(step: Step) -> list[str]:
    """The columns a step needs: without one of them, it is skipped."""
    if isinstance(step, ColumnStep):
        found = [step.column]
    elif isinstance(step, FilterStep):
        found = _condition_columns(step.conditions)
    elif isinstance(step, KpiStep):
        found = _definition_columns(step.definition)
    elif isinstance(step, TimeSeriesStep):
        a = step.analysis
        found = [c for c in (a.time, a.value, a.group) if c]
        found += [d.column for d in a.drivers or [] if d.kind != "calendar"]
    elif isinstance(step, MlStep):
        found = [step.target, *_validation_columns(step), *_condition_columns(step.conditions)]
    else:
        found = []
    return list(dict.fromkeys(found))


def categories_read(step: Step) -> list[str]:
    """Columns whose categories a step depends on (a new category is worth a warning)."""
    if isinstance(step, FilterStep):
        return [c.column for c in step.conditions if c.op == "in"]
    if isinstance(step, KpiStep):
        return [g.column for g in step.definition.groups]
    if isinstance(step, TimeSeriesStep):
        return [step.analysis.group] if step.analysis.group else []
    if isinstance(step, MlStep):
        return [step.target]
    return []


def _renamed_conditions(conditions: list[Condition], to: Callable[[str], str]) -> list[Condition]:
    return [
        c.model_copy(
            update={
                "column": to(c.column),
                "columns": [to(name) for name in c.columns],
                "order_by": to(c.order_by) if c.order_by else None,
            }
        )
        for c in conditions
    ]


def renamed(step: Step, mapping: dict[str, str | None]) -> Step:
    """The step reading the file's columns (recipe name → file name)."""

    def to(name: str) -> str:
        return mapping.get(name) or name

    if isinstance(step, ColumnStep):
        return step.model_copy(update={"column": to(step.column)})
    if isinstance(step, FilterStep):
        return step.model_copy(update={"conditions": _renamed_conditions(step.conditions, to)})
    if isinstance(step, KpiStep):
        raw = step.definition.model_dump(by_alias=True)
        for measure in raw["measures"]:
            if measure.get("column"):
                measure["column"] = to(measure["column"])
        for group in raw["groups"]:
            group["column"] = to(group["column"])
        if raw.get("time"):
            raw["time"]["column"] = to(raw["time"]["column"])
        return step.model_copy(update={"definition": Definition.model_validate(raw)})
    if isinstance(step, TimeSeriesStep):
        a = step.analysis
        analysis = a.model_copy(
            update={
                "time": to(a.time) if a.time else None,
                "value": to(a.value) if a.value else None,
                "group": to(a.group) if a.group else None,
            }
        )
        return step.model_copy(update={"analysis": analysis})
    if isinstance(step, MlStep):
        v = step.validation
        validation = v.model_copy(
            update={
                "group": to(v.group) if v.group else None,
                "time_column": to(v.time_column) if v.time_column else None,
                "period_column": to(v.period_column) if v.period_column else None,
                "entity_column": to(v.entity_column) if v.entity_column else None,
            }
        )
        preparation = step.preparation
        if preparation is not None:
            columns = {
                to(name): choice
                for name, choice in preparation.columns.items()
                if mapping.get(name, name) is not None
            }
            preparation = preparation.model_copy(update={"columns": columns})
        return step.model_copy(
            update={
                "target": to(step.target),
                "validation": validation,
                "preparation": preparation,
                "conditions": _renamed_conditions(step.conditions, to),
                "inputs": [to(name) for name in step.inputs],  # a missing one is a warning
            }
        )
    return step


# --- Sentences ---------------------------------------------------------------------------------


def _list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _condition(condition: Condition) -> str:
    column, op = condition.column, condition.op
    if condition.part and op == "part_in":
        text = f"{column} ({condition.part.replace('_', ' ')}) is {' or '.join(condition.values)}"
    elif op == "in":
        text = f"{column} is {' or '.join(condition.values)}"
    elif op == "between":
        text = f"{column} is between {condition.low} and {condition.high}"
    elif op == "is_null":
        text = f"{column} is empty"
    elif op == "unique":
        text = f"one row per {column}"
    elif op == "duplicated":
        text = f"{column} is duplicated"
    else:
        words = {
            "eq": "is",
            "gt": ">",
            "gte": "≥",
            "lt": "<",
            "lte": "≤",
            "contains": "contains",
            "starts_with": "starts with",
        }
        text = f"{column} {words.get(op, op)} {condition.value}"
    return f"not ({text})" if condition.negate else text


def _measure(definition: Definition) -> str:
    parts = []
    for measure in definition.measures:
        label = AGG_LABELS.get(measure.agg, measure.agg)
        parts.append(label if measure.column is None else f"{label} of {measure.column}")
    if definition.ratio is not None:
        return f"ratio of {_list(parts)}"
    return _list(parts)


def model_label(key: str | None) -> str:
    if key is None:
        return "the default model"
    try:
        from quantilo.ml.models import CATALOGUE

        spec = CATALOGUE.get(key)
        return spec.label if spec else key
    except Exception:  # the ML extra may be missing: the key still says which model
        return key


def sentence(step: Step, filters: dict[str, str] | None = None) -> str:
    """One readable line per step. `filters`: filter step id → its name."""
    names = filters or {}
    if isinstance(step, ColumnStep):
        parts = []
        if step.type is not None:
            parts.append(f"Treat {step.column} as {TYPE_LABELS.get(step.type, step.type)}")
        if step.role is not None:
            role = step.role.replace("_", " ") if step.role else "no role"
            parts.append(
                f"give {step.column} the role {role}"
                if step.role
                else f"give {step.column} no role"
            )
        if step.order:
            more = "…" if len(step.order) > 6 else ""
            parts.append(f"order {step.column} as {', '.join(step.order[:6])}{more}")
        if step.excluded is True:
            parts.append(f"leave {step.column} out of the analyses")
        elif step.excluded is False:
            parts.append(f"bring {step.column} back into the analyses")
        text = "; ".join(parts)
        return text[0].upper() + text[1:]
    if isinstance(step, FilterStep):
        where = " and ".join(_condition(c) for c in step.conditions)
        text = f"Filter “{step.name}”: keep rows where {where}"
        return f"{text} (the next steps use these rows)" if step.active else text
    on = ""
    used = getattr(step, "filter", None)
    if used:
        on = f", on the rows of “{names.get(used, used)}”"
    if isinstance(step, KpiStep):
        definition = step.definition
        grain = ""
        if definition.time is not None:
            g = definition.time.grain
            grain = f" per {g}" if g != "auto" else f" over {definition.time.column}"
        split = f", by {_list([g.column for g in definition.groups])}" if definition.groups else ""
        return f"KPI “{step.name}”: {_measure(definition)}{grain}{split}{on}"
    if isinstance(step, TimeSeriesStep):
        a = step.analysis
        what = a.value or "the number of rows"
        period = f" per {a.period}" if a.period != "auto" else ""
        group = f" for {a.group} = {a.series}" if a.group and a.series else ""
        if not step.forecast:
            return f"Analyse {what}{period}{group} (“{step.name}”){on}"
        models = [FORECAST_LABELS.get(m, m) for m in (a.models or DEFAULT_FORECAST)]
        ahead = f", {a.horizon} periods ahead" if a.horizon else ""
        return f"Forecast {what}{period}{group}{ahead}, with {_list(models)}{on}"
    if isinstance(step, MlStep):
        v = step.validation
        how = VALIDATIONS.get(v.method, v.method).format(folds=v.folds)
        classes = f" (cut into {len(step.classes.names)} classes)" if step.classes else ""
        rows = (
            f", on rows where {' and '.join(_condition(c) for c in step.conditions)}"
            if step.conditions
            else ""
        )
        model = model_label(step.model)
        return f"Train {model} to predict {step.target}{classes}, judged by {how}{rows}"
    if isinstance(step, OutputStep):
        files = [
            f
            for f, wanted in (("the HTML report", step.report), ("a CSV of the rows", step.csv))
            if wanted
        ]
        return f"Write {_list(files)}{on}"
    return step.kind
