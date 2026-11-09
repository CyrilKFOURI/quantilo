"""Recipes (030): matching, schema, replay, key figures, cancellation, Python export."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quantilo as dx
from quantilo.recipes import RecipeError, parse
from quantilo.recipes.matching import distance, match_columns, normalize
from quantilo.recipes.model import ColumnStep, Recipe, RecipeColumn
from quantilo.recipes.run import compare_figures, plan
from quantilo.recipes.script import to_python

FORECAST = {
    "time": "date",
    "value": "montant",
    "period": "month",
    "aggregate": "sum",
    "models": ["naive", "seasonal_naive"],
    "horizon": 3,
}
REVENUE = {"measures": [{"id": "m1", "agg": "sum", "column": "montant"}]}
ORDERS = {
    "measures": [{"id": "m1", "agg": "count"}],
    "time": {"column": "date", "grain": "month"},
}


def sales_frame(seed: int, months: int = 24) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.date_range("2024-01-01", periods=months * 30, freq="D")
    n = len(days)
    frame = pd.DataFrame(
        {
            "id_interne": [f"T{seed}-{i:05d}" for i in range(n)],
            "date": days.strftime("%Y-%m-%d"),
            "pays": rng.choice(["FR", "BE", "DE"], n),
            "indice": rng.integers(1, 6, n),
            "nb_articles": rng.integers(1, 6, n),
            "montant": (rng.gamma(2.0, 40.0, n) + 10 * np.sin(np.arange(n) / 30)).round(2),
        }
    )
    frame["churn"] = np.where(frame["montant"] + rng.normal(0, 30, n) > 90, "yes", "no")
    return frame


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("recipes")


def open_csv(frame: pd.DataFrame, folder: Path, name: str) -> dx.Dataset:
    path = folder / f"{name}.csv"
    frame.to_csv(path, index=False)
    return dx.open(path, cache_dir=folder / "cache")


@pytest.fixture(scope="module")
def october(cache: Path) -> dx.Dataset:
    return open_csv(sales_frame(1), cache, "octobre")


@pytest.fixture(scope="module")
def recipe(october: dx.Dataset) -> Recipe:
    october.retype("indice", "categorical_nominal").retype("nb_articles", "numeric_continuous")
    filtered = october.where("pays", "in", ["FR", "BE"])
    model = october.ml("churn").train()
    return dx.Recipe.from_dataset(
        filtered,
        "Ventes mensuelles",
        kpis={"Revenue": REVENUE, "Orders": ORDERS},
        timeseries={"Monthly sales": FORECAST},
        models=[model],
    )


# --- Matching ------------------------------------------------------------------------------------


def expected(*names: str, kind: str = "numeric_continuous") -> list[RecipeColumn]:
    return [RecipeColumn(name=n, type=kind) for n in names]  # type: ignore[arg-type]


def test_matching_same_name_normalized_typo_missing_and_new_columns() -> None:
    columns = [*expected("montant", "Région", "date_vente", "pays", "absent")]
    columns[2] = RecipeColumn(name="date_vente", type="datetime")
    columns[3] = RecipeColumn(name="pays", type="categorical_nominal")
    actual = [
        ("Montant", "numeric_continuous"),
        ("region", "numeric_continuous"),
        ("date_vnete", "datetime"),
        ("pays", "categorical_nominal"),
        ("en_plus", "text"),
    ]

    found = {m.recipe: m for m in match_columns(columns, actual).columns}
    matching = match_columns(columns, actual)

    assert found["pays"].status == "same"
    assert (found["montant"].status, found["montant"].column) == ("matched", "Montant")
    assert (found["Région"].status, found["Région"].column) == ("matched", "region")
    assert (found["date_vente"].status, found["date_vente"].column) == ("proposed", "date_vnete")
    assert found["date_vente"].distance == 2
    assert found["absent"].status == "missing"
    assert matching.new_columns == ["en_plus"]
    assert matching.mapping()["date_vente"] is None  # a proposal waits for the user
    assert matching.mapping(accept_proposals=True)["date_vente"] == "date_vnete"


def test_a_close_name_of_another_type_is_not_proposed() -> None:
    found = match_columns(
        expected("montant"), [("montnat", "text"), ("montan", "numeric_continuous")]
    ).columns[0]
    assert (found.status, found.column) == ("proposed", "montan")
    alone = match_columns(expected("montant"), [("montnat", "text")]).columns[0]
    assert alone.status == "missing"


def test_normalize_and_distance() -> None:
    assert normalize(" Chiffre d'Affaires_TTC ") == "chiffred'affairesttc"
    assert normalize("Région-Nord") == normalize("region nord")
    assert distance("montant", "montnat") == 2
    assert distance("montant", "montant") == 0
    assert distance("montant", "prix") > 2


# --- Schema ------------------------------------------------------------------------------------


def test_json_round_trip(recipe: Recipe) -> None:
    again = parse(recipe.to_json())
    assert again == recipe
    assert json.loads(recipe.to_json())["schema"] == 1


def test_a_recipe_exported_before_the_rename_still_opens(recipe: Recipe) -> None:
    raw = json.loads(recipe.to_json())
    assert raw["format"] == "quantilo/recipe"
    raw["format"] = "cyfouri-data-explorer/recipe"

    assert parse(json.dumps(raw)) == recipe


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (lambda r: r.update(colour="red"), "colour"),
        (lambda r: r["steps"][0].update(colour="red"), "steps[0].colour"),
        (lambda r: r.update(schema=2), "schema"),
        (lambda r: r.update(format="something"), "format"),
        (lambda r: r["steps"][0].update(id="__import__('os')"), "steps[0].id"),
    ],
)
def test_a_hand_edited_recipe_is_refused_with_the_field_named(
    recipe: Recipe, change: object, field: str
) -> None:
    raw = json.loads(recipe.to_json())
    change(raw)  # type: ignore[operator]

    with pytest.raises(RecipeError) as refused:
        parse(json.dumps(raw))

    assert refused.value.field == field
    assert field.split(".")[-1].split("[")[0] in refused.value.message


def test_unknown_field_nested_in_a_lenient_model_is_refused(recipe: Recipe) -> None:
    raw = json.loads(recipe.to_json())
    ml = next(step for step in raw["steps"] if step["kind"] == "ml")
    ml["validation"]["evil"] = 1

    with pytest.raises(RecipeError, match="evil"):
        parse(json.dumps(raw))


def test_not_json_is_refused() -> None:
    with pytest.raises(RecipeError, match="invalid JSON"):
        parse("{not json")
    with pytest.raises(RecipeError, match="JSON object"):
        parse("[1, 2]")


# --- Replay ------------------------------------------------------------------------------------


def test_the_recipe_keeps_the_work_as_steps(recipe: Recipe) -> None:
    kinds = [step.kind for step in recipe.steps]
    assert kinds == ["column", "column", "filter", "kpi", "kpi", "timeseries", "ml"]
    filter_step = recipe.steps[2]
    assert filter_step.active  # type: ignore[union-attr]
    assert all(getattr(s, "filter", None) == filter_step.id for s in recipe.steps[3:6])
    pays = next(c for c in recipe.columns if c.name == "pays")
    assert pays.categories == ["BE", "DE", "FR"]
    assert next(c for c in recipe.columns if c.name == "montant").categories is None
    assert "Treat indice as categorical" in [
        s.split(";")[0] for s in (p.sentence for p in plan(recipe, {}))
    ]


def test_replay_on_november_runs_every_step(recipe: Recipe, cache: Path) -> None:
    november = open_csv(sales_frame(2), cache, "novembre")

    report = recipe.run(november)

    assert [s.status for s in report.steps] == ["done"] * 7, [
        (s.sentence, s.message, s.warnings) for s in report.steps
    ]
    assert report.status == "done"
    assert november.column("indice").type == "categorical_nominal"
    kinds = sorted(f.kind for f in report.figures)
    assert kinds == ["forecast", "kpi", "kpi", "model"]
    assert all(f.value is not None for f in report.figures)
    revenue = next(f for f in report.figures if f.label == "Revenue")
    rows = november.unfiltered().where("pays", "in", ["FR", "BE"]).to_pandas()
    assert revenue.value == pytest.approx(rows["montant"].sum())


def test_key_figures_next_to_the_previous_run(recipe: Recipe, cache: Path) -> None:
    first = recipe.run(open_csv(sales_frame(3), cache, "m3")).figures
    second = recipe.run(open_csv(sales_frame(4), cache, "m4")).figures

    changes = compare_figures(second, first)

    revenue = next(c for c in changes if c.label == "Revenue")
    assert revenue.previous == next(f.value for f in first if f.label == "Revenue")
    assert revenue.change == pytest.approx(revenue.value - revenue.previous)  # type: ignore[operator]
    assert revenue.change_pct == pytest.approx(revenue.change / abs(revenue.previous))  # type: ignore[operator]


def test_renamed_and_missing_columns(recipe: Recipe, cache: Path) -> None:
    frame = sales_frame(5).rename(columns={"montant": "Montant", "pays": "pyas"})
    november = open_csv(frame.drop(columns=["nb_articles"]), cache, "renamed")

    matching = recipe.match(november)
    found = {m.recipe: m for m in matching.columns}
    assert found["montant"].status == "matched"
    assert found["pays"].status == "proposed"  # a typo: waits for confirmation
    assert found["nb_articles"].status == "missing"

    report = recipe.run(november)  # proposal not accepted: pays is missing
    status = {s.id: s for s in report.steps}
    assert status["s2"].status == "skipped"
    assert "nb_articles" in (status["s2"].message or "")
    assert status["s3"].status == "skipped"
    for key in ("s4", "s5", "s6"):
        assert status[key].status == "skipped"
        assert "filter" in (status[key].message or "")
    assert status["s1"].status == "done"
    assert status["s7"].status == "warning"  # trained, without two of its columns
    assert "missing" in status["s7"].warnings[0]

    accepted = recipe.run(november, {"pays": "pyas"})
    assert [s.status for s in accepted.steps if s.kind == "kpi"] == ["done", "done"]
    revenue = next(f for f in accepted.figures if f.label == "Revenue")
    assert revenue.value is not None


def test_a_date_correction_that_no_longer_reads_is_not_applied(cache: Path) -> None:
    frame = pd.DataFrame({"jour": [f"2024-01-{d:02d}" for d in range(1, 31)] * 2, "x": range(60)})
    frame.loc[:23, "jour"] = [f"not a date {i}" for i in range(24)]  # 40 %
    ds = open_csv(frame, cache, "dates")
    assert ds.column("jour").type != "datetime"
    recipe = Recipe(
        name="Dates",
        columns=[
            RecipeColumn(name="jour", type="datetime"),
            RecipeColumn(name="x", type="numeric_discrete"),
        ],
        steps=[ColumnStep(id="s1", column="jour", type="datetime")],
    )

    report = recipe.run(ds)

    step = report.steps[0]
    assert step.status == "warning"
    assert "40%" in (step.message or "")
    assert ds.column("jour").type != "datetime"


def test_new_categories_are_flagged(recipe: Recipe, cache: Path) -> None:
    frame = sales_frame(6)
    frame.loc[:40, "pays"] = "IT"
    report = recipe.run(open_csv(frame, cache, "italy"))

    filter_step = next(s for s in report.steps if s.kind == "filter")
    assert filter_step.status == "warning"
    assert any("1 new category in pays: IT" in w for w in filter_step.warnings)


def test_cancelling_keeps_the_steps_done(recipe: Recipe, cache: Path) -> None:
    from quantilo.recipes.library import DatasetTarget
    from quantilo.recipes.run import run

    november = open_csv(sales_frame(7), cache, "cancel")
    calls = {"n": 0}

    def cancelled() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    report = run(recipe, DatasetTarget(november), cancelled=cancelled)

    statuses = [s.status for s in report.steps]
    assert statuses[:3] == ["done", "done", "done"]
    assert set(statuses[3:]) == {"cancelled"}
    assert report.status == "cancelled"


def test_turned_off_steps_and_skipped_kinds(recipe: Recipe) -> None:
    off = recipe.model_copy(deep=True)
    off.steps[0].enabled = False
    planned = plan(off, {c.name: c.name for c in recipe.columns}, {"ml": "Not here."})
    assert planned[0].skip == "Turned off in the recipe."
    assert planned[-1].skip == "Not here."
    assert all(p.skip is None for p in planned[1:-1])


# --- Python export -----------------------------------------------------------------------------


def test_the_exported_script_gives_the_same_key_figures(
    recipe: Recipe, cache: Path, tmp_path: Path
) -> None:
    path = cache / "novembre_script.csv"
    sales_frame(8).to_csv(path, index=False)
    report = recipe.run(dx.open(path, cache_dir=tmp_path / "app"))
    script = tmp_path / "recipe.py"
    script.write_text(to_python(recipe, str(path)), encoding="utf-8")

    done = subprocess.run(
        [sys.executable, str(script), str(path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={**__import__("os").environ, "QUANTILO_CACHE": str(tmp_path / "py")},
    )
    printed = json.loads(done.stdout[done.stdout.index("{") :])

    assert set(printed) == {f.id for f in report.figures}
    for figure in report.figures:
        assert printed[figure.id]["value"] == pytest.approx(figure.value), figure.label


def test_script_quotes_every_name_and_value() -> None:
    evil = 'a"\nimport os; os.system("x") #'
    recipe = Recipe(
        name='Bad """ name\nimport os',
        columns=[RecipeColumn(name=evil, type="numeric_continuous")],
        steps=[ColumnStep(id="s1", column=evil, type="categorical_nominal")],
    )

    code = to_python(recipe, "x.csv")

    tree = ast.parse(code)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    retype = next(c for c in calls if getattr(c.func, "attr", "") == "retype")
    assert isinstance(retype.args[0], ast.Constant) and retype.args[0].value == evil
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.Import | ast.ImportFrom)]
    assert sorted(a.name for n in imports for a in n.names) == ["json", "quantilo", "sys"]


def test_a_recipe_keeps_the_forecast_drivers(tmp_path: Path) -> None:
    # K11 (040): drivers (028 A4) came after recipes and were lost on replay
    sample = Path(__file__).resolve().parents[3] / "samples" / "sales_timeseries.csv"
    ds = dx.open(sample, cache_dir=tmp_path)
    drivers = [{"column": "units"}, {"column": "promo", "kind": "flag"}]
    analysis = {
        "time": "date",
        "value": "revenue",
        "period": "week",
        "horizon": 4,
        "models": ["naive", "tslm"],
        "drivers": drivers,
    }
    recipe = dx.Recipe.from_dataset(ds, "Weekly revenue", timeseries={"Revenue": analysis})
    step = next(s for s in recipe.steps if s.kind == "timeseries")

    assert [d.column for d in step.analysis.drivers or []] == ["units", "promo"]
    again = parse(recipe.to_json())
    assert any(s.kind == "timeseries" and s.analysis.drivers for s in again.steps)
    report = recipe.run(ds)
    assert all(s.status in ("done", "warning") for s in report.steps)
    script = recipe.to_python(str(sample))
    assert "drivers=[" in script and "'units'" in script and "'flag'" in script
