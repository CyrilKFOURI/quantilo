# quantilo

The engine of [Quantilo](https://github.com/CyrilKFOURI/quantilo), a data analysis app, for notebooks and scripts: column typing, profiles, the right test for every pair of variables, correlations with graded insights, time series, machine learning, segments and unusual rows, KPIs, synthetic data and the HTML report. The same code computes the app's results, so the numbers match to the last digit.

```python
import quantilo as dx

ds = dx.open("titanic.csv")
ds.types                       # detected types, roles and the reasons
ds.variable("Age")             # statistics, normality tests, outliers
ds.compare("Age", "Survived")  # the test follows the types, with an effect size
ds.correlations()              # mixed association matrix, ranked pairs, insights
ds.report("titanic.html")      # the app's report: one self-contained file
```

## Install

From the repository (not on PyPI yet):

```bash
pip install -e packages/quantilo             # the core
pip install -e "packages/quantilo[ml]"       # + XGBoost, LightGBM, Optuna
pip install -e "packages/quantilo[deep]"     # + PyTorch: CTGAN, TVAE, TimeGAN
pip install -e "packages/quantilo[plot]"     # + plotly: the app's charts
```

On another computer, straight from GitHub (the repository is private: be signed in to GitHub, for example with `gh auth login`):

```bash
pip install "quantilo[ml,plot] @ git+https://github.com/CyrilKFOURI/quantilo.git@feature/python-library#subdirectory=packages/quantilo"
```

Add `--upgrade` to get the latest version. The notebooks in `examples/` read the sample files of `samples/` at the repository root.

Python 3.12 or later. The core needs DuckDB, pyarrow, NumPy, SciPy, statsmodels, scikit-learn and pydantic; no server, no database.

On Windows, results use statistical notation (ρ, η, χ², ≥, →) that the old console encoding (cp1252) cannot print: `print()` then raises `UnicodeEncodeError` in a script. Run Python in UTF-8 mode, once for all scripts with `setx PYTHONUTF8 1`, or per run with `python -X utf8 script.py`. Jupyter is not affected.

## Open data

```python
ds = dx.open("sales.csv")                   # CSV, TSV, Parquet, Excel, or a folder of them
ds = dx.open("export.csv", delimiter=";", decimal=",", encoding="latin-1")
ds = dx.open("big.parquet", sample_pct=10)  # a reproducible 10 % sample
ds = dx.from_pandas(frame, "customers")     # or dx.from_arrow(table)
```

The CSV dialect is detected (separator, decimal comma, encoding). The source is copied once to Parquet in a cache folder (`~/.cache/quantilo`, `%LOCALAPPDATA%` on Windows, or `QUANTILO_CACHE`); opening an unchanged file again reuses the copy and the detected types.

## Types

Every column gets a type: `numeric_continuous`, `numeric_discrete`, `categorical_nominal`, `categorical_ordinal`, `boolean`, `datetime`, `text`, `identifier` or `ignored`. The type is read from a knowledge base of names and value checks: French decimals, `oui/non`, `21/02/2019`, `5%`, ZIP codes. Each detection says why it chose that type.

```python
ds.types                                    # every column (dx.to_pandas(ds.types) for a table)
ds.column("zip_code").reasons
ds.retype("Pclass", "categorical_ordinal")  # same rules as the app's Columns page
ds.set_order("size", ["S", "M", "L"])
ds.exclude("Name", "Ticket")                # out of the overview, correlations and report
ds.reset_types()
```

## Analyses

```python
ds.overview()                         # rows, types, missing cells, duplicates, quality alerts
ds.variable("Fare")                   # describe, normality (5 tests), KDE, ECDF, QQ
ds.fit("Fare")                        # probability laws ranked by AIC, transformations
ds.outliers("Fare", method="iqr")     # IQR, 3×IQR, z-score, robust z, and the rows
ds.compare("Fare", "Pclass")          # Welch / Mann-Whitney / ANOVA / Kruskal… + effect size
ds.compare("Age", "Fare", color="Sex")
ds.correlations(method="spearman")
ds.rows(limit=20, mode="duplicates")  # first, sample, duplicates, missing
ds.to_pandas()                        # every row, each column in its type
ds.export("clean.parquet")            # CSV, Parquet or Excel
```

## Filters

`where()` returns a filtered view and leaves `ds` alone. Conditions add up (AND); the values of `in` combine with OR. The operators are the app's filter bar: `eq`, `gt`, `gte`, `lt`, `lte`, `between`, `in`, `contains`, `starts_with`, `is_null`, `part_in`, `unique`, `duplicated`.

```python
women = ds.where("Sex", "in", ["female"])
adults = women.where("Age", "between", (18, 65))
adults.variable("Fare")
len(adults)
```

## Time series

```python
series = ds.timeseries(time="date", value="sales")  # period, aggregation, season: automatic
series.setup()                                      # the choices made, and why
series.overview()                                   # trend (Mann-Kendall, Sen's slope), rolling mean and spread
series.decomposition()                              # stl, mstl, classical_additive, classical_multiplicative
series.stationarity()                               # ADF, KPSS, Phillips-Perron and the suggested differencing
series.autocorrelation()                            # ACF, PACF, Ljung-Box, Durbin-Watson
series.volatility()
series.breaks()
series.residuals()
series.forecast(horizon=12)                         # naive models, ETS, ARIMA… ranked by MASE on rolling origins
series.drivers()                                    # columns that move with the series, future values in the file
series.forecast(horizon=12, models=["sarimax", "tslm", "gbm_lags"],
                drivers=["visits",                              # past only: lagged by the horizon
                         dx.Driver("price", known=True),        # future values from the file's future rows
                         dx.Driver("weekday", kind="calendar"),
                         dx.Driver("holiday", kind="flag")])    # each driver judged with / without it
series.scenario([dx.Driver("price", known=True)], {"price": [9.9] * 12})  # what if the price changes
series.scenario([dx.Driver("price", known=True)],
                changes=[dx.ScenarioChange("price", "percent", -10, start="2025-12", end="2025-12")])

stores = ds.timeseries("date", "montant", group="magasin")
stores.groups()                                     # how many series, the hierarchy above (store → region)
stores.forecast_groups(horizon=3, reconcile="mint_shrink")  # best model per store, totals that add up
stores.forecast_groups(global_model=True, limit=200)        # + one LightGBM over every series; 200 largest
```

## Machine learning

```python
ds.ml_targets()                         # every column as a target: regression, binary, multiclass
ml = ds.ml("Survived")
ml.preview()                            # the automatic preparation, column by column, and why
model = ml.train()                      # the app's first model: linear or logistic
model = ml.train("random_forest", settings={"n_estimators": 300})
model.score, model.scores               # held-out metrics
model.explain()                         # permutation importance, SHAP, partial dependence
model.predict(new_rows)                 # a Dataset or a pandas DataFrame
ml.compare()                            # one model per suitable family, best first
ml.tune("xgboost", budget_seconds=120)  # Optuna (the ml extra)
found = ml.auto(minutes=5)              # baselines, families, tuning, ensemble: found.leaderboard, found.model
```

Where a model is wrong, on its held-out predictions only (test rows, or out of fold):

```python
errors = model.errors()                 # "misclassified" / "log_loss" for classes
errors.slices                           # subgroups notably worse, in sentences, with their filters
errors.to_pandas()                      # one row per subgroup: rows, error, ratio, q-value (BH)
errors.by_column, errors.worst          # error per category or fifth; the 50 worst predictions
errors.hints                            # more data, an excluded column, the log, class weights
model.compare_errors(other)             # where another run fixes or worsens the errors
model.fairness("sex")                   # metrics side by side for each group (display only)
dx.plot(errors, column="age")           # the error along one column
```

On new data, as the app's *Predict on new data*:

```python
model.check(new_rows)                   # Ready, Check or Blocked: missing columns, types, unknown categories, ranges
model.predict(new_rows, keep=["id"])    # predicted, p(class) or a 90 % interval, less_reliable and why
model.prediction(new_rows)              # + the scores when new_rows has the target, the distribution
model.export("churn_lightgbm.zip")      # model.joblib + model_card.json + predict.py, no data rows
model = dx.load_model("churn_lightgbm.zip")   # checks the card first; warns when versions differ
```

Panel data (the same entities at successive periods: ids × eras, shops × weeks) is validated walk-forward by period:

```python
ml = ds.ml("target", validation={"method": "panel_walk_forward", "period_column": "era", "embargo": 4})
ml.panel()                              # period, entity, periods, the embargo the target calls for
model = ml.train()                      # model.panel: score per period, summary, folds, exposures, alerts
model.panel.summary                     # mean, std, mean / std, share of good periods, worst, longest bad streak
model.neutralize(["feature_1"], 1.0)    # remove the part a column explains, score every period again
dx.plot(model, "periods")               # also "cumulative" and "folds"
## Segments and unusual rows

```python
ds.segments().plan()                    # columns used or left out, and why; the method Auto takes
found = ds.segments().fit()             # k = 2 to 10 tried, the k the scores agree on, stability
found.k, found.choice.why               # "4 segments: best silhouette, and the elbow is at 4."
found.segments[0].traits                # "income: 2.1× the average", "plan: 78 % are Premium…"
found.to_pandas()                       # the profile table: one column per segment
found.to_dataset()                      # the rows with a `segment` column, as a new dataset
ds.segments(method="kprototypes", k=3)  # also kmeans, gmm, ward, hdbscan
odd = ds.unusual_rows()                 # Isolation Forest, Mahalanobis (MCD), LOF, or "segment"
odd.threshold, odd.listed[0].reasons    # where the scores break; why each row is unusual
```

## KPIs

```python
suggested = ds.kpi_suggestions()
ds.kpi(suggested.kpis[0].definition)  # grain, totals and comparisons chosen by rules
```

## What changed between two versions

```python
october, november = dx.open("sales_2026_10.csv"), dx.open("sales_2026_11.csv")
changes = november.changes(october)  # schema (guessed renames), quality, dates, distributions
changes.headline                     # "Compared with sales_2026_10: +4,812 rows (+3.1%), …"
changes.level                        # none, check or breaking
dx.to_pandas(changes)                # PSI, normalised Wasserstein, KS, χ² per column
november.changes(october, used={"amount": ["Revenue KPI"]})  # a used column gone: breaking
november.row_changes(october, key="order_id")  # added, removed, modified rows (both files)
november.kpi_changes(october, {"Revenue": definition})   # periods in both files: revisions
november.relation_changes(october)   # the 10 strongest relations, before and after
november.snapshot()                  # the photo the app keeps of every version
```

PSI is read on the old version's deciles (categories: on the values), empty bins floored at 1e-4: below 0.1 stable, 0.1 to 0.25 moderate, above 0.25 strong.

## Experiments (A/B tests)

```python
ds.experiment_candidates()              # variant columns (2 to 5 values), the control guessed, metrics
exp = ds.experiment(variant="group", metrics=["converted"], unit=None)
result = exp.analyze()                  # SRM check, z-test / Welch / Poisson / delta method, Holm
result.metrics[0].comparisons[0].sentence   # "B reaches 12.4 % vs 11.1 % for A: +1.3 points…"
ds.experiment("group", [{"column": "revenue", "denominator": "orders"},   # a ratio of sums
                        {"column": "spend", "pre": "spend_before"}],      # CUPED
              unit="user_id", segment="device", time="date").analyze()
dx.sample_size(baseline=0.10, effect=0.05)   # units per group to detect a +5 % lift on 10 %
dx.sample_size(kind="mean", baseline=50, sd=20, effect=2, relative=False, daily=1_000)
```

## Recipes

```python
october.retype("code_postal", "categorical_nominal")
recipe = dx.Recipe.from_dataset(
    october.where("pays", "in", ["FR", "BE"]),   # its conditions become the active filter
    "Monthly sales",
    kpis={"Revenue": {"measures": [{"id": "m1", "agg": "sum", "column": "montant"}]}},
    timeseries={"Sales": {"time": "date", "value": "montant", "period": "month", "horizon": 3}},
    models=[october.ml("churn").train("lightgbm")],
)
recipe.match(november)              # same name, matched (case, accents, _), proposed (≤ 2 edits)
report = recipe.run(november)       # step by step: done, warning, skipped (why), failed
report.figures                      # KPI values, next forecast value, model scores
open("monthly_sales.py", "w").write(recipe.to_python())   # the same with this library
dx.Recipe.from_json(recipe.to_json())   # the app's .recipe.json (versioned, validated)
```

A type correction that leaves more than 10 % of a new file's values unreadable is not applied (a warning); a step whose column is missing is skipped, and so are the steps on the rows of a filter that could not be applied.

## Synthetic data

```python
synth = ds.synthetic()
synth.plan()              # per column: model, regenerate, fake or leave out
made = synth.generate(
    method="cart", rows=10_000
)                         # also copula, bayes, independent, ctgan, tvae, timegan
made.report.scores        # diagnostic, quality, detection, privacy
made.utility("Survived")  # train on synthetic, test on real
made.to_dataset().compare("Age", "Fare")
made.save("synthetic.parquet")
```

## Charts

The app's charts, drawn by the same rules and in its colours, as interactive plotly figures (the `plot` extra). In a notebook, the figure shows in the cell.

```python
ds.plot("Fare")                         # the column's card: histogram, Top N bars, timeline…
ds.plot("Fare", "laws")                 # also box, kde, ecdf, violin, qq; pie for a category
dx.plot(ds.compare("Fare", "Pclass"))   # the Compare chart; views: box, violin, means
dx.plot(ds.correlations())              # the association matrix, clustered
dx.plot(series.decomposition())         # every time series section, forecast included
dx.plot(model, "roc")                   # predicted, residuals, confusion, roc, pr, calibration…
dx.plot(model.explain(), "shap")        # permutation, shap, dependence, waterfall
dx.plot(made, "pca")                    # synthetic: mean, std, pca, links, dcr, scores
dx.plot(ds.kpi(definition), mode="change_pct")   # "share": the % of total under it
dx.plot(result)                         # an experiment's intervals; "cumulative" over time
dx.plot(dx.sample_size(baseline=0.1, effect=0.05))   # units per group ↔ detectable lift
dx.plot([made, other])                  # several synthetic runs, scores side by side
dx.plot(changes, column="amount")       # a column before and after (default: the biggest shift)
dx.plot(found, "scores")                # segments: map, tsne, scores, stability, dendrogram
dx.plot(odd, threshold=0.6)             # unusual rows: the sorted scores and the threshold
dx.plot(result, theme="dark", title="…", height=400)
```

A view that does not fit the result says which ones do. `fig.write_html("chart.html")` saves a figure as an offline page.

## Results

Results are dataclasses, the same ones the app's API returns. Two helpers turn them into plain data:

```python
dx.to_dict(ds.variable("Age"))   # dicts and lists, JSON-ready (NaN → None)
dx.to_pandas(ds.correlations())  # the association matrix as a DataFrame
```

`to_pandas` knows rows, the association matrix, per-column statistics, laws, outliers, tests, targets, models, permutation importance, forecasts, KPI values, the synthetic plan, segment profiles, unusual rows and the types.

Every error is a `dx.ExplorerError` with a message you can act on. A mistyped column suggests the closest names:

```python
ds.variable("Agee")  # ColumnNotFound: Column 'Agee' not found; did you mean 'Age'?
```

## License

MIT
