# User guide

Quantilo is a data analysis app: a local, offline tool to explore a table of data (CSV, Parquet or Excel) with real statistics: the right chart and the right test for each column, explained in plain words. Nothing leaves your machine, and your original file is never modified: the app works on its own copy.

## Quick start

1. **Home** → drop a file, choose one, or browse to a file or a folder (all files of a folder with the same columns are stacked into one dataset). Demo datasets are listed under *Samples*.
2. Check the import preview (separator, encoding, header row, decimal separator) and click **Import**.
3. Open **Columns** to check the type detected for each column, then **Validate types**.
4. Explore: **Overview** (every variable at a glance), a **variable page** (deep dive into one column), **Compare** (two or three variables), **Correlations** (every pair at once), **Data** (the rows themselves, and exports).
5. Press **F** at any time to filter the rows: the filter applies to every page.

## Pages

### Home

- The list of your datasets: rows, columns, size, last opened. Rename, re-import (instant when the source file hasn't changed) or delete a dataset from its menu.
- Tick two datasets and click **Compare versions** to see what changed between them (see *Changes*).
- Import a **file** or a **folder**. Large files are converted once to Parquet; after that, every page reads the copy, which is fast even on millions of rows.

### Columns

Each column gets an **analysis type**, which decides which charts and statistical tests apply:

| Type | Meaning | Shown as |
|---|---|---|
| Continuous | A measured quantity (price, weight, duration) | Histogram, density, box plot; mean, quantiles, normality tests |
| Discrete | Few whole numbers (counts, ratings) | Bars per value |
| Category | Labels without order (country, product) | Top values, frequencies |
| Ordered category | Labels with an order (low < medium < high) | Bars in that order; edit the order with *Edit order* |
| Boolean | Two states (yes / no) | Two bars |
| Date / time | Moments in time | Volumes per period; date parts (year, month, weekday…) |
| Free text | Sentences | Lengths and frequent words |
| Identifier | One value per row (ids, emails) | Checked for duplicates only |
| Ignored | Empty, constant or not useful | Left out of analyses |

- **Detection** explains why a type was chosen ("100 % of values read as numbers"). Columns marked *Check* or *Unsure* deserve a look.
- **Change a type** from its badge. If some values can't be read in the new type, the app tells you how many become empty before you confirm.
- **Use** (checkbox) includes or excludes a column from every analysis. **Undo** (Ctrl+Z) reverts the last change; *Re-run detection* starts over (your manual corrections are kept).
- The **role** (amount, city, email…) is a finer hint used for display; it doesn't change the statistics.

### Overview

- **Summary strip**: rows (filtered / total), columns, missing values, duplicate rows and columns, size.
- **Alerts**: things worth a look — heavily skewed variables, high cardinality, constant or mostly empty columns, unreadable values.
- **One card per variable**, three per row: its chart and key numbers. Sort (file order, name, most missing, most distinct, most skewed), search, choose the *Top N* categories shown (the rest is grouped as *Others*) and the number of cards per page.
- **Export report** (top right) downloads one HTML file about the whole dataset, after your filters: key findings ranked by importance, a table of every column at a glance, a card per variable (figures, distribution with density, box plot and QQ-plot for numbers, percentiles or most frequent values, what stands out, most related columns), the most strongly linked pairs drawn as in Compare with their test, the correlation matrix and insights, missing values, and a note on each method. It opens offline in any browser, holds no whole row (only counts, statistics and the most frequent values) and prints well to PDF. Large datasets take a while: the button shows the progress and can be cancelled.
- **Save as recipe** (top right) keeps what you did on this dataset to replay it on next month's file (see *Recipes*). When a recipe fits the file you just imported, a line on top offers to run it.
- On a card: **full screen** (every category, with a table of counts, % and cumulative %; for numbers, the full statistics and normality tests), **change the chart**, **download PNG**, **open the variable page**. Click a bar or a range to filter on it.

### Variable page

Everything about one column, in tabs. Use the arrows to go to the previous or next variable, and *Compare with…* to cross it with another one.

- **Summary** — counts (rows, missing, distinct), center and spread, quantiles, shape (see *Metrics* below). Charts: histogram (choose the number of bins), density, ECDF, box, violin, QQ-plot.
- **Normality** — five tests of "could these values come from a normal distribution?", with a verdict and the α level you choose.
- **Distribution** — candidate laws fitted to the data (normal, log-normal, exponential, gamma, Weibull…, Poisson for counts), ranked by AIC, drawn over the histogram; and transformations (log, square root, Box-Cox, Yeo-Johnson) previewed for symmetry. The data itself is never modified.
- **Outliers** — four rules to flag extreme values, the most extreme rows, and two buttons: *Show only these rows* / *Exclude them* (adds a filter).
- **Categories** (categories, booleans) — every value with count, %, cumulative %. **Dates** — range, granularity, rows per period, empty periods, rows per weekday and hour. **Text** — lengths and most frequent words. **Identifiers** — duplicates.

### Compare
- **Whole numbers with few values** (at most 12, like a class 1–3 or a number of children) are read as categories: box plots instead of a scatter. *As numbers* under the picker switches back.
- **Color on groups**: a number against a category can be split by a color of at most 6 values — box plots, violins or means side by side, one per color, with a legend. The tests still compare the groups, all colors together.

Pick an **X** and a **Y**: the chart and the statistical tests follow their types.

| X × Y | Chart | Tests |
|---|---|---|
| Number × number | Scatter with linear fit (or density for many rows), 3D with a third number | Pearson r, Spearman ρ, Kendall τ |
| Number × category | Box, violin, or means ± 95 % CI per group | Welch t-test / Mann-Whitney (2 groups); ANOVA, Welch ANOVA, Kruskal-Wallis (3+ groups); Levene for equal spreads. The *Recommended* test is chosen from normality and spreads |
| Category × category | Heatmap of counts, or stacked 100 % bars | χ² test of independence, Fisher's exact test (2×2), Cramér's V |
| Date × number | Time series: one aggregated value per period, rolling mean and ± 1σ band | Mann-Kendall trend test, Sen's slope |
| Date × category | Counts per period and category | — |

- **Color** splits a numeric pair or a series by a category. **Swap** exchanges X and Y.
- **Time series** (date × number): rows are always **aggregated per period first** — a rolling mean over raw rows would mean nothing when a date has several rows. Above the chart, change:
  - **Period** — *Auto* uses the usual gap between dates, coarsened beyond 400 periods (3 years of daily data → weeks);
  - **Aggregate** — *Auto* sums amounts and quantities (by their column role) and averages everything else; also median, min, max, count;
  - **Window** of the rolling mean and standard deviation, in periods (*Auto*: 7 days, 4 weeks, 3 months…), **Trailing** (only looks back) or **Centered**;
  - which layers to draw: values, rolling mean, ± 1σ band (single series only).
  An empty period is a gap — except for sums and counts, where it is 0. A first or last period only partly covered by the data (a week starting mid-week) is drawn but left out of the trend test. *Data used* lists every period with its value, rolling mean, rolling σ and number of rows.
- Each test shows its statistic, p-value, **effect size** with its magnitude (negligible, small, medium, large) and a sentence to read it. Statistics use every row; only the points drawn may be sampled (it is said under the chart).
- **Analyze as an experiment →** appears above the chart when one column has 2 to 5 values and the other is a number or a yes/no column: it opens Experiments with that column as the variant and the other as the metric.

### Time series

*Modules › Time series* (needs a date column) analyses **one series** in depth. It opens already set up: the first date, the first amount or quantity column (otherwise the first number), and the automatic period and aggregate of *Compare*. *Analyze as time series →* in *Compare* opens it on the same choices.

- **Settings bar** (stays at the top): Time, Value, Period, Aggregate, **Group** (one series per category, up to 8; *Series* picks the one analysed), **Gaps** — leave empty periods empty, interpolate them, fill them with 0 or carry the last value forward. Methods that need a complete series interpolate on their own and say so.
- **Sections**, each opening on its conclusion (also shown in the side contents):
  - **Overview** — span, number of periods, empty periods, change between the first and last quarter; the curve with its rolling mean and ± 1σ band, a range slider to zoom, and *Show all series* to overlay the groups. Above the chart, **Rolling means** lists each mean with the colour of its curve; **+ Add** suggests windows that suit the period (4, 13, 26, 52 weeks…) or takes any number, up to five means; **Position** sets trailing or centered windows; **Show** turns the values, the means and the ± 1σ band (around the first mean) on or off.
  - **Seasonality** — boxes by weekday, month, quarter or hour (whichever the period allows), with the label furthest from the average once the trend is removed; one line per year to compare the latest year with the previous ones.
  - **Decomposition** — trend, seasonal and remainder, stacked on one time axis, with **STL** (robust, default), **MSTL** (several cycles, e.g. weekly and yearly), or **classical** additive / multiplicative moving averages. *Season length* overrides the cycle (7 days, 52 weeks, 12 months…). The **strengths** of trend and seasonality run from 0 (none) to 1 (dominant): above 0.64 is strong.
- **Methods**: every section lists the methods it ran; **+ Add method** offers every method that answers the same question, and your choice is remembered. Each method shows its conclusion, p-value, statistic, what it estimates, and *What it does* with its null hypothesis.

| Section | Methods (★ = shown by default) |
|---|---|
| Trend | ★ Mann-Kendall · ★ Sen's slope · Mann-Kendall corrected for autocorrelation (Hamed–Rao) · Seasonal Mann-Kendall · Cox–Stuart · Spearman ρ with time · Linear regression on time |
| Seasonality | ★ Kruskal–Wallis between seasons (ε²) · ★ Periodogram (dominant cycle length) · Friedman · QS test · Fisher's g · Autocorrelation at the seasonal lag |
| Stationarity | ★ ADF · ★ KPSS · ★ Phillips–Perron · DF-GLS · Zivot–Andrews (one break) · Variance ratio · Range unit-root |
| Autocorrelation | ★ Ljung–Box · ★ Durbin–Watson · Box–Pierce · Breusch–Godfrey · Runs test · Turning points · Bartels rank test |
| Volatility | ★ Breusch–Pagan · ★ ARCH-LM · White · Goldfeld–Quandt · McLeod–Li · Levene · Brown–Forsythe · Bartlett · Fligner–Killeen |
| Breaks | ★ Pettitt · CUSUM · Chow · Buishand range · SNHT |

- **Residuals** are what is left once trend and seasonality are removed (the STL remainder; a straight line's residuals when the series is too short for two cycles). With fewer than 7 cycles, the seasonal part is estimated as a plain average per position of the cycle, so it does not swallow the noise.
- **Volatility** — the residuals and their rolling standard deviation (type its window, e.g. 10), and tests of whether the variance changes over time or comes in clusters (ARCH). When the spread grows with the level, a log transformation is suggested.
- **Residuals** section — the full distribution, exactly as on a variable page: shape statistics (skewness, excess kurtosis), QQ-plot against a normal law, the normality tests, and the fitted laws ranked by AIC with transformations. Choose what to analyse: **Residuals**, the **Series** itself, or its **Returns** (relative change from one period to the next).
- **Breaks & anomalies** — change points found by **PELT** (exact) or **binary segmentation**, on the series without its seasonality. *Change in*: the **mean**, the **mean and variance**, or the **trend** (a straight line per segment: a change of slope). *Auto* uses the trend when the series has one, because cutting a steady trend into mean levels gives a staircase of false breaks. *Sensitivity* sets how strong a change must be. **Anomalies** are periods whose residual is unusual: robust z-score, IQR rule, rolling z-score, Hampel filter, the **generalized ESD test** (a formal test at α = 5 %, strict), **isolation forest** and **local outlier factor** (unusual windows of residuals) or a **masked autoencoder** (a small neural network rebuilds each period from its neighbours; periods it cannot rebuild break the pattern); threshold 3, 3.5 or 4 (not used by the ESD test). Every anomaly is marked on the chart; the table lists the 5 strongest, with **Show all** for the rest. Point anomalies are listed here; *contextual* ones (normal values at the wrong time of year) are caught because residuals have the season removed; *collective* ones (a lasting shift) are the breaks above; *Show rows* opens the Data page filtered on that period.
- An incomplete first or last period (a week that starts mid-week) is left out of every analysis when values are summed or counted: its total is too low.
- **Forecast** — runs when you click **Run forecast** (it fits each model several times). Choose the **horizon** (default: 30 days, 12 weeks, 12 months, 8 quarters… at most a quarter of the series) and the models: ★ **Naive**, ★ **Seasonal naive**, ★ **ETS (automatic)** (every exponential-smoothing form, lowest AICc), ★ **ARIMA (automatic)** (differences from the stationarity tests, then a stepwise search of the orders), plus Drift, Mean, Simple exponential smoothing, Holt, Holt-Winters, Theta, **SARIMA** with your own orders, **linear regression** on a trend and season indicators, and machine learning on the previous periods (**gradient boosting**, **neural network**; their intervals are *conformal*: taken from their own validation errors). For series with **many zeros** (sales of a slow product, tickets, breakdowns), the forecast says so, classifies the demand (smooth, erratic, intermittent, lumpy, from the average gap between non-zero values and the variation of their sizes) and offers the models built for it: **Croston**, **Croston corrected (SBA)**, **TSB** (follows a demand that stops) and **ADIDA** (aggregate, then spread). **Combination (best 3)** averages the three models that validated best; it is scored without hindsight (at each origin it only knows how the models did before). Each model is scored on several **rolling origins**: it forecasts periods it has not seen, from several starting points. The table ranks the models by **MASE** (error relative to the naive reference: below 1 beats it) with MAE, RMSE, the gain over the reference, and the **coverage** of the 95 % interval (*All metrics* adds MAPE, sMAPE, **WAPE** — the share of the volume missed, usable with zeros —, RMSSE, bias, 80 % coverage and the Winkler interval score). **How the models were scored** draws, for the chosen model, each forecast it made from an earlier date next to what really happened. The chart shows the history, the chosen model's forecast and its 80 % and 95 % intervals; click a model to draw it, *Show the other models* to overlay them, *Export CSV* to download the forecast. A Ljung–Box check says whether the model's residuals are white noise (the model captured the structure). You can cancel a running forecast. **Drivers** (the panel under the methods, closed until you open it) add outside variables that may explain the series: another numeric or yes/no column, summed or averaged per period like the series (a yes/no column becomes the share of rows that are yes), a **0/1 flag** you provide (holidays, events), or a **calendar** variable in one click (day of week, month, end of month, when they make sense for the period). For each column, say whether it is **known in advance** (a planned price, a scheduled promotion) — its future values then come from rows of the file dated after the series (a date and the driver, the series left empty), a constant, or its last value repeated (with a warning) — or **past only** (visits): it then enters lagged by the horizon, so the forecast never uses a value it could not have known. **SARIMAX** (the automatic ARIMA's orders, refitted with the drivers), the **linear regression** and **gradient boosting** use them; the other models ignore them, and the panel says so (and offers to add SARIMAX when none of the chosen models uses drivers). When another column moves with the series (correlation of their period-to-period changes of 0.5 or more), a line suggests it: *Add driver*. After the run, each driver model shows its MASE without drivers, and the **Drivers, model by model** table gives each driver's coefficient (or importance for gradient boosting), its p-value, the MASE without it on the same rolling origins and a verdict: *Helps* (at least 5 % lower MASE with it), *Unclear* or *Does not help*. In the validation a known-in-advance driver uses its real past values (except *last value repeated*, which is repeated from each origin too), so its scores are an upper bound when its future values are guesses. In the browser-only version SARIMAX is locked like ARIMA; the regression and gradient boosting take drivers.
- **Save analysis** (under the page description) keeps the current setup — time, value, group, period, aggregate, gaps, season — with the forecast models, horizon and drivers, under a name; **Saved (N)** reopens one (drivers included) or deletes it. Saved analyses are the time series steps of a recipe (see *Recipes*).
- **Scenarios (what if)** — once a forecast has a driver **known in advance** (a price, a promotion, a budget), the **Scenarios** panel under the results changes its future values: **± x %** or **set to** a value, over every forecast period or **from / to** chosen periods ("price −10 % in December"). Pick the model (SARIMAX, linear regression or gradient boosting; the others are greyed out: they do not use drivers). The chart draws the base forecast and each scenario, and a table gives each scenario's total over the horizon and its gap with the base. With the linear regression or SARIMAX, the periods you did not change keep exactly the base forecast. Up to 3 named scenarios, kept in the page's link (reload it or share it to find them again).
- **Every group** — when a **Group** is chosen, the Forecast section offers **Forecast every group**: every value of the group column (every store, product, country) gets its own forecast with the ticked methods (★ Seasonal naive, ETS and Combination by default; Croston-type models are added to series with intermittent demand; ARIMA only with *Include ARIMA (slow)*; drivers are not used here), each scored on its own rolling origins, and the best model is kept **per series**. Beyond 200 series (the largest by volume), the others are pooled into **Others** and forecast as one series, which the result says. A series with less than two cycles or 12 periods of history gets the seasonal naive forecast and a *short history* tag. Series run in parallel, one process per core (16 at most; one after the other in the browser version, 30 series at most); a progress bar counts them and you can cancel. **Global model** adds one gradient boosting (LightGBM) trained on every series at once (lags, rolling means, position in the cycle, the series as a category) — often better when there are many short series; it replaces a series' best model where it scores better, and the result says for how many. **Totals** make the forecasts add up: the app recognises a **hierarchy** when each value of the group belongs to a single value of another column (store → region → Total), shown as a chain you can edit, then **Bottom-up** (parents = sum of their children), **Top-down** (the total split by each series' historical share) or **MinT** (the combination with the smallest total error, from the covariance of the models' errors; *Auto* picks it when the history is long enough, bottom-up otherwise). After reconciliation every parent equals the sum of its children at every date; a table compares each level's MASE before and after. Only for sums and counts. The results: a sentence ("ETS is best for 41 % of stores"), the best model by number of series, small charts of the 12 largest series (history and forecast with its 95 % band), and a table of every series — volume, demand pattern, best model, MASE, WAPE, total forecast over the horizon, change against the same periods a year earlier — with sorting and search. Click a series to open its usual detailed forecast. **Export CSV** downloads every forecast in long format (series, level, period, forecast, 80 % and 95 % bounds).

- **Stationarity** — does the series keep a stable level and spread? ADF (H0: unit root) and KPSS (H0: stationary) ask opposite questions, so together they conclude: *stationary*, *not stationary*, *trend- or difference-stationary*, or *inconclusive*. Pick a **transformation** — log, difference, seasonal difference, log + difference, difference ×2, seasonal + difference — and the tests run on it; the simplest one that makes the series stationary is marked **Suggested**. With a strong trend, the tests are run around the trend.
- **Autocorrelation** — on the series as transformed above: the **ACF** (correlation with the value k periods earlier) and **PACF** (the same once shorter lags are accounted for; Yule-Walker, OLS or Burg), with the significance band ± 1.96/√n, and a reading of their shapes: slow decay → trend or unit root; spikes every cycle → seasonality; PACF stopping after lag p → an AR(p) model is a good start; nothing beyond chance → white noise. The **AR(p) model** is fitted with p chosen by AIC — or the order you pick — with its coefficients, the AIC / BIC of every order, and a Ljung–Box check of its residuals.

### Machine learning

*Modules › Machine learning* trains a model in one click. The only choice is **what to predict**: every column is listed with the problem it defines — a **number** (regression), **two classes** or **several classes** — or why it cannot be predicted (identifiers, free text, dates, mostly empty columns, categories with almost one value per row). Suggested targets come first.

- Choosing a target **trains a first model at once**: linear regression for a number, logistic regression for classes. Change its **essential settings** on the left (regularization: none, Ridge or Lasso and its strength; for classes, the strength C and *Balance classes*, on by default when the rarest class is under 20 %), then click **Train** again. Each training is kept in **Runs** (★ = best score): click one to see it again.
- **Preparation** is automatic and explained: each column gets decisions from named rules — left out (the target itself, identifiers, free text, constant or mostly empty columns, columns that *look like they contain the answer*, near-duplicates of another number), missing values filled (mean or median, plus a “was missing” indicator above 5 %), skewed numbers log-transformed or Yeo-Johnson, numbers standardised (robust scaling with many outliers), categories one-hot encoded (up to 15) or target-encoded, ordered categories coded in the order set in Columns, dates split into year, month, weekday… (month, weekday and hour as sine / cosine). Everything that learns from the data is fitted on the training rows only.
- **Model** opens a searchable list of every model, by family: linear models (OLS, ridge, lasso, elastic net, Huber, quantile, Bayesian, Poisson / Gamma / Tweedie, logistic, SGD…), discriminant analysis, nearest neighbours, support vector machines, Naive Bayes, trees, random forest and extra trees, boosting (histogram gradient boosting, gradient boosting, AdaBoost; XGBoost, LightGBM and CatBoost when installed), neural network (MLP), kernel ridge and Gaussian process. *Recommended for this data* comes first, with the reason. Models too slow for the data are greyed out with why.
- Under the model, its **essential settings**; **All settings** opens a side panel with every setting, grouped, each with why its default and its technical name. Turn on **Expert** for the optimizer (solver, learning rate, momentum, tolerance…). Options that cannot go together are greyed out with the reason (e.g. an L1 penalty needs the liblinear or saga solver).
- **Validation**: a train / test split (default 80 / 20), k-fold or repeated k-fold cross-validation, group k-fold (rows of one group never on both sides), a time-based split (test on the most recent rows), leave-one-out or panel walk-forward (below); the primary metric; the seed; and how many rows to train on. With cross-validation, every row is predicted by a model that had not seen it, scores read *mean ± standard deviation* over the folds, and the saved model is retrained on every row. *All metrics* lists every metric of the problem.
- **Panel data** (the same entities seen period after period: customers × months, shops × weeks, ids × eras): when a column of periods (a date, or a growing whole number such as `era`) comes back on many rows, often with an entity column (an id seen at each period), a banner says so — “This looks like panel data: 574 eras × ~5,000 ids. Random splits would overestimate the score.” — because random rows share their entity and their period with the test rows. **Use panel validation** picks **Panel walk-forward**: periods in order, each fold (3 to 10) trains on earlier periods and is tested on the next block, with an **embargo** of periods left out in between. When the target looks ahead (a 20-day return on weekly rows), the embargo it calls for is proposed with the reason. The window is *expanding* (every past period) or *rolling* (the last N); large panels train on whole periods only (1 period in 4 beyond 200 periods by default), and a row sample keeps whole periods too. The period and the entity are left out of the features. Numbers are then judged by the **Spearman correlation per period** by default (Pearson and Spearman are now regression metrics too), and the run's score is the **mean**, the **mean / std** (steadiness) or the **worst period**, also used by *Compare models* and *Tune*.
  - The **Periods** tab gives the mean, standard deviation, mean / std, share of good periods (above 0 for a correlation, 0.5 for an AUC), worst period and longest bad streak; the score of every period (bars) and their running total (line); the folds on the time axis; *Export the periods (CSV)*; and the panel checks: a column from the future (far more linked to the next period's target than to the current one, or only filled in the last periods), a mean score far above what the 3 best columns give alone, an embargo shorter than the target's horizon.
  - **Dependence on single columns**: the correlation of the predictions with each number column, period by period; the largest and the 5 most followed columns. **Neutralize** removes from the predictions the part the ticked columns explain (0 to 100 %), period by period, then scores every period again, side by side with the run: a steadier model can be worth more than a stronger one on average.
- **Long trainings**: *How long will it take?* times the model on a sample and extrapolates; *Cancel* stops a training at its next step (nothing is saved). Iterative models (boosting, neural network) show their **learning curve**: training and held-out scores, iteration by iteration, with the one kept by early stopping. For two classes with a split, the **decision threshold** can be chosen on the training rows (best F1 or best balanced accuracy) instead of 0.5. Boosting models accept **monotonic constraints** (“size:+1, age:-1”): the prediction may only rise with size and only fall with age.
- **Compare models** trains a selection of models (one per suitable family by default) with their defaults on the same rows and validation, and ranks them; each becomes a run you can open. Tick two runs or more in **Runs** to see them side by side: every metric, the settings that differ, and the predictions (or ROC curves) overlaid. Only runs with the same target, classes and validation can be compared.
- **Tune** searches the best settings of the model within a budget (1, 5 or 15 minutes, or a number of trials), with 3-fold cross-validation on the training rows only: the test rows judge the result once, when you click **Use these settings**. You see the best score so far, the gain over the defaults and every trial live, and can stop at any time and keep the best. Expert: the search algorithm (TPE, random, grid, CMA-ES, NSGA-II), the early stop of bad trials, and the search space (bounds, log scale, fixed settings). After 20 trials or more, the settings that mattered most are shown.
- **Auto** finds the best model it can within a time budget (2, 5, 10 or 30 minutes, or your own), on its own and step by step: **baselines** (a constant model — the mean or the most frequent class — and the linear model, to know where you start from), **one model per family** with its defaults, cheapest first (a family too slow for the budget is skipped, and the journal says why), a **reduction** (on large data the weakest families stop early, on a quarter then half of the rows; the 3 best go on), **tuning** of those 3 families with Optuna (the time left shared by rank) and an **ensemble**. The ensemble picks, again and again, the model whose out-of-fold predictions most improve the average (greedy selection, Caruana et al. 2004; a model can be picked several times: that is its weight), or, as an option, a regularized linear model combines them (*stacking*). It is judged on the same folds as the single models and **kept only if it beats the best single model by more than one standard deviation between folds**; otherwise the single model is recommended — simpler and as good. Every model is judged the way you chose, panel walk-forward and its criterion included; with a train / test split, the models are compared on 5 folds of the training rows and the test rows only judge the model kept (the decision threshold, if any, is chosen on training folds). You follow the current step, the clock, a journal and a live leaderboard (model, family, settings, score ± std between folds, seconds), and can **Stop and keep the best** at any time: the best model so far is saved as a run. At the end, the kept model is a normal run — explainable, usable on new data — with its gain in plain words (“23% lower error than the linear model”) and the Auto's journal and leaderboard kept with it. With the same budget the ranking is the same, up to how many Optuna trials fit in the time.
- **Explain** shows why the model predicts what it predicts (computed once per run, then kept): what the model relies on (how much the score drops when each column is shuffled), how each column moves the predictions (SHAP values: exact for linear models, and for XGBoost / LightGBM when installed; other models say why they have none), the effect of one column on the average prediction (partial dependence, with how many rows sit at each value), and the tree itself for a decision tree. Columns so linked that their importance is shared are flagged. In **Predictions**, click a row to see how each column pushed its prediction up or down from the average.
- **Features** shows coefficients for linear models and the importance of each input for trees and boosting.
- **Errors** shows where the model is wrong, on its held-out predictions only (the test rows of a split, or the out-of-fold prediction of every row with cross-validation and panel walk-forward — never a training row). The error is the absolute error for a number, the share of wrong predictions (or the log-loss) for classes.
  - **Subgroups where the model is wrong** are found for you: every category and fifth of the columns that matter most for the error, their pairs, and a shallow tree fitted on the error. Each is tested against the other rows, with false discoveries held at 5 % (Benjamini–Hochberg), so a model whose errors are spread evenly gets “No subgroup is notably worse than average.” A subgroup needs at least 30 rows and 1 % of the rows and an error at least 20 % above average; the costliest come first (rows × extra error), each in one sentence: “When region = South and age < 25 (4.2 % of rows), the model is wrong 31 % of the time vs 12 % overall.” Click one for its worst rows and **Open in Data** / **Open in Compare**, which open those pages filtered on the group (every row of it, training rows included).
  - **What might help**: careful hints the app never applies itself — more data for a small badly predicted group, a column the preparation excluded that defines a group (include it in *Preparation › Columns*), errors growing with the value (predict its log, or use a Poisson or Gamma regression), a rare class often missed (class weights, or a lower decision threshold for two classes).
  - **Error by column**: one small chart per column, the mean error in each category or fifth against the average (dashed), the columns where it varies most first; bars of fewer than 30 rows are muted.
  - **Worst predictions**: the 50 held-out rows predicted worst; click one to see how each column pushed its prediction (as in *Predictions*).
  - **Metrics by group**: choose a sensitive column (sex, age…) to put the model's metrics for each of its groups side by side — error rate, predicted positive rate and recall for two classes; mean error, prediction and actual value for a number — with the largest gap. Display only: nothing is corrected.
  - With **two runs** ticked in *Runs*, *Where the errors change* lists the subgroups where the second run fixes the first one's errors, and where it makes them worse.
- **Predict as Classes** (number targets) cuts the number into ordered classes — *Low · Medium · High* — and trains a classification. Choose 2 to 10 classes and how to cut: *Equal counts* (same number of rows per class, the default), *Equal width* (same range; extremes beyond the 1st / 99th percentiles go to the end classes), *Natural breaks* (limits where the data has gaps) or *Custom* (drag the limits on the histogram, or move them with the arrow keys, Shift for bigger steps). *Round values* gives round limits (12,000 rather than 12,487.33). Rename a class by typing in its name. A class with too few rows is flagged with a one-click fix; a class with fewer than 5 rows cannot be trained. Training starts by itself 600 ms after your last change. Results add *Right or off by one*: the share of predictions in the right class or the next one. A count from 1 to 10 (a rating) can be predicted with one class per value.
- **Preparation · Edit** opens the preparation editor. *Automatic* applies the rules; change anything and it becomes *Custom* (*Reset all* goes back). The changes apply to the next training and are saved with its run (opening a run brings its choices back).
  - **Columns** — one row per column: use it or not, how missing values are filled (with an optional “was missing” flag), transform, encoding, scaling. Grey “Auto: …” values come from the rules (ⓘ shows which rule and why); your choices show a dot and a reset button. Choices a column or the model cannot take are greyed out with the reason (e.g. log needs every value ≥ 0; linear models need missing values filled). Left-out columns are listed last with their reason; most can be used anyway — a column that seems to contain the answer asks for confirmation.
  - **Selection** — optional statistical selection of the prepared inputs: low variance, correlated pairs, variance inflation (VIF), mutual information, univariate F test, L1 regularization, recursive elimination with cross-validation, permutation importance. Each has its threshold and shows how many inputs it kept; all are fitted on the training rows only.
  - **Missing & outliers** — the default for missing numbers (auto, mean, median, 0, KNN imputer, iterative imputer), the “was missing” threshold, rare categories, the one-hot limit, target-encoding smoothing, and outliers (left as they are by default; clip to the 1st–99th percentiles or to 1.5 IQR, or drop extreme training rows beyond 3 IQR — never test rows).
  - **Imbalance** (classes) — auto (class weights when the rarest class is under 20 %), none, class weights, random over- or under-sampling of the training rows, with the class counts before and after. The test rows keep the real shares.
  - **Preview** — the first 20 training rows exactly as the model receives them, with the name of every input (`city_Paris`, `signup_date_month_sin`…).
- **Validation**: 80 % of the rows to train, 20 % to test (stratified for classes), at most 200,000 training rows (a reproducible sample beyond).
- **Results** start with a sentence comparing the model with a naive reference — always predicting the average, or always answering the most frequent class — and say plainly when the model does no better.
  - *Regression*: R², MAE, RMSE (next to the average's), predicted against actual, residuals against predicted, residual distribution, and a Breusch–Pagan check of whether the error grows with the value.
  - *Classification*: accuracy (next to the majority class), balanced accuracy, macro F1, ROC AUC, log loss, the confusion matrix (counts or % of each actual class), precision / recall / F1 per class; for two classes, ROC, precision–recall and calibration curves and a **threshold** slider that updates the confusion matrix live.
  - *Features*: coefficients with 95 % interval, p-value and VIF (linear regression without penalty), the effect per unit of the original column, odds ratios for logistic regression.
  - *Predictions*: test rows with actual, predicted and error or probability; *Export CSV*.
- **Use this model**, under a run's results, puts the model to work on new rows (this month's customers, the cars to price).
  - **Predict on new data**: pick an imported dataset (a new file is imported from Home first, then appears in the list); it does not need the target column. The data is **checked first**, against what the model was trained on: *Ready*, *Check* or *Blocked*, with the sentence that explains it and one line per column of the model, the most important first. A missing column, or a column whose values mostly cannot be read in the model's type (text where a number was expected: the conversion is tried first), **blocks**. Categories the model never saw are counted (they follow the preparation's rule, e.g. grouped with the rare categories), and so are the rows outside the training range; either on 1 % of the rows or more asks you to *Check*. The data is also **compared with the training data** (as on the *Changes* page): a *Shift* column gives each model column's PSI, and a strong shift (PSI above 0.25) on a column the model reads asks you to *Check*, since the validation scores may not hold; *See what changed* opens the full comparison. Extra columns are ignored and listed. Choose the columns to keep next to the predictions (identifiers by default), then **Predict**: every row of the dataset, 100,000 at a time, with *Cancel*.
  - The result: rows predicted and the share **less reliable** (an important column of the row has a category the model never saw or a value outside its training range); for a number, a **90 % conformal interval** (low and high on each row, from the validation errors: about 9 rows in 10 fall within it); the **distribution of the predictions** against the validation's (a shift shows at a glance); when the data has the answer, its **scores next to the validation's**, to see whether the model holds. The rows come 100 per page (*Less reliable only* to look at those); click one to see why it got its prediction. *Export CSV* downloads every row; **Save as dataset** imports the data with its predictions as a new dataset. Every prediction stays listed under the run (date, data, rows, check).
  - **Export model (.zip)**: the pipeline (`model.joblib`), a **model card** (`model_card.json`: target, problem, the columns expected with their types, known categories and ranges, the validation scores, the versions of Python and the libraries, the dates; no data rows) and `predict.py`. With the Python library, `model = dx.load_model("churn_lightgbm.zip")` then `model.predict(dx.open("customers_december.csv"))`, or `python predict.py new.csv predictions.csv`. `load_model` checks the card before loading the pipeline, refuses a zip without one and warns when a library version differs. The app itself never imports a model file: a joblib file can run code when it is opened.

### Segments

*Modules › Segments* answers "what kinds of customers, products or vehicles do I have?" without knowing anything about clustering: the app picks the columns, prepares them, finds a number of groups that holds up, describes each group in words, and lets you reuse the groups anywhere. The active filters apply (segment only the French customers).

- **Ready at once**: the first visit runs a segmentation with the app's choices. On the left, the columns used and the ones left out with their reason (*customer_id excluded: Identifier*); identifiers, raw dates, free text, near-constant columns and columns more than 95 % empty are left out. **Edit** opens the preparation: the machine learning rules with a *distance* profile — missing values filled, very skewed numbers log-transformed, every number **scaled robustly** (median and IQR; otherwise the column with the largest numbers would decide everything), categories one-hot encoded (the 20 most frequent, the rest as *Other*). Your changes apply to the next run and to *Unusual rows*.
- **Method** (Auto says which it takes and why): *k-means* ★ for mostly numbers (MiniBatch k-means beyond 100,000 rows), *k-prototypes* (Huang, 1998) from 30 % of category columns — categories are compared as they are, without one-hot —, *Gaussian mixture* (groups of different shapes, a probability of belonging, k chosen by BIC), *Hierarchical (Ward)* up to 20,000 rows (with a dendrogram; the tree is built on at most 5,000 rows), *HDBSCAN* (finds the number of groups itself and leaves noise as *Not grouped*, never hidden).
- **How many segments**: k = 2 to 10 are tried and four small charts show the silhouette, Calinski–Harabasz, Davies–Bouldin and the inertia (look for the elbow); the app keeps the k they agree on and says so (*4 segments: best silhouette, and the elbow is at 4*). Switch *Segments* to **Set** and drag the slider: releasing it runs again with that k.
- **Stability**: the segmentation is redone on 10 subsamples of 80 % of the rows and compared with the adjusted Rand index. Below 0.6, *These segments are not stable: the data may not have clear groups.* A low silhouette is flagged too: read such segments as tendencies.
- **Portrait of each segment**: its size, the columns that set it apart, in their own units (*income: 2.1× the average*, *plan: 78 % are Premium, vs 21 % overall*), a proposed name from its two strongest traits (*High income · Premium*) that you can rename, and simple **rules** learnt by a depth-3 decision tree, with the share of the segment they describe.
- **Map**: the rows in 2D (principal components, with the share of the variance each axis explains; *Draw with t-SNE* on demand, on at most 5,000 rows), coloured by segment, centres named.
- **Profiles**: one line per column, the mean or the most frequent category of each segment; strong gaps are highlighted. *Export profiles (CSV)*.
- **Save as dataset** creates a new dataset with the same rows (filtered as they were) and a `segment` column holding each row's segment name; it opens in Compare, KPI, Machine learning and Time series like any other. The source is not changed.
- **Runs**: every segmentation is kept (columns, preparation, method, k, scores). Tick 2 to 5 to compare them, with how much their segments agree.
- **Large data**: beyond 200,000 rows the segments are fitted on a reproducible sample of 200,000 rows, then every row goes to its nearest segment (the page says so). k is tried on at most 50,000 rows, the chosen k is fitted on the whole sample.

### Unusual rows

A row can be normal on each column and absurd as a whole: 25 years old with 40 years of seniority, a small city car at 400,000 €, a customer with 0 orders and 3,000 € of revenue. The *Unusual rows* section of Segments finds these rows and says **why** each one is unusual. It uses the same columns and preparation as Segments. The variable page's *Outliers* tab links here.

- **Methods** (Auto chooses): *Robust Mahalanobis distance* (MCD) when every column is a number, 30 at most, roughly elliptical once prepared; *Isolation Forest* (200 trees, seed fixed, fitted on at most 200,000 rows, every row scored) otherwise; *Local Outlier Factor* on demand (rows isolated in their own neighbourhood); *Distance to its segment* once a segmentation exists (unusual **for its group**: a luxury car is ordinary among luxury cars, not among city cars).
- **How many rows**: nothing to guess. The scores are sorted from the most unusual row down and the threshold sits where the curve **breaks** (steep, then flat); without a clear break, the 1 % most unusual rows. Move it with the slider or click the curve: the count and the table follow at once, without scoring again.
- **Why this row**: the three columns that matter most, as a sentence (*price is in the top 0.2 %*, *orders = 0 while revenue = 3,012: this combination appears in 0.05 % of rows*). A column's contribution is how much the score falls when its value is replaced by the column's median (or most frequent category) — within the row's segment for the segment method.
- **Table**: the rows above the threshold (the 500 most unusual are listed), score, reasons and every column; click the reasons to open the row's card, each value placed on its column (percentile, or share of its category).
- **Mark** a row *Expected* (normal to you) or *Error*. Marks are kept (under the identifier when the dataset has one, else a fingerprint of the row) and exported; rows marked *Expected* leave the next search (*1 row marked expected hidden*).
- **Save as dataset** adds `unusual_score` and `is_unusual` (at or above the threshold); **Export rows (CSV)** gives the rows above the threshold with their reasons and marks.

In the browser version, Segments and Unusual rows are available: beyond 50,000 rows they fit on a sample, and t-SNE is locked.

### KPI

A business indicator in one click — revenue per month, distinct customers per week, average basket per quarter — as key figures, a chart and a table.

- **Suggested** KPIs open the page, read from your columns: *Rows*, *Total revenue* (amounts and quantities are summed), *Distinct customer_id* (identifiers that repeat), *Revenue per customer_id*, *% churned* (yes/no columns), *Average rating*. The first one is already computed; click another to see it.
- **Split by** chips under the chart add the best columns to split by (countries, regions, statuses first, then the columns with 3 to 12 values).
- **The builder** asks four questions: *Measure* (rows, or the sum, average, median, min, max, distinct count or % true of a column — up to 4 measures, plus a **ratio** of two, e.g. revenue ÷ distinct customers), *Over time* (date column, grain and period: all dates, the last N periods, year to date or a custom range; *Use the page filters* adds the filters to that period), *Split by* (one or two columns; beyond the 8 largest groups, the rest becomes *Other*). Changes take effect with **Calculate**.
- **Everything is chosen for you, with the reason**: the grain (the finest giving at most 60 periods), the chart (bars or areas for volumes that add up, lines for averages and shares, stacked bars per group, side-by-side bars for negative values or 3 periods or fewer, lines when one group makes over 70 % of the total, small multiples for two splits, horizontal bars or a heatmap without dates), the table layout (periods across up to 13 periods, in rows beyond), the number format. *Why this view* lists the choices.
- **Key figures**: the last complete period of each measure, its change against the previous period and the same period last year (arrows in neutral ink: a rise is not always good news), and a mini line of the last 24 periods. An incomplete period (the current month) is marked † and left out.
- **Totals are recomputed**: the total of an average, a median or a distinct count is computed on all the rows, never added up (a customer who bought in two countries counts once). *Other* is computed from its rows too.
- **Customize** changes any automatic choice — chart type, a *% of total* chart below, value labels, *Show as* (value, % of total / row / column, change vs previous, % change vs previous or last year, running total), table layout, totals, shading, compare columns, Top N, hiding *Other*, the order of the groups, and for each measure its name, decimals, compact numbers, prefix and suffix (€, km). Each setting says what Auto chose; options that would mislead are greyed out with the reason (stacking an average, a running total of a mean). A changed setting has a dot and *Reset to auto*. Display changes are instant: only the data changes recompute.
- **Table**: *Swap* rows and columns, sort by clicking a column, *Copy* (paste into a spreadsheet) or *Download CSV*.
- **Save** a KPI by name; saved KPIs are listed on the left (rename, duplicate, delete). The page's address keeps the KPI shown.

### Experiments

An A/B test, read the way a statistician would: "did the new checkout convert more, and is it more than chance?", then "how many users do I need next time?". Two tabs: **Analyze** and **Plan**. The demo dataset *ab_test* is a checkout test with several sessions per user.

- **Describe the test** in one bar: the **Variant** column (2 to 5 values; columns named variant, group, arm, treatment… are offered first), the **Control** (guessed: *control*, *A*, *0*, *baseline*…, else the largest group), one or more **Metrics**, and the **Unit**. The page opens on a suggested test, already computed; every change recomputes it, and the page filters apply. The address keeps the test shown.
- **Each metric's kind is guessed and said**: 0/1 or yes/no is a **rate** (z-test of two proportions, Newcomb's interval of the difference); numbers are a **mean** (Welch's t, which does not assume equal variances; when the values are very skewed, Mann–Whitney and a bootstrap interval are shown as checks); whole numbers from 0 are a **count** (a Poisson test when the variance is close to the mean, Welch's t otherwise). A metric's menu turns it into a **ratio of sums** (revenue ÷ orders, delta method with a bootstrap check), makes it the **primary** metric (the others are secondary), or forces its kind.
- **Unit**: when a user has several rows, choose the column naming the user; rows are combined per user first (a rate is 1 if any row is, counts and amounts are summed). Without it, rows are not independent and the intervals are too narrow: the page notices an identifier that repeats and offers **Use user_id as the unit**. Users seen in two variants are left out, with a warning.
- **Sample ratio check (SRM)** first: a χ² test of the users per variant against the planned split (50/50 by default; set it under **More**). Below p = 0.001, a red banner warns that the groups are not the size you planned and the results may be biased.
- **One sentence per variant and metric**: "new_checkout reaches 25.0 % vs 19.8 % for control: +5.2 points (+26.4 %), 95 % CI [+1.6 ; +8.9]. Unlikely to be chance (p = 0.01, adjusted for 2 comparisons)." with a verdict: *Significant · higher / lower*, *Could be chance*, or *Not enough data*. A chart shows each variant's difference with its interval against zero, and a table gives the values, the difference and the lift with their intervals, p, the adjusted p, the **power** reached and the **detectable lift**.
- **Several variants or metrics**: p-values are always adjusted with **Holm's correction** over every comparison to the control, and the page says so ("p adjusted for 4 comparisons (Holm)").
- **Not enough data is not "no effect"**: when a result is not significant and the power is below 50 %, the sentence says "Not enough data to conclude: with this sample you could only detect a lift of ±36 %".
- **CUPED** (in a metric's menu, *CUPED with*): the same measure before the test (last month's spend) removes part of the noise without biasing the difference; the page says by how much ("CUPED reduced the variance by 38 %").
- **More**: **Subgroups** (the effect per country or device, up to 10 levels with at least 20 users per variant, with the warning that subgroups are hypotheses to confirm), **Over time** (the cumulative difference day by day with its interval, and why stopping as soon as p < 0.05 inflates false positives), α.
- **Plan**: a rate or a mean, its current value (typed, or read from a column of the dataset with *Read from*), the effect worth detecting (a % lift or an absolute difference), α, power, number of variants, the ratio of variant to control users and, optionally, the users entering the test per day. The page gives the users per group, the total and the duration, and a curve of the smallest lift each sample size detects. Sizes are statsmodels' `NormalIndPower`; with more than two variants, α is shared between the comparisons.
- **Understand this test** under each metric explains the test, the interval, the power and the correction.

### Synthetic data

An artificial copy of the data — same distributions, same links between columns, none of its rows — to share, demo or test without exposing real rows. The page says how faithful the copy is and how well it protects the real rows.

- **Ready to generate**: each column gets a treatment with its rule and reason — numbers, categories and dates are *modelled* (real bounds, decimals and levels kept), identifiers get *new values in the same format* (never a real one), free text, e-mails and phones are *left out* (e-mails and phones can be replaced by *fake values* of the same shape), categories with over 200 levels have their rare levels grouped into *Other*. Change any treatment, set bounds on numbers, then **Generate**.
- **Methods**: *Sequential CART* ★ (columns generated one after the other, each from a decision tree on the previous ones, taking real values of the same leaf of at least 5 rows), *Gaussian copula* (fast, keeps monotonic links), *Bayesian network* (a Chow-Liu tree of the strongest links, shown with the run), *Independent columns* (reference: every link is lost), and two neural networks, *CTGAN* and *TVAE*, trained on your data — on the graphics card when one is ready (a *GPU · card* / *CPU* switch next to Generate), with a progress bar per epoch and *Epochs* (Auto: 300, fewer on large data to stay within minutes). They need the optional PyTorch install (`uv sync --extra deep`); without it they are greyed out. *Smooth numbers* (on) and *Drop copies of unique real rows* (on) keep synthetic rows away from real ones. The same seed and settings give the same data. 20 % of the real rows are held out from the generator to check privacy and utility.
- **Four figures**, each with a verdict: *Diagnostic* (valid values: bounds, levels, unique identifiers, whole numbers — expected 100 %), *Quality* (computed as SDMetrics: mean of *Column shapes* and *Column pair trends*, only pairs strongly linked in the real data counting), *Detection* (can a classifier tell real from synthetic rows? 100 % = no), *Privacy* (no copied rows, no memorisation).
- **TimeGAN** (sequences): on data with dates, it learns how each history moves over time — a store day after day, a customer purchase after purchase. The bar proposes the date column, the entity (the column that names a history), the length of the sequences, how many to generate and the training steps (about 2 minutes on a GPU); every synthetic history gets a new entity and dates at the real step. The **Sequences** tab compares the autocorrelation of real and synthetic sequences, shows how often a recurrent classifier tells them apart, how well they predict the next step, and a few sequences side by side.
- **Columns**: from the most damaged to the least, with the largest difference in words; open one to see real and synthetic side by side — counts and shares of each set, histograms on the same intervals or cumulative curves, paired statistics with their gap, most frequent, smallest and largest values with their frequency in the other set; aligned levels for categories. *Overlay a target* shows whether its link with each column is kept.
- **Links**: the association matrices of the real and synthetic data and their gap, the pairs that changed most; click one to compare the pair.
- **Checks**: every diagnostic check, the details of the quality score, and coverage (range, levels, empty rate, outliers).
- **Privacy**: copies of real rows that no other row shares (the ones that can point to a person), exact and near copies of any row, protection against memorisation (synthetic rows should be no closer to the training rows than new real rows), distances to the closest real row, rare values reproduced, and the *disclosure risk* (knowing some columns of a person, can the synthetic rows reveal a sensitive one?). Synthetic data is not a differential-privacy guarantee.
- **Utility**: pick a column; one model trained on real rows and one on synthetic rows are tested on the same held-out real rows — "keeps 92 % of the real data's predictive power".
- **Overview**: means and spreads of every number, real against synthetic; principal components of both sets; the columns left out.
- **Data**: the first 100 synthetic rows, **Save as dataset** (a new dataset with the same column types, marked *Synthetic, from …*), or download CSV / Parquet.
- **Runs**: every generation is kept; reopen one, tick 2 to 5 to compare their scores, delete.

### Assistant

An optional AI analyst docked on the right of every dataset page (**Ask the assistant**, or **Ctrl+J**). Ask in plain words — "what drives Survived?", "which columns need cleaning?" — and it runs the app's own analyses, reads their results and explains them: the overview, column profiles, Compare, correlations and counts, and also the modules — it trains, compares, tunes and explains machine-learning models (and runs the Auto, after asking you how many minutes it may take) and predicts new data with them, analyses and forecasts time series, suggests and computes KPIs, analyzes A/B tests and computes the sample size of the next one, generates synthetic data with its fidelity and privacy report, and writes the HTML report (with a link to download it). Long analyses show their progress under their step. Every figure it gives comes from one of those analyses: each step is listed under its answer, with a link that opens the analysis in the app. It follows your filters and knows which page you are on.

- **Setup**: it works with your own API key from Anthropic (Claude), Google Gemini, OpenAI, OpenRouter, Groq, Mistral or DeepSeek, or with a local model server that speaks the OpenAI format (Ollama, LM Studio: give its address, no key needed). Paste the key in the assistant's settings: the app recognises Anthropic, Gemini, OpenAI, OpenRouter and Groq keys by their shape and switches to that provider; for the others, choose the provider first. The key is saved on this computer and never shown again. You can also set it in the environment before starting the app (`GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`…, optionally prefixed with `QUANTILO_`). The Docker image includes what it needs; from source, run `uv sync --extra agent` in `backend/`.
- **Models and cost**: Claude models are listed by name; for other providers, the settings list the models your key can use (the newest stable one of the provider's main family is the default; for Gemini, a Flash model, since free keys have no quota for Pro). Prices are only known for Claude: with Claude the panel shows dollars against your budget, with other providers it counts tokens, up to 2 million per conversation.
- **What leaves your computer**: your messages, the column names and types, and the statistics the analyses return (counts, means, tests, the most frequent values) — never whole rows.
- **Cost**: each conversation shows what it has spent; it stops at its budget (1 $ by default, change it in the settings). Pick the model and how hard it thinks (effort) there too: Opus 5.5 is the most capable, Sonnet 5.5 costs half, Haiku 5.5 is the quickest and cheapest.
- **It asks before changing anything.** Saving a KPI or creating a dataset from synthetic data or from predictions shows a card: **Confirm** or **Dismiss**. Trained models and generated runs are kept like the ones you make yourself.
- **Stop** ends an answer after the current step (and cancels a training, forecast or generation in progress). Conversations are kept per dataset; start a new one with **+**.
- **Figures are checked.** After each answer, the app looks for every number of the answer in the results of the analyses it ran (no extra call to the model). A figure found nowhere is listed under the answer ("1 figure not found in the analyses: 46.8"): it may be computed by the model, like a difference, or wrong. Check it before relying on it.
- **Dataset notes** (the notebook button in the panel's header): short facts every conversation on this dataset starts with, such as the target, a unit or what a code means. Write them yourself, or let the assistant offer one: it asks you first, like any change. Notes go with the dataset.
- **Method**: for broad requests (a first look at the data, preparing a model, a forecast, KPIs, synthetic data), the assistant reads a short method written for that kind of work before it starts; the step shows as "Reading the … method".
- Long conversations stay light: results from older questions are sent to the model as their one-line summary.

### Correlations

Every pair of variables measured at once, whatever their types.

- **Association matrix** (full width): blue = positive, red = negative (numeric pairs only); categories use Cramér's V and η, from 0 to 1. Choose *Pearson* or *Spearman* for numeric pairs, *Group related variables* (clustering) to put linked variables next to each other, and which columns to include (60 at most). Click a cell to open *Compare* on that pair. Above 200,000 rows, the matrix is computed on a reproducible sample of 200,000 rows.
- **Insights**: sentences grouped by theme — *Redundant variables*, *Relationships*, *Distributions*, *Data quality* — each with a **View** link to the chart that proves it.
- **Evidence level** of each link (badge with bars). *Show links* chooses what you see: *Strong*, *+ Moderate* (default) or *+ Weak*. Distribution and quality insights are always shown.
- **Focus on a variable**: what relates to it, best evidence first.
- **Strongest pairs**: table with measure, value, adjusted p-value, evidence and n.

### Data

The rows behind every chart, after your filters.

- **Scroll** through millions of rows (loaded 100 at a time). The first column is the row number in the file.
- **Rows to show**: *All rows*, *Random sample* (10,000 rows, always the same), *Duplicates* (identical rows, copies next to each other), *With missing values*.
- **Search** any column (case-insensitive). **Sort** by clicking a column name (ascending → descending → file order).
- Column menu (⋮): sort, *Filter on this column…*, *Pin to the left*, *Hide column*, *Open the variable page*. Hidden and pinned columns are remembered per dataset.
- Cells: numbers right-aligned, empty values as a grey "—", a value that can't be read in the column's type in orange italics (hover for details).
- **Export** what you see (filters, mode, search, sort) as **CSV**, **Parquet** or **Excel** (Excel holds 1,048,575 rows at most), with the visible columns or all of them; or the **column summary** (one row per column: type, missing, distinct, min, quartiles, max, mean, std, top value), or the full **HTML report** (see Overview).

### Changes

What changed between two versions of a file, before you trust the numbers: a column gone, a type broken, new categories, a distribution that moved, rows edited, past KPI values corrected in the source.

- **Two ways in**. After a **re-import**, the app compares the new version with the previous one: the *Overview* shows a one-line summary on top (*See what changed* opens this page; the cross hides it until the next re-import). Or tick **two datasets** on *Home* and click **Compare versions** (October against November, a training file against a file to predict): the one imported last is compared with the other. On this page, *Compared with* switches between the previous version and any other dataset.
- **Summary**: a level — *No notable change*, *Check* or *Breaking* — and a sentence such as "Compared with 2026-10-01: +4,812 rows (+3.1%), 1 column removed, 2 distributions shifted, 3 new categories.", with the reasons. *Breaking* means something you saved depends on what changed (a column used by a saved KPI is gone, or no longer the same kind of type).
- **Schema**: columns added, removed, **probably renamed** (same kind of values and the same values, or a close name such as *chiffre_affaires* → *ca*; never applied for you), types that changed, columns now constant or empty, and what uses them.
- **Quality**: rows, duplicates, dates in the future or months with no rows, and per column the share of missing and of unusual values (beyond 1.5 IQR, as on the variable page) before → after.
- **Distributions**: every number and category, the strongest shift first, ranked by **PSI** (an effect size: on a million rows every difference is "significant", so p-values don't rank). Click a row for the two versions drawn together (before in grey, after outlined), mean, median and standard deviation before → after, and the values that appeared or disappeared with their share. Dates show their range and rows per period.
- **Rows** (needs the previous rows, see *Settings › Storage*): rows matched on an identifier column, added, removed, modified and unchanged, how many values changed per column, examples, and **Download the differences (CSV)** with the old and new value of every change.
- **KPIs**: each saved KPI on both versions with its change; *Past periods revised* counts the periods present in both files whose value changed (a correction made back in the source), listed below.
- **Relations**: the 10 strongest relations of each version (as on *Correlations*) with their value in the other; a relation that appears or disappears is flagged.

The summary, Schema, Quality and Distributions work from a small statistical photo the app keeps of every import, even when the previous rows are gone; Rows and Relations need the previous rows.

### Recipes

A **recipe** keeps the work done on a dataset — corrected types, filters, KPIs, time series analyses, models, outputs — and **replays** it on another file with the same columns: next month's file gets the same treatment in one click, with a report of what worked, what changed and how the key figures moved.

- **Create**: *Overview › Save as recipe*. Name it and tick what to take (everything is ticked, except the output files): *Columns* (each type, role or order you corrected, each column you excluded), *Filters* (your saved filters; one of them can be **active**: the KPIs, time series and outputs are then computed on its rows), *KPIs* (saved in the KPI builder), *Time series* (analyses kept with *Save analysis*), *Machine learning* (the latest run of each target and model), *Output* (an HTML report, a CSV of the filtered rows). A recipe holds column names and settings, never rows; for a category read by a step it keeps the known values (at most 50) to flag new ones.
- **Recipes** (sidebar, under *Home*) lists them: steps, last run and its status. *Import a recipe* reads a `.recipe.json` file.
- **Run on…**: pick a dataset already imported, or import a new file. The app then pairs each column of the recipe with a column of the file: the **same name**; the same name once case, spaces, accents and `_` are ignored (*Matched*, flagged); otherwise the closest name, **at most 2 letters apart and with the same detected type** (*Proposed*: tick it to accept); otherwise *Missing*. You can pick another column on any line. Steps that read a missing column are listed as *will be skipped*, with the reason; the others still run. Columns of the file the recipe doesn't know stay as detected and are listed.
- After an import, the *Overview* suggests the recipes the file fits: "This file looks like *Monthly sales* (14 of 14 columns match). Run the recipe?"
- **The run** happens in the background, step by step (Columns → Filters → KPIs → Time series → Machine learning → Output); *Cancel* stops it: the steps already done stay, the rest are marked *cancelled*. A step that fails doesn't stop the others, except the ones that depend on it (the KPIs computed on a filter that could not be applied are skipped). A type correction that no longer reads the new file (more than 10% of the values unreadable, e.g. a date column full of text) is **not applied** and flagged ⚠️ with the rate. Models are **trained again** on the new rows (not reused). Nothing is overwritten: KPIs, analyses and runs are added to the dataset, marked with the recipe's name.
- **Report**: on top, *what changed* since the previous run's file (or the previous version of the same dataset), with its level and a link to *Changes*; then the **key figures** — each KPI's value, each model's score, the next forecast value — next to the previous run's, with the change; then each step: ✅ done, ⚠️ done with a warning ("3 new categories in *region*"), ⏭️ skipped (why), ❌ failed (the error), with a link to the page it filled.
- **Baseline**: when a recipe is made, it reads the key figures of the dataset it was made on (nothing changed, models not retrained), so the first run on a new file already has something to compare with.
- **History**: the recipe page draws each key figure over the runs (one small chart per figure) and lists every run; *Download CSV* exports the history.
- **Edit**: untick or delete steps on the recipe page. Each change makes a **new version**; every run remembers the version it used.
- **Export as Python** writes a readable script doing the same with the `quantilo` library (`dx.open`, `retype`, `where`, `kpi`, `timeseries(...).forecast`, `ml(...).train`, `report`), to run without the app (Windows Task Scheduler, cron, a notebook); it prints the same key figures. **Export file** saves the `.recipe.json` to send to someone who has the app. An imported file is checked against its schema: an unknown field or schema version is refused with the field named, and nothing in it is ever run as code.
- In the browser version, recipes work; their machine-learning steps are skipped (machine learning needs the full app).

### Settings

- **Appearance**: theme (system, light, dark), density (compact, comfortable), animations (system, reduce, full).
- **Analysis defaults**: starting values of *Top categories per chart*, *Cards per page* and the normality tests' *α*, for datasets you haven't customised yet (each page can still change them). Remembered in this browser.
- **Type detection**: the thresholds used by the next detections (imports and *Re-run detection*): how many distinct whole numbers still count as *discrete* (20), the share of values that must read as numbers or dates to convert a text column (95 %), the share of distinct values that makes an *identifier* (95 %), the share of empty values that makes a column *ignored* (95 %). Validated types and manual corrections are kept. *Back to defaults* restores them.
- **Compute**: the graphics card found (*Check again* looks again) and *Use the GPU when available*, for the deep synthetic-data models coming next (no effect yet).
- **Storage**: where the app keeps its copies (the workspace), the disk used in total and per dataset, **Keep the previous version** — *Always*, *Under 2 GB* (default; *Under 200 MB* in the browser version) or *Never* — the rows of the previous version kept at a re-import for the row by row comparison (statistics are always kept), with the space they take, and *Clear caches* to drop computed results kept in memory (rebuilt on demand).
- **About**: version and licence.
- The ↺ button next to a setting resets it to its default.

### Anywhere

- **Ctrl+K** (or the *Search or jump to…* button at the top) opens the command palette: type a few letters to go to a page, a variable of the open dataset or another dataset, or to run an action (open or clear the filters, change the theme, show the shortcuts). Arrows to move, Enter to run, Esc to close.
- **?** lists the keyboard shortcuts. **Guide** (bottom of the sidebar) opens this guide.

## Filters

Press **F** or click **Filter** (top right). Filters follow you on every page, live in the URL (shareable, reload-safe) and belong to one dataset.

- **AND between filters**, **OR between the values ticked** in one filter. Every condition has its opposite ("is none of", "outside", "is not empty"…). Empty values never pass a negated condition.
- Conditions by type: numbers (=, >, ≥, <, ≤, between, outside); categories (is any of, contains, starts with, equals); dates (before, after, between, falls on a weekday / month / quarter / hour); every type: is empty.
- **Unique values (one row each)** — keeps one row per value. *Based on*: **This column**, or **Several columns (key)** to tick the columns of a composite key (e.g. `vehicle_id` + `month`). *Keep*: the **first** or **last** row, in file order or **ordered by** a column (e.g. the latest date).
- **Duplicates only** — the rows whose value (or key) appears more than once, all copies. **Values appearing once** — the opposite: values present a single time (rows of duplicated values are removed entirely; to keep one row of each, use *Unique values*).
- Unique / duplicates conditions apply **after** the value conditions: *country = FR* + *unique vehicle_id (last)* gives the last French row of each vehicle.
- **Save** a set of filters under a name and apply it later. Click a chart (bar, range, outliers) to add a filter from it. Tick *Show active filters under the top bar* to keep them visible as chips.

## Metrics

### Describing one variable

| Metric | What it tells |
|---|---|
| Missing | Empty cells, plus null tokens like "N/A", "-", "null" |
| Unreadable | Values present in the file but impossible to read in the chosen type (e.g. "abc" in a number column); counted as missing in analyses |
| Distinct | Number of different values ("≈" when estimated, above 200,000 values) |
| Mean | The average; pulled by extreme values |
| Median | The middle value: half of the values are below. Robust to extremes |
| Std dev | Standard deviation: typical distance to the mean, in the variable's unit |
| CV | Coefficient of variation = std dev / mean: spread relative to the level (unitless) |
| MAD | Median absolute deviation: typical distance to the median, robust to outliers |
| Min, Q1, Q3, Max | Smallest value, 25 % quantile, 75 % quantile, largest value |
| IQR | Q3 − Q1: the range of the middle half of the values |
| P1, P5, P10, P90, P95, P99 | Values below which 1 %, 5 %, 10 %, 90 %, 95 %, 99 % of the data lie |
| Zeros, negatives, infinite | Counts of values equal to 0, below 0, or infinite (infinite values are left out of the statistics) |
| Sum | Total of the values |
| Skewness | Asymmetry: 0 = symmetric, > 0 = long tail to the right (a few very large values), < 0 = long tail to the left. Beyond ±1 it is clearly asymmetric; beyond ±2, strongly |
| Kurtosis | Tail weight compared with a normal distribution (excess kurtosis, normal = 0): > 0 = more extreme values than a normal, < 0 = fewer |

### Normality tests

Each test asks whether the values could come from a normal (bell-shaped) distribution. A **p-value below α** (0.05 by default) means "probably not normal".

| Test | How it works |
|---|---|
| Shapiro–Wilk | Compares the sorted values with those a normal would give. The most powerful general test for small and medium samples |
| D'Agostino–Pearson K² | Combines skewness and kurtosis: rejects data too asymmetric or with tails too heavy or too light |
| Anderson–Darling | Distance between the empirical and the normal distribution, with extra weight on the tails. Its p-value is bounded between 0.01 and 0.15, shown with "≤" or "≥" |
| Jarque–Bera | Checks that skewness and kurtosis match a normal's. Best with large samples |
| Lilliefors | Largest gap between the empirical distribution and a normal with the same mean and std dev (Kolmogorov–Smirnov corrected for estimated parameters) |

With thousands of rows, tests detect even tiny departures from normality: the **QQ-plot** tells whether they matter. Points on the line = normal; curving away at the ends = heavier or lighter tails.

### Fitting laws

| Metric | Meaning |
|---|---|
| AIC | Akaike information criterion: fit quality penalised by the number of parameters. Lower is better; only differences between laws matter (a gap above 10 is decisive) |
| BIC | Like AIC, with a stronger penalty on large samples |
| KS | Largest gap between the empirical and fitted distributions (0 = perfect) |
| λ (Box-Cox, Yeo-Johnson) | The power that makes the data most symmetric (λ = 0 ≈ log, 0.5 ≈ square root, 1 = unchanged) |

### Outlier rules

| Rule | Flags a value when |
|---|---|
| IQR × 1.5 | It is below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR (Tukey; the box plot's whiskers) |
| IQR × 3 | It is beyond 3 × IQR from the quartiles: only the most extreme values |
| z-score | It is more than 3 std devs from the mean. Assumes a roughly normal shape |
| Robust z | Its distance to the median, in MADs, is above 3.5 (Iglewicz & Hoaglin). Not fooled by the outliers themselves |

### Relationships between two variables

| Measure | Pair | Range | Reading |
|---|---|---|---|
| Pearson r | Number × number | −1 … 1 | Strength of a **linear** relationship; sign = direction |
| Spearman ρ | Number × number | −1 … 1 | Strength of a **monotonic** relationship (on ranks): robust to outliers and curves |
| Kendall τ | Number × number | −1 … 1 | Share of concordant minus discordant pairs; smaller values than ρ for the same strength |
| R² | Linear fit | 0 … 1 | Share of Y's variance explained by the line |
| η (correlation ratio) | Number × category | 0 … 1 | How much the number differs between groups; η² = share of variance explained by the groups |
| Cramér's V | Category × category | 0 … 1 | Strength of association (bias-corrected); 1 = one determines the other |
| Cohen's d | Number × 2 groups | 0 … ∞ | Difference of means, in pooled std devs |
| η² | Number × 3+ groups (ANOVA) | 0 … 1 | Share of the variance explained by the groups |
| ε² | Number × 3+ groups (Kruskal-Wallis) | 0 … 1 | Same idea, on ranks |
| Rank-biserial r | Mann-Whitney | −1 … 1 | How often one group's values exceed the other's |
| Mann-Kendall τ | Date × number | −1 … 1 | Over every pair of periods, how often the later value is higher; −1 = always falls, 1 = always rises |
| Sen's slope | Date × number | any | Median of the slopes between every pair of periods: the typical change per period, robust to outliers; with its 95 % interval and the change per year relative to the average level |

**Effect sizes** (Cohen's conventions) say whether a link matters in practice:

| Magnitude | r, ρ, τ, V | η | η², ε² | d |
|---|---|---|---|---|
| Small | ≥ 0.1 | ≥ 0.10 | ≥ 0.01 | ≥ 0.2 |
| Medium | ≥ 0.3 | ≥ 0.24 | ≥ 0.06 | ≥ 0.5 |
| Large | ≥ 0.5 | ≥ 0.37 | ≥ 0.14 | ≥ 0.8 |

The **Mann-Kendall test** runs on one value per period. When successive values are strongly linked (lag-1 autocorrelation above 0.5: a cycle or a slow drift), its p-value is optimistic, and the panel says so.

Cramér's V thresholds are lowered for tables with more than two rows and columns (Cohen's correction for the degrees of freedom).

### p-values and evidence

- A **p-value** is the probability of seeing a link at least this strong if there were none. Below 0.05 = *significant*: unlikely to be chance. It says nothing about the size of the link: with many rows, tiny effects become significant — look at the effect size.
- **Adjusted p-value** (*adj. p*, Benjamini-Hochberg): the Correlations page tests hundreds of pairs at once, and about 5 % of them would pass p < 0.05 by chance alone. The adjustment controls that share of false discoveries.
- **Evidence level** of a link crosses its effect size with its adjusted p-value:

| Level | Rule | Reading |
|---|---|---|
| Strong | Large effect and adj. p < 0.001 | A clear link, very unlikely to be chance |
| Moderate | At least medium effect and adj. p < 0.05 | A real link of notable size |
| Weak | Small effect with adj. p < 0.05, or notable effect with adj. p ≥ 0.05 | Real but minor, or possibly chance (the sentence says which) |
| None | Negligible effect, or small and not significant | No insight |

- **Redundant variables**: |r| or V above 0.95 — two columns carry almost the same information; keep one of them in models.

### Comparing two versions

| Measure | For | Reading |
|---|---|---|
| PSI (population stability index) | Numbers (on the old version's deciles), categories (on their values) | How far the shares moved: below 0.1 *stable*, 0.1 to 0.25 *moderate*, above 0.25 *strong* (Siddiqi, 2006). Empty bins count as 0.01 % so the index stays finite |
| Wasserstein distance | Numbers | How far values moved on average, in units of the old interquartile range (0.5 IQR = half the spread of the middle half of the data) |
| Kolmogorov-Smirnov D | Numbers | Largest gap between the two cumulative distributions (0 to 1), with its p-value |
| χ² | Categories | Test that both versions share the same proportions, with its p-value |

Numbers are compared from 101 quantiles kept per version (D is exact to about ±0.01). Distributions are ranked by PSI, not by p-value: on large files every difference is significant. A category where most rows hold a value outside its 50 most frequent (invoice numbers, customers) is marked *Many values*: new values are expected there, so it is listed last and not counted as shifted.

### Insights thresholds

| Insight | Rule |
|---|---|
| Skewed | \|skewness\| > 2 |
| Imbalanced | One category above 70 % of the rows |
| Missing | 20 % of the values or more |
| Relationship | Evidence level weak or above (see above) |
| Redundant | \|r\|, \|ρ\| or V > 0.95, significant |

## Keyboard shortcuts

| Keys | Action |
|---|---|
| F | Open the filters |
| Ctrl+K | Command palette: go to a page or a variable, run an action |
| ? | List the shortcuts |
| Ctrl+Z | Undo the last change on the Columns page |
| Esc | Close a panel, a menu or a dialog |

## Good to know

- **Large files**: every page works on millions of rows. Some heavy steps use a reproducible sample (always the same rows), and the page says so: points of a scatter, normality tests (5,000 values), law fitting (10,000 values), the correlation matrix (200,000 rows). Counts, means and test statistics use every row unless stated.
- **Types drive everything**: if a chart looks wrong, check the column's type on the Columns page first.
- **Privacy**: the app runs on your machine and never sends your data anywhere.
