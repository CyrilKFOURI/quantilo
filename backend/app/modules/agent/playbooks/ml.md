# Preparing and judging a model

1. **Target first** (`list_ml_targets`). Check the task (classification or regression) and, for classification, the class balance. Strong imbalance changes the metric: use balanced accuracy, F1 or ROC AUC, never plain accuracy.
2. **Panel data.** If the same entities come back period after period (ids × dates, `era` and `id`), call `check_panel` and follow the `panel` method instead: random splits overestimate the score there.
3. **Leakage.** Before training, look for columns that give the answer away: identifiers, columns filled after the outcome (a "survived_date", a "refund_amount" to predict churn), near copies of the target. A suspiciously perfect score (AUC ≥ 0.99) almost always means leakage: say so and name the suspect.
4. **Baseline.** Every score is read against the baseline the app reports (majority class, mean). A model that barely beats it has learned little.
5. **Train** (`train_model`) with the app's recommended model first; `compare_models` only if the user wants the best one. Validation is cross-validated: quote the mean and the spread.
6. **Tune** (`tune_model`) only after a reasonable first model, and say how much it gained. A gain inside the spread is noise.
   **Auto** (`auto_ml`) does compare, tune and ensemble in one go when the user wants "the best model": first ask how many minutes it may take (2, 5, 10 or 30), never pick the budget yourself. Quote the kept model, its gain over the linear model and, when an ensemble was tried, why it was kept or not.
7. **Explain** (`explain_model`). Importance says what the model uses, not what causes the outcome. Read SHAP directions in plain words ("higher Fare pushes towards survival"). A free-text column is read through its words (`words`): quote the terms that push up and down, say whether the reading is exact (linear model) or a comparison of rows with and without the word, and treat a word that names the answer ("refunded" to predict refunds) as leakage.
8. **Where it is wrong** (`model_errors`). Quote the subgroups' sentences as they are (they passed a false-discovery control), the hints as suggestions, and the metrics by group as a display, not a fairness verdict.
9. **Limits.** Sample size, missing values imputed, columns excluded, and that the model reflects this data only.

Training takes time: say what you are doing while it runs, and do not retrain without a reason.
