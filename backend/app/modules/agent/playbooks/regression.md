# Regression: effects other things equal

1. **Start from the links.** `rank_associations_with` on the target first; then `fit_regression` with the target and its strongest links (or no columns: the tool picks the 5 strongest itself and says why). Add the controls the user names; a coefficient always reads "other columns held fixed".
2. **Name the model and the standard errors with their reason**, from `model_reason` and `standard_errors.reason`: HC3 by default (the Breusch–Pagan test informs, it never chooses), clustered by entity for panel data, HAC for one dated series (give the lag), WLS when a column has the role weight, logistic for a yes/no target (read the average marginal effect in probability points, the odds ratio second).
3. **Quote effects with their interval and adjusted p** (`p_adjusted`, `multiple_testing`). A coefficient significant before but not after the correction "no longer holds". Small standardized effects are "real but minor" even with a tiny p.
4. **Read the checks before concluding.** Red or orange checks change the reading:
   - *Straight lines* (RESET): an effect may bend; point to Two variables for the named column.
   - *Repeated columns* (VIF > 10): two columns share one effect at random; keep one, or say the pair's joint effect is what is known.
   - *Influential rows*: a few rows may drive a coefficient; the Influence section lists them.
   - *Out of sample*: R² well above the out-of-sample R² means the model fits noise.
   - *Few groups* (< 40 clusters): clustered intervals are too narrow.
   - *Left out*: a column equal to a combination of others is dropped, with the reason; say so.
5. **Normality** matters little for the coefficients' intervals with hundreds of rows; do not alarm the user about it.
6. **Association is not causation**, even with controls: name the plausible confounders left out, and reverse causality when it is plausible (sales drive advertising as much as the reverse).
7. **Panel data** (a `panel` check, R-REG-14): say that fixed effects compare each entity with itself, and point to Advanced > Panel. Quote its recommendation with the reason: the Mundlak test decides between fixed and random effects (p < 0.05: random effects would be biased); the classic Hausman test only informs. Name the columns fixed effects absorb (they do not change within an entity).
8. **Instruments (IV / 2SLS)** are in Advanced > More, and only when the user names both the endogenous column and the instrument: never propose an instrument yourself. If the user does, read the effective F first (below 10: weak, trust the Anderson–Rubin test), then Wu–Hausman (is least squares biased?), then Hansen J when there are more instruments than endogenous columns.
9. Give the link to the Regression tab: its Advanced sections show every kind of standard error side by side, the panel estimators, the terms and the influential rows.
