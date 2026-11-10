# Time series

1. **Check the series** (`analyze_time_series`): one date column, one measure, a regular frequency. Gaps, duplicates and the last incomplete period come first: they distort everything after.
   - Empty periods: the Kalman gap filling (Gaps: Kalman) estimates them with an uncertainty band; prefer it to linear interpolation for long or seasonal gaps, and say which periods were filled.
   - Smoothing (the Smooth row): a trailing moving average, EWMA and Kalman *filtered* are causal (they use only the past); a centred mean, LOWESS, Savitzky–Golay, Kalman *smoothed* and HP look at future values: for reading the level only, never as a model input or a forecast. HP's last points move when new data arrives (Hamilton 2018).
2. **Trend and seasonality.** Read the decomposition: the size of the seasonal swing against the trend, and the period found. Name the season in calendar words ("peaks every December").
3. **Stationarity.** ADF and KPSS together: they can disagree. A non-stationary series needs differencing before ARIMA-like models; the app handles it, but say it.
4. **Forecast** (`forecast`) with a horizon that the history supports: at most about a third of its length, and at least two seasons of history for a seasonal model. Always give the interval, not only the central line, and the backtest error the app reports.
5. **Limits.** A forecast extends the past; breaks (a launch, a crisis, a change of definition) are invisible to it.

Answer with the trend in one sentence, the seasonality, then the forecast with its interval and error.
