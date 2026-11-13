import { parseWindows } from "./Windows";
import type { ForecastTuning, SeriesParams } from "../../api/client";

export const PERIODS = [
  "hour",
  "day",
  "week",
  "month",
  "quarter",
  "year",
] as const;
export const AGGREGATES = [
  "sum",
  "mean",
  "median",
  "min",
  "max",
  "count",
] as const;
export const GAPS = ["leave", "interpolate", "zero", "carry", "kalman"] as const;
export const DECOMPOSITIONS = [
  "stl",
  "mstl",
  "classical_additive",
  "classical_multiplicative",
] as const;
export type Decomposition = (typeof DECOMPOSITIONS)[number];
export const TRANSFORMS = [
  "none",
  "log",
  "diff",
  "seasonal_diff",
  "log_diff",
  "diff2",
  "seasonal_diff_diff",
] as const;
export type Transform = (typeof TRANSFORMS)[number];
export const PACF = ["ywm", "ols", "burg"] as const;

/** A named what-if scenario (039): one known-in-advance driver changed over a range. */
export type StoredScenario = {
  name: string;
  column: string;
  how: "percent" | "value";
  amount: number;
  start?: string;
  end?: string;
};
export const MAX_SCENARIOS = 3;

/** Scenarios kept in the URL (until analyses are saved, spec 030). */
export function parseScenarios(value: unknown): StoredScenario[] | undefined {
  if (!Array.isArray(value)) return undefined;
  const found: StoredScenario[] = [];
  for (const item of value.slice(0, MAX_SCENARIOS)) {
    if (!item || typeof item !== "object") continue;
    const raw = item as Record<string, unknown>;
    const amount = Number(raw.amount);
    if (typeof raw.column !== "string" || !Number.isFinite(amount)) continue;
    found.push({
      name: typeof raw.name === "string" ? raw.name.slice(0, 60) : "",
      column: raw.column,
      how: raw.how === "value" ? "value" : "percent",
      amount,
      start: text(raw.start),
      end: text(raw.end),
    });
  }
  return found.length ? found : undefined;
}

/** URL of the Time series page: the series parameters, plus display choices. */
export type TimeSeriesSearch = SeriesParams & {
  method?: Decomposition;
  windows?: string; // overview rolling windows, e.g. "7,15" (the first draws the band)
  spread?: number; // volatility rolling window
  ar?: number; // autocorrelation: AR order to fit (default: chosen by AIC)
  transform?: Transform; // stationarity and autocorrelation sections
  pacf?: (typeof PACF)[number];
  rsource?: "series" | "returns"; // residuals section (default: residuals)
  segment?: "binseg"; // change-point search (default: PELT)
  cost?: "mean" | "mean_var" | "trend"; // default: chosen from the data
  sens?: "low" | "high";
  anomaly?: "iqr" | "rolling_z" | "hampel" | "gesd" | "isolation_forest" | "lof" | "autoencoder" | "kalman";
  threshold?: number;
  all?: boolean; // overlay every group on the overview
  scen?: StoredScenario[]; // what-if scenarios of the forecast (039)
  smodel?: string; // the driver model the scenarios use
  // Smooth row (064): parameters forced in "Options" (undefined = automatic)
  ewma_alpha?: number;
  lowess_frac?: number;
  savgol_window?: number;
  savgol_order?: number;
  kalman_model?: (typeof KALMAN_FORMS)[number];
  hp_lambda?: number;
  // Forecast ⚙ Options (028 A7): undefined = the default of each model
  arima_p?: number;
  arima_q?: number;
  arima_sp?: number;
  arima_sq?: number;
  arima_d?: number;
  arima_ic?: (typeof FORECAST_OPTIONS.arima_ic)[number];
  gbm_grid?: (typeof FORECAST_OPTIONS.gbm_grid)[number];
  cp_scale?: number;
  cp_count?: number;
  fc_limit?: number;
};

/** The forecast's ⚙ Options (028 A7): URL key and the values offered after the default. */
export const FORECAST_OPTIONS = {
  arima_p: [0, 1, 2, 3, 4, 5],
  arima_q: [0, 1, 2, 3, 4, 5],
  arima_sp: [0, 1, 2],
  arima_sq: [0, 1, 2],
  arima_d: [0, 1, 2],
  arima_ic: ["aicc", "aic", "bic"],
  gbm_grid: ["fixed", "small", "wide"],
  cp_scale: [0.001, 0.01, 0.05, 0.1, 0.5],
  cp_count: [0, 5, 10, 25, 50],
  fc_limit: [10, 30, 60, 120, 300],
} as const;
export type ForecastOption = keyof typeof FORECAST_OPTIONS;
/** Each option's default in the engine (`forecast.Tuning`); the time limit has none. */
export const FORECAST_DEFAULTS: Record<ForecastOption, number | string | null> = {
  arima_p: 3,
  arima_q: 3,
  arima_sp: 1,
  arima_sq: 1,
  arima_d: 2,
  arima_ic: "aicc",
  gbm_grid: "fixed",
  cp_scale: 0.05,
  cp_count: 25,
  fc_limit: null,
};
/** The models each option tunes (the time limit: all of them). */
export const FORECAST_OPTION_MODELS: Record<ForecastOption, string[] | null> = {
  arima_p: ["arima_auto", "sarimax"],
  arima_q: ["arima_auto", "sarimax"],
  arima_sp: ["arima_auto", "sarimax"],
  arima_sq: ["arima_auto", "sarimax"],
  arima_d: ["arima_auto", "sarimax"],
  arima_ic: ["arima_auto", "sarimax"],
  gbm_grid: ["gbm_lags"],
  cp_scale: ["prophet_like"],
  cp_count: ["prophet_like"],
  fc_limit: null,
};
const TUNING_KEYS = {
  arima_p: "max_p",
  arima_q: "max_q",
  arima_sp: "max_seasonal_p",
  arima_sq: "max_seasonal_q",
  arima_d: "max_d",
  arima_ic: "criterion",
  gbm_grid: "gbm_grid",
  cp_scale: "changepoint_scale",
  cp_count: "changepoints",
  fc_limit: "time_limit",
} as const satisfies Record<ForecastOption, keyof ForecastTuning>;

/** The options set in the URL, as the forecast's `tuning` (only the keys that were changed). */
export function forecastTuning(search: TimeSeriesSearch): ForecastTuning {
  const found: Record<string, number | string> = {};
  for (const option of Object.keys(FORECAST_OPTIONS) as ForecastOption[]) {
    const value = search[option];
    if (value !== undefined) found[TUNING_KEYS[option]] = value;
  }
  return found as ForecastTuning;
}

export const KALMAN_FORMS = ["llevel", "lltrend", "llevel_seasonal", "lltrend_seasonal"] as const;
/** The Smooth row's overridable parameters: URL key and the values offered after "Auto". */
export const SMOOTH_OPTIONS = {
  ewma_alpha: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
  lowess_frac: [0.1, 0.2, 0.3, 0.5, 0.67, 0.8],
  savgol_window: [5, 7, 11, 15, 21, 31, 51],
  savgol_order: [1, 2, 3, 4],
  kalman_model: KALMAN_FORMS,
  hp_lambda: [6.25, 100, 1600, 14400, 129600],
} as const;
export type SmoothOption = keyof typeof SMOOTH_OPTIONS;

const among = (value: unknown, allowed: readonly number[]) => {
  const number = Number(value);
  return value !== undefined && allowed.includes(number) ? number : undefined;
};

const text = (value: unknown) =>
  typeof value === "string" && value ? value : undefined;
const oneOf = <T extends string>(
  value: unknown,
  allowed: readonly T[],
): T | undefined =>
  typeof value === "string" && (allowed as readonly string[]).includes(value)
    ? (value as T)
    : undefined;
const integer = (value: unknown, low: number, high: number) => {
  const number = Number(value);
  return Number.isInteger(number) && number >= low && number <= high
    ? number
    : undefined;
};

export function validateTimeSeriesSearch(
  search: Record<string, unknown>,
): TimeSeriesSearch {
  return {
    time: text(search.time),
    value: text(search.value),
    group: text(search.group),
    series: text(search.series),
    period: oneOf(search.period, PERIODS),
    agg: oneOf(search.agg, AGGREGATES),
    window: integer(search.window, 2, 365),
    align: oneOf(search.align, ["trailing", "centered"] as const),
    gaps: oneOf(search.gaps, GAPS),
    season: integer(search.season, 2, 400),
    method: oneOf(search.method, DECOMPOSITIONS),
    windows: typeof search.windows === "string" || typeof search.windows === "number" ? String(search.windows) : undefined,
    spread: integer(search.spread, 2, 365),
    ar: integer(search.ar, 1, 24),
    transform: oneOf(search.transform, TRANSFORMS),
    pacf: oneOf(search.pacf, PACF),
    rsource: oneOf(search.rsource, ["series", "returns"] as const),
    segment: oneOf(search.segment, ["binseg"] as const),
    cost: oneOf(search.cost, ["mean", "mean_var", "trend"] as const),
    sens: oneOf(search.sens, ["low", "high"] as const),
    anomaly: oneOf(search.anomaly, ["iqr", "rolling_z", "hampel", "gesd", "isolation_forest", "lof", "autoencoder", "kalman"] as const),
    threshold: [3, 4].includes(Number(search.threshold))
      ? Number(search.threshold)
      : undefined,
    all: search.all === true || search.all === "true" ? true : undefined,
    scen: parseScenarios(search.scen),
    smodel: oneOf(search.smodel, ["sarimax", "tslm", "gbm_lags", "prophet_like"] as const),
    ewma_alpha: among(search.ewma_alpha, SMOOTH_OPTIONS.ewma_alpha),
    lowess_frac: among(search.lowess_frac, SMOOTH_OPTIONS.lowess_frac),
    savgol_window: among(search.savgol_window, SMOOTH_OPTIONS.savgol_window),
    savgol_order: among(search.savgol_order, SMOOTH_OPTIONS.savgol_order),
    kalman_model: oneOf(search.kalman_model, KALMAN_FORMS),
    hp_lambda: among(search.hp_lambda, SMOOTH_OPTIONS.hp_lambda),
    arima_p: among(search.arima_p, FORECAST_OPTIONS.arima_p),
    arima_q: among(search.arima_q, FORECAST_OPTIONS.arima_q),
    arima_sp: among(search.arima_sp, FORECAST_OPTIONS.arima_sp),
    arima_sq: among(search.arima_sq, FORECAST_OPTIONS.arima_sq),
    arima_d: among(search.arima_d, FORECAST_OPTIONS.arima_d),
    arima_ic: oneOf(search.arima_ic, FORECAST_OPTIONS.arima_ic),
    gbm_grid: oneOf(search.gbm_grid, FORECAST_OPTIONS.gbm_grid),
    cp_scale: among(search.cp_scale, FORECAST_OPTIONS.cp_scale),
    cp_count: among(search.cp_count, FORECAST_OPTIONS.cp_count),
    fc_limit: among(search.fc_limit, FORECAST_OPTIONS.fc_limit),
  };
}

/** The parameters sent to the API (display-only choices left out). */
export function seriesParams(search: TimeSeriesSearch): SeriesParams {
  const params: TimeSeriesSearch = { ...search };
  const first = parseWindows(search.windows)[0];
  if (first) params.window = first;
  delete params.method;
  delete params.windows;
  delete params.spread;
  delete params.ar;
  delete params.transform;
  delete params.pacf;
  for (const key of [
    "rsource",
    "segment",
    "cost",
    "sens",
    "anomaly",
    "threshold",
  ] as const)
    delete params[key];
  delete params.all;
  delete params.scen;
  delete params.smodel;
  for (const key of Object.keys(SMOOTH_OPTIONS) as SmoothOption[]) delete params[key];
  for (const key of Object.keys(FORECAST_OPTIONS) as ForecastOption[]) delete params[key];
  return params;
}

/** The Smooth row's forced parameters, as query values of the smoothing call. */
export function smoothParams(search: TimeSeriesSearch): Record<string, string> {
  const found: Record<string, string> = {};
  for (const key of Object.keys(SMOOTH_OPTIONS) as SmoothOption[]) {
    const value = search[key];
    if (value !== undefined) found[key] = String(value);
  }
  return found;
}
