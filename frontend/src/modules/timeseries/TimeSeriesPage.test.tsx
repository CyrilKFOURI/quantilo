import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot(
    { label },
    ref,
  ) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const periods = Array.from(
  { length: 12 },
  (_, i) => `2024-${String(i + 1).padStart(2, "0")}`,
);
const line = (label: string | null) => ({
  label,
  values: periods.map((_, i) => 100 + 10 * i),
  counts: periods.map(() => 30),
  rolling_mean: periods.map((_, i) => (i < 2 ? null : 90 + 10 * i)),
  rolling_std: periods.map(() => 4),
});
const setup = {
  time: "day",
  value: "units",
  group: null,
  series: null,
  dates: ["day"],
  numbers: ["units", "revenue"],
  categories: ["store"],
  groups: [],
  period: "month",
  aggregate: "sum",
  period_auto: true,
  aggregate_auto: true,
  coarsened: false,
  first_partial: false,
  last_partial: false,
  gaps: "leave",
  season: 12,
  periods: 12,
  rows: 360,
  total: 360,
};
const method = (key: string, overrides: Record<string, unknown> = {}) => ({
  key,
  statistic: 3.2,
  p_value: 0.001,
  n: 12,
  significant: true,
  value: 0.8,
  low: null,
  high: null,
  note: null,
  ...overrides,
});
const overview = {
  setup,
  periods,
  analyzed: line(null),
  lines: [],
  figures: {
    start: periods[0],
    end: periods[11],
    periods: 12,
    missing: 0,
    change: 42.3,
  },
  window: 3,
  align: "trailing",
  methods: [
    method("mann_kendall"),
    method("sen", {
      statistic: null,
      p_value: null,
      value: 10,
      low: 8,
      high: 12,
    }),
  ],
  message: null,
};
const seasonality = {
  setup,
  detrend: "straight line",
  profiles: [
    {
      key: "month",
      labels: ["Jan", "Feb"],
      values: [
        [1, 2],
        [3, 4],
      ],
      deviation: [-20, 38],
      test: method("kruskal_season", { value: 0.3 }),
      strongest: "Feb",
    },
  ],
  overlay: null,
  methods: [method("kruskal_season", { value: 0.3 })],
  filled: 0,
  message: null,
};
const decomposition = {
  setup,
  periods,
  observed: line(null).values,
  result: {
    method: "stl",
    seasons: [12],
    trend: periods.map(() => 1),
    seasonal: [periods.map(() => 0)],
    resid: periods.map(() => 0),
    strength_trend: 0.9,
    strength_seasonal: 0.2,
    notes: [],
  },
  filled: 0,
  message: null,
};

const stationarity = {
  setup,
  transform: "none",
  regression: "c",
  periods,
  transformed: line(null).values,
  synthesis: { verdict: "unit_root", adf_rejects: false, kpss_rejects: true },
  suggestion: { transform: "diff", verdict: "stationary" },
  methods: [
    method("adf", { significant: false, p_value: 0.6, value: null }),
    method("kpss", { value: null }),
  ],
  filled: 0,
  message: null,
};
const autocorrelation = {
  setup,
  transform: "none",
  correlogram: {
    lags: [1, 2, 3],
    acf: [0.9, 0.8, 0.7],
    pacf: [0.9, 0.1, 0.05],
    band: 0.2,
    readings: ["pacf_cutoff"],
    ar_order: 1,
  },
  methods: [method("ljung_box", { value: 1 })],
  filled: 0,
  message: null,
};

const residualSummary = {
  rows: 12,
  count: 12,
  missing: 0,
  infinite: 0,
  zeros: 0,
  negatives: 4,
  distinct: 12,
  distinct_approx: false,
  mean: 0.1,
  std: 2.5,
  cv: null,
  min: -4,
  p1: -4,
  p5: -4,
  p10: -3,
  q1: -1,
  median: 0,
  q3: 1,
  p90: 3,
  p95: 4,
  p99: 4,
  max: 4,
  iqr: 2,
  mad: 1,
  skewness: 0.2,
  kurtosis: 4.5,
  sum: 1,
};
const volatility = {
  setup,
  residuals_from: "STL remainder",
  periods,
  residuals: periods.map(() => 1),
  rolling_std: periods.map(() => 1),
  window: 3,
  rising: true,
  log_helps: true,
  methods: [method("breusch_pagan"), method("arch_lm")],
  filled: 0,
  message: null,
};
const residuals = {
  setup,
  distribution: {
    source: "residuals",
    method: "STL remainder",
    describe: residualSummary,
    normality: {
      n: 12,
      sample: 12,
      tests: [
        { key: "shapiro", statistic: 0.8, p_value: 0.001, n: 12, note: null },
      ],
      reason: null,
    },
    density: null,
    qq: null,
    bins: [
      { start: -4, end: 0, count: 6 },
      { start: 0, end: 4, count: 6 },
    ],
    fit: {
      n: 12,
      grid: [-4, 0, 4],
      sample_quantiles: [-4, 0, 4],
      laws: [
        {
          key: "student_t",
          label: "Student's t",
          discrete: false,
          params: { df: 3 },
          log_likelihood: -20,
          aic: 46,
          bic: 47,
          ks: 0.1,
          density: [0.1, 0.4, 0.1],
          quantiles: [-4, 0, 4],
        },
      ],
      transforms: [],
      reason: null,
    },
  },
  filled: 0,
  message: null,
};
const breaks = {
  setup,
  periods,
  values: line(null).values,
  deseasonalized: true,
  breaks: {
    search: "pelt",
    cost: "trend",
    points: [6],
    segments: [
      { start: 0, end: 6, mean: 1, slope: 2, first: 0, last: 10 },
      { start: 6, end: 12, mean: 2, slope: 5, first: 12, last: 37 },
    ],
    cost_auto: true,
  },
  points: [
    {
      index: 6,
      period: periods[6],
      before: 10,
      after: 12,
      slope_before: 2,
      slope_after: 5,
    },
  ],
  methods: [method("pettitt", { value: 7 })],
  anomalies: [
    {
      index: 3,
      period: periods[3],
      start: "2024-04-01T00:00:00",
      end: "2024-04-30T23:59:59",
      value: 500,
      expected: 130,
      score: 7.2,
    },
  ],
  anomaly_method: "robust_z",
  threshold: 3.5,
  residuals_from: "STL remainder",
  filled: 0,
  message: null,
};

const future = ["2025-01", "2025-02", "2025-03"];
const modelResult = (
  key: string,
  mase: number,
  vs: number | null,
  info: string | null,
) => ({
  key,
  info,
  scores: {
    mae: 10,
    rmse: 12,
    mape: 5,
    smape: 5,
    wape: 6,
    mase,
    rmsse: 1,
    bias: 0,
    coverage80: 0.8,
    coverage95: 0.94,
    winkler95: 50,
  },
  vs_reference: vs,
  forecast: [1, 2, 3],
  low80: [0, 1, 2],
  high80: [2, 3, 4],
  low95: [-1, 0, 1],
  high95: [3, 4, 5],
  residual_check:
    key === "ets_auto"
      ? method("ljung_box", { significant: false, p_value: 0.4 })
      : null,
  seconds: 0.4,
  error: null,
  backtests: [
    { origin: 6, forecast: [1, 2, 3], low95: [0, 1, 2], high95: [2, 3, 4] },
  ],
  uses_drivers: key === "tslm",
  mase_without_drivers: key === "tslm" ? 0.95 : null,
  drivers:
    key === "tslm"
      ? [
          {
            driver: "visits",
            coefficient: 2.5,
            p_value: 0.0001,
            importance: null,
            mase_without: 0.95,
            verdict: "helps",
          },
        ]
      : [],
});
const forecastJob = {
  id: "job1",
  status: "done",
  done: 2,
  total: 2,
  current: null,
  error: null,
  result: {
    horizon: 3,
    periods,
    history: line(null).values,
    future,
    reference: "seasonal_naive",
    origins: 3,
    results: [
      modelResult("ets_auto", 0.7, 30, "ETS(A,A,A)"),
      modelResult("tslm", 0.8, 20, "Trend + seasonal dummies + 1 driver"),
      modelResult("seasonal_naive", 1, 0, null),
    ],
    best: "ets_auto",
    demand: {
      zeros_pct: 62,
      adi: 2.6,
      cv2: 0.3,
      kind: "intermittent",
      intermittent: true,
    },
    drivers: [],
    driver_notes: [
      "visits: its last value (1200) is repeated over the forecast, which assumes it stays flat.",
    ],
  },
};
const drivers = {
  setup,
  horizon: 3,
  candidates: [
    {
      column: "visits",
      flag: false,
      correlation: 0.72,
      suggested: true,
      future_periods: 0,
    },
    {
      column: "price",
      flag: false,
      correlation: 0.1,
      suggested: false,
      future_periods: 3,
    },
    {
      column: "holiday",
      flag: true,
      correlation: null,
      suggested: false,
      future_periods: 3,
    },
  ],
  calendar: [],
};

const smoothing = (keys: string[]) => ({
  setup,
  periods,
  curves: keys.map((key) => ({
    key,
    causal: key === "kalman_filtered",
    values: periods.map((_, i) => 100 + 10 * i),
    low: periods.map((_, i) => 90 + 10 * i),
    high: periods.map((_, i) => 110 + 10 * i),
    params: [
      { key: "model", value: "llevel_seasonal", auto: true, reason: "lowest AIC among local level, local linear trend" },
    ],
    info: "Local level + seasonal (12), chosen by AIC · noise σ = 4.1",
    notes: [],
    error: null,
  })),
  suggestions: keys.length
    ? []
    : [{ key: "kalman_smoothed", reason: "2 of 12 periods are empty (17%): Kalman estimates them with an uncertainty band." }],
  unavailable: {},
  message: null,
});

const columns = [
  column({
    name: "day",
    type: "datetime",
    detected_type: "datetime",
    role: null,
  }),
  column({ name: "units", position: 1, role: "quantity" }),
];

const regimes = {
  setup: overview.setup,
  periods: ["2024-01-01", "2024-01-02", "2024-01-03"],
  filled: 0,
  regimes: {
    on: "changes",
    on_reason: "R-RG-01: the series has a unit root (ADF p = 0.8).",
    k: 2,
    k_reason: "R-RG-02: 2 regimes, the lower BIC.",
    start: 1,
    fits: [],
    regimes: [
      { name: "calm", mean: 0, std: 1, share: 0.7, expected_duration: 40, stay: 0.975 },
      { name: "turbulent", mean: 0, std: 4, share: 0.3, expected_duration: 12, stay: 0.92 },
    ],
    transition: [[0.975, 0.025], [0.08, 0.92]],
    probabilities: [[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]],
    most_likely: [0, 0, 1],
    values: [1, -1, 6],
    switches: 1,
    sentence: "2 regimes of the changes, 1 switches; the series ends in the turbulent regime.",
    message: null,
  },
};

function setupApi() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /modules/timeseries/datasets/d1/overview": overview,
    "GET /modules/timeseries/datasets/d1/smoothing": smoothing([]),
    "GET /modules/timeseries/datasets/d1/seasonality": seasonality,
    "GET /modules/timeseries/datasets/d1/decomposition": decomposition,
    "GET /modules/timeseries/datasets/d1/stationarity": stationarity,
    "GET /modules/timeseries/datasets/d1/autocorrelation": autocorrelation,
    "GET /modules/timeseries/datasets/d1/volatility": volatility,
    "GET /modules/timeseries/datasets/d1/residuals": residuals,
    "GET /modules/timeseries/datasets/d1/breaks": breaks,
    "GET /modules/timeseries/datasets/d1/regimes": regimes,
    "GET /modules/timeseries/datasets/d1/drivers": drivers,
    "POST /modules/timeseries/datasets/d1/forecast": {
      ...forecastJob,
      status: "running",
      done: 0,
      result: null,
    },
    "GET /modules/timeseries/forecast-jobs/job1": forecastJob,
  });
}

/** 098: a page with a chosen series (opened empty otherwise), its sections open. */
const SERIES = "/d/d1/timeseries?time=day&value=units";

beforeEach(() => {
  localStorage.setItem("de.ts.open", JSON.stringify(["trend", "seasonality", "decomposition", "stationarity", "autocorrelation", "volatility", "residuals", "breaks", "forecast"]));
});

afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

describe("Time series page", () => {
  it("opens configured, with a verdict per section", async () => {
    setupApi();
    renderAt(SERIES);

    const overviewSection = await screen.findByRole("region", {
      name: "Overview",
    });
    expect(
      within(overviewSection).getAllByText("Rising trend").length,
    ).toBeGreaterThan(0);
    expect(within(overviewSection).getByText("+42.3 %")).toBeInTheDocument();
    const trend = screen.getByRole("region", { name: "Trend" });
    expect(
      within(trend).getByText("A monotonic trend is present."),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Value")).toHaveValue("units");

    const season = screen.getByRole("region", { name: "Seasonality" });
    expect(
      await within(season).findByText("Strong yearly (by month) pattern"),
    ).toBeInTheDocument();
    expect(
      within(season).getByText(
        "Feb is 38 % above average once the trend is removed.",
      ),
    ).toBeInTheDocument();

    const parts = screen.getByRole("region", { name: "Decomposition" });
    expect(
      await within(parts).findByText("Trend strong, seasonality weak"),
    ).toBeInTheDocument();
  });

  it("adds a method from the catalogue and asks the server for it", async () => {
    const calls = setupApi();
    renderAt(SERIES);
    await screen.findByRole("region", { name: "Overview" });
    const trend = screen.getByRole("region", { name: "Trend" });

    fireEvent.click(
      within(trend).getByRole("button", { name: "Add method" }),
    );
    fireEvent.click(
      screen.getByRole("menuitemcheckbox", { name: /Cox–Stuart/ }),
    );

    await waitFor(() => {
      const last = calls.filter((c) => c.key.endsWith("/overview")).at(-1);
      expect(last?.params.get("methods")).toBe("mann_kendall,sen,cox_stuart");
    });
  });

  it("keeps the configuration in the URL", async () => {
    const calls = setupApi();
    const router = renderAt(
      "/d/d1/timeseries?time=day&value=revenue&period=week&gaps=interpolate",
    );
    await screen.findByRole("region", { name: "Overview" });

    const sent = calls.find((c) => c.key.endsWith("/overview"));
    expect(sent?.params.get("value")).toBe("revenue");
    expect(sent?.params.get("period")).toBe("week");
    expect(sent?.params.get("gaps")).toBe("interpolate");

    fireEvent.change(screen.getByLabelText("Aggregate"), {
      target: { value: "mean" },
    });
    await waitFor(() =>
      expect(router.state.location.search).toMatchObject({ agg: "mean" }),
    );
  });

  it("tests stationarity, suggests a transformation and reads the correlogram", async () => {
    const calls = setupApi();
    const router = renderAt(SERIES);

    const section = await screen.findByRole("region", { name: "Stationarity" });
    expect(
      await within(section).findByText(
        "Not stationary: ADF and KPSS agree on a unit root. Suggested: difference, which makes it stationary.",
      ),
    ).toBeInTheDocument();
    expect(
      within(section).getByRole("radio", { name: /Difference\s*Suggested/ }),
    ).toBeInTheDocument();

    const memory = screen.getByRole("region", { name: "Autocorrelation" });
    expect(
      await within(memory).findByText("Short memory, AR(1)"),
    ).toBeInTheDocument();
    expect(
      within(memory).getByText(/The PACF stops after lag 1/),
    ).toBeInTheDocument();

    fireEvent.click(
      within(section).getByRole("radio", { name: /Difference\s*Suggested/ }),
    );
    await waitFor(() =>
      expect(router.state.location.search).toMatchObject({ transform: "diff" }),
    );
    await waitFor(() => {
      const last = calls
        .filter((c) => c.key.endsWith("/autocorrelation"))
        .at(-1);
      expect(last?.params.get("transform")).toBe("diff");
    });
  });

  it("shows volatility, the distribution of residuals, breaks and anomalies", async () => {
    setupApi();
    renderAt(SERIES);

    const spread = await screen.findByRole("region", { name: "Volatility" });
    expect(
      await within(spread).findByText("Variability changes (rising)"),
    ).toBeInTheDocument();
    expect(
      within(spread).getByText(/a log transformation usually stabilises it/),
    ).toBeInTheDocument();

    const shape = screen.getByRole("region", { name: "Residuals" });
    expect(
      await within(shape).findByText("Not normal · best law: Student's t"),
    ).toBeInTheDocument();
    expect(within(shape).getByText("Excess kurtosis")).toBeInTheDocument();

    const regime = screen.getByRole("region", { name: "Breaks & anomalies" });
    expect(
      await within(regime).findByText("1 breaks · 1 anomalies"),
    ).toBeInTheDocument();
    expect(
      within(regime).getByText(
        /The trend changes from 2 to 5 per month from 2024-07/,
      ),
    ).toBeInTheDocument();
    const rows = within(regime).getByRole("link", { name: "Show rows" });
    expect(decodeURIComponent(rows.getAttribute("href") ?? "")).toContain(
      "2024-04-30T23:59:59",
    );
  });

  it("runs a forecast on demand and ranks the models", async () => {
    const calls = setupApi();
    renderAt(SERIES);
    const section = await screen.findByRole("region", { name: "Forecast" });
    expect(calls.some((c) => c.key.endsWith("/forecast"))).toBe(false); // not run automatically

    fireEvent.click(
      within(section).getByRole("button", { name: "Run forecast" }),
    );

    expect(
      await within(section).findByText(
        "ETS (automatic): 30 % better than Seasonal naive",
      ),
    ).toBeInTheDocument();
    expect(within(section).getByText("ETS(A,A,A)")).toBeInTheDocument();
    expect(
      within(section).getByText(/its residuals look like white noise/),
    ).toBeInTheDocument();
    const sent = calls.find((c) => c.key.endsWith("/forecast"));
    expect(sent?.params.get("horizon")).toBe("3"); // at most a quarter of the 12 periods
    expect(sent?.params.get("methods")).toBe(
      "naive,seasonal_naive,ets_auto,arima_auto",
    );

    // 028: zeros suggest Croston-type models; the rolling origins can be shown
    expect(
      within(section).getByText(
        "62 % of the periods are zero: intermittent demand.",
      ),
    ).toBeInTheDocument();
    fireEvent.click(
      within(section).getByRole("button", { name: "Add Croston-type models" }),
    );
    expect(within(section).getByText("Croston")).toBeInTheDocument();
    fireEvent.click(
      within(section).getByRole("button", {
        name: "How the models were scored",
      }),
    );
    expect(
      within(section).getByRole("button", { name: "Hide the scoring" }),
    ).toBeInTheDocument();
  });

  it("adds drivers from the suggestion and the panel, and shows what each brings", async () => {
    const calls = setupApi();
    renderAt(SERIES);
    const section = await screen.findByRole("region", { name: "Forecast" });
    const panel = within(section).getByRole("button", { name: "Drivers" });
    expect(panel).toHaveAttribute("aria-expanded", "false"); // collapsed by default

    expect(
      await within(section).findByText(
        "units and visits move together (r = 0.72 between their changes): use visits as a driver.",
      ),
    ).toBeInTheDocument();
    fireEvent.click(within(section).getByRole("button", { name: "Add driver" }));
    expect(within(section).queryByText(/move together/)).not.toBeInTheDocument();
    await waitFor(() =>
      expect(
        within(section).getByRole("button", { name: /Drivers/ }),
      ).toHaveAttribute("aria-expanded", "true"),
    );
    // no future values for visits: past only, lagged by the horizon
    expect(within(section).getByText(/Enters lagged by 3 periods/)).toBeInTheDocument();
    expect(
      within(section).getByText("None of the chosen models uses drivers."),
    ).toBeInTheDocument();
    fireEvent.click(
      within(section).getByRole("button", {
        name: "Add SARIMAX (ARIMA + drivers)",
      }),
    );
    // price has future rows: known in advance, from the file
    fireEvent.change(within(section).getByLabelText("Add a column…"), {
      target: { value: "price" },
    });
    expect(
      within(section).getByRole("option", { name: "From the file (3 of 3)" }),
    ).toBeInTheDocument();
    fireEvent.change(
      within(section).getByLabelText("Add a 0/1 flag (holidays, events)…"),
      { target: { value: "holiday" } },
    );

    fireEvent.click(
      within(section).getByRole("button", { name: "Run forecast" }),
    );

    expect(
      await within(section).findByText("Drivers, model by model"),
    ).toBeInTheDocument();
    const sent = calls.find((c) => c.key.endsWith("/forecast"));
    expect(sent?.params.get("methods")).toContain("sarimax");
    expect(sent?.body).toEqual({
      drivers: [
        { column: "visits", kind: "column", known: false, future: "file", value: null },
        { column: "price", kind: "column", known: true, future: "file", value: null },
        { column: "holiday", kind: "flag", known: true, future: "file", value: null },
      ],
    });
    expect(within(section).getByText("Without drivers: MASE 0.95")).toBeInTheDocument();
    expect(within(section).getByText("Helps")).toBeInTheDocument();
    expect(within(section).getByText("+2.5 per unit")).toBeInTheDocument();
    expect(within(section).getByText(/its last value \(1200\) is repeated/)).toBeInTheDocument();
  });

  it("is a card of the Toolbox once the dataset has a date, not a sidebar link (098)", async () => {
    setupApi();
    renderAt("/d/d1/toolbox");

    const card = await screen.findByRole("link", { name: /^Time series/ });
    expect(card).toHaveAttribute("href", "/d/d1/timeseries");
    const nav = screen.getByRole("navigation", { name: "Main navigation" });
    expect(within(nav).queryByRole("link", { name: "Time series" })).not.toBeInTheDocument();
  });
});

describe("Opened from the Toolbox, folded sections (098)", () => {
  it("opens empty: only the date and value pickers, nothing computed until both are chosen", async () => {
    localStorage.clear();
    const calls = setupApi();
    const router = renderAt("/d/d1/timeseries");

    expect(await screen.findByRole("heading", { name: "Pick a series" })).toBeInTheDocument();
    expect(screen.queryByRole("region", { name: "Overview" })).not.toBeInTheDocument();
    const computed = () => calls.filter((c) => /\/modules\/timeseries\/datasets\/d1\/(?!analyses)/.test(c.key));
    expect(computed()).toHaveLength(0);

    fireEvent.change(screen.getByLabelText("Date"), { target: { value: "day" } });
    await waitFor(() => expect(router.state.location.search).toMatchObject({ time: "day" }));
    expect(screen.getByRole("heading", { name: "Pick a series" })).toBeInTheDocument();
    expect(computed()).toHaveLength(0);

    fireEvent.change(screen.getByLabelText("Value"), { target: { value: "units" } });
    expect(await screen.findByRole("region", { name: "Overview" })).toBeInTheDocument();
    expect(router.state.location.search).toMatchObject({ time: "day", value: "units" });
    expect(calls.find((c) => c.key.endsWith("/overview"))?.params.get("value")).toBe("units");
  });

  it("closes every section by default, computes one only when opened and remembers it", async () => {
    localStorage.clear();
    const calls = setupApi();
    renderAt(SERIES);
    await screen.findByRole("region", { name: "Overview" });

    const title = screen.getByRole("button", { name: /^Seasonality/ });
    expect(title).toHaveAttribute("aria-expanded", "false");
    for (const name of [/^Trend/, /^Stationarity/, /^Breaks & anomalies/, /^Forecast/]) {
      expect(screen.getByRole("button", { name })).toHaveAttribute("aria-expanded", "false");
    }
    // the series itself is shown; no section is asked for
    expect(calls.some((c) => c.key.endsWith("/overview"))).toBe(true);
    for (const section of ["seasonality", "decomposition", "stationarity", "autocorrelation", "volatility", "residuals", "breaks", "drivers"]) {
      expect(calls.some((c) => c.key.endsWith(`/${section}`))).toBe(false);
    }

    fireEvent.click(title);
    expect(title).toHaveAttribute("aria-expanded", "true");
    const season = screen.getByRole("region", { name: "Seasonality" });
    expect(await within(season).findAllByText("Strong yearly (by month) pattern")).not.toHaveLength(0);
    expect(calls.some((c) => c.key.endsWith("/seasonality"))).toBe(true);
    expect(calls.some((c) => c.key.endsWith("/stationarity"))).toBe(false);
    expect(JSON.parse(localStorage.getItem("de.ts.open") ?? "[]")).toEqual(["seasonality"]);

    // closed again, its verdict stays on its line
    fireEvent.click(title);
    expect(title).toHaveAttribute("aria-expanded", "false");
    expect(within(season).getByText("Strong yearly (by month) pattern")).toBeInTheDocument();
  });

  it("still opens when the browser keeps nothing", async () => {
    localStorage.clear();
    const getItem = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    const setItem = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    setupApi();
    renderAt(SERIES);
    await screen.findByRole("region", { name: "Overview" });
    const title = screen.getByRole("button", { name: /^Volatility/ });
    fireEvent.click(title);
    expect(title).toHaveAttribute("aria-expanded", "true");
    getItem.mockRestore();
    setItem.mockRestore();
  });
});

describe("Smooth row and Kalman gaps (064)", () => {
  it("starts with the moving average, suggests Kalman and adds it on demand", async () => {
    const calls = setupApi();
    renderAt(SERIES);
    const overviewSection = await screen.findByRole("region", { name: "Overview" });
    const row = within(overviewSection).getByRole("group", { name: "Smooth" });

    expect(within(row).getByText("Moving average")).toBeInTheDocument();
    expect(within(row).getByText("Causal")).toBeInTheDocument();
    // nothing but the rules is asked for when the page opens
    await waitFor(() => {
      const sent = calls.find((c) => c.key.endsWith("/smoothing"));
      expect(sent?.params.get("methods")).toBe("");
    });
    expect(await within(overviewSection).findByText(/2 of 12 periods are empty/)).toBeInTheDocument();

    fireEvent.click(within(overviewSection).getByRole("button", { name: "Add Kalman (smoothed)" }));
    await waitFor(() => {
      const last = calls.filter((c) => c.key.endsWith("/smoothing")).at(-1);
      expect(last?.params.get("methods")).toBe("kalman_smoothed");
    });
    expect(within(row).getByText("Uses future values")).toBeInTheDocument();
  });

  it("keeps niche smoothers two folds deep", async () => {
    setupApi();
    renderAt(SERIES);
    const overviewSection = await screen.findByRole("region", { name: "Overview" });
    fireEvent.click(within(overviewSection).getByRole("button", { name: "Add a smoothing method" }));

    const more = screen.getByText("More").closest("details");
    // the Advanced of the smoothing list (Breaks has its own, 062)
    const advanced = more?.parentElement?.closest("details") ?? null;
    expect(advanced).not.toHaveAttribute("open");
    expect(more).not.toHaveAttribute("open");
    expect(advanced?.contains(more ?? null)).toBe(true);
    expect(more?.textContent).toMatch(/Savitzky–Golay/);
    expect(more?.textContent).toMatch(/Hodrick–Prescott/);
    expect(advanced?.textContent).toMatch(/EWMA/);
  });

  it("suggests Kalman for empty periods and fills them on request", async () => {
    const calls = mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": columns,
      "GET /column-roles": roles,
      "GET /modules/timeseries/datasets/d1/overview": {
        ...overview,
        figures: { ...overview.figures, missing: 2 },
      },
      "GET /modules/timeseries/datasets/d1/smoothing": smoothing([]),
    });
    const router = renderAt(SERIES);

    expect(await screen.findByText("2 empty periods. Kalman fills them with an uncertainty band.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Use Kalman" }));
    await waitFor(() => expect(router.state.location.search).toMatchObject({ gaps: "kalman" }));
    await waitFor(() => {
      const last = calls.filter((c) => c.key.endsWith("/overview")).at(-1);
      expect(last?.params.get("gaps")).toBe("kalman");
    });
  });
});

describe("Regimes (062)", () => {
  it("are folded under Advanced in Breaks and fitted only when asked", async () => {
    const calls = setupApi();
    renderAt(SERIES);
    const summary = await screen.findByText("Regimes (Markov switching)");
    const asked = () => calls.filter((c) => c.key.endsWith("/regimes"));
    expect(asked()).toHaveLength(0);
    fireEvent.click(summary);
    fireEvent.change(screen.getByLabelText("Regimes"), { target: { value: "2" } });
    fireEvent.click(screen.getByRole("button", { name: "Find regimes" }));
    expect(await screen.findByText(/ends in the turbulent regime/)).toBeInTheDocument();
    expect(asked().at(-1)?.params.get("k")).toBe("2");
    expect(screen.getByText(/R-RG-01/)).toBeInTheDocument();
  });
});
