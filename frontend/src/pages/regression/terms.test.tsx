import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

afterEach(() => vi.unstubAllGlobals());

const columns = [
  column({ name: "price", position: 0 }),
  column({ name: "area", position: 1, role: null }),
  column({ name: "city", position: 2, type: "categorical_nominal", detected_type: "categorical_nominal", role: null }),
  column({ name: "sold", position: 3, type: "datetime", detected_type: "datetime", role: null }),
];

const coefficient = (overrides: Record<string, unknown>) => ({
  term: "area",
  column: "area",
  level: null,
  estimate: 2980.5,
  std_error: 40.2,
  ci_low: 2901.7,
  ci_high: 3059.3,
  statistic: 74.1,
  p_value: 1e-40,
  q_value: 2e-40,
  evidence: "strong",
  standardized: 0.91,
  std_low: 0.89,
  std_high: 0.93,
  odds_ratio: null,
  marginal: null,
  marginal_low: null,
  marginal_high: null,
  sentence: "Each extra unit of `area` adds 2,980 to `price`.",
  ratio: null,
  ratio_low: null,
  ratio_high: null,
  outcome: null,
  group: "area",
  ...overrides,
});

const regression = {
  target: "price",
  positive: null,
  model: "ols",
  model_label: "Least squares (OLS)",
  model_reason: "A number to explain: least squares.",
  columns: ["area", "city"],
  focus: null,
  terms: [
    coefficient({ term: "(Intercept)", column: "(Intercept)", estimate: 50000, evidence: "none", q_value: null, standardized: null, group: "" }),
    coefficient({}),
    coefficient({ term: "city = Lyon", column: "city", level: "Lyon", estimate: -60000, group: "city", sentence: "With `city` = Lyon, `price` is 60,000 lower." }),
  ],
  se: { kind: "hc3", label: "HC3", rule: "R-REG-SE-05", reason: "HC3 always.", lag: null, groups: [], clusters: null, warning: null },
  weights: null,
  quality: {
    rows: 1200, loaded: 1200, total: 1200, sampled: false, missing: [], r2: 0.97, adj_r2: 0.969, oos_r2: 0.968,
    oos_how: "5 random folds", aic: 28000, bic: 28020, f: 3000, f_p: 0, df_model: 2, df_resid: 1197, pseudo_r2: null,
  },
  multiple: { method: "holm", label: "Holm", tests: 2, significant: 2, kept: 2, alpha: 0.05, auto: true, reason: "Holm.", sentence: "All 2 effects stay." },
  sentence: "Each extra unit of `area` adds 2,980 to `price`.",
  checks: [
    { key: "specification", status: "warn", title: "Straight lines", sentence: "`area` bends the most. Try a curve on it.", rule: "R-REG-10", action: "term", column: "area", family: null, term: "spline:area" },
    { key: "influence", status: "warn", title: "Influential rows", sentence: "38 rows weigh heavily. Huber's robust fit shows whether they change the coefficients.", rule: "R-REG-11", action: "robust", column: null, family: null, term: null },
  ],
  references: [{ column: "city", level: "Paris", levels: 1, others: 0, reason: "most frequent level", coding: "reference" }],
  left_out: [],
  structure: { kind: "none", date: null, entity: null, period: null, periods: 0, entities: 0, message: null },
  hints: [],
  plots: null,
  family: null,
};

const terms = {
  references: regression.references,
  left_out: [],
  vifs: [],
  condition: 3.1,
  sentence: "2 terms estimated.",
  specs: ["spline:area"],
  center: false,
  coding: "reference",
  notes: [{ key: "spline:area", label: "curve of `area`", status: "used", reason: "restricted cubic spline, 5 knots", rule: "R-REG-TERM-02" }],
  tests: [
    { key: "spline:area", question: "Is the effect of `area` a curve?", columns: 3, statistic: 41, df_num: 3, df_den: 1190, p_value: 1e-9, status: "ok", sentence: "`area` is a curve (F = 41, p < 0.001)." },
  ],
  curves: [{ column: "area", x: [1, 2], lines: [{ label: "fitted", y: [1, 2], low: [0, 1], high: [2, 3] }], sentence: "Along `area`, the fitted `price` goes from 1 to 2.", scale: "`price`" }],
  slopes: [],
  contrasts: [],
  numbers: ["area"],
};

const partial = {
  column: "area", available: true, reason: "", x_resid: [0, 1], y_resid: [0, 2], slope: 2, x: [1, 2], component_resid: [1, 2], line_x: [1, 2], line_y: [1, 2],
  rows: [0, 1], points: 2, of: 2, sentence: "Once the other columns are taken out, the slope of `area` is 2 (its coefficient).",
};

const robust = {
  available: true,
  reason: "",
  terms: ["(Intercept)", "area"],
  columns: [
    { kind: "ols", label: "Least squares", estimates: [1, 2], low: [0, 1], high: [2, 3], p_values: [0.01, 0.001], note: "HC3 standard errors" },
    { kind: "huber", label: "Huber", estimates: [1, 1.5], low: [0, 1], high: [2, 2], p_values: [0.01, 0.001], note: "k = 1.345" },
    { kind: "tukey", label: "Tukey biweight", estimates: [1, 1.4], low: [0, 1], high: [2, 2], p_values: [0.01, 0.001], note: "c = 4.685" },
    { kind: "quantile", label: "Quantile τ = 0.5", estimates: [1, 1.8], low: [0, 1], high: [2, 3], p_values: [0.01, 0.001], note: "the median" },
  ],
  quantile: 0.5,
  quantiles: [0.1, 0.25, 0.5, 0.75, 0.9],
  moved: ["area"],
  downweighted: 120,
  rejected: 8,
  rows: 1200,
  quantile_rows: 1200,
  laws: [{ name: "Normal", aic: 100, df: null, best: false }, { name: "Student t", aic: 90, df: 3.4, best: true }],
  law_sentence: "The residuals follow a Student t law best.",
  tail_xi: 0.31,
  tail_sentence: "The residuals have a heavy tail (Hill ξ = 0.31).",
  sentence: "Huber weighs 120 rows less than least squares does; `area` moves by more than one standard error.",
  notes: [],
};

const quantiles = {
  available: true,
  reason: "",
  taus: [0.1, 0.5, 0.9],
  curves: [{ term: "area", taus: [0.1, 0.5, 0.9], estimates: [1, 2, 3], low: [0.5, 1.5, 2.5], high: [1.5, 2.5, 3.5], ols: 2, ols_low: 1.8, ols_high: 2.2, difference: 2, p_value: 0.001, q_value: 0.001, sentence: "The effect of `area` grows along the distribution." }],
  tested: 1,
  changing: 1,
  rows: 1200,
  sampled: false,
  multiple: null,
  sentence: "1 of 1 effects change along the distribution of `price`.",
  t_errors: { available: true, reason: "", converged: true, df: 3.2, scale: 1, terms: [], estimates: [], low: [], high: [], aic: 1, ols_aic: 2, sentence: "With Student-t errors, ν = 3.2: heavy tails." },
};

const more = { white: null, white_sentence: "White's test.", removed: 0, shifts: [], shift_sentence: "No influential row to leave out." };
const families = { current: "ols", alternatives: [], sentence: "No other family fits this target." };

function routes() {
  return {
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/regression": regression,
    "GET /datasets/d1/regression/terms": terms,
    "GET /datasets/d1/regression/partial": partial,
    "GET /datasets/d1/regression/robust": robust,
    "GET /datasets/d1/regression/quantiles": quantiles,
    "GET /datasets/d1/regression/more": more,
    "GET /datasets/d1/regression/families": families,
  };
}

const last = (calls: { key: string; params: URLSearchParams }[], key: string) => [...calls].reverse().find((c) => c.key === key);

describe("Regression: terms (070)", () => {
  it("adds the curve R-REG-10 proposes, and asks for Terms only once it is opened", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=price&columns=area,city");
    await screen.findByText(/bends the most/);
    expect(calls.some((c) => c.key === "GET /datasets/d1/regression/terms")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Try a curve" }));
    await waitFor(() => expect(last(calls, "GET /datasets/d1/regression")?.params.get("terms")).toBe("spline:area"));
    expect(screen.getByRole("link", { name: "See the curve →" })).toBeInTheDocument();
  });

  it("builds an interaction, centers the numbers and shows the shape of a column on demand", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=price&columns=area,city&terms=spline:area");
    await screen.findByText(/bends the most/);
    fireEvent.click(screen.getByRole("button", { name: /^Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /^Terms/ }));
    expect(await screen.findByText(/is a curve \(F = 41/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Effect curve of area" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove spline:area" })).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Kind" }), { target: { value: "interaction" } });
    const first = screen.getByRole("combobox", { name: "Column" });
    expect(within(first).getByRole("option", { name: "sold" })).toBeInTheDocument(); // a date can interact (its trend)
    fireEvent.change(first, { target: { value: "area" } });
    fireEvent.change(screen.getByRole("combobox", { name: "With" }), { target: { value: "city" } });
    fireEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() =>
      expect(last(calls, "GET /datasets/d1/regression")?.params.get("terms")).toBe("spline:area,interaction:area|city"),
    );

    fireEvent.change(screen.getByRole("combobox", { name: "Center numbers" }), { target: { value: "on" } });
    await waitFor(() => expect(last(calls, "GET /datasets/d1/regression")?.params.get("center")).toBe("true"));
    expect(screen.getByRole("button", { name: "Reset to auto: Center numbers" })).toBeInTheDocument();

    expect(calls.some((c) => c.key === "GET /datasets/d1/regression/partial")).toBe(false);
    fireEvent.change(screen.getByRole("combobox", { name: "Show the shape of…" }), { target: { value: "area" } });
    expect(await screen.findByText(/its coefficient/)).toBeInTheDocument();
    expect(last(calls, "GET /datasets/d1/regression/partial")?.params.get("column")).toBe("area");
    expect(screen.getByRole("img", { name: "Component + residual" })).toBeInTheDocument();
  });
});

describe("Regression: robust and quantile (072)", () => {
  it("opens Robust & quantile from the influence check, changes the quantile, and asks for the τ grid only in More", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=price&columns=area,city");
    await screen.findByText(/weigh heavily/);
    window.HTMLElement.prototype.scrollIntoView = vi.fn();
    expect(calls.some((c) => c.key === "GET /datasets/d1/regression/robust")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Try…" }));
    expect(await screen.findByText(/Huber weighs 120 rows/)).toBeInTheDocument();
    expect(last(calls, "GET /datasets/d1/regression/robust")?.params.get("quantile")).toBe("0.5");
    expect(screen.getByText("moves")).toBeInTheDocument();
    expect(screen.getByText(/heavy tail/)).toBeInTheDocument();

    fireEvent.change(screen.getByRole("combobox", { name: "Quantile" }), { target: { value: "0.9" } });
    await waitFor(() => expect(last(calls, "GET /datasets/d1/regression/robust")?.params.get("quantile")).toBe("0.9"));
    expect(calls.some((c) => c.key === "GET /datasets/d1/regression/quantiles")).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /^More/ }));
    expect(await screen.findByText(/1 of 1 effects change/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Effect of area along the distribution" })).toBeInTheDocument();
    expect(screen.getByText(/ν = 3.2: heavy tails/)).toBeInTheDocument();
  });
});
