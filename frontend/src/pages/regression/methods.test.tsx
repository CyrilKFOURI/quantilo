import { fireEvent, screen, within } from "@testing-library/react";
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
  column({ name: "tenure", position: 0 }),
  column({ name: "Charges", position: 1, role: null }),
  column({ name: "Churn", position: 2, type: "boolean", detected_type: "boolean", role: null }),
  column({ name: "Other", position: 3, role: null }),
];

const coefficient = (term: string, estimate: number) => ({
  term,
  column: term,
  level: null,
  estimate,
  std_error: 0.1,
  ci_low: estimate - 0.2,
  ci_high: estimate + 0.2,
  statistic: 20,
  p_value: 1e-6,
  q_value: term === "(Intercept)" ? null : 1e-6,
  evidence: term === "(Intercept)" ? "none" : "strong",
  standardized: null,
  std_low: null,
  std_high: null,
  odds_ratio: null,
  marginal: null,
  marginal_low: null,
  marginal_high: null,
  sentence: `Each extra unit of \`${term}\` adds ${estimate}.`,
});

const regression = {
  target: "tenure",
  positive: null,
  model: "ols",
  model_label: "Least squares (OLS)",
  model_reason: "A number to explain.",
  columns: ["Charges"],
  focus: null,
  terms: [coefficient("(Intercept)", 30), coefficient("Charges", -0.2)],
  se: { kind: "hc3", label: "HC3", rule: "R-REG-SE-05", reason: "HC3, always.", lag: null, groups: [], clusters: null, warning: null },
  weights: null,
  quality: { rows: 600, loaded: 600, total: 600, sampled: false, missing: [], r2: 0.1, adj_r2: 0.1, oos_r2: 0.09, oos_how: "5 folds", aic: 1, bic: 1, f: 10, f_p: 0, df_model: 1, df_resid: 598, pseudo_r2: null },
  multiple: { method: "none", label: "None", tests: 1, significant: 1, kept: 1, alpha: 0.05, auto: true, reason: "", sentence: "One test." },
  sentence: "Each extra unit of `Charges` adds -0.2.",
  checks: [
    {
      key: "time_to_event",
      status: "warn",
      title: "Time to event",
      sentence: "`tenure` reads as a duration and `Churn` as the event that ends it: least squares on a duration still running are biased. Open More for the Cox model.",
      rule: "R-REG-06",
      action: "more",
      column: null,
    },
  ],
  references: [],
  left_out: [],
  structure: { kind: "none", date: null, entity: null, period: null, periods: 0, entities: 0, message: null },
  hints: [],
  plots: null,
};

const more = { white: null, white_sentence: "White's test: nothing.", removed: 0, shifts: [], shift_sentence: "No influential rows." };
const families = { alternatives: [], sentence: "No other family." };

const method = (kind: string, title: string, proposed: boolean, available: boolean, reason: string) => ({ kind, title, proposed, available, reason, rule: "R" });
const methods = {
  methods: [
    method("survival", "Survival (Cox)", true, true, "R-REG-06: the target reads as a duration with its event."),
    method("outputs", "Several outcomes", false, true, "R-REG-XS-04: the same columns explain other numeric columns too."),
    method("cross_section", "Fama–MacBeth and IC", false, false, "Needs a panel: entities seen over periods (R-REG-XS-01)."),
  ],
  numbers: ["Other"],
  events: ["Churn"],
  slopes: ["Charges"],
  sigma: null,
  sigma_x: null,
  event: { column: "Churn", reason: "R-SRV-01: `tenure` reads as a duration and `Churn` as the event." },
  group: null,
};

const survival = {
  available: true,
  reason: "",
  duration: "tenure",
  event: "Churn = True",
  event_reason: "R-SRV-03: Churn = True read as the event.",
  rows: 600,
  events: 500,
  censored: 100,
  sampled: false,
  terms: [
    { term: "Charges", coefficient: 0.02, std_error: 0.002, hazard_ratio: 1.02, ci_low: 1.016, ci_high: 1.024, p_value: 1e-9, ph_p: 0.4, proportional: true, sentence: "Each extra unit of `Charges` multiplies the hazard of `Churn = True` by 1.02." },
  ],
  lr: 80,
  lr_df: 1,
  lr_p: 1e-12,
  sentence: "Cox model of `tenure` until `Churn = True`: 500 events, 100 still followed (censored).",
  ph_sentence: "No effect changes clearly with time (Schoenfeld residuals, p ≥ 0.01): proportional hazards hold (R-REG-SURV-02).",
  notes: [],
};

const outputs = {
  available: true,
  reason: "",
  outputs: ["tenure", "Other"],
  terms: ["(Intercept)", "Charges"],
  coefficients: [
    [
      { estimate: 30, std_error: 1, p_value: 1e-9 },
      { estimate: 0.1, std_error: 1, p_value: 0.9 },
    ],
    [
      { estimate: -0.2, std_error: 0.01, p_value: 1e-9 },
      { estimate: 0.01, std_error: 0.02, p_value: 0.6 },
    ],
  ],
  r2: [0.1, 0.0],
  rows: 600,
  tests: [{ term: "Charges", df: 1, wilks: 0.9, wilks_f: 30, wilks_df1: 2, wilks_df2: 597, wilks_p: 1e-9, pillai: 0.1, pillai_f: 30, pillai_df1: 2, pillai_df2: 597, pillai_p: 1e-9 }],
  grs: null,
  grs_reason: "GRS needs one row per date (R-REG-XS-05).",
  sentence: "`Charges` moves the 2 outcomes jointly (Pillai's trace 0.100, p < 0.001).",
  notes: [],
};

function routes() {
  return {
    "GET /health": health,
    "GET /datasets": [dataset()],
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": columns,
    "GET /column-roles": roles,
    "GET /datasets/d1/regression": regression,
    "GET /datasets/d1/regression/more": more,
    "GET /datasets/d1/regression/families": families,
    "GET /datasets/d1/regression/quantiles": { available: false, reason: "No.", taus: [], curves: [], rows: [], t_errors: null, sentence: "", notes: [] },
    "GET /datasets/d1/regression/selection": { available: false, reason: "No." },
    "GET /datasets/d1/regression/methods": methods,
    "GET /datasets/d1/regression/survival": survival,
    "GET /datasets/d1/regression/outputs": outputs,
  };
}

const asked = (calls: { key: string }[], path: string) => calls.some((call) => call.key === `GET /datasets/d1/regression/${path}`);

describe("Regression: methods of More (074–076)", () => {
  it("lists the methods only once More is open, and computes one only once it is opened", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=tenure&columns=Charges");
    await screen.findByText(/least squares on a duration still running are biased/);
    expect(asked(calls, "methods")).toBe(false);
    window.HTMLElement.prototype.scrollIntoView = vi.fn();

    fireEvent.click(screen.getByRole("button", { name: "Try…" }));
    const list = await screen.findByRole("region", { name: "Methods for these data" });
    expect(asked(calls, "methods")).toBe(true);
    expect(await within(list).findByText("Proposed by a rule")).toBeInTheDocument();
    expect(within(list).getByText(/Needs a panel/)).toBeInTheDocument();
    expect(asked(calls, "survival")).toBe(false);

    fireEvent.click(within(list).getByRole("button", { name: /^Survival \(Cox\)/ }));
    expect(await screen.findByText(/500 events, 100 still followed/)).toBeInTheDocument();
    expect(asked(calls, "survival")).toBe(true);
    expect(screen.getByRole("columnheader", { name: "Hazard ratio" })).toBeInTheDocument();
    expect(screen.getByText(/proportional hazards hold/)).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Event" })).toHaveValue("__rule");
    expect(asked(calls, "outputs")).toBe(false);
  });

  it("asks for several outcomes only once one is added", async () => {
    const calls = mockApi(routes());
    renderAt("/d/d1/regression?target=tenure&columns=Charges");
    await screen.findByText(/least squares on a duration still running are biased/);
    fireEvent.click(screen.getByRole("button", { name: /^Advanced/ }));
    fireEvent.click(screen.getByRole("button", { name: /^More/ }));
    const list = await screen.findByRole("region", { name: "Methods for these data" });
    fireEvent.click(await within(list).findByRole("button", { name: /^Several outcomes/ }));
    expect(screen.getByText(/Add at least one other numeric column/)).toBeInTheDocument();
    expect(asked(calls, "outputs")).toBe(false);
    fireEvent.change(screen.getByRole("combobox", { name: "Add an outcome…" }), { target: { value: "Other" } });
    expect(await screen.findByText(/moves the 2 outcomes jointly/)).toBeInTheDocument();
    expect(asked(calls, "outputs")).toBe(true);
    expect(screen.getByText(/GRS needs one row per date/)).toBeInTheDocument();
  });
});
