import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { MlRun } from "../../api/client";
import { hasScorecard, ScorecardTab } from "./Scorecard";

const gap = {
  metric: "ROC AUC",
  model: 0.84,
  readable: 0.82,
  gap: 0.02,
  low: -0.004,
  high: 0.041,
  within_noise: true,
  model_ece: 0.03,
  readable_ece: 0.02,
  model_brier: 0.14,
  readable_brier: 0.15,
  rows: 1409,
  sentence: "A 3-condition scorecard is 2.0 pts of ROC AUC below the run's model: within noise (R-XAI-07).",
};
const card = {
  lines: [
    { condition: "Contract = Month-to-month", column: "Contract", points: 3, share: 0.55 },
    { condition: "tenure ≤ 12", column: "tenure", points: 2, share: 0.31 },
  ],
  steps: [
    { total: 0, risk: 0.05, rows: 600, observed: 0.06 },
    { total: 2, risk: 0.2, rows: 100, observed: 0.18 },
    { total: 3, risk: 0.3, rows: 400, observed: 0.31 },
    { total: 5, risk: 0.6, rows: 309, observed: 0.58 },
  ],
  intercept: -3,
  scale: 0.6,
  auc: 0.82,
  learnt_auc: 0.83,
  ece: 0.02,
  brier: 0.15,
  log_loss: 0.45,
  columns: 2,
  sentence: "2 conditions on 2 columns, points from +2 to +3: ROC AUC 0.820 on 1,409 rows it did not learn from.",
};
const calls = { sparse: 0, optimal: 0, classes: [] as boolean[] };

vi.mock("../../api/hooks", () => ({
  useMlScorecard: () => ({
    data: {
      positive: "Yes",
      conditions: 5,
      bound: 5,
      best: card,
      others: [{ ...card, lines: [{ condition: "OnlineSecurity = No", column: "OnlineSecurity", points: 2, share: 0.5 }] }],
      gap,
      learnt: 5634,
      checked: 1409,
      candidates: 60,
      source: "learnt on the training rows, judged on the test rows (R-XAI-08)",
      notes: ["R-XAI-06: 60 conditions tried.", "Rows learnt on the training rows (R-XAI-08)."],
      seconds: 2.1,
    },
    isError: false,
    isPlaceholderData: false,
  }),
  useMlSparseTree: () => {
    calls.sparse += 1;
    return {
      data: {
        depth: 4,
        leaves: 2,
        max_leaves: 12,
        alpha: 0.001,
        measure: "ROC AUC",
        score: 0.8,
        nodes: [
          { id: 0, depth: 0, value: "No (73%)", samples: 100, left: 1, right: 2, left_rule: "tenure ≤ 16.5", right_rule: "tenure > 16.5" },
          { id: 1, depth: 1, value: "Yes (55%)", samples: 40, left: null, right: null, left_rule: null, right_rule: null },
          { id: 2, depth: 1, value: "No (90%)", samples: 60, left: null, right: null, left_rule: null, right_rule: null },
        ],
        pruning: [],
        gap: null,
        learnt: 5634,
        checked: 1409,
        sentence: "A tree of 1 level and 2 leaves (greedy CART, not optimal) (R-XAI-09).",
      },
      isError: false,
      isPlaceholderData: false,
    };
  },
  useMlClassCards: (_id: string, _run: string, _c: number, _p: number, _m: boolean, enabled: boolean) => {
    calls.classes.push(enabled);
    return enabled
      ? {
          data: {
            classes: [{ label: "low", card, auc: 0.9, model_auc: 0.93 }],
            accuracy: 0.81,
            model_accuracy: 0.86,
            conditions: 5,
            bound: 5,
            learnt: 700,
            checked: 300,
            source: "test rows",
            sentence: "3 cards of 5 conditions at most, one per class (R-XAI-17).",
            notes: [],
            seconds: 4.2,
            monotone: false,
          },
          isError: false,
          isPlaceholderData: false,
        }
      : { data: undefined, isError: false };
  },
  useInterpretAvailable: () => ({
    data: [
      { key: "gosdt", label: "Optimal tree (GOSDT)", available: false, reason: "R-XAI-10: gosdt has no wheel for Python 3.13 (only 3.9 to 3.12).", install: 'pip install "quantilo[interpret]"' },
      { key: "treefarms", label: "Rashomon set (TreeFARMS)", available: false, reason: "R-XAI-10: treefarms is not installed (optional extra `interpret`).", install: 'pip install "quantilo[interpret]"' },
    ],
    isError: false,
  }),
  useMlOptimal: (_id: string, _run: string, _kind: string, _depth: number, enabled: boolean) => {
    if (enabled) calls.optimal += 1;
    return { data: undefined, isError: false };
  },
}));

const run = { id: "r1", model: "random_forest", task: "binary", classification: { positive: "Yes" }, panel: null } as unknown as MlRun;

function open(summary: string) {
  const details = screen.getByText(summary).closest("details") as HTMLDetailsElement;
  details.open = true;
  fireEvent(details, new Event("toggle"));
}

describe("Scorecard (121, 122)", () => {
  it("is a tab for two classes, and a Readable model tab for more classes and numbers (121 bis, 122 bis)", () => {
    expect(hasScorecard(run)).toBe(true);
    expect(hasScorecard({ ...run, classification: { positive: null } } as unknown as MlRun)).toBe(false);
    expect(hasScorecard({ ...run, task: "regression", classification: null } as unknown as MlRun)).toBe(true);
    expect(hasScorecard({ ...run, task: "multiclass", classification: { positive: null } } as unknown as MlRun)).toBe(true);
  });

  it("builds one card per class only on its button (R-XAI-17)", () => {
    calls.classes = [];
    render(<ScorecardTab run={{ ...run, task: "multiclass" } as unknown as MlRun} datasetId="d" />);
    expect(calls.classes.every((on) => !on)).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: "Build one card per class" }));
    expect(calls.classes.at(-1)).toBe(true);
    expect(screen.getByText(/one per class \(R-XAI-17\)/)).toBeInTheDocument();
    expect(screen.getByText(/low against the others/)).toBeInTheDocument();
  });

  it("shows a number's sparse tree in front and greys the optimal tree with its reason (R-XAI-19)", () => {
    calls.sparse = 0;
    render(<ScorecardTab run={{ ...run, task: "regression", classification: null } as unknown as MlRun} datasetId="d" />);
    expect(calls.sparse).toBeGreaterThan(0);
    expect(screen.getByRole("list", { name: "A sparse decision tree" })).toBeInTheDocument();
    expect(screen.getAllByText(/learn classes only/).length).toBeGreaterThan(0);
  });

  it("shows the card, the risk of each total and the gap to the model, conclusion first", () => {
    render(<ScorecardTab run={run} datasetId="d" />);
    expect(screen.getByText(/within noise \(R-XAI-07\)/)).toBeInTheDocument();
    const table = screen.getAllByRole("table", { name: "Scorecard" })[0];
    expect(table).toHaveTextContent("Contract = Month-to-month");
    expect(table).toHaveTextContent("+3");
    expect(screen.getByRole("table", { name: "Risk for each total" })).toHaveTextContent("60.0 %");
    expect(screen.getByText("Other scorecards almost as good (1)")).toBeInTheDocument();
  });

  it("asks for the tree only once Advanced is open, and greys the optional solvers with their reason", () => {
    calls.sparse = 0;
    render(<ScorecardTab run={run} datasetId="d" />);
    expect(calls.sparse).toBe(0);
    open("Advanced");
    expect(calls.sparse).toBeGreaterThan(0);
    expect(screen.getByRole("list", { name: "A sparse decision tree" })).toHaveTextContent("tenure ≤ 16.5");
    expect(screen.queryByText(/no wheel for Python/)).not.toBeInTheDocument();
    open("More");
    expect(screen.getByText(/no wheel for Python 3.13/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Find the optimal tree" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Find the Rashomon set" })).toBeDisabled();
    expect(calls.optimal).toBe(0);
  });
});
