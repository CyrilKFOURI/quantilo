import { fireEvent, render, screen } from "@testing-library/react";
import { forwardRef } from "react";
import { describe, expect, it, vi } from "vitest";

import type { MlRun } from "../../api/client";
import { WhatWouldChange } from "./Counterfactual";
import { ExplainAdvanced } from "./ExplainAdvanced";
import { aleFigure } from "./figures";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const ale = {
  column: "tenure",
  kind: "number",
  grid: [0, 10, 20],
  curves: { yes: [0.2, 0, -0.2] },
  counts: [40, 60],
  output: "Probability of “yes”",
  rows: 100,
  order: null,
  sentence: "Over its real values, tenure moves the yes by 0.4 at most.",
  rule: "R-XAI-01: …",
  monotonic: null,
};
const calls = { ale: [] as (string | null)[], counterfactual: 0 };

vi.mock("../../api/hooks", () => ({
  useMlAle: (_id: string, _run: string, column: string | null) => {
    calls.ale.push(column);
    return { data: ale, isError: false };
  },
  useMlSurrogate: () => ({
    data: {
      depth: 3,
      measure: "agreement",
      fidelity: 0.6,
      rows: 70,
      checked: 30,
      leaves: 2,
      sentence: "A tree of 3 levels gives the model's class on 60% of the rows.",
      warning: "R-XAI-04: fidelity 60 %, under 80 %.",
      nodes: [
        { id: 0, depth: 0, value: "no (60%)", samples: 70, left: 1, right: 2, left_rule: "tenure ≤ 12", right_rule: "tenure > 12" },
        { id: 1, depth: 1, value: "yes (70%)", samples: 30, left: null, right: null, left_rule: null, right_rule: null },
        { id: 2, depth: 1, value: "no (80%)", samples: 40, left: null, right: null, left_rule: null, right_rule: null },
      ],
    },
    isError: false,
  }),
  useMlMonotonic: () => ({ data: [], isError: false }),
  useMlCounterfactual: () => {
    calls.counterfactual += 1;
    return {
      data: {
        row: 4,
        goal: "Reach the class “no”",
        output: "probability of “no”",
        before: 0.3,
        after: 0.6,
        reached: true,
        changes: [{ column: "tenure", before: "3", after: "14", output: 0.6 }],
        sentence: "tenure 3 → 14: the model gives the class “no”.",
        columns: [
          { column: "tenure", editable: true, locked: null },
          { column: "age", editable: false, locked: "a personal attribute (age)" },
        ],
        table: [
          { column: "tenure", row: "3", changed: "14", nearest: "15" },
          { column: "age", row: "40", changed: "40", nearest: "41" },
        ],
        nearest_row: 9,
        distance: 0.05,
        typical: 0.1,
        plausible: true,
        plausibility: "Plausible: the closest real row with that outcome (row 10)…",
        target_class: "no",
        target_value: null,
        classes: ["no", "yes"],
      },
      isError: false,
      isPlaceholderData: false,
    };
  },
}));

const run = {
  id: "r1",
  model: "random_forest",
  task: "binary",
  settings: {},
  preparation: { decisions: [{ column: "tenure", type: "numeric_discrete", use: true }, { column: "age", type: "numeric_discrete", use: true }] },
} as unknown as MlRun;

const explanation = {
  dependence: [{ column: "tenure", extrapolates: "R-XAI-01: …", linked: "months_left" }],
} as never;

describe("Explain › Advanced (125)", () => {
  it("opens on the column whose partial dependence extrapolates, and flags a weak surrogate", () => {
    render(<ExplainAdvanced datasetId="d" run={run} found={explanation} column={null} onColumn={() => {}} />);
    expect(calls.ale.at(-1)).toBe("tenure");
    expect(screen.getByText(/Shown first by R-XAI-01/)).toBeInTheDocument();
    expect(screen.getByText(/R-XAI-04/)).toBeInTheDocument();
    expect(screen.getByText("tenure ≤ 12", { exact: false })).toBeInTheDocument();
  });

  it("computes a counterfactual only once opened, with locked columns said", () => {
    const { container } = render(<WhatWouldChange datasetId="d" run={run} row={4} />);
    expect(calls.counterfactual).toBe(0);
    const details = container.querySelector("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(calls.counterfactual).toBeGreaterThan(0);
    expect(screen.getByText(/tenure 3 → 14/)).toBeInTheDocument();
    expect(screen.getByText(/R-XAI-03/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /age/ })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: /tenure/ })).toBeChecked();
  });

  it("draws one bar per interval under the curve", () => {
    const figure = aleFigure(ale as never)({ palette: ["#111"], line: "#ccc", ink3: "#999" } as never);
    expect(figure.data).toHaveLength(2);
    expect((figure.data[1] as { x: number[] }).x).toEqual([5, 15]);
  });
});
