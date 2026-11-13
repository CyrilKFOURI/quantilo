import { fireEvent, render, screen } from "@testing-library/react";
import { forwardRef } from "react";
import { describe, expect, it, vi } from "vitest";

import type { MlRun } from "../../api/client";
import { ExplainMore } from "./ExplainMore";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

const asked = { shap: [] as boolean[], pairs: [] as boolean[], anchor: [] as boolean[], prototypes: [] as boolean[] };

const shap = {
  method: "Permutation sampling (any model, estimated)",
  scale: "probability",
  base: 0.3,
  positive: "yes",
  rows: [4, 7],
  features: [{ column: "tenure", mean_abs: 0.2, values: [0.25, -0.1], colors: [0.1, 0.9], reading: null }],
  columns: ["tenure", "contract"],
  matrix: [
    [0.25, 0.05],
    [-0.1, 0.02],
  ],
  labels: [
    ["3", "month"],
    ["60", "year"],
  ],
  note: null,
};

vi.mock("../../api/hooks", () => ({
  useMlPermutationShap: (_id: string, _run: string, _pairs: number, _background: number, enabled: boolean) => {
    asked.shap.push(enabled);
    return enabled
      ? { data: { shap, half_width: [[0.03, 0.01], [0.02, 0.01]], orders: 20, background: 50, additivity: 0, widest: 0.03, sentence: "Estimated on 2 rows: tenure moves the probability most." }, isError: false }
      : { data: undefined, isError: false };
  },
  useMlInteractions: (_id: string, _run: string, _columns: number, enabled: boolean) => {
    asked.pairs.push(enabled);
    return enabled
      ? {
          data: {
            columns: ["tenure", "contract", "age"],
            pairs: [
              { first: "tenure", second: "contract", h: 0.4, share: 0.16, joint: 0.3, weak: false },
              { first: "age", second: "contract", h: 0, share: 0, joint: 0.001, weak: true },
            ],
            rows: 250,
            output: "Probability of “yes”",
            sentence: "The strongest interaction is tenure × contract (H = 0.40).",
            skipped: [],
            note: null,
          },
          isError: false,
        }
      : { data: undefined, isError: false };
  },
  useMlAnchor: (_id: string, _run: string, _row: number | null, _threshold: number, _conditions: number, enabled: boolean) => {
    asked.anchor.push(enabled);
    return enabled
      ? {
          data: {
            row: 4,
            predicted: "yes",
            outcome: "the class “yes”",
            conditions: [
              { column: "tenure", text: "tenure ≤ 9", value: "3" },
              { column: "contract", text: "contract = month", value: "month" },
            ],
            rule: "IF tenure ≤ 9 AND contract = month THEN the class “yes”",
            precision: 0.97,
            precision_low: 0.96,
            coverage: 0.18,
            samples: 2000,
            reached: true,
            threshold: 0.95,
            evaluated: 12,
            sentence: "IF tenure ≤ 9 AND contract = month THEN the class “yes”: 97% of similar rows.",
            left_out: ["customer_id (an identifier)"],
          },
          isError: false,
        }
      : { data: undefined, isError: false };
  },
  useMlPrototypes: (_id: string, _run: string, _count: number, _criticisms: number, enabled: boolean) => {
    asked.prototypes.push(enabled);
    return enabled
      ? {
          data: {
            columns: ["tenure"],
            groups: [{ label: "yes", rows: 300, prototypes: [{ row: 10, values: ["2"], predicted: "yes", share: 0.6, witness: null }], criticisms: [{ row: 11, values: ["70"], predicted: "no", share: null, witness: 0.1 }], mmd: [0.1] }],
            width: 2,
            sentence: "1 real rows summarise the class “yes”.",
            left_out: [],
          },
          isError: false,
        }
      : { data: undefined, isError: false };
  },
}));

const run = {
  id: "r1",
  model: "knn",
  task: "binary",
  settings: {},
  predictions: [
    { row: 4, actual: "yes", predicted: "yes" },
    { row: 7, actual: "no", predicted: "no" },
  ],
} as unknown as MlRun;

describe("Explain › Advanced › More (126)", () => {
  it("asks for nothing until a button is pressed, then shows each result", () => {
    render(<ExplainMore datasetId="d" run={run} found={{ shap: null } as never} />);
    expect([...asked.shap, ...asked.pairs, ...asked.anchor, ...asked.prototypes].every((on) => !on)).toBe(true);

    fireEvent.click(screen.getByRole("button", { name: "Estimate SHAP values" }));
    expect(asked.shap.at(-1)).toBe(true);
    expect(screen.getByText(/tenure moves the probability most/)).toBeInTheDocument();
    expect(screen.getByText("± 0.03", { exact: false })).toBeInTheDocument();
    expect(asked.pairs.at(-1)).toBe(false); // the other blocks still wait

    fireEvent.click(screen.getByRole("button", { name: "Measure interactions" }));
    expect(screen.getByText("tenure × contract")).toBeInTheDocument();
    expect(screen.getByText(/1 pair has a joint effect too weak to judge/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Find a rule for row 5/ }));
    expect(screen.getByText("tenure ≤ 9")).toBeInTheDocument();
    expect(screen.getByText(/covers 18% of the rows/)).toBeInTheDocument();
    expect(screen.getByText(/customer_id \(an identifier\)/)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Find prototypes" }));
    expect(screen.getByText("Prototype · 60%")).toBeInTheDocument();
    expect(screen.getByText("Criticism")).toBeInTheDocument();
  });

  it("explains every setting in a cell with its reason", () => {
    render(<ExplainMore datasetId="d" run={run} found={{ shap: null } as never} />);
    const reasons = [...document.querySelectorAll("[title]")].map((e) => e.getAttribute("title") ?? "");
    for (const code of ["R-XAI-12", "R-XAI-13", "R-XAI-14", "R-XAI-15"]) expect(reasons.some((r) => r.includes(code))).toBe(true);
  });
});
