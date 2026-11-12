import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { MlRun } from "../../api/client";
import { IntervalNote, intervalExplain, intervalValue } from "./IntervalNote";
import { paretoFigure } from "./tuneFigures";

type Interval = NonNullable<MlRun["interval"]>;

const interval = (overrides: Partial<Interval> = {}): Interval => ({
  method: "split",
  level: 0.9,
  half_width: 12.3,
  rows: 240,
  calibration_rows: 120,
  check_rows: 120,
  coverage: 0.9,
  mean_width: 24.6,
  order: "random",
  tolerance: 0.055,
  rules: ["R-ML-CONF-01: …", "R-ML-CONF-02: …"],
  ...overrides,
});

describe("057 prediction interval", () => {
  it("says the width and the coverage checked on held-out rows", () => {
    render(<IntervalNote interval={interval()} />);
    const note = screen.getByTestId("interval-note");
    expect(note.textContent).toContain("90 % prediction interval: ± 12.3 around each prediction (split conformal).");
    expect(note.textContent).toContain("On 120 held-out rows that did not set it, 90 % fell inside (target 90 %).");
    expect(screen.queryByText(/may not be alike/)).toBeNull();
  });

  it("warns when the coverage falls short of the level beyond chance (R-ML-CONF-03)", () => {
    render(<IntervalNote interval={interval({ coverage: 0.7, order: "time" })} />);
    expect(screen.getByText(/only 70 % fell inside/)).toBeTruthy();
    expect(screen.getByText(/may not be alike/)).toBeTruthy();
  });

  it("an adaptive interval says its mean width", () => {
    render(<IntervalNote interval={interval({ method: "cqr", mean_width: 30 })} />);
    expect(screen.getByTestId("interval-note").textContent).toContain("adaptive, 30 wide on average");
  });

  it("nothing without an interval", () => {
    const { container } = render(<IntervalNote interval={null} />);
    expect(container.textContent).toBe("");
  });

  it("the prediction tile shows ± or Adaptive, with the checked coverage", () => {
    const card = { level: 0.95, half_width: 5, errors: 400, method: "split" as const, coverage: 0.951 };
    expect(intervalValue(card)).toBe("± 5");
    expect(intervalExplain(card)).toContain("covers 95 % of the 400 validation errors");
    expect(intervalExplain(card)).toContain("95.1 % inside");
    expect(intervalValue({ ...card, method: "cqr" })).toBe("Adaptive");
  });
});

describe("066 Pareto front", () => {
  it("joins the front and keeps the other trials apart", () => {
    const trials = [
      { number: 0, value: 0.8, state: "complete", seconds: 1, pareto: true },
      { number: 1, value: 0.9, state: "complete", seconds: 3, pareto: true },
      { number: 2, value: 0.7, state: "complete", seconds: 4, pareto: false },
      { number: 3, value: null, state: "pruned", seconds: 0, pareto: false },
    ];
    const figure = paretoFigure(trials, "R²")({ palette: ["#111", "#222"], ink3: "#999", numeric: "#333", categorical: "#444" } as never);
    const [others, front] = figure.data as { x: number[]; y: number[] }[];
    expect(front!.x).toEqual([1, 3]);
    expect(others!.y).toEqual([0.7]);
  });
});
