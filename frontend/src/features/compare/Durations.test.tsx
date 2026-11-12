import { fireEvent, screen, waitFor } from "@testing-library/react";
import { forwardRef } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { column, roles } from "../../test/columns";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { survivalFigure } from "./Durations";
import type { Survival } from "../../api/client";

vi.mock("../../charts/Plot", () => ({
  Plot: forwardRef<HTMLDivElement, { label: string }>(function Plot({ label }, ref) {
    return <div ref={ref} role="img" aria-label={label} />;
  }),
}));

afterEach(() => vi.unstubAllGlobals());

const choices = { durations: ["tenure"], events: ["Churn"], dates: [], groups: ["Contract"] };
const curve = (label: string, median: number | null) => ({
  label,
  n: 100,
  events: 40,
  times: [0, 5, 10],
  survival: [1, 0.7, 0.4],
  lower: [1, 0.6, 0.3],
  upper: [1, 0.8, 0.5],
  at_risk: [100, 70, 40],
  median,
  median_lower: median === null ? null : 8,
  median_upper: null,
});
const detected: Survival = {
  setup: {
    kind: "duration",
    duration: "tenure",
    event: "Churn",
    start: null,
    end: null,
    event_value: "True",
    event_reason: "R-SRV-03: Churn = True read as the event.",
    unit: null,
    detected: true,
    reason: "R-SRV-01: 'tenure' reads as a duration and 'Churn' as the event. Check that your data was read correctly.",
  },
  choices,
  group: null,
  rows: 100,
  events: 40,
  censored: 60,
  dropped: 0,
  curves: [curve("All rows", 10)],
  log_rank: null,
  others: 0,
  sentence: "Half reach the event by 10.",
  message: null,
};
const none = { setup: null, choices: { entities: [], times: [], statuses: [] }, entities: 0, rows: 0, chain: null, step_days: null, others: 0, sentence: null, message: null };

function routes(survival: object, transitions: object = none) {
  return mockApi({
    "GET /health": health,
    "GET /datasets/d1": dataset(),
    "GET /datasets/d1/columns": [column({ name: "tenure", position: 0 })],
    "GET /column-roles": roles,
    "GET /datasets/d1/survival": survival,
    "GET /datasets/d1/transitions": transitions,
  });
}

describe("Durations in Compare (062)", () => {
  it("shows the survival curve and the rule's reason when a rule detects it", async () => {
    routes(detected);
    renderAt("/d/d1/compare");
    expect(await screen.findByText("Half reach the event by 10.")).toBeInTheDocument();
    expect(screen.getAllByText(/R-SRV-01/).length).toBeGreaterThan(0);
    expect(screen.getByRole("img", { name: "Survival curves with 95% intervals" })).toBeInTheDocument();
  });

  it("shows nothing when no rule fires (onion rule)", async () => {
    const calls = routes({ ...detected, setup: null, curves: [], sentence: null });
    renderAt("/d/d1/compare");
    await waitFor(() => expect(calls.some((c) => c.key === "GET /datasets/d1/survival")).toBe(true));
    expect(screen.queryByText("Durations")).not.toBeInTheDocument();
  });

  it("asks for the curves by group", async () => {
    const calls = routes(detected);
    renderAt("/d/d1/compare");
    const group = await screen.findByLabelText("Compare curves by");
    fireEvent.change(group, { target: { value: "Contract" } });
    await waitFor(() =>
      expect(calls.some((c) => c.key === "GET /datasets/d1/survival" && c.params.get("group") === "Contract")).toBe(true),
    );
  });

  it("folds the transitions under Advanced and computes them only when opened", async () => {
    const setup = { entity: "customer_id", time: "month", status: "status", detected: true, reason: "R-MKV-01: found." };
    const chain = {
      states: ["churned", "paid"],
      counts: [[3, 0], [2, 8]],
      probabilities: [[1, 0], [0.2, 0.8]],
      absorbing: ["churned"],
      never_left: [],
      stay_steps: [null, 5],
      stationary: null,
      stationary_note: null,
      steps_to_absorption: { paid: 5 },
      absorption: { paid: { churned: 1 } },
      transitions: 13,
    };
    const calls = routes(
      { ...detected, setup: null },
      { ...none, setup, choices: { entities: ["customer_id"], times: ["month"], statuses: ["status"] }, chain, sentence: "churned is never left." },
    );
    renderAt("/d/d1/compare");
    const summary = await screen.findByText("Status transitions (Markov chain)");
    const full = () => calls.filter((c) => c.key === "GET /datasets/d1/transitions" && !c.params.get("detect_only"));
    expect(full()).toHaveLength(0);
    const details = summary.closest("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(await screen.findByText("churned is never left.")).toBeInTheDocument();
    expect(full().length).toBeGreaterThan(0);
  });
});

describe("survivalFigure", () => {
  it("draws a step line and its band per curve", () => {
    const tokens = { palette: ["#111", "#222"], numeric: "#000", ink3: "#999" } as never;
    const built = survivalFigure({ ...detected, curves: [curve("a", 10), curve("b", null)] })(tokens);
    expect(built.data).toHaveLength(4);
    expect((built.data[1] as { line: { shape: string } }).line.shape).toBe("hv");
  });
});
