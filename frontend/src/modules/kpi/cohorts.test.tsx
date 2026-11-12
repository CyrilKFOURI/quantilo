import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { CohortResult, FunnelResult, KpiArchetype, KpiDefinition, KpiSuggestions } from "../../api/client";
import { ArchetypeBanner, packOf } from "./ArchetypeBanner";
import { cohortMatrix, conversions, offsetLabels } from "./cohortFigures";

type Suggested = KpiSuggestions["kpis"][number];

const cohort = {
  grain: "month",
  cohorts: [
    { start: "2024-01-01", label: "Jan 2024", size: 3, retained: [1, 1 / 3, 2 / 3] },
    { start: "2024-02-01", label: "Feb 2024", size: 1, retained: [1, 1, null] },
    { start: "2024-03-01", label: "Mar 2024", size: 1, retained: [1, null, null] },
  ],
  periods: [],
} as unknown as CohortResult;

describe("Retention by cohort (060)", () => {
  it("draws a triangle: cohorts down, offsets across, empty where not observed", () => {
    const { x, y, z } = cohortMatrix(cohort);
    expect(offsetLabels(cohort)).toEqual(["+0", "+1", "+2"]);
    expect(x).toHaveLength(3);
    expect(y[0]).toBe("Jan 2024 · 3");
    expect(z[2]).toEqual([1, null, null]);
  });
});

describe("Funnel (060)", () => {
  it("lists the conversion from each stage to the next", () => {
    const funnel = {
      stages: [
        { label: "Lead", reached: 4, of_first: 1, conversion: null },
        { label: "Qualified", reached: 3, of_first: 0.75, conversion: 0.75 },
        { label: "Won", reached: 1, of_first: 0.25, conversion: 1 / 3 },
      ],
    } as unknown as FunnelResult;
    expect(conversions(funnel)).toEqual([
      { from: "Lead", to: "Qualified", share: 0.75 },
      { from: "Qualified", to: "Won", share: 1 / 3 },
    ]);
  });
});

function kpi(name: string): Suggested {
  return {
    rule: "R-SUG-02",
    name,
    score: 80,
    reasons: [`R-SUG-02 · ${name} is an amount: summed`],
    definition: { measures: [{ id: "m1", agg: "sum", column: name }] } as unknown as KpiDefinition,
  } as Suggested;
}

describe("The kind of dataset (061)", () => {
  const archetype = {
    id: "sales",
    label: "sales data",
    message: "Looks like sales data · 3 KPIs",
    kpis: ["Sales", "Margin", "Orders"],
    reasons: ["R-ARC-01 · Looks like sales data: Sales (amount)", "R-ARC-02 · nothing is saved until you confirm"],
  } as KpiArchetype;
  const kpis = [kpi("Orders"), kpi("Rows"), kpi("Sales"), kpi("Margin")];

  it("keeps the pack in its order", () => {
    expect(packOf(archetype, kpis).map((k) => k.name)).toEqual(["Sales", "Margin", "Orders"]);
  });

  it("saves nothing until the user confirms, and only what is kept", async () => {
    const onSave = vi.fn().mockResolvedValue(undefined);
    render(<ArchetypeBanner archetype={archetype} kpis={kpis} saving={false} onSave={onSave} />);
    expect(screen.getByText("Looks like sales data · 3 KPIs")).toBeTruthy();
    expect(onSave).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Review" }));
    fireEvent.click(screen.getByLabelText("Margin"));
    fireEvent.click(screen.getByRole("button", { name: "Save 2 KPIs" }));
    expect(onSave).toHaveBeenCalledWith([
      { name: "Sales", definition: kpis[2]!.definition },
      { name: "Orders", definition: kpis[0]!.definition },
    ]);
    expect(await screen.findByText(/2 KPIs saved/)).toBeTruthy();
  });

  it("goes away when it is not this kind", () => {
    render(<ArchetypeBanner archetype={archetype} kpis={kpis} saving={false} onSave={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Not this" }));
    expect(screen.queryByText("Looks like sales data · 3 KPIs")).toBeNull();
  });
});
