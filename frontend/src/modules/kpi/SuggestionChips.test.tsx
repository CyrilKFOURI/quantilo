import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { Column, KpiDefinition, KpiSuggestions } from "../../api/client";
import { column } from "../../test/columns";
import { defaultAgg } from "./definition";
import { HierarchyChip, splitReason, SuggestionChips } from "./SuggestionChips";

type Suggested = KpiSuggestions["kpis"][number];

function kpi(name: string, score: number, rule = "R-SUG-02"): Suggested {
  const definition = { measures: [{ id: "m1", agg: "sum", column: name }] } as unknown as KpiDefinition;
  return {
    rule,
    name,
    score,
    reasons: [`${rule} · ${name} is an amount that adds up over rows: summed`, `R-SUG-16 · Score ${score}/100: weight ${score} for this kind of KPI`],
    definition,
  } as Suggested;
}

describe("Suggested KPIs, explained (048)", () => {
  it("keeps six chips and lists the rest under More (n)", () => {
    const kpis = Array.from({ length: 9 }, (_, i) => kpi(`m${i}`, 90 - i));
    const onPick = vi.fn();
    render(<SuggestionChips kpis={kpis} isActive={() => false} onPick={onPick} />);

    const row = screen.getByRole("group", { name: "Try:" });
    expect(within(row).getAllByRole("button", { pressed: false })).toHaveLength(6);
    const more = screen.getByRole("button", { name: "More (3)" });
    fireEvent.click(more);
    const panel = screen.getByRole("dialog", { name: "More suggested KPIs" });
    expect(within(panel).getAllByRole("button")).toHaveLength(3);
    fireEvent.click(within(panel).getByText("m7"));
    expect(onPick).toHaveBeenCalledWith(kpis[7]!.definition);
  });

  it("tells why a KPI is suggested, with its score and rules", () => {
    render(<SuggestionChips kpis={[kpi("Sales", 90)]} isActive={() => true} onPick={() => {}} />);

    const chip = screen.getByRole("button", { name: "Sales" });
    expect(chip).toHaveAttribute("aria-pressed", "true");
    const tip = document.getElementById(chip.getAttribute("aria-describedby")!)!;
    expect(tip).toHaveAttribute("role", "tooltip");
    expect(tip).toHaveTextContent("Why this KPI · 90/100");
    expect(tip).toHaveTextContent("Sales is an amount that adds up over rows: summed");
    expect(tip).toHaveTextContent("R-SUG-02");
    expect(screen.queryByRole("button", { name: /More/ })).toBeNull();
  });

  it("splits a reason into its rule and its sentence", () => {
    expect(splitReason("R-SUG-12 · Sales ÷ distinct Order ID: the average value")).toEqual({ rule: "R-SUG-12", text: "Sales ÷ distinct Order ID: the average value" });
    expect(splitReason("plain words")).toEqual({ rule: null, text: "plain words" });
  });

  it("offers a hierarchy as one chip with its reason", () => {
    const onPick = vi.fn();
    const reason = "R-SUG-18 · Each State belongs to one Region (100% of rows): split by Region, then State";
    render(<HierarchyChip hierarchy={{ parent: "Region", child: "State", share: 1, reason }} onPick={onPick} />);

    const chip = screen.getByRole("button", { name: "+ Region › State" });
    fireEvent.click(chip);
    expect(onPick).toHaveBeenCalled();
    expect(document.getElementById(chip.getAttribute("aria-describedby")!)).toHaveTextContent("Each State belongs to one Region");
  });
});

describe("Default aggregate of a column (R-MES-01, 048)", () => {
  const of = (overrides: Record<string, unknown>) => defaultAgg(column(overrides) as unknown as Column);

  it("sums flows and averages prices", () => {
    expect(of({ name: "Sales", role: "amount" })).toBe("sum");
    expect(of({ name: "shipping_cost", role: "cost" })).toBe("sum");
    expect(of({ name: "sales_target", role: "target" })).toBe("sum");
    expect(of({ name: "Price", role: "amount" })).toBe("mean");
    expect(of({ name: "UnitPrice", role: "unit_price" })).toBe("mean");
    expect(of({ name: "Salaire moyen", role: "amount" })).toBe("mean");
    expect(of({ name: "rating", role: "rating" })).toBe("mean");
  });
});
