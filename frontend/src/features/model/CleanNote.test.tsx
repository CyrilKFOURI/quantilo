import { createMemoryHistory, createRootRoute, createRouter, RouterProvider } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { Relation, RelationCheck } from "./api";
import { CleanNote } from "./CleanNote";

const relation = (status: Relation["status"]) => ({ status, child_id: "orders", parent_id: "customers" }) as Relation;
const check = (orphans: number, duplicates: number) => ({ orphan_rows: orphans, duplicate_keys: duplicates }) as RelationCheck;

function show(status: Relation["status"], orphans: number, duplicates = 0) {
  const root = createRootRoute({
    component: () => <CleanNote relation={relation(status)} check={check(orphans, duplicates)} childName="orders" parentName="customers" />,
  });
  const router = createRouter({ routeTree: root, history: createMemoryHistory({ initialEntries: ["/"] }) });
  return render(<RouterProvider router={router} />);
}

describe("CleanNote (N32)", () => {
  it("says Clean lists the orphan rows of an accepted relation, with a link", async () => {
    show("accepted", 17);
    expect(await screen.findByText(/These 17 rows stay in orders/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open Clean of orders" }).getAttribute("href")).toBe("/d/orders/clean");
  });

  it("points to the dimension for a repeated key", async () => {
    show("accepted", 0, 2);
    expect(await screen.findByText(/lists the repeated keys \(T-02\)/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open Clean of customers" }).getAttribute("href")).toBe("/d/customers/clean");
  });

  it("says nothing for a proposal or a clean relation", async () => {
    const { container } = show("suggested", 17);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(container.textContent).toBe("");
    show("accepted", 0, 0);
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByRole("link")).toBeNull();
  });
});
