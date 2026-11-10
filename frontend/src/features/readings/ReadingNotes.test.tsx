import { createMemoryHistory, createRootRoute, createRouter, RouterProvider } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ReadingNotes } from "./ReadingNotes";

function show(notes: string[]) {
  const root = createRootRoute({ component: () => <ReadingNotes datasetId="d1" notes={notes} label="How the columns were read" /> });
  const router = createRouter({ routeTree: root, history: createMemoryHistory({ initialEntries: ["/"] }) });
  render(<RouterProvider router={router} />);
}

describe("ReadingNotes (093)", () => {
  it("lists each rule with its reason and links to the Columns page", async () => {
    show(["R-SUG-25 · patient_nbr is never summed or averaged: its name ends in 'nbr'."]);
    expect(await screen.findByText(/R-SUG-25 · patient_nbr/)).toBeTruthy();
    const link = screen.getByRole("link", { name: "Check how the columns were read" });
    expect(link.getAttribute("href")).toBe("/d/d1/columns");
  });

  it("shows nothing without notes", () => {
    const { container } = render(<ReadingNotes datasetId="d1" notes={[]} label="x" />);
    expect(container.textContent).toBe("");
  });
});
