import { createMemoryHistory, createRootRoute, createRouter, RouterProvider } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { fireEvent } from "@testing-library/react";

import { columnReadingNotes, ReadingNotes } from "./ReadingNotes";

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

  it("says a hint instead of the link, and shows the rest on demand (092)", () => {
    const notes = ["a (R-TYP-01)", "b (R-TYP-03)", "c (R-ING-01)"];
    render(<ReadingNotes notes={notes} label="Read" hint="Check that your data was read correctly." limit={2} />);
    expect(screen.getByText("Check that your data was read correctly.")).toBeTruthy();
    expect(screen.queryByText("c (R-ING-01)")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Show all 3" }));
    expect(screen.getByText("c (R-ING-01)")).toBeTruthy();
  });

  it("keeps only the reasons given by a reading rule (092)", () => {
    const notes = columnReadingNotes([
      { name: "dep", reasons: ["1% of values, such as 2A, cannot be read as numbers: kept as text (R-TYP-01)", "105 distinct values"] },
      { name: "price", reasons: ["name suggests amount"] },
    ]);
    expect(notes).toEqual(["dep: 1% of values, such as 2A, cannot be read as numbers: kept as text (R-TYP-01)"]);
  });
});
