import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Dataset } from "../../api/client";
import { OPEN_PALETTE_EVENT } from "../../features/command/Shortcuts";
import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { recentDatasets } from "./NotFoundPage";

afterEach(() => vi.unstubAllGlobals());

const gone = () => Response.json({ error: { code: "not_found", message: "Dataset not found" } }, { status: 404 });

describe("recentDatasets", () => {
  it("keeps ready datasets, last opened first, three at most", () => {
    const rows = [
      dataset({ id: "a", last_opened_at: "2026-11-01T10:00:00Z" }),
      dataset({ id: "b", last_opened_at: "2026-11-05T10:00:00Z" }),
      dataset({ id: "c", status: "error", last_opened_at: "2026-11-09T10:00:00Z" }),
      dataset({ id: "d", last_opened_at: null, imported_at: "2026-11-03T10:00:00Z" }),
      dataset({ id: "e", last_opened_at: "2026-10-01T10:00:00Z" }),
    ] as Dataset[];
    expect(recentDatasets(rows).map((row) => row.id)).toEqual(["b", "d", "a"]);
  });
});

describe("Not found (081)", () => {
  it("an unknown address keeps the shell and offers home, search and recent datasets", async () => {
    mockApi({ "GET /health": health, "GET /datasets": [dataset({ name: "superstore" })] });
    renderAt("/no-such-page");

    expect(await screen.findByRole("heading", { level: 1, name: "Page not found" })).toBeInTheDocument();
    expect(screen.getByText("/no-such-page")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Go to datasets" })).toHaveAttribute("href", "/");
    expect(await screen.findByRole("link", { name: /superstore/ })).toHaveAttribute("href", "/d/d1");
    // the app's navigation is still there
    expect(screen.getByRole("link", { name: "Guide" })).toBeInTheDocument();

    const opened = vi.fn();
    window.addEventListener(OPEN_PALETTE_EVENT, opened);
    fireEvent.click(within(screen.getByRole("main")).getByRole("button", { name: /^Search/ }));
    expect(opened).toHaveBeenCalledTimes(1);
    window.removeEventListener(OPEN_PALETTE_EVENT, opened);
  });

  it("an unknown or deleted dataset says so on every dataset page", async () => {
    mockApi({ "GET /health": health, "GET /datasets": [], "GET /datasets/gone": gone });
    renderAt("/d/gone/correlations");

    expect(await screen.findByRole("heading", { level: 1, name: "Dataset not found" })).toBeInTheDocument();
    expect(screen.getByText(/may have been deleted/)).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Opened recently" })).not.toBeInTheDocument();
  });
});
