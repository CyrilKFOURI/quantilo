import { fireEvent, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { legalHref } from "../../features/legal/documents";
import { dataStatement } from "../../features/legal/LegalLinks";
import { health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

afterEach(() => vi.unstubAllGlobals());

const routes = { "GET /health": health, "GET /datasets": [], "GET /settings": {}, "GET /storage": {} };

describe("Legal pages (081)", () => {
  it("shows the privacy page with tabs to the other texts", async () => {
    mockApi(routes);
    const router = renderAt("/legal/privacy");

    expect(await screen.findByRole("heading", { level: 1, name: "Privacy" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "The AI assistant (optional, app on your computer only)" })).toBeInTheDocument();
    const tabs = screen.getByRole("tablist", { name: "Legal" });
    expect(within(tabs).getByRole("tab", { name: "Privacy" })).toHaveAttribute("aria-selected", "true");

    fireEvent.click(within(tabs).getByRole("tab", { name: "Legal notice" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Legal notice" })).toBeInTheDocument();
    expect(router.state.location.pathname).toBe("/legal/notice");
    expect(screen.getAllByText(/Vercel Inc\./).length).toBeGreaterThan(0);
    // anonymity option (LCEN art. 1-1 II): no postal address or phone on the page
    expect(screen.queryByText(/Antony/)).toBeNull();
    expect(screen.getAllByText(/LCEN, art\. 1-1 II/).length).toBe(2);
  });

  it("an unknown legal page is a not-found page", async () => {
    mockApi(routes);
    renderAt("/legal/cookies");
    expect(await screen.findByRole("heading", { level: 1, name: "Page not found" })).toBeInTheDocument();
  });

  it("the Guide links to the three texts, the sidebar does not", async () => {
    mockApi(routes);
    renderAt("/help");
    const footer = await screen.findByRole("navigation", { name: "Legal" });
    expect(within(footer).getByRole("link", { name: "Privacy" })).toHaveAttribute("href", "/legal/privacy");
    expect(within(footer).getByRole("link", { name: "Terms of use" })).toHaveAttribute("href", "/legal/terms");
    expect(within(footer).getByRole("link", { name: "Legal notice" })).toHaveAttribute("href", "/legal/notice");
    expect(screen.getAllByRole("link", { name: "Privacy" })).toHaveLength(1);
  });

  it("the Datasets page says where the data goes and what the AI assistant sends", async () => {
    mockApi(routes);
    renderAt("/");
    expect(await screen.findByText(/Only the optional AI assistant sends your questions/)).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Legal" })).toBeInTheDocument();
  });

  it("links inside the texts: legal pages stay in the app, only https addresses are kept", () => {
    const href = legalHref((path) => `#${path}`);
    expect(href("terms")).toBe("#/legal/terms");
    expect(href("https://vercel.com/help")).toBe("https://vercel.com/help");
    expect(href("javascript:alert(1)")).toBeNull();
    expect(dataStatement(true)).toMatch(/not part of the browser version/);
  });
});
