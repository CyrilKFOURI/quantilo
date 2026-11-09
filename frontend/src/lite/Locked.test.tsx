import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./flag", () => ({ LITE: true, FULL_APP_URL: "https://example.test" }));

describe("A page the browser version cannot run", () => {
  it("names the page it stands for", async () => {
    const { modules } = await import("../modules/registry");
    const ml = modules.find((module) => module.id === "ml");
    expect(ml?.locked).toBe(true);

    const Page = ml!.component;
    render(<Page />);

    expect(screen.getByRole("heading", { name: "Machine learning needs the full app" })).toBeInTheDocument();
  });
});
