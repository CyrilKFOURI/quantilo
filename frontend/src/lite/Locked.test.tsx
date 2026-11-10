import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

vi.mock("./flag", () => ({ LITE: true, FULL_APP_URL: "https://example.test" }));

describe("A page the browser version cannot run", () => {
  it("names the page it stands for", async () => {
    const { modules } = await import("../modules/registry");
    const synthetic = modules.find((module) => module.id === "synthetic");
    expect(synthetic?.locked).toBe(true);

    const Page = synthetic!.component;
    render(<Page />);

    expect(screen.getByRole("heading", { name: "Synthetic data needs the full app" })).toBeInTheDocument();
  });

  it("leaves machine learning open: its scikit-learn models run in the browser (102)", async () => {
    const { modules } = await import("../modules/registry");
    expect(modules.find((module) => module.id === "ml")?.locked).toBeUndefined();
  });
});
