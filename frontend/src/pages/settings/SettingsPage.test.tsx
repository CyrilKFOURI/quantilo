import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { readDefaults } from "../../lib/defaults";
import { health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

const detection = { parse_rate: 0.95, empty_rate: 0.95, discrete_max: 20, identifier_unique: 0.95 };

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [],
    "GET /settings": { detection, compute: { use_gpu: null } },
    "GET /settings/gpu": { state: "ready", name: "NVIDIA GeForce RTX 4050 Laptop", memory_mb: 6141, driver: "560.94", torch: false, message: "NVIDIA GeForce RTX 4050 Laptop · 6 GB · ready" },
    "PUT /settings": (body: unknown) => body,
    "GET /storage": { workspace: "C:\\Users\\me\\Quantilo", total_bytes: 3_145_728, datasets: [{ id: "d1", name: "sales", bytes: 2_097_152 }] },
    "POST /storage/clear-caches": null,
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Settings page", () => {
  it("shows the GPU found and saves the choice to use it", async () => {
    const calls = setup();
    renderAt("/settings");

    expect(await screen.findByText("NVIDIA GeForce RTX 4050 Laptop · 6 GB · ready")).toBeInTheDocument();
    const choice = screen.getByRole("radiogroup", { name: "Use the GPU when available" });
    expect(within(choice).getByRole("radio", { name: "On" })).toHaveAttribute("aria-checked", "true");
    fireEvent.click(within(choice).getByRole("radio", { name: "Off" }));
    await waitFor(() => {
      const put = calls.find((c) => c.key === "PUT /settings");
      expect(put?.body).toEqual({ detection, compute: { use_gpu: false } });
    });
  });

  it("saves detection thresholds as a whole", async () => {
    const calls = setup();
    renderAt("/settings");

    const discrete = await screen.findByRole("spinbutton", { name: "Discrete numbers" });
    const save = screen.getByRole("button", { name: "Save thresholds" });
    expect(save).toBeDisabled();
    fireEvent.change(discrete, { target: { value: "8" } });
    fireEvent.change(screen.getByRole("spinbutton", { name: "Identifier" }), { target: { value: "90" } });
    fireEvent.click(save);

    await waitFor(() => {
      const put = calls.find((c) => c.key === "PUT /settings");
      expect(put?.body).toEqual({ detection: { ...detection, discrete_max: 8, identifier_unique: 0.9 }, compute: { use_gpu: null } });
    });
  });

  it("remembers analysis defaults and appearance in this browser", async () => {
    setup();
    renderAt("/settings");

    fireEvent.change(await screen.findByRole("combobox", { name: "Top categories per chart" }), { target: { value: "20" } });
    fireEvent.click(screen.getByRole("radio", { name: "0.01" }));
    fireEvent.click(screen.getByRole("radio", { name: "Reduce" }));

    expect(readDefaults()).toMatchObject({ topN: 20, alpha: "0.01", pageSize: 12 });
    expect(document.documentElement.dataset.motion).toBe("reduce");
    fireEvent.click(screen.getByRole("button", { name: "Reset Top categories per chart to default" }));
    expect(readDefaults().topN).toBe(10);
    delete document.documentElement.dataset.motion;
  });

  it("shows disk usage and clears the caches", async () => {
    const calls = setup();
    renderAt("/settings");

    expect(await screen.findByText("3.0 MB")).toBeInTheDocument();
    expect(screen.getByText("sales")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Clear caches" }));

    await waitFor(() => expect(calls.some((c) => c.key === "POST /storage/clear-caches")).toBe(true));
    expect(await screen.findByRole("button", { name: "Cleared" })).toBeInTheDocument();
    expect(screen.getByText("0.1.0")).toBeInTheDocument();
  });
});
