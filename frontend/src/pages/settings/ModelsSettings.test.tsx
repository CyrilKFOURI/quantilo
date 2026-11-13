import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AiModel, AiModels } from "../../modules/timeseries/forecastAi";
import { aiBadge, aiReasons } from "../../modules/timeseries/forecastAi";
import { health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";

const detection = { parse_rate: 0.95, empty_rate: 0.95, discrete_max: 20, identifier_unique: 0.95 };

function model(key: string, overrides: Partial<AiModel> = {}): AiModel {
  return {
    key,
    title: key,
    family: "chronos_bolt",
    pretrained: true,
    state: "needs_download",
    reason: null,
    size: 190_889_945,
    size_text: "191 MB",
    licence: "Apache-2.0",
    parameters: "48 M",
    note: "",
    source: "https://huggingface.co/amazon/chronos-bolt-small/tree/772f3d25",
    folder: "C:\\Quantilo\\models\\forecast\\chronos_bolt_small",
    drivers: false,
    missing: [],
    install: null,
    download: "idle",
    progress: 0,
    error: null,
    ...overrides,
  };
}

const models: AiModels = {
  models: [
    model("chronos_bolt_tiny", { state: "ready", size_text: "35 MB", progress: 1 }),
    model("chronos_bolt_small"),
    model("nhits", { pretrained: false, family: "neural", size: 0, size_text: "", source: "", folder: "" }),
  ],
  folder: "C:\\Quantilo\\models\\forecast",
  offline: false,
  offline_forced: false,
  install: 'pip install "quantilo[forecast-ai]"',
};

function setup() {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [],
    "GET /settings": { detection, compute: { use_gpu: null } },
    "GET /settings/gpu": { state: "not_found", name: null, memory_mb: null, driver: null, torch: false, message: "No graphics card" },
    "GET /storage": { workspace: "C:\\Quantilo", total_bytes: 0, datasets: [] },
    "GET /modules/timeseries/forecast-ai": models,
    "POST /modules/timeseries/forecast-ai/chronos_bolt_small/download": model("chronos_bolt_small", { download: "running" }),
    "DELETE /modules/timeseries/forecast-ai/chronos_bolt_tiny": model("chronos_bolt_tiny"),
    "PUT /modules/timeseries/forecast-ai/offline": (body: unknown) => ({ ...models, offline: (body as { offline: boolean }).offline }),
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("Settings › Models (028 B1)", () => {
  it("downloads a model only after a confirmation that shows its size and source", async () => {
    const calls = setup();
    renderAt("/settings");

    fireEvent.click(await screen.findByRole("button", { name: "Download · 191 MB" }));
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText("Your data is not sent anywhere: forecasting runs on this computer.")).toBeInTheDocument();
    expect(within(dialog).getByText(/huggingface\.co\/amazon\/chronos-bolt-small/)).toBeInTheDocument();
    expect(calls.some((c) => c.key.startsWith("POST"))).toBe(false);

    fireEvent.click(within(dialog).getByRole("button", { name: "Download · 191 MB" }));
    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/timeseries/forecast-ai/chronos_bolt_small/download")).toBe(true));
  });

  it("deletes a downloaded model and sets Never download", async () => {
    const calls = setup();
    renderAt("/settings");

    expect(await screen.findByText("On this computer · 35 MB")).toBeInTheDocument();
    expect(screen.getByText("Trained on your series: nothing to download")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    const never = screen.getByRole("radiogroup", { name: "Never download" });
    fireEvent.click(within(never).getByRole("radio", { name: "On" }));

    await waitFor(() => {
      expect(calls.some((c) => c.key === "DELETE /modules/timeseries/forecast-ai/chronos_bolt_tiny")).toBe(true);
      expect(calls.find((c) => c.key === "PUT /modules/timeseries/forecast-ai/offline")?.body).toEqual({ offline: true });
    });
  });
});

describe("AI model rules in the forecast list", () => {
  it("greys a model with its reason and offers the download of the others", () => {
    const list = [
      model("chronos_2", { state: "needs_extra", install: 'pip install "quantilo[forecast-ai]"' }),
      model("timesfm_2p5", { state: "offline", folder: "C:\\m\\timesfm_2p5" }),
      model("chronos_bolt_small"),
    ];

    const reasons = aiReasons(list, 120);

    expect(reasons.chronos_2).toContain('pip install "quantilo[forecast-ai]"');
    expect(reasons.timesfm_2p5).toContain("C:\\m\\timesfm_2p5");
    expect(reasons.chronos_bolt_small).toBeUndefined();
    expect(reasons.nhits).toContain("at least 500 periods (this series has 120)");
    expect(aiReasons(list, 600).nhits).toBeUndefined();
    expect(aiBadge(list[2])).toBe("Download · 191 MB");
    expect(aiBadge(model("x", { state: "ready" }))).toBe("Pretrained · zero-shot");
    expect(aiBadge(model("x", { download: "running", progress: 0.4 }))).toBe("Downloading 40 %");
  });
});
