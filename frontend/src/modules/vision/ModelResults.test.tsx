import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import {
  anomalyPredictions,
  anomalyRun,
  anomalyTask,
  classificationPredictions,
  classificationRun,
  multilabelPredictions,
  multilabelRun,
  regressionPredictions,
  regressionRun,
  runSummary,
} from "./modelTestData";
import { imageIndex, imageReport, qualityRules } from "./testData";

const sizes = ["offsetHeight", "offsetWidth", "clientHeight", "clientWidth"] as const;
const saved = sizes.map((key) => Object.getOwnPropertyDescriptor(HTMLElement.prototype, key));
beforeAll(() => {
  for (const key of sizes) {
    Object.defineProperty(HTMLElement.prototype, key, { configurable: true, get: () => (key.endsWith("Height") ? 900 : 1200) });
  }
});
afterAll(() => {
  sizes.forEach((key, index) => {
    const descriptor = saved[index];
    if (descriptor) Object.defineProperty(HTMLElement.prototype, key, descriptor);
  });
});
afterEach(() => vi.unstubAllGlobals());

function setup(extra: Record<string, unknown>) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset({ format: "images", name: "parts" })],
    "GET /datasets/d1": dataset({ format: "images", name: "parts" }),
    "GET /datasets/d1/columns": [],
    "GET /datasets/d1/images/index": imageIndex(),
    "GET /datasets/d1/images/report": imageReport,
    "GET /images/rules": qualityRules,
    "GET /datasets/d1/images/decisions": { draft: {}, applied: {}, pending: 0, version: 1 },
    ...extra,
  });
}

describe("Model tab (113)", () => {
  it("says the task and its rule, trains only when asked, and greys a detector with its reason", async () => {
    const calls = setup({
      "GET /modules/vision/datasets/d1/task": anomalyTask,
      "GET /modules/vision/datasets/d1/runs": [],
      "POST /modules/vision/datasets/d1/runs": { id: "bbbbbbbbbbbb", dataset_id: "d1", status: "running", stage: "features", progress: 0.1, message: "Descriptors 1 / 6 images", run_id: null, error: null, long_run: null },
      "GET /modules/vision/jobs/bbbbbbbbbbbb": { id: "bbbbbbbbbbbb", dataset_id: "d1", status: "running", stage: "features", progress: 0.3, message: "Descriptors 2 / 6 images", run_id: null, error: null, long_run: null },
      "POST /modules/vision/weights/resnet18/download": { key: "resnet18", title: "ResNet-18", size: "47 MB", licence: "BSD-3-Clause", note: "", downloaded: false, download: "running", progress: 0, error: null },
    });
    renderAt("/d/d1/vision?tab=model");
    expect(await screen.findByRole("heading", { name: "Anomaly detection" })).toBeInTheDocument();
    expect(screen.getByText("R-VML-02")).toBeInTheDocument();
    // R-113-01: opening the tab reads the rule, nothing is trained
    expect(calls.some((c) => c.key.startsWith("POST"))).toBe(false);

    // Advanced: PatchCore greyed with its reason; a download asks first (R-113-04, R-113-05)
    expect(screen.getByRole("radio", { name: "PatchCore" })).toBeDisabled();
    expect(screen.getByText(/Download the weights first: Wide ResNet-50-2/)).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole("button", { name: "Download ResNet-18 (47 MB)" })[0] as HTMLElement);
    const dialog = await screen.findByRole("dialog");
    expect(within(dialog).getByText(/47 MB · licence BSD-3-Clause/)).toBeInTheDocument();
    expect(calls.some((c) => c.key.includes("/download"))).toBe(false);
    fireEvent.click(within(dialog).getByRole("button", { name: "Download 47 MB" }));
    await waitFor(() => expect(calls.some((c) => c.key === "POST /modules/vision/weights/resnet18/download")).toBe(true));

    // choose the autoencoder, train, and follow the job
    fireEvent.click(screen.getByRole("radio", { name: "Convolutional autoencoder" }));
    fireEvent.click(screen.getByRole("button", { name: "Train models" }));
    await waitFor(() => expect(calls.find((c) => c.key === "POST /modules/vision/datasets/d1/runs")?.body).toMatchObject({ detector: "autoencoder", deep: null }));
    expect(await screen.findByText("Descriptors 2 / 6 images")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeInTheDocument();
  });
});

describe("Results tab (113)", () => {
  it("moves the anomaly threshold in the page and the counts follow (R-VTH-03)", async () => {
    const calls = setup({
      "GET /modules/vision/datasets/d1/runs": [runSummary("anomaly")],
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa": anomalyRun,
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa/predictions": anomalyPredictions,
    });
    renderAt("/d/d1/vision?tab=results");
    expect(await screen.findByText(/At 0.600, 1 of 1 defects are found and 0 of 2 normal images/)).toBeInTheDocument();
    expect(screen.getAllByText("Image AUROC").length).toBeGreaterThan(0);
    const before = calls.length;
    fireEvent.keyDown(screen.getByRole("slider", { name: "Threshold" }), { key: "Home" });
    expect(await screen.findByText(/At 0.200, 1 of 1 defects are found and 2 of 2 normal images/)).toBeInTheDocument();
    expect(calls.slice(before).filter((c) => c.key.includes("/modules/vision"))).toHaveLength(0);
    expect(screen.getByRole("button", { name: /Back to the rule's threshold/ })).toBeInTheDocument();
    // every model of the run, the default first
    expect(screen.getAllByRole("button", { pressed: true })[0]).toHaveTextContent("Patch PCA + nearest normals");
  });

  it("shows every classification metric and the images of a confusion cell", async () => {
    setup({
      "GET /modules/vision/datasets/d1/runs": [runSummary("classification")],
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa": classificationRun,
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa/predictions": classificationPredictions,
    });
    renderAt("/d/d1/vision?tab=results");
    expect(await screen.findByRole("heading", { name: "Every metric" })).toBeInTheDocument();
    expect(screen.getAllByText("Cohen's kappa").length).toBeGreaterThan(0);
    expect(screen.getByText(/Temperature scaling \(T = 1.60\)/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "1 image of dog predicted cat" }));
    expect(await screen.findByRole("list", { name: "Images of this cell" })).toBeInTheDocument();
    expect(within(screen.getByRole("list", { name: "Images of this cell" })).getAllByRole("button")).toHaveLength(1);
  });

  it("shows a regression's metrics against the training mean (c3, R-VRG-02)", async () => {
    setup({
      "GET /modules/vision/datasets/d1/runs": [{ ...runSummary("classification"), task: "regression", metric: "r2", score: 0.81 }],
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa": regressionRun,
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa/predictions": regressionPredictions,
    });
    renderAt("/d/d1/vision?tab=results");
    expect(await screen.findByRole("heading", { name: "Predicted against true weight_g" })).toBeInTheDocument();
    expect(screen.getByText(/R² 0.810 on 3 images the model did not learn from; 62 % less error/)).toBeInTheDocument();
    expect(screen.getAllByText("R²").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "Predicted too high" })).toBeInTheDocument();
  });

  it("shows a multi-label run's metrics, label by label (c3, R-VMLB-02)", async () => {
    setup({
      "GET /modules/vision/datasets/d1/runs": [{ ...runSummary("classification"), task: "multilabel", metric: "macro_f1", score: 0.75 }],
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa": multilabelRun,
      "GET /modules/vision/datasets/d1/runs/aaaaaaaaaaaa/predictions": multilabelPredictions,
    });
    renderAt("/d/d1/vision?tab=results");
    const table = await screen.findByRole("table", { name: "Every label" });
    expect(within(table).getByRole("rowheader", { name: "stripe" })).toBeInTheDocument();
    expect(screen.getAllByText("Macro F1").length).toBeGreaterThan(0);
    expect(screen.getByRole("heading", { name: "Most labels wrong" })).toBeInTheDocument();
  });

  it("says there is no run yet and leads to Model", async () => {
    setup({ "GET /modules/vision/datasets/d1/runs": [] });
    renderAt("/d/d1/vision?tab=results");
    expect(await screen.findByText("No run yet")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Open Model/ })).toBeInTheDocument();
  });
});
