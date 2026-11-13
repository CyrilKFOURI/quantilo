import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { imageIndex, imageReport, qualityRules } from "./testData";

afterEach(() => vi.unstubAllGlobals());

const RUN = "abcdef012345";
const run = {
  id: RUN,
  created_at: "2026-11-12T10:00:00+00:00",
  task: "classification",
  chosen_by: "rule",
  default: "deep_linear",
  default_title: "Linear probe on DINOv2 ViT-S/14 (frozen)",
  metric: "balanced_accuracy",
  score: 0.95,
  models: ["logistic", "deep_linear"],
  deep: true,
  scored: 2,
  seconds: 12,
  sentence: "",
};
const ref = (image_id: string, label: string, similarity: number) => ({ row: 0, image_id, label, similarity, distance: null, share: null, witness: null });
const models = [
  { key: "logistic", title: "Logistic regression on classic descriptors", default: false },
  { key: "deep_linear", title: "Linear probe on DINOv2 ViT-S/14 (frozen)", default: true },
];
const why = {
  run_id: RUN,
  image_id: "eeeeeeeeeeeeeeee",
  row: 5,
  task: "classification",
  model: models[1],
  models,
  fold: 0,
  prediction: {
    label: "cat",
    probability: 0.86,
    probabilities: [
      { label: "cat", probability: 0.86 },
      { label: "dog", probability: 0.14 },
    ],
    score: null,
    threshold: null,
    flagged: null,
    truth: "dog",
    correct: false,
    tested: true,
    sentence: "Linear probe on DINOv2 ViT-S/14 (frozen) says cat with probability 0.86. It is wrong: the image is labelled dog.",
  },
  evidence: {
    space: "the DINOv2 ViT-S/14 vectors the probe reads, standardised (cosine)",
    nearest: [ref("dddddddddddddddd", "dog", 0.61), ref("aaaaaaaaaaaaaaaa", "cat", 0.5)],
    sentence: "1 of the 2 nearest training images are labelled cat; the others: dog 1 (similarity 0.61 to 0.50, R-VEX-01).",
    prototype: { ...ref("bbbbbbbbbbbbbbbb", "cat", 0.41), share: 0.44 },
    prototype_sentence: "The closest of the 3 prototypes of cat (k-medoids), which stands for 44% of the class's training images (similarity 0.41, R-VEX-02).",
  },
  counterfactual: {
    image: ref("dddddddddddddddd", "dog", 0.61),
    own_label: "cat",
    probability: 0.97,
    sentence: "The nearest training image labelled dog, the second most probable class (similarity 0.61); the model gives it 0.97 dog (R-VEX-04).",
  },
  methods: [
    { key: "occlusion", title: "Occlusion", level: "advanced", available: true, reason: null, cost: "about 5 s", own: false },
    { key: "grad_cam", title: "Grad-CAM", level: "advanced", available: true, reason: null, cost: "about 1 s", own: false },
    { key: "rollout", title: "Attention rollout", level: "advanced", available: true, reason: null, cost: "about 1 s", own: false },
    { key: "saliency", title: "Vanilla saliency", level: "more", available: true, reason: null, cost: "about 1 s", own: false },
    { key: "smoothgrad", title: "SmoothGrad", level: "more", available: true, reason: null, cost: "about 5 s", own: false },
    { key: "integrated_gradients", title: "Integrated Gradients", level: "more", available: true, reason: null, cost: "about 5 s", own: false },
    { key: "sanity", title: "Sanity test (random weights)", level: "more", available: true, reason: null, cost: "about 15 s", own: false },
  ],
  seconds: 0.4,
  notes: [],
};
const gradCam = {
  method: "grad_cam",
  title: "Grad-CAM",
  unit: "weighted activation",
  sentence: "Grad-CAM for cat on the patch tokens of DINOv2 ViT-S/14 (16 x 16 cells, R-VEX-06).",
  warning: "This heatmap is a post-hoc explanation: it shows where the model's output is sensitive, not the reasoning it used.",
  own: false,
  blocky: false,
  box: [0, 0, 1, 1],
  target: "cat",
  rows: 2,
  columns: 2,
  low: 0,
  high: 0.07,
  png: "data:image/png;base64,AA",
  seconds: 0.05,
  cached: false,
  notes: [],
  extra: {},
  patch: null,
};
const overview = {
  run_id: RUN,
  task: "classification",
  model: models[1],
  models,
  space: "the DINOv2 ViT-S/14 vectors the probe reads, standardised (cosine)",
  classes: [
    { label: "cat", size: 4, prototypes: [{ ...ref("aaaaaaaaaaaaaaaa", "cat", 1), share: 0.75 }], criticisms: [{ ...ref("bbbbbbbbbbbbbbbb", "cat", 0), witness: 0.07 }], notes: [] },
    { label: "dog", size: 2, prototypes: [{ ...ref("dddddddddddddddd", "dog", 1), share: 1 }], criticisms: [], notes: [] },
  ],
  sentence: "1 prototypes per class (k-medoids) and the images they represent worst (MMD-critic criticisms), in the space the model reads (R-VEX-02, R-VEX-03).",
  families: [
    { key: "texture", title: "Texture (LBP)", drop: 0.237, spread: 0.049, columns: 236 },
    { key: "colour", title: "Colour histogram (HSV)", drop: 0.132, spread: 0.012, columns: 72 },
  ],
  families_sentence: "The classic logistic model leans most on texture (LBP): shuffling it across the 2 test images costs 0.237 of balanced accuracy (base 0.805, 5 shuffles, R-VEX-13).",
  seconds: 1,
  notes: [],
};

function setup(runs: unknown[] = [run]) {
  return mockApi({
    "GET /health": health,
    "GET /datasets": [dataset({ format: "images", name: "pets" })],
    "GET /datasets/d1": dataset({ format: "images", name: "pets" }),
    "GET /datasets/d1/columns": [],
    "GET /datasets/d1/images/index": imageIndex(),
    "GET /datasets/d1/images/report": imageReport,
    "GET /images/rules": qualityRules,
    "GET /datasets/d1/images/decisions": { draft: {}, applied: {}, pending: 0, version: 1 },
    "GET /modules/vision/datasets/d1/runs": runs,
    [`GET /modules/vision/datasets/d1/runs/${RUN}/explain`]: overview,
    [`GET /modules/vision/datasets/d1/runs/${RUN}/explain/images/eeeeeeeeeeeeeeee`]: why,
    [`GET /modules/vision/datasets/d1/runs/${RUN}/explain/images/eeeeeeeeeeeeeeee/map`]: gradCam,
  });
}

describe("Why? (117)", () => {
  it("opens from the image sheet and shows the model's evidence first, nothing post-hoc", async () => {
    const calls = setup();
    renderAt("/d/d1/vision?image=dog%2F1.png");
    fireEvent.click(await screen.findByRole("button", { name: /Why\?/ }));
    const dialog = await screen.findByRole("dialog", { name: "Why? 1.png" });
    expect(await within(dialog).findByText(/says cat with probability 0.86/)).toBeInTheDocument();
    expect(within(dialog).getByText(/Wrong: labelled dog/)).toBeInTheDocument();
    expect(within(dialog).getByText(/1 of the 2 nearest training images are labelled cat/)).toBeInTheDocument();
    expect(within(dialog).getByText(/stands for 44%/)).toBeInTheDocument();
    expect(within(dialog).getByRole("group", { name: /The image and its evidence/ })).toBeInTheDocument();
    // the onion rule: no map is asked for before Advanced is opened and a method chosen
    expect(calls.some((call) => call.key.endsWith("/map"))).toBe(false);
    expect(within(dialog).queryByText(/Counterfactual example/)).not.toBeInTheDocument();
  });

  it("computes a post-hoc map only when chosen, with its written caveat", async () => {
    const calls = setup();
    renderAt("/d/d1/vision?why=dog%2F1.png");
    const dialog = await screen.findByRole("dialog");
    fireEvent.click(await within(dialog).findByText("Advanced"));
    expect(await within(dialog).findByText(/the model gives it 0.97 dog/)).toBeInTheDocument();
    expect(within(dialog).getAllByText(/post-hoc explanation/).length).toBeGreaterThan(0);
    expect(calls.some((call) => call.key.endsWith("/map"))).toBe(false);
    fireEvent.click(within(dialog).getByRole("radio", { name: /Grad-CAM, about 1 s/ }));
    expect(await within(dialog).findByText(/Grad-CAM for cat on the patch tokens/)).toBeInTheDocument();
    const map = calls.find((call) => call.key.endsWith("/map"));
    expect(map?.params.get("method")).toBe("grad_cam");
    expect(within(dialog).getByRole("radiogroup", { name: "Show" })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/Opacity/)).toHaveValue("60");
    // [ and ] change the opacity
    fireEvent.keyDown(window, { key: "]" });
    await waitFor(() => expect(within(dialog).getByLabelText(/Opacity/)).toHaveValue("70"));
    // More stays closed: no sanity test was run
    expect(calls.some((call) => call.key.endsWith("/sanity"))).toBe(false);
  });

  it("says how to get a model when the set has none", async () => {
    setup([]);
    renderAt("/d/d1/vision?why=dog%2F1.png");
    expect(await within(await screen.findByRole("dialog")).findByText("No model to explain yet")).toBeInTheDocument();
  });
});

describe("Explain tab (117)", () => {
  it("shows the prototypes, criticisms and descriptor families of the run", async () => {
    setup();
    renderAt("/d/d1/vision?tab=explain");
    expect(await screen.findByText("Prototypes and criticisms")).toBeInTheDocument();
    expect(screen.getByText("75 % of the class")).toBeInTheDocument();
    expect(screen.getByText("witness 0.070")).toBeInTheDocument();
    expect(screen.getByText("None: the prototypes cover the class.")).toBeInTheDocument();
    expect(screen.getByText(/leans most on texture \(LBP\)/)).toBeInTheDocument();
    expect(screen.getByRole("table", { name: "What the classic model reads" })).toBeInTheDocument();
  });
});
