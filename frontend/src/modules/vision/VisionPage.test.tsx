import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import {
  afterAll,
  afterEach,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { dataset, health, mockApi } from "../../test/mockApi";
import { renderAt } from "../../test/render";
import { imageIndex, imageReport, qualityRules } from "./testData";

// jsdom lays nothing out: give elements a size so the virtualized grid shows its rows.
const sizes = [
  "offsetHeight",
  "offsetWidth",
  "clientHeight",
  "clientWidth",
] as const;
const saved = sizes.map((key) =>
  Object.getOwnPropertyDescriptor(HTMLElement.prototype, key),
);
beforeAll(() => {
  for (const key of sizes) {
    Object.defineProperty(HTMLElement.prototype, key, {
      configurable: true,
      get: () => (key.endsWith("Height") ? 900 : 1200),
    });
  }
});
afterAll(() => {
  sizes.forEach((key, index) => {
    const descriptor = saved[index];
    if (descriptor)
      Object.defineProperty(HTMLElement.prototype, key, descriptor);
  });
});
afterEach(() => vi.unstubAllGlobals());

function setup(
  decisions = { draft: {}, applied: {}, pending: 0, version: 1 },
  extra: Record<string, unknown> = {},
) {
  return mockApi({
    ...extra,
    "GET /health": health,
    "GET /datasets": [dataset({ format: "images", name: "pets" })],
    "GET /datasets/d1": dataset({ format: "images", name: "pets" }),
    "GET /datasets/d1/columns": [],
    "GET /datasets/d1/images/index": imageIndex(),
    "GET /datasets/d1/images/report": imageReport,
    "GET /images/rules": qualityRules,
    "GET /datasets/d1/images/decisions": decisions,
    "PUT /datasets/d1/images/decisions": (body: unknown) => {
      const sent = body as {
        decisions?: Record<string, string | null>;
        labels?: Record<string, string | null>;
        pins?: Record<string, boolean | null>;
        thresholds?: Record<string, number>;
      };
      const keep = <T,>(part: Record<string, T | null> | undefined) =>
        Object.fromEntries(
          Object.entries(part ?? {}).filter(([, value]) => value !== null && value !== false),
        ) as Record<string, T>;
      const draft = keep(sent.decisions);
      const labels = keep(sent.labels);
      const pins = keep(sent.pins);
      const changed = sent.thresholds ? 1 : 0;
      const count = Object.keys(draft).length + Object.keys(labels).length + Object.keys(pins).length + changed;
      return {
        draft,
        applied: {},
        labels_draft: labels,
        pins_draft: pins,
        thresholds_draft: sent.thresholds ?? {},
        thresholds_applied: {},
        pending: count,
        pending_by: { decisions: Object.keys(draft).length, labels: Object.keys(labels).length, pins: Object.keys(pins).length, thresholds: changed },
        version: 1,
      };
    },
    "POST /datasets/d1/images/decisions/apply": {
      version: 2,
      excluded: 3,
      decided: 1,
    },
    "GET /images/map/methods": mapMethods,
    "GET /datasets/d1/images/map": {
      state: "ready",
      method: "classic",
      dims: 5,
      map: imageMap,
    },
    "GET /datasets/d1/images/prepare": prepare,
    "PUT /datasets/d1/images/prepare": (body: unknown) => ({
      ...prepare,
      saved: true,
      recipe: (body as { recipe: unknown }).recipe,
    }),
    "POST /datasets/d1/images/prepare/preview": (body: unknown) => {
      const { image_id, step } = body as { image_id: string; step: string };
      return {
        step,
        image_id,
        enabled: false,
        geometric: step === "resize",
        before: "data:image/png;base64,AA",
        after: "data:image/png;base64,BB",
        before_size: [400, 300],
        after_size: [224, 224],
        before_measures: { brightness: 0.5, contrast: 0.2, sharpness: 300 },
        after_measures: { brightness: 0.52, contrast: 0.31, sharpness: 280 },
        notes: [],
      };
    },
  });
}

const imageMap = {
  method: "classic",
  title: "Classic descriptors",
  projection: "pca",
  clustering: "kmeans",
  rows: [0, 2, 4, 5],
  x: [0, 1, 2, 3],
  y: [0, 1, 0, 1],
  clusters: [0, 0, 0, -1, 1, 1],
  k: 2,
  k_forced: false,
  scores: [
    { k: 2, silhouette: 0.31 },
    { k: 3, silhouette: 0.2 },
  ],
  silhouette: 0.31,
  explained: [0.12, 0.08],
  reading: { ari: 0.9, neighbours_same: 0.9, chance: 0.5, labels: 2 },
  sentence: "Clusters match labels: adjusted Rand 0.90.",
  notes: [],
  points: 4,
  of: 5,
};

const mapMethods = {
  projections: [
    { key: "pacmap", title: "PaCMAP", licence: "Apache-2.0", available: true, reason: null, install: null },
    { key: "trimap", title: "TriMAP", licence: "Apache-2.0", available: false, reason: "R-XAI-10: trimap is not installed (optional extra `interpret`).", install: 'pip install "quantilo[interpret]"' },
  ],
  installed: false,
  reason: "Needs the optional extra vision; missing: torch (R-VDP-01).",
  install: "pip install quantilo[vision]",
  downloads: {},
  methods: [
    {
      key: "classic",
      title: "Classic descriptors",
      detail: "Colour, shape and texture.",
      level: "first",
      dims: 2072,
      available: true,
      downloaded: true,
      size: null,
      licence: null,
      note: null,
      reason: null,
    },
    {
      key: "dinov2_s14",
      title: "DINOv2 ViT-S/14",
      detail: "A frozen network.",
      level: "first",
      dims: 768,
      available: false,
      downloaded: false,
      size: "88 MB",
      licence: "Apache-2.0",
      note: null,
      reason: "Needs the optional extra vision (R-VDP-01).",
    },
  ],
};

const recipe = {
  kind: "vision-prepare",
  version: 1,
  task: "classification",
  size: 224,
  fit: "crop",
  trim: false,
  white_balance: false,
  denoise: "off",
  sigma: 1,
  equalise: "off",
  clahe_clip: 2,
  clahe_tiles: 8,
  sharpen: false,
  sharpen_radius: 1,
  sharpen_amount: 1,
  augment: true,
  flip: true,
  rotate: 0,
  jitter: 0,
  crop_scale: 0.8,
};
const prepare = {
  recipe,
  default: recipe,
  saved: false,
  task: "classification",
  borders: { share: 0, looked_at: 5, suggest: false },
};

describe("Vision page (107, 108)", () => {
  it("shows the set's figures, its tabs and every image", async () => {
    setup();
    renderAt("/d/d1/vision?stats=false");
    expect(
      await screen.findByRole("heading", { name: "Vision" }),
    ).toBeInTheDocument();
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    expect(within(grid).getAllByRole("option")).toHaveLength(6);
    expect(
      within(grid).getByRole("option", { name: "blur.png, cat, R-VQ-04" }),
    ).toBeInTheDocument();
    // the second exact copy is left out by its rule, said on the tile
    expect(within(grid).getByText("Excluded · R-VQ-09")).toBeInTheDocument();
    // every tab works since 117 (Explain)
    expect(screen.getByRole("tab", { name: "Explain" })).not.toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Model" })).not.toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(screen.getByRole("tab", { name: "Map" })).not.toHaveAttribute(
      "aria-disabled",
      "true",
    );
    expect(screen.getAllByText("Left out of models").length).toBeGreaterThan(0);
    // the sidebar lists Vision for an image set
    expect(
      screen.getAllByRole("link", { name: "Vision" }).length,
    ).toBeGreaterThan(0);
  });

  it("filters by a rule's chip and selects with Shift", async () => {
    setup();
    const router = renderAt("/d/d1/vision?stats=false");
    fireEvent.click(await screen.findByRole("button", { name: /^Blurry/ }));
    await waitFor(() =>
      expect(
        screen.getByRole("listbox", { name: "1 image" }),
      ).toBeInTheDocument(),
    );
    expect(router.state.location.search).toMatchObject({ quality: "R-VQ-04" });
    fireEvent.click(screen.getByRole("button", { name: "Clear all" }));
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    const options = within(grid).getAllByRole("option");
    fireEvent.click(options[0] as HTMLElement);
    fireEvent.click(options[2] as HTMLElement, { shiftKey: true });
    expect(screen.getByText("3 selected")).toBeInTheDocument();
    expect(
      options.filter(
        (option) => option.getAttribute("aria-selected") === "true",
      ),
    ).toHaveLength(3);
  });

  it("opens an image's sheet with its measures, rules and copies, and goes to the next", async () => {
    setup();
    const router = renderAt("/d/d1/vision?stats=false");
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    fireEvent.doubleClick(
      within(grid).getByRole("option", { name: "0.png, cat, R-VQ-09" }),
    );
    const sheet = await screen.findByRole("dialog");
    expect(
      within(sheet).getByText("200 × 200 PNG, 7.8 KB"),
    ).toBeInTheDocument();
    expect(within(sheet).getByText("Exact copies")).toBeInTheDocument();
    expect(within(sheet).getByText("Exact duplicate")).toBeInTheDocument();
    expect(within(sheet).getByText("Used by models.")).toBeInTheDocument();
    expect(within(sheet).getByText("1 of 6")).toBeInTheDocument();
    expect(router.state.location.search).toMatchObject({ image: "cat/0.png" });
    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() =>
      expect(router.state.location.search).toMatchObject({
        image: "cat/copy.png",
      }),
    );
    expect(
      await within(screen.getByRole("dialog")).findByText(
        /Left out of models by R-VQ-09/,
      ),
    ).toBeInTheDocument();
  });

  it("never shows a location, only that it is not read", async () => {
    setup();
    renderAt("/d/d1/vision?stats=false&image=%22dog%2F1.png%22");
    const sheet = await screen.findByRole("dialog");
    expect(
      within(sheet).getByText(/never read nor shown \(R-IMG-05\)/),
    ).toBeInTheDocument();
  });

  it("lists the quality findings and saves decisions as a version (108)", async () => {
    const calls = setup();
    renderAt("/d/d1/vision?tab=quality");
    expect(
      await screen.findByText(
        "4 images with an issue (R-VQ-01, R-VQ-04, R-VQ-09).",
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /R-VQ-04\s*Blurry/ }),
    ).toBeInTheDocument();
    expect(screen.getByText(/blur_sharpness 100/)).toBeInTheDocument();
    expect(
      screen.getByText("Rules that found nothing (1)"),
    ).toBeInTheDocument();
    // keep the copy, leave the first out: one request
    const strips = screen.getAllByRole("button", { name: "Keep this one" });
    fireEvent.click(strips[1] as HTMLElement);
    await waitFor(() =>
      expect(
        calls.some((call) => call.key === "PUT /datasets/d1/images/decisions"),
      ).toBe(true),
    );
    const put = calls.find(
      (call) => call.key === "PUT /datasets/d1/images/decisions",
    )?.body as { decisions: Record<string, string> };
    expect(put.decisions).toEqual({
      "cat/0.png": "exclude",
      "cat/copy.png": "keep",
    });
    fireEvent.click(
      await screen.findByRole("button", { name: "Save as version 2" }),
    );
    expect(
      await screen.findByText(
        /Saved as version 2: 3 images left out of models/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: "See what changed" }),
    ).toBeInTheDocument();
  });

  it("labels the selection with a digit and pins it with P (c3)", async () => {
    const calls = setup();
    renderAt("/d/d1/vision?stats=false");
    const grid = await screen.findByRole("listbox", { name: "6 images" });
    fireEvent.click(within(grid).getByRole("option", { name: "blur.png, cat, R-VQ-04" }));
    // R-VED-01: the labels in alphabetical order, cat 1 and dog 2
    fireEvent.keyDown(grid, { key: "2" });
    await waitFor(() =>
      expect(
        within(grid).getByRole("option", { name: /blur\.png, dog, was cat at the import/ }),
      ).toBeInTheDocument(),
    );
    const put = calls.find((call) => call.key === "PUT /datasets/d1/images/decisions")?.body as { labels: Record<string, string> };
    expect(put.labels).toEqual({ "cat/blur.png": "dog" });
    fireEvent.keyDown(grid, { key: "p" });
    await waitFor(() => expect(within(grid).getByRole("option", { name: /Pinned/ })).toBeInTheDocument());
    expect(await screen.findByText(/changes? not saved./)).toBeInTheDocument();
    // the Pinned filter shows it alone
    fireEvent.change(screen.getByRole("combobox", { name: "Quality" }), { target: { value: "pinned" } });
    await waitFor(() => expect(screen.getByRole("listbox", { name: "1 image" })).toBeInTheDocument());
  });

  it("previews other thresholds, then puts them in the draft (c3)", async () => {
    const calls = setup(undefined, {
      "GET /images/thresholds": { blur_z: 3, blur_sharpness: 100, min_side: 32, min_relative: 8 },
      "POST /datasets/d1/images/quality/preview": {
        quality: { ...imageReport.quality, sentence: "5 images with an issue (R-VQ-04)." },
        counts: { "R-VQ-04": 5 },
        current_counts: { "R-VQ-04": 1 },
        current_flagged: 4,
        current_excluded: 2,
      },
    });
    renderAt("/d/d1/vision?tab=quality");
    expect(calls.some((call) => call.key === "GET /images/thresholds")).toBe(false);
    fireEvent.click(await screen.findByText("Thresholds"));
    const field = await screen.findByRole("spinbutton", { name: "blur_sharpness" });
    fireEvent.change(field, { target: { value: "300" } });
    fireEvent.click(screen.getByRole("button", { name: "Preview" }));
    expect(await screen.findByText(/5 images with an issue \(R-VQ-04\)\./)).toBeInTheDocument();
    const preview = calls.find((call) => call.key === "POST /datasets/d1/images/quality/preview")?.body as { thresholds: Record<string, number> };
    expect(preview.thresholds).toEqual({ blur_sharpness: 300 });
    fireEvent.click(screen.getByRole("button", { name: "Use these thresholds" }));
    await waitFor(() => expect(calls.some((call) => call.key === "PUT /datasets/d1/images/decisions")).toBe(true));
    const put = calls.find((call) => call.key === "PUT /datasets/d1/images/decisions")?.body as { thresholds: Record<string, number> };
    expect(put.thresholds).toEqual({ blur_sharpness: 300 });
    expect(await screen.findByText(/Saving runs the quality rules again/)).toBeInTheDocument();
  });

  it("reads R-VQ-16 and R-VQ-17 only once Advanced is opened, and searches on request (b2)", async () => {
    const semantic = {
      available: true,
      reason: null,
      network: "DINOv2 ViT-S/14",
      cosine: 0.95,
      cached: 0,
      images: 6,
      groups: [],
    };
    const calls = setup(undefined, {
      "GET /datasets/d1/images/semantic": { ...semantic, state: "missing" },
      "POST /datasets/d1/images/semantic": {
        ...semantic,
        state: "ready",
        sentence:
          "1 group of near-copies seen by DINOv2 ViT-S/14 that the hashes of R-VQ-10 missed (2 images, cosine at least 0.95).",
        groups: [
          {
            group: 0,
            kind: "semantic",
            rows: [4, 5],
            keep: null,
            cosine: 0.962,
            two_labels: false,
            leak: true,
          },
        ],
        leaks: 1,
      },
      "GET /modules/vision/datasets/d1/mislabelled": {
        run_id: "r1",
        created_at: "",
        model: "Linear probe",
        message:
          "1 image whose label gets less than 10 % from a model that never saw it, while another class gets more than 80 %: check its label.",
        reason: "Confident learning.",
        count: 1,
        checked: 6,
        rows: [4],
        images: [
          {
            row: 4,
            label: "dog",
            suggested: "cat",
            suggested_proba: 0.93,
            label_proba: 0.04,
          },
        ],
        notes: [],
        parameters: { given: 0.1, other: 0.8 },
      },
    });
    renderAt("/d/d1/vision?tab=quality");
    const summary = await screen.findByText(
      /Advanced: rules that need a model/,
    );
    expect(
      calls.some(
        (call) =>
          call.key.includes("/semantic") || call.key.includes("/mislabelled"),
      ),
    ).toBe(false);
    const details = summary.closest("details") as HTMLDetailsElement;
    details.open = true;
    fireEvent(details, new Event("toggle"));
    expect(await screen.findByText(/check its label/)).toBeInTheDocument();
    expect(screen.getByText("dog → cat 93 %")).toBeInTheDocument();
    fireEvent.click(
      await screen.findByRole("button", { name: "Find semantic near-copies" }),
    );
    expect(
      await screen.findByText(/1 group of near-copies seen by DINOv2/),
    ).toBeInTheDocument();
    expect(screen.getByText("lowest cosine 0.962")).toBeInTheDocument();
    expect(screen.getByText(/spans train and test/)).toBeInTheDocument();
  });

  it("shows the map with its sentence and legend, and sends a group to the gallery (109)", async () => {
    setup();
    const router = renderAt("/d/d1/vision?tab=map");
    expect(
      await screen.findByText("Clusters match labels: adjusted Rand 0.90."),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /Map of 4 images, coloured by Label/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/The two axes keep 12 % and 8 %/),
    ).toBeInTheDocument();
    // the deep option is greyed with its reason, never downloaded without asking
    expect(
      screen.getByText("Needs the optional extra vision (R-VDP-01)."),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /Download/ }),
    ).not.toBeInTheDocument();
    fireEvent.click(
      screen.getByRole("button", { name: "Select every image of dog" }),
    );
    expect(await screen.findByText("2 images selected")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show in gallery" }));
    expect(
      await screen.findByRole("listbox", { name: "2 images" }),
    ).toBeInTheDocument();
    expect(router.state.location.search).not.toHaveProperty("tab");
    fireEvent.click(screen.getByRole("button", { name: /From the map: 2/ }));
    expect(
      await screen.findByRole("listbox", { name: "6 images" }),
    ).toBeInTheDocument();
  });

  it("offers PaCMAP and TriMAP in Advanced › More, greyed with their reason (c3, R-VMAP-12)", async () => {
    const calls = setup();
    const router = renderAt("/d/d1/vision?tab=map");
    expect(await screen.findByText("Clusters match labels: adjusted Rand 0.90.")).toBeInTheDocument();
    const open = (summary: HTMLElement) => {
      const details = summary.closest("details") as HTMLDetailsElement;
      details.open = true;
      fireEvent(details, new Event("toggle"));
    };
    open(screen.getByText("t-SNE, HDBSCAN, the number of groups"));
    open(await screen.findByText("More: PaCMAP, TriMAP"));
    expect(await screen.findByText(/trimap is not installed/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Draw with TriMAP" })).toBeDisabled();
    // nothing was asked before the click
    expect(calls.some((call) => call.key === "GET /datasets/d1/images/map" && call.params.get("projection") === "pacmap")).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Draw with PaCMAP" }));
    await waitFor(() => expect(router.state.location.search).toMatchObject({ proj: "pacmap" }));
  });

  it("previews a step before and after on sample images and saves the recipe (110)", async () => {
    const calls = setup();
    renderAt("/d/d1/vision?tab=prepare");
    expect(
      await screen.findByText(
        /Images enter the model at 224 × 224 px, centre-cropped/,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("heading", { name: /Input size/ }),
    ).toBeInTheDocument();
    await waitFor(() =>
      expect(
        screen.getAllByRole("group", { name: /before and after Input size/ }),
      ).toHaveLength(5),
    );
    expect(screen.getAllByText("400 × 300 px").length).toBeGreaterThan(0);
    // the augmentations are on (classification): the Advanced steps show, in their places
    expect(
      screen.getByRole("button", { name: /Hide Advanced steps/ }),
    ).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(
      screen.getByRole("checkbox", { name: "Use Equalise contrast" }),
    );
    fireEvent.click(screen.getByRole("button", { name: /^Equalise contrast/ }));
    expect(
      await screen.findByRole("heading", { name: /Equalise contrast/ }),
    ).toBeInTheDocument();
    expect((await screen.findAllByText("(+0.11)")).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Save recipe" }));
    await waitFor(() =>
      expect(
        calls.some((call) => call.key === "PUT /datasets/d1/images/prepare"),
      ).toBe(true),
    );
    const put = calls.find(
      (call) => call.key === "PUT /datasets/d1/images/prepare",
    )?.body as { recipe: { equalise: string } };
    expect(put.recipe.equalise).toBe("clahe");
    expect(await screen.findByText("Saved recipe")).toBeInTheDocument();
  });

  it("is not listed for a table", async () => {
    mockApi({
      "GET /health": health,
      "GET /datasets": [dataset()],
      "GET /datasets/d1": dataset(),
      "GET /datasets/d1/columns": [],
      "GET /datasets/d1/overview": { error: "skip" },
    });
    renderAt("/d/d1/vision");
    expect(
      await screen.findByText(/This dataset is a table, not an image set/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("link", { name: "Vision" }),
    ).not.toBeInTheDocument();
  });
});
