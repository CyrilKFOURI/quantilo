import { describe, expect, it } from "vitest";

import type { ImageMap } from "./api";
import { colouring, insidePolygon, NONE, rowsOfEntry, zoomView, START } from "./mapModel";
import { toRows } from "./model";
import { delta, isOn, measureText, pickSamples, sameRecipe, stepInfo, stepsOn, toggled, type Recipe } from "./prepModel";
import { validateVisionSearch } from "./search";
import { imageIndex } from "./testData";

const rows = toRows(imageIndex());

function map(overrides: Partial<ImageMap> = {}): ImageMap {
  return {
    method: "classic",
    title: "Classic descriptors",
    projection: "pca",
    clustering: "kmeans",
    rows: [0, 2, 4, 5],
    x: [0, 1, 2, 3],
    y: [0, 1, 0, 1],
    clusters: [0, 0, 1, -1, 1, 1],
    k: 2,
    k_forced: false,
    scores: [],
    silhouette: 0.3,
    explained: [0.1, 0.05],
    reading: null,
    sentence: "",
    notes: [],
    points: 4,
    of: 5,
    ...overrides,
  };
}

describe("map colouring (109, R-VMAP-11)", () => {
  it("colours labels by name, never by count", () => {
    const found = colouring(rows, map(), "label");
    expect(found.kind).toBe("categorical");
    if (found.kind !== "categorical") return;
    expect(found.entries.map((e) => [e.key, e.slot, e.count])).toEqual([
      ["cat", 0, 2],
      ["dog", 1, 2],
    ]);
    expect(found.keyOf).toEqual(["cat", "cat", "dog", "dog"]);
  });

  it("puts images without a group in a neutral entry and sorts groups by number", () => {
    const found = colouring(rows, map({ clusters: [1, 0, 1, -1, 0, -1] }), "cluster");
    if (found.kind !== "categorical") throw new Error("categorical expected");
    expect(found.entries.map((e) => e.key)).toEqual(["0", "1", NONE]);
    expect(found.entries.at(-1)?.slot).toBe(-1);
  });

  it("folds more than 8 categories into Other", () => {
    const ten = Array.from({ length: 10 }, (_, i) => ({ ...(rows[0] as (typeof rows)[number]), i, key: `x/${i}.png`, label: `L${i}` }));
    const index = ten.map((_, i) => i);
    const found = colouring(ten, map({ rows: index, x: index, y: index, clusters: index }), "label");
    if (found.kind !== "categorical") throw new Error("categorical expected");
    expect(found.entries).toHaveLength(8);
    expect(found.entries.at(-1)).toMatchObject({ key: "Other", slot: -1, count: 3 });
  });

  it("colours a measure on the sequential scale (log for sharpness)", () => {
    const found = colouring(rows, map(), "sharpness");
    if (found.kind !== "sequential") throw new Error("sequential expected");
    expect(found.log).toBe(true);
    expect(found.min).toBeCloseTo(12);
    expect(found.max).toBeCloseTo(4000);
    expect(found.level[3]).toBe(1);
  });

  it("selects every image of a group, drawn or not (R-VMAP-08)", () => {
    const m = map();
    expect(rowsOfEntry(rows, m, "cluster", "1", ["0", "1", NONE])).toEqual([2, 4, 5]);
    expect(rowsOfEntry(rows, m, "label", "cat", ["cat", "dog"])).toEqual([0, 1, 2, 3]);
  });

  it("finds points inside a lasso and keeps the zoom within limits", () => {
    const square = [{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 10 }, { x: 0, y: 10 }];
    expect(insidePolygon(5, 5, square)).toBe(true);
    expect(insidePolygon(15, 5, square)).toBe(false);
    expect(zoomView(START, 0.5, 10, 10)).toEqual(START);
    const zoomed = zoomView(START, 2, 100, 50);
    expect(zoomed).toEqual({ scale: 2, x: -100, y: -50 });
    expect(zoomView(zoomed, 100, 0, 0).scale).toBe(24);
  });
});

const recipe: Recipe = {
  task: "classification", size: 224, fit: "crop", trim: false, white_balance: false, denoise: "off", sigma: 1,
  equalise: "off", clahe_clip: 2, clahe_tiles: 8, sharpen: false, sharpen_radius: 1, sharpen_amount: 1,
  augment: true, flip: true, rotate: 0, jitter: 0, crop_scale: 0.8,
};

describe("preparation (110)", () => {
  it("switches steps on and off with their usual setting", () => {
    expect(isOn(recipe, "equalise")).toBe(false);
    const on = toggled(recipe, "equalise", true);
    expect(on.equalise).toBe("clahe");
    expect(toggled(on, "equalise", false).equalise).toBe("off");
    expect(stepsOn(recipe)).toBe(1); // the augmentations
    const none = toggled({ ...recipe, augment: false, flip: false, crop_scale: 1 }, "augment", true);
    expect(none).toMatchObject({ augment: true, flip: true, crop_scale: 0.8 });
    expect(sameRecipe(recipe, { ...recipe })).toBe(true);
    expect(sameRecipe(recipe, on)).toBe(false);
  });

  it("picks six images by a said rule: typical, extremes, random (R-VPR-05)", () => {
    const samples = pickSamples(rows, stepInfo("sharpen"), 0);
    // the broken file is never picked; 5 decodable images here
    expect(samples).toHaveLength(5);
    expect(new Set(samples.map((s) => s.row.key)).size).toBe(5);
    expect(samples.slice(0, 2).map((s) => s.why)).toEqual(["typical", "typical"]);
    expect(samples.find((s) => s.why === "lowest")?.row.key).toBe("cat/blur.png");
    expect(samples.find((s) => s.why === "highest")?.row.key).toBe("dog/1.png");
    expect(samples.every((s) => s.row.key !== "cat/broken.png")).toBe(true);
    expect(pickSamples(rows, stepInfo("sharpen"), 0)).toEqual(samples); // seeded
  });

  it("shows one image six times for the augmentations", () => {
    const draws = pickSamples(rows, stepInfo("augment"), 2);
    expect(draws).toHaveLength(6);
    expect(new Set(draws.map((s) => s.row.key)).size).toBe(1);
    expect(draws.map((s) => s.seed)).toEqual([12, 13, 14, 15, 16, 17]);
  });

  it("writes changes with their sign", () => {
    expect(delta(0.5, 0.62)).toBe("+0.12");
    expect(delta(0.5, 0.47)).toBe("−0.03");
    expect(delta(0.5, 0.5)).toBe("0");
    expect(delta(undefined, 1)).toBeNull();
    expect(measureText("sharpness", 1234.4)).toBe("1,234");
    expect(measureText("contrast", undefined)).toBe("—");
  });

  it("keeps the map and prepare settings in the URL, and drops wrong ones", () => {
    expect(validateVisionSearch({ tab: "map", color: "sharpness", method: "dinov2_s14", proj: "tsne", k: 4, step: "equalise" })).toMatchObject({
      tab: "map",
      color: "sharpness",
      method: "dinov2_s14",
      proj: "tsne",
      k: 4,
      step: "equalise",
    });
    expect(validateVisionSearch({ color: "nope", method: "../x", k: 40, step: "glow" })).toMatchObject({
      color: undefined,
      method: undefined,
      k: undefined,
      step: undefined,
    });
  });
});
