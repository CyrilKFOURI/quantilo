import { describe, expect, it } from "vitest";

import { balanceSentence, measureSentence } from "./DatasetStats";
import {
  binRows,
  clickSelection,
  exclusionCause,
  filterAndSort,
  groupsOf,
  isExcluded,
  labelSeries,
  OTHER,
  positionsInRect,
  rankOf,
  sizeBucket,
  sortRows,
  toRows,
  type ImageRow,
} from "./model";
import { formatRange, parseRange, validateVisionSearch } from "./search";
import { imageIndex } from "./testData";

const rows = toRows(imageIndex());

describe("the gallery's index (107)", () => {
  it("reads one row per image, in table order, keyed by rel_path", () => {
    expect(rows).toHaveLength(6);
    expect(rows.map((row) => row.i)).toEqual([0, 1, 2, 3, 4, 5]);
    expect(rows[1]?.key).toBe("cat/copy.png");
    // exact copies share an image_id: the key stays unique
    expect(rows[0]?.id).toBe(rows[1]?.id);
    expect(new Set(rows.map((row) => row.key)).size).toBe(6);
    expect(rows[3]?.decodeOk).toBe(false);
  });

  it("says why an image is left out, the user's decision winning (R-108-05)", () => {
    const copy = rows[1] as ImageRow;
    expect(isExcluded(copy, {})).toBe(true);
    expect(exclusionCause(copy, {})).toEqual({ kind: "rule", code: "R-VQ-09" });
    expect(isExcluded(copy, { [copy.key]: "keep" })).toBe(false);
    expect(exclusionCause(copy, { [copy.key]: "keep" }).kind).toBe("kept");
    const plain = rows[2] as ImageRow;
    expect(isExcluded(plain, { [plain.key]: "exclude" })).toBe(true);
    expect(exclusionCause(plain, { [plain.key]: "exclude" }).kind).toBe("user");
  });
});

describe("filters and sorting", () => {
  it("filters by label, split, rule, size, name and a measure's range", () => {
    const keys = (filters: Parameters<typeof filterAndSort>[1]) =>
      filterAndSort(rows, filters, {}, "file", false).map((row) => row.key);
    expect(keys({ labels: ["dog"] })).toEqual(["dog/0.png", "dog/1.png"]);
    expect(keys({ splits: ["test"] })).toEqual(["dog/1.png"]);
    expect(keys({ quality: "R-VQ-04" })).toEqual(["cat/blur.png"]);
    expect(keys({ quality: "any" })).toHaveLength(4); // the kept copy carries R-VQ-09 too
    expect(keys({ quality: "none" })).toHaveLength(2);
    expect(keys({ quality: "excluded" })).toEqual(["cat/copy.png", "cat/broken.png"]);
    expect(filterAndSort(rows, { quality: "decided" }, { "dog/0.png": "keep" }, "file", false)).toHaveLength(1);
    expect(keys({ size: "large" })).toEqual(["dog/1.png"]);
    expect(keys({ q: "BLUR" })).toEqual(["cat/blur.png"]);
    expect(keys({ range: { measure: "brightness", low: 0.4, high: 0.6 } })).toEqual(["cat/0.png", "cat/copy.png", "dog/0.png"]);
  });

  it("sorts with missing values last and ties in table order", () => {
    const sorted = sortRows(rows, "sharpness");
    expect(sorted.at(-1)?.key).toBe("cat/broken.png");
    expect(sorted[0]?.key).toBe("cat/blur.png");
    expect(sortRows(rows, "sharpness", true)[0]?.key).toBe("dog/1.png");
    expect(sortRows(rows, "label").map((row) => row.label)).toEqual(["cat", "cat", "cat", "cat", "dog", "dog"]);
  });

  it("buckets sizes by the short side", () => {
    expect(rows.map(sizeBucket)).toEqual(["small", "small", "medium", null, "medium", "large"]);
  });
});

describe("statistics", () => {
  it("bins a measure in equal classes, sharpness on a log scale, per label when asked", () => {
    const bins = binRows(rows, "brightness", { count: 4, labels: ["cat", "dog"] });
    expect(bins.counts.reduce((a, b) => a + b, 0)).toBe(5); // the broken file has no value
    expect(bins.edges).toHaveLength(5);
    expect(bins.byLabel?.get("cat")?.reduce((a, b) => a + b, 0)).toBe(3);
    const sharp = binRows(rows, "sharpness", { count: 3 });
    expect(sharp.log).toBe(true);
    expect(sharp.edges[0]).toBeCloseTo(12, 6);
    expect(sharp.edges[3]).toBeCloseTo(4000, 4);
  });

  it("ranks a value as the share of images strictly below it (R-107-08)", () => {
    const sorted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    expect(rankOf(sorted, 10)).toBe(0.9);
    expect(rankOf(sorted, 1)).toBe(0);
    expect(rankOf(sorted, null)).toBeNull();
  });

  it("gives each label a fixed palette slot, eight at most", () => {
    expect(labelSeries(["b", "a"])).toEqual(["a", "b"]);
    const many = labelSeries(Array.from({ length: 10 }, (_, k) => `l${k}`));
    expect(many).toHaveLength(8);
    expect(many.at(-1)).toBe(OTHER);
  });

  it("writes the conclusion above the charts", () => {
    expect(balanceSentence([["cat", 30], ["dog", 6]])).toBe("Imbalanced: cat has 5.0× the images of dog (R-VQ-13).");
    expect(balanceSentence([["cat", 4], ["dog", 3]])).toMatch(/^Classes are balanced/);
    expect(measureSentence("sharpness", [20, 300, 500])).toBe("Median 300; 1 images under 100, the usual blur threshold.");
  });

  it("groups the copies", () => {
    const groups = groupsOf(rows);
    expect([...groups.values()].map((members) => members.map((row) => row.key))).toEqual([["cat/0.png", "cat/copy.png"]]);
  });
});

describe("selection", () => {
  const keysAt = (set: Set<string>) => rows.filter((row) => set.has(row.key)).map((row) => row.i);

  it("selects one, a range with Shift, adds with Ctrl", () => {
    const one = clickSelection(new Set(), rows, 2, null, { shift: false, toggle: false });
    expect(keysAt(one)).toEqual([2]);
    const range = clickSelection(one, rows, 5, 2, { shift: true, toggle: false });
    expect(keysAt(range)).toEqual([2, 3, 4, 5]);
    const toggled = clickSelection(range, rows, 3, 2, { shift: false, toggle: true });
    expect(keysAt(toggled)).toEqual([2, 4, 5]);
  });

  it("finds the tiles under a rectangle by arithmetic (R-107-05)", () => {
    const grid = { columns: 4, cellWidth: 100, rowHeight: 130, gap: 10, count: 10 };
    expect(positionsInRect({ x0: 50, y0: 50, x1: 160, y1: 200 }, grid)).toEqual([0, 1, 4, 5]);
    // starting in the gap after column 0 does not take column 0
    expect(positionsInRect({ x0: 105, y0: 10, x1: 120, y1: 20 }, grid)).toEqual([1]);
    expect(positionsInRect({ x0: 0, y0: 290, x1: 999, y1: 300 }, grid)).toEqual([8, 9]);
  });
});

describe("the page's URL (R-107-11)", () => {
  it("keeps only valid values, defaults left out", () => {
    const search = validateVisionSearch({
      tab: "quality",
      labels: ["cat"],
      quality: "R-VQ-04",
      sort: "sharpness",
      desc: true,
      density: "l",
      range: "brightness:0.6:0.2",
      stats: false,
      size: "huge",
    });
    expect(search).toMatchObject({ tab: "quality", labels: ["cat"], quality: "R-VQ-04", sort: "sharpness", desc: true, density: "l", stats: false });
    expect(search.size).toBeUndefined();
    expect(parseRange(search.range)).toEqual({ measure: "brightness", low: 0.2, high: 0.6 });
    expect(validateVisionSearch({ tab: "gallery", sort: "file", quality: "nope", range: "x:1:2" })).toEqual(
      expect.objectContaining({ tab: undefined, sort: undefined, quality: undefined, range: undefined }),
    );
    expect(formatRange({ measure: "sharpness", low: 12.345678, high: 99.99999 })).toBe("sharpness:12.35:100");
  });
});
