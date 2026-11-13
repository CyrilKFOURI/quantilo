import type { ImageIndex } from "./api";

/** What the user decided for an image (108): left out of models, or kept despite its rule. */
export type Decision = "exclude" | "keep";

/** One image of the set, read from the columnar index (107). `i` is its row in the table. */
export type ImageRow = {
  i: number;
  id: string;
  key: string; // rel_path: unique, where exact copies share their image_id
  name: string;
  label: string | null;
  /** c3, R-VED-02: the label read at the import (equal to `label` until one is edited). */
  labelOriginal: string | null;
  /** c3, R-VED-03: marked to come back to; no effect on models. */
  pinned: boolean;
  split: string | null;
  format: string | null;
  width: number | null;
  height: number | null;
  bytes: number | null;
  megapixels: number | null;
  aspect: number | null;
  brightness: number | null;
  contrast: number | null;
  sharpness: number | null;
  saturation: number | null;
  colourfulness: number | null;
  entropy: number | null;
  lumaP05: number | null;
  lumaP95: number | null;
  clippedDark: number | null;
  clippedLight: number | null;
  colorMode: string | null;
  bitDepth: number | null;
  frames: number | null;
  camera: string | null;
  dateTaken: string | null;
  dominant: string | null;
  decodeOk: boolean;
  decodeError: string | null;
  issues: string[];
  ruleExcluded: boolean;
  group: number | null;
  hasMask: boolean;
  hasLocation: boolean;
  isNormal: boolean | null;
  isGrayscale: boolean | null;
};

export function toRows(index: ImageIndex): ImageRow[] {
  const rows: ImageRow[] = [];
  for (let i = 0; i < index.count; i += 1) {
    const key = index.rel_path[i] ?? String(i);
    rows.push({
      i,
      id: index.image_id[i] ?? "",
      key,
      name: index.file_name[i] ?? key,
      label: index.label[i] ?? null,
      labelOriginal: index.label_original[i] ?? index.label[i] ?? null,
      pinned: Boolean(index.pinned[i]),
      split: index.split[i] ?? null,
      format: index.format[i] ?? null,
      width: index.width[i] ?? null,
      height: index.height[i] ?? null,
      bytes: index.file_bytes[i] ?? null,
      megapixels: index.megapixels[i] ?? null,
      aspect: index.aspect_ratio[i] ?? null,
      brightness: index.brightness[i] ?? null,
      contrast: index.contrast[i] ?? null,
      sharpness: index.sharpness[i] ?? null,
      saturation: index.saturation[i] ?? null,
      colourfulness: index.colourfulness[i] ?? null,
      entropy: index.entropy[i] ?? null,
      lumaP05: index.luma_p05[i] ?? null,
      lumaP95: index.luma_p95[i] ?? null,
      clippedDark: index.clipped_dark[i] ?? null,
      clippedLight: index.clipped_light[i] ?? null,
      colorMode: index.color_mode[i] ?? null,
      bitDepth: index.bit_depth[i] ?? null,
      frames: index.frames[i] ?? null,
      camera: index.camera[i] ?? null,
      dateTaken: index.date_taken[i] ?? null,
      dominant: index.dominant_colour[i] ?? null,
      decodeOk: index.decode_ok[i] !== false,
      decodeError: index.decode_error[i] ?? null,
      issues: index.issues[i] ?? [],
      ruleExcluded: Boolean(index.rule_excluded[i]),
      group: index.duplicate_group[i] ?? null,
      hasMask: Boolean(index.has_mask[i]),
      hasLocation: Boolean(index.has_location[i]),
      isNormal: index.is_normal[i] ?? null,
      isGrayscale: index.is_grayscale[i] ?? null,
    });
  }
  return rows;
}

export type Draft = Readonly<Record<string, Decision>>;

/** c3: the labels and pins of the draft (R-VED-01, R-VED-03), by rel_path. */
export type Edits = { labels: Readonly<Record<string, string>>; pins: Readonly<Record<string, boolean>> };
export const NO_EDITS: Edits = { labels: {}, pins: {} };

/** The rows as the draft shows them: a label edited (else the import's), pinned or not. The table
 *  holds the saved state; the draft is read against the import's label, so a label taken back in
 *  the draft shows the original again. */
export function withEdits(rows: readonly ImageRow[], edits: Edits): ImageRow[] {
  return rows.map((row) => {
    const label = edits.labels[row.key] ?? row.labelOriginal;
    const pinned = Boolean(edits.pins[row.key]);
    return label === row.label && pinned === row.pinned ? row : { ...row, label, pinned };
  });
}

/** R-VED-01: the labels keys 1 to 9 give, in alphabetical order (the import's and the new ones). */
export function labelChoices(rows: readonly ImageRow[]): string[] {
  const labels = new Set<string>();
  for (const row of rows) {
    if (row.label !== null) labels.add(row.label);
    if (row.labelOriginal !== null) labels.add(row.labelOriginal);
  }
  return [...labels].sort((a, b) => a.localeCompare(b));
}

/** The label patch of a key pressed on a selection: 1-9 the n-th label, 0 back to the import's. */
export function labelPatch(keys: readonly string[], digit: number, choices: readonly string[]): Record<string, string | null> | null {
  if (digit === 0) return Object.fromEntries(keys.map((key) => [key, null]));
  const label = choices[digit - 1];
  if (label === undefined) return null;
  return Object.fromEntries(keys.map((key) => [key, label]));
}

/** R-108-05: `keep` wins over a rule's exclusion, `exclude` over a flag; no decision: the rule. */
export function isExcluded(row: ImageRow, draft: Draft): boolean {
  const decision = draft[row.key];
  return decision === "exclude" || (row.ruleExcluded && decision !== "keep");
}

/** Why an image is (or is not) left out: by a rule (its code), by the user, or kept by the user. */
export function exclusionCause(row: ImageRow, draft: Draft): { kind: "rule" | "user" | "kept" | "none"; code?: string } {
  const decision = draft[row.key];
  if (decision === "exclude") return { kind: "user" };
  if (decision === "keep") return { kind: row.ruleExcluded ? "kept" : "none" };
  if (row.ruleExcluded) return { kind: "rule", code: EXCLUDING.find((code) => row.issues.includes(code)) };
  return { kind: "none" };
}

/** The rules that leave an image out of models (105): broken file, second exact copy, tiny class. */
export const EXCLUDING = ["R-VQ-01", "R-VQ-09", "R-VQ-14"];

// --- measures ------------------------------------------------------------------------------------

export const MEASURES = ["brightness", "contrast", "sharpness", "saturation", "colourfulness", "entropy"] as const;
export type Measure = (typeof MEASURES)[number];

/** The histogram of sharpness is read on log10(1 + x): the variance of the Laplacian spreads over
 *  four orders of magnitude (R-107-09). */
export const LOG_MEASURES: ReadonlySet<Measure> = new Set(["sharpness"]);

export function measureOf(row: ImageRow, measure: Measure): number | null {
  return row[measure];
}

/** The decoded values of a measure, sorted, for ranks and quantiles. */
export function sortedValues(rows: readonly ImageRow[], measure: Measure): number[] {
  const values: number[] = [];
  for (const row of rows) {
    const value = row[measure];
    if (value !== null && Number.isFinite(value)) values.push(value);
  }
  return values.sort((a, b) => a - b);
}

/** R-107-08: the share of images strictly below this value ("sharper than 92 % of images"). */
export function rankOf(sorted: readonly number[], value: number | null): number | null {
  if (value === null || sorted.length < 2) return null;
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((sorted[middle] ?? 0) < value) low = middle + 1;
    else high = middle;
  }
  return low / sorted.length;
}

export function quantile(sorted: readonly number[], q: number): number | null {
  if (sorted.length === 0) return null;
  const position = (sorted.length - 1) * q;
  const below = Math.floor(position);
  const above = Math.ceil(position);
  const a = sorted[below] ?? 0;
  const b = sorted[above] ?? a;
  return a + (b - a) * (position - below);
}

export type Bins = {
  /** Edges in the measure's own units (count + 1 of them). */
  edges: number[];
  counts: number[];
  /** Per label (the label order given), when asked. */
  byLabel: Map<string, number[]> | null;
  log: boolean;
};

const forward = (value: number, log: boolean) => (log ? Math.log10(1 + Math.max(value, 0)) : value);
const backward = (value: number, log: boolean) => (log ? 10 ** value - 1 : value);

/** Equal-width classes (on log10(1 + x) for a log measure), the last one closed. */
export function binRows(
  rows: readonly ImageRow[],
  measure: Measure,
  options: { count?: number; labels?: readonly string[] | null } = {},
): Bins {
  const count = options.count ?? 24;
  const log = LOG_MEASURES.has(measure);
  const values = rows.map((row) => row[measure]);
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (value === null || !Number.isFinite(value)) continue;
    const v = forward(value, log);
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!Number.isFinite(min)) return { edges: [], counts: [], byLabel: null, log };
  if (max === min) max = min + 1e-6;
  const width = (max - min) / count;
  const edges = Array.from({ length: count + 1 }, (_, k) => backward(min + k * width, log));
  const counts = new Array<number>(count).fill(0);
  const labelIndex = options.labels ? new Map(options.labels.map((label, k) => [label, k])) : null;
  const byLabel = options.labels ? new Map(options.labels.map((label) => [label, new Array<number>(count).fill(0)])) : null;
  values.forEach((value, k) => {
    if (value === null || !Number.isFinite(value)) return;
    const bin = Math.min(count - 1, Math.floor((forward(value, log) - min) / width));
    counts[bin] = (counts[bin] ?? 0) + 1;
    if (byLabel && labelIndex) {
      const label = rows[k]?.label ?? OTHER;
      const series = byLabel.get(labelIndex.has(label) ? label : OTHER);
      if (series) series[bin] = (series[bin] ?? 0) + 1;
    }
  });
  return { edges, counts, byLabel, log };
}

/** The label series a chart colours (R-107-09): the palette's 8 fixed slots, by label name, the rest
 *  folded into "Other". Colour follows the label, never its rank. */
export const OTHER = "Other";
export function labelSeries(labels: readonly string[]): string[] {
  const sorted = [...labels].sort((a, b) => a.localeCompare(b));
  return sorted.length <= 8 ? sorted : [...sorted.slice(0, 7), OTHER];
}

export function countBy<T extends string | null>(rows: readonly ImageRow[], read: (row: ImageRow) => T): Map<T, number> {
  const counts = new Map<T, number>();
  for (const row of rows) {
    const key = read(row);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return counts;
}

/** Images sharing a duplicate group (R-VQ-09/10): group id -> rows, in table order. */
export function groupsOf(rows: readonly ImageRow[]): Map<number, ImageRow[]> {
  const groups = new Map<number, ImageRow[]>();
  for (const row of rows) {
    if (row.group === null) continue;
    const members = groups.get(row.group);
    if (members) members.push(row);
    else groups.set(row.group, [row]);
  }
  for (const [group, members] of groups) if (members.length < 2) groups.delete(group);
  return groups;
}

// --- filters and sorting -------------------------------------------------------------------------

export type SizeBucket = "small" | "medium" | "large";
/** By the short side: under 256 px, 256 to 1,023 px, 1,024 px and over. */
export function sizeBucket(row: ImageRow): SizeBucket | null {
  if (row.width === null || row.height === null) return null;
  const side = Math.min(row.width, row.height);
  return side < 256 ? "small" : side < 1024 ? "medium" : "large";
}

/** "any" issue, "none", "excluded", "decided" (has a decision), "pinned" (c3), or one rule's code. */
export type QualityFilter = "any" | "none" | "excluded" | "decided" | "pinned" | `R-VQ-${string}`;

export type Range = { measure: Measure; low: number; high: number };

export type Filters = {
  q?: string;
  labels?: readonly string[];
  splits?: readonly string[];
  quality?: QualityFilter;
  size?: SizeBucket;
  range?: Range;
};

export function matches(row: ImageRow, filters: Filters, draft: Draft): boolean {
  if (filters.q) {
    const needle = filters.q.toLowerCase();
    if (!row.key.toLowerCase().includes(needle)) return false;
  }
  if (filters.labels?.length && !filters.labels.includes(row.label ?? "")) return false;
  if (filters.splits?.length && !filters.splits.includes(row.split ?? "")) return false;
  if (filters.size && sizeBucket(row) !== filters.size) return false;
  if (filters.range) {
    const value = row[filters.range.measure];
    if (value === null || value < filters.range.low || value > filters.range.high) return false;
  }
  switch (filters.quality) {
    case undefined:
      return true;
    case "any":
      return row.issues.length > 0;
    case "none":
      return row.issues.length === 0;
    case "excluded":
      return isExcluded(row, draft);
    case "decided":
      return draft[row.key] !== undefined;
    case "pinned":
      return row.pinned;
    default:
      return row.issues.includes(filters.quality);
  }
}

export const SORTS = ["file", "name", "label", "brightness", "contrast", "sharpness", "saturation", "megapixels", "bytes", "aspect"] as const;
export type SortKey = (typeof SORTS)[number];

function sortValue(row: ImageRow, key: SortKey): number | string | null {
  switch (key) {
    case "file":
      return row.i;
    case "name":
      return row.key;
    case "label":
      return row.label;
    case "bytes":
      return row.bytes;
    default:
      return row[key];
  }
}

/** Sorted copy; missing values always last, ties in table order. */
export function sortRows(rows: readonly ImageRow[], key: SortKey, descending = false): ImageRow[] {
  const direction = descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    const x = sortValue(a, key);
    const y = sortValue(b, key);
    if (x === null && y === null) return a.i - b.i;
    if (x === null) return 1;
    if (y === null) return -1;
    const order = typeof x === "string" || typeof y === "string" ? String(x).localeCompare(String(y)) : x - y;
    return order === 0 ? a.i - b.i : order * direction;
  });
}

export function filterAndSort(
  rows: readonly ImageRow[],
  filters: Filters,
  draft: Draft,
  sort: SortKey,
  descending: boolean,
): ImageRow[] {
  const kept = rows.filter((row) => matches(row, filters, draft));
  return sort === "file" && !descending ? kept : sortRows(kept, sort, descending);
}

/** Codes that fired, with how many images each, in code order (quick chips). */
export function issueCounts(rows: readonly ImageRow[]): [string, number][] {
  const counts = new Map<string, number>();
  for (const row of rows) for (const code of row.issues) counts.set(code, (counts.get(code) ?? 0) + 1);
  return [...counts.entries()].sort(([a], [b]) => a.localeCompare(b));
}

// --- selection -----------------------------------------------------------------------------------

/** The selection after a click on position `at` of the shown list (R-107: click, Shift, Ctrl). */
export function clickSelection(
  current: ReadonlySet<string>,
  shown: readonly ImageRow[],
  at: number,
  anchor: number | null,
  modifiers: { shift: boolean; toggle: boolean },
): Set<string> {
  const row = shown[at];
  if (!row) return new Set(current);
  if (modifiers.shift && anchor !== null) {
    const [from, to] = anchor < at ? [anchor, at] : [at, anchor];
    const next = modifiers.toggle ? new Set(current) : new Set<string>();
    for (let k = from; k <= to; k += 1) {
      const key = shown[k]?.key;
      if (key !== undefined) next.add(key);
    }
    return next;
  }
  if (modifiers.toggle) {
    const next = new Set(current);
    if (next.has(row.key)) next.delete(row.key);
    else next.add(row.key);
    return next;
  }
  return new Set([row.key]);
}

/** Positions covered by a rectangle over a regular grid (content coordinates, R-107-05). */
export function positionsInRect(
  rect: { x0: number; y0: number; x1: number; y1: number },
  grid: { columns: number; cellWidth: number; rowHeight: number; gap: number; count: number },
): number[] {
  const strideX = grid.cellWidth + grid.gap;
  const strideY = grid.rowHeight + grid.gap;
  const left = Math.min(rect.x0, rect.x1);
  const right = Math.max(rect.x0, rect.x1);
  const top = Math.min(rect.y0, rect.y1);
  const bottom = Math.max(rect.y0, rect.y1);
  const firstColumn = Math.max(0, Math.floor(left / strideX));
  const lastColumn = Math.min(grid.columns - 1, Math.floor(right / strideX));
  const firstRow = Math.max(0, Math.floor(top / strideY));
  const lastRow = Math.floor(bottom / strideY);
  const out: number[] = [];
  for (let r = firstRow; r <= lastRow; r += 1) {
    for (let c = firstColumn; c <= lastColumn; c += 1) {
      // the gap to the right of / under a cell does not count as the cell
      if (c * strideX + grid.cellWidth < left || r * strideY + grid.rowHeight < top) continue;
      const position = r * grid.columns + c;
      if (position < grid.count) out.push(position);
    }
  }
  return out;
}
