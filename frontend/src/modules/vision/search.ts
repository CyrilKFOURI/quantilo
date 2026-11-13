import { COLOUR_BY, type ColourBy } from "./mapModel";
import { MEASURES, SORTS, type Filters, type Measure, type QualityFilter, type Range, type SizeBucket, type SortKey } from "./model";
import { STEP_IDS, type StepId } from "./prepModel";

export const TABS = ["gallery", "quality", "map", "prepare", "model", "results", "explain"] as const;
export type VisionTab = (typeof TABS)[number];
/** The tabs that work today (107 to 110, 113, 117). */
export const READY_TABS: ReadonlySet<VisionTab> = new Set(["gallery", "quality", "map", "prepare", "model", "results", "explain"]);

export type Density = "s" | "m" | "l";

/** The page's URL (R-107-11): tab, filters, sort, density, the open image. Defaults stay out. */
export type VisionSearch = {
  tab?: VisionTab;
  q?: string;
  labels?: string[];
  splits?: string[];
  quality?: QualityFilter;
  size?: SizeBucket;
  range?: string; // "brightness:0.2:0.25"
  sort?: SortKey;
  desc?: boolean;
  density?: Density;
  image?: string; // rel_path of the image whose sheet is open
  stats?: boolean; // false: the Dataset stats are folded
  // Map (109)
  color?: ColourBy;
  method?: string; // "dinov2_s14"... (classic by default)
  proj?: "tsne" | "pacmap" | "trimap"; // c3, R-VMAP-12: PaCMAP and TriMAP from Advanced › More
  groups?: "hdbscan";
  k?: number;
  // Prepare (110)
  step?: StepId;
  run?: string; // 113: the run shown in Results (default: the latest)
  model?: string; // 113: the model shown (default: the run's default)
  threshold?: number; // 113: a threshold moved by the user (R-VTH-03)
  job?: string; // 113: the training job followed (kept across a reload or a remount)
  // 117: the image whose "Why?" is open (it explains the run and model above)
  why?: string;
};

const text = (value: unknown, max = 400) =>
  typeof value === "string" && value.length > 0 && value.length <= max ? value : undefined;

const list = (value: unknown) => {
  const items = Array.isArray(value) ? value : typeof value === "string" ? [value] : [];
  const kept = items.filter((item): item is string => typeof item === "string" && item.length <= 200).slice(0, 50);
  return kept.length > 0 ? kept : undefined;
};

export function parseRange(raw: string | undefined): Range | undefined {
  if (!raw) return undefined;
  const [measure, low, high] = raw.split(":");
  const a = Number(low);
  const b = Number(high);
  if (!MEASURES.includes(measure as Measure) || !Number.isFinite(a) || !Number.isFinite(b)) return undefined;
  return { measure: measure as Measure, low: Math.min(a, b), high: Math.max(a, b) };
}

export function formatRange(range: Range): string {
  const round = (value: number) => Number(value.toPrecision(4));
  return `${range.measure}:${round(range.low)}:${round(range.high)}`;
}

const QUALITY = /^(any|none|excluded|decided|pinned|R-VQ-\d{2})$/;

export function validateVisionSearch(search: Record<string, unknown>): VisionSearch {
  const tab = TABS.find((value) => value === search.tab);
  const sort = SORTS.find((value) => value === search.sort);
  return {
    tab: tab && tab !== "gallery" ? tab : undefined,
    q: text(search.q, 200),
    labels: list(search.labels),
    splits: list(search.splits),
    quality: typeof search.quality === "string" && QUALITY.test(search.quality) ? (search.quality as QualityFilter) : undefined,
    size: search.size === "small" || search.size === "medium" || search.size === "large" ? search.size : undefined,
    range: parseRange(text(search.range, 80)) ? text(search.range, 80) : undefined,
    sort: sort && sort !== "file" ? sort : undefined,
    desc: search.desc === true ? true : undefined,
    density: search.density === "s" || search.density === "l" ? search.density : undefined,
    image: text(search.image, 1000),
    stats: search.stats === false ? false : undefined,
    color: COLOUR_BY.find((value) => value === search.color),
    method: typeof search.method === "string" && /^[a-z0-9_]{1,32}$/.test(search.method) && search.method !== "classic" ? search.method : undefined,
    proj: search.proj === "tsne" || search.proj === "pacmap" || search.proj === "trimap" ? search.proj : undefined,
    groups: search.groups === "hdbscan" ? "hdbscan" : undefined,
    k: typeof search.k === "number" && Number.isInteger(search.k) && search.k >= 2 && search.k <= 12 ? search.k : undefined,
    step: STEP_IDS.find((value) => value === search.step && value !== "resize"),
    run: typeof search.run === "string" && /^[0-9a-f]{12}$/.test(search.run) ? search.run : undefined,
    model: typeof search.model === "string" && /^[a-z][a-z0-9_]{0,31}$/.test(search.model) ? search.model : undefined,
    threshold: typeof search.threshold === "number" && Number.isFinite(search.threshold) ? search.threshold : undefined,
    job: typeof search.job === "string" && /^[0-9a-f]{12}$/.test(search.job) ? search.job : undefined,
    why: text(search.why, 1000),
  };
}

export function filtersOf(search: VisionSearch): Filters {
  return {
    q: search.q,
    labels: search.labels,
    splits: search.splits,
    quality: search.quality,
    size: search.size,
    range: parseRange(search.range),
  };
}

export function activeFilterCount(search: VisionSearch): number {
  return [search.q, search.labels?.length, search.splits?.length, search.quality, search.size, search.range].filter(Boolean).length;
}

/** The URL patch that clears every filter (tab, sort, density and the open image stay). */
export const CLEARED: VisionSearch = {
  q: undefined,
  labels: undefined,
  splits: undefined,
  quality: undefined,
  size: undefined,
  range: undefined,
};
