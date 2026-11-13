import type { ImageRow, Measure } from "./model";

/** 110: the preparation recipe (quantilo/vision/transforms.py, `Recipe`), settings only. */
export type Recipe = {
  kind?: "vision-prepare";
  version?: number;
  task: "classification" | "anomaly" | "clusters";
  size: number;
  fit: "crop" | "pad";
  trim: boolean;
  white_balance: boolean;
  denoise: "off" | "gaussian" | "median";
  sigma: number;
  equalise: "off" | "global" | "clahe";
  clahe_clip: number;
  clahe_tiles: number;
  sharpen: boolean;
  sharpen_radius: number;
  sharpen_amount: number;
  augment: boolean;
  flip: boolean;
  rotate: number;
  jitter: number;
  crop_scale: number;
};

export const STEP_IDS = ["decode", "trim", "resize", "white_balance", "denoise", "equalise", "sharpen", "augment", "normalise"] as const;
export type StepId = (typeof STEP_IDS)[number];

export type StepInfo = {
  id: StepId;
  /** First plan, or folded under Advanced (the onion rule). */
  level: "first" | "advanced";
  /** Always applied (no switch). */
  always: boolean;
  /** The measure whose extremes the previews show (R-VPR-05); "aspect" for the size step. */
  measure: Measure | "aspect" | null;
  /** The step changes the geometry: no pixel difference (R-VPR-08). */
  geometric: boolean;
  preview: boolean;
};

/** R-VPR-01: the fixed, reasoned order. */
export const STEPS: readonly StepInfo[] = [
  { id: "decode", level: "first", always: true, measure: null, geometric: false, preview: true },
  { id: "trim", level: "advanced", always: false, measure: "aspect", geometric: true, preview: true },
  { id: "resize", level: "first", always: true, measure: "aspect", geometric: true, preview: true },
  { id: "white_balance", level: "advanced", always: false, measure: "colourfulness", geometric: false, preview: true },
  { id: "denoise", level: "advanced", always: false, measure: "sharpness", geometric: false, preview: true },
  { id: "equalise", level: "advanced", always: false, measure: "contrast", geometric: false, preview: true },
  { id: "sharpen", level: "advanced", always: false, measure: "sharpness", geometric: false, preview: true },
  { id: "augment", level: "advanced", always: false, measure: null, geometric: true, preview: true },
  { id: "normalise", level: "first", always: true, measure: null, geometric: false, preview: false },
];

export const stepInfo = (id: StepId): StepInfo => STEPS.find((step) => step.id === id) ?? (STEPS[0] as StepInfo);

export function isOn(recipe: Recipe, id: StepId): boolean {
  switch (id) {
    case "trim":
      return recipe.trim;
    case "white_balance":
      return recipe.white_balance;
    case "denoise":
      return recipe.denoise !== "off";
    case "equalise":
      return recipe.equalise !== "off";
    case "sharpen":
      return recipe.sharpen;
    case "augment":
      return recipe.augment && (recipe.flip || recipe.rotate > 0 || recipe.jitter > 0 || recipe.crop_scale < 1);
    default:
      return true;
  }
}

/** The recipe with a step switched on (its usual setting) or off. */
export function toggled(recipe: Recipe, id: StepId, on: boolean): Recipe {
  switch (id) {
    case "trim":
      return { ...recipe, trim: on };
    case "white_balance":
      return { ...recipe, white_balance: on };
    case "denoise":
      return { ...recipe, denoise: on ? (recipe.denoise === "off" ? "gaussian" : recipe.denoise) : "off" };
    case "equalise":
      return { ...recipe, equalise: on ? (recipe.equalise === "off" ? "clahe" : recipe.equalise) : "off" };
    case "sharpen":
      return { ...recipe, sharpen: on };
    case "augment":
      return on && !(recipe.flip || recipe.rotate > 0 || recipe.jitter > 0 || recipe.crop_scale < 1)
        ? { ...recipe, augment: true, flip: true, crop_scale: 0.8 }
        : { ...recipe, augment: on };
    default:
      return recipe;
  }
}

/** How many optional steps are on. */
export function stepsOn(recipe: Recipe): number {
  return STEPS.filter((step) => !step.always && isOn(recipe, step.id)).length;
}

export function sameRecipe(a: Recipe | null | undefined, b: Recipe | null | undefined): boolean {
  if (!a || !b) return a === b;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  return [...keys].every((key) => (a as Record<string, unknown>)[key] === (b as Record<string, unknown>)[key]);
}

// --- the preview images (R-VPR-05) --------------------------------------------------------------

export type Sample = { row: ImageRow; why: "typical" | "lowest" | "highest" | "random"; seed: number };

/** A small seeded generator (mulberry32): the same Shuffle gives the same images. */
export function random(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const TYPICAL: readonly Measure[] = ["brightness", "contrast", "sharpness", "saturation", "colourfulness"];

function valueOf(row: ImageRow, measure: Measure | "aspect"): number | null {
  if (measure === "aspect") return row.aspect === null ? null : Math.abs(Math.log(row.aspect));
  const value = row[measure];
  if (value === null) return null;
  return measure === "sharpness" ? Math.log10(1 + Math.max(0, value)) : value;
}

/** 6 images: 2 typical (nearest the median of the standardised measures), 2 extremes of the
 *  step's measure (lowest, highest; for the size step the most elongated), 2 at random (seed).
 *  The augmentation: one typical image, 6 draws. */
export function pickSamples(rows: readonly ImageRow[], step: StepInfo, seed: number): Sample[] {
  const pool = rows.filter((row) => row.decodeOk && row.id);
  if (pool.length === 0) return [];
  // standardised measures
  const stats = TYPICAL.map((measure) => {
    const values = pool.map((row) => valueOf(row, measure)).filter((v): v is number => v !== null).sort((a, b) => a - b);
    const median = values[Math.floor(values.length / 2)] ?? 0;
    const mean = values.reduce((s, v) => s + v, 0) / Math.max(1, values.length);
    const sd = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, values.length)) || 1;
    return { measure, median, sd };
  });
  const distance = (row: ImageRow) =>
    stats.reduce((sum, { measure, median, sd }) => {
      const v = valueOf(row, measure);
      return sum + (v === null ? 4 : ((v - median) / sd) ** 2);
    }, 0);
  const typical = [...pool].sort((a, b) => distance(a) - distance(b) || a.i - b.i);
  const first = typical[0] as ImageRow;
  if (step.id === "augment") return Array.from({ length: 6 }, (_, k) => ({ row: first, why: "random" as const, seed: seed * 6 + k }));

  const chosen: Sample[] = [];
  const taken = new Set<number>();
  const take = (row: ImageRow | undefined, why: Sample["why"]) => {
    if (!row || taken.has(row.i) || chosen.length >= 6) return;
    taken.add(row.i);
    chosen.push({ row, why, seed: 0 });
  };
  take(typical[0], "typical");
  take(typical[1], "typical");
  if (step.measure) {
    const measured = pool.filter((row) => valueOf(row, step.measure as Measure | "aspect") !== null);
    const sorted = [...measured].sort((a, b) => (valueOf(a, step.measure!) ?? 0) - (valueOf(b, step.measure!) ?? 0) || a.i - b.i);
    const low = sorted[0] ? valueOf(sorted[0], step.measure) : null;
    const high = sorted.at(-1) ? valueOf(sorted.at(-1) as ImageRow, step.measure) : null;
    if (low === null || high === null || high - low <= 1e-9) {
      // every image has the same value (Beans: all 500 x 500): no extreme to show, random ones instead
    } else if (step.measure === "aspect") {
      take(sorted.at(-1), "highest");
      take(sorted.at(-2), "highest");
    } else {
      take(sorted[0], "lowest");
      take(sorted.at(-1), "highest");
    }
  }
  const next = random(seed + 1);
  for (let attempt = 0; chosen.length < Math.min(6, pool.length) && attempt < 200; attempt += 1) {
    take(pool[Math.floor(next() * pool.length)], "random");
  }
  return chosen;
}

// --- captions ------------------------------------------------------------------------------------

export const SHOWN_MEASURES: readonly Measure[] = ["brightness", "contrast", "sharpness"];

/** The change of a measure, with its sign ("+0.12", "−0.03", "0"); null when not comparable. */
export function delta(before: number | undefined, after: number | undefined, digits = 2): string | null {
  if (before === undefined || after === undefined || !Number.isFinite(before) || !Number.isFinite(after)) return null;
  const change = after - before;
  const text = Math.abs(change).toFixed(digits);
  if (Number(text) === 0) return "0";
  return `${change > 0 ? "+" : "−"}${text}`;
}

/** A measure in its own units: the lumas and contrast as 0-1, sharpness as an integer. */
export function measureText(measure: Measure, value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  return measure === "sharpness" || measure === "colourfulness" ? Math.round(value).toLocaleString("en-US") : value.toFixed(2);
}
