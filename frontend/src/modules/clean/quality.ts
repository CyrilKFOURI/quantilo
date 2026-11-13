import { locale } from "../../lib/i18n";
import type { Step, Suggestion, Suggestions } from "./api";

/** N37: a column's quality as measured by the server with the Basic profile. */
export type ColumnQuality = NonNullable<Suggestions["quality"]>[number];

/** The parts of a column's bar, in the order they are drawn. */
export const PARTS = ["valid", "toFix", "missing", "unreadable"] as const;
export type Part = (typeof PARTS)[number];

/** A column's bar: its cells split into four parts that add up to its rows. A cell is counted
 *  once: empty first, then unreadable for its type, then touched by a fix not ticked yet; the rest
 *  is valid. `fixed`: the cells the ticked fixes change (shown in the tip, never a part). */
export type QualityBar = {
  column: string;
  rows: number;
  parts: Record<Part, number>;
  /** Unreadable values not counted (too many distinct values for the profile). */
  uncounted: boolean;
  fixed: number;
  pending: Suggestion[];
  unreadableValues: ColumnQuality["unreadable_values"];
  reading: string | null;
};

/** The suggestions whose step is in the draft, turned on. */
function tickedIds(steps: Step[]): Set<string> {
  return new Set(steps.filter((step) => step.enabled && step.suggestion).map((step) => step.suggestion as string));
}

/** Every column's bar, from the measure and the fixes found, as the draft stands: ticking a fix
 *  moves its cells out of "to fix" (a fix is the column's when it is its first column, as in the
 *  "By column" view). */
export function qualityBars(quality: ColumnQuality[], found: Suggestion[], steps: Step[]): QualityBar[] {
  const ticked = tickedIds(steps);
  const byColumn = new Map<string, { pending: Suggestion[]; fixed: number }>();
  for (const suggestion of found) {
    const column = suggestion.columns[0];
    if (!column || !suggestion.step || suggestion.cells <= 0) continue;
    const entry = byColumn.get(column) ?? { pending: [], fixed: 0 };
    if (ticked.has(suggestion.id)) entry.fixed += suggestion.cells;
    else entry.pending.push(suggestion);
    byColumn.set(column, entry);
  }
  return quality.map((measured) => {
    const rows = Math.max(0, measured.rows);
    const missing = Math.min(rows, measured.missing);
    const unreadable = Math.min(rows - missing, measured.unreadable ?? 0);
    const entry = byColumn.get(measured.column) ?? { pending: [], fixed: 0 };
    const touched = entry.pending.reduce((sum, suggestion) => sum + suggestion.cells, 0);
    const toFix = Math.min(rows - missing - unreadable, touched);
    return {
      column: measured.column,
      rows,
      parts: { valid: rows - missing - unreadable - toFix, toFix, missing, unreadable },
      uncounted: measured.unreadable === null,
      fixed: entry.fixed,
      pending: entry.pending,
      unreadableValues: measured.unreadable_values,
      reading: measured.reading,
    };
  });
}

/** "92 %", in the page's language; a part above 0 never reads 0 % (it says "< 1 %"). */
export function shareOf(count: number, rows: number): string {
  if (rows <= 0) return "";
  const share = count / rows;
  const format = new Intl.NumberFormat(locale(), { style: "percent", maximumFractionDigits: 0 });
  if (count > 0 && share < 0.005) return `< ${format.format(0.01)}`;
  if (count < rows && share > 0.995) return `> ${format.format(0.99)}`;
  return format.format(share);
}

/** Columns past which the overview is folded at first (onion rule, N37). */
export const FOLD_OVER = 30;
