import type { Pin, PinLabel } from "./api";

/** Pure helpers of the pinned rows (099). Row numbers are the file's, from 0, as the server
 *  keeps them; the pages show them from 1. */

export const COLORS = [1, 2, 3, 4, 5, 6, 7, 8] as const;
export const MAX_AT_ONCE = 500;

/** The palette slot as a CSS colour: the theme's --chart-N token (R-PIN-03), never a raw hex. */
export const colorVar = (slot: number) => `var(--chart-${slot})`;

/** The first colour no label uses yet, else the next one round the palette. */
export function nextColor(labels: PinLabel[]): number {
  const used = new Set(labels.map((label) => label.color));
  return COLORS.find((slot) => !used.has(slot)) ?? (labels.length % COLORS.length) + 1;
}

/** "12, 40-45" (rows as shown, from 1) → [11, 39, …, 44]; null when a part is not a row number
 *  or the list goes beyond `MAX_AT_ONCE` rows. */
export function parseRows(text: string): number[] | null {
  const found = new Set<number>();
  for (const part of text.split(/[,;\s]+/).filter(Boolean)) {
    const range = /^(\d+)\s*[-–]\s*(\d+)$/.exec(part) ?? /^(\d+)$/.exec(part);
    if (!range) return null;
    const first = Number(range[1]);
    const last = Number(range[2] ?? range[1]);
    if (first < 1 || last < first || last - first >= MAX_AT_ONCE) return null;
    for (let row = first; row <= last; row += 1) found.add(row - 1);
    if (found.size > MAX_AT_ONCE) return null;
  }
  return [...found].sort((a, b) => a - b);
}

/** The pins found, by row number (the rows not found after a re-import are left out). */
export function pinsByRow(pins: Pin[] | undefined): Map<number, Pin> {
  return new Map((pins ?? []).filter((pin) => pin.found).map((pin) => [pin.row, pin]));
}

/** Row number → label of every pin found: what a `label_rows` step carries (R-PIN-15). */
export function labelsOf(pins: Pin[] | undefined): Record<string, string> {
  return Object.fromEntries((pins ?? []).filter((pin) => pin.found).map((pin) => [String(pin.row), pin.label]));
}
