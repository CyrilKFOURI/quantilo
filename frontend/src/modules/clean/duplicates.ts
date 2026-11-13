import type { DuplicateGroup, DuplicateGroups, Step, StepOf, Suggestion } from "./api";

/** N35: the review of a removal of duplicates group by group (Splink / dedupe's review of pairs,
 *  Informatica's survivorship): per group, the row kept (the rule's, or one the user chose), the
 *  group said not duplicates (every row kept), and, only when asked, the kept row's empty cells
 *  filled from its group. The state is read from the draft's step, never kept apart; every write
 *  gives the step back with exactly what the review decided. Pure functions. */

export type DuplicateStep = StepOf<"drop_duplicates">;

/** The rules whose fix removes duplicates: exact copies, a key repeated, near and likely ones. */
export const DUPLICATE_RULES = new Set(["B-14", "B-23", "A-15", "A-16"]);

export const duplicateOf = (step: Step | null | undefined): DuplicateStep | null => (step?.op === "drop_duplicates" ? step : null);

/** A fix that removes duplicates, reviewed by group. */
export function isDuplicateFix(suggestion: Suggestion): boolean {
  return DUPLICATE_RULES.has(suggestion.rule) && duplicateOf(suggestion.step) !== null;
}

/** The step as the rule wrote it (the review cleared): the groups it makes do not move while the
 *  user reviews them. */
export function ruleStep(step: DuplicateStep): DuplicateStep {
  return { ...step, survivors: null, not_duplicates: null, merge: false };
}

/** The draft for the groups: with this fix (turned on), the review cleared, and nothing after it. */
export function stepsForGroups(steps: Step[], stepId: string): Step[] {
  const at = steps.findIndex((step) => step.id === stepId);
  return steps.slice(0, at + 1).map((step) => {
    const dup = step.id === stepId ? duplicateOf(step) : null;
    return dup ? ({ ...ruleStep(dup), enabled: true } as Step) : step;
  });
}

const rowsOf = (group: DuplicateGroup) => group.rows.map((row) => row.row);
const inGroup = (group: DuplicateGroup) => {
  const rows = new Set(rowsOf(group));
  return (row: number) => rows.has(row);
};

/** The row the step keeps in this group: the one the user chose, else the rule's. */
export function survivorOf(group: DuplicateGroup, step: DuplicateStep | null): number {
  const chosen = (step?.survivors ?? []).find(inGroup(group));
  return chosen ?? group.order[0] ?? group.first;
}

/** The group is said not duplicates: every row of it stays. */
export function isApart(group: DuplicateGroup, step: DuplicateStep | null): boolean {
  const has = inGroup(group);
  return (step?.not_duplicates ?? []).some((rows) => rows.some(has));
}

/** How the kept row is chosen in a group: by the step's rule, the first, the most complete, the
 *  latest by a date column, or a row picked by hand. */
export type Choice = "rule" | "first" | "most_complete" | `latest:${string}` | "pick";

/** The row each choice keeps (`pick` has none: the user's row). */
export function rowOfChoice(group: DuplicateGroup, choice: Choice): number | null {
  if (choice === "rule") return group.order[0] ?? group.first;
  if (choice === "first") return group.first;
  if (choice === "most_complete") return group.most_complete;
  if (choice.startsWith("latest:")) return group.latest[choice.slice("latest:".length)] ?? null;
  return null;
}

/** The choice that names the kept row: the rule when it is its row, else the first that keeps it,
 *  else a row picked by hand. */
export function choiceOf(group: DuplicateGroup, survivor: number, dates: string[]): Choice {
  const choices: Choice[] = ["rule", "first", "most_complete", ...dates.map((date) => `latest:${date}` as const)];
  return choices.find((choice) => rowOfChoice(group, choice) === survivor) ?? "pick";
}

/** The step with `row` kept in this group (the rule's own row: no choice written). */
export function withSurvivor(step: DuplicateStep, group: DuplicateGroup, row: number): DuplicateStep {
  const has = inGroup(group);
  const survivors = (step.survivors ?? []).filter((other) => !has(other));
  if (row !== (group.order[0] ?? group.first)) survivors.push(row);
  return { ...step, survivors: survivors.length ? survivors.sort((a, b) => a - b) : null };
}

/** The step with this group said not duplicates (`apart`: its rows all kept) or duplicates again. */
export function withApart(step: DuplicateStep, group: DuplicateGroup, apart: boolean): DuplicateStep {
  const has = inGroup(group);
  const groups = (step.not_duplicates ?? []).filter((rows) => !rows.some(has));
  if (apart) groups.push(rowsOf(group));
  return { ...step, not_duplicates: groups.length ? groups.sort((a, b) => (a[0] ?? 0) - (b[0] ?? 0)) : null };
}

/** The step with every shown group set to `apart` (Tick all / Untick all of the shown groups). */
export function withAllApart(step: DuplicateStep, groups: DuplicateGroup[], apart: boolean): DuplicateStep {
  return groups.reduce((next, group) => withApart(next, group, apart), step);
}

/** The rows the step removes now: the rule's, less those of the groups said not duplicates (a
 *  group shown counts all its rows, `size`; one not shown, the rows the step lists). */
export function removedNow(found: Pick<DuplicateGroups, "removed" | "groups">, step: DuplicateStep | null): number {
  const kept = (step?.not_duplicates ?? []).reduce((sum, rows) => {
    const group = found.groups.find((one) => rows.some(inGroup(one)));
    return sum + Math.max(0, (group?.size ?? rows.length) - 1);
  }, 0);
  return Math.max(0, found.removed - kept);
}

/** The review changed something: rows chosen, groups kept whole, or cells merged. */
export function isReviewed(step: DuplicateStep | null): boolean {
  return Boolean(step && (step.survivors?.length || step.not_duplicates?.length || step.merge));
}

const blank = (cell: string | null) => cell === null || cell.trim() === "";

/** With merge on: the kept row's empty cells, each with the value it takes and the row it comes
 *  from (the first other row with one, in the rule's order), by column index. */
export function mergedCells(group: DuplicateGroup, survivor: number): Map<number, { value: string; from: number }> {
  const filled = new Map<number, { value: string; from: number }>();
  const kept = group.rows.find((row) => row.row === survivor);
  if (!kept) return filled;
  const others = group.order.filter((row) => row !== survivor).map((number) => group.rows.find((row) => row.row === number));
  kept.cells.forEach((cell, at) => {
    if (!blank(cell)) return;
    for (const other of others) {
      const value = other?.cells[at] ?? null;
      if (other && value !== null && !blank(value)) {
        filled.set(at, { value, from: other.row });
        return;
      }
    }
  });
  return filled;
}
