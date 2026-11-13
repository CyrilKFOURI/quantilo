import type { Step, StepOf, Suggestion } from "./api";
import { MERGE_RULES } from "./groups";

/** N38: the review of a merge of spellings, value by value (OpenRefine's cluster dialog): each
 *  group with its target and its values, every one ticked or not; the step written is exactly what
 *  remains ticked. Pure functions: the state is read from the draft's step, never kept apart. */

/** One value of a group: as shown, its rows (null: an older server did not say), and the keys of
 *  the step's map it stands for (raw and normalized spellings, C-23). */
export type ReviewValue = { value: string; rows: number | null; spellings: string[] };
export type ReviewGroup = { target: string; rows: number | null; values: ReviewValue[] };
/** Per group, per value: ticked; per group: the value its ticked members become. */
export type ReviewState = { ticked: boolean[][]; targets: string[] };

type ValueMap = StepOf<"map_values">["map"];

const mapOf = (step: Step | null | undefined): ValueMap | null => (step?.op === "map_values" ? step.map : null);

/** The groups of a merge: the engine's (N38), else, from an older server, its map by target. */
export function reviewGroups(suggestion: Suggestion): ReviewGroup[] {
  const clusters = (suggestion.clusters as Suggestion["clusters"] | undefined) ?? [];
  if (clusters.length) return clusters.map((group) => ({ target: group.target, rows: group.rows, values: group.values.map((value) => ({ value: value.value, rows: value.rows, spellings: value.spellings })) }));
  const by = new Map<string, ReviewValue[]>();
  for (const [key, target] of Object.entries(mapOf(suggestion.step) ?? {})) {
    if (target === null) continue;
    by.set(target, [...(by.get(target) ?? []), { value: key, rows: null, spellings: [key] }]);
  }
  return [...by.entries()].map(([target, values]) => ({ target, rows: null, values }));
}

/** A merge of spellings (A-01 to A-03, or any suggestion that sends its groups) whose step maps values. */
export function isReviewable(suggestion: Suggestion): boolean {
  const clusters = (suggestion.clusters as Suggestion["clusters"] | undefined) ?? [];
  return suggestion.step?.op === "map_values" && (MERGE_RULES.has(suggestion.rule) || clusters.length > 0) && reviewGroups(suggestion).length > 0;
}

/** The review as the draft holds it: a value is ticked when one of its keys is in the step's map
 *  (none when the suggestion is not ticked), its group's target the value it is mapped to. */
export function readReview(groups: ReviewGroup[], step: Step | null | undefined): ReviewState {
  const map = mapOf(step);
  const ticked = groups.map((group) => group.values.map((value) => map !== null && value.spellings.some((key) => key in map)));
  const targets = groups.map((group) => {
    if (map === null) return group.target;
    const key = group.values.flatMap((value) => value.spellings).find((spelling) => typeof map[spelling] === "string");
    const target = key === undefined ? (map[group.target] ?? null) : map[key];
    return typeof target === "string" ? target : group.target;
  });
  return { ticked, targets };
}

/** The map a review writes: each ticked value's keys → its group's target, and the target of the
 *  engine too when the user changed it (the whole group becomes the new value, as in OpenRefine).
 *  An empty target leaves its group out. Empty map: nothing to merge, no step. */
export function writeReview(groups: ReviewGroup[], state: ReviewState): Record<string, string> {
  const map: Record<string, string> = {};
  groups.forEach((group, index) => {
    const target = (state.targets[index] ?? group.target).trim();
    const ticked = state.ticked[index] ?? [];
    if (!target || !ticked.some(Boolean)) return;
    group.values.forEach((value, at) => {
      if (!ticked[at]) return;
      for (const key of value.spellings) if (key !== target) map[key] = target;
    });
    if (target !== group.target) map[group.target] = target;
  });
  return map;
}

/** How much of a review is ticked: groups and values, and the cells it changes (null when an
 *  older server sent no counts). */
export function reviewFacts(groups: ReviewGroup[], state: ReviewState): { groups: number; values: number; total: number; cells: number | null } {
  let cells: number | null = 0;
  let values = 0;
  let total = 0;
  let full = 0;
  groups.forEach((group, index) => {
    const ticked = state.ticked[index] ?? [];
    const any = ticked.some(Boolean);
    if (any) full += 1;
    group.values.forEach((value, at) => {
      total += 1;
      if (!ticked[at]) return;
      values += 1;
      cells = cells === null || value.rows === null ? null : cells + value.rows;
    });
    const target = (state.targets[index] ?? group.target).trim();
    if (any && target && target !== group.target) cells = cells === null || group.rows === null ? null : cells + group.rows;
  });
  return { groups: full, values, total, cells };
}

/** Every value ticked (true), some (mixed), none (false): the merge's own tick. */
export function reviewTick(state: ReviewState): boolean | "mixed" {
  const all = state.ticked.flat();
  const on = all.filter(Boolean).length;
  return on === 0 ? false : on === all.length ? true : "mixed";
}
