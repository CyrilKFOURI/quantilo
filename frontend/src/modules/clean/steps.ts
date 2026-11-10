import { t, type MessageKey } from "../../lib/i18n";
import type { Op, Step, Suggestion } from "./api";

// One line per step lives in features/ (the Changes page lists a version's steps too, 095).
export { describeStep, opLabel, stepColumns } from "../../features/lineage/steps";

/** The M3 batch of transformations (study § 3), as the "+ Add step" catalogue shows them. */
export type CatalogGroup = "columns" | "text" | "values" | "rows";

export const CATALOG: { op: Op; group: CatalogGroup }[] = [
  { op: "rename_column", group: "columns" },
  { op: "drop_columns", group: "columns" },
  { op: "duplicate_column", group: "columns" },
  { op: "change_type", group: "columns" },
  { op: "replace_values", group: "text" },
  { op: "normalize_spaces", group: "text" },
  { op: "remove_invisible", group: "text" },
  { op: "change_case", group: "text" },
  { op: "map_values", group: "values" },
  { op: "fill", group: "values" },
  { op: "fill_empty", group: "values" },
  { op: "null_values", group: "values" },
  { op: "null_outside", group: "values" },
  { op: "pad_zeros", group: "values" },
  { op: "filter_rows", group: "rows" },
  { op: "drop_duplicates", group: "rows" },
  { op: "drop_empty_rows", group: "rows" },
  { op: "drop_header_rows", group: "rows" },
];

export const opHint = (op: Op) => t(`clean.opHint.${op}` as MessageKey);

/** Ops reading one column (`column`) or several (`columns`). */
const SINGLE: Op[] = ["rename_column", "duplicate_column", "change_type", "replace_values", "map_values", "fill", "fill_empty", "null_values", "null_outside", "pad_zeros"];
const MANY: Op[] = ["drop_columns", "normalize_spaces", "remove_invisible", "change_case"];

export function nextId(steps: Step[]): string {
  const used = steps.map((step) => Number(step.id.slice(1))).filter(Number.isFinite);
  return `s${Math.max(0, ...used) + 1}`;
}

const base = (id: string) => ({ id, enabled: true, origin: "manual" as const, rule: null, suggestion: null });

/** A new step of `op` on `column` (the first column when none is chosen), ready to edit. */
export function newStep(op: Op, id: string, column: string): Step {
  const common = base(id);
  switch (op) {
    case "rename_column":
      return { ...common, op, column, to: `${column} 2` };
    case "duplicate_column":
      return { ...common, op, column, to: `${column} copy` };
    case "drop_columns":
      return { ...common, op, columns: [column] };
    case "change_type":
      return { ...common, op, column, to: "number", decimal: ".", formats: [], excel_serial: false, true_values: [], false_values: [] };
    case "normalize_spaces":
    case "remove_invisible":
      return { ...common, op, columns: [column] };
    case "change_case":
      return { ...common, op, columns: [column], case: "title" };
    case "replace_values":
      return { ...common, op, column, find: "?", replace: "", match: "exact", case_sensitive: true };
    case "map_values":
      return { ...common, op, column, map: { "": "" } };
    case "fill":
      return { ...common, op, column, direction: "down" };
    case "fill_empty":
      return { ...common, op, column, value: "" };
    case "null_values":
      return { ...common, op, column, values: ["?"], match: "token" };
    case "null_outside":
      return { ...common, op, column, min: 0, max: null };
    case "pad_zeros":
      return { ...common, op, column, length: 5 };
    case "filter_rows":
      return { ...common, op, conditions: [{ column, op: "empty", value: null, values: null }], combine: "all", action: "remove" };
    case "drop_duplicates":
      return { ...common, op, keys: null, keep: "first" };
    case "drop_empty_rows":
      return { ...common, op };
    case "drop_header_rows":
      return { ...common, op };
  }
}

export function isSingle(op: Op) {
  return SINGLE.includes(op);
}

export function isMany(op: Op) {
  return MANY.includes(op);
}

/** The draft with a suggestion's step at its place (C-06): before the first suggested step that
 *  comes after it; manual steps keep their place. Its id changes if another step has it. */
export function withSuggestion(steps: Step[], suggestion: Suggestion, all: Suggestion[]): Step[] {
  if (!suggestion.step || steps.some((step) => step.suggestion === suggestion.id)) return steps;
  const step = { ...suggestion.step } as Step;
  if (steps.some((other) => other.id === step.id)) step.id = nextId(steps);
  const order = new Map(all.map((item) => [item.id, item.order]));
  const at = steps.findIndex((other) => (order.get(other.suggestion ?? "") ?? -1) > suggestion.order);
  return at < 0 ? [...steps, step] : [...steps.slice(0, at), step, ...steps.slice(at)];
}

export function withoutSuggestion(steps: Step[], suggestionId: string): Step[] {
  return steps.filter((step) => step.suggestion !== suggestionId);
}

export function move(steps: Step[], id: string, by: -1 | 1): Step[] {
  const index = steps.findIndex((step) => step.id === id);
  const target = index + by;
  if (index < 0 || target < 0 || target >= steps.length) return steps;
  const next = [...steps];
  [next[index], next[target]] = [next[target]!, next[index]!];
  return next;
}

export function replaceStep(steps: Step[], step: Step): Step[] {
  return steps.map((other) => (other.id === step.id ? step : other));
}

// --- Undo (C-11) ---------------------------------------------------------------------------------

export type History = { past: Step[][]; present: Step[]; future: Step[][] };
const LIMIT = 100;

export const historyOf = (present: Step[]): History => ({ past: [], present, future: [] });

export function push(history: History, next: Step[]): History {
  if (next === history.present) return history;
  return { past: [...history.past, history.present].slice(-LIMIT), present: next, future: [] };
}

export function undo(history: History): History {
  const previous = history.past.at(-1);
  if (!previous) return history;
  return { past: history.past.slice(0, -1), present: previous, future: [history.present, ...history.future] };
}

export function redo(history: History): History {
  const [next, ...rest] = history.future;
  if (!next) return history;
  return { past: [...history.past, history.present], present: next, future: rest };
}
