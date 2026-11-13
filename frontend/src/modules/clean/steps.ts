import { t, type MessageKey } from "../../lib/i18n";
import type { Op, Step, StepOf, Suggestion } from "./api";

// One line per step lives in features/ (the Changes page lists a version's steps too, 095).
export { describeStep, opLabel, stepColumns } from "../../features/lineage/steps";

/** The M3 and M5 batches of transformations (study § 3), as the "+ Add step" catalogue shows them. */
export type CatalogGroup = "columns" | "text" | "values" | "dates" | "rows" | "calculate" | "structure";

export const CATALOG: { op: Op; group: CatalogGroup }[] = [
  { op: "rename_column", group: "columns" },
  { op: "drop_columns", group: "columns" },
  { op: "duplicate_column", group: "columns" },
  { op: "assemble_date", group: "columns" },
  { op: "change_type", group: "columns" },
  { op: "split_column", group: "columns" },
  { op: "merge_columns", group: "columns" },
  // N23: '12 kg', '33 cl' read as numbers in grams, litres or metres
  { op: "parse_quantity", group: "columns" },
  // N36: one number per entity (a name and what tells it apart), the names unchanged
  { op: "entity_column", group: "columns" },
  { op: "replace_values", group: "text" },
  { op: "normalize_spaces", group: "text" },
  { op: "remove_invisible", group: "text" },
  { op: "change_case", group: "text" },
  { op: "extract_text", group: "text" },
  { op: "format_text", group: "text" },
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
  // M5 (B1): calculations, and the conditional step (C-45)
  { op: "set_where", group: "calculate" },
  { op: "conditional_column", group: "calculate" },
  { op: "computed_column", group: "calculate" },
  { op: "transform_number", group: "calculate" },
  { op: "swap_values", group: "calculate" },
  // C1: dates, rows and header, structure (C-52 to C-59)
  { op: "shift_date", group: "dates" },
  { op: "convert_timezone", group: "dates" },
  { op: "keep_rows", group: "rows" },
  { op: "promote_header", group: "rows" },
  { op: "split_rows", group: "rows" },
  { op: "unpivot", group: "structure" },
  { op: "pivot", group: "structure" },
  { op: "group_by", group: "structure" },
  { op: "append_rows", group: "structure" },
  { op: "merge_dataset", group: "structure" },
];

/** C-59: steps that reshape the table (its rows no longer come one by one from the original). */
export const RESHAPE_OPS: Op[] = ["split_rows", "unpivot", "pivot", "group_by", "append_rows"];

export const opHint = (op: Op) => t(`clean.opHint.${op}` as MessageKey);

/** Ops reading one column (`column`) or several (`columns`). */
const SINGLE: Op[] = [
  "rename_column",
  "duplicate_column",
  "change_type",
  "replace_values",
  "map_values",
  "fill",
  "fill_empty",
  "null_values",
  "null_outside",
  "pad_zeros",
  "transform_number",
  "split_column",
  "extract_text",
  "format_text",
  "parse_quantity",
  "entity_column",
  "set_where",
  "shift_date",
  "convert_timezone",
  "split_rows",
  "pivot",
];
const MANY: Op[] = ["drop_columns", "normalize_spaces", "remove_invisible", "change_case", "merge_columns", "swap_values"];

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
    case "assemble_date":
      return { ...common, op, year: column, month: column, day: null, hour: null, minute: null, time: null, to: "date" };
    case "drop_columns":
      return { ...common, op, columns: [column] };
    case "change_type":
      return { ...common, op, column, to: "number", decimal: ".", formats: [], excel_serial: false, true_values: [], false_values: [] };
    case "normalize_spaces":
      return { ...common, op, columns: [column], keep_newlines: false };
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
      return { ...common, op, column, length: 5, conditions: [] };
    case "filter_rows":
      return { ...common, op, conditions: [{ column, op: "empty", value: null, values: null }], combine: "all", action: "remove" };
    case "drop_duplicates":
      return { ...common, op, keys: null, keep: "first", order_by: null, groups: null, survivors: null, not_duplicates: null, merge: false };
    case "drop_empty_rows":
      return { ...common, op, share: null };
    case "drop_header_rows":
      return { ...common, op };
    case "label_rows":
      return { ...common, op, to: PIN_COLUMN, labels: {}, other: null };
    // 101: steps of the Expert checks, made by a suggestion (not in the catalogue)
    case "set_cells":
      return { ...common, op, column, cells: {} };
    case "drop_rows":
      return { ...common, op, rows: [] };
    // M5 (B1)
    case "transform_number":
      return { ...common, op, column, how: "round", decimals: 0, factor: 1, min: null, max: null, width: null };
    case "split_column":
      return { ...common, op, column, by: "delimiter", delimiter: ",", positions: [], to: [`${column} 1`, `${column} 2`] };
    case "merge_columns":
      return { ...common, op, columns: [column], separator: " ", to: `${column} merged` };
    case "extract_text":
      return { ...common, op, column, how: "first", count: 3, start: "", end: "", pattern: null, to: null };
    case "format_text":
      return { ...common, op, column, prefix: "", suffix: "", pad_to: 5, pad_with: "0" };
    case "parse_quantity":
      return { ...common, op, column, to_grams: `${column}_g`, to_litres: `${column}_l`, to_metres: null, unit_to: `${column}_unit` };
    case "entity_column":
      return { ...common, op, column, anchor: null, anchor_kind: "key", to: `${column}_entity`, groups: [], not_entities: null, split: null };
    case "set_where":
      return { ...common, op, column, cases: [newCase(column)] };
    case "conditional_column":
      return { ...common, op, to: "group", cases: [newCase(column)], otherwise: null };
    case "computed_column":
      return { ...common, op, to: "computed", formula: quoted(column) };
    case "swap_values":
      return { ...common, op, columns: [column, column], when: `${quoted(column)} > ${quoted(column)}` };
    // C1
    case "shift_date":
      return { ...common, op, column, amount: 1, unit: "hours" };
    case "convert_timezone":
      return { ...common, op, column, from_zone: "UTC", to_zone: "Europe/Paris" };
    case "keep_rows":
      return { ...common, op, how: "first", count: 100 };
    case "promote_header":
      return { ...common, op, row: 1 };
    case "split_rows":
      return { ...common, op, column, delimiter: ",", trim: true };
    case "unpivot":
      return { ...common, op, columns: [column], names_to: "attribute", values_to: "value", drop_empty: false };
    case "pivot":
      return { ...common, op, column, value_column: null, aggregate: "count", headers: [] };
    case "group_by":
      return { ...common, op, keys: [column], aggregations: [{ column: null, fn: "count", to: "rows" }] };
    case "append_rows":
      return { ...common, op, dataset: "", label: null, source_column: null, this_label: null };
    case "merge_dataset":
      return { ...common, op, dataset: "", label: null, keys: [column], other_keys: [column], add: null, how: "left", prefix: "" };
  }
}

export type Case = StepOf<"set_where">["cases"][number];

/** A case of a conditional step (C-45): one condition to start from, the value to give. */
export function newCase(column: string): Case {
  return { conditions: [{ column, op: "eq", value: "", values: null }], combine: "all", when: null, value: "", formula: null };
}

const FORMULA_WORDS = new Set(["and", "or", "not", "is", "null", "true", "false", "case", "when", "then", "else", "end", "between", "in"]);

/** A column name as a formula writes it: bare when it can be, else in double quotes (formula.py). */
export function quoted(name: string): string {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !FORMULA_WORDS.has(name.toLowerCase()) ? name : `"${name.replaceAll('"', '""')}"`;
}

/** 099 (R-PIN-15): the column of the pinned rows' labels. */
export const PIN_COLUMN = "pin_label";

/** The draft with the labels of the pinned rows as a column: the `label_rows` step already there
 *  takes the current labels, else a new one is added at the end (after the other steps, so the
 *  column is not touched by them). */
export function withPinnedLabels(steps: Step[], labels: Record<string, string>, columns: string[]): Step[] {
  const existing = steps.find((step) => step.op === "label_rows");
  if (existing) return replaceStep(steps, { ...existing, labels } as Step);
  let to = PIN_COLUMN;
  for (let n = 2; columns.includes(to); n += 1) to = `${PIN_COLUMN}_${n}`;
  return [...steps, { ...base(nextId(steps)), op: "label_rows", to, labels, other: null }];
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
