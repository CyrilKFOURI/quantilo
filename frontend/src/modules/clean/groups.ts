import type { Step, Suggestion } from "./api";

/** The groups of the Suggestions tab (096, CL-04): by what a fix does to the dataset, from the
 *  lossless to the judgement calls, which are all gathered in "decision" whatever their effect.
 *  N04: when the reliable fixes are applied by default (N03), they come first, in "auto". */
export type Impact = Suggestion["impact"];
export type GroupId = "auto" | Impact | "decision";

export const GROUP_ORDER: GroupId[] = ["auto", "format", "values", "types", "rows", "columns", "decision"];

/** The effect of a step by its op, when an older server sends no `impact` (mirrors OP_IMPACT of
 *  cleaning/model.py). */
const OP_IMPACT: Record<Step["op"], Impact> = {
  rename_column: "columns",
  drop_columns: "columns",
  duplicate_column: "columns",
  assemble_date: "columns",
  label_rows: "columns",
  change_type: "types",
  normalize_spaces: "format",
  remove_invisible: "format",
  change_case: "format",
  replace_values: "values",
  map_values: "values",
  set_cells: "values",
  fill: "values",
  fill_empty: "values",
  null_values: "values",
  null_outside: "values",
  pad_zeros: "values",
  transform_number: "values",
  split_column: "columns",
  merge_columns: "columns",
  extract_text: "values",
  format_text: "values",
  parse_quantity: "columns",
  set_where: "values",
  conditional_column: "columns",
  computed_column: "columns",
  swap_values: "values",
  filter_rows: "rows",
  drop_duplicates: "rows",
  entity_column: "columns", // N36: an entity number, the names unchanged
  drop_rows: "rows",
  drop_empty_rows: "rows",
  drop_header_rows: "rows",
  // C1
  shift_date: "values",
  convert_timezone: "values",
  keep_rows: "rows",
  promote_header: "columns",
  split_rows: "rows",
  unpivot: "columns",
  pivot: "columns",
  group_by: "rows",
  append_rows: "rows",
  merge_dataset: "columns",
};

export function impactOf(suggestion: Suggestion): Impact {
  return (suggestion.impact as Impact | undefined) ?? (suggestion.step ? OP_IMPACT[suggestion.step.op] : "values");
}

/** `applied`: the mode applies the reliable fixes (N03), so a suggestion marked `auto` (N02) goes
 *  to "Applied automatically" whatever its effect. */
export function groupOf(suggestion: Suggestion, applied = false): GroupId {
  if (applied && suggestion.auto) return "auto";
  const risk = (suggestion.risk as Suggestion["risk"] | undefined) ?? (suggestion.mode === "A" ? "safe" : "review");
  return risk === "review" ? "decision" : impactOf(suggestion);
}

export type Group = { id: GroupId; items: Suggestion[] };

/** The non-empty groups, in their order; each keeps the suggestions' own order (C-06). */
export function groupSuggestions(suggestions: Suggestion[], applied = false): Group[] {
  const by = new Map<GroupId, Suggestion[]>();
  for (const suggestion of [...suggestions].sort((a, b) => a.order - b.order)) {
    const id = groupOf(suggestion, applied);
    by.set(id, [...(by.get(id) ?? []), suggestion]);
  }
  return GROUP_ORDER.flatMap((id) => {
    const items = by.get(id);
    return items ? [{ id, items }] : [];
  });
}

/** C1 (the option of 096, CL-04): the suggestions by the column they touch first ("everything
 *  about `city`"), the columns in the order of their first fix; a fix of the whole table (rows with
 *  no column named) goes in the group `null`, last. Each fix appears once. */
export type ColumnGroup = { column: string | null; items: Suggestion[] };

export function groupByColumn(suggestions: Suggestion[]): ColumnGroup[] {
  const by = new Map<string | null, Suggestion[]>();
  for (const suggestion of [...suggestions].sort((a, b) => a.order - b.order)) {
    const column = suggestion.columns[0] ?? null;
    by.set(column, [...(by.get(column) ?? []), suggestion]);
  }
  const named = [...by.entries()].filter(([column]) => column !== null);
  const whole = by.get(null);
  return [...named.map(([column, items]) => ({ column, items })), ...(whole ? [{ column: null, items: whole }] : [])];
}

/** Suggestions by layer: Basic in the open, Advanced folded, Advanced › More folded inside it. */
export function byLayer(suggestions: Suggestion[]) {
  return {
    basic: suggestions.filter((suggestion) => suggestion.layer === "basic"),
    advanced: suggestions.filter((suggestion) => suggestion.layer === "advanced"),
    expert: suggestions.filter((suggestion) => suggestion.layer === "expert"),
  };
}

/** The ticked suggestions of a list: those whose step is in the draft. */
export function tickedIn(items: Suggestion[], steps: Step[]): number {
  const ticked = new Set(steps.map((step) => step.suggestion));
  return items.filter((item) => ticked.has(item.id)).length;
}

/** N04: the reliable fixes of a list that are in the draft, and every cell they change. */
export function autoFacts(items: Suggestion[], steps: Step[]): { applied: number; cells: number; rows: number } {
  const ticked = new Set(steps.map((step) => step.suggestion));
  return {
    applied: items.filter((item) => ticked.has(item.id)).length,
    cells: items.reduce((sum, item) => sum + item.cells, 0),
    rows: items.reduce((sum, item) => sum + item.rows, 0),
  };
}

// --- N39: repetitive suggestions as one line -----------------------------------------------------

/** A line of a group: one fix, or the same fix on several columns (N39, display only: each fix
 *  keeps its own step, the engine is unchanged). */
export type Line = { kind: "one"; suggestion: Suggestion } | { kind: "batch"; key: string; items: Suggestion[] };

/** The same fix on this many columns or more is shown as one line. */
export const MIN_BATCH = 3;

/** The merges of spellings are reviewed one by one, value by value (N38): never batched. */
export const MERGE_RULES = new Set(["A-01", "A-02", "A-03"]);

const WORD = /[\p{L}\p{N}_]/u;

/** `text` with its last whole mention of `name` replaced by `by`, null when `name` is not in it
 *  as a whole ("city" in "Trim extra spaces in city", not in "velocity"). */
export function replaceColumn(text: string, name: string, by: string): string | null {
  if (!name) return null;
  let at = text.lastIndexOf(name);
  while (at >= 0) {
    const before = at === 0 ? "" : text.charAt(at - 1);
    const after = text.charAt(at + name.length);
    if (!WORD.test(before) && !WORD.test(after)) return text.slice(0, at) + by + text.slice(at + name.length);
    at = at === 0 ? -1 : text.lastIndexOf(name, at - 1);
  }
  return null;
}

/** What makes two suggestions "the same fix on another column": the rule, the op, the catalogue
 *  entry, the rest of the id ("B-09:temp:-200:repeated" → ":-200:repeated") and the engine's title
 *  once its column is masked ("Remove the empty column …" is not "Remove the constant column …").
 *  Null: not batched (no step, several columns, a merge, a title that does not name its column). */
export function batchKey(suggestion: Suggestion): string | null {
  const column = suggestion.columns[0];
  if (!suggestion.step || suggestion.columns.length !== 1 || column === undefined || MERGE_RULES.has(suggestion.rule)) return null;
  const shape = replaceColumn(suggestion.title, column, "\u0000");
  if (shape === null) return null;
  const head = `${suggestion.rule}:${column}`;
  const rest = suggestion.id.startsWith(head) ? suggestion.id.slice(head.length) : suggestion.id;
  return [suggestion.rule, suggestion.step.op, suggestion.catalogue ?? "", suggestion.auto ? "auto" : "", rest, shape].join("|");
}

/** A group's lines: the fixes in their order, those repeated on MIN_BATCH columns or more
 *  gathered at the place of the first one. */
export function linesOf(items: Suggestion[]): Line[] {
  const keys = items.map(batchKey);
  const counts = new Map<string, number>();
  for (const key of keys) if (key !== null) counts.set(key, (counts.get(key) ?? 0) + 1);
  const lines: Line[] = [];
  const batches = new Map<string, Suggestion[]>();
  items.forEach((suggestion, index) => {
    const key = keys[index];
    if (key === null || key === undefined || (counts.get(key) ?? 0) < MIN_BATCH) {
      lines.push({ kind: "one", suggestion });
      return;
    }
    const batch = batches.get(key);
    if (batch) {
      batch.push(suggestion);
      return;
    }
    const fresh = [suggestion];
    batches.set(key, fresh);
    lines.push({ kind: "batch", key, items: fresh });
  });
  return lines;
}

/** Cells and rows a list of fixes changes, summed. */
export function totals(items: Suggestion[]): { cells: number; rows: number } {
  return { cells: items.reduce((sum, item) => sum + item.cells, 0), rows: items.reduce((sum, item) => sum + item.rows, 0) };
}
