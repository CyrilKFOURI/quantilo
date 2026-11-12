import type { Step, Suggestion } from "./api";

/** The groups of the Suggestions tab (096, CL-04): by what a fix does to the dataset, from the
 *  lossless to the judgement calls, which are all gathered in "decision" whatever their effect. */
export type Impact = Suggestion["impact"];
export type GroupId = Impact | "decision";

export const GROUP_ORDER: GroupId[] = ["format", "values", "types", "rows", "columns", "decision"];

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
  fill: "values",
  fill_empty: "values",
  null_values: "values",
  null_outside: "values",
  pad_zeros: "values",
  filter_rows: "rows",
  drop_duplicates: "rows",
  drop_empty_rows: "rows",
  drop_header_rows: "rows",
};

export function impactOf(suggestion: Suggestion): Impact {
  return (suggestion.impact as Impact | undefined) ?? (suggestion.step ? OP_IMPACT[suggestion.step.op] : "values");
}

export function groupOf(suggestion: Suggestion): GroupId {
  const risk = (suggestion.risk as Suggestion["risk"] | undefined) ?? (suggestion.mode === "A" ? "safe" : "review");
  return risk === "review" ? "decision" : impactOf(suggestion);
}

export type Group = { id: GroupId; items: Suggestion[] };

/** The non-empty groups, in their order; each keeps the suggestions' own order (C-06). */
export function groupSuggestions(suggestions: Suggestion[]): Group[] {
  const by = new Map<GroupId, Suggestion[]>();
  for (const suggestion of [...suggestions].sort((a, b) => a.order - b.order)) {
    const id = groupOf(suggestion);
    by.set(id, [...(by.get(id) ?? []), suggestion]);
  }
  return GROUP_ORDER.flatMap((id) => {
    const items = by.get(id);
    return items ? [{ id, items }] : [];
  });
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
