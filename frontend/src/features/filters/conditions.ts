import type { AnalysisType } from "../../api/client";
import type { Condition } from "../../api/filters";

/** What the user picks in the editor: an operator plus "is not". */
export type Choice = {
  id: string;
  label: string;
  op: Condition["op"];
  negate: boolean;
  /** Which inputs the choice needs: one value typed ("value") or picked from the data ("one"),
   *  several ticked ("values") or pasted ("paste"), a number of days ("days")... */
  input: "none" | "value" | "range" | "values" | "one" | "paste" | "part" | "rows" | "days";
  /** An operator the column's type no longer offers, kept for a saved condition (R-FLT-07). */
  legacy?: boolean;
};

// Operator sets per type (100, R-FLT-01 to 06): each type offers what makes sense for it.
const NUMBER: Choice[] = [
  { id: "eq", label: "=", op: "eq", negate: false, input: "value" },
  { id: "neq", label: "≠", op: "eq", negate: true, input: "value" },
  { id: "gt", label: ">", op: "gt", negate: false, input: "value" },
  { id: "gte", label: "≥", op: "gte", negate: false, input: "value" },
  { id: "lt", label: "<", op: "lt", negate: false, input: "value" },
  { id: "lte", label: "≤", op: "lte", negate: false, input: "value" },
  { id: "between", label: "between", op: "between", negate: false, input: "range" },
  { id: "outside", label: "outside", op: "between", negate: true, input: "range" },
];
const VALUES: Choice[] = [
  { id: "in", label: "is any of", op: "in", negate: false, input: "values" },
  { id: "not_in", label: "is none of", op: "in", negate: true, input: "values" },
];
/** A category is picked from its values, never typed (R-FLT-03). */
const CATEGORY: Choice[] = [
  { id: "is", label: "is", op: "in", negate: false, input: "one" },
  { id: "is_not", label: "is not", op: "in", negate: true, input: "one" },
  ...VALUES,
];
const TEXT: Choice[] = [
  { id: "contains", label: "contains", op: "contains", negate: false, input: "value" },
  { id: "not_contains", label: "doesn't contain", op: "contains", negate: true, input: "value" },
  { id: "starts_with", label: "starts with", op: "starts_with", negate: false, input: "value" },
];
/** An identifier is looked up, alone or from a pasted list (R-FLT-05). */
const IDENTIFIER: Choice[] = [
  { id: "eq", label: "equals", op: "eq", negate: false, input: "value" },
  { id: "in", label: "is any of (paste a list)", op: "in", negate: false, input: "paste" },
  { id: "not_in", label: "is none of (paste a list)", op: "in", negate: true, input: "paste" },
];
const DATE: Choice[] = [
  { id: "lt", label: "before", op: "lt", negate: false, input: "value" },
  { id: "gt", label: "after", op: "gt", negate: false, input: "value" },
  { id: "between", label: "between", op: "between", negate: false, input: "range" },
  { id: "outside", label: "outside", op: "between", negate: true, input: "range" },
  { id: "last_days", label: "in the last N days", op: "last_days", negate: false, input: "days" },
  { id: "part_in", label: "falls on", op: "part_in", negate: false, input: "part" },
  { id: "part_not_in", label: "doesn't fall on", op: "part_in", negate: true, input: "part" },
];
const MISSING: Choice[] = [
  { id: "is_null", label: "is missing", op: "is_null", negate: false, input: "none" },
  { id: "not_null", label: "is not missing", op: "is_null", negate: true, input: "none" },
];

/** Act on rows as a whole: one row per value, or the repeated values only (any column type). */
const ROWS: Choice[] = [
  { id: "unique", label: "unique values (one row each)", op: "unique", negate: false, input: "rows" },
  { id: "duplicated", label: "duplicates only", op: "duplicated", negate: false, input: "rows" },
  { id: "not_duplicated", label: "values appearing once", op: "duplicated", negate: true, input: "rows" },
];

const CATEGORICAL = new Set<AnalysisType>(["categorical_nominal", "categorical_ordinal", "boolean"]);
const NUMERIC = new Set<AnalysisType>(["numeric_continuous", "numeric_discrete"]);

export function choicesFor(type: AnalysisType): Choice[] {
  if (type === "numeric_continuous") return [...NUMBER, ...MISSING, ...ROWS];
  if (type === "numeric_discrete") return [...VALUES, ...NUMBER, ...MISSING, ...ROWS];
  if (CATEGORICAL.has(type)) return [...CATEGORY, ...MISSING, ...ROWS];
  if (type === "datetime") return [...DATE, ...MISSING, ...ROWS];
  if (type === "text") return [...TEXT, ...MISSING, ...ROWS];
  if (type === "identifier") return [...IDENTIFIER, ...MISSING, ...ROWS];
  return [];
}

/** An older condition rewritten as today's equivalent, same rows (R-FLT-07): "equals" on a
 *  category compares its label, exactly as "is" does. */
export function upgraded(condition: Condition, type: AnalysisType): Condition {
  if (condition.op !== "eq" || !CATEGORICAL.has(type)) return condition;
  if (condition.value === null || condition.value === undefined) return condition;
  const { value, ...rest } = condition;
  return { ...rest, op: "in", values: [String(value)] };
}

/** Words of an older operator, [is, is not]. */
const OLDER: Partial<Record<Condition["op"], [string, string]>> = {
  eq: ["equals", "doesn't equal"],
  gt: ["after", "not after"],
  gte: ["on or after", "before"],
  lt: ["before", "not before"],
  lte: ["on or before", "after"],
  between: ["between", "outside"],
  contains: ["contains", "doesn't contain"],
  starts_with: ["starts with", "doesn't start with"],
  in: ["is any of", "is none of"],
};
const OLDER_SYMBOLS: Partial<Record<Condition["op"], [string, string]>> = {
  gt: [">", "≤"],
  gte: ["≥", "<"],
  lt: ["<", "≥"],
  lte: ["≤", ">"],
};

function olderInput(op: Condition["op"], type: AnalysisType): Choice["input"] {
  if (op === "between") return "range";
  if (op === "in") return type === "identifier" || type === "text" ? "paste" : "values";
  return "value";
}

/** The choice a saved condition uses although its type no longer offers it: shown for that
 *  condition only, so it still opens, reads and applies the same (R-FLT-07). */
function olderChoice(condition: Condition, type: AnalysisType): Choice | undefined {
  const words = (NUMERIC.has(type) ? OLDER_SYMBOLS[condition.op] : undefined) ?? OLDER[condition.op];
  if (!words) return undefined;
  return {
    id: `older_${condition.op}_${condition.negate ? "not" : "is"}`,
    label: `${words[condition.negate ? 1 : 0]} (older condition)`,
    op: condition.op,
    negate: condition.negate,
    input: olderInput(condition.op, type),
    legacy: true,
  };
}

/** The editor's choice for a condition: "is" for a single category, "is any of" for several. */
export function choiceOf(condition: Condition, type: AnalysisType): Choice | undefined {
  const current = upgraded(condition, type);
  const matches = choicesFor(type).filter((choice) => choice.op === current.op && choice.negate === current.negate);
  if (matches.length > 1) {
    const single = (current.values?.length ?? 0) <= 1;
    return matches.find((choice) => (choice.input === "one") === single) ?? matches[0];
  }
  return matches[0] ?? olderChoice(current, type);
}

/** The choices shown for a column: its type's, plus the older one a saved condition uses. */
export function choicesWith(type: AnalysisType, condition: Condition | undefined): Choice[] {
  const choices = choicesFor(type);
  const current = condition ? choiceOf(condition, type) : undefined;
  return current?.legacy ? [current, ...choices] : choices;
}

/** Values pasted from elsewhere: one per line, or separated by commas, semicolons or tabs. */
export function pastedValues(text: string): string[] {
  return [
    ...new Set(
      text
        .split(/[\n\r,;\t]+/)
        .map((value) => value.trim())
        .filter(Boolean),
    ),
  ];
}

/** "2024-12-30" minus `days` days, as an ISO date. */
export function daysBefore(iso: string, days: number): string {
  const moment = new Date(`${iso}T00:00:00Z`);
  moment.setUTCDate(moment.getUTCDate() - days);
  return moment.toISOString().slice(0, 10);
}

/** Date parts offered by "falls on", with their values as the SQL part returns them. */
export const PARTS: Record<string, { label: string; values: { value: string; label: string }[] }> = {
  day_of_week: {
    label: "Day of week",
    values: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((label, i) => ({ value: String(i + 1), label })),
  },
  month: {
    label: "Month",
    values: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"].map((label, i) => ({
      value: String(i + 1),
      label,
    })),
  },
  quarter: { label: "Quarter", values: [1, 2, 3, 4].map((q) => ({ value: String(q), label: `Q${q}` })) },
  hour: { label: "Hour", values: Array.from({ length: 24 }, (_, h) => ({ value: String(h), label: `${h}h` })) },
};

const SYMBOLS: Partial<Record<Condition["op"], [string, string]>> = {
  eq: ["=", "≠"],
  gt: [">", "≤"],
  gte: ["≥", "<"],
  lt: ["<", "≥"],
  lte: ["≤", ">"],
};

function list(values: string[], max = 3): string {
  const shown = values.slice(0, max).join(", ");
  return values.length > max ? `${shown} +${values.length - max}` : shown;
}

function short(value: Condition["value"]): string {
  return typeof value === "string" && /^\d{4}-\d{2}-\d{2}T00:00/.test(value) ? value.slice(0, 10) : String(value ?? "");
}

/** "country ∈ FR, BE", "age ≥ 30", "signup_date on Mon", "vehicle_id · unique (last by date)"... */
export function describe(condition: Condition): string {
  const { column, op, negate } = condition;
  if (op === "unique" || op === "duplicated") {
    const key = [column, ...(condition.columns ?? [])].join(" + ");
    if (op === "duplicated") return `${key} · ${negate ? "appearing once" : "duplicates only"}`;
    const order = condition.order_by ? ` by ${condition.order_by}` : "";
    return `${key} · unique (${condition.keep ?? "first"}${order})`;
  }
  const symbols = SYMBOLS[op];
  if (symbols) return `${column} ${symbols[negate ? 1 : 0]} ${short(condition.value)}`;
  if (op === "between") {
    return `${column} ${negate ? "outside" : "in"} [${short(condition.low)} – ${short(condition.high)}]`;
  }
  if (op === "in") return `${column} ${negate ? "∉" : "∈"} ${list(condition.values ?? [])}`;
  if (op === "contains") return `${column} ${negate ? "doesn't contain" : "contains"} "${condition.value}"`;
  if (op === "starts_with") return `${column} ${negate ? "doesn't start with" : "starts with"} "${condition.value}"`;
  if (op === "is_null") return `${column} ${negate ? "is not missing" : "is missing"}`;
  if (op === "last_days") {
    const days = `last ${String(condition.value ?? "")} days (to ${short(condition.high)})`;
    return `${column} · ${negate ? `not in the ${days}` : days}`;
  }
  const part = PARTS[condition.part ?? ""];
  const labels = (condition.values ?? []).map(
    (value) => part?.values.find((item) => item.value === value)?.label ?? value,
  );
  return `${column} ${negate ? "not on" : "on"} ${list(labels)}`;
}
