import type { AnalysisType } from "../../api/client";
import type { Condition } from "../../api/filters";
import { t, type MessageKey } from "../../lib/i18n";

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
// Built on each call so the words follow the chosen language (127).
const NUMBER = (): Choice[] => [
  { id: "eq", label: "=", op: "eq", negate: false, input: "value" },
  { id: "neq", label: "≠", op: "eq", negate: true, input: "value" },
  { id: "gt", label: ">", op: "gt", negate: false, input: "value" },
  { id: "gte", label: "≥", op: "gte", negate: false, input: "value" },
  { id: "lt", label: "<", op: "lt", negate: false, input: "value" },
  { id: "lte", label: "≤", op: "lte", negate: false, input: "value" },
  { id: "between", label: t("fig.filter.between"), op: "between", negate: false, input: "range" },
  { id: "outside", label: t("fig.filter.outside"), op: "between", negate: true, input: "range" },
];
const VALUES = (): Choice[] => [
  { id: "in", label: t("fig.filter.isAnyOf"), op: "in", negate: false, input: "values" },
  { id: "not_in", label: t("fig.filter.isNoneOf"), op: "in", negate: true, input: "values" },
];
/** A category is picked from its values, never typed (R-FLT-03). */
const CATEGORY = (): Choice[] => [
  { id: "is", label: t("fig.filter.is"), op: "in", negate: false, input: "one" },
  { id: "is_not", label: t("fig.filter.isNot"), op: "in", negate: true, input: "one" },
  ...VALUES(),
];
const TEXT = (): Choice[] => [
  { id: "contains", label: t("fig.filter.contains"), op: "contains", negate: false, input: "value" },
  { id: "not_contains", label: t("fig.filter.notContains"), op: "contains", negate: true, input: "value" },
  { id: "starts_with", label: t("fig.filter.startsWith"), op: "starts_with", negate: false, input: "value" },
];
/** An identifier is looked up, alone or from a pasted list (R-FLT-05). */
const IDENTIFIER = (): Choice[] => [
  { id: "eq", label: t("fig.filter.equals"), op: "eq", negate: false, input: "value" },
  { id: "in", label: t("fig.filter.isAnyOfPaste"), op: "in", negate: false, input: "paste" },
  { id: "not_in", label: t("fig.filter.isNoneOfPaste"), op: "in", negate: true, input: "paste" },
];
const DATE = (): Choice[] => [
  { id: "lt", label: t("fig.filter.before"), op: "lt", negate: false, input: "value" },
  { id: "gt", label: t("fig.filter.after"), op: "gt", negate: false, input: "value" },
  { id: "between", label: t("fig.filter.between"), op: "between", negate: false, input: "range" },
  { id: "outside", label: t("fig.filter.outside"), op: "between", negate: true, input: "range" },
  { id: "last_days", label: t("fig.filter.lastDays"), op: "last_days", negate: false, input: "days" },
  { id: "part_in", label: t("fig.filter.fallsOn"), op: "part_in", negate: false, input: "part" },
  { id: "part_not_in", label: t("fig.filter.notFallsOn"), op: "part_in", negate: true, input: "part" },
];
const MISSING = (): Choice[] => [
  { id: "is_null", label: t("fig.filter.isMissing"), op: "is_null", negate: false, input: "none" },
  { id: "not_null", label: t("fig.filter.isNotMissing"), op: "is_null", negate: true, input: "none" },
];

/** Act on rows as a whole: one row per value, or the repeated values only (any column type). */
const ROWS = (): Choice[] => [
  { id: "unique", label: t("fig.filter.unique"), op: "unique", negate: false, input: "rows" },
  { id: "duplicated", label: t("fig.filter.duplicates"), op: "duplicated", negate: false, input: "rows" },
  { id: "not_duplicated", label: t("fig.filter.once"), op: "duplicated", negate: true, input: "rows" },
];

const CATEGORICAL = new Set<AnalysisType>(["categorical_nominal", "categorical_ordinal", "boolean"]);
const NUMERIC = new Set<AnalysisType>(["numeric_continuous", "numeric_discrete"]);

export function choicesFor(type: AnalysisType): Choice[] {
  if (type === "numeric_continuous") return [...NUMBER(), ...MISSING(), ...ROWS()];
  if (type === "numeric_discrete") return [...VALUES(), ...NUMBER(), ...MISSING(), ...ROWS()];
  if (CATEGORICAL.has(type)) return [...CATEGORY(), ...MISSING(), ...ROWS()];
  if (type === "datetime") return [...DATE(), ...MISSING(), ...ROWS()];
  if (type === "text") return [...TEXT(), ...MISSING(), ...ROWS()];
  if (type === "identifier") return [...IDENTIFIER(), ...MISSING(), ...ROWS()];
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
const OLDER: Partial<Record<Condition["op"], [MessageKey, MessageKey]>> = {
  eq: ["fig.filter.equals", "fig.filter.notEquals"],
  gt: ["fig.filter.after", "fig.filter.notAfter"],
  gte: ["fig.filter.onOrAfter", "fig.filter.before"],
  lt: ["fig.filter.before", "fig.filter.notBefore"],
  lte: ["fig.filter.onOrBefore", "fig.filter.after"],
  between: ["fig.filter.between", "fig.filter.outside"],
  contains: ["fig.filter.contains", "fig.filter.notContains"],
  starts_with: ["fig.filter.startsWith", "fig.filter.notStartsWith"],
  in: ["fig.filter.isAnyOf", "fig.filter.isNoneOf"],
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
  const words = (NUMERIC.has(type) ? OLDER_SYMBOLS[condition.op] : undefined) ?? OLDER[condition.op]?.map((key) => t(key));
  if (!words) return undefined;
  return {
    id: `older_${condition.op}_${condition.negate ? "not" : "is"}`,
    label: t("fig.filter.older", { words: words[condition.negate ? 1 : 0] ?? "" }),
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

type Part = { label: string; values: { value: string; label: string; english: string }[] };

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAY_KEYS: MessageKey[] = ["fig.part.mon", "fig.part.tue", "fig.part.wed", "fig.part.thu", "fig.part.fri", "fig.part.sat", "fig.part.sun"];
const MONTH_KEYS: MessageKey[] = [
  "fig.part.jan", "fig.part.feb", "fig.part.mar", "fig.part.apr", "fig.part.may", "fig.part.jun",
  "fig.part.jul", "fig.part.aug", "fig.part.sep", "fig.part.oct", "fig.part.nov", "fig.part.dec",
];

/** Date parts offered by "falls on", with their values as the SQL part returns them. The labels
 *  follow the chosen language (127); `english` is the label a chart bar carries, matched on click. */
export function parts(): Record<string, Part> {
  return {
    day_of_week: {
      label: t("part.day_of_week"),
      values: DAYS.map((english, i) => ({ value: String(i + 1), label: t(DAY_KEYS[i] ?? "fig.part.mon"), english })),
    },
    month: {
      label: t("part.month"),
      values: MONTHS.map((english, i) => ({ value: String(i + 1), label: t(MONTH_KEYS[i] ?? "fig.part.jan"), english })),
    },
    quarter: {
      label: t("part.quarter"),
      values: [1, 2, 3, 4].map((q) => ({ value: String(q), label: t("fig.part.quarterN", { n: String(q) }), english: `Q${q}` })),
    },
    hour: {
      label: t("part.hour"),
      values: Array.from({ length: 24 }, (_, h) => ({ value: String(h), label: t("fig.part.hourN", { n: String(h) }), english: `${h}h` })),
    },
  };
}

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
    if (op === "duplicated") return `${key} · ${negate ? t("fig.filter.desc.once") : t("fig.filter.duplicates")}`;
    const order = condition.order_by ? t("fig.filter.desc.by", { column: condition.order_by }) : "";
    const keep = condition.keep === "last" ? t("fig.filter.desc.last") : t("fig.filter.desc.first");
    return `${key} · ${t("fig.filter.desc.unique", { keep, order })}`;
  }
  const symbols = SYMBOLS[op];
  if (symbols) return `${column} ${symbols[negate ? 1 : 0]} ${short(condition.value)}`;
  if (op === "between") {
    return `${column} ${negate ? t("fig.filter.outside") : t("fig.filter.desc.in")} [${short(condition.low)} – ${short(condition.high)}]`;
  }
  if (op === "in") return `${column} ${negate ? "∉" : "∈"} ${list(condition.values ?? [])}`;
  if (op === "contains") return `${column} ${negate ? t("fig.filter.notContains") : t("fig.filter.contains")} "${condition.value}"`;
  if (op === "starts_with") return `${column} ${negate ? t("fig.filter.notStartsWith") : t("fig.filter.startsWith")} "${condition.value}"`;
  if (op === "is_null") return `${column} ${negate ? t("fig.filter.isNotMissing") : t("fig.filter.isMissing")}`;
  if (op === "last_days") {
    const days = t("fig.filter.desc.lastDays", { days: String(condition.value ?? ""), to: short(condition.high) });
    return `${column} · ${negate ? t("fig.filter.desc.notInThe", { days }) : days}`;
  }
  const part = parts()[condition.part ?? ""];
  const labels = (condition.values ?? []).map(
    (value) => part?.values.find((item) => item.value === value)?.label ?? value,
  );
  return `${column} ${negate ? t("fig.filter.desc.notOn") : t("fig.filter.desc.on")} ${list(labels)}`;
}
