import type { Condition } from "../../api/filters";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import type { CheckKind, CleanCheck, RowCondition, Step, Suggestion } from "./api";
import { allOf, allowed, operand, type Types } from "./rows";

/** N31: declared checks, as dbt tests or Great Expectations: what must hold in the data, run again
 *  at each import and each version; a check never fixes anything. These are the page's coded rules
 *  around them: a new check, the check a step or a rule's suggestion stands for, its sentence, and
 *  the Data filter of its failing rows when there is one. */

export const CHECK_KINDS: CheckKind[] = ["not_empty", "unique", "in_list", "between", "pattern", "compare", "sum", "determines", "condition"];

/** The columns a kind takes (mirrors ARITY of quantilo/cleaning/checks.py). */
export const ARITY: Record<CheckKind, [number, number]> = {
  not_empty: [1, 50],
  unique: [1, 4],
  in_list: [1, 1],
  between: [1, 1],
  pattern: [1, 1],
  compare: [2, 2],
  sum: [3, 21],
  determines: [2, 2],
  condition: [0, 0],
};

export function nextCheckId(checks: CleanCheck[]): string {
  const used = checks.map((check) => Number(check.id.slice(1))).filter(Number.isFinite);
  return `c${Math.max(0, ...used) + 1}`;
}

const BASE: Omit<CleanCheck, "id" | "kind" | "columns"> = {
  name: null,
  enabled: true,
  values: [],
  min: null,
  max: null,
  pattern: null,
  op: "le",
  tolerance: 0,
  conditions: [],
  combine: "all",
  expect: "all",
  origin: "user",
  rule: null,
};

/** A new check of `kind` on the first columns given, the rest left to fill. */
export function blankCheck(kind: CheckKind, id: string, columns: string[]): CleanCheck {
  const [least] = ARITY[kind];
  const picked = kind === "condition" ? [] : columns.slice(0, Math.max(least, 1));
  const conditions: RowCondition[] = kind === "condition" && columns[0] ? [{ column: columns[0], op: "not_empty", value: null, values: null }] : [];
  return { ...BASE, id, kind, columns: picked, conditions };
}

/** The check a step stands for: a filter that removes rows → "no row meets these conditions" (a
 *  filter that keeps them: "every row does"); duplicates removed by a key → that key is unique. */
export function checkOfStep(step: Step, id: string): CleanCheck | null {
  const from = { ...BASE, id, origin: "step" as const, rule: step.rule ?? null };
  if (step.op === "filter_rows") return { ...from, kind: "condition", columns: [], conditions: step.conditions, combine: step.combine, expect: step.action === "remove" ? "none" : "all" };
  if (step.op === "drop_duplicates" && step.keys && step.keys.length >= 1 && step.keys.length <= 4) return { ...from, kind: "unique", columns: step.keys };
  return null;
}

/** The check a rule's suggestion stands for (coded per rule): A-10 "a determines b", A-11 "the
 *  first date is not after the second", else the check of its step. */
export function checkOfSuggestion(suggestion: Suggestion, id: string): CleanCheck | null {
  const from = { ...BASE, id, origin: "rule" as const, rule: suggestion.rule };
  const [a, b] = suggestion.columns;
  if (suggestion.rule === "A-10" && a && b && suggestion.columns.length === 2) return { ...from, kind: "determines", columns: [a, b] };
  if (suggestion.rule === "A-11" && a && b && suggestion.columns.length === 2) return { ...from, kind: "compare", columns: [a, b], op: "le" };
  const step = suggestion.step ? checkOfStep(suggestion.step as Step, id) : null;
  return step ? { ...step, origin: "rule", rule: suggestion.rule } : null;
}

/** Two checks that test the same thing (a recipe's check already declared here). */
export function sameCheck(a: CleanCheck, b: CleanCheck): boolean {
  const body = (check: CleanCheck) => JSON.stringify({ ...check, id: "", name: null, origin: "user", rule: null, enabled: true });
  return body(a) === body(b);
}

const SYMBOL: Record<CleanCheck["op"], string> = { lt: "<", le: "≤", eq: "=", ne: "≠", ge: "≥", gt: ">" };

function listed(values: string[]): string {
  const shown = values.slice(0, 5).join(", ");
  return values.length > 5 ? t("clean.checks.andMore", { values: shown, count: formatCount(values.length - 5) }) : shown;
}

function conditionText(condition: RowCondition): string {
  const value = condition.op === "in" ? listed((condition.values ?? []).map(String)) : String(condition.value ?? "");
  return t(`clean.checks.op.${condition.op}` as MessageKey, { column: condition.column, value });
}

/** The check in a sentence of the page's language ("order_id is unique"); its own name if given. */
export function describeCheck(check: CleanCheck): string {
  if (check.name) return check.name;
  const [a = "", b = ""] = check.columns;
  switch (check.kind) {
    case "not_empty":
      return t(check.columns.length > 1 ? "clean.checks.say.not_empty.many" : "clean.checks.say.not_empty", { columns: check.columns.join(", ") });
    case "unique":
      return t("clean.checks.say.unique", { columns: check.columns.join(" + ") });
    case "in_list":
      return t("clean.checks.say.in_list", { column: a, values: listed(check.values) });
    case "between":
      if (check.min !== null && check.max !== null) return t("clean.checks.say.between", { column: a, min: check.min, max: check.max });
      return check.min !== null ? t("clean.checks.say.atLeast", { column: a, min: check.min }) : t("clean.checks.say.atMost", { column: a, max: check.max ?? "" });
    case "pattern":
      return t("clean.checks.say.pattern", { column: a, pattern: check.pattern ?? "" });
    case "compare":
      return `${a} ${SYMBOL[check.op]} ${b}`;
    case "sum":
      return check.tolerance ? t("clean.checks.say.sumWithin", { total: a, parts: check.columns.slice(1).join(" + "), tolerance: String(check.tolerance) }) : `${a} = ${check.columns.slice(1).join(" + ")}`;
    case "determines":
      return t("clean.checks.say.determines", { a, b });
    case "condition": {
      const parts = check.conditions.map(conditionText).join(check.combine === "all" ? t("clean.checks.and") : t("clean.checks.or"));
      return t(check.expect === "all" ? "clean.checks.say.every" : "clean.checks.say.none", { conditions: parts });
    }
  }
}

/** The failing rows as a filter of the Data page, when the check can be written as one (Data
 *  combines its conditions with AND); null otherwise (two columns compared, a pattern, a sum). */
export function checkFilter(check: CleanCheck, types: Types): Condition[] | null {
  const [column] = check.columns;
  switch (check.kind) {
    case "not_empty":
      return column && check.columns.length === 1 && allowed(types, column, "is_null") ? [{ column, op: "is_null", negate: false }] : null;
    case "unique":
      return column ? [{ column, op: "duplicated", columns: check.columns.slice(1), negate: false }] : null;
    case "in_list":
      return column && allowed(types, column, "in") && check.values.length <= 1_000 ? [{ column, op: "in", values: check.values, negate: true }] : null;
    case "between": {
      if (!column) return null;
      const low = check.min === null ? null : operand(types, column, check.min);
      const high = check.max === null ? null : operand(types, column, check.max);
      if (low !== null && high !== null) return allowed(types, column, "between") ? [{ column, op: "between", low, high, negate: true }] : null;
      if (low !== null) return allowed(types, column, "lt") ? [{ column, op: "lt", value: low, negate: false }] : null;
      if (high !== null) return allowed(types, column, "gt") ? [{ column, op: "gt", value: high, negate: false }] : null;
      return null;
    }
    case "condition":
      return check.expect === "none" ? allOf(types, check.conditions, check.combine) : null;
    default:
      return null;
  }
}
