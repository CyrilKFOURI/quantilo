import type { AnalysisType } from "../../api/client";
import type { Condition } from "../../api/filters";
import type { RowCondition, Step } from "./api";

/** N41: the rows a fix touches, as a filter of the Data page when its step can be written as
 *  one; null when it names rows by number (More checks), by a formula, by several cases, or by a
 *  test the column's type does not offer as a filter. */

/** The operators each type takes as a filter (mirrors ALLOWED of quantilo/filters/sql.py). */
const ALLOWED: Record<AnalysisType, Condition["op"][]> = {
  numeric_continuous: ["eq", "gt", "gte", "lt", "lte", "between", "is_null"],
  numeric_discrete: ["eq", "gt", "gte", "lt", "lte", "between", "in", "is_null"],
  categorical_nominal: ["in", "eq", "contains", "starts_with", "is_null"],
  categorical_ordinal: ["in", "eq", "contains", "starts_with", "is_null"],
  boolean: ["in", "is_null"],
  text: ["eq", "contains", "starts_with", "in", "is_null"],
  identifier: ["eq", "contains", "starts_with", "in", "is_null"],
  datetime: ["eq", "gt", "gte", "lt", "lte", "between", "part_in", "last_days", "is_null"],
} as Record<AnalysisType, Condition["op"][]>;

/** A filter takes 1,000 values at most (MAX_VALUES of quantilo/filters/model.py). */
const MAX_VALUES = 1_000;

export type Types = Record<string, AnalysisType | undefined>;

const NUMERIC = new Set<string>(["numeric_continuous", "numeric_discrete"]);

export function allowed(types: Types, column: string, op: Condition["op"]): boolean {
  const type = types[column];
  return type !== undefined && (ALLOWED[type] ?? []).includes(op);
}

/** A value as the filter compares it: a number on a number column. */
export function operand(types: Types, column: string, value: string | number): string | number | null {
  if (!NUMERIC.has(types[column] ?? "")) return String(value);
  const number = typeof value === "number" ? value : Number(value.trim());
  return value !== "" && Number.isFinite(number) ? number : null;
}

/** The rows holding one of `values`: "is any of", or "=" for one value of a column without it. */
function anyOf(types: Types, column: string, values: string[]): Condition | null {
  if (!values.length || values.length > MAX_VALUES) return null;
  if (allowed(types, column, "in")) return { column, op: "in", values, negate: false };
  const value = values.length === 1 ? operand(types, column, values[0]!) : null;
  return value !== null && allowed(types, column, "eq") ? { column, op: "eq", value, negate: false } : null;
}

const COMPARE: Partial<Record<RowCondition["op"], Condition["op"]>> = { lt: "lt", le: "lte", gt: "gt", ge: "gte", contains: "contains" };

/** A condition of a step (RowCondition) as a Data filter, when the column's type offers it. */
function fromRow(types: Types, condition: RowCondition): Condition | null {
  const { column } = condition;
  if (condition.op === "empty" || condition.op === "not_empty")
    return allowed(types, column, "is_null") ? { column, op: "is_null", negate: condition.op === "not_empty" } : null;
  if (condition.op === "in") return anyOf(types, column, (condition.values ?? []).filter((value): value is string => typeof value === "string"));
  if (condition.value === null || condition.value === undefined) return null;
  const op = condition.op === "eq" || condition.op === "ne" ? "eq" : COMPARE[condition.op];
  if (!op || !allowed(types, column, op)) return null;
  const value = op === "contains" ? String(condition.value) : operand(types, column, condition.value as string | number);
  return value === null ? null : { column, op, value, negate: condition.op === "ne" };
}

/** Every condition, or none when one cannot be a filter (Data combines them with AND). */
export function allOf(types: Types, conditions: RowCondition[], combine: "all" | "any"): Condition[] | null {
  if (!conditions.length || (combine === "any" && conditions.length > 1)) return null;
  const found = conditions.map((condition) => fromRow(types, condition));
  return found.every((condition): condition is Condition => condition !== null) ? found : null;
}

/** The Data filter of the rows a step touches, or null (see above). */
export function filterOf(step: Step, types: Types): Condition[] | null {
  const one = (condition: Condition | null) => (condition ? [condition] : null);
  switch (step.op) {
    case "map_values":
      return one(anyOf(types, step.column, Object.keys(step.map)));
    case "null_values":
      return step.match === "exact" ? one(anyOf(types, step.column, step.values.filter((value): value is string => typeof value === "string"))) : null;
    case "replace_values":
      return step.match === "exact" && step.case_sensitive ? one(anyOf(types, step.column, [step.find])) : null;
    case "filter_rows":
      // the rows it removes; a "keep" step touches the others, which a filter of AND cannot say
      return step.action === "remove" ? allOf(types, step.conditions, step.combine) : null;
    case "set_where":
    case "conditional_column": {
      const [only, ...more] = step.cases;
      return only && !more.length && !only.when ? allOf(types, only.conditions, only.combine) : null;
    }
    default:
      return null;
  }
}
