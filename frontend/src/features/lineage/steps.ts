import type { components } from "../../api/schema";
import { t, type MessageKey } from "../../lib/i18n";

/* 095: a cleaning step in one line, shared by the Clean module (its steps list) and the Changes
   page (the steps that explain a cleaned version): a page never imports a module. */

type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;
export type CleanStep = Complete<components["schemas"]["CleanDraft"]>["steps"][number];
export type CleanOp = CleanStep["op"];

export const opLabel = (op: CleanOp) => t(`clean.op.${op}` as MessageKey);

/** The columns a step reads. */
export function stepColumns(step: CleanStep): string[] {
  if (step.op === "entity_column") return step.anchor ? [step.column, step.anchor] : [step.column];
  if ("column" in step) return [step.column];
  if ("columns" in step) return step.columns;
  if (step.op === "filter_rows") return step.conditions.map((condition) => condition.column);
  if (step.op === "drop_duplicates") return step.keys ?? [];
  // N19: a date + time step reads its year, month and day from one date column: named once
  if (step.op === "assemble_date") return [...new Set([step.year, step.month, step.day, step.hour, step.minute, step.time].filter((name): name is string => Boolean(name)))];
  if (step.op === "group_by") return [...step.keys, ...step.aggregations.flatMap((aggregation) => (aggregation.column ? [aggregation.column] : []))];
  if (step.op === "merge_dataset") return step.keys;
  return [];
}

/** One line for the list: "Trim extra spaces · description". */
export function describeStep(step: CleanStep): string {
  const columns = stepColumns(step);
  const label = opLabel(step.op);
  if (step.op === "rename_column") return t("clean.describe.rename", { from: step.column, to: step.to });
  if (step.op === "change_type") return `${label} · ${step.column} → ${t(`clean.type.${step.to}` as MessageKey)}`;
  if (step.op === "filter_rows") return `${t(step.action === "keep" ? "clean.action.keep" : "clean.action.remove")} · ${columns.join(", ")}`;
  if (step.op === "drop_duplicates") return describeDuplicates(step, label);
  if (step.op === "label_rows") return t("clean.describe.labelRows", { to: step.to, rows: String(Object.keys(step.labels).length) });
  // 101: the steps of More checks, by row number
  if (step.op === "set_cells") return t("clean.describe.setCells", { cells: String(Object.keys(step.cells).length), column: step.column });
  if (step.op === "drop_rows") return t("clean.describe.dropRows", { rows: String(step.rows.length) });
  // B1: M5 transformations and the conditional step
  // N29: a free text's spaces trimmed, its line breaks kept
  if (step.op === "normalize_spaces" && step.keep_newlines) return t("clean.describe.keepNewlines", { columns: columns.join(", ") });
  if (step.op === "drop_empty_rows" && step.share !== null) return t("clean.describe.mostlyEmpty", { share: String(Math.round(step.share * 100)) });
  if (step.op === "split_column") return t("clean.describe.split", { column: step.column, parts: step.to.join(", ") });
  if (step.op === "merge_columns") return t("clean.describe.merge", { columns: step.columns.join(", "), to: step.to });
  if (step.op === "parse_quantity")
    return t("clean.describe.quantity", { column: step.column, to: [step.to_grams, step.to_litres, step.to_metres].filter((name): name is string => Boolean(name)).join(", ") });
  if (step.op === "entity_column") return describeEntities(step);
  if (step.op === "set_where") return t("clean.describe.setWhere", { column: step.column, cases: String(step.cases.length) });
  if (step.op === "conditional_column") return t("clean.describe.conditional", { to: step.to, cases: String(step.cases.length) });
  if (step.op === "computed_column") return t("clean.describe.computed", { to: step.to, formula: step.formula });
  if (step.op === "swap_values") return t("clean.describe.swap", { first: step.columns[0] ?? "", second: step.columns[1] ?? "" });
  if (step.op === "transform_number") return `${t(`clean.how.${step.how}` as MessageKey)} · ${step.column}`;
  // C1: dates, rows and header, structure
  if (step.op === "shift_date") return t("clean.describe.shift", { column: step.column, amount: `${step.amount > 0 ? "+" : ""}${step.amount}`, unit: t(`clean.unit.${step.unit}` as MessageKey) });
  if (step.op === "convert_timezone") return t("clean.describe.zone", { column: step.column, from: step.from_zone, to: step.to_zone });
  if (step.op === "keep_rows") return t(`clean.describe.keep.${step.how}` as MessageKey, { count: String(step.count) });
  if (step.op === "split_rows") return t("clean.describe.splitRows", { column: step.column, delimiter: step.delimiter });
  if (step.op === "unpivot") return t("clean.describe.unpivot", { columns: step.columns.join(", "), names: step.names_to, values: step.values_to });
  if (step.op === "pivot") return t("clean.describe.pivot", { column: step.column, fn: t(`clean.aggregate.${step.aggregate}` as MessageKey), values: step.value_column ?? t("clean.field.rowsCounted") });
  if (step.op === "group_by") return t("clean.describe.groupBy", { keys: step.keys.join(", "), count: String(step.aggregations.length) });
  if (step.op === "append_rows") return t("clean.describe.append", { name: step.label ?? step.dataset });
  if (step.op === "merge_dataset") return t("clean.describe.mergeDataset", { name: step.label ?? step.dataset, keys: step.keys.join(", "), how: t(`clean.joinHow.${step.how}` as MessageKey) });
  return columns.length ? `${label} · ${columns.join(", ")}` : label;
}

/** N36: an entity number, its clusters (A-20) and what their review kept apart or split. */
function describeEntities(step: Extract<CleanStep, { op: "entity_column" }>): string {
  const head = step.anchor ? t("clean.describe.entity", { column: step.column, anchor: step.anchor, to: step.to }) : t("clean.describe.entityAlone", { column: step.column, to: step.to });
  const notes = [
    step.groups.length ? t("clean.describe.entityClusters", { count: String(step.groups.length) }) : "",
    step.not_entities?.length ? t("clean.describe.entityApart", { count: String(step.not_entities.length) }) : "",
    Object.keys(step.split ?? {}).length ? t("clean.describe.entitySplit", { count: String(Object.keys(step.split ?? {}).length) }) : "",
  ].filter(Boolean);
  return notes.length ? `${head} · ${notes.join(", ")}` : head;
}

/** N35: a removal of duplicates, its keys or its listed groups (A-16), and what its review kept. */
function describeDuplicates(step: Extract<CleanStep, { op: "drop_duplicates" }>, label: string): string {
  const head = step.keys?.length ? `${label} · ${step.keys.join(", ")}` : step.groups?.length ? t("clean.describe.dupGroups", { label, count: String(step.groups.length) }) : label;
  const notes = [
    step.not_duplicates?.length ? t("clean.describe.dupApart", { count: String(step.not_duplicates.length) }) : "",
    step.survivors?.length ? t("clean.describe.dupChosen", { count: String(step.survivors.length) }) : "",
    step.merge ? t("clean.describe.dupMerge") : "",
  ].filter(Boolean);
  return notes.length ? `${head} · ${notes.join(", ")}` : head;
}
