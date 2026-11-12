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
  if ("column" in step) return [step.column];
  if ("columns" in step) return step.columns;
  if (step.op === "filter_rows") return step.conditions.map((condition) => condition.column);
  if (step.op === "drop_duplicates") return step.keys ?? [];
  if (step.op === "assemble_date") return [step.year, step.month, step.day, step.hour, step.minute].filter((name): name is string => Boolean(name));
  return [];
}

/** One line for the list: "Trim extra spaces · description". */
export function describeStep(step: CleanStep): string {
  const columns = stepColumns(step);
  const label = opLabel(step.op);
  if (step.op === "rename_column") return t("clean.describe.rename", { from: step.column, to: step.to });
  if (step.op === "change_type") return `${label} · ${step.column} → ${t(`clean.type.${step.to}` as MessageKey)}`;
  if (step.op === "filter_rows") return `${t(step.action === "keep" ? "clean.action.keep" : "clean.action.remove")} · ${columns.join(", ")}`;
  if (step.op === "drop_duplicates") return step.keys?.length ? `${label} · ${step.keys.join(", ")}` : label;
  return columns.length ? `${label} · ${columns.join(", ")}` : label;
}
