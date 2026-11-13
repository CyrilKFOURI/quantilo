import { describeStep, stepColumns, type CleanStep } from "../../features/lineage/steps";
import { formatCount } from "../../lib/format";
import { locale, t, type MessageKey } from "../../lib/i18n";
import { replaceColumn } from "./groups";

/** Rules whose title says one thing about its columns: written by code in the page's language. */
const BY_RULE: Partial<Record<string, MessageKey>> = {
  "B-01": "clean.title.B-01",
  "B-02": "clean.title.B-02",
  "B-05": "clean.title.B-05",
  "B-06": "clean.title.B-06",
  "B-18": "clean.title.B-18",
  "B-20": "clean.title.B-20",
};

/** 127: a suggestion's title comes from the engine in English. In another language the page
 *  writes it from its rule ("Réparer l'encodage de brand") or else its step ("… · description"). */
export function suggestionTitle(suggestion: { title: string; rule?: string; step?: unknown }): string {
  if (locale() === "en" || !suggestion.step) return suggestion.title;
  const step = suggestion.step as CleanStep;
  const key = suggestion.rule ? BY_RULE[suggestion.rule] : undefined;
  if (key) return t(key, { column: stepColumns(step).join(", ") });
  return describeStep(step);
}

/** N39: the title of the same fix on several columns, in the page's language: "Remove 48 empty
 *  columns", else the first fix's title with its column replaced ("Trim extra spaces in 30
 *  columns"), else "… and 29 more columns". */
export function batchTitle(items: { title: string; rule?: string; step?: unknown; columns: string[] }[]): string {
  const first = items[0];
  if (!first) return "";
  const count = formatCount(items.length);
  if (first.rule === "B-19" && first.title.startsWith("Remove the empty column")) return t("clean.batch.B-19.empty", { count });
  if (first.rule === "B-19" && first.title.startsWith("Remove the constant column")) return t("clean.batch.B-19.constant", { count });
  const title = suggestionTitle(first);
  return (
    replaceColumn(title, first.columns[0] ?? "", t("clean.batch.columns", { count })) ??
    t("clean.batch.more", { title, count: formatCount(items.length - 1) })
  );
}
