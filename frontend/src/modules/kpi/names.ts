/** Suggested KPI names and the kind of dataset in the interface language (M01). The engine names a
 *  suggestion in English, by its rule ("Average basket", R-SUG-12) or from its measure ("Total
 *  income"); the page reads that name back with the English pattern of `catalog.json` and writes it
 *  with the French one, the column names unchanged. A name it cannot read stays as the engine wrote it. */

import type { KpiArchetype, KpiDefinition, KpiSuggestions } from "../../api/client";
import { findEngineText, locale, readTemplate, t } from "../../lib/i18n";
import catalog from "../../lib/i18n/catalog.json";

type Measure = KpiDefinition["measures"][number];
/** What naming a suggestion reads: its rule, its name, the shape of its definition. */
type Named = Pick<KpiSuggestions["kpis"][number], "rule" | "name"> & {
  definition: { measures: Pick<Measure, "agg" | "key" | "until">[]; ratio?: Pick<NonNullable<KpiDefinition["ratio"]>, "kind"> | null };
};

/** The keys that may name a suggestion: its rule's own name first, then its measure's pattern. */
export function nameKeys(kpi: Omit<Named, "name">): string[] {
  const keys = [`kpi.suggestName.rule.${kpi.rule}`];
  const { measures, ratio } = kpi.definition;
  const first = measures[0];
  if (ratio) keys.push(`kpi.suggestName.ratio.${ratio.kind ?? "ratio"}`);
  else if (first?.until) keys.push(`kpi.suggestName.delay.${first.agg}`);
  else if (first?.agg === "share") keys.push(first.key ? "kpi.suggestName.agg.shareOf" : "kpi.suggestName.agg.share");
  else if (first) keys.push(`kpi.suggestName.agg.${first.agg}`);
  return keys;
}

/** French "de" before a vowel: "Moyenne d'age", "% d'order_id". */
const elide = (text: string) => (locale() === "fr" ? text.replace(/(^|[\s(])de (?=[aeiouyàâäéèêëîïôöûüœ])/gi, "$1d'") : text);

/** A suggestion's name as the page shows (and saves) it. */
export function kpiName(kpi: Named): string {
  const found = findEngineText(nameKeys(kpi), kpi.name);
  if (found === null) return kpi.name;
  const text = elide(found);
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** "Looks like sales data · 5 KPIs" by the kind's id. */
export function archetypeMessage(archetype: Pick<KpiArchetype, "id" | "label" | "message">): string {
  const kind = findEngineText([`kpi.archetype.kind.${archetype.id}`], archetype.label);
  const read = readTemplate(catalog["kpi.archetype.looksLike"], archetype.message);
  if (kind === null || !read || read.kind !== archetype.label) return archetype.message;
  return t("kpi.archetype.looksLike", { kind, count: read.count ?? "" });
}
