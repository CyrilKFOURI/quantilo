import { Link } from "@tanstack/react-router";
import { Table2 } from "lucide-react";

import { useColumns } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { Step, Suggestion } from "./api";
import { useReview } from "./context";
import { filterOf } from "./rows";

/** N32: an information about rows without a step (T-01 orphan keys, T-02 repeated keys) names
 *  them by its `where`; this opens them as a Data filter, where they can be looked at or pinned
 *  (099). Nothing when the rows cannot be written as a filter. */
export function WhereRows({ suggestion }: { suggestion: Suggestion }) {
  const context = useReview();
  const where = suggestion.step ? [] : (suggestion.where ?? []);
  const datasetId = context?.datasetId ?? "";
  const columns = useColumns(datasetId, Boolean(context) && where.length > 0);
  if (!context || !where.length || !columns.data) return null;
  const types = Object.fromEntries(columns.data.map((column) => [column.name, column.type]));
  const step = { op: "filter_rows", id: "where", conditions: where, combine: "all", action: "remove" } as unknown as Step;
  const conditions = filterOf(step, types);
  if (!conditions) return null;
  const count = where.reduce((total, condition) => total + (condition.values?.length ?? 0), 0);
  const label = t("cleanMisc.where.open");
  return (
    <Link
      to="/d/$datasetId/data"
      params={{ datasetId }}
      search={{ filters: { dataset: datasetId, conditions } }}
      aria-label={label}
      title={`${label} (${t("cleanMisc.where.hint", { count: formatCount(count) })})`}
      className="inline-flex size-(--control-h) shrink-0 items-center justify-center rounded-md text-ink-2 transition-[background-color,filter] hover:bg-surface-2 hover:text-ink-1"
    >
      <Table2 size={15} strokeWidth={1.5} aria-hidden />
    </Link>
  );
}
