import { Filter, X } from "lucide-react";
import { useEffect, useState } from "react";

import { useColumns, useDerivedColumns, useOverview } from "../../api/hooks";
import { withDerived } from "../derived/derived";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Popover } from "../../ui/Popover";
import { describe } from "./conditions";
import { FilterPanel } from "./FilterPanel";
import { useFilters } from "./useFilters";

/** Dispatched on window (by the command palette) to open the filter panel. */
export const OPEN_FILTERS_EVENT = "de:open-filters";

type FilterButtonProps = {
  datasetId: string;
  showChips: boolean;
  onShowChips: (show: boolean) => void;
};

/** The top bar's Filter button: count of active filters, filtered rows, the panel. Shortcut F. */
export function FilterButton({ datasetId, showChips, onShowChips }: FilterButtonProps) {
  const [open, setOpen] = useState(false);
  const { conditions } = useFilters(datasetId);
  const columns = useColumns(datasetId, true);
  // 058: derived columns ("review › Has question") under their source, fetched with the panel
  const derived = useDerivedColumns(datasetId, open);
  const overview = useOverview(datasetId, conditions.length > 0);
  const counts = conditions.length ? overview.data : undefined;

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target instanceof Element ? event.target : null;
      const typing = target?.closest("input, select, textarea, [contenteditable=true]");
      if (typing || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key === "f" || event.key === "F") {
        event.preventDefault();
        setOpen(true);
      }
    };
    const onOpen = () => setOpen(true);
    window.addEventListener("keydown", onKey);
    window.addEventListener(OPEN_FILTERS_EVENT, onOpen);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener(OPEN_FILTERS_EVENT, onOpen);
    };
  }, []);

  return (
    <div className="flex items-center gap-2.5">
      {counts && (
        <span className="text-xs text-ink-2 tabular-nums" aria-live="polite">
          {t("filters.rows", { rows: formatCount(counts.rows), total: formatCount(counts.total_rows) })}
        </span>
      )}
      <Popover
        open={open}
        onOpenChange={setOpen}
        label={t("filters.title")}
        trigger={
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-haspopup="dialog"
            title={t("filters.hint")}
            className={`inline-flex h-(--control-h) items-center gap-1.5 rounded-md border px-2.5 text-[13px] transition-colors ${
              conditions.length
                ? "border-accent bg-accent-soft text-ink-1"
                : "border-line bg-surface-1 text-ink-2 hover:bg-surface-2 hover:text-ink-1"
            }`}
          >
            <Filter size={14} strokeWidth={1.5} aria-hidden />
            {t("filters.button")}
            {conditions.length > 0 && (
              <span className="rounded-full bg-accent px-1.5 text-[11px] font-medium text-accent-ink tabular-nums">
                {conditions.length}
              </span>
            )}
          </button>
        }
      >
        <FilterPanel
          datasetId={datasetId}
          columns={withDerived(columns.data ?? [], derived.data)}
          rows={counts?.rows}
          totalRows={counts?.total_rows}
          showChips={showChips}
          onShowChips={onShowChips}
        />
      </Popover>
    </div>
  );
}

/** Optional row of chips under the top bar (off by default, switched on in the panel). */
export function FilterChips({ datasetId }: { datasetId: string }) {
  const { conditions, remove, clear } = useFilters(datasetId);
  if (!conditions.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-line bg-surface-0 px-6 py-1.5" aria-label={t("filters.active")}>
      {conditions.map((condition, index) => (
        <span key={index} className="inline-flex items-center gap-1 rounded-full border border-line bg-surface-1 py-0.5 pr-1 pl-2.5 font-mono text-xs text-ink-1">
          {describe(condition)}
          <button
            type="button"
            aria-label={t("filters.remove", { filter: describe(condition) })}
            onClick={() => remove(index)}
            className="inline-flex size-4 items-center justify-center rounded-full text-ink-3 hover:bg-surface-2 hover:text-ink-1"
          >
            <X size={12} strokeWidth={1.75} aria-hidden />
          </button>
        </span>
      ))}
      <button type="button" onClick={clear} className="text-xs text-ink-3 hover:text-ink-1">
        {t("filters.clear")}
      </button>
    </div>
  );
}
