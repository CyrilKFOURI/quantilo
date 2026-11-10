import { Activity, ArrowDown, ArrowLeftRight, ArrowUp, Copy, Download, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";

import type { KpiResult } from "../../api/client";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import type { Format } from "./definition";
import { formatShown } from "./format";
import { buildTable, DIVERGING_MODES, toDelimited, type TableCell, type TableOptions } from "./pivot";

function background(cell: TableCell): string | undefined {
  if (cell.shade === null || cell.total) return undefined;
  if (DIVERGING_MODES.has(cell.mode)) {
    const color = cell.shade >= 0 ? "var(--chart-1)" : "var(--chart-2)";
    return `color-mix(in oklab, ${color} ${Math.round(Math.abs(cell.shade) * 32)}%, transparent)`;
  }
  return `color-mix(in oklab, var(--chart-1) ${Math.round(4 + cell.shade * 30)}%, transparent)`;
}

/** 054: an icon before the value and the sentence for screen readers; never colour alone. */
function CellMark({ flag }: { flag: NonNullable<TableCell["flag"]> }) {
  const Icon = flag.kind === "anomaly" ? Activity : TriangleAlert;
  return (
    <>
      <Icon size={11} strokeWidth={2} aria-hidden className={`mr-1 inline align-[-1px] ${flag.severity === "warning" ? "text-warn" : "text-ink-3"}`} />
      <span className="sr-only">{flag.message} </span>
    </>
  );
}

export function PivotTable({
  result,
  options,
  formats,
  canSwap,
  onSwap,
  onSort,
  fileName,
}: {
  result: KpiResult;
  options: TableOptions;
  formats: Record<string, Format>;
  canSwap: boolean;
  onSwap: () => void;
  onSort: (sort: string | null) => void;
  fileName: string;
}) {
  const table = useMemo(() => buildTable(result, options), [result, options]);
  const [copied, setCopied] = useState(false);
  const formatOf = (measure: string) => result.measures.find((m) => m.id === measure)?.format;
  const lastHead = table.head.length - 1;
  const [sortColumn, sortDirection] = (options.sort ?? "").split(":");

  const copy = () => {
    void navigator.clipboard?.writeText(toDelimited(table, "\t")).then(() => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    });
  };
  const download = () => {
    const blob = new Blob([toDelimited(table, ",")], { type: "text/csv;charset=utf-8" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${fileName.replace(/[^\w-]+/g, "_")}.csv`;
    link.click();
    URL.revokeObjectURL(link.href);
  };
  const sortBy = (column: number) => {
    if (!table.sortable) return;
    const current = sortColumn === String(column) ? sortDirection : null;
    onSort(current === "desc" ? `${column}:asc` : current === "asc" ? null : `${column}:desc`);
  };

  return (
    <section aria-labelledby="kpi-table" className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id="kpi-table" className="text-[15px] font-semibold text-ink-1">
          {t("kpi.table")}
        </h2>
        <div className="flex items-center gap-1">
          {canSwap && (
            <Button variant="ghost" icon={ArrowLeftRight} onClick={onSwap} title={t("kpi.swap")}>
              {t("kpi.swap.short")}
            </Button>
          )}
          <Button variant="ghost" icon={Copy} onClick={copy}>
            {copied ? t("kpi.copied") : t("kpi.copy")}
          </Button>
          <Button variant="ghost" icon={Download} onClick={download}>
            {t("kpi.csv")}
          </Button>
        </div>
      </div>
      {table.note && <p className="text-xs text-ink-3">{table.note}</p>}
      {/* A wide table (periods across) scrolls sideways; its values never break letter by letter. */}
      <div className="max-h-[560px] overflow-auto overscroll-x-contain border-t border-line">
        <table data-keep-line className="w-full border-separate border-spacing-0 text-[13px] tabular-nums">
          <thead>
            {table.head.map((row, r) => (
              <tr key={r}>
                {r === 0 && (
                  <th
                    rowSpan={table.head.length}
                    scope="col"
                    className="sticky top-0 left-0 z-20 border-b border-line bg-surface-2 px-3 py-2 text-left text-xs font-medium whitespace-nowrap text-ink-2"
                  >
                    {table.corner}
                  </th>
                )}
                {row.map((head, i) => {
                  const sortable = table.sortable && r === lastHead;
                  const sorted = sortable && sortColumn === String(i);
                  return (
                    <th
                      key={i}
                      colSpan={head.span}
                      scope="col"
                      aria-sort={sorted ? (sortDirection === "asc" ? "ascending" : "descending") : undefined}
                      className="sticky z-10 border-b border-line bg-surface-2 px-3 py-2 text-right text-xs font-medium whitespace-nowrap text-ink-2"
                      style={{ top: `${r * 33}px` }}
                    >
                      {sortable ? (
                        <button type="button" onClick={() => sortBy(i)} className="inline-flex items-center gap-1 hover:text-ink-1">
                          {head.label}
                          {sorted && (sortDirection === "asc" ? <ArrowUp size={12} aria-hidden /> : <ArrowDown size={12} aria-hidden />)}
                        </button>
                      ) : (
                        head.label
                      )}
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.rows.map((row) => {
              if (row.kind === "section") {
                return (
                  <tr key={row.key}>
                    <th colSpan={1 + (table.rows.find((x) => x.cells.length)?.cells.length ?? 1)} scope="rowgroup" className="sticky left-0 border-b border-line bg-surface-1 px-3 pt-3 pb-1 text-left text-xs font-semibold text-ink-1">
                      {row.label}
                    </th>
                  </tr>
                );
              }
              const strong = row.kind === "total" || row.kind === "subtotal";
              return (
                <tr key={row.key} className={row.kind === "total" ? "bg-surface-2" : "hover:bg-surface-2/60"}>
                  <th
                    scope="row"
                    className={`sticky left-0 border-b border-line bg-surface-1 py-1.5 pr-3 text-left whitespace-nowrap ${strong ? "font-semibold text-ink-1" : "font-normal text-ink-1"} ${row.partial ? "italic" : ""}`}
                    style={{ paddingLeft: `${12 + row.indent * 16}px` }}
                  >
                    {row.label}
                  </th>
                  {row.cells.map((cell, i) => {
                    const auto = formatOf(cell.measure);
                    return (
                      <td
                        key={i}
                        className={`border-b border-line px-3 py-1.5 text-right whitespace-nowrap ${cell.total || strong ? "font-semibold" : ""} ${cell.partial ? "italic text-ink-2" : "text-ink-1"}`}
                        style={{ background: background(cell) }}
                        title={cell.flag?.message}
                      >
                        {cell.flag && <CellMark flag={cell.flag} />}
                        {auto ? formatShown(cell.value, cell.mode, auto, formats[cell.measure] ?? {}, "full") : ""}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {result.periods.some((p) => p.partial) && <p className="text-xs text-ink-3">{t("kpi.partialNote")}</p>}
      {options.hideOther && result.splits.some((s) => s.others > 0) && <p className="text-xs text-ink-3">{t("kpi.hiddenOther")}</p>}
    </section>
  );
}
