import { useVirtualizer } from "@tanstack/react-virtual";
import { ArrowDownWideNarrow, ArrowUpNarrowWide } from "lucide-react";
import { useRef, useState } from "react";

import type { Cell, RowsView } from "../../api/client";
import { ROW_BLOCK, useRowBlocks } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Select } from "../../ui/fields";
import { CellView, isRightAligned } from "./cells";
import type { TableColumn } from "./DataTable";
import type { SortState } from "./HeaderCell";

const FIELD_HEIGHT = 20;
const CARD_GAP = 8;
const integer = new Intl.NumberFormat("en-US");

type RecordListProps = {
  datasetId: string;
  columns: TableColumn[];
  total: number;
  view: RowsView;
  /** Fields on a card: the first visible columns, pinned ones first. */
  fields: number;
  onSort: (column: string, sort: SortState) => void;
};

/** The rows on a phone or a tablet (041): one card per row with its first fields, the whole row in a sheet.
 *  Nothing scrolls sideways; sorting takes a menu instead of the column headers. */
export function RecordList({ datasetId, columns, total, view, fields, onSort }: RecordListProps) {
  const scroller = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState<number | null>(null);
  const shown = columns.slice(0, fields);
  const cardHeight = 34 + shown.length * FIELD_HEIGHT + 12;
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: total,
    getScrollElement: () => scroller.current,
    estimateSize: () => cardHeight + CARD_GAP,
    overscan: 6,
    initialRect: { width: 360, height: 640 },
  });
  const items = virtualizer.getVirtualItems();
  const firstBlock = Math.floor((items[0]?.index ?? 0) / ROW_BLOCK);
  const lastBlock = Math.floor((Math.max(items[items.length - 1]?.index ?? 0, open ?? 0)) / ROW_BLOCK);
  const wanted = Array.from({ length: lastBlock - firstBlock + 1 }, (_, k) => firstBlock + k);
  const results = useRowBlocks(datasetId, view, wanted, total > 0);
  const blocks = new Map(wanted.map((block, k) => [block, results[k]?.data]));
  const rowAt = (index: number) => blocks.get(Math.floor(index / ROW_BLOCK))?.rows[index % ROW_BLOCK];
  const opened = open === null ? undefined : rowAt(open);

  return (
    <div className="flex h-full flex-col gap-2">
      <div className="flex items-center gap-2">
        <Select
          aria-label={t("data.sortMenu")}
          value={view.sort ?? ""}
          onChange={(event) => onSort(event.target.value, event.target.value ? (view.desc ? "desc" : "asc") : null)}
          className="min-w-0 flex-1"
        >
          <option value="">{t("data.sortClear")}</option>
          {columns.map(({ column }) => (
            <option key={column.name} value={column.name}>
              {t("data.sortBy", { name: column.name })}
            </option>
          ))}
        </Select>
        <Button
          iconOnly
          icon={view.desc ? ArrowDownWideNarrow : ArrowUpNarrowWide}
          disabled={!view.sort}
          aria-label={view.desc ? t("data.sorted.desc") : t("data.sorted.asc")}
          onClick={() => view.sort && onSort(view.sort, view.desc ? "asc" : "desc")}
        />
      </div>
      <div ref={scroller} role="list" aria-label={t("data.table")} className="relative min-h-0 flex-1 overflow-y-auto">
        <div style={{ height: virtualizer.getTotalSize() }} className="relative">
          {items.map((item) => {
            const row = rowAt(item.index);
            return (
              <div
                key={item.key}
                role="listitem"
                className="absolute inset-x-0"
                style={{ top: 0, height: cardHeight, transform: `translateY(${item.start}px)` }}
              >
                <button
                  type="button"
                  onClick={() => setOpen(item.index)}
                  aria-label={t("data.openRow", { row: row ? integer.format(row[0]) : "" })}
                  className="flex h-full w-full flex-col rounded-lg border border-line bg-surface-1 px-3 py-2 text-left text-[13px] active:bg-surface-2"
                >
                  <span className="text-xs text-ink-3 tabular-nums">#{row ? integer.format(row[0]) : ""}</span>
                  <span className="mt-1 block w-full">
                    {shown.map(({ column, index }) => (
                      <span key={column.name} className="flex h-5 items-center gap-3">
                        <span className="w-2/5 shrink-0 truncate text-ink-3">{column.name}</span>
                        <span className={`min-w-0 flex-1 truncate ${isRightAligned(column.type) ? "tabular-nums" : ""}`}>
                          {row === undefined ? (
                            <span aria-hidden className="inline-block h-2.5 w-3/5 animate-pulse rounded bg-surface-2" />
                          ) : (
                            <CellView value={(row[index + 1] as Cell | undefined) ?? null} type={column.type} />
                          )}
                        </span>
                      </span>
                    ))}
                  </span>
                </button>
              </div>
            );
          })}
        </div>
      </div>
      <Dialog
        open={open !== null}
        onOpenChange={(next) => !next && setOpen(null)}
        title={t("data.rowTitle", { row: opened ? integer.format(opened[0]) : "" })}
      >
        <dl className="divide-y divide-line text-[13px]">
          {columns.map(({ column, index }) => (
            <div key={column.name} className="flex gap-3 py-2">
              <dt className="w-2/5 shrink-0 text-ink-3">{column.name}</dt>
              <dd className="min-w-0 flex-1">
                <CellView value={(opened?.[index + 1] as Cell | undefined) ?? null} type={column.type} />
              </dd>
            </div>
          ))}
        </dl>
      </Dialog>
    </div>
  );
}
