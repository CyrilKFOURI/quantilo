import { useVirtualizer } from "@tanstack/react-virtual";
import { useMemo, useRef, type CSSProperties } from "react";

import type { Cell, Column, RowsView } from "../../api/client";
import { ROW_BLOCK, useRowBlocks } from "../../api/hooks";
import { t } from "../../lib/i18n";
import { colorVar } from "../pins/rows";
import { CellView, columnWidth, isRightAligned } from "./cells";
import { HeaderCell, type SortState } from "./HeaderCell";

const ROW_HEIGHT = 32;
const HEADER_HEIGHT = 56;
const NUMBER_WIDTH = 72;
const integer = new Intl.NumberFormat("en-US");

export type TableColumn = { column: Column; index: number; bars: { label: string; count: number }[] };

type DataTableProps = {
  datasetId: string;
  columns: TableColumn[]; // visible, pinned first; `index` = position in the rows sent by the server
  pinned: string[];
  total: number;
  view: RowsView;
  onSort: (column: string, sort: SortState) => void;
  onFilter: (column: Column) => void;
  onHide: (name: string) => void;
  onPin: (name: string) => void;
  /** 099: pinned rows (row number in the file, from 0) and their label's colour. */
  marks?: ReadonlyMap<number, { label: string; color: number }>;
  /** 099, while Advanced · Pinned rows is open: the row numbers become buttons that select. */
  selected?: ReadonlySet<number>;
  onToggleRow?: (row: number) => void;
};

/** Virtualized rows (only the visible ones exist in the page), sticky header and pinned columns. */
export function DataTable(props: DataTableProps) {
  const { datasetId, columns, pinned, total, view, marks, selected, onToggleRow } = props;
  const sort = view.sort ? { column: view.sort, desc: view.desc } : null;
  const scroller = useRef<HTMLDivElement>(null);
  // The compiler skips memoizing this component (the virtualizer's functions change on scroll).
  // eslint-disable-next-line react-hooks/incompatible-library
  const virtualizer = useVirtualizer({
    count: total,
    getScrollElement: () => scroller.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 12,
    initialRect: { width: 1200, height: 640 },
  });
  const items = virtualizer.getVirtualItems();
  const firstBlock = Math.floor((items[0]?.index ?? 0) / ROW_BLOCK);
  const lastBlock = Math.floor((items[items.length - 1]?.index ?? 0) / ROW_BLOCK);
  const wanted = Array.from({ length: lastBlock - firstBlock + 1 }, (_, k) => firstBlock + k);
  const results = useRowBlocks(datasetId, view, wanted, total > 0);
  const blocks = new Map(wanted.map((block, k) => [block, results[k]?.data]));

  const layout = useMemo(() => {
    let left = NUMBER_WIDTH;
    return columns.map((item) => {
      const width = columnWidth(item.column.name, item.column.type);
      const isPinned = pinned.includes(item.column.name);
      const style: CSSProperties = { width, minWidth: width };
      if (isPinned) {
        Object.assign(style, { position: "sticky", left, zIndex: 1 });
        left += width;
      }
      return { ...item, width, isPinned, style };
    });
  }, [columns, pinned]);
  const width = NUMBER_WIDTH + layout.reduce((sum, item) => sum + item.width, 0);
  const lastPinned = layout.filter((item) => item.isPinned).pop()?.column.name;
  const numberStyle: CSSProperties = { width: NUMBER_WIDTH, minWidth: NUMBER_WIDTH, position: "sticky", left: 0, zIndex: 1 };

  const rowAt = (index: number) => blocks.get(Math.floor(index / ROW_BLOCK))?.rows[index % ROW_BLOCK];

  return (
    <div
      ref={scroller}
      role="grid"
      aria-label={t("data.table")}
      aria-rowcount={total + 1}
      aria-colcount={columns.length + 1}
      className="relative h-full overflow-auto rounded-lg border border-line bg-surface-1 text-[13px]"
    >
      <div style={{ width, height: virtualizer.getTotalSize() + HEADER_HEIGHT }} className="relative">
        <div role="row" aria-rowindex={1} className="sticky top-0 z-20 flex border-b border-line bg-surface-2" style={{ height: HEADER_HEIGHT, width }}>
          <div role="columnheader" style={numberStyle} className="flex items-end bg-surface-2 px-2.5 pb-2 text-xs text-ink-3">
            #
          </div>
          {layout.map((item) => (
            <div
              key={item.column.name}
              role="columnheader"
              aria-sort={sort?.column === item.column.name ? (sort.desc ? "descending" : "ascending") : undefined}
              style={item.style}
              className={`bg-surface-2 ${item.column.name === lastPinned ? "border-r border-line" : ""}`}
            >
              <HeaderCell
                datasetId={datasetId}
                column={item.column}
                bars={item.bars}
                sort={sort?.column === item.column.name ? (sort.desc ? "desc" : "asc") : null}
                pinned={item.isPinned}
                onSort={(next) => props.onSort(item.column.name, next)}
                onFilter={() => props.onFilter(item.column)}
                onHide={() => props.onHide(item.column.name)}
                onPin={() => props.onPin(item.column.name)}
              />
            </div>
          ))}
        </div>
        {items.map((virtualRow) => {
          const row = rowAt(virtualRow.index);
          return (
            <div
              key={virtualRow.key}
              role="row"
              aria-rowindex={virtualRow.index + 2}
              className="group absolute left-0 flex border-b border-line/60"
              style={{ top: HEADER_HEIGHT, height: ROW_HEIGHT, width, transform: `translateY(${virtualRow.start}px)` }}
            >
              <RowHeader number={row?.[0]} style={numberStyle} marks={marks} selected={selected} onToggle={onToggleRow} />
              {layout.map((item) => {
                const value = row?.[item.index + 1] as Cell | undefined;
                return (
                  <div
                    key={item.column.name}
                    role="gridcell"
                    style={item.style}
                    className={`flex items-center overflow-hidden bg-surface-1 px-2.5 whitespace-nowrap group-hover:bg-surface-2 ${
                      isRightAligned(item.column.type) ? "justify-end tabular-nums" : ""
                    } ${item.column.name === lastPinned ? "border-r border-line" : ""}`}
                  >
                    {row === undefined ? (
                      <span aria-hidden className="h-2.5 w-3/5 animate-pulse rounded bg-surface-2" />
                    ) : (
                      <span className="truncate">
                        <CellView value={value ?? null} type={item.column.type} />
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/** The row number, kept on the left; 099: the colour of its pin, and a selection button while
 *  rows are being pinned. */
function RowHeader({
  number,
  style,
  marks,
  selected,
  onToggle,
}: {
  number: number | undefined; // from 1, as shown
  style: CSSProperties;
  marks?: ReadonlyMap<number, { label: string; color: number }>;
  selected?: ReadonlySet<number>;
  onToggle?: (row: number) => void;
}) {
  const row = number === undefined ? undefined : number - 1;
  const mark = row === undefined ? undefined : marks?.get(row);
  const chosen = row !== undefined && Boolean(selected?.has(row));
  const shown = number === undefined ? "" : integer.format(number);
  return (
    <div
      role="rowheader"
      style={style}
      title={mark ? t("pins.marked", { label: mark.label }) : undefined}
      className={`relative flex items-center justify-end px-2.5 text-xs text-ink-3 tabular-nums group-hover:bg-surface-2 ${chosen ? "bg-accent-soft" : "bg-surface-1"}`}
    >
      {mark && <span aria-hidden className="absolute inset-y-1 left-0 w-1 rounded-r-sm" style={{ background: colorVar(mark.color) }} />}
      {onToggle && row !== undefined ? (
        <button
          type="button"
          aria-pressed={chosen}
          aria-label={t("pins.selectRow", { row: shown })}
          onClick={() => onToggle(row)}
          className={`rounded px-1 tabular-nums hover:text-ink-1 hover:underline ${chosen ? "font-medium text-ink-1" : ""}`}
        >
          {shown}
        </button>
      ) : (
        shown
      )}
      {mark && <span className="sr-only">{t("pins.marked", { label: mark.label })}</span>}
    </div>
  );
}
