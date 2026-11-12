import type { CSSProperties } from "react";

import type { TableChoice } from "../../api/client";

export type GridArea = {
  id: string;
  top: number;
  left: number;
  bottom: number;
  right: number;
  headerRows: number[];
  droppedRows: number[];
  color: number; // 1..8, the palette token --chart-<n>
};

/** "A", "B", … "AA": the column letters of a sheet. */
export function columnLetter(index: number): string {
  let letters = "";
  let n = index;
  while (n > 0) {
    const rest = (n - 1) % 26;
    letters = String.fromCharCode(65 + rest) + letters;
    n = Math.floor((n - 1) / 26);
  }
  return letters;
}

/** The areas to draw for the tables of one sheet, coloured with the palette in their order. */
export function gridAreas(tables: TableChoice[]): GridArea[] {
  return tables.flatMap((table, index) =>
    table.area
      ? [
          {
            id: table.id,
            top: table.area.top,
            left: table.area.left,
            bottom: table.area.bottom,
            right: table.area.right,
            headerRows: table.area.header_rows,
            droppedRows: table.area.dropped_rows ?? [],
            color: (index % 8) + 1,
          },
        ]
      : [],
  );
}

function cellStyle(area: GridArea, row: number, col: number, active: boolean): CSSProperties {
  const tone = `var(--chart-${area.color})`;
  const tint = `color-mix(in srgb, ${tone} ${active ? 24 : 12}%, transparent)`;
  const edge = `${active ? 2 : 1}px solid ${tone}`;
  const style: CSSProperties = { backgroundColor: tint };
  if (area.headerRows.includes(row)) {
    // header rows hatched (study § 6)
    style.backgroundImage = `repeating-linear-gradient(135deg, color-mix(in srgb, ${tone} 22%, transparent) 0 3px, transparent 3px 7px)`;
  }
  if (row === area.top) style.borderTop = edge;
  if (row === area.bottom) style.borderBottom = edge;
  if (col === area.left) style.borderLeft = edge;
  if (col === area.right) style.borderRight = edge;
  return style;
}

type SheetGridProps = {
  cells: (string | null)[][];
  areas: GridArea[];
  activeId: string | null;
  onPick: (id: string) => void;
  label: string;
};

/** The sheet as it is (080b, left pane): each detected table outlined and tinted, header rows
 * hatched, rows set aside struck through; a click in a table shows its result. */
export function SheetGrid({ cells, areas, activeId, onPick, label }: SheetGridProps) {
  const width = Math.max(0, ...cells.map((row) => row.length));
  const areaAt = (row: number, col: number) =>
    areas.find((a) => row >= a.top && row <= a.bottom && col >= a.left && col <= a.right) ?? null;
  return (
    <div className="max-h-[52dvh] overflow-auto rounded-md border border-line" role="region" aria-label={label}>
      <table className="w-max border-collapse font-mono text-[11.5px] tabular-nums">
        <thead className="sticky top-0 z-10 bg-surface-2 text-ink-3">
          <tr>
            <th scope="col" className="sticky left-0 z-10 bg-surface-2 px-1.5" />
            {Array.from({ length: width }, (_, index) => (
              <th key={index} scope="col" className="min-w-10 px-1.5 py-0.5 text-center font-normal">
                {columnLetter(index + 1)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cells.map((row, r) => (
            <tr key={r}>
              <th scope="row" className="sticky left-0 bg-surface-2 px-1.5 text-right font-normal text-ink-3">
                {r + 1}
              </th>
              {Array.from({ length: width }, (_, c) => {
                const value = row[c] ?? null;
                const area = areaAt(r + 1, c + 1);
                const dropped = area?.droppedRows.includes(r + 1) ?? false;
                return (
                  <td
                    key={c}
                    title={value ?? undefined}
                    style={area ? cellStyle(area, r + 1, c + 1, area.id === activeId) : undefined}
                    onClick={area ? () => onPick(area.id) : undefined}
                    className={`max-w-40 truncate border border-line/60 px-1.5 py-0.5 ${area ? "cursor-pointer" : ""} ${dropped ? "text-ink-3 line-through" : area ? "text-ink-1" : "text-ink-2"}`}
                  >
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
