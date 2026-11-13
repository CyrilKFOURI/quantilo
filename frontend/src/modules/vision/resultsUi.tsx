import { useState, type ReactNode } from "react";

import { imageUrl } from "../../features/import/imageSetApi";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { InfoTip } from "../../ui/InfoTip";
import type { ImageRow } from "./model";

/** 113: the pieces the Results tab is made of (tiles, metric tables, sections, galleries). */

export type Tile = { label: string; value: string; detail?: string; tip?: string; main?: boolean };

/** The main numbers, in a strip of 1 px separated cells (as the set's figures, 107): the main
 *  metric first and a little larger (R-113-07). Two per row on a phone. */
export function Tiles({ tiles, label }: { tiles: Tile[]; label: string }) {
  return (
    <dl aria-label={label} className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,9.5rem),1fr))] gap-px overflow-hidden rounded-md border border-line bg-line max-sm:grid-cols-2 max-sm:[&>div:last-child:nth-child(odd)]:col-span-2">
      {tiles.map((tile) => (
        <div key={tile.label} className="min-w-0 bg-surface-1 px-3.5 py-3">
          <dt className="flex items-center gap-1 text-xs text-ink-3">
            <span className="truncate">{tile.label}</span>
            {tile.tip && <InfoTip label={tile.label} text={tile.tip} />}
          </dt>
          <dd className={`mt-0.5 font-semibold text-ink-1 tabular-nums ${tile.main ? "text-2xl" : "text-xl"}`}>{tile.value}</dd>
          {tile.detail && <dd className="mt-0.5 truncate text-xs text-ink-3 tabular-nums" title={tile.detail}>{tile.detail}</dd>}
        </div>
      ))}
    </dl>
  );
}

export type MetricRow = { label: string; value: string; detail?: string; tip?: string };

/** Every metric, in two columns on a laptop: name (with its explanation), value, interval or
 *  reason. Numbers right-aligned and tabular. */
export function MetricTable({ rows, caption }: { rows: MetricRow[]; caption: string }) {
  return (
    <table className="w-full border-collapse text-[13px]">
      <caption className="sr-only">{caption}</caption>
      <tbody className="divide-y divide-line border-y border-line">
        {rows.map((row) => (
          <tr key={row.label}>
            <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink-2">
              <span className="inline-flex items-center gap-1">
                {row.label}
                {row.tip && <InfoTip label={row.label} text={row.tip} />}
              </span>
            </th>
            <td className="py-1.5 text-right font-medium whitespace-nowrap text-ink-1 tabular-nums">{row.value}</td>
            <td className="hidden py-1.5 pl-3 text-right text-xs whitespace-nowrap text-ink-3 tabular-nums sm:table-cell">{row.detail ?? ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A titled block of the tab; `conclusion` is the sentence that says what the chart shows
 *  (Datawrapper style), above it. */
export function Block({ id, title, conclusion, children, aside }: { id: string; title: string; conclusion?: ReactNode; children: ReactNode; aside?: ReactNode }) {
  return (
    <section aria-labelledby={id} className="min-w-0">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 id={id} className="text-[13px] font-medium text-ink-1">
          {title}
        </h3>
        {aside}
      </div>
      {conclusion && <p className="mt-1 max-w-[72ch] text-[13px] text-pretty text-ink-2">{conclusion}</p>}
      <div className="mt-2.5">{children}</div>
    </section>
  );
}

export type GalleryItem = { row: number; caption: string; detail?: string; tone?: "bad" | "warn" | "ok" };

const TONE = { bad: "text-bad", warn: "text-warn", ok: "text-ok" } as const;

/** A strip of thumbnails behind a list of the run's rows (errors, most unusual...). The first 12
 *  show, the rest on demand; a click opens the image. */
export function Gallery({
  datasetId,
  rows,
  items,
  empty,
  onOpen,
  label,
}: {
  datasetId: string;
  rows: readonly ImageRow[];
  items: GalleryItem[];
  empty: string;
  onOpen: (row: number, list: number[]) => void;
  label: string;
}) {
  const [all, setAll] = useState(false);
  if (items.length === 0) return <p className="text-[13px] text-ink-3">{empty}</p>;
  const shown = all ? items : items.slice(0, 12);
  const list = items.map((item) => item.row);
  return (
    <div>
      <ul aria-label={label} className="grid grid-cols-[repeat(auto-fill,minmax(112px,1fr))] gap-2 max-sm:grid-cols-3">
        {shown.map((item) => {
          const image = rows[item.row];
          return (
            <li key={item.row} className="min-w-0">
              <button
                type="button"
                onClick={() => onOpen(item.row, list)}
                className="group block w-full text-left focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
                aria-label={`${image?.name ?? item.row}: ${item.caption}${item.detail ? `, ${item.detail}` : ""}`}
              >
                <span className="block aspect-square overflow-hidden rounded-[5px] bg-surface-2 outline outline-1 -outline-offset-1 outline-ink-1/10">
                  {image && (
                    <img
                      src={imageUrl(datasetId, image.id)}
                      alt=""
                      loading="lazy"
                      decoding="async"
                      className="size-full object-cover transition-transform duration-200 group-hover:scale-[1.03] motion-reduce:transition-none"
                    />
                  )}
                </span>
                <span className={`mt-1 block truncate text-xs ${item.tone ? TONE[item.tone] : "text-ink-1"}`}>{item.caption}</span>
                {item.detail && <span className="block truncate text-xs text-ink-3 tabular-nums">{item.detail}</span>}
              </button>
            </li>
          );
        })}
      </ul>
      {items.length > 12 && (
        <button type="button" onClick={() => setAll((value) => !value)} className="mt-2 text-[13px] text-accent hover:underline">
          {all ? t("vision.results.showFewer") : t("vision.results.showAll", { count: formatCount(items.length) })}
        </button>
      )}
    </div>
  );
}
