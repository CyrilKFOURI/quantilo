import { ChevronDown } from "lucide-react";
import { createContext, useContext, useId, useMemo, useState, type ReactNode } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { TipBox } from "../../ui/InfoTip";
import { SegmentedControl } from "../../ui/SegmentedControl";
import type { Step, Suggestion } from "./api";
import { FOLD_OVER, PARTS, qualityBars, shareOf, type ColumnQuality, type Part, type QualityBar } from "./quality";

/** N37, the quality bar of each column (Power Query's Valid / Error / Empty, Dataprep's green /
 *  red / grey bar). Four parts in a fixed order, the status colours of the tokens (UI_UX: `--ok`,
 *  `--warn`, `--bad` are for data quality), never alone: the unreadable part is hatched, every
 *  part is named with its cells in the tip, the bar's name and the table view. */

/** Tailwind sees whole class names only: one per part, written out. */
const FILL: Record<Part, string> = {
  valid: "bg-ok",
  toFix: "bg-warn",
  missing: "bg-ink-3/45",
  unreadable: "bg-[repeating-linear-gradient(135deg,var(--bad)_0_2px,transparent_2px_4px)] ring-1 ring-bad ring-inset",
};

const LABEL: Record<Part, MessageKey> = {
  valid: "clean.quality.part.valid",
  toFix: "clean.quality.part.toFix",
  missing: "clean.quality.part.missing",
  unreadable: "clean.quality.part.unreadable",
};

const HINT: Record<Part, MessageKey> = {
  valid: "clean.quality.partHint.valid",
  toFix: "clean.quality.partHint.toFix",
  missing: "clean.quality.partHint.missing",
  unreadable: "clean.quality.partHint.unreadable",
};

const QualityContext = createContext<Map<string, QualityBar> | null>(null);

/** Given once by the page: the measure (null while the suggestions are computed), every fix found
 *  and the draft, so that each bar follows the draft's ticks. */
export function QualityProvider({ quality, found, steps, children }: { quality: ColumnQuality[] | null | undefined; found: Suggestion[]; steps: Step[]; children: ReactNode }) {
  const bars = useMemo(() => (quality ? new Map(qualityBars(quality, found, steps).map((bar) => [bar.column, bar])) : null), [quality, found, steps]);
  return <QualityContext.Provider value={bars}>{children}</QualityContext.Provider>;
}

function useBars(): QualityBar[] {
  const bars = useContext(QualityContext);
  return bars ? [...bars.values()] : [];
}

function useBar(column: string): QualityBar | null {
  return useContext(QualityContext)?.get(column) ?? null;
}

/** The section of a column in the "By column" view, which the overview scrolls to. */
export function columnAnchor(column: string): string {
  return `clean-column-${[...column].map((char) => (/[\w-]/.test(char) ? char : `_${char.codePointAt(0)}`)).join("")}`;
}

/** "orders: 92 % valid, 5 % to fix, 3 % empty, 0 % unreadable". */
function barLabel(bar: QualityBar): string {
  return t("clean.quality.bar", {
    column: bar.column,
    valid: shareOf(bar.parts.valid, bar.rows),
    toFix: shareOf(bar.parts.toFix, bar.rows),
    missing: shareOf(bar.parts.missing, bar.rows),
    unreadable: bar.uncounted ? t("clean.quality.notCounted") : shareOf(bar.parts.unreadable, bar.rows),
  });
}

function Swatch({ part }: { part: Part }) {
  return <span className={`inline-block h-2 w-3 shrink-0 rounded-full ${FILL[part]}`} aria-hidden />;
}

/** The thin bar (8 px, 4 px in a table header): one block per part above 0, a 2 px gap between. */
function Bar({ bar, dense = false }: { bar: QualityBar; dense?: boolean }) {
  return (
    <span className={`flex w-full gap-0.5 ${dense ? "h-1" : "h-2"}`} aria-hidden>
      {bar.rows === 0 ? (
        <span className="h-full w-full rounded-full bg-surface-2" />
      ) : (
        PARTS.filter((part) => bar.parts[part] > 0).map((part) => <span key={part} className={`h-full min-w-1 rounded-full ${FILL[part]}`} style={{ flexGrow: bar.parts[part], flexBasis: 0 }} />)
      )}
    </span>
  );
}

/** What the tip of a bar says: each part with its cells and share, and how it is counted. */
function TipBody({ bar }: { bar: QualityBar }) {
  return (
    <span className="block space-y-1.5">
      <span className="block font-mono text-ink-1">{bar.column}</span>
      <span className="block space-y-0.5 tabular-nums">
        {PARTS.map((part) => (
          <span key={part} className="flex items-center gap-2">
            <Swatch part={part} />
            <span className="flex-1 text-ink-1">{t(LABEL[part])}</span>
            <span>{part === "unreadable" && bar.uncounted ? t("clean.quality.notCounted") : t("clean.quality.count", { count: formatCount(bar.parts[part]), share: shareOf(bar.parts[part], bar.rows) })}</span>
          </span>
        ))}
      </span>
      {bar.fixed > 0 && <span className="block">{t("clean.quality.fixed", { count: formatCount(bar.fixed) })}</span>}
      {bar.uncounted && <span className="block">{t("clean.quality.uncounted")}</span>}
      <span className="block text-ink-3">{t("clean.quality.once")}</span>
    </span>
  );
}

/** A column's bar with its tip, on hover or keyboard focus (the bar is a stop of Tab). */
export function QualityMeter({ bar, dense = false, className = "" }: { bar: QualityBar; dense?: boolean; className?: string }) {
  const id = useId();
  return (
    <span className={`group relative flex items-center ${className}`}>
      <span role="img" tabIndex={0} aria-label={barLabel(bar)} aria-describedby={id} className={`flex w-full items-center rounded-sm focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent ${dense ? "py-0.5" : "py-1"}`}>
        <Bar bar={bar} dense={dense} />
      </span>
      <TipBox id={id}>
        <TipBody bar={bar} />
      </TipBox>
    </span>
  );
}

/** The bar of a column where one is shown (a section of "By column", a header of a preview);
 *  nothing while the measure is not there or for a column it does not know (added by a step). */
export function ColumnBar({ column, dense = false, withShare = false, className = "" }: { column: string; dense?: boolean; withShare?: boolean; className?: string }) {
  const bar = useBar(column);
  if (!bar) return null;
  if (!withShare) return <QualityMeter bar={bar} dense={dense} className={className} />;
  return (
    <span className={`flex items-center gap-2 ${className}`}>
      <QualityMeter bar={bar} dense={dense} className="w-32 shrink-0" />
      <span className="text-xs whitespace-nowrap text-ink-3 tabular-nums">{t("clean.quality.validShare", { share: shareOf(bar.parts.valid, bar.rows) })}</span>
    </span>
  );
}

/** The parts' names in the order they are drawn (no numbers: those are in each tip and the table). */
function Legend() {
  return (
    <ul aria-label={t("clean.quality.legend")} className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-ink-2">
      {PARTS.map((part) => (
        <li key={part} className="flex items-center gap-1.5" title={t(HINT[part])}>
          <Swatch part={part} />
          {t(LABEL[part])}
        </li>
      ))}
    </ul>
  );
}

type View = "bars" | "table";

/** N37: at the head of "By column", a line per column with its bar; folded at first past
 *  `FOLD_OVER` columns (onion rule). A line opens its details (the values not read, its fixes). */
export function QualityOverview() {
  const bars = useBars();
  const [open, setOpen] = useState<boolean | null>(null);
  const [view, setView] = useState<View>("bars");
  const id = useId();
  if (!bars.length) return null;
  const shown = open ?? bars.length <= FOLD_OVER;
  const flagged = bars.filter((bar) => bar.parts.toFix + bar.parts.missing + bar.parts.unreadable > 0).length;
  return (
    <section aria-labelledby={`${id}-title`} className="rounded-lg border border-line bg-surface-1">
      <button type="button" aria-expanded={shown} aria-controls={id} onClick={() => setOpen(!shown)} className="flex min-h-11 w-full min-w-0 items-center gap-2 px-3 py-2 text-left">
        <ChevronDown size={15} strokeWidth={1.5} className={`shrink-0 text-ink-3 transition-transform ${shown ? "" : "-rotate-90"}`} aria-hidden />
        <span id={`${id}-title`} className="shrink-0 text-[13px] font-medium text-ink-1">
          {t("clean.quality.title")}
        </span>
        <span className="min-w-0 truncate text-xs text-ink-3 tabular-nums">{t("clean.quality.hint", { count: formatCount(bars.length), flagged: formatCount(flagged) })}</span>
      </button>
      <div id={id} hidden={!shown} className="space-y-3 border-t border-line px-3 py-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <Legend />
          <SegmentedControl
            label={t("clean.quality.view")}
            value={view}
            options={[
              { value: "bars", label: t("clean.quality.view.bars") },
              { value: "table", label: t("clean.quality.view.table") },
            ]}
            onChange={setView}
          />
        </div>
        {view === "bars" ? (
          <ul className="divide-y divide-line">
            {bars.map((bar) => (
              <OverviewRow key={bar.column} bar={bar} />
            ))}
          </ul>
        ) : (
          <QualityTable bars={bars} />
        )}
        <p className="max-w-prose text-xs text-ink-3">{t("clean.quality.measured")}</p>
      </div>
    </section>
  );
}

function OverviewRow({ bar }: { bar: QualityBar }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const unreadable = bar.parts.unreadable;
  return (
    <li className="py-1">
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)_7rem]">
        <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)} aria-label={t("clean.quality.details", { column: bar.column })} className="flex min-h-8 min-w-0 items-center gap-1.5 rounded-sm text-left">
          <ChevronDown size={14} strokeWidth={1.5} className={`shrink-0 text-ink-3 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
          <span className="min-w-0 truncate font-mono text-[12px] text-ink-1">{bar.column}</span>
        </button>
        <QualityMeter bar={bar} className="col-span-2 row-start-2 sm:col-span-1 sm:col-start-2 sm:row-start-1" />
        <span className="col-start-2 row-start-1 text-right text-xs whitespace-nowrap text-ink-2 tabular-nums sm:col-start-3">
          {t("clean.quality.validShare", { share: shareOf(bar.parts.valid, bar.rows) })}
        </span>
      </div>
      {open && (
        <div id={id} className="space-y-2 pt-1.5 pb-2 pl-5 text-xs text-ink-2">
          <p className="tabular-nums">
            {PARTS.map((part) => `${t(LABEL[part])} ${part === "unreadable" && bar.uncounted ? t("clean.quality.notCounted") : formatCount(bar.parts[part])}`).join(" · ")}
          </p>
          {unreadable > 0 && bar.unreadableValues.length > 0 && (
            <div>
              <p>{t("clean.quality.values", { reading: t(`clean.quality.reading.${bar.reading ?? "number"}` as MessageKey) })}</p>
              <ul className="mt-1 flex flex-wrap gap-1.5">
                {bar.unreadableValues.map((value) => (
                  <li key={value.value} className="rounded-sm border border-line bg-surface-0 px-1.5 py-0.5 font-mono text-[11px] text-ink-1">
                    {value.value} <span className="text-ink-3 tabular-nums">× {formatCount(value.rows)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {bar.uncounted && <p>{t("clean.quality.uncounted")}</p>}
          {bar.pending.length > 0 ? (
            <Button variant="ghost" className="-ml-3" onClick={() => document.getElementById(columnAnchor(bar.column))?.scrollIntoView({ block: "start", behavior: "smooth" })}>
              {t("clean.quality.fixes", { count: formatCount(bar.pending.length) })}
            </Button>
          ) : (
            <p className="text-ink-3">{t("clean.quality.noFix")}</p>
          )}
        </div>
      )}
    </li>
  );
}

/** The same numbers as a table (dataviz: every chart has one), for a screen reader or a copy. */
function QualityTable({ bars }: { bars: QualityBar[] }) {
  return (
    <div className="max-h-[28rem] overflow-auto rounded-md border border-line">
      <table className="w-full border-collapse text-[12px] tabular-nums">
        <caption className="sr-only">{t("clean.quality.caption")}</caption>
        <thead className="sticky top-0 bg-surface-1">
          <tr className="border-b border-line text-left text-[11px] text-ink-3">
            <th scope="col" className="px-2 py-1.5 font-normal">
              {t("clean.quality.col.column")}
            </th>
            {PARTS.map((part) => (
              <th key={part} scope="col" className="px-2 py-1.5 text-right font-normal">
                {t(LABEL[part])}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {bars.map((bar) => (
            <tr key={bar.column} className="border-b border-line last:border-b-0">
              <th scope="row" className="max-w-56 truncate px-2 py-1 text-left font-mono font-normal text-ink-1">
                {bar.column}
              </th>
              {PARTS.map((part) => (
                <td key={part} className="px-2 py-1 text-right whitespace-nowrap text-ink-2">
                  {part === "unreadable" && bar.uncounted ? t("clean.quality.notCounted") : t("clean.quality.count", { count: formatCount(bar.parts[part]), share: shareOf(bar.parts[part], bar.rows) })}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
