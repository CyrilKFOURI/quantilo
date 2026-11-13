import { ChevronDown } from "lucide-react";
import { useMemo, useState } from "react";

import { Plot, type PointClick } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { formatBytes, formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { LITE } from "../../lite/flag";
import { BLUR_SHARPNESS } from "../../lite/images";
import { Checkbox } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { histogramFigure, labelsFigure, sizeFigure } from "./figures";
import {
  binRows,
  countBy,
  isExcluded,
  labelSeries,
  quantile,
  sortedValues,
  type Draft,
  type ImageRow,
  type Measure,
  type QualityFilter,
  type Range,
} from "./model";

const CHART_HEIGHT = 168;
// R-VQ-04's absolute blur threshold: 100 at 512 pixels, 300 at the browser's 256 (R-VIS-WEB-02)
const BLUR = LITE ? BLUR_SHARPNESS : 100;
const HISTOGRAMS: { measure: Measure; title: MessageKey; unit: MessageKey }[] = [
  { measure: "brightness", title: "vision.measure.brightness", unit: "vision.stats.unit.brightness" },
  { measure: "contrast", title: "vision.measure.contrast", unit: "vision.stats.unit.contrast" },
  { measure: "sharpness", title: "vision.measure.sharpness", unit: "vision.stats.unit.sharpness" },
  { measure: "saturation", title: "vision.measure.saturation", unit: "vision.stats.unit.saturation" },
];

type StripProps = {
  rows: readonly ImageRow[];
  draft: Draft;
  onQuality: (quality: QualityFilter | undefined) => void;
  quality: QualityFilter | undefined;
};

/** The set in a row of figures (§ 4.3); a figure that names a subset filters the gallery on it. */
export function StatsStrip({ rows, draft, onQuality, quality }: StripProps) {
  const labels = countBy(rows, (row) => row.label);
  labels.delete(null);
  const splits = countBy(rows, (row) => row.split);
  splits.delete(null);
  const formats = [...countBy(rows, (row) => row.format).entries()]
    .filter(([format]) => format !== null)
    .sort((a, b) => b[1] - a[1])
    .map(([format]) => format)
    .join(", ");
  const withIssue = rows.filter((row) => row.issues.length > 0).length;
  const excluded = rows.filter((row) => isExcluded(row, draft)).length;
  const bytes = rows.reduce((sum, row) => sum + (row.bytes ?? 0), 0);
  const plain: [string, string][] = [
    [t("vision.stats.images"), formatCount(rows.length)],
    [t("vision.stats.labels"), formatCount(labels.size)],
    [t("vision.stats.splits"), splits.size > 0 ? [...splits.keys()].join(", ") : t("vision.none")],
    [t("vision.stats.formats"), formats || "—"],
    [t("vision.stats.size"), formatBytes(bytes)],
  ];
  const filtering: [string, number, QualityFilter][] = [
    [t("vision.stats.withIssue"), withIssue, "any"],
    [t("vision.stats.excluded"), excluded, "excluded"],
  ];
  return (
    <dl className="flex flex-wrap gap-px overflow-hidden rounded-md border border-line bg-line text-xs">
      {plain.map(([label, value]) => (
        <div key={label} className="min-w-0 flex-[1_1_110px] bg-surface-1 px-3 py-2">
          <dt className="text-ink-3">{label}</dt>
          <dd className="truncate text-[13px] font-medium text-ink-1 tabular-nums" title={value}>
            {value}
          </dd>
        </div>
      ))}
      {filtering.map(([label, count, filter]) => {
        const on = quality === filter;
        return (
          <div key={label} className="min-w-0 flex-[1_1_130px] bg-surface-1">
            <button
              type="button"
              aria-pressed={on}
              disabled={count === 0 && !on}
              onClick={() => onQuality(on ? undefined : filter)}
              className={`block size-full px-3 py-2 text-left transition-colors enabled:hover:bg-surface-2 disabled:cursor-default ${on ? "bg-accent-soft" : ""}`}
            >
              <dt className="text-ink-3">{label}</dt>
              <dd className={`text-[13px] font-medium tabular-nums ${count > 0 ? "text-ink-1" : "text-ink-3"}`}>
                {formatCount(count)}
                {count > 0 && <span className="ml-1.5 text-xs font-normal text-accent">{on ? t("vision.stats.shown") : t("vision.stats.show")}</span>}
              </dd>
            </button>
          </div>
        );
      })}
    </dl>
  );
}

type StatsProps = {
  rows: readonly ImageRow[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onRange: (range: Range) => void;
  onLabel: (label: string) => void;
};

/** The conclusion above a histogram (Datawrapper): the median and the middle half of the set. */
export function measureSentence(measure: Measure, sorted: readonly number[]): string {
  const median = quantile(sorted, 0.5);
  const low = quantile(sorted, 0.25);
  const high = quantile(sorted, 0.75);
  if (median === null || low === null || high === null) return t("vision.stats.noValue");
  const vars = { median: formatNumber(median), low: formatNumber(low), high: formatNumber(high) };
  if (measure === "sharpness") {
    const blurry = sorted.filter((value) => value < BLUR).length;
    return t(blurry > 0 ? "vision.stats.sentence.sharpnessLow" : "vision.stats.sentence.sharpness", {
      ...vars,
      count: formatCount(blurry),
      threshold: formatCount(BLUR),
    });
  }
  return t("vision.stats.sentence.measure", vars);
}

export function balanceSentence(counts: readonly [string, number][]): string {
  if (counts.length < 2) return counts.length === 1 ? t("vision.stats.oneLabel") : t("vision.stats.noLabel");
  const [largest, most] = counts[0] ?? ["", 0];
  const [smallest, least] = counts[counts.length - 1] ?? ["", 1];
  const ratio = most / Math.max(least, 1);
  const vars = { largest, smallest, ratio: ratio.toFixed(1) };
  // R-VQ-13: over 3, accuracy would hide the rare class
  return t(ratio > 3 ? "vision.stats.imbalanced" : "vision.stats.balanced", vars);
}

/**
 * 107 (§ 4.3): the set's statistics, computed in the page from the index (nothing is asked of the
 * server). A click on a bar filters the gallery: a histogram on its range, the labels on one label.
 */
export function DatasetStats({ rows, open, onOpenChange, onRange, onLabel }: StatsProps) {
  const [byLabel, setByLabel] = useState(false);
  const labelCounts = useMemo(() => {
    const counts = [...countBy(rows, (row) => row.label).entries()].filter((entry): entry is [string, number] => entry[0] !== null);
    return counts.sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  }, [rows]);
  const series = useMemo(() => labelSeries(labelCounts.map(([label]) => label)), [labelCounts]);
  const colour = byLabel && series.length > 1;
  const histograms = useMemo(
    () =>
      HISTOGRAMS.map(({ measure, title, unit }) => {
        const bins = binRows(rows, measure, { labels: colour ? series : null });
        return { measure, title, unit, bins, sentence: measureSentence(measure, sortedValues(rows, measure)) };
      }),
    [rows, colour, series],
  );
  const splits = useMemo(() => {
    const names = [...new Set(rows.map((row) => row.split).filter((split): split is string => split !== null))].sort();
    const table = new Map<string, Map<string, number>>();
    for (const row of rows) {
      if (row.label === null || row.split === null) continue;
      const line = table.get(row.label) ?? new Map<string, number>();
      line.set(row.split, (line.get(row.split) ?? 0) + 1);
      table.set(row.label, line);
    }
    return { names, table };
  }, [rows]);
  const sized = rows.filter((row) => row.width && row.height).length;

  return (
    <section aria-label={t("vision.stats.title")} className="rounded-lg border border-line bg-surface-1">
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => onOpenChange(!open)}
          className="inline-flex min-h-7 items-center gap-1.5 text-[13px] font-medium text-ink-1 pointer-coarse:min-h-11"
        >
          <ChevronDown size={15} strokeWidth={1.5} aria-hidden className={`text-ink-3 transition-transform duration-150 ${open ? "" : "-rotate-90"}`} />
          {t("vision.stats.title")}
        </button>
        {open && series.length > 1 && (
          <Checkbox
            label={t("vision.stats.byLabel")}
            checked={byLabel}
            onChange={(event) => setByLabel(event.target.checked)}
            className="text-xs text-ink-2"
          />
        )}
      </div>
      {open && (
        <div className="space-y-5 border-t border-line px-3 pt-3 pb-4">
          <div className="grid grid-cols-1 gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
            <Card
              title={t("vision.stats.sizeTitle")}
              sentence={sizeSentence(rows)}
              note={sized > 5000 ? t("vision.stats.sampled", { count: formatCount(sized) }) : undefined}
            >
              <Plot figure={sizeFigure(rows)} height={CHART_HEIGHT + 24} label={t("vision.stats.sizeTitle")} />
            </Card>
            {histograms.map(({ measure, title, unit, bins, sentence }) => (
              <Card
                key={measure}
                title={t(title)}
                sentence={sentence}
                tip={measure === "sharpness" ? t("vision.stats.sharpnessTip") : undefined}
              >
                <Plot
                  figure={histogramFigure(bins, { unit: t(unit), series: colour ? series : null })}
                  height={CHART_HEIGHT}
                  label={t(title)}
                  onPointClick={(point: PointClick) => {
                    const range = point.customdata as [number, number] | undefined;
                    if (range) onRange({ measure, low: range[0], high: range[1] });
                  }}
                />
              </Card>
            ))}
            {labelCounts.length > 0 && (
              <Card title={t("vision.stats.labelsTitle")} sentence={balanceSentence(labelCounts)}>
                <Plot
                  figure={labelsFigure(labelCounts.slice(0, 12), series)}
                  height={Math.max(CHART_HEIGHT, Math.min(12, labelCounts.length) * 26 + 16)}
                  label={t("vision.stats.labelsTitle")}
                  onPointClick={(point) => typeof point.customdata === "string" && onLabel(point.customdata)}
                />
              </Card>
            )}
          </div>
          {splits.names.length > 1 && splits.table.size > 0 && (
            <div className="min-w-0 overflow-x-auto">
              <table className="w-full max-w-2xl text-xs">
                <caption className="mb-1 text-left text-[13px] font-medium text-ink-1">{t("vision.stats.splitTitle")}</caption>
                <thead>
                  <tr className="text-ink-3">
                    <th scope="col" className="py-1 pr-3 text-left font-normal">
                      {t("vision.field.label")}
                    </th>
                    {splits.names.map((name) => (
                      <th key={name} scope="col" className="py-1 pl-3 text-right font-mono font-normal">
                        {name}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[...splits.table.entries()]
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([label, counts]) => (
                      <tr key={label} className="border-t border-line">
                        <th scope="row" className="truncate py-1 pr-3 text-left font-mono font-normal text-ink-1">
                          {label}
                        </th>
                        {splits.names.map((name) => (
                          <td key={name} className="py-1 pl-3 text-right text-ink-2 tabular-nums">
                            {formatCount(counts.get(name) ?? 0)}
                          </td>
                        ))}
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="text-xs text-ink-3">{t("vision.stats.clickHint")}</p>
        </div>
      )}
    </section>
  );
}

function sizeSentence(rows: readonly ImageRow[]): string {
  const sized = rows.filter((row) => row.width && row.height);
  if (sized.length === 0) return t("vision.stats.noValue");
  const shapes = countBy(sized, (row) => `${row.width}×${row.height}`);
  const [top, count] = [...shapes.entries()].sort((a, b) => b[1] - a[1])[0] ?? ["", 0];
  if (count === sized.length) return t("vision.stats.sizeAll", { size: top.replace("×", " × ") });
  const share = Math.round((count / sized.length) * 100);
  return share >= 50
    ? t("vision.stats.sizeMost", { size: top.replace("×", " × "), share: `${share} %` })
    : t("vision.stats.sizeMixed", { count: formatCount(shapes.size) });
}

function Card({
  title,
  sentence,
  note,
  tip,
  children,
}: {
  title: string;
  sentence: string;
  note?: string;
  tip?: string;
  children: React.ReactNode;
}) {
  return (
    <figure className="min-w-0 space-y-1.5">
      <figcaption className="space-y-0.5">
        <span className="flex items-center gap-1 text-xs text-ink-3">
          {title}
          {tip && <InfoTip text={tip} label={title} />}
        </span>
        <span className="block text-[13px] leading-snug font-medium text-pretty text-ink-1">{sentence}</span>
      </figcaption>
      {children}
      {note && <p className="text-xs text-ink-3">{note}</p>}
    </figure>
  );
}
