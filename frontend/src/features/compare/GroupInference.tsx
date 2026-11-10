import { useState } from "react";

import type { Comparison, CompareGroup, GroupDifference } from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { MultipleTestingNote, NoLongerHolds, noLongerHolds } from "../inference/MultipleTesting";
import { formatP } from "../variable/normality";

const DIFFERENCE_ROWS = 20;

const interval = (low: number | null, high: number | null) =>
  low === null || high === null ? "—" : `[${formatNumber(low)} ; ${formatNumber(high)}]`;

const groupLabel = (group: CompareGroup) => (group.color ? `${group.label} · ${group.color}` : group.label);

function methodLabel(method: string | null): string {
  if (method === "bootstrap") return t("compare.intervals.bootstrap");
  if (method === "welch") return t("compare.intervals.welch");
  return t("compare.intervals.student");
}

/** Each group's mean and median with their 95% intervals, and what gave each interval (R-BS-06). */
export function GroupIntervals({ result }: { result: Comparison }) {
  if (result.groups.length === 0) return null;
  return (
    <section className="mt-4 space-y-2" aria-labelledby="compare-intervals">
      <h3 id="compare-intervals" className="text-sm font-semibold">
        {t("compare.intervals.title")}
      </h3>
      {result.intervals && <p className="text-[13px] text-ink-2">{result.intervals}</p>}
      <div className="rounded-md border border-line">
        {/* 041: a phone keeps the group, the mean and the method (n and median hidden). */}
        <table className="w-full border-collapse text-[13px] max-sm:[&_tr>:nth-child(2)]:hidden max-sm:[&_tr>:nth-child(4)]:hidden">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("compare.intervals.group")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">n</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("compare.intervals.mean")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("compare.intervals.median")}</th>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("compare.intervals.method")}</th>
            </tr>
          </thead>
          <tbody>
            {result.groups.map((group) => (
              <tr key={`${group.label}|${group.color ?? ""}`} className="border-t border-line align-top">
                <td className="px-3 py-1.5 font-mono text-xs text-ink-1">{groupLabel(group)}</td>
                <td className="px-3 py-1.5 text-right text-ink-3 tabular-nums">{formatCount(group.n)}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  <span className="text-ink-1">{formatNumber(group.mean)}</span>{" "}
                  <span className="text-xs whitespace-nowrap text-ink-3">{interval(group.ci_low, group.ci_high)}</span>
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums" title={group.median_reason ?? undefined}>
                  <span className="text-ink-1">{formatNumber(group.median)}</span>{" "}
                  <span className="text-xs whitespace-nowrap text-ink-3">{interval(group.median_low, group.median_high)}</span>
                </td>
                <td className="px-3 py-1.5" title={group.ci_reason ?? undefined}>
                  <span className={group.ci_method === "bootstrap" ? "text-ink-1" : "text-ink-2"}>{methodLabel(group.ci_method)}</span>
                  {/* 041: on a phone the reason stays in the cell's title, the group name keeps its width. */}
                  {group.ci_reason && <span className="block text-[11px] text-ink-3 max-sm:hidden">{group.ci_reason}</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const byAdjusted = (a: GroupDifference, b: GroupDifference) => (a.p_adjusted ?? 1) - (b.p_adjusted ?? 1);

/** Every pair of groups: B − A with its interval, Welch's p and the p adjusted for testing every pair. */
export function PairwiseDifferences({ result }: { result: Comparison }) {
  const [all, setAll] = useState(false);
  if (result.differences.length === 0) return null;
  const sorted = [...result.differences].sort(byAdjusted);
  const shown = all ? sorted : sorted.slice(0, DIFFERENCE_ROWS);
  return (
    <section className="mt-4 space-y-2" aria-labelledby="compare-differences">
      <h3 id="compare-differences" className="text-sm font-semibold">
        {t("compare.differences.title", { count: String(result.differences.length) })}
      </h3>
      {result.multiple && <MultipleTestingNote multiple={result.multiple} />}
      <div className="rounded-md border border-line">
        {/* 041: a phone keeps the pair, the difference, its interval and adj. p (p and method hidden). */}
        <table className="w-full border-collapse text-[13px] max-sm:[&_tr>:nth-child(4)]:hidden max-sm:[&_tr>:nth-child(6)]:hidden">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("compare.differences.pair")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("compare.differences.diff")}</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">95% CI</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">p</th>
              <th scope="col" className="px-3 py-1.5 text-right font-normal">adj. p</th>
              <th scope="col" className="px-3 py-1.5 font-normal">{t("compare.intervals.method")}</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((d) => (
              <tr key={`${d.a}|${d.b}`} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs text-ink-1">
                  {d.b} − {d.a}
                </td>
                <td className="px-3 py-1.5 text-right text-ink-1 tabular-nums">{formatNumber(d.diff)}</td>
                <td className="px-3 py-1.5 text-right text-xs text-ink-3 tabular-nums">{interval(d.low, d.high)}</td>
                <td className="px-3 py-1.5 text-right whitespace-nowrap text-ink-3 tabular-nums">{formatP(d.p_value)}</td>
                <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">
                  <span className="whitespace-nowrap">{formatP(d.p_adjusted)}</span>
                  {noLongerHolds(d.p_value, d.p_adjusted) && <NoLongerHolds />}
                </td>
                <td className="px-3 py-1.5 text-ink-2" title={d.reason}>
                  {methodLabel(d.interval)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {sorted.length > DIFFERENCE_ROWS && (
        <button type="button" onClick={() => setAll(!all)} className="text-xs text-accent hover:underline">
          {all ? t("compare.differences.fewer") : t("compare.differences.all", { count: String(sorted.length) })}
        </button>
      )}
    </section>
  );
}
