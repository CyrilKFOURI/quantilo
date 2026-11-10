import type { Robust } from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { StatTable, type StatRow } from "./StatTable";

/** Robust center and spread (056, R-ROB-1): shown when outliers pull the mean, with the reason. */
export function RobustBlock({ robust }: { robust: Robust }) {
  const rows: StatRow[] = [
    { key: "median", label: t("variable.robust.median"), value: formatNumber(robust.median), explain: t("explain.median") },
    { key: "mad", label: t("variable.robust.mad"), value: formatNumber(robust.mad_sigma), explain: t("variable.robust.explainMad") },
    { key: "iqr", label: t("variable.robust.iqr"), value: formatNumber(robust.iqr_sigma), explain: t("variable.robust.explainIqr") },
    ...(robust.huber_center !== null
      ? [
          { key: "huberCenter", label: t("variable.robust.huberCenter"), value: formatNumber(robust.huber_center), explain: t("variable.robust.explainHuber") },
          { key: "huberScale", label: t("variable.robust.huberScale"), value: formatNumber(robust.huber_scale), explain: t("variable.robust.explainHuber") },
        ]
      : []),
  ];
  const shift = robust.mean_shift;
  return (
    <section className="space-y-2" aria-label={t("variable.robust.title")}>
      <div className="max-w-prose space-y-0.5">
        <p className="text-[13px] text-ink-1">{robust.reason}</p>
        {shift !== null && Math.abs(shift) >= 0.01 && (
          <p className="text-[13px] text-ink-2">
            {t(shift > 0 ? "variable.robust.above" : "variable.robust.below", {
              share: `${Math.abs(100 * shift).toFixed(0)}%`,
            })}
          </p>
        )}
        {robust.note && <p className="text-xs text-ink-3">{robust.note}</p>}
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatTable title={t("variable.robust.title")} rows={rows} />
      </div>
      {robust.sample > 0 && <p className="text-xs text-ink-3">{t("variable.robust.sample", { n: formatCount(robust.sample) })}</p>}
    </section>
  );
}
