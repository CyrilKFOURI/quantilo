import type { MlRun } from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { t } from "../../lib/i18n";
import { InfoTip } from "../../ui/InfoTip";

type Conformal = NonNullable<MlRun["interval"]>;
/** The interval a model card carries (Predict on new data). */
type CardInterval = { level: number; half_width: number; errors: number; method: "split" | "cqr"; coverage?: number | null };

const share = (value: number) => `${(value * 100).toFixed(1).replace(/\.0$/, "")} %`;
const level = (value: number) => `${Math.round(value * 100)} %`;

/** The prediction interval of a number (057), under the regression results: its width, and the
 *  coverage checked on held-out rows that did not set it (R-ML-CONF-01 to 03). */
export function IntervalNote({ interval }: { interval: Conformal | null | undefined }) {
  if (!interval) return null;
  const below = interval.coverage !== null && interval.coverage < interval.level - interval.tolerance;
  const width =
    interval.method === "cqr"
      ? t("ml.interval.adaptive", { width: formatNumber(interval.mean_width) })
      : t("ml.interval.fixed", { width: formatNumber(interval.half_width) });
  return (
    <div className="space-y-1 text-[13px]" data-testid="interval-note">
      <p className="text-ink-1">
        {t("ml.interval.sentence", { level: level(interval.level), width })}{" "}
        {interval.coverage !== null && (
          <span className={below ? "text-warn" : "text-ink-2"}>
            {t(below ? "ml.interval.coverageLow" : "ml.interval.coverage", {
              coverage: share(interval.coverage),
              rows: interval.check_rows.toLocaleString("en-US"),
              level: level(interval.level),
            })}
          </span>
        )}
        {(interval.rules ?? []).length > 0 && <InfoTip label={t("ml.interval.why")} text={(interval.rules ?? []).join(" ")} />}
      </p>
      {below && <p className="text-xs text-ink-2">{t("ml.interval.lowHint")}</p>}
    </div>
  );
}

/** The tile value of a prediction's interval on new data: ± a width, or adaptive (CQR). */
export function intervalValue(interval: CardInterval): string {
  return interval.method === "cqr" ? t("ml.interval.adaptiveShort") : `± ${formatNumber(interval.half_width)}`;
}

/** Its explanation: the coverage checked on the validation, when known. */
export function intervalExplain(interval: CardInterval): string {
  const base =
    interval.method === "cqr"
      ? t("ml.interval.cardAdaptive", { level: level(interval.level), errors: interval.errors.toLocaleString("en-US") })
      : t("ml.interval.cardSplit", { level: level(interval.level), errors: interval.errors.toLocaleString("en-US") });
  return interval.coverage !== null && interval.coverage !== undefined ? `${base} ${t("ml.interval.cardCoverage", { coverage: share(interval.coverage) })}` : base;
}
