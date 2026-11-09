import { useMemo, useState } from "react";

import type { Chart, Fit, Law, NumericProfile, Transform } from "../../api/client";
import { useChart, useFit } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { InfoTip } from "../../ui/InfoTip";
import { Select } from "../../ui/fields";
import { lawsFigure, miniHistogram, qqFigure } from "./figures";
import { formatP } from "./normality";
import { Loading } from "../../ui/Loading";

const OVERLAID = 3;

type FitPanelProps = { datasetId: string; column: string; profile: NumericProfile };

export function FitPanel({ datasetId, column, profile }: FitPanelProps) {
  const fit = useFit(datasetId, column, true);
  const chart = useChart(datasetId, column, { view: "histogram" }, true);

  if (fit.isError) return <p className="text-bad">{fit.error.message}</p>;
  if (!fit.data || !chart.data) {
    return (
      <Loading label={t("fit.loading")} className="h-72 rounded-md" />
    );
  }
  if (fit.data.reason || !fit.data.laws.length) {
    return <p className="text-ink-2">{t("fit.none", { reason: fit.data.reason ?? "" })}</p>;
  }
  return <FitResults fit={fit.data} count={profile.describe.count} column={column} chart={chart.data} />;
}

/** Laws ranked by AIC, overlaid on the histogram, a QQ-plot against any of them, and
 *  transformations. Also used for time series residuals. */
export function FitResults({
  fit,
  count,
  column,
  chart,
}: {
  fit: Fit;
  count: number;
  column: string;
  chart: Pick<Chart, "bins">;
}) {
  const [qqLaw, setQqLaw] = useState(fit.laws[0]?.key ?? "");
  const best = fit.laws[0] as Law;
  const top = useMemo(() => fit.laws.slice(0, OVERLAID), [fit.laws]);
  const overlay = useMemo(
    () => lawsFigure(chart, fit.grid, top, count),
    [chart, fit.grid, top, count],
  );
  const selected = fit.laws.find((law) => law.key === qqLaw) ?? best;
  const qq = useMemo(
    () => qqFigure(selected.quantiles, fit.sample_quantiles, { slope: 1, intercept: 0 }, `${selected.label} quantiles`),
    [selected, fit.sample_quantiles],
  );
  const logHelps = best.key === "lognormal" && fit.transforms.some((item) => item.key === "log");

  return (
    <div className="space-y-5">
      <p className="text-ink-1">
        <span className="font-medium">{t("fit.best", { law: best.label })}</span>
        {logHelps && <span className="text-ink-2"> {t("fit.bestLog")}</span>}
      </p>

      <div className="grid gap-4 lg:grid-cols-2">
        <figure className="min-w-0 rounded-md border border-line p-2">
          <figcaption className="mb-1 px-1 text-xs text-ink-3">{t("fit.overlay")}</figcaption>
          <Plot figure={overlay} height={280} label={t("fit.overlay")} />
        </figure>
        <figure className="min-w-0 rounded-md border border-line p-2">
          <figcaption className="mb-1 flex items-center gap-2 px-1 text-xs text-ink-3">
            {t("fit.qqAgainst")}
            <Select value={selected.key} onChange={(event) => setQqLaw(event.target.value)} className="h-7 text-xs">
              {fit.laws.map((law) => (
                <option key={law.key} value={law.key}>
                  {law.label}
                </option>
              ))}
            </Select>
          </figcaption>
          <Plot figure={qq} height={280} label={`${t("fit.qqAgainst")} ${selected.label}: ${column}`} />
        </figure>
      </div>

      <section>
        <h3 className="mb-1.5 text-xs font-medium text-ink-3">{t("fit.title")}</h3>
        <div className="overflow-x-auto rounded-md border border-line">
          <table className="w-full border-collapse text-[13px]">
            <thead className="bg-surface-2 text-left text-xs text-ink-3">
              <tr>
                <th scope="col" className="px-3 py-1.5 font-normal">#</th>
                <th scope="col" className="px-3 py-1.5 font-normal">{t("fit.law")}</th>
                <th scope="col" className="px-3 py-1.5 font-normal">{t("fit.params")}</th>
                <HeaderWithTip label={t("fit.aic")} tip={t("fit.explainAic")} />
                <HeaderWithTip label={t("fit.bic")} tip={t("fit.explainBic")} />
                <HeaderWithTip label={t("fit.ks")} tip={t("fit.explainKs")} />
              </tr>
            </thead>
            <tbody>
              {fit.laws.map((law, index) => (
                <tr key={law.key} className="border-t border-line">
                  <td className="px-3 py-1.5 text-ink-3 tabular-nums">{index + 1}</td>
                  <td className="px-3 py-1.5 whitespace-nowrap text-ink-1">
                    {index < OVERLAID && (
                      <span
                        aria-hidden
                        className="mr-1.5 inline-block size-2 rounded-full align-middle"
                        style={{ background: `var(--chart-${index + 1})` }}
                      />
                    )}
                    {law.label}
                    {law.discrete && <span className="ml-1.5 text-xs text-ink-3">{t("fit.discrete")}</span>}
                  </td>
                  <td className="px-3 py-1.5 font-mono text-xs text-ink-2">
                    {Object.entries(law.params)
                      .map(([name, value]) => `${name}=${formatNumber(value)}`)
                      .join(", ")}
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums">{formatNumber(law.aic)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">{formatNumber(law.bic)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-2 tabular-nums">{law.ks.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-1.5 text-xs text-ink-3">{t("fit.sample", { n: formatCount(fit.n) })}</p>
      </section>

      {fit.transforms.length > 0 && <Transforms transforms={fit.transforms} />}
    </div>
  );
}

function HeaderWithTip({ label, tip }: { label: string; tip: string }) {
  return (
    <th scope="col" className="px-3 py-1.5 text-right font-normal">
      <span className="inline-flex items-center gap-1">
        {label}
        <InfoTip label={label} text={tip} />
      </span>
    </th>
  );
}

function Transforms({ transforms }: { transforms: Transform[] }) {
  const mostSymmetric = transforms.reduce<Transform | null>((best, item) => {
    if (item.skewness_after === null) return best;
    if (!best || best.skewness_after === null) return item;
    return Math.abs(item.skewness_after) < Math.abs(best.skewness_after) ? item : best;
  }, null);

  return (
    <section>
      <div className="mb-1.5 flex flex-wrap items-baseline gap-x-3">
        <h3 className="text-xs font-medium text-ink-3">{t("fit.transforms")}</h3>
        <p className="text-xs text-ink-3">{t("fit.transformsHint")}</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {transforms.map((item) => (
          <TransformCard key={item.key} transform={item} best={item === mostSymmetric} />
        ))}
      </div>
    </section>
  );
}

function TransformCard({ transform, best }: { transform: Transform; best: boolean }) {
  const figure = useMemo(() => miniHistogram(transform.bins), [transform.bins]);
  const name = t(`fit.transform.${transform.key as "log" | "sqrt" | "boxcox" | "yeojohnson"}`);
  return (
    <article className={`rounded-md border p-2.5 ${best ? "border-accent" : "border-line"}`}>
      <header className="flex items-baseline justify-between gap-2">
        <h4 className="font-medium text-ink-1">{name}</h4>
        <span className="text-xs text-ink-3">
          {transform.lmbda !== null && t("fit.lambda", { value: transform.lmbda.toFixed(2) })}
          {best && <span className="ml-1.5 text-accent">{t("fit.bestTransform")}</span>}
        </span>
      </header>
      <Plot figure={figure} height={90} label={name} />
      <dl className="mt-1 grid grid-cols-[1fr_auto_auto] gap-x-2 text-xs tabular-nums">
        <dt className="text-ink-3">{t("fit.skewness")}</dt>
        <dd className="text-ink-3 line-through decoration-ink-3/50">{formatNumber(transform.skewness_before)}</dd>
        <dd className="text-ink-1">{formatNumber(transform.skewness_after)}</dd>
        <dt className="text-ink-3">{t("fit.shapiro")}</dt>
        <dd className="text-ink-3 line-through decoration-ink-3/50">{formatP(transform.shapiro_before)}</dd>
        <dd className="text-ink-1">{formatP(transform.shapiro_after)}</dd>
      </dl>
    </article>
  );
}
