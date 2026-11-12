import { useMemo } from "react";

import type { RegressionParams, RegressionResampling, RegressionSelection, RegressionStability } from "../../api/client";
import { useRegressionSection } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { cusumFigure, rollingFigure } from "./figures";
import { Sentence } from "./Sentence";

const SMALL_HEIGHT = 220;
const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums text-right";

function Pending({ error }: { error: Error | null }) {
  return error ? <ErrorNote>{error.message}</ErrorNote> : <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />;
}

const interval = (low: number | null | undefined, high: number | null | undefined) => `[${formatNumber(low ?? null)} ; ${formatNumber(high ?? null)}]`;

/** Advanced › Stability over time (073): CUSUM and CUSUM of squares, a Chow test at the strongest
 *  date, rolling coefficients and the factor reading (R-REG-15, R-REG-STAB-01…04). */
export function StabilitySection({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useRegressionSection(datasetId, "stability", params, true);
  const data: RegressionStability | undefined = query.data;
  const figures = useMemo(() => (data ? data.rolling.map((curve) => [curve, rollingFigure(curve)] as const) : []), [data]);
  const cusum = useMemo(() => (data?.cusum ? cusumFigure(data.cusum, "CUSUM") : null), [data]);
  const squares = useMemo(() => (data?.cusum_squares ? cusumFigure(data.cusum_squares, "CUSUM²") : null), [data]);
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-[13px] text-ink-2">{data.reason}</p>;
  return (
    <div className="space-y-3 text-[13px]">
      <p className={`max-w-prose ${data.stable ? "text-ink-2" : "text-warn"}`}>
        <Sentence text={data.sentence} />
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        {cusum && (
          <figure className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
            <figcaption className="mb-1 text-xs text-ink-2">{t("regression.stability.cusum")}</figcaption>
            <Plot figure={cusum} height={SMALL_HEIGHT} label={t("regression.stability.cusum")} />
          </figure>
        )}
        {squares && (
          <figure className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
            <figcaption className="mb-1 text-xs text-ink-2">{t("regression.stability.squares")}</figcaption>
            <Plot figure={squares} height={SMALL_HEIGHT} label={t("regression.stability.squares")} />
          </figure>
        )}
      </div>
      {data.chow && (
        <p className="max-w-prose text-ink-2">
          <Sentence text={data.chow.sentence} />
        </p>
      )}
      {figures.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-xs font-medium text-ink-2">{t("regression.stability.rolling")}</h3>
          <p className="text-xs text-ink-3">
            <Sentence text={data.window_reason} />
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {figures.map(([curve, figure]) => (
              <figure key={curve.term} className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
                <figcaption className="mb-1 text-xs text-ink-2">
                  <Sentence text={curve.sentence} />
                </figcaption>
                <Plot figure={figure} height={SMALL_HEIGHT} label={t("regression.stability.rollingOf", { term: curve.term })} />
              </figure>
            ))}
          </div>
          {data.rolled > data.rolling.length && <p className="text-xs text-ink-3">{t("regression.stability.shown", { shown: String(data.rolling.length), of: String(data.rolled) })}</p>}
        </div>
      )}
      {data.factor && (
        <div className="space-y-1">
          <h3 className="text-xs font-medium text-ink-2">{t("regression.stability.factor")}</h3>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.factor.sentence} />
          </p>
        </div>
      )}
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          <Sentence text={note} />
        </p>
      ))}
    </div>
  );
}

/** Advanced › Resampling (077): bootstrap intervals beside the first plane's, the wild cluster
 *  bootstrap with few clusters, a permutation test (R-REG-RS-01…05). */
export function ResamplingSection({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useRegressionSection(datasetId, "resampling", params, true);
  const data: RegressionResampling | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-[13px] text-ink-2">{data.reason}</p>;
  return (
    <div className="space-y-3 text-[13px]">
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <p className="text-xs text-ink-3">
        <Sentence text={data.scheme_reason} /> {t("regression.resampling.draws", { draws: formatCount(data.draws) })}
      </p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.term")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.estimate")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.resampling.formula")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.resampling.bootstrap")}</th>
              {data.wild && <th scope="col" className={`${th} text-right`}>{t("regression.resampling.wild")}</th>}
            </tr>
          </thead>
          <tbody>
            {data.terms.map((term) => (
              <tr key={term.term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">{term.term}</td>
                <td className={`${td} text-ink-1`}>{formatNumber(term.estimate)}</td>
                <td className={`${td} whitespace-nowrap text-ink-2`} title={`p ${formatP(term.p_value)}`}>{interval(term.low, term.high)}</td>
                <td className={`${td} whitespace-nowrap text-ink-1`} title={`SE ${formatNumber(term.boot_se)}`}>{interval(term.boot_low, term.boot_high)}</td>
                {data.wild && <td className={`${td} text-ink-2`}>{term.wild_p === null ? "—" : formatP(term.wild_p)}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        <Sentence text={data.wild_reason} />
      </p>
      <p className="text-ink-2">
        <span className="font-medium text-ink-1">{t("regression.resampling.permutation")}</span> ·{" "}
        {data.permutation ? <Sentence text={data.permutation.sentence} /> : <Sentence text={data.permutation_reason} />}
      </p>
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          <Sentence text={note} />
        </p>
      ))}
    </div>
  );
}

/** Advanced › More › Selection (077, R-REG-16, R-REG-SEL-01…04): asked for only once More is open. */
export function SelectionBlock({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useRegressionSection(datasetId, "selection", params, true);
  const data: RegressionSelection | undefined = query.data;
  return (
    <section id="regression-selection" className="space-y-2 border-t border-line pt-3 text-[13px]" aria-label={t("regression.selection.title")}>
      <div>
        <h3 className="text-[13px] font-medium text-ink-1">{t("regression.selection.title")}</h3>
        <p className="max-w-prose text-xs text-ink-2">{t("regression.selection.hint")}</p>
      </div>
      {!data ? (
        <Pending error={query.error} />
      ) : !data.available ? (
        <p className="text-ink-2">{data.reason}</p>
      ) : (
        <>
          <p className="max-w-prose text-ink-2">
            <Sentence text={data.sentence} />
          </p>
          <p className="text-xs text-ink-3">
            <Sentence text={data.penalty_reason} />
          </p>
          {data.double ? (
            <p className="max-w-prose text-ink-2">
              <Sentence text={data.double.sentence} />{" "}
              <span className="text-ink-3">{t("regression.selection.firstPlane", { value: formatNumber(data.double.ols), interval: interval(data.double.ols_low, data.double.ols_high) })}</span>
            </p>
          ) : (
            <p className="text-xs text-ink-3">
              <Sentence text={data.double_reason} />
            </p>
          )}
          <div className="max-h-96 overflow-auto rounded-md border border-line">
            <table className="w-full border-collapse">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th scope="col" className={th}>{t("regression.term")}</th>
                  <th scope="col" className={`${th} text-right`}>Lasso</th>
                  <th scope="col" className={`${th} text-right`}>{t("regression.selection.post")}</th>
                  <th scope="col" className={`${th} text-right`}>{t("regression.selection.share")}</th>
                </tr>
              </thead>
              <tbody>
                {data.terms.map((term) => (
                  <tr key={term.term} className="border-t border-line">
                    <td className={`px-3 py-1.5 font-mono text-xs ${term.kept ? "text-ink-1" : "text-ink-3"}`}>{term.term}</td>
                    <td className={`${td} ${term.kept ? "text-ink-1" : "text-ink-3"}`}>{term.kept ? formatNumber(term.lasso) : "0"}</td>
                    <td className={`${td} whitespace-nowrap text-ink-2`}>{term.post === null ? "—" : `${formatNumber(term.post)} ${interval(term.post_low, term.post_high)}`}</td>
                    <td className={`${td} ${term.stable ? "text-ink-1" : "text-ink-3"}`}>{term.share === null ? "—" : `${Math.round(term.share * 100)} %`}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.notes.map((note) => (
            <p key={note} className="text-xs text-ink-3">
              <Sentence text={note} />
            </p>
          ))}
        </>
      )}
    </section>
  );
}
