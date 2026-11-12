import { useMemo, useState } from "react";

import type { RegressionParams, RegressionQuantiles, RegressionRobust } from "../../api/client";
import { useRegressionRobust, useRegressionSection } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Cell } from "../../modules/kpi/Customize";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { quantileFigure } from "./figures";
import { Sentence } from "./Sentence";

const MEDIAN = 0.5;
const SMALL_HEIGHT = 220;
const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums text-right";

function Pending({ error }: { error: Error | null }) {
  return error ? <ErrorNote>{error.message}</ErrorNote> : <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />;
}

/** Advanced › Robust & quantile (072): Huber, Tukey and one quantile beside least squares
 *  (R-REG-ROB-01…03, R-REG-QR-01), the law and tail of the residuals (R-REG-QR-03). */
export function RobustSection({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const [quantile, setQuantile] = useState(MEDIAN);
  const query = useRegressionRobust(datasetId, params, quantile, true);
  const data: RegressionRobust | undefined = query.data;
  if (!data) return <Pending error={query.error} />;
  if (!data.available) return <p className="text-[13px] text-ink-2">{data.reason}</p>;
  const moved = new Set(data.moved);
  return (
    <div className="space-y-3 text-[13px]">
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <Cell label={t("regression.robust.quantile")} why={t("regression.robust.quantileWhy")} changed={quantile !== MEDIAN} onReset={() => setQuantile(MEDIAN)}>
          <Select aria-label={t("regression.robust.quantile")} value={String(quantile)} onChange={(event) => setQuantile(Number(event.target.value))} className="h-8 w-full min-w-0 text-[13px]">
            {data.quantiles.map((q) => (
              <option key={q} value={String(q)}>
                {q === MEDIAN ? t("regression.robust.median") : t("regression.robust.tau", { value: String(q) })}
              </option>
            ))}
          </Select>
        </Cell>
      </div>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.term")}</th>
              {data.columns.map((column) => (
                <th key={column.kind} scope="col" className={`${th} text-right whitespace-nowrap`} title={column.note}>
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.terms.map((term, row) => (
              <tr key={term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">
                  {term}
                  {moved.has(term) && <span className="ml-1.5 font-sans text-[11px] text-warn">{t("regression.robust.moves")}</span>}
                </td>
                {data.columns.map((column) => (
                  <td key={column.kind} className={`${td} ${column.kind === "ols" ? "text-ink-1" : "text-ink-2"}`} title={`p ${formatP(column.p_values[row] ?? null)}`}>
                    {formatNumber(column.estimates[row] ?? null)}
                    <span className="block text-xs whitespace-nowrap text-ink-3">
                      [{formatNumber(column.low[row] ?? null)} ; {formatNumber(column.high[row] ?? null)}]
                    </span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="space-y-0.5 text-xs text-ink-3">
        {data.columns.map((column) => (
          <li key={column.kind}>
            <span className="text-ink-2">{column.label}</span> · <Sentence text={column.note} />
          </li>
        ))}
        {data.notes.map((note) => (
          <li key={note}>
            <Sentence text={note} />
          </li>
        ))}
      </ul>
      <div>
        <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.robust.residuals")}</h3>
        <p className="text-ink-2">
          <Sentence text={data.law_sentence} /> <Sentence text={data.tail_sentence} />
        </p>
        {data.laws.length > 0 && (
          <p className="mt-0.5 text-xs text-ink-3 tabular-nums">
            {data.laws.map((law) => `${law.name}${law.df !== null ? ` (ν = ${law.df.toFixed(1)})` : ""}: AIC ${formatNumber(law.aic)}`).join(" · ")}
          </p>
        )}
      </div>
    </div>
  );
}

/** Advanced › More › Along the distribution (072, R-REG-QR-02, R-REG-ROB-04): asked for only
 *  once More is open. */
export function QuantilesBlock({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useRegressionSection(datasetId, "quantiles", params, true);
  const data: RegressionQuantiles | undefined = query.data;
  const figures = useMemo(() => (data ? data.curves.map((curve) => [curve, quantileFigure(curve)] as const) : []), [data]);
  return (
    <section className="space-y-2 border-t border-line pt-3 text-[13px]" aria-label={t("regression.quantiles.title")}>
      <div>
        <h3 className="text-[13px] font-medium text-ink-1">{t("regression.quantiles.title")}</h3>
        <p className="max-w-prose text-xs text-ink-2">{t("regression.quantiles.hint")}</p>
      </div>
      {!data ? (
        <Pending error={query.error} />
      ) : !data.available ? (
        <p className="text-ink-2">{data.reason}</p>
      ) : (
        <>
          <p className="text-ink-2">
            <Sentence text={data.sentence} />
          </p>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {figures.map(([curve, figure]) => (
              <figure key={curve.term} className="min-w-0 rounded-lg border border-line bg-surface-1 p-3">
                <figcaption className="mb-1 text-xs text-ink-2">
                  <Sentence text={curve.sentence} />
                </figcaption>
                <Plot figure={figure} height={SMALL_HEIGHT} label={t("regression.quantiles.curveOf", { term: curve.term })} />
              </figure>
            ))}
          </div>
          {data.tested > data.curves.length && <p className="text-xs text-ink-3">{t("regression.quantiles.shown", { shown: String(data.curves.length), tested: String(data.tested) })}</p>}
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th scope="col" className={th}>{t("regression.term")}</th>
                  <th scope="col" className={`${th} text-right`}>τ = 0.1</th>
                  <th scope="col" className={`${th} text-right`}>τ = 0.9</th>
                  <th scope="col" className={`${th} text-right`}>{t("regression.quantiles.ols")}</th>
                  <th scope="col" className={`${th} text-right`}>adj. p</th>
                </tr>
              </thead>
              <tbody>
                {data.curves.map((curve) => (
                  <tr key={curve.term} className="border-t border-line">
                    <td className="px-3 py-1.5 font-mono text-xs">{curve.term}</td>
                    <td className={`${td} text-ink-1`}>{formatNumber(curve.estimates[0] ?? null)}</td>
                    <td className={`${td} text-ink-1`}>{formatNumber(curve.estimates[curve.estimates.length - 1] ?? null)}</td>
                    <td className={`${td} text-ink-2`}>{formatNumber(curve.ols)}</td>
                    <td className={`${td} text-ink-2`}>{formatP(curve.q_value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.sampled && <p className="text-xs text-ink-3">{t("regression.quantiles.sampled", { rows: formatCount(data.rows) })}</p>}
          {data.t_errors && (
            <p className="text-ink-2">
              <span className="font-medium text-ink-1">{t("regression.quantiles.tErrors")}</span> ·{" "}
              {data.t_errors.available ? <Sentence text={data.t_errors.sentence} /> : <Sentence text={data.t_errors.reason} />}
            </p>
          )}
        </>
      )}
    </section>
  );
}
