import { useState } from "react";

import type { RegressionPanel, RegressionPanelEstimator, RegressionParams } from "../../api/client";
import { useRegressionSection } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Cell } from "../../modules/kpi/Customize";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Sentence } from "./Sentence";

type Key = RegressionPanelEstimator["key"];

const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums text-right";

/** Advanced › Panel (071): every panel estimator, the recommended one preselected by rule
 *  (R-REG-PAN-01, 02) in a native select with its reason, compared with pooled OLS. */
export function PanelSection({ datasetId, params }: { datasetId: string; params: RegressionParams }) {
  const query = useRegressionSection(datasetId, "panel", params, true);
  const data: RegressionPanel | undefined = query.data;
  const [picked, setPicked] = useState<Key | null>(null);
  if (!data) return query.error ? <ErrorNote>{query.error.message}</ErrorNote> : <Loading label={t("regression.section.loading")} compact className="h-32 rounded-md" />;
  if (!data.available) return <p className="text-[13px] text-ink-2">{data.reason}</p>;
  const available = data.estimators.filter((e) => e.available);
  const missing = data.estimators.filter((e) => !e.available);
  const recommended = data.recommended ?? available[0]?.key ?? "pooled";
  const key = picked && available.some((e) => e.key === picked) ? picked : recommended;
  const chosen = available.find((e) => e.key === key) ?? available[0];
  const pooled = data.estimators.find((e) => e.key === "pooled");
  const why = key === recommended ? `${data.recommended_rule ? `${data.recommended_rule}: ` : ""}${data.recommended_reason}` : chosen?.reason;
  return (
    <div className="space-y-3 text-[13px]">
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(0,16rem)_minmax(0,1fr)] sm:items-end">
        <Cell label={t("regression.panel.estimator")} why={why} changed={key !== recommended} onReset={() => setPicked(null)}>
          <Select aria-label={t("regression.panel.estimator")} value={key} onChange={(event) => setPicked(event.target.value as Key)} className="h-8 w-full min-w-0 text-[13px]">
            {available.map((estimator) => (
              <option key={estimator.key} value={estimator.key}>
                {estimator.key === recommended ? t("regression.panel.recommended", { label: estimator.label }) : estimator.label}
              </option>
            ))}
          </Select>
        </Cell>
        {why && (
          <p className="text-xs text-ink-2">
            <Sentence text={why} />
          </p>
        )}
      </div>
      {chosen && pooled && <Comparison data={data} pooled={pooled} chosen={chosen} />}
      {chosen && chosen.dropped.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.panel.dropped")}</h3>
          <ul className="space-y-0.5 text-ink-2">
            {chosen.dropped.map((item) => (
              <li key={item.term}>
                <code className="font-mono text-xs text-ink-1">{item.term}</code> · <Sentence text={item.reason} />
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.tests.length > 0 && (
        <div>
          <h3 className="mb-1 text-xs font-medium text-ink-2">{t("regression.panel.tests")}</h3>
          <ul className="space-y-1 text-ink-2">
            {data.tests.map((test) => (
              <li key={test.key}>
                <span className="font-medium text-ink-1">{test.title}</span> · <Sentence text={test.sentence} />
                {test.rule && <span className="text-xs text-ink-3"> ({test.rule})</span>}
              </li>
            ))}
          </ul>
        </div>
      )}
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          <Sentence text={note} />
        </p>
      ))}
      {missing.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-3">
          {missing.map((estimator) => (
            <li key={estimator.key}>
              <span className="text-ink-2">{estimator.label}</span> · {t("regression.panel.unavailable")}: <Sentence text={estimator.reason} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function Comparison({ data, pooled, chosen }: { data: RegressionPanel; pooled: RegressionPanelEstimator; chosen: RegressionPanelEstimator }) {
  const columns = chosen.key === "pooled" ? [pooled] : [pooled, chosen];
  const details = t("regression.panel.details", {
    rows: formatCount(chosen.rows),
    groups: formatCount(chosen.groups),
    r2kind: chosen.r2_kind,
    r2: chosen.r2 === null ? "—" : chosen.r2.toFixed(3),
    se: chosen.se_label,
  });
  return (
    <div className="space-y-1">
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.term")}</th>
              {columns.map((estimator) => (
                <th key={estimator.key} scope="col" className={`${th} text-right whitespace-nowrap`} title={estimator.reason}>
                  {estimator === chosen ? <span className="font-medium text-ink-1">{estimator.label}</span> : estimator.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.terms.map((term, row) => (
              <tr key={term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">{term}</td>
                {columns.map((estimator) => {
                  const c = estimator.coefficients[row];
                  return (
                    <td key={estimator.key} className={`${td} ${estimator === chosen ? "text-ink-1" : "text-ink-2"}`} title={c?.p_value !== null && c?.p_value !== undefined ? `p ${formatP(c.p_value)}` : undefined}>
                      {c?.estimate === null || c?.estimate === undefined ? (
                        <span className="text-xs text-ink-3">{t("regression.panel.notEstimated")}</span>
                      ) : (
                        <>
                          {formatNumber(c.estimate)}
                          <span className="block text-xs text-ink-3">
                            [{formatNumber(c.ci_low)} ; {formatNumber(c.ci_high)}]
                          </span>
                        </>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        {details}
        {chosen.theta !== null && ` · ${t("regression.panel.theta", { value: chosen.theta.toFixed(2) })}`}
      </p>
    </div>
  );
}
