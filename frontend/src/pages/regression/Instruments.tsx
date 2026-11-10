import { AlertTriangle, CheckCircle2, Info, X, XCircle } from "lucide-react";
import { useState } from "react";

import type { AnalysisType, Regression, RegressionIv, RegressionParams } from "../../api/client";
import { useColumns, useRegressionInstruments } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Sentence } from "./Sentence";

type Status = RegressionIv["tests"][number]["status"];

const USABLE = new Set<AnalysisType>(["numeric_continuous", "numeric_discrete", "boolean", "categorical_nominal", "categorical_ordinal"]);
const STATUS: Record<Status, { icon: typeof Info; tone: string; label: MessageKey }> = {
  ok: { icon: CheckCircle2, tone: "text-ok", label: "regression.status.ok" },
  warn: { icon: AlertTriangle, tone: "text-warn", label: "regression.status.warn" },
  bad: { icon: XCircle, tone: "text-bad", label: "regression.status.bad" },
  info: { icon: Info, tone: "text-ink-3", label: "regression.status.info" },
};
const th = "px-3 py-1.5 font-normal";
const td = "px-3 py-1.5 tabular-nums text-right";

/** Advanced › More › Instruments (071): IV / 2SLS, asked for only once the user has named the
 *  endogenous column and at least one instrument (R-REG-IV-01, never guessed). */
export function Instruments({ datasetId, params, result }: { datasetId: string; params: RegressionParams; result: Regression }) {
  const columns = useColumns(datasetId, true);
  const [endogenous, setEndogenous] = useState("");
  const [instruments, setInstruments] = useState<string[]>([]);
  const inModel = new Set([result.target, ...result.columns]);
  const candidates = (columns.data ?? [])
    .filter((c) => !c.excluded && USABLE.has(c.type) && !inModel.has(c.name))
    .sort((a, b) => a.position - b.position);
  const endogenousNow = result.columns.includes(endogenous) ? endogenous : "";
  const chosen = instruments.filter((name) => candidates.some((c) => c.name === name));
  const query = useRegressionInstruments(datasetId, params, endogenousNow ? [endogenousNow] : [], chosen);
  const field = "h-8 w-full min-w-0 text-[13px]";
  return (
    <section className="space-y-3 border-t border-line pt-3">
      <div>
        <h3 className="text-[13px] font-medium text-ink-1">{t("regression.iv.title")}</h3>
        <p className="max-w-prose text-xs text-ink-2">{t("regression.iv.hint")}</p>
      </div>
      {result.model === "logit" ? (
        <p className="text-xs text-ink-3">{t("regression.iv.logit")}</p>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <Field label={t("regression.iv.endogenous")}>
              <Select aria-label={t("regression.iv.endogenous")} value={endogenousNow} onChange={(event) => setEndogenous(event.target.value)} className={field}>
                <option value="">{t("regression.iv.pick")}</option>
                {result.columns.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            </Field>
            <Field label={t("regression.iv.instruments")}>
              <Select
                aria-label={t("regression.iv.addInstrument")}
                value=""
                onChange={(event) => event.target.value && setInstruments([...chosen, event.target.value])}
                className={field}
              >
                <option value="">{t("regression.iv.addInstrument")}</option>
                {candidates
                  .filter((c) => !chosen.includes(c.name))
                  .map((c) => (
                    <option key={c.name} value={c.name}>
                      {c.name}
                    </option>
                  ))}
              </Select>
            </Field>
            {chosen.length > 0 && (
              <ul className="flex flex-wrap items-end gap-1.5 sm:col-span-1">
                {chosen.map((name) => (
                  <li key={name}>
                    <button
                      type="button"
                      onClick={() => setInstruments(chosen.filter((item) => item !== name))}
                      aria-label={t("regression.iv.remove", { name })}
                      className="inline-flex h-8 items-center gap-1 rounded-md border border-line bg-surface-2 px-2 font-mono text-xs text-ink-1 hover:text-accent"
                    >
                      {name}
                      <X size={12} strokeWidth={1.75} aria-hidden />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          {!endogenousNow || chosen.length === 0 ? (
            <p className="text-xs text-ink-3">{t("regression.iv.waiting")}</p>
          ) : query.error ? (
            <ErrorNote>{query.error.message}</ErrorNote>
          ) : !query.data ? (
            <Loading label={t("regression.iv.loading")} compact className="h-32 rounded-md" />
          ) : (
            <IvResult data={query.data} />
          )}
        </>
      )}
    </section>
  );
}

function IvResult({ data }: { data: RegressionIv }) {
  return (
    <div className="space-y-3 text-[13px]">
      <p className="max-w-prose text-ink-2">
        <Sentence text={data.sentence} />
      </p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={th}>{t("regression.term")}</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.iv.estimate")}</th>
              <th scope="col" className={`${th} text-right`}>p</th>
              <th scope="col" className={`${th} text-right`}>{t("regression.iv.leastSquares")}</th>
            </tr>
          </thead>
          <tbody>
            {data.coefficients.map((c) => (
              <tr key={c.term} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs">
                  {c.term}
                  {c.endogenous && <span className="ml-1.5 font-sans text-[11px] text-ink-3">{t("regression.iv.endogenousTag")}</span>}
                </td>
                <td className={`${td} text-ink-1`}>
                  {formatNumber(c.estimate)}
                  <span className="block text-xs text-ink-3">
                    [{formatNumber(c.ci_low)} ; {formatNumber(c.ci_high)}]
                  </span>
                </td>
                <td className={`${td} whitespace-nowrap text-ink-2`}>{formatP(c.p_value)}</td>
                <td className={`${td} text-ink-2`}>
                  {formatNumber(c.ols)}
                  <span className="block text-xs text-ink-3">
                    [{formatNumber(c.ols_low)} ; {formatNumber(c.ols_high)}]
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">
        {formatCount(data.rows)} rows · <Sentence text={t("regression.iv.se", { label: data.se.label, reason: data.se.reason })} />
      </p>
      <div>
        <h4 className="mb-1 text-xs font-medium text-ink-2">{t("regression.iv.firstStage")}</h4>
        <ul className="space-y-1">
          {data.first_stages.map((stage) => (
            <Line key={stage.term} status={stage.status} text={stage.sentence} />
          ))}
        </ul>
      </div>
      <div>
        <h4 className="mb-1 text-xs font-medium text-ink-2">{t("regression.iv.tests")}</h4>
        <ul className="space-y-1">
          {data.tests.map((test) => (
            <Line key={test.key} status={test.status} title={test.title} text={test.sentence} />
          ))}
        </ul>
      </div>
      {data.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          <Sentence text={note} />
        </p>
      ))}
    </div>
  );
}

function Line({ status, title, text }: { status: Status; title?: string; text: string }) {
  const style = STATUS[status];
  const Icon = style.icon;
  return (
    <li className="grid grid-cols-[4.5rem_minmax(0,1fr)] gap-x-2 text-ink-2">
      <span className={`inline-flex items-center gap-1 pt-px text-xs font-medium ${style.tone}`}>
        <Icon size={14} strokeWidth={1.75} aria-hidden />
        {t(style.label)}
      </span>
      <span>
        {title && <span className="font-medium text-ink-1">{title} · </span>}
        <Sentence text={text} />
      </span>
    </li>
  );
}
