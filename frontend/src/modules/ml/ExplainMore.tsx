import { useState } from "react";

import type { MlAnchor, MlExplanation, MlInteractions, MlPermutationShap, MlPrototypes, MlRun } from "../../api/client";
import { useMlAnchor, useMlInteractions, useMlPermutationShap, useMlPrototypes } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";
import { beeswarmFigure } from "./figures";

type Explanation = NonNullable<MlExplanation>;

const PAIRS = 10;
const BACKGROUND = 50;
const COLUMNS = 10;
const THRESHOLD = 0.95;
const CONDITIONS = 4;
const PROTOTYPES = 4;
const CRITICISMS = 2;
const ROWS_SHOWN = 10;

function Block({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h4 className="text-[13px] font-medium text-ink-1">{title}</h4>
      <p className="max-w-prose text-xs text-ink-2">{hint}</p>
      {children}
    </section>
  );
}

function Settings({ children }: { children: React.ReactNode }) {
  return <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">{children}</div>;
}

function Choice<T extends number>({ label, value, options, onChange, show }: { label: string; value: T; options: T[]; onChange: (value: T) => void; show?: (value: T) => string }) {
  return (
    <Select aria-label={label} value={value} onChange={(e) => onChange(Number(e.target.value) as T)} className="w-full text-xs">
      {options.map((n) => (
        <option key={n} value={n}>
          {show ? show(n) : n}
        </option>
      ))}
    </Select>
  );
}

function Compute({ onClick, label }: { onClick: () => void; label: string }) {
  return (
    <button type="button" onClick={onClick} className="rounded-md border border-line px-3 py-1.5 text-xs text-ink-1 hover:bg-surface-2 coarse:min-h-11">
      {label}
    </button>
  );
}

const pct = (value: number) => `${Math.round(value * 100)}%`;

/** Explain › Advanced › More (126): slow methods for a targeted gain, each computed only on its
 * own button (rule of the onion: nothing runs when the page or this section opens). */
export function ExplainMore({ datasetId, run, found }: { datasetId: string; run: MlRun; found: Explanation }) {
  return (
    <div className="mt-3 space-y-6">
      <p className="max-w-prose text-xs text-ink-2">{t("ml.explain.moreHint")}</p>
      <PermutationShapBlock datasetId={datasetId} run={run} exact={Boolean(found.shap)} />
      <InteractionsBlock datasetId={datasetId} run={run} />
      <AnchorBlock datasetId={datasetId} run={run} />
      <PrototypesBlock datasetId={datasetId} run={run} />
    </div>
  );
}

function PermutationShapBlock({ datasetId, run, exact }: { datasetId: string; run: MlRun; exact: boolean }) {
  const [pairs, setPairs] = useState(PAIRS);
  const [background, setBackground] = useState(BACKGROUND);
  const [asked, setAsked] = useState(false);
  const found = useMlPermutationShap(datasetId, run.id, pairs, background, asked);
  return (
    <Block title={t("ml.explain.permShap")} hint={t(exact ? "ml.explain.permShapHintExact" : "ml.explain.permShapHint")}>
      <Settings>
        <Cell label={t("ml.explain.permShapOrders")} why={t("ml.explain.permShapOrdersWhy")} changed={pairs !== PAIRS} onReset={() => setPairs(PAIRS)}>
          <Choice label={t("ml.explain.permShapOrders")} value={pairs} options={[5, 10, 20]} onChange={setPairs} show={(n) => String(2 * n)} />
        </Cell>
        <Cell label={t("ml.explain.permShapBackground")} why={t("ml.explain.permShapBackgroundWhy")} changed={background !== BACKGROUND} onReset={() => setBackground(BACKGROUND)}>
          <Choice label={t("ml.explain.permShapBackground")} value={background} options={[20, 50, 100]} onChange={setBackground} />
        </Cell>
      </Settings>
      {!asked ? (
        <Compute onClick={() => setAsked(true)} label={t("ml.explain.permShapRun")} />
      ) : found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.explain.permShapComputing")} compact className="h-40 rounded-md" />
      ) : (
        <PermutationShapResult data={found.data} />
      )}
    </Block>
  );
}

function PermutationShapResult({ data }: { data: MlPermutationShap }) {
  const shap = data.shap;
  const [row, setRow] = useState<number | null>(null);
  const chosen = row ?? shap.rows[0] ?? null;
  const index = chosen === null ? -1 : shap.rows.indexOf(chosen);
  const values = shap.matrix[index] ?? [];
  const halves = data.half_width[index] ?? [];
  const labels = shap.labels[index] ?? [];
  const order = values.map((_, j) => j).sort((a, b) => Math.abs(values[b] ?? 0) - Math.abs(values[a] ?? 0)).slice(0, ROWS_SHOWN);
  return (
    <div className="space-y-3">
      <p className="max-w-prose text-[13px] text-ink-1">{data.sentence}</p>
      <figure className="min-w-0 rounded-md border border-line bg-surface-1 p-2" aria-label={t("ml.explain.permShap")}>
        <Plot figure={beeswarmFigure(shap)} height={Math.max(260, 26 * shap.features.length + 80)} label={t("ml.explain.permShap")} />
      </figure>
      {shap.positive && <p className="text-xs text-ink-3">{t("ml.explain.permShapPositive", { positive: shap.positive })}</p>}
      {shap.note && <p className="max-w-prose text-xs text-ink-3">{shap.note}</p>}
      {chosen !== null && (
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-xs text-ink-2">
            {t("ml.explain.permShapRow")}
            <Select aria-label={t("ml.explain.permShapRow")} value={chosen} onChange={(e) => setRow(Number(e.target.value))} className="h-7 text-xs">
              {shap.rows.slice(0, 200).map((r) => (
                <option key={r} value={r}>
                  {t("ml.explain.rowNumber", { row: String(r + 1) })}
                </option>
              ))}
            </Select>
          </label>
          <table className="w-full max-w-xl text-[13px] tabular-nums">
            <thead>
              <tr className="text-left text-xs text-ink-3">
                <th className="py-1 font-normal">{t("ml.explain.column")}</th>
                <th className="py-1 text-right font-normal">{t("ml.explain.permShapContribution")}</th>
              </tr>
            </thead>
            <tbody>
              {order.map((j) => (
                <tr key={shap.columns[j]} className="border-t border-line">
                  <td className="py-1 text-ink-1">
                    {shap.columns[j]} = <span className="text-ink-2">{labels[j]}</span>
                  </td>
                  <td className="py-1 text-right text-ink-1">
                    {(values[j] ?? 0) >= 0 ? "+" : ""}
                    {formatNumber(values[j] ?? 0)} <span className="text-xs text-ink-3">± {formatNumber(halves[j] ?? 0)}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-ink-3">{t("ml.explain.permShapAdditive", { base: formatNumber(shap.base) })}</p>
        </div>
      )}
    </div>
  );
}

function InteractionsBlock({ datasetId, run }: { datasetId: string; run: MlRun }) {
  const [columns, setColumns] = useState(COLUMNS);
  const [asked, setAsked] = useState(false);
  const found = useMlInteractions(datasetId, run.id, columns, asked);
  return (
    <Block title={t("ml.explain.interactions")} hint={t("ml.explain.interactionsHint")}>
      <Settings>
        <Cell label={t("ml.explain.interactionsColumns")} why={t("ml.explain.interactionsColumnsWhy")} changed={columns !== COLUMNS} onReset={() => setColumns(COLUMNS)}>
          <Choice label={t("ml.explain.interactionsColumns")} value={columns} options={[5, 10, 15]} onChange={setColumns} />
        </Cell>
      </Settings>
      {!asked ? (
        <Compute onClick={() => setAsked(true)} label={t("ml.explain.interactionsRun")} />
      ) : found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.explain.interactionsComputing")} compact className="h-32 rounded-md" />
      ) : (
        <InteractionsResult data={found.data} />
      )}
    </Block>
  );
}

function InteractionsResult({ data }: { data: MlInteractions }) {
  const strong = data.pairs.filter((p) => !p.weak);
  const weak = data.pairs.length - strong.length;
  return (
    <div className="space-y-2">
      <p className="max-w-prose text-[13px] text-ink-1">{data.sentence}</p>
      <ul className="max-w-xl space-y-0.5 text-[13px] tabular-nums" aria-label={t("ml.explain.interactions")}>
        {strong.slice(0, 15).map((p) => (
          <li key={`${p.first}-${p.second}`} className="grid grid-cols-[minmax(0,1fr)_6rem_3rem] items-center gap-3 border-t border-line py-1">
            <span className="min-w-0 truncate text-ink-1">
              {p.first} × {p.second}
            </span>
            <span className="h-1.5 rounded-sm bg-surface-2" aria-hidden="true">
              <span className="block h-1.5 rounded-sm bg-accent" style={{ width: `${Math.round(p.h * 100)}%` }} />
            </span>
            <span className="text-right text-ink-2">{p.h.toFixed(2)}</span>
          </li>
        ))}
      </ul>
      <p className="text-xs text-ink-3">
        {t("ml.explain.interactionsRows", { rows: data.rows.toLocaleString("en-US"), columns: String(data.columns.length) })}
        {weak > 0 && ` ${t("ml.explain.interactionsWeak", { count: String(weak) })}`}
      </p>
      {data.note && <p className="text-xs text-ink-3">{data.note}</p>}
    </div>
  );
}

function AnchorBlock({ datasetId, run }: { datasetId: string; run: MlRun }) {
  const rows = run.predictions.map((p) => p.row);
  const [row, setRow] = useState<number | null>(null);
  const [threshold, setThreshold] = useState(THRESHOLD);
  const [conditions, setConditions] = useState(CONDITIONS);
  const [asked, setAsked] = useState(false);
  const chosen = row ?? rows[0] ?? null;
  const found = useMlAnchor(datasetId, run.id, chosen, threshold, conditions, asked);
  if (chosen === null) return null;
  const predicted = run.predictions.find((p) => p.row === chosen)?.predicted;
  return (
    <Block title={t("ml.explain.anchor")} hint={t("ml.explain.anchorHint")}>
      <Settings>
        <Cell label={t("ml.explain.anchorRow")} why={t("ml.explain.anchorRowWhy")} changed={row !== null && row !== rows[0]} onReset={() => setRow(null)}>
          <Select aria-label={t("ml.explain.anchorRow")} value={chosen} onChange={(e) => setRow(Number(e.target.value))} className="w-full text-xs">
            {run.predictions.map((p) => (
              <option key={p.row} value={p.row}>
                {t("ml.explain.anchorRowOption", { row: String(p.row + 1), predicted: p.predicted })}
              </option>
            ))}
          </Select>
        </Cell>
        <Cell label={t("ml.explain.anchorPrecision")} why={t("ml.explain.anchorPrecisionWhy")} changed={threshold !== THRESHOLD} onReset={() => setThreshold(THRESHOLD)}>
          <Choice label={t("ml.explain.anchorPrecision")} value={threshold} options={[0.8, 0.9, 0.95, 0.99]} onChange={setThreshold} show={pct} />
        </Cell>
        <Cell label={t("ml.explain.anchorConditions")} why={t("ml.explain.anchorConditionsWhy")} changed={conditions !== CONDITIONS} onReset={() => setConditions(CONDITIONS)}>
          <Choice label={t("ml.explain.anchorConditions")} value={conditions} options={[1, 2, 3, 4, 5, 6]} onChange={setConditions} />
        </Cell>
      </Settings>
      {!asked ? (
        <Compute onClick={() => setAsked(true)} label={t("ml.explain.anchorRun", { row: String(chosen + 1), predicted: predicted ?? "" })} />
      ) : found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.explain.anchorComputing")} compact className="h-24 rounded-md" />
      ) : (
        <AnchorResult data={found.data} />
      )}
    </Block>
  );
}

function AnchorResult({ data }: { data: MlAnchor }) {
  return (
    <div className="space-y-2">
      <div className={`max-w-prose rounded-md border px-3 py-2 text-[13px] ${data.reached ? "border-line bg-surface-1" : "border-warn/40 bg-warn/8"}`}>
        <p className="font-mono text-xs text-ink-2">IF</p>
        <ul className="ml-3 space-y-0.5">
          {data.conditions.map((c, i) => (
            <li key={c.column} className="text-ink-1">
              {i > 0 && <span className="font-mono text-xs text-ink-2">AND </span>}
              {c.text}
              <span className="text-xs text-ink-3"> ({t("ml.explain.anchorValue", { value: c.value })})</span>
            </li>
          ))}
        </ul>
        <p className="mt-1 text-ink-1">
          <span className="font-mono text-xs text-ink-2">THEN </span>
          {data.outcome}
        </p>
      </div>
      <p className="max-w-prose text-[13px] text-ink-1">{data.sentence}</p>
      <p className="text-xs text-ink-3">
        {t("ml.explain.anchorStats", { precision: pct(data.precision), low: pct(data.precision_low), coverage: pct(data.coverage), samples: data.samples.toLocaleString("en-US"), tried: String(data.evaluated) })}
      </p>
      {data.left_out.length > 0 && <p className="text-xs text-ink-3">{t("ml.explain.leftOut", { columns: data.left_out.join("; ") })}</p>}
    </div>
  );
}

function PrototypesBlock({ datasetId, run }: { datasetId: string; run: MlRun }) {
  const [count, setCount] = useState(PROTOTYPES);
  const [criticisms, setCriticisms] = useState(CRITICISMS);
  const [asked, setAsked] = useState(false);
  const found = useMlPrototypes(datasetId, run.id, count, criticisms, asked);
  return (
    <Block title={t("ml.explain.prototypes")} hint={t("ml.explain.prototypesHint")}>
      <Settings>
        <Cell label={t("ml.explain.prototypesCount")} why={t("ml.explain.prototypesCountWhy")} changed={count !== PROTOTYPES} onReset={() => setCount(PROTOTYPES)}>
          <Choice label={t("ml.explain.prototypesCount")} value={count} options={[1, 2, 3, 4, 6, 8]} onChange={setCount} />
        </Cell>
        <Cell label={t("ml.explain.criticismsCount")} why={t("ml.explain.criticismsCountWhy")} changed={criticisms !== CRITICISMS} onReset={() => setCriticisms(CRITICISMS)}>
          <Choice label={t("ml.explain.criticismsCount")} value={criticisms} options={[0, 1, 2, 3, 5]} onChange={setCriticisms} />
        </Cell>
      </Settings>
      {!asked ? (
        <Compute onClick={() => setAsked(true)} label={t("ml.explain.prototypesRun")} />
      ) : found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.explain.prototypesComputing")} compact className="h-32 rounded-md" />
      ) : (
        <PrototypesResult data={found.data} />
      )}
    </Block>
  );
}

function PrototypesResult({ data }: { data: MlPrototypes }) {
  return (
    <div className="space-y-4">
      <p className="max-w-prose text-[13px] text-ink-1">{data.sentence}</p>
      {data.groups.map((group) => (
        <div key={group.label} className="min-w-0 space-y-1">
          <p className="text-xs text-ink-2">{t("ml.explain.prototypesGroup", { label: group.label, rows: group.rows.toLocaleString("en-US") })}</p>
          <div className="overflow-x-auto">
            <table className="w-full text-[13px] tabular-nums">
              <thead>
                <tr className="text-left text-xs text-ink-3">
                  <th className="py-1 pr-3 font-normal">{t("ml.explain.prototypesKind")}</th>
                  <th className="py-1 pr-3 font-normal">{t("ml.explain.prototypesRow")}</th>
                  {data.columns.map((c) => (
                    <th key={c} className="py-1 pr-3 font-normal">
                      {c}
                    </th>
                  ))}
                  <th className="py-1 pr-3 font-normal">{t("ml.explain.prototypesPredicted")}</th>
                </tr>
              </thead>
              <tbody>
                {group.prototypes.map((p) => (
                  <tr key={`p-${p.row}`} className="border-t border-line">
                    <td className="py-1 pr-3 text-ink-1">{t("ml.explain.prototypeShare", { share: pct(p.share ?? 0) })}</td>
                    <td className="py-1 pr-3 text-ink-2">{p.row + 1}</td>
                    {p.values.map((v, i) => (
                      <td key={data.columns[i]} className="py-1 pr-3 text-ink-1">
                        {v}
                      </td>
                    ))}
                    <td className="py-1 pr-3 text-ink-1">{p.predicted}</td>
                  </tr>
                ))}
                {group.criticisms.map((c) => (
                  <tr key={`c-${c.row}`} className="border-t border-line">
                    <td className="py-1 pr-3 text-ink-2">{t("ml.explain.criticism")}</td>
                    <td className="py-1 pr-3 text-ink-2">{c.row + 1}</td>
                    {c.values.map((v, i) => (
                      <td key={data.columns[i]} className="py-1 pr-3 text-ink-1">
                        {v}
                      </td>
                    ))}
                    <td className="py-1 pr-3 text-ink-1">{c.predicted}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}
      {data.left_out.length > 0 && <p className="text-xs text-ink-3">{t("ml.explain.leftOut", { columns: data.left_out.join("; ") })}</p>}
    </div>
  );
}
