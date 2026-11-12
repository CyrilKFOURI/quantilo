import { useState } from "react";

import type { MlExplanation, MlRun, MlSurrogate } from "../../api/client";
import { useMlAle, useMlMonotonic, useMlSurrogate } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";
import { aleFigure } from "./figures";

type Explanation = NonNullable<MlExplanation>;
type SurrogateNode = MlSurrogate["nodes"][number];

const BINS = 20;
const DEPTH = 3;
/** Models whose training takes monotonic constraints (Settings › Advanced, 019). */
const CONSTRAINED = new Set(["hist_gradient_boosting", "xgboost", "lightgbm"]);
const SKIPPED_TYPES = new Set(["text", "datetime", "identifier"]);

function Block({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h4 className="text-[13px] font-medium text-ink-1">{title}</h4>
      <p className="max-w-prose text-xs text-ink-2">{hint}</p>
      {children}
    </section>
  );
}

/** Explain › Advanced (125): nothing here is computed before the user opens it. */
export function ExplainAdvanced({
  datasetId,
  run,
  found,
  column,
  onColumn,
}: {
  datasetId: string;
  run: MlRun;
  found: Explanation;
  column: string | null;
  onColumn: (column: string | null) => void;
}) {
  return (
    <div className="mt-3 space-y-6">
      <AleBlock datasetId={datasetId} run={run} found={found} column={column} onColumn={onColumn} />
      <SurrogateBlock datasetId={datasetId} run={run} />
      <MonotonicBlock datasetId={datasetId} run={run} />
    </div>
  );
}

function AleBlock({ datasetId, run, found, column, onColumn }: { datasetId: string; run: MlRun; found: Explanation; column: string | null; onColumn: (column: string | null) => void }) {
  const [bins, setBins] = useState(BINS);
  const inputs = run.preparation.decisions.filter((d) => d.use && !SKIPPED_TYPES.has(d.type)).map((d) => d.column);
  // R-XAI-01: the column whose partial dependence extrapolates is the one to read first
  const linked = found.dependence.find((d) => d.extrapolates);
  const suggested = linked?.column ?? found.dependence[0]?.column ?? inputs[0] ?? null;
  const chosen = column ?? suggested;
  const ale = useMlAle(datasetId, run.id, chosen, bins);
  if (!chosen) return null;
  return (
    <Block title={t("ml.explain.ale")} hint={t("ml.explain.aleHint")}>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
        <Cell
          label={t("ml.explain.column")}
          why={linked && chosen === linked.column ? t("ml.explain.aleWhyLinked", { other: linked.linked ?? "" }) : t("ml.explain.aleWhyColumn")}
          changed={column !== null && column !== suggested}
          onReset={() => onColumn(null)}
        >
          <Select aria-label={t("ml.explain.aleColumn")} value={chosen} onChange={(e) => onColumn(e.target.value)} className="w-full text-xs">
            {[...new Set([chosen, ...inputs])].map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </Select>
        </Cell>
        <Cell label={t("ml.explain.aleBins")} why={t("ml.explain.aleBinsWhy")} changed={bins !== BINS} onReset={() => setBins(BINS)}>
          <Select aria-label={t("ml.explain.aleBins")} value={bins} onChange={(e) => setBins(Number(e.target.value))} className="w-full text-xs">
            {[10, 20, 30, 50].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
      </div>
      {ale.isError ? (
        <ErrorNote>{ale.error.message}</ErrorNote>
      ) : !ale.data ? (
        <Loading label={t("ml.explain.aleComputing")} compact className="h-40 rounded-md" />
      ) : (
        <div className="space-y-2">
          <p className="max-w-prose text-[13px] text-ink-1">{ale.data.sentence}</p>
          {ale.data.rule && <p className="max-w-prose text-xs text-ink-2">{t("ml.explain.aleRule")}</p>}
          <figure className="min-w-0 rounded-md border border-line bg-surface-1 p-2" aria-label={t("ml.explain.ale")}>
            <Plot figure={aleFigure(ale.data)} height={320} label={t("ml.explain.ale")} />
          </figure>
          <p className="text-xs text-ink-3">
            {t(ale.data.kind === "number" ? "ml.explain.aleRowsNumber" : "ml.explain.aleRowsCategory", { rows: ale.data.rows.toLocaleString("en-US"), order: ale.data.order ?? "" })}
          </p>
          {ale.data.monotonic && (
            <p className={`text-xs ${ale.data.monotonic.violations ? "text-bad" : "text-ok"}`}>{ale.data.monotonic.sentence}</p>
          )}
        </div>
      )}
    </Block>
  );
}

function SurrogateBlock({ datasetId, run }: { datasetId: string; run: MlRun }) {
  const [depth, setDepth] = useState(DEPTH);
  const found = useMlSurrogate(datasetId, run.id, depth, true);
  return (
    <Block title={t("ml.explain.surrogate")} hint={t("ml.explain.surrogateHint")}>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
        <Cell label={t("ml.explain.surrogateDepth")} why={t("ml.explain.surrogateDepthWhy")} changed={depth !== DEPTH} onReset={() => setDepth(DEPTH)}>
          <Select aria-label={t("ml.explain.surrogateDepth")} value={depth} onChange={(e) => setDepth(Number(e.target.value))} className="w-full text-xs">
            {[1, 2, 3, 4, 5].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
      </div>
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.explain.surrogateComputing")} compact className="h-32 rounded-md" />
      ) : (
        <div className="space-y-2">
          <p className="max-w-prose text-[13px] text-ink-1">{found.data.sentence}</p>
          {found.data.warning && (
            <p role="note" className="max-w-prose rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-xs text-ink-1">
              {found.data.warning}
            </p>
          )}
          <SurrogateTree nodes={found.data.nodes} />
        </div>
      )}
    </Block>
  );
}

function SurrogateTree({ nodes }: { nodes: SurrogateNode[] }) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const render = (node: SurrogateNode, rule: string | null): React.ReactNode => {
    const label = (
      <span className="text-[13px]">
        {rule && <span className="font-mono text-xs text-ink-2">{rule} → </span>}
        <span className="text-ink-1">{node.value}</span>
        <span className="text-xs text-ink-3"> · {t("ml.explain.samples", { count: node.samples.toLocaleString("en-US") })}</span>
      </span>
    );
    const left = node.left === null ? undefined : byId.get(node.left);
    const right = node.right === null ? undefined : byId.get(node.right);
    if (!left || !right) return <li key={node.id} className="py-0.5 pl-4">{label}</li>;
    return (
      <li key={node.id}>
        <details open>
          <summary className="cursor-pointer py-0.5">{label}</summary>
          <ul className="ml-3 border-l border-line pl-2">
            {render(left, node.left_rule)}
            {render(right, node.right_rule)}
          </ul>
        </details>
      </li>
    );
  };
  const root = nodes[0];
  return <ul className="rounded-md border border-line bg-surface-1 p-2" aria-label={t("ml.explain.surrogate")}>{root && render(root, null)}</ul>;
}

function MonotonicBlock({ datasetId, run }: { datasetId: string; run: MlRun }) {
  const text = String(run.settings.monotonic ?? "").trim();
  const found = useMlMonotonic(datasetId, run.id, Boolean(text));
  if (!CONSTRAINED.has(run.model)) return null;
  if (!text) return <p className="max-w-prose text-xs text-ink-2">{t("ml.explain.monotonicNone")}</p>;
  return (
    <Block title={t("ml.explain.monotonic")} hint={t("ml.explain.monotonicHint", { constraints: text })}>
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("ml.explain.monotonicComputing")} compact className="h-16 rounded-md" />
      ) : (
        <ul className="space-y-1 text-[13px]">
          {found.data.map((m) => (
            <li key={m.column} className={m.violations ? "text-bad" : "text-ok"}>
              {m.sentence}
            </li>
          ))}
        </ul>
      )}
    </Block>
  );
}
