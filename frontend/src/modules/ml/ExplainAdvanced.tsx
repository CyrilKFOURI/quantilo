import { useState } from "react";

import type { MlExplanation, MlRun, MlSurrogate } from "../../api/client";
import { useMlAle, useMlAle2d, useMlMonotonic, useMlSurrogate } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";
import { ExplainMore } from "./ExplainMore";
import { ale2dFigure, aleFigure } from "./figures";

type Explanation = NonNullable<MlExplanation>;
/** A tree's node as the surrogate, the sparse tree and the optimal tree give it (125, 122). */
export type TreeNode = Pick<MlSurrogate["nodes"][number], "id" | "value" | "samples" | "left" | "right" | "left_rule" | "right_rule">;

const BINS = 20;
const DEPTH = 3;
/** Models whose training takes monotonic constraints (Settings › Advanced, 019). */
const CONSTRAINED = new Set(["hist_gradient_boosting", "xgboost", "lightgbm"]);
const SKIPPED_TYPES = new Set(["text", "datetime", "identifier"]);
const NUMBER_TYPES = new Set(["numeric_continuous", "numeric_discrete"]);

function Block({ title, hint, children }: { title: string; hint: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h4 className="text-[13px] font-medium text-ink-1">{title}</h4>
      <p className="max-w-prose text-xs text-ink-2">{hint}</p>
      {children}
    </section>
  );
}

/** Explain › Advanced (125): nothing here is computed before the user opens it; More (126) is
 * folded inside it, and each of its methods waits for its own button. */
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
  const [more, setMore] = useState(false);
  return (
    <div className="mt-3 space-y-6">
      <AleBlock datasetId={datasetId} run={run} found={found} column={column} onColumn={onColumn} />
      <SurrogateBlock datasetId={datasetId} run={run} />
      <MonotonicBlock datasetId={datasetId} run={run} />
      <details className="text-xs" onToggle={(event) => setMore(event.currentTarget.open)}>
        <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("ml.explain.more")}</summary>
        {more && <ExplainMore datasetId={datasetId} run={run} found={found} />}
      </details>
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
  // 125 bis (R-XAI-20): a second number, chosen by the user; nothing is computed before
  const [second, setSecond] = useState<string | null>(null);
  const numbers = run.preparation.decisions.filter((d) => d.use && NUMBER_TYPES.has(d.type)).map((d) => d.column);
  const pair = second && chosen && second !== chosen && numbers.includes(chosen) ? second : null;
  const surface = useMlAle2d(datasetId, run.id, pair ? chosen : null, pair);
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
        {numbers.includes(chosen) && numbers.length >= 2 && (
          <Cell label={t("ml.explain.ale2d")} why={t("ml.explain.ale2dWhy")} changed={second !== null} onReset={() => setSecond(null)}>
            <Select aria-label={t("ml.explain.ale2d")} value={second ?? ""} onChange={(e) => setSecond(e.target.value || null)} className="w-full text-xs">
              <option value="">{t("ml.explain.ale2dNone")}</option>
              {numbers
                .filter((c) => c !== chosen)
                .map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
            </Select>
          </Cell>
        )}
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
      {pair &&
        (surface.isError ? (
          <ErrorNote>{surface.error.message}</ErrorNote>
        ) : !surface.data ? (
          <Loading label={t("ml.explain.ale2dComputing")} compact className="h-40 rounded-md" />
        ) : (
          <div className="space-y-2">
            <p className="max-w-prose text-[13px] text-ink-1">{surface.data.sentence}</p>
            <figure className="min-w-0 rounded-md border border-line bg-surface-1 p-2" aria-label={t("ml.explain.ale2d")}>
              <Plot figure={ale2dFigure(surface.data)} height={340} label={t("ml.explain.ale2dLabel", { first: surface.data.first, second: surface.data.second })} />
            </figure>
            <p className="text-xs text-ink-3">{t("ml.explain.ale2dRows", { rows: surface.data.rows.toLocaleString("en-US") })}</p>
          </div>
        ))}
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

/** A tree as nested questions, each branch folded open. */
export function SurrogateTree({ nodes, label = t("ml.explain.surrogate") }: { nodes: TreeNode[]; label?: string }) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const render = (node: TreeNode, rule: string | null): React.ReactNode => {
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
  return <ul className="rounded-md border border-line bg-surface-1 p-2" aria-label={label}>{root && render(root, null)}</ul>;
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
