import { useState } from "react";

import type { MlExplanation, MlRun } from "../../api/client";
import { useMlExplanation } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { beeswarmFigure, dependenceFigure, permutationFigure, waterfallFigure } from "./figures";
import { Loading } from "../../ui/Loading";

type Explanation = NonNullable<MlExplanation>;
type TreeNode = NonNullable<Explanation["tree"]>[number];
type TextWords = Explanation["words"][number];
type Word = TextWords["up"][number];

function Section({ title, hint, children }: { title: string; hint?: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-[13px] font-medium text-ink-1">{title}</h3>
      {hint && <p className="max-w-prose text-xs text-ink-2">{hint}</p>}
      {children}
    </section>
  );
}

function Frame({ label, height, children }: { label: string; height: number; children: (h: number) => React.ReactNode }) {
  return (
    <figure className="min-w-0 rounded-md border border-line bg-surface-1 p-2" aria-label={label}>
      {children(height)}
    </figure>
  );
}

/** Fetches the explanation (computed in the background the first time) and renders `children`. */
export function WithExplanation({ datasetId, run, children }: { datasetId: string; run: MlRun; children: (found: Explanation) => React.ReactNode }) {
  const state = useMlExplanation(datasetId, run.id, true);
  if (state.isError) return <ErrorNote>{state.error.message}</ErrorNote>;
  if (!state.data || state.data.status === "running")
    return <Loading label={t("ml.explain.computing")} className="h-40 rounded-md" />;
  if (state.data.status === "failed" || !state.data.explanation) return <ErrorNote>{state.data.error ?? t("ml.explain.failed")}</ErrorNote>;
  return <>{children(state.data.explanation)}</>;
}

export function ExplainTab({ datasetId, run }: { datasetId: string; run: MlRun }) {
  return <WithExplanation datasetId={datasetId} run={run}>{(found) => <ExplainBody found={found} />}</WithExplanation>;
}

function ExplainBody({ found }: { found: Explanation }) {
  const [column, setColumn] = useState<string | null>(null);
  const permutation = found.permutation;
  const shap = found.shap;
  const chosen = found.dependence.find((d) => d.column === column) ?? found.dependence[0];
  return (
    <div className="space-y-6">
      {found.warnings.map((w) => (
        <p key={w} className="text-xs text-ink-3">
          {w}
        </p>
      ))}
      {permutation && (
        <Section title={t("ml.explain.permutation")} hint={t("ml.explain.permutationHint", { metric: t(`ml.metric.${permutation.metric}` as MessageKey), rows: permutation.rows.toLocaleString("en-US") })}>
          <Frame label={t("ml.explain.permutation")} height={Math.max(220, 24 * Math.min(15, permutation.items.length) + 60)}>
            {(h) => <Plot figure={permutationFigure(permutation.items, t(`ml.metric.${permutation.metric}` as MessageKey))} height={h} label={t("ml.explain.permutation")} />}
          </Frame>
        </Section>
      )}
      {found.pairs.length > 0 && (
        <ul className="space-y-1 rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-xs text-ink-1" aria-label={t("ml.explain.pairs")}>
          {found.pairs.slice(0, 8).map((p) => (
            <li key={`${p.first}-${p.second}`}>
              {t("ml.explain.pair", { first: p.first, second: p.second, measure: p.measure, strength: p.strength.toFixed(2) })}
            </li>
          ))}
        </ul>
      )}
      <Section title={t("ml.explain.shap")} hint={shap ? t("ml.explain.shapHint", { method: shap.method, scale: shap.scale, rows: shap.rows.length.toLocaleString("en-US") }) : undefined}>
        {shap ? (
          <>
            <Frame label={t("ml.explain.shap")} height={Math.max(260, 26 * shap.features.length + 80)}>
              {(h) => <Plot figure={beeswarmFigure(shap)} height={h} label={t("ml.explain.shap")} />}
            </Frame>
            <ul className="space-y-0.5 text-[13px] text-ink-2">
              {shap.features.flatMap((f) => (f.reading ? [<li key={f.column}>{f.reading}</li>] : [])).slice(0, 6)}
            </ul>
            {shap.positive && <p className="text-xs text-ink-3">{t("ml.explain.shapPositive", { positive: shap.positive })}</p>}
          </>
        ) : (
          <p className="max-w-prose text-[13px] text-ink-2">{found.shap_unavailable}</p>
        )}
      </Section>
      {found.words.map((w) => (
        <WordsSection key={w.column} words={w} />
      ))}
      {chosen && (
        <Section title={t("ml.explain.dependence")} hint={t("ml.explain.dependenceHint")}>
          <label className="flex items-center gap-2 text-xs text-ink-2">
            {t("ml.explain.column")}
            <Select aria-label={t("ml.explain.column")} value={chosen.column} onChange={(e) => setColumn(e.target.value)} className="h-7 text-xs">
              {found.dependence.map((d) => (
                <option key={d.column} value={d.column}>
                  {d.column}
                </option>
              ))}
            </Select>
          </label>
          <Frame label={t("ml.explain.dependence")} height={320}>
            {(h) => <Plot figure={dependenceFigure(chosen)} height={h} label={t("ml.explain.dependence")} />}
          </Frame>
        </Section>
      )}
      {found.tree && (
        <Section title={t("ml.explain.tree")} hint={t("ml.explain.treeHint")}>
          <TreeView nodes={found.tree} />
        </Section>
      )}
    </div>
  );
}

/** The terms of one free-text column that move the predictions most (052). */
function WordsSection({ words }: { words: TextWords }) {
  const hint = t(words.method === "linear" ? "ml.explain.wordsLinear" : "ml.explain.wordsContrast", {
    output: words.output,
    terms: words.terms.toLocaleString("en-US"),
  });
  return (
    <Section title={t("ml.explain.words", { column: words.column })} hint={hint}>
      <p className="max-w-prose text-[13px] text-ink-1">{words.sentence}</p>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <WordList label={t("ml.explain.wordsUp")} items={words.up} />
        <WordList label={t("ml.explain.wordsDown")} items={words.down} />
      </div>
    </Section>
  );
}

function WordList({ label, items }: { label: string; items: Word[] }) {
  return (
    <div className="min-w-0">
      <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={label}>
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="py-1 pr-3 font-normal">{label}</th>
            <th className="py-1 pr-3 text-right font-normal">{t("ml.explain.wordsEffect")}</th>
            <th className="py-1 text-right font-normal">{t("ml.explain.wordsRows")}</th>
          </tr>
        </thead>
        <tbody>
          {items.length === 0 ? (
            <tr className="border-t border-line">
              <td colSpan={3} className="py-1 text-ink-3">
                {t("ml.explain.wordsNone")}
              </td>
            </tr>
          ) : (
            items.map((w) => (
              <tr key={w.term} className="border-t border-line">
                <td className="max-w-0 truncate py-1 pr-3 font-mono text-xs text-ink-1" title={w.term}>
                  {w.term}
                  {w.towards && <span className="ml-1.5 font-sans text-[11px] text-ink-3">→ {w.towards}</span>}
                </td>
                <td className="py-1 pr-3 text-right text-ink-1">
                  {w.effect >= 0 ? "+" : ""}
                  {formatNumber(w.effect)}
                </td>
                <td className="py-1 text-right text-ink-2">{`${Math.round(w.rows * 100) || "<1"} %`}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

function TreeView({ nodes }: { nodes: TreeNode[] }) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const render = (node: TreeNode, rule: string | null): React.ReactNode => {
    const label = (
      <span className="text-[13px]">
        {rule && <span className="font-mono text-xs text-ink-2">{rule} → </span>}
        <span className="text-ink-1">{node.value}</span>
        <span className="text-xs text-ink-3"> · {t("ml.explain.samples", { count: node.samples.toLocaleString("en-US") })}</span>
      </span>
    );
    if (node.feature === null || node.left === null || node.right === null) return <li key={node.id} className="py-0.5 pl-4">{label}</li>;
    const left = byId.get(node.left);
    const right = byId.get(node.right);
    return (
      <li key={node.id}>
        <details open={node.depth < 2}>
          <summary className="cursor-pointer py-0.5">{label}</summary>
          <ul className="ml-3 border-l border-line pl-2">
            {left && render(left, `${node.feature} ≤ ${formatNumber(node.threshold)}`)}
            {right && render(right, `${node.feature} > ${formatNumber(node.threshold)}`)}
          </ul>
        </details>
      </li>
    );
  };
  const root = nodes[0];
  return <ul className="rounded-md border border-line bg-surface-1 p-2">{root && render(root, null)}</ul>;
}

/** The waterfall of one row: from the average prediction to this row's, top contributions first. */
export function RowExplanation({ datasetId, run, row }: { datasetId: string; run: MlRun; row: number }) {
  return (
    <WithExplanation datasetId={datasetId} run={run}>
      {(found) => {
        const shap = found.shap;
        if (!shap) return <p className="text-[13px] text-ink-2">{found.shap_unavailable}</p>;
        const index = shap.rows.indexOf(row);
        if (index < 0) return <p className="text-[13px] text-ink-2">{t("ml.explain.rowMissing")}</p>;
        const values = shap.matrix[index] ?? [];
        const labels = shap.labels[index] ?? [];
        const order = values.map((_, j) => j).sort((a, b) => Math.abs(values[b] ?? 0) - Math.abs(values[a] ?? 0));
        const top = order.slice(0, 10);
        const rest = order.slice(10);
        const steps = top.map((j) => ({ label: `${shap.columns[j]} = ${labels[j]}`, value: values[j] ?? 0 }));
        if (rest.length) steps.push({ label: t("ml.explain.others", { count: String(rest.length) }), value: rest.reduce((sum, j) => sum + (values[j] ?? 0), 0) });
        const strongest = top.slice(0, 3).map((j) => `${shap.columns[j]} (${(values[j] ?? 0) >= 0 ? "+" : ""}${formatNumber(values[j] ?? 0)})`);
        return (
          <div className="space-y-2">
            <p className="text-[13px] text-ink-1">{t("ml.explain.rowSentence", { row: String(row + 1), columns: strongest.join(", ") })}</p>
            <Frame label={t("ml.explain.row", { row: String(row + 1) })} height={Math.max(240, 26 * (steps.length + 2) + 60)}>
              {(h) => <Plot figure={waterfallFigure(shap.base, steps, shap.scale)} height={h} label={t("ml.explain.row", { row: String(row + 1) })} />}
            </Frame>
            {shap.positive && <p className="text-xs text-ink-3">{t("ml.explain.shapPositive", { positive: shap.positive })}</p>}
          </div>
        );
      }}
    </WithExplanation>
  );
}
