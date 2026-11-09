import { useState } from "react";

import type { SyntheticPair, SyntheticRun } from "../../api/client";
import { useSyntheticPair } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { linksFigure, pairFigure } from "./figures";
import { pct } from "./format";
import { Table } from "./Report";
import { Loading } from "../../ui/Loading";

const MEASURES: Record<string, string> = { pearson: "r", spearman: "ρ", cramers_v: "V", eta: "η" };

function CrossTable({ pair, side }: { pair: SyntheticPair; side: "real" | "synthetic" }) {
  const table = side === "real" ? pair.real_table : pair.synthetic_table;
  return (
    <Table caption={t(side === "real" ? "synthetic.real" : "synthetic.synthetic")} head={["", ...pair.columns]}>
      {pair.rows.map((row, i) => (
        <tr key={row} className="border-b border-line/60">
          <td className="py-1 pr-3 text-ink-2">{row}</td>
          {(table[i] ?? []).map((value, j) => (
            <td key={j} className="py-1 pr-3 text-right">
              {pct(value, 1)}
            </td>
          ))}
        </tr>
      ))}
    </Table>
  );
}

/** Three matrices (real, synthetic, |gap|), the pairs that changed most, and one pair opened. */
export function LinksTab({ run, datasetId }: { run: SyntheticRun; datasetId: string }) {
  const links = run.report.links;
  const first = links.worst[0];
  const [pair, setPair] = useState<[string, string] | null>(first ? [first.a, first.b] : null);
  const detail = useSyntheticPair(datasetId, run.id, pair);
  if (links.columns.length < 2) return <p className="text-[13px] text-ink-3">{t("synthetic.noPairs")}</p>;
  const size = Math.min(420, 90 + 22 * links.columns.length);
  const open = (x: unknown, y: unknown) => {
    if (typeof x === "string" && typeof y === "string" && x !== y) setPair([y, x]);
  };
  return (
    <div className="space-y-6">
      <div className="grid gap-6 lg:grid-cols-3">
        {(["real", "synthetic", "gap"] as const).map((which) => (
          <figure key={which} className="min-w-0 space-y-1">
            <figcaption className="text-[13px] font-medium text-ink-1">{t(`synthetic.links.${which}` as MessageKey)}</figcaption>
            <Plot figure={linksFigure(links, which)} height={size} label={t(`synthetic.links.${which}` as MessageKey)} onPointClick={(p) => open(p.x, p.y)} />
          </figure>
        ))}
      </div>
      <p className="text-xs text-ink-3">{t("synthetic.linksHelp")}</p>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,3fr)]">
        <Table caption={t("synthetic.worstPairs")} head={[t("synthetic.pair"), t("synthetic.real"), t("synthetic.synthetic"), t("synthetic.gap")]}>
          {links.worst.map((p) => {
            const active = pair?.[0] === p.a && pair[1] === p.b;
            return (
              <tr key={`${p.a}-${p.b}`} className={`border-b border-line/60 ${active ? "bg-surface-1" : ""}`}>
                <td className="py-1.5 pr-3">
                  <button type="button" onClick={() => setPair([p.a, p.b])} aria-pressed={active} className="text-left text-accent hover:underline">
                    {p.a} · {p.b}
                  </button>{" "}
                  <span className="text-xs text-ink-3">{MEASURES[p.measure]}</span>
                </td>
                <td className="py-1.5 pr-3 text-right">{p.real === null ? "–" : p.real.toFixed(2)}</td>
                <td className="py-1.5 pr-3 text-right">{p.synthetic === null ? "–" : p.synthetic.toFixed(2)}</td>
                <td className="py-1.5 pr-3 text-right text-warn">{p.gap === null ? "–" : p.gap.toFixed(2)}</td>
              </tr>
            );
          })}
        </Table>
        <div className="min-w-0 space-y-2">
          {pair && <h3 className="text-[13px] font-medium text-ink-1">{pair.join(" · ")}</h3>}
          {detail.data?.kind === "categories" ? (
            <div className="grid gap-4 md:grid-cols-2">
              <CrossTable pair={detail.data} side="real" />
              <CrossTable pair={detail.data} side="synthetic" />
            </div>
          ) : detail.data ? (
            <Plot figure={pairFigure(detail.data)} height={300} label={pair?.join(" · ") ?? ""} />
          ) : (
            pair && <Loading label={t("loading.chart")} className="h-64" />
          )}
        </div>
      </div>
    </div>
  );
}
