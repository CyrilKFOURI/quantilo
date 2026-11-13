import type { Data } from "plotly.js-dist-min";

import { Plot, type Figure } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { VisionModel } from "./modelApi";
import { Block } from "./resultsUi";

/**
 * 118: what an Advanced › More model adds to Results: how its training went (the loss of each
 * epoch, the held-out loss that stopped it, the epoch kept) and, for the concept bottleneck,
 * how well each concept is read and how it pushes each class (R-VFT-04, R-VPP-03, R-VCB-04).
 */

type Epoch = { epoch: number; train_loss: number; val_loss: number | null; val_balanced_accuracy: number | null };
type Log = { history: Epoch[]; best_epoch: number; stopped: string; device: string; precision: string; epochs: number; seconds: number };
type ConceptScore = { key: string; name: string; kind: string; metric: string; value: number | null; weak: boolean; known: number; setting: string; weights: Record<string, number> };

function historyFigure(log: Log): Figure {
  return (tokens) => {
    const x = log.history.map((e) => e.epoch);
    const series: Data[] = [
      { type: "scatter", mode: "lines+markers", name: t("vision.more.trainLoss"), x, y: log.history.map((e) => e.train_loss), line: { color: tokens.palette[0] ?? tokens.numeric, width: 2 }, marker: { size: 5 } },
    ];
    if (log.history.some((e) => e.val_loss !== null)) {
      series.push({ type: "scatter", mode: "lines+markers", name: t("vision.more.heldLoss"), x, y: log.history.map((e) => e.val_loss), line: { color: tokens.palette[1] ?? tokens.ink2, width: 2 }, marker: { size: 5 } });
    }
    return {
      data: series,
      layout: {
        xaxis: { title: { text: t("vision.more.epoch"), font: { size: 11, color: tokens.ink3 } }, dtick: 1, showgrid: false, zeroline: false },
        yaxis: { showgrid: true, gridcolor: tokens.line, zeroline: false },
        shapes: [{ type: "line", x0: log.best_epoch, x1: log.best_epoch, yref: "paper", y0: 0, y1: 1, line: { color: tokens.ink3, width: 1, dash: "dot" } }],
        showlegend: true,
        legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11, color: tokens.ink2 } },
        margin: { l: 8, r: 8, t: 24, b: 8 },
      },
    };
  };
}

export function MoreResults({ model }: { model: VisionModel }) {
  const logs = (model.training ?? []) as unknown as Log[];
  const concepts = (model.concepts ?? []) as unknown as ConceptScore[];
  const log = logs[0];
  const classes = concepts.length > 0 ? Object.keys(concepts[0]?.weights ?? {}) : [];
  return (
    <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
      {log && log.history.length > 0 && (
        <Block id="vision-more-training" title={t("vision.more.trainingTitle")} conclusion={log.stopped}>
          <Plot figure={historyFigure(log)} height={220} label={t("vision.more.trainingTitle")} />
          <p className="mt-1 text-xs text-ink-3 tabular-nums">{t("vision.more.trainingFacts", { epochs: formatCount(log.epochs), best: formatCount(log.best_epoch), device: log.device, precision: log.precision, seconds: log.seconds.toFixed(0) })}</p>
        </Block>
      )}
      {concepts.length > 0 && (
        <Block id="vision-more-concepts" title={t("vision.more.conceptQuality")} conclusion={t("vision.more.conceptQualityNote")}>
          <div className="max-w-full overflow-x-auto">
            <table className="w-full text-xs tabular-nums">
              <caption className="sr-only">{t("vision.more.conceptQuality")}</caption>
              <thead>
                <tr className="text-left text-ink-3">
                  <th scope="col" className="py-1 pr-3 font-normal">{t("vision.more.concept")}</th>
                  <th scope="col" className="py-1 pr-3 text-right font-normal">{t("vision.more.readScore")}</th>
                  {classes.map((label) => (
                    <th key={label} scope="col" className="py-1 pr-3 text-right font-mono font-normal">
                      {label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {concepts.map((c) => (
                  <tr key={c.key} className="border-t border-line">
                    <th scope="row" className="py-2 pr-3 text-left font-normal text-ink-1">
                      {c.name}
                      {c.weak && <span className="ml-1.5 text-warn">{t("vision.more.weak")}</span>}
                    </th>
                    <td className="py-2 pr-3 text-right text-ink-1">
                      {c.value === null ? "—" : c.value.toFixed(3)} <span className="text-ink-3">{c.metric === "roc_auc" ? "AUC" : "R²"}</span>
                    </td>
                    {classes.map((label) => {
                      const w = c.weights[label] ?? 0;
                      return (
                        <td key={label} className={`py-2 pr-3 text-right ${w === 0 ? "text-ink-3" : "text-ink-1"}`}>
                          {w > 0 ? "+" : w < 0 ? "−" : ""}
                          {Math.abs(w).toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Block>
      )}
    </div>
  );
}
