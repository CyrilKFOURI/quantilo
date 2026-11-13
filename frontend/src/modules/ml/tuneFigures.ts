import type { Data } from "plotly.js-dist-min";

import type { Figure } from "../../charts/Plot";
import { t } from "../../lib/i18n";

type ParetoTrial = { number: number; value: number | null; state: string; seconds: number; pareto: boolean };

const title = (text: string) => ({ title: { text, font: { size: 11 } } });

/** Two objectives (066): every trial by its seconds and score, the Pareto front joined by a line
 *  (no trial on it is beaten on both the score and the time). */
export function paretoFigure(trials: ParetoTrial[], metric: string): Figure {
  return (tokens) => {
    const done = trials.filter((t) => t.state === "complete" && t.value !== null);
    const front = done.filter((t) => t.pareto).sort((a, b) => a.seconds - b.seconds);
    const others = done.filter((t) => !t.pareto);
    return {
      data: [
        {
          type: "scatter",
          mode: "markers",
          name: t("fig.ml.trial"),
          x: others.map((t) => t.seconds),
          y: others.map((t) => t.value),
          text: others.map((t) => String(t.number)),
          marker: { color: tokens.ink3, size: 6 },
          hovertemplate: t("fig.ml.hoverTrialSeconds"),
        } as Data,
        {
          type: "scatter",
          mode: "lines+markers",
          name: t("fig.ml.paretoFront"),
          x: front.map((t) => t.seconds),
          y: front.map((t) => t.value),
          text: front.map((t) => String(t.number)),
          line: { color: tokens.palette[1] ?? tokens.categorical, width: 2, shape: "hv" },
          marker: { color: tokens.palette[0] ?? tokens.numeric, size: 9 },
          hovertemplate: t("fig.ml.hoverTrialSeconds"),
        } as Data,
      ],
      layout: { showlegend: true, xaxis: title(t("fig.ml.secondsPerTrial")), yaxis: title(metric) },
    };
  };
}
