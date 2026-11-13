import { useMemo } from "react";

import type { Figure } from "../../charts/Plot";
import { Plot } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import type { ImageRow } from "./model";
import type { Prediction, VisionModel } from "./modelApi";
import { barsFigure } from "./resultFigures";
import { metric, percent } from "./scores";
import { Block, Gallery, MetricTable, Tiles, type GalleryItem } from "./resultsUi";

type Props = {
  datasetId: string;
  rows: readonly ImageRow[];
  model: VisionModel;
  predictions: readonly Prediction[];
  target?: string | null;
  onOpen: (row: number, list: number[]) => void;
};

type Interval = { low: number; high: number } | undefined;
const range = (found: Interval) => (found ? t("vision.results.ci", { range: `[${metric(found.low)}; ${metric(found.high)}]` }) : undefined);

/** c3, R-VRG-02: predicted against true, one dot per scored image, the line of perfect predictions
 *  and the training mean (the baseline) as references. */
function predictedFigure(truth: number[], predicted: number[], mean: number, target: string): Figure {
  return (tokens) => {
    const low = Math.min(...truth, ...predicted);
    const high = Math.max(...truth, ...predicted);
    return {
      data: [
        { type: "scatter", mode: "lines", x: [low, high], y: [low, high], line: { color: tokens.ink3, width: 1, dash: "dot" }, hoverinfo: "skip", name: t("fig.vision.perfect") },
        { type: "scatter", mode: "lines", x: [low, high], y: [mean, mean], line: { color: tokens.line, width: 1 }, hoverinfo: "skip", name: t("fig.mean") },
        {
          type: truth.length > 2000 ? "scattergl" : "scatter",
          mode: "markers",
          x: truth,
          y: predicted,
          marker: { color: tokens.palette[0], size: 6, opacity: 0.7 },
          hovertemplate: t("fig.vision.hoverTruePredicted"),
          name: target,
        },
      ],
      layout: {
        xaxis: { title: { text: t("vision.target.trueAxis", { target }), font: { size: 11, color: tokens.ink3 } }, showgrid: true, gridcolor: tokens.line, zeroline: false },
        yaxis: { title: { text: t("vision.target.predictedAxis"), font: { size: 11, color: tokens.ink3 } }, showgrid: true, gridcolor: tokens.line, zeroline: false },
        showlegend: false,
        margin: { l: 8, r: 8, t: 8, b: 8 },
        annotations: [
          { x: high, y: high, text: t("vision.target.perfect"), showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 11, color: tokens.ink3 } },
          { x: high, y: mean, text: t("vision.target.mean"), showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 11, color: tokens.ink3 } },
        ],
      },
    };
  };
}

/** c3, R-VRG-02: the metrics of a regression on images, the baseline beside each. */
export function RegressionResults({ datasetId, rows, model, predictions, target, onOpen }: Props) {
  const report = model.regression;
  const key = `${model.key}_pred`;
  const pairs = useMemo(() => {
    const truth: number[] = [];
    const predicted: number[] = [];
    for (const p of predictions) {
      const y = p.truth;
      const value = p[key];
      if (typeof y === "number" && typeof value === "number") {
        truth.push(y);
        predicted.push(value);
      }
    }
    return { truth, predicted };
  }, [predictions, key]);
  const byRow = useMemo(() => new Map(predictions.map((p) => [p.row, p])), [predictions]);
  if (!report) return null;
  const name = target ?? t("vision.target.value");
  const item = (row: number): GalleryItem => {
    const found = byRow.get(row);
    const y = typeof found?.truth === "number" ? found.truth : null;
    const value = typeof found?.[key] === "number" ? (found[key] as number) : null;
    return {
      row,
      caption: t("vision.target.predicted", { value: value === null ? "—" : formatNumber(value) }),
      detail: t("vision.target.truth", { value: y === null ? "—" : formatNumber(y) }),
      tone: "bad",
    };
  };
  const gain = report.baseline_mae > 0 ? 1 - report.mae / report.baseline_mae : 0;
  return (
    <div className="flex flex-col gap-8">
      <Tiles
        label={t("vision.results.mainMetrics")}
        tiles={[
          { label: t("vision.target.m.r2"), value: metric(report.r2), detail: range(report.intervals.r2), tip: t("vision.target.tip.r2"), main: true },
          { label: t("vision.target.m.mae"), value: formatNumber(report.mae), detail: t("vision.target.baseline", { value: formatNumber(report.baseline_mae) }), tip: t("vision.target.tip.mae") },
          { label: t("vision.target.m.rmse"), value: formatNumber(report.rmse), detail: t("vision.target.baseline", { value: formatNumber(report.baseline_rmse) }), tip: t("vision.target.tip.rmse") },
          { label: t("vision.target.m.spearman"), value: metric(report.spearman ?? null), tip: t("vision.target.tip.spearman") },
        ]}
      />
      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block
          id="vision-predicted"
          title={t("vision.target.predictedTitle", { target: name })}
          conclusion={t("vision.target.predictedSentence", { r2: metric(report.r2), gain: percent(gain, 0), n: formatCount(report.n) })}
        >
          <Plot figure={predictedFigure(pairs.truth, pairs.predicted, report.train_mean, name)} height={340} label={t("vision.target.predictedTitle", { target: name })} />
        </Block>
        <Block id="vision-regression-all" title={t("vision.results.allMetrics")}>
          <MetricTable
            caption={t("vision.results.allMetrics")}
            rows={[
              { label: t("vision.target.m.r2"), value: metric(report.r2), detail: range(report.intervals.r2), tip: t("vision.target.tip.r2") },
              { label: t("vision.target.m.mae"), value: formatNumber(report.mae), detail: range(report.intervals.mae), tip: t("vision.target.tip.mae") },
              { label: t("vision.target.m.rmse"), value: formatNumber(report.rmse), tip: t("vision.target.tip.rmse") },
              { label: t("vision.target.m.median"), value: formatNumber(report.median_ae), tip: t("vision.target.tip.median") },
              { label: t("vision.target.m.spearman"), value: metric(report.spearman ?? null), tip: t("vision.target.tip.spearman") },
              { label: t("vision.target.m.baselineMae"), value: formatNumber(report.baseline_mae), detail: t("vision.target.trainMean", { value: formatNumber(report.train_mean) }), tip: t("vision.target.tip.baseline") },
            ]}
          />
        </Block>
      </div>
      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-over" title={t("vision.target.over")} conclusion={t("vision.target.overSentence")}>
          <Gallery datasetId={datasetId} rows={rows} items={(model.galleries.over ?? []).map(item)} empty={t("vision.results.noImage")} onOpen={onOpen} label={t("vision.target.over")} />
        </Block>
        <Block id="vision-under" title={t("vision.target.under")} conclusion={t("vision.target.underSentence")}>
          <Gallery datasetId={datasetId} rows={rows} items={(model.galleries.under ?? []).map(item)} empty={t("vision.results.noImage")} onOpen={onOpen} label={t("vision.target.under")} />
        </Block>
      </div>
    </div>
  );
}

/** c3, R-VMLB-02: the metrics of a multi-label model, then each label's. */
export function MultilabelResults({ datasetId, rows, model, predictions, onOpen }: Props) {
  const report = model.multilabel;
  const byRow = useMemo(() => new Map(predictions.map((p) => [p.row, p])), [predictions]);
  if (!report) return null;
  const labelsOf = (value: unknown) => (Array.isArray(value) ? value.map(String).join(", ") || "—" : "—");
  const wrong = (model.galleries.most_wrong ?? []).map(
    (row): GalleryItem => ({
      row,
      caption: t("vision.target.predicted", { value: labelsOf(byRow.get(row)?.[`${model.key}_pred`]) }),
      detail: t("vision.target.truth", { value: labelsOf(byRow.get(row)?.truth) }),
      tone: "bad",
    }),
  );
  const f1 = report.per_label.map((label) => ({ name: label.label, value: label.f1 }));
  return (
    <div className="flex flex-col gap-8">
      <Tiles
        label={t("vision.results.mainMetrics")}
        tiles={[
          { label: t("vision.target.m.macroF1"), value: metric(report.macro_f1), detail: [range(report.intervals.macro_f1), t("vision.results.baseline", { value: metric(report.baseline_macro_f1) })].filter(Boolean).join(" · "), tip: t("vision.target.tip.macroF1"), main: true },
          { label: t("vision.target.m.microF1"), value: metric(report.micro_f1), tip: t("vision.target.tip.microF1") },
          { label: t("vision.target.m.exact"), value: metric(report.exact_match), detail: t("vision.results.baseline", { value: metric(report.baseline_exact_match) }), tip: t("vision.target.tip.exact") },
          { label: t("vision.target.m.hamming"), value: metric(report.hamming_loss), detail: t("vision.results.baseline", { value: metric(report.baseline_hamming_loss) }), tip: t("vision.target.tip.hamming") },
        ]}
      />
      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-labels-f1" title={t("vision.target.byLabel")} conclusion={report.sentence}>
          <Plot figure={barsFigure(f1, { range: [0, 1.1] })} height={Math.max(160, 26 * f1.length + 30)} label={t("vision.target.byLabel")} />
        </Block>
        <Block id="vision-labels-table" title={t("vision.target.labelTable")}>
          <div className="max-w-full overflow-x-auto">
            <table className="w-full border-collapse text-[13px] tabular-nums">
              <caption className="sr-only">{t("vision.target.labelTable")}</caption>
              <thead>
                <tr className="text-xs text-ink-3">
                  <th scope="col" className="py-1.5 pr-3 text-left font-normal">{t("vision.target.label")}</th>
                  <th scope="col" className="px-2 py-1.5 text-right font-normal">F1</th>
                  <th scope="col" className="px-2 py-1.5 text-right font-normal">{t("vision.target.precision")}</th>
                  <th scope="col" className="px-2 py-1.5 text-right font-normal">{t("vision.target.recall")}</th>
                  <th scope="col" className="px-2 py-1.5 text-right font-normal max-sm:hidden">AUROC</th>
                  <th scope="col" className="py-1.5 pl-2 text-right font-normal">{t("vision.target.support")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line border-y border-line">
                {report.per_label.map((label) => (
                  <tr key={label.label}>
                    <th scope="row" className="py-1.5 pr-3 text-left font-normal text-ink-1">{label.label}</th>
                    <td className="px-2 py-1.5 text-right text-ink-1">{metric(label.f1)}</td>
                    <td className="px-2 py-1.5 text-right text-ink-2">{metric(label.precision)}</td>
                    <td className="px-2 py-1.5 text-right text-ink-2">{metric(label.recall)}</td>
                    <td className="px-2 py-1.5 text-right text-ink-2 max-sm:hidden">{metric(label.auroc ?? null)}</td>
                    <td className="py-1.5 pl-2 text-right text-ink-3">
                      {formatCount(label.support)} <span className="text-xs">({percent(label.prevalence, 0)})</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-ink-3">{t("vision.target.thresholdNote", { threshold: String(report.threshold) })}</p>
        </Block>
      </div>
      <Block id="vision-most-wrong" title={t("vision.target.mostWrong")} conclusion={t("vision.target.mostWrongSentence")}>
        <Gallery datasetId={datasetId} rows={rows} items={wrong} empty={t("vision.results.noImage")} onOpen={onOpen} label={t("vision.target.mostWrong")} />
      </Block>
    </div>
  );
}
