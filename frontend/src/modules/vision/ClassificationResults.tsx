import { useMemo, useState } from "react";

import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { ConfusionMatrix } from "./ConfusionMatrix";
import type { ImageRow } from "./model";
import type { ClassificationReport, Prediction, VisionModel } from "./modelApi";
import { curvesFigure, reliabilityFigure } from "./resultFigures";
import { cellRows, classCurves, metric, num, percent, text } from "./scores";
import { useChartTokens } from "./Threshold";
import { Block, Gallery, MetricTable, Tiles, type GalleryItem, type MetricRow, type Tile } from "./resultsUi";

type Props = {
  datasetId: string;
  rows: readonly ImageRow[];
  model: VisionModel;
  report: ClassificationReport;
  predictions: readonly Prediction[];
  onOpen: (row: number, list: number[]) => void;
};

const interval = (report: ClassificationReport, key: string) => {
  const found = report.intervals[key];
  return found ? `[${metric(found.low)}; ${metric(found.high)}]` : undefined;
};

/** 113 (§ 7.1): every classification metric of the chosen model, the main one first. */
export function ClassificationResults({ datasetId, rows, model, report, predictions, onOpen }: Props) {
  const [mode, setMode] = useState<"counts" | "rows">("counts");
  const [cell, setCell] = useState<{ truth: string; predicted: string } | null>(null);
  const [scaled, setScaled] = useState<"trained" | "scaled">("trained");
  const labels = report.labels;
  const key = model.key;
  const byRow = useMemo(() => new Map(predictions.map((p) => [p.row, p])), [predictions]);
  const curves = useMemo(() => classCurves(predictions, key, labels), [predictions, key, labels]);
  const main = report.main.metric;
  const mainValue = report[main];
  const floor = main === "balanced_accuracy" ? report.baseline.balanced_accuracy : report.baseline.accuracy;
  const ece = report.calibration?.reliability.ece ?? null;
  const topK = Object.entries(report.top_k).sort((a, b) => Number(a[0]) - Number(b[0]));

  const tiles: Tile[] = [
    {
      label: t(main === "balanced_accuracy" ? "vision.results.m.balanced_accuracy" : "vision.results.m.accuracy"),
      value: metric(mainValue),
      detail: [interval(report, main) && t("vision.results.ci", { range: interval(report, main) ?? "" }), t("vision.results.baseline", { value: metric(floor) })].filter(Boolean).join(" · "),
      tip: report.main.reason,
      main: true,
    },
    { label: t("vision.results.m.f1_macro"), value: metric(report.f1_macro), detail: interval(report, "f1_macro") && t("vision.results.ci", { range: interval(report, "f1_macro") ?? "" }), tip: t("vision.results.tip.f1_macro") },
    { label: t("vision.results.m.roc_auc"), value: metric(report.roc_auc), detail: labels.length > 2 ? t("vision.results.macroOvr") : undefined, tip: t("vision.results.tip.roc_auc") },
    { label: t("vision.results.m.ece"), value: metric(ece), detail: calibrationDetail(report), tip: t("vision.results.tip.ece") },
    ...topK.slice(0, 1).map(([k, value]) => ({ label: t("vision.results.m.topk", { k }), value: metric(value), tip: t("vision.results.tip.topk", { k }) })),
  ];

  const all: MetricRow[] = [
    { label: t("vision.results.m.accuracy"), value: metric(report.accuracy), detail: interval(report, "accuracy"), tip: t("vision.results.tip.accuracy") },
    { label: t("vision.results.m.balanced_accuracy"), value: metric(report.balanced_accuracy), detail: interval(report, "balanced_accuracy"), tip: t("vision.results.tip.balanced_accuracy") },
    { label: t("vision.results.m.f1_macro"), value: metric(report.f1_macro), detail: interval(report, "f1_macro"), tip: t("vision.results.tip.f1_macro") },
    { label: t("vision.results.m.kappa"), value: metric(report.kappa), detail: interval(report, "kappa"), tip: t("vision.results.tip.kappa") },
    { label: t("vision.results.m.roc_auc"), value: metric(report.roc_auc), detail: interval(report, "roc_auc") ?? (labels.length > 2 ? t("vision.results.macroOvr") : undefined), tip: t("vision.results.tip.roc_auc") },
    { label: t("vision.results.m.pr_auc"), value: metric(report.pr_auc), detail: labels.length > 2 ? t("vision.results.macroOvr") : undefined, tip: t("vision.results.tip.pr_auc") },
    { label: t("vision.results.m.log_loss"), value: metric(report.log_loss), tip: t("vision.results.tip.log_loss") },
    { label: t("vision.results.m.brier"), value: metric(report.brier), tip: t("vision.results.tip.brier") },
    { label: t("vision.results.m.ece"), value: metric(ece), detail: t("vision.results.bins", { count: String(report.calibration?.reliability.counts.length ?? 15) }), tip: t("vision.results.tip.ece") },
    { label: t("vision.results.m.mce"), value: metric(report.calibration?.reliability.mce ?? null), tip: t("vision.results.tip.mce") },
    ...topK.map(([k, value]) => ({ label: t("vision.results.m.topk", { k }), value: metric(value), tip: t("vision.results.tip.topk", { k }) })),
    { label: t("vision.results.m.baselineAcc"), value: metric(report.baseline.accuracy), detail: report.baseline.label, tip: report.baseline.reason },
    { label: t("vision.results.m.baselineBal"), value: metric(report.baseline.balanced_accuracy), tip: report.baseline.reason },
    { label: t("vision.results.m.n"), value: formatCount(report.n) },
  ];

  const errorItems = (list: number[]): GalleryItem[] =>
    list.map((row) => {
      const p = byRow.get(row);
      const truth = text(p?.truth);
      const predicted = text(p?.[`${key}_pred`]);
      return {
        row,
        caption: truth === predicted ? truth : `${truth} → ${predicted}`,
        detail: t("vision.results.confidence", { value: percent(num(p?.[`${key}_proba`]), 0) }),
        tone: truth === predicted ? undefined : ("bad" as const),
      };
    });
  const sure = (row: number) => num(byRow.get(row)?.[`${key}_proba`]);
  const cellItems = cell ? errorItems(cellRows(predictions, key, cell.truth, cell.predicted).sort((a, b) => sure(b) - sure(a))) : [];
  const worst = [...report.per_class].sort((a, b) => a.recall - b.recall)[0];
  const calibration = report.calibration;
  const after = calibration?.applied ? (calibration.reliability_after ?? null) : null;
  const reliability = after && scaled === "scaled" ? after : calibration?.reliability;

  return (
    <div className="flex flex-col gap-8">
      <Tiles tiles={tiles} label={t("vision.results.mainMetrics")} />

      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-all-metrics" title={t("vision.results.allMetrics")} conclusion={t("vision.results.intervalNote")}>
          <MetricTable rows={all} caption={t("vision.results.allMetrics")} />
        </Block>
        <Block
          id="vision-per-class"
          title={t("vision.results.perClass")}
          conclusion={worst ? t("vision.results.perClassSentence", { label: worst.label, recall: percent(worst.recall, 0) }) : undefined}
        >
          <div className="max-w-full overflow-x-auto">
            <table className="w-full border-collapse text-[13px] tabular-nums">
              <thead>
                <tr className="text-xs text-ink-3">
                  <th scope="col" className="py-1.5 pr-2 text-left font-normal">{t("vision.results.class")}</th>
                  <th scope="col" className="px-1.5 py-1.5 text-right font-normal">{t("vision.results.precision")}</th>
                  <th scope="col" className="px-1.5 py-1.5 text-right font-normal">{t("vision.results.recall")}</th>
                  <th scope="col" className="px-1.5 py-1.5 text-right font-normal">F1</th>
                  <th scope="col" className="px-1.5 py-1.5 text-right font-normal">{t("vision.results.support")}</th>
                  <th scope="col" className="px-1.5 py-1.5 text-right font-normal max-sm:hidden">ROC AUC</th>
                  <th scope="col" className="py-1.5 pl-1.5 text-right font-normal max-sm:hidden">PR AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line border-y border-line">
                {report.per_class.map((c) => (
                  <tr key={c.label}>
                    <th scope="row" className="max-w-40 truncate py-1.5 pr-2 text-left font-medium text-ink-1" title={c.label}>{c.label}</th>
                    <td className="px-1.5 py-1.5 text-right">{metric(c.precision)}</td>
                    <td className="px-1.5 py-1.5 text-right">{metric(c.recall)}</td>
                    <td className="px-1.5 py-1.5 text-right">{metric(c.f1)}</td>
                    <td className="px-1.5 py-1.5 text-right text-ink-2">{formatCount(c.support)}</td>
                    <td className="px-1.5 py-1.5 text-right text-ink-2 max-sm:hidden">{metric(report.roc_auc_by_class[c.label] ?? null)}</td>
                    <td className="py-1.5 pl-1.5 text-right text-ink-2 max-sm:hidden">{metric(report.pr_auc_by_class[c.label] ?? null)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Block>
      </div>

      <Block
        id="vision-confusion"
        title={t("vision.results.confusion")}
        conclusion={t("vision.results.confusionSentence")}
        aside={
          <SegmentedControl
            label={t("vision.results.confusionMode")}
            value={mode}
            onChange={setMode}
            options={[
              { value: "counts", label: t("vision.results.counts") },
              { value: "rows", label: t("vision.results.byRow") },
            ]}
          />
        }
      >
        <ConfusionMatrix labels={labels} counts={report.confusion} normalized={report.confusion_normalized} mode={mode} selected={cell} onSelect={setCell} />
        {cell && (
          <div className="mt-4">
            <p className="mb-2 text-[13px] text-ink-2">
              {t("vision.results.cellShown", { count: formatCount(cellItems.length), truth: cell.truth, predicted: cell.predicted })}{" "}
              <button type="button" onClick={() => setCell(null)} className="text-accent hover:underline">
                {t("vision.results.clearCell")}
              </button>
            </p>
            <Gallery datasetId={datasetId} rows={rows} items={cellItems} empty={t("vision.results.noImage")} onOpen={onOpen} label={t("vision.results.cellImages")} />
          </div>
        )}
      </Block>

      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-roc" title={t("vision.results.rocTitle")} conclusion={t("vision.results.rocSentence", { value: metric(report.roc_auc) })}>
          <CurveChart curves={curves.flatMap((c) => (c.roc ? [{ name: c.label, curve: c.roc }] : []))} kind="roc" label={t("vision.results.rocTitle")} />
        </Block>
        <Block id="vision-pr" title={t("vision.results.prTitle")} conclusion={t("vision.results.prSentence", { value: metric(report.pr_auc) })}>
          <CurveChart curves={curves.flatMap((c) => (c.pr ? [{ name: c.label, curve: c.pr }] : []))} kind="pr" label={t("vision.results.prTitle")} />
        </Block>
      </div>

      {reliability && (
        <div className="grid gap-x-10 gap-y-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
          <Block
            id="vision-reliability"
            title={t("vision.results.reliability")}
            conclusion={t("vision.results.reliabilitySentence", { ece: percent(reliability.ece), mce: percent(reliability.mce) })}
            aside={
              after ? (
                <SegmentedControl
                  label={t("vision.results.reliability")}
                  value={scaled}
                  onChange={setScaled}
                  options={[
                    { value: "trained", label: t("vision.results.asTrained") },
                    { value: "scaled", label: t("vision.results.afterScaling") },
                  ]}
                />
              ) : undefined
            }
          >
            <Plot figure={reliabilityFigure(reliability)} height={260} label={t("vision.results.reliability")} />
          </Block>
          <div className="flex flex-col justify-center gap-2 text-[13px] text-pretty text-ink-2">
            <p>{report.calibration?.reason}</p>
            {calibration?.applied && calibration.temperature != null ? (
              <p>
                {t("vision.results.temperatureApplied", {
                  before: metric(calibration.reliability.ece),
                  after: metric(calibration.ece_after ?? null),
                  lossBefore: metric(report.log_loss),
                  lossAfter: metric(calibration.log_loss_after ?? null),
                  t: calibration.temperature.toFixed(2),
                })}
              </p>
            ) : (
              calibration?.suggest_temperature &&
              calibration.temperature != null && <p>{t("vision.results.temperature", { t: calibration.temperature.toFixed(2), after: metric(calibration.ece_after ?? null) })}</p>
            )}
          </div>
        </div>
      )}

      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-errors" title={t("vision.results.confidentErrors")} conclusion={t("vision.results.confidentErrorsSentence")}>
          <Gallery datasetId={datasetId} rows={rows} items={errorItems(model.galleries.confident_errors ?? [])} empty={t("vision.results.noError")} onOpen={onOpen} label={t("vision.results.confidentErrors")} />
        </Block>
        <Block id="vision-uncertain" title={t("vision.results.uncertain")} conclusion={t("vision.results.uncertainSentence")}>
          <Gallery datasetId={datasetId} rows={rows} items={errorItems(model.galleries.uncertain ?? [])} empty={t("vision.results.noImage")} onOpen={onOpen} label={t("vision.results.uncertain")} />
        </Block>
      </div>

      {report.notes.length > 0 && (
        <ul className="flex flex-col gap-1 text-xs text-ink-3">
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** R-VM-03 / R-VM-06: what the ECE tile says under its value. */
function calibrationDetail(report: ClassificationReport): string {
  const calibration = report.calibration;
  if (calibration?.applied) return t("vision.results.calibratedApplied", { after: metric(calibration.ece_after ?? null) });
  return calibration?.suggest_temperature ? t("vision.results.calibrate") : t("vision.results.calibrated");
}

function CurveChart({ curves, kind, label }: { curves: { name: string; curve: NonNullable<ReturnType<typeof classCurves>[number]["roc"]> }[]; kind: "roc" | "pr"; label: string }) {
  const tokens = useChartTokens();
  if (curves.length === 0) return <p className="text-[13px] text-ink-3">{t("vision.results.noCurve")}</p>;
  return (
    <div>
      <Plot figure={curvesFigure(curves, kind)} height={280} label={label} />
      {curves.length > 1 && (
        <ul className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-2" aria-label={t("vision.results.legend")}>
          {curves.map(({ name, curve }, slot) => (
            <li key={name} className="inline-flex items-center gap-1.5 tabular-nums">
              <span aria-hidden className="inline-block h-0.5 w-4 rounded-full" style={{ backgroundColor: tokens.palette[slot % tokens.palette.length] }} />
              {name} <span className="text-ink-3">{curve.area.toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
