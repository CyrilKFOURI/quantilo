import { RotateCcw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { AnomalyView } from "./AnomalyView";
import type { ImageRow } from "./model";
import type { Prediction, VisionModel } from "./modelApi";
import { barsFigure, curvesFigure } from "./resultFigures";
import { anomalyRows, bestF1Threshold, countsAt, histogram, metric, percent, prCurve, rocCurve } from "./scores";
import { Block, Gallery, MetricTable, Tiles, type GalleryItem, type MetricRow, type Tile } from "./resultsUi";
import { ThresholdHistogram } from "./Threshold";

type Props = {
  datasetId: string;
  runId: string;
  rows: readonly ImageRow[];
  model: VisionModel;
  predictions: readonly Prediction[];
  /** R-113-11: a threshold moved by the user, kept in the URL; undefined = the rule's. */
  threshold: number | undefined;
  onThreshold: (value: number | undefined) => void;
  /** 117: open an image's "Why?" (this view closes first: one dialog at a time). */
  onWhy?: (row: number) => void;
};

const short = (value: number) => (Number.isFinite(value) ? (Math.abs(value) >= 100 ? value.toFixed(0) : value.toPrecision(3)) : "—");

/**
 * 113 (§ 7.2, R-VTH): every anomaly metric of the chosen model. The threshold of the rule is the
 * starting point; the user moves it on the histogram and the counts, F1 and galleries follow at
 * once, computed in the page on the run's scores (R-VTH-03, R-113-08).
 */
export function AnomalyResults({ datasetId, runId, rows, model, predictions, threshold: kept, onThreshold, onWhy }: Props) {
  // The line follows the pointer at once; the URL is written once the hand rests (250 ms).
  const [threshold, setThreshold] = useState<number | undefined>(kept);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);
  const move = (value: number | undefined) => {
    setThreshold(value);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => onThreshold(value), value === undefined ? 0 : 250);
  };
  const report = model.anomaly ?? null;
  const choice = report?.threshold ?? model.threshold ?? null;
  const data = useMemo(() => anomalyRows(predictions, model.key), [predictions, model.key]);
  const defects = useMemo(() => data.map((d) => d.defect), [data]);
  const scores = useMemo(() => data.map((d) => d.score), [data]);
  const evaluable = report !== null;
  const rule = choice?.threshold ?? (evaluable ? bestF1Threshold(defects, scores) : null);
  const current = threshold ?? rule ?? 0;
  const [open, setOpen] = useState<{ row: number; list: number[] } | null>(null);

  // R-VTH-01: the counts come from the half of the test the threshold was not chosen on.
  const evaluation = choice?.code === "R-VTH-01" && choice.evaluation.length > 0 ? choice.evaluation : null;
  const counted = useMemo(() => {
    const index = evaluation ?? data.map((_, i) => i);
    return { defects: index.map((i) => defects[i] === true), scores: index.map((i) => scores[i] ?? Number.NaN) };
  }, [evaluation, data, defects, scores]);
  const at = countsAt(counted.defects, counted.scores, current);
  const hist = useMemo(() => histogram(defects, scores, 32), [defects, scores]);
  const roc = useMemo(() => (evaluable ? rocCurve(defects, scores) : null), [evaluable, defects, scores]);
  const pr = useMemo(() => (evaluable ? prCurve(defects, scores) : null), [evaluable, defects, scores]);
  const nDefects = counted.defects.filter(Boolean).length;
  const nNormals = counted.defects.length - nDefects;
  const flaggedAll = data.filter((d) => d.score >= current).length;

  const byScore = [...data].sort((a, b) => b.score - a.score);
  const item = (d: (typeof data)[number]): GalleryItem => ({
    row: d.row,
    caption: d.truth,
    detail: t("vision.results.scoreShort", { value: short(d.score) }),
    tone: d.defect ? (d.score >= current ? undefined : ("bad" as const)) : d.score >= current ? ("warn" as const) : undefined,
  });
  const flagged = byScore.filter((d) => d.score >= current).map(item);
  const missed = [...byScore].reverse().filter((d) => d.defect && d.score < current).map(item);
  const alarms = byScore.filter((d) => !d.defect && d.score >= current).map(item);
  const openRow = (row: number, list: number[]) => setOpen({ row, list });

  const tiles: Tile[] = evaluable
    ? [
        {
          label: t("vision.results.m.auroc"),
          value: metric(report.auroc),
          detail: report.auroc_interval ? t("vision.results.ci", { range: `[${metric(report.auroc_interval.low)}; ${metric(report.auroc_interval.high)}]` }) : undefined,
          tip: t("vision.results.tip.auroc"),
          main: true,
        },
        { label: t("vision.results.m.ap"), value: metric(report.ap), tip: t("vision.results.tip.ap") },
        { label: t("vision.results.m.f1At"), value: metric(at.f1), detail: t("vision.results.atValue", { value: short(current) }), tip: t("vision.results.tip.f1At") },
        { label: t("vision.results.m.pixelAuroc"), value: metric(report.pixel?.auroc ?? null), detail: report.pixel ? undefined : t("vision.results.noPixel"), tip: t("vision.results.tip.pixelAuroc") },
        { label: t("vision.results.m.aupro"), value: metric(report.pixel?.aupro ?? null), detail: report.pixel ? t("vision.results.upTo", { limit: percent(report.pixel.limit, 0) }) : t("vision.results.noPixel"), tip: t("vision.results.tip.aupro") },
      ]
    : [
        { label: t("vision.results.m.flagged"), value: formatCount(flaggedAll), detail: t("vision.results.ofImages", { count: formatCount(data.length) }), main: true },
        { label: t("vision.results.m.threshold"), value: short(current), detail: choice?.code },
      ];

  const all: MetricRow[] = evaluable
    ? [
        { label: t("vision.results.m.auroc"), value: metric(report.auroc), detail: report.auroc_interval ? `[${metric(report.auroc_interval.low)}; ${metric(report.auroc_interval.high)}]` : undefined, tip: t("vision.results.tip.auroc") },
        { label: t("vision.results.m.ap"), value: metric(report.ap), tip: t("vision.results.tip.ap") },
        { label: t("vision.results.m.precision"), value: metric(at.precision), detail: t("vision.results.atValue", { value: short(current) }) },
        { label: t("vision.results.m.recall"), value: metric(at.recall), detail: t("vision.results.atValue", { value: short(current) }) },
        { label: t("vision.results.m.f1At"), value: metric(at.f1), detail: t("vision.results.atValue", { value: short(current) }) },
        { label: t("vision.results.m.falseAlarm"), value: percent(at.falseAlarm), detail: t("vision.results.atValue", { value: short(current) }), tip: t("vision.results.tip.falseAlarm") },
        ...(report.pixel
          ? [
              { label: t("vision.results.m.pixelAuroc"), value: metric(report.pixel.auroc), tip: t("vision.results.tip.pixelAuroc") },
              { label: t("vision.results.m.aupro"), value: metric(report.pixel.aupro), detail: t("vision.results.upTo", { limit: percent(report.pixel.limit, 0) }), tip: t("vision.results.tip.aupro") },
              { label: t("vision.results.m.pixelF1"), value: metric(report.pixel.f1 ?? null), detail: report.pixel.threshold !== null && report.pixel.threshold !== undefined ? t("vision.results.atValue", { value: short(report.pixel.threshold) }) : undefined },
              { label: t("vision.results.m.pixelIou"), value: metric(report.pixel.iou ?? null) },
              { label: t("vision.results.m.regions"), value: formatCount(report.pixel.regions), tip: t("vision.results.tip.regions") },
            ]
          : []),
        { label: t("vision.results.m.normals"), value: formatCount(report.n_normal) },
        { label: t("vision.results.m.defects"), value: formatCount(report.n_defect) },
      ]
    : [];

  const byType = Object.entries(report?.auroc_by_type ?? {}).sort((a, b) => a[1] - b[1]);
  const sentence = evaluable
    ? t("vision.results.thresholdSentence", {
        value: short(current),
        found: formatCount(at.tp),
        defects: formatCount(nDefects),
        alarms: formatCount(at.fp),
        normals: formatCount(nNormals),
        f1: metric(at.f1),
      })
    : t("vision.results.thresholdSentenceNoDefect", { value: short(current), flagged: formatCount(flaggedAll), count: formatCount(data.length) });

  const openPrediction = open ? predictions.find((p) => p.row === open.row) : undefined;

  return (
    <div className="flex flex-col gap-8">
      <Tiles tiles={tiles} label={t("vision.results.mainMetrics")} />

      <Block
        id="vision-threshold"
        title={t("vision.results.thresholdTitle")}
        conclusion={sentence}
        aside={
          threshold !== undefined && rule !== null ? (
            <Button variant="ghost" icon={RotateCcw} onClick={() => move(undefined)} className="h-7 text-xs">
              {t("vision.results.backToRule", { value: short(rule) })}
            </Button>
          ) : undefined
        }
      >
        <div className="grid gap-x-8 gap-y-4 lg:grid-cols-[minmax(0,1fr)_16rem]">
          <div className="min-w-0">
            <ThresholdHistogram histogram={hist} threshold={current} rule={rule} onChange={(value) => move(value)} single={!evaluable} label={t("vision.results.thresholdChart")} />
            <p className="mt-1 text-xs text-pretty text-ink-3">
              {choice ? (choice.message.includes(choice.code) ? choice.message : `${choice.message} (${choice.code})`) : ""}
            </p>
          </div>
          {evaluable && (
            <div className="flex flex-col gap-3">
              <table className="w-full border-collapse text-[13px] tabular-nums" aria-label={t("vision.results.counts2x2")}>
                <thead>
                  <tr className="text-xs text-ink-3">
                    <th scope="col" className="py-1 text-left font-normal" />
                    <th scope="col" className="py-1 text-right font-normal">{t("vision.results.flaggedCol")}</th>
                    <th scope="col" className="py-1 text-right font-normal">{t("vision.results.notFlaggedCol")}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line border-y border-line">
                  <tr>
                    <th scope="row" className="py-1.5 text-left font-normal text-ink-2">{t("vision.results.defectsRow")}</th>
                    <td className="py-1.5 text-right font-medium text-ink-1">{formatCount(at.tp)}</td>
                    <td className={`py-1.5 text-right ${at.fn > 0 ? "font-medium text-bad" : "text-ink-2"}`}>{formatCount(at.fn)}</td>
                  </tr>
                  <tr>
                    <th scope="row" className="py-1.5 text-left font-normal text-ink-2">{t("vision.results.normalsRow")}</th>
                    <td className={`py-1.5 text-right ${at.fp > 0 ? "font-medium text-warn" : "text-ink-2"}`}>{formatCount(at.fp)}</td>
                    <td className="py-1.5 text-right text-ink-2">{formatCount(at.tn)}</td>
                  </tr>
                </tbody>
              </table>
              <dl className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[13px] tabular-nums">
                <dt className="text-ink-3">{t("vision.results.precision")}</dt>
                <dd className="text-right text-ink-1">{metric(at.precision)}</dd>
                <dt className="text-ink-3">{t("vision.results.recall")}</dt>
                <dd className="text-right text-ink-1">{metric(at.recall)}</dd>
                <dt className="text-ink-3">F1</dt>
                <dd className="text-right font-medium text-ink-1">{metric(at.f1)}</dd>
                <dt className="text-ink-3">{t("vision.results.m.falseAlarm")}</dt>
                <dd className="text-right text-ink-1">{percent(at.falseAlarm)}</dd>
              </dl>
            </div>
          )}
        </div>
      </Block>

      {evaluable && (
        <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
          <Block id="vision-anomaly-metrics" title={t("vision.results.allMetrics")}>
            <MetricTable rows={all} caption={t("vision.results.allMetrics")} />
          </Block>
          {byType.length > 0 && (
            <Block
              id="vision-by-type"
              title={t("vision.results.byType")}
              conclusion={
                metric(byType[0]?.[1] ?? null) === metric(byType[byType.length - 1]?.[1] ?? null)
                  ? t("vision.results.byTypeSame", { value: metric(byType[0]?.[1] ?? null) })
                  : t("vision.results.byTypeSentence", { type: byType[0]?.[0] ?? "", value: metric(byType[0]?.[1] ?? null) })
              }
            >
              <Plot figure={barsFigure(byType.map(([name, value]) => ({ name, value })).reverse(), { range: [0, 1.08] })} height={Math.max(120, 36 * byType.length + 40)} label={t("vision.results.byType")} />
            </Block>
          )}
        </div>
      )}

      {evaluable && (roc || pr) && (
        <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
          {roc && (
            <Block id="vision-anomaly-roc" title={t("vision.results.rocTitle")} conclusion={t("vision.results.rocSentence", { value: metric(report.auroc) })}>
              <Plot figure={curvesFigure([{ name: model.title, curve: roc }], "roc")} height={260} label={t("vision.results.rocTitle")} />
            </Block>
          )}
          {pr && (
            <Block id="vision-anomaly-pr" title={t("vision.results.prTitle")} conclusion={t("vision.results.prSentence", { value: metric(report.ap) })}>
              <Plot figure={curvesFigure([{ name: model.title, curve: pr }], "pr", nDefects / Math.max(1, counted.defects.length))} height={260} label={t("vision.results.prTitle")} />
            </Block>
          )}
        </div>
      )}

      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-flagged" title={t("vision.results.flaggedTitle", { count: formatCount(flagged.length) })} conclusion={t("vision.results.flaggedSentence")}>
          <Gallery datasetId={datasetId} rows={rows} items={flagged} empty={t("vision.results.noFlagged")} onOpen={openRow} label={t("vision.results.flaggedTitle", { count: formatCount(flagged.length) })} />
        </Block>
        {evaluable ? (
          <div className="flex flex-col gap-8">
            <Block id="vision-missed" title={t("vision.results.missedTitle", { count: formatCount(missed.length) })} conclusion={t("vision.results.missedSentence")}>
              <Gallery datasetId={datasetId} rows={rows} items={missed} empty={t("vision.results.noMissed")} onOpen={openRow} label={t("vision.results.missedTitle", { count: formatCount(missed.length) })} />
            </Block>
            <Block id="vision-alarms" title={t("vision.results.alarmsTitle", { count: formatCount(alarms.length) })} conclusion={t("vision.results.alarmsSentence")}>
              <Gallery datasetId={datasetId} rows={rows} items={alarms} empty={t("vision.results.noAlarm")} onOpen={openRow} label={t("vision.results.alarmsTitle", { count: formatCount(alarms.length) })} />
            </Block>
          </div>
        ) : (
          <Block id="vision-unusual" title={t("vision.results.unusualTitle")} conclusion={t("vision.results.unusualSentence")}>
            <Gallery datasetId={datasetId} rows={rows} items={byScore.map(item)} empty={t("vision.results.noImage")} onOpen={openRow} label={t("vision.results.unusualTitle")} />
          </Block>
        )}
      </div>

      {report && report.notes.length > 0 && (
        <ul className="flex flex-col gap-1 text-xs text-ink-3">
          {report.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}

      {open && openPrediction && (
        <AnomalyView
          datasetId={datasetId}
          runId={runId}
          model={model}
          rows={rows}
          prediction={openPrediction}
          threshold={current}
          list={open.list}
          onNavigate={(row) => setOpen({ row, list: open.list })}
          onClose={() => setOpen(null)}
          onWhy={
            onWhy &&
            ((row) => {
              setOpen(null);
              onWhy(row);
            })
          }
        />
      )}
    </div>
  );
}
