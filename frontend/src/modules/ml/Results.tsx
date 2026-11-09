import { useMemo, useState } from "react";

import { api, type MlRun } from "../../api/client";
import { Plot } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { pLabel } from "../../features/variable/normality";
import { t, type MessageKey } from "../../lib/i18n";
import { Checkbox } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { ErrorsTab } from "./Errors";
import { ExplainTab, RowExplanation } from "./Explain";
import { periodVerdict, PeriodsTab } from "./PanelResults";
import { Panel, ScoreBand } from "./ResultParts";
import {
  coefficientsFigure,
  confusionFigure,
  classCurvesFigure,
  curveFigure,
  predictedFigure,
  residualHistogram,
  residualsFigure,
  importancesFigure,
  learningCurveFigure,
} from "./figures";

type Tab = "performance" | "periods" | "features" | "explain" | "errors" | "predictions";
const TABS = ["performance", "periods", "features", "explain", "errors", "predictions"] as const;
const pct = (value: number | null | undefined, digits = 0) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)} %`;

/** The result in one sentence, before any number. */
export function verdict(run: MlRun): string {
  // a panel is judged period by period (035), not on all its rows at once
  if (run.panel) return periodVerdict(run.panel);
  if (run.regression) {
    const r = run.regression;
    // no real gain: under 1 % better than the average, or an R² under 1 %
    if (r.skill === null || r.skill < 0.01 || (r.r2 !== null && r.r2 < 0.01)) {
      const overfit = run.settings.regularization === "none" && run.preparation.inputs * 10 > run.train_rows;
      const text = t("ml.verdict.noBetterAverage", { target: run.target });
      return overfit ? `${text} ${t("ml.verdict.overfit", { inputs: String(run.preparation.inputs), rows: String(run.train_rows) })}` : text;
    }
    return t("ml.verdict.regression", { r2: pct(r.r2), target: run.target, skill: pct(r.skill) });
  }
  const c = run.classification;
  if (!c) return "";
  const chance = c.roc_auc !== null && c.roc_auc < 0.6;
  const noGain =
    c.accuracy !== null &&
    c.baseline_accuracy !== null &&
    c.accuracy <= c.baseline_accuracy &&
    (c.balanced_accuracy ?? 0) <= 1 / c.labels.length + 0.05;
  if (chance || noGain) {
    const text = t("ml.verdict.noBetterMajority", { majority: c.majority });
    return c.roc_auc !== null ? `${text} ${t("ml.verdict.auc", { auc: c.roc_auc.toFixed(2) })}` : text;
  }
  const text = t("ml.verdict.classification", { accuracy: pct(c.accuracy), majority: c.majority, baseline: pct(c.baseline_accuracy) });
  return c.within_one !== null ? `${text} ${t("ml.verdict.withinOne", { share: pct(c.within_one) })}` : text;
}

export function Results({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const [tab, setTab] = useState<Tab>("performance");
  return (
    <div className="@container space-y-4">
      <p className="text-[17px] leading-snug font-medium text-ink-1">{verdict(run)}</p>
      <ScoreTiles run={run} />
      <AllMetrics run={run} />
      <p className="text-xs text-ink-3">
        {t("ml.rowsUsed", {
          train: run.train_rows.toLocaleString("en-US"),
          test: run.test_rows.toLocaleString("en-US"),
        })}
        {run.sampled ? ` · ${t("ml.sampled", { rows: run.rows.toLocaleString("en-US") })}` : ""}
        {` · ${run.seconds.toFixed(1)} s`}
      </p>
      <SegmentedControl
        label={t("ml.tabs")}
        value={tab}
        options={(run.panel ? TABS : TABS.filter((value) => value !== "periods")).map((value) => ({ value, label: t(`ml.tab.${value}`) }))}
        onChange={setTab}
      />
      {tab === "performance" &&
        (run.regression ? (
          <RegressionPerformance run={run} />
        ) : (
          <div className="space-y-4">
            {run.classification?.threshold != null && run.classification.threshold !== 0.5 && run.validation.threshold !== "default" && (
              <p className="text-xs text-ink-2">
                {t("ml.thresholdChosen", {
                  threshold: run.classification.threshold.toFixed(2),
                  rule: t(`ml.validation.threshold.${run.validation.threshold}` as MessageKey),
                })}
              </p>
            )}
            <ClassificationPerformance run={run} />
          </div>
        ))}
      {tab === "periods" && <PeriodsTab run={run} datasetId={datasetId} />}
      {tab === "features" && <Features run={run} />}
      {tab === "explain" && <ExplainTab run={run} datasetId={datasetId} />}
      {tab === "errors" && <ErrorsTab run={run} datasetId={datasetId} />}
      {tab === "predictions" && <Predictions run={run} datasetId={datasetId} />}
    </div>
  );
}

function ScoreTiles({ run }: { run: MlRun }) {
  if (run.regression) {
    const r = run.regression;
    return (
      <ScoreBand
        items={[
          { label: "R²", value: formatNumber(r.r2), explain: t("ml.explain.r2") },
          { label: "MAE", value: formatNumber(r.mae), note: `avg ${formatNumber(r.baseline_mae)}`, explain: t("ml.explain.mae") },
          { label: "RMSE", value: formatNumber(r.rmse), note: `avg ${formatNumber(r.baseline_rmse)}`, explain: t("ml.explain.rmse") },
          { label: t("ml.vsAverage"), value: pct(r.skill), explain: t("ml.explain.skill") },
          ...(r.mape !== null ? [{ label: "MAPE", value: `${r.mape.toFixed(1)} %` }] : []),
        ]}
      />
    );
  }
  const c = run.classification;
  if (!c) return null;
  return (
    <ScoreBand
      items={[
        { label: t("ml.accuracy"), value: pct(c.accuracy, 1), note: `${t("ml.majority")} ${pct(c.baseline_accuracy)}`, explain: t("ml.explain.accuracy") },
        { label: t("ml.balancedAccuracy"), value: pct(c.balanced_accuracy, 1), explain: t("ml.explain.balanced") },
        ...(c.within_one !== null
          ? [
              {
                label: t("ml.withinOne"),
                value: pct(c.within_one, 1),
                note: c.mean_class_error !== null ? t("ml.meanClassError", { error: c.mean_class_error.toFixed(2) }) : undefined,
                explain: t("ml.explain.withinOne"),
              },
            ]
          : []),
        { label: "F1 (macro)", value: formatNumber(c.f1_macro), explain: t("ml.explain.f1") },
        ...(c.roc_auc !== null ? [{ label: "ROC AUC", value: formatNumber(c.roc_auc), explain: t("ml.explain.auc") }] : []),
        ...(c.log_loss !== null ? [{ label: "Log loss", value: formatNumber(c.log_loss), explain: t("ml.explain.logloss") }] : []),
      ]}
    />
  );
}

const HEIGHT = 240;

function LearningCurvePanel({ run }: { run: MlRun }) {
  const curve = run.learning_curve;
  if (!curve) return null;
  return (
    <Panel title={t("ml.learningCurve", { label: curve.label })}>
      <Plot figure={learningCurveFigure(curve)} height={HEIGHT} label={t("ml.learningCurve", { label: curve.label })} />
    </Panel>
  );
}

function RegressionPerformance({ run }: { run: MlRun }) {
  const r = run.regression;
  if (!r) return null;
  return (
    <div className="space-y-3">
      <div className="grid gap-x-5 gap-y-4 @2xl:grid-cols-2 @6xl:grid-cols-3">
        <Panel title={t("ml.predictedVsActual")} wide>
          <Plot figure={predictedFigure(r, run.target)} height={HEIGHT} label={t("ml.predictedVsActual")} />
        </Panel>
        <Panel title={t("ml.residualDistribution")}>
          <Plot figure={residualHistogram(r)} height={HEIGHT} label={t("ml.residualDistribution")} />
        </Panel>
        <Panel title={t("ml.residuals")}>
          <Plot figure={residualsFigure(r)} height={HEIGHT} label={t("ml.residuals")} />
        </Panel>
        <LearningCurvePanel run={run} />
      </div>
      {r.heteroscedasticity_p !== null && (
        <p className={`text-[13px] ${r.heteroscedasticity_p < 0.05 ? "text-warn" : "text-ink-2"}`}>
          {t(r.heteroscedasticity_p < 0.05 ? "ml.heteroscedastic" : "ml.homoscedastic", { p: pLabel(r.heteroscedasticity_p) })}
        </p>
      )}
    </div>
  );
}

/** Confusion matrix and scores of a binary classifier at a chosen probability threshold. */
function atThreshold(truth: number[], probability: number[], threshold: number) {
  let tp = 0;
  let fp = 0;
  let tn = 0;
  let fn = 0;
  truth.forEach((actual, i) => {
    const positive = (probability[i] ?? 0) >= threshold;
    if (actual === 1 && positive) tp++;
    else if (actual === 1) fn++;
    else if (positive) fp++;
    else tn++;
  });
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = tp + fn ? tp / (tp + fn) : 0;
  return { tp, fp, tn, fn, precision, recall, accuracy: (tp + tn) / (truth.length || 1) };
}

function ClassificationPerformance({ run }: { run: MlRun }) {
  const c = run.classification;
  const [percent, setPercent] = useState(false);
  const [threshold, setThreshold] = useState(c?.threshold ?? 0.5);
  const binary = c?.positive ? c : null;
  const scored = useMemo(
    () => (binary ? atThreshold(binary.truth, binary.probability, threshold) : null),
    [binary, threshold],
  );
  if (!c) return null;
  const negative = binary ? (c.labels.find((l) => l !== binary.positive) ?? "") : "";
  const labels = binary && binary.positive ? [negative, binary.positive] : c.labels;
  const matrix = scored ? [[scored.tn, scored.fp], [scored.fn, scored.tp]] : c.confusion;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-4 text-xs">
        <Checkbox label={t("ml.percentByRow")} checked={percent} onChange={(event) => setPercent(event.target.checked)} />
        {binary && scored && (
          <label className="flex items-center gap-2 text-ink-2">
            {t("ml.threshold", { positive: binary.positive ?? "" })}
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.05}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
              aria-label={t("ml.thresholdLabel")}
              className="accent-(--accent)"
            />
            <span className="tabular-nums text-ink-1">{threshold.toFixed(2)}</span>
            <span className="text-ink-3">
              {t("ml.atThreshold", { precision: pct(scored.precision), recall: pct(scored.recall), accuracy: pct(scored.accuracy) })}
            </span>
          </label>
        )}
      </div>
      <div className="grid gap-x-5 gap-y-4 @2xl:grid-cols-2 @6xl:grid-cols-3">
        <Panel title={t("ml.confusion")}>
          <Plot figure={confusionFigure(labels, matrix, percent)} height={HEIGHT} label={t("ml.confusion")} />
        </Panel>
        <Panel title={t("ml.perClass")}>
          <div className="max-h-[240px] overflow-auto">
          <table className="w-full border-collapse text-[13px] tabular-nums">
            <thead className="text-left text-xs text-ink-3">
              <tr>
                <th className="px-3 py-1.5 font-normal">{t("ml.class")}</th>
                <th className="px-3 py-1.5 text-right font-normal">{t("ml.precision")}</th>
                <th className="px-3 py-1.5 text-right font-normal">{t("ml.recall")}</th>
                <th className="px-3 py-1.5 text-right font-normal">F1</th>
                <th className="px-3 py-1.5 text-right font-normal">{t("ml.support")}</th>
              </tr>
            </thead>
            <tbody>
              {c.report.map((row) => (
                <tr key={row.label} className="border-t border-line">
                  <td className="px-3 py-1.5 text-ink-1">{row.label}</td>
                  <td className="px-3 py-1.5 text-right">{pct(row.precision)}</td>
                  <td className="px-3 py-1.5 text-right">{pct(row.recall)}</td>
                  <td className="px-3 py-1.5 text-right">{formatNumber(row.f1)}</td>
                  <td className="px-3 py-1.5 text-right text-ink-2">{row.support.toLocaleString("en-US")}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        </Panel>
      {c.roc && c.pr && c.calibration && (
        <>
          <Panel title={t("ml.roc")}>
            <Plot figure={curveFigure(c.roc.x, c.roc.y, "False positive rate", "True positive rate", "diagonal")} height={HEIGHT} label={t("ml.roc")} />
          </Panel>
          <Panel title={t("ml.pr")}>
            <Plot
              figure={curveFigure(c.pr.x, c.pr.y, "Recall", "Precision", "flat", binary ? binary.truth.filter((v) => v === 1).length / (binary.truth.length || 1) : 0)}
              height={HEIGHT}
              label={t("ml.pr")}
            />
          </Panel>
          <Panel title={t("ml.calibration")}>
            <Plot figure={curveFigure(c.calibration.x, c.calibration.y, "Predicted probability", "Observed share", "diagonal")} height={HEIGHT} label={t("ml.calibration")} />
          </Panel>
        </>
      )}
      {c.class_curves.length > 0 && (
        <>
            <Panel title={`${t("ml.roc")} · ${t("ml.classCurves")}`}>
              <Plot
                figure={classCurvesFigure(
                  c.class_curves.map((k) => ({ name: k.auc === null ? k.label : `${k.label} · AUC ${k.auc.toFixed(2)}`, x: k.roc.x, y: k.roc.y })),
                  "False positive rate",
                  "True positive rate",
                  true,
                )}
                height={HEIGHT}
                label={t("ml.roc")}
              />
            </Panel>
            <Panel title={`${t("ml.pr")} · ${t("ml.classCurves")}`}>
              <Plot
                figure={classCurvesFigure(
                  c.class_curves.map((k) => ({ name: `${k.label} · ${t("ml.classShare", { share: (k.share * 100).toFixed(0) })}`, x: k.pr.x, y: k.pr.y })),
                  "Recall",
                  "Precision",
                  false,
                )}
                height={HEIGHT}
                label={t("ml.pr")}
              />
            </Panel>
            {c.class_curves.some((k) => k.calibration) && (
              <Panel title={`${t("ml.calibration")} · ${t("ml.classCurves")}`}>
                <Plot
                  figure={classCurvesFigure(
                    c.class_curves.flatMap((k) => (k.calibration ? [{ name: k.label, x: k.calibration.x, y: k.calibration.y }] : [])),
                    "Predicted probability",
                    "Observed share",
                    true,
                  )}
                  height={HEIGHT}
                  label={t("ml.calibration")}
                />
              </Panel>
            )}
        </>
      )}
        <LearningCurvePanel run={run} />
      </div>
    </div>
  );
}

function AllMetrics({ run }: { run: MlRun }) {
  const cv = run.cross_validation;
  const folds = cv && Object.keys(cv.mean).length > 0 ? cv : null;
  const keys = Object.keys(run.scores).filter((k) => run.scores[k] !== null || folds?.mean[k] != null);
  if (!keys.length) return null;
  const show = (value: number | null | undefined) => (value === null || value === undefined ? "—" : formatNumber(value));
  return (
    <details className="text-[13px]">
      <summary className="cursor-pointer text-xs text-ink-2">
        {t("ml.allMetrics")}
        {cv ? ` · ${t("ml.crossValidated", { method: t(`ml.validation.method.${cv.method}` as MessageKey) })}` : ""}
      </summary>
      <table className="mt-2 w-full max-w-xl border-collapse tabular-nums">
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="py-1 font-normal">{t("ml.metricName")}</th>
            <th className="py-1 text-right font-normal">{t("ml.metricValue")}</th>
            {folds && <th className="py-1 text-right font-normal">{t("ml.foldMean")}</th>}
          </tr>
        </thead>
        <tbody>
          {keys.map((key) => (
            <tr key={key} className={`border-t border-line ${key === run.primary ? "font-medium text-ink-1" : "text-ink-2"}`}>
              <td className="py-1">
                {t(`ml.metric.${key}` as MessageKey)}
                {key === run.primary ? " ★" : ""}
              </td>
              <td className="py-1 text-right">{show(run.scores[key])}</td>
              {folds && (
                <td className="py-1 text-right">
                  {show(folds.mean[key])}
                  {folds.std[key] != null ? ` ± ${formatNumber(folds.std[key] as number)}` : ""}
                </td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

function Importances({ run }: { run: MlRun }) {
  return (
    <div className="space-y-3">
      <p className="text-[13px] text-ink-2">{t("ml.importancesHint")}</p>
      <div className="grid gap-5 @4xl:grid-cols-2">
      <Panel title={t("ml.importances")}>
        <Plot figure={importancesFigure(run.importances)} height={Math.max(220, 24 * Math.min(15, run.importances.length) + 60)} label={t("ml.importances")} />
      </Panel>
      <div className="max-h-96 overflow-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{t("ml.feature")}</th>
              <th className="px-3 py-1.5 text-right font-normal">{t("ml.importance")}</th>
            </tr>
          </thead>
          <tbody>
            {run.importances.map((i) => (
              <tr key={i.feature} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs text-ink-1">{i.feature}</td>
                <td className="px-3 py-1.5 text-right">{(i.importance * 100).toFixed(1)} %</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </div>
    </div>
  );
}

function Features({ run }: { run: MlRun }) {
  const coefficients = run.coefficients;
  if (!coefficients.length && run.importances.length) return <Importances run={run} />;
  if (!coefficients.length) return <p className="text-ink-2">{t("ml.noCoefficients")}</p>;
  const withStats = coefficients.some((c) => c.p_value !== null);
  const odds = coefficients.some((c) => c.odds_ratio !== null);
  return (
    <div className="space-y-3">
      <p className="text-[13px] text-ink-2">{t(run.regression ? "ml.coefficientsHint" : "ml.coefficientsLogisticHint")}</p>
      <div className="grid gap-5 @4xl:grid-cols-2">
      <Panel title={t("ml.topCoefficients")}>
        <Plot figure={coefficientsFigure(coefficients)} height={Math.max(220, 24 * Math.min(15, coefficients.length) + 60)} label={t("ml.topCoefficients")} />
      </Panel>
      <div className="max-h-96 overflow-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{t("ml.feature")}</th>
              <th className="px-3 py-1.5 text-right font-normal">{t("ml.coefficient")}</th>
              {withStats && (
                <>
                  <th className="px-3 py-1.5 text-right font-normal">{t("ml.ci")}</th>
                  <th className="px-3 py-1.5 text-right font-normal">p</th>
                  <th className="px-3 py-1.5 text-right font-normal" title={t("ml.vifHint")}>VIF</th>
                </>
              )}
              <th className="px-3 py-1.5 text-right font-normal">{odds ? t("ml.oddsRatio") : t("ml.perUnit")}</th>
            </tr>
          </thead>
          <tbody>
            {coefficients.map((c) => (
              <tr key={c.feature} className="border-t border-line">
                <td className="px-3 py-1.5 font-mono text-xs text-ink-1">{c.feature}</td>
                <td className="px-3 py-1.5 text-right">{formatNumber(c.estimate)}</td>
                {withStats && (
                  <>
                    <td className="px-3 py-1.5 text-right text-ink-2">
                      {c.low === null ? "—" : `${formatNumber(c.low)} – ${formatNumber(c.high)}`}
                    </td>
                    <td className={`px-3 py-1.5 text-right ${c.p_value !== null && c.p_value < 0.05 ? "text-ink-1" : "text-ink-3"}`}>
                      {c.p_value === null ? "—" : pLabel(c.p_value)}
                    </td>
                    <td className={`px-3 py-1.5 text-right ${c.vif !== null && c.vif > 10 ? "text-warn" : "text-ink-2"}`}>
                      {formatNumber(c.vif)}
                    </td>
                  </>
                )}
                <td className="px-3 py-1.5 text-right text-ink-2">{formatNumber(odds ? c.odds_ratio : c.per_unit)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      </div>
    </div>
  );
}

function Predictions({ run, datasetId }: { run: MlRun; datasetId: string }) {
  const regression = Boolean(run.regression);
  const [row, setRow] = useState<number | null>(null);
  return (
    <div className="space-y-2">
      {row !== null && <RowExplanation datasetId={datasetId} run={run} row={row} />}
      <div className="flex items-center justify-between text-xs text-ink-3">
        <span>{t("ml.predictionsHint", { rows: run.predictions.length.toLocaleString("en-US") })}</span>
        <a href={api.mlPredictionsUrl(datasetId, run.id)} className="text-accent hover:underline">
          {t("ml.exportPredictions")}
        </a>
      </div>
      <div className="max-h-96 overflow-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{t("ml.row")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ml.actual")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ml.predicted")}</th>
              <th className="px-3 py-1.5 text-right font-normal">{regression ? t("ml.error") : t("ml.probability")}</th>
            </tr>
          </thead>
          <tbody>
            {run.predictions.map((p) => (
              <tr
                key={p.row}
                onClick={() => setRow(p.row)}
                aria-selected={row === p.row}
                className={`cursor-pointer border-t border-line hover:bg-surface-2 ${row === p.row ? "bg-accent-soft" : ""} ${p.correct === false ? "text-warn" : ""}`}
              >
                <td className="px-3 py-1.5 text-ink-3">
                  <button type="button" className="underline-offset-2 hover:underline" aria-label={t("ml.explain.explainRow", { row: String(p.row + 1) })}>
                    {p.row + 1}
                  </button>
                </td>
                <td className="px-3 py-1.5">{p.actual}</td>
                <td className="px-3 py-1.5">{p.predicted}</td>
                <td className="px-3 py-1.5 text-right">{regression ? formatNumber(p.error) : pct(p.probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export type { MessageKey };
