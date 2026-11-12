import { useState } from "react";

import type { MlChoicesInput, MlClasses, MlModel, MlSearchParam, MlValidation } from "../../api/client";
import { useMlStopTuning, useMlTune, useMlTuning } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { importanceBars, trialsFigure } from "./figures";
import { paretoFigure } from "./tuneFigures";
import { CriterionSelect, metricName, PANEL } from "./SettingsDrawer";

type Budget = "60" | "300" | "900" | "trials";
const SAMPLERS = ["tpe", "random", "grid", "cmaes", "nsga2"] as const;
const PRUNERS = ["median", "halving", "hyperband", "none"] as const;
/** R-ML-TUNE-01: these samplers optimise a single objective. */
const SINGLE_OBJECTIVE = new Set<string>(["grid", "cmaes"]);

/** The model's search space as the API serves it, with nothing fixed yet. */
function initialSpace(model: MlModel): MlSearchParam[] {
  return model.space.map((p) => ({ key: p.key, kind: p.kind, low: p.low, high: p.high, log: p.log, choices: [...p.choices], fixed: null }));
}

export function TunePanel({
  open,
  onOpenChange,
  datasetId,
  target,
  model,
  settings,
  defaults,
  choices,
  classes,
  validation,
  onUse,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  target: string;
  model: MlModel;
  settings: Record<string, unknown>;
  defaults: Record<string, unknown>;
  choices: MlChoicesInput;
  classes: MlClasses | null;
  validation: MlValidation;
  onUse: (settings: Record<string, unknown>) => void;
}) {
  const [budget, setBudget] = useState<Budget>("60");
  const [trials, setTrials] = useState(30);
  const [expert, setExpert] = useState(false);
  const [sampler, setSampler] = useState<(typeof SAMPLERS)[number]>("tpe");
  const [pruner, setPruner] = useState<(typeof PRUNERS)[number]>("median");
  const [warmStart, setWarmStart] = useState(false);
  const [objectives, setObjectives] = useState<"score" | "score_time">("score");
  const [space, setSpace] = useState<MlSearchParam[]>(() => initialSpace(model));
  const [jobId, setJobId] = useState<string | null>(null);
  // on a panel, every trial is judged on the criterion over the test periods
  const [judged, setJudged] = useState<MlValidation>(validation);
  const start = useMlTune(datasetId);
  const job = useMlTuning(jobId);
  const stop = useMlStopTuning();
  const labels = Object.fromEntries(model.settings.map((s) => [s.key, s.label]));
  const progress = job.data?.progress ?? null;
  const running = start.isPending || (jobId !== null && (!job.data || job.data.status === "running"));

  const patch = (index: number, change: Partial<MlSearchParam>) => setSpace(space.map((p, i) => (i === index ? { ...p, ...change } : p)));
  const launch = () => {
    // the settings outside the search stay as the user set them
    const searched = new Set(space.map((p) => p.key));
    const fixed = Object.fromEntries(Object.entries(settings).filter(([key]) => !searched.has(key)));
    start.mutate(
      {
        target,
        classes,
        validation: judged,
        body: {
          model: model.key,
          budget_seconds: budget === "trials" ? null : Number(budget),
          trials: budget === "trials" ? trials : null,
          sampler,
          pruner,
          space,
          warm_start: expert && warmStart,
          objectives: expert ? objectives : "score",
          settings: fixed,
          preparation: choices,
        },
      },
      { onSuccess: (created) => setJobId(created.id) },
    );
  };
  const gain =
    progress && progress.best_value !== null && progress.default_value !== null
      ? (progress.greater_is_better ? 1 : -1) * (progress.best_value - progress.default_value)
      : null;

  return (
    <Dialog open={open} onOpenChange={onOpenChange} size="lg" title={t("ml.tune.title", { model: model.label })} description={t("ml.tune.description")}>
      {!jobId ? (
        <div className="space-y-4 text-[13px]">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs text-ink-2">{t("ml.tune.budget")}</span>
            <SegmentedControl
              label={t("ml.tune.budget")}
              value={budget}
              onChange={setBudget}
              options={[
                { value: "60", label: "1 min" },
                { value: "300", label: "5 min" },
                { value: "900", label: "15 min" },
                { value: "trials", label: t("ml.tune.trials") },
              ]}
            />
            {budget === "trials" && (
              <TextInput
                aria-label={t("ml.tune.trialCount")}
                type="number"
                min={2}
                max={2000}
                value={trials}
                onChange={(e) => setTrials(Math.max(2, Math.min(2000, Number(e.target.value) || 2)))}
                className="h-7 w-20 text-xs"
              />
            )}
            <Checkbox className="ml-auto text-xs text-ink-2" label={t("ml.settings.expert")} checked={expert} onChange={(e) => setExpert(e.target.checked)} />
          </div>
          {judged.method === PANEL && (
            <label className="flex items-center gap-2 text-xs text-ink-2">
              {t("ml.validation.criterion")}
              <CriterionSelect validation={judged} onValidation={setJudged} />
              <span className="text-ink-3">{t("ml.validation.criterionHint")}</span>
            </label>
          )}
          {expert && (
            <div className="flex flex-wrap gap-4">
              <label className="flex items-center gap-2 text-xs text-ink-2">
                {t("ml.tune.sampler")}
                <Select aria-label={t("ml.tune.sampler")} value={sampler} onChange={(e) => setSampler(e.target.value as typeof sampler)} className="h-7 text-xs">
                  {SAMPLERS.map((s) => (
                    <option key={s} value={s} disabled={objectives === "score_time" && SINGLE_OBJECTIVE.has(s)}>
                      {t(`ml.tune.sampler.${s}` as MessageKey)}
                    </option>
                  ))}
                </Select>
              </label>
              <label className="flex items-center gap-2 text-xs text-ink-2">
                {t("ml.tune.pruner")}
                <Select
                  aria-label={t("ml.tune.pruner")}
                  value={objectives === "score_time" ? "none" : pruner}
                  disabled={objectives === "score_time"}
                  onChange={(e) => setPruner(e.target.value as typeof pruner)}
                  className="h-7 text-xs"
                >
                  {PRUNERS.map((p) => (
                    <option key={p} value={p}>
                      {t(`ml.tune.pruner.${p}` as MessageKey)}
                    </option>
                  ))}
                </Select>
              </label>
              <details className="w-full text-xs" open={warmStart || objectives !== "score"}>
                <summary className="cursor-pointer text-ink-2">{t("ml.tune.more")}</summary>
                <div className="mt-2 space-y-2">
                  <Checkbox className="text-xs text-ink-2" label={t("ml.tune.warmStart")} checked={warmStart} onChange={(e) => setWarmStart(e.target.checked)} />
                  <p className="text-[11px] text-ink-3">{t("ml.tune.warmStartHint")}</p>
                  <label className="flex items-center gap-2 text-xs text-ink-2">
                    {t("ml.tune.objectives")}
                    <Select
                      aria-label={t("ml.tune.objectives")}
                      value={objectives}
                      onChange={(e) => {
                        const next = e.target.value as typeof objectives;
                        setObjectives(next);
                        // R-ML-TUNE-01: two objectives need a sampler that handles them
                        if (next === "score_time" && SINGLE_OBJECTIVE.has(sampler)) setSampler("nsga2");
                      }}
                      className="h-7 text-xs"
                    >
                      <option value="score">{t("ml.tune.objectives.score")}</option>
                      <option value="score_time">{t("ml.tune.objectives.score_time")}</option>
                    </Select>
                  </label>
                  {objectives === "score_time" && <p className="text-[11px] text-ink-3">{t("ml.tune.objectivesHint")}</p>}
                </div>
              </details>
            </div>
          )}
          <section className="space-y-1.5">
            <h3 className="text-[11px] font-medium tracking-wide text-ink-3 uppercase">{t("ml.tune.space")}</h3>
            <table className="w-full border-collapse text-xs">
              <thead className="text-left text-ink-3">
                <tr>
                  <th className="py-1 font-normal">{t("ml.tune.setting")}</th>
                  <th className="py-1 font-normal">{t("ml.tune.range")}</th>
                  <th className="py-1 font-normal">{t("ml.tune.log")}</th>
                  <th className="py-1 font-normal">{t("ml.tune.fixed")}</th>
                </tr>
              </thead>
              <tbody>
                {space.map((p, i) => {
                  const isFixed = p.fixed !== null && p.fixed !== undefined;
                  return (
                    <tr key={p.key} className="border-t border-line">
                      <td className="py-1.5 pr-3 text-ink-1">
                        {labels[p.key] ?? p.key} <span className="font-mono text-[10px] text-ink-3">{p.key}</span>
                      </td>
                      <td className="py-1.5 pr-3">
                        {isFixed ? (
                          <span className="text-ink-2">{String(p.fixed)}</span>
                        ) : p.kind === "categorical" ? (
                          <span className="text-ink-2">{p.choices.join(" · ")}</span>
                        ) : (
                          <span className="inline-flex items-center gap-1">
                            <TextInput aria-label={t("ml.tune.low", { setting: labels[p.key] ?? p.key })} type="number" step="any" defaultValue={p.low ?? ""} onBlur={(e) => patch(i, { low: Number(e.target.value) })} className="h-6 w-20 text-xs" />
                            –
                            <TextInput aria-label={t("ml.tune.high", { setting: labels[p.key] ?? p.key })} type="number" step="any" defaultValue={p.high ?? ""} onBlur={(e) => patch(i, { high: Number(e.target.value) })} className="h-6 w-20 text-xs" />
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 pr-3">{p.kind !== "categorical" && !isFixed && <Checkbox aria-label={t("ml.tune.log")} checked={p.log} onChange={(e) => patch(i, { log: e.target.checked })} />}</td>
                      <td className="py-1.5">
                        <Checkbox
                          aria-label={t("ml.tune.fixFor", { setting: labels[p.key] ?? p.key })}
                          checked={isFixed}
                          onChange={(e) => patch(i, { fixed: e.target.checked ? (settings[p.key] ?? defaults[p.key] ?? null) : null })}
                        />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-xs text-ink-3">{t("ml.tune.spaceHint")}</p>
          </section>
          <div className="flex justify-end border-t border-line pt-3">
            <Button variant="primary" onClick={launch} disabled={start.isPending}>
              {t("ml.tune.start")}
            </Button>
          </div>
          {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
        </div>
      ) : (
        <div className="space-y-4 text-[13px]">
          {job.data?.status === "failed" && <ErrorNote>{job.data.error}</ErrorNote>}
          {progress && (
            <>
              <div className="flex flex-wrap items-center gap-x-6 gap-y-1" aria-live="polite">
                <span>
                  {t("ml.tune.best")} <strong className="tabular-nums">{formatNumber(progress.best_value)}</strong> {metricName(progress.metric)}
                </span>
                <span className="text-ink-2">
                  {t("ml.tune.defaults")} {formatNumber(progress.default_value)}
                </span>
                {gain !== null && (
                  <span className={gain > 0 ? "text-ink-1" : "text-ink-3"}>
                    {t("ml.tune.gain", { gain: `${gain >= 0 ? "+" : ""}${formatNumber(gain)}`, metric: metricName(progress.metric) })}
                  </span>
                )}
                <span className="text-xs text-ink-3">
                  {t("ml.tune.counter", { trials: String(progress.trials.length), seconds: progress.seconds.toFixed(0) })}
                </span>
                {running && (
                  <Button className="ml-auto" onClick={() => jobId && stop.mutate(jobId)} disabled={stop.isPending}>
                    {t("ml.tune.stop")}
                  </Button>
                )}
              </div>
              {progress.trials.length > 0 && (
                <figure className="rounded-md border border-line bg-surface-1 p-2">
                  <Plot figure={trialsFigure(progress.trials, progress.greater_is_better, metricName(progress.metric))} height={240} label={t("ml.tune.chart")} />
                </figure>
              )}
              {progress.objectives === "score_time" && progress.trials.some((trial) => trial.pareto === true) && (
                <figure className="space-y-1 rounded-md border border-line bg-surface-1 p-2">
                  <figcaption className="text-[13px] text-ink-1">{t("ml.tune.paretoSentence", { count: String(progress.trials.filter((trial) => trial.pareto).length) })}</figcaption>
                  <Plot figure={paretoFigure(progress.trials, metricName(progress.metric))} height={240} label={t("ml.tune.pareto")} />
                </figure>
              )}
              {((progress.warm ?? []).length > 0 || (progress.rules ?? []).length > 0) && (
                <ul className="space-y-0.5 text-xs text-ink-2" aria-label={t("ml.tune.notes")}>
                  {(progress.warm ?? []).map((reason, i) => (
                    <li key={`warm-${i}`}>{t("ml.tune.warmTrial", { number: String(i), reason })}</li>
                  ))}
                  {(progress.rules ?? []).map((rule) => (
                    <li key={rule}>{rule}</li>
                  ))}
                </ul>
              )}
            </>
          )}
          {!progress && running && <p className="text-ink-3">{t("ml.tune.preparing")}</p>}
          {progress && job.data?.status !== "running" && Object.keys(progress.best_params).length > 0 && (
            <section className="space-y-2">
              <h3 className="text-[13px] font-medium">{t(job.data?.status === "stopped" ? "ml.tune.stoppedTitle" : "ml.tune.doneTitle")}</h3>
              <table className="w-full max-w-xl border-collapse text-xs tabular-nums">
                <thead className="text-left text-ink-3">
                  <tr>
                    <th className="py-1 font-normal">{t("ml.tune.setting")}</th>
                    <th className="py-1 text-right font-normal">{t("ml.tune.found")}</th>
                    <th className="py-1 text-right font-normal">{t("ml.tune.default")}</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(progress.best_params).map(([key, value]) => {
                    const before = settings[key] ?? defaults[key];
                    const changed = String(before) !== String(value);
                    return (
                      <tr key={key} className={`border-t border-line ${changed ? "text-ink-1" : "text-ink-3"}`}>
                        <td className="py-1">
                          {changed && <span className="mr-1 inline-block size-1.5 rounded-full bg-accent" aria-hidden />}
                          {labels[key] ?? key}
                        </td>
                        <td className="py-1 text-right">{typeof value === "number" ? formatNumber(value) : String(value)}</td>
                        <td className="py-1 text-right">{typeof before === "number" ? formatNumber(before) : String(before ?? "—")}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {Object.keys(progress.importances).length > 0 && (
                <figure className="max-w-xl rounded-md border border-line bg-surface-1 p-2">
                  <Plot figure={importanceBars(progress.importances, labels)} height={60 + 26 * Object.keys(progress.importances).length} label={t("ml.tune.importance")} />
                </figure>
              )}
              <p className="text-xs text-ink-3">{t("ml.tune.useHint")}</p>
              <Button
                variant="primary"
                onClick={() => {
                  onUse({ ...settings, ...progress.best_params });
                  onOpenChange(false);
                }}
              >
                {t("ml.tune.use")}
              </Button>
            </section>
          )}
        </div>
      )}
    </Dialog>
  );
}
