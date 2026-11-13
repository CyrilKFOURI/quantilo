import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

import type {
  MlAutoEntry,
  MlAutoProgress,
  MlCatalogue,
  MlChoicesInput,
  MlClasses,
  MlValidation,
} from "../../api/client";
import { useMlAuto, useMlAutoJob, useMlStopAuto } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import {
  Checkbox,
  ErrorNote,
  ProgressBar,
  Select,
  TextInput,
} from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { entryName, familyName, modelName } from "./catalogText";
import { CriterionSelect, metricName, PANEL } from "./SettingsDrawer";
import { Loading } from "../../ui/Loading";

type Budget = "1" | "2" | "5" | "10" | "30" | "custom";

const DESKTOP_BUDGETS: { value: Budget; label: string }[] = [
  { value: "2", label: "2 min" },
  { value: "5", label: "5 min" },
  { value: "10", label: "10 min" },
  { value: "30", label: "30 min" },
];
/** R-WEB-03: in the browser version the page waits for the Auto, so it stays short (5 minutes at most). */
const BROWSER_BUDGETS: { value: Budget; label: string }[] = [
  { value: "1", label: "1 min" },
  { value: "2", label: "2 min" },
  { value: "5", label: "5 min" },
];
const CROSS = ["kfold", "repeated_kfold", "group_kfold", "loo", PANEL];

/** "1:05" from seconds. */
export function clock(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

/** Every model tried, best first; those without a score (skipped, failed) last. */
export function ranked(progress: MlAutoProgress): MlAutoEntry[] {
  const sign = progress.greater_is_better ? 1 : -1;
  const scored = progress.entries.filter(
    (e) => e.status === "done" && e.score !== null,
  );
  scored.sort((a, b) => sign * ((b.score ?? 0) - (a.score ?? 0)));
  return [...scored, ...progress.entries.filter((e) => !scored.includes(e))];
}

const STATUS_TONE: Record<string, string> = {
  done: "text-ink-2",
  running: "text-accent",
  waiting: "text-ink-3",
  skipped: "text-ink-3",
  stopped: "text-warn",
  failed: "text-bad",
};

function settingsText(
  entry: MlAutoEntry,
  progress: MlAutoProgress,
): { short: string; full: string } {
  if (entry.model === "ensemble" && progress.ensemble) {
    const members = progress.ensemble.members
      .map((m) =>
        progress.ensemble?.method === "greedy"
          ? `${modelName(m.model, m.label)} ×${m.weight}`
          : modelName(m.model, m.label),
      )
      .join(", ");
    return { short: members, full: members };
  }
  const pairs = Object.entries(entry.settings);
  if (!pairs.length)
    return { short: t("ml.auto.defaults"), full: t("ml.auto.defaults") };
  const full = pairs
    .map(
      ([key, value]) =>
        `${key} = ${typeof value === "number" ? formatNumber(value) : String(value)}`,
    )
    .join(", ");
  return { short: t("ml.auto.changed", { count: String(pairs.length) }), full };
}

function Leaderboard({ progress }: { progress: MlAutoProgress }) {
  const metric = metricName(progress.metric);
  return (
    <section className="space-y-1.5">
      <h3 className="text-[11px] font-medium tracking-wide text-ink-3 uppercase">
        {t("ml.auto.leaderboard")}
      </h3>
      <table className="w-full border-collapse text-[13px] tabular-nums">
        <thead className="text-left text-xs text-ink-3">
          <tr>
            <th className="py-1.5 font-normal">{t("ml.model")}</th>
            <th className="py-1.5 font-normal">{t("ml.auto.settings")}</th>
            <th className="py-1.5 text-right font-normal">
              {metric} {t("ml.auto.score")}
            </th>
            <th className="py-1.5 text-right font-normal">
              {t("ml.compare.seconds")}
            </th>
          </tr>
        </thead>
        <tbody>
          {ranked(progress).map((entry) => {
            const kept = progress.best === entry.id;
            const settings = settingsText(entry, progress);
            return (
              <tr
                key={entry.id}
                className={`border-t border-line align-top ${kept ? "font-medium text-ink-1" : "text-ink-2"}`}
              >
                <td className="py-1.5 pr-3">
                  {entryName(entry.model, entry.label)}
                  {kept && (
                    <span className="ml-1.5 rounded-sm bg-accent-soft px-1 text-[11px] font-normal text-accent">
                      {t("ml.auto.kept")}
                    </span>
                  )}
                  <span className="block text-[11px] font-normal text-ink-3">
                    {familyName(entry.family)}
                  </span>
                  {entry.note && (
                    <span
                      className={`block text-xs font-normal ${entry.status === "failed" ? "text-bad" : "text-ink-3"}`}
                    >
                      {entry.note}
                    </span>
                  )}
                </td>
                <td
                  className="max-w-56 py-1.5 pr-3 text-xs font-normal text-ink-3"
                  title={settings.full}
                >
                  {settings.short}
                </td>
                <td className="py-1.5 text-right whitespace-nowrap">
                  {entry.status === "done" ? (
                    <>
                      {formatNumber(entry.score)}
                      {entry.std !== null && (
                        <span className="text-xs font-normal text-ink-3">
                          {" "}
                          ± {formatNumber(entry.std)}
                        </span>
                      )}
                      {entry.share < 1 && (
                        <span className="block text-[11px] font-normal text-ink-3">
                          {t("ml.auto.rows", {
                            share: String(Math.round(entry.share * 100)),
                          })}
                        </span>
                      )}
                    </>
                  ) : (
                    <span
                      className={`text-xs font-normal ${STATUS_TONE[entry.status] ?? "text-ink-3"}`}
                    >
                      {t(`ml.auto.status.${entry.status}` as MessageKey)}
                    </span>
                  )}
                </td>
                <td className="py-1.5 text-right text-ink-3">
                  {entry.seconds ? entry.seconds.toFixed(1) : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function Steps({ progress }: { progress: MlAutoProgress }) {
  return (
    <ol className="grid gap-x-4 gap-y-2 sm:grid-cols-5">
      {progress.steps.map((step, index) => (
        <li
          key={step.key}
          className={`border-t-2 pt-1.5 ${step.status === "running" ? "border-accent" : step.status === "done" ? "border-ink-3" : "border-line"}`}
        >
          <span className="block text-[11px] text-ink-3">
            {index + 1} ·{" "}
            <span className={STATUS_TONE[step.status] ?? ""}>
              {t(`ml.auto.status.${step.status}` as MessageKey)}
            </span>
            {step.seconds > 0 && ` · ${clock(step.seconds)}`}
          </span>
          <span className="block text-[13px] text-ink-1">
            {t(`ml.auto.step.${step.key}` as MessageKey)}
          </span>
          <span className="block text-xs text-ink-3">
            {step.note || t(`ml.auto.stepHint.${step.key}` as MessageKey)}
          </span>
        </li>
      ))}
    </ol>
  );
}

/** AutoML within a time budget (038): the steps as they run, the journal, the leaderboard. */
export function AutoPanel({
  open,
  onOpenChange,
  datasetId,
  target,
  catalogue,
  choices,
  classes,
  validation,
  jobId,
  onJob,
  onRun,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  datasetId: string;
  target: string;
  catalogue: MlCatalogue;
  choices: MlChoicesInput;
  classes: MlClasses | null;
  validation: MlValidation;
  /** The Auto being followed: kept by the page, so closing the window does not lose it. */
  jobId: string | null;
  onJob: (jobId: string | null) => void;
  onRun: (runId: string) => void;
}) {
  const client = useQueryClient();
  const limits = catalogue.browser;
  const [budget, setBudget] = useState<Budget>(limits ? "2" : "5");
  const [custom, setCustom] = useState(15);
  const [metric, setMetric] = useState(
    validation.metric ?? catalogue.default_metric,
  );
  const [judged, setJudged] = useState<MlValidation>(validation);
  const [stacking, setStacking] = useState(false);
  const start = useMlAuto(datasetId);
  const job = useMlAutoJob(jobId);
  const stop = useMlStopAuto();
  const progress = job.data?.progress ?? null;
  const running =
    start.isPending ||
    (jobId !== null && (!job.data || job.data.status === "running"));
  const finished = job.data !== undefined && job.data.status !== "running";
  const minutes = budget === "custom" ? custom : Number(budget);
  const split = !CROSS.includes(validation.method);
  const panel = judged.method === PANEL;
  // the kept model is a new run: show it in the run list once the Auto ends
  useEffect(() => {
    if (finished)
      void client.invalidateQueries({
        queryKey: ["datasets", datasetId, "ml", "runs", target],
      });
  }, [finished, client, datasetId, target]);

  const launch = () =>
    start.mutate(
      {
        target,
        classes,
        validation: { ...judged, metric },
        body: { minutes, stacking, preparation: choices },
      },
      { onSuccess: (created) => onJob(created.id) },
    );
  const current = progress?.steps.find((s) => s.status === "running");
  const kept = progress?.entries.find((e) => e.id === progress.best);

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      size="lg"
      title={t("ml.auto.title")}
      description={t("ml.auto.description")}
    >
      {!jobId ? (
        <div className="space-y-4 text-[13px]">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs text-ink-2">{t("ml.auto.budget")}</span>
            <SegmentedControl
              label={t("ml.auto.budget")}
              value={budget}
              onChange={setBudget}
              options={limits ? BROWSER_BUDGETS : [...DESKTOP_BUDGETS, { value: "custom", label: t("ml.auto.custom") }]}
            />
            {budget === "custom" && (
              <label className="flex items-center gap-2 text-xs text-ink-2">
                <TextInput
                  aria-label={t("ml.auto.minutes")}
                  type="number"
                  min={1}
                  max={240}
                  value={custom}
                  onChange={(e) =>
                    setCustom(
                      Math.max(1, Math.min(240, Number(e.target.value) || 1)),
                    )
                  }
                  className="h-7 w-20 text-xs"
                />
                {t("ml.auto.minutes").toLowerCase()}
              </label>
            )}
          </div>
          {limits && (
            <p className="text-xs text-ink-3">{t("ml.auto.browserBudget", { minutes: String(limits.auto_minutes) })}</p>
          )}
          <div className="flex flex-wrap items-center gap-4">
            <label className="flex items-center gap-2 text-xs text-ink-2">
              {t("ml.auto.metric")}
              <Select
                aria-label={t("ml.auto.metric")}
                value={metric}
                onChange={(e) => setMetric(e.target.value)}
                className="h-7 text-xs"
              >
                {catalogue.metrics.map((m) => (
                  <option key={m} value={m}>
                    {metricName(m)}
                  </option>
                ))}
              </Select>
            </label>
            {panel && (
              <label className="flex items-center gap-2 text-xs text-ink-2">
                {t("ml.validation.criterion")}
                <CriterionSelect validation={judged} onValidation={setJudged} />
              </label>
            )}
            <Checkbox
              className="text-xs text-ink-2"
              label={t("ml.auto.stacking")}
              checked={stacking}
              onChange={(e) => setStacking(e.target.checked)}
            />
          </div>
          {split && <p className="text-xs text-ink-3">{t("ml.auto.split")}</p>}
          <div className="flex justify-end border-t border-line pt-3">
            <Button
              variant="primary"
              onClick={launch}
              disabled={start.isPending}
            >
              {t("ml.auto.start")}
            </Button>
          </div>
          {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
        </div>
      ) : (
        <div className="space-y-4 text-[13px]">
          {job.data?.status === "failed" && (
            <ErrorNote>{job.data.error}</ErrorNote>
          )}
          {!progress && running && (
            <Loading label={t("ml.auto.loading")} className="h-24 rounded-lg" />
          )}
          {progress && (
            <>
              <div className="space-y-1.5">
                <div
                  className="flex flex-wrap items-center gap-x-4 gap-y-1"
                  aria-live="polite"
                >
                  <span className="text-ink-1">
                    {current
                      ? t(`ml.auto.step.${current.key}` as MessageKey)
                      : finished
                        ? t(
                            `ml.auto.status.${job.data?.status ?? "done"}` as MessageKey,
                          )
                        : t("ml.auto.loading")}
                  </span>
                  <span className="text-xs text-ink-3 tabular-nums">
                    {t("ml.auto.clock", {
                      elapsed: clock(progress.seconds),
                      budget: clock(progress.budget_seconds),
                    })}
                  </span>
                  {progress.trials > 0 && (
                    <span className="text-xs text-ink-3">
                      {t("ml.auto.trials", { trials: String(progress.trials) })}
                    </span>
                  )}
                  {running && (
                    <Button
                      className="ml-auto"
                      onClick={() => jobId && stop.mutate(jobId)}
                      disabled={stop.isPending}
                    >
                      {t("ml.auto.stop")}
                    </Button>
                  )}
                </div>
                <ProgressBar
                  value={
                    finished ? 1 : progress.seconds / progress.budget_seconds
                  }
                  label={t("ml.auto.progress")}
                />
                {progress.compared_on && (
                  <p className="text-xs text-ink-3">{progress.compared_on}</p>
                )}
              </div>
              <Steps progress={progress} />
              {finished && (
                <section className="space-y-1 border-t border-line pt-3">
                  <h3 className="text-[13px] font-medium text-ink-1">
                    {job.data?.status === "stopped"
                      ? t("ml.auto.stoppedTitle")
                      : kept
                        ? t("ml.auto.doneTitle", { model: kept.label })
                        : t("ml.auto.failedTitle")}
                  </h3>
                  {progress.gain && (
                    <p className="text-ink-1">{progress.gain}.</p>
                  )}
                  {progress.ensemble && (
                    <p className="text-xs text-ink-2">
                      {progress.ensemble.reason}
                    </p>
                  )}
                  {progress.test_score !== null && (
                    <p className="text-xs text-ink-2">
                      {t("ml.auto.test", {
                        score: `${metricName(progress.metric)} ${formatNumber(progress.test_score)}`,
                      })}
                    </p>
                  )}
                  <div className="flex flex-wrap gap-2 pt-1">
                    {job.data?.run_id && (
                      <Button
                        variant="primary"
                        onClick={() => {
                          onRun(job.data?.run_id as string);
                          onOpenChange(false);
                        }}
                      >
                        {t("ml.auto.open")}
                      </Button>
                    )}
                    <Button variant="ghost" onClick={() => onJob(null)}>
                      {t("ml.auto.again")}
                    </Button>
                  </div>
                </section>
              )}
              <Leaderboard progress={progress} />
              <section className="space-y-1.5">
                <h3 className="text-[11px] font-medium tracking-wide text-ink-3 uppercase">
                  {t("ml.auto.journal")}
                </h3>
                <ol className="max-h-48 space-y-0.5 overflow-auto rounded-md border border-line bg-surface-0 p-2 font-mono text-[11px] text-ink-2">
                  {progress.log.map((line, index) => (
                    <li
                      key={index}
                      className="grid grid-cols-[3rem_minmax(0,1fr)] gap-2"
                    >
                      <span className="text-right text-ink-3 tabular-nums">
                        {clock(line.at)}
                      </span>
                      <span>{line.message}</span>
                    </li>
                  ))}
                </ol>
              </section>
            </>
          )}
        </div>
      )}
    </Dialog>
  );
}
