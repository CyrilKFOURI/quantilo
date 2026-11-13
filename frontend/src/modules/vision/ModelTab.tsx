import { useQueryClient } from "@tanstack/react-query";
import { Check, ChevronRight, CircleSlash, Download, Play, Square, Star, Trash2 } from "lucide-react";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import { formatCount, formatRelativeTime } from "../../lib/format";
import { LITE } from "../../lite/flag";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, ProgressBar, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import {
  modelKeys,
  useCancelRun,
  useDeleteRun,
  useDownloadWeights,
  useRuns,
  useStartRun,
  useTask,
  useVisionJob,
  type DetectorKey,
  type DetectorOut,
  type RunSummary,
  type Task,
  type TaskOut,
  type VisionJob,
  type WeightOut,
} from "./modelApi";
import type { MoreKey } from "./moreApi";
import { MoreOptions } from "./MoreOptions";
import { metric } from "./scores";

type ModelProps = {
  datasetId: string;
  jobId: string | null;
  onJob: (jobId: string | null) => void;
  /** A finished (or chosen) run: open it in Results. */
  onOpenRun: (runId: string) => void;
};

const TASK_LABEL: Record<Task, MessageKey> = {
  classification: "vision.model.task.classification",
  anomaly: "vision.model.task.anomaly",
  clusters: "vision.model.task.clusters",
  regression: "vision.model.task.regression",
  multilabel: "vision.model.task.multilabel",
};

/**
 * 113: the Model tab. The task the rule reads (R-VML) comes first with its code and sentence, then
 * the split and what a run trains, the default marked; one button starts the job (progress, stop).
 * Advanced (folded) lets the user impose another possible task, keep the classic models only, or
 * choose the anomaly detector, each deep option greyed with its reason (R-113-04) and its weights
 * downloaded only after a confirmation that shows their size and licence (R-113-05). Opening the
 * tab only reads the rule (R-113-01).
 */
export function Model({ datasetId, jobId, onJob, onOpenRun }: ModelProps) {
  const [task, setTask] = useState<Task | undefined>(undefined);
  // c3, R-VML-05: the column a regression predicts (its first candidate when not chosen)
  const [target, setTarget] = useState<string | undefined>(undefined);
  const [classicOnly, setClassicOnly] = useState(false);
  const [detector, setDetector] = useState<DetectorKey>("auto");
  // 118: Advanced › More models ticked by the user, and the bottleneck's concepts (null = all)
  const [more, setMore] = useState<MoreKey[]>([]);
  const [concepts, setConcepts] = useState<string[] | null>(null);
  const query = useTask(datasetId, task, true, task === "regression" ? target : undefined);
  const runs = useRuns(datasetId, true);
  const job = useVisionJob(jobId);
  const start = useStartRun(datasetId);
  const cancel = useCancelRun();
  const client = useQueryClient();
  const running = job.data?.status === "running" || start.isPending;

  // A run that ends: the list of runs is read again, and a finished one opens in Results.
  const finished = job.data?.status === "done" ? job.data.run_id : null;
  const opened = useRef<string | null>(null);
  useEffect(() => {
    if (!finished || opened.current === finished) return;
    opened.current = finished;
    void client.invalidateQueries({ queryKey: modelKeys.runs(datasetId) }).then(() => onOpenRun(finished));
  }, [finished, client, datasetId, onOpenRun]);

  if (query.isPending) return <Loading label={t("vision.model.loading")} className="h-72 rounded-lg" />;
  if (query.isError) return <ErrorNote>{(query.error as Error).message}</ErrorNote>;
  const out = query.data;
  const chosenMore = out.choice.task === "classification" ? more.filter((key) => out.more?.some((o) => o.key === key && o.available)) : [];

  const launch = (confirmLong = false) =>
    start.mutate(
      {
        task: task ?? null,
        deep: classicOnly ? false : null,
        detector: classicOnly ? "none" : detector,
        confirm_long: confirmLong,
        threshold: null,
        seed: 0,
        more: chosenMore,
        concepts: chosenMore.includes("concept_bottleneck") ? concepts : null,
        target: out.choice.task === "regression" ? (out.choice.target ?? null) : null,
      },
      { onSuccess: (next) => onJob(next.id) },
    );

  return (
    <div className="flex flex-col gap-6">
      <TaskHeader out={out} />

      <section aria-labelledby="vision-model-plan" className="grid gap-x-8 gap-y-5 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <h3 id="vision-model-plan" className="text-[13px] font-medium text-ink-1">
            {t("vision.model.trains")}
          </h3>
          <ModelList out={out} detector={classicOnly ? "none" : detector} classicOnly={classicOnly} more={chosenMore} />
        </div>
        {out.split && (
          <div className="min-w-0">
            <h3 className="text-[13px] font-medium text-ink-1">{t("vision.model.split")}</h3>
            <dl className="mt-2 grid grid-cols-3 gap-x-4 gap-y-1 text-[13px] max-sm:grid-cols-3">
              <Fact label={t("vision.model.learnt")} value={formatCount(out.split.train)} />
              <Fact label={t("vision.model.scored")} value={formatCount(out.split.scored)} />
              <Fact label={out.split.folds > 1 ? t("vision.model.folds") : t("vision.model.heldOut")} value={formatCount(out.split.folds > 1 ? out.split.folds : out.split.holdout)} />
            </dl>
            <p className="mt-2 max-w-prose text-[13px] text-pretty text-ink-2">
              {out.split.reason} <RuleCode code={out.split.code} />
            </p>
            {out.split.leaks > 0 && (
              <p className="mt-1.5 text-[13px] text-warn">{t("vision.model.leaks", { count: formatCount(out.split.leaks) })}</p>
            )}
            {out.split.notes.map((note) => (
              <p key={note} className="mt-1 max-w-prose text-xs text-ink-3">
                {note}
              </p>
            ))}
          </div>
        )}
      </section>

      <RunBar
        out={out}
        job={job.data ?? null}
        running={running}
        error={(start.error as Error | null) ?? null}
        onStart={() => launch(false)}
        onAnyway={() => launch(true)}
        onStop={() => jobId && cancel.mutate(jobId)}
      />

      <Advanced
        datasetId={datasetId}
        out={out}
        task={task}
        onTask={setTask}
        target={out.choice.target ?? undefined}
        onTarget={setTarget}
        classicOnly={classicOnly}
        onClassicOnly={setClassicOnly}
        detector={detector}
        onDetector={setDetector}
        more={more}
        onMore={setMore}
        concepts={concepts}
        onConcepts={setConcepts}
      />

      <RunList datasetId={datasetId} runs={runs.data ?? []} onOpen={onOpenRun} />
    </div>
  );
}

function RuleCode({ code }: { code: string }) {
  return <span className="font-mono text-xs whitespace-nowrap text-ink-3">{code}</span>;
}

function Fact({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-ink-3">{label}</dt>
      <dd className="text-[15px] font-medium text-ink-1 tabular-nums">{value}</dd>
    </div>
  );
}

/** R-113-02: the task and the rule that chose it, in the words of the rule. */
function TaskHeader({ out }: { out: TaskOut }) {
  const choice = out.choice;
  const classes = Object.entries(choice.classes);
  const leftOut = Object.entries(choice.left_out);
  return (
    <section aria-labelledby="vision-task" className="border-b border-line pb-5">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 id="vision-task" className="text-xl font-semibold text-balance text-ink-1">
          {choice.task ? t(TASK_LABEL[choice.task]) : t("vision.model.task.none")}
        </h2>
        <RuleCode code={choice.code} />
        {choice.chosen_by === "user" && <span className="text-xs text-ink-3">{t("vision.model.chosenByYou")}</span>}
      </div>
      <p className="mt-1.5 max-w-[72ch] text-[14px] text-pretty text-ink-2">{choice.message}</p>
      {choice.chosen_by === "user" && choice.rule_message && (
        <p className="mt-1 max-w-[72ch] text-[13px] text-ink-3">
          {t("vision.model.ruleReads", { message: choice.rule_message })} <RuleCode code={choice.rule_code} />
        </p>
      )}
      {choice.blocked && <p className="mt-2 text-[13px] text-warn">{choice.blocked}</p>}
      <dl className="mt-4 flex flex-wrap gap-x-8 gap-y-3">
        <Fact label={t("vision.model.usable")} value={`${formatCount(choice.n_usable)} / ${formatCount(choice.n_images)}`} />
        {choice.task === "classification" && <Fact label={t("vision.model.classes")} value={formatCount(classes.length)} />}
        {choice.task === "regression" && (
          <>
            <Fact label={t("vision.model.target")} value={<span className="font-mono">{choice.target}</span>} />
            <Fact label={t("vision.model.known")} value={formatCount(choice.n_target)} />
          </>
        )}
        {choice.task === "multilabel" && <Fact label={t("vision.model.labels")} value={formatCount(choice.label_names.length)} />}
        {choice.task === "anomaly" && (
          <>
            <Fact label={t("vision.model.normals")} value={formatCount(choice.n_normal)} />
            <Fact label={t("vision.model.defects")} value={formatCount(choice.n_defect)} />
            <Fact label={t("vision.model.pixelMetrics")} value={choice.pixel ? t("vision.yes") : t("vision.no")} />
          </>
        )}
      </dl>
      {choice.task === "multilabel" && (
        <p className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[13px] text-ink-1">
          {choice.label_names.map((label) => (
            <span key={label} className="whitespace-nowrap">
              {label}
            </span>
          ))}
        </p>
      )}
      {classes.length > 0 && (
        <p className="mt-3 flex flex-wrap gap-x-3 gap-y-1 text-[13px] text-ink-2">
          {classes.map(([label, count]) => (
            <span key={label} className="whitespace-nowrap">
              <span className="text-ink-1">{label}</span> <span className="text-ink-3 tabular-nums">{formatCount(count)}</span>
            </span>
          ))}
        </p>
      )}
      {leftOut.length > 0 && (
        <p className="mt-1.5 text-[13px] text-ink-3">
          {t("vision.model.leftOut", { classes: leftOut.map(([label, count]) => `${label} (${formatCount(count)})`).join(", ") })}
        </p>
      )}
      {choice.notes.filter((note) => !choice.message.includes(note)).map((note) => (
        <p key={note} className="mt-1 max-w-[72ch] text-xs text-ink-3">
          {note}
        </p>
      ))}
    </section>
  );
}

/** What a run trains, the default first and marked (R-113-03). */
function ModelList({ out, detector, classicOnly, more }: { out: TaskOut; detector: DetectorKey; classicOnly: boolean; more: readonly MoreKey[] }) {
  const task = out.choice.task;
  const chosen = detector === "auto" ? out.default_detector : detector === "none" ? null : detector;
  const deep = out.detectors.find((d) => d.key === chosen && d.available);
  const titles = out.models.filter((title) => !out.detectors.some((d) => d.title === title));
  const classic = classicOnly ? titles.filter((title) => !title.includes("DINOv2")) : titles;
  const defaultTitle = deep?.title ?? defaultOf(task, classic, out.deep.used && !classicOnly);
  // the default first (R-113-03), the others in the order the run trains them
  const all = deep ? [deep.title, ...classic] : classic;
  const ticked = (out.more ?? []).filter((o) => more.includes(o.key)).map((o) => o.title);
  const items = [...(defaultTitle ? [defaultTitle, ...all.filter((title) => title !== defaultTitle)] : all), ...ticked];
  if (items.length === 0) return <p className="mt-2 text-[13px] text-ink-3">{t("vision.model.nothing")}</p>;
  return (
    <ul className="mt-2 divide-y divide-line border-y border-line">
      {items.map((title) => {
        const isDefault = title === defaultTitle;
        return (
          <li key={title} className="flex min-w-0 items-center gap-2 py-2 text-[13px]">
            {isDefault ? <Star size={14} strokeWidth={1.5} className="shrink-0 text-accent" aria-label={t("vision.model.default")} /> : <span className="w-3.5 shrink-0" aria-hidden />}
            <span className={`min-w-0 flex-1 ${isDefault ? "font-medium text-ink-1" : "text-ink-2"}`}>{title}</span>
            {isDefault && <span className="text-xs text-ink-3">{t("vision.model.default")}</span>}
            {ticked.includes(title) && <span className="text-xs text-ink-3">{t("vision.more.level")}</span>}
          </li>
        );
      })}
    </ul>
  );
}

/** The default the engine will pick before it runs: the DINOv2 probe when it is there
 *  (R-VCL-06), the ensemble for anomalies (R-VAN-03); for classic classification, the better of
 *  the two by inner cross-validation, known only after the run (R-VCL-04). */
function defaultOf(task: Task | null | undefined, titles: string[], deep: boolean): string | undefined {
  if (task === "classification") return deep ? titles.find((title) => title.startsWith("Linear probe")) : undefined;
  if (task === "anomaly") return titles.find((title) => title.includes("average rank"));
  return titles[0];
}

/** Seconds since it was mounted (it is mounted while a job runs), shown after 3 s (042). */
function Elapsed() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, []);
  return seconds >= 3 ? <> · {t("vision.model.elapsed", { seconds: formatCount(seconds) })}</> : null;
}

const STAGE: Record<VisionJob["stage"], MessageKey> = {
  features: "vision.model.stage.features",
  embedding: "vision.model.stage.embedding",
  training: "vision.model.stage.training",
  saving: "vision.model.stage.saving",
};

function RunBar({
  out,
  job,
  running,
  error,
  onStart,
  onAnyway,
  onStop,
}: {
  out: TaskOut;
  job: VisionJob | null;
  running: boolean;
  error: Error | null;
  onStart: () => void;
  onAnyway: () => void;
  onStop: () => void;
}) {
  const blocked = !out.choice.task;
  return (
    <section aria-label={t("vision.model.run")} className="rounded-lg border border-line bg-surface-1 p-4" aria-live="polite">
      {running ? (
        <div className="flex flex-col gap-2.5">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <p className="text-[13px] font-medium text-ink-1">{job ? t(STAGE[job.stage]) : t("vision.model.starting")}</p>
            <p className="text-xs text-ink-3 tabular-nums">
              {job ? `${Math.round(job.progress * 100)} %` : ""}
              <Elapsed />
            </p>
          </div>
          <ProgressBar value={job?.progress ?? 0} label={t("vision.model.progress")} />
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="min-w-0 text-[13px] text-ink-2 tabular-nums">{job?.message ?? t("vision.model.starting")}</p>
            <Button variant="secondary" icon={Square} onClick={onStop} disabled={!job}>
              {t("vision.model.stop")}
            </Button>
          </div>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
          <Button variant="primary" icon={Play} onClick={onStart} disabled={blocked} className="max-sm:w-full max-sm:justify-center">
            {t("vision.model.train")}
          </Button>
          <p className="min-w-0 flex-1 basis-64 text-[13px] text-pretty text-ink-3">{blocked ? out.choice.blocked : t("vision.model.trainHint")}</p>
        </div>
      )}
      {!running && job?.status === "cancelled" && <p className="mt-3 text-[13px] text-ink-2">{job.error}</p>}
      {!running && job?.status === "failed" && (
        <div className="mt-3 flex flex-col gap-2">
          <ErrorNote>{job.error}</ErrorNote>
          {job.long_run && (
            <div className="flex flex-wrap items-center gap-3">
              <p className="text-[13px] text-ink-2">{t("vision.model.longRun", { estimate: job.long_run })}</p>
              <Button variant="secondary" onClick={onAnyway}>
                {t("vision.model.runAnyway")}
              </Button>
            </div>
          )}
        </div>
      )}
      {error && (
        <div className="mt-3">
          <ErrorNote>{error.message}</ErrorNote>
        </div>
      )}
    </section>
  );
}

function Advanced({
  datasetId,
  out,
  task,
  onTask,
  target,
  onTarget,
  classicOnly,
  onClassicOnly,
  detector,
  onDetector,
  more,
  onMore,
  concepts,
  onConcepts,
}: {
  datasetId: string;
  out: TaskOut;
  task: Task | undefined;
  onTask: (task: Task | undefined) => void;
  target: string | undefined;
  onTarget: (target: string | undefined) => void;
  classicOnly: boolean;
  onClassicOnly: (value: boolean) => void;
  detector: DetectorKey;
  onDetector: (value: DetectorKey) => void;
  more: readonly MoreKey[];
  onMore: (next: MoreKey[]) => void;
  concepts: readonly string[] | null;
  onConcepts: (next: string[] | null) => void;
}) {
  const choice = out.choice;
  const ruleTask = choice.chosen_by === "user" ? choice.rule_task : choice.task;
  const options: Task[] = [...new Set([ruleTask, choice.task, ...choice.alternatives].filter((value): value is Task => Boolean(value)))];
  const [asking, setAsking] = useState<WeightOut | null>(null);
  const download = useDownloadWeights(datasetId);
  const weights = new Map(out.weights.map((w) => [w.key, w]));
  return (
    <details className="group/advanced rounded-lg border border-line bg-surface-1 open:pb-1 [&_summary::-webkit-details-marker]:hidden">
      <summary className="flex cursor-pointer list-none items-center gap-2 rounded-lg px-4 py-3 text-[13px] font-medium text-ink-1 hover:bg-surface-2">
        <ChevronRight size={15} strokeWidth={1.5} aria-hidden className="text-ink-3 transition-transform duration-150 group-open/advanced:rotate-90 motion-reduce:transition-none" />
        {t("vision.model.advanced")}
        <span className="font-normal text-ink-3">{t("vision.model.advancedHint")}</span>
      </summary>
      <div className="flex flex-col gap-5 border-t border-line px-4 pt-4 pb-3">
        <div className="grid gap-4 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)]">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-ink-2">{t("vision.model.taskLabel")}</span>
            <Select value={task ?? ""} onChange={(event) => onTask((event.target.value || undefined) as Task | undefined)}>
              <option value="">{t("vision.model.taskRule", { task: ruleTask ? t(TASK_LABEL[ruleTask]) : "—" })}</option>
              {options
                .filter((value) => value !== ruleTask)
                .map((value) => (
                  <option key={value} value={value}>
                    {t(TASK_LABEL[value])}
                  </option>
                ))}
            </Select>
          </label>
          <p className="self-end text-xs text-pretty text-ink-3">{t("vision.model.taskHint")}</p>
        </div>
        {choice.task === "regression" && choice.numeric_targets.length > 0 && (
          <div className="grid gap-4 sm:grid-cols-[minmax(0,14rem)_minmax(0,1fr)]">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-ink-2">{t("vision.model.targetLabel")}</span>
              <Select value={target ?? choice.numeric_targets[0]} onChange={(event) => onTarget(event.target.value)}>
                {choice.numeric_targets.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            </label>
            <p className="self-end text-xs text-pretty text-ink-3">{t("vision.model.targetHint")}</p>
          </div>
        )}
        {(choice.task === "regression" || choice.task === "multilabel") && (
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs text-ink-2">{t("vision.model.vectors")}</span>
            <SegmentedControl
              label={t("vision.model.vectors")}
              value={classicOnly ? "classic" : "auto"}
              onChange={(value) => onClassicOnly(value === "classic")}
              options={[
                { value: "auto", label: t("vision.model.vectorsAuto") },
                { value: "classic", label: t("vision.model.vectorsClassic") },
              ]}
            />
          </div>
        )}

        {choice.task === "classification" && (
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-3">
              <span className="text-xs text-ink-2">{t("vision.model.vectors")}</span>
              <SegmentedControl
                label={t("vision.model.vectors")}
                value={classicOnly ? "classic" : "auto"}
                onChange={(value) => onClassicOnly(value === "classic")}
                options={[
                  { value: "auto", label: t("vision.model.vectorsAuto") },
                  { value: "classic", label: t("vision.model.vectorsClassic") },
                ]}
              />
            </div>
            <OptionRow
              title={t("vision.model.dinoTitle")}
              note={t("vision.model.dinoNote")}
              available={out.deep.used}
              reason={out.deep.reason ?? null}
              weights={["dinov2_s14"].map((key) => weights.get(key)).filter((w): w is WeightOut => Boolean(w))}
              installed={out.installed}
              install={out.install}
              onDownload={setAsking}
            />
          </div>
        )}

        {choice.task === "anomaly" && (
          <fieldset className="flex flex-col gap-1">
            <legend className="mb-1.5 text-xs text-ink-2">{t("vision.model.detector")}</legend>
            <DetectorChoice
              value="auto"
              current={detector}
              onChange={onDetector}
              title={t("vision.model.detectorAuto", { name: out.default_detector ? (out.detectors.find((d) => d.key === out.default_detector)?.title ?? "") : t("vision.model.detectorClassicName") })}
              note={t("vision.model.detectorAutoNote")}
            />
            {out.detectors.map((d) => (
              <DetectorChoice
                key={d.key}
                value={d.key as DetectorKey}
                current={detector}
                onChange={onDetector}
                detector={d}
                weights={d.weights.map((key) => weights.get(key)).filter((w): w is WeightOut => Boolean(w))}
                installed={out.installed}
                install={out.install}
                onDownload={setAsking}
              />
            ))}
            <DetectorChoice value="none" current={detector} onChange={onDetector} title={t("vision.model.detectorNone")} note={t("vision.model.detectorNoneNote")} />
          </fieldset>
        )}
        {choice.task === "classification" && <MoreOptions out={out} more={more} onMore={onMore} concepts={concepts} onConcepts={onConcepts} />}
        {download.isError && <ErrorNote>{(download.error as Error).message}</ErrorNote>}
      </div>

      {asking && (
        <Dialog
          open
          onOpenChange={(open) => !open && setAsking(null)}
          size="sm"
          title={t("vision.model.downloadTitle", { title: asking.title })}
          description={t("vision.model.downloadBody", { size: asking.size, licence: asking.licence })}
          footer={
            <>
              <Button variant="ghost" onClick={() => setAsking(null)}>
                {t("vision.model.cancel")}
              </Button>
              <Button
                variant="primary"
                icon={Download}
                onClick={() => {
                  download.mutate(asking.key);
                  setAsking(null);
                }}
              >
                {t("vision.model.downloadConfirm", { size: asking.size })}
              </Button>
            </>
          }
        >
          <div className="flex flex-col gap-2 text-[13px] text-ink-2">
            {asking.note && <p>{asking.note}</p>}
            <p>{t("vision.model.downloadSafety")}</p>
          </div>
        </Dialog>
      )}
    </details>
  );
}

function WeightsLine({ weights, installed, install, onDownload }: { weights: WeightOut[]; installed: boolean; install: string; onDownload: (w: WeightOut) => void }) {
  // 119: nothing to install in the browser version (the option's reason says "Desktop only")
  if (!installed) return LITE ? null : <p className="text-xs text-ink-3">{t("vision.model.installHint", { command: install })}</p>;
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      {weights.map((w) =>
        w.downloaded ? (
          <span key={w.key} className="inline-flex items-center gap-1 text-xs text-ok">
            <Check size={13} strokeWidth={1.75} aria-hidden /> {t("vision.model.weightsHere", { title: w.title })}
          </span>
        ) : w.download === "running" ? (
          <span key={w.key} className="inline-flex min-w-48 flex-col gap-1 text-xs text-ink-2">
            {t("vision.model.downloading", { title: w.title, percent: String(Math.round(w.progress * 100)) })}
            <ProgressBar value={w.progress} label={t("vision.model.downloading", { title: w.title, percent: String(Math.round(w.progress * 100)) })} />
          </span>
        ) : (
          <span key={w.key} className="inline-flex flex-wrap items-center gap-2">
            <Button variant="secondary" icon={Download} onClick={() => onDownload(w)} className="h-7 text-xs">
              {t("vision.model.download", { title: w.title, size: w.size })}
            </Button>
            {w.error && <span className="text-xs text-bad">{w.error}</span>}
          </span>
        ),
      )}
    </div>
  );
}

function OptionRow({
  title,
  note,
  available,
  reason,
  weights,
  installed,
  install,
  onDownload,
}: {
  title: string;
  note: string;
  available: boolean;
  reason: string | null;
  weights: WeightOut[];
  installed: boolean;
  install: string;
  onDownload: (w: WeightOut) => void;
}) {
  return (
    <div className="flex flex-col gap-1.5 rounded-md border border-line px-3 py-2.5">
      <p className="flex flex-wrap items-center gap-2 text-[13px]">
        {available ? <Check size={14} strokeWidth={1.75} className="text-ok" aria-hidden /> : <CircleSlash size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />}
        <span className={available ? "font-medium text-ink-1" : "text-ink-2"}>{title}</span>
        <span className="text-xs text-ink-3">{available ? t("vision.model.ready") : t("vision.model.unavailable")}</span>
      </p>
      <p className="text-xs text-pretty text-ink-3">{note}</p>
      {!available && reason && <p className="text-xs text-pretty text-ink-2">{reason}</p>}
      {weights.length > 0 && <WeightsLine weights={weights} installed={installed} install={install} onDownload={onDownload} />}
    </div>
  );
}

/** One radio of the anomaly detector, greyed with its reason when it cannot run (R-113-04). */
function DetectorChoice({
  value,
  current,
  onChange,
  title,
  note,
  detector,
  weights = [],
  installed = true,
  install = "",
  onDownload,
}: {
  value: DetectorKey;
  current: DetectorKey;
  onChange: (value: DetectorKey) => void;
  title?: string;
  note?: string;
  detector?: DetectorOut;
  weights?: WeightOut[];
  installed?: boolean;
  install?: string;
  onDownload?: (w: WeightOut) => void;
}) {
  const id = useId();
  const disabled = detector ? !detector.available : false;
  const level = detector?.level === "advanced" ? t("vision.model.levelAdvanced") : detector?.level === "more" ? t("vision.more.level") : detector ? t("vision.model.levelFirst") : null;
  return (
    <div className={`flex gap-3 rounded-md px-2 py-2 ${current === value ? "bg-surface-2" : ""}`}>
      <input
        id={id}
        type="radio"
        name="vision-detector"
        aria-label={title ?? detector?.title}
        className="mt-0.5 size-3.5 shrink-0 accent-(--accent) disabled:opacity-50"
        checked={current === value}
        disabled={disabled}
        onChange={() => onChange(value)}
      />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <label htmlFor={id} className={`flex flex-wrap items-baseline gap-x-2 text-[13px] ${disabled ? "text-ink-3" : "cursor-pointer text-ink-1"}`}>
          <span className="font-medium">{title ?? detector?.title}</span>
          {level && <span className="text-xs font-normal text-ink-3">{level}</span>}
          {detector?.note && <InfoTip label={t("vision.model.about", { title: detector.title })} text={detector.note} />}
        </label>
        {note && <p className="text-xs text-pretty text-ink-3">{note}</p>}
        {detector && !detector.available && detector.reason && <p className="text-xs text-pretty text-ink-2">{detector.reason}</p>}
        {detector && weights.length > 0 && onDownload && <WeightsLine weights={weights} installed={installed} install={install} onDownload={onDownload} />}
      </div>
    </div>
  );
}

const METRIC_LABEL: Record<string, MessageKey> = {
  balanced_accuracy: "vision.results.m.balanced_accuracy",
  accuracy: "vision.results.m.accuracy",
  auroc: "vision.results.m.auroc",
  r2: "vision.target.m.r2",
  macro_f1: "vision.target.m.macroF1",
};

function RunList({ datasetId, runs, onOpen }: { datasetId: string; runs: RunSummary[]; onOpen: (runId: string) => void }) {
  const remove = useDeleteRun(datasetId);
  if (runs.length === 0) return null;
  return (
    <section aria-labelledby="vision-runs">
      <h3 id="vision-runs" className="text-[13px] font-medium text-ink-1">
        {t("vision.model.runs", { count: formatCount(runs.length) })}
      </h3>
      <ul className="mt-2 divide-y divide-line border-y border-line">
        {runs.map((run) => (
          <li key={run.id} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-4 gap-y-1 py-2.5 text-[13px] sm:grid-cols-[minmax(0,1fr)_8rem_7rem_auto]">
            <div className="min-w-0">
              <p className="truncate text-ink-1">{run.default_title ?? t(TASK_LABEL[run.task])}</p>
              <p className="text-xs text-ink-3">
                {formatRelativeTime(run.created_at)} · {t(TASK_LABEL[run.task])} · {t("vision.model.runScored", { count: formatCount(run.scored) })}
              </p>
            </div>
            <p className="text-right tabular-nums max-sm:hidden">
              {run.metric && run.score !== null && run.score !== undefined ? (
                <>
                  <span className="text-ink-1">{metric(run.score)}</span>{" "}
                  <span className="text-xs text-ink-3">{METRIC_LABEL[run.metric] ? t(METRIC_LABEL[run.metric] as MessageKey) : run.metric}</span>
                </>
              ) : (
                <span className="text-ink-3">—</span>
              )}
            </p>
            <p className="text-right text-xs text-ink-3 tabular-nums max-sm:hidden">{t("vision.model.seconds", { seconds: run.seconds.toFixed(0) })}</p>
            <div className="flex items-center justify-end gap-1">
              <Button variant="secondary" onClick={() => onOpen(run.id)} className="h-7 text-xs">
                {t("vision.model.open")}
              </Button>
              <Button variant="ghost" iconOnly icon={Trash2} aria-label={t("vision.model.delete")} title={t("vision.model.delete")} onClick={() => remove.mutate(run.id)} />
            </div>
          </li>
        ))}
      </ul>
    </section>
  );
}
