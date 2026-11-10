import { useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import type { MlCandidate, MlChoicesInput, MlClasses, MlContext, MlPanelCheck, MlRun, MlValidation } from "../../api/client";
import {
  useDataset,
  useMlCatalog,
  useMlCancel,
  useMlDistribution,
  useMlEstimate,
  useMlJob,
  useMlPanel,
  useMlRun,
  useMlRuns,
  useMlTargets,
  useMlTrain,
} from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { DecisionsBanner } from "../../features/decisions/DecisionsBanner";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { classCounts, DEFAULT_SETUP, edgesFor, namesFor, type Setup } from "./bins";
import { ClassesPanel, setupFromRun, type Cut } from "./ClassesPanel";
import { AutoPanel } from "./AutoPanel";
import { CompareModels } from "./CompareModels";
import { BrowserNote } from "./desktopOnly";
import { ModelPicker } from "./ModelPicker";
import { incomparable, RunCompare } from "./RunCompare";
import { TunePanel } from "./TunePanel";
import { PreparationEditor, preparationSummary } from "./PreparationEditor";
import { disabledOptions, isVisible, resolveValues, type Values } from "./rules";
import { DEFAULT_VALIDATION, metricName, PANEL, panelValidation, SettingField, SettingsDrawer } from "./SettingsDrawer";
import { PredictSection } from "./Predict";
import { Results } from "./Results";
import { Loading } from "../../ui/Loading";

export type MlSearch = { target?: string; run?: string; prediction?: string };

export function validateMlSearch(search: Record<string, unknown>): MlSearch {
  const text = (value: unknown) => (typeof value === "string" && value ? value : undefined);
  return { target: text(search.target), run: text(search.run), prediction: text(search.prediction) };
}

const taskName = (task: string) => t(`ml.task.${task}` as MessageKey);

export function MlPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const search = useSearch({ strict: false }) as MlSearch;
  const navigate = useNavigate();
  const set = (patch: MlSearch) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, ...patch }), replace: true });
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const targets = useMlTargets(datasetId, ready);
  const candidate = targets.data?.candidates.find((c) => c.name === search.target && c.task);

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <header className="max-w-prose">
        <h1 className="text-xl font-semibold tracking-tight">{t("ml.title")}</h1>
        <p className="mt-1 text-ink-2">{t("ml.description")}</p>
      </header>
      <DecisionsBanner datasetId={datasetId} ready={ready} />
      {targets.isError ? (
        <ErrorNote>{targets.error.message}</ErrorNote>
      ) : !targets.data ? (
        <Loading label={t("loading.targets")} className="h-72 rounded-lg" />
      ) : !candidate ? (
        <TargetPicker candidates={targets.data.candidates} onPick={(name) => set({ target: name, run: undefined })} />
      ) : (
        <Workbench
          key={candidate.name}
          datasetId={datasetId}
          candidate={candidate}
          candidates={targets.data.candidates}
          rows={targets.data.rows}
          runId={search.run}
          predictionId={search.prediction}
          onTarget={(name) => set({ target: name, run: undefined, prediction: undefined })}
          onRun={(run) => set({ run, prediction: undefined })}
          onPrediction={(prediction) => set({ prediction })}
        />
      )}
    </div>
  );
}

function TargetPicker({ candidates, onPick }: { candidates: MlCandidate[]; onPick: (name: string) => void }) {
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const shown = candidates.filter((c) => !needle || c.name.toLowerCase().includes(needle));
  return (
    <section className="mx-auto w-full max-w-2xl space-y-3 rounded-lg border border-line bg-surface-1 p-5" aria-labelledby="ml-question">
      <h2 id="ml-question" className="text-lg font-semibold tracking-tight">
        {t("ml.question")}
      </h2>
      <p className="text-[13px] text-ink-2">{t("ml.questionHint")}</p>
      <div className="relative">
        <Search size={14} strokeWidth={1.5} className="pointer-events-none absolute top-1/2 left-2 -translate-y-1/2 text-ink-3" aria-hidden />
        <TextInput aria-label={t("ml.searchColumns")} placeholder={t("ml.searchColumns")} value={query} onChange={(e) => setQuery(e.target.value)} className="w-full pl-7" />
      </div>
      <ul className="max-h-[420px] divide-y divide-line overflow-y-auto rounded-md border border-line">
        {shown.map((c) => (
          <li key={c.name}>
            <button
              type="button"
              disabled={!c.task}
              onClick={() => onPick(c.name)}
              className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:bg-transparent"
            >
              <span className="min-w-0">
                <span className="font-mono text-xs font-medium text-ink-1">{c.name}</span>
                {c.suggested && c.task && (
                  <span className="ml-2 rounded-full bg-accent-soft px-1.5 py-0.5 text-[10px] text-ink-1">{t("ml.suggested")}</span>
                )}
                {c.reason && <span className="block text-xs text-ink-3">{c.reason}</span>}
              </span>
              {c.task && <span className="shrink-0 text-xs text-ink-2">{taskName(c.task)}</span>}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function Workbench({
  datasetId,
  candidate,
  candidates,
  rows,
  runId,
  predictionId,
  onTarget,
  onRun,
  onPrediction,
}: {
  datasetId: string;
  candidate: MlCandidate;
  candidates: MlCandidate[];
  rows: number;
  runId: string | undefined;
  predictionId: string | undefined;
  onTarget: (name: string) => void;
  onRun: (id: string) => void;
  onPrediction: (id: string | undefined) => void;
}) {
  const client = useQueryClient();
  const task = candidate.task ?? "regression";
  const runs = useMlRuns(datasetId, candidate.name);
  const shownRunId = runId ?? runs.data?.[0]?.id ?? null;
  const run = useMlRun(datasetId, shownRunId);

  // A number can be predicted as ordered classes (018): mode and cut follow the shown run until
  // the user changes them.
  const numeric = task === "regression";
  const fewValues = candidate.type === "numeric_discrete" && (candidate.classes ?? Infinity) <= 10;
  const runClasses = run.data?.classes ?? null;
  const [modeEdited, setModeEdited] = useState<"number" | "classes" | null>(null);
  const [setupEdited, setSetupEdited] = useState<Setup | null>(null);
  const mode = numeric ? (modeEdited ?? (runClasses ? "classes" : "number")) : "number";
  const distribution = useMlDistribution(datasetId, candidate.name, numeric && mode === "classes");
  const setup = useMemo<Setup>(
    () => setupEdited ?? (runClasses ? setupFromRun(runClasses) : fewValues ? { ...DEFAULT_SETUP, method: "values" } : DEFAULT_SETUP),
    [setupEdited, runClasses, fewValues],
  );
  const cut = useMemo<Cut | null>(() => {
    const shape = distribution.data;
    if (!shape || shape.rows === 0) return null;
    const edges = edgesFor(setup, shape);
    return { edges, names: namesFor(setup, edges.length + 1, shape, formatNumber), counts: classCounts(shape, edges) };
  }, [distribution.data, setup]);
  const classes = useMemo<MlClasses | null>(
    () => (mode === "classes" && cut && cut.names.length >= 2 ? { method: setup.method, edges: cut.edges, names: cut.names } : null),
    [mode, cut, setup.method],
  );
  const effectiveTask = mode === "classes" ? (cut?.names.length === 2 ? "binary" : "multiclass") : task;

  // Validation: what the user changed, else what the shown run used.
  const [validationEdited, setValidationEdited] = useState<MlValidation | null>(null);
  const validation = validationEdited ?? run.data?.validation ?? DEFAULT_VALIDATION;
  // Panel data (035): recognised, or checked with the columns the user declared.
  const panelMode = validation.method === PANEL;
  const panel = useMlPanel(datasetId, candidate.name, panelMode ? validation.period_column : null, panelMode ? validation.entity_column : null);
  const context = useMemo<MlContext>(() => {
    const others = candidates.filter((c) => c.name !== candidate.name);
    const categorical = others.filter((c) => ["categorical_nominal", "categorical_ordinal", "boolean"].includes(c.type)).length;
    const used = Math.min(rows, validation.max_rows ?? rows);
    const cross = ["kfold", "repeated_kfold", "group_kfold", "loo", PANEL].includes(validation.method);
    return {
      task: effectiveTask,
      rows: cross ? used : used * (1 - validation.test_size),
      minority: null,
      features: others.length,
      categoricalShare: others.length ? categorical / others.length : 0,
      missingShare: others.length ? others.reduce((sum, c) => sum + c.missing_share, 0) / others.length : 0,
    };
  }, [candidates, candidate.name, rows, validation.max_rows, validation.method, validation.test_size, effectiveTask]);
  const catalog = useMlCatalog(context);
  const reference = effectiveTask === "regression" ? "linear_regression" : "logistic_regression";
  const [modelEdited, setModelEdited] = useState<string | null>(null);
  const runModel = run.data && run.data.task === effectiveTask ? run.data.model : null;
  const models = catalog.data?.models;
  const model = models?.find((m) => m.key === (modelEdited ?? runModel ?? reference)) ?? models?.find((m) => m.key === reference);
  const [values, setValues] = useState<Values>({});
  const [drawer, setDrawer] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [automl, setAutoml] = useState(false);
  const [autoJob, setAutoJob] = useState<string | null>(null);
  const [tuning, setTuning] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const train = useMlTrain(datasetId);
  const cancel = useMlCancel();
  const estimate = useMlEstimate(datasetId);
  const job = useMlJob(jobId);
  const started = useRef(false);
  const [editing, setEditing] = useState(false);
  // Preparation choices: what the user changed, else what the shown run used.
  const [edited, setEdited] = useState<MlChoicesInput | null>(null);
  const runChoices = run.data?.preparation.choices;
  const choices = useMemo<MlChoicesInput>(() => edited ?? runChoices ?? {}, [edited, runChoices]);

  const runSettings = run.data?.model === model?.key ? run.data?.settings : undefined;
  // Shown: what the user changed, else what the shown run used, else the data-dependent defaults;
  // then the catalogue's rules, as the server applies them.
  const resolved = useMemo(
    () => (model ? resolveValues(model, runSettings ?? {}, values) : { values: {} as Values, error: null }),
    [model, runSettings, values],
  );
  const settings = resolved.values;
  const resetSetting = (key?: string) => {
    if (key === undefined) return setValues({});
    const rest = { ...values };
    delete rest[key];
    setValues(rest);
  };

  // Only the settings the user changed (for this model): the server resolves the other defaults.
  const changed = Object.fromEntries(Object.entries(values).filter(([key]) => model?.settings.some((s) => s.key === key)));
  const changedSettings = Object.keys(changed).length ? changed : null;
  const launch = () =>
    train.mutate(
      { target: candidate.name, model: model?.key ?? null, settings: changedSettings, choices, classes, validation },
      {
        onSuccess: (created) => {
          setJobId(created.id);
          setEdited(null);
          setDrawer(false);
        },
      },
    );
  const pickRun = (id: string) => {
    setEdited(null);
    setModelEdited(null);
    setValues({});
    setValidationEdited(null);
    setModeEdited(null);
    setSetupEdited(null);
    onRun(id);
  };
  const switchMode = (next: "number" | "classes") => {
    const existing = runs.data?.find((r) => (next === "classes") === Boolean(r.classes));
    if (existing && !(next === "classes" && setupEdited)) return pickRun(existing.id);
    setModeEdited(next);
    if (next === "number")
      train.mutate({ target: candidate.name, model: null, settings: null, choices }, { onSuccess: (created) => setJobId(created.id) });
  };

  // Classes retrain on their own 600 ms after the last change (not at every click).
  const payload = classes ? JSON.stringify(classes) : null;
  const shownPayload = runClasses ? JSON.stringify({ method: runClasses.method, edges: runClasses.edges, names: runClasses.names }) : null;
  const sent = useRef<string | null>(null);
  const modelReady = Boolean(model?.tasks.includes(effectiveTask));
  useEffect(() => {
    if (!payload || !classes || !modelReady || payload === shownPayload || payload === sent.current) return;
    const timer = window.setTimeout(() => {
      sent.current = payload;
      train.mutate(
        { target: candidate.name, model: model?.key ?? null, settings: changedSettings, choices, classes, validation },
        { onSuccess: (created) => setJobId(created.id) },
      );
    }, 600);
    return () => window.clearTimeout(timer);
  }, [payload, shownPayload, modelReady, classes, candidate.name, model, choices, train, changedSettings, validation]);

  // The reference model trains on its own the first time a target is chosen.
  useEffect(() => {
    if (runs.data && runs.data.length === 0 && !started.current) {
      started.current = true;
      train.mutate({ target: candidate.name, model: null, settings: null }, { onSuccess: (created) => setJobId(created.id) });
    }
  }, [runs.data, candidate.name, train]);

  // When a training ends, show its run and refresh the list.
  const finished = job.data?.status === "done" ? job.data.run_id : null;
  useEffect(() => {
    if (finished) {
      void client.invalidateQueries({ queryKey: ["datasets", datasetId, "ml", "runs", candidate.name] });
      onRun(finished);
    }
  }, [finished, client, datasetId, candidate.name, onRun]);

  const training = train.isPending || job.data?.status === "running";
  const preparation = run.data?.preparation;

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_minmax(0,1fr)]">
      <aside
        className="space-y-5 lg:sticky lg:top-4 lg:max-h-[calc(100dvh-2rem)] lg:self-start lg:overflow-y-auto lg:pr-2"
        aria-label={t("ml.settings")}
      >
        <section className="space-y-2">
          <label className="flex items-center justify-between gap-3">
            <span className="text-xs text-ink-3">{t("ml.predict")}</span>
            <Select aria-label={t("ml.predict")} value={candidate.name} onChange={(event) => onTarget(event.target.value)} className="h-8 max-w-48 text-[13px]">
              {candidates
                .filter((c) => c.task)
                .map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name}
                  </option>
                ))}
            </Select>
          </label>
          {numeric && (
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-ink-3">{t("ml.predictAs")}</span>
              <SegmentedControl
                label={t("ml.predictAs")}
                value={mode}
                options={[
                  { value: "number", label: t("ml.predictAs.number") },
                  { value: "classes", label: t("ml.predictAs.classes") },
                ]}
                onChange={switchMode}
              />
            </div>
          )}
          <p className="text-xs text-ink-2">{t(`ml.taskLong.${effectiveTask}` as MessageKey)}</p>
          {mode === "number" && fewValues ? (
            <p className="flex flex-wrap items-center gap-2 text-xs text-ink-2">
              {t("ml.classes.suggest")}
              <Button
                variant="ghost"
                onClick={() => {
                  setSetupEdited({ ...DEFAULT_SETUP, method: "values" });
                  setModeEdited("classes");
                }}
              >
                {t("ml.classes.suggestAction")}
              </Button>
            </p>
          ) : (
            candidate.reason && mode === "number" && <p className="text-xs text-ink-3">{candidate.reason}</p>
          )}
          {mode === "classes" &&
            (distribution.data && cut ? (
              <ClassesPanel target={candidate.name} shape={distribution.data} setup={setup} cut={cut} onSetup={setSetupEdited} />
            ) : distribution.isError ? (
              <ErrorNote>{distribution.error.message}</ErrorNote>
            ) : (
              <Loading label={t("loading.distribution")} className="h-40 rounded-md" />
            ))}
        </section>
        {catalog.data && model && (
          <section className="space-y-2">
            <span className="text-xs text-ink-3">{t("ml.model")}</span>
            <ModelPicker
              catalogue={catalog.data}
              current={model}
              onPick={(key) => {
                setModelEdited(key);
                setValues({});
              }}
            />
            <p className="text-xs text-ink-3">{model.description}</p>
            {model.settings
              .filter((s) => s.level === "essential" && isVisible(s, settings))
              .map((s) => (
                <SettingField
                  key={s.key}
                  setting={s}
                  value={settings[s.key]}
                  changed={s.key in values}
                  disabled={disabledOptions(model, settings)[s.key]}
                  onChange={(value) => setValues({ ...values, [s.key]: value })}
                  onReset={() => resetSetting(s.key)}
                />
              ))}
            <span className="flex flex-wrap gap-1">
              <Button variant="ghost" onClick={() => setDrawer(true)}>
                {t("ml.settings.all")}
              </Button>
              {/* 102: tuning (Optuna) is left to the desktop app in the browser version (R-WEB-04) */}
              {catalog.data.browser?.tuning !== false && (
                <Button
                  variant="ghost"
                  onClick={() => setTuning(true)}
                  disabled={model.space.length === 0 || training}
                  title={model.space.length === 0 ? t("ml.tune.nothing") : undefined}
                >
                  {t("ml.tune.button")}
                </Button>
              )}
              <Button variant="ghost" onClick={() => setComparing(true)} disabled={training}>
                {t("ml.compare.button")}
              </Button>
              <Button variant="ghost" onClick={() => setAutoml(true)} disabled={training} title={t("ml.auto.buttonHint")}>
                {t("ml.auto.button")}
              </Button>
            </span>
            <BrowserNote catalogue={catalog.data} />
          </section>
        )}
        {catalog.data && comparing && (
          <CompareModels
            open={comparing}
            onOpenChange={setComparing}
            datasetId={datasetId}
            target={candidate.name}
            catalogue={catalog.data}
            choices={choices}
            classes={classes}
            validation={validation}
            onRun={pickRun}
          />
        )}
        {catalog.data && automl && (
          <AutoPanel
            open={automl}
            onOpenChange={setAutoml}
            datasetId={datasetId}
            target={candidate.name}
            catalogue={catalog.data}
            choices={choices}
            classes={classes}
            validation={validation}
            jobId={autoJob}
            onJob={setAutoJob}
            onRun={pickRun}
          />
        )}
        {model && tuning && (
          <TunePanel
            key={model.key}
            open={tuning}
            onOpenChange={setTuning}
            datasetId={datasetId}
            target={candidate.name}
            model={model}
            settings={settings}
            defaults={Object.fromEntries(model.settings.map((s) => [s.key, s.default]))}
            choices={choices}
            classes={classes}
            validation={validation}
            onUse={(found) => {
              const chosen = Object.fromEntries(Object.entries(found).filter(([key]) => model.settings.some((s) => s.key === key)));
              setValues(chosen);
              train.mutate(
                { target: candidate.name, model: model.key, settings: chosen, choices, classes, validation },
                { onSuccess: (created) => setJobId(created.id) },
              );
            }}
          />
        )}
        <section className="space-y-1 text-xs">
          <div className="flex items-center justify-between gap-2">
            <p className="text-ink-3">
              {t("ml.preparation")} · {(edited ? Object.keys(edited).length > 0 : preparation?.custom) ? t("ml.prep.custom") : t("ml.prep.automatic")}
            </p>
            <Button variant="ghost" onClick={() => setEditing(true)} disabled={!model}>
              {t("ml.prep.edit")}
            </Button>
          </div>
          <p className="text-ink-1">{preparation ? preparationSummary(preparation) : t("ml.preparationAuto")}</p>
          {edited && <p className="text-warn">{t("ml.prep.pending")}</p>}
          <p className="flex items-center justify-between gap-2 pt-1 text-ink-3">
            <span>{validationSummary(validation)}</span>
            <Button variant="ghost" aria-label={t("ml.validation.edit")} onClick={() => setDrawer(true)}>
              {t("ml.prep.edit")}
            </Button>
          </p>
        </section>
        {model && (
          <SettingsDrawer
            open={drawer}
            onOpenChange={setDrawer}
            model={model}
            values={settings}
            chosen={values}
            onChange={(key, value) => setValues({ ...values, [key]: value })}
            onReset={resetSetting}
            validation={validation}
            onValidation={setValidationEdited}
            metrics={catalog.data?.metrics ?? []}
            candidates={candidates}
            task={effectiveTask}
            onPreparation={() => {
              setDrawer(false);
              setEditing(true);
            }}
            onTrain={launch}
            training={training}
            error={resolved.error}
            panel={panel.data}
          />
        )}
        <PreparationEditor
          open={editing}
          onOpenChange={setEditing}
          datasetId={datasetId}
          target={candidate.name}
          task={task}
          model={model?.key ?? null}
          settings={changedSettings}
          choices={choices}
          onChoices={setEdited}
          classes={classes}
          validation={panelMode ? validation : null}
        />
        {resolved.error && <ErrorNote>{resolved.error}</ErrorNote>}
        {model && (model.badges.some((b) => b.startsWith("Slow")) || context.rows > 50_000 || validation.method !== "split") && (
          <p className="flex flex-wrap items-center gap-2 text-xs text-ink-2">
            <Button
              variant="ghost"
              disabled={estimate.isPending || training}
              onClick={() =>
                estimate.mutate({ target: candidate.name, model: model.key, settings: changedSettings, choices, classes, validation })
              }
            >
              {estimate.isPending ? t("ml.estimate.running") : t("ml.estimate.button")}
            </Button>
            {estimate.data && <span>{t("ml.estimate.result", { duration: duration(estimate.data.seconds), fits: `${estimate.data.fits} fit${estimate.data.fits > 1 ? "s" : ""}` })}</span>}
            {estimate.isError && <span className="text-bad">{estimate.error.message}</span>}
          </p>
        )}
        <span className="flex gap-2">
          <Button variant="primary" onClick={launch} disabled={training || !model || Boolean(resolved.error)}>
            {training ? t(`ml.stage.${job.data?.stage ?? "loading"}` as MessageKey) : t("ml.train")}
          </Button>
          {training && jobId && (
            <Button onClick={() => cancel.mutate(jobId)} disabled={cancel.isPending}>
              {t("ml.cancel")}
            </Button>
          )}
        </span>
        {job.data?.status === "cancelled" && <p className="text-xs text-ink-3">{job.data.error}</p>}
        {train.isError && <ErrorNote>{train.error.message}</ErrorNote>}
        {job.data?.status === "failed" && <ErrorNote>{job.data.error}</ErrorNote>}
        {runs.data && runs.data.length > 0 && (
          <RunStrip
            runs={runs.data}
            shown={run.data}
            onPick={pickRun}
            labels={Object.fromEntries((models ?? []).map((m) => [m.key, m.label]))}
            selected={selected}
            onSelect={setSelected}
          />
        )}
      </aside>

      <div className={`min-w-0 space-y-5 ${training ? "opacity-60" : ""}`} aria-busy={training}>
        {panel.data?.panel && (
          <PanelBanner
            check={panel.data}
            validation={validation}
            onUse={() => setValidationEdited(panelValidation(validation, panel.data))}
            onEdit={() => setDrawer(true)}
          />
        )}
        {run.data ? (
          <>
            <Results run={run.data} datasetId={datasetId} />
            <PredictSection run={run.data} datasetId={datasetId} predictionId={predictionId} onPrediction={onPrediction} />
          </>
        ) : (
          <div className="flex h-72 items-center justify-center rounded-lg border border-dashed border-line text-ink-3">
            {training ? t("ml.trainingFirst") : t("ml.noRunYet")}
          </div>
        )}
        {runs.data && selected.length >= 2 && (
          <RunCompare datasetId={datasetId} summaries={runs.data.filter((r) => selected.includes(r.id))} />
        )}
      </div>
    </div>
  );
}

/** 75 → "1 min 15 s". */
function duration(seconds: number): string {
  if (seconds < 1) return "< 1 s";
  if (seconds < 60) return `${Math.round(seconds)} s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes} min ${Math.round(seconds % 60)} s`;
  return `${Math.floor(minutes / 60)} h ${minutes % 60} min`;
}

/** "This looks like panel data…" with the way to validate by period; under a random split, why
 *  its score is too good (the same entities and pairs on both sides). */
function PanelBanner({ check, validation, onUse, onEdit }: { check: MlPanelCheck; validation: MlValidation; onUse: () => void; onEdit: () => void }) {
  if (validation.method === PANEL) return null;
  const random = ["split", "kfold", "repeated_kfold", "loo"].includes(validation.method);
  const period = check.period ?? "";
  const entity = check.entity;
  return (
    <section className="space-y-1.5 rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-[13px] text-ink-1" aria-label={t("ml.validation.method.panel_walk_forward")}>
      <p className="flex flex-wrap items-center gap-2">
        <span className="min-w-0 flex-1">{check.message}</span>
        <Button variant="primary" onClick={onUse}>
          {t("ml.panel.use")}
        </Button>
        <Button variant="ghost" onClick={onEdit}>
          {t("ml.panel.edit")}
        </Button>
      </p>
      {random && entity && (validation.method !== "group_kfold" || validation.group !== entity) && (
        <p className="text-xs text-ink-2">{t("ml.panel.bothSides", { entities: `${entity}s` })}</p>
      )}
      {random && entity && check.duplicates > 0 && (
        <p className="text-xs text-ink-2">{t("ml.panel.duplicates", { count: check.duplicates.toLocaleString("en-US"), entity, period })}</p>
      )}
    </section>
  );
}

function validationSummary(validation: MlValidation): string {
  const method = t(`ml.validation.method.${validation.method}` as MessageKey);
  const detail =
    validation.method === "split" || validation.method === "time_split"
      ? ` · ${Math.round(validation.test_size * 100)} % test`
      : validation.method === "loo"
        ? ""
        : ` · ${validation.folds} folds`;
  const panel =
    validation.method === PANEL
      ? t("ml.validation.panelSummary", { period: validation.period_column ?? "—", embargo: String(validation.embargo) })
      : "";
  const rows = validation.max_rows === null ? t("ml.rows.all") : t("ml.rows.atMost", { rows: validation.max_rows.toLocaleString("en-US") });
  return `${t("ml.validationLabel")}: ${method}${detail}${panel} · ${rows}${validation.metric ? ` · ${metricName(validation.metric)}` : ""}`;
}

function RunStrip({
  runs,
  shown,
  onPick,
  labels,
  selected,
  onSelect,
}: {
  runs: NonNullable<ReturnType<typeof useMlRuns>["data"]>;
  shown: MlRun | undefined;
  onPick: (id: string) => void;
  labels: Record<string, string>;
  selected: string[];
  onSelect: (ids: string[]) => void;
}) {
  const first = runs.find((r) => r.id === selected[0]);
  // Only runs scored the same way compare (R² and accuracy do not).
  const metric = shown?.primary ?? runs[0]?.metric;
  const comparable = runs.filter((r) => r.score !== null && r.metric === metric);
  const best =
    comparable.length > 1
      ? [...comparable].sort((a, b) => (a.greater_is_better ? (b.score ?? 0) - (a.score ?? 0) : (a.score ?? 0) - (b.score ?? 0)))[0]
      : undefined;
  return (
    <section aria-label={t("ml.runs")} className="space-y-1.5 border-t border-line pt-4">
      <h3 className="text-xs text-ink-3">{t("ml.runs")}</h3>
      <div className="space-y-0.5">
        {runs.map((r, index) => {
          const reason = first && first.id !== r.id ? incomparable(first, r) : null;
          return (
            <span key={r.id} className="flex items-start gap-1.5">
              <input
                type="checkbox"
                className="mt-2 size-3.5 accent-(--accent)"
                aria-label={t("ml.runCompare.pick", { run: String(runs.length - index) })}
                title={reason ?? t("ml.runCompare.pickHint")}
                disabled={Boolean(reason)}
                checked={selected.includes(r.id)}
                onChange={(e) => onSelect(e.target.checked ? [...selected, r.id] : selected.filter((id) => id !== r.id))}
              />
              <button
                type="button"
                onClick={() => onPick(r.id)}
                aria-pressed={shown?.id === r.id}
                className={`min-w-0 flex-1 rounded-md border px-2 py-1.5 text-left text-xs ${shown?.id === r.id ? "border-accent bg-accent-soft" : "border-transparent hover:bg-surface-1"}`}
              >
                <span className="block truncate text-ink-1">
                  #{runs.length - index} {r.label || labels[r.model] || r.model}
                  {typeof r.settings.regularization === "string" && r.settings.regularization !== "none"
                    ? ` · ${t(`ml.option.${r.settings.regularization}` as MessageKey)}`
                    : ""}
                  {r.classes ? ` · ${t("ml.classes.runLabel", { count: String(r.classes.names.length) })}` : ""}
                  {best?.id === r.id ? " ★" : ""}
                </span>
                <span className="block text-ink-3 tabular-nums">
                  {t(`ml.metric.${r.metric}` as MessageKey)} {formatNumber(r.score)} · {r.seconds.toFixed(1)} s
                </span>
              </button>
            </span>
          );
        })}
      </div>
    </section>
  );
}
