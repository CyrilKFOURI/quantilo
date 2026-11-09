import { useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { ChevronDown, ChevronRight, Cpu, History, Play, Square, Trash2, Zap } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import type {
  SyntheticColumnPlan,
  SyntheticMethod,
  SyntheticPlan,
  SyntheticRequest,
  SyntheticRunSummary,
  SyntheticTreatment,
} from "../../api/client";
import {
  useDataset,
  useSyntheticActions,
  useSyntheticCancel,
  useSyntheticGenerate,
  useSyntheticJob,
  useSyntheticPlan,
  useSyntheticRun,
  useSyntheticRuns,
} from "../../api/hooks";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Checkbox, ErrorNote, ProgressBar, Select, TextInput } from "../../ui/fields";
import { SettingsBar } from "../../ui/ParamPanel";
import { Popover } from "../../ui/Popover";
import { pct } from "./format";
import { Report } from "./Report";
import { RunCompare } from "./RunCompare";
import { SequenceSettings, type SequenceDraft } from "./SequencesTab";
import type { SyntheticSearch, Tab } from "./search";
import { Loading } from "../../ui/Loading";

const small = "h-7 text-xs";
const STAGES = ["loading", "generating", "evaluating", "saving"] as const;

type Draft = Omit<SyntheticRequest, "sequence"> & {
  sequence?: SequenceDraft;
  treatments: Record<string, SyntheticTreatment>;
  bounds: Record<string, { minimum?: number | null; maximum?: number | null }>;
};
const EMPTY: Draft = { treatments: {}, bounds: {} };

function Group({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5 max-sm:w-full" role="group" aria-label={label}>
      <span className="text-ink-3 max-sm:basis-full">{label}</span>
      {children}
    </div>
  );
}

const Divider = () => <span className="hidden h-5 w-px bg-line sm:block" aria-hidden />;

/** "NVIDIA GeForce RTX 4050 Laptop GPU" → "RTX 4050". */
const shortGpu = (name: string) => /(RTX|GTX|RX|Arc)\s*\w+/i.exec(name)?.[0] ?? name.replace(/^NVIDIA (GeForce )?/, "");

const treatmentLabel = (value: SyntheticTreatment) => t(`synthetic.treatment.${value}` as MessageKey);

/** The plan of each column: what happens to it, the rule and why; bounds for numbers. */
function PlanTable({
  plan,
  draft,
  onChange,
}: {
  plan: SyntheticPlan;
  draft: Draft;
  onChange: (next: Draft) => void;
}) {
  const treatmentOf = (c: SyntheticColumnPlan) => draft.treatments[c.name] ?? c.treatment;
  const setTreatment = (c: SyntheticColumnPlan, value: SyntheticTreatment) => {
    const treatments = { ...draft.treatments };
    if (value === c.treatment) delete treatments[c.name];
    else treatments[c.name] = value;
    onChange({ ...draft, treatments });
  };
  const setBound = (name: string, side: "minimum" | "maximum", raw: string) => {
    const value = raw === "" ? null : Number(raw);
    const bound = { ...draft.bounds[name], [side]: Number.isFinite(value) ? value : null };
    onChange({ ...draft, bounds: { ...draft.bounds, [name]: bound } });
  };
  return (
    // 041: on a phone each column is a stacked block: name and treatment, then bounds, then why.
    <table className="w-full text-[13px] max-sm:block max-sm:[&_tbody]:block max-sm:[&_thead]:hidden max-sm:[&_td]:p-0 max-sm:[&_td:nth-child(n+3)]:col-span-2 max-sm:[&_tr]:grid max-sm:[&_tr]:grid-cols-[minmax(0,1fr)_auto] max-sm:[&_tr]:items-center max-sm:[&_tr]:gap-x-3 max-sm:[&_tr]:gap-y-1.5 max-sm:[&_tr]:py-2">
      <caption className="sr-only">{t("synthetic.columns")}</caption>
      <thead>
        <tr className="border-b border-line text-left text-xs text-ink-3">
          <th className="py-1.5 pr-3 font-normal">{t("synthetic.column")}</th>
          <th className="py-1.5 pr-3 font-normal">{t("synthetic.treatment")}</th>
          <th className="py-1.5 pr-3 font-normal">{t("synthetic.bounds")}</th>
          <th className="py-1.5 font-normal">{t("synthetic.why")}</th>
        </tr>
      </thead>
      <tbody>
        {plan.columns.map((c) => {
          const treatment = treatmentOf(c);
          const changed = treatment !== c.treatment;
          const numeric = treatment === "model" && (c.type === "numeric_continuous" || c.type === "numeric_discrete");
          return (
            <tr key={c.name} className={`border-b border-line/60 ${treatment === "leave" ? "text-ink-3" : ""}`}>
              <td className="py-1.5 pr-3">
                <span className="font-medium text-ink-1">{c.name}</span>{" "}
                <span className="text-xs text-ink-3">{t(`type.${c.type}` as MessageKey)}</span>
              </td>
              <td className="py-1.5 pr-3">
                {c.options.length > 1 ? (
                  <Select
                    aria-label={t("synthetic.treatmentOf", { column: c.name })}
                    value={treatment}
                    onChange={(e) => setTreatment(c, e.target.value as SyntheticTreatment)}
                    className={`${small} ${changed ? "border-accent" : ""}`}
                  >
                    {[c.treatment, ...c.options.filter((o) => o !== c.treatment)].map((o) => (
                      <option key={o} value={o}>
                        {treatmentLabel(o)}
                        {o === c.treatment ? ` (${t("synthetic.auto")})` : ""}
                      </option>
                    ))}
                  </Select>
                ) : (
                  <span className="text-xs">{treatmentLabel(treatment)}</span>
                )}
              </td>
              <td className="py-1.5 pr-3">
                {numeric && (
                  <span className="inline-flex items-center gap-1 text-xs text-ink-3">
                    <TextInput
                      type="number"
                      aria-label={t("synthetic.minimumOf", { column: c.name })}
                      placeholder={t("synthetic.min")}
                      value={draft.bounds[c.name]?.minimum ?? ""}
                      onChange={(e) => setBound(c.name, "minimum", e.target.value)}
                      className={`${small} w-20`}
                    />
                    –
                    <TextInput
                      type="number"
                      aria-label={t("synthetic.maximumOf", { column: c.name })}
                      placeholder={t("synthetic.max")}
                      value={draft.bounds[c.name]?.maximum ?? ""}
                      onChange={(e) => setBound(c.name, "maximum", e.target.value)}
                      className={`${small} w-20`}
                    />
                  </span>
                )}
              </td>
              <td className="py-1.5 text-xs text-ink-2">
                <span className="mr-1.5 font-mono text-[11px] text-ink-3">{c.rule}</span>
                {changed ? t("synthetic.changedByYou") : c.reason}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Saved runs: open one, tick 2 to 5 to compare, delete. */
function RunsMenu({
  runs,
  current,
  compared,
  onOpen,
  onCompare,
  onDelete,
}: {
  runs: SyntheticRunSummary[];
  current: string | undefined;
  compared: string[];
  onOpen: (id: string) => void;
  onCompare: (ids: string[]) => void;
  onDelete: (run: SyntheticRunSummary) => void;
}) {
  const [open, setOpen] = useState(false);
  const [ticked, setTicked] = useState<string[]>(compared);
  const toggle = (id: string) =>
    setTicked(ticked.includes(id) ? ticked.filter((x) => x !== id) : [...ticked, id].slice(-5));
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("synthetic.runs")}
      trigger={
        <Button icon={History} onClick={() => setOpen(!open)} aria-expanded={open}>
          {t("synthetic.runs")}
          <span className="rounded-full bg-surface-2 px-1.5 text-[11px] text-ink-2">{runs.length}</span>
          <ChevronDown size={14} aria-hidden />
        </Button>
      }
    >
      <div className="max-h-96 overflow-auto p-1.5 text-[13px]">
        {runs.length === 0 && <p className="px-2 py-3 text-xs text-ink-3">{t("synthetic.noRuns")}</p>}
        <ul>
          {runs.map((run) => (
            <li key={run.id} className="group flex items-center gap-1 rounded-md px-1 hover:bg-surface-2">
              <Checkbox
                aria-label={t("synthetic.tickToCompare", { method: run.method_label })}
                checked={ticked.includes(run.id)}
                onChange={() => toggle(run.id)}
              />
              <button
                type="button"
                onClick={() => {
                  setOpen(false);
                  onOpen(run.id);
                }}
                aria-current={run.id === current ? "page" : undefined}
                className={`grid min-w-0 flex-1 grid-cols-[1fr_auto] gap-x-3 px-1.5 py-1.5 text-left ${run.id === current ? "font-medium text-ink-1" : "text-ink-2"}`}
              >
                <span className="truncate">
                  {run.method_label} · {run.rows.toLocaleString("en")} {t("synthetic.rowsShort")}
                </span>
                <span className="text-xs text-ink-3 tabular-nums">
                  {t("synthetic.quality")} {pct(run.scores.quality)}
                </span>
                <span className="col-span-2 text-[11px] font-normal text-ink-3">
                  {new Date(run.created_at).toLocaleString("en", { dateStyle: "medium", timeStyle: "short" })}
                  {run.created_dataset_id ? ` · ${t("synthetic.savedAsDataset")}` : ""}
                </span>
              </button>
              <button
                type="button"
                aria-label={t("synthetic.deleteRun")}
                title={t("synthetic.deleteRun")}
                onClick={() => {
                  setOpen(false);
                  onDelete(run);
                }}
                className="inline-flex size-7 shrink-0 items-center justify-center rounded text-ink-3 hover:text-ink-1"
              >
                <Trash2 size={13} aria-hidden />
              </button>
            </li>
          ))}
        </ul>
        {runs.length >= 2 && (
          <div className="flex items-center justify-between gap-2 border-t border-line px-2 pt-2 pb-1">
            <span className="text-xs text-ink-3">{t("synthetic.compareHint")}</span>
            <Button
              disabled={ticked.length < 2}
              onClick={() => {
                setOpen(false);
                onCompare(ticked);
              }}
            >
              {t("synthetic.compare")} {ticked.length >= 2 ? `(${ticked.length})` : ""}
            </Button>
          </div>
        )}
      </div>
    </Popover>
  );
}

function Progress({
  stage,
  share,
  message,
  onCancel,
  cancelling,
}: {
  stage: string;
  share: number | null;
  message: string | null;
  onCancel: () => void;
  cancelling: boolean;
}) {
  const index = STAGES.indexOf(stage as (typeof STAGES)[number]);
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 border-y border-line py-4" aria-live="polite">
      <ol className="flex flex-wrap items-center gap-x-1 gap-y-1 text-[13px]">
        {STAGES.map((s, i) => (
          <li key={s} className="flex items-center gap-1">
            {i > 0 && <ChevronRight size={12} className="text-ink-3" aria-hidden />}
            <span
              className={i === index ? "font-medium text-ink-1" : i < index ? "text-ink-2" : "text-ink-3"}
              aria-current={i === index ? "step" : undefined}
            >
              {i === index && <span className="mr-1.5 inline-block size-1.5 animate-pulse rounded-full bg-accent align-middle" aria-hidden />}
              {t(`synthetic.stage.${s}` as MessageKey)}
            </span>
          </li>
        ))}
      </ol>
      <Button icon={Square} variant="ghost" onClick={onCancel} disabled={cancelling}>
        {t("synthetic.cancel")}
      </Button>
      {share !== null && (
        <div className="flex w-full items-center gap-3">
          <div className="max-w-md flex-1">
            <ProgressBar value={share} label={t("synthetic.training")} />
          </div>
          <span className="text-xs text-ink-2 tabular-nums">{message}</span>
        </div>
      )}
    </div>
  );
}

export function SyntheticPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const search = useSearch({ strict: false }) as SyntheticSearch;
  const navigate = useNavigate();
  const client = useQueryClient();
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const plan = useSyntheticPlan(datasetId, ready);
  const runs = useSyntheticRuns(datasetId, ready);
  const actions = useSyntheticActions(datasetId);
  const generate = useSyntheticGenerate(datasetId);
  const cancel = useSyntheticCancel();
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useSyntheticJob(jobId);
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [columnsOpen, setColumnsOpen] = useState<boolean | null>(null);
  const [compared, setCompared] = useState<string[]>([]);

  const status = job.data?.status;
  const running = Boolean(jobId) && (status === "running" || !job.data);
  useEffect(() => {
    if (status === "done") void client.invalidateQueries({ queryKey: ["datasets", datasetId, "synthetic", "runs"] });
  }, [status, client, datasetId]);

  const finished = status === "done" ? (job.data?.run_id ?? undefined) : undefined;
  const runId = search.run ?? finished ?? runs.data?.[0]?.id;
  const run = useSyntheticRun(datasetId, runId);
  const go = (next: SyntheticSearch) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, run: next.run, tab: next.tab }) });

  const p = plan.data;
  const method: SyntheticMethod = draft.method ?? p?.method ?? "cart";
  const rows = draft.rows ?? p?.default_rows ?? 1000;
  const seed = draft.seed ?? 42;
  const smooth = draft.smooth ?? p?.smooth ?? true;
  const dropCopies = draft.drop_copies ?? true;
  const option = p?.methods.find((m) => m.key === method);
  const deep = Boolean(option?.badge);
  const device = draft.device ?? "auto";
  const onGpu = Boolean(p?.gpu) && device !== "cpu";
  const timegan = method === "timegan";
  const dates = p ? p.columns.filter((c) => c.type === "datetime" && (draft.treatments[c.name] ?? c.treatment) === "model").map((c) => c.name) : [];
  const modelled = p ? p.columns.filter((c) => (draft.treatments[c.name] ?? c.treatment) === "model").length : 0;
  const showColumns = columnsOpen ?? !runId;

  const start = () => {
    const bounds = Object.fromEntries(
      Object.entries(draft.bounds).filter(([, b]) => b.minimum != null || b.maximum != null),
    );
    const body: SyntheticRequest = { method, rows, seed, smooth, drop_copies: dropCopies, treatments: draft.treatments, bounds };
    if (deep) {
      body.epochs = timegan ? null : (draft.epochs ?? null);
      body.device = device;
    }
    if (timegan) body.sequence = draft.sequence ?? {};
    generate.mutate(body, {
      onSuccess: (created) => {
        setJobId(created.id);
        go({ run: undefined, tab: search.tab });
      },
    });
  };

  if (dataset.data && !ready) return <p className="px-(--gutter) py-6 text-ink-3">{t("synthetic.notReady")}</p>;

  return (
    <div className="flex flex-col">
      <header className="flex flex-wrap items-start justify-between gap-4 px-(--gutter) pt-6 pb-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{t("synthetic.title")}</h1>
          <p className="mt-1 max-w-prose text-ink-2">{t("synthetic.description")}</p>
        </div>
        <RunsMenu
          runs={runs.data ?? []}
          current={runId}
          compared={compared}
          onOpen={(id) => {
            setCompared([]);
            go({ run: id, tab: search.tab });
          }}
          onCompare={setCompared}
          onDelete={(r) => {
            if (!window.confirm(t("synthetic.deleteConfirm", { method: r.method_label }))) return;
            actions.remove.mutate(r.id, { onSuccess: () => runId === r.id && go({ run: undefined, tab: search.tab }) });
          }}
        />
      </header>

      <div className="space-y-4 px-(--gutter) pb-10">
        {plan.error && <ErrorNote>{plan.error.message}</ErrorNote>}
        {p && (
          <>
            <SettingsBar
              label={t("synthetic.settings")}
              summary={[option?.label ?? method, !timegan && t("synthetic.rowsCount", { rows: rows.toLocaleString("en") }), t("synthetic.seedShort", { seed: String(seed) })]}
              phoneAction={
                <Button variant="primary" icon={Play} onClick={start} disabled={running || generate.isPending || Boolean(p.error) || modelled === 0} className="w-full justify-center">
                  {t("synthetic.generate")}
                </Button>
              }
            >
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
                <Group label={t("synthetic.method")}>
                  <Select
                    aria-label={t("synthetic.method")}
                    value={method}
                    onChange={(e) => setDraft({ ...draft, method: e.target.value as SyntheticMethod })}
                    className={small}
                  >
                    {p.methods.map((m) => (
                      <option key={m.key} value={m.key} disabled={!m.available} title={m.unavailable ?? undefined}>
                        {m.label}
                        {m.key === p.method ? " ★" : ""}
                        {m.badge ? ` · ${m.available ? m.badge : t("synthetic.needsPytorch")}` : ""}
                      </option>
                    ))}
                  </Select>
                </Group>
                {timegan && p.sequence && option && (
                  <>
                    <Divider />
                    <SequenceSettings
                      plan={p.sequence}
                      dates={dates}
                      draft={draft.sequence ?? {}}
                      steps={option.epochs ?? 1000}
                      onChange={(sequence) => setDraft({ ...draft, sequence })}
                    />
                  </>
                )}
                {deep && option && (
                  <>
                    {!timegan && (
                      <>
                        <Divider />
                        <Group label={t("synthetic.epochs")}>
                          <TextInput
                            type="number"
                            min={10}
                            max={1000}
                            aria-label={t("synthetic.epochs")}
                            placeholder={onGpu || !p.gpu ? t("synthetic.epochsAuto", { epochs: String(option.epochs) }) : t("synthetic.auto")}
                            value={draft.epochs ?? ""}
                            onChange={(e) => setDraft({ ...draft, epochs: e.target.value === "" ? null : Math.min(1000, Math.max(10, Number(e.target.value))) })}
                            className={`${small} w-24 text-right`}
                          />
                        </Group>
                      </>
                    )}
                    <button
                      type="button"
                      onClick={() => p.gpu && setDraft({ ...draft, device: onGpu ? "cpu" : "auto" })}
                      disabled={!p.gpu}
                      aria-pressed={onGpu}
                      title={p.gpu ? t("synthetic.deviceToggle") : t("synthetic.noGpu")}
                      className={`inline-flex h-7 items-center gap-1 rounded-full border px-2.5 ${onGpu ? "border-accent text-accent" : "border-line text-ink-2"} disabled:opacity-60`}
                    >
                      {onGpu ? <Zap size={12} aria-hidden /> : <Cpu size={12} aria-hidden />}
                      {onGpu ? `GPU · ${shortGpu(p.gpu ?? "")}` : "CPU"}
                    </button>
                  </>
                )}
                {!timegan && (
                  <>
                    <Divider />
                    <Group label={t("synthetic.rows")}>
                      <TextInput
                        type="number"
                        min={100}
                        max={1_000_000}
                        aria-label={t("synthetic.rows")}
                        value={rows}
                        onChange={(e) => setDraft({ ...draft, rows: Math.min(1_000_000, Math.max(100, Number(e.target.value) || 100)) })}
                        className={`${small} w-24 text-right`}
                      />
                    </Group>
                  </>
                )}
                <Divider />
                <Group label={t("synthetic.seed")}>
                  <TextInput
                    type="number"
                    min={0}
                    aria-label={t("synthetic.seed")}
                    value={seed}
                    onChange={(e) => setDraft({ ...draft, seed: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
                    className={`${small} w-16 text-right`}
                  />
                </Group>
                <Divider />
                <span title={p.smooth_reason}>
                  <Checkbox
                    label={t("synthetic.smooth")}
                    checked={smooth}
                    disabled={method !== "cart"}
                    onChange={(e) => setDraft({ ...draft, smooth: e.target.checked })}
                    className="text-ink-2"
                  />
                </span>
                <span title={t("synthetic.dropCopiesHint")}>
                  <Checkbox
                    label={t("synthetic.dropCopies")}
                    checked={dropCopies}
                    onChange={(e) => setDraft({ ...draft, drop_copies: e.target.checked })}
                    className="text-ink-2"
                  />
                </span>
                <Divider />
                <button
                  type="button"
                  onClick={() => setColumnsOpen(!showColumns)}
                  aria-expanded={showColumns}
                  className="inline-flex items-center gap-1 text-ink-2 hover:text-ink-1"
                >
                  {showColumns ? <ChevronDown size={13} aria-hidden /> : <ChevronRight size={13} aria-hidden />}
                  {t("synthetic.modelledOf", { modelled: String(modelled), total: String(p.columns.length) })}
                </button>
                <div className="ml-auto max-sm:hidden">
                  <Button variant="primary" icon={Play} onClick={start} disabled={running || generate.isPending || Boolean(p.error) || modelled === 0}>
                    {t("synthetic.generate")}
                  </Button>
                </div>
              </div>
            </SettingsBar>
            <p className="text-xs text-ink-3">
              <span className="font-medium text-ink-2">{t("synthetic.why")} </span>
              {draft.method && draft.method !== p.method
                ? p.methods.find((m) => m.key === draft.method)?.why
                : p.method_reason}
              {timegan && p.sequence && !p.sequence.error ? ` ${p.sequence.reason} · ${p.sequence.step}` : ""}
              {" · "}
              {t("synthetic.learnsFrom", { rows: p.train_rows.toLocaleString("en") })}
            </p>
            {timegan && p.sequence?.warning && <p className="border-l-2 border-warn pl-3 text-xs text-ink-2">{p.sequence.warning}</p>}
            {p.error && <ErrorNote>{p.error}</ErrorNote>}
            {showColumns && <PlanTable plan={p} draft={draft} onChange={setDraft} />}
          </>
        )}

        {generate.error && <ErrorNote>{generate.error.message}</ErrorNote>}
        {running && job.data && (
          <Progress
            stage={job.data.stage}
            share={job.data.progress}
            message={job.data.message}
            onCancel={() => jobId && cancel.mutate(jobId)}
            cancelling={cancel.isPending}
          />
        )}
        {status === "failed" && <ErrorNote>{job.data?.error}</ErrorNote>}
        {status === "cancelled" && <p className="text-[13px] text-ink-2">{job.data?.error}</p>}

        {compared.length >= 2 && <RunCompare datasetId={datasetId} runIds={compared} onClose={() => setCompared([])} />}

        {!running && run.data && (
          <Report
            run={run.data}
            datasetId={datasetId}
            tab={search.tab ?? "columns"}
            onTab={(tab: Tab) => go({ run: runId, tab })}
          />
        )}
        {!running && runId && run.isLoading && <Loading label={t("loading.report")} className="h-64" />}
        {!running && !runId && p && !p.error && (
          <p className="border-y border-line py-6 text-center text-[13px] text-ink-3">{t("synthetic.empty")}</p>
        )}
      </div>
    </div>
  );
}
