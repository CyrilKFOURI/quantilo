import { useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";

import type { MlChoicesInput } from "../../api/client";
import { useDataset } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { ErrorNote, ProgressBar, Select } from "../../ui/fields";
import { PageHeader } from "../../ui/PageHeader";
import { ParamPanel } from "../../ui/ParamPanel";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Tabs } from "../../ui/Tabs";
import {
  useCancelSegmentJob,
  useSegmentJob,
  useSegmentPlan,
  useSegmentRun,
  useSegmentRuns,
  useStartSegments,
  type SegmentJob,
  type SegmentPlan,
  type SegmentRequest,
  type SegmentRunSummary,
} from "./api";
import { ColumnsDialog } from "./ColumnsDialog";
import { Results, RunCompare } from "./Results";
import type { Section, SegmentsSearch } from "./search";
import { UnusualSection } from "./UnusualSection";
import { Loading } from "../../ui/Loading";

/** The columns and preparation both sections share (unset: the app's choice). */
export type Shared = { columns: string[] | null; preparation: MlChoicesInput | null };

export function SegmentsPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const search = useSearch({ strict: false }) as SegmentsSearch;
  const navigate = useNavigate();
  const set = (patch: SegmentsSearch) =>
    void navigate({ to: ".", search: (previous: Record<string, unknown>) => ({ ...previous, ...patch }), replace: true });
  const dataset = useDataset(datasetId);
  const ready = dataset.data?.status === "ready";
  const [shared, setShared] = useState<Shared>({ columns: null, preparation: null });
  const section: Section = search.section ?? "segments";

  return (
    <div className="flex flex-col gap-5 px-(--gutter) py-6">
      <PageHeader title={t("segments.title")} description={t("segments.description")} />
      <Tabs
        label={t("segments.sections")}
        value={section}
        items={[
          { value: "segments", label: t("segments.section.segments") },
          { value: "unusual", label: t("segments.section.unusual") },
        ]}
        onChange={(next) => set({ section: next === "segments" ? undefined : next })}
      >
        {!ready ? (
          <Loading label={t("dataset.loading")} className="h-72 rounded-lg" />
        ) : section === "segments" ? (
          <SegmentsWorkbench datasetId={datasetId} runId={search.run} onRun={(run) => set({ run })} shared={shared} onShared={setShared} />
        ) : (
          <UnusualSection datasetId={datasetId} runId={search.unusual} onRun={(unusual) => set({ unusual })} shared={shared} />
        )}
      </Tabs>
    </div>
  );
}

/** "Choosing k · k = 4", with the share done when the stage reports one. */
export function JobStatus({ job, onCancel }: { job: SegmentJob; onCancel: () => void }) {
  return (
    <div className="space-y-1.5" role="status">
      <p className="flex items-center justify-between gap-2 text-xs text-ink-2">
        <span>
          {t(`segments.stage.${job.stage}` as MessageKey)}
          {job.message ? ` · ${job.message}` : ""}
        </span>
        <Button variant="ghost" onClick={onCancel}>
          {t("segments.cancel")}
        </Button>
      </p>
      {job.progress !== null && <ProgressBar value={job.progress} label={t("segments.progress")} />}
    </div>
  );
}

function SegmentsWorkbench({
  datasetId,
  runId,
  onRun,
  shared,
  onShared,
}: {
  datasetId: string;
  runId: string | undefined;
  onRun: (id: string) => void;
  shared: Shared;
  onShared: (next: Shared) => void;
}) {
  const client = useQueryClient();
  const runs = useSegmentRuns(datasetId, true);
  const shownId = runId ?? runs.data?.[0]?.id ?? null;
  const run = useSegmentRun(datasetId, shownId);
  const phone = useMedia(PHONE);
  const [method, setMethod] = useState<SegmentRequest["method"]>("auto");
  const [k, setK] = useState<number | null>(null);
  const [editing, setEditing] = useState(false);
  const [selected, setSelected] = useState<string[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const request: SegmentRequest = { columns: shared.columns, preparation: shared.preparation, method, k };
  const plan = useSegmentPlan(datasetId, { columns: shared.columns, preparation: shared.preparation }, true);
  const start = useStartSegments(datasetId);
  const cancel = useCancelSegmentJob();
  const job = useSegmentJob(jobId);
  const started = useRef(false);

  const launch = (body: SegmentRequest = request) => start.mutate(body, { onSuccess: (created) => setJobId(created.id) });

  // A first segmentation runs on its own, with the app's choices.
  useEffect(() => {
    if (runs.data && runs.data.length === 0 && !started.current) {
      started.current = true;
      start.mutate({}, { onSuccess: (created) => setJobId(created.id) });
    }
  }, [runs.data, start]);

  // When a segmentation ends, show its run and refresh the list (once per job).
  const finished = job.data?.status === "done" && job.data.kind === "segments" ? job.data.run_id : null;
  const shownFinished = useRef<string | null>(null);
  useEffect(() => {
    if (finished && shownFinished.current !== finished) {
      shownFinished.current = finished;
      void client.invalidateQueries({ queryKey: ["datasets", datasetId, "segments", "runs"] });
      onRun(finished);
    }
  }, [finished, client, datasetId, onRun]);

  const running = start.isPending || job.data?.status === "running";
  const result = run.data?.result;

  const settings = (
    <>
      <ColumnsSummary plan={plan.data} error={plan.error?.message} onEdit={() => setEditing(true)} />
      <section className="space-y-1.5">
        <label className="flex items-center justify-between gap-3">
          <span className="text-xs text-ink-3">{t("segments.method")}</span>
          <Select aria-label={t("segments.method")} value={method} onChange={(e) => setMethod(e.target.value as SegmentRequest["method"])} className="h-8 max-w-48 text-[13px]">
            <option value="auto">{plan.data ? t("segments.method.autoIs", { method: t(`segments.method.${plan.data.method}` as MessageKey) }) : t("segments.method.auto")}</option>
            {plan.data?.methods.map((m) => (
              <option key={m.key} value={m.key} disabled={!m.available}>
                {m.label}
              </option>
            ))}
          </Select>
        </label>
        <p className="text-xs text-ink-2">
          {method === "auto" ? plan.data?.method_why : plan.data?.methods.find((m) => m.key === method)?.reason}
        </p>
      </section>
      <KControl k={k} disabled={method === "hdbscan" || running} onK={setK} onCommit={(value) => launch({ ...request, k: value })} />
    </>
  );

  return (
    <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
      <aside className="space-y-5 lg:sticky lg:top-4 lg:max-h-[calc(100dvh-2rem)] lg:self-start lg:overflow-y-auto lg:pr-2" aria-label={t("segments.settings")}>
        {plan.data && (
          <ColumnsDialog
            open={editing}
            onOpenChange={setEditing}
            plan={plan.data}
            shared={shared}
            onShared={onShared}
          />
        )}
        {phone ? (
          // 041: on a phone the settings fold into one line; Segment stays in sight.
          <ParamPanel
            label={t("segments.settings")}
            layout="stack"
            summary={[
              method === "auto"
                ? plan.data && t(`segments.method.${plan.data.method}` as MessageKey)
                : plan.data?.methods.find((m) => m.key === method)?.label,
              k !== null && t("segments.kShort", { k: String(k) }),
            ]}
          >
            {settings}
          </ParamPanel>
        ) : (
          settings
        )}
        <span className="flex gap-2">
          <Button variant="primary" onClick={() => launch()} disabled={running}>
            {running ? t("segments.running") : t("segments.run")}
          </Button>
        </span>
        {job.data?.status === "running" && jobId && <JobStatus job={job.data} onCancel={() => cancel.mutate(jobId)} />}
        {job.data?.status === "cancelled" && <p className="text-xs text-ink-3">{job.data.error}</p>}
        {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
        {job.data?.status === "failed" && <ErrorNote>{job.data.error}</ErrorNote>}
        {runs.data && runs.data.length > 0 && (
          <RunStrip runs={runs.data} shown={shownId} onPick={onRun} selected={selected} onSelect={setSelected} />
        )}
      </aside>
      <div className={`min-w-0 space-y-6 ${running ? "opacity-60" : ""}`} aria-busy={running}>
        {result && run.data ? (
          <Results datasetId={datasetId} run={run.data} />
        ) : run.isError ? (
          <ErrorNote>{run.error.message}</ErrorNote>
        ) : runs.isPending || run.isLoading ? (
          <Loading label={t("loading.segments")} className="h-72 rounded-lg" />
        ) : (
          <div className="flex h-72 items-center justify-center rounded-lg border border-dashed border-line text-ink-3">
            {running ? t("segments.first") : t("segments.none")}
          </div>
        )}
        {selected.length >= 2 && <RunCompare datasetId={datasetId} runIds={selected} />}
      </div>
    </div>
  );
}

function ColumnsSummary({ plan, error, onEdit }: { plan: SegmentPlan | undefined; error: string | undefined; onEdit: () => void }) {
  if (error) return <ErrorNote>{error}</ErrorNote>;
  if (!plan) return <Loading label={t("loading.columns")} className="h-16 rounded-md" compact />;
  const used = plan.preparation.decisions.filter((d) => d.use);
  const left = plan.preparation.decisions.filter((d) => !d.use);
  return (
    <section className="space-y-1.5 text-xs">
      <div className="flex items-center justify-between gap-2">
        <p className="text-ink-3">
          {t("segments.columns")} · {plan.preparation.custom ? t("segments.custom") : t("segments.automatic")}
        </p>
        <Button variant="ghost" onClick={onEdit}>
          {t("segments.edit")}
        </Button>
      </div>
      <p className="text-ink-1">{t("segments.columnsUsed", { used: String(used.length), left: String(left.length) })}</p>
      <ul className="space-y-0.5 text-ink-2">
        {left.slice(0, 4).map((d) => (
          <li key={d.column}>
            <span className="font-mono text-ink-1">{d.column}</span> {t("segments.excluded")}: {d.why[0]}
          </li>
        ))}
        {left.length > 4 && <li className="text-ink-3">{t("segments.more", { count: String(left.length - 4) })}</li>}
      </ul>
      <p className="text-ink-3">
        {t("segments.rows", { rows: formatCount(plan.rows) })}
        {plan.sampled ? ` · ${t("segments.sampled", { rows: formatCount(plan.fit_rows) })}` : ""}
      </p>
    </section>
  );
}

/** Auto, or a number of segments set with a slider that runs again when released. */
function KControl({ k, disabled, onK, onCommit }: { k: number | null; disabled: boolean; onK: (k: number | null) => void; onCommit: (k: number) => void }) {
  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs text-ink-3">{t("segments.k")}</span>
        <SegmentedControl
          label={t("segments.k")}
          value={k === null ? "auto" : "set"}
          options={[
            { value: "auto", label: t("segments.k.auto") },
            { value: "set", label: t("segments.k.set") },
          ]}
          onChange={(value) => onK(value === "auto" ? null : 4)}
        />
      </div>
      {k !== null && (
        <label className="flex items-center gap-3">
          <input
            type="range"
            min={2}
            max={10}
            step={1}
            value={k}
            disabled={disabled}
            aria-label={t("segments.k.slider")}
            className="flex-1 accent-(--accent)"
            onChange={(e) => onK(Number(e.target.value))}
            onPointerUp={(e) => onCommit(Number((e.target as HTMLInputElement).value))}
            onKeyUp={(e) => {
              if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(e.key)) onCommit(Number((e.target as HTMLInputElement).value));
            }}
          />
          <span className="w-6 text-right text-[13px] tabular-nums">{k}</span>
        </label>
      )}
    </section>
  );
}

function RunStrip({
  runs,
  shown,
  onPick,
  selected,
  onSelect,
}: {
  runs: SegmentRunSummary[];
  shown: string | null;
  onPick: (id: string) => void;
  selected: string[];
  onSelect: (ids: string[]) => void;
}) {
  return (
    <section aria-label={t("segments.runs")} className="space-y-1.5 border-t border-line pt-4">
      <h3 className="text-xs text-ink-3">{t("segments.runs")}</h3>
      <div className="space-y-0.5">
        {runs.map((r, index) => (
          <span key={r.id} className="flex items-start gap-1.5">
            <input
              type="checkbox"
              className="mt-2 size-3.5 accent-(--accent)"
              aria-label={t("segments.runCompare.pick", { run: String(runs.length - index) })}
              title={t("segments.runCompare.hint")}
              checked={selected.includes(r.id)}
              disabled={!selected.includes(r.id) && selected.length >= 5}
              onChange={(e) => onSelect(e.target.checked ? [...selected, r.id] : selected.filter((id) => id !== r.id))}
            />
            <button
              type="button"
              onClick={() => onPick(r.id)}
              aria-pressed={shown === r.id}
              className={`min-w-0 flex-1 rounded-md border px-2 py-1.5 text-left text-xs ${shown === r.id ? "border-accent bg-accent-soft" : "border-transparent hover:bg-surface-1"}`}
            >
              <span className="block truncate text-ink-1">
                #{runs.length - index} {r.method_label} · {t("segments.kCount", { k: String(r.k) })}
                {r.filtered ? ` · ${t("segments.filtered")}` : ""}
              </span>
              <span className="block text-ink-3 tabular-nums">
                {t("segments.silhouette")} {formatNumber(r.silhouette)} · {t("segments.stability")} {formatNumber(r.stability)} · {r.seconds.toFixed(1)} s
              </span>
            </button>
          </span>
        ))}
      </div>
    </section>
  );
}
