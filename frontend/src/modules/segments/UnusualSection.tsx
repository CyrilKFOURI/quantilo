import { useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { Database, Download } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import {
  segmentsApi,
  useCancelSegmentJob,
  useSegmentJob,
  useSegmentRuns,
  useStartUnusual,
  useUnusualActions,
  useUnusualRow,
  useUnusualRun,
  useUnusualRuns,
  type Mark,
  type UnusualMethod,
  type UnusualRow,
  type UnusualRun,
} from "./api";
import { rowsAbove, unusualFigure } from "./figures";
import { JobStatus, type Shared } from "./SegmentsPage";
import { Loading } from "../../ui/Loading";

const METHODS: UnusualMethod[] = ["auto", "isolation_forest", "mahalanobis", "lof", "segment"];

export function UnusualSection({
  datasetId,
  runId,
  onRun,
  shared,
}: {
  datasetId: string;
  runId: string | undefined;
  onRun: (id: string) => void;
  shared: Shared;
}) {
  const client = useQueryClient();
  const runs = useUnusualRuns(datasetId, true);
  const segmentations = useSegmentRuns(datasetId, true);
  const shownId = runId ?? runs.data?.[0]?.id ?? null;
  const run = useUnusualRun(datasetId, shownId);
  const [method, setMethod] = useState<UnusualMethod>("auto");
  const [jobId, setJobId] = useState<string | null>(null);
  const start = useStartUnusual(datasetId);
  const cancel = useCancelSegmentJob();
  const job = useSegmentJob(jobId);
  const started = useRef(false);
  const latestSegmentation = segmentations.data?.[0];

  const launch = () =>
    start.mutate(
      {
        columns: shared.columns,
        preparation: shared.preparation,
        method,
        segments_run: method === "segment" ? (latestSegmentation?.id ?? null) : null,
      },
      { onSuccess: (created) => setJobId(created.id) },
    );

  // The first visit finds unusual rows on its own, with the app's choices.
  useEffect(() => {
    if (runs.data && runs.data.length === 0 && !started.current) {
      started.current = true;
      start.mutate({ columns: shared.columns, preparation: shared.preparation }, { onSuccess: (created) => setJobId(created.id) });
    }
  }, [runs.data, start, shared]);

  const finished = job.data?.status === "done" && job.data.kind === "unusual" ? job.data.run_id : null;
  const shownFinished = useRef<string | null>(null);
  useEffect(() => {
    if (finished && shownFinished.current !== finished) {
      shownFinished.current = finished;
      void client.invalidateQueries({ queryKey: ["datasets", datasetId, "segments", "unusual-runs"] });
      onRun(finished);
    }
  }, [finished, client, datasetId, onRun]);

  const running = start.isPending || job.data?.status === "running";
  return (
    <div className="grid gap-6 lg:grid-cols-[300px_minmax(0,1fr)]">
      <aside className="space-y-5 lg:sticky lg:top-4 lg:self-start" aria-label={t("unusual.settings")}>
        <section className="space-y-1.5">
          <label className="flex items-center justify-between gap-3">
            <span className="text-xs text-ink-3">{t("unusual.method")}</span>
            <Select aria-label={t("unusual.method")} value={method} onChange={(e) => setMethod(e.target.value as UnusualMethod)} className="h-8 max-w-52 text-[13px]">
              {METHODS.map((m) => (
                <option key={m} value={m} disabled={m === "segment" && !latestSegmentation}>
                  {t(`unusual.method.${m}` as MessageKey)}
                </option>
              ))}
            </Select>
          </label>
          <p className="text-xs text-ink-2">
            {method === "segment"
              ? latestSegmentation
                ? t("unusual.segmentFrom", { run: `${latestSegmentation.method_label} · ${latestSegmentation.k}` })
                : t("unusual.segmentNeedsRun")
              : t(`unusual.methodHelp.${method}` as MessageKey)}
          </p>
          <p className="text-xs text-ink-3">{shared.preparation ? t("unusual.columnsCustom") : t("unusual.columnsAuto")}</p>
        </section>
        <Button variant="primary" onClick={launch} disabled={running}>
          {running ? t("segments.running") : t("unusual.run")}
        </Button>
        {job.data?.status === "running" && jobId && <JobStatus job={job.data} onCancel={() => cancel.mutate(jobId)} />}
        {job.data?.status === "cancelled" && <p className="text-xs text-ink-3">{job.data.error}</p>}
        {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
        {job.data?.status === "failed" && <ErrorNote>{job.data.error}</ErrorNote>}
        {runs.data && runs.data.length > 0 && (
          <section aria-label={t("unusual.runs")} className="space-y-1 border-t border-line pt-4">
            <h3 className="text-xs text-ink-3">{t("unusual.runs")}</h3>
            {runs.data.map((r, index) => (
              <button
                key={r.id}
                type="button"
                onClick={() => onRun(r.id)}
                aria-pressed={shownId === r.id}
                className={`block w-full rounded-md border px-2 py-1.5 text-left text-xs ${shownId === r.id ? "border-accent bg-accent-soft" : "border-transparent hover:bg-surface-1"}`}
              >
                <span className="block text-ink-1">
                  #{runs.data.length - index} {r.method_label}
                </span>
                <span className="block text-ink-3 tabular-nums">
                  {t("unusual.runLine", { rows: formatCount(r.proposed), of: formatCount(r.rows) })} · {r.seconds.toFixed(1)} s
                </span>
              </button>
            ))}
          </section>
        )}
      </aside>
      <div className={`min-w-0 space-y-6 ${running ? "opacity-60" : ""}`} aria-busy={running}>
        {run.data ? (
          <UnusualResults key={run.data.id} datasetId={datasetId} run={run.data} />
        ) : run.isError ? (
          <ErrorNote>{run.error.message}</ErrorNote>
        ) : runs.isPending || run.isLoading ? (
          <Loading label={t("loading.unusual")} className="h-72 rounded-lg" />
        ) : (
          <div className="flex h-72 items-center justify-center rounded-lg border border-dashed border-line text-ink-3">
            {running ? t("unusual.first") : t("unusual.none")}
          </div>
        )}
      </div>
    </div>
  );
}

function UnusualResults({ datasetId, run }: { datasetId: string; run: UnusualRun }) {
  const result = run.result;
  const listed = result.listed;
  const [threshold, setThreshold] = useState(result.threshold.value);
  const [open, setOpen] = useState<UnusualRow | null>(null);
  const actions = useUnusualActions(datasetId, run.id);
  // the proposed threshold may keep more rows than are listed: its count comes from every row
  const kept = threshold === result.threshold.value ? result.threshold.rows : rowsAbove(result, threshold);
  const shown = listed.filter((row) => row.score >= threshold);
  const low = listed.at(-1)?.score ?? threshold;
  const high = listed[0]?.score ?? threshold;
  const created = actions.save.data?.dataset_id;
  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h2 className="text-lg font-semibold tracking-tight">
          {t("unusual.headline", { rows: formatCount(kept), of: formatCount(result.rows) })}
        </h2>
        <p className="text-[13px] text-ink-2">
          {result.method_why} {result.threshold.why}
        </p>
        {result.hidden_expected > 0 && (
          <p className="text-[13px] text-ink-2" role="status">
            {t(result.hidden_expected === 1 ? "unusual.hiddenOne" : "unusual.hiddenMany", { count: String(result.hidden_expected) })}
          </p>
        )}
        {result.sampled && <p className="text-xs text-ink-3">{t("unusual.sampled", { rows: formatCount(result.fit_rows) })}</p>}
      </header>
      <section aria-labelledby="unusual-curve" className="space-y-2">
        <h3 id="unusual-curve" className="flex items-center gap-1 text-sm font-medium">
          {t("unusual.curve")}
          <InfoTip label={t("unusual.curve")} text={t("unusual.curveHelp")} />
        </h3>
        <Plot
          figure={unusualFigure(result, threshold)}
          height={260}
          label={t("unusual.curve")}
          onPointClick={(point) => typeof point.y === "number" && setThreshold(Math.min(Math.max(point.y, low), high))}
        />
        <label className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-2">
          <span className="shrink-0">{t("unusual.threshold")}</span>
          <input
            type="range"
            min={low}
            max={high}
            step={(high - low) / 500 || 0.001}
            value={threshold}
            aria-label={t("unusual.threshold")}
            aria-valuetext={t("unusual.thresholdValue", { score: formatNumber(threshold), rows: formatCount(kept) })}
            className="min-w-40 flex-1 accent-(--accent)"
            onChange={(e) => setThreshold(Number(e.target.value))}
          />
          <span className="shrink-0 tabular-nums" aria-live="polite">
            {t("unusual.thresholdValue", { score: formatNumber(threshold), rows: formatCount(kept) })}
          </span>
          <Button variant="ghost" onClick={() => setThreshold(result.threshold.value)} disabled={threshold === result.threshold.value}>
            {t("unusual.proposed")}
          </Button>
        </label>
        <p className="text-xs text-ink-3">{t("unusual.listedNote", { rows: formatCount(listed.length) })}</p>
      </section>
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" icon={Database} onClick={() => actions.save.mutate(threshold)} disabled={actions.save.isPending}>
          {t("segments.save")}
        </Button>
        <a
          href={segmentsApi.exportUrl(datasetId, run.id, threshold)}
          download
          className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2"
        >
          <Download size={15} strokeWidth={1.5} aria-hidden />
          {t("unusual.export")}
        </a>
        {created && (
          <p className="text-[13px] text-ink-2" role="status">
            {t("segments.savedAs")}{" "}
            <Link to="/d/$datasetId" params={{ datasetId: created }} className="text-accent hover:underline">
              {t("segments.openDataset")}
            </Link>
          </p>
        )}
      </div>
      {actions.save.error && <ErrorNote>{actions.save.error.message}</ErrorNote>}
      <RowsTable
        rows={shown}
        columns={result.columns}
        marks={run.marks}
        onOpen={setOpen}
        onMark={(row, mark) => actions.mark.mutate({ key: row.key, mark })}
      />
      {actions.mark.isError && <ErrorNote>{actions.mark.error.message}</ErrorNote>}
      {open && <RowCard datasetId={datasetId} runId={run.id} row={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

function RowsTable({
  rows,
  columns,
  marks,
  onOpen,
  onMark,
}: {
  rows: UnusualRow[];
  columns: string[];
  marks: Record<string, Mark>;
  onOpen: (row: UnusualRow) => void;
  onMark: (row: UnusualRow, mark: Mark | null) => void;
}) {
  if (rows.length === 0) return <p className="text-[13px] text-ink-3">{t("unusual.noRows")}</p>;
  return (
    <div className="max-h-[560px] overflow-auto border-y border-line">
      <table className="w-full text-[13px]">
        <caption className="sr-only">{t("unusual.table")}</caption>
        <thead className="sticky top-0 bg-surface-0">
          <tr className="border-b border-line text-xs text-ink-3">
            <th className="py-1.5 pr-3 text-right font-normal">#</th>
            <th className="py-1.5 pr-3 text-right font-normal">{t("unusual.score")}</th>
            <th className="py-1.5 pr-3 text-left font-normal">{t("unusual.mark")}</th>
            <th className="py-1.5 pr-3 text-left font-normal">{t("unusual.reasons")}</th>
            {columns.map((c) => (
              <th key={c} className="py-1.5 pr-3 text-left font-mono font-normal whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const mark = marks[row.key] ?? null;
            return (
              <tr key={row.row} className={`border-b border-line align-top last:border-0 ${mark === "expected" ? "text-ink-3" : ""}`}>
                <td className="py-1.5 pr-3 text-right tabular-nums">{row.rank}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{formatNumber(row.score)}</td>
                <td className="py-1.5 pr-3 whitespace-nowrap">
                  {(["expected", "error"] as const).map((m) => (
                    <button
                      key={m}
                      type="button"
                      aria-pressed={mark === m}
                      onClick={() => onMark(row, mark === m ? null : m)}
                      className={`mr-1 rounded-md border px-1.5 py-0.5 text-xs ${mark === m ? "border-accent bg-accent-soft text-ink-1" : "border-line text-ink-2 hover:bg-surface-2"}`}
                    >
                      {t(`unusual.mark.${m}` as MessageKey)}
                    </button>
                  ))}
                </td>
                <td className="min-w-80 py-1.5 pr-3">
                  <button type="button" onClick={() => onOpen(row)} className="text-left hover:underline" aria-label={t("unusual.openRow", { row: String(row.row) })}>
                    {row.combination && <span className="block text-ink-1">{row.combination}</span>}
                    {row.reasons.map((r) => (
                      <span key={r.column} className="block text-ink-2">
                        {r.text}
                      </span>
                    ))}
                  </button>
                </td>
                {columns.map((c) => (
                  <td key={c} className="py-1.5 pr-3 whitespace-nowrap text-ink-1 tabular-nums">
                    {display(row.values[c])}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function display(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return formatNumber(value);
  return String(value).replace("T00:00:00", "");
}

/** The whole row, each value placed on its column: percentile for numbers, share for categories. */
function RowCard({ datasetId, runId, row, onClose }: { datasetId: string; runId: string; row: UnusualRow; onClose: () => void }) {
  const card = useUnusualRow(datasetId, runId, row.row);
  return (
    <Dialog open onOpenChange={(open) => !open && onClose()} title={t("unusual.rowTitle", { row: String(row.row), rank: String(row.rank) })} description={row.combination ?? row.reasons[0]?.text}>
      {card.isError ? (
        <ErrorNote>{card.error.message}</ErrorNote>
      ) : !card.data ? (
        <Loading label={t("loading.row")} className="h-40 rounded-md" />
      ) : (
        <table className="w-full text-[13px]">
          <caption className="sr-only">{t("unusual.rowValues")}</caption>
          <tbody>
            {card.data.map((v) => {
              const reason = row.reasons.some((r) => r.column === v.column);
              const where = v.percentile ?? v.share;
              return (
                <tr key={v.column} className="border-b border-line last:border-0">
                  <td className={`py-1.5 pr-3 font-mono text-xs ${reason ? "font-semibold text-ink-1" : "text-ink-2"}`}>{v.column}</td>
                  <td className="py-1.5 pr-3 tabular-nums">{display(v.value)}</td>
                  <td className="w-48 py-1.5 pr-3">
                    {where !== null && (
                      <span className="relative block h-1.5 rounded-full bg-surface-2" aria-hidden>
                        <span className="absolute top-1/2 size-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent" style={{ left: `${Math.round(where * 100)}%` }} />
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 text-xs text-ink-2">
                    {v.percentile !== null
                      ? t("unusual.percentile", { share: `${Math.round(v.percentile * 1000) / 10}`, median: display(v.median) })
                      : v.share !== null
                        ? t("unusual.share", { share: `${Math.round(v.share * 1000) / 10}` })
                        : ""}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Dialog>
  );
}
