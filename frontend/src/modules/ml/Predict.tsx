import { useQueryClient } from "@tanstack/react-query";
import { Link } from "@tanstack/react-router";
import { AlertTriangle, CheckCircle2, Download, Play, XCircle } from "lucide-react";
import { useEffect, useState } from "react";

import { api, LOWER_IS_BETTER, type MlColumnCheck, type MlLevel, type MlModelCheck, type MlPrediction, type MlRun } from "../../api/client";
import {
  useColumns,
  useDatasets,
  useMlCancelPredict,
  useMlCheck,
  useMlPredict,
  useMlPrediction,
  useMlPredictionRowExplanation,
  useMlPredictionRows,
  useMlPredictions,
  useMlPredictJob,
  useMlSavePrediction,
} from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, ProgressBar, Select } from "../../ui/fields";
import { waterfallFigure } from "./figures";
import { distributionFigure } from "./predictFigures";
import { Panel, ScoreBand } from "./ResultParts";
import { Loading } from "../../ui/Loading";

const PAGE = 100;
const pct = (share: number) => (share > 0 && share < 0.01 ? "< 1 %" : `${(share * 100).toFixed(0)} %`);
const when = (iso: string) =>
  new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

const LEVEL_ICON = { ready: CheckCircle2, check: AlertTriangle, blocked: XCircle } as const;
const LEVEL_TONE = { ready: "text-ok", check: "text-warn", blocked: "text-bad" } as const;

/** Ready · Check · Blocked: an icon and a word, never the colour alone. */
function LevelTag({ level }: { level: MlLevel }) {
  const Icon = LEVEL_ICON[level];
  return (
    <span className={`inline-flex items-center gap-1 text-xs font-medium ${LEVEL_TONE[level]}`}>
      <Icon size={14} strokeWidth={1.5} aria-hidden />
      {t(`ml.predict.level.${level}` as MessageKey)}
    </span>
  );
}

/** Under a run's results: predict on new data, export the model, the predictions made (037). */
export function PredictSection({
  run,
  datasetId,
  predictionId,
  onPrediction,
}: {
  run: MlRun;
  datasetId: string;
  predictionId: string | undefined;
  onPrediction: (id: string | undefined) => void;
}) {
  const [open, setOpen] = useState(false);
  const made = useMlPredictions(datasetId, run.id);
  const shown = made.data?.some((p) => p.id === predictionId) ? predictionId : undefined;
  return (
    <section className="space-y-3 border-t border-line pt-4" aria-labelledby="ml-use-model">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="ml-use-model" className="text-sm font-semibold text-ink-1">
            {t("ml.predict.title")}
          </h2>
          <p className="text-xs text-ink-3">{t("ml.predict.noImport")}</p>
        </div>
        <span className="flex gap-2">
          <Button variant="primary" icon={Play} onClick={() => setOpen(true)}>
            {t("ml.predict.open")}
          </Button>
          <a
            href={api.mlModelZipUrl(datasetId, run.id)}
            className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2"
          >
            <Download size={15} strokeWidth={1.5} aria-hidden />
            {t("ml.predict.export")}
          </a>
        </span>
      </div>
      {made.data && made.data.length > 0 && (
        <ul className="divide-y divide-line rounded-md border border-line text-[13px]" aria-label={t("ml.predict.made")}>
          {made.data.map((p) => (
            <li key={p.id}>
              <button
                type="button"
                onClick={() => onPrediction(p.id === shown ? undefined : p.id)}
                aria-pressed={p.id === shown}
                className={`flex w-full flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 text-left tabular-nums hover:bg-surface-2 ${p.id === shown ? "bg-accent-soft" : ""}`}
              >
                <span className="text-ink-3">{when(p.created_at)}</span>
                <span className="font-medium text-ink-1">{p.source_name}</span>
                <span className="text-ink-2">{t("ml.predict.rows", { rows: p.rows.toLocaleString("en-US") })}</span>
                <LevelTag level={p.level as MlLevel} />
                <span className="text-ink-3">{t("ml.predict.lessReliableShare", { share: pct(p.less_reliable / (p.rows || 1)) })}</span>
              </button>
            </li>
          ))}
        </ul>
      )}
      {shown && <PredictionResult key={shown} datasetId={datasetId} run={run} predictionId={shown} />}
      {open && (
        <PredictDialog
          run={run}
          datasetId={datasetId}
          onClose={() => setOpen(false)}
          onDone={(id) => {
            setOpen(false);
            onPrediction(id);
          }}
        />
      )}
    </section>
  );
}

function PredictDialog({
  run,
  datasetId,
  onClose,
  onDone,
}: {
  run: MlRun;
  datasetId: string;
  onClose: () => void;
  onDone: (predictionId: string) => void;
}) {
  const client = useQueryClient();
  const datasets = useDatasets();
  const ready = (datasets.data ?? []).filter((d) => d.status === "ready");
  const [picked, setPicked] = useState<string | null>(null);
  const source = picked ?? ready.find((d) => d.id !== datasetId)?.id ?? ready[0]?.id ?? null;
  const check = useMlCheck(datasetId, run.id, source);
  const columns = useColumns(source ?? "", Boolean(source));
  const [kept, setKept] = useState<Record<string, string[]>>({});
  const identifiers = (columns.data ?? []).filter((c) => c.type === "identifier").map((c) => c.name);
  const keep = source ? (kept[source] ?? identifiers) : [];
  const predict = useMlPredict(datasetId, run.id);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useMlPredictJob(jobId);
  const cancel = useMlCancelPredict();
  const running = predict.isPending || job.data?.status === "running" || (jobId !== null && !job.data);
  const finished = job.data?.status === "done" ? job.data.prediction_id : null;
  useEffect(() => {
    if (!finished) return;
    void client.invalidateQueries({ queryKey: ["datasets", datasetId, "ml", "predictions", run.id] });
    onDone(finished);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished]);

  const toggle = (name: string, on: boolean) =>
    source && setKept({ ...kept, [source]: on ? [...keep, name] : keep.filter((k) => k !== name) });
  const start = () => {
    if (!source) return;
    predict.mutate({ source, keep }, { onSuccess: (started) => setJobId(started.id) });
  };

  return (
    <Dialog
      open
      onOpenChange={(next) => !next && !running && onClose()}
      title={t("ml.predict.dialogTitle", { target: run.target })}
      description={t("ml.predict.dialogHint")}
      size="lg"
      footer={
        <>
          {running && job.data && (
            <span className="mr-auto flex min-w-64 items-center gap-3 text-xs text-ink-2 tabular-nums">
              <ProgressBar value={job.data.rows / (job.data.total || 1)} label={t("ml.predict.progressLabel")} />
              <span className="shrink-0">
                {t("ml.predict.progress", { rows: job.data.rows.toLocaleString("en-US"), total: job.data.total.toLocaleString("en-US") })}
              </span>
            </span>
          )}
          {running && jobId ? (
            <Button onClick={() => cancel.mutate(jobId)} disabled={cancel.isPending}>
              {t("ml.cancel")}
            </Button>
          ) : (
            <Button onClick={onClose}>{t("ml.predict.close")}</Button>
          )}
          <Button variant="primary" onClick={start} disabled={!check.data || check.data.level === "blocked" || running}>
            {running ? t("ml.predict.running") : t("ml.predict.run")}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <div className="flex flex-wrap items-end gap-4">
          <Field label={t("ml.predict.data")}>
            <Select value={source ?? ""} onChange={(event) => setPicked(event.target.value)} className="min-w-64" disabled={running}>
              {ready.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.id === datasetId ? t("ml.predict.thisData", { name: d.name }) : d.name}
                </option>
              ))}
            </Select>
          </Field>
          <p className="pb-1.5 text-xs text-ink-3">
            {t("ml.predict.newFile")}{" "}
            <Link to="/" className="text-accent hover:underline">
              {t("ml.predict.importFile")}
            </Link>
          </p>
        </div>
        {check.isError && <ErrorNote>{check.error.message}</ErrorNote>}
        {source && !check.data && !check.isError && <Loading label={t("loading.check")} className="h-40 rounded-md" />}
        {check.data && <CheckReport check={check.data} datasetId={datasetId} sourceId={source} />}
        {columns.data && columns.data.length > 0 && (
          <fieldset className="space-y-1.5">
            <legend className="text-xs text-ink-2">{t("ml.predict.keep")}</legend>
            <div className="flex max-h-28 flex-wrap gap-x-4 gap-y-1 overflow-y-auto text-[13px]">
              {columns.data.map((c) => (
                <Checkbox
                  key={c.name}
                  label={<span className="font-mono text-xs">{c.name}</span>}
                  checked={keep.includes(c.name)}
                  onChange={(event) => toggle(c.name, event.target.checked)}
                  disabled={running}
                />
              ))}
            </div>
          </fieldset>
        )}
        {predict.isError && <ErrorNote>{predict.error.message}</ErrorNote>}
        {job.data && (job.data.status === "failed" || job.data.status === "cancelled") && (
          <ErrorNote>{job.data.error ?? job.data.status}</ErrorNote>
        )}
      </div>
    </Dialog>
  );
}

/** "Check: the model can predict, but…" → "The model can predict, but…" (the tag says the level). */
const withoutLevel = (message: string) => {
  const rest = message.replace(/^(Ready|Check|Blocked): /, "");
  return rest.charAt(0).toUpperCase() + rest.slice(1);
};

/** The check: its level and sentence, then every column of the model, the most important first. */
export function CheckReport({
  check,
  datasetId,
  sourceId,
}: {
  check: MlModelCheck;
  /** The training dataset and the data to predict: link to 031's comparison of the two. */
  datasetId?: string;
  sourceId?: string | null;
}) {
  const drift = new Map((check.drift?.columns ?? []).map((c) => [c.column, c]));
  return (
    <div className="space-y-2">
      <p className="flex flex-wrap items-baseline gap-2 text-[13px] text-ink-1">
        <LevelTag level={check.level} />
        <span>{withoutLevel(check.message)}</span>
      </p>
      {check.problems
        .filter((p) => p.includes("PSI"))
        .map((p) => (
          <p key={p} className="text-xs text-warn">
            {p}
          </p>
        ))}
      {check.drift && (
        <p className="text-xs text-ink-2">
          {t("ml.predict.drift", { headline: check.drift.headline })}{" "}
          {datasetId && sourceId && (
            <Link
              to="/d/$datasetId/changes"
              params={{ datasetId: sourceId }}
              search={{ vs: datasetId }}
              className="text-accent hover:underline"
            >
              {t("ml.predict.driftLink")}
            </Link>
          )}
        </p>
      )}
      {check.target_present && <p className="text-xs text-ink-2">{t("ml.predict.targetPresent")}</p>}
      <div className="max-h-72 overflow-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th className="px-3 py-1.5 font-normal">{t("ml.predict.column")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ml.predict.type")}</th>
              <th className="px-3 py-1.5 text-right font-normal">{t("ml.importance")}</th>
              <th className="px-3 py-1.5 text-right font-normal">{t("ml.predict.unknown")}</th>
              <th className="px-3 py-1.5 text-right font-normal">{t("ml.predict.outOfRange")}</th>
              {check.drift && <th className="px-3 py-1.5 text-right font-normal">{t("ml.predict.shift")}</th>}
              <th className="px-3 py-1.5 font-normal">{t("ml.predict.note")}</th>
            </tr>
          </thead>
          <tbody>
            {check.columns.map((c) => (
              <CheckRow key={c.column} column={c} drift={check.drift ? (drift.get(c.column) ?? null) : undefined} />
            ))}
          </tbody>
        </table>
      </div>
      {check.ignored.length > 0 && (
        <p className="text-xs text-ink-3">{t("ml.predict.ignored", { columns: check.ignored.join(", ") })}</p>
      )}
    </div>
  );
}

type DriftColumn = NonNullable<MlModelCheck["drift"]>["columns"][number];
const SHIFT_TONE = { stable: "text-ink-2", moderate: "text-warn", strong: "text-bad" } as const;

/** `drift`: undefined when there was no comparison (no column), null for a column not compared. */
function CheckRow({ column: c, drift }: { column: MlColumnCheck; drift?: DriftColumn | null }) {
  const tone = c.found === null ? "text-bad" : c.note ? "text-warn" : "text-ink-2";
  return (
    <tr className="border-t border-line align-top">
      <td className="px-3 py-1.5 font-mono text-xs text-ink-1">
        {c.column}
        {c.important && <span className="sr-only"> {t("ml.predict.important")}</span>}
      </td>
      <td className="px-3 py-1.5 text-xs text-ink-2">
        {t(`type.${c.expected}` as MessageKey)}
        {c.found !== null && c.found !== c.expected ? ` ← ${t(`type.${c.found}` as MessageKey)}` : ""}
      </td>
      <td className="px-3 py-1.5 text-right text-ink-2">{c.importance === null ? "—" : pct(c.importance)}</td>
      <td className="px-3 py-1.5 text-right">
        {typeof c.unknown_terms === "number" ? (
          <span title={t("ml.predict.unknownTermsHint", { training: pct(c.training_unknown_terms ?? 0) })}>
            {t("ml.predict.unknownTerms", { share: pct(c.unknown_terms) })}
          </span>
        ) : c.unknown_rows ? (
          pct(c.unknown_share)
        ) : (
          "—"
        )}
      </td>
      <td className="px-3 py-1.5 text-right">{c.out_of_range ? pct(c.out_of_range) : "—"}</td>
      {drift !== undefined && (
        <td
          className={`px-3 py-1.5 text-right ${drift?.drift && !drift.long_tail ? SHIFT_TONE[drift.drift] : "text-ink-3"}`}
          title={drift?.long_tail ? t("changes.longTailHelp") : undefined}
        >
          {drift?.psi == null ? "—" : drift.long_tail ? t("changes.longTail") : `PSI ${drift.psi.toFixed(2)}`}
        </td>
      )}
      <td className={`px-3 py-1.5 text-xs ${tone}`}>{c.note ?? t("ml.predict.ok")}</td>
    </tr>
  );
}

/** A prediction: key figures, scores against the validation when the data had the answer, the
 *  distribution of the predictions against the validation's, and the rows. */
function PredictionResult({ datasetId, run, predictionId }: { datasetId: string; run: MlRun; predictionId: string }) {
  const found = useMlPrediction(datasetId, predictionId);
  if (found.isError) return <ErrorNote>{found.error.message}</ErrorNote>;
  if (!found.data?.summary) return <Loading label={t("loading.predictions")} className="h-60 rounded-md" />;
  const prediction = found.data;
  const summary = prediction.summary!;
  const share = prediction.less_reliable / (prediction.rows || 1);
  return (
    <div className="@container space-y-4">
      <p className="text-[15px] leading-snug font-medium text-ink-1">
        {t("ml.predict.sentence", { rows: prediction.rows.toLocaleString("en-US"), name: prediction.source_name })}{" "}
        {prediction.less_reliable > 0 ? t("ml.predict.sentenceLessReliable", { share: pct(share) }) : t("ml.predict.sentenceReliable")}
      </p>
      <ScoreBand
        items={[
          { label: t("ml.predict.rowsLabel"), value: prediction.rows.toLocaleString("en-US") },
          { label: t("ml.predict.lessReliable"), value: pct(share), explain: t("ml.predict.lessReliableHint") },
          { label: t("ml.predict.checkLabel"), value: t(`ml.predict.level.${prediction.level}` as MessageKey) },
          ...(summary.interval
            ? [
                {
                  label: t("ml.predict.interval", { level: (summary.interval.level * 100).toFixed(0) }),
                  value: `± ${formatNumber(summary.interval.half_width)}`,
                  explain: t("ml.predict.intervalHint", { errors: summary.interval.errors.toLocaleString("en-US") }),
                },
              ]
            : []),
        ]}
      />
      {prediction.check && prediction.check.level !== "ready" && (
        <p className="flex items-baseline gap-2 text-xs text-ink-2">
          <LevelTag level={prediction.check.level} />
          {withoutLevel(prediction.check.message)}
        </p>
      )}
      <div className="grid gap-x-5 gap-y-4 @4xl:grid-cols-2">
        <Panel title={t("ml.predict.distribution")}>
          <Plot figure={distributionFigure(summary.distribution)} height={240} label={t("ml.predict.distribution")} />
        </Panel>
        {summary.scores && <ScoresAgainstValidation prediction={prediction} />}
      </div>
      <PredictionTable datasetId={datasetId} run={run} prediction={prediction} />
    </div>
  );
}

function ScoresAgainstValidation({ prediction }: { prediction: MlPrediction }) {
  const summary = prediction.summary!;
  const scores = summary.scores ?? {};
  const keys = [summary.primary, ...Object.keys(scores).filter((k) => k !== summary.primary)].filter(
    (k) => scores[k] != null || summary.validation_scores[k] != null,
  );
  const show = (value: number | null | undefined) => (value === null || value === undefined ? "—" : formatNumber(value));
  const now = scores[summary.primary];
  const before = summary.validation_scores[summary.primary];
  const lower = LOWER_IS_BETTER.has(summary.primary);
  const holds = now != null && before != null && (lower ? now <= before * 1.1 : now >= before - Math.abs(before) * 0.1);
  return (
    <Panel title={t("ml.predict.scores", { rows: summary.scored_rows.toLocaleString("en-US") })}>
      {now != null && before != null && (
        <p className={`mb-1 text-xs ${holds ? "text-ink-2" : "text-warn"}`}>
          {t(holds ? "ml.predict.holds" : "ml.predict.drops", { metric: t(`ml.metric.${summary.primary}` as MessageKey) })}
        </p>
      )}
      <div className="max-h-[220px] overflow-auto">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="text-left text-xs text-ink-3">
            <tr>
              <th className="py-1 font-normal">{t("ml.metricName")}</th>
              <th className="py-1 text-right font-normal">{t("ml.predict.thisFile")}</th>
              <th className="py-1 text-right font-normal">{t("ml.predict.validation")}</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => (
              <tr key={key} className={`border-t border-line ${key === summary.primary ? "font-medium text-ink-1" : "text-ink-2"}`}>
                <td className="py-1">
                  {t(`ml.metric.${key}` as MessageKey)}
                  {key === summary.primary ? " ★" : ""}
                </td>
                <td className="py-1 text-right">{show(scores[key])}</td>
                <td className="py-1 text-right">{show(summary.validation_scores[key])}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

function PredictionTable({ datasetId, run, prediction }: { datasetId: string; run: MlRun; prediction: MlPrediction }) {
  const [page, setPage] = useState(0);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const [row, setRow] = useState<number | null>(null);
  const rows = useMlPredictionRows(datasetId, prediction.id, page * PAGE, PAGE, onlyFlagged);
  const save = useMlSavePrediction(datasetId);
  const created = save.data?.dataset_id ?? prediction.created_dataset_id;
  const data = rows.data;
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE)) : 1;
  const at = (name: string) => data?.columns.indexOf(name) ?? -1;
  return (
    <div className="space-y-2">
      {row !== null && <PredictedRowExplanation datasetId={datasetId} predictionId={prediction.id} row={row} />}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs text-ink-3">
        <span className="flex items-center gap-4">
          <span>{t("ml.predict.rowsHint")}</span>
          <Checkbox
            label={t("ml.predict.onlyLessReliable")}
            checked={onlyFlagged}
            onChange={(event) => {
              setOnlyFlagged(event.target.checked);
              setPage(0);
            }}
          />
        </span>
        <span className="flex items-center gap-3">
          <a href={api.mlPredictionCsvUrl(datasetId, prediction.id)} className="text-accent hover:underline">
            {t("ml.exportPredictions")}
          </a>
          {created ? (
            <Link to="/d/$datasetId" params={{ datasetId: created }} className="text-accent hover:underline">
              {t("ml.predict.openDataset")}
            </Link>
          ) : (
            <Button onClick={() => save.mutate(prediction.id)} disabled={save.isPending || !prediction.source_id}>
              {save.isPending ? t("ml.predict.saving") : t("ml.predict.save")}
            </Button>
          )}
        </span>
      </div>
      {save.isError && <ErrorNote>{save.error.message}</ErrorNote>}
      {rows.isError && <ErrorNote>{rows.error.message}</ErrorNote>}
      {data && (
        <>
          <div className="max-h-96 overflow-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px] tabular-nums">
              <thead className="sticky top-0 bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  {data.columns.map((name) => (
                    <th key={name} className={`px-3 py-1.5 font-normal ${name === "why" ? "" : "whitespace-nowrap"}`}>
                      {name === "row" ? t("ml.row") : name === "less_reliable" ? t("ml.predict.lessReliable") : name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.rows.map((values) => {
                  const number = Number(values[0]) - 1;
                  const flagged = values[at("less_reliable")] === true;
                  return (
                    <tr
                      key={number}
                      onClick={() => setRow(number)}
                      aria-selected={row === number}
                      className={`cursor-pointer border-t border-line hover:bg-surface-2 ${row === number ? "bg-accent-soft" : ""}`}
                    >
                      {values.map((value, j) => (
                        <td
                          key={data.columns[j]}
                          className={`px-3 py-1.5 ${data.columns[j] === "why" ? "min-w-48" : "whitespace-nowrap"} ${j === 0 ? "text-ink-3" : ""} ${flagged && data.columns[j] === "why" ? "text-warn" : ""}`}
                        >
                          {j === 0 ? (
                            <button type="button" className="underline-offset-2 hover:underline" aria-label={t("ml.explain.explainRow", { row: String(number + 1) })}>
                              {number + 1}
                            </button>
                          ) : (
                            <Cell value={value} name={data.columns[j] ?? ""} />
                          )}
                        </td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="flex items-center justify-end gap-2 text-xs text-ink-3 tabular-nums">
            <span>{t("ml.predict.page", { page: String(page + 1), pages: String(pages), total: data.total.toLocaleString("en-US") })}</span>
            <Button onClick={() => setPage(page - 1)} disabled={page === 0}>
              {t("ml.predict.previous")}
            </Button>
            <Button onClick={() => setPage(page + 1)} disabled={page + 1 >= pages}>
              {t("ml.predict.next")}
            </Button>
          </div>
        </>
      )}
      {run.task !== "regression" && run.classification?.threshold != null && (
        <p className="text-xs text-ink-3">{t("ml.predict.threshold", { threshold: run.classification.threshold.toFixed(2), positive: run.classification.positive ?? "" })}</p>
      )}
    </div>
  );
}

function Cell({ value, name }: { value: unknown; name: string }) {
  if (value === null || value === undefined) return <span className="text-ink-3">—</span>;
  if (name === "less_reliable") return <>{value === true ? t("ml.predict.yes") : ""}</>;
  if (typeof value === "number") return <>{name.startsWith("p(") ? `${(value * 100).toFixed(1)} %` : formatNumber(value)}</>;
  return <>{String(value)}</>;
}

/** The waterfall of one predicted row, as on the Predictions tab (020). */
function PredictedRowExplanation({ datasetId, predictionId, row }: { datasetId: string; predictionId: string; row: number }) {
  const found = useMlPredictionRowExplanation(datasetId, predictionId, row);
  if (found.isError) return <ErrorNote>{found.error.message}</ErrorNote>;
  if (!found.data) return <p className="text-[13px] text-ink-3">{t("ml.predict.explaining")}</p>;
  const e = found.data;
  if (e.unavailable || e.base === null) return <p className="text-[13px] text-ink-2">{e.unavailable}</p>;
  const label = t("ml.explain.row", { row: String(row + 1) });
  const strongest = e.steps.slice(0, 3).map((s) => `${s.label} (${s.value >= 0 ? "+" : ""}${formatNumber(s.value)})`);
  return (
    <div className="space-y-2">
      <p className="text-[13px] text-ink-1">{t("ml.predict.rowSentence", { row: String(row + 1), columns: strongest.join(", ") })}</p>
      <Plot figure={waterfallFigure(e.base, e.steps, e.scale ?? "prediction")} height={Math.max(240, 26 * (e.steps.length + 2) + 60)} label={label} />
      {e.positive && <p className="text-xs text-ink-3">{t("ml.explain.shapPositive", { positive: e.positive })}</p>}
    </div>
  );
}
