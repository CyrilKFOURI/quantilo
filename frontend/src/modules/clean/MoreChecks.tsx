import { useQueryClient } from "@tanstack/react-query";
import { useId, useState } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import {
  cleanKeys,
  useCancelExpert,
  useExpertJob,
  useExpertMethods,
  useExpertResults,
  useRunExpert,
  type CellToMark,
  type ExpertMethod,
  type ExpertOptions,
  type ExpertResult,
  type Step,
  type Suggestion,
} from "./api";
import { groupSuggestions } from "./groups";
import { Groups, type TickMany, type Toggle } from "./Suggestions";

/** Advanced › More checks (101, M6): the methods that learn from the data. The list is read from
 *  the columns only; a method runs at its Run, one at a time, and can be stopped (C-29, C-31).
 *  What it finds is a suggestion never ticked (C-30), in "Needs your decision". */
export function MoreChecks({ datasetId, steps, onToggle, onTickMany }: { datasetId: string; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const methods = useExpertMethods(datasetId);
  const results = useExpertResults(datasetId);
  if (methods.isError) return <ErrorNote>{methods.error.message}</ErrorNote>;
  if (!methods.data) return <Loading label={t("clean.expert.loading")} className="h-24 rounded-md" compact />;
  return (
    <div className="space-y-3">
      <p className="max-w-prose text-xs text-ink-2">{t("clean.expert.intro")}</p>
      <ul className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface-1">
        {methods.data.map((method) => (
          <li key={method.id}>
            <Check datasetId={datasetId} method={method} result={results[method.id] ?? null} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
          </li>
        ))}
      </ul>
    </div>
  );
}

const ALL = "";
type Mark = "error" | "ok" | "skip";

function Check({
  datasetId,
  method,
  result,
  steps,
  onToggle,
  onTickMany,
}: {
  datasetId: string;
  method: ExpertMethod;
  result: ExpertResult | null;
  steps: Step[];
  onToggle: Toggle;
  onTickMany: TickMany;
}) {
  const headingId = useId();
  const title = t(`clean.expert.${method.id}.title` as MessageKey);
  const [column, setColumn] = useState(method.id === "fill_empty" ? ALL : (method.columns[0] ?? ""));
  const [imputer, setImputer] = useState<"knn" | "iterative">("knn");
  const [marks, setMarks] = useState<Record<number, Mark>>({});
  const [ranOn, setRanOn] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const run = useRunExpert(datasetId);
  const job = useExpertJob(datasetId, jobId);
  const cancel = useCancelExpert();
  const client = useQueryClient();
  const running = run.isPending || (jobId !== null && (!job.data || job.data.status === "running"));
  const stopped = job.data?.status === "cancelled" ? job.data.error : null;
  const failed = run.error?.message ?? (job.data?.status === "failed" ? job.data.error : null);

  const start = (options: ExpertOptions) =>
    run.mutate(
      { method: method.id, options },
      {
        onSuccess: (started) => {
          setRanOn(options.column ?? null);
          setJobId(started.id);
        },
      },
    );
  const options = (): ExpertOptions => {
    if (method.id === "fill_empty") return { imputer, columns: column === ALL ? null : [column] };
    if (method.id === "unusual_rows") return {};
    return { column };
  };
  const clear = () => {
    client.setQueryData(cleanKeys.expert(datasetId, method.id), null);
    setMarks({});
    setJobId(null);
  };
  const toMark = method.id === "cell_errors" && result && !running ? result.to_mark : [];
  const counted = Object.values(marks);
  const ready = counted.includes("error") && counted.includes("ok");
  const learn = () =>
    start({
      column: ranOn ?? column,
      marks: Object.entries(marks)
        .filter(([, mark]) => mark !== "skip")
        .map(([row, mark]) => ({ row: Number(row), error: mark === "error" })),
    });

  return (
    <section aria-labelledby={headingId} className="space-y-2.5 px-3 py-3">
      <header className="flex flex-wrap items-start gap-x-3 gap-y-2">
        <div className="min-w-0 flex-1 basis-64">
          <h4 id={headingId} className="text-[13px] font-medium text-ink-1">
            {title} <span className="font-normal text-ink-3">{method.rule}</span>
          </h4>
          <p className="mt-0.5 max-w-prose text-xs text-ink-2">{t(`clean.expert.${method.id}.hint` as MessageKey)}</p>
          <p className="mt-0.5 text-[11px] text-ink-3">{t(`clean.expert.${method.id}.cost` as MessageKey)}</p>
        </div>
      </header>
      {!method.available ? (
        <p className="text-xs text-ink-3">{t("clean.expert.unavailable", { why: method.why_not ?? "" })}</p>
      ) : (
        <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,12rem),1fr))] items-end gap-3">
          {(method.id === "suspicious_labels" || method.id === "cell_errors") && (
            <Field label={t(method.id === "suspicious_labels" ? "clean.expert.target" : "clean.expert.column")}>
              <Select value={column} onChange={(event) => setColumn(event.target.value)} disabled={running}>
                {method.columns.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </Select>
            </Field>
          )}
          {method.id === "fill_empty" && (
            <>
              <Field label={t("clean.expert.columns")}>
                <Select value={column} onChange={(event) => setColumn(event.target.value)} disabled={running}>
                  <option value={ALL}>{t("clean.expert.allColumns")}</option>
                  {method.columns.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </Select>
              </Field>
              <div className="flex flex-col gap-1">
                <span className="text-xs text-ink-2">{t("clean.expert.imputer")}</span>
                <SegmentedControl
                  label={t("clean.expert.imputer")}
                  value={imputer}
                  onChange={setImputer}
                  options={[
                    { value: "knn", label: t("clean.expert.imputer.knn") },
                    { value: "iterative", label: t("clean.expert.imputer.iterative") },
                  ]}
                />
              </div>
            </>
          )}
          <span className="flex items-center gap-2">
            <Button variant="secondary" disabled={running} onClick={() => start(options())}>
              {result ? t("clean.expert.runAgain") : t("clean.expert.run")}
            </Button>
            {result && !running && (
              <Button variant="ghost" title={t("clean.expert.clearHint")} onClick={clear}>
                {t("clean.expert.clear")}
              </Button>
            )}
          </span>
        </div>
      )}
      {running && (
        <div className="flex items-center gap-3">
          <Loading label={t("clean.expert.running", { title })} className="h-9 flex-1 rounded-md" compact />
          {jobId && (
            <Button variant="ghost" onClick={() => cancel.mutate(jobId)}>
              {t("clean.expert.cancel")}
            </Button>
          )}
        </div>
      )}
      {stopped && !running && <p className="text-xs text-ink-3">{stopped}</p>}
      {failed && !running && <ErrorNote>{failed}</ErrorNote>}
      {result && !running && <Result result={result} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />}
      {toMark.length > 0 && <Marks cells={toMark} marks={marks} onMark={(row, mark) => setMarks({ ...marks, [row]: mark })} ready={ready} onLearn={learn} count={counted.filter((mark) => mark !== "skip").length} />}
    </section>
  );
}

function Result({ result, steps, onToggle, onTickMany }: { result: ExpertResult; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const facts = t(result.sampled ? "clean.expert.rowsSampled" : "clean.expert.rows", { rows: formatCount(result.rows), seconds: String(result.seconds) });
  return (
    <div className="space-y-2" role="status">
      {result.note && <p className="max-w-prose text-xs text-ink-2">{result.note}</p>}
      {!result.note && result.suggestions.length === 0 && result.to_mark.length === 0 && <p className="text-xs text-ink-2">{t("clean.expert.nothing")}</p>}
      {result.suggestions.length > 0 && <Groups groups={groupSuggestions(result.suggestions as Suggestion[])} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />}
      <p className="text-[11px] text-ink-3 tabular-nums">{facts}</p>
    </div>
  );
}

/** One cell per group to mark Error or OK (S7); the marks stay until the results are cleared. */
function Marks({ cells, marks, onMark, ready, onLearn, count }: { cells: CellToMark[]; marks: Record<number, Mark>; onMark: (row: number, mark: Mark) => void; ready: boolean; onLearn: () => void; count: number }) {
  return (
    <div className="space-y-2">
      <div>
        <p className="text-[13px] font-medium text-ink-1">{t("clean.expert.marks.title")}</p>
        <p className="text-xs text-ink-2">{t("clean.expert.marks.hint")}</p>
      </div>
      <ul className="divide-y divide-line rounded-md border border-line">
        {cells.map((cell) => (
          <li key={cell.row} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-2.5 py-1.5">
            <span className="w-16 shrink-0 text-[11px] text-ink-3 tabular-nums">{t("clean.expert.marks.row", { row: formatCount(cell.row + 1) })}</span>
            <span className="min-w-0 flex-1 basis-32 truncate font-mono text-xs text-ink-1">{cell.value ?? t("clean.empty")}</span>
            <span className="min-w-0 basis-48 text-[11px] text-ink-3">{t("clean.expert.marks.alike", { hint: cell.hint, count: formatCount(cell.alike) })}</span>
            <SegmentedControl
              label={t("clean.expert.marks.label", { row: formatCount(cell.row + 1) })}
              value={marks[cell.row] ?? "skip"}
              onChange={(mark) => onMark(cell.row, mark)}
              options={[
                { value: "error", label: t("clean.expert.marks.error") },
                { value: "ok", label: t("clean.expert.marks.ok") },
                { value: "skip", label: t("clean.expert.marks.skip") },
              ]}
            />
          </li>
        ))}
      </ul>
      <Button variant="secondary" disabled={!ready} onClick={onLearn}>
        {t("clean.expert.marks.learn", { count: String(count) })}
      </Button>
    </div>
  );
}
