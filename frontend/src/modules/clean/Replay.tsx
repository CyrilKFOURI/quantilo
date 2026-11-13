import { CircleCheck, CircleMinus, CircleSlash } from "lucide-react";
import { useState, type ChangeEvent } from "react";

import { useDatasets } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useAdoptReplay, useLineage, useReplay, type CleanRecipe, type ReplayReport, type ReplayStep, type Step } from "./api";
import { describeStep } from "./steps";

const FILE = "file";
const ICONS = { applies: CircleCheck, unchanged: CircleMinus, skipped: CircleSlash } as const;

/** Replay a cleaning on this dataset (101, M5): a cleaned version's steps or a recipe file, checked
 *  step by step. Nothing is saved: "Use as draft" replaces the draft, read before any version
 *  (C-43). */
export function ReplayDialog({ datasetId, onClose, onUse }: { datasetId: string; onClose: () => void; onUse: (steps: Step[]) => void }) {
  const lineage = useLineage();
  const datasets = useDatasets();
  const names = new Map((datasets.data ?? []).map((dataset) => [dataset.id, dataset.name]));
  const [source, setSource] = useState("");
  const [recipe, setRecipe] = useState<CleanRecipe | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);
  const [proposals, setProposals] = useState(false);
  const replay = useReplay(datasetId);
  const adopt = useAdoptReplay(datasetId);
  const report = replay.data;
  const ready = source === FILE ? recipe !== null : source !== "";

  const read = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    void file.text().then((text) => {
      try {
        setRecipe(JSON.parse(text) as CleanRecipe);
        setFileError(null);
      } catch (error) {
        setRecipe(null);
        setFileError(t("clean.replay.badFile", { error: error instanceof Error ? error.message : String(error) }));
      }
    });
  };
  const check = () => replay.mutate(source === FILE ? { recipe, accept_proposals: proposals } : { version_id: source, accept_proposals: proposals });

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={t("clean.replay.title")}
      size="lg"
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("import.cancel")}
          </Button>
          <Button
            variant="primary"
            disabled={!report || replay.isPending}
            title={t("clean.replay.useHint")}
            onClick={() => {
              if (!report) return;
              onUse(report.adapted as Step[]);
              adopt.mutate(report); // N05: the recipe's dismissals are kept for this dataset
              onClose();
            }}
          >
            {t("clean.replay.use")}
          </Button>
        </>
      }
    >
      <div className="space-y-4">
        <p className="max-w-prose text-[13px] text-ink-2">{t("clean.replay.intro")}</p>
        <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,14rem),1fr))] items-end gap-3">
          <Field label={t("clean.replay.source")}>
            <Select value={source} onChange={(event) => setSource(event.target.value)}>
              <option value="">{t("clean.replay.pick")}</option>
              {lineage.map((item) => (
                <option key={item.dataset_id} value={item.dataset_id}>
                  {names.get(item.dataset_id) ?? t("clean.versions.name", { seq: String(item.seq) })}
                </option>
              ))}
              <option value={FILE}>{t("clean.replay.file")}</option>
            </Select>
          </Field>
          {source === FILE && (
            <Field label={t("clean.replay.fileLabel")}>
              <input type="file" accept=".json,application/json" onChange={read} className="text-[13px] text-ink-2 file:mr-2 file:rounded-md file:border file:border-line file:bg-surface-1 file:px-2 file:py-1 file:text-ink-1" />
            </Field>
          )}
          <Button variant="secondary" disabled={!ready || replay.isPending} onClick={check}>
            {t("clean.replay.check")}
          </Button>
        </div>
        <Checkbox label={t("clean.replay.proposals")} checked={proposals} onChange={(event) => setProposals(event.target.checked)} className="text-[13px] text-ink-2" />
        {fileError && <ErrorNote>{fileError}</ErrorNote>}
        {replay.error && <ErrorNote>{replay.error.message}</ErrorNote>}
        {replay.isPending && <Loading label={t("clean.replay.checking")} className="h-24 rounded-md" compact />}
        {report && !replay.isPending && <Report report={report} />}
      </div>
    </Dialog>
  );
}

export function Report({ report }: { report: ReplayReport }) {
  const columns = report.columns.filter((column) => column.status !== "same");
  return (
    <section aria-label={t("clean.replay.summary", { applies: String(report.applies), skipped: String(report.skipped) })} className="space-y-3">
      <p className="text-[13px] text-ink-1 tabular-nums">
        <span className="font-medium">{t("clean.replay.summary", { applies: String(report.applies), skipped: String(report.skipped) })}</span>
        {report.made_on && <span className="text-ink-3"> · {t("clean.replay.made", { recipe: report.recipe, name: report.made_on })}</span>}
      </p>
      {(columns.length > 0 || report.new_columns.length > 0) && (
        <div>
          <p className="text-xs font-medium text-ink-1">{t("clean.replay.columns")}</p>
          <ul className="mt-1 space-y-0.5 text-xs text-ink-2">
            {columns.map((column) => (
              <li key={column.recipe}>{t(`clean.replay.column.${column.status as "matched" | "proposed" | "missing"}`, { recipe: column.recipe, column: column.column ?? "" })}</li>
            ))}
            {report.new_columns.length > 0 && <li>{t("clean.replay.newColumns", { columns: report.new_columns.join(", ") })}</li>}
          </ul>
        </div>
      )}
      <ol className="divide-y divide-line rounded-lg border border-line">
        {report.steps.map((step) => (
          <StepLine key={step.id} step={step} adapted={report.adapted.find((item) => item.id === step.id) as Step | undefined} />
        ))}
      </ol>
      {report.check_report ? (
        <p className="text-xs text-ink-2">
          {t("clean.replay.checks", {
            summary: [
              t("clean.checks.pass", { count: formatCount(report.check_report.passed) }),
              report.check_report.failed > 0 && t(report.check_report.failed === 1 ? "clean.checks.failOne" : "clean.checks.fail", { count: formatCount(report.check_report.failed), rows: formatCount(report.check_report.failing_rows) }),
              report.check_report.errors > 0 && t("clean.checks.errors", { count: formatCount(report.check_report.errors) }),
            ]
              .filter(Boolean)
              .join(" · "),
          })}{" "}
          {t("clean.replay.checksAdopt")}
        </p>
      ) : null}
      {(report.hidden as number | undefined) ? <p className="text-xs text-ink-2">{t("clean.replay.hidden", { count: formatCount(report.hidden) })}</p> : null}
      {report.new_problems.length > 0 && (
        <div>
          <p className="text-xs font-medium text-ink-1">{t("clean.replay.newProblems")}</p>
          <ul className="mt-1 space-y-0.5 text-xs text-ink-2">
            {report.new_problems.map((problem) => (
              <li key={problem.id}>
                {problem.title} <span className="text-ink-3">· {problem.rule}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

function StepLine({ step, adapted }: { step: ReplayStep; adapted: Step | undefined }) {
  const Icon = ICONS[step.status];
  const counts = [step.cells > 0 && t("clean.count.cells", { count: formatCount(step.cells) }), step.rows > 0 && t("clean.count.rows", { count: formatCount(step.rows) })].filter(Boolean);
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-2 px-3 py-1.5">
      <Icon size={15} strokeWidth={1.5} className={`mt-0.5 ${step.status === "skipped" ? "text-warn" : "text-ink-3"}`} aria-hidden />
      <div className="min-w-0">
        <p className="text-[13px] text-ink-1">
          {adapted ? describeStep(adapted) : step.op}
          <span className="ml-2 text-xs text-ink-3">
            {t(`clean.replay.status.${step.status}`)}
            {counts.length > 0 && ` · ${counts.join(" · ")}`}
          </span>
        </p>
        {step.message && <p className="text-xs text-ink-2">{step.message}</p>}
      </div>
    </li>
  );
}
