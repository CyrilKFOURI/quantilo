import { Link } from "@tanstack/react-router";
import { Download, Square } from "lucide-react";

import type { ChangeLevel } from "../../api/client";
import { recipeUrls, useRecipeActions, useRecipeRun, type StepResult } from "../../api/recipes";
import { formatNumber } from "../../charts/figures";
import { LevelBadge } from "../../features/changes/ChangesBanner";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, ProgressBar } from "../../ui/fields";
import { Change, StatusTag, figureKindLabel, stepRoute } from "./parts";
import { Loading } from "../../ui/Loading";

/** A run's report: progress while it runs (cancellable), then what changed, the key figures
 *  next to the previous run's, and each step with its status, reason and link. */
export function RunReport({ recipeId, runId }: { recipeId: string; runId: string }) {
  const run = useRecipeRun(runId);
  const { cancel } = useRecipeActions(recipeId);
  if (run.isError) return <ErrorNote>{run.error.message}</ErrorNote>;
  const data = run.data;
  if (!data) return <Loading label={t("loading.report")} className="h-40 rounded-lg" />;
  const steps = data.report?.steps ?? [];
  const finished = steps.filter((step) => !["waiting", "running"].includes(step.status)).length;
  const running = data.status === "running";
  const datasetId = data.dataset_id;

  return (
    <section aria-label={t("recipes.report.title")} className="space-y-5 rounded-lg border border-line bg-surface-1 p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold">
            {data.kind === "baseline"
              ? t("recipes.report.baseline", { dataset: data.dataset_name })
              : t("recipes.report.on", { dataset: data.dataset_name })}
          </h2>
          <p className="mt-0.5 text-xs text-ink-3">
            {new Date(data.created_at).toLocaleString()} · {t("recipes.version", { version: String(data.version) })}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusTag status={data.status} />
          {running && (
            <Button icon={Square} onClick={() => cancel.mutate(runId)} disabled={cancel.isPending}>
              {t("recipes.report.cancel")}
            </Button>
          )}
        </div>
      </header>

      {running && steps.length > 0 && (
        <div className="space-y-1">
          <ProgressBar value={finished / steps.length} label={t("recipes.report.progress")} />
          <p className="text-xs text-ink-3">
            {t("recipes.report.stepsDone", { done: String(finished), total: String(steps.length) })}
          </p>
        </div>
      )}
      {data.error && <ErrorNote>{data.error}</ErrorNote>}

      {data.changes && datasetId && (
        <div className="flex items-start gap-3 rounded-md border border-line px-3 py-2.5">
          <LevelBadge level={data.changes.level as ChangeLevel} />
          <p className="min-w-0 flex-1 text-ink-1">{data.changes.headline}</p>
          <Link
            to="/d/$datasetId/changes"
            params={{ datasetId }}
            search={{ vs: data.changes.vs ?? undefined, tab: undefined, column: undefined }}
            className="shrink-0 text-xs text-accent hover:underline"
          >
            {t("changes.seeDetail")}
          </Link>
        </div>
      )}

      {data.compared.length > 0 && (
        <div className="overflow-x-auto">
          <h3 className="mb-1.5 text-xs text-ink-3">{t("recipes.report.figures")}</h3>
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-line text-left text-xs text-ink-3">
                <th scope="col" className="py-1.5 pr-3 font-normal">{t("recipes.report.figure")}</th>
                <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("recipes.report.value")}</th>
                <th scope="col" className="px-3 py-1.5 text-right font-normal">{t("recipes.report.previous")}</th>
                <th scope="col" className="py-1.5 pl-3 text-right font-normal">{t("recipes.report.change")}</th>
              </tr>
            </thead>
            <tbody>
              {data.compared.map((figure) => (
                <tr key={figure.id} className="border-b border-line last:border-0">
                  <td className="py-1.5 pr-3">
                    <span className="text-ink-1">{figure.label}</span>
                    <span className="ml-2 text-xs text-ink-3">
                      {figureKindLabel(figure.kind)}
                      {figure.detail ? ` · ${figure.detail}` : ""}
                    </span>
                  </td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-ink-1">{formatNumber(figure.value)}</td>
                  <td className="px-3 py-1.5 text-right tabular-nums text-ink-2">{formatNumber(figure.previous)}</td>
                  <td className="py-1.5 pl-3 text-right"><Change figure={figure} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {steps.length > 0 && (
        <div>
          <h3 className="mb-1.5 text-xs text-ink-3">{t("recipes.report.steps")}</h3>
          <ol className="divide-y divide-line">
            {steps.map((step) => (
              <StepRow key={step.id} step={step} runId={runId} datasetId={datasetId} />
            ))}
          </ol>
        </div>
      )}

      {data.report && data.report.new_columns.length > 0 && (
        <p className="text-xs text-ink-2">
          {t("recipes.report.newColumns")}{" "}
          <span className="font-mono">{data.report.new_columns.join(", ")}</span>
        </p>
      )}
    </section>
  );
}

function StepRow({ step, runId, datasetId }: { step: StepResult; runId: string; datasetId: string | null }) {
  const route = step.link && datasetId ? stepRoute(datasetId, step.link) : null;
  return (
    <li className="flex flex-wrap items-start gap-x-3 gap-y-1 py-2">
      <span className="w-36 shrink-0">
        <StatusTag status={step.status} />
      </span>
      <div className="min-w-0 flex-1">
        <p className="text-ink-1">{step.sentence}</p>
        {step.message && <p className="text-xs text-ink-2">{step.message}</p>}
        {step.warnings.map((warning) => (
          <p key={warning} className="text-xs text-ink-2">
            {warning}
          </p>
        ))}
        {step.files.length > 0 && (
          <p className="mt-1 flex flex-wrap gap-3">
            {step.files.map((name) => (
              <a
                key={name}
                href={recipeUrls.output(runId, name)}
                download
                className="inline-flex items-center gap-1 text-xs text-accent hover:underline"
              >
                <Download size={12} strokeWidth={1.75} aria-hidden />
                {name}
              </a>
            ))}
          </p>
        )}
      </div>
      {route && step.link && (
        <Link
          to={route.to}
          params={route.params as never}
          search={route.search as never}
          className="shrink-0 text-xs text-accent hover:underline"
        >
          {step.link.label || t("recipes.report.open")}
        </Link>
      )}
    </li>
  );
}
