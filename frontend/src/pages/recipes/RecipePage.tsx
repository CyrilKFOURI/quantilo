import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { FileCode2, FileJson, Play, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";

import { recipeUrls, useRecipe, useRecipeActions, type FigureHistory } from "../../api/recipes";
import { Plot } from "../../charts/Plot";
import { formatRelativeTime } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { Checkbox, ErrorNote } from "../../ui/fields";
import { historyFigure } from "./figures";
import { StatusTag, figureKindLabel, kindLabel } from "./parts";
import { RunDialog } from "./RunDialog";
import { RunReport } from "./RunReport";
import { Loading } from "../../ui/Loading";

export type RecipeSearch = { run?: string; report?: string };

const ID = /^[\w-]{1,64}$/;

export function validateRecipeSearch(search: Record<string, unknown>): RecipeSearch {
  const id = (value: unknown) => (typeof value === "string" && ID.test(value) ? value : undefined);
  return { run: id(search.run), report: id(search.report) };
}

const linkClass =
  "inline-flex h-(--control-h) shrink-0 items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2";

export function RecipePage() {
  const { recipeId } = useParams({ strict: false }) as { recipeId: string };
  const search = useSearch({ strict: false }) as RecipeSearch;
  const navigate = useNavigate();
  const recipe = useRecipe(recipeId);
  const actions = useRecipeActions(recipeId);
  const [running, setRunning] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const set = (patch: RecipeSearch) =>
    void navigate({ to: ".", search: (previous: RecipeSearch) => ({ ...previous, ...patch }), replace: true });

  if (recipe.isError) {
    return (
      <div className="mx-auto max-w-5xl px-(--gutter) py-8">
        <ErrorNote>{recipe.error.message}</ErrorNote>
      </div>
    );
  }
  const data = recipe.data;
  if (!data) return <Loading label={t("loading.recipe")} className="mx-(--gutter) my-8 h-64 rounded-lg" />;
  const shown = search.report ?? data.runs[0]?.id;
  const charts = data.history.filter((line) => line.points.filter((p) => p.value !== null).length >= 2);
  const dialogOpen = running || Boolean(search.run);
  const editError = actions.edit.error ?? actions.remove.error;

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-7 px-(--gutter) py-8">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-xs text-ink-3">
            <Link to="/recipes" className="hover:text-accent hover:underline">
              {t("recipes.title")}
            </Link>
          </p>
          <h1 className="mt-0.5 text-xl font-semibold tracking-tight">{data.name}</h1>
          <p className="mt-1 text-ink-2">
            {t("recipes.version", { version: String(data.version) })}
            {data.source ? ` · ${t("recipes.madeOn", { dataset: data.source })}` : ""}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="primary" icon={Play} onClick={() => setRunning(true)}>
            {t("recipes.runOn")}
          </Button>
          <a href={recipeUrls.python(recipeId)} download className={linkClass}>
            <FileCode2 size={15} strokeWidth={1.5} aria-hidden />
            {t("recipes.exportPython")}
          </a>
          <a href={recipeUrls.file(recipeId)} download className={linkClass}>
            <FileJson size={15} strokeWidth={1.5} aria-hidden />
            {t("recipes.exportFile")}
          </a>
          <Button variant="ghost" icon={Trash2} iconOnly aria-label={t("recipes.delete")} title={t("recipes.delete")} onClick={() => setDeleting(true)} />
        </div>
      </header>
      {editError && <ErrorNote>{editError.message}</ErrorNote>}

      {charts.length > 0 && (
        <section aria-label={t("recipes.history.title")}>
          <h2 className="mb-2 text-xs text-ink-3">{t("recipes.history.title")}</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {charts.map((line) => (
              <HistoryCard key={line.id} line={line} />
            ))}
          </div>
        </section>
      )}

      <section aria-label={t("recipes.runs.title")}>
        <div className="mb-2 flex items-center justify-between gap-3">
          <h2 className="text-xs text-ink-3">{t("recipes.runs.title")}</h2>
          {data.runs.length > 0 && (
            <a href={recipeUrls.history(recipeId)} download className="text-xs text-accent hover:underline">
              {t("recipes.runs.csv")}
            </a>
          )}
        </div>
        {data.runs.length === 0 ? (
          <p className="text-ink-2">{t("recipes.runs.none")}</p>
        ) : (
          <div className="grid gap-5 lg:grid-cols-[minmax(0,280px)_minmax(0,1fr)]">
            <ul className="space-y-0.5" aria-label={t("recipes.runs.title")}>
              {data.runs.map((run) => (
                <li key={run.id}>
                  <button
                    type="button"
                    onClick={() => set({ report: run.id })}
                    aria-current={run.id === shown ? "true" : undefined}
                    className={`flex w-full flex-col items-start gap-0.5 rounded-md px-2.5 py-1.5 text-left hover:bg-surface-2 ${
                      run.id === shown ? "bg-accent-soft" : ""
                    }`}
                  >
                    <span className="flex w-full items-center justify-between gap-2">
                      <span className="truncate text-ink-1">{run.dataset_name}</span>
                      <StatusTag status={run.status} />
                    </span>
                    <span className="text-xs text-ink-3">
                      {formatRelativeTime(run.created_at)} · v{run.version}
                      {run.kind === "baseline" ? ` · ${t("recipes.runs.baseline")}` : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
            {shown && <RunReport key={shown} recipeId={recipeId} runId={shown} />}
          </div>
        )}
      </section>

      <section aria-label={t("recipes.steps.title")}>
        <h2 className="mb-2 text-xs text-ink-3">{t("recipes.steps.title")}</h2>
        <ol className="divide-y divide-line rounded-lg border border-line bg-surface-1">
          {data.step_views.map((step) => (
            <li key={step.id} className="flex items-start gap-3 px-3 py-2">
              <Checkbox
                checked={step.enabled}
                disabled={actions.edit.isPending}
                aria-label={t("recipes.steps.toggle", { step: step.sentence })}
                onChange={(event) => actions.edit.mutate({ enabled: { [step.id]: event.target.checked } })}
              />
              <span className="w-28 shrink-0 text-xs text-ink-3">{kindLabel(step.kind)}</span>
              <span className={`min-w-0 flex-1 ${step.enabled ? "text-ink-1" : "text-ink-3 line-through"}`}>
                {step.sentence}
              </span>
              <Button
                variant="ghost"
                icon={Trash2}
                iconOnly
                aria-label={t("recipes.steps.remove", { step: step.sentence })}
                title={t("recipes.steps.removeShort")}
                disabled={actions.edit.isPending}
                onClick={() => actions.edit.mutate({ remove: [step.id] })}
              />
            </li>
          ))}
        </ol>
        <p className="mt-1.5 text-xs text-ink-3">{t("recipes.steps.versioned")}</p>
      </section>

      {data.recipe && data.recipe.columns.length > 0 && (
        <section aria-label={t("recipes.columns.title")}>
          <h2 className="mb-2 text-xs text-ink-3">
            {t("recipes.columns.title")} ({data.recipe.columns.length})
          </h2>
          <ul className="flex flex-wrap gap-1.5">
            {data.recipe.columns.map((column) => (
              <li key={column.name} className="rounded-md border border-line bg-surface-1 px-2 py-0.5 text-xs">
                <span className="font-mono text-ink-1">{column.name}</span>
                <span className="ml-1.5 text-ink-3">{t(`type.${column.type}` as never)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {dialogOpen && (
        <RunDialog
          recipe={data}
          initialDataset={search.run ?? null}
          onClose={() => {
            setRunning(false);
            set({ run: undefined });
          }}
          onStarted={(runId) => {
            setRunning(false);
            set({ run: undefined, report: runId });
          }}
        />
      )}
      <Dialog
        open={deleting}
        onOpenChange={setDeleting}
        title={t("recipes.deleteTitle", { name: data.name })}
        description={t("recipes.deleteBody")}
        size="sm"
        footer={
          <div className="flex justify-end gap-2">
            <Button onClick={() => setDeleting(false)}>{t("recipes.cancel")}</Button>
            <Button
              variant="danger"
              disabled={actions.remove.isPending}
              onClick={() =>
                actions.remove.mutate(undefined, { onSuccess: () => void navigate({ to: "/recipes" }) })
              }
            >
              {t("recipes.delete")}
            </Button>
          </div>
        }
      >
        <p className="text-ink-2">{t("recipes.deleteDetail")}</p>
      </Dialog>
    </div>
  );
}

function HistoryCard({ line }: { line: FigureHistory }) {
  const figure = useMemo(() => historyFigure(line), [line]);
  const runs = line.points.filter((point) => point.value !== null).map((point) => point.dataset_name);
  return (
    <figure className="rounded-lg border border-line bg-surface-1 px-3 pt-2.5 pb-1">
      <figcaption className="text-[13px] font-medium text-ink-1">
        {line.label}
        <span className="ml-1.5 text-xs font-normal text-ink-3">
          {figureKindLabel(line.kind)}
          {line.detail && line.kind !== "forecast" ? ` · ${line.detail}` : ""}
        </span>
      </figcaption>
      <Plot figure={figure} height={130} label={t("recipes.history.chart", { figure: line.label })} />
      <p className="truncate pb-1 text-xs text-ink-3" title={runs.join(" → ")}>
        {t("recipes.history.span", { first: runs[0] ?? "", last: runs.at(-1) ?? "", count: String(runs.length) })}
      </p>
    </figure>
  );
}
