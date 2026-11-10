import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { Redo2, RotateCcw, Undo2 } from "lucide-react";
import { useEffect, useState } from "react";

import { useColumns, useDataset } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Tabs } from "../../ui/Tabs";
import {
  useCancelJob,
  useCleanJob,
  useCleanSettings,
  useDraft,
  useFamily,
  useLineage,
  usePreview,
  useResetDraft,
  useSaveDraft,
  useSaveVersion,
  useSuggestions,
  type CleanDraft,
  type Step,
  type Suggestion,
} from "./api";
import type { CleanSearch, CleanTab } from "./search";
import { historyOf, push, redo, undo, withSuggestion, withoutSuggestion, type History } from "./steps";
import { StepsEditor } from "./StepsEditor";
import { SuggestionsList } from "./Suggestions";
import { Versions } from "./Versions";

export function CleanPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const dataset = useDataset(datasetId);
  const lineage = useLineage();
  const parent = lineage.find((item) => item.dataset_id === datasetId);
  return (
    <div className="flex flex-col">
      <header className="px-(--gutter) pt-6 pb-4">
        <h1 className="text-xl font-semibold tracking-tight">{t("clean.title")}</h1>
        <p className="mt-1 max-w-prose text-ink-2">{t("clean.description")}</p>
      </header>
      <div className="px-(--gutter) pb-10">
        {dataset.isError ? (
          <ErrorNote>{dataset.error.message}</ErrorNote>
        ) : !dataset.data ? (
          <Loading label={t("loading.default")} className="h-40 rounded-lg" />
        ) : dataset.data.status !== "ready" ? (
          <p className="text-ink-2">{t("clean.notReady")}</p>
        ) : parent ? (
          // C-13: the page always starts from the original
          <div className="max-w-prose space-y-3 rounded-lg border border-line bg-surface-1 p-4">
            <p className="text-[13px] text-ink-1">{t("clean.isVersion", { name: t("clean.versions.name", { seq: String(parent.seq) }) })}</p>
            <Link to="/d/$datasetId/clean" params={{ datasetId: parent.parent_id }} className="text-[13px] text-accent hover:underline">
              {t("clean.openOriginal")}
            </Link>
          </div>
        ) : (
          <Editor key={datasetId} datasetId={datasetId} name={dataset.data.name} />
        )}
      </div>
    </div>
  );
}

function typing(target: EventTarget | null) {
  return target instanceof HTMLElement && (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));
}

function Editor({ datasetId, name }: { datasetId: string; name: string }) {
  const search = useSearch({ strict: false }) as CleanSearch;
  const navigate = useNavigate();
  const tab: CleanTab = search.tab ?? "suggestions";
  const settings = useCleanSettings();
  const [asked, setAsked] = useState(false);
  const off = settings.data?.mode === "off" && !asked;
  const suggestions = useSuggestions(datasetId, settings.isFetched && !off);
  const draft = useDraft(datasetId, settings.isFetched);
  const columns = useColumns(datasetId, true);
  const family = useFamily(datasetId);
  const saveDraft = useSaveDraft(datasetId);
  const reset = useResetDraft(datasetId);

  // The page edits its own copy of the draft (undo history); a new draft from the server (first
  // load, Reset to automatic) starts a new history. Derived, never set in an effect.
  const [local, setLocal] = useState<{ base: CleanDraft | undefined; history: History }>({ base: undefined, history: historyOf([]) });
  const server = draft.data;
  const history = local.base === server ? local.history : historyOf(server?.steps ?? []);
  const steps = history.present;
  const commit = (next: History) => {
    setLocal({ base: server, history: next });
    if (next.present !== history.present) saveDraft.mutate(next.present);
  };
  const update = (next: Step[]) => commit(push(history, next));

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey) || event.key.toLowerCase() !== "z" || typing(event.target)) return;
      event.preventDefault();
      const next = event.shiftKey ? redo(history) : undo(history);
      if (next !== history) {
        setLocal({ base: server, history: next });
        saveDraft.mutate(next.present);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [history, server, saveDraft]);

  const whole = usePreview(datasetId, server ? steps : undefined, null);
  const problems = whole.data?.problems ?? {};
  const found = suggestions.data?.suggestions ?? [];
  const nextSeq = Math.max(0, ...(family.data?.versions ?? []).map((version) => version.seq)) + 1;
  const go = (next: CleanTab) => void navigate({ to: ".", search: next === "suggestions" ? {} : { tab: next }, replace: true });

  const toggle = (suggestion: Suggestion, on: boolean) =>
    update(on ? withSuggestion(steps, suggestion, found) : withoutSuggestion(steps, suggestion.id));
  const tickAll = (on: boolean) =>
    update(on ? found.reduce((all, suggestion) => withSuggestion(all, suggestion, found), steps) : steps.filter((step) => !step.suggestion));

  if (draft.isError) return <ErrorNote>{draft.error.message}</ErrorNote>;
  if (!server) return <Loading label={t("clean.loadingDraft")} className="h-48 rounded-lg" />;

  return (
    <div className="space-y-4">
      <Banner
        datasetId={datasetId}
        steps={steps}
        preview={whole}
        problems={Object.keys(problems).length}
        nextSeq={nextSeq}
        automatic={server.automatic && (local.base !== server || history.past.length === 0)}
        onReset={() => reset.mutate()}
        resetting={reset.isPending}
        history={history}
        onHistory={commit}
        onSaved={() => go("versions")}
      />
      {saveDraft.isError && <ErrorNote>{saveDraft.error.message}</ErrorNote>}
      <Tabs
        label={t("clean.tabs")}
        value={tab}
        onChange={go}
        items={[
          { value: "suggestions", label: t("clean.tab.suggestions"), badge: suggestions.data ? String(found.length) : undefined },
          { value: "steps", label: t("clean.tab.steps"), badge: String(steps.length) },
          { value: "versions", label: t("clean.tab.versions"), badge: family.data ? String(family.data.versions.length) : undefined },
        ]}
      >
        {tab === "suggestions" &&
          (off ? (
            <div className="flex flex-wrap items-center gap-3 rounded-lg border border-line bg-surface-1 p-4 text-[13px] text-ink-2">
              {t("clean.off")}
              <Button variant="primary" onClick={() => setAsked(true)}>
                {t("clean.find")}
              </Button>
            </div>
          ) : suggestions.isError ? (
            <ErrorNote>{suggestions.error.message}</ErrorNote>
          ) : !suggestions.data ? (
            <Loading label={t("clean.loadingSuggestions")} className="h-48 rounded-lg" />
          ) : (
            <SuggestionsList suggestions={found} steps={steps} onToggle={toggle} onTickAll={tickAll} />
          ))}
        {tab === "steps" && (
          <StepsEditor
            datasetId={datasetId}
            steps={steps}
            problems={problems}
            suggestions={found}
            columns={(columns.data ?? []).map((column) => column.name)}
            onChange={update}
          />
        )}
        {tab === "versions" && <Versions family={family.data} originalName={name} />}
      </Tabs>
    </div>
  );
}

function Banner({
  datasetId,
  steps,
  preview,
  problems,
  nextSeq,
  automatic,
  onReset,
  resetting,
  history,
  onHistory,
  onSaved,
}: {
  datasetId: string;
  steps: Step[];
  preview: ReturnType<typeof usePreview>;
  problems: number;
  nextSeq: number;
  automatic: boolean;
  onReset: () => void;
  resetting: boolean;
  history: History;
  onHistory: (history: History) => void;
  onSaved: () => void;
}) {
  const save = useSaveVersion(datasetId);
  const [jobId, setJobId] = useState<string | null>(null);
  const [savedSeq, setSavedSeq] = useState(nextSeq);
  const job = useCleanJob(jobId);
  const cancel = useCancelJob();
  const running = save.isPending || job.data?.status === "running" || (Boolean(jobId) && !job.data);
  const done = job.data?.status === "done" ? job.data : null;
  const failed = save.error?.message ?? (job.data && ["failed", "cancelled"].includes(job.data.status) ? job.data.error : null);
  const enabled = steps.filter((step) => step.enabled);
  const auto = enabled.filter((step) => step.origin === "auto").length;
  const facts = [
    steps.length === 0
      ? t("clean.banner.noSteps")
      : steps.length === 1
        ? t("clean.banner.oneStep")
        : t("clean.banner.steps", { count: String(steps.length), automatic: String(auto) }),
    preview.data && steps.length > 0 && t("clean.banner.cells", { count: formatCount(preview.data.cells_changed) }),
    preview.data && preview.data.rows_removed > 0 && t("clean.banner.rows", { count: formatCount(preview.data.rows_removed) }),
    problems > 0 && t("clean.banner.problems", { count: String(problems) }),
  ].filter(Boolean);

  return (
    <section aria-label={t("clean.banner.label")} className="space-y-2 rounded-lg border border-line bg-surface-1 px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <p className="min-w-0 flex-1 text-[13px] text-ink-1 tabular-nums max-sm:basis-full" aria-live="polite">
          <span className="font-medium">{t("clean.banner.label")}</span> · {facts.join(" · ")}
          {preview.isFetching && <span className="ml-2 text-xs text-ink-3">{t("clean.banner.computing")}</span>}
        </p>
        <span className="flex flex-wrap items-center gap-1.5">
          <Button variant="ghost" icon={Undo2} iconOnly aria-label={t("clean.undo")} title={`${t("clean.undo")} (Ctrl+Z)`} disabled={!history.past.length} onClick={() => onHistory(undo(history))} />
          <Button variant="ghost" icon={Redo2} iconOnly aria-label={t("clean.redo")} title={`${t("clean.redo")} (Ctrl+Shift+Z)`} disabled={!history.future.length} onClick={() => onHistory(redo(history))} />
          {!automatic && (
            <Button icon={RotateCcw} onClick={onReset} disabled={resetting}>
              {t("clean.reset")}
            </Button>
          )}
          <Button
            variant="primary"
            disabled={running || enabled.length === 0 || problems > 0}
            onClick={() =>
              save.mutate(steps, {
                onSuccess: (started) => {
                  setSavedSeq(nextSeq);
                  setJobId(started.id);
                },
              })
            }
          >
            {t("clean.save", { seq: String(nextSeq) })}
          </Button>
        </span>
      </div>
      {running && (
        <div className="flex items-center gap-3">
          <Loading label={job.data?.stage === "importing" ? t("clean.importing") : t("clean.saving")} className="h-9 flex-1 rounded-md" compact />
          {jobId && (
            <Button variant="ghost" onClick={() => cancel.mutate(jobId)}>
              {t("clean.cancel")}
            </Button>
          )}
        </div>
      )}
      {done?.created_dataset_id && (
        <p className="flex flex-wrap items-center gap-2 text-[13px] text-ok" role="status">
          {t("clean.saved", { seq: String(savedSeq) })}
          <Link to="/d/$datasetId" params={{ datasetId: done.created_dataset_id }} className="text-accent hover:underline">
            {t("clean.open")}
          </Link>
          <button type="button" onClick={onSaved} className="text-accent hover:underline">
            {t("clean.tab.versions")}
          </button>
        </p>
      )}
      {failed && <ErrorNote>{failed}</ErrorNote>}
    </section>
  );
}
