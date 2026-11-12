import { Link, useNavigate, useParams, useSearch } from "@tanstack/react-router";
import { ArrowRight, Redo2, RotateCcw, Undo2 } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { useColumns, useDataset } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button, buttonClass } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
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
import { PinnedColumn } from "./PinnedColumn";
import type { CleanSearch, CleanTab } from "./search";
import { historyOf, push, redo, undo, withSuggestion, withoutSuggestion, type History } from "./steps";
import { StepsEditor } from "./StepsEditor";
import { SuggestionsList } from "./Suggestions";
import { Versions } from "./Versions";

export function CleanPage() {
  const { datasetId } = useParams({ strict: false }) as { datasetId: string };
  const dataset = useDataset(datasetId);
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
        ) : (
          // 096: Clean works on the dataset open, a cleaned version included (v1 → v2)
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
  const lineage = useLineage();
  const saveDraft = useSaveDraft(datasetId);
  const reset = useResetDraft(datasetId);
  const current = lineage.find((item) => item.dataset_id === datasetId);
  // Versions names the original, which is another dataset when a version is open (096)
  const original = useDataset(current ? current.parent_id : undefined);
  const originalName = current ? (original.data?.name ?? "") : name;

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
  // "Tick all in group" (096): one change, so one Ctrl+Z undoes it
  const tickMany = (many: Suggestion[], on: boolean) => {
    const ids = new Set(many.map((suggestion) => suggestion.id));
    update(on ? many.reduce((all, suggestion) => withSuggestion(all, suggestion, found), steps) : steps.filter((step) => !step.suggestion || !ids.has(step.suggestion)));
  };

  if (draft.isError) return <ErrorNote>{draft.error.message}</ErrorNote>;
  if (!server) return <Loading label={t("clean.loadingDraft")} className="h-48 rounded-lg" />;

  return (
    <div className="space-y-4">
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[13px] text-ink-1">
        <span className="font-medium">{current ? t("clean.working.version", { name: t("clean.versions.name", { seq: String(current.seq) }) }) : t("clean.working.original")}</span>
        <ArrowRight size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />
        <span className="text-ink-2">{family.data ? t("clean.working.next", { seq: String(nextSeq) }) : "…"}</span>
        <InfoTip label={t("clean.working.original")} text={t("clean.working.hint")} />
      </p>
      <Banner
        datasetId={datasetId}
        steps={steps}
        preview={whole}
        problems={Object.keys(problems).length}
        nextSeq={nextSeq}
        fixes={suggestions.data ? found.length : off ? 0 : null}
        onReview={tab === "suggestions" ? null : () => go("suggestions")}
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
            <SuggestionsList suggestions={found} steps={steps} onToggle={toggle} onTickMany={tickMany} />
          ))}
        {tab === "steps" && (
          <div className="space-y-4">
            <StepsEditor
              datasetId={datasetId}
              steps={steps}
              problems={problems}
              suggestions={found}
              columns={(columns.data ?? []).map((column) => column.name)}
              onChange={update}
            />
            {/* 099: the labels of pinned rows as a column, under Advanced */}
            <PinnedColumn datasetId={datasetId} steps={steps} columns={(columns.data ?? []).map((column) => column.name)} onChange={update} />
          </div>
        )}
        {tab === "versions" && <Versions family={family.data} originalName={originalName} />}
      </Tabs>
    </div>
  );
}

/** One of the three moves of the banner (CL-08): its number, its verb, then what it holds. */
function Move({ number, title, children }: { number: number; title: string; children: ReactNode }) {
  return (
    <li className="flex min-w-0 flex-col gap-1.5 px-3 py-2.5">
      <p className="flex items-center gap-2 text-[13px] font-medium text-ink-1">
        <span className="inline-flex size-5 shrink-0 items-center justify-center rounded-full border border-line text-[11px] font-normal text-ink-2 tabular-nums" aria-hidden>
          {number}
        </span>
        {title}
      </p>
      {children}
    </li>
  );
}

function Banner({
  datasetId,
  steps,
  preview,
  problems,
  nextSeq,
  fixes,
  onReview,
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
  /** Suggestions found (null while they are computed). */
  fixes: number | null;
  /** Back to the Suggestions tab, from another tab. */
  onReview: (() => void) | null;
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
  const review =
    fixes === null
      ? t("clean.flow.reviewLoading")
      : fixes === 0
        ? t("clean.flow.reviewNone")
        : fixes === 1
          ? t("clean.flow.reviewOne")
          : t("clean.flow.reviewBody", { count: formatCount(fixes) });

  return (
    <section aria-label={t("clean.banner.label")} className="rounded-lg border border-line bg-surface-1">
      <ol aria-label={t("clean.flow.label")} className="grid divide-line max-lg:divide-y lg:grid-cols-[minmax(0,1fr)_minmax(0,1.5fr)_minmax(0,1fr)] lg:divide-x">
        <Move number={1} title={t("clean.flow.review")}>
          <p className="text-xs text-ink-2 tabular-nums">{review}</p>
          {onReview && (
            <span>
              <Button variant="ghost" className="-ml-3" onClick={onReview}>
                {t("clean.flow.show")}
              </Button>
            </span>
          )}
        </Move>
        <Move number={2} title={t("clean.flow.apply")}>
          <p className="text-xs text-ink-2 tabular-nums" aria-live="polite">
            {facts.join(" · ")}
            {preview.isFetching && <span className="ml-2 text-ink-3">{t("clean.banner.computing")}</span>}
          </p>
          <span className="-ml-2 flex flex-wrap items-center gap-1">
            <Button variant="ghost" icon={Undo2} iconOnly aria-label={t("clean.undo")} title={`${t("clean.undo")} (Ctrl+Z)`} disabled={!history.past.length} onClick={() => onHistory(undo(history))} />
            <Button variant="ghost" icon={Redo2} iconOnly aria-label={t("clean.redo")} title={`${t("clean.redo")} (Ctrl+Shift+Z)`} disabled={!history.future.length} onClick={() => onHistory(redo(history))} />
            {!automatic && (
              <Button variant="ghost" icon={RotateCcw} onClick={onReset} disabled={resetting}>
                {t("clean.reset")}
              </Button>
            )}
          </span>
        </Move>
        <Move number={3} title={t("clean.flow.save")}>
          <span>
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
          <p className="text-xs text-ink-3">{t("clean.flow.saveHint")}</p>
        </Move>
      </ol>
      {(running || done?.created_dataset_id || failed) && (
        <div className="space-y-2 border-t border-line px-3 py-2.5">
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
            <div className="flex flex-wrap items-center gap-2" role="status">
              <p className="mr-1 text-[13px] text-ok">{t("clean.saved", { seq: String(savedSeq) })}</p>
              <Link to="/d/$datasetId" params={{ datasetId: done.created_dataset_id }} className={buttonClass("secondary")}>
                {t("clean.open")}
              </Link>
              <Button variant="ghost" onClick={onSaved}>
                {t("clean.tab.versions")}
              </Button>
            </div>
          )}
          {failed && <ErrorNote>{failed}</ErrorNote>}
        </div>
      )}
    </section>
  );
}
