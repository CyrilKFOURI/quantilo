import { keepPreviousData, useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import { request } from "../../api/client";
import type { components } from "../../api/schema";
import { nextCheckId, sameCheck } from "./checkRules";
import { CLEAN_ENABLED } from "./flag";

/** Hooks of the Clean module (065), kept here rather than in api/hooks.ts (another branch edits
 *  it). Routes: /api/v1/modules/clean (backend/app/modules/clean/router.py). */

type Schemas = components["schemas"];
/** The server sends every field, defaults included: make them all required, deeply. */
type Complete<T> = T extends (infer Item)[] ? Complete<Item>[] : T extends object ? { [K in keyof T]-?: Complete<T[K]> } : T;

export type CleanDraft = Complete<Schemas["CleanDraft"]>;
export type Step = CleanDraft["steps"][number];
export type Op = Step["op"];
export type StepOf<O extends Op> = Extract<Step, { op: O }>;
export type RowCondition = Complete<Schemas["RowCondition"]>;
export type Suggestion = Complete<Schemas["CleanSuggestion"]>;
export type Suggestions = Complete<Schemas["CleanSuggestions"]>;
export type CleanPreview = Complete<Schemas["CleanPreview"]>;
export type PreviewRow = CleanPreview["sample"][number];
export type ColumnDiff = CleanPreview["columns"][number];
export type CleanJob = Complete<Schemas["CleanJob"]>;
export type CleanFamily = Complete<Schemas["CleanFamily"]>;
export type CleanVersion = CleanFamily["versions"][number];
export type CleanReport = Complete<Schemas["CleanReport"]>;
export type LineageItem = Complete<Schemas["CleanLineageItem"]>;
export type CleanMode = Complete<Schemas["CleanSettings"]>["mode"];
// 101: the Expert checks (More checks) and the replay of a cleaning
export type ExpertMethod = Complete<Schemas["ExpertMethod"]>;
export type MethodId = ExpertMethod["id"];
export type ExpertOptions = Partial<Schemas["ExpertOptions"]>; // the server gives every default
export type ExpertResult = Complete<Schemas["ExpertResult"]>;
export type CellToMark = ExpertResult["to_mark"][number];
export type ExpertJob = Complete<Schemas["ExpertJob"]>;
export type CleanRecipe = Schemas["CleanRecipe"];
export type ReplayReport = Complete<Schemas["ReplayReport"]>;
export type ReplayStep = ReplayReport["steps"][number];
export type ReplayRequest = { version_id?: string | null; recipe?: CleanRecipe | null; accept_proposals?: boolean };
// N05: "Don't suggest again"
export type CleanDismissal = Complete<Schemas["CleanDismissal"]>;
export type DismissScope = CleanDismissal["scope"];
export type HiddenSuggestion = Complete<Schemas["HiddenSuggestion"]>;
export type DismissBody = Schemas["Dismissal"];
export type Dismissed = Complete<Schemas["Dismissed"]>;

// N31: declared checks
export type CleanCheck = Complete<Schemas["CleanCheck"]>;
export type CheckKind = CleanCheck["kind"];
export type CleanChecks = Complete<Schemas["CleanChecks"]>;
export type CheckReport = Complete<Schemas["CheckReport"]>;
export type CheckResult = CheckReport["results"][number];
export type CheckRun = Complete<Schemas["CheckRun"]>;
export type FailingRows = Complete<Schemas["FailingRows"]>;

// N46: the cleaning journal
export type JournalEntry = Complete<Schemas["JournalEntry"]>;
export type CleanJournal = Complete<Schemas["CleanJournal"]>;

// N35: the groups of a removal of duplicates, side by side
export type DuplicateGroups = Complete<Schemas["DuplicateGroups"]>;
export type DuplicateGroup = DuplicateGroups["groups"][number];
// N36: the clusters of an entity number (A-20), side by side
export type EntityClusters = Complete<Schemas["EntityClusters"]>;
export type EntityCluster = EntityClusters["clusters"][number];
export type EntityRecordView = EntityCluster["records"][number];

/** N05: what the dismissals hide, and the dismissals that apply here (an older server sends none). */
export function hiddenOf(found: Suggestions | undefined): HiddenSuggestion[] {
  return (found?.hidden as HiddenSuggestion[] | undefined) ?? [];
}
export function dismissalsOf(found: Suggestions | undefined): CleanDismissal[] {
  return (found?.dismissals as CleanDismissal[] | undefined) ?? [];
}

const BASE = "/modules/clean";

export const cleanApi = {
  settings: () => request<{ mode: CleanMode }>("GET", `${BASE}/settings`),
  updateSettings: (mode: CleanMode) => request<{ mode: CleanMode }>("PUT", `${BASE}/settings`, { mode }),
  lineage: () => request<LineageItem[]>("GET", `${BASE}/lineage`),
  suggestions: (id: string) => request<Suggestions>("GET", `${BASE}/datasets/${id}/suggestions`),
  draft: (id: string) => request<CleanDraft>("GET", `${BASE}/datasets/${id}/draft`),
  saveDraft: (id: string, steps: Step[]) => request<CleanDraft>("PUT", `${BASE}/datasets/${id}/draft`, { schema: 1, steps }),
  resetDraft: (id: string) => request<CleanDraft>("DELETE", `${BASE}/datasets/${id}/draft`),
  preview: (id: string, steps: Step[], stepId: string | null) =>
    request<CleanPreview>("POST", `${BASE}/datasets/${id}/preview`, { steps, step_id: stepId }),
  duplicates: (id: string, steps: Step[], stepId: string, limit: number) =>
    request<DuplicateGroups>("POST", `${BASE}/datasets/${id}/duplicates`, { steps, step_id: stepId, limit }),
  entities: (id: string, steps: Step[], stepId: string, limit: number) =>
    request<EntityClusters>("POST", `${BASE}/datasets/${id}/entities`, { steps, step_id: stepId, limit }),
  save: (id: string, steps: Step[]) => request<CleanJob>("POST", `${BASE}/datasets/${id}/versions`, { steps }),
  family: (id: string) => request<CleanFamily>("GET", `${BASE}/datasets/${id}/versions`),
  job: (jobId: string) => request<CleanJob>("GET", `${BASE}/jobs/${jobId}`),
  cancel: (jobId: string) => request<CleanJob>("DELETE", `${BASE}/jobs/${jobId}`),
  advanced: (id: string) => request<Suggestions>("GET", `${BASE}/datasets/${id}/suggestions/advanced`),
  expertMethods: (id: string) => request<ExpertMethod[]>("GET", `${BASE}/datasets/${id}/expert`),
  runExpert: (id: string, method: MethodId, options: ExpertOptions) => request<ExpertJob>("POST", `${BASE}/datasets/${id}/expert/${method}`, options),
  expertJob: (jobId: string) => request<ExpertJob>("GET", `${BASE}/expert-jobs/${jobId}`),
  cancelExpert: (jobId: string) => request<ExpertJob>("DELETE", `${BASE}/expert-jobs/${jobId}`),
  recipe: (versionId: string) => request<CleanRecipe>("GET", `${BASE}/versions/${versionId}/recipe`),
  replay: (id: string, body: ReplayRequest) => request<ReplayReport>("POST", `${BASE}/datasets/${id}/replay`, body),
  /** B1 (C-47): Rebuild from original, a job like a save. */
  rebuild: (versionId: string) => request<CleanJob>("POST", `${BASE}/versions/${versionId}/rebuild`),
  dismiss: (id: string, body: DismissBody) => request<Dismissed>("POST", `${BASE}/datasets/${id}/dismissed`, body),
  restore: (dismissalId: string) => request<null>("DELETE", `${BASE}/dismissed/${dismissalId}`),
  checks: (id: string) => request<CleanChecks>("GET", `${BASE}/datasets/${id}/checks`),
  saveChecks: (id: string, checks: CleanCheck[]) => request<CleanChecks>("PUT", `${BASE}/datasets/${id}/checks`, { checks }),
  checkRows: (id: string, checkId: string) => request<FailingRows>("GET", `${BASE}/datasets/${id}/checks/${checkId}/rows`),
  draftJournal: (id: string) => request<JournalEntry[]>("GET", `${BASE}/datasets/${id}/journal`),
  journal: (versionId: string) => request<CleanJournal>("GET", `${BASE}/versions/${versionId}/journal`),
};

/** N45, N46: download links; the browser saves the file the server names. */
export const cleanUrls = {
  sql: (versionId: string) => `/api/v1${BASE}/versions/${versionId}/sql`,
  python: (versionId: string) => `/api/v1${BASE}/versions/${versionId}/python`,
  journalMarkdown: (versionId: string) => `/api/v1${BASE}/versions/${versionId}/journal/markdown`,
  journalHtml: (versionId: string) => `/api/v1${BASE}/versions/${versionId}/journal/html`,
};

export const cleanKeys = {
  settings: ["clean", "settings"] as const,
  lineage: ["clean", "lineage"] as const,
  suggestions: (id: string) => ["datasets", id, "clean", "suggestions"] as const,
  draft: (id: string) => ["datasets", id, "clean", "draft"] as const,
  family: (id: string) => ["datasets", id, "clean", "family"] as const,
  advanced: (id: string) => ["datasets", id, "clean", "advanced"] as const,
  checks: (id: string) => ["datasets", id, "clean", "checks"] as const,
  methods: (id: string) => ["datasets", id, "clean", "methods"] as const,
  /** 101: the last result of a check, kept while the page lives (decision 4). */
  expert: (id: string, method: MethodId) => ["datasets", id, "clean", "expert", method] as const,
};

export function useCleanSettings() {
  return useQuery({ queryKey: cleanKeys.settings, queryFn: cleanApi.settings, enabled: CLEAN_ENABLED });
}

export function useUpdateCleanSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: cleanApi.updateSettings,
    onSuccess: (settings) => {
      client.setQueryData(cleanKeys.settings, settings);
      // N03: a draft not saved yet is the mode's default (the reliable fixes, or none)
      void client.invalidateQueries({ predicate: ({ queryKey }) => queryKey[0] === "datasets" && queryKey[2] === "clean" && queryKey[3] === "draft" });
    },
  });
}

/** Every cleaned version and its parent. An error (module off, an older server) reads as "no
 *  versions": the home page and the top bar then look as they did before 065. */
export function useLineage() {
  const query = useQuery({
    queryKey: cleanKeys.lineage,
    queryFn: cleanApi.lineage,
    enabled: CLEAN_ENABLED,
    retry: false,
    staleTime: 10_000,
  });
  return query.data ?? [];
}

export function useSuggestions(id: string, enabled: boolean) {
  return useQuery({ queryKey: cleanKeys.suggestions(id), queryFn: () => cleanApi.suggestions(id), enabled, staleTime: Infinity });
}

/** Read once: the page then edits its own copy (undo history), saved at every change. */
export function useDraft(id: string, enabled: boolean) {
  return useQuery({
    queryKey: cleanKeys.draft(id),
    queryFn: () => cleanApi.draft(id),
    enabled,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  });
}

export function useSaveDraft(id: string) {
  return useMutation({ mutationFn: (steps: Step[]) => cleanApi.saveDraft(id, steps) });
}

export function useResetDraft(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => cleanApi.resetDraft(id),
    onSuccess: (draft) => client.setQueryData(cleanKeys.draft(id), draft),
  });
}

export function usePreview(id: string, steps: Step[] | undefined, stepId: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "clean", "preview", steps, stepId],
    queryFn: () => cleanApi.preview(id, steps ?? [], stepId),
    enabled: steps !== undefined,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
  });
}

/** N35: the groups of one removal of duplicates, as `steps` leave the rows before it (the first
 *  `limit` groups); kept while more are fetched. */
export function useDuplicateGroups(id: string, steps: Step[] | undefined, stepId: string | null, limit: number) {
  return useQuery({
    queryKey: ["datasets", id, "clean", "duplicates", steps, stepId, limit],
    queryFn: () => cleanApi.duplicates(id, steps ?? [], stepId ?? "", limit),
    enabled: steps !== undefined && stepId !== null,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    retry: false,
  });
}

/** N36: the clusters of one entity number, as `steps` leave the rows before it (the first `limit`
 *  clusters); kept while more are fetched. */
export function useEntityClusters(id: string, steps: Step[] | undefined, stepId: string | null, limit: number) {
  return useQuery({
    queryKey: ["datasets", id, "clean", "entities", steps, stepId, limit],
    queryFn: () => cleanApi.entities(id, steps ?? [], stepId ?? "", limit),
    enabled: steps !== undefined && stepId !== null,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    retry: false,
  });
}

export function useFamily(id: string, enabled = true) {
  return useQuery({ queryKey: cleanKeys.family(id), queryFn: () => cleanApi.family(id), enabled: enabled && CLEAN_ENABLED, retry: false });
}

/** Start a save and follow it (a job, then the new dataset's import). */
export function useSaveVersion(id: string) {
  return useMutation({ mutationFn: (steps: Step[]) => cleanApi.save(id, steps) });
}

export function useCleanJob(jobId: string | null) {
  const client = useQueryClient();
  return useQuery({
    queryKey: ["clean", "job", jobId],
    queryFn: async () => {
      const job = await cleanApi.job(jobId ?? "");
      if (job.status !== "running") {
        void client.invalidateQueries({ queryKey: ["datasets"] });
        void client.invalidateQueries({ queryKey: cleanKeys.lineage });
      }
      return job;
    },
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 500 : false),
    refetchIntervalInBackground: true,
  });
}

export function useCancelJob() {
  return useMutation({ mutationFn: cleanApi.cancel });
}

// --- 101: Advanced on demand, More checks, replay ----------------------------------------------

/** The Advanced layer, asked for when it is opened (C-16); `enabled` false only reads the cache. */
export function useAdvanced(id: string, enabled: boolean) {
  return useQuery({ queryKey: cleanKeys.advanced(id), queryFn: () => cleanApi.advanced(id), enabled, staleTime: Infinity });
}

/** The methods of More checks, read from the columns only (C-29). */
export function useExpertMethods(id: string) {
  return useQuery({ queryKey: cleanKeys.methods(id), queryFn: () => cleanApi.expertMethods(id), staleTime: Infinity });
}

export const METHOD_IDS: MethodId[] = ["unusual_rows", "suspicious_labels", "fill_empty", "cell_errors"];

/** The last result of each check, from the cache only: never a request (C-29). */
export function useExpertResults(id: string): Partial<Record<MethodId, ExpertResult>> {
  const results = useQueries({
    queries: METHOD_IDS.map((method) => ({
      queryKey: cleanKeys.expert(id, method),
      queryFn: () => Promise.reject(new Error("Run the check first.")),
      enabled: false,
      staleTime: Infinity,
    })),
  });
  // a cleared result is null in the cache (Clear): read as no result
  return Object.fromEntries(METHOD_IDS.map((method, index) => [method, (results[index]?.data as ExpertResult | null | undefined) ?? undefined]));
}

/** Follow a run; once it is over its result goes to the cache, where the page reads it. */
export function useExpertJob(id: string, jobId: string | null) {
  const client = useQueryClient();
  return useQuery({
    queryKey: ["clean", "expert-job", jobId],
    queryFn: async () => {
      const job = await cleanApi.expertJob(jobId ?? "");
      if (job.status === "done" && job.result) client.setQueryData(cleanKeys.expert(id, job.method), job.result);
      return job;
    },
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 400 : false),
    refetchIntervalInBackground: true,
  });
}

export function useRunExpert(id: string) {
  return useMutation({ mutationFn: ({ method, options }: { method: MethodId; options: ExpertOptions }) => cleanApi.runExpert(id, method, options) });
}

export function useCancelExpert() {
  return useMutation({ mutationFn: cleanApi.cancelExpert });
}

/** N05: the suggestions shown change; the draft is the page's (it takes out the steps hidden). */
function refreshFound(client: ReturnType<typeof useQueryClient>, id: string) {
  void client.invalidateQueries({ queryKey: cleanKeys.suggestions(id) });
  void client.invalidateQueries({ queryKey: cleanKeys.advanced(id) });
}

export function useDismiss(id: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: (body: DismissBody) => cleanApi.dismiss(id, body), onSuccess: () => refreshFound(client, id) });
}

export function useRestore(id: string) {
  const client = useQueryClient();
  return useMutation({ mutationFn: cleanApi.restore, onSuccess: () => refreshFound(client, id) });
}

/** "Use as draft" of a replay: what the recipe carries besides its steps is kept for this dataset
 *  (N05: its dismissals, the same one twice kept once; N31: its checks not declared here yet). */
export function useAdoptReplay(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (report: ReplayReport) => {
      for (const dismissal of (report.dismissed as DismissBody[] | undefined) ?? []) await cleanApi.dismiss(id, dismissal);
      const brought = (report.checks as CleanCheck[] | undefined) ?? [];
      if (!brought.length) return;
      const here = (await cleanApi.checks(id)).checks;
      const added = brought.filter((check) => !here.some((other) => sameCheck(other, check)));
      if (!added.length) return;
      const all = added.reduce<CleanCheck[]>((list, check) => [...list, { ...check, id: nextCheckId(list) }], here);
      client.setQueryData(cleanKeys.checks(id), await cleanApi.saveChecks(id, all));
    },
    onSuccess: () => refreshFound(client, id),
  });
}

/** N31: the family's checks and what they find here (run by the server when the data changed). An
 *  error (an older server) reads as no checks. */
export function useChecks(id: string, enabled = true) {
  return useQuery({ queryKey: cleanKeys.checks(id), queryFn: () => cleanApi.checks(id), enabled: enabled && CLEAN_ENABLED, retry: false, staleTime: 30_000 });
}

export function useSaveChecks(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (checks: CleanCheck[]) => cleanApi.saveChecks(id, checks),
    onSuccess: (found) => {
      client.setQueryData(cleanKeys.checks(id), found);
      void client.invalidateQueries({ queryKey: cleanKeys.family(id) });
    },
  });
}

/** N31: add one check to the family's (read first: the page may not have them yet). */
export function useAddCheck(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: async (check: CleanCheck) => {
      const here = (await cleanApi.checks(id)).checks;
      const known = here.find((other) => sameCheck(other, check));
      if (known) return { found: client.getQueryData<CleanChecks>(cleanKeys.checks(id)) ?? null, id: known.id };
      const added = { ...check, id: nextCheckId(here) };
      return { found: await cleanApi.saveChecks(id, [...here, added]), id: added.id };
    },
    onSuccess: ({ found }) => {
      if (found) client.setQueryData(cleanKeys.checks(id), found);
      else void client.invalidateQueries({ queryKey: cleanKeys.checks(id) });
    },
  });
}

/** N46: the changes of the draft since the last version; an error (an older server) reads as none. */
export function useDraftJournal(id: string) {
  return useQuery({ queryKey: ["datasets", id, "clean", "journal"], queryFn: () => cleanApi.draftJournal(id), enabled: CLEAN_ENABLED, retry: false, staleTime: 5_000 });
}

export function useJournal(versionId: string) {
  return useQuery({ queryKey: ["datasets", versionId, "clean", "version-journal"], queryFn: () => cleanApi.journal(versionId), retry: false });
}

export function useCheckRows(id: string, checkId: string) {
  return useQuery({ queryKey: [...cleanKeys.checks(id), "rows", checkId], queryFn: () => cleanApi.checkRows(id, checkId), staleTime: 30_000 });
}

export function useReplay(id: string) {
  return useMutation({ mutationFn: (body: ReplayRequest) => cleanApi.replay(id, body) });
}
