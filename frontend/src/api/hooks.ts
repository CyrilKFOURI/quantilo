import {
  keepPreviousData,
  useMutation,
  useQueries,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { useActiveFilters, type Condition } from "./filters";
import {
  api,
  type ChartParams,
  type CompareParams,
  type DriverSpec,
  type ScenarioChange,
  type SeriesParams,
  type SeriesSectionKey,
  type CorrelationMethod,
  type ColumnChange,
  type Dataset,
  type ImportOptions,
  type KpiDefinition,
  type ComputeSettings,
  type ServerSettings,
  type VersionSettings,
  type SyntheticRequest,
  type MlChoicesInput,
  type MlClasses,
  type MlContext,
  type MlValidation,
  type MlTuneRequest,
  type MlAutoRequest,
  type OutlierMethod,
  type DetectionSettings,
  type RowsView,
  type SourceOptions,
} from "./client";

export const queryKeys = {
  health: ["health"] as const,
  datasets: ["datasets"] as const,
  dataset: (id: string) => ["datasets", id] as const,
  schema: (id: string) => ["datasets", id, "schema"] as const,
  head: (id: string) => ["datasets", id, "head"] as const,
  job: (id: string) => ["jobs", id] as const,
  preview: (path: string, options: SourceOptions) =>
    ["preview", path, options] as const,
  browse: (path: string | null) => ["browse", path] as const,
  samples: ["samples"] as const,
  columns: (id: string) => ["datasets", id, "columns"] as const,
  roles: ["column-roles"] as const,
  overview: (id: string, filters: Condition[] = []) =>
    ["datasets", id, "overview", filters] as const,
  chart: (
    id: string,
    column: string,
    params: ChartParams,
    filters: Condition[] = [],
  ) => ["datasets", id, "chart", column, params, filters] as const,
  summaries: (id: string, filters: Condition[]) =>
    ["datasets", id, "summaries", filters] as const,
  variable: (id: string, column: string, filters: Condition[] = []) =>
    ["datasets", id, "variable", column, filters] as const,
  fit: (id: string, column: string, filters: Condition[] = []) =>
    ["datasets", id, "variable", column, "fit", filters] as const,
  outliers: (
    id: string,
    column: string,
    method: OutlierMethod,
    filters: Condition[] = [],
  ) =>
    ["datasets", id, "variable", column, "outliers", method, filters] as const,
  savedFilters: (id: string) => ["datasets", id, "saved-filters"] as const,
  correlations: (
    id: string,
    method: CorrelationMethod,
    columns: string[] | null,
    filters: Condition[],
  ) => ["datasets", id, "correlations", method, columns, filters] as const,
  compare: (id: string, params: CompareParams, filters: Condition[]) =>
    ["datasets", id, "compare", params, filters] as const,
  series: (
    id: string,
    section: string,
    params: SeriesParams,
    extra: Record<string, string>,
    filters: Condition[],
  ) => ["datasets", id, "timeseries", section, params, extra, filters] as const,
  settings: ["settings"] as const,
  storage: ["storage"] as const,
  versions: (id: string, stamp: string) => ["datasets", id, "versions", stamp] as const,
  changes: (id: string, vs: string | null, part: string, stamp: string, extra = "") =>
    ["datasets", id, "changes", vs, part, stamp, extra] as const,
  rows: (id: string, view: RowsView, filters: Condition[], block: number) =>
    ["datasets", id, "rows", view, filters, block] as const,
};

const isImporting = (datasets: Dataset[] | undefined) =>
  datasets?.some((dataset) => dataset.status === "importing") ?? false;

export function useHealth() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: api.health,
    refetchInterval: 15_000,
    retry: false,
  });
}

export function useDatasets() {
  return useQuery({
    queryKey: queryKeys.datasets,
    queryFn: api.datasets,
    refetchInterval: (query) => (isImporting(query.state.data) ? 1_000 : false),
    refetchIntervalInBackground: true,
  });
}

export function useDataset(id: string | undefined) {
  return useQuery({
    queryKey: queryKeys.dataset(id ?? ""),
    queryFn: () => api.dataset(id ?? ""),
    enabled: Boolean(id),
    refetchInterval: (query) =>
      query.state.data?.status === "importing" ? 1_000 : false,
  });
}

export function useSchema(id: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.schema(id),
    queryFn: () => api.schema(id),
    enabled,
  });
}

export function useHead(id: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.head(id),
    queryFn: () => api.head(id),
    enabled,
  });
}

export function useJob(id: string | null) {
  return useQuery({
    queryKey: queryKeys.job(id ?? ""),
    queryFn: () => api.job(id ?? ""),
    enabled: Boolean(id),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "done" || status === "failed" ? false : 500;
    },
    // Keep following an import when the user switches to another tab.
    refetchIntervalInBackground: true,
  });
}

export function usePreview(path: string | null, options: SourceOptions) {
  return useQuery({
    queryKey: queryKeys.preview(path ?? "", options),
    queryFn: () => api.preview(path ?? "", options),
    enabled: Boolean(path),
    retry: false,
    staleTime: Infinity,
    placeholderData: (previous) => previous,
  });
}

export function useBrowse(path: string | null, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.browse(path),
    queryFn: () => api.browse(path),
    enabled,
    retry: false,
  });
}

export function useSamples() {
  return useQuery({
    queryKey: queryKeys.samples,
    queryFn: api.samples,
    staleTime: Infinity,
  });
}

export function useUpload() {
  return useMutation({ mutationFn: api.upload });
}

export function useImportDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: {
      path: string;
      name: string | null;
      options: ImportOptions;
    }) => api.importDataset(input.path, input.name, input.options),
    onSettled: () => client.invalidateQueries({ queryKey: queryKeys.datasets }),
  });
}

export function useRenameDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { id: string; name: string }) =>
      api.renameDataset(input.id, input.name),
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.datasets }),
  });
}

export function useDeleteDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.deleteDataset,
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.datasets }),
  });
}

export function useReimportDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.reimportDataset,
    onSuccess: () => client.invalidateQueries({ queryKey: queryKeys.datasets }),
  });
}

export function useOpenDataset() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.openDataset,
    onSuccess: (dataset) => {
      client.setQueryData(queryKeys.dataset(dataset.id), dataset);
      void client.invalidateQueries({
        queryKey: queryKeys.datasets,
        exact: true,
      });
    },
  });
}

export function useColumns(id: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.columns(id),
    queryFn: () => api.columns(id),
    enabled,
  });
}

export function useRoles() {
  return useQuery({
    queryKey: queryKeys.roles,
    queryFn: api.roles,
    staleTime: Infinity,
  });
}

/** Column corrections; the server answers with all columns, written straight to the cache. */
export function useUpdateColumns(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (changes: ColumnChange[]) => api.updateColumns(id, changes),
    onSuccess: (columns) => {
      client.setQueryData(queryKeys.columns(id), columns);
      void client.invalidateQueries({
        queryKey: queryKeys.dataset(id),
        exact: true,
      });
      void client.invalidateQueries({ queryKey: queryKeys.overview(id) });
      void client.invalidateQueries({ queryKey: ["datasets", id, "chart"] });
      void client.invalidateQueries({ queryKey: ["datasets", id, "variable"] });
    },
  });
}

export function useDetectColumns(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.detectColumns(id),
    onSuccess: (columns) => client.setQueryData(queryKeys.columns(id), columns),
  });
}

export function useReviewTypes(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.reviewTypes(id),
    onSuccess: (dataset) => {
      client.setQueryData(queryKeys.dataset(id), dataset);
      void client.invalidateQueries({
        queryKey: queryKeys.datasets,
        exact: true,
      });
    },
  });
}

/** Overview of the rows that pass the active filters. */
export function useOverview(id: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.overview(id, filters),
    queryFn: () => api.overview(id, filters),
    enabled,
    placeholderData: (previous) => previous,
  });
}

/** Chart data over the filtered rows; `unfiltered` reads every row (e.g. values to pick from). */
export function useChart(
  id: string,
  column: string,
  params: ChartParams,
  enabled: boolean,
  options: { unfiltered?: boolean } = {},
) {
  const active = useActiveFilters(id);
  const filters = options.unfiltered ? [] : active;
  return useQuery({
    queryKey: queryKeys.chart(id, column, params, filters),
    queryFn: () => api.chart(id, column, params, filters),
    enabled,
    staleTime: 60_000,
    placeholderData: (previous) => previous,
  });
}

export function useVariable(id: string, column: string, enabled = true) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.variable(id, column, filters),
    queryFn: () => api.variable(id, column, filters),
    enabled,
    staleTime: 60_000,
  });
}

export function useFit(id: string, column: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.fit(id, column, filters),
    queryFn: () => api.fit(id, column, filters),
    enabled,
    staleTime: 60_000,
  });
}

export function useOutliers(
  id: string,
  column: string,
  method: OutlierMethod,
  enabled: boolean,
) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.outliers(id, column, method, filters),
    queryFn: () => api.outliers(id, column, method, filters),
    enabled,
    staleTime: 60_000,
    placeholderData: (previous) => previous,
  });
}

/** Column summaries over the filtered rows (only fetched while filters are active). */
export function useColumnSummaries(id: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.summaries(id, filters),
    queryFn: () => api.summaries(id, filters),
    enabled: enabled && filters.length > 0,
    placeholderData: (previous) => previous,
  });
}

export function useKpiSuggestions(id: string, enabled: boolean) {
  return useQuery({
    queryKey: ["datasets", id, "kpi", "suggestions"],
    queryFn: () => api.kpiSuggestions(id),
    enabled,
    staleTime: 60_000,
  });
}

/** A KPI's numbers. The display settings (`view`) are left out of the key: changing them only
 *  redraws, it never recomputes. With "Use the page filters" off, the filters are not sent. */
export function useKpiCompute(id: string, definition: KpiDefinition | null) {
  const active = useActiveFilters(id);
  const filters = definition?.time && definition.time.page_filters === false ? [] : active;
  const data = definition ? { ...definition, view: undefined } : null;
  return useQuery({
    queryKey: ["datasets", id, "kpi", "compute", data, filters],
    queryFn: () => api.kpiCompute(id, { ...(definition as KpiDefinition), view: undefined }, filters),
    enabled: Boolean(definition),
    staleTime: 60_000,
    placeholderData: (previous) => previous,
    retry: false,
  });
}

const kpisKey = (id: string) => ["datasets", id, "kpi", "definitions"];

export function useKpis(id: string, enabled: boolean) {
  return useQuery({ queryKey: kpisKey(id), queryFn: () => api.kpis(id), enabled });
}

/** Create, update, duplicate or delete a saved KPI; the list refreshes afterwards. */
export function useKpiActions(id: string) {
  const client = useQueryClient();
  const onSuccess = () => client.invalidateQueries({ queryKey: kpisKey(id) });
  return {
    create: useMutation({
      mutationFn: (input: { name: string; definition: KpiDefinition }) =>
        api.createKpi(id, input.name, input.definition),
      onSuccess,
    }),
    update: useMutation({
      mutationFn: (input: { kpiId: string; name?: string; definition?: KpiDefinition }) =>
        api.updateKpi(id, input.kpiId, { name: input.name, definition: input.definition }),
      onSuccess,
    }),
    duplicate: useMutation({
      mutationFn: (kpiId: string) => api.duplicateKpi(id, kpiId),
      onSuccess,
    }),
    remove: useMutation({ mutationFn: (kpiId: string) => api.deleteKpi(id, kpiId), onSuccess }),
  };
}

export function useSavedFilters(id: string) {
  return useQuery({
    queryKey: queryKeys.savedFilters(id),
    queryFn: () => api.savedFilters(id),
  });
}

export function useSaveFilter(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (input: { name: string; conditions: Condition[] }) =>
      api.saveFilter(id, input.name, input.conditions),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: queryKeys.savedFilters(id) }),
  });
}

export function useDeleteSavedFilter(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (filterId: string) => api.deleteSavedFilter(id, filterId),
    onSuccess: () =>
      client.invalidateQueries({ queryKey: queryKeys.savedFilters(id) }),
  });
}

/** Two or three columns compared over the filtered rows. */
export function useCompare(id: string, params: CompareParams | null) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.compare(id, params ?? { x: "", y: "" }, filters),
    queryFn: () => api.compare(id, params as CompareParams, filters),
    enabled: Boolean(params),
    staleTime: 60_000,
    placeholderData: (previous) => previous,
    retry: false,
  });
}

/** One section of the Time series page; `enabled` lets a section wait until it is on screen. */
export function useSeriesSection<T>(
  id: string,
  section: SeriesSectionKey,
  params: SeriesParams,
  extra: Record<string, string>,
  enabled = true,
) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.series(id, section, params, extra, filters),
    queryFn: () => api.seriesSection<T>(id, section, params, extra, filters),
    enabled,
    staleTime: 60_000,
    placeholderData: (previous) => previous,
    retry: false,
  });
}

/** Start (or reuse) a forecast job for the current series and filters. */
export function useStartForecast(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: ({
      params,
      extra,
      drivers,
    }: {
      params: SeriesParams;
      extra: Record<string, string>;
      drivers?: DriverSpec[];
    }) => api.startForecast(id, params, extra, filters, drivers),
  });
}

/** Follow a forecast job until it ends (also while the tab is in the background). */
export function useForecastJob(jobId: string | null) {
  return useQuery({
    queryKey: ["forecast-jobs", jobId ?? ""],
    queryFn: () => api.forecastJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 500 : false,
    refetchIntervalInBackground: true,
    retry: false,
  });
}

/** Start (or reuse) a forecast of every group (039). */
export function useStartGroupForecast(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: ({
      params,
      extra,
    }: {
      params: SeriesParams;
      extra: Record<string, string>;
    }) => api.startGroupForecast(id, params, extra, filters),
  });
}

/** Follow a forecast of every group until it ends (also in the background). */
export function useGroupJob(jobId: string | null) {
  return useQuery({
    queryKey: ["group-forecast-jobs", jobId ?? ""],
    queryFn: () => api.groupJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      query.state.data?.status === "running" || !query.state.data ? 700 : false,
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useCancelGroupJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.cancelGroupJob(jobId),
    onSuccess: (job) => client.setQueryData(["group-forecast-jobs", job.id], job),
  });
}

/** One driver model's forecast per list of what-if changes ([] = the base forecast), 039. */
export function useScenarios(
  id: string,
  params: SeriesParams,
  extra: Record<string, string>,
  drivers: DriverSpec[],
  scenarios: ScenarioChange[][],
  enabled = true,
) {
  const filters = useActiveFilters(id);
  return useQueries({
    queries: scenarios.map((changes) => ({
      queryKey: ["datasets", id, "timeseries", "scenario", params, extra, filters, drivers, changes],
      queryFn: () => api.scenario(id, params, extra, filters, drivers, changes),
      enabled,
      staleTime: 60_000,
      retry: false,
    })),
    // Structurally shared: the same object while nothing changed (stable for useMemo).
    combine: (found) => ({
      data: found.map((r) => r.data),
      error: found.find((r) => r.error)?.error ?? null,
      pending: found.some((r) => r.isFetching),
    }),
  });
}

export function useCancelForecast() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.cancelForecast(jobId),
    onSuccess: (job) => client.setQueryData(["forecast-jobs", job.id], job),
  });
}

/** Is this a panel (entities × periods)? Its period, entity and the embargo the target calls for. */
export function useMlPanel(id: string, target: string, period: string | null = null, entity: string | null = null) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "ml", "panel", target, period, entity, filters],
    queryFn: () => api.mlPanel(id, target, period, entity, filters),
    staleTime: 60_000,
    retry: false,
  });
}

/** A panel run's predictions without the part some columns explain, every period scored again. */
export function useMlNeutralize(id: string) {
  return useMutation({
    mutationFn: (args: { runId: string; columns: string[]; proportion: number }) => api.mlNeutralize(id, args.runId, args.columns, args.proportion),
  });
}

/** Every column as a possible target, through the active filters. */
export function useMlTargets(id: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "ml", "targets", filters],
    queryFn: () => api.mlTargets(id, filters),
    enabled,
    staleTime: 60_000,
  });
}

export function useMlCatalog(context: MlContext | null) {
  return useQuery({
    queryKey: ["ml", "catalog", context],
    queryFn: () => api.mlCatalog(context as MlContext),
    enabled: Boolean(context),
    staleTime: Infinity,
    placeholderData: keepPreviousData,
  });
}

export function useMlTrain(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: ({
      target,
      model,
      settings,
      choices,
      classes,
      validation,
    }: {
      target: string;
      model: string | null;
      settings: Record<string, unknown> | null;
      choices?: MlChoicesInput | null;
      classes?: MlClasses | null;
      validation?: MlValidation | null;
    }) => api.mlTrain(id, target, model, settings, filters, choices, classes, validation),
  });
}

/** The preparation the next training would use, with its first prepared rows. */
export function useMlPreparation(
  id: string,
  target: string,
  model: string | null,
  settings: Record<string, unknown> | null,
  choices: MlChoicesInput,
  enabled: boolean,
  classes: MlClasses | null = null,
  validation: MlValidation | null = null,
) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "ml", "preparation", target, model, settings, choices, classes, validation, filters],
    queryFn: () => api.mlPreparation(id, target, model, settings, choices, filters, classes, validation),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 60_000,
    retry: false,
  });
}

/** Where the run is wrong (036): computed on demand on its held-out predictions, then cached. */
export function useMlErrors(id: string, runId: string, measure: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "errors", runId, measure ?? ""],
    queryFn: () => api.mlErrors(id, runId, measure),
    staleTime: Infinity,
    retry: false,
  });
}

/** Subgroups where the second run fixes or worsens the first one's errors. */
export function useMlErrorsCompared(id: string, runId: string, otherId: string) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "errors-compared", runId, otherId],
    queryFn: () => api.mlErrorsCompared(id, runId, otherId),
    staleTime: Infinity,
    retry: false,
  });
}

/** A run's metrics per group of a sensitive column (none chosen: nothing fetched). */
export function useMlFairness(id: string, runId: string, column: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "fairness", runId, column ?? ""],
    queryFn: () => api.mlFairness(id, runId, column ?? ""),
    enabled: Boolean(column),
    staleTime: Infinity,
    retry: false,
  });
}

/** Why the run predicts what it predicts: computed in the background, polled until done. */
export function useMlExplanation(id: string, runId: string | null, enabled: boolean) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "explanation", runId ?? ""],
    queryFn: () => api.mlExplanation(id, runId ?? ""),
    enabled: enabled && Boolean(runId),
    refetchInterval: (query) => (query.state.data?.status === "running" ? 800 : false),
    refetchIntervalInBackground: true,
    staleTime: Infinity,
    retry: false,
  });
}

/** Stop a training at its next checkpoint. */
export function useMlCancel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.mlCancel(jobId),
    onSuccess: (job) => client.setQueryData(["ml", "jobs", job.id], job),
  });
}

/** How long a training would take (fits on 500 and 2 000 rows, extrapolated). */
export function useMlEstimate(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: (args: {
      target: string;
      model: string | null;
      settings: Record<string, unknown> | null;
      choices: MlChoicesInput | null;
      classes: MlClasses | null;
      validation: MlValidation | null;
    }) => api.mlEstimate(id, args.target, args.model, args.settings, args.choices, args.classes, args.validation, filters),
  });
}

/** Train several models one after the other on the same rows. */
export function useMlCompare(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: (args: { target: string; models: string[]; choices: MlChoicesInput | null; classes: MlClasses | null; validation: MlValidation | null }) =>
      api.mlCompare(id, args.target, args.models, args.choices, args.classes, args.validation, filters),
  });
}

export function useMlComparison(jobId: string | null) {
  return useQuery({
    queryKey: ["ml", "compare-jobs", jobId ?? ""],
    queryFn: () => api.mlComparison(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "done" ? false : 700),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

/** Search the best settings of a model within a budget. */
export function useMlTune(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: (args: { target: string; body: MlTuneRequest; classes: MlClasses | null; validation: MlValidation | null }) =>
      api.mlTune(id, args.target, args.body, args.classes, args.validation, filters),
  });
}

export function useMlTuning(jobId: string | null) {
  return useQuery({
    queryKey: ["ml", "tune-jobs", jobId ?? ""],
    queryFn: () => api.mlTuning(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data && query.state.data.status !== "running" ? false : 700),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useMlStopTuning() {
  return useMutation({ mutationFn: (jobId: string) => api.mlStopTuning(jobId) });
}

/** The Auto (038): the best model within a time budget, followed step by step. */
export function useMlAuto(id: string) {
  const filters = useActiveFilters(id);
  return useMutation({
    mutationFn: (args: { target: string; body: MlAutoRequest; classes: MlClasses | null; validation: MlValidation | null }) =>
      api.mlAuto(id, args.target, args.body, args.classes, args.validation, filters),
  });
}

export function useMlAutoJob(jobId: string | null) {
  return useQuery({
    queryKey: ["ml", "auto-jobs", jobId ?? ""],
    queryFn: () => api.mlAutoJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data && query.state.data.status !== "running" ? false : 700),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useMlStopAuto() {
  return useMutation({ mutationFn: (jobId: string) => api.mlStopAuto(jobId) });
}

/** A numeric target's distribution, to cut it into classes in the browser. */
export function useMlDistribution(id: string, column: string, enabled: boolean) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: ["datasets", id, "ml", "distribution", column, filters],
    queryFn: () => api.mlDistribution(id, column, filters),
    enabled,
    staleTime: 60_000,
  });
}

/** Follow a training until it ends (also while the tab is in the background). */
export function useMlJob(jobId: string | null) {
  return useQuery({
    queryKey: ["ml", "jobs", jobId ?? ""],
    queryFn: () => api.mlJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 400 : false),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useMlRuns(id: string, target: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "runs", target ?? ""],
    queryFn: () => api.mlRuns(id, target ?? ""),
    enabled: Boolean(target),
  });
}

export function useMlRun(id: string, runId: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "run", runId ?? ""],
    queryFn: () => api.mlRun(id, runId ?? ""),
    enabled: Boolean(runId),
    staleTime: Infinity,
  });
}

// --- Predict on new data (037) -------------------------------------------------------------

/** Can the run's model predict this dataset? Ready, Check or Blocked, column by column. */
export function useMlCheck(id: string, runId: string, source: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "check", runId, source ?? ""],
    queryFn: () => api.mlCheck(id, runId, source ?? ""),
    enabled: Boolean(source),
    retry: false,
  });
}

export function useMlPredict(id: string, runId: string) {
  return useMutation({
    mutationFn: ({ source, keep }: { source: string; keep: string[] }) => api.mlPredict(id, runId, source, keep),
  });
}

export function useMlPredictJob(jobId: string | null) {
  return useQuery({
    queryKey: ["ml", "predict-jobs", jobId ?? ""],
    queryFn: () => api.mlPredictJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 500 : false),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

/** Stop a prediction after its current block of rows. */
export function useMlCancelPredict() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.mlCancelPredict(jobId),
    onSuccess: (job) => client.setQueryData(["ml", "predict-jobs", job.id], job),
  });
}

export function useMlPredictions(id: string, runId: string) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "predictions", runId],
    queryFn: () => api.mlPredictions(id, runId),
    retry: false,
  });
}

export function useMlPrediction(id: string, predictionId: string | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "prediction", predictionId ?? ""],
    queryFn: () => api.mlPrediction(id, predictionId ?? ""),
    enabled: Boolean(predictionId),
    staleTime: Infinity,
    retry: false,
  });
}

export function useMlPredictionRows(id: string, predictionId: string, offset: number, limit: number, lessReliable: boolean) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "prediction", predictionId, "rows", offset, limit, lessReliable],
    queryFn: () => api.mlPredictionRows(id, predictionId, offset, limit, lessReliable),
    placeholderData: keepPreviousData,
    staleTime: Infinity,
  });
}

export function useMlPredictionRowExplanation(id: string, predictionId: string, row: number | null) {
  return useQuery({
    queryKey: ["datasets", id, "ml", "prediction", predictionId, "explain", row ?? -1],
    queryFn: () => api.mlPredictionRowExplanation(id, predictionId, row ?? 0),
    enabled: row !== null,
    staleTime: Infinity,
    retry: false,
  });
}

/** The predicted rows with their predictions, imported as a new dataset. */
export function useMlSavePrediction(id: string) {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (predictionId: string) => api.mlSavePrediction(id, predictionId),
    onSuccess: () => {
      void client.invalidateQueries({ queryKey: queryKeys.datasets });
      void client.invalidateQueries({ queryKey: ["datasets", id, "ml", "predictions"] });
    },
  });
}

/** Association matrix and insights over the filtered rows. */
export function useCorrelations(
  id: string,
  method: CorrelationMethod,
  columns: string[] | null,
  enabled: boolean,
) {
  const filters = useActiveFilters(id);
  return useQuery({
    queryKey: queryKeys.correlations(id, method, columns, filters),
    queryFn: () => api.correlations(id, method, columns, filters),
    enabled,
    staleTime: 60_000,
    placeholderData: (previous) => previous,
  });
}

export const ROW_BLOCK = 100;

/** Blocks of rows of the Data page (block k = rows k*100 … k*100+99 of the view). */
export function useRowBlocks(
  id: string,
  view: RowsView,
  blocks: number[],
  enabled: boolean,
) {
  const filters = useActiveFilters(id);
  return useQueries({
    queries: blocks.map((block) => ({
      queryKey: queryKeys.rows(id, view, filters, block),
      queryFn: () => api.rows(id, view, block * ROW_BLOCK, ROW_BLOCK, filters),
      enabled,
      staleTime: 60_000,
      placeholderData: block === 0 ? keepPreviousData : undefined,
    })),
  });
}

export function useSettings() {
  return useQuery({ queryKey: queryKeys.settings, queryFn: api.settings });
}

export function useUpdateSettings() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (change: { detection?: DetectionSettings; compute?: ComputeSettings; versions?: VersionSettings }) => {
      const current = client.getQueryData<ServerSettings>(queryKeys.settings);
      if (!current) throw new Error("Settings not loaded yet.");
      return api.updateSettings({ ...current, ...change });
    },
    onSuccess: (saved) => client.setQueryData(queryKeys.settings, saved),
  });
}

/** The graphics card the deep models could use; refreshing looks again ("Check again"). */
export function useGpu() {
  return useQuery({ queryKey: ["settings", "gpu"], queryFn: () => api.gpu(), staleTime: Infinity });
}

export function useRefreshGpu() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: () => api.gpu(true),
    onSuccess: (status) => client.setQueryData(["settings", "gpu"], status),
  });
}

// --- Synthetic data ---------------------------------------------------------------------------

const syntheticKey = (id: string) => ["datasets", id, "synthetic"];

export function useSyntheticPlan(id: string, enabled: boolean) {
  return useQuery({ queryKey: [...syntheticKey(id), "plan"], queryFn: () => api.syntheticPlan(id), enabled });
}

export function useSyntheticRuns(id: string, enabled: boolean) {
  return useQuery({ queryKey: [...syntheticKey(id), "runs"], queryFn: () => api.syntheticRuns(id), enabled });
}

export function useSyntheticRun(id: string, runId: string | undefined) {
  return useQuery({
    queryKey: [...syntheticKey(id), "run", runId],
    queryFn: () => api.syntheticRun(id, runId ?? ""),
    enabled: Boolean(runId),
    retry: false,
  });
}

export function useSyntheticGenerate(id: string) {
  return useMutation({ mutationFn: (body: SyntheticRequest) => api.syntheticGenerate(id, body) });
}

/** Follows a generation until it ends (also while the tab is hidden). */
export function useSyntheticJob(jobId: string | null) {
  return useQuery({
    queryKey: ["synthetic-jobs", jobId],
    queryFn: () => api.syntheticJob(jobId ?? ""),
    enabled: Boolean(jobId),
    refetchInterval: (query) => (query.state.data?.status === "running" || !query.state.data ? 500 : false),
    refetchIntervalInBackground: true,
    retry: false,
  });
}

export function useSyntheticCancel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => api.syntheticCancel(jobId),
    onSuccess: (job) => client.setQueryData(["synthetic-jobs", job.id], job),
  });
}

export function useSyntheticCompare(id: string, runIds: string[]) {
  return useQuery({
    queryKey: [...syntheticKey(id), "compare", runIds],
    queryFn: () => api.syntheticCompare(id, runIds),
    enabled: runIds.length >= 2,
  });
}

export function useSyntheticPreview(id: string, runId: string, enabled: boolean) {
  return useQuery({
    queryKey: [...syntheticKey(id), "preview", runId],
    queryFn: () => api.syntheticPreview(id, runId),
    enabled,
  });
}

export function useSyntheticColumn(
  id: string,
  runId: string,
  column: string | null,
  bins: number | null,
  target: string | null,
) {
  return useQuery({
    queryKey: [...syntheticKey(id), "column", runId, column, bins, target],
    queryFn: () => api.syntheticColumn(id, runId, column ?? "", bins, target),
    enabled: Boolean(column),
    placeholderData: (previous) => (previous?.column === column ? previous : undefined),
  });
}

export function useSyntheticPair(id: string, runId: string, pair: [string, string] | null) {
  return useQuery({
    queryKey: [...syntheticKey(id), "pair", runId, pair],
    queryFn: () => api.syntheticPair(id, runId, pair?.[0] ?? "", pair?.[1] ?? ""),
    enabled: Boolean(pair),
  });
}

/** Run actions; each refreshes the run list and the run (its stored report changes). */
export function useSyntheticActions(id: string) {
  const client = useQueryClient();
  const refresh = () => client.invalidateQueries({ queryKey: syntheticKey(id) });
  return {
    remove: useMutation({ mutationFn: (runId: string) => api.syntheticDelete(id, runId), onSuccess: refresh }),
    save: useMutation({
      mutationFn: (runId: string) => api.syntheticSave(id, runId),
      onSuccess: () => {
        void refresh();
        void client.invalidateQueries({ queryKey: queryKeys.datasets });
      },
    }),
    utility: useMutation({
      mutationFn: (input: { runId: string; target: string }) => api.syntheticUtility(id, input.runId, input.target),
      onSuccess: refresh,
    }),
    disclosure: useMutation({
      mutationFn: (input: { runId: string; known: string[]; sensitive: string }) =>
        api.syntheticDisclosure(id, input.runId, input.known, input.sensitive),
      onSuccess: refresh,
    }),
  };
}

export function useSyntheticOrigin(id: string | undefined) {
  return useQuery({
    queryKey: ["datasets", id, "synthetic-origin"],
    queryFn: () => api.syntheticOrigin(id ?? ""),
    enabled: Boolean(id),
    staleTime: Infinity,
  });
}


/** A dataset's data and types: when either changes, comparisons are read again. */
function stampOf(dataset: Dataset | undefined): string {
  return dataset ? `${dataset.imported_at ?? ""}|${dataset.types_version}` : "";
}

/** Imported versions of a dataset, newest first (031). */
export function useVersions(dataset: Dataset | undefined) {
  const id = dataset?.id ?? "";
  return useQuery({
    queryKey: queryKeys.versions(id, stampOf(dataset)),
    queryFn: () => api.versions(id),
    enabled: dataset?.status === "ready",
  });
}

/** What changed since the previous version, or since dataset `vs`. */
export function useChanges(dataset: Dataset | undefined, vs: string | null, enabled = true) {
  const id = dataset?.id ?? "";
  return useQuery({
    queryKey: queryKeys.changes(id, vs, "summary", stampOf(dataset)),
    queryFn: () => api.changes(id, vs),
    enabled: enabled && dataset?.status === "ready",
    retry: false,
    staleTime: 60_000,
  });
}

export function useChangeRows(dataset: Dataset | undefined, vs: string | null, key: string | null, enabled: boolean) {
  const id = dataset?.id ?? "";
  return useQuery({
    queryKey: queryKeys.changes(id, vs, "rows", stampOf(dataset), key ?? ""),
    queryFn: () => api.changeRows(id, vs, key),
    enabled,
    retry: false,
    staleTime: 60_000,
    placeholderData: keepPreviousData,
  });
}

export function useChangeKpis(dataset: Dataset | undefined, vs: string | null, enabled: boolean) {
  const id = dataset?.id ?? "";
  return useQuery({
    queryKey: queryKeys.changes(id, vs, "kpis", stampOf(dataset)),
    queryFn: () => api.changeKpis(id, vs),
    enabled,
    retry: false,
    staleTime: 60_000,
  });
}

export function useChangeRelations(dataset: Dataset | undefined, vs: string | null, enabled: boolean) {
  const id = dataset?.id ?? "";
  return useQuery({
    queryKey: queryKeys.changes(id, vs, "relations", stampOf(dataset)),
    queryFn: () => api.changeRelations(id, vs),
    enabled,
    retry: false,
    staleTime: 60_000,
  });
}

export function useStorage() {
  return useQuery({ queryKey: queryKeys.storage, queryFn: api.storage });
}

/** Empties the server's in-memory caches and this page's cached results. */
export function useClearCaches() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: api.clearCaches,
    onSuccess: () => client.invalidateQueries({ queryKey: ["datasets"] }),
  });
}
