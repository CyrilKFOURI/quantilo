import { filtersParam, type Condition } from "./filters";
import type { components } from "./schema";

type Schemas = components["schemas"];
export type Health = Schemas["HealthOut"];
export type Dataset = Schemas["DatasetOut"];
export type Job = Schemas["JobOut"];
export type Preview = Schemas["Preview"];
export type SourceOptions = Schemas["SourceOptions"];
export type ImportOptions = Schemas["ImportOptions"];
export type CsvOptions = Schemas["CsvOptions"];
export type ImportResponse = Schemas["ImportResponse"];
export type Listing = Schemas["Listing"];
export type Sample = Schemas["SampleOut"];
export type Upload = Schemas["UploadOut"];
export type ColumnInfo = Schemas["ColumnInfo"];
export type Head = Schemas["HeadOut"];
export type Column = Schemas["ColumnOut"];
/** Fields with a server default (like `reset`) are optional when sending a change. */
export type ColumnChange = Partial<Omit<Schemas["ColumnChange"], "name">> & {
  name: string;
};
export type AnalysisType = Column["type"];
export type Role = Schemas["RoleOut"];
export type Conversion = Schemas["ConversionOut"];
// Fields with server defaults are always present in responses.
export type Alert = Required<Schemas["Alert"]>;
export type Overview = Omit<Required<Schemas["Overview"]>, "alerts"> & {
  alerts: Alert[];
};
export type Chart = Required<Schemas["Chart"]>;
export type ChartParams = {
  view?: string | null;
  topN?: number;
  full?: boolean;
  step?: string | null;
  bins?: number | null;
};

/** The server sends every field, defaults included: make them all required, deeply. */
type Complete<T> = T extends (infer Item)[]
  ? Complete<Item>[]
  : T extends object
    ? { [K in keyof T]-?: Complete<T[K]> }
    : T;
export type VariableProfile = Complete<Schemas["VariableProfile"]>;
export type NumericProfile = NonNullable<VariableProfile["numeric"]>;
export type Describe = NumericProfile["describe"];
export type Normality = NumericProfile["normality"];
export type NormalityTest = Normality["tests"][number];
export type CategoricalProfile = NonNullable<VariableProfile["categorical"]>;
export type DateProfile = NonNullable<VariableProfile["date"]>;
export type TextProfile = NonNullable<VariableProfile["text"]>;
export type IdentifierProfile = NonNullable<VariableProfile["identifier"]>;
export type Fit = Complete<Schemas["Fit"]>;
export type Law = Fit["laws"][number];
export type Transform = Fit["transforms"][number];
export type Outliers = Complete<Schemas["Outliers"]>;
export type Tail = Complete<Schemas["Tail"]>;
export type TailSeries = NonNullable<Tail["hill"]>;
export type Robust = NonNullable<NumericProfile["robust"]>;
export type ColumnSummary = Schemas["ColumnSummary"];
export type SavedFilter = Schemas["SavedFilter"];
export type Comparison = Complete<Schemas["Comparison"]>;
export type BivariateTest = Comparison["tests"][number];
export type CompareGroup = Comparison["groups"][number];
export type GroupDifference = Comparison["differences"][number];
export type MultipleTesting = NonNullable<Comparison["multiple"]>;
/** How the p-values of a view are adjusted (051): auto follows the rule, the others force it. */
export type Correction = "auto" | "bh" | "holm" | "bonferroni";
export type Correlations = Complete<Schemas["Correlations"]>;
export type AssociationMatrix = Correlations["matrix"];
export type Insight = Correlations["insights"][number];
export type CorrelationMethod = AssociationMatrix["method"];
/** The structure of the numeric columns (078): PCA with parallel analysis, then factor analysis. */
export type Structure = Complete<Schemas["Structure"]>;
export type StructureRule = NonNullable<Correlations["structure"]>;
export type Factors = Complete<Schemas["Factors"]>;
// Regression (067, 068)
export type Regression = Complete<Schemas["RegressionReport"]>;
export type RegressionTerm = Regression["terms"][number];
export type RegressionCheck = Regression["checks"][number];
export type RegressionSuggestion = Complete<Schemas["RegressionSuggestion"]>;
export type RegressionStandardErrors = Complete<Schemas["StandardErrorTable"]>;
export type RegressionInfluence = Complete<Schemas["InfluenceTable"]>;
export type RegressionTerms = Complete<Schemas["TermsTable"]>;
export type RegressionMore = Complete<Schemas["MoreChecks"]>;
// Panel and instruments (071)
export type RegressionPanel = Complete<Schemas["PanelTable"]>;
export type RegressionPanelEstimator = RegressionPanel["estimators"][number];
export type RegressionIv = Complete<Schemas["IvTable"]>;
export type RegressionSe = Regression["se"]["kind"];
export type RegressionSection = "standard-errors" | "influence" | "terms" | "more" | "panel";
export type RegressionSections = {
  "standard-errors": RegressionStandardErrors;
  influence: RegressionInfluence;
  terms: RegressionTerms;
  more: RegressionMore;
  panel: RegressionPanel;
};
/** What the page asks for: absent fields follow the rules (R-REG-20, 21, SE-04). */
export type RegressionParams = {
  target?: string;
  columns?: string[];
  weights?: string;
  cluster?: string;
  se?: RegressionSe | "auto";
  focus?: string;
};

function regressionQuery(params: RegressionParams, filters?: Condition[]): URLSearchParams {
  const query = new URLSearchParams(filtersParam(filters));
  if (params.target) query.set("target", params.target);
  if (params.columns) query.set("columns", params.columns.join(","));
  if (params.weights !== undefined) query.set("weights", params.weights);
  if (params.cluster) query.set("cluster", params.cluster);
  if (params.se && params.se !== "auto") query.set("se", params.se);
  if (params.focus) query.set("focus", params.focus);
  return query;
}
export type SeriesOverview = Complete<Schemas["SeriesOverview"]>;
export type SeriesSeasonality = Complete<Schemas["SeriesSeasonality"]>;
export type SeriesDecomposition = Complete<Schemas["SeriesDecomposition"]>;
export type SeriesStationarity = Complete<Schemas["SeriesStationarity"]>;
export type SeriesAutocorrelation = Complete<Schemas["SeriesAutocorrelation"]>;
export type SeriesVolatility = Complete<Schemas["SeriesVolatility"]>;
export type SeriesResiduals = Complete<Schemas["SeriesResiduals"]>;
export type SeriesBreaks = Complete<Schemas["SeriesBreaks"]>;
export type SeriesSmoothing = Complete<Schemas["SeriesSmoothing"]>;
export type SmoothCurve = SeriesSmoothing["curves"][number];
export type ForecastJob = Complete<Schemas["ForecastJob"]>;
export type SeriesDrivers = Complete<Schemas["SeriesDrivers"]>;
/** A driver of a forecast (028 A4): a column, a 0/1 flag or a calendar part. */
export type DriverSpec = Schemas["Driver"];
/** Every group of a series at once (039): setup, job and result. */
export type GroupsSetup = Complete<Schemas["GroupsSetup"]>;
export type GroupJob = Complete<Schemas["GroupJob"]>;
export type GroupForecast = NonNullable<GroupJob["result"]>;
export type GroupRow = GroupForecast["rows"][number];
/** A what-if change of a known-in-advance driver (039). */
export type ScenarioChange = Schemas["ScenarioChange"];
export type DriverForecast = Complete<Schemas["DriverForecast"]>;
export type MlTargets = Complete<Schemas["Targets"]>;
export type MlCandidate = MlTargets["candidates"][number];
export type MlModel = Complete<Schemas["ModelView"]>;
export type MlSetting = MlModel["settings"][number];
export type MlCatalogue = Complete<Schemas["Catalogue"]>;
export type MlValidation = Complete<Schemas["Validation"]>;
export type MlExplanationState = Complete<Schemas["ExplanationState"]>;
export type MlExplanation = MlExplanationState["explanation"];
export type MlCompareJob = Complete<Schemas["CompareJob"]>;
export type MlTuneJob = Complete<Schemas["TuneJob"]>;
export type MlAutoJob = Complete<Schemas["AutoJob"]>;
export type MlAutoProgress = NonNullable<MlAutoJob["progress"]>;
export type MlAutoEntry = MlAutoProgress["entries"][number];
export type MlAutoRequest = { minutes: number; models?: string[] | null; trials?: number | null; stacking: boolean; preparation?: MlChoicesInput | null };
export type MlSearchParam = Complete<Schemas["SearchParam"]>;
export type MlEstimate = Complete<Schemas["Estimate"]>;
export type MlTuneRequest = {
  model: string;
  budget_seconds: number | null;
  trials: number | null;
  sampler: string;
  pruner: string;
  space: MlSearchParam[] | null;
  settings: Record<string, unknown>;
  preparation?: MlChoicesInput | null;
};
/** Metrics where lower is better (as the server's evaluate.LOWER_IS_BETTER). */
export const LOWER_IS_BETTER = new Set(["mae", "rmse", "mape", "median_ae", "max_error", "log_loss", "brier"]);
/** What the catalogue's recommendations and defaults depend on. */
export type MlContext = { task: string; rows: number; minority: number | null; features: number; categoricalShare: number; missingShare: number };
export type MlJob = Complete<Schemas["TrainingJob"]>;
export type MlRun = Complete<Schemas["Run"]>;
export type MlRunSummary = Complete<Schemas["RunSummary"]>;
export type MlTask = MlRun["task"];
export type MlPreparation = MlRun["preparation"];
export type MlDecision = MlPreparation["decisions"][number];
export type MlChoices = Complete<Schemas["Choices"]>;
export type MlColumnChoice = MlChoices["columns"][string];
export type MlSelectionChoice = MlChoices["selection"][number];
/** What the preparation editor sends: anything left out is automatic. */
export type MlChoicesInput = Partial<Omit<MlChoices, "columns">> & { columns?: Record<string, Partial<MlColumnChoice>> };
export type MlPreparationPreview = Complete<Schemas["PreparationPreview"]>;
export type MlDistribution = Complete<Schemas["TargetDistribution"]>;
export type MlClasses = { method: string; edges: number[]; names: string[] };
/** Panel data (035): recognition, a run's periods, its neutralization. */
export type MlPanelCheck = Complete<Schemas["PanelCheck"]>;
export type MlPanel = NonNullable<MlRun["panel"]>;
export type MlPeriodScore = MlPanel["periods"][number];
export type MlPeriodSummary = MlPanel["summary"];
export type MlNeutralized = Complete<Schemas["Neutralized"]>;
export type MlErrors = Complete<Schemas["ErrorAnalysis"]>;
export type MlErrorMeasure = MlErrors["measure"];
export type MlSlice = MlErrors["slices"][number];
export type MlColumnErrors = MlErrors["by_column"][number];
export type MlErrorPrediction = MlErrors["worst"][number];
export type MlErrorComparison = Complete<Schemas["ErrorComparison"]>;
export type MlFairness = Complete<Schemas["Fairness"]>;
// Predict on new data (037)
export type MlModelCheck = Complete<Schemas["ModelCheck"]>;
export type MlColumnCheck = MlModelCheck["columns"][number];
export type MlLevel = MlModelCheck["level"];
export type MlPredictionJob = Complete<Schemas["PredictionJob"]>;
export type MlPredictionSummary = Complete<Schemas["PredictionSummary"]>;
export type MlPrediction = Complete<Schemas["Prediction"]>;
export type MlPredictionReport = NonNullable<MlPrediction["summary"]>;
export type MlPredictionDistribution = MlPredictionReport["distribution"];
export type MlPredictionRows = Complete<Schemas["PredictionRows"]>;
export type MlRowExplanation = Complete<Schemas["RowExplanation"]>;
/** A KPI as sent and saved: "auto" (or a missing field) lets the app choose. Fields with a
 *  server default are optional, nested ones included. */
type KpiFormat = Partial<Schemas["Format"]>;
type KpiTableView = Partial<Schemas["TableView"]>;
export type KpiDefinition = Omit<
  Partial<Schemas["Definition-Input"]>,
  "measures" | "ratio" | "time" | "view" | "groups"
> & {
  measures: (Omit<Schemas["Measure"], "format"> & { format?: KpiFormat })[];
  ratio?: (Omit<Partial<Schemas["Ratio"]>, "format"> & { numerator: string; denominator: string; format?: KpiFormat }) | null;
  time?: (Omit<Partial<Schemas["Time"]>, "period"> & { column: string; period?: Partial<Schemas["Period"]> }) | null;
  groups?: { column: string }[];
  view?: Omit<Partial<Schemas["View"]>, "table"> & { table?: KpiTableView };
};
export type KpiResult = Complete<Schemas["KpiResult"]>;
export type KpiSuggestions = Complete<Schemas["Suggestions"]>;
export type SavedKpi = Complete<Schemas["SavedKpi"]>;
export type KpiDecision = KpiResult["resolved"]["layout"];
export type KpiMeasure = KpiResult["measures"][number];
export type KpiSeries = KpiResult["series"][number];
export type KpiTile = KpiResult["tiles"][number];
export type KpiNumberFormat = KpiMeasure["format"];
export type SeriesSetup = SeriesOverview["setup"];
export type SeriesLine = SeriesOverview["analyzed"];
export type MethodResult = SeriesOverview["methods"][number];
export type SeasonalProfile = SeriesSeasonality["profiles"][number];
export type SeriesSectionKey =
  | "overview"
  | "smoothing"
  | "seasonality"
  | "decomposition"
  | "stationarity"
  | "autocorrelation"
  | "volatility"
  | "residuals"
  | "breaks"
  | "groups"
  | "drivers";
/** Time series page parameters; undefined = automatic. */
export type SeriesParams = {
  time?: string;
  value?: string;
  group?: string;
  series?: string;
  period?: string;
  agg?: string;
  window?: number;
  align?: string;
  gaps?: string;
  season?: number;
};
export type CompareParams = {
  x: string;
  y: string;
  z?: string | null;
  color?: string | null;
  period?: string;
  agg?: string;
  window?: number;
  align?: string;
  /** Whole-number columns kept as numbers (otherwise few values read as categories). */
  numbers?: string[];
};
export type OutlierMethod = Outliers["method"];
export type AppSettings = Schemas["AppSettings"];
export type DetectionSettings = Required<Schemas["DetectionSettings"]>;
export type ComputeSettings = Schemas["ComputeSettings"];
export type VersionSettings = Required<Schemas["VersionSettings"]>;
export type KeepPolicy = VersionSettings["keep"];
export type ServerSettings = { detection: DetectionSettings; compute: ComputeSettings; versions: VersionSettings };
/** What changed between two versions (031). */
export type Changes = Complete<Schemas["Changes"]>;
export type ChangeLevel = Changes["level"];
export type Shift = Changes["distributions"][number];
export type LevelShift = Shift["levels"][number];
export type SchemaChange = Changes["schema"][number];
export type QualityChange = Changes["quality"][number];
export type DateChange = Changes["dates"][number];
export type VersionInfo = Complete<Schemas["VersionInfo"]>;
export type RowsAnswer = Complete<Schemas["RowsAnswer"]>;
export type RowChanges = NonNullable<RowsAnswer["result"]>;
export type KpiChange = Complete<Schemas["KpiChange"]>;
export type RelationsAnswer = Complete<Schemas["RelationsAnswer"]>;
export type RelationChange = RelationsAnswer["pairs"][number];
export type GpuStatus = Complete<Schemas["GpuStatus"]>;
export type SyntheticPlan = Complete<Schemas["Plan"]>;
export type SyntheticColumnPlan = SyntheticPlan["columns"][number];
export type SyntheticMethod = SyntheticPlan["method"];
export type SyntheticTreatment = SyntheticColumnPlan["treatment"];
export type SyntheticJob = Complete<Schemas["SyntheticJob"]>;
export type ReportJob = Complete<Schemas["ReportJob"]>;
export type SyntheticRun = Complete<Schemas["SyntheticRun"]>;
export type SyntheticReport = SyntheticRun["report"];
export type SyntheticRunSummary = Complete<Schemas["SyntheticRunSummary"]>;
export type SyntheticPreview = Complete<Schemas["SyntheticPreview"]>;
export type SyntheticColumnDetail = Complete<Schemas["ColumnDetail"]>;
export type SyntheticPair = Complete<Schemas["PairDetail"]>;
export type SyntheticUtility = Complete<Schemas["Utility"]>;
export type SyntheticDisclosure = Complete<Schemas["Disclosure"]>;
export type SyntheticOrigin = Complete<Schemas["Origin"]>;
/** What the user changed from the plan; everything else stays automatic. */
export type SyntheticRequest = {
  method?: SyntheticMethod;
  rows?: number;
  seed?: number;
  smooth?: boolean;
  drop_copies?: boolean;
  treatments?: Record<string, SyntheticTreatment>;
  bounds?: Record<string, { minimum?: number | null; maximum?: number | null }>;
  epochs?: number | null;
  device?: "auto" | "cpu" | "gpu";
  sequence?: {
    time?: string | null;
    entity?: string | null;
    no_entity?: boolean;
    length?: number | null;
    sequences?: number | null;
    steps?: number | null;
  };
};
export type StorageInfo = Schemas["StorageInfo"];
export type RowsColumn = Schemas["RowsColumn"];
/** A cell: the typed value, or the stored text when the column's type cannot read it. */
export type Cell = string | number | boolean | null | { invalid: string };
export type RowsPage = Omit<Schemas["RowsPage"], "rows"> & {
  rows: [number, ...Cell[]][];
};
export type RowsMode = "first" | "sample" | "duplicates" | "missing";
export type RowsView = {
  mode: RowsMode;
  search: string;
  sort: string | null;
  desc: boolean;
};
export type ExportFormat = "csv" | "parquet" | "xlsx";

function viewQuery(view: RowsView, filters?: Condition[]): URLSearchParams {
  const query = new URLSearchParams({
    mode: view.mode,
    ...filtersParam(filters),
  });
  if (view.search.trim()) query.set("search", view.search.trim());
  if (view.sort) query.set("sort", view.sort);
  if (view.sort && view.desc) query.set("desc", "true");
  return query;
}

/** Download links: the browser streams the file (no copy of a large export in memory). */
export function exportUrl(
  id: string,
  format: ExportFormat,
  view: RowsView,
  columns: string[] | null,
  filters?: Condition[],
) {
  const query = viewQuery(view, filters);
  query.set("format", format);
  if (columns) query.set("columns", columns.join(","));
  return `/api/v1/datasets/${id}/export?${query}`;
}

export function summaryUrl(id: string, filters?: Condition[]) {
  return `/api/v1/datasets/${id}/export/summary?${new URLSearchParams(filtersParam(filters))}`;
}

/** `vs`: another dataset's id, or `@n` for an earlier import n of this dataset (095). */
function versusQuery(vs: string | null, extra: Record<string, string> = {}): string {
  const query = new URLSearchParams(extra);
  if (vs?.startsWith("@")) query.set("version", vs.slice(1));
  else if (vs) query.set("vs", vs);
  const text = query.toString();
  return text ? `?${text}` : "";
}

/** Every difference of the row by row comparison, as CSV (031); without a key, paired as the
 *  Rows tab pairs them by default (the cleaning's row map, 095). */
export function changeRowsUrl(id: string, vs: string | null, key: string | null) {
  return `/api/v1/datasets/${id}/changes/rows/export${versusQuery(vs, key ? { key } : {})}`;
}

export function reportFileUrl(jobId: string) {
  return `/api/v1/reports/jobs/${jobId}/file`;
}

function chartQuery(
  column: string,
  params: ChartParams,
  filters?: Condition[],
): string {
  const query = new URLSearchParams({ column, ...filtersParam(filters) });
  if (params.view) query.set("view", params.view);
  if (params.topN) query.set("top_n", String(params.topN));
  if (params.full) query.set("full", "true");
  if (params.step) query.set("step", params.step);
  if (params.bins) query.set("bins", String(params.bins));
  return query.toString();
}

export class ApiError extends Error {
  readonly status: number;
  /** The error body's code and extra fields (e.g. `field` to fix, `suggestion`). */
  readonly details: Record<string, unknown>;

  constructor(status: number, message: string, details: Record<string, unknown> = {}) {
    super(message);
    this.status = status;
    this.details = details;
  }
}

export async function apiError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as {
      error?: { message?: string } & Record<string, unknown>;
      detail?: unknown;
    };
    if (body.error?.message) return new ApiError(response.status, body.error.message, body.error);
  } catch {
    // Not JSON: fall back to the status text.
  }
  return new ApiError(response.status, `Request failed (${response.status})`);
}

function mlQuery(
  target: string,
  model: string | null,
  settings: Record<string, unknown> | null,
  filters?: Condition[],
  classes?: MlClasses | null,
) {
  const query = new URLSearchParams({ target, ...filtersParam(filters) });
  if (model) query.set("model", model);
  if (settings) query.set("settings", JSON.stringify(settings));
  if (classes) query.set("classes", JSON.stringify(classes));
  return query;
}

export async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const response = await fetch(`/api/v1${path}`, {
    method,
    headers: {
      Accept: "application/json",
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw await apiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("GET", "/health"),

  datasets: () => request<Dataset[]>("GET", "/datasets"),
  dataset: (id: string) => request<Dataset>("GET", `/datasets/${id}`),
  importDataset: (path: string, name: string | null, options: ImportOptions) =>
    request<ImportResponse>("POST", "/datasets", { path, name, options }),
  renameDataset: (id: string, name: string) =>
    request<Dataset>("PATCH", `/datasets/${id}`, { name }),
  deleteDataset: (id: string) => request<void>("DELETE", `/datasets/${id}`),
  reimportDataset: (id: string) =>
    request<Job>("POST", `/datasets/${id}/reimport`),
  openDataset: (id: string) => request<Dataset>("POST", `/datasets/${id}/open`),
  schema: (id: string) =>
    request<ColumnInfo[]>("GET", `/datasets/${id}/schema`),
  head: (id: string, limit = 20) =>
    request<Head>("GET", `/datasets/${id}/head?limit=${limit}`),

  columns: (id: string) => request<Column[]>("GET", `/datasets/${id}/columns`),
  updateColumns: (id: string, changes: ColumnChange[]) =>
    request<Column[]>("PATCH", `/datasets/${id}/columns`, { changes }),
  detectColumns: (id: string) =>
    request<Column[]>("POST", `/datasets/${id}/columns/detect`),
  conversion: (id: string, column: string, type: AnalysisType) =>
    request<Conversion>(
      "GET",
      `/datasets/${id}/columns/conversion?column=${encodeURIComponent(column)}&type=${type}`,
    ),
  reviewTypes: (id: string) =>
    request<Dataset>("POST", `/datasets/${id}/columns/review`),
  roles: () => request<Role[]>("GET", "/column-roles"),
  overview: (id: string, filters?: Condition[]) =>
    request<Overview>(
      "GET",
      `/datasets/${id}/overview?${new URLSearchParams(filtersParam(filters))}`,
    ),
  chart: (
    id: string,
    column: string,
    params: ChartParams,
    filters?: Condition[],
  ) =>
    request<Chart>(
      "GET",
      `/datasets/${id}/columns/chart?${chartQuery(column, params, filters)}`,
    ),
  summaries: (id: string, filters: Condition[]) =>
    request<Record<string, ColumnSummary>>(
      "GET",
      `/datasets/${id}/columns/summaries?${new URLSearchParams(filtersParam(filters))}`,
    ),
  variable: (id: string, column: string, filters?: Condition[]) =>
    request<VariableProfile>(
      "GET",
      `/datasets/${id}/variable?${new URLSearchParams({ column, ...filtersParam(filters) })}`,
    ),
  fit: (id: string, column: string, filters?: Condition[]) =>
    request<Fit>(
      "GET",
      `/datasets/${id}/variable/fit?${new URLSearchParams({ column, ...filtersParam(filters) })}`,
    ),
  outliers: (
    id: string,
    column: string,
    method: OutlierMethod,
    filters?: Condition[],
  ) =>
    request<Outliers>(
      "GET",
      `/datasets/${id}/variable/outliers?${new URLSearchParams({ column, method, ...filtersParam(filters) })}`,
    ),
  tail: (id: string, column: string, filters?: Condition[]) =>
    request<Tail>(
      "GET",
      `/datasets/${id}/variable/tail?${new URLSearchParams({ column, ...filtersParam(filters) })}`,
    ),
  compare: (id: string, params: CompareParams, filters?: Condition[]) => {
    const query = new URLSearchParams({
      x: params.x,
      y: params.y,
      ...filtersParam(filters),
    });
    if (params.z) query.set("z", params.z);
    if (params.color) query.set("color", params.color);
    if (params.period) query.set("period", params.period);
    if (params.agg) query.set("agg", params.agg);
    if (params.window) query.set("window", String(params.window));
    if (params.align) query.set("align", params.align);
    if (params.numbers?.length) query.set("numbers", params.numbers.join(","));
    return request<Comparison>("GET", `/datasets/${id}/compare?${query}`);
  },
  seriesSection: <T>(
    id: string,
    section: SeriesSectionKey,
    params: SeriesParams,
    extra: Record<string, string>,
    filters?: Condition[],
  ) => {
    const query = new URLSearchParams({ ...filtersParam(filters), ...extra });
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<T>(
      "GET",
      `/modules/timeseries/datasets/${id}/${section}?${query}`,
    );
  },
  startForecast: (
    id: string,
    params: SeriesParams,
    extra: Record<string, string>,
    filters?: Condition[],
    drivers?: DriverSpec[],
  ) => {
    const query = new URLSearchParams({ ...filtersParam(filters), ...extra });
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<ForecastJob>(
      "POST",
      `/modules/timeseries/datasets/${id}/forecast?${query}`,
      drivers?.length ? { drivers } : undefined,
    );
  },
  startGroupForecast: (
    id: string,
    params: SeriesParams,
    extra: Record<string, string>,
    filters?: Condition[],
  ) => {
    const query = new URLSearchParams({ ...filtersParam(filters), ...extra });
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<GroupJob>(
      "POST",
      `/modules/timeseries/datasets/${id}/forecast-groups?${query}`,
    );
  },
  groupJob: (jobId: string) =>
    request<GroupJob>("GET", `/modules/timeseries/group-forecast-jobs/${jobId}`),
  cancelGroupJob: (jobId: string) =>
    request<GroupJob>("DELETE", `/modules/timeseries/group-forecast-jobs/${jobId}`),
  /** The long CSV, read through the API (works in the browser version too). */
  groupCsv: async (jobId: string) => {
    const response = await fetch(
      `/api/v1/modules/timeseries/group-forecast-jobs/${jobId}/forecast.csv`,
    );
    if (!response.ok) throw await apiError(response);
    return response.text();
  },
  scenario: (
    id: string,
    params: SeriesParams,
    extra: Record<string, string>,
    filters: Condition[] | undefined,
    drivers: DriverSpec[],
    changes: ScenarioChange[],
  ) => {
    const query = new URLSearchParams({ ...filtersParam(filters), ...extra });
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<DriverForecast>(
      "POST",
      `/modules/timeseries/datasets/${id}/scenario?${query}`,
      { drivers, changes },
    );
  },
  forecastJob: (jobId: string) =>
    request<ForecastJob>("GET", `/modules/timeseries/forecast-jobs/${jobId}`),
  cancelForecast: (jobId: string) =>
    request<ForecastJob>(
      "DELETE",
      `/modules/timeseries/forecast-jobs/${jobId}`,
    ),
  mlTargets: (id: string, filters?: Condition[]) =>
    request<MlTargets>("GET", `/modules/ml/datasets/${id}/targets?${new URLSearchParams(filtersParam(filters))}`),
  mlCatalog: (context: MlContext) => {
    const query = new URLSearchParams({
      task: context.task,
      rows: String(Math.round(context.rows)),
      features: String(context.features),
      categorical_share: context.categoricalShare.toFixed(3),
      missing_share: context.missingShare.toFixed(3),
    });
    if (context.minority !== null) query.set("minority", String(context.minority));
    return request<MlCatalogue>("GET", `/modules/ml/catalog?${query}`);
  },
  mlTrain: (
    id: string,
    target: string,
    model: string | null,
    settings: Record<string, unknown> | null,
    filters?: Condition[],
    choices?: MlChoicesInput | null,
    classes?: MlClasses | null,
    validation?: MlValidation | null,
  ) => {
    const query = mlQuery(target, model, settings, filters, classes);
    if (validation) query.set("validation", JSON.stringify(validation));
    return request<MlJob>("POST", `/modules/ml/datasets/${id}/runs?${query}`, choices ?? undefined);
  },
  mlPreparation: (
    id: string,
    target: string,
    model: string | null,
    settings: Record<string, unknown> | null,
    choices: MlChoicesInput,
    filters?: Condition[],
    classes?: MlClasses | null,
    validation?: MlValidation | null,
  ) => {
    const query = mlQuery(target, model, settings, filters, classes);
    if (validation) query.set("validation", JSON.stringify(validation));
    return request<MlPreparationPreview>("POST", `/modules/ml/datasets/${id}/preparation?${query}`, choices);
  },
  mlDistribution: (id: string, column: string, filters?: Condition[]) =>
    request<MlDistribution>(
      "GET",
      `/modules/ml/datasets/${id}/targets/${encodeURIComponent(column)}/distribution?${new URLSearchParams(filtersParam(filters))}`,
    ),
  mlJob: (jobId: string) => request<MlJob>("GET", `/modules/ml/jobs/${jobId}`),
  mlCancel: (jobId: string) => request<MlJob>("DELETE", `/modules/ml/jobs/${jobId}`),
  mlEstimate: (
    id: string,
    target: string,
    model: string | null,
    settings: Record<string, unknown> | null,
    choices: MlChoicesInput | null,
    classes: MlClasses | null,
    validation: MlValidation | null,
    filters?: Condition[],
  ) => {
    const query = mlQuery(target, model, settings, filters, classes);
    if (validation) query.set("validation", JSON.stringify(validation));
    return request<MlEstimate>("POST", `/modules/ml/datasets/${id}/estimate?${query}`, choices ?? undefined);
  },
  mlRuns: (id: string, target: string) =>
    request<MlRunSummary[]>("GET", `/modules/ml/datasets/${id}/runs?${new URLSearchParams({ target })}`),
  mlRun: (id: string, runId: string) => request<MlRun>("GET", `/modules/ml/datasets/${id}/runs/${runId}`),
  mlExplanation: (id: string, runId: string) =>
    request<MlExplanationState>("GET", `/modules/ml/datasets/${id}/runs/${runId}/explanation`),
  mlErrors: (id: string, runId: string, measure: string | null) =>
    request<MlErrors>("GET", `/modules/ml/datasets/${id}/runs/${runId}/errors${measure ? `?${new URLSearchParams({ measure })}` : ""}`),
  mlErrorsCompared: (id: string, runId: string, otherId: string) =>
    request<MlErrorComparison>("GET", `/modules/ml/datasets/${id}/runs/${runId}/errors/compare/${otherId}`),
  mlFairness: (id: string, runId: string, column: string) =>
    request<MlFairness>("GET", `/modules/ml/datasets/${id}/runs/${runId}/errors/fairness?${new URLSearchParams({ column })}`),
  mlCompare: (
    id: string,
    target: string,
    models: string[],
    choices: MlChoicesInput | null,
    classes: MlClasses | null,
    validation: MlValidation | null,
    filters?: Condition[],
  ) => {
    const query = mlQuery(target, null, null, filters, classes);
    if (validation) query.set("validation", JSON.stringify(validation));
    return request<MlCompareJob>("POST", `/modules/ml/datasets/${id}/compare?${query}`, { models, preparation: choices });
  },
  mlComparison: (jobId: string) => request<MlCompareJob>("GET", `/modules/ml/compare-jobs/${jobId}`),
  mlTune: (id: string, target: string, body: MlTuneRequest, classes: MlClasses | null, validation: MlValidation | null, filters?: Condition[]) => {
    const query = mlQuery(target, null, null, filters, classes);
    if (validation) query.set("validation", JSON.stringify(validation));
    return request<MlTuneJob>("POST", `/modules/ml/datasets/${id}/tune?${query}`, body);
  },
  mlTuning: (jobId: string) => request<MlTuneJob>("GET", `/modules/ml/tune-jobs/${jobId}`),
  mlStopTuning: (jobId: string) => request<MlTuneJob>("DELETE", `/modules/ml/tune-jobs/${jobId}`),
  mlAuto: (id: string, target: string, body: MlAutoRequest, classes: MlClasses | null, validation: MlValidation | null, filters?: Condition[]) => {
    const query = mlQuery(target, null, null, filters, classes);
    if (validation) query.set("validation", JSON.stringify(validation));
    return request<MlAutoJob>("POST", `/modules/ml/datasets/${id}/auto?${query}`, body);
  },
  mlAutoJob: (jobId: string) => request<MlAutoJob>("GET", `/modules/ml/auto-jobs/${jobId}`),
  mlStopAuto: (jobId: string) => request<MlAutoJob>("DELETE", `/modules/ml/auto-jobs/${jobId}`),
  mlPredictionsUrl: (id: string, runId: string) => `/api/v1/modules/ml/datasets/${id}/runs/${runId}/predictions.csv`,
  mlPanel: (id: string, target: string, period: string | null, entity: string | null, filters?: Condition[]) => {
    const query = new URLSearchParams({ target, ...filtersParam(filters) });
    if (period) query.set("period", period);
    if (entity) query.set("entity", entity);
    return request<MlPanelCheck>("GET", `/modules/ml/datasets/${id}/panel?${query}`);
  },
  mlNeutralize: (id: string, runId: string, columns: string[], proportion: number) =>
    request<MlNeutralized>("POST", `/modules/ml/datasets/${id}/runs/${runId}/neutralize`, { columns, proportion }),
  mlPeriodsUrl: (id: string, runId: string) => `/api/v1/modules/ml/datasets/${id}/runs/${runId}/periods.csv`,
  // Predict on new data, export the model (037)
  mlCheck: (id: string, runId: string, source: string) =>
    request<MlModelCheck>("GET", `/modules/ml/datasets/${id}/runs/${runId}/check?${new URLSearchParams({ source })}`),
  mlPredict: (id: string, runId: string, source: string, keep: string[]) =>
    request<MlPredictionJob>("POST", `/modules/ml/datasets/${id}/runs/${runId}/predictions`, { source, keep }),
  mlPredictJob: (jobId: string) => request<MlPredictionJob>("GET", `/modules/ml/predict-jobs/${jobId}`),
  mlCancelPredict: (jobId: string) => request<MlPredictionJob>("DELETE", `/modules/ml/predict-jobs/${jobId}`),
  mlPredictions: (id: string, runId: string) =>
    request<MlPredictionSummary[]>("GET", `/modules/ml/datasets/${id}/runs/${runId}/predictions`),
  mlPrediction: (id: string, predictionId: string) =>
    request<MlPrediction>("GET", `/modules/ml/datasets/${id}/predictions/${predictionId}`),
  mlPredictionRows: (id: string, predictionId: string, offset: number, limit: number, lessReliable: boolean) => {
    const query = new URLSearchParams({ offset: String(offset), limit: String(limit) });
    if (lessReliable) query.set("less_reliable", "true");
    return request<MlPredictionRows>("GET", `/modules/ml/datasets/${id}/predictions/${predictionId}/rows?${query}`);
  },
  mlPredictionRowExplanation: (id: string, predictionId: string, row: number) =>
    request<MlRowExplanation>("GET", `/modules/ml/datasets/${id}/predictions/${predictionId}/rows/${row}/explanation`),
  mlSavePrediction: (id: string, predictionId: string) =>
    request<{ dataset_id: string }>("POST", `/modules/ml/datasets/${id}/predictions/${predictionId}/dataset`),
  mlPredictionCsvUrl: (id: string, predictionId: string) =>
    `/api/v1/modules/ml/datasets/${id}/predictions/${predictionId}/predictions.csv`,
  mlModelZipUrl: (id: string, runId: string) => `/api/v1/modules/ml/datasets/${id}/runs/${runId}/model.zip`,
  correlations: (
    id: string,
    method: CorrelationMethod,
    columns: string[] | null,
    filters?: Condition[],
    correction: Correction = "auto",
  ) => {
    const query = new URLSearchParams({ method, ...filtersParam(filters) });
    if (columns) query.set("columns", columns.join(","));
    if (correction !== "auto") query.set("correction", correction);
    return request<Correlations>(
      "GET",
      `/datasets/${id}/correlations?${query}`,
    );
  },
  structure: (id: string, columns: string[] | null, filters?: Condition[]) => {
    const query = new URLSearchParams(filtersParam(filters));
    if (columns) query.set("columns", columns.join(","));
    return request<Structure>("GET", `/datasets/${id}/correlations/structure?${query}`);
  },
  factors: (id: string, columns: string[] | null, filters?: Condition[]) => {
    const query = new URLSearchParams(filtersParam(filters));
    if (columns) query.set("columns", columns.join(","));
    return request<Factors>("GET", `/datasets/${id}/correlations/factors?${query}`);
  },
  regression: (id: string, params: RegressionParams, filters?: Condition[]) =>
    request<Regression>("GET", `/datasets/${id}/regression?${regressionQuery(params, filters)}`),
  regressionSuggestion: (id: string, target: string | undefined, filters?: Condition[]) =>
    request<RegressionSuggestion>(
      "GET",
      `/datasets/${id}/regression/suggestion?${regressionQuery({ target }, filters)}`,
    ),
  regressionSection: <S extends RegressionSection>(id: string, section: S, params: RegressionParams, filters?: Condition[]) =>
    request<RegressionSections[S]>("GET", `/datasets/${id}/regression/${section}?${regressionQuery(params, filters)}`),
  regressionInstruments: (id: string, params: RegressionParams, endogenous: string[], instruments: string[], filters?: Condition[]) => {
    const query = regressionQuery(params, filters);
    query.set("endogenous", endogenous.join(","));
    query.set("instruments", instruments.join(","));
    return request<RegressionIv>("GET", `/datasets/${id}/regression/instruments?${query}`);
  },
  rows: (
    id: string,
    view: RowsView,
    offset: number,
    limit: number,
    filters?: Condition[],
  ) => {
    const query = viewQuery(view, filters);
    query.set("offset", String(offset));
    query.set("limit", String(limit));
    return request<RowsPage>("GET", `/datasets/${id}/rows?${query}`);
  },
  kpiSuggestions: (id: string) =>
    request<KpiSuggestions>("GET", `/modules/kpi/datasets/${id}/suggestions`),
  kpiCompute: (id: string, definition: KpiDefinition, filters?: Condition[]) =>
    request<KpiResult>(
      "POST",
      `/modules/kpi/datasets/${id}/compute?${new URLSearchParams(filtersParam(filters))}`,
      definition,
    ),
  kpis: (id: string) => request<SavedKpi[]>("GET", `/modules/kpi/datasets/${id}/definitions`),
  createKpi: (id: string, name: string, definition: KpiDefinition) =>
    request<SavedKpi>("POST", `/modules/kpi/datasets/${id}/definitions`, { name, definition }),
  updateKpi: (id: string, kpiId: string, change: { name?: string; definition?: KpiDefinition }) =>
    request<SavedKpi>("PUT", `/modules/kpi/datasets/${id}/definitions/${kpiId}`, change),
  duplicateKpi: (id: string, kpiId: string) =>
    request<SavedKpi>("POST", `/modules/kpi/datasets/${id}/definitions/${kpiId}/duplicate`),
  deleteKpi: (id: string, kpiId: string) =>
    request<void>("DELETE", `/modules/kpi/datasets/${id}/definitions/${kpiId}`),
  savedFilters: (id: string) =>
    request<SavedFilter[]>("GET", `/datasets/${id}/saved-filters`),
  saveFilter: (id: string, name: string, conditions: Condition[]) =>
    request<SavedFilter>("POST", `/datasets/${id}/saved-filters`, {
      name,
      conditions,
    }),
  deleteSavedFilter: (id: string, filterId: string) =>
    request<void>("DELETE", `/datasets/${id}/saved-filters/${filterId}`),

  settings: () => request<ServerSettings>("GET", "/settings"),
  updateSettings: (settings: ServerSettings) =>
    request<ServerSettings>("PUT", "/settings", settings),
  gpu: (refresh = false) => request<GpuStatus>("GET", `/settings/gpu${refresh ? "?refresh=true" : ""}`),
  syntheticPlan: (id: string) => request<SyntheticPlan>("GET", `/modules/synthetic/datasets/${id}/plan`),
  syntheticGenerate: (id: string, body: SyntheticRequest) =>
    request<SyntheticJob>("POST", `/modules/synthetic/datasets/${id}/runs`, body),
  reportStart: (id: string, filters: Condition[]) =>
    request<ReportJob>("POST", `/datasets/${id}/report?${new URLSearchParams(filtersParam(filters))}`),
  reportJob: (jobId: string) => request<ReportJob>("GET", `/reports/jobs/${jobId}`),
  reportCancel: (jobId: string) => request<ReportJob>("DELETE", `/reports/jobs/${jobId}`),
  syntheticJob: (jobId: string) => request<SyntheticJob>("GET", `/modules/synthetic/jobs/${jobId}`),
  syntheticCancel: (jobId: string) => request<SyntheticJob>("DELETE", `/modules/synthetic/jobs/${jobId}`),
  syntheticRuns: (id: string) => request<SyntheticRunSummary[]>("GET", `/modules/synthetic/datasets/${id}/runs`),
  syntheticRun: (id: string, runId: string) =>
    request<SyntheticRun>("GET", `/modules/synthetic/datasets/${id}/runs/${runId}`),
  syntheticCompare: (id: string, runIds: string[]) =>
    request<SyntheticRunSummary[]>(
      "GET",
      `/modules/synthetic/datasets/${id}/runs/compare?runs=${runIds.map(encodeURIComponent).join(",")}`,
    ),
  syntheticDelete: (id: string, runId: string) =>
    request<void>("DELETE", `/modules/synthetic/datasets/${id}/runs/${runId}`),
  syntheticPreview: (id: string, runId: string) =>
    request<SyntheticPreview>("GET", `/modules/synthetic/datasets/${id}/runs/${runId}/preview`),
  syntheticDownloadUrl: (id: string, runId: string, format: "csv" | "parquet") =>
    `/api/v1/modules/synthetic/datasets/${id}/runs/${runId}/download?format=${format}`,
  syntheticSave: (id: string, runId: string) =>
    request<{ dataset_id: string }>("POST", `/modules/synthetic/datasets/${id}/runs/${runId}/dataset`),
  syntheticUtility: (id: string, runId: string, target: string) =>
    request<SyntheticUtility>("POST", `/modules/synthetic/datasets/${id}/runs/${runId}/utility`, { target }),
  syntheticDisclosure: (id: string, runId: string, known: string[], sensitive: string) =>
    request<SyntheticDisclosure>("POST", `/modules/synthetic/datasets/${id}/runs/${runId}/disclosure`, {
      known,
      sensitive,
    }),
  syntheticColumn: (id: string, runId: string, column: string, bins: number | null, target: string | null) => {
    const query = new URLSearchParams();
    if (bins) query.set("bins", String(bins));
    if (target) query.set("target", target);
    return request<SyntheticColumnDetail>(
      "GET",
      `/modules/synthetic/datasets/${id}/runs/${runId}/columns/${encodeURIComponent(column)}?${query}`,
    );
  },
  syntheticPair: (id: string, runId: string, a: string, b: string) =>
    request<SyntheticPair>("GET", `/modules/synthetic/datasets/${id}/runs/${runId}/pairs?${new URLSearchParams({ a, b })}`),
  syntheticOrigin: (id: string) => request<SyntheticOrigin | null>("GET", `/modules/synthetic/origin/${id}`),
  storage: () => request<StorageInfo>("GET", "/storage"),
  versions: (id: string) => request<VersionInfo[]>("GET", `/datasets/${id}/versions`),
  changes: (id: string, vs: string | null) =>
    request<Changes>("GET", `/datasets/${id}/changes${versusQuery(vs)}`),
  changeRows: (id: string, vs: string | null, key: string | null) =>
    request<RowsAnswer>("GET", `/datasets/${id}/changes/rows${versusQuery(vs, key ? { key } : {})}`),
  changeKpis: (id: string, vs: string | null) =>
    request<KpiChange[]>("GET", `/datasets/${id}/changes/kpis${versusQuery(vs)}`),
  changeRelations: (id: string, vs: string | null) =>
    request<RelationsAnswer>("GET", `/datasets/${id}/changes/relations${versusQuery(vs)}`),
  clearCaches: () => request<void>("POST", "/storage/clear-caches"),

  job: (id: string) => request<Job>("GET", `/jobs/${id}`),

  preview: (path: string, options: SourceOptions) =>
    request<Preview>("POST", "/sources/preview", { path, options }),
  browse: (path: string | null) =>
    request<Listing>(
      "GET",
      `/browse${path ? `?path=${encodeURIComponent(path)}` : ""}`,
    ),
  samples: () => request<Sample[]>("GET", "/samples"),

  upload: async (file: File): Promise<Upload> => {
    const response = await fetch(
      `/api/v1/uploads?filename=${encodeURIComponent(file.name)}`,
      {
        method: "PUT",
        body: file,
      },
    );
    if (!response.ok) throw await apiError(response);
    return (await response.json()) as Upload;
  },
};
