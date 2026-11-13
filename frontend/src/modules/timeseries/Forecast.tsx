import type { Data } from "plotly.js-dist-min";
import { useMemo, useState } from "react";

import type {
  DriverSpec,
  ForecastJob,
  SeriesDrivers,
  SeriesParams,
} from "../../api/client";
import {
  useCancelForecast,
  useForecastJob,
  useSeriesSection,
  useStartForecast,
} from "../../api/hooks";
import { Plot, type Figure } from "../../charts/Plot";
import { formatNumber, withAlpha } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import {
  Checkbox,
  ErrorNote,
  ProgressBar,
  Select,
  TextInput,
} from "../../ui/fields";
import {
  DriverResults,
  DriverSuggestion,
  DriversPanel,
  driverLabel,
  shownDrivers,
  usableDrivers,
} from "./Drivers";
import { EveryGroup } from "./EveryGroup";
import { componentsFigure } from "./figures";
import { ForecastOptions } from "./ForecastOptions";
import {
  aiBadge,
  aiReasons,
  aiTitle,
  useAiModels,
  useDownloadModel,
  useStopDownload,
  type AiModel,
  type ModelSuggestion,
} from "./forecastAi";
import { DownloadDialog, DownloadProgress } from "./ModelDownload";
import { MethodsRow, useMethods } from "./Methods";
import { ScenariosPanel } from "./Scenarios";
import {
  forecastTuning,
  type StoredScenario,
  type TimeSeriesSearch,
} from "./search";

type Result = NonNullable<ForecastJob["result"]>;
type ModelResult = Result["results"][number];
type Verdict = { text: string; tone: "ok" | "none"; detail?: string } | null;

const NO_SCENARIOS: StoredScenario[] = [];
const NO_SEARCH: TimeSeriesSearch = {};
const HORIZONS = [3, 6, 8, 12, 24, 30, 52, 90];
const INTERMITTENT = ["croston", "sba", "tsb", "adida"];
/** Models with several seasons, suggested when the history covers two cycles or more (028 A6). */
const SEVERAL_SEASONS = ["prophet_like", "stlf"];
const CYCLE_NAMES = new Set(["daily", "weekly", "yearly"]);
const cycleName = (name: string) =>
  CYCLE_NAMES.has(name) ? t(`ts.cycle.${name}` as MessageKey) : name;
const modelName = (key: string) => t(`tsm.${key}` as MessageKey);

function forecastFigure(
  result: Result,
  shown: ModelResult,
  others: ModelResult[],
  what: string,
): Figure {
  return (tokens) => {
    const color = tokens.palette[0] ?? tokens.date;
    const future = result.future;
    const band = (
      low: number[],
      high: number[],
      alpha: number,
      name: string,
    ): Data[] => [
      {
        type: "scatter",
        mode: "lines",
        x: future,
        y: low,
        line: { width: 0 },
        hoverinfo: "skip",
        showlegend: false,
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        name,
        x: future,
        y: high,
        fill: "tonexty",
        fillcolor: withAlpha(color, alpha),
        line: { width: 0 },
        hoverinfo: "skip",
      } as Data,
    ];
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          name: what,
          x: result.periods,
          y: result.history,
          line: { color: tokens.ink2, width: 1.5 },
          hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
        } as Data,
        ...band(shown.low95, shown.high95, 0.12, "95 % interval"),
        ...band(shown.low80, shown.high80, 0.22, "80 % interval"),
        ...others.map(
          (model, i) =>
            ({
              type: "scatter",
              mode: "lines",
              name: modelName(model.key),
              x: future,
              y: model.forecast,
              line: {
                color: tokens.palette[(i + 1) % 8] ?? tokens.ink3,
                width: 1.25,
                dash: "dot",
              },
              hovertemplate: `${modelName(model.key)}: %{y:,.4g}<extra></extra>`,
            }) as Data,
        ),
        {
          type: "scatter",
          mode: "lines+markers",
          name: shown.info ?? modelName(shown.key),
          x: future,
          y: shown.forecast,
          line: { color, width: 2.5 },
          marker: { size: 5 },
          hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
        hovermode: "x unified",
        xaxis: {
          type: "date",
          rangeslider: { visible: result.periods.length > 30, thickness: 0.08 },
        },
      },
    };
  };
}

/** Each rolling origin of a model: what it forecast from there, against what happened. */
function backtestFigure(result: Result, model: ModelResult): Figure {
  return (tokens) => {
    const first = Math.min(...model.backtests.map((b) => b.origin));
    const start = Math.max(0, first - 2 * result.horizon);
    const data: Data[] = [
      {
        type: "scatter",
        mode: "lines",
        name: t("ts.backtest.actual"),
        x: result.periods.slice(start),
        y: result.history.slice(start),
        line: { color: tokens.ink2, width: 1.5 },
        hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
      } as Data,
    ];
    model.backtests.forEach((backtest, i) => {
      const color = tokens.palette[i % 8] ?? tokens.date;
      const x = result.periods.slice(
        backtest.origin,
        backtest.origin + backtest.forecast.length,
      );
      const name = t("ts.backtest.origin", {
        period: result.periods[backtest.origin] ?? "",
      });
      if (backtest.low95.every((v) => v !== null)) {
        data.push(
          {
            type: "scatter",
            mode: "lines",
            x,
            y: backtest.low95,
            line: { width: 0 },
            hoverinfo: "skip",
            showlegend: false,
            legendgroup: name,
          } as Data,
          {
            type: "scatter",
            mode: "lines",
            x,
            y: backtest.high95,
            fill: "tonexty",
            fillcolor: withAlpha(color, 0.1),
            line: { width: 0 },
            hoverinfo: "skip",
            showlegend: false,
            legendgroup: name,
          } as Data,
        );
      }
      data.push({
        type: "scatter",
        mode: "lines+markers",
        name,
        legendgroup: name,
        x,
        y: backtest.forecast,
        line: { color, width: 2 },
        marker: { size: 4 },
        hovertemplate: `${name}: %{y:,.4g}<extra></extra>`,
      } as Data);
    });
    return {
      data,
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.12, font: { size: 11 } },
        hovermode: "x unified",
        xaxis: { type: "date" },
      },
    };
  };
}

function downloadCsv(result: Result, model: ModelResult) {
  const header = "period,forecast,low80,high80,low95,high95,model";
  const label = (model.info ?? model.key).replaceAll('"', "'");
  const lines = result.future.map((period, i) =>
    [
      period,
      model.forecast[i],
      model.low80[i],
      model.high80[i],
      model.low95[i],
      model.high95[i],
      `"${label}"`,
    ].join(","),
  );
  const blob = new Blob([[header, ...lines].join("\n")], { type: "text/csv" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `forecast-${model.key}.csv`;
  link.click();
  URL.revokeObjectURL(link.href);
}

/** The components of a Prophet-like fit (trend, seasons, holidays, drivers), as rows of the
 * components chart, over the periods they cover. */
function componentsOf(result: Result, model: ModelResult) {
  const periods = result.periods.slice(model.components_start);
  return {
    periods,
    rows: model.components.map((c) => ({
      kind: c.kind,
      values: c.values,
      name:
        c.kind === "trend"
          ? t("fig.ts.trend")
          : c.kind === "season"
            ? cycleName(c.name)
            : driverLabel(c.name),
    })),
  };
}

/** 028 B2, B4: the engine's rule on the series' length, with its numbers; the button ticks the
 *  model (or asks for its download), or the reason it cannot run here is shown instead. */
function AiSuggestion({
  suggestion,
  chosen,
  reason,
  onAdd,
}: {
  suggestion: ModelSuggestion;
  chosen: string[];
  reason: (key: string) => string | undefined;
  onAdd: (key: string) => void;
}) {
  const key = suggestion.models[0];
  if (!key || chosen.includes(key)) return null;
  const why = reason(key);
  const text =
    suggestion.kind === "short_series"
      ? t("fcai.suggest.short", {
          periods: String(suggestion.periods),
          threshold: String(suggestion.threshold),
        })
      : t("fcai.suggest.long", { periods: String(suggestion.periods) });
  return (
    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-line bg-surface-1 px-3 py-2 text-[13px]">
      <span className="text-ink-1">{text}</span>
      {why ? (
        <span className="text-xs text-ink-3">{why}</span>
      ) : (
        <Button variant="ghost" onClick={() => onAdd(key)}>
          {t("fcai.suggest.try", { title: aiTitle(key) })}
        </Button>
      )}
    </div>
  );
}

const percent = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : `${(value * 100).toFixed(0)} %`;

export function ForecastSection({
  datasetId,
  params,
  period,
  what,
  defaultHorizon,
  scenarios = NO_SCENARIOS,
  scenarioModel,
  onScenarios = () => {},
  onPickSeries = () => {},
  drivers: chosenDrivers,
  onDrivers: setDrivers,
  search = NO_SEARCH,
  onOptions = () => {},
  periods,
}: {
  datasetId: string;
  params: SeriesParams;
  period: string;
  what: string;
  defaultHorizon: number;
  /** What-if scenarios kept in the URL (039). */
  scenarios?: StoredScenario[];
  scenarioModel?: string;
  onScenarios?: (scenarios: StoredScenario[] | undefined, model?: string) => void;
  /** A series picked among every group: it becomes the analysed one (039). */
  onPickSeries?: (label: string) => void;
  /** The chosen drivers, kept by the page so a saved analysis keeps them (030). */
  drivers: DriverSpec[];
  onDrivers: (drivers: DriverSpec[]) => void;
  /** The page's URL: the ⚙ Options of the models (028 A7) live there. */
  search?: TimeSeriesSearch;
  onOptions?: (patch: Partial<TimeSeriesSearch>) => void;
  /** Periods of the series: the trained networks need 500 (028 B4). */
  periods?: number;
}) {
  const [models, setModels] = useMethods("forecast");
  const aiModels = useAiModels();
  const download = useDownloadModel();
  const stopDownload = useStopDownload();
  const [asking, setAsking] = useState<AiModel | null>(null);
  const aiByKey = useMemo(
    () => new Map((aiModels.data?.models ?? []).map((m) => [m.key, m])),
    [aiModels.data],
  );
  const aiUnavailable = useMemo(
    () => aiReasons(aiModels.data?.models, periods),
    [aiModels.data, periods],
  );
  /** 028 B1: a pretrained model without its weights asks for the download before it is ticked. */
  const pickModel = (key: string) => {
    const found = aiByKey.get(key);
    if (found?.pretrained && found.state === "needs_download") {
      if (found.download !== "running") setAsking(found);
      return true;
    }
    return false;
  };
  const addSuggested = (key: string) => {
    if (!pickModel(key)) setModels([...models, key]);
  };
  const downloading = models
    .map((key) => aiByKey.get(key))
    .filter((m): m is AiModel => Boolean(m && m.pretrained && m.download !== "idle" && m.state !== "ready"));
  const [horizon, setHorizon] = useState<number | null>(null);
  const [order, setOrder] = useState("1,1,1");
  const [seasonalOrder, setSeasonalOrder] = useState("0,1,1");
  const [jobId, setJobId] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [overlay, setOverlay] = useState(false);
  const [allMetrics, setAllMetrics] = useState(false);
  const [scoring, setScoring] = useState(false);
  const [driversOpen, setDriversOpen] = useState(false);
  const [components, setComponents] = useState(false);
  const start = useStartForecast(datasetId);
  const cancel = useCancelForecast();
  const job = useForecastJob(jobId);
  const h = horizon ?? defaultHorizon;
  const candidates = useSeriesSection<SeriesDrivers>(
    datasetId,
    "drivers",
    params,
    { horizon: String(h) },
  ).data;
  const drivers = usableDrivers(chosenDrivers, candidates);
  const listed = shownDrivers(chosenDrivers, candidates);
  const tuning = useMemo(() => forecastTuning(search), [search]);
  const addDriver = (driver: DriverSpec) => {
    setDrivers([...listed, driver]);
    setDriversOpen(true);
  };

  const run = () =>
    start.mutate(
      {
        params,
        extra: {
          horizon: String(h),
          methods: models.join(","),
          ...(models.includes("sarima")
            ? { order, sorder: seasonalOrder }
            : {}),
        },
        drivers,
        tuning,
      },
      { onSuccess: (started) => setJobId(started.id) },
    );

  const state = job.data;
  const waiting = downloading.some((m) => m.download === "running");
  const running = state?.status === "running" || start.isPending;
  const result = state?.status === "done" ? state.result : null;
  const shown = result
    ? (result.results.find((r) => r.key === selected && r.scores) ??
      result.results.find((r) => r.key === result.best))
    : undefined;
  const figure = useMemo(
    () =>
      result && shown
        ? forecastFigure(
            result,
            shown,
            overlay
              ? result.results.filter((r) => r.scores && r.key !== shown.key)
              : [],
            what,
          )
        : null,
    [result, shown, overlay, what],
  );
  const scoringFigure = useMemo(
    () =>
      result && shown?.backtests.length ? backtestFigure(result, shown) : null,
    [result, shown],
  );
  const parts = useMemo(
    () => (result && shown?.components?.length ? componentsOf(result, shown) : null),
    [result, shown],
  );
  const partsFigure = useMemo(
    () =>
      result && parts
        ? componentsFigure(parts.periods, result.future, parts.rows)
        : null,
    [result, parts],
  );
  const demand = result?.demand?.intermittent ? result.demand : null;
  const missingIntermittent = INTERMITTENT.filter(
    (key) => !models.includes(key),
  );
  const cycles = result?.cycles ?? [];
  const missingSeasons =
    cycles.length >= 2
      ? SEVERAL_SEASONS.filter((key) => !models.includes(key))
      : [];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-ink-2">
          {t("ts.forecast.horizon")}
          <Select
            value={String(h)}
            onChange={(event) => setHorizon(Number(event.target.value))}
            className="h-7 text-xs"
          >
            {[...new Set([defaultHorizon, ...HORIZONS])]
              .sort((a, b) => a - b)
              .map((value) => (
                <option key={value} value={value}>
                  {value} {t(`time.period.${period}` as MessageKey)}
                  {value === defaultHorizon
                    ? ` (${t("ts.forecast.default")})`
                    : ""}
                </option>
              ))}
          </Select>
        </label>
        {models.includes("sarima") && (
          <>
            <label className="flex items-center gap-1.5 text-ink-2">
              {t("ts.forecast.order")}
              <TextInput
                value={order}
                onChange={(event) => setOrder(event.target.value)}
                className="h-7 w-20 text-xs"
              />
            </label>
            <label className="flex items-center gap-1.5 text-ink-2">
              {t("ts.forecast.seasonalOrder")}
              <TextInput
                value={seasonalOrder}
                onChange={(event) => setSeasonalOrder(event.target.value)}
                className="h-7 w-20 text-xs"
              />
            </label>
          </>
        )}
        <Button variant="primary" onClick={run} disabled={running || waiting}>
          {running ? t("ts.forecast.running") : t("ts.forecast.run")}
        </Button>
      </div>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <MethodsRow
          section="forecast"
          chosen={models}
          onChange={setModels}
          unavailable={aiUnavailable}
          onPick={pickModel}
          tag={(key) => aiBadge(aiByKey.get(key))}
        />
        <ForecastOptions models={models} search={search} onChange={onOptions} />
      </div>
      <DownloadProgress models={downloading} onStop={(key) => stopDownload.mutate(key)} />
      {download.isError && <ErrorNote>{download.error.message}</ErrorNote>}
      {asking && (
        <DownloadDialog
          model={asking}
          onClose={() => setAsking(null)}
          onConfirm={(key) => {
            download.mutate(key);
            if (!models.includes(key)) setModels([...models, key]);
          }}
        />
      )}
      <DriverSuggestion
        candidates={candidates}
        drivers={drivers}
        onAdd={addDriver}
      />
      <DriversPanel
        candidates={candidates}
        drivers={listed}
        onChange={setDrivers}
        open={driversOpen}
        onOpenChange={setDriversOpen}
        models={models}
        onAddModel={(key) => setModels([...models, key])}
        horizon={h}
      />

      {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
      {state?.status === "running" && (
        <div className="flex max-w-md items-center gap-3">
          <div className="flex-1">
            <ProgressBar
              value={state.total ? state.done / state.total : 0}
              label={t("ts.forecast.progress")}
            />
            <p className="mt-1 text-xs text-ink-3">
              {t("ts.forecast.fitting", {
                model: state.current ? modelName(state.current) : "…",
                done: String(state.done),
                total: String(state.total),
              })}
            </p>
          </div>
          <Button variant="ghost" onClick={() => jobId && cancel.mutate(jobId)}>
            {t("ts.forecast.cancel")}
          </Button>
        </div>
      )}
      {state?.status === "cancelled" && (
        <p className="text-ink-2">{t("ts.forecast.cancelled")}</p>
      )}
      {state?.status === "failed" && <ErrorNote>{state.error}</ErrorNote>}
      {!state && !running && (
        <p className="text-ink-3">{t("ts.forecast.idle")}</p>
      )}

      {result && shown && figure && (
        <div className="space-y-3">
          {forecastVerdict(state) && (
            <p className="font-medium text-ink-1">
              {forecastVerdict(state)?.text}
            </p>
          )}
          {result.driver_notes.length > 0 && (
            <ul className="space-y-0.5 text-[13px] text-warn">
              {result.driver_notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
          {demand && (
            <div className="rounded-md border border-line bg-surface-1 px-3 py-2 text-[13px]">
              <p className="text-ink-1">
                {t("ts.demand.title", {
                  pct: String(Math.round(demand.zeros_pct)),
                  kind: t(`ts.demand.kind.${demand.kind}` as MessageKey),
                })}
              </p>
              <p className="text-xs text-ink-3">
                {t("ts.demand.detail", {
                  adi: demand.adi.toFixed(1),
                  cv2: demand.cv2.toFixed(2),
                })}
              </p>
              {missingIntermittent.length > 0 && (
                <div className="mt-1.5 flex flex-wrap items-center gap-2">
                  <span className="text-xs text-ink-2">
                    {t("ts.demand.suggest")}
                  </span>
                  <Button
                    variant="ghost"
                    onClick={() =>
                      setModels([...models, ...missingIntermittent])
                    }
                  >
                    {t("ts.demand.add")}
                  </Button>
                </div>
              )}
            </div>
          )}
          {result.suggestions?.map((suggestion) => (
            <AiSuggestion
              key={suggestion.kind}
              suggestion={suggestion}
              chosen={models}
              reason={(key) => aiUnavailable[key]}
              onAdd={addSuggested}
            />
          ))}
          {missingSeasons.length > 0 && (
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-line bg-surface-1 px-3 py-2 text-[13px]">
              <span className="text-ink-1">
                {t("ts.cycles.found", {
                  count: String(cycles.length),
                  cycles: cycles.map(cycleName).join(", "),
                })}
              </span>
              <Button
                variant="ghost"
                onClick={() => setModels([...models, ...missingSeasons])}
              >
                {t("ts.cycles.add")}
              </Button>
            </div>
          )}
          <div className="rounded-lg border border-line bg-surface-1 p-3">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="text-ink-3">
                {t("ts.forecast.validation", {
                  origins: String(result.origins),
                  horizon: String(result.horizon),
                })}
              </span>
              <span className="flex items-center gap-3">
                <Checkbox
                  label={t("ts.forecast.overlay")}
                  checked={overlay}
                  onChange={(event) => setOverlay(event.target.checked)}
                />
                <Button
                  variant="ghost"
                  onClick={() => downloadCsv(result, shown)}
                >
                  {t("ts.forecast.export")}
                </Button>
              </span>
            </div>
            <Plot
              figure={figure}
              height={380}
              label={t("ts.section.forecast")}
            />
            {scoringFigure && (
              <div className="mt-2 border-t border-line pt-2">
                <Button variant="ghost" onClick={() => setScoring(!scoring)}>
                  {scoring ? t("ts.backtest.hide") : t("ts.backtest.show")}
                </Button>
                {scoring && (
                  <>
                    <p className="mt-1 text-xs text-ink-3">
                      {t("ts.backtest.explain", {
                        model: shown.info ?? modelName(shown.key),
                      })}
                    </p>
                    <Plot
                      figure={scoringFigure}
                      height={320}
                      label={t("ts.backtest.show")}
                    />
                  </>
                )}
              </div>
            )}
            {partsFigure && parts && (
              <div className="mt-2 border-t border-line pt-2">
                <Button
                  variant="ghost"
                  onClick={() => setComponents(!components)}
                >
                  {components
                    ? t("ts.components.hide")
                    : t("ts.components.show")}
                </Button>
                {components && (
                  <>
                    <p className="mt-1 text-xs text-ink-3">
                      {t("ts.components.explain", {
                        model: shown.info ?? modelName(shown.key),
                      })}
                    </p>
                    <Plot
                      figure={partsFigure}
                      height={Math.max(220, 110 * parts.rows.length)}
                      label={t("ts.components.show")}
                    />
                  </>
                )}
              </div>
            )}
          </div>
          {shown.residual_check && (
            <p
              className={`text-[13px] ${shown.residual_check.significant ? "text-warn" : "text-ink-2"}`}
            >
              {t(
                shown.residual_check.significant
                  ? "ts.forecast.residualsLinked"
                  : "ts.forecast.residualsWhite",
                { model: shown.info ?? modelName(shown.key) },
              )}
            </p>
          )}
          <div className="flex items-center justify-between">
            <h3 className="text-[13px] font-medium">
              {t("ts.forecast.scores")}
            </h3>
            <Checkbox
              label={t("ts.forecast.allMetrics")}
              checked={allMetrics}
              onChange={(event) => setAllMetrics(event.target.checked)}
              className="text-xs"
            />
          </div>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px] tabular-nums">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th className="px-3 py-1.5 font-normal">
                    {t("ts.forecast.model")}
                  </th>
                  <th
                    className="px-3 py-1.5 text-right font-normal"
                    title={t("ts.forecast.maseHint")}
                  >
                    MASE
                  </th>
                  <th className="px-3 py-1.5 text-right font-normal">
                    {t("ts.forecast.vs", {
                      reference: modelName(result.reference),
                    })}
                  </th>
                  <th className="px-3 py-1.5 text-right font-normal">MAE</th>
                  <th className="px-3 py-1.5 text-right font-normal">RMSE</th>
                  {allMetrics && (
                    <>
                      <th className="px-3 py-1.5 text-right font-normal">
                        MAPE
                      </th>
                      <th className="px-3 py-1.5 text-right font-normal">
                        sMAPE
                      </th>
                      <th
                        className="px-3 py-1.5 text-right font-normal"
                        title={t("ts.forecast.wapeHint")}
                      >
                        {t("ts.forecast.wape")}
                      </th>
                      <th className="px-3 py-1.5 text-right font-normal">
                        RMSSE
                      </th>
                      <th className="px-3 py-1.5 text-right font-normal">
                        {t("ts.forecast.bias")}
                      </th>
                      <th className="px-3 py-1.5 text-right font-normal">
                        {t("ts.forecast.coverage80")}
                      </th>
                      <th className="px-3 py-1.5 text-right font-normal">
                        {t("ts.forecast.winkler")}
                      </th>
                    </>
                  )}
                  <th className="px-3 py-1.5 text-right font-normal">
                    {t("ts.forecast.coverage95")}
                  </th>
                  <th className="px-3 py-1.5 text-right font-normal">
                    {t("ts.forecast.time")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {result.results.map((model) => {
                  const s = model.scores;
                  const active = model.key === shown.key;
                  return (
                    <tr
                      key={model.key}
                      onClick={() => s && setSelected(model.key)}
                      className={`border-t border-line ${s ? "cursor-pointer hover:bg-surface-1" : "text-ink-3"} ${active ? "bg-accent-soft" : ""}`}
                      aria-selected={active}
                    >
                      <td className="px-3 py-1.5">
                        <span className="text-ink-1">
                          {modelName(model.key)}
                        </span>
                        {model.key === result.best && (
                          <span className="ml-1.5 rounded-full bg-accent-soft px-1.5 text-[10px] text-ink-1">
                            {t("ts.forecast.best")}
                          </span>
                        )}
                        {model.info && (
                          <span className="block font-mono text-[11px] text-ink-3">
                            {model.info}
                          </span>
                        )}
                        {model.error && (
                          <span
                            className={`block text-xs ${model.timed_out ? "text-warn" : "text-ink-3"}`}
                          >
                            {model.timed_out && `${t("ts.fopt.timedOut")} · `}
                            {model.error}
                          </span>
                        )}
                        {model.uses_drivers &&
                          model.mase_without_drivers !== null && (
                            <span className="block text-xs text-ink-3">
                              {t("ts.drivers.without", {
                                mase: formatNumber(model.mase_without_drivers),
                              })}
                            </span>
                          )}
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        {formatNumber(s?.mase)}
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        {model.vs_reference === null
                          ? "—"
                          : `${model.vs_reference >= 0 ? "+" : "−"}${Math.abs(model.vs_reference).toFixed(0)} %`}
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        {formatNumber(s?.mae)}
                      </td>
                      <td className="px-3 py-1.5 text-right">
                        {formatNumber(s?.rmse)}
                      </td>
                      {allMetrics && (
                        <>
                          <td className="px-3 py-1.5 text-right">
                            {s?.mape === null || !s
                              ? "—"
                              : `${s.mape.toFixed(1)} %`}
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            {s?.smape === null || !s
                              ? "—"
                              : `${s.smape.toFixed(1)} %`}
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            {s?.wape === null || !s
                              ? "—"
                              : `${s.wape.toFixed(1)} %`}
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            {formatNumber(s?.rmsse)}
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            {formatNumber(s?.bias)}
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            {percent(s?.coverage80)}
                          </td>
                          <td className="px-3 py-1.5 text-right">
                            {formatNumber(s?.winkler95)}
                          </td>
                        </>
                      )}
                      <td className="px-3 py-1.5 text-right">
                        {percent(s?.coverage95)}
                      </td>
                      <td className="px-3 py-1.5 text-right text-ink-3">
                        {model.seconds.toFixed(1)} s
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-xs text-ink-3">{t("ts.forecast.reading")}</p>
          <DriverResults result={result} />
          {drivers.length > 0 && (
            <ScenariosPanel
              datasetId={datasetId}
              params={params}
              drivers={drivers}
              horizon={h}
              models={models}
              future={result.future}
              periods={result.periods}
              values={result.history}
              what={what}
              tuning={tuning}
              scenarios={scenarios}
              model={scenarioModel}
              onChange={onScenarios}
            />
          )}
        </div>
      )}
      {params.group && (
        <EveryGroup
          datasetId={datasetId}
          params={params}
          horizon={h}
          models={models}
          onPick={onPickSeries}
        />
      )}
    </div>
  );
}

export function forecastVerdict(job: ForecastJob | undefined): Verdict {
  const result = job?.status === "done" ? job.result : null;
  if (!result || !result.best) return null;
  const best = result.results.find((r) => r.key === result.best);
  if (!best) return null;
  const gain = best.vs_reference;
  const name = modelName(best.key);
  if (gain === null || gain <= 0)
    return {
      text: t("ts.forecast.noBetter", {
        reference: modelName(result.reference),
      }),
      tone: "none",
    };
  return {
    text: t("ts.forecast.verdict", {
      model: name,
      gain: gain.toFixed(0),
      reference: modelName(result.reference),
    }),
    tone: "ok",
  };
}
