import type { Data } from "plotly.js-dist-min";
import { ArrowDown, ArrowUp, Plus, X } from "lucide-react";
import { useMemo, useState } from "react";

import {
  api,
  type GroupForecast,
  type GroupRow,
  type GroupsSetup,
  type SeriesParams,
} from "../../api/client";
import {
  useCancelGroupJob,
  useGroupJob,
  useSeriesSection,
  useStartGroupForecast,
} from "../../api/hooks";
import { Plot, type Figure } from "../../charts/Plot";
import { formatNumber, withAlpha } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { Checkbox, ErrorNote, ProgressBar, Select, TextInput } from "../../ui/fields";
import { LOCKED } from "./Methods";

type Reconcile = "none" | "auto" | "bottom_up" | "top_down" | "mint_shrink";
type SortKey = "volume" | "label" | "mase" | "wape" | "total" | "change";

const SLOW = ["arima_auto", "sarima", "sarimax"];
const SMALL_MULTIPLES = 12;
const PAGE = 50;
const RECONCILE: Reconcile[] = ["none", "bottom_up", "top_down", "mint_shrink", "auto"];

const modelName = (key: string | null) =>
  key ? t(`tsm.${key}` as MessageKey) : "—";
const pct = (value: number | null) => {
  if (value === null) return "—";
  const shown = Math.abs(value).toFixed(1);
  return shown === "0.0" ? "0.0 %" : `${value > 0 ? "+" : "−"}${shown} %`;
};

/** The methods sent: the ticked ones without the slow ARIMA family unless asked for. */
export function groupMethods(models: string[], slow: boolean): string[] {
  const kept = models.filter((key) => !SLOW.includes(key) && !LOCKED.has(key));
  if (slow && !LOCKED.has("arima_auto")) kept.push("arima_auto");
  return kept;
}

/** "ETS is best for 41 % of stores; the global model wins for 7 of 30." */
export function groupVerdict(result: GroupForecast): string {
  const top = result.best_models[0];
  if (!top) return t("ts.groups.noneScored");
  let text = t("ts.groups.verdict", {
    model: modelName(top.model),
    pct: top.share.toFixed(0),
    group: result.group,
  });
  if (result.global_model)
    text += ` ${t("ts.groups.globalWins", {
      wins: String(result.global_model.wins),
      series: String(result.global_model.series),
    })}`;
  return text;
}

/** One small chart: the history in grey, the forecast in the first palette color, its 95 % band. */
function smallFigure(result: GroupForecast, row: GroupRow): Figure {
  return (tokens) => {
    const color = tokens.palette[0] ?? tokens.ink1;
    return {
      data: [
        {
          type: "scatter",
          mode: "lines",
          x: result.periods,
          y: row.history,
          line: { color: tokens.ink3, width: 1.25 },
          hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: result.future,
          y: row.low95,
          line: { width: 0 },
          hoverinfo: "skip",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: result.future,
          y: row.high95,
          fill: "tonexty",
          fillcolor: withAlpha(color, 0.16),
          line: { width: 0 },
          hoverinfo: "skip",
        } as Data,
        {
          type: "scatter",
          mode: "lines",
          x: result.future,
          y: row.forecast,
          line: { color, width: 2 },
          hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
        } as Data,
      ],
      layout: {
        hovermode: "x unified",
        xaxis: { type: "date", showticklabels: true, tickfont: { size: 10 } },
        yaxis: { tickfont: { size: 10 } },
      },
    };
  };
}

/** Which model is best for how many series: horizontal bars, one color (a single measure). */
function bestFigure(result: GroupForecast): Figure {
  return (tokens) => {
    const shares = [...result.best_models].reverse();
    return {
      data: [
        {
          type: "bar",
          orientation: "h",
          x: shares.map((s) => s.series),
          y: shares.map((s) => modelName(s.model)),
          text: shares.map((s) => `${s.share.toFixed(0)} %`),
          textposition: "outside",
          textfont: { color: tokens.ink2, size: 11 },
          marker: { color: tokens.palette[0] ?? tokens.ink1 },
          hovertemplate: "%{y}: %{x} series<extra></extra>",
          cliponaxis: false,
        } as Data,
      ],
      layout: { xaxis: { showgrid: true }, yaxis: { showgrid: false }, margin: { l: 8, r: 40, t: 4, b: 8 } },
    };
  };
}

function download(text: string, name: string) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

/** "store → region → Total": the levels recognised, each removable; others can be added. */
function LevelsTree({
  setup,
  levels,
  onChange,
}: {
  setup: GroupsSetup;
  levels: string[];
  onChange: (levels: string[]) => void;
}) {
  const parents = setup.candidates.filter((c) => c.parent);
  const missing = parents.filter((c) => !levels.includes(c.column));
  const values = (column: string) =>
    parents.find((c) => c.column === column)?.values ?? 0;
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-xs" aria-label={t("ts.groups.levels")}>
      <span className="text-ink-3">{t("ts.groups.levels")}</span>
      <span className="font-mono text-ink-1">{setup.group}</span>
      {levels.map((level) => (
        <span key={level} className="inline-flex items-center gap-1.5">
          <span className="text-ink-3" aria-hidden>
            →
          </span>
          <span className="inline-flex h-6 items-center gap-1 rounded-full border border-line bg-surface-1 pr-1 pl-2.5 font-mono text-ink-1">
            {level}
            <button
              type="button"
              className="rounded-full p-0.5 text-ink-3 hover:bg-surface-2 hover:text-ink-1"
              aria-label={t("ts.groups.removeLevel", { level })}
              onClick={() => onChange(levels.filter((l) => l !== level))}
            >
              <X size={12} strokeWidth={1.5} aria-hidden />
            </button>
          </span>
        </span>
      ))}
      <span className="text-ink-3" aria-hidden>
        →
      </span>
      <span className="text-ink-1">{t("ts.groups.total")}</span>
      {missing.length > 0 && (
        <span className="ml-1 inline-flex items-center gap-1 text-ink-3">
          <Plus size={12} strokeWidth={1.5} aria-hidden />
          <Select
            value=""
            aria-label={t("ts.groups.addLevel")}
            onChange={(event) => {
              const next = [...levels, event.target.value].sort(
                (a, b) => values(b) - values(a),
              );
              onChange(next);
            }}
            className="h-6 text-xs"
          >
            <option value="">{t("ts.groups.addLevel")}</option>
            {missing.map((c) => (
              <option key={c.column} value={c.column}>
                {c.column} ({c.values})
              </option>
            ))}
          </Select>
        </span>
      )}
    </div>
  );
}

function SeriesTable({
  result,
  onPick,
}: {
  result: GroupForecast;
  onPick: (label: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; down: boolean }>({ key: "volume", down: true });
  const [all, setAll] = useState(false);
  const hierarchy = result.hierarchy !== null;
  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const found = result.rows.filter(
      (r) => !needle || r.label.toLowerCase().includes(needle),
    );
    const value = (r: GroupRow): number | string =>
      sort.key === "label" ? r.label.toLowerCase() : (r[sort.key] ?? -Infinity);
    return [...found].sort((a, b) => {
      const x = value(a);
      const y = value(b);
      const order = x < y ? -1 : x > y ? 1 : 0;
      return sort.down ? -order : order;
    });
  }, [result.rows, query, sort]);
  const shown = all ? rows : rows.slice(0, PAGE);
  const header = (key: SortKey, label: string, right = true) => (
    <th className={`px-3 py-1.5 font-normal ${right ? "text-right" : ""}`}>
      <button
        type="button"
        className="inline-flex items-center gap-1 hover:text-ink-1"
        onClick={() =>
          setSort(sort.key === key ? { key, down: !sort.down } : { key, down: key !== "label" })
        }
      >
        {label}
        {sort.key === key &&
          (sort.down ? (
            <ArrowDown size={12} strokeWidth={1.5} aria-hidden />
          ) : (
            <ArrowUp size={12} strokeWidth={1.5} aria-hidden />
          ))}
      </button>
    </th>
  );
  const pickable = (r: GroupRow) => !r.other && r.level === result.group;
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[13px] font-medium">{t("ts.groups.table")}</h3>
        <span className="flex items-center gap-2">
          <TextInput
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={t("ts.groups.search")}
            aria-label={t("ts.groups.search")}
            className="h-7 w-48 text-xs"
          />
        </span>
      </div>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              {header("label", result.group, false)}
              {hierarchy && <th className="px-3 py-1.5 font-normal">{t("ts.groups.level")}</th>}
              {header("volume", t("ts.groups.volume"))}
              <th className="px-3 py-1.5 font-normal">{t("ts.groups.demand")}</th>
              <th className="px-3 py-1.5 font-normal">{t("ts.groups.best")}</th>
              {header("mase", "MASE")}
              {header("wape", t("ts.forecast.wape"))}
              {header("total", t("ts.groups.forecastTotal", { horizon: String(result.horizon) }))}
              {header("change", t("ts.groups.vsLastYear"))}
            </tr>
          </thead>
          <tbody>
            {shown.map((row) => (
              <tr
                key={`${row.level}:${row.label}`}
                onClick={() => pickable(row) && onPick(row.label)}
                className={`border-t border-line ${pickable(row) ? "cursor-pointer hover:bg-surface-1" : ""}`}
              >
                <td className="px-3 py-1.5">
                  <span className={row.other ? "text-ink-2" : "text-ink-1"}>{row.label}</span>
                  {row.short_history && (
                    <span className="ml-1.5 rounded-full bg-surface-2 px-1.5 text-[10px] text-ink-2">
                      {t("ts.groups.short")}
                    </span>
                  )}
                </td>
                {hierarchy && <td className="px-3 py-1.5 font-mono text-xs text-ink-2">{row.level}</td>}
                <td className="px-3 py-1.5 text-right">{formatNumber(row.volume)}</td>
                <td className="px-3 py-1.5 text-ink-2">
                  {row.demand ? t(`ts.demand.kind.${row.demand}` as MessageKey) : "—"}
                </td>
                <td className="px-3 py-1.5">
                  <span className="text-ink-1">{modelName(row.best)}</span>
                  {row.error && <span className="block text-xs text-ink-3">{row.error}</span>}
                </td>
                <td className="px-3 py-1.5 text-right">{formatNumber(row.mase)}</td>
                <td className="px-3 py-1.5 text-right">
                  {row.wape === null ? "—" : `${row.wape.toFixed(1)} %`}
                </td>
                <td className="px-3 py-1.5 text-right">{formatNumber(row.total)}</td>
                <td className="px-3 py-1.5 text-right">{pct(row.change)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > PAGE && (
        <Button variant="ghost" onClick={() => setAll(!all)}>
          {all ? t("ts.groups.showFewer") : t("ts.groups.showAll", { n: String(rows.length) })}
        </Button>
      )}
    </div>
  );
}

function Results({
  result,
  jobId,
  onPick,
}: {
  result: GroupForecast;
  jobId: string;
  onPick: (label: string) => void;
}) {
  const [exportError, setExportError] = useState<string | null>(null);
  const best = useMemo(() => bestFigure(result), [result]);
  const smalls = useMemo(
    () =>
      result.rows
        .filter((r) => r.level === result.group)
        .sort((a, b) => b.volume - a.volume)
        .slice(0, SMALL_MULTIPLES)
        .map((row) => ({ row, figure: smallFigure(result, row) })),
    [result],
  );
  const hierarchy = result.hierarchy;
  return (
    <div className="space-y-4">
      <p className="font-medium text-ink-1">{groupVerdict(result)}</p>
      {result.notes.length > 0 && (
        <ul className="space-y-0.5 text-[13px] text-ink-2">
          {result.notes.map((note) => (
            <li key={note}>{note}</li>
          ))}
        </ul>
      )}
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]">
        <div>
          <h3 className="mb-1 text-[13px] font-medium">{t("ts.groups.bestModels")}</h3>
          <Plot
            figure={best}
            height={Math.max(120, 28 * result.best_models.length + 40)}
            label={t("ts.groups.bestModels")}
          />
        </div>
        {hierarchy && (
          <div>
            <h3 className="mb-1 text-[13px] font-medium">
              {t("ts.groups.reconciled", { method: t(`ts.groups.method.${hierarchy.method}` as MessageKey) })}
            </h3>
            <div className="overflow-x-auto rounded-md border border-line">
              <table className="w-full border-collapse text-[13px] tabular-nums">
                <thead className="bg-surface-2 text-left text-xs text-ink-3">
                  <tr>
                    <th className="px-3 py-1.5 font-normal">{t("ts.groups.level")}</th>
                    <th className="px-3 py-1.5 text-right font-normal">{t("ts.groups.nodes")}</th>
                    <th className="px-3 py-1.5 text-right font-normal">{t("ts.groups.before")}</th>
                    <th className="px-3 py-1.5 text-right font-normal">{t("ts.groups.after")}</th>
                  </tr>
                </thead>
                <tbody>
                  {hierarchy.errors.map((e) => (
                    <tr key={e.level} className="border-t border-line">
                      <td className="px-3 py-1.5 font-mono text-xs text-ink-1">{e.level}</td>
                      <td className="px-3 py-1.5 text-right">{e.nodes}</td>
                      <td className="px-3 py-1.5 text-right">{formatNumber(e.mase_before)}</td>
                      <td className="px-3 py-1.5 text-right">{formatNumber(e.mase_after)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="mt-1 text-xs text-ink-3">
              {t("ts.groups.coherent", { gap: hierarchy.gap.toExponential(1) })}
            </p>
          </div>
        )}
      </div>
      <div>
        <h3 className="mb-1 text-[13px] font-medium">
          {t("ts.groups.largest", { n: String(smalls.length), group: result.group })}
        </h3>
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
          {smalls.map(({ row, figure }) => (
            <div key={row.label} className="rounded-md border border-line bg-surface-1 p-2">
              <div className="flex items-baseline justify-between gap-2 text-xs">
                {row.other ? (
                  <span className="truncate text-ink-2">{row.label}</span>
                ) : (
                  <button
                    type="button"
                    className="truncate text-left font-medium text-ink-1 hover:text-accent"
                    onClick={() => onPick(row.label)}
                    title={t("ts.groups.open", { label: row.label })}
                  >
                    {row.label}
                  </button>
                )}
                <span className="shrink-0 text-ink-3">{modelName(row.best)}</span>
              </div>
              <Plot figure={figure} height={140} label={row.label} />
            </div>
          ))}
        </div>
      </div>
      <SeriesTable result={result} onPick={onPick} />
      <div className="flex items-center gap-3">
        <Button
          variant="secondary"
          onClick={() => {
            setExportError(null);
            api
              .groupCsv(jobId)
              .then((text) => download(text, "forecast-every-group.csv"))
              .catch((error: Error) => setExportError(error.message));
          }}
        >
          {t("ts.forecast.export")}
        </Button>
        <span className="text-xs text-ink-3">{t("ts.groups.exportHint")}</span>
      </div>
      {exportError && <ErrorNote>{exportError}</ErrorNote>}
    </div>
  );
}

/** "Forecast every group" (039): every series of the group column, coherent totals. */
export function EveryGroup({
  datasetId,
  params,
  horizon,
  models,
  onPick,
}: {
  datasetId: string;
  params: SeriesParams;
  horizon: number;
  models: string[];
  onPick: (label: string) => void;
}) {
  const setupQuery = useSeriesSection<GroupsSetup>(datasetId, "groups", params, {});
  const setup = setupQuery.data;
  const [slow, setSlow] = useState(false);
  const [global, setGlobal] = useState(false);
  const [reconcile, setReconcile] = useState<Reconcile | null>(null);
  const [levels, setLevels] = useState<string[] | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const start = useStartGroupForecast(datasetId);
  const cancel = useCancelGroupJob();
  const job = useGroupJob(jobId);

  if (!setup) {
    return setupQuery.error ? <ErrorNote>{setupQuery.error.message}</ErrorNote> : null;
  }
  const chosenReconcile: Reconcile = !setup.summable
    ? "none"
    : (reconcile ?? "auto");
  const chosenLevels = levels ?? setup.levels;
  const run = () => {
    const methods = groupMethods(models, slow);
    start.mutate(
      {
        params,
        extra: {
          horizon: String(horizon),
          ...(methods.length ? { methods: methods.join(",") } : {}),
          ...(global ? { global: "true" } : {}),
          reconcile: chosenReconcile,
          ...(chosenReconcile !== "none" ? { levels: chosenLevels.join(",") } : {}),
        },
      },
      { onSuccess: (started) => setJobId(started.id) },
    );
  };
  const state = job.data;
  const running = state?.status === "running" || start.isPending;
  const shown = Math.min(setup.groups, setup.limit);

  return (
    <div className="space-y-3 border-t border-line pt-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-[14px] font-medium text-ink-1">{t("ts.groups.title")}</h3>
        <span className="text-xs text-ink-3">
          {setup.groups > setup.limit
            ? t("ts.groups.countLimited", {
                n: String(setup.groups),
                group: setup.group,
                limit: String(setup.limit),
              })
            : t("ts.groups.count", { n: String(setup.groups), group: setup.group })}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <Button variant="primary" onClick={run} disabled={running}>
          {running ? t("ts.forecast.running") : t("ts.groups.run")}
        </Button>
        {!LOCKED.has("arima_auto") && (
          <Checkbox
            label={t("ts.groups.arima")}
            checked={slow}
            onChange={(event) => setSlow(event.target.checked)}
          />
        )}
        <Checkbox
          label={t("ts.groups.global")}
          checked={global}
          onChange={(event) => setGlobal(event.target.checked)}
        />
      </div>
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <span className="text-ink-3">{t("ts.groups.totals")}</span>
        {setup.summable ? (
          <SegmentedControl
            label={t("ts.groups.totals")}
            value={chosenReconcile}
            options={RECONCILE.map((value) => ({
              value,
              label: t(`ts.groups.reconcile.${value}` as MessageKey),
            }))}
            onChange={setReconcile}
          />
        ) : (
          <span className="text-ink-2">{t("ts.groups.notSummable")}</span>
        )}
      </div>
      {chosenReconcile !== "none" && (
        <LevelsTree setup={setup} levels={chosenLevels} onChange={setLevels} />
      )}
      {setup.browser && (
        <p className="text-xs text-ink-3">
          {t("ts.groups.browser", { n: String(setup.limit) })}
        </p>
      )}
      {start.isError && <ErrorNote>{start.error.message}</ErrorNote>}
      {state?.status === "running" && (
        <div className="flex max-w-md items-center gap-3">
          <div className="flex-1">
            <ProgressBar
              value={state.total ? state.done / state.total : 0}
              label={t("ts.groups.progress")}
            />
            <p className="mt-1 text-xs text-ink-3" aria-live="polite">
              {state.stage === "series"
                ? t("ts.groups.series", { done: String(state.done), total: String(state.total) })
                : t(`ts.groups.stage.${state.stage}` as MessageKey)}
            </p>
          </div>
          <Button variant="ghost" onClick={() => jobId && cancel.mutate(jobId)}>
            {t("ts.forecast.cancel")}
          </Button>
        </div>
      )}
      {state?.status === "cancelled" && <p className="text-ink-2">{t("ts.forecast.cancelled")}</p>}
      {state?.status === "failed" && <ErrorNote>{state.error}</ErrorNote>}
      {!state && !running && (
        <p className="text-xs text-ink-3">
          {t("ts.groups.idle", { n: String(shown), group: setup.group })}
        </p>
      )}
      {state?.status === "done" && state.result && jobId && (
        <Results result={state.result} jobId={jobId} onPick={onPick} />
      )}
    </div>
  );
}
