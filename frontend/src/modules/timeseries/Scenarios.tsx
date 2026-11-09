import type { Data } from "plotly.js-dist-min";
import { Plus, X } from "lucide-react";
import { useMemo } from "react";

import type {
  DriverForecast,
  DriverSpec,
  ScenarioChange,
  SeriesParams,
} from "../../api/client";
import { useScenarios } from "../../api/hooks";
import { Plot, type Figure } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { DRIVER_MODELS } from "./Drivers";
import { LOCKED } from "./Methods";
import { MAX_SCENARIOS, type StoredScenario } from "./search";

const modelName = (key: string) => t(`tsm.${key}` as MessageKey);

/** Drivers whose future values a scenario can change: known in advance, not calendar parts. */
export function changeableDrivers(drivers: DriverSpec[]): DriverSpec[] {
  return drivers.filter(
    (d) => d.kind === "flag" || (d.kind === "column" && d.known),
  );
}

export function toChange(scenario: StoredScenario): ScenarioChange {
  return {
    column: scenario.column,
    how: scenario.how,
    amount: scenario.amount,
    start: scenario.start ?? null,
    end: scenario.end ?? null,
  };
}

const sum = (values: number[]) => values.reduce((a, b) => a + b, 0);

/** Base forecast and one line per scenario; colors follow the scenario's slot, never its rank. */
function scenarioFigure(
  history: { periods: string[]; values: (number | null)[] },
  base: DriverForecast,
  scenarios: { name: string; forecast: DriverForecast }[],
  what: string,
): Figure {
  return (tokens) => {
    const keep = Math.min(history.periods.length, 3 * base.future.length);
    const data: Data[] = [
      {
        type: "scatter",
        mode: "lines",
        name: what,
        x: history.periods.slice(-keep),
        y: history.values.slice(-keep),
        line: { color: tokens.ink2, width: 1.5 },
        hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
      } as Data,
      {
        type: "scatter",
        mode: "lines+markers",
        name: t("ts.scenario.base"),
        x: base.future,
        y: base.forecast,
        line: { color: tokens.palette[0] ?? tokens.ink1, width: 2 },
        marker: { size: 8 },
        hovertemplate: `${t("ts.scenario.base")}: %{y:,.4g}<extra></extra>`,
      } as Data,
      ...scenarios.map(
        ({ name, forecast }, i) =>
          ({
            type: "scatter",
            mode: "lines+markers",
            name,
            x: forecast.future,
            y: forecast.forecast,
            line: { color: tokens.palette[i + 1] ?? tokens.ink3, width: 2, dash: "dash" },
            marker: { size: 8 },
            hovertemplate: `${name}: %{y:,.4g}<extra></extra>`,
          }) as Data,
      ),
    ];
    return {
      data,
      layout: {
        showlegend: true,
        legend: { orientation: "h", x: 0, y: 1.14, font: { size: 11 } },
        hovermode: "x unified",
        xaxis: { type: "date" },
      },
    };
  };
}

/** "What if": change a known-in-advance driver (± x %, a fixed value, over a range of periods)
 *  and see the forecast next to the base one (039 story 6). Kept in the URL. */
export function ScenariosPanel({
  datasetId,
  params,
  drivers,
  horizon,
  models,
  future,
  periods,
  values,
  what,
  scenarios,
  model,
  onChange,
}: {
  datasetId: string;
  params: SeriesParams;
  drivers: DriverSpec[];
  horizon: number;
  models: string[];
  future: string[];
  periods: string[];
  values: (number | null)[];
  what: string;
  scenarios: StoredScenario[];
  model: string | undefined;
  onChange: (scenarios: StoredScenario[] | undefined, model?: string) => void;
}) {
  const known = changeableDrivers(drivers);
  const offered = [...new Set([...models, ...DRIVER_MODELS])];
  const usable = (key: string) => DRIVER_MODELS.includes(key) && !LOCKED.has(key);
  const chosen =
    model && usable(model)
      ? model
      : (models.find(usable) ?? "tslm"); // the regression: fast, and only changed periods move
  const extra = { horizon: String(horizon), model: chosen };
  const lists = useMemo(
    () => [[] as ScenarioChange[], ...scenarios.map((s) => [toChange(s)])],
    [scenarios],
  );
  const results = useScenarios(datasetId, params, extra, drivers, lists, known.length > 0);
  const base = results.data[0];
  const done = useMemo(
    () =>
      scenarios.flatMap((s, i) => {
        const forecast = results.data[i + 1];
        const name = s.name || t("ts.scenario.unnamed", { n: String(i + 1) });
        return forecast ? [{ name, forecast }] : [];
      }),
    [scenarios, results.data],
  );
  const figure = useMemo(
    () => (base ? scenarioFigure({ periods, values }, base, done, what) : null),
    [base, periods, values, done, what],
  );
  const failed = results.error;

  if (!known.length) {
    return (
      <div className="rounded-md border border-line bg-surface-1 px-3 py-2 text-[13px] text-ink-2">
        <h3 className="mb-0.5 font-medium text-ink-1">{t("ts.scenario.title")}</h3>
        {t("ts.scenario.needsKnown")}
      </div>
    );
  }

  const update = (i: number, patch: Partial<StoredScenario>) =>
    onChange(
      scenarios.map((s, j) => (j === i ? { ...s, ...patch } : s)),
      chosen,
    );
  const add = () =>
    onChange(
      [
        ...scenarios,
        {
          name: "",
          column: known[0]?.column ?? "",
          how: "percent",
          amount: -10,
        },
      ],
      chosen,
    );

  return (
    <div className="space-y-2 rounded-lg border border-line bg-surface-1 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[13px] font-medium text-ink-1">{t("ts.scenario.title")}</h3>
        <label className="flex items-center gap-1.5 text-xs text-ink-2">
          {t("ts.scenario.model")}
          <Select
            value={chosen}
            onChange={(event) => onChange(scenarios.length ? scenarios : undefined, event.target.value)}
            className="h-7 text-xs"
          >
            {offered.map((key) => (
              <option key={key} value={key} disabled={!usable(key)}>
                {modelName(key)}
                {!DRIVER_MODELS.includes(key)
                  ? ` (${t("ts.scenario.noDrivers")})`
                  : LOCKED.has(key)
                    ? ` (${t("lite.locked")})`
                    : ""}
              </option>
            ))}
          </Select>
        </label>
      </div>
      <p className="text-xs text-ink-3">{t("ts.scenario.explain")}</p>
      {scenarios.map((scenario, i) => (
        <div key={i} className="flex flex-wrap items-center gap-2 text-xs">
          <TextInput
            value={scenario.name}
            placeholder={t("ts.scenario.unnamed", { n: String(i + 1) })}
            aria-label={t("ts.scenario.name")}
            onChange={(event) => update(i, { name: event.target.value })}
            className="h-7 w-44 text-xs"
          />
          <Select
            value={scenario.column}
            aria-label={t("ts.scenario.driver")}
            onChange={(event) => update(i, { column: event.target.value })}
            className="h-7 font-mono text-xs"
          >
            {known.map((d) => (
              <option key={d.column} value={d.column}>
                {d.column}
              </option>
            ))}
          </Select>
          <Select
            value={scenario.how}
            aria-label={t("ts.scenario.how")}
            onChange={(event) => update(i, { how: event.target.value === "value" ? "value" : "percent" })}
            className="h-7 text-xs"
          >
            <option value="percent">{t("ts.scenario.percent")}</option>
            <option value="value">{t("ts.scenario.value")}</option>
          </Select>
          <TextInput
            type="number"
            value={String(scenario.amount)}
            aria-label={t("ts.scenario.amount")}
            onChange={(event) => update(i, { amount: Number(event.target.value) || 0 })}
            className="h-7 w-20 text-right text-xs tabular-nums"
          />
          {scenario.how === "percent" && <span className="text-ink-3">%</span>}
          <span className="text-ink-3">{t("ts.scenario.from")}</span>
          <Select
            value={scenario.start ?? ""}
            aria-label={t("ts.scenario.from")}
            onChange={(event) => update(i, { start: event.target.value || undefined })}
            className="h-7 text-xs"
          >
            <option value="">{t("ts.scenario.first")}</option>
            {future.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </Select>
          <span className="text-ink-3">{t("ts.scenario.to")}</span>
          <Select
            value={scenario.end ?? ""}
            aria-label={t("ts.scenario.to")}
            onChange={(event) => update(i, { end: event.target.value || undefined })}
            className="h-7 text-xs"
          >
            <option value="">{t("ts.scenario.last")}</option>
            {future.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </Select>
          <Button
            variant="ghost"
            icon={X}
            iconOnly
            aria-label={t("ts.scenario.remove")}
            title={t("ts.scenario.remove")}
            onClick={() => {
              const next = scenarios.filter((_, j) => j !== i);
              onChange(next.length ? next : undefined, chosen);
            }}
          />
        </div>
      ))}
      {scenarios.length < MAX_SCENARIOS && (
        <Button variant="ghost" icon={Plus} onClick={add}>
          {t("ts.scenario.add")}
        </Button>
      )}
      {failed && <ErrorNote>{failed.message}</ErrorNote>}
      {figure && base && (
        <>
          <Plot figure={figure} height={300} label={t("ts.scenario.title")} />
          {done.length > 0 && (
            <div className="overflow-x-auto rounded-md border border-line">
              <table className="w-full border-collapse text-[13px] tabular-nums">
                <thead className="bg-surface-2 text-left text-xs text-ink-3">
                  <tr>
                    <th className="px-3 py-1.5 font-normal">{t("ts.scenario.name")}</th>
                    <th className="px-3 py-1.5 text-right font-normal">{t("ts.scenario.total")}</th>
                    <th className="px-3 py-1.5 text-right font-normal">{t("ts.scenario.gap")}</th>
                    <th className="px-3 py-1.5 text-right font-normal">{t("ts.scenario.gapPct")}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-t border-line text-ink-2">
                    <td className="px-3 py-1.5">{t("ts.scenario.base")}</td>
                    <td className="px-3 py-1.5 text-right">{formatNumber(sum(base.forecast))}</td>
                    <td className="px-3 py-1.5 text-right">—</td>
                    <td className="px-3 py-1.5 text-right">—</td>
                  </tr>
                  {done.map(({ name, forecast }) => {
                    const gap = sum(forecast.forecast) - sum(base.forecast);
                    const whole = sum(base.forecast);
                    return (
                      <tr key={name} className="border-t border-line">
                        <td className="px-3 py-1.5 text-ink-1">{name}</td>
                        <td className="px-3 py-1.5 text-right">{formatNumber(sum(forecast.forecast))}</td>
                        <td className="px-3 py-1.5 text-right">
                          {gap >= 0 ? "+" : "−"}
                          {formatNumber(Math.abs(gap))}
                        </td>
                        <td className="px-3 py-1.5 text-right">
                          {whole
                            ? `${gap >= 0 ? "+" : "−"}${Math.abs((gap / whole) * 100).toFixed(1)} %`
                            : "—"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          {base.notes.length > 0 && (
            <ul className="space-y-0.5 text-xs text-warn">
              {base.notes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          )}
        </>
      )}
    </div>
  );
}
