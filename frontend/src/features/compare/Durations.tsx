import type { Data, Layout } from "plotly.js-dist-min";
import { useMemo, useState } from "react";

import type { DurationParams, MarkovChain, Survival, Transitions } from "../../api/client";
import { useSurvival, useTransitions } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { formatNumber, withAlpha } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";

/** 062: durations in Compare. Survival shows on the first plane only when a rule detects a
 *  duration and an event, or a start and an end (R-SRV-01/02); status transitions sit folded under
 *  "Advanced" when R-MKV-01 finds a repeating key and a status, computed when opened. Nothing
 *  shows when no rule fires (owner's onion rule). */
export function DurationsSection({ datasetId, ready }: { datasetId: string; ready: boolean }) {
  const [params, setParams] = useState<DurationParams>({});
  const survival = useSurvival(datasetId, params, ready);
  const detected = useTransitions(datasetId, { detect_only: "true" }, ready);
  const showSurvival = Boolean(survival.data?.setup) || Object.values(params).some(Boolean);
  const showTransitions = Boolean(detected.data?.setup);
  if (!showSurvival && !showTransitions) return null;
  return (
    <section aria-labelledby="durations-title" className="space-y-4 border-t border-line pt-5">
      <header className="max-w-prose">
        <h2 id="durations-title" className="text-base font-semibold tracking-tight">
          {t("compare.durations.title")}
        </h2>
        <p className="text-[13px] text-ink-2">{t("compare.durations.question")}</p>
      </header>
      {showSurvival &&
        (survival.isError ? (
          <ErrorNote>{survival.error.message}</ErrorNote>
        ) : !survival.data ? (
          <Loading label={t("compare.survival.loading")} className="h-80 rounded-lg" />
        ) : (
          <SurvivalBody data={survival.data} params={params} onParams={setParams} />
        ))}
      {showTransitions && detected.data && <TransitionsPanel datasetId={datasetId} detected={detected.data} />}
    </section>
  );
}

function unitLabel(unit: string | null | undefined): string {
  return unit ? ` (${unit})` : "";
}

export function survivalFigure(data: Survival) {
  return (tokens: ChartTokens): { data: Data[]; layout: Partial<Layout> } => {
    const traces: Data[] = [];
    data.curves.forEach((curve, i) => {
      const color = data.curves.length === 1 ? tokens.numeric : (tokens.palette[i % 8] ?? tokens.ink3);
      traces.push({
        type: "scatter",
        mode: "lines",
        x: [...curve.times, ...[...curve.times].reverse()],
        y: [...curve.upper, ...[...curve.lower].reverse()],
        fill: "toself",
        fillcolor: withAlpha(color, 0.15),
        line: { width: 0, shape: "hv", color },
        hoverinfo: "skip",
        showlegend: false,
        name: `${curve.label} 95%`,
      } as Data);
      traces.push({
        type: "scatter",
        mode: "lines",
        name: curve.label,
        x: curve.times,
        y: curve.survival,
        customdata: curve.at_risk,
        line: { color, width: 2, shape: "hv" },
        hovertemplate: `${curve.label}<br>%{x:,.4g}: %{y:.1%} (%{customdata} at risk)<extra></extra>`,
      } as Data);
    });
    return {
      data: traces,
      layout: {
        showlegend: data.curves.length > 1,
        legend: { orientation: "h" as const, x: 0, y: 1.1, font: { size: 11 } },
        xaxis: { title: { text: `${t("compare.survival.time")}${unitLabel(data.setup?.unit)}`, font: { size: 11 } } },
        yaxis: { title: { text: t("compare.survival.share"), font: { size: 11 } }, tickformat: ".0%", range: [0, 1.02] },
      },
    };
  };
}

function SurvivalBody({
  data,
  params,
  onParams,
}: {
  data: Survival;
  params: DurationParams;
  onParams: (params: DurationParams) => void;
}) {
  const figure = useMemo(() => survivalFigure(data), [data]);
  const setup = data.setup;
  const set = (patch: DurationParams) => onParams({ ...params, ...patch });
  const kind = params.start || params.end ? "dates" : params.duration || params.event ? "duration" : setup?.kind;
  const unit = unitLabel(setup?.unit);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <h3 className="text-sm font-semibold">{t("compare.survival.title")}</h3>
        {setup && (
          <InfoTip
            label={t("compare.survival.why")}
            text={[setup.reason, setup.event_reason].filter(Boolean).join(" ")}
          />
        )}
      </div>
      {setup?.detected && <p className="text-xs text-ink-3">{setup.reason}</p>}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label={t("compare.survival.group")}>
          <Select value={params.group ?? ""} onChange={(e) => set({ group: e.target.value || undefined })}>
            <option value="">{t("compare.none")}</option>
            {data.choices.groups.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Field>
        {kind === "dates" ? (
          <>
            <ColumnSelect
              label={t("compare.survival.start")}
              value={params.start ?? setup?.start ?? ""}
              options={data.choices.dates}
              onChange={(start) => set({ start, end: params.end ?? setup?.end ?? undefined, duration: undefined, event: undefined })}
            />
            <ColumnSelect
              label={t("compare.survival.end")}
              value={params.end ?? setup?.end ?? ""}
              options={data.choices.dates}
              onChange={(end) => set({ end, start: params.start ?? setup?.start ?? undefined, duration: undefined, event: undefined })}
            />
          </>
        ) : (
          <>
            <ColumnSelect
              label={t("compare.survival.duration")}
              value={params.duration ?? setup?.duration ?? ""}
              options={data.choices.durations}
              onChange={(duration) => set({ duration, event: params.event ?? setup?.event ?? undefined })}
            />
            <ColumnSelect
              label={t("compare.survival.event")}
              value={params.event ?? setup?.event ?? ""}
              options={data.choices.events}
              onChange={(event) => set({ event, duration: params.duration ?? setup?.duration ?? undefined, event_value: undefined })}
            />
          </>
        )}
      </div>
      {data.message ? (
        <p className="rounded-lg border border-line bg-surface-1 px-4 py-6 text-center text-ink-2">{data.message}</p>
      ) : (
        <>
          {data.sentence && <p className="font-medium text-ink-1 text-pretty">{data.sentence}</p>}
          <Plot figure={figure} height={360} label={t("compare.survival.chartLabel")} />
          <p className="text-xs text-ink-3">
            {t("compare.survival.counts", {
              rows: formatCount(data.rows),
              events: formatCount(data.events),
              censored: formatCount(data.censored),
            })}
            {data.dropped > 0 && ` ${t("compare.survival.dropped", { count: formatCount(data.dropped) })}`}
            {data.others > 0 && ` ${t("compare.survival.others", { count: String(data.others) })}`}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full min-w-0 text-sm tabular-nums">
              <thead className="text-left text-xs text-ink-2">
                <tr>
                  <th className="py-1 pr-3 font-normal">{t("compare.survival.curve")}</th>
                  <th className="py-1 pr-3 text-right font-normal">{t("compare.survival.rows")}</th>
                  <th className="py-1 pr-3 text-right font-normal">{t("compare.survival.events")}</th>
                  <th className="py-1 pr-3 text-right font-normal">{t("compare.survival.median")}{unit}</th>
                  <th className="py-1 text-right font-normal">{t("compare.survival.interval")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {data.curves.map((curve) => (
                  <tr key={curve.label}>
                    <td className="py-1 pr-3">{curve.label}</td>
                    <td className="py-1 pr-3 text-right">{formatCount(curve.n)}</td>
                    <td className="py-1 pr-3 text-right">{formatCount(curve.events)}</td>
                    <td className="py-1 pr-3 text-right">{amount(curve.median)}</td>
                    <td className="py-1 text-right">
                      {curve.median_lower === null ? "—" : `${amount(curve.median_lower)} – ${amount(curve.median_upper)}`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.log_rank && (
            <p className="text-xs text-ink-2">
              {t("compare.survival.logRank", {
                statistic: formatNumber(data.log_rank.statistic),
                df: String(data.log_rank.df),
                p: data.log_rank.p_value < 0.0001 ? "< 0.0001" : `= ${data.log_rank.p_value.toPrecision(2)}`,
              })}
            </p>
          )}
        </>
      )}
    </div>
  );
}

const amount = (value: number | null) => (value === null ? t("compare.survival.notReached") : formatNumber(value));

function ColumnSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (name: string) => void;
}) {
  return (
    <Field label={label}>
      <Select value={value} onChange={(e) => onChange(e.target.value)}>
        {!options.includes(value) && <option value={value}>{value || t("compare.choose")}</option>}
        {options.map((name) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </Select>
    </Field>
  );
}

/** "Advanced": the Markov chain of a status, computed only once opened. */
function TransitionsPanel({ datasetId, detected }: { datasetId: string; detected: Transitions }) {
  const [open, setOpen] = useState(false);
  const [params, setParams] = useState<DurationParams>({});
  const setup = detected.setup;
  const chosen = {
    entity: params.entity ?? setup?.entity ?? undefined,
    time: params.time ?? setup?.time ?? undefined,
    status: params.status ?? setup?.status ?? undefined,
  };
  const result = useTransitions(datasetId, chosen, open);
  return (
    <details className="group rounded-md border border-line" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="flex min-h-11 cursor-pointer items-center gap-2 px-3 text-sm">
        <span className="text-ink-2">{t("compare.advanced")}</span>
        <span className="font-medium">{t("compare.transitions.title")}</span>
      </summary>
      {open && (
        <div className="space-y-3 border-t border-line px-3 py-3">
          {setup && <p className="text-xs text-ink-3">{setup.reason}</p>}
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <ColumnSelect
              label={t("compare.transitions.entity")}
              value={chosen.entity ?? ""}
              options={detected.choices.entities}
              onChange={(entity) => setParams({ ...chosen, entity })}
            />
            <ColumnSelect
              label={t("compare.transitions.time")}
              value={chosen.time ?? ""}
              options={detected.choices.times}
              onChange={(time) => setParams({ ...chosen, time })}
            />
            <ColumnSelect
              label={t("compare.transitions.status")}
              value={chosen.status ?? ""}
              options={detected.choices.statuses}
              onChange={(status) => setParams({ ...chosen, status })}
            />
          </div>
          {result.isError ? (
            <ErrorNote>{result.error.message}</ErrorNote>
          ) : !result.data ? (
            <Loading label={t("compare.transitions.loading")} className="h-40 rounded-lg" />
          ) : result.data.message || !result.data.chain ? (
            <p className="text-ink-2">{result.data.message}</p>
          ) : (
            <ChainBody data={result.data} chain={result.data.chain} />
          )}
        </div>
      )}
    </details>
  );
}

const percent = (value: number) => `${(value * 100).toFixed(value > 0 && value < 0.1 ? 1 : 0)}%`;

function ChainBody({ data, chain }: { data: Transitions; chain: MarkovChain }) {
  return (
    <div className="space-y-3">
      {data.sentence && <p className="font-medium text-ink-1 text-pretty">{data.sentence}</p>}
      <p className="text-xs text-ink-3">
        {t("compare.transitions.counts", {
          transitions: formatCount(chain.transitions),
          entities: formatCount(data.entities),
        })}
        {data.step_days !== null && ` ${t("compare.transitions.step", { days: formatNumber(data.step_days) })}`}
      </p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-0 text-sm tabular-nums" aria-label={t("compare.transitions.matrix")}>
          <caption className="pb-1 text-left text-xs text-ink-2">{t("compare.transitions.matrix")}</caption>
          <thead className="text-xs text-ink-2">
            <tr>
              <th className="py-1 pr-3 text-left font-normal">{t("compare.transitions.from")}</th>
              {chain.states.map((state) => (
                <th key={state} className="py-1 pr-3 text-right font-normal">
                  {state}
                </th>
              ))}
              <th className="py-1 text-right font-normal">{t("compare.transitions.stay")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {chain.states.map((state, i) => (
              <tr key={state}>
                <th scope="row" className="py-1 pr-3 text-left font-normal">
                  {state}
                  {chain.absorbing.includes(state) && (
                    <span className="ml-1 text-xs text-ink-3">{t("compare.transitions.absorbing")}</span>
                  )}
                </th>
                {(chain.probabilities[i] ?? []).map((p, j) => (
                  <td key={j} className={`py-1 pr-3 text-right ${i === j ? "font-medium text-ink-1" : "text-ink-2"}`}>
                    {percent(p)}
                  </td>
                ))}
                <td className="py-1 text-right text-ink-2">
                  {chain.stay_steps[i] === null || chain.stay_steps[i] === undefined
                    ? "—"
                    : formatNumber(chain.stay_steps[i] as number)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {chain.stationary && (
        <p className="text-sm">
          {t("compare.transitions.stationary")}{" "}
          {chain.states.map((state, i) => `${state} ${percent(chain.stationary?.[i] ?? 0)}`).join(" · ")}
        </p>
      )}
      {Object.keys(chain.steps_to_absorption).length > 0 && (
        <p className="text-sm">
          {t("compare.transitions.toAbsorbing", { states: chain.absorbing.join(", ") })}{" "}
          {Object.entries(chain.steps_to_absorption)
            .map(([state, steps]) => `${state} ${formatNumber(steps)}`)
            .join(" · ")}
        </p>
      )}
      {chain.stationary_note && <p className="text-xs text-ink-3">{chain.stationary_note}</p>}
    </div>
  );
}
