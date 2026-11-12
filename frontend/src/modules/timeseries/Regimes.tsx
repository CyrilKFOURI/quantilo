import type { Data, Layout } from "plotly.js-dist-min";
import { useMemo, useState } from "react";

import type { SeriesParams, SeriesRegimes } from "../../api/client";
import { useSeriesSection } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import type { ChartTokens } from "../../charts/theme";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

type Choice = { k: string; on: string };

/** 062: regimes of the series (Markov switching, statsmodels MarkovRegression). Slow for a
 *  modest gain: folded under "Advanced" and computed only when asked (owner's onion rule). */
export function RegimesPanel({ datasetId, params }: { datasetId: string; params: SeriesParams }) {
  const [asked, setAsked] = useState<Choice | null>(null);
  const [choice, setChoice] = useState<Choice>({ k: "", on: "auto" });
  const extra: Record<string, string> = asked ? { on: asked.on, ...(asked.k ? { k: asked.k } : {}) } : {};
  const regimes = useSeriesSection<SeriesRegimes>(datasetId, "regimes", params, extra, asked !== null);
  return (
    <details className="group rounded-md border border-line">
      <summary className="flex min-h-11 cursor-pointer items-center gap-2 px-3 text-sm">
        <span className="text-ink-2">{t("ts.advanced")}</span>
        <span className="font-medium">{t("ts.regimes.title")}</span>
      </summary>
      <div className="space-y-3 border-t border-line px-3 py-3">
        <p className="text-xs text-ink-2 text-pretty">{t("ts.regimes.intro")}</p>
        <div className="flex flex-wrap items-end gap-3">
          <Field label={t("ts.regimes.k")}>
            <Select value={choice.k} onChange={(e) => setChoice({ ...choice, k: e.target.value })} className="w-36">
              <option value="">{t("ts.regimes.kAuto")}</option>
              <option value="2">2</option>
              <option value="3">3</option>
            </Select>
          </Field>
          <Field label={t("ts.regimes.on")}>
            <Select value={choice.on} onChange={(e) => setChoice({ ...choice, on: e.target.value })} className="w-44">
              <option value="auto">{t("ts.regimes.onAuto")}</option>
              <option value="level">{t("ts.regimes.onLevel")}</option>
              <option value="changes">{t("ts.regimes.onChanges")}</option>
            </Select>
          </Field>
          <Button onClick={() => setAsked({ ...choice })}>{t("ts.regimes.run")}</Button>
        </div>
        {asked &&
          (regimes.isError ? (
            <ErrorNote>{regimes.error.message}</ErrorNote>
          ) : !regimes.data || regimes.isFetching ? (
            <Loading label={t("ts.regimes.loading")} className="h-64 rounded-lg" />
          ) : (
            <RegimesBody data={regimes.data} />
          ))}
      </div>
    </details>
  );
}

export function regimesFigure(data: SeriesRegimes) {
  return (tokens: ChartTokens): { data: Data[]; layout: Partial<Layout> } => {
    const found = data.regimes;
    const traces: Data[] = [
      {
        type: "scatter",
        mode: "lines",
        x: data.periods,
        y: found.values,
        line: { color: tokens.ink3, width: 1 },
        hoverinfo: "skip",
        showlegend: false,
      } as Data,
    ];
    found.regimes.forEach((regime, i) => {
      const index = found.most_likely.flatMap((r, j) => (r === i ? [j] : []));
      traces.push({
        type: "scattergl",
        mode: "markers",
        name: regime.name,
        x: index.map((j) => data.periods[j]),
        y: index.map((j) => found.values[j]),
        marker: { color: tokens.palette[i % 8] ?? tokens.ink2, size: 5 },
        hovertemplate: `${regime.name}<br>%{x}: %{y:,.4g}<extra></extra>`,
      } as Data);
    });
    const last = found.regimes.length - 1;
    traces.push({
      type: "scatter",
      mode: "lines",
      name: t("ts.regimes.probability", { name: found.regimes[last]?.name ?? "" }),
      x: data.periods,
      y: found.probabilities[last] ?? [],
      yaxis: "y2",
      line: { color: tokens.accent, width: 1.5 },
      hovertemplate: "%{x}: %{y:.0%}<extra></extra>",
    } as Data);
    return {
      data: traces,
      layout: {
        showlegend: true,
        legend: { orientation: "h" as const, x: 0, y: 1.12, font: { size: 11 } },
        yaxis: { domain: [0.32, 1] },
        yaxis2: { domain: [0, 0.24], range: [0, 1], tickformat: ".0%", anchor: "x" },
      },
    };
  };
}

function RegimesBody({ data }: { data: SeriesRegimes }) {
  const figure = useMemo(() => regimesFigure(data), [data]);
  const found = data.regimes;
  if (found.message) return <p className="text-ink-2">{found.message}</p>;
  return (
    <div className="space-y-3">
      {found.sentence && <p className="font-medium text-ink-1 text-pretty">{found.sentence}</p>}
      <p className="text-xs text-ink-3">
        {found.on_reason} {found.k_reason}
      </p>
      <Plot figure={figure} height={380} label={t("ts.regimes.chartLabel")} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-0 text-sm tabular-nums">
          <thead className="text-left text-xs text-ink-2">
            <tr>
              <th className="py-1 pr-3 font-normal">{t("ts.regimes.regime")}</th>
              <th className="py-1 pr-3 text-right font-normal">{t("ts.regimes.mean")}</th>
              <th className="py-1 pr-3 text-right font-normal">{t("ts.regimes.std")}</th>
              <th className="py-1 pr-3 text-right font-normal">{t("ts.regimes.share")}</th>
              <th className="py-1 text-right font-normal">{t("ts.regimes.duration")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line">
            {found.regimes.map((regime) => (
              <tr key={regime.name}>
                <td className="py-1 pr-3">{regime.name}</td>
                <td className="py-1 pr-3 text-right">{formatNumber(regime.mean)}</td>
                <td className="py-1 pr-3 text-right">{formatNumber(regime.std)}</td>
                <td className="py-1 pr-3 text-right">{Math.round(regime.share * 100)}%</td>
                <td className="py-1 text-right">
                  {regime.expected_duration === null ? "—" : formatNumber(regime.expected_duration)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
