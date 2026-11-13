import type { Data } from "plotly.js-dist-min";

import type {
  SeriesAutocorrelation,
  SeriesStationarity,
} from "../../api/client";
import { Plot, type Figure } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { MethodsTable } from "./Methods";
import { TRANSFORMS, type Transform } from "./search";

export type { SeriesAutocorrelation, SeriesStationarity };
type Correlogram = NonNullable<SeriesAutocorrelation["correlogram"]>;
type Verdict = { text: string; tone: "ok" | "none"; detail?: string } | null;

const transformName = (how: string) => t(`ts.transform.${how}` as MessageKey);

/** None · Log · Difference…, the suggested one marked. */
export function TransformPicker({
  value,
  suggested,
  onChange,
}: {
  value: Transform;
  suggested: string | null;
  onChange: (value: Transform) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={t("ts.transform")}
      className="inline-flex flex-wrap rounded-md border border-line bg-surface-2 p-0.5 text-xs"
    >
      {TRANSFORMS.map((how) => (
        <button
          key={how}
          type="button"
          role="radio"
          aria-checked={value === how}
          onClick={() => onChange(how)}
          className={`inline-flex h-6 items-center gap-1 rounded px-2 ${
            value === how
              ? "bg-surface-1 text-ink-1 shadow-sm"
              : "text-ink-2 hover:text-ink-1"
          }`}
        >
          {transformName(how)}
          {suggested === how && (
            <span className="rounded-full bg-accent-soft px-1.5 text-[10px] text-ink-1">
              {t("ts.suggested")}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

function transformedFigure(data: SeriesStationarity): Figure {
  return (tokens) => ({
    data: [
      {
        type: "scatter",
        mode: "lines",
        x: data.periods,
        y: data.transformed,
        line: { color: tokens.palette[0] ?? tokens.date, width: 1.5 },
        hovertemplate: "%{x}: %{y:,.4g}<extra></extra>",
      } as Data,
    ],
    layout: {
      showlegend: false,
      xaxis: { type: "date" },
      yaxis: { zeroline: true, zerolinecolor: tokens.line },
    },
  });
}

export function StationarityBody({ data }: { data: SeriesStationarity }) {
  if (data.message) return <p className="text-ink-2">{data.message}</p>;
  return (
    <div className="space-y-3">
      <div className="rounded-lg border border-line bg-surface-1 p-3">
        <p className="mb-1 text-xs text-ink-3">
          {t("ts.transformed", {
            how: transformName(data.transform).toLowerCase(),
          })}
          {" · "}
          {t(data.regression === "ct" ? "ts.aroundTrend" : "ts.aroundLevel")}
        </p>
        <Plot
          figure={transformedFigure(data)}
          height={220}
          label={t("ts.section.stationarity")}
        />
      </div>
      <MethodsTable results={data.methods ?? []} />
      {data.filled ? (
        <p className="text-xs text-ink-3">
          {t("ts.filled", { count: String(data.filled) })}
        </p>
      ) : null}
    </div>
  );
}

export function stationarityVerdict(data: SeriesStationarity): Verdict {
  const synthesis = data.synthesis;
  if (!synthesis) return null;
  const verdict = t(`ts.stationarity.${synthesis.verdict}` as MessageKey);
  const suggestion = data.suggestion;
  const advice =
    synthesis.verdict === "stationary" ||
    !suggestion ||
    suggestion.transform === data.transform
      ? ""
      : ` ${t("ts.stationarity.after", { how: transformName(suggestion.transform).toLowerCase() })}`;
  const short = t(`ts.stationarity.short.${synthesis.verdict}` as MessageKey);
  const hint = advice
    ? ` · ${t("ts.stationarity.try", { how: transformName(suggestion?.transform ?? "").toLowerCase() })}`
    : "";
  return {
    text: `${short}${hint}`,
    detail: `${verdict}${advice}`,
    tone: synthesis.verdict === "stationary" ? "ok" : "none",
  };
}

function barsFigure(lags: number[], values: number[], band: number): Figure {
  return (tokens) => ({
    data: [
      {
        type: "bar",
        x: lags,
        y: values,
        marker: {
          color: values.map((v) =>
            Math.abs(v) > band
              ? (tokens.palette[0] ?? tokens.numeric)
              : tokens.ink3,
          ),
        },
        hovertemplate: t("fig.ts.hoverLag"),
      } as Data,
      {
        type: "scatter",
        mode: "lines",
        x: [0, Math.max(...lags) + 1, null, 0, Math.max(...lags) + 1],
        y: [band, band, null, -band, -band],
        line: { color: tokens.ink3, width: 1, dash: "dot" },
        hoverinfo: "skip",
      } as Data,
    ],
    layout: {
      showlegend: false,
      bargap: 0.35,
      xaxis: { title: { text: t("fig.ts.lag"), font: { size: 11 } } },
      yaxis: { range: [-1, 1] },
    },
  });
}

export function AutocorrelationBody({
  data,
  pacf,
  onPacf,
  arOrder,
  onAr,
}: {
  data: SeriesAutocorrelation;
  pacf: string;
  onPacf: (value: string) => void;
  arOrder: number | undefined;
  onAr: (order: number | undefined) => void;
}) {
  const gram = data.correlogram;
  if (data.message || !gram)
    return <p className="text-ink-2">{data.message}</p>;
  return (
    <div className="space-y-3">
      <ul className="space-y-1 text-[13px] text-ink-1">
        {(gram.readings ?? []).map((reading) => (
          <li key={reading}>
            •{" "}
            {t(`ts.reading.${reading}` as MessageKey, {
              order: String(gram.ar_order ?? ""),
            })}
          </li>
        ))}
      </ul>
      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-lg border border-line bg-surface-1 p-3">
          <h3 className="text-[13px] font-medium">{t("ts.acf")}</h3>
          <p className="mb-1 text-xs text-ink-3">{t("ts.acfHint")}</p>
          <Plot
            figure={barsFigure(gram.lags, gram.acf, gram.band)}
            height={240}
            label={t("ts.acf")}
          />
        </div>
        <div className="rounded-lg border border-line bg-surface-1 p-3">
          <div className="flex items-center justify-between gap-2">
            <h3 className="text-[13px] font-medium">{t("ts.pacf")}</h3>
            <Select
              aria-label={t("ts.pacfMethod")}
              value={pacf}
              onChange={(event) => onPacf(event.target.value)}
              className="h-7 text-xs"
            >
              <option value="ywm">Yule-Walker</option>
              <option value="ols">OLS</option>
              <option value="burg">Burg</option>
            </Select>
          </div>
          <p className="mb-1 text-xs text-ink-3">{t("ts.pacfHint")}</p>
          <Plot
            figure={barsFigure(
              gram.lags.slice(0, gram.pacf.length),
              gram.pacf,
              gram.band,
            )}
            height={240}
            label={t("ts.pacf")}
          />
        </div>
      </div>
      {data.ar && <ArPanel fit={data.ar} order={arOrder} onOrder={onAr} />}
      <MethodsTable results={data.methods ?? []} />
      {data.filled ? (
        <p className="text-xs text-ink-3">
          {t("ts.filled", { count: String(data.filled) })}
        </p>
      ) : null}
    </div>
  );
}

export function autocorrelationVerdict(data: SeriesAutocorrelation): Verdict {
  const gram: Correlogram | null | undefined = data.correlogram;
  if (!gram) return null;
  const readings = gram.readings ?? [];
  if (readings.includes("white_noise"))
    return { text: t("ts.memory.none"), tone: "none" };
  const first = readings[0];
  return {
    text: first
      ? t(`ts.memory.${first}` as MessageKey, {
          order: String(gram.ar_order ?? ""),
        })
      : t("ts.memory.some"),
    tone: "ok",
  };
}

type ArFitView = NonNullable<SeriesAutocorrelation["ar"]>;

/** An AR(p) model: p chosen by AIC or by the user, its coefficients, and the AIC / BIC of every
 *  order to help choose. */
function ArPanel({ fit, order, onOrder }: { fit: ArFitView; order: number | undefined; onOrder: (order: number | undefined) => void }) {
  const best = [...fit.orders].sort((a, b) => (a.aic ?? Infinity) - (b.aic ?? Infinity))[0];
  return (
    <div className="space-y-2 rounded-lg border border-line bg-surface-1 p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-[13px] font-medium">{t("ts.ar.title", { order: String(fit.order) })}</h3>
        <label className="flex items-center gap-1.5 text-xs text-ink-2">
          {t("ts.ar.order")}
          <Select
            value={order ? String(order) : ""}
            onChange={(event) => onOrder(event.target.value ? Number(event.target.value) : undefined)}
            className="h-7 text-xs"
          >
            <option value="">{t("ts.ar.auto", { order: String(best?.order ?? fit.order) })}</option>
            {fit.orders.map((o) => (
              <option key={o.order} value={o.order}>
                {o.order}
              </option>
            ))}
          </Select>
        </label>
      </div>
      <p className="text-xs text-ink-3">
        {fit.chosen_by === "AIC" ? t("ts.ar.byAic") : t("ts.ar.byYou")}
        {fit.sigma !== null ? ` · σ = ${fit.sigma.toPrecision(4)}` : ""}
      </p>
      {best && best.order === fit.orders[fit.orders.length - 1]?.order && (
        <p className="text-[13px] text-warn">{t("ts.ar.atBoundary", { order: String(best.order) })}</p>
      )}
      <p className={`text-[13px] ${fit.residuals.significant ? "text-warn" : "text-ink-2"}`}>
        {fit.residuals.significant ? t("ts.ar.residualsLinked") : t("ts.ar.residualsWhite")}
      </p>
      <div className="grid gap-3 xl:grid-cols-2">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="text-left text-xs text-ink-3">
            <tr>
              <th className="py-1 font-normal">{t("ts.ar.term")}</th>
              <th className="py-1 text-right font-normal">{t("ts.ar.coefficient")}</th>
              <th className="py-1 text-right font-normal">{t("ts.ar.stdError")}</th>
              <th className="py-1 text-right font-normal">p</th>
            </tr>
          </thead>
          <tbody>
            {fit.coefficients.map((c) => (
              <tr key={c.term} className="border-t border-line">
                <td className="py-1">{c.term}</td>
                <td className="py-1 text-right">{c.estimate.toPrecision(4)}</td>
                <td className="py-1 text-right text-ink-2">{c.std_error === null ? "—" : c.std_error.toPrecision(3)}</td>
                <td className={`py-1 text-right ${c.p_value !== null && c.p_value < 0.05 ? "text-ink-1" : "text-ink-3"}`}>
                  {c.p_value === null ? "—" : c.p_value < 0.001 ? "< 0.001" : c.p_value.toFixed(3)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead className="text-left text-xs text-ink-3">
            <tr>
              <th className="py-1 font-normal">p</th>
              <th className="py-1 text-right font-normal">AIC</th>
              <th className="py-1 text-right font-normal">BIC</th>
            </tr>
          </thead>
          <tbody>
            {fit.orders.map((o) => (
              <tr key={o.order} className={`border-t border-line ${o.order === fit.order ? "bg-accent-soft" : ""}`}>
                <td className="py-1">{o.order}{o.order === best?.order ? " ★" : ""}</td>
                <td className="py-1 text-right">{o.aic === null ? "—" : o.aic.toFixed(1)}</td>
                <td className="py-1 text-right text-ink-2">{o.bic === null ? "—" : o.bic.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
