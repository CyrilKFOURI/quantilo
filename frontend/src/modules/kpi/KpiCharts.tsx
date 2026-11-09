import type { KpiMeasure, KpiResult } from "../../api/client";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import type { Format, ShowAs } from "./definition";
import { kpiFigure, shareFigure, type ChartInput } from "./figures";
import { effectiveMode } from "./pivot";

const OTHER = "Other";
const PANELS_MAX = 6;

/** The chart drawn for a measure: the one chosen in Customize when this measure allows it,
 *  otherwise the automatic one. */
export function chartFor(measure: KpiMeasure, chosen: string): string {
  if (chosen === "auto") return measure.chart.value;
  const option = measure.chart.options.find((o) => o.value === chosen);
  return option?.allowed ? chosen : measure.chart.value;
}

export const chartName = (chart: string) => t(`kpi.chart.${chart}` as MessageKey);

export function KpiCharts({
  result,
  chart,
  share,
  showAs,
  labelsMode,
  hideOther,
  line,
  spacing,
  labels,
  formats,
}: {
  result: KpiResult;
  chart: string;
  share: boolean;
  showAs: ShowAs;
  labelsMode: "auto" | "on" | "off";
  hideOther: boolean;
  line: boolean;
  spacing: "narrow" | "normal" | "wide";
  labels: Record<string, string>;
  formats: Record<string, Format>;
}) {
  const multiples = result.resolved.time && result.splits.length === 2;
  const panels = multiples
    ? (result.splits[0]?.levels ?? []).filter((l) => !(hideOther && l === OTHER)).slice(0, PANELS_MAX)
    : [];
  return (
    <div className="space-y-6">
      {result.measures.map((measure) => {
        const drawn = chartFor(measure, chart);
        if (drawn === "none") return null;
        const input: ChartInput = {
          result,
          measure: measure.id,
          chart: drawn,
          mode: effectiveMode(showAs, measure.additive, result.resolved.time),
          format: measure.format,
          custom: formats[measure.id] ?? {},
          labels: labelsMode,
          hideOther,
          line,
          spacing,
        };
        const name = labels[measure.id] ?? measure.label;
        const height = result.resolved.time ? 240 : Math.min(520, 70 + 26 * (result.splits[0]?.levels.length ?? 4));
        return (
          <figure key={measure.id} className="space-y-1">
            {result.measures.length > 1 && <figcaption className="text-[13px] font-medium text-ink-1">{name}</figcaption>}
            {multiples ? (
              <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                {panels.map((panel) => (
                  <div key={panel}>
                    <p className="text-xs font-medium text-ink-2">{panel}</p>
                    <Plot figure={kpiFigure({ ...input, panel })} height={200} label={`${name} · ${panel}`} />
                  </div>
                ))}
              </div>
            ) : (
              <Plot figure={kpiFigure(input)} height={height} label={`${name}: ${chartName(drawn)}`} />
            )}
            {share && measure.share.allowed && !multiples && (
              <div>
                <p className="text-xs text-ink-2">{t("kpi.shareChart", { measure: name })}</p>
                <Plot figure={shareFigure(input)} height={140} label={t("kpi.shareChart", { measure: name })} />
              </div>
            )}
          </figure>
        );
      })}
    </div>
  );
}
