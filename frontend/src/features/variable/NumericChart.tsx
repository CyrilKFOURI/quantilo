import { useMemo, useState } from "react";

import type { AnalysisType, NumericProfile } from "../../api/client";
import { useChart } from "../../api/hooks";
import { barsFigure, histogramFigure } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { Select } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { boxFromProfile, ecdfFigure, kdeFigure, qqFigure, violinFigure } from "./figures";
import { Loading } from "../../ui/Loading";

const VIEWS = ["histogram", "kde", "ecdf", "box", "violin", "qq"] as const;
type View = (typeof VIEWS)[number];
const BIN_CHOICES = [10, 20, 30, 50, 80, 120];
const HEIGHT = 340;

type NumericChartProps = {
  datasetId: string;
  column: string;
  type: AnalysisType;
  profile: NumericProfile;
};

/** The variable page's main chart: six ways to look at one distribution. */
export function NumericChart({ datasetId, column, type, profile }: NumericChartProps) {
  const [view, setView] = useStoredState<View>("de.variable.view", "histogram");
  const [bins, setBins] = useState<number | null>(null);
  // A discrete variable's "histogram" is one bar per value (the chart endpoint sends bars).
  const discrete = type === "numeric_discrete";
  const chart = useChart(datasetId, column, { view: "histogram", bins: discrete ? null : bins }, view === "histogram");
  const describe = profile.describe;

  const figure = useMemo(() => {
    if (view === "kde") return kdeFigure(profile);
    if (view === "ecdf") return ecdfFigure(profile);
    if (view === "violin") return violinFigure(profile);
    if (view === "box") return boxFromProfile(profile);
    if (view === "qq" && profile.qq) {
      return qqFigure(profile.qq.theoretical, profile.qq.sample, profile.qq, t("fig.variable.normalQuantilesZ"));
    }
    if (!chart.data) return null;
    return chart.data.kind === "bars" ? barsFigure(chart.data, type) : histogramFigure(chart.data, type);
  }, [view, profile, chart.data, type]);

  const sampled = profile.sample.length < describe.count;
  const note =
    (view === "violin" || view === "box") && sampled
      ? t("variable.sampleNote", { sample: formatCount(profile.sample.length), n: formatCount(describe.count) })
      : view === "qq"
        ? t("variable.qqNote")
        : null;

  return (
    <section className="rounded-lg border border-line bg-surface-1 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <SegmentedControl
          label={t("variable.views")}
          value={view}
          options={VIEWS.map((value) => ({ value, label: t(`variable.view.${value}`) }))}
          onChange={setView}
        />
        {view === "histogram" && !discrete && (
          <label className="flex items-center gap-1.5 text-xs text-ink-3">
            {t("variable.bins")}
            <Select
              value={bins ?? ""}
              onChange={(event) => setBins(event.target.value ? Number(event.target.value) : null)}
              className="h-7 text-xs"
            >
              <option value="">
                {t("variable.binsAuto")}
                {chart.data && !bins ? ` (${chart.data.bins.length})` : ""}
              </option>
              {BIN_CHOICES.map((choice) => (
                <option key={choice} value={choice}>
                  {choice}
                </option>
              ))}
            </Select>
          </label>
        )}
      </div>
      {figure ? (
        <Plot figure={figure} height={HEIGHT} label={t("chart.label", { name: column })} />
      ) : (
        <Loading label={t("loading.chart")} style={{ height: HEIGHT }} className="rounded-md" />
      )}
      {note && <p className="mt-1.5 text-xs text-ink-3">{note}</p>}
    </section>
  );
}
