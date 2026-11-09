import { forwardRef, useCallback, useMemo, useState } from "react";

import type { AnalysisType, Chart } from "../../api/client";
import { boxFigure, barsFigure, histogramFigure, pieFigure, timelineFigure } from "../../charts/figures";
import { Plot, type PlotHandle, type PointClick } from "../../charts/Plot";
import { ChartFilterMenu } from "../filters/ChartFilterMenu";
import { actionsFor, type ChartAction } from "../filters/fromChart";
import { useFilters } from "../filters/useFilters";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";

type ChartViewProps = {
  chart: Chart;
  type: AnalysisType;
  name: string;
  height: number;
  /** Set to let a click on a bar or bin filter the dataset. */
  datasetId?: string;
};

/** Draws prepared chart data; non-chart kinds (identifier, high cardinality) get a short summary. */
export const ChartView = forwardRef<PlotHandle, ChartViewProps>(function ChartView(
  { chart, type, name, height, datasetId },
  ref,
) {
  const [menu, setMenu] = useState<{ x: number; y: number; actions: ChartAction[] } | null>(null);
  const filters = useFilters(datasetId ?? "");
  const onPointClick = useCallback(
    (point: PointClick) => {
      const actions = actionsFor(name, chart, point);
      if (actions.length) setMenu({ x: point.clientX, y: point.clientY, actions });
    },
    [name, chart],
  );
  const closeMenu = useCallback(() => setMenu(null), []);
  const figure = useMemo(() => {
    if (chart.kind === "histogram") {
      if (chart.view === "box") return boxFigure(chart, type);
      return histogramFigure(chart, chart.view === "lengths" ? "text" : type, chart.view === "lengths" ? "texts" : "rows");
    }
    if (chart.kind === "timeline") return timelineFigure(chart);
    if (chart.kind === "bars") return chart.view === "pie" ? pieFigure(chart) : barsFigure(chart, type);
    return null;
  }, [chart, type]);

  if (figure) {
    return (
      <>
        <Plot
          ref={ref}
          figure={figure}
          height={height}
          label={t("chart.label", { name })}
          onPointClick={datasetId ? onPointClick : undefined}
        />
        {menu && (
          <ChartFilterMenu
            x={menu.x}
            y={menu.y}
            actions={menu.actions}
            onClose={closeMenu}
            onPick={(action) => {
              filters.add(action.condition);
              setMenu(null);
            }}
          />
        )}
      </>
    );
  }

  if (chart.kind === "identifier") {
    return (
      <div style={{ height }} className="flex flex-col justify-center gap-3 px-1">
        <Metric label={t("chart.unique")} value={formatCount(chart.distinct)} />
        <Metric
          label={t("chart.duplicated")}
          value={formatCount(chart.duplicates)}
          tone={chart.duplicates ? "text-warn" : "text-ok"}
        />
        <p className="truncate font-mono text-xs text-ink-3 max-lg:max-w-full">{chart.examples.join(" · ")}</p>
      </div>
    );
  }
  return (
    <div style={{ height }} className="flex flex-col items-center justify-center gap-2 px-4 text-center">
      <p className="text-ink-2">{chart.message ?? t("chart.empty")}</p>
      {chart.kind === "high_cardinality" && (
        <p className="truncate font-mono text-xs text-ink-3 max-lg:max-w-full">{chart.examples.join(" · ")}</p>
      )}
    </div>
  );
});

function Metric({ label, value, tone = "text-ink-1" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-3">
      <span className="text-xs text-ink-3">{label}</span>
      <span className={`text-lg font-semibold tabular-nums ${tone}`}>{value}</span>
    </div>
  );
}
