import { useState } from "react";

import type { Column } from "../../api/client";
import { useChart } from "../../api/hooks";
import { categoryBars } from "../../charts/figures";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Dialog } from "../../ui/Dialog";
import { CategoryTable } from "../variable/CategoryTable";
import { VariableDetails } from "../variable/VariableDetails";
import { ChartView } from "./ChartView";
import { ViewSwitch } from "./ViewSwitch";
import { Loading } from "../../ui/Loading";

/** Beyond this many categories, bars stop being readable: the table carries the detail. */
const MAX_FULL_BARS = 120;
/** Tall bar charts scroll inside this height, so the details below stay in sight. */
const VISIBLE_CHART_HEIGHT = 420;

type FullscreenChartProps = {
  datasetId: string;
  column: Column;
  view: string;
  step: string | null;
  onClose: () => void;
};

export function FullscreenChart({ datasetId, column, view: initialView, step: initialStep, onClose }: FullscreenChartProps) {
  const [view, setView] = useState<string | null>(initialView);
  const [step, setStep] = useState<string | null>(initialStep);
  const chart = useChart(datasetId, column.name, { view, step, full: true }, true);
  const data = chart.data;
  const categorical = data?.kind === "bars" && view !== "month" && view !== "day_of_week" && view !== "hour";
  const barCount = data ? categoryBars(data).length : 0;
  const height = categorical ? Math.min(Math.max(360, barCount * 22), 2600) : 440;

  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={column.name}
      description={data ? t("fullscreen.summary", { rows: formatCount(data.rows), distinct: formatCount(data.distinct) }) : undefined}
      size="lg"
    >
      {data && data.views.length > 1 && (
        <div className="mb-3">
          <ViewSwitch chart={data} onView={setView} onStep={setStep} />
        </div>
      )}
      {data ? (
        <div className="space-y-5">
          {!(categorical && barCount > MAX_FULL_BARS) && (
            <div className="overflow-y-auto" style={{ maxHeight: categorical ? VISIBLE_CHART_HEIGHT : undefined }}>
              <ChartView chart={data} type={column.type} name={column.name} height={height} zoom="explore" />
            </div>
          )}
          {categorical && <CategoryTable chart={data} />}
          <VariableDetails datasetId={datasetId} column={column.name} />
        </div>
      ) : (
        <Loading label={t("loading.chart")} className="h-96 rounded-md" />
      )}
    </Dialog>
  );
}
