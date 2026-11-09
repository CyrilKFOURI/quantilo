import { Link } from "@tanstack/react-router";
import { ArrowUpRight, Download, Maximize2 } from "lucide-react";
import { useRef, useState } from "react";

import type { AnalysisType, Column } from "../../api/client";
import { useChart } from "../../api/hooks";
import type { PlotHandle } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { useInView } from "../../lib/useInView";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { TypeBadge } from "../columns/TypeBadge";
import { cardStats } from "./cardStats";
import { ChartView } from "./ChartView";
import { FullscreenChart } from "./FullscreenChart";
import { ViewSwitch } from "./ViewSwitch";
import { Loading } from "../../ui/Loading";

type VariableCardProps = {
  datasetId: string;
  column: Column;
  topN: number;
  onChangeType: (type: AnalysisType) => void;
};

export const CARD_CHART_HEIGHT = 190;
/** On a phone the cards go two per row (041): a shorter chart. */
const PHONE_CHART_HEIGHT = 130;

export function VariableCard({ datasetId, column, topN, onChangeType }: VariableCardProps) {
  const { ref, seen } = useInView<HTMLElement>();
  const [view, setView] = useState<string | null>(null);
  const [step, setStep] = useState<string | null>(null);
  const [fullscreen, setFullscreen] = useState(false);
  const plot = useRef<PlotHandle>(null);
  const chart = useChart(datasetId, column.name, { view, topN, step }, seen);
  const data = chart.data;
  const stats = cardStats(column, data);
  const height = useMedia(PHONE) ? PHONE_CHART_HEIGHT : CARD_CHART_HEIGHT;

  return (
    <article ref={ref} className="flex min-w-0 flex-col rounded-lg border border-line bg-surface-1">
      <header className="flex items-start gap-2 px-3 pt-2.5 max-sm:gap-1 max-sm:px-2">
        <div className="min-w-0 flex-1">
          <h3 className="truncate font-mono text-[13px] font-medium text-ink-1" title={column.name}>
            {column.name}
          </h3>
          <div className="-ml-1.5">
            <TypeBadge type={column.type} detectedType={column.detected_type} reasons={column.reasons} onChange={onChangeType} />
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-0.5">
          <Button
            variant="ghost"
            icon={Download}
            iconOnly
            className="max-sm:hidden"
            aria-label={t("card.download")}
            title={t("card.download")}
            disabled={!data}
            onClick={() => void plot.current?.toImage(column.name)}
          />
          <Button
            variant="ghost"
            icon={Maximize2}
            iconOnly
            aria-label={t("card.fullscreen", { name: column.name })}
            title={t("card.fullscreenHint")}
            disabled={!data}
            onClick={() => setFullscreen(true)}
          />
          <Link
            to="/d/$datasetId/variable/$column"
            params={{ datasetId, column: column.name }}
            aria-label={t("card.details")}
            title={t("card.details")}
            className="inline-flex size-(--control-h) items-center justify-center rounded-md text-ink-2 hover:bg-surface-2 hover:text-ink-1"
          >
            <ArrowUpRight size={15} strokeWidth={1.5} aria-hidden />
          </Link>
        </div>
      </header>

      {data && data.views.length > 1 && (
        <div className="px-3 pt-1.5 max-sm:px-2">
          <ViewSwitch chart={data} onView={setView} onStep={setStep} />
        </div>
      )}

      <div className="px-2 pt-1 max-sm:px-1">
        {data ? (
          <ChartView ref={plot} chart={data} type={column.type} name={column.name} height={height} datasetId={datasetId} />
        ) : chart.isError ? (
          <p style={{ height }} className="flex items-center justify-center px-4 text-center text-xs text-bad">
            {chart.error.message}
          </p>
        ) : (
          <Loading label={t("loading.chart")} style={{ height }} className="m-1 rounded-md" compact />
        )}
      </div>

      <dl className="mt-auto flex flex-wrap gap-x-4 gap-y-1 border-t border-line px-3 py-2 text-xs max-sm:gap-x-3 max-sm:px-2">
        {stats.map((stat) => (
          <div key={stat.label} className="flex min-w-0 items-baseline gap-1.5">
            <dt className="truncate text-ink-3">{stat.label}</dt>
            <dd className="text-ink-1 tabular-nums">{stat.value}</dd>
          </div>
        ))}
      </dl>

      {fullscreen && data && (
        <FullscreenChart datasetId={datasetId} column={column} view={data.view} step={step} onClose={() => setFullscreen(false)} />
      )}
    </article>
  );
}
