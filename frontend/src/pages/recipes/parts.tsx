import {
  Ban,
  CircleCheck,
  CircleDashed,
  CircleX,
  LoaderCircle,
  SkipForward,
  TriangleAlert,
  type LucideIcon,
} from "lucide-react";

import type { FigureChange, StepLink } from "../../api/recipes";
import { formatNumber } from "../../charts/figures";
import { t, type MessageKey } from "../../lib/i18n";

type Meta = { icon: LucideIcon; tone: string; label: MessageKey };

/** Every status has an icon and a word: colour is never the only signal. */
export const STATUS: Record<string, Meta> = {
  waiting: { icon: CircleDashed, tone: "text-ink-3", label: "recipes.status.waiting" },
  running: { icon: LoaderCircle, tone: "text-ink-2", label: "recipes.status.running" },
  done: { icon: CircleCheck, tone: "text-ok", label: "recipes.status.done" },
  warning: { icon: TriangleAlert, tone: "text-warn", label: "recipes.status.warning" },
  skipped: { icon: SkipForward, tone: "text-ink-3", label: "recipes.status.skipped" },
  failed: { icon: CircleX, tone: "text-bad", label: "recipes.status.failed" },
  cancelled: { icon: Ban, tone: "text-ink-3", label: "recipes.status.cancelled" },
};

export function StatusTag({ status }: { status: string }) {
  const meta = STATUS[status] ?? STATUS.waiting!;
  const Icon = meta.icon;
  return (
    <span className="inline-flex shrink-0 items-center gap-1.5 text-xs text-ink-1">
      <Icon
        size={14}
        strokeWidth={1.75}
        aria-hidden
        className={`${meta.tone} ${status === "running" ? "motion-safe:animate-spin" : ""}`}
      />
      {t(meta.label)}
    </span>
  );
}

/** Where a step's link goes in the app (the page it filled). */
export function stepRoute(
  datasetId: string,
  link: StepLink,
): { to: string; params: Record<string, string>; search: Record<string, string> } | null {
  const p = link.params;
  const pick = (keys: string[]) =>
    Object.fromEntries(keys.filter((key) => p[key]).map((key) => [key, p[key] as string]));
  switch (link.page) {
    case "columns":
      return { to: "/d/$datasetId/columns", params: { datasetId }, search: {} };
    case "data":
      return { to: "/d/$datasetId/data", params: { datasetId }, search: {} };
    case "kpi":
      return { to: "/d/$datasetId/kpi", params: { datasetId }, search: pick(["kpi"]) };
    case "ml":
      return { to: "/d/$datasetId/ml", params: { datasetId }, search: pick(["target", "run"]) };
    case "timeseries":
      return {
        to: "/d/$datasetId/timeseries",
        params: { datasetId },
        search: pick(["time", "value", "group", "series", "period", "agg", "gaps", "season"]),
      };
    default:
      return null;
  }
}

/** "+12.4 (+3.1%)" with an arrow; "—" without a previous value. */
export function Change({ figure }: { figure: FigureChange }) {
  if (figure.change === null || figure.previous === null) {
    return <span className="text-ink-3">—</span>;
  }
  const up = figure.change > 0;
  const flat = figure.change === 0;
  const arrow = flat ? "→" : up ? "↑" : "↓";
  const sign = up ? "+" : "";
  const pct =
    figure.change_pct === null ? "" : ` (${sign}${(figure.change_pct * 100).toFixed(1)}%)`;
  // Better or worse, for a model score only (a KPI going up is not good or bad by itself).
  const judged =
    figure.higher_is_better === null || flat
      ? null
      : (figure.higher_is_better === up ? "better" : "worse");
  return (
    <span className="tabular-nums text-ink-1">
      <span aria-hidden>{arrow} </span>
      {sign}
      {Math.abs(figure.change) < 1
        ? figure.change.toLocaleString("en-US", { maximumSignificantDigits: 3 })
        : formatNumber(figure.change)}
      {pct}
      {judged && (
        <span className="ml-1.5 text-xs text-ink-3">
          {t(judged === "better" ? "recipes.better" : "recipes.worse")}
        </span>
      )}
    </span>
  );
}

/** A step's kind (Columns, Filters…), or a key figure's (KPI, Forecast, Model score). */
export function kindLabel(kind: string): string {
  return t(`recipes.kind.${kind}` as MessageKey);
}

export function figureKindLabel(kind: string): string {
  return t(`recipes.figure.${kind}` as MessageKey);
}
