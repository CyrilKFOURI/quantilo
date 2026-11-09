import type { AnalysisType, Changes, DateChange, LevelShift, Shift } from "../../api/client";
import { Plot } from "../../charts/Plot";
import { formatNumber } from "../../charts/figures";
import { TYPE_META } from "../../features/columns/columnTypes";
import { periodsFigure, shiftFigure, shiftHeight } from "../../features/changes/figures";
import { formatP } from "../../features/variable/normality";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { InfoTip } from "../../ui/InfoTip";
import { Empty, pct } from "./SchemaTabs";

const DRIFT: Record<NonNullable<Shift["drift"]>, { label: MessageKey; tone: string }> = {
  stable: { label: "changes.drift.stable", tone: "text-ink-3" },
  moderate: { label: "changes.drift.moderate", tone: "text-warn" },
  strong: { label: "changes.drift.strong", tone: "text-bad" },
};

const head = "px-3 py-1.5 font-normal";
const cell = "px-3 py-1.5";

/** Two shapes in 72 × 20 px: the old version filled grey, the new one as an accent line. */
function MiniShift({ shift }: { shift: Shift }) {
  const width = 72;
  const height = 20;
  const numeric = shift.histogram && !(shift.kind === "numeric" && shift.levels.length > 0);
  const label = t("changes.mini", { column: shift.column });
  if (numeric && shift.histogram) {
    const { before, after } = shift.histogram;
    const max = Math.max(...before, ...after, 1e-9);
    const step = width / Math.max(before.length, 1);
    const line = (values: number[]) =>
      values.map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(height - (v / max) * height).toFixed(1)}h${step.toFixed(1)}`).join("");
    return (
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} className="shrink-0">
        <path d={`${line(before)}V${height}H0Z`} className="fill-ink-3 opacity-40" />
        <path d={line(after)} className="fill-none stroke-accent" strokeWidth={1.25} />
      </svg>
    );
  }
  const levels = shift.levels.slice(0, 6);
  if (levels.length === 0) return <span className="inline-block h-5 w-[72px]" aria-hidden />;
  const max = Math.max(...levels.flatMap((l) => [l.before, l.after]), 1e-9);
  const slot = width / levels.length;
  const bar = Math.max(slot / 2 - 1, 1);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} role="img" aria-label={label} className="shrink-0">
      {levels.map((level, i) => {
        const b = (level.before / max) * height;
        const a = (level.after / max) * height;
        return (
          <g key={level.value}>
            <rect x={i * slot} y={height - b} width={bar} height={b} className="fill-ink-3 opacity-60" />
            <rect x={i * slot + bar + 0.5} y={height - a} width={bar} height={a} className="fill-accent" />
          </g>
        );
      })}
    </svg>
  );
}

export function DriftBadge({ shift }: { shift: Shift }) {
  if (shift.long_tail)
    return (
      <span className="text-xs text-ink-3" title={t("changes.longTailHelp")}>
        {t("changes.longTail")}
      </span>
    );
  if (!shift.drift) return <span className="text-xs text-ink-3">—</span>;
  const meta = DRIFT[shift.drift];
  return <span className={`text-xs font-medium ${meta.tone}`}>{t(meta.label)}</span>;
}

export function DistributionsTab({
  changes,
  types,
  column,
  onColumn,
}: {
  changes: Changes;
  types: Map<string, AnalysisType>;
  column: string | undefined;
  onColumn: (column: string) => void;
}) {
  const shifts = changes.distributions;
  const chosen = shifts.find((shift) => shift.column === column) ?? shifts[0];
  const tone = (name: string, type: AnalysisType) => TYPE_META[types.get(name) ?? type].tone;
  if (shifts.length === 0 && changes.dates.length === 0) return <Empty>{t("changes.distributions.none")}</Empty>;
  return (
    <div className="flex flex-col gap-5">
      {shifts.length > 0 && (
        <section>
          <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
            <h2 className="text-sm font-semibold">{t("changes.distributions.title")}</h2>
            <span className="text-xs text-ink-3">{changes.thresholds}</span>
          </div>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th scope="col" className={`${head} w-full`}>{t("changes.col.column")}</th>
                  <th scope="col" className={`${head} text-right`}>
                    <span className="inline-flex items-center gap-1">
                      PSI <InfoTip label={t("changes.psiLabel")} text={t("changes.psiHelp")} />
                    </span>
                  </th>
                  <th scope="col" className={head}>{t("changes.col.drift")}</th>
                  <th scope="col" className={`${head} text-right`}>
                    <span className="inline-flex items-center gap-1">
                      {t("changes.col.distance")} <InfoTip label={t("changes.distanceLabel")} text={t("changes.distanceHelp")} />
                    </span>
                  </th>
                  <th scope="col" className={`${head} text-right`}>{t("changes.col.test")}</th>
                  <th scope="col" className={head}><span className="sr-only">{t("changes.col.shape")}</span></th>
                </tr>
              </thead>
              <tbody>
                {shifts.map((shift) => {
                  const selected = shift.column === chosen?.column;
                  return (
                    <tr
                      key={shift.column}
                      aria-selected={selected}
                      className={`cursor-pointer border-t border-line ${selected ? "bg-accent-soft" : "hover:bg-surface-2"}`}
                      onClick={() => onColumn(shift.column)}
                    >
                      <td className={cell}>
                        <button type="button" onClick={() => onColumn(shift.column)} className={`font-mono text-xs font-medium ${tone(shift.column, shift.type)}`}>
                          {shift.column}
                        </button>
                        {shift.focus && <span className="ml-2 text-xs text-ink-3">{t("changes.focus")}</span>}
                        {shift.appeared.length > 0 && (
                          <span className="ml-2 text-xs text-ink-2">{t("changes.newValues", { count: String(shift.appeared.length) })}</span>
                        )}
                      </td>
                      <td className={`${cell} text-right text-ink-1 tabular-nums`}>{shift.psi === null ? "—" : shift.psi.toFixed(3)}</td>
                      <td className={cell}>
                        <DriftBadge shift={shift} />
                      </td>
                      <td className={`${cell} text-right text-ink-2 tabular-nums`}>
                        {shift.wasserstein === null ? "—" : `${shift.wasserstein.toFixed(2)} IQR`}
                      </td>
                      <td className={`${cell} text-right whitespace-nowrap text-ink-2 tabular-nums`}>
                        {shift.ks !== null
                          ? `KS ${shift.ks.toFixed(3)} · p ${formatP(shift.ks_p)}`
                          : shift.chi2 !== null
                            ? `χ² ${formatNumber(shift.chi2)} · p ${formatP(shift.chi2_p)}`
                            : "—"}
                      </td>
                      <td className={cell}>
                        <MiniShift shift={shift} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {chosen && <ShiftDetail shift={chosen} />}
      {changes.dates.map((dates) => (
        <DatesDetail key={dates.column} dates={dates} />
      ))}
    </div>
  );
}

function Moment({ label, before, after }: { label: string; before: number | null; after: number | null }) {
  return (
    <div>
      <dt className="text-xs text-ink-3">{label}</dt>
      <dd className="text-ink-1 tabular-nums">
        {formatNumber(before)} → {formatNumber(after)}
      </dd>
    </div>
  );
}

function LevelList({ title, levels, side }: { title: string; levels: LevelShift[]; side: "before" | "after" }) {
  if (levels.length === 0) return null;
  return (
    <div>
      <h4 className="text-xs text-ink-3">{title}</h4>
      <ul className="mt-0.5 space-y-0.5 text-[13px]">
        {levels.map((level) => (
          <li key={level.value} className="flex gap-2">
            <span className="truncate text-ink-1">{level.value}</span>
            <span className="text-ink-2 tabular-nums">{pct(100 * level[side])}</span>
            {!level.certain && <span className="text-xs text-ink-3" title={t("changes.uncertainHint")}>{t("changes.uncertain")}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ShiftDetail({ shift }: { shift: Shift }) {
  const figure = shiftFigure(shift);
  return (
    <section className="rounded-lg border border-line bg-surface-1 p-4">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
        <h3 className="font-mono text-sm font-semibold">{shift.column}</h3>
        <DriftBadge shift={shift} />
        <span className="text-xs text-ink-3">
          {t("changes.values", { before: formatCount(shift.rows_before), after: formatCount(shift.rows_after) })}
        </span>
      </div>
      {(shift.histogram || shift.levels.length > 0) && (
        <Plot figure={figure} height={shiftHeight(shift)} label={t("changes.chartLabel", { column: shift.column })} />
      )}
      <div className="mt-3 flex flex-wrap gap-x-10 gap-y-3">
        {shift.before && shift.after && (
          <dl className="grid grid-cols-3 gap-x-6 text-[13px]">
            <Moment label={t("changes.mean")} before={shift.before.mean} after={shift.after.mean} />
            <Moment label={t("changes.median")} before={shift.before.median} after={shift.after.median} />
            <Moment label={t("changes.std")} before={shift.before.std} after={shift.after.std} />
          </dl>
        )}
        <LevelList title={t("changes.appeared")} levels={shift.appeared} side="after" />
        <LevelList title={t("changes.disappeared")} levels={shift.disappeared} side="before" />
        {shift.moves.length > 0 && (
          <div>
            <h4 className="text-xs text-ink-3">{t("changes.moves")}</h4>
            <ul className="mt-0.5 space-y-0.5 text-[13px]">
              {shift.moves.slice(0, 5).map((level) => (
                <li key={level.value} className="flex gap-2">
                  <span className="truncate text-ink-1">{level.value}</span>
                  <span className="text-ink-2 tabular-nums">
                    {pct(100 * level.before)} → {pct(100 * level.after)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </section>
  );
}

function DatesDetail({ dates }: { dates: DateChange }) {
  return (
    <section className="rounded-lg border border-line bg-surface-1 p-4">
      <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
        <h3 className="font-mono text-sm font-semibold">{dates.column}</h3>
        <span className="text-xs text-ink-3">
          {t("changes.dates.range", {
            before: `${dates.before_min ?? "—"} – ${dates.before_max ?? "—"}`,
            after: `${dates.after_min ?? "—"} – ${dates.after_max ?? "—"}`,
          })}
        </span>
      </div>
      {dates.periods.length > 0 && (
        <Plot figure={periodsFigure(dates)} height={260} label={t("changes.periodsLabel", { column: dates.column })} />
      )}
      {(dates.future > 0 || dates.gaps.length > 0) && (
        <p className="mt-2 text-xs text-warn">
          {dates.future > 0 && t("changes.quality.future", { count: formatCount(dates.future) })}
          {dates.future > 0 && dates.gaps.length > 0 && " · "}
          {dates.gaps.length > 0 && t("changes.quality.gaps", { periods: dates.gaps.join(", ") })}
        </p>
      )}
    </section>
  );
}
