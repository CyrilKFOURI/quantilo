import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

import type { SyntheticColumnDetail, SyntheticRun } from "../../api/client";
import { useSyntheticColumn } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { t, type MessageKey } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { ecdfFigure, histogramFigure, levelsFigure, targetFigure } from "./figures";
import { count, num, pct, signedPct, tone, toneText } from "./format";
import { Table } from "./Report";
import { Loading } from "../../ui/Loading";

type Line = SyntheticRun["report"]["columns"][number];

/** Two overlaid bar rows in a few pixels: real filled grey, synthetic outlined accent. */
function Mini({ real, synthetic }: { real: number[]; synthetic: number[] }) {
  const top = Math.max(...real, ...synthetic, 1e-9);
  const width = 96;
  const height = 22;
  const step = width / Math.max(real.length, 1);
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden className="shrink-0">
      {real.map((value, i) => (
        <rect key={`r${i}`} x={i * step + 0.5} y={height - (value / top) * height} width={step - 1} height={(value / top) * height} fill="var(--ink-2)" opacity={0.45} />
      ))}
      {synthetic.map((value, i) => (
        <rect
          key={`s${i}`}
          x={i * step + 0.5}
          y={height - (value / top) * height}
          width={step - 1}
          height={(value / top) * height}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.25}
        />
      ))}
    </svg>
  );
}

function Counts({ detail }: { detail: SyntheticColumnDetail }) {
  return (
    <Table caption={t("synthetic.counts")} head={["", t("synthetic.real"), "%", t("synthetic.synthetic"), "%"]}>
      {detail.counts.map((c) => (
        <tr key={c.label} className="border-b border-line/60">
          <td className="py-1.5 pr-3 text-ink-2">{t(`synthetic.count.${c.label.toLowerCase()}` as MessageKey)}</td>
          <td className="py-1.5 pr-3 text-right">{count(c.real)}</td>
          <td className="py-1.5 pr-3 text-right text-ink-3">{pct(c.real_share, 1)}</td>
          <td className="py-1.5 pr-3 text-right">{count(c.synthetic)}</td>
          <td className="py-1.5 pr-3 text-right text-ink-3">{pct(c.synthetic_share, 1)}</td>
        </tr>
      ))}
    </Table>
  );
}

function Values({ caption, rows }: { caption: string; rows: SyntheticColumnDetail["frequent_real"] }) {
  return (
    <Table caption={caption} head={[t("synthetic.value"), t("synthetic.real"), t("synthetic.synthetic")]}>
      {rows.map((r) => (
        <tr key={r.value} className="border-b border-line/60">
          <td className="py-1 pr-3">{r.value}</td>
          <td className={`py-1 pr-3 text-right ${r.real === 0 ? "text-ink-3" : ""}`}>{pct(r.real_share, 2)}</td>
          <td className={`py-1 pr-3 text-right ${r.synthetic === 0 ? "text-ink-3" : ""}`}>{pct(r.synthetic_share, 2)}</td>
        </tr>
      ))}
    </Table>
  );
}

function Detail({ run, datasetId, line, target }: { run: SyntheticRun; datasetId: string; line: Line; target: string | null }) {
  const [bins, setBins] = useState<"auto" | "10" | "30">("auto");
  const [view, setView] = useState<"histogram" | "ecdf">("histogram");
  const detail = useSyntheticColumn(datasetId, run.id, line.column, bins === "auto" ? null : Number(bins), target);
  const d = detail.data;
  if (!d) return <Loading label={t("loading.chart")} className="h-48" />;
  const dates = d.kind === "date";
  const categorical = d.kind === "category";
  return (
    <div className={`space-y-6 py-4 ${detail.isFetching ? "opacity-60" : ""}`}>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
        <div className="min-w-0 space-y-2">
          {categorical ? (
            <Plot figure={levelsFigure(d)} height={Math.min(520, 80 + 24 * d.levels.length)} label={t("synthetic.levelsChart", { column: d.column })} />
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <SegmentedControl
                  label={t("synthetic.view")}
                  value={view}
                  onChange={setView}
                  options={[
                    { value: "histogram", label: t("synthetic.histogram") },
                    { value: "ecdf", label: t("synthetic.ecdf") },
                  ]}
                />
                {view === "histogram" && (
                  <SegmentedControl
                    label={t("synthetic.bins")}
                    value={bins}
                    onChange={setBins}
                    options={[
                      { value: "auto", label: t("synthetic.auto") },
                      { value: "10", label: "10" },
                      { value: "30", label: "30" },
                    ]}
                  />
                )}
              </div>
              <Plot
                figure={view === "histogram" ? histogramFigure(d, dates) : ecdfFigure(d, dates)}
                height={260}
                label={t(view === "histogram" ? "synthetic.histogramOf" : "synthetic.ecdfOf", { column: d.column })}
              />
            </>
          )}
          {d.target && (
            <figure className="space-y-1">
              <figcaption className="text-xs text-ink-2">{t("synthetic.targetBy", { measure: d.target.measure, column: d.column })}</figcaption>
              <Plot figure={targetFigure(d.target)} height={220} label={t("synthetic.targetBy", { measure: d.target.measure, column: d.column })} />
            </figure>
          )}
        </div>
        <div className="min-w-0 space-y-6">
          <Counts detail={d} />
          {!categorical && (
            <Table caption={t("synthetic.stats")} head={["", t("synthetic.real"), t("synthetic.synthetic"), "Δ", "Δ %"]}>
              {d.stats.map((s) => (
                <tr key={s.label} className="border-b border-line/60">
                  <td className="py-1 pr-3 text-ink-2">{s.label}</td>
                  <td className="py-1 pr-3 text-right">{dates && s.label !== "Skewness" && s.label !== "Kurtosis" ? dateOrSpan(s.label, s.real) : num(s.real)}</td>
                  <td className="py-1 pr-3 text-right">{dates && s.label !== "Skewness" && s.label !== "Kurtosis" ? dateOrSpan(s.label, s.synthetic) : num(s.synthetic)}</td>
                  <td className="py-1 pr-3 text-right text-ink-2">{dates ? days(s.delta) : num(s.delta)}</td>
                  <td className={`py-1 pr-3 text-right ${Math.abs(s.delta_share ?? 0) > 0.1 ? "text-warn" : "text-ink-3"}`}>{signedPct(s.delta_share)}</td>
                </tr>
              ))}
            </Table>
          )}
        </div>
      </div>
      {categorical ? (
        <Table caption={t("synthetic.levels")} head={[t("synthetic.value"), t("synthetic.real"), t("synthetic.synthetic"), "Δ", ""]}>
          {d.levels.map((r) => (
            <tr key={r.value} className="border-b border-line/60">
              <td className="py-1 pr-3">{r.value}</td>
              <td className="py-1 pr-3 text-right">{pct(r.real_share, 1)}</td>
              <td className="py-1 pr-3 text-right">{pct(r.synthetic_share, 1)}</td>
              <td className={`py-1 pr-3 text-right ${Math.abs(r.delta) > 0.02 ? "text-warn" : "text-ink-3"}`}>{signedPct(r.delta)}</td>
              <td className="py-1 pr-3 text-right text-xs text-bad">{r.status ? t(r.status === "invented" ? "synthetic.invented" : "synthetic.missingInSynthetic") : ""}</td>
            </tr>
          ))}
        </Table>
      ) : (
        <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-4">
          <Values caption={t("synthetic.frequentReal")} rows={d.frequent_real} />
          <Values caption={t("synthetic.frequentSynthetic")} rows={d.frequent_synthetic} />
          <Values caption={t("synthetic.smallest")} rows={d.smallest} />
          <Values caption={t("synthetic.largest")} rows={d.largest} />
        </div>
      )}
    </div>
  );
}

const SPANS = new Set(["Std. deviation", "IQR"]);
const dateOrSpan = (label: string, seconds: number | null) =>
  seconds === null ? "–" : SPANS.has(label) ? days(seconds) : new Date(seconds * 1000).toISOString().slice(0, 10);
const days = (seconds: number | null) => (seconds === null ? "–" : `${(seconds / 86_400).toFixed(1)} d`);

/** Columns from the most damaged to the least; a click opens real and synthetic side by side. */
export function ColumnsTab({ run, datasetId }: { run: SyntheticRun; datasetId: string }) {
  const lines = run.report.columns;
  const [open, setOpen] = useState<string | null>(lines[0]?.column ?? null);
  const [target, setTarget] = useState<string>("");
  const targets = run.report.learned.filter((l) => l.kind !== "date" && (l.kind !== "category" || l.levels.length <= 2 || l.levels.includes("True")));
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-ink-3">{t("synthetic.overlayTarget")}</span>
        <Select aria-label={t("synthetic.overlayTarget")} value={target} onChange={(e) => setTarget(e.target.value)} className="h-7 text-xs">
          <option value="">{t("synthetic.none")}</option>
          {targets.map((l) => (
            <option key={l.name} value={l.name}>
              {l.name}
            </option>
          ))}
        </Select>
        <span className="ml-auto inline-flex items-center gap-3 text-ink-3">
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-2.5 bg-ink-2/45" aria-hidden /> {t("synthetic.real")}
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="inline-block size-2.5 border-[1.5px] border-accent" aria-hidden /> {t("synthetic.synthetic")}
          </span>
        </span>
      </div>
      <ul className="border-t border-line">
        {lines.map((line) => {
          const expanded = open === line.column;
          return (
            <li key={line.column} className="border-b border-line">
              <button
                type="button"
                onClick={() => setOpen(expanded ? null : line.column)}
                aria-expanded={expanded}
                className="grid w-full grid-cols-[16px_minmax(0,1.4fr)_64px_96px_minmax(0,2fr)] items-center gap-3 py-2 text-left text-[13px] hover:bg-surface-1"
              >
                {expanded ? <ChevronDown size={14} className="text-ink-3" aria-hidden /> : <ChevronRight size={14} className="text-ink-3" aria-hidden />}
                <span className="truncate">
                  <span className="font-medium text-ink-1">{line.column}</span>{" "}
                  <span className="text-xs text-ink-3">{t(`type.${line.type}` as MessageKey)}</span>
                </span>
                <span className={`text-right tabular-nums ${toneText[tone(line.score)]}`} title={line.metric === "KS" ? "KSComplement" : "TVComplement"}>
                  {pct(line.score)}
                </span>
                <Mini real={line.real} synthetic={line.synthetic} />
                <span className="truncate text-xs text-ink-2">{line.biggest}</span>
              </button>
              {expanded && <Detail run={run} datasetId={datasetId} line={line} target={target || null} />}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
