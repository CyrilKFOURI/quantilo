import { Plus, X } from "lucide-react";
import { useState, type ReactNode } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { Checkbox, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { MAX_WINDOWS } from "./Windows";

export type Layers = { values: boolean; mean: boolean; band: boolean; bands?: boolean };
type Align = "trailing" | "centered";

/** Windows worth suggesting for each period (a week, a month, a quarter, a year…). */
const SUGGESTED: Record<string, number[]> = {
  hour: [6, 24, 168],
  day: [7, 14, 30, 90],
  week: [4, 13, 26, 52],
  month: [3, 6, 12],
  quarter: [2, 4, 8],
  year: [2, 3, 5],
};

/** The colour of each mean's curve: the first one is the series' own colour (--chart-1), the
 *  others follow the overview figure (palette index i + 1, i.e. --chart-(i + 2)). */
export function meanColor(index: number): string {
  return index === 0 ? "var(--chart-1)" : `var(--chart-${((index + 1) % 8) + 1})`;
}

/** "7 weeks", "1 month". */
export function span(window: number, period: string): string {
  const unit = t(`time.period.${period}` as MessageKey);
  return `${window} ${unit}${window > 1 ? "s" : ""}`;
}

/** The overview's settings in two named rows: which rolling means (each with the colour of its
 *  curve), where the window sits, and what the chart shows. */
export function RollingControls({
  windows,
  auto,
  period,
  maximum,
  align,
  layers,
  onWindows,
  onAlign,
  onLayers,
  smooth,
  average = true,
  hasBands = false,
}: {
  windows: number[];
  auto: number | null;
  period: string;
  maximum: number;
  align: Align;
  layers: Layers;
  onWindows: (next: number[]) => void;
  onAlign: (next: Align) => void;
  onLayers: (next: Layers) => void;
  /** The Smooth row of methods (064), drawn first. */
  smooth?: ReactNode;
  /** The moving average is ticked: its windows and position are shown. */
  average?: boolean;
  /** Another smoother or Kalman-filled periods have a 95 % band to show or hide. */
  hasBands?: boolean;
}) {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState("");
  const limit = Math.min(365, maximum);
  const shown = windows.length ? windows : auto !== null ? [auto] : [];
  const add = (value: number) => {
    if (!Number.isInteger(value) || value < 2 || value > limit || shown.includes(value)) return;
    const base = windows.length ? windows : auto !== null ? [auto] : [];
    onWindows([...base, value].slice(0, MAX_WINDOWS));
    setDraft("");
    setAdding(false);
  };
  const typed = Number(draft);
  const valid = Number.isInteger(typed) && typed >= 2 && typed <= limit && !shown.includes(typed);
  const suggestions = (SUGGESTED[period] ?? []).filter((w) => w <= limit && !shown.includes(w));
  const full = shown.length >= MAX_WINDOWS;
  const first = shown[0];

  return (
    <div className="space-y-2 text-xs">
      {smooth}
      {average && (
      <>
      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={t("ts.rolling.means")}>
          <span className="w-24 shrink-0 text-ink-3">{t("ts.rolling.means")}</span>
          {shown.map((w, i) => {
            const automatic = windows.length === 0;
            return (
              <span
                key={w}
                className={`inline-flex h-7 items-center gap-1.5 rounded-full border pr-1 pl-2.5 ${automatic ? "border-dashed border-line text-ink-2" : "border-line bg-surface-2 text-ink-1"}`}
              >
                <span className="size-2 rounded-full" style={{ background: meanColor(i) }} aria-hidden />
                {automatic ? t("ts.rolling.auto", { span: span(w, period) }) : span(w, period)}
                {!automatic && (
                  <button
                    type="button"
                    aria-label={t("ts.rolling.remove", { span: span(w, period) })}
                    onClick={() => onWindows(windows.filter((x) => x !== w))}
                    className="inline-flex size-5 items-center justify-center rounded-full text-ink-3 hover:bg-surface-1 hover:text-ink-1"
                  >
                    <X size={11} strokeWidth={2} aria-hidden />
                  </button>
                )}
                {automatic && <span className="pr-1.5" />}
              </span>
            );
          })}
          {!full &&
            (adding ? (
              <span className="inline-flex flex-wrap items-center gap-1 rounded-full border border-accent/50 py-0.5 pr-1 pl-2">
                {suggestions.map((w) => (
                  <button
                    key={w}
                    type="button"
                    onClick={() => add(w)}
                    className="rounded-full px-2 py-0.5 text-ink-1 hover:bg-accent-soft"
                  >
                    {span(w, period)}
                  </button>
                ))}
                <TextInput
                  autoFocus
                  type="number"
                  min={2}
                  max={limit}
                  value={draft}
                  placeholder={t("ts.rolling.other")}
                  aria-label={t("ts.rolling.input", { period: t(`time.period.${period}` as MessageKey) })}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") add(typed);
                    if (e.key === "Escape") setAdding(false);
                  }}
                  className="h-6 w-16 text-xs"
                />
                <button
                  type="button"
                  disabled={!valid}
                  onClick={() => add(typed)}
                  aria-label={t("ts.rolling.confirm")}
                  className="inline-flex size-6 items-center justify-center rounded-full text-accent hover:bg-accent-soft disabled:text-ink-3 disabled:hover:bg-transparent"
                >
                  <Plus size={12} strokeWidth={2} aria-hidden />
                </button>
                <button
                  type="button"
                  onClick={() => setAdding(false)}
                  aria-label={t("ts.rolling.close")}
                  className="inline-flex size-6 items-center justify-center rounded-full text-ink-3 hover:bg-surface-2 hover:text-ink-1"
                >
                  <X size={12} strokeWidth={2} aria-hidden />
                </button>
              </span>
            ) : (
              <button
                type="button"
                onClick={() => setAdding(true)}
                className="inline-flex h-7 items-center gap-1 rounded-full px-2.5 text-accent hover:bg-accent-soft"
              >
                <Plus size={12} strokeWidth={2} aria-hidden />
                {t("ts.rolling.add")}
              </button>
            ))}
      </div>
      <div className="flex items-center gap-1.5">
        <span className="w-24 shrink-0 text-ink-3">{t("ts.rolling.position")}</span>
        <SegmentedControl
            label={t("time.align")}
            value={align}
            onChange={onAlign}
            options={[
              { value: "trailing", label: t("time.align.trailing") },
            { value: "centered", label: t("time.align.centered") },
          ]}
        />
      </div>
      </>
      )}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5" role="group" aria-label={t("ts.rolling.show")}>
        <span className="w-24 shrink-0 text-ink-3">{t("ts.rolling.show")}</span>
        <Checkbox label={t("time.layer.values")} checked={layers.values} onChange={(e) => onLayers({ ...layers, values: e.target.checked })} />
        {average && (
          <Checkbox label={t("ts.rolling.layerMeans")} checked={layers.mean} onChange={(e) => onLayers({ ...layers, mean: e.target.checked })} />
        )}
        {average && (
          <Checkbox
            label={first ? t("ts.rolling.layerBand", { span: span(first, period) }) : t("time.layer.band")}
            checked={layers.band}
            onChange={(e) => onLayers({ ...layers, band: e.target.checked })}
          />
        )}
        {hasBands && (
          <Checkbox
            label={t("ts.smooth.layerBands")}
            checked={layers.bands ?? true}
            onChange={(e) => onLayers({ ...layers, bands: e.target.checked })}
          />
        )}
      </div>
    </div>
  );
}
