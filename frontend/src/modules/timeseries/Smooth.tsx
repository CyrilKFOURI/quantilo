import { Plus, SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import type { SeriesSmoothing, SmoothCurve } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Popover } from "../../ui/Popover";
import { Cell } from "../kpi/Customize";
import { MethodsRow } from "./Methods";
import { SMOOTH_OPTIONS, type SmoothOption, type TimeSeriesSearch } from "./search";

/** Methods that use only data up to t (R-SM-02); the moving average depends on its position. */
const CAUSAL = new Set(["ewma", "kalman_filtered"]);

export function isCausal(key: string, align: "trailing" | "centered"): boolean {
  return key === "moving_average" ? align === "trailing" : CAUSAL.has(key);
}

/** "Causal" or "Uses future values", after a smoother's name. */
export function CausalTag({ causal }: { causal: boolean }) {
  return (
    <span
      title={t(causal ? "ts.smooth.causalHint" : "ts.smooth.futureHint")}
      className={`rounded-full px-1.5 text-[10px] leading-4 ${causal ? "bg-accent-soft text-ink-1" : "bg-surface-2 text-ink-2"}`}
    >
      {t(causal ? "ts.smooth.causal" : "ts.smooth.future")}
    </span>
  );
}

/** "Advanced › More" smoothers: never suggested on the first plane (owner's onion rule). */
const DEEP = new Set(["savgol", "hp"]);

/** Which overridable parameters each smoother has, and the curve parameter they set. */
const OPTIONS_OF: Record<string, { option: SmoothOption; param: string }[]> = {
  ewma: [{ option: "ewma_alpha", param: "alpha" }],
  lowess: [{ option: "lowess_frac", param: "frac" }],
  savgol: [
    { option: "savgol_window", param: "window" },
    { option: "savgol_order", param: "order" },
  ],
  kalman_filtered: [{ option: "kalman_model", param: "model" }],
  kalman_smoothed: [{ option: "kalman_model", param: "model" }],
  hp: [{ option: "hp_lambda", param: "lambda" }],
};

const smootherName = (key: string) => t(`tsm.${key}` as MessageKey);

function shown(option: SmoothOption, value: string | number): string {
  if (option === "kalman_model") return t(`ts.kalman.${value}` as MessageKey);
  return typeof value === "number" ? value.toLocaleString("en-US", { maximumFractionDigits: 4 }) : value;
}

/** The parameters of the ticked smoothers (064, R-SM-09): Auto's value and why, a native list. */
function SmoothOptions({
  chosen,
  curves,
  search,
  onChange,
}: {
  chosen: string[];
  curves: SmoothCurve[];
  search: TimeSeriesSearch;
  onChange: (patch: Partial<TimeSeriesSearch>) => void;
}) {
  const [open, setOpen] = useState(false);
  const seen = new Set<SmoothOption>();
  const cells = chosen.flatMap((key) =>
    (OPTIONS_OF[key] ?? []).flatMap(({ option, param }) => {
      if (seen.has(option)) return [];
      seen.add(option);
      const found = curves.find((c) => c.key === key)?.params.find((p) => p.key === param);
      return [{ key, option, found }];
    }),
  );
  if (!cells.length) return null;
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("ts.smooth.options")}
      align="start"
      pinBelowLaptop
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="inline-flex h-6 items-center gap-1 rounded-full px-2 text-ink-2 hover:bg-surface-2 hover:text-ink-1"
        >
          <SlidersHorizontal size={12} strokeWidth={2} aria-hidden />
          {t("ts.smooth.options")}
        </button>
      }
    >
      <div className="grid grid-cols-1 gap-3 p-3 sm:grid-cols-3">
        {cells.map(({ key, option, found }) => {
          const forced = search[option];
          const label = t(`ts.smooth.param.${option}` as MessageKey);
          return (
            <Cell
              key={option}
              label={label}
              why={found ? `${smootherName(key)}: ${found.reason}` : null}
              changed={forced !== undefined}
              onReset={() => onChange({ [option]: undefined })}
            >
              <Select
                aria-label={label}
                value={forced === undefined ? "" : String(forced)}
                onChange={(event) =>
                  onChange({
                    [option]:
                      event.target.value === ""
                        ? undefined
                        : option === "kalman_model"
                          ? event.target.value
                          : Number(event.target.value),
                  })
                }
                className="h-8 w-full min-w-0 text-[13px]"
              >
                <option value="">
                  {found?.auto
                    ? t("time.auto", { value: shown(option, found.value) })
                    : t("ts.smooth.auto")}
                </option>
                {SMOOTH_OPTIONS[option].map((value) => (
                  <option key={value} value={String(value)}>
                    {shown(option, value)}
                  </option>
                ))}
              </Select>
            </Cell>
          );
        })}
      </div>
    </Popover>
  );
}

/** The Smooth row (064): methods with their causal tag, options, suggestions and what each chose. */
export function SmoothRow({
  chosen,
  onChosen,
  data,
  failed,
  align,
  search,
  onChange,
}: {
  chosen: string[];
  onChosen: (keys: string[]) => void;
  data: SeriesSmoothing | undefined;
  failed: string | null;
  align: "trailing" | "centered";
  search: TimeSeriesSearch;
  onChange: (patch: Partial<TimeSeriesSearch>) => void;
}) {
  const curves = data?.curves ?? [];
  const waiting = chosen.some((key) => key !== "moving_average" && !curves.some((c) => c.key === key));
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <MethodsRow
          section="smooth"
          label={t("ts.smooth")}
          addLabel={t("ts.smooth.add")}
          align="end"
          chosen={chosen}
          onChange={onChosen}
          unavailable={data?.unavailable}
          badge={(key) => <CausalTag causal={isCausal(key, align)} />}
        />
        <SmoothOptions chosen={chosen} curves={curves} search={search} onChange={onChange} />
      </div>
      {failed && <p className="text-bad">{failed}</p>}
      {waiting && !failed && (
        <Loading compact label={t("ts.smooth.computing")} className="h-9 rounded-md" />
      )}
      {(data?.suggestions ?? []).filter((s) => !DEEP.has(s.key)).map((suggestion) => (
        <p key={suggestion.key} className="flex flex-wrap items-center gap-x-2 text-ink-2">
          <span>
            {t("ts.smooth.suggested", { name: smootherName(suggestion.key) })}{" "}
            <span className="text-ink-3">{suggestion.reason}</span>
          </span>
          <button
            type="button"
            onClick={() => onChosen([...chosen, suggestion.key])}
            aria-label={t("ts.smooth.addSuggested", { name: smootherName(suggestion.key) })}
            className="inline-flex h-6 items-center gap-1 rounded-full px-2 text-accent hover:bg-accent-soft"
          >
            <Plus size={12} strokeWidth={2} aria-hidden />
            {t("ts.smooth.add.short")}
          </button>
        </p>
      ))}
      {curves
        .filter((curve) => curve.key !== "moving_average")
        .map((curve) => (
          <p key={curve.key} className="text-ink-3">
            <span className="text-ink-2">{smootherName(curve.key)}:</span>{" "}
            {curve.error ?? curve.info}
            {curve.notes.map((note) => (
              <span key={note} className="block">
                {note}
              </span>
            ))}
          </p>
        ))}
    </div>
  );
}
