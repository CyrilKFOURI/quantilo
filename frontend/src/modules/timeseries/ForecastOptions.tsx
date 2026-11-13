import { SlidersHorizontal } from "lucide-react";
import { useState } from "react";

import { t, type MessageKey } from "../../lib/i18n";
import { Select } from "../../ui/fields";
import { Popover } from "../../ui/Popover";
import { Cell } from "../kpi/Customize";
import {
  FORECAST_DEFAULTS,
  FORECAST_OPTION_MODELS,
  FORECAST_OPTIONS,
  type ForecastOption,
  type TimeSeriesSearch,
} from "./search";

/** How a value of an option reads. */
function shown(option: ForecastOption, value: number | string | null): string {
  if (value === null) return t("ts.fopt.limit.none");
  if (option === "arima_ic") return t(`ts.fopt.ic.${value}` as MessageKey);
  if (option === "gbm_grid") return t(`ts.fopt.grid.${value}` as MessageKey);
  if (option === "fc_limit") return t("ts.fopt.seconds", { value: String(value) });
  return typeof value === "number"
    ? value.toLocaleString("en-US", { maximumFractionDigits: 4 })
    : value;
}

/** The options worth showing: those of the ticked models, and the time limit. */
export function forecastOptionsFor(models: string[]): ForecastOption[] {
  return (Object.keys(FORECAST_OPTIONS) as ForecastOption[]).filter((option) => {
    const tuned = FORECAST_OPTION_MODELS[option];
    return tuned === null || tuned.some((key) => models.includes(key));
  });
}

/** "⚙ Options" of the forecast (028 A7, onion rule: closed, nothing changed by default): ARIMA
 * bounds and criterion, gradient-boosting grid, Prophet-like changepoints, time limit. Each
 * cell gives its default and why; a changed one has a dot and a reset. Kept in the URL. */
export function ForecastOptions({
  models,
  search,
  onChange,
}: {
  models: string[];
  search: TimeSeriesSearch;
  onChange: (patch: Partial<TimeSeriesSearch>) => void;
}) {
  const [open, setOpen] = useState(false);
  const options = forecastOptionsFor(models);
  const changed = options.filter((option) => search[option] !== undefined).length;
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      label={t("ts.fopt.title")}
      align="start"
      pinBelowLaptop
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="inline-flex h-6 items-center gap-1 rounded-full px-2 text-xs text-ink-2 hover:bg-surface-2 hover:text-ink-1"
        >
          <SlidersHorizontal size={12} strokeWidth={2} aria-hidden />
          {t("ts.fopt.title")}
          {changed > 0 && <span className="text-ink-3 tabular-nums">· {changed}</span>}
        </button>
      }
    >
      <div className="max-w-xl space-y-2 p-3">
        <p className="text-xs text-ink-3">{t("ts.fopt.explain")}</p>
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          {options.map((option) => {
            const forced = search[option];
            const label = t(`ts.fopt.param.${option}` as MessageKey);
            return (
              <Cell
                key={option}
                label={label}
                why={t(`ts.fopt.why.${option}` as MessageKey)}
                changed={forced !== undefined}
                onReset={() => onChange({ [option]: undefined })}
              >
                <Select
                  aria-label={label}
                  value={forced === undefined ? "" : String(forced)}
                  onChange={(event) => {
                    const raw = event.target.value;
                    const value =
                      raw === ""
                        ? undefined
                        : option === "arima_ic" || option === "gbm_grid"
                          ? raw
                          : Number(raw);
                    onChange({ [option]: value });
                  }}
                  className="h-8 w-full min-w-0 text-[13px]"
                >
                  <option value="">
                    {t("ts.fopt.default", {
                      value: shown(option, FORECAST_DEFAULTS[option]),
                    })}
                  </option>
                  {FORECAST_OPTIONS[option]
                    .filter((value) => value !== FORECAST_DEFAULTS[option])
                    .map((value) => (
                      <option key={value} value={String(value)}>
                        {shown(option, value)}
                      </option>
                    ))}
                </Select>
              </Cell>
            );
          })}
        </div>
      </div>
    </Popover>
  );
}
