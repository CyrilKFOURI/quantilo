import { Check, CircleCheck, CircleMinus, Plus, X } from "lucide-react";
import { useState, type ReactNode } from "react";

import type { MethodResult } from "../../api/client";
import { formatNumber } from "../../charts/figures";
import { pLabel } from "../../features/variable/normality";
import { t, type MessageKey } from "../../lib/i18n";
import { useStoredState } from "../../lib/useStoredState";
import { LITE } from "../../lite/flag";
import { LockBadge } from "../../lite/Locked";
import { Popover } from "../../ui/Popover";

/** Methods of a section (docs/TIMESERIES_METHODS.md), the defaults first. */
export const CATALOGUE = {
  trend: {
    all: [
      "mann_kendall",
      "sen",
      "hamed_rao",
      "seasonal_mk",
      "cox_stuart",
      "spearman_time",
      "linear_trend",
    ],
    defaults: ["mann_kendall", "sen"],
  },
  seasonality: {
    all: [
      "kruskal_season",
      "periodogram",
      "friedman",
      "qs",
      "fisher_g",
      "seasonal_acf",
    ],
    defaults: ["kruskal_season", "periodogram"],
  },
  stationarity: {
    all: [
      "adf",
      "kpss",
      "phillips_perron",
      "df_gls",
      "zivot_andrews",
      "variance_ratio",
      "rur",
    ],
    defaults: ["adf", "kpss", "phillips_perron"],
  },
  volatility: {
    all: [
      "breusch_pagan",
      "arch_lm",
      "white",
      "goldfeld_quandt",
      "mcleod_li",
      "levene",
      "brown_forsythe",
      "bartlett",
      "fligner",
    ],
    defaults: ["breusch_pagan", "arch_lm"],
  },
  breaks: {
    all: ["pettitt", "cusum", "chow", "buishand", "snht"],
    defaults: ["pettitt"],
  },
  forecast: {
    all: [
      "naive",
      "seasonal_naive",
      "ets_auto",
      "arima_auto",
      "drift",
      "mean",
      "ses",
      "holt",
      "holt_winters",
      "theta",
      "sarima",
      "sarimax",
      "tslm",
      "prophet_like",
      "stlf",
      "structural",
      "gbm_lags",
      "mlp_lags",
      "croston",
      "sba",
      "tsb",
      "adida",
      "chronos_bolt_tiny",
      "chronos_bolt_small",
      "chronos_bolt_base",
      "chronos_2",
      "timesfm_2p5",
      "nhits",
      "nbeats",
      "combination",
    ],
    defaults: ["naive", "seasonal_naive", "ets_auto", "arima_auto"],
  },
  smooth: {
    all: [
      "moving_average",
      "ewma",
      "kalman_filtered",
      "kalman_smoothed",
      "lowess",
      "savgol",
      "hp",
    ],
    defaults: ["moving_average"],
  },
  autocorrelation: {
    all: [
      "ljung_box",
      "durbin_watson",
      "box_pierce",
      "breusch_godfrey",
      "runs",
      "turning_points",
      "bartels",
    ],
    defaults: ["ljung_box", "durbin_watson"],
  },
} as const;
export type Section = keyof typeof CATALOGUE;

/** Headings of the "+ Add method" list, for the sections with many kinds of method. `fold`
 *  puts a group under "Advanced" (folded) or "Advanced › More" (folded inside it): the more
 *  niche or slow a method is for a modest gain, the deeper it goes (owner's onion rule, 064). */
type Group = { label: MessageKey; keys: string[]; fold?: "advanced" | "more" };
const GROUPS: Partial<Record<Section, Group[]>> = {
  smooth: [
    { label: "ts.group.simple", keys: ["moving_average"] },
    {
      label: "ts.group.models",
      keys: ["ewma", "kalman_filtered", "kalman_smoothed", "lowess"],
      fold: "advanced",
    },
    { label: "ts.group.filters", keys: ["savgol", "hp"], fold: "more" },
  ],
  forecast: [
    {
      label: "ts.group.baselines",
      keys: ["naive", "seasonal_naive", "drift", "mean"],
    },
    {
      label: "ts.group.statistical",
      keys: [
        "ets_auto",
        "ses",
        "holt",
        "holt_winters",
        "theta",
        "arima_auto",
        "sarima",
        "sarimax",
        "tslm",
        "prophet_like",
        "stlf",
      ],
    },
    { label: "ts.group.ml", keys: ["gbm_lags", "mlp_lags"] },
    {
      label: "ts.group.intermittent",
      keys: ["croston", "sba", "tsb", "adida"],
    },
    { label: "ts.group.combination", keys: ["combination"] },
    // 028 phase B (onion rule): never ticked by default, folded; the trained networks deeper.
    {
      label: "ts.group.pretrained",
      keys: [
        "chronos_bolt_tiny",
        "chronos_bolt_small",
        "chronos_bolt_base",
        "chronos_2",
        "timesfm_2p5",
      ],
      fold: "advanced",
    },
    { label: "ts.group.stateSpace", keys: ["structural"], fold: "more" },
    { label: "ts.group.deep", keys: ["nhits", "nbeats"], fold: "more" },
  ],
};

/** Too slow without native threads: locked in the browser-only build. SARIMAX runs the
 * automatic ARIMA search at every origin, then refits once per driver. */
export const LOCKED = new Set(LITE ? ["arima_auto", "sarima", "sarimax"] : []);
/** The browser version cannot run PyTorch: the AI models (028 phase B) say so instead. */
const LITE_AI: Partial<Record<string, MessageKey>> = LITE
  ? Object.fromEntries(
      ["chronos_bolt_tiny", "chronos_bolt_small", "chronos_bolt_base", "chronos_2", "timesfm_2p5", "nhits", "nbeats"].map(
        (key) => [key, "fcai.state.browser" as MessageKey],
      ),
    )
  : {};
for (const key of Object.keys(LITE_AI)) LOCKED.add(key);

/** Methods whose value is a change per period. */
const SLOPES = new Set(["sen", "linear_trend"]);

const name = (key: string) => t(`tsm.${key}` as MessageKey);

/** The chosen methods of a section, remembered for every series. */
export function useMethods(
  section: Section,
): [string[], (keys: string[]) => void] {
  const [stored, setStored] = useStoredState<string[] | null>(
    `de.ts.methods.${section}`,
    null,
  );
  const known = CATALOGUE[section].all as readonly string[];
  const chosen = (stored ?? [...CATALOGUE[section].defaults]).filter(
    (key) => known.includes(key) && !LOCKED.has(key),
  );
  return [
    chosen.length ? chosen : [...CATALOGUE[section].defaults],
    (keys) => setStored(keys),
  ];
}

/** "Methods  [ADF ✕] [KPSS ✕]  + Add method": the defaults are ticked, the rest can be added. */
export function MethodsRow({
  section,
  chosen,
  onChange,
  badge,
  unavailable = {},
  addLabel,
  label,
  align = "start",
  onPick,
  tag,
}: {
  section: Section;
  chosen: string[];
  onChange: (keys: string[]) => void;
  /** A tag after a chosen method's name (Smooth: causal or not). */
  badge?: (key: string) => ReactNode;
  /** Methods that do not apply here, with the reason (shown instead of their sentence). */
  unavailable?: Record<string, string>;
  /** Accessible name of "+ Add method" when a section has two rows. */
  addLabel?: string;
  /** The row's name (default "Methods"). */
  label?: string;
  /** "end" when "+ Add method" sits far right of its row: the list then opens leftwards. */
  align?: "start" | "end";
  /** Called before ticking a method; true when it handled the click itself (a pretrained model
   *  whose weights must be downloaded first opens its confirmation, 028 B1). */
  onPick?: (key: string) => boolean;
  /** A short tag after a method's name in the list ("Download · 35 MB"). */
  tag?: (key: string) => string | null;
}) {
  const [open, setOpen] = useState(false);
  const all = CATALOGUE[section].all as readonly string[];
  const blocked = (key: string) => LOCKED.has(key) || key in unavailable;
  const toggle = (key: string) => {
    const next = chosen.includes(key)
      ? chosen.filter((k) => k !== key)
      : [...chosen, key];
    if (next.length) onChange(all.filter((k) => next.includes(k)));
  };
  const groups = GROUPS[section];
  const plain = groups?.filter((g) => !g.fold) ?? [];
  const advanced = groups?.filter((g) => g.fold === "advanced") ?? [];
  const more = groups?.filter((g) => g.fold === "more") ?? [];
  const keysOf = (list: Group[]) => list.flatMap((g) => g.keys);
  const item = (key: string, list: Group[]) => {
    const on = chosen.includes(key);
    const group = list.find((g) => g.keys[0] === key);
    const locked = blocked(key);
    return (
      <li key={key}>
        {group && (
          <p className="px-2.5 pt-2 pb-0.5 text-[11px] font-medium tracking-wide text-ink-3 uppercase">
            {t(group.label)}
          </p>
        )}
        <button
          type="button"
          role="menuitemcheckbox"
          aria-checked={on}
          aria-disabled={locked || undefined}
          onClick={() => {
            if (locked) return;
            if (!on && onPick?.(key)) {
              setOpen(false);
              return;
            }
            toggle(key);
          }}
          className={`flex w-full items-start gap-2 rounded-md px-2.5 py-1.5 text-left ${locked ? "cursor-not-allowed opacity-60" : "hover:bg-surface-2"}`}
        >
          <span className="mt-0.5 size-3.5 shrink-0 text-accent">
            {on && <Check size={14} strokeWidth={2} aria-hidden />}
          </span>
          <span className="min-w-0">
            <span className="flex items-center gap-1.5 text-[13px] text-ink-1">
              {name(key)}
              {LOCKED.has(key) && <LockBadge />}
              {badge?.(key)}
              {!locked && tag?.(key) && (
                <span className="rounded-full border border-line px-1.5 text-[11px] text-ink-2">
                  {tag(key)}
                </span>
              )}
            </span>
            <span className="block text-xs text-ink-3">
              {LOCKED.has(key)
                ? t(LITE_AI[key] ?? "lite.methodLocked")
                : (unavailable[key] ?? t(`tsm.${key}.what` as MessageKey))}
            </span>
          </span>
        </button>
      </li>
    );
  };
  const folded = (list: Group[]) => keysOf(list).some((key) => chosen.includes(key));
  return (
    <div
      className="flex flex-wrap items-center gap-1.5 text-xs"
      role="group"
      aria-label={label ?? t("ts.methods")}
    >
      <span className="mr-1 text-ink-3">{label ?? t("ts.methods")}</span>
      {chosen.map((key) => (
        <span
          key={key}
          className="inline-flex h-6 items-center gap-1 rounded-full border border-line bg-surface-1 pr-1 pl-2.5 text-ink-1"
        >
          {name(key)}
          {badge?.(key)}
          {chosen.length > 1 && (
            <button
              type="button"
              aria-label={t("ts.removeMethod", { name: name(key) })}
              onClick={() => toggle(key)}
              className="rounded-full p-0.5 text-ink-3 hover:bg-surface-2 hover:text-ink-1"
            >
              <X size={11} strokeWidth={2} aria-hidden />
            </button>
          )}
        </span>
      ))}
      <Popover
        open={open}
        onOpenChange={setOpen}
        label={t("ts.addMethod")}
        align={align}
        pinBelowLaptop
        trigger={
          <button
            type="button"
            onClick={() => setOpen(!open)}
            aria-expanded={open}
            aria-label={addLabel}
            className="inline-flex h-6 items-center gap-1 rounded-full px-2 text-accent hover:bg-accent-soft"
          >
            <Plus size={12} strokeWidth={2} aria-hidden />
            {t("ts.addMethod")}
          </button>
        }
      >
        <ul className="max-h-80 overflow-y-auto p-1">
          {(groups ? keysOf(plain) : all).map((key) => item(key, plain))}
          {(advanced.length > 0 || more.length > 0) && (
            <li>
              <details open={folded(advanced) || folded(more) || undefined}>
                <summary className="cursor-pointer px-2.5 pt-2 pb-1 text-[11px] font-medium tracking-wide text-ink-2 uppercase hover:text-ink-1">
                  {t("ts.fold.advanced")}
                </summary>
                <ul className="pl-2">
                  {keysOf(advanced).map((key) => item(key, advanced))}
                  {more.length > 0 && (
                    <li>
                      <details open={folded(more) || undefined}>
                        <summary className="cursor-pointer px-2.5 pt-2 pb-1 text-[11px] font-medium tracking-wide text-ink-2 uppercase hover:text-ink-1">
                          {t("ts.fold.more")}
                        </summary>
                        <ul className="pl-2">{keysOf(more).map((key) => item(key, more))}</ul>
                      </details>
                    </li>
                  )}
                </ul>
              </details>
            </li>
          )}
        </ul>
      </Popover>
    </div>
  );
}

/** One line per method: conclusion in words, then the numbers. */
export function MethodsTable({
  results,
  unit,
}: {
  results: MethodResult[];
  unit?: string;
}) {
  if (!results.length) return null;
  return (
    <ul className="divide-y divide-line rounded-md border border-line bg-surface-1">
      {results.map((result) => (
        <MethodLine key={result.key} result={result} unit={unit} />
      ))}
    </ul>
  );
}

function conclusion(result: MethodResult): string {
  if (result.significant === null) return result.note ?? t("ts.notRun");
  return t(
    `tsm.${result.key}.${result.significant ? "yes" : "no"}` as MessageKey,
  );
}

function MethodLine({ result, unit }: { result: MethodResult; unit?: string }) {
  const tone = result.significant ? "text-ok" : "text-ink-2";
  const Icon = result.significant ? CircleCheck : CircleMinus;
  const run = result.significant !== null || result.value !== null;
  return (
    <li className="px-3 py-2">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <span className="font-medium text-ink-1">{name(result.key)}</span>
        {run && (
          <span className="flex flex-wrap gap-x-3 text-xs text-ink-2 tabular-nums">
            {result.p_value !== null && (
              <span className="text-ink-1">{pLabel(result.p_value)}</span>
            )}
            {result.statistic !== null && (
              <span>
                <span className="text-ink-3">{t("ts.statistic")}</span>{" "}
                {formatNumber(result.statistic)}
              </span>
            )}
            {result.value !== null && (
              <span>
                <span className="text-ink-3">
                  {t(`tsm.${result.key}.value` as MessageKey)}
                </span>{" "}
                {formatNumber(result.value)}
                {result.low !== null && result.high !== null && (
                  <span className="text-ink-3">
                    {" "}
                    [{formatNumber(result.low)}, {formatNumber(result.high)}]
                  </span>
                )}
                {unit && SLOPES.has(result.key) ? ` ${unit}` : ""}
              </span>
            )}
            <span>
              <span className="text-ink-3">n</span>{" "}
              {result.n.toLocaleString("en-US")}
            </span>
          </span>
        )}
      </div>
      <p
        className={`mt-0.5 flex items-start gap-1.5 text-[13px] ${run ? tone : "text-ink-3"}`}
      >
        {run && (
          <Icon
            size={14}
            strokeWidth={1.75}
            className="mt-0.5 shrink-0"
            aria-hidden
          />
        )}
        <span>
          {run ? conclusion(result) : (result.note ?? t("ts.notRun"))}
        </span>
      </p>
      <details className="mt-1 text-xs text-ink-2">
        <summary className="cursor-pointer text-ink-3 hover:text-ink-1">
          {t("ts.how")}
        </summary>
        <p className="mt-1 leading-relaxed">
          {t(`tsm.${result.key}.what` as MessageKey)}{" "}
          <span className="text-ink-3">
            {t(`tsm.${result.key}.h0` as MessageKey)}
          </span>
        </p>
        {run && result.note && <p className="mt-1 text-ink-3">{result.note}</p>}
      </details>
    </li>
  );
}
