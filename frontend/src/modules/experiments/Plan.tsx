import { useMemo, useState, type ReactNode } from "react";

import { ApiError } from "../../api/client";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { ErrorNote, Select, TextInput } from "../../ui/fields";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { useBaseline, useExperimentCandidates, useSampleSize, type SampleSizeRequest } from "./api";
import { sampleSizeFigure } from "./figures";
import { number } from "./format";
import { Loading } from "../../ui/Loading";

const small = "h-7 text-xs";

type Kind = "rate" | "mean";
type Draft = {
  kind: Kind;
  column: string | null; // the metric whose current value fills the baseline
  baseline: string | null; // typed by the user (null: the column's value)
  sd: string | null;
  effect: string;
  relative: boolean;
  alpha: string;
  power: string;
  variants: string;
  ratio: string;
  daily: string;
};

const START: Draft = {
  kind: "rate",
  column: null,
  baseline: null,
  sd: null,
  effect: "5",
  relative: true,
  alpha: "0.05",
  power: "0.8",
  variants: "2",
  ratio: "1",
  daily: "",
};

function Labeled({ label, children, hint }: { label: string; children: ReactNode; hint?: string }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-xs text-ink-2">{label}</span>
      {children}
      {hint && <span className="text-[11px] text-ink-3">{hint}</span>}
    </label>
  );
}

const numeric = (text: string | null) => (text === null || text.trim() === "" ? null : Number(text));

/** Before running a test: units per group to detect an effect, the duration, and the curve. */
export function Plan({ datasetId, ready }: { datasetId: string; ready: boolean }) {
  const [draft, setDraft] = useState<Draft>(START);
  const set = (patch: Partial<Draft>) => setDraft((current) => ({ ...current, ...patch }));
  const candidates = useExperimentCandidates(datasetId, ready);
  const baseline = useBaseline(datasetId, draft.column);
  const read = baseline.data && baseline.data.column === draft.column ? baseline.data : null;

  // the column's current value, as the field shows it (a rate in %), unless the user typed one
  const readBaseline = read?.value == null ? null : draft.kind === "rate" ? String(+(read.value * 100).toFixed(2)) : String(+read.value.toPrecision(4));
  const baselineText = draft.baseline ?? readBaseline ?? (draft.kind === "rate" ? "10" : "");
  const sdText = draft.sd ?? (read?.sd == null ? "" : String(+read.sd.toPrecision(4)));

  const request: SampleSizeRequest | null = useMemo(() => {
    const base = numeric(baselineText);
    const effect = numeric(draft.effect);
    if (base === null || effect === null || !Number.isFinite(base) || !Number.isFinite(effect) || effect === 0) return null;
    const sd = numeric(sdText);
    if (draft.kind === "mean" && (sd === null || !(sd > 0))) return null;
    const daily = numeric(draft.daily);
    const rate = draft.kind === "rate";
    return {
      kind: draft.kind,
      baseline: rate ? base / 100 : base,
      sd: rate ? null : sd,
      effect: draft.relative ? effect / 100 : rate ? effect / 100 : effect,
      relative: draft.relative,
      alpha: Number(draft.alpha),
      power: Number(draft.power),
      variants: Number(draft.variants),
      ratio: numeric(draft.ratio) ?? 1,
      daily: daily && daily > 0 ? Math.round(daily) : null,
    };
  }, [baselineText, sdText, draft]);
  const plan = useSampleSize(request);
  const error = plan.error instanceof ApiError ? plan.error : null;
  const figure = useMemo(() => (plan.data ? sampleSizeFigure(plan.data) : null), [plan.data]);
  const columns = (candidates.data?.metrics ?? []).filter((m) => (draft.kind === "rate" ? m.kind === "rate" : m.kind !== "rate"));
  const effectUnit = draft.relative ? t("experiments.plan.liftUnit") : draft.kind === "rate" ? t("experiments.plan.pointsUnit") : "";

  return (
    <div className="space-y-5">
      <p className="max-w-prose text-[13px] text-ink-2">{t("experiments.plan.intro")}</p>
      <div className="flex flex-wrap items-end gap-x-4 gap-y-3 border-y border-line bg-surface-1 px-3 py-3 text-[13px]">
        <Labeled label={t("experiments.plan.metric")}>
          <SegmentedControl
            label={t("experiments.plan.metric")}
            value={draft.kind}
            options={[
              { value: "rate", label: t("experiments.kind.rate") },
              { value: "mean", label: t("experiments.kind.mean") },
            ]}
            onChange={(kind) => set({ kind, column: null, baseline: null, sd: null })}
          />
        </Labeled>
        <Labeled label={t("experiments.plan.readFrom")}>
          <Select
            className={`${small} w-40`}
            value={draft.column ?? ""}
            onChange={(e) => set({ column: e.target.value || null, baseline: null, sd: null })}
          >
            <option value="">{t("experiments.plan.typeIt")}</option>
            {columns.map((m) => (
              <option key={m.column} value={m.column}>
                {m.column}
              </option>
            ))}
          </Select>
        </Labeled>
        <Labeled label={t(draft.kind === "rate" ? "experiments.plan.currentRate" : "experiments.plan.currentMean")}>
          <TextInput type="number" className={`${small} w-24 tabular-nums`} value={baselineText} onChange={(e) => set({ baseline: e.target.value })} />
        </Labeled>
        {draft.kind === "mean" && (
          <Labeled label={t("experiments.plan.sd")}>
            <TextInput type="number" min={0} className={`${small} w-24 tabular-nums`} value={sdText} onChange={(e) => set({ sd: e.target.value })} />
          </Labeled>
        )}
        <Labeled label={t("experiments.plan.detect")}>
          <div className="flex items-center gap-1.5">
            <TextInput
              type="number"
              className={`${small} w-20 tabular-nums`}
              value={draft.effect}
              onChange={(e) => set({ effect: e.target.value })}
              aria-label={t("experiments.plan.effect")}
            />
            <SegmentedControl
              label={t("experiments.plan.effectKind")}
              value={draft.relative ? "relative" : "absolute"}
              options={[
                { value: "relative", label: t("experiments.plan.relative") },
                { value: "absolute", label: t(draft.kind === "rate" ? "experiments.plan.points" : "experiments.plan.absolute") },
              ]}
              onChange={(value) => set({ relative: value === "relative" })}
            />
          </div>
        </Labeled>
        <Labeled label="α">
          <Select aria-label={t("experiments.alpha")} className={`${small} w-20`} value={draft.alpha} onChange={(e) => set({ alpha: e.target.value })}>
            {["0.01", "0.05", "0.1"].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </Select>
        </Labeled>
        <Labeled label={t("experiments.plan.power")}>
          <Select className={`${small} w-24`} value={draft.power} onChange={(e) => set({ power: e.target.value })}>
            {["0.8", "0.9", "0.95"].map((v) => (
              <option key={v} value={v}>
                {Math.round(Number(v) * 100)} %
              </option>
            ))}
          </Select>
        </Labeled>
        <Labeled label={t("experiments.plan.variants")}>
          <Select className={`${small} w-16`} value={draft.variants} onChange={(e) => set({ variants: e.target.value })}>
            {["2", "3", "4", "5"].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </Select>
        </Labeled>
        <Labeled label={t("experiments.plan.ratio")}>
          <TextInput type="number" min={0.05} step={0.1} className={`${small} w-16 tabular-nums`} value={draft.ratio} onChange={(e) => set({ ratio: e.target.value })} />
        </Labeled>
        <Labeled label={t("experiments.plan.daily")}>
          <TextInput
            type="number"
            min={1}
            className={`${small} w-24 tabular-nums`}
            value={draft.daily}
            placeholder={t("experiments.plan.optional")}
            onChange={(e) => set({ daily: e.target.value })}
          />
        </Labeled>
      </div>
      {effectUnit && <p className="-mt-3 text-[11px] text-ink-3">{effectUnit}</p>}

      {error ? (
        <ErrorNote>{error.message}</ErrorNote>
      ) : !request ? (
        <p className="text-[13px] text-ink-2">{t("experiments.plan.incomplete")}</p>
      ) : !plan.data || !figure ? (
        <Loading label={t("loading.plan")} className="h-[300px]" />
      ) : (
        <section aria-labelledby="experiment-plan" className={`space-y-3 ${plan.isFetching ? "opacity-60 transition-opacity" : ""}`}>
          <h2 id="experiment-plan" className="sr-only">
            {t("experiments.plan.result")}
          </h2>
          <dl className="flex flex-wrap gap-x-10 gap-y-3">
            <div>
              <dt className="text-xs text-ink-3">{t(plan.data.ratio === 1 ? "experiments.plan.perGroup" : "experiments.plan.inControl")}</dt>
              <dd className="text-2xl font-semibold tracking-tight text-ink-1 tabular-nums">{plan.data.control.toLocaleString("en-US")}</dd>
            </div>
            {plan.data.ratio !== 1 && (
              <div>
                <dt className="text-xs text-ink-3">{t("experiments.plan.inEachVariant")}</dt>
                <dd className="text-2xl font-semibold tracking-tight text-ink-1 tabular-nums">{plan.data.per_variant.toLocaleString("en-US")}</dd>
              </div>
            )}
            <div>
              <dt className="text-xs text-ink-3">{t("experiments.plan.total")}</dt>
              <dd className="text-2xl font-semibold tracking-tight text-ink-1 tabular-nums">{plan.data.total.toLocaleString("en-US")}</dd>
            </div>
            {plan.data.days !== null && (
              <div>
                <dt className="text-xs text-ink-3">{t("experiments.plan.duration")}</dt>
                <dd className="text-2xl font-semibold tracking-tight text-ink-1 tabular-nums">
                  {t("experiments.plan.days", { days: plan.data.days.toLocaleString("en-US") })}
                </dd>
              </div>
            )}
            <div>
              <dt className="text-xs text-ink-3">{t("experiments.plan.effectSize")}</dt>
              <dd className="text-2xl font-semibold tracking-tight text-ink-2 tabular-nums">{number(plan.data.effect_size)}</dd>
            </div>
          </dl>
          <p className="max-w-prose text-[13px] text-ink-1">{plan.data.sentence}</p>
          <div>
            <h3 className="text-[13px] font-medium text-ink-1">{t("experiments.plan.curveTitle")}</h3>
            <Plot figure={figure} height={300} label={t("experiments.plan.curveLabel")} />
            <p className="mt-1 text-xs text-ink-3">{t("experiments.plan.curveNote")}</p>
          </div>
        </section>
      )}
    </div>
  );
}
