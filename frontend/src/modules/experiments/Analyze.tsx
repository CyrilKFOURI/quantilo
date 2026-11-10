import { ChevronDown, Plus, SlidersHorizontal, TriangleAlert } from "lucide-react";
import { useState, type ReactNode } from "react";

import { ApiError } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { ErrorNote, Field, Select, TextInput } from "../../ui/fields";
import { ParamPanel } from "../../ui/ParamPanel";
import { Popover } from "../../ui/Popover";
import {
  useExperiment,
  useExperimentCandidates,
  type ExperimentCandidates,
  type ExperimentDesign,
  type ExperimentMetric,
} from "./api";
import { PolicySection } from "./Policy";
import { BanditSection, CumulativeSection, MetricSection, SegmentsSection } from "./Results";
import type { ExperimentsSearch } from "./search";
import { Loading } from "../../ui/Loading";

const small = "h-7 text-xs";
const KINDS = ["auto", "rate", "mean", "count", "ratio"] as const;
const ALPHAS = [0.01, 0.05, 0.1];

/** One setting. On a phone (046) the settings sit three per row, label above: `wide` takes the
 *  whole row (the metrics and their chips, the planned shares). */
function Group({ label, wide = false, children }: { label: string; wide?: boolean; children: ReactNode }) {
  return (
    <div
      className={`flex flex-wrap items-center gap-1.5 max-sm:min-w-0 max-sm:gap-1 max-sm:[&_select]:w-full max-sm:[&_select]:max-w-none ${wide ? "max-sm:col-span-3" : "max-sm:flex-col max-sm:items-stretch"}`}
      role="group"
      aria-label={label}
    >
      <span className={`text-ink-3 max-sm:text-[11px] ${wide ? "max-sm:basis-full" : ""}`}>{label}</span>
      {children}
    </div>
  );
}

/** A row of settings: wrapped on a laptop, three per row on a phone (046). */
const ROW = "flex flex-wrap items-center gap-x-3 gap-y-2 max-sm:grid max-sm:grid-cols-3 max-sm:items-end";

const Divider = () => <span className="hidden h-5 w-px bg-line sm:block" aria-hidden />;

/** The experiment shown: the URL's, else a seed from another page, else the suggested one. */
export function designOf(search: ExperimentsSearch, candidates: ExperimentCandidates | undefined): ExperimentDesign | null {
  if (search.design) return search.design;
  if (search.variant && search.metric)
    return { variant: search.variant, metrics: [{ column: search.metric }], unit: search.unit ?? null };
  const suggested = candidates?.suggested;
  if (!suggested) return null;
  return { variant: suggested.variant, metrics: suggested.metrics.map((m) => ({ column: m.column })), unit: suggested.unit };
}

const metricLabel = (m: ExperimentMetric) => (m.denominator ? `${m.column} ÷ ${m.denominator}` : m.column);

/** A metric of the bar: its name, and a menu to make it primary, a ratio, CUPED-adjusted, or remove it. */
function MetricChip({
  metric,
  primary,
  numbers,
  onChange,
  onPrimary,
  onRemove,
}: {
  metric: ExperimentMetric;
  primary: boolean;
  numbers: string[];
  onChange: (next: ExperimentMetric) => void;
  onPrimary: () => void;
  onRemove: (() => void) | null;
}) {
  const [open, setOpen] = useState(false);
  const others = numbers.filter((n) => n !== metric.column);
  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      align="start"
      label={t("experiments.metricMenu", { metric: metricLabel(metric) })}
      trigger={
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className={`inline-flex h-7 items-center gap-1 rounded-md border px-2 font-mono text-xs ${
            primary ? "border-accent text-ink-1" : "border-line text-ink-2 hover:text-ink-1"
          }`}
        >
          {metricLabel(metric)}
          {metric.pre && <span className="font-sans text-[11px] text-ink-3">CUPED</span>}
          {primary && <span className="font-sans text-[11px] text-accent">{t("experiments.primary")}</span>}
          <ChevronDown size={12} aria-hidden />
        </button>
      }
    >
      <div className="grid gap-3 p-3 text-[13px] sm:grid-cols-2">
        <Field label={t("experiments.readAs")}>
          <Select value={metric.kind ?? "auto"} onChange={(e) => onChange({ ...metric, kind: e.target.value as ExperimentMetric["kind"] })}>
            {KINDS.map((kind) => (
              <option key={kind} value={kind} disabled={kind === "ratio" && !metric.denominator}>
                {t(`experiments.kind.${kind}` as MessageKey)}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("experiments.ratioOf")}>
          <Select
            value={metric.denominator ?? ""}
            onChange={(e) => onChange({ ...metric, denominator: e.target.value || null, kind: e.target.value ? "ratio" : "auto" })}
          >
            <option value="">{t("experiments.none")}</option>
            {others.map((name) => (
              <option key={name} value={name}>
                ÷ {name}
              </option>
            ))}
          </Select>
        </Field>
        <Field label={t("experiments.cupedWith")}>
          <Select value={metric.pre ?? ""} onChange={(e) => onChange({ ...metric, pre: e.target.value || null })} disabled={Boolean(metric.denominator)}>
            <option value="">{t("experiments.none")}</option>
            {others.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Field>
        <div className="flex items-end gap-2">
          {!primary && (
            <Button
              className={small}
              onClick={() => {
                setOpen(false);
                onPrimary();
              }}
            >
              {t("experiments.makePrimary")}
            </Button>
          )}
          {onRemove && (
            <Button
              variant="ghost"
              className={small}
              onClick={() => {
                setOpen(false);
                onRemove();
              }}
            >
              {t("experiments.remove")}
            </Button>
          )}
        </div>
        <p className="text-xs text-ink-3 sm:col-span-2">{t("experiments.cupedHint")}</p>
      </div>
    </Popover>
  );
}

function SetupBar({
  design,
  candidates,
  levels,
  onChange,
}: {
  design: ExperimentDesign;
  candidates: ExperimentCandidates;
  levels: string[];
  onChange: (next: ExperimentDesign) => void;
}) {
  const phone = useMedia(PHONE);
  const [more, setMore] = useState(Boolean(design.segment || design.time || design.expected));
  const numbers = candidates.metrics.map((m) => m.column);
  const used = new Set(design.metrics.map((m) => m.column));
  const primary = design.primary ?? design.metrics[0]?.column;
  const variantNames = candidates.variants.map((v) => v.column);
  const variants = variantNames.includes(design.variant) ? variantNames : [design.variant, ...variantNames];
  const units = candidates.units.map((u) => u.column);
  const setMetric = (index: number, next: ExperimentMetric) =>
    onChange({ ...design, metrics: design.metrics.map((m, i) => (i === index ? next : m)) });
  const share = (level: string) => design.expected?.[level];
  const fields = (
    <>
    <div className={ROW}>
      <Group label={t("experiments.variant")}>
        <Select
          aria-label={t("experiments.variant")}
          className={`${small} max-w-48`}
          value={design.variant}
          onChange={(e) => onChange({ ...design, variant: e.target.value, control: null, expected: null })}
        >
          {variants.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </Select>
      </Group>
      <Group label={t("experiments.control")}>
        <Select
          aria-label={t("experiments.control")}
          className={`${small} max-w-40`}
          value={design.control ?? ""}
          onChange={(e) => onChange({ ...design, control: e.target.value || null })}
        >
          <option value="">{t("experiments.guessed")}</option>
          {levels.map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </Select>
      </Group>
      <Divider />
      <Group label={t("experiments.metrics")} wide>
        {design.metrics.map((metric, i) => (
          <MetricChip
            key={`${metric.column}-${metric.denominator ?? ""}-${i}`}
            metric={metric}
            primary={metric.column === primary}
            numbers={numbers}
            onChange={(next) => setMetric(i, next)}
            onPrimary={() => onChange({ ...design, primary: metric.column })}
            onRemove={
              design.metrics.length > 1
                ? () =>
                    onChange({
                      ...design,
                      metrics: design.metrics.filter((_, j) => j !== i),
                      primary: design.primary === metric.column ? null : design.primary,
                    })
                : null
            }
          />
        ))}
        {design.metrics.length < 10 && (
          <label className="relative inline-flex items-center">
            <span className="sr-only">{t("experiments.addMetric")}</span>
            <Plus size={12} className="pointer-events-none absolute left-2 text-ink-3" aria-hidden />
            <Select
              className={`${small} w-32 pl-6 text-ink-2`}
              value=""
              onChange={(e) => e.target.value && onChange({ ...design, metrics: [...design.metrics, { column: e.target.value }] })}
            >
              <option value="">{t("experiments.addMetric")}</option>
              {candidates.metrics
                .filter((m) => !used.has(m.column) && m.column !== design.variant)
                .map((m) => (
                  <option key={m.column} value={m.column}>
                    {m.column} · {t(`experiments.kind.${m.kind}` as MessageKey)}
                  </option>
                ))}
            </Select>
          </label>
        )}
      </Group>
      <Divider />
      <Group label={t("experiments.unit")}>
        <Select
          aria-label={t("experiments.unit")}
          className={`${small} max-w-40`}
          value={design.unit ?? ""}
          onChange={(e) => onChange({ ...design, unit: e.target.value || null })}
        >
          <option value="">{t("experiments.eachRow")}</option>
          {[...new Set([...(design.unit ? [design.unit] : []), ...units])].map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </Select>
      </Group>
      <Button variant={more ? "secondary" : "ghost"} icon={SlidersHorizontal} className={`${small} ml-auto max-sm:ml-0 max-sm:self-end`} onClick={() => setMore(!more)} aria-expanded={more}>
        {t("experiments.more")}
      </Button>
    </div>
    {more && (
      <div className={`${ROW} border-t border-line pt-2`}>
        <Group label={t("experiments.planned")} wide>
          {levels.map((level) => (
            <label key={`${design.variant}-${level}`} className="inline-flex items-center gap-1 text-xs text-ink-2">
              {level}
              <TextInput
                type="number"
                min={1}
                className={`${small} w-16 tabular-nums`}
                aria-label={t("experiments.plannedShare", { level })}
                placeholder={String(Math.round(100 / Math.max(levels.length, 1)))}
                defaultValue={share(level) ?? ""}
                onBlur={(e) => {
                  const entered = Number(e.target.value);
                  const expected = { ...(design.expected ?? {}) };
                  if (e.target.value === "" || !Number.isFinite(entered) || entered <= 0) delete expected[level];
                  else expected[level] = entered;
                  const complete = levels.every((l) => expected[l] !== undefined);
                  onChange({ ...design, expected: complete ? expected : null });
                }}
              />
            </label>
          ))}
        </Group>
        <Divider />
        <Group label={t("experiments.subgroups")}>
          <Select
            aria-label={t("experiments.subgroups")}
            className={`${small} max-w-40`}
            value={design.segment ?? ""}
            onChange={(e) => onChange({ ...design, segment: e.target.value || null })}
          >
            <option value="">{t("experiments.none")}</option>
            {candidates.segments
              .filter((s) => s !== design.variant)
              .map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
          </Select>
        </Group>
        <Group label={t("experiments.overTime")}>
          <Select
            aria-label={t("experiments.overTime")}
            className={`${small} max-w-40`}
            value={design.time ?? ""}
            onChange={(e) => onChange({ ...design, time: e.target.value || null })}
            disabled={candidates.times.length === 0}
          >
            <option value="">{t("experiments.none")}</option>
            {candidates.times.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Group>
        <Group label="α">
          <Select
            aria-label={t("experiments.alpha")}
            className={`${small} w-20`}
            value={String(design.alpha ?? 0.05)}
            onChange={(e) => onChange({ ...design, alpha: Number(e.target.value) })}
          >
            {ALPHAS.map((alpha) => (
              <option key={alpha} value={String(alpha)}>
                {alpha}
              </option>
            ))}
          </Select>
        </Group>
      </div>
    )}
    </>
  );
  if (phone) {
    // 041: on a phone the setup folds into one line ("Ship Mode · Sales · Customer ID").
    return (
      <ParamPanel
        label={t("experiments.setup")}
        layout="stack"
        summary={[design.variant, design.metrics.map((m) => m.column).join(", "), design.unit]}
      >
        <div className="space-y-2 text-[13px]">{fields}</div>
      </ParamPanel>
    );
  }
  return <div className="space-y-2 border-y border-line bg-surface-1 px-3 py-2.5 text-[13px]">{fields}</div>;
}

/** No experiment to open: why (R-EXP-01), and the observed categories one may compare anyway. */
function NoVariant({ candidates, onDesign }: { candidates: ExperimentCandidates | undefined; onDesign: (design: ExperimentDesign) => void }) {
  const metric = candidates?.metrics[0]?.column;
  const observed = candidates?.note && metric ? candidates.variants.filter((v) => v.column !== metric) : [];
  if (!candidates?.note || !metric || observed.length === 0)
    return <p className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-ink-2">{t("experiments.noCandidates")}</p>;
  return (
    <section className="space-y-3 rounded-lg border border-dashed border-line px-4 py-6 text-[13px]" aria-label={t("experiments.observational")}>
      <p className="text-ink-1">{candidates.note}</p>
      <ul className="space-y-2">
        {observed.map((variant) => (
          <li key={variant.column} className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <Button onClick={() => onDesign({ variant: variant.column, metrics: [{ column: metric }], unit: null })}>
              {t("experiments.compareAnyway", { column: variant.column })}
            </Button>
            <span className="min-w-0 text-xs text-ink-3">{variant.why}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function Analyze({
  datasetId,
  ready,
  search,
  onDesign,
}: {
  datasetId: string;
  ready: boolean;
  search: ExperimentsSearch;
  onDesign: (design: ExperimentDesign) => void;
}) {
  const candidates = useExperimentCandidates(datasetId, ready);
  const design = designOf(search, candidates.data);
  const experiment = useExperiment(datasetId, design);
  const result = experiment.data;
  const error = experiment.error instanceof ApiError ? experiment.error : null;

  if (candidates.isPending) return <Loading label={t("loading.columns")} className="h-[420px]" />;
  if (candidates.isError) return <ErrorNote>{candidates.error.message}</ErrorNote>;
  if (!design || !candidates.data) return <NoVariant candidates={candidates.data} onDesign={onDesign} />;

  const fromCandidates = candidates.data.variants.find((v) => v.column === design.variant)?.levels;
  const levels = fromCandidates ?? result?.variants.map((v) => v.label) ?? [];
  const sameDesign = result && result.design.variant === design.variant;

  return (
    <div className="space-y-4">
      <SetupBar design={design} candidates={candidates.data} levels={levels} onChange={onDesign} />
      {error ? (
        <ErrorNote>{error.message}</ErrorNote>
      ) : !result || !sameDesign ? (
        <Loading label={t("loading.results")} className="h-[420px]" />
      ) : (
        <div className={`space-y-4 ${experiment.isFetching ? "opacity-60 transition-opacity" : ""}`} aria-busy={experiment.isFetching}>
          {result.srm.alarm ? (
            <p role="alert" className="flex items-start gap-2 rounded-md border border-bad/30 bg-bad/8 px-3 py-2 text-[13px] font-medium text-bad">
              <TriangleAlert size={15} strokeWidth={1.75} className="mt-0.5 shrink-0" aria-hidden />
              <span>
                {result.srm.message} <span className="font-normal">(χ² p = {result.srm.p_value === null ? "—" : result.srm.p_value.toExponential(1)})</span>
              </span>
            </p>
          ) : null}
          {result.warnings.map((warning) => (
            <div key={warning} className="flex flex-wrap items-center gap-x-3 gap-y-1 border-l-2 border-accent pl-3 text-xs text-ink-2">
              <span>{warning}</span>
              {result.suggested_unit && warning.startsWith(result.suggested_unit) && (
                <Button className={small} onClick={() => onDesign({ ...design, unit: result.suggested_unit })}>
                  {t("experiments.useUnit", { unit: result.suggested_unit })}
                </Button>
              )}
            </div>
          ))}
          <p className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3 tabular-nums">
            <span>
              {t(result.units === result.rows ? "experiments.rows" : "experiments.unitsOfRows", {
                units: result.units.toLocaleString("en-US"),
                rows: result.rows.toLocaleString("en-US"),
                unit: result.design.unit ?? "",
              })}
            </span>
            <span>
              {result.variants
                .map((v) => `${v.label} ${v.units.toLocaleString("en-US")} (${(v.share * 100).toFixed(1)} %)`)
                .join(" · ")}
            </span>
            {!result.srm.alarm && <span>{result.srm.message}</span>}
            <span className="text-ink-2">{result.correction}</span>
          </p>
          {result.metrics.map((metric, i) => (
            <MetricSection key={`${metric.label}-${i}`} metric={metric} result={result} />
          ))}
          {result.bandit && <BanditSection bandit={result.bandit} />}
          {result.segments && <SegmentsSection segments={result.segments} />}
          {result.policy_offer && (
            <PolicySection
              datasetId={datasetId}
              design={design}
              offer={result.policy_offer}
              metric={result.metrics.find((m) => m.primary)?.label ?? ""}
            />
          )}
          {result.cumulative && <CumulativeSection cumulative={result.cumulative} />}
        </div>
      )}
    </div>
  );
}
