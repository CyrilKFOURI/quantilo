import { CircleCheck, CircleHelp } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { ApiError } from "../../api/client";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { usePolicy, type ExperimentDesign, type PolicyLearning, type PolicyOffer } from "./api";
import { policyValuesFigure, qiniFigure } from "./figures";
import { value } from "./format";

/** A column of the units, on or off for the rule (all on by default). */
export function CovariateChips({
  columns,
  excluded,
  onChange,
  label,
}: {
  columns: string[];
  excluded: string[];
  onChange: (next: string[]) => void;
  label: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label={label}>
      <span className="text-xs text-ink-3">{label}</span>
      {columns.map((name) => {
        const on = !excluded.includes(name);
        return (
          <button
            key={name}
            type="button"
            aria-pressed={on}
            onClick={() => onChange(on ? [...excluded, name] : excluded.filter((n) => n !== name))}
            className={`h-7 rounded-md border px-2 font-mono text-xs ${on ? "border-accent bg-accent-soft text-ink-1" : "border-line text-ink-3 line-through hover:text-ink-1"}`}
          >
            {name}
          </button>
        );
      })}
    </div>
  );
}

function Verdict({ verdict }: { verdict: PolicyLearning["verdict"] }) {
  if (verdict === "worth_testing")
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap text-ok">
        <CircleCheck size={13} strokeWidth={1.75} aria-hidden />
        {t("experiments.policy.worth")}
      </span>
    );
  return (
    <span className="inline-flex items-center gap-1 text-xs font-medium whitespace-nowrap text-ink-2">
      <CircleHelp size={13} strokeWidth={1.75} aria-hidden />
      {t("experiments.policy.noBetter")}
    </span>
  );
}

/** "Advanced › More": the Qini curves, requested only once this section is opened (the scores
 *  are kept by the server: no new cross-fitting). */
function QiniMore({ datasetId, design, covariates }: { datasetId: string; design: ExperimentDesign; covariates: string[] | null }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="group text-xs" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("experiments.policy.qiniOpen")}</summary>
      {open && <QiniCurves datasetId={datasetId} design={design} covariates={covariates} />}
    </details>
  );
}

function QiniCurves({ datasetId, design, covariates }: { datasetId: string; design: ExperimentDesign; covariates: string[] | null }) {
  const policy = usePolicy(datasetId, design, covariates, true, true);
  const learnt = policy.data;
  const curves = useMemo(() => (learnt ? learnt.qini.map((curve) => ({ curve, figure: qiniFigure(curve, learnt.kind) })) : []), [learnt]);
  if (policy.error instanceof ApiError) return <ErrorNote>{policy.error.message}</ErrorNote>;
  if (!learnt) return <Loading label={t("experiments.policy.qiniLoading")} className="mt-3 h-64 rounded-lg" compact />;
  return (
    <div className="mt-3 grid gap-4 lg:grid-cols-2">
      {curves.map(({ curve, figure }) => (
        <div key={curve.variant} className="min-w-0">
          <h3 className="text-[13px] font-medium text-ink-1">{t("experiments.policy.qiniTitle", { variant: curve.variant })}</h3>
          <p className="text-xs text-ink-2">{curve.sentence}</p>
          <Plot figure={figure} height={260} label={t("experiments.policy.qiniLabel", { variant: curve.variant })} />
        </div>
      ))}
    </div>
  );
}

function Learnt({ learnt, more }: { learnt: PolicyLearning; more: ReactNode }) {
  const best = learnt.values.find((v) => v.key === learnt.best_constant)?.value ?? null;
  const values = useMemo(
    () => policyValuesFigure(learnt.values.map((v) => ({ label: v.label, value: v.value, low: v.low, high: v.high })), learnt.kind, best),
    [learnt, best],
  );
  const left = learnt.covariates.filter((c) => !c.used);
  return (
    <div className="mt-3 space-y-4">
      <div className="space-y-1.5">
        <p className="text-[15px] font-medium text-ink-1">{learnt.rule}</p>
        <p className="flex flex-wrap items-baseline gap-x-2 text-[13px] leading-relaxed text-ink-1">
          <Verdict verdict={learnt.verdict} />
          <span className="min-w-0">{learnt.sentence}</span>
        </p>
      </div>
      {learnt.leaves.length > 1 && (
        <ul className="space-y-0.5 text-[13px]" aria-label={t("experiments.policy.leavesLabel")}>
          {learnt.leaves.map((leaf) => (
            <li key={`${leaf.conditions}-${leaf.variant}`} className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 border-b border-line/60 py-1">
              <span className="min-w-0 text-ink-2">
                {leaf.conditions || t("experiments.policy.everyone")} → <span className="font-medium text-ink-1">{leaf.variant}</span>
              </span>
              <span className="text-xs text-ink-3 tabular-nums">
                {t("experiments.units", { count: leaf.units.toLocaleString("en-US") })} · {(leaf.share * 100).toFixed(0)} %
              </span>
            </li>
          ))}
        </ul>
      )}
      <div className="max-w-4xl min-w-0">
        <h3 className="text-[13px] font-medium text-ink-1">{t("experiments.policy.valuesTitle")}</h3>
        <p className="text-xs text-ink-3">{t("experiments.policy.valuesNote", { best: learnt.best_constant, value: value(learnt.kind, best) })}</p>
        <Plot figure={values} height={70 + 44 * learnt.values.length} label={t("experiments.policy.valuesLabel")} />
      </div>
      {left.length > 0 && (
        <ul className="space-y-0.5 text-xs text-ink-3" aria-label={t("experiments.policy.leftOut")}>
          {left.map((c) => (
            <li key={c.column}>
              <span className="font-mono text-ink-2">{c.column}</span>: {c.reason}
            </li>
          ))}
        </ul>
      )}
      <p className="max-w-prose text-xs text-ink-3">
        {learnt.method}
        {learnt.sampled && ` ${t("experiments.policy.sampled", { units: learnt.units.toLocaleString("en-US") })}`}
      </p>
      {more}
    </div>
  );
}

function PolicyBody({ datasetId, design, offer }: { datasetId: string; design: ExperimentDesign; offer: PolicyOffer }) {
  const [excluded, setExcluded] = useState<string[]>([]);
  const covariates = excluded.length ? offer.covariates.filter((c) => !excluded.includes(c)) : null;
  const policy = usePolicy(datasetId, design, covariates, true);
  const error = policy.error instanceof ApiError ? policy.error : null;
  return (
    <div className="mt-3 space-y-3">
      <CovariateChips columns={offer.covariates} excluded={excluded} onChange={setExcluded} label={t("experiments.policy.covariates")} />
      {error ? (
        <ErrorNote>{error.message}</ErrorNote>
      ) : !policy.data ? (
        <Loading label={t("experiments.policy.loading")} className="h-72 rounded-lg" />
      ) : (
        <div className={policy.isFetching ? "opacity-60 transition-opacity" : ""} aria-busy={policy.isFetching}>
          <Learnt learnt={policy.data} more={<QiniMore datasetId={datasetId} design={design} covariates={covariates} />} />
        </div>
      )}
    </div>
  );
}

/** 055 § B: a short rule giving each unit a variant, learnt on doubly robust scores, measured by
 *  cross-validation against everyone on one variant. Onion rule: the first plane only says whether
 *  it can be computed; the rule is "Advanced" (computed when opened), the Qini curves
 *  "Advanced › More" (requested only when opened). */
export function PolicySection({ datasetId, design, offer, metric }: { datasetId: string; design: ExperimentDesign; offer: PolicyOffer; metric: string }) {
  const [open, setOpen] = useState(false);
  return (
    <section aria-labelledby="experiment-policy" className="space-y-2 border-t border-line pt-5">
      <div className="space-y-0.5">
        <div className="flex flex-wrap items-baseline gap-x-2">
          <h2 id="experiment-policy" className="text-[15px] font-semibold text-ink-1">
            {t("experiments.policy.title")}
          </h2>
          <span className="font-mono text-xs text-ink-3">{metric}</span>
        </div>
        <p className="max-w-prose text-xs text-ink-3">{t("experiments.policy.why")}</p>
      </div>
      {offer.available ? (
        <details className="group text-xs" onToggle={(e) => setOpen(e.currentTarget.open)}>
          <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("experiments.policy.open")}</summary>
          {open && (
            <>
              <p className="mt-2 max-w-prose border-l-2 border-warn pl-3 text-xs text-ink-2">{t("experiments.policy.caveat")}</p>
              <PolicyBody datasetId={datasetId} design={design} offer={offer} />
            </>
          )}
        </details>
      ) : (
        <p className="max-w-prose text-xs text-ink-2">{offer.reason}</p>
      )}
    </section>
  );
}
