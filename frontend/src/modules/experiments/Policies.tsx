import { Plus, TriangleAlert, X } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";

import { ApiError } from "../../api/client";
import { useMlRuns } from "../../api/hooks";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { ParamPanel } from "../../ui/ParamPanel";
import { useEvaluatePolicy, type DecisionsRequest, type LoggedDecisions, type PolicyEstimate, type PolicyEvaluation } from "./api";
import { policyValuesFigure } from "./figures";
import { gap, interval, value } from "./format";
import { Matching } from "./Matching";
import { CovariateChips } from "./Policy";

const small = "h-7 text-xs";
const ESTIMATORS = ["dm", "ips", "snips", "dr"] as const;

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

const ROW = "flex flex-wrap items-center gap-x-3 gap-y-2 max-sm:grid max-sm:grid-cols-3 max-sm:items-end";

/** The setup: action, reward, propensity (a column or estimated), covariates and saved models. */
function SetupBar({
  decisions,
  request,
  covariates,
  onChange,
}: {
  decisions: LoggedDecisions;
  request: DecisionsRequest;
  covariates: string[];
  onChange: (next: DecisionsRequest) => void;
}) {
  const phone = useMedia(PHONE);
  const excluded = request.covariates ? covariates.filter((c) => !request.covariates?.includes(c)) : [];
  const fields = (
    <div className="space-y-2">
      <div className={ROW}>
        <Group label={t("experiments.policies.action")}>
          <Select
            aria-label={t("experiments.policies.action")}
            className={`${small} max-w-44`}
            value={request.action}
            onChange={(e) => onChange({ ...request, action: e.target.value, models: [] })}
          >
            {decisions.action_columns.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Group>
        <Group label={t("experiments.policies.reward")}>
          <Select
            aria-label={t("experiments.policies.reward")}
            className={`${small} max-w-44`}
            value={request.reward}
            onChange={(e) => onChange({ ...request, reward: e.target.value, models: [] })}
          >
            {decisions.reward_columns.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Group>
        <Group label={t("experiments.policies.propensity")}>
          <Select
            aria-label={t("experiments.policies.propensity")}
            className={`${small} max-w-44`}
            value={request.propensity ?? ""}
            onChange={(e) => onChange({ ...request, propensity: e.target.value || null })}
          >
            <option value="">{t("experiments.policies.estimate")}</option>
            {decisions.propensity_columns.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </Group>
      </div>
      <div className="border-t border-line pt-2">
        <CovariateChips
          columns={covariates}
          excluded={excluded}
          onChange={(next) => onChange({ ...request, covariates: next.length ? covariates.filter((c) => !next.includes(c)) : null })}
          label={t("experiments.policies.covariates")}
        />
      </div>
    </div>
  );
  if (phone)
    return (
      <ParamPanel label={t("experiments.setup")} layout="stack" summary={[request.action, request.reward, request.propensity ?? t("experiments.policies.estimate")]}>
        <div className="space-y-2 text-[13px]">{fields}</div>
      </ParamPanel>
    );
  return <div className="border-y border-line bg-surface-1 px-3 py-2.5 text-[13px]">{fields}</div>;
}

/** Saved models of the dataset (037) evaluated as policies: one that predicts the action, or the
 *  reward from the action. Each one is evaluated only once added. */
function ModelsPicker({ datasetId, request, onChange }: { datasetId: string; request: DecisionsRequest; onChange: (next: DecisionsRequest) => void }) {
  const byAction = useMlRuns(datasetId, request.action);
  const byReward = useMlRuns(datasetId, request.reward);
  const runs = [...(byAction.data ?? []), ...(byReward.data ?? [])];
  const chosen = request.models ?? [];
  const runLabel = (id: string) => {
    const run = runs.find((r) => r.id === id);
    return run ? `${run.label || run.model} → ${run.target}` : id;
  };
  return (
    <div className="flex flex-wrap items-center gap-1.5 text-[13px]">
      <Group label={t("experiments.policies.models")} wide>
          {chosen.map((id) => (
            <span key={id} className="inline-flex h-7 items-center gap-1 rounded-md border border-line px-2 text-xs text-ink-2">
              {runLabel(id)}
              <button
                type="button"
                className="text-ink-3 hover:text-ink-1"
                aria-label={t("experiments.policies.removeModel", { model: runLabel(id) })}
                onClick={() => onChange({ ...request, models: chosen.filter((m) => m !== id) })}
              >
                <X size={12} aria-hidden />
              </button>
            </span>
          ))}
          {chosen.length < 5 && (
            <label className="relative inline-flex items-center">
              <span className="sr-only">{t("experiments.policies.addModel")}</span>
              <Plus size={12} className="pointer-events-none absolute left-2 text-ink-3" aria-hidden />
              <Select
                className={`${small} w-44 pl-6 text-ink-2`}
                value=""
                disabled={runs.length === 0}
                onChange={(e) => e.target.value && onChange({ ...request, models: [...chosen, e.target.value] })}
              >
                <option value="">{runs.length ? t("experiments.policies.addModel") : t("experiments.policies.noModel")}</option>
                {runs
                  .filter((r) => !chosen.includes(r.id))
                  .map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label || r.model} → {r.target}
                    </option>
                  ))}
              </Select>
            </label>
          )}
        </Group>
    </div>
  );
}

function estimate(policy: PolicyEstimate, key: (typeof ESTIMATORS)[number]) {
  return policy.estimates.find((e) => e.estimator === key) ?? null;
}

function PoliciesTable({ evaluation }: { evaluation: PolicyEvaluation }) {
  const kind = evaluation.kind;
  const head = "px-2 py-1.5 text-right font-normal text-ink-3 first:pl-0 first:text-left";
  const cell = "px-2 py-1.5 text-right whitespace-nowrap tabular-nums first:pl-0 first:text-left";
  const shown = (v: number | null, low: number | null, high: number | null) => (
    <>
      {value(kind, v)} {low !== null && high !== null && <span className="text-ink-3">{interval(value(kind, low), value(kind, high))}</span>}
    </>
  );
  return (
    <div className="overflow-x-auto">
      {/* 041: a phone keeps the policy, DR, the gain and the effective sample. */}
      <table
        className="w-full min-w-[640px] text-xs max-sm:[&_tr>:nth-child(2)]:hidden max-sm:[&_tr>:nth-child(3)]:hidden max-sm:[&_tr>:nth-child(4)]:hidden"
        aria-label={t("experiments.policies.tableLabel")}
      >
        <thead>
          <tr className="border-b border-line">
            <th className={head}>{t("experiments.policies.col.policy")}</th>
            <th className={head}>DM</th>
            <th className={head}>IPS</th>
            <th className={head}>SNIPS</th>
            <th className={head}>DR</th>
            <th className={head}>{t("experiments.policies.col.gain")}</th>
            <th className={head}>{t("experiments.policies.col.ess")}</th>
          </tr>
        </thead>
        <tbody>
          {evaluation.policies.map((policy) => {
            const reference = policy.source === "logged";
            return (
              <tr key={policy.key} className="border-b border-line/60">
                <td className={`${cell} ${reference ? "text-ink-2" : ""}`}>
                  {policy.label}
                  {!policy.reliable && <span className="text-warn"> · {t("experiments.policies.unreliable")}</span>}
                </td>
                {ESTIMATORS.map((key) => {
                  const e = estimate(policy, key);
                  if (reference)
                    return key === "dr" ? (
                      <td key={key} className={cell}>
                        {shown(policy.value, policy.low, policy.high)}
                      </td>
                    ) : (
                      <td key={key} className={`${cell} text-ink-3`}>
                        —
                      </td>
                    );
                  return (
                    <td key={key} className={`${cell} ${key === "dr" ? "font-medium" : ""}`}>
                      {e ? shown(e.value, e.low, e.high) : "—"}
                    </td>
                  );
                })}
                <td className={cell}>
                  {reference ? (
                    <span className="text-ink-3">{t("experiments.policies.reference")}</span>
                  ) : (
                    <>
                      {gap(kind, policy.gain)}{" "}
                      {policy.gain_low !== null && policy.gain_high !== null && (
                        <span className="text-ink-3">{interval(gap(kind, policy.gain_low), gap(kind, policy.gain_high))}</span>
                      )}
                    </>
                  )}
                </td>
                <td className={cell}>
                  {policy.ess === null ? "—" : Math.round(policy.ess).toLocaleString("en-US")}
                  {policy.ess_share !== null && !reference && <span className="text-ink-3"> ({(policy.ess_share * 100).toFixed(0)} %)</span>}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Evaluation({ evaluation, advanced }: { evaluation: PolicyEvaluation; advanced: ReactNode }) {
  const logged = evaluation.policies.find((p) => p.source === "logged");
  const figure = useMemo(
    () => policyValuesFigure(evaluation.policies.map((p) => ({ label: p.label, value: p.value, low: p.low, high: p.high })), evaluation.kind, logged?.value ?? null),
    [evaluation, logged],
  );
  const d = evaluation.diagnostics;
  return (
    <div className="space-y-5">
      <section aria-labelledby="policies-diagnostics" className="space-y-2">
        <h2 id="policies-diagnostics" className="text-[15px] font-semibold text-ink-1">
          {t("experiments.policies.diagnostics")}
        </h2>
        {evaluation.warnings.map((warning) => (
          <p key={warning} role="note" className="flex max-w-prose items-start gap-2 border-l-2 border-warn pl-3 text-xs text-ink-1">
            <TriangleAlert size={13} strokeWidth={1.75} className="mt-0.5 shrink-0 text-warn" aria-hidden />
            <span>{warning}</span>
          </p>
        ))}
        <p className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink-3 tabular-nums">
          <span>{t(d.propensity === "column" ? "experiments.policies.fromColumn" : "experiments.policies.estimated")}</span>
          <span>{t("experiments.policies.decisions", { count: evaluation.units.toLocaleString("en-US") })}</span>
          <span>{t("experiments.policies.clipped", { count: d.clipped.toLocaleString("en-US") })}</span>
        </p>
        <ul className="max-w-prose space-y-0.5 text-xs text-ink-2">
          {d.sentences.map((sentence) => (
            <li key={sentence}>{sentence}</li>
          ))}
        </ul>
      </section>
      <section aria-labelledby="policies-values" className="space-y-3 border-t border-line pt-5">
        <h2 id="policies-values" className="text-[15px] font-semibold text-ink-1">
          {t("experiments.policies.valuesTitle", { reward: evaluation.reward })}
        </h2>
        <ul className="space-y-1 text-[13px] leading-relaxed text-ink-1">
          {evaluation.policies.map((p) => (
            <li key={p.key} className={p.reliable ? "" : "text-ink-2"}>
              {p.sentence}
            </li>
          ))}
        </ul>
        <div className="max-w-4xl">
          <Plot figure={figure} height={70 + 44 * evaluation.policies.length} label={t("experiments.policies.chartLabel")} />
        </div>
        {advanced}
      </section>
    </div>
  );
}

/** "Advanced": every estimator side by side, the saved models (nothing new is computed until a
 *  model is added) and the matched groups (123, computed once asked). */
function Advanced({
  datasetId,
  evaluation,
  request,
  onChange,
}: {
  datasetId: string;
  evaluation: PolicyEvaluation;
  request: DecisionsRequest;
  onChange: (next: DecisionsRequest) => void;
}) {
  return (
    <details className="group text-xs">
      <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("experiments.policies.advanced")}</summary>
      <div className="mt-3 space-y-3">
        <PoliciesTable evaluation={evaluation} />
        <ModelsPicker datasetId={datasetId} request={request} onChange={onChange} />
        {evaluation.skipped.map((why) => (
          <p key={why} className="text-xs text-ink-3">
            {t("experiments.policies.skipped", { why })}
          </p>
        ))}
        <p className="max-w-prose text-xs text-ink-3">
          {evaluation.method}
          {evaluation.sampled && ` ${t("experiments.policy.sampled", { units: evaluation.units.toLocaleString("en-US") })}`}
        </p>
        {evaluation.matching && (
          <Matching key={`${request.action}-${request.reward}`} datasetId={datasetId} request={request} offer={evaluation.matching} actions={evaluation.actions} />
        )}
        <LearnMore datasetId={datasetId} request={request} />
      </div>
    </details>
  );
}

/** "Advanced › More": a rule learnt from the log (a cross-validated tree), requested only once
 *  this section is opened. */
function LearnMore({ datasetId, request }: { datasetId: string; request: DecisionsRequest }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="group text-xs" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("experiments.policies.learnOpen")}</summary>
      {open && <LearntRule datasetId={datasetId} request={request} />}
    </details>
  );
}

function LearntRule({ datasetId, request }: { datasetId: string; request: DecisionsRequest }) {
  const learnt = useEvaluatePolicy(datasetId, { ...request, learn: true });
  if (learnt.error instanceof ApiError) return <ErrorNote>{learnt.error.message}</ErrorNote>;
  if (!learnt.data) return <Loading label={t("experiments.policies.learnLoading")} className="mt-3 h-24 rounded-lg" compact />;
  const rule = learnt.data.policies.find((p) => p.source === "learned");
  return (
    <div className="mt-3 space-y-1.5">
      {learnt.data.rule && <p className="text-[13px] font-medium text-ink-1">{learnt.data.rule}</p>}
      {rule && <p className="text-[13px] leading-relaxed text-ink-1">{rule.sentence}</p>}
      {rule && <p className="max-w-prose text-xs text-ink-3">{rule.description}</p>}
    </div>
  );
}

/** 055 § C: what other policies would have earned on the logged decisions. Onion rule: the
 *  diagnostics and the DR value of each simple policy first; every estimator and the saved
 *  models under "Advanced"; a rule learnt from the log under "Advanced › More". */
export function Policies({ datasetId, decisions }: { datasetId: string; decisions: LoggedDecisions }) {
  const [request, setRequest] = useState<DecisionsRequest>(() => ({
    action: decisions.action ?? "",
    reward: decisions.reward ?? "",
    propensity: decisions.propensity,
    covariates: null,
    models: [],
  }));
  const evaluation = useEvaluatePolicy(datasetId, request.action && request.reward ? request : null);
  const error = evaluation.error instanceof ApiError ? evaluation.error : null;
  const taken = [request.action, request.reward, request.propensity];
  const covariates = decisions.covariates.filter((c) => !taken.includes(c));
  return (
    <div className="space-y-4">
      <div className="space-y-0.5">
        <h2 className="text-[15px] font-semibold text-ink-1">{t("experiments.policies.title")}</h2>
        <p className="max-w-prose text-xs text-ink-3">{decisions.sentence}</p>
      </div>
      <SetupBar decisions={decisions} request={request} covariates={covariates} onChange={setRequest} />
      {error ? (
        <ErrorNote>{error.message}</ErrorNote>
      ) : !evaluation.data ? (
        <Loading label={t("experiments.policies.loading")} className="h-[420px]" />
      ) : (
        <div className={evaluation.isFetching ? "opacity-60 transition-opacity" : ""} aria-busy={evaluation.isFetching}>
          <Evaluation
            evaluation={evaluation.data}
            advanced={<Advanced datasetId={datasetId} evaluation={evaluation.data} request={request} onChange={setRequest} />}
          />
        </div>
      )}
    </div>
  );
}
