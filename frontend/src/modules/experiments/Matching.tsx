import { Check, TriangleAlert } from "lucide-react";
import { useMemo, useState } from "react";

import { ApiError } from "../../api/client";
import { Plot } from "../../charts/Plot";
import { t } from "../../lib/i18n";
import { PHONE, useMedia } from "../../lib/useMedia";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";
import { useMatchedGroups, type DecisionsRequest, type MatchedGroups, type MatchingOffer } from "./api";
import { policyValuesFigure } from "./figures";
import { gap, interval } from "./format";

const BINS = 4;
const CHART_GROUPS = 12;
const head = "px-2 py-1.5 text-right font-normal whitespace-nowrap text-ink-3 first:pl-0 first:text-left";
const cell = "px-2 py-1.5 text-right whitespace-nowrap tabular-nums first:pl-0 first:text-left";
const smd = (v: number | null) => (v === null ? "—" : v.toFixed(2));

/** 123, under Policies › Advanced: units matched exactly on their covariates (FLAME). The rule
 *  (R-MAT-01) says whether it applies and why; nothing is computed before "Match the units". */
export function Matching({
  datasetId,
  request,
  offer,
  actions,
}: {
  datasetId: string;
  /** Policies: the decision log; Analyze (123 bis): the variant and the primary metric. */
  request: Pick<DecisionsRequest, "action" | "reward" | "covariates">;
  offer: MatchingOffer;
  actions: string[];
}) {
  const [asked, setAsked] = useState(false);
  const [treated, setTreated] = useState<string | null>(null);
  const [control, setControl] = useState<string | null>(null);
  const [method, setMethod] = useState<"flame" | "dame">("flame");
  const [bins, setBins] = useState(BINS);
  const body = asked && offer.available ? { treatment: request.action, outcome: request.reward, covariates: request.covariates ?? null, treated, control, method, bins } : null;
  const found = useMatchedGroups(datasetId, body);
  return (
    <section aria-labelledby="policies-matching" className="space-y-3 border-t border-line pt-4">
      <div className="space-y-0.5">
        <h3 id="policies-matching" className="text-[13px] font-semibold text-ink-1">
          {t("experiments.matching.title")}
        </h3>
        <p className="max-w-prose text-xs text-ink-2">{offer.reason}</p>
      </div>
      {offer.available && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
            <Cell label={t("experiments.matching.treated")} why={t("experiments.matching.treatedWhy")} changed={treated !== null} onReset={() => setTreated(null)}>
              <Select aria-label={t("experiments.matching.treated")} value={treated ?? ""} onChange={(e) => setTreated(e.target.value || null)} className="w-full text-xs">
                <option value="">{t("experiments.matching.treatedAuto")}</option>
                {actions.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </Select>
            </Cell>
            {actions.length > 2 && (
              <Cell label={t("experiments.matching.control")} why={t("experiments.matching.controlWhy")} changed={control !== null} onReset={() => setControl(null)}>
                <Select aria-label={t("experiments.matching.control")} value={control ?? ""} onChange={(e) => setControl(e.target.value || null)} className="w-full text-xs">
                  <option value="">{t("experiments.matching.treatedAuto")}</option>
                  {actions.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </Select>
              </Cell>
            )}
            <Cell label={t("experiments.matching.method")} why={t("experiments.matching.methodWhy")} changed={method !== "flame"} onReset={() => setMethod("flame")}>
              <Select aria-label={t("experiments.matching.method")} value={method} onChange={(e) => setMethod(e.target.value as "flame" | "dame")} className="w-full text-xs">
                <option value="flame">{t("experiments.matching.flame")}</option>
                <option value="dame">{t("experiments.matching.dame")}</option>
              </Select>
            </Cell>
            <Cell label={t("experiments.matching.bins")} why={t("experiments.matching.binsWhy")} changed={bins !== BINS} onReset={() => setBins(BINS)}>
              <Select aria-label={t("experiments.matching.bins")} value={bins} onChange={(e) => setBins(Number(e.target.value))} className="w-full text-xs">
                {[2, 3, 4, 5, 6, 8, 10].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </Select>
            </Cell>
          </div>
          {!asked ? (
            <Button variant="secondary" onClick={() => setAsked(true)}>
              {t("experiments.matching.run")}
            </Button>
          ) : found.error instanceof ApiError ? (
            <ErrorNote>{found.error.message}</ErrorNote>
          ) : !found.data ? (
            <Loading label={t("experiments.matching.loading")} className="h-32 rounded-lg" compact />
          ) : (
            <div className={found.isFetching ? "opacity-60 transition-opacity" : ""} aria-busy={found.isFetching}>
              <Matched result={found.data} />
            </div>
          )}
        </>
      )}
    </section>
  );
}

function Matched({ result }: { result: MatchedGroups }) {
  const kind = result.kind;
  const phone = useMedia(PHONE);
  const shown = Math.min(result.groups.length, CHART_GROUPS);
  const figure = useMemo(
    () =>
      policyValuesFigure(
        // 041: a phone names each group by its number; its conditions are in the table below
        result.groups.slice(0, CHART_GROUPS).map((g) => ({ label: phone ? `#${g.id}` : `#${g.id} ${g.conditions.join(" · ")}`, value: g.effect, low: g.low, high: g.high })),
        result.kind,
        result.effects.find((e) => e.key === "ate")?.value ?? null,
        t(result.kind === "rate" ? "experiments.matching.axisRate" : "experiments.matching.axis", { outcome: result.outcome }),
        true,
      ),
    [result, phone],
  );
  return (
    <div className="space-y-5">
      <p className="max-w-prose text-[13px] leading-relaxed text-ink-1">{result.sentence}</p>
      {result.warnings.map((warning) => (
        <p key={warning} role="note" className="flex max-w-prose items-start gap-2 border-l-2 border-warn pl-3 text-xs text-ink-1">
          <TriangleAlert size={13} strokeWidth={1.75} className="mt-0.5 shrink-0 text-warn" aria-hidden />
          <span>{warning}</span>
        </p>
      ))}
      <dl className="grid grid-cols-1 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3">
        {result.effects.map((e) => (
          <div key={e.key} className="min-w-0 space-y-0.5 bg-surface-1 px-3 py-2">
            <dt className="text-[11px] text-ink-2">{e.label}</dt>
            <dd className={`text-[15px] tabular-nums ${e.key === "ate" ? "font-semibold text-ink-1" : "text-ink-1"}`}>{gap(kind, e.value)}</dd>
            <dd className="text-xs text-ink-3 tabular-nums">
              {e.low !== null && e.high !== null ? interval(gap(kind, e.low), gap(kind, e.high)) : "—"} · {t("experiments.matching.units", { count: e.units.toLocaleString("en-US") })}
            </dd>
          </div>
        ))}
      </dl>
      {shown > 0 && (
        <figure className="max-w-4xl space-y-1">
          <figcaption className="text-xs text-ink-2">{t("experiments.matching.chartCaption", { count: String(shown) })}</figcaption>
          <Plot figure={figure} height={70 + 30 * shown} label={t("experiments.matching.chartLabel")} />
        </figure>
      )}
      <GroupsTable result={result} />
      <LevelsTable result={result} />
      <BalanceTable result={result} />
      <p className="max-w-prose text-xs text-ink-3">
        {result.stopped} {result.method}
      </p>
    </div>
  );
}

function GroupsTable({ result }: { result: MatchedGroups }) {
  return (
    <div className="space-y-1">
      <h4 className="text-xs font-medium text-ink-1">{t("experiments.matching.groupsTitle", { shown: String(result.groups.length), count: result.group_count.toLocaleString("en-US") })}</h4>
      <div className="overflow-x-auto">
        {/* 041: a phone keeps the group, its effect and its sizes. */}
        <table className="w-full min-w-[560px] text-xs max-sm:min-w-0 max-sm:[&_tr>:nth-child(2)]:hidden max-sm:[&_tr>:nth-child(6)]:hidden" aria-label={t("experiments.matching.groupsTitle", { shown: String(result.groups.length), count: String(result.group_count) })}>
          <thead>
            <tr className="border-b border-line">
              <th className={head}>{t("experiments.matching.col.group")}</th>
              <th className={head}>{t("experiments.matching.col.level")}</th>
              <th className={head}>{result.treated}</th>
              <th className={head}>{result.control}</th>
              <th className={head}>{t("experiments.matching.col.effect")}</th>
              <th className={head}>{t("experiments.matching.col.rows")}</th>
            </tr>
          </thead>
          <tbody>
            {result.groups.map((g) => (
              <tr key={g.id} className="border-b border-line/60 align-top">
                <td className={`${cell} whitespace-normal`}>
                  <span className="text-ink-3">#{g.id} </span>
                  {g.conditions.join(" · ")}
                </td>
                <td className={cell}>{g.level}</td>
                <td className={cell}>{g.treated.toLocaleString("en-US")}</td>
                <td className={cell}>{g.control.toLocaleString("en-US")}</td>
                <td className={cell}>
                  {gap(result.kind, g.effect)} {g.low !== null && g.high !== null && <span className="text-ink-3 max-sm:hidden">{interval(gap(result.kind, g.low), gap(result.kind, g.high))}</span>}
                </td>
                <td className={`${cell} max-w-48 truncate font-mono text-[11px] text-ink-2`} title={g.rows.join(", ")}>
                  {g.rows.join(", ")}
                  {g.treated + g.control > g.rows.length ? "…" : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function LevelsTable({ result }: { result: MatchedGroups }) {
  return (
    <div className="space-y-1">
      <h4 className="text-xs font-medium text-ink-1">{t("experiments.matching.levelsTitle")}</h4>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] text-xs max-sm:min-w-0 max-sm:[&_tr>:nth-child(4)]:hidden" aria-label={t("experiments.matching.levelsTitle")}>
          <thead>
            <tr className="border-b border-line">
              <th className={head}>{t("experiments.matching.col.level")}</th>
              <th className={head}>{t("experiments.matching.col.matchedOn")}</th>
              <th className={head}>{t("experiments.matching.col.dropped")}</th>
              <th className={head}>{t("experiments.matching.col.error")}</th>
              <th className={head}>{t("experiments.matching.col.groups")}</th>
              <th className={head}>{t("experiments.matching.col.newly", { treated: result.treated, control: result.control })}</th>
            </tr>
          </thead>
          <tbody>
            {result.levels.map((level) => (
              <tr key={level.level} className="border-b border-line/60 align-top">
                <td className={cell}>{level.level}</td>
                <td className={`${cell} whitespace-normal`}>{level.covariates.join(", ")}</td>
                <td className={cell}>{level.dropped ?? "—"}</td>
                <td className={cell}>{`${(level.prediction_error * 100).toFixed(1)} %`}</td>
                <td className={cell}>{level.groups.toLocaleString("en-US")}</td>
                <td className={cell}>
                  {level.treated.toLocaleString("en-US")} / {level.control.toLocaleString("en-US")}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="max-w-prose text-xs text-ink-3">{t("experiments.matching.errorHelp")}</p>
    </div>
  );
}

function BalanceTable({ result }: { result: MatchedGroups }) {
  const how = new Map(result.covariates.map((c) => [c.column, c.how]));
  return (
    <div className="space-y-1">
      <h4 className="text-xs font-medium text-ink-1">{t("experiments.matching.balanceTitle")}</h4>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[480px] text-xs max-sm:min-w-0 max-sm:[&_tr>:nth-child(5)]:hidden" aria-label={t("experiments.matching.balanceTitle")}>
          <thead>
            <tr className="border-b border-line">
              <th className={head}>{t("experiments.matching.col.column")}</th>
              <th className={head}>{t("experiments.matching.col.before")}</th>
              <th className={head}>{t("experiments.matching.col.after")}</th>
              <th className={head}>{t("experiments.matching.col.balanced")}</th>
              <th className={head}>{t("experiments.matching.col.how")}</th>
            </tr>
          </thead>
          <tbody>
            {result.balance.map((row) => (
              <tr key={row.column} className="border-b border-line/60">
                <td className={cell}>{row.column}</td>
                <td className={cell}>{smd(row.before)}</td>
                <td className={cell}>{smd(row.after)}</td>
                <td className={cell}>
                  {row.balanced ? (
                    <Check size={13} className="ml-auto text-ok" aria-label={t("experiments.matching.balancedYes")} />
                  ) : (
                    <span className="text-warn">{t("experiments.matching.balancedNo")}</span>
                  )}
                </td>
                <td className={`${cell} whitespace-normal text-ink-3`}>
                  {how.get(row.column)}
                  {row.dropped_at !== null && ` · ${t("experiments.matching.droppedAt", { level: String(row.dropped_at) })}`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="max-w-prose text-xs text-ink-3">{t("experiments.matching.balanceHelp")}</p>
    </div>
  );
}
