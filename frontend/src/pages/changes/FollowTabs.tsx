import { Link } from "@tanstack/react-router";

import type { Dataset, KpiChange, RelationChange } from "../../api/client";
import { useChangeKpis, useChangeRelations } from "../../api/hooks";
import { formatNumber } from "../../charts/figures";
import { METRIC_LABELS } from "../../features/correlations/figures";
import { t, type MessageKey } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { NotKept } from "./RowsTab";
import { Empty } from "./SchemaTabs";
import { Loading } from "../../ui/Loading";

const head = "px-3 py-1.5 font-normal";
const cell = "px-3 py-1.5";

const change = (value: number | null) =>
  value === null ? "—" : `${value > 0 ? "+" : value < 0 ? "−" : ""}${Math.abs(100 * value).toFixed(1)}%`;

const KPI_STATUS: Record<KpiChange["status"], MessageKey> = {
  ok: "changes.kpis.ok",
  broken: "changes.kpis.broken",
  unavailable: "changes.kpis.unavailable",
};

export function KpisTab({ dataset, vs }: { dataset: Dataset; vs: string | null }) {
  const kpis = useChangeKpis(dataset, vs, true);
  if (kpis.isError) return <ErrorNote>{kpis.error.message}</ErrorNote>;
  if (!kpis.data) return <Loading label={t("loading.kpis")} className="h-32 rounded-lg" />;
  if (kpis.data.length === 0) {
    return (
      <Empty>
        {t("changes.kpis.none")}{" "}
        <Link to="/d/$datasetId/kpi" params={{ datasetId: dataset.id }} className="text-accent hover:underline">
          {t("changes.kpis.open")}
        </Link>
      </Empty>
    );
  }
  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={`${head} w-full`}>{t("changes.kpis.name")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.before")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.after")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.kpis.change")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.kpis.revised")}</th>
            </tr>
          </thead>
          <tbody>
            {kpis.data.map((kpi) => (
              <tr key={kpi.name} className="border-t border-line align-top">
                <td className={cell}>
                  <span className="text-ink-1">{kpi.name}</span>
                  <span className="ml-2 text-xs text-ink-3">{kpi.measure}</span>
                  {kpi.status !== "ok" && (
                    <p className={`text-xs ${kpi.status === "broken" ? "text-bad" : "text-ink-3"}`}>
                      {t(KPI_STATUS[kpi.status])}
                      {kpi.problem ? `: ${kpi.problem}` : ""}
                    </p>
                  )}
                </td>
                <td className={`${cell} text-right text-ink-2 tabular-nums`}>{formatNumber(kpi.before)}</td>
                <td className={`${cell} text-right text-ink-1 tabular-nums`}>{formatNumber(kpi.after)}</td>
                <td className={`${cell} text-right text-ink-1 tabular-nums`}>{change(kpi.change)}</td>
                <td className={`${cell} text-right tabular-nums ${kpi.revised ? "text-warn" : "text-ink-3"}`}>
                  {kpi.common.length ? `${kpi.revised} / ${kpi.common.length}` : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="max-w-prose text-xs text-ink-3">{t("changes.kpis.hint")}</p>
      {kpis.data
        .filter((kpi) => kpi.revised > 0)
        .map((kpi) => (
          <Revisions key={kpi.name} kpi={kpi} />
        ))}
    </div>
  );
}

function Revisions({ kpi }: { kpi: KpiChange }) {
  const revised = kpi.common.filter((period) => period.change !== null && period.change !== 0);
  return (
    <section>
      <h2 className="mb-2 text-sm font-semibold">{t("changes.kpis.revisedTitle", { name: kpi.name })}</h2>
      <div className="max-w-xl overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={`${head} w-full`}>{t("changes.kpis.period")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.before")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.after")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.kpis.change")}</th>
            </tr>
          </thead>
          <tbody>
            {revised.map((period) => (
              <tr key={period.label} className="border-t border-line">
                <td className={`${cell} text-ink-1`}>{period.label}</td>
                <td className={`${cell} text-right text-ink-2 tabular-nums`}>{formatNumber(period.before)}</td>
                <td className={`${cell} text-right text-ink-1 tabular-nums`}>{formatNumber(period.after)}</td>
                <td className={`${cell} text-right text-ink-1 tabular-nums`}>{change(period.change)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const RELATION_STATUS: Record<RelationChange["status"], { label: MessageKey; tone: string }> = {
  appeared: { label: "changes.relations.appeared", tone: "text-warn" },
  disappeared: { label: "changes.relations.disappeared", tone: "text-warn" },
  changed: { label: "changes.relations.changed", tone: "text-ink-1" },
  stable: { label: "changes.relations.stable", tone: "text-ink-3" },
};

export function RelationsTab({ dataset, vs }: { dataset: Dataset; vs: string | null }) {
  const relations = useChangeRelations(dataset, vs, true);
  if (relations.isError) return <ErrorNote>{relations.error.message}</ErrorNote>;
  if (!relations.data) return <Loading label={t("loading.relations")} className="h-32 rounded-lg" />;
  if (!relations.data.available) return <NotKept reason={relations.data.reason ?? ""} />;
  if (relations.data.pairs.length === 0) return <Empty>{t("changes.relations.none")}</Empty>;
  const value = (v: number | null) => (v === null ? "—" : v.toFixed(2));
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-ink-3">{t("changes.relations.hint")}</p>
      <div className="overflow-x-auto rounded-md border border-line">
        <table className="w-full border-collapse text-[13px]">
          <thead className="bg-surface-2 text-left text-xs text-ink-3">
            <tr>
              <th scope="col" className={head}>A</th>
              <th scope="col" className={head}>B</th>
              <th scope="col" className={head}>{t("correlations.metric")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.before")}</th>
              <th scope="col" className={`${head} text-right`}>{t("changes.after")}</th>
              <th scope="col" className={head}>{t("changes.relations.status")}</th>
            </tr>
          </thead>
          <tbody>
            {relations.data.pairs.map((pair) => {
              const meta = RELATION_STATUS[pair.status];
              return (
                <tr key={`${pair.a}|${pair.b}`} className="border-t border-line">
                  <td className={`${cell} font-mono text-xs text-ink-1`}>{pair.a}</td>
                  <td className={`${cell} font-mono text-xs text-ink-1`}>{pair.b}</td>
                  <td className={`${cell} text-ink-2`}>{METRIC_LABELS[pair.metric as keyof typeof METRIC_LABELS] ?? pair.metric}</td>
                  <td className={`${cell} text-right text-ink-2 tabular-nums`}>{value(pair.before)}</td>
                  <td className={`${cell} text-right text-ink-1 tabular-nums`}>{value(pair.after)}</td>
                  <td className={`${cell} text-xs font-medium ${meta.tone}`}>{t(meta.label)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
