import { useId, useState } from "react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useEntityClusters, type EntityCluster, type EntityClusters, type Step, type Suggestion } from "./api";
import { useReview, type Replace } from "./context";
import { entitiesNow, entityOf, isApart, stepsForClusters, takenOut, withApart, withTaken, type EntityStep } from "./entities";
import { visible } from "./parts";
import { withFix } from "./SeeRows";

/** Clusters fetched at first, and by each "Show more". */
const PAGE = 20;
const MAX_CLUSTERS = 200;
/** Spellings of a cluster shown side by side; the rest are counted. */
const SHOWN_RECORDS = 6;

/** N36: an entity number (A-20) reviewed cluster by cluster, with N35's panel: each cluster's
 *  spellings side by side (a column per spelling, the cells of its first row, those that differ
 *  highlighted); untick a cluster whose spellings are not one entity and each keeps its own
 *  number, or tick the spellings that are another one to split them off together. The names are
 *  never changed. Every change writes the step again (one change, one Ctrl+Z); the clusters are
 *  those of the rule, read once. */
export function EntityReview({ suggestion, step, onReplace }: { suggestion: Suggestion; step: Step | undefined; onReplace: Replace }) {
  const context = useReview();
  const [limit, setLimit] = useState(PAGE);
  const fix = context ? withFix(suggestion, context) : null;
  const stepId = fix?.step?.id ?? null;
  const query = useEntityClusters(context?.datasetId ?? "", fix && stepId ? stepsForClusters(fix.steps, stepId) : undefined, stepId, limit);
  const base = entityOf(step ?? suggestion.step);
  const current = entityOf(step);
  if (!base) return null;
  const write = (next: EntityStep) => onReplace(suggestion, next as Step);

  return (
    <section aria-label={t("clean.ent.label")} aria-busy={query.isFetching} className="mx-3 mt-1 mb-2 rounded-md border border-line bg-surface-0 sm:ml-9">
      {query.data ? <Summary found={query.data} step={current} /> : <p className="border-b border-line px-3 py-1.5 text-xs text-ink-2">{t("clean.ent.label")}</p>}
      <p className="px-3 pt-1.5 text-xs text-ink-3">{t("clean.ent.hint", { to: base.to })}</p>
      {query.isError ? (
        <div className="p-3">
          <ErrorNote>{query.error.message}</ErrorNote>
        </div>
      ) : !query.data ? (
        <div className="p-3">
          <Loading compact label={t("clean.ent.loading")} className="h-24 rounded-md" />
        </div>
      ) : query.data.clusters.length === 0 ? (
        <p className="px-3 py-3 text-xs text-ink-2">{t("clean.ent.none")}</p>
      ) : (
        <>
          <ol className="mt-1.5 max-h-[32rem] divide-y divide-line overflow-y-auto border-t border-line">
            {query.data.clusters.map((cluster, index) => (
              <ClusterCard key={cluster.index} index={index + 1} cluster={cluster} found={query.data} step={current} base={base} onWrite={write} />
            ))}
          </ol>
          <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line px-3 py-1.5 text-xs text-ink-3 tabular-nums">
            <span>{t("clean.ent.shown", { shown: formatCount(query.data.clusters.length), count: formatCount(query.data.total_clusters) })}</span>
            {query.data.clusters.length < query.data.total_clusters && limit < MAX_CLUSTERS && (
              <Button variant="ghost" className="-ml-3" disabled={query.isFetching} onClick={() => setLimit(Math.min(MAX_CLUSTERS, limit + PAGE))}>
                {t("clean.ent.more", { count: formatCount(Math.min(PAGE, query.data.total_clusters - query.data.clusters.length)) })}
              </Button>
            )}
          </footer>
        </>
      )}
    </section>
  );
}

/** "51 clusters · 282 rows · 53 entity numbers · 1 kept apart · 1 split", said again when it changes. */
function Summary({ found, step }: { found: EntityClusters; step: EntityStep | null }) {
  const apart = step?.not_entities?.length ?? 0;
  const split = Object.keys(step?.split ?? {}).length;
  const parts = [
    t("clean.ent.clusters", { count: formatCount(found.total_clusters) }),
    t("clean.ent.rows", { count: formatCount(found.total_rows) }),
    t("clean.ent.numbers", { count: formatCount(entitiesNow(found, step)) }),
    apart > 0 && t("clean.ent.apartCount", { count: formatCount(apart) }),
    split > 0 && t("clean.ent.splitCount", { count: formatCount(split) }),
  ].filter(Boolean);
  return (
    <header className="border-b border-line px-3 py-1.5">
      <p className="text-xs text-ink-2 tabular-nums" aria-live="polite">
        {parts.join(" · ")}
      </p>
      {!step && <p className="text-xs text-ink-3">{t("clean.ent.notTicked")}</p>}
    </header>
  );
}

/** One cluster: its tick (one entity or not) and its spellings side by side: a column per
 *  spelling (its rows, a tick to split it off), the cells of its first row, the differing first. */
function ClusterCard({ index, cluster, found, step, base, onWrite }: { index: number; cluster: EntityCluster; found: EntityClusters; step: EntityStep | null; base: EntityStep; onWrite: (next: EntityStep) => void }) {
  const [equal, setEqual] = useState(false);
  const tickId = useId();
  const tableId = useId();
  const apart = step ? isApart(cluster, step) : false;
  const taken = takenOut(cluster, step);
  const shown = cluster.records.slice(0, SHOWN_RECORDS);
  const differing = new Set(cluster.differing);
  const columns = found.columns.map((name, at) => ({ name, at }));
  const first = columns.filter((column) => differing.has(column.at));
  const rest = columns.filter((column) => !differing.has(column.at));
  const listed = equal || first.length === 0 ? [...first, ...rest] : first;
  const status = apart
    ? t("clean.ent.apart", { count: formatCount(cluster.records.length) })
    : taken.size > 0
      ? t("clean.ent.split", { count: formatCount(taken.size), total: formatCount(cluster.records.length) })
      : t("clean.ent.one", { count: formatCount(cluster.records.length), rows: formatCount(cluster.rows) });

  return (
    <li className="px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <label htmlFor={tickId} className="flex min-h-7 cursor-pointer items-center gap-2 text-[13px] text-ink-1">
          <input id={tickId} type="checkbox" className="size-3.5 shrink-0 accent-(--accent)" aria-label={t("clean.ent.tickLabel", { index: formatCount(index) })} checked={!apart} onChange={(event) => onWrite(withApart(base, cluster, !event.target.checked))} />
          <span className="font-medium">{t("clean.ent.cluster", { index: formatCount(index) })}</span>
        </label>
        <span className="text-xs text-ink-2 tabular-nums">{status}</span>
      </div>
      <div className="mt-1.5 sm:pl-5.5">
        <table id={tableId} className="w-full table-fixed border-collapse text-[12px]">
          <caption className="sr-only">{t("clean.ent.caption", { index: formatCount(index) })}</caption>
          <thead>
            <tr className="border-b border-line">
              <th scope="col" className="w-28 py-1 pr-2 text-left text-[11px] font-normal text-ink-3 sm:w-40">
                {t("clean.dup.column")}
              </th>
              {shown.map((record) => (
                <th key={record.index} scope="col" className="py-1 pr-2 text-left align-bottom font-normal">
                  <span className="block truncate font-mono text-ink-1" title={record.spelling ?? record.name}>
                    {visible(record.spelling ?? record.name)}
                  </span>
                  <span className="block text-[11px] text-ink-3 tabular-nums">
                    {record.first === null ? t("clean.ent.gone") : t("clean.ent.recordRows", { count: formatCount(record.rows), row: formatCount(record.first + 1) })}
                  </span>
                  <label className={`flex min-h-7 items-center gap-1.5 text-[11px] ${apart ? "cursor-default text-ink-3" : "cursor-pointer text-ink-2"}`}>
                    <input
                      type="checkbox"
                      className="size-3.5 shrink-0 accent-(--accent)"
                      aria-label={t("clean.ent.splitLabel", { spelling: record.spelling ?? record.name, index: formatCount(index) })}
                      checked={taken.has(record.index)}
                      disabled={apart}
                      onChange={(event) => onWrite(withTaken(base, cluster, record.index, event.target.checked))}
                    />
                    {t("clean.ent.splitOff")}
                  </label>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {listed.map(({ name, at }) => {
              const differs = differing.has(at);
              return (
                <tr key={name} className="border-b border-line last:border-b-0">
                  <th scope="row" className="py-1 pr-2 text-left align-top font-mono font-normal break-words text-ink-2">
                    {name}
                    {differs && <span className="sr-only"> ({t("clean.dup.differs")})</span>}
                  </th>
                  {shown.map((record) => (
                    <td key={record.index} className={`py-1 pr-2 align-top font-mono break-words ${taken.has(record.index) ? "text-ink-3" : differs ? "text-ink-1" : "text-ink-2"} ${differs ? "bg-warn/8" : ""}`}>
                      {visible(record.cells[at] ?? null)}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-1 flex flex-wrap items-center gap-x-3 text-[11px] text-ink-3">
          {rest.length > 0 && first.length > 0 && (
            <button type="button" aria-expanded={equal} aria-controls={tableId} onClick={() => setEqual(!equal)} className="min-h-7 rounded-sm text-ink-2 underline-offset-2 hover:underline">
              {t(equal ? "clean.dup.hideEqual" : "clean.dup.showEqual", { count: formatCount(rest.length) })}
            </button>
          )}
          {cluster.records.length > shown.length && <span>{t("clean.ent.moreRecords", { count: formatCount(cluster.records.length - shown.length) })}</span>}
        </p>
      </div>
    </li>
  );
}
