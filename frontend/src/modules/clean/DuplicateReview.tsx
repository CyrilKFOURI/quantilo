import { useId, useState } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useDuplicateGroups, type DuplicateGroup, type DuplicateGroups, type Step, type Suggestion } from "./api";
import { useReview, type Replace } from "./context";
import {
  choiceOf,
  duplicateOf,
  isApart,
  mergedCells,
  removedNow,
  rowOfChoice,
  stepsForGroups,
  survivorOf,
  withApart,
  withSurvivor,
  type Choice,
  type DuplicateStep,
} from "./duplicates";
import { visible } from "./parts";
import { withFix } from "./SeeRows";

/** Groups fetched at first, and by each "Show more". */
const PAGE = 20;
const MAX_GROUPS = 200;
/** Rows of a group shown side by side; the rest are counted. */
const SHOWN_ROWS = 6;

/** N35: a removal of duplicates reviewed group by group, as Splink's and dedupe's review of pairs
 *  (without the learning) and Informatica's survivorship: each group's rows side by side (a column
 *  per row, its cells that differ highlighted), the row kept marked, chosen by a rule (first, most
 *  complete, latest by a date) or by hand; untick a group whose rows are not duplicates and they
 *  all stay; merge the kept row's empty cells from its group only if asked. Every change writes
 *  the step again (one change, one Ctrl+Z); the groups are those of the rule, read once. */
export function DuplicateReview({ suggestion, step, onReplace }: { suggestion: Suggestion; step: Step | undefined; onReplace: Replace }) {
  const context = useReview();
  const [limit, setLimit] = useState(PAGE);
  const fix = context ? withFix(suggestion, context) : null;
  const stepId = fix?.step?.id ?? null;
  const query = useDuplicateGroups(context?.datasetId ?? "", fix && stepId ? stepsForGroups(fix.steps, stepId) : undefined, stepId, limit);
  const base = duplicateOf(step ?? suggestion.step);
  const current = duplicateOf(step);
  if (!base) return null;
  const write = (next: DuplicateStep) => onReplace(suggestion, next as Step);

  return (
    <section aria-label={t("clean.dup.label")} aria-busy={query.isFetching} className="mx-3 mt-1 mb-2 rounded-md border border-line bg-surface-0 sm:ml-9">
      {query.data ? (
        <Summary found={query.data} step={current} />
      ) : (
        <p className="border-b border-line px-3 py-1.5 text-xs text-ink-2">{t("clean.dup.label")}</p>
      )}
      <p className="px-3 pt-1.5 text-xs text-ink-3">{t("clean.dup.hint")}</p>
      <MergeOption step={base} on={Boolean(current?.merge)} onChange={(merge) => write({ ...base, merge })} />
      {query.isError ? (
        <div className="p-3">
          <ErrorNote>{query.error.message}</ErrorNote>
        </div>
      ) : !query.data ? (
        <div className="p-3">
          <Loading compact label={t("clean.dup.loading")} className="h-24 rounded-md" />
        </div>
      ) : query.data.groups.length === 0 ? (
        <p className="px-3 py-3 text-xs text-ink-2">{t("clean.dup.none")}</p>
      ) : (
        <>
          <ol className="max-h-[32rem] divide-y divide-line overflow-y-auto">
            {query.data.groups.map((group, index) => (
              <GroupCard key={group.first} index={index + 1} group={group} found={query.data} step={current} base={base} onWrite={write} />
            ))}
          </ol>
          <footer className="flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-line px-3 py-1.5 text-xs text-ink-3 tabular-nums">
            <span>{t("clean.dup.shown", { shown: formatCount(query.data.groups.length), count: formatCount(query.data.total_groups) })}</span>
            {query.data.groups.length < query.data.total_groups && limit < MAX_GROUPS && (
              <Button variant="ghost" className="-ml-3" disabled={query.isFetching} onClick={() => setLimit(Math.min(MAX_GROUPS, limit + PAGE))}>
                {t("clean.dup.more", { count: formatCount(Math.min(PAGE, query.data.total_groups - query.data.groups.length)) })}
              </Button>
            )}
          </footer>
        </>
      )}
    </section>
  );
}

/** "12 groups · 9 rows removed of 21 · 1 kept whole", said again when it changes. */
function Summary({ found, step }: { found: DuplicateGroups; step: DuplicateStep | null }) {
  const apart = step?.not_duplicates?.length ?? 0;
  const parts = [
    t("clean.dup.groups", { count: formatCount(found.total_groups) }),
    t("clean.dup.removed", { count: formatCount(step ? removedNow(found, step) : 0), total: formatCount(found.total_rows) }),
    apart > 0 && t("clean.dup.apartCount", { count: formatCount(apart) }),
    (step?.survivors?.length ?? 0) > 0 && t("clean.dup.chosenCount", { count: formatCount(step?.survivors?.length ?? 0) }),
  ].filter(Boolean);
  return (
    <header className="border-b border-line px-3 py-1.5">
      <p className="text-xs text-ink-2 tabular-nums" aria-live="polite">
        {parts.join(" · ")}
      </p>
      {!step && <p className="text-xs text-ink-3">{t("clean.dup.notTicked")}</p>}
    </header>
  );
}

/** Survivorship can also merge fields, only when the user asks: the reason says what it does. */
function MergeOption({ step, on, onChange }: { step: DuplicateStep; on: boolean; onChange: (on: boolean) => void }) {
  const id = useId();
  if (!step.keys && !step.groups) return null; // exact copies (B-14): nothing to merge
  return (
    <div className="flex items-start gap-2 px-3 pt-1.5 pb-2">
      <input id={id} type="checkbox" className="mt-0.5 size-3.5 shrink-0 accent-(--accent)" checked={on} aria-describedby={`${id}-why`} onChange={(event) => onChange(event.target.checked)} />
      <div className="min-w-0">
        <label htmlFor={id} className="cursor-pointer text-[13px] text-ink-1">
          {t("clean.dup.merge")}
        </label>
        <p id={`${id}-why`} className="text-xs text-ink-3">
          {t("clean.dup.mergeWhy")}
        </p>
      </div>
    </div>
  );
}

const ruleLabel = (found: DuplicateGroups) =>
  found.keep === "latest" && found.order_by ? t("clean.dup.rule.latest", { column: found.order_by }) : t(`clean.dup.rule.${found.keep}` as MessageKey);

/** One group: its tick (duplicates or not), how its kept row is chosen, and its rows side by side:
 *  a column per row, the cells that differ first and highlighted, the equal ones folded. */
function GroupCard({ index, group, found, step, base, onWrite }: { index: number; group: DuplicateGroup; found: DuplicateGroups; step: DuplicateStep | null; base: DuplicateStep; onWrite: (next: DuplicateStep) => void }) {
  const [equal, setEqual] = useState(false);
  const tickId = useId();
  const tableId = useId();
  const apart = step ? isApart(group, step) : false;
  const survivor = survivorOf(group, step);
  const choice = choiceOf(group, survivor, found.dates);
  const shown = group.rows.slice(0, SHOWN_ROWS);
  if (!shown.some((row) => row.row === survivor)) {
    const kept = group.rows.find((row) => row.row === survivor);
    if (kept) shown[shown.length - 1] = kept;
  }
  const filled = step?.merge && !apart ? mergedCells(group, survivor) : new Map<number, { value: string; from: number }>();
  const differing = new Set(group.differing);
  const columns = found.columns.map((name, at) => ({ name, at }));
  const first = columns.filter((column) => differing.has(column.at) || filled.has(column.at));
  const rest = columns.filter((column) => !differing.has(column.at) && !filled.has(column.at));
  const listed = equal || first.length === 0 ? [...first, ...rest] : first;
  const groupName = `${tableId}-keep`;
  const options: Choice[] = ["rule", "first", "most_complete", ...found.dates.map((date) => `latest:${date}` as const)];
  const label = (option: Choice) => {
    if (option === "rule") return t("clean.dup.choice.rule", { rule: ruleLabel(found) });
    if (option === "pick") return t("clean.dup.choice.pick", { row: formatCount(survivor + 1) });
    if (option.startsWith("latest:")) return t("clean.dup.choice.latest", { column: option.slice("latest:".length) });
    return t(`clean.dup.choice.${option}` as MessageKey);
  };
  const keep = (row: number) => onWrite(withSurvivor(base, group, row));

  return (
    <li className="px-3 py-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <label htmlFor={tickId} className="flex min-h-7 cursor-pointer items-center gap-2 text-[13px] text-ink-1">
          <input id={tickId} type="checkbox" className="size-3.5 shrink-0 accent-(--accent)" aria-label={t("clean.dup.tickLabel", { index: formatCount(index) })} checked={!apart} onChange={(event) => onWrite(withApart(base, group, !event.target.checked))} />
          <span className="font-medium">{t("clean.dup.group", { index: formatCount(index) })}</span>
        </label>
        <span className="text-xs text-ink-2 tabular-nums">
          {apart ? t("clean.dup.apart", { count: formatCount(group.size) }) : t("clean.dup.groupRemoves", { count: formatCount(group.size - 1), total: formatCount(group.size) })}
        </span>
        <label className="ml-auto flex items-center gap-2 text-xs text-ink-2">
          {t("clean.dup.keepBy")}
          <Select
            aria-label={t("clean.dup.keepByLabel", { index: formatCount(index) })}
            value={choice}
            disabled={apart}
            onChange={(event) => {
              const row = rowOfChoice(group, event.target.value as Choice);
              if (row !== null) keep(row);
            }}
            className="max-w-56 text-xs"
          >
            {options.map((option) => (
              <option key={option} value={option}>
                {label(option)}
              </option>
            ))}
            {choice === "pick" && (
              <option value="pick" disabled>
                {label("pick")}
              </option>
            )}
          </Select>
        </label>
      </div>
      <div className="mt-1.5 sm:pl-5.5">
        <table id={tableId} className="w-full table-fixed border-collapse text-[12px]">
          <caption className="sr-only">{t("clean.dup.caption", { index: formatCount(index) })}</caption>
          <thead>
            <tr className="border-b border-line">
              <th scope="col" className="w-28 py-1 pr-2 text-left text-[11px] font-normal text-ink-3 sm:w-40">
                {t("clean.dup.column")}
              </th>
              {shown.map((row) => {
                const kept = apart || row.row === survivor;
                return (
                  <th key={row.row} scope="col" className="py-1 pr-2 text-left align-bottom font-normal">
                    <label className={`flex min-h-7 items-center gap-1.5 ${apart ? "cursor-default" : "cursor-pointer"}`}>
                      <input
                        type="radio"
                        name={groupName}
                        className="size-3.5 shrink-0 accent-(--accent)"
                        aria-label={t("clean.dup.keepThis", { row: formatCount(row.row + 1), index: formatCount(index) })}
                        checked={row.row === survivor}
                        disabled={apart}
                        onChange={() => keep(row.row)}
                      />
                      <span className="text-ink-1 tabular-nums">{t("clean.dup.row", { row: formatCount(row.row + 1) })}</span>
                      <span className={`text-[11px] ${kept ? "text-ok" : "text-ink-3"}`}>{t(kept ? "clean.dup.kept" : "clean.dup.dropped")}</span>
                    </label>
                  </th>
                );
              })}
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
                  {shown.map((row) => {
                    const fill = row.row === survivor ? filled.get(at) : undefined;
                    const cell = row.cells[at] ?? null;
                    const ink = row.row === survivor || apart ? (differs ? "text-ink-1" : "text-ink-2") : "text-ink-3";
                    return (
                      <td key={row.row} className={`py-1 pr-2 align-top font-mono break-words ${ink} ${differs ? "bg-warn/8" : ""}`}>
                        {fill ? (
                          <>
                            <span className="text-ink-3 line-through">{visible(cell)}</span> <span className="text-ink-1">{visible(fill.value)}</span>{" "}
                            <span className="font-sans text-[11px] text-ink-3">{t("clean.dup.mergeFrom", { row: formatCount(fill.from + 1) })}</span>
                          </>
                        ) : (
                          visible(cell)
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="mt-1 flex flex-wrap items-center gap-x-3 text-[11px] text-ink-3">
          {first.length === 0 ? (
            <span>{t("clean.dup.allEqual")}</span>
          ) : (
            rest.length > 0 && (
              <button type="button" aria-expanded={equal} aria-controls={tableId} onClick={() => setEqual(!equal)} className="min-h-7 rounded-sm text-ink-2 underline-offset-2 hover:underline">
                {t(equal ? "clean.dup.hideEqual" : "clean.dup.showEqual", { count: formatCount(rest.length) })}
              </button>
            )
          )}
          {group.size > shown.length && <span>{t("clean.dup.moreRows", { count: formatCount(group.size - shown.length) })}</span>}
          {filled.size > 0 && <span>{t("clean.dup.mergeCount", { count: formatCount(filled.size) })}</span>}
        </p>
      </div>
    </li>
  );
}
