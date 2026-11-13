import { Binary, Check, ChevronDown, Columns3, Redo2, Replace, Rows3, Scale, Type, Undo2, type LucideIcon } from "lucide-react";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { suggestionReason } from "../../lib/i18n/reasons";
import { Button, buttonClass } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { hiddenOf, useAdvanced, type CleanDismissal, type HiddenSuggestion, type Step, type Suggestion } from "./api";
import { autoFacts, byLayer, groupByColumn, groupOf, groupSuggestions, impactOf, linesOf, tickedIn, totals, type Group, type GroupId, type Line } from "./groups";
import { useReview } from "./context";
import { MakeCheckButton } from "./Checks";
import { ColumnBar, columnAnchor, QualityOverview } from "./ColumnQuality";
import { checkOfSuggestion } from "./checkRules";
import { DismissButton, DismissedList } from "./Dismiss";
import { DuplicateReview } from "./DuplicateReview";
import { duplicateOf, isDuplicateFix, isReviewed } from "./duplicates";
import { entityOf, isEntityFix, isEntityReviewed } from "./entities";
import { EntityReview } from "./EntityReview";
import { MapReview } from "./MapReview";
import { moveFocus, reviewKey, rowKeys } from "./keys";
import { MoreChecks } from "./MoreChecks";
import { SeeRows } from "./SeeRows";
import { WhereRows } from "./WhereRows";
import { Tick, visible } from "./parts";
import { isReviewable, readReview, reviewGroups, reviewTick } from "./review";
import { batchTitle, suggestionTitle } from "./titles";

/** A group's icon names its effect on the dataset; it always sits beside its label (UI_UX). */
const ICONS: Record<GroupId, LucideIcon> = {
  auto: Check,
  format: Type,
  values: Replace,
  types: Binary,
  rows: Rows3,
  columns: Columns3,
  decision: Scale,
};

/** Fixes shown at first in a group; the rest behind "Show N more" (CL-06). */
const SHOWN = 8;

export type Toggle = (suggestion: Suggestion, on: boolean) => void;
export type TickMany = (suggestions: Suggestion[], on: boolean) => void;
/** C1: the groups by effect (096, the default) or by column. */
export type GroupBy = "effect" | "column";

/** The Suggestions tab, classified by effect (096): a summary line, then the groups from the
 *  lossless to the judgement calls. A ticked fix is a step of the draft. Empty groups are not
 *  rendered. Advanced, folded, is computed when it is opened (097, C-16) and holds More checks,
 *  folded too, whose methods run one by one (101, C-29): owner's onion rule. `applied` (N03: the
 *  mode applies the reliable fixes): those come first, folded, in "Applied automatically" (N04).
 *  `dismissed` (N05): the dismissals that apply here and what they hide, folded last. */
export function SuggestionsList({
  datasetId,
  suggestions,
  steps,
  applied = false,
  dismissed,
  onToggle,
  onTickMany,
}: {
  datasetId: string;
  suggestions: Suggestion[];
  steps: Step[];
  applied?: boolean;
  dismissed?: { dismissals: CleanDismissal[]; hidden: HiddenSuggestion[] };
  onToggle: Toggle;
  onTickMany: TickMany;
}) {
  const { basic } = byLayer(suggestions);
  const groups = groupSuggestions(basic, applied);
  const [by, setBy] = useState<GroupBy>("effect");
  const known = useAdvanced(datasetId, false).data; // the cache only: never asked before it opens
  const list = useRef<HTMLDivElement>(null);
  useEffect(() => {
    // N42: j / k from one fix to the next, wherever the focus is on the page (not in a field)
    const onKey = (event: KeyboardEvent) => {
      const key = event.key.toLowerCase();
      if ((key !== "j" && key !== "k") || event.shiftKey || !reviewKey(event) || !list.current) return;
      if (moveFocus(list.current, key === "j" ? 1 : -1)) event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return (
    <div ref={list} className="space-y-4">
      {basic.length === 0 ? (
        <p className="rounded-lg border border-line bg-surface-1 px-3 py-4 text-[13px] text-ink-2">{t("clean.layer.empty")}</p>
      ) : (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <Summary groups={groups} total={basic.length} />
            <SegmentedControl
              label={t("clean.groupBy.label")}
              value={by}
              options={[
                { value: "effect", label: t("clean.groupBy.effect") },
                { value: "column", label: t("clean.groupBy.column") },
              ]}
              onChange={setBy}
            />
          </div>
          {by === "column" && <QualityOverview />}
          <Groups groups={groups} items={basic} by={by} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
        </>
      )}
      <Layer
        title={t("clean.layer.advanced")}
        hint={known ? (known.suggestions.length ? t("clean.layer.count", { count: formatCount(known.suggestions.length) }) : t("clean.layer.none")) : `${t("clean.layer.advancedHint")} ${t("clean.layer.advancedOpen")}`}
      >
        <Advanced datasetId={datasetId} by={by} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
      </Layer>
      {dismissed && dismissed.dismissals.length > 0 && <DismissedLayer datasetId={datasetId} dismissals={dismissed.dismissals} hidden={[...dismissed.hidden, ...hiddenOf(known)]} />}
    </div>
  );
}

/** N05: Dismissed, folded last (onion rule): what "Don't suggest again" hides here, and why. */
function DismissedLayer({ datasetId, dismissals, hidden }: { datasetId: string; dismissals: CleanDismissal[]; hidden: HiddenSuggestion[] }) {
  return (
    <Layer title={t("clean.dismissed.title")} hint={t("clean.dismissed.count", { count: formatCount(hidden.length) })}>
      <DismissedList datasetId={datasetId} dismissals={dismissals} hidden={hidden} />
    </Layer>
  );
}

/** Mounted when Advanced opens: its suggestions are asked for then, once (C-16). */
function Advanced({ datasetId, by, steps, onToggle, onTickMany }: { datasetId: string; by: GroupBy; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const advanced = useAdvanced(datasetId, true);
  const found = advanced.data?.suggestions ?? [];
  return (
    <div className="space-y-3 p-3">
      {advanced.isError ? (
        <ErrorNote>{advanced.error.message}</ErrorNote>
      ) : !advanced.data ? (
        <Loading label={t("clean.layer.loadingAdvanced")} className="h-24 rounded-md" compact />
      ) : found.length > 0 ? (
        <Groups groups={groupSuggestions(found)} items={found} by={by} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
      ) : (
        <p className="text-[13px] text-ink-3">{t("clean.layer.none")}</p>
      )}
      <Layer title={t("clean.layer.expert")} hint={t("clean.layer.expertHint")} nested>
        <div className="p-3">
          <MoreChecks datasetId={datasetId} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
        </div>
      </Layer>
    </div>
  );
}

function part(id: GroupId, count: number): string {
  const one = count === 1 && ["values", "types", "decision"].includes(id);
  return t((one ? `clean.summary.${id}.one` : `clean.summary.${id}`) as MessageKey, { count: formatCount(count) });
}

/** "12 fixes: 7 safe formatting · 3 value changes · 2 need your decision". */
function Summary({ groups, total }: { groups: Group[]; total: number }) {
  return (
    <p className="text-[13px] text-ink-2 tabular-nums">
      <span className="font-medium text-ink-1">{total === 1 ? t("clean.summary.fix") : t("clean.summary.fixes", { count: formatCount(total) })}</span>
      {": "}
      {groups.map((group) => part(group.id, group.items.length)).join(" · ")}
    </p>
  );
}

export function Groups({ groups, items, by = "effect", steps, onToggle, onTickMany }: { groups: Group[]; items?: Suggestion[]; by?: GroupBy; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const ticked = new Set(steps.map((step) => step.suggestion));
  const sections: Section[] =
    by === "column"
      ? groupByColumn(items ?? groups.flatMap((group) => group.items)).map(({ column, items: fixes }) => ({
          key: `column:${column ?? ""}`,
          title: column ?? t("clean.groupBy.table"),
          mono: column !== null,
          hint: null,
          icon: column === null ? Rows3 : Columns3,
          warn: false,
          batch: false, // one section per column: nothing to gather
          items: fixes,
          column, // N37: its quality bar under the title
        }))
      : groups.map((group) => ({
          key: group.id,
          title: t(`clean.group.${group.id}` as MessageKey),
          mono: false,
          hint: t(`clean.group.${group.id}.hint` as MessageKey),
          icon: ICONS[group.id],
          warn: group.id === "decision",
          batch: true,
          items: group.items,
        }));
  return (
    <div className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface-1">
      {sections.map((section) =>
        section.key === "auto" ? (
          <AutoSection key={section.key} section={section} ticked={ticked} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
        ) : (
          <GroupSection key={section.key} section={section} ticked={ticked} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
        ),
      )}
    </div>
  );
}

/** `batch`: the same fix on several columns is one line (N39). */
type Section = { key: string; title: string; mono: boolean; hint: string | null; icon: LucideIcon; warn: boolean; batch: boolean; items: Suggestion[]; column?: string | null };

const linesFor = (section: Section): Line[] => (section.batch ? linesOf(section.items) : section.items.map((suggestion) => ({ kind: "one", suggestion })));

function GroupSection({ section, ticked, steps, onToggle, onTickMany }: { section: Section; ticked: Set<string | null>; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const [all, setAll] = useState(false);
  const headingId = useId();
  const Icon = section.icon;
  const tickable = section.items.filter((item) => item.step);
  const count = tickedIn(tickable, steps);
  const full = tickable.length > 0 && count === tickable.length;
  const lines = linesFor(section);
  const shown = all ? lines : lines.slice(0, SHOWN);
  const decision = section.key === "decision";
  return (
    <section aria-labelledby={headingId} id={section.column ? columnAnchor(section.column) : undefined} className="scroll-mt-4">
      <header className="flex flex-wrap items-start gap-x-3 gap-y-2 px-3 pt-3 pb-2">
        <Icon size={16} strokeWidth={1.5} className={`mt-0.5 shrink-0 ${section.warn ? "text-warn" : "text-ink-2"}`} aria-hidden />
        <div className="min-w-0 flex-1 basis-56">
          <h3 id={headingId} className="flex items-baseline gap-2 text-[13px] font-medium text-ink-1">
            <span className={section.mono ? "min-w-0 truncate font-mono" : ""}>{section.title}</span>
            <span className="font-normal text-ink-3 tabular-nums">{formatCount(section.items.length)}</span>
          </h3>
          {section.hint && <p className="mt-0.5 max-w-prose text-xs text-ink-2">{section.hint}</p>}
          {section.column && <ColumnBar column={section.column} withShare className="mt-1" />}
        </div>
        {tickable.length > 0 && (
          <div className="flex items-center gap-3 max-sm:w-full max-sm:justify-between max-sm:pl-7">
            <span className="text-xs text-ink-3 tabular-nums">{t("clean.group.ticked", { ticked: formatCount(count), count: formatCount(tickable.length) })}</span>
            <Button onClick={() => onTickMany(tickable, !full)} aria-label={`${t(full ? "clean.group.untick" : "clean.group.tick")}: ${section.title}`}>
              {t(full ? "clean.group.untick" : "clean.group.tick")}
            </Button>
          </div>
        )}
      </header>
      <ul className="pb-1.5">
        {shown.map((line) =>
          line.kind === "batch" ? (
            <BatchRow key={line.key} items={line.items} ticked={ticked} decision={decision} onToggle={onToggle} onTickMany={onTickMany} />
          ) : (
            <FixRow key={line.suggestion.id} suggestion={line.suggestion} checked={ticked.has(line.suggestion.id)} decision={groupOf(line.suggestion) === "decision"} onToggle={(on) => onToggle(line.suggestion, on)} />
          ),
        )}
      </ul>
      {lines.length > SHOWN && (
        <div className="pr-3 pb-2.5 pl-10">
          <Button variant="ghost" onClick={() => setAll(!all)} aria-expanded={all}>
            {all ? t("clean.group.fewer") : t("clean.group.more", { count: formatCount(lines.length - SHOWN) })}
          </Button>
        </div>
      )}
    </section>
  );
}

/** N04: the reliable fixes applied by default (N03), first and folded: their count and cells in
 *  the heading, "Undo all automatic fixes" beside it; open, each fix with its rule, why it is
 *  reliable and its own Undo (it leaves the draft; Ctrl+Z puts it back). */
function AutoSection({ section, ticked, steps, onToggle, onTickMany }: { section: Section; ticked: Set<string | null>; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const [open, setOpen] = useState(false);
  const headingId = useId();
  const listId = useId();
  const tickable = section.items.filter((item) => item.step);
  const facts = autoFacts(tickable, steps);
  const counts = [
    t("clean.group.auto.applied", { applied: formatCount(facts.applied), count: formatCount(tickable.length) }),
    facts.cells > 0 && t("clean.count.cells", { count: formatCount(facts.cells) }),
    facts.rows > 0 && t("clean.count.rows", { count: formatCount(facts.rows) }),
  ].filter(Boolean);
  const any = facts.applied > 0;
  return (
    <section aria-labelledby={headingId}>
      <header className="flex flex-wrap items-start gap-x-3 gap-y-2 px-3 pt-3 pb-2">
        <button type="button" aria-expanded={open} aria-controls={listId} onClick={() => setOpen(!open)} className="flex min-h-9 min-w-0 flex-1 basis-56 items-start gap-2 text-left">
          <ChevronDown size={15} strokeWidth={1.5} className={`mt-0.5 shrink-0 text-ink-3 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
          <Check size={16} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ok" aria-hidden />
          <span className="min-w-0">
            <span id={headingId} className="flex items-baseline gap-2 text-[13px] font-medium text-ink-1">
              {section.title}
              <span className="font-normal text-ink-3 tabular-nums">{formatCount(section.items.length)}</span>
            </span>
            <span className="mt-0.5 block text-xs text-ink-3 tabular-nums">{counts.join(" · ")}</span>
            {section.hint && <span className="mt-0.5 block max-w-prose text-xs text-ink-2">{section.hint}</span>}
          </span>
        </button>
        {tickable.length > 0 && (
          <div className="flex items-center max-sm:w-full max-sm:justify-end">
            <Button icon={any ? Undo2 : Redo2} onClick={() => onTickMany(tickable, !any)}>
              {t(any ? "clean.auto.undoAll" : "clean.auto.redoAll")}
            </Button>
          </div>
        )}
      </header>
      {open && (
        <ul id={listId} className="pb-1.5">
          {linesFor(section).map((line) =>
            line.kind === "batch" ? (
              <AutoBatchRow key={line.key} items={line.items} ticked={ticked} onToggle={onToggle} onTickMany={onTickMany} />
            ) : (
              <AutoRow key={line.suggestion.id} suggestion={line.suggestion} applied={ticked.has(line.suggestion.id)} onToggle={(on) => onToggle(line.suggestion, on)} />
            ),
          )}
        </ul>
      )}
    </section>
  );
}

/** Why a fix was applied without asking, in the page's language (the measure of N02); an older
 *  server sends none: the engine's reason then says it. */
function reliability(suggestion: Suggestion): string | null {
  const measured = suggestion.measured as number | null | undefined;
  if (measured === null || measured === undefined) return null;
  return measured === 0 ? t("clean.auto.lossless") : t("clean.auto.measured", { n: formatCount(measured) });
}

/** One applied fix: what, its rule and why it is reliable, a before → after sample, its cells or
 *  rows, and Undo (Apply again once undone). */
function AutoRow({ suggestion, applied, onToggle }: { suggestion: Suggestion; applied: boolean; onToggle: (on: boolean) => void }) {
  const [seeing, setSeeing] = useState(false);
  const title = suggestionTitle(suggestion);
  const counts = [
    suggestion.cells > 0 && t("clean.count.cells", { count: formatCount(suggestion.cells) }),
    suggestion.rows > 0 && t("clean.count.rows", { count: formatCount(suggestion.rows) }),
  ].filter(Boolean);
  const example = suggestion.examples[0];
  const why = reliability(suggestion);
  const action = t(applied ? "clean.auto.undo" : "clean.auto.redo");
  const keys = rowKeys({ onTick: () => suggestion.step && onToggle(!applied), onOpen: suggestion.step ? () => setSeeing(true) : undefined });
  return (
    <li {...keys} aria-label={title} className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5 py-1.5 pr-3 pl-10 hover:bg-surface-2 focus-visible:-outline-offset-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,15rem)_auto]">
      <div className="col-start-1 row-start-1 min-w-0">
        <p className="text-[13px]">
          <span className={applied ? "text-ink-1" : "text-ink-3 line-through"}>{title}</span>
          {!applied && <span className="ml-2 text-xs text-ink-3">{t("clean.auto.undone")}</span>}
        </p>
        <p className="mt-0.5 text-xs text-ink-2">
          <span className="font-mono text-ink-3">{suggestion.rule}</span>
          {" · "}
          {why ?? suggestionReason(suggestion)}
        </p>
      </div>
      {example ? (
        <p className="col-start-1 row-start-2 min-w-0 truncate font-mono text-[11px] text-ink-3 sm:col-start-2 sm:row-start-1 sm:mt-0.5" title={`${visible(example.before)} → ${visible(example.after)}`}>
          {visible(example.before)} → {visible(example.after)}
        </p>
      ) : (
        <span className="hidden sm:col-start-2 sm:row-start-1 sm:block" aria-hidden />
      )}
      <div className="col-start-2 row-start-1 flex items-center gap-1.5 text-[11px] whitespace-nowrap text-ink-3 tabular-nums sm:col-start-3">
        <span>{counts.join(" · ")}</span>
        <InfoTip label={`${t("clean.why")}: ${title}`} text={`${suggestion.rule}: ${suggestionReason(suggestion, why)}`} />
        <SeeRows suggestion={suggestion} open={seeing} onOpenChange={setSeeing} />
        <DismissButton suggestion={suggestion} />
        <Button variant="ghost" icon={applied ? Undo2 : Redo2} disabled={!suggestion.step} aria-label={`${action}: ${title}`} onClick={() => onToggle(!applied)}>
          {action}
        </Button>
      </div>
    </li>
  );
}

/** One fix on one compact line: what, a before → after sample, how many cells or rows, and the
 *  rule with its reason behind the tip. A decision also writes its effect and reason (CL-07). */
function FixRow({ suggestion, checked, decision, onToggle }: { suggestion: Suggestion; checked: boolean; decision: boolean; onToggle: (on: boolean) => void }) {
  const id = useId();
  const reviewId = useId();
  const [reviewing, setReviewing] = useState(false);
  const [seeing, setSeeing] = useState(false);
  const context = useReview();
  const counts = [
    suggestion.cells > 0 && t("clean.count.cells", { count: formatCount(suggestion.cells) }),
    suggestion.rows > 0 && t("clean.count.rows", { count: formatCount(suggestion.rows) }),
  ].filter(Boolean);
  const example = suggestion.examples[0];
  // N38: a merge of spellings is reviewed value by value; its tick says how much of it is kept
  const review = context && isReviewable(suggestion) ? context : null;
  // N31: a rule whose finding is a lasting rule of the data (a → b, start ≤ end, a key) can be declared
  const check = context ? checkOfSuggestion(suggestion, "c1") : null;
  const step = review?.steps.find((other) => other.suggestion === suggestion.id);
  const groups = review ? reviewGroups(suggestion) : [];
  // N35: a removal of duplicates is reviewed group by group; a group kept whole: partly kept
  const duplicates = context && !review && isDuplicateFix(suggestion) ? context : null;
  const dupStep = duplicates ? duplicateOf(duplicates.steps.find((other) => other.suggestion === suggestion.id)) : null;
  // N36: an entity number is reviewed cluster by cluster, in the same panel; a cluster kept apart: partly kept
  const entities = context && !review && !duplicates && isEntityFix(suggestion) ? context : null;
  const entStep = entities ? entityOf(entities.steps.find((other) => other.suggestion === suggestion.id)) : null;
  const partly = Boolean(dupStep?.not_duplicates?.length || entStep?.not_entities?.length);
  const state = review ? reviewTick(readReview(groups, step)) : partly ? "mixed" : checked;
  // a merge (or a removal of duplicates, an entity number) partly kept, ticked again: the whole fix, as found
  const reviewer = review ?? duplicates ?? entities;
  const reviewed = review ? step : (dupStep ?? entStep);
  const tick = (on: boolean) => (reviewer && on && reviewed ? reviewer.onReplace(suggestion, suggestion.step) : onToggle(on));
  const keys = rowKeys({
    onTick: () => suggestion.step && tick(state !== true),
    onOpen: review || duplicates || entities ? () => setReviewing(!reviewing) : suggestion.step ? () => setSeeing(true) : undefined,
  });
  return (
    <li>
      <div {...keys} role="group" aria-label={suggestionTitle(suggestion)} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5 px-3 py-1.5 hover:bg-surface-2 focus-visible:-outline-offset-2 sm:grid-cols-[auto_minmax(0,1fr)_minmax(0,15rem)_auto]">
        <Tick
          id={id}
          state={state}
          className="col-start-1 row-start-1 disabled:opacity-50"
          disabled={!suggestion.step}
          onChange={tick}
        />
        <div className="col-start-2 row-start-1 min-w-0">
          <label htmlFor={id} className="cursor-pointer text-[13px] text-ink-1">
            {suggestionTitle(suggestion)}
          </label>
          {review && (
            <BatchFold
              open={reviewing}
              listId={reviewId}
              label={groups.length === 1 ? t("clean.review.openOne") : t("clean.review.open", { count: formatCount(groups.length) })}
              status={state === "mixed" ? t("clean.review.partly") : ""}
              onOpen={() => setReviewing(!reviewing)}
            />
          )}
          {duplicates && (
            <BatchFold
              open={reviewing}
              listId={reviewId}
              label={t("clean.dup.open")}
              status={state === "mixed" ? t("clean.review.partly") : isReviewed(dupStep) ? t("clean.dup.reviewed") : ""}
              onOpen={() => setReviewing(!reviewing)}
            />
          )}
          {entities && (
            <BatchFold
              open={reviewing}
              listId={reviewId}
              label={t("clean.ent.open")}
              status={state === "mixed" ? t("clean.review.partly") : isEntityReviewed(entStep) ? t("clean.dup.reviewed") : ""}
              onOpen={() => setReviewing(!reviewing)}
            />
          )}
          {decision && (
            <p className="mt-0.5 text-xs text-ink-2">
              <span className="text-ink-1">{t(`clean.impact.${impactOf(suggestion)}` as MessageKey)}</span>
              {" · "}
              {suggestionReason(suggestion, reliability(suggestion))}
            </p>
          )}
        </div>
        {example ? (
          <p className="col-start-2 row-start-2 min-w-0 truncate font-mono text-[11px] text-ink-3 sm:col-start-3 sm:row-start-1 sm:mt-0.5" title={`${visible(example.before)} → ${visible(example.after)}`}>
            {visible(example.before)} → {visible(example.after)}
          </p>
        ) : (
          <span className="hidden sm:col-start-3 sm:row-start-1 sm:block" aria-hidden />
        )}
        <div className="col-start-3 row-start-1 flex items-center gap-1.5 pt-px text-[11px] whitespace-nowrap text-ink-3 tabular-nums sm:col-start-4">
          <span>{counts.join(" · ")}</span>
          <InfoTip label={`${t("clean.why")}: ${suggestionTitle(suggestion)}`} text={`${suggestion.rule}: ${suggestionReason(suggestion, reliability(suggestion))}`} />
          <SeeRows suggestion={suggestion} open={seeing} onOpenChange={setSeeing} />
          <WhereRows suggestion={suggestion} />
          {context && check && <MakeCheckButton datasetId={context.datasetId} check={check} iconOnly />}
          <DismissButton suggestion={suggestion} />
        </div>
      </div>
      {review && reviewing && (
        <div id={reviewId}>
          <MapReview suggestion={suggestion} step={step} onReplace={review.onReplace} />
        </div>
      )}
      {duplicates && reviewing && (
        <div id={reviewId}>
          <DuplicateReview suggestion={suggestion} step={dupStep ?? undefined} onReplace={duplicates.onReplace} />
        </div>
      )}
      {entities && reviewing && (
        <div id={reviewId}>
          <EntityReview suggestion={suggestion} step={entStep ?? undefined} onReplace={entities.onReplace} />
        </div>
      )}
    </li>
  );
}

/** "1,234 cells · 6 rows" (what is shown when nothing is). */
function countsOf({ cells, rows }: { cells: number; rows: number }): string {
  return [cells > 0 && t("clean.count.cells", { count: formatCount(cells) }), rows > 0 && t("clean.count.rows", { count: formatCount(rows) })].filter(Boolean).join(" · ");
}

/** The columns of a batch for its tip: "city, region, state and 27 more". */
function columnsList(items: Suggestion[]): string {
  const names = items.map((item) => item.columns[0] ?? "").filter(Boolean);
  const shown = names.slice(0, 5).join(", ");
  return names.length > 5 ? t("clean.batch.andMore", { columns: shown, count: formatCount(names.length - 5) }) : shown;
}

/** The fold of a batch: "30 columns", open or not, beside what is ticked. */
function BatchFold({ open, listId, label, status, onOpen }: { open: boolean; listId: string; label: string; status: string; onOpen: () => void }) {
  return (
    <p className="mt-0.5 flex flex-wrap items-center gap-x-2 text-xs text-ink-3 tabular-nums">
      <button type="button" aria-expanded={open} aria-controls={listId} onClick={onOpen} className={buttonClass("ghost", "-ml-3 text-xs")}>
        <ChevronDown size={14} strokeWidth={1.5} className={`shrink-0 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
        {label}
      </button>
      {status && <span>{status}</span>}
    </p>
  );
}

/** N39: the same fix on several columns, one line: one tick for all (mixed when some are), the
 *  cells summed; open, each column's own line, ticked one by one. */
function BatchRow({ items, ticked, decision, onToggle, onTickMany }: { items: Suggestion[]; ticked: Set<string | null>; decision: boolean; onToggle: Toggle; onTickMany: TickMany }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const listId = useId();
  const count = items.filter((item) => ticked.has(item.id)).length;
  const state = count === 0 ? false : count === items.length ? true : "mixed";
  const title = batchTitle(items);
  const first = items[0]!;
  const example = first.examples[0];
  const keys = rowKeys({ onTick: () => onTickMany(items, state !== true), onOpen: () => setOpen(!open) });
  return (
    <li>
      <div {...keys} role="group" aria-label={title} className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5 px-3 py-1.5 hover:bg-surface-2 focus-visible:-outline-offset-2 sm:grid-cols-[auto_minmax(0,1fr)_minmax(0,15rem)_auto]">
        <Tick id={id} state={state} className="col-start-1 row-start-1" onChange={(on) => onTickMany(items, on)} />
        <div className="col-start-2 row-start-1 min-w-0">
          <label htmlFor={id} className="cursor-pointer text-[13px] text-ink-1">
            {title}
          </label>
          <BatchFold open={open} listId={listId} label={t("clean.batch.columns", { count: formatCount(items.length) })} status={t("clean.group.ticked", { ticked: formatCount(count), count: formatCount(items.length) })} onOpen={() => setOpen(!open)} />
          {decision && (
            <p className="mt-0.5 text-xs text-ink-2">
              <span className="text-ink-1">{t(`clean.impact.${impactOf(first)}` as MessageKey)}</span>
              {" · "}
              {t("clean.batch.decision")}
            </p>
          )}
        </div>
        {example ? (
          <p className="col-start-2 row-start-2 min-w-0 truncate font-mono text-[11px] text-ink-3 sm:col-start-3 sm:row-start-1 sm:mt-0.5" title={`${visible(example.before)} → ${visible(example.after)}`}>
            {visible(example.before)} → {visible(example.after)}
          </p>
        ) : (
          <span className="hidden sm:col-start-3 sm:row-start-1 sm:block" aria-hidden />
        )}
        <div className="col-start-3 row-start-1 flex items-center gap-1.5 pt-px text-[11px] whitespace-nowrap text-ink-3 tabular-nums sm:col-start-4">
          <span>{countsOf(totals(items))}</span>
          <InfoTip label={`${t("clean.why")}: ${title}`} text={`${first.rule}: ${t("clean.batch.why", { count: formatCount(items.length), columns: columnsList(items) })}`} />
        </div>
      </div>
      {open && (
        <ul id={listId} aria-label={title} className="mb-1 ml-8 border-l border-line">
          {items.map((suggestion) => (
            <FixRow key={suggestion.id} suggestion={suggestion} checked={ticked.has(suggestion.id)} decision={decision} onToggle={(on) => onToggle(suggestion, on)} />
          ))}
        </ul>
      )}
    </li>
  );
}

/** N39 in Applied automatically: the same reliable fix on several columns, one line with one
 *  Undo (Apply again once all are undone); open, each column with its own Undo. */
function AutoBatchRow({ items, ticked, onToggle, onTickMany }: { items: Suggestion[]; ticked: Set<string | null>; onToggle: Toggle; onTickMany: TickMany }) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  const applied = items.filter((item) => ticked.has(item.id)).length;
  const title = batchTitle(items);
  const first = items[0]!;
  const example = first.examples[0];
  const any = applied > 0;
  const action = t(any ? "clean.auto.undo" : "clean.auto.redo");
  const keys = rowKeys({ onTick: () => onTickMany(items, !any), onOpen: () => setOpen(!open) });
  return (
    <li>
      <div {...keys} role="group" aria-label={title} className="grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5 py-1.5 pr-3 pl-10 hover:bg-surface-2 focus-visible:-outline-offset-2 sm:grid-cols-[minmax(0,1fr)_minmax(0,15rem)_auto]">
        <div className="col-start-1 row-start-1 min-w-0">
          <p className={`text-[13px] ${any ? "text-ink-1" : "text-ink-3 line-through"}`}>{title}</p>
          <p className="mt-0.5 text-xs text-ink-2">
            <span className="font-mono text-ink-3">{first.rule}</span>
            {" · "}
            {reliability(first) ?? suggestionReason(first)}
          </p>
          <BatchFold open={open} listId={listId} label={t("clean.batch.columns", { count: formatCount(items.length) })} status={t("clean.group.auto.applied", { applied: formatCount(applied), count: formatCount(items.length) })} onOpen={() => setOpen(!open)} />
        </div>
        {example ? (
          <p className="col-start-1 row-start-2 min-w-0 truncate font-mono text-[11px] text-ink-3 sm:col-start-2 sm:row-start-1 sm:mt-0.5" title={`${visible(example.before)} → ${visible(example.after)}`}>
            {visible(example.before)} → {visible(example.after)}
          </p>
        ) : (
          <span className="hidden sm:col-start-2 sm:row-start-1 sm:block" aria-hidden />
        )}
        <div className="col-start-2 row-start-1 flex items-center gap-1.5 text-[11px] whitespace-nowrap text-ink-3 tabular-nums sm:col-start-3">
          <span>{countsOf(totals(items))}</span>
          <InfoTip label={`${t("clean.why")}: ${title}`} text={`${first.rule}: ${t("clean.batch.why", { count: formatCount(items.length), columns: columnsList(items) })}`} />
          <Button variant="ghost" icon={any ? Undo2 : Redo2} aria-label={`${action}: ${title}`} onClick={() => onTickMany(items, !any)}>
            {action}
          </Button>
        </div>
      </div>
      {open && (
        <ul id={listId} aria-label={title} className="mb-1 ml-8 border-l border-line">
          {items.map((suggestion) => (
            <AutoRow key={suggestion.id} suggestion={suggestion} applied={ticked.has(suggestion.id)} onToggle={(on) => onToggle(suggestion, on)} />
          ))}
        </ul>
      )}
    </li>
  );
}

function Layer({ title, hint, nested = false, children }: { title: string; hint: string; nested?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  // once opened, kept mounted (hidden when folded): a check running inside keeps its result (101)
  const [seen, setSeen] = useState(false);
  const id = useId();
  return (
    <section className={`rounded-lg border border-line ${nested ? "bg-surface-0" : "bg-surface-1"}`}>
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => { setOpen(!open); setSeen(true); }} className="flex min-h-11 w-full min-w-0 items-center gap-2 px-3 py-2 text-left">
        <ChevronDown size={15} strokeWidth={1.5} className={`shrink-0 text-ink-3 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
        <span className="shrink-0 text-[13px] font-medium text-ink-1">{title}</span>
        <span className="min-w-0 truncate text-xs text-ink-3">{hint}</span>
      </button>
      {seen && (
        <div id={id} hidden={!open} className="border-t border-line">
          {children}
        </div>
      )}
    </section>
  );
}
