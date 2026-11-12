import { Binary, ChevronDown, Columns3, Replace, Rows3, Scale, Type, type LucideIcon } from "lucide-react";
import { useId, useState, type ReactNode } from "react";

import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { InfoTip } from "../../ui/InfoTip";
import type { Step, Suggestion } from "./api";
import { byLayer, groupSuggestions, impactOf, tickedIn, type Group, type GroupId } from "./groups";

/** A group's icon names its effect on the dataset; it always sits beside its label (UI_UX). */
const ICONS: Record<GroupId, LucideIcon> = {
  format: Type,
  values: Replace,
  types: Binary,
  rows: Rows3,
  columns: Columns3,
  decision: Scale,
};

/** Fixes shown at first in a group; the rest behind "Show N more" (CL-06). */
const SHOWN = 8;

type Toggle = (suggestion: Suggestion, on: boolean) => void;
type TickMany = (suggestions: Suggestion[], on: boolean) => void;

/** The Suggestions tab, classified by effect (096): a summary line, then the groups from the
 *  lossless to the judgement calls. A ticked fix is a step of the draft. Empty groups and empty
 *  layers are not rendered (owner's request). */
export function SuggestionsList({ suggestions, steps, onToggle, onTickMany }: { suggestions: Suggestion[]; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const { basic, advanced, expert } = byLayer(suggestions);
  const groups = groupSuggestions(basic);
  return (
    <div className="space-y-4">
      {basic.length === 0 ? (
        <p className="rounded-lg border border-line bg-surface-1 px-3 py-4 text-[13px] text-ink-2">{t("clean.layer.empty")}</p>
      ) : (
        <>
          <Summary groups={groups} total={basic.length} />
          <Groups groups={groups} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
        </>
      )}
      {/* Owner's onion rule: Advanced folded, Advanced › More folded inside it; neither is shown
          when it has nothing (CL-05). */}
      {advanced.length + expert.length > 0 && (
        <Layer title={t("clean.layer.advanced")} hint={advanced.length > 0 ? t("clean.layer.count", { count: formatCount(advanced.length) }) : t("clean.layer.advancedHint")}>
          {advanced.length > 0 ? (
            <div className="p-3">
              <Groups groups={groupSuggestions(advanced)} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
            </div>
          ) : (
            <p className="px-3 py-3 text-[13px] text-ink-3">{t("clean.layer.none")}</p>
          )}
          {expert.length > 0 && (
            <div className="px-3 pb-3">
              <Layer title={t("clean.layer.expert")} hint={t("clean.layer.count", { count: formatCount(expert.length) })} nested>
                <div className="p-3">
                  <Groups groups={groupSuggestions(expert)} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
                </div>
              </Layer>
            </div>
          )}
        </Layer>
      )}
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

function Groups({ groups, steps, onToggle, onTickMany }: { groups: Group[]; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const ticked = new Set(steps.map((step) => step.suggestion));
  return (
    <div className="divide-y divide-line overflow-hidden rounded-lg border border-line bg-surface-1">
      {groups.map((group) => (
        <GroupSection key={group.id} group={group} ticked={ticked} steps={steps} onToggle={onToggle} onTickMany={onTickMany} />
      ))}
    </div>
  );
}

function GroupSection({ group, ticked, steps, onToggle, onTickMany }: { group: Group; ticked: Set<string | null>; steps: Step[]; onToggle: Toggle; onTickMany: TickMany }) {
  const [all, setAll] = useState(false);
  const headingId = useId();
  const Icon = ICONS[group.id];
  const tickable = group.items.filter((item) => item.step);
  const count = tickedIn(tickable, steps);
  const full = tickable.length > 0 && count === tickable.length;
  const shown = all ? group.items : group.items.slice(0, SHOWN);
  const decision = group.id === "decision";
  return (
    <section aria-labelledby={headingId}>
      <header className="flex flex-wrap items-start gap-x-3 gap-y-2 px-3 pt-3 pb-2">
        <Icon size={16} strokeWidth={1.5} className={`mt-0.5 shrink-0 ${decision ? "text-warn" : "text-ink-2"}`} aria-hidden />
        <div className="min-w-0 flex-1 basis-56">
          <h3 id={headingId} className="flex items-baseline gap-2 text-[13px] font-medium text-ink-1">
            {t(`clean.group.${group.id}` as MessageKey)}
            <span className="font-normal text-ink-3 tabular-nums">{formatCount(group.items.length)}</span>
          </h3>
          <p className="mt-0.5 max-w-prose text-xs text-ink-2">{t(`clean.group.${group.id}.hint` as MessageKey)}</p>
        </div>
        {tickable.length > 0 && (
          <div className="flex items-center gap-3 max-sm:w-full max-sm:justify-between max-sm:pl-7">
            <span className="text-xs text-ink-3 tabular-nums">{t("clean.group.ticked", { ticked: formatCount(count), count: formatCount(tickable.length) })}</span>
            <Button onClick={() => onTickMany(tickable, !full)} aria-label={`${t(full ? "clean.group.untick" : "clean.group.tick")}: ${t(`clean.group.${group.id}` as MessageKey)}`}>
              {t(full ? "clean.group.untick" : "clean.group.tick")}
            </Button>
          </div>
        )}
      </header>
      <ul className="pb-1.5">
        {shown.map((suggestion) => (
          <FixRow key={suggestion.id} suggestion={suggestion} checked={ticked.has(suggestion.id)} decision={decision} onToggle={(on) => onToggle(suggestion, on)} />
        ))}
      </ul>
      {group.items.length > SHOWN && (
        <div className="pr-3 pb-2.5 pl-10">
          <Button variant="ghost" onClick={() => setAll(!all)} aria-expanded={all}>
            {all ? t("clean.group.fewer") : t("clean.group.more", { count: formatCount(group.items.length - SHOWN) })}
          </Button>
        </div>
      )}
    </section>
  );
}

/** An example value as it is: empty in words, blank as "", spaces at the ends made visible. */
function visible(value: string | null): string {
  if (value === null) return t("clean.empty");
  if (value.trim() === "") return `"${value}"`;
  return value.replace(/^ +| +$/g, (spaces) => "·".repeat(spaces.length));
}

/** One fix on one compact line: what, a before → after sample, how many cells or rows, and the
 *  rule with its reason behind the tip. A decision also writes its effect and reason (CL-07). */
function FixRow({ suggestion, checked, decision, onToggle }: { suggestion: Suggestion; checked: boolean; decision: boolean; onToggle: (on: boolean) => void }) {
  const id = useId();
  const counts = [
    suggestion.cells > 0 && t("clean.count.cells", { count: formatCount(suggestion.cells) }),
    suggestion.rows > 0 && t("clean.count.rows", { count: formatCount(suggestion.rows) }),
  ].filter(Boolean);
  const example = suggestion.examples[0];
  return (
    <li className="grid grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-x-3 gap-y-0.5 px-3 py-1.5 hover:bg-surface-2 sm:grid-cols-[auto_minmax(0,1fr)_minmax(0,15rem)_auto]">
      <input
        id={id}
        type="checkbox"
        className="col-start-1 row-start-1 mt-0.5 size-3.5 shrink-0 accent-(--accent)"
        checked={checked}
        disabled={!suggestion.step}
        onChange={(event) => onToggle(event.target.checked)}
      />
      <div className="col-start-2 row-start-1 min-w-0">
        <label htmlFor={id} className="cursor-pointer text-[13px] text-ink-1">
          {suggestion.title}
        </label>
        {decision && (
          <p className="mt-0.5 text-xs text-ink-2">
            <span className="text-ink-1">{t(`clean.impact.${impactOf(suggestion)}` as MessageKey)}</span>
            {" · "}
            {suggestion.reason}
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
        <InfoTip label={`${t("clean.why")}: ${suggestion.title}`} text={`${suggestion.rule}: ${suggestion.reason}`} />
      </div>
    </li>
  );
}

function Layer({ title, hint, nested = false, children }: { title: string; hint: string; nested?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <section className={`rounded-lg border border-line ${nested ? "bg-surface-0" : "bg-surface-1"}`}>
      <button type="button" aria-expanded={open} aria-controls={id} onClick={() => setOpen(!open)} className="flex min-h-11 w-full min-w-0 items-center gap-2 px-3 py-2 text-left">
        <ChevronDown size={15} strokeWidth={1.5} className={`shrink-0 text-ink-3 transition-transform ${open ? "" : "-rotate-90"}`} aria-hidden />
        <span className="shrink-0 text-[13px] font-medium text-ink-1">{title}</span>
        <span className="min-w-0 truncate text-xs text-ink-3">{hint}</span>
      </button>
      {open && (
        <div id={id} className="border-t border-line">
          {children}
        </div>
      )}
    </section>
  );
}
