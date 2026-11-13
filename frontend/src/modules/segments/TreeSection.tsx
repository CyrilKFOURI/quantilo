import { TriangleAlert } from "lucide-react";
import { useState } from "react";

import { useInterpretAvailable } from "../../api/hooks";
import { useFilters } from "../../features/filters/useFilters";
import { Folded } from "../../features/text/Folded";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Cell } from "../kpi/Customize";
import { SurrogateTree } from "../ml/ExplainAdvanced";
import { useOptimalSegmentTree, useSegmentTree, type SegmentRun, type SegmentTree, type SegmentTreeRules } from "./api";

const pct = (share: number | null) => (share === null ? "—" : `${Math.round(share * 100)} %`);

/** 124, "Advanced · Describe segments with a tree": one sparse tree (the CART of 122) that predicts
 *  each row's segment. Folded, and nothing is asked before its button (onion rule). */
export function TreeSection({ datasetId, run }: { datasetId: string; run: SegmentRun }) {
  return (
    <Folded title={t("segments.tree.advanced")} hint={t("segments.tree.hint")}>
      <TreeBody datasetId={datasetId} run={run} />
    </Folded>
  );
}

function TreeBody({ datasetId, run }: { datasetId: string; run: SegmentRun }) {
  const [asked, setAsked] = useState(false);
  const [depth, setDepth] = useState<number | null>(null);
  const [leaves, setLeaves] = useState<number | null>(null);
  const found = useSegmentTree(datasetId, run.id, depth, leaves, asked);
  if (!asked)
    return (
      <div className="space-y-2">
        <p className="max-w-prose text-xs text-ink-2">{t("segments.tree.intro")}</p>
        <Button variant="secondary" onClick={() => setAsked(true)}>
          {t("segments.tree.run")}
        </Button>
      </div>
    );
  const shownDepth = depth ?? found.data?.depth ?? 3;
  const shownLeaves = leaves ?? found.data?.max_leaves ?? 8;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-[repeat(3,minmax(0,1fr))]">
        <Cell label={t("segments.tree.depth")} why={t("segments.tree.depthWhy")} changed={depth !== null} onReset={() => setDepth(null)}>
          <Select aria-label={t("segments.tree.depth")} value={shownDepth} onChange={(e) => setDepth(Number(e.target.value))} className="w-full text-xs">
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
        <Cell label={t("segments.tree.leaves")} why={t("segments.tree.leavesWhy")} changed={leaves !== null} onReset={() => setLeaves(null)}>
          <Select aria-label={t("segments.tree.leaves")} value={shownLeaves} onChange={(e) => setLeaves(Number(e.target.value))} className="w-full text-xs">
            {[2, 3, 4, 6, 8, 10, 12, 16, 20].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </Select>
        </Cell>
      </div>
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !found.data ? (
        <Loading label={t("segments.tree.loading")} compact className="h-32 rounded-md" />
      ) : (
        <div className={found.isPlaceholderData ? "opacity-60 transition-opacity" : ""} aria-busy={found.isFetching}>
          <Described tree={found.data} datasetId={datasetId} />
          <OptimalMore datasetId={datasetId} run={run} tree={found.data} />
        </div>
      )}
    </div>
  );
}

/** R-SGT-05 (124 bis): a rule's conditions added to the page filters, every page then reads the
 *  rows it describes; the note says when empty numbers are left out. */
function RuleFilters({ segment, datasetId }: { segment: SegmentTreeRules; datasetId: string }) {
  const filters = useFilters(datasetId);
  const [applied, setApplied] = useState<string | null>(null);
  if (!segment.filters?.length) return null;
  return (
    <ul className="space-y-1">
      {segment.filters.map((rule) => (
        <li key={rule.rule} className="flex flex-wrap items-center gap-2 text-xs">
          <button
            type="button"
            className="rounded-md border border-line px-2 py-1 text-ink-1 hover:bg-surface-2 coarse:min-h-11"
            aria-label={t("segments.tree.filterRule", { rule: rule.rule })}
            onClick={() => {
              filters.set([...filters.conditions, ...rule.conditions]);
              setApplied(rule.rule);
            }}
          >
            {t("segments.tree.filter")}
          </button>
          <span className="min-w-0 truncate font-mono text-[11px] text-ink-2">{rule.rule}</span>
          {applied === rule.rule && <span className="text-ink-3">{t("segments.tree.filtered")}</span>}
          {rule.note && <span className="w-full text-[11px] text-ink-3">{rule.note}</span>}
        </li>
      ))}
    </ul>
  );
}

/** R-SGT-06 (124 bis): under More, GOSDT's optimal tree of one segment against the others;
 *  greyed with its reason when the `interpret` extra cannot run here (R-XAI-10). */
function OptimalMore({ datasetId, run, tree }: { datasetId: string; run: SegmentRun; tree: SegmentTree }) {
  const [open, setOpen] = useState(false);
  return (
    <details className="mt-4 border-t border-line pt-3 text-xs" onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className="cursor-pointer text-[13px] text-accent select-none hover:underline">{t("segments.tree.more")}</summary>
      {open && <OptimalBody datasetId={datasetId} run={run} tree={tree} />}
    </details>
  );
}

function OptimalBody({ datasetId, run, tree }: { datasetId: string; run: SegmentRun; tree: SegmentTree }) {
  const available = useInterpretAvailable(true);
  const [chosen, setChosen] = useState<number>(tree.segments[0]?.segment ?? 0);
  const [asked, setAsked] = useState<number | null>(null);
  const found = useOptimalSegmentTree(datasetId, run.id, asked, 3);
  const gosdt = available.data?.find((a) => a.key === "gosdt");
  if (available.isError) return <ErrorNote>{available.error.message}</ErrorNote>;
  if (!available.data) return <Loading label={t("ml.scorecard.checking")} compact className="mt-2 h-12 rounded-md" />;
  return (
    <div className="mt-2 space-y-2">
      <p className="max-w-prose text-ink-2">{t("segments.tree.optimalHint")}</p>
      <div className="flex flex-wrap items-center gap-2">
        <Select aria-label={t("segments.tree.optimalSegment")} value={chosen} onChange={(e) => setChosen(Number(e.target.value))} className="h-8 max-w-56 text-xs">
          {tree.segments.map((s) => (
            <option key={s.segment} value={s.segment}>
              {s.name}
            </option>
          ))}
        </Select>
        <button
          type="button"
          disabled={!gosdt?.available}
          title={gosdt?.available ? undefined : (gosdt?.reason ?? undefined)}
          onClick={() => setAsked(chosen)}
          className="rounded-md border border-line px-3 py-1.5 text-xs text-ink-1 hover:bg-surface-2 disabled:cursor-not-allowed disabled:text-ink-3 coarse:min-h-11"
        >
          {t("segments.tree.optimalRun")}
        </button>
      </div>
      {!gosdt?.available && <p className="max-w-prose text-ink-2">{gosdt?.reason}</p>}
      {!gosdt?.available && gosdt?.install && (
        <p className="text-ink-3">
          {t("ml.scorecard.install")} <code className="font-mono text-ink-2">{gosdt.install}</code>
        </p>
      )}
      {asked !== null &&
        (found.isError ? (
          <ErrorNote>{found.error.message}</ErrorNote>
        ) : !found.data ? (
          <Loading label={t("ml.scorecard.solving")} compact className="h-24 rounded-md" />
        ) : (
          <div className="space-y-2">
            <p className="max-w-prose text-[13px] text-ink-1">{found.data.sentence}</p>
            <SurrogateTree nodes={found.data.nodes} label={t("segments.tree.optimalLabel")} />
          </div>
        ))}
    </div>
  );
}

function Described({ tree, datasetId }: { tree: SegmentTree; datasetId: string }) {
  return (
    <div className="space-y-4">
      <div className="space-y-1">
        <p className="text-[15px] font-semibold tabular-nums text-ink-1">{t("segments.tree.fidelity", { share: pct(tree.fidelity) })}</p>
        <p className="max-w-prose text-[13px] text-ink-1">{tree.sentence}</p>
        {tree.warning && (
          <p role="note" className="flex max-w-prose items-start gap-2 border-l-2 border-warn pl-3 text-xs text-ink-1">
            <TriangleAlert size={13} strokeWidth={1.75} className="mt-0.5 shrink-0 text-warn" aria-hidden />
            <span>{tree.warning}</span>
          </p>
        )}
      </div>
      <div className="grid gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-2 xl:grid-cols-3">
        {tree.segments.map((segment) => (
          <article key={segment.segment} className="min-w-0 space-y-1.5 bg-surface-1 p-3" aria-label={segment.name}>
            <header className="flex items-center gap-1.5 text-[13px] font-medium text-ink-1">
              <span
                className="size-2.5 shrink-0 rounded-full"
                style={{ background: segment.segment < 0 ? "var(--ink-3)" : `var(--chart-${(segment.segment % 8) + 1})` }}
                aria-hidden
              />
              <span className="min-w-0 truncate">{segment.name}</span>
            </header>
            {segment.rules.map((rule) => (
              <p key={rule.text} className="font-mono text-[11px] break-words text-ink-1">
                {rule.text}
                <span className="font-sans text-ink-3"> · {t("segments.tree.ruleShare", { rows: rule.rows.toLocaleString("en-US"), share: pct(rule.precision) })}</span>
              </p>
            ))}
            <p className="text-xs text-ink-2">{segment.text}</p>
            <RuleFilters segment={segment} datasetId={datasetId} />
          </article>
        ))}
      </div>
      <SurrogateTree nodes={tree.nodes} label={t("segments.tree.label")} />
      <p className="max-w-prose text-xs text-ink-3">{t("segments.tree.rows", { learnt: tree.learnt.toLocaleString("en-US"), checked: tree.checked.toLocaleString("en-US") })}</p>
    </div>
  );
}
