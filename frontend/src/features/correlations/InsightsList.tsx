import { Link } from "@tanstack/react-router";
import { ArrowRight, BarChart3, Copy, Link2, TriangleAlert, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { AnalysisType, Insight } from "../../api/client";
import { t, type MessageKey } from "../../lib/i18n";
import { SegmentedControl } from "../../ui/SegmentedControl";
import { TYPE_META } from "../columns/columnTypes";
import { EvidenceBadge, isShown, type ShownLevels, useShownLevels } from "./evidence";

type Group = { id: string; title: MessageKey; icon: LucideIcon; kinds: Insight["kind"][] };

const GROUPS: Group[] = [
  { id: "redundant", title: "insights.group.redundant", icon: Copy, kinds: ["redundant"] },
  { id: "links", title: "insights.group.links", icon: Link2, kinds: ["correlated", "group_difference", "associated"] },
  { id: "shape", title: "insights.group.shape", icon: BarChart3, kinds: ["skewed", "imbalanced"] },
  { id: "quality", title: "insights.group.quality", icon: TriangleAlert, kinds: ["missing", "codes", "spike"] },
];

/** The sentence with each variable name set in its type's colour. */
export function highlight(text: string, columns: string[], typeOf: (name: string) => AnalysisType | undefined): ReactNode[] {
  const names = [...new Set(columns)].sort((a, b) => b.length - a.length);
  if (!names.length) return [text];
  const escaped = names.map((name) => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const parts = text.split(new RegExp(`(${escaped.join("|")})`, "g"));
  return parts.map((part, index) => {
    if (!names.includes(part)) return part;
    const type = typeOf(part);
    const tone = type ? TYPE_META[type].tone : "text-ink-1";
    return (
      <span key={index} className={`rounded bg-surface-2 px-1 py-px font-mono text-[12px] font-medium ${tone}`}>
        {part}
      </span>
    );
  });
}

type InsightsListProps = {
  datasetId: string;
  insights: Insight[];
  typeOf: (name: string) => AnalysisType | undefined;
};

/** Insights grouped by theme; each sentence links to the chart that backs it. */
export function InsightsList({ datasetId, insights, typeOf }: InsightsListProps) {
  const [shown, setShown] = useShownLevels();
  const links = insights.filter((i) => i.evidence);
  const count = (level: ShownLevels) => links.filter((i) => i.evidence && isShown(i.evidence, level)).length;
  const visible = insights.filter((i) => !i.evidence || isShown(i.evidence, shown));
  const hidden = insights.length - visible.length;
  const groups = GROUPS.map((group) => ({ ...group, items: visible.filter((i) => group.kinds.includes(i.kind)) })).filter(
    (group) => group.items.length,
  );

  return (
    <section aria-label={t("insights.title")} className="space-y-2.5">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="flex items-baseline gap-2">
          <h2 className="text-sm font-semibold">{t("insights.title")}</h2>
          <span className="text-xs text-ink-3 tabular-nums">{visible.length}</span>
        </div>
        {links.length > 0 && (
          <div className="flex flex-wrap items-center gap-2 sm:ml-auto">
            <span className="text-xs text-ink-3">{t("evidence.show")}</span>
            <SegmentedControl
              label={t("evidence.showLabel")}
              value={shown}
              options={[
                { value: "strong", label: t("evidence.showStrong", { count: String(count("strong")) }) },
                { value: "moderate", label: t("evidence.showModerate", { count: String(count("moderate")) }) },
                { value: "weak", label: t("evidence.showAll", { count: String(count("weak")) }) },
              ]}
              onChange={setShown}
            />
          </div>
        )}
      </div>
      {groups.length === 0 ? (
        <p className="rounded-lg border border-line bg-surface-1 px-4 py-3 text-ink-2">{t("insights.none")}</p>
      ) : (
        <div className="grid gap-3 lg:grid-cols-2">
          {groups.map((group) => {
            const Icon = group.icon;
            return (
              <article key={group.id} className="rounded-lg border border-line bg-surface-1">
                <header className="flex items-center gap-2 border-b border-line px-3.5 py-2">
                  <Icon size={14} strokeWidth={1.5} className="text-ink-3" aria-hidden />
                  <h3 className="text-xs font-medium tracking-wide text-ink-2 uppercase">{t(group.title)}</h3>
                  <span className="text-xs text-ink-3 tabular-nums">{group.items.length}</span>
                </header>
                <ul className="divide-y divide-line">
                  {group.items.map((insight) => (
                    <li key={insight.id} className="group flex items-start gap-3 px-3.5 py-2.5">
                      <span
                        aria-hidden
                        className={`mt-2 size-1.5 shrink-0 rounded-full ${insight.severity === "warning" ? "bg-warn" : "bg-accent"}`}
                      />
                      <p className="min-w-0 flex-1 text-[13px] leading-relaxed text-ink-1">
                        {highlight(insight.text, insight.columns, typeOf)}
                      </p>
                      {insight.evidence && (
                        <span className="mt-0.5">
                          <EvidenceBadge level={insight.evidence} />
                        </span>
                      )}
                      <InsightLink datasetId={datasetId} insight={insight} />
                    </li>
                  ))}
                </ul>
              </article>
            );
          })}
        </div>
      )}
      {hidden > 0 && (
        <p className="text-xs text-ink-3">
          {t("evidence.hidden", { count: String(hidden) })}{" "}
          <button type="button" className="text-ink-2 underline-offset-2 hover:text-accent hover:underline" onClick={() => setShown("weak")}>
            {t("evidence.showEverything")}
          </button>
        </p>
      )}
    </section>
  );
}

function InsightLink({ datasetId, insight }: { datasetId: string; insight: Insight }) {
  const className =
    "mt-0.5 inline-flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-ink-3 transition-colors group-hover:text-accent hover:bg-surface-2";
  const content = (
    <>
      {t("insights.view")}
      <ArrowRight size={12} strokeWidth={1.75} aria-hidden />
    </>
  );
  if (insight.link.page === "compare" && insight.link.y) {
    return (
      <Link
        to="/d/$datasetId/compare"
        params={{ datasetId }}
        search={(previous) => ({ ...previous, x: insight.link.x, y: insight.link.y ?? undefined, z: undefined, color: undefined })}
        className={className}
      >
        {content}
      </Link>
    );
  }
  return (
    <Link to="/d/$datasetId/variable/$column" params={{ datasetId, column: insight.link.x }} className={className}>
      {content}
    </Link>
  );
}
