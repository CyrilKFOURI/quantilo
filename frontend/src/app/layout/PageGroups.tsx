import { Link, useParams } from "@tanstack/react-router";
import { Lock } from "lucide-react";
import type { ReactNode } from "react";

import { t, type MessageKey } from "../../lib/i18n";

/** 046: pages that answer the same question share one entry in the sidebar and a row of tabs. */
export type PageGroup = "overview" | "relationships" | "ml";

type Tab = { labelKey: MessageKey; to: string; locked?: boolean };

export const GROUPS: Record<PageGroup, Tab[]> = {
  overview: [
    { labelKey: "tab.summary", to: "/d/$datasetId" },
    { labelKey: "nav.columns", to: "/d/$datasetId/columns" },
    { labelKey: "nav.data", to: "/d/$datasetId/data" },
    { labelKey: "nav.changes", to: "/d/$datasetId/changes" },
  ],
  relationships: [
    { labelKey: "tab.twoVariables", to: "/d/$datasetId/compare" },
    { labelKey: "nav.correlations", to: "/d/$datasetId/correlations" },
    { labelKey: "tab.regression", to: "/d/$datasetId/regression" },
  ],
  ml: [
    // Both run in the browser version too (102: scikit-learn models only).
    { labelKey: "tab.predict", to: "/d/$datasetId/ml" },
    { labelKey: "module.segments.label", to: "/d/$datasetId/segments" },
  ],
};

/** The group a page of the open dataset belongs to, from its address. */
export function groupOf(pathname: string): PageGroup | null {
  const match = /^\/d\/[^/]+(\/[^/]*)?\/?$/.exec(pathname);
  if (!match) return null;
  const rest = match[1] ?? "";
  if (rest === "" || rest === "/" || rest === "/columns" || rest === "/data" || rest === "/changes") return "overview";
  if (rest === "/compare" || rest === "/correlations" || rest === "/regression") return "relationships";
  if (rest === "/ml" || rest === "/segments") return "ml";
  return null;
}

/** Where the group's sidebar entry leads: its first tab the build can run. */
export function groupEntry(group: PageGroup): string {
  const tabs = GROUPS[group];
  return (tabs.find((tab) => !tab.locked) ?? tabs[0]!).to;
}

/** A page of a group: the group's tabs above it, the page below fills the rest of the height. */
export function Grouped({ group, children }: { group: PageGroup; children: ReactNode }) {
  const { datasetId } = useParams({ strict: false });
  return (
    <div className="flex h-full flex-col">
      {datasetId && (
        <nav aria-label={t("nav.tabs")} className="shrink-0 overflow-x-auto border-b border-line px-(--gutter) pt-3">
          <ul className="flex gap-5">
            {GROUPS[group].map((tab) => (
              <li key={tab.to} className="shrink-0">
                <Link
                  to={tab.to as never}
                  params={{ datasetId } as never}
                  activeOptions={{ exact: true, includeSearch: false }}
                  className="-mb-px inline-flex items-center gap-1.5 border-b-2 border-transparent pb-2 text-[13px] whitespace-nowrap text-ink-2 hover:text-ink-1"
                  activeProps={{ className: "!border-accent font-medium !text-ink-1", "aria-current": "page" }}
                >
                  {t(tab.labelKey)}
                  {tab.locked && <Lock size={11} strokeWidth={1.75} aria-label={t("lite.locked")} className="text-ink-3" />}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
      )}
      <div className="min-h-0 flex-1">{children}</div>
    </div>
  );
}
