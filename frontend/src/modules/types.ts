import type { LucideIcon } from "lucide-react";
import type { ReactElement } from "react";

import type { AnalysisType } from "../api/client";
import type { PageGroup } from "../app/layout/PageGroups";
import type { MessageKey } from "../lib/i18n";

type Base = {
  id: string;
  labelKey: MessageKey;
  icon: LucideIcon;
  component: () => ReactElement;
  /** Shown with a lock: the browser-only build cannot run it (lite/flag.ts). */
  locked?: boolean;
  /** Parses the page's own URL parameters. */
  validateSearch?: (search: Record<string, unknown>) => Record<string, unknown>;
  /** 046: a tab of a group of pages (its tabs above it); the group has one sidebar entry. */
  group?: PageGroup;
  /** 046: false when the page is reached from its group's tabs only, not from the sidebar. */
  nav?: boolean;
  /** 065: listed in the sidebar's Dataset section (under Overview) instead of the analyses. */
  section?: "dataset";
};

/** A panel a module docks beside every page (e.g. the assistant); it decides when to show itself. */
export type PanelManifest = {
  id: string;
  component: () => ReactElement | null;
};

/** A column type, or "images" (106): an analysis type of the dataset itself, an image set
 * (`format = 'images'`), whose columns stay ordinary columns for the tabular pages. */
export type Requirement = AnalysisType | "images";

/** Whether a dataset meets a page's requirement. */
export function meets(
  requirement: Requirement,
  format: string | undefined,
  columns: { type: AnalysisType; excluded: boolean }[],
): boolean {
  if (requirement === "images") return format === "images";
  return columns.some((column) => column.type === requirement && !column.excluded);
}

/** What a module declares to appear in the sidebar and own a route. */
export type ModuleManifest = Base &
  (
    | { scope: "global"; path: `/modules/${string}` }
    | {
        /** A page of the open dataset, greyed out without the column types it needs. */
        scope: "dataset";
        path: `/d/$datasetId/${string}`;
        /** Column types the page needs, or "images": the dataset is an image set (106). */
        requires?: Requirement[];
        requiresKey?: MessageKey;
      }
  );
