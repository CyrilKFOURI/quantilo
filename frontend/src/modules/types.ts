import type { LucideIcon } from "lucide-react";
import type { ReactElement } from "react";

import type { AnalysisType } from "../api/client";
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
};

/** A panel a module docks beside every page (e.g. the assistant); it decides when to show itself. */
export type PanelManifest = {
  id: string;
  component: () => ReactElement | null;
};

/** What a module declares to appear in the sidebar and own a route. */
export type ModuleManifest = Base &
  (
    | { scope: "global"; path: `/modules/${string}` }
    | {
        /** A page of the open dataset, greyed out without the column types it needs. */
        scope: "dataset";
        path: `/d/$datasetId/${string}`;
        requires?: AnalysisType[];
        requiresKey?: MessageKey;
      }
  );
