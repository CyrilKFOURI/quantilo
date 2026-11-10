import { createElement } from "react";

import { agentPanel } from "./agent/manifest";
import { cleanModule } from "./clean/manifest";
import { CLEAN_ENABLED } from "./clean/flag";
import { experimentsModule } from "./experiments/manifest";
import { kpiModule } from "./kpi/manifest";
import { mlModule } from "./ml/manifest";
import { segmentsModule } from "./segments/manifest";
import { syntheticModule } from "./synthetic/manifest";
import { timeseriesModule } from "./timeseries/manifest";
import { LITE } from "../lite/flag";
import { LockedPage } from "../lite/Locked";
import { t } from "../lib/i18n";
import type { ModuleManifest, PanelManifest } from "./types";

/** In the browser-only build, heavy modules stay listed but open a page that names them and
 *  explains why not ("Machine learning needs the full app"). */
const lockedInLite = (module: ModuleManifest): ModuleManifest =>
  LITE
    ? { ...module, locked: true, component: () => createElement(LockedPage, { page: t(module.labelKey) }) }
    : module;

/** Active modules. Adding a module = creating its folder and listing its manifest here. */
export const modules: ModuleManifest[] = [
  // the Dataset section, under Overview (065); VITE_CLEAN_ENABLED=0 leaves it out
  ...(CLEAN_ENABLED ? [cleanModule] : []),
  timeseriesModule,
  kpiModule,
  lockedInLite(mlModule),
  segmentsModule, // a tab of Machine learning; in the browser version too: sampled beyond 50,000 rows, t-SNE locked
  // light (SciPy and statsmodels): runs in the browser-only build too
  experimentsModule,
  lockedInLite(syntheticModule),
];

/** Panels docked beside the pages. Adding one = listing its manifest here. */
export const panels: PanelManifest[] = LITE ? [] : [agentPanel];
