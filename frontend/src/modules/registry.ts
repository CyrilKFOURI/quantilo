import { agentPanel } from "./agent/manifest";
import { experimentsModule } from "./experiments/manifest";
import { kpiModule } from "./kpi/manifest";
import { mlModule } from "./ml/manifest";
import { segmentsModule } from "./segments/manifest";
import { syntheticModule } from "./synthetic/manifest";
import { timeseriesModule } from "./timeseries/manifest";
import { LITE } from "../lite/flag";
import { LockedPage } from "../lite/Locked";
import type { ModuleManifest, PanelManifest } from "./types";

/** In the browser-only build, heavy modules stay listed but open a page that explains why not. */
const lockedInLite = (module: ModuleManifest): ModuleManifest =>
  LITE ? { ...module, locked: true, component: LockedPage } : module;

/** Active modules. Adding a module = creating its folder and listing its manifest here. */
export const modules: ModuleManifest[] = [
  timeseriesModule,
  lockedInLite(mlModule),
  segmentsModule, // in the browser version too: sampled beyond 50,000 rows, t-SNE locked
  kpiModule,
  lockedInLite(syntheticModule),
  // light (SciPy and statsmodels): runs in the browser-only build too
  experimentsModule,
];

/** Panels docked beside the pages. Adding one = listing its manifest here. */
export const panels: PanelManifest[] = LITE ? [] : [agentPanel];
