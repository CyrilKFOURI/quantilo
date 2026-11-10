import { LineChart } from "lucide-react";

import type { ModuleManifest } from "../types";
import { validateTimeSeriesSearch } from "./search";
import { TimeSeriesPage } from "./TimeSeriesPage";

export const timeseriesModule: ModuleManifest = {
  id: "timeseries",
  scope: "dataset",
  labelKey: "module.timeseries.label",
  icon: LineChart,
  path: "/d/$datasetId/timeseries",
  requires: ["datetime"],
  requiresKey: "module.timeseries.needsDate",
  component: TimeSeriesPage,
  validateSearch: validateTimeSeriesSearch,
  nav: false, // 098: a manual tool, opened from the Toolbox (or prefilled from an analysis)
};
