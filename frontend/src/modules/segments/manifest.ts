import { Shapes } from "lucide-react";

import type { ModuleManifest } from "../types";
import { validateSegmentsSearch } from "./search";
import { SegmentsPage } from "./SegmentsPage";

export const segmentsModule: ModuleManifest = {
  id: "segments",
  scope: "dataset",
  labelKey: "module.segments.label",
  icon: Shapes,
  path: "/d/$datasetId/segments",
  component: SegmentsPage,
  group: "ml",
  nav: false, // a tab of Machine learning (046)
  validateSearch: validateSegmentsSearch,
};
