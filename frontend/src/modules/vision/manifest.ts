import { Images } from "lucide-react";

import type { ModuleManifest } from "../types";
import { validateVisionSearch } from "./search";
import { VisionPage } from "./VisionPage";

/** Vision (107): an entry of the Dataset section, under Overview, only for an image set
 *  (`format = 'images'`, R-107-01); hidden, not greyed out, for a table. */
export const visionModule: ModuleManifest = {
  id: "vision",
  scope: "dataset",
  labelKey: "module.vision.label",
  icon: Images,
  path: "/d/$datasetId/vision",
  component: VisionPage,
  section: "dataset",
  requires: ["images"],
  onlyWhenMet: true,
  validateSearch: validateVisionSearch,
};
