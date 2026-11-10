import { Eraser } from "lucide-react";

import type { ModuleManifest } from "../types";
import { CleanPage } from "./CleanPage";
import { validateCleanSearch } from "./search";

/** Clean (065): an entry of the Dataset section, under Overview (the one exception to "nothing new
 *  in the sidebar", approved by the owner). */
export const cleanModule: ModuleManifest = {
  id: "clean",
  scope: "dataset",
  labelKey: "module.clean.label",
  icon: Eraser,
  path: "/d/$datasetId/clean",
  component: CleanPage,
  section: "dataset",
  validateSearch: validateCleanSearch,
};
