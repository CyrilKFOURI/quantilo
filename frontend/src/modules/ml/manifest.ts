import { BrainCircuit } from "lucide-react";

import type { ModuleManifest } from "../types";
import { MlPage, validateMlSearch } from "./MlPage";

export const mlModule: ModuleManifest = {
  id: "ml",
  scope: "dataset",
  labelKey: "module.ml.label",
  icon: BrainCircuit,
  path: "/d/$datasetId/ml",
  component: MlPage,
  group: "ml",
  validateSearch: validateMlSearch,
};
