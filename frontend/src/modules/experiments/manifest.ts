import { FlaskConical } from "lucide-react";

import type { ModuleManifest } from "../types";
import { ExperimentsPage } from "./ExperimentsPage";
import { validateExperimentsSearch } from "./search";

export const experimentsModule: ModuleManifest = {
  id: "experiments",
  scope: "dataset",
  labelKey: "module.experiments.label",
  icon: FlaskConical,
  path: "/d/$datasetId/experiments",
  component: ExperimentsPage,
  validateSearch: validateExperimentsSearch,
};
