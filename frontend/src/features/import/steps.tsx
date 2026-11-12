import type { ComponentType } from "react";

import { ImageSetStep } from "./ImageSetStep";
import type { ImportStepKind, ImportStepProps } from "./stepTypes";
import { TablesStep } from "./TablesStep";

/** 106: an image, a folder or a ZIP of images, a Parquet file with an image column (R-VIS-SRC). */
export const IMPORT_STEPS: Partial<Record<ImportStepKind, ComponentType<ImportStepProps>>> = {
  tables: TablesStep,
  image: ImageSetStep,
};
