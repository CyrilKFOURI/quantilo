import type { ComponentType } from "react";

import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import type { ImportStepKind, ImportStepProps } from "./stepTypes";
import { TablesStep } from "./TablesStep";

/** Until the image step exists (086), an image is named and not imported. */
function ImageNotice({ preview, onClose }: ImportStepProps) {
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={preview.suggested_name}
      description={preview.detected?.reason}
      size="sm"
      footer={
        <Button variant="primary" onClick={onClose}>
          {t("import.close")}
        </Button>
      }
    >
      <p className="text-ink-2">
        {t("preview.detected.image", { label: preview.detected?.label ?? "Image" })}
      </p>
    </Dialog>
  );
}

export const IMPORT_STEPS: Partial<Record<ImportStepKind, ComponentType<ImportStepProps>>> = {
  tables: TablesStep,
  image: ImageNotice,
};
