import { Tags } from "lucide-react";
import { useState } from "react";

import { usePins } from "../../features/pins/api";
import { PinPanel } from "../../features/pins/PinPanel";
import { labelsOf } from "../../features/pins/rows";
import { Folded } from "../../features/text/Folded";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import type { Step } from "./api";
import { withPinnedLabels } from "./steps";

/** 099, under the Steps tab: pin rows by number, compare them, and put their labels in the draft
 *  as a `label_rows` step (R-PIN-15). Folded: nothing is asked of the server before it opens. */
export function PinnedColumn({ datasetId, steps, columns, onChange }: { datasetId: string; steps: Step[]; columns: string[]; onChange: (steps: Step[]) => void }) {
  return (
    <Folded title={t("clean.pins.title")} hint={t("clean.pins.hint")}>
      <PinPanel datasetId={datasetId} hint={t("pins.cleanHint")} footer={<AddLabels datasetId={datasetId} steps={steps} columns={columns} onChange={onChange} />} />
    </Folded>
  );
}

function AddLabels({ datasetId, steps, columns, onChange }: { datasetId: string; steps: Step[]; columns: string[]; onChange: (steps: Step[]) => void }) {
  const pins = usePins(datasetId);
  const labels = labelsOf(pins.data?.pins);
  const count = Object.keys(labels).length;
  const existing = steps.find((step) => step.op === "label_rows");
  const [added, setAdded] = useState<string | null>(null);
  const add = () => {
    const next = withPinnedLabels(steps, labels, columns);
    const step = next.find((item) => item.op === "label_rows");
    onChange(next);
    setAdded(step && "to" in step ? step.to : null);
  };
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 border-t border-line pt-3">
      <Button variant="primary" icon={Tags} disabled={count === 0} onClick={add}>
        {existing ? t("clean.pins.update") : t("clean.pins.add")}
      </Button>
      <p className="max-w-prose text-xs text-ink-3" role="status">
        {count === 0 ? t("clean.pins.none") : added ? t("clean.pins.added", { to: added }) : t("clean.opHint.label_rows")}
      </p>
    </div>
  );
}
