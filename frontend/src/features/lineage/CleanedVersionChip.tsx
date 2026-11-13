import { useLocation, useNavigate } from "@tanstack/react-router";
import { Eraser } from "lucide-react";

import { t } from "../../lib/i18n";
import { samePageOn, useWorkingDataset } from "./working";

/** C-62: in the top bar, on an original that has a ready cleaned version: "Cleaned v2 exists",
 *  a button to the same page on that version. Not on ML (it has its own note) nor on Clean (it
 *  says what is being cleaned). Nothing without a cleaned version. */
export function CleanedVersionChip({ datasetId }: { datasetId: string }) {
  const { latest } = useWorkingDataset(datasetId);
  const { pathname, searchStr } = useLocation();
  const navigate = useNavigate();
  if (!latest || /\/(ml|clean)(\/|$)/.test(pathname)) return null;
  const name = t("clean.versions.name", { seq: String(latest.seq) });
  const label = t("clean.cleanedChip.text", { name });
  return (
    <button
      type="button"
      onClick={() => void navigate({ href: `${samePageOn(pathname, datasetId, latest.dataset_id)}${searchStr}` })}
      title={t("clean.cleanedChip.hint", { name })}
      aria-label={`${label}. ${t("clean.cleanedChip.open", { name })}`}
      className="inline-flex h-7 shrink-0 items-center gap-1.5 self-center rounded-full border border-line bg-surface-1 px-2.5 text-xs text-ink-2 hover:border-accent hover:text-ink-1 max-sm:size-9 max-sm:justify-center max-sm:px-0"
    >
      <Eraser size={13} strokeWidth={1.75} className="shrink-0 text-accent" aria-hidden />
      <span className="max-sm:hidden">{label}</span>
    </button>
  );
}
