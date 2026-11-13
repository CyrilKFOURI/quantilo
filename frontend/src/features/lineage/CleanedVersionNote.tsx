import { Link } from "@tanstack/react-router";
import { Eraser } from "lucide-react";

import { t } from "../../lib/i18n";
import { useWorkingDataset } from "./working";

/** B1 (study § 5, "ML on the latest cleaned version", as a shortcut): opened on an original that
 *  has cleaned versions, a page says so and links to the same page on the latest ready one. Never
 *  a silent switch: the user picks it. Nothing is shown without a cleaned version. */
export function CleanedVersionNote({ datasetId, page }: { datasetId: string; page: string }) {
  const { latest } = useWorkingDataset(datasetId); // C-62
  if (!latest) return null;
  return (
    <p className="flex max-w-prose flex-wrap items-center gap-x-2 gap-y-1 border-l-2 border-accent pl-3 text-[13px] text-ink-2" title={t("clean.cleanedNote.hint")}>
      <Eraser size={14} strokeWidth={1.75} className="shrink-0 text-ink-3" aria-hidden />
      <span>{t("clean.cleanedNote.text", { name: t("clean.versions.name", { seq: String(latest.seq) }), steps: String(latest.steps.filter((step) => step.enabled).length) })}</span>
      <Link to={`/d/$datasetId/${page}` as never} params={{ datasetId: latest.dataset_id } as never} className="text-accent hover:underline">
        {t("clean.cleanedNote.use")}
      </Link>
    </p>
  );
}
