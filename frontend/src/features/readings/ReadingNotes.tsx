import { Link } from "@tanstack/react-router";
import { Info } from "lucide-react";
import { t } from "../../lib/i18n";

/** How the rules read the data, each with its code and reason (093, owner rule: no silent
 * correction), and where to check or change it: the Columns page. */
export function ReadingNotes({ datasetId, notes, label }: { datasetId: string; notes: string[]; label: string }) {
  if (notes.length === 0) return null;
  return (
    <section aria-label={label} className="min-w-0 rounded-md border border-line bg-surface-1 px-3 py-2 text-xs text-ink-2">
      <ul className="space-y-1">
        {notes.map((note) => (
          <li key={note} className="flex min-w-0 items-baseline gap-2">
            <Info size={12} strokeWidth={1.5} aria-hidden className="shrink-0 translate-y-0.5 text-ink-3" />
            <span className="min-w-0 break-words">{note}</span>
          </li>
        ))}
      </ul>
      <Link
        to="/d/$datasetId/columns"
        params={{ datasetId }}
        className="mt-1 inline-flex min-h-6 items-center text-accent hover:underline pointer-coarse:min-h-11"
      >
        {t("readings.checkColumns")}
      </Link>
    </section>
  );
}
