import { Link } from "@tanstack/react-router";
import { Info } from "lucide-react";
import { useState } from "react";

import { t } from "../../lib/i18n";

type ReadingNotesProps = {
  notes: string[];
  label: string;
  /** Links to this dataset's Columns page, where a reading is checked or changed (093). */
  datasetId?: string;
  /** Said under the notes instead of the link, where the link would lead to the same page or
   * the dataset does not exist yet (092: the Columns page, the import preview). */
  hint?: string;
  /** Notes shown before a "Show all" button (092: a wide survey reads dozens of columns). */
  limit?: number;
};

/** How the rules read the data, each with its code and reason (093, owner rule: no silent
 * correction), and where to check or change it: the Columns page. */
export function ReadingNotes({ datasetId, notes, label, hint, limit }: ReadingNotesProps) {
  const [all, setAll] = useState(false);
  if (notes.length === 0) return null;
  const shown = limit && !all ? notes.slice(0, limit) : notes;
  return (
    <section aria-label={label} className="min-w-0 rounded-md border border-line bg-surface-1 px-3 py-2 text-xs text-ink-2">
      <ul className="space-y-1">
        {shown.map((note) => (
          <li key={note} className="flex min-w-0 items-baseline gap-2">
            <Info size={12} strokeWidth={1.5} aria-hidden className="shrink-0 translate-y-0.5 text-ink-3" />
            <span className="min-w-0 break-words">{note}</span>
          </li>
        ))}
      </ul>
      {shown.length < notes.length && (
        <button
          type="button"
          className="mt-1 inline-flex min-h-6 items-center text-accent hover:underline pointer-coarse:min-h-11"
          onClick={() => setAll(true)}
        >
          {t("readings.showAll", { count: String(notes.length) })}
        </button>
      )}
      {datasetId ? (
        <Link
          to="/d/$datasetId/columns"
          params={{ datasetId }}
          className="mt-1 inline-flex min-h-6 items-center text-accent hover:underline pointer-coarse:min-h-11"
        >
          {t("readings.checkColumns")}
        </Link>
      ) : (
        hint && <p className="mt-1 font-medium text-ink-1">{hint}</p>
      )}
    </section>
  );
}

/** "R-TYP-01", "R-ING-01": a reason given by a reading rule of the import or the typing (092). */
const READING_RULE = /\(R-(?:TYP|ING)-\d+\)/;

/** Each column read by a rule, with that rule's reason: "dep: … (R-TYP-01)". */
export function columnReadingNotes(columns: { name: string; reasons: string[] }[]): string[] {
  return columns.flatMap((column) =>
    column.reasons.filter((reason) => READING_RULE.test(reason)).map((reason) => `${column.name}: ${reason}`),
  );
}
