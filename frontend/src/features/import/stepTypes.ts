import type { Preview } from "../../api/client";

/** What the import screen shows once the file is read (082: told by its first bytes). One step per
 * kind of source; a new kind (images, 086) adds its component to `IMPORT_STEPS`. */
export type ImportStepKind = "default" | "tables" | "image";

export type ImportStepProps = {
  path: string;
  preview: Preview;
  onClose: () => void;
  onImported?: (datasetId: string) => void;
  project: { id: string; name: string } | null;
  /** Back to the one-table dialog (whole sheets, the main table of a document). */
  onSimple: () => void;
};

export const DOCUMENT_FORMATS = ["json", "jsonl", "xml"];

/** The step for this preview. `mode` is the user's choice: "simple" keeps the one-table dialog,
 * "tables" opens the tables of a workbook even when the detection found a single sure one. */
export function importStepFor(preview: Preview, mode: "auto" | "simple" | "tables"): ImportStepKind {
  if (preview.detected?.kind === "image") return "image";
  if (mode === "simple") return "default";
  const tables = preview.tables ?? [];
  if (DOCUMENT_FORMATS.includes(preview.format)) return tables.length > 1 ? "tables" : "default";
  if (preview.format !== "excel" || tables.length === 0) return "default";
  if (mode === "tables") return "tables";
  // 080b: one sure table keeps the current dialog (study § 6); several, or a doubt, show them all
  return tables.length > 1 || tables[0]?.level !== "high" ? "tables" : "default";
}
