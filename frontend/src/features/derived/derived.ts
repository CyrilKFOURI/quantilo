import type { Column } from "../../api/client";
import { t } from "../../lib/i18n";

/** 058: a derived column ("review › Length"), computed from its source when it is read. */
export function isDerived(column: Pick<Column, "parser">): boolean {
  return column.parser.kind === "derived";
}

export function sourceOf(column: Pick<Column, "parser">): string | null {
  return column.parser.derived?.source ?? null;
}

/** The columns with each source followed by its derived columns (R-DER-08): a picker lists
 *  `review`, `review › Length`, `review › Words`… then the next column. */
export function withDerived(columns: Column[], derived: Column[] | undefined): Column[] {
  if (!derived?.length) return columns;
  const bySource = new Map<string, Column[]>();
  for (const column of derived) {
    const source = sourceOf(column);
    if (!source) continue;
    bySource.set(source, [...(bySource.get(source) ?? []), column]);
  }
  return columns.flatMap((column) => [column, ...(bySource.get(column.name) ?? [])]);
}

/** Why a derived column exists, with its rule (R-DER-08): "Derived from 'review': number of
 *  characters (R-DER-01). Check that your data was read correctly." Null for a stored column. */
export function derivedNote(column: Pick<Column, "parser" | "reasons"> | undefined): string | null {
  if (!column || !isDerived(column)) return null;
  const reason = column.reasons[0] ?? "";
  const sentence = reason ? reason.charAt(0).toUpperCase() + reason.slice(1) : "";
  return `${sentence}. ${t("derived.check")}`;
}
