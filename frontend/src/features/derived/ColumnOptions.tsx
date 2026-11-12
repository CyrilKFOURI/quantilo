import type { Column } from "../../api/client";
import { t } from "../../lib/i18n";
import { derivedNote, isDerived, sourceOf } from "./derived";

type Group = { source: string | null; columns: Column[] };

/** Consecutive derived columns of one source, in a group of their own (R-DER-08). */
function groups(columns: Column[]): Group[] {
  const found: Group[] = [];
  for (const column of columns) {
    const source = isDerived(column) ? sourceOf(column) : null;
    const last = found.at(-1);
    if (last && source !== null && last.source === source) last.columns.push(column);
    else found.push({ source, columns: [column] });
  }
  return found;
}

/** The `<option>`s of a column picker: each source's derived columns ("review › Length") in an
 *  `<optgroup>` right under it, each with its rule as a tooltip. */
export function ColumnOptions({
  columns,
  unavailable,
}: {
  columns: Column[];
  /** Why a column cannot be picked here (null: it can). */
  unavailable?: (column: Column) => string | null;
}) {
  const option = (column: Column) => {
    const reason = unavailable?.(column) ?? null;
    return (
      <option
        key={column.name}
        value={column.name}
        disabled={reason !== null}
        title={reason ?? derivedNote(column) ?? undefined}
      >
        {column.name}
        {reason ? ` (${reason})` : ""}
      </option>
    );
  };
  return (
    <>
      {groups(columns).map((group, index) =>
        group.source === null ? (
          group.columns.map(option)
        ) : (
          <optgroup key={`${group.source}-${index}`} label={t("derived.group", { source: group.source })}>
            {group.columns.map(option)}
          </optgroup>
        ),
      )}
    </>
  );
}
