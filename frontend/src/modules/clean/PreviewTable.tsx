import { TriangleAlert } from "lucide-react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import type { CleanPreview } from "./api";

/** "54 cells changed in description · 6 rows removed": the conclusion, then the first rows
 *  touched with their changed cells marked (C-10). */
export function PreviewPanel({
  preview,
  loading,
  error,
  title,
}: {
  preview: CleanPreview | undefined;
  loading: boolean;
  error: Error | null;
  title: string;
}) {
  if (error) return <ErrorNote>{error.message}</ErrorNote>;
  if (!preview) return <Loading label={t("clean.preview.loading")} className="h-48 rounded-lg" compact />;
  const changed = preview.columns.filter((column) => column.cells > 0);
  const parts: string[] = [];
  if (preview.cells_changed > 0)
    parts.push(
      t("clean.preview.cells", {
        count: formatCount(preview.cells_changed),
        columns: changed.length <= 3 ? changed.map((column) => column.column).join(", ") : `${changed.length} ${t("clean.field.columns").toLowerCase()}`,
      }),
    );
  if (preview.rows_removed > 0) parts.push(t("clean.preview.rows", { count: formatCount(preview.rows_removed) }));
  if (preview.added.length) parts.push(t("clean.preview.added", { columns: preview.added.join(", ") }));
  if (preview.dropped.length) parts.push(t("clean.preview.dropped", { columns: preview.dropped.join(", ") }));
  return (
    <section aria-label={title} aria-busy={loading} className={`space-y-2 transition-opacity ${loading ? "opacity-60" : ""}`}>
      <h3 className="text-xs text-ink-3">{title}</h3>
      <p className="text-[13px] text-ink-1">{parts.length ? parts.join(" · ") : t("clean.preview.none")}</p>
      {preview.sample.length > 0 && <SampleTable preview={preview} />}
    </section>
  );
}

function SampleTable({ preview }: { preview: CleanPreview }) {
  return (
    <div className="space-y-1">
      <p className="text-xs text-ink-3">{t("clean.preview.first", { count: String(preview.sample.length) })}</p>
      {/* Full width on a phone, scrolling inside its frame (046 Data table); values stay on one line. */}
      <div className="max-h-[28rem] overflow-auto rounded-lg border border-line bg-surface-1">
        <table data-keep-line className="w-full border-collapse text-[12px] tabular-nums">
          <thead className="sticky top-0 bg-surface-1">
            <tr className="border-b border-line text-left text-[11px] text-ink-3">
              <th scope="col" className="px-2 py-1.5 text-right font-normal">{t("clean.preview.row")}</th>
              {preview.shown.map((column) => (
                <th key={column} scope="col" className="px-2 py-1.5 font-mono font-normal whitespace-nowrap">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.sample.map((row) => (
              <tr key={row.row} className={`border-b border-line align-top last:border-b-0 ${row.removed ? "text-ink-3 line-through" : ""}`}>
                <th scope="row" className="px-2 py-1 text-right font-normal whitespace-nowrap text-ink-3">
                  {formatCount(row.row + 1)}
                  {row.removed && <span className="ml-1 no-underline">· {t("clean.preview.removed")}</span>}
                </th>
                {preview.shown.map((column, index) => {
                  const changed = row.changed[index];
                  return (
                    <td key={column} className={`px-2 py-1 whitespace-nowrap ${changed ? "bg-warn/8" : ""}`}>
                      {changed ? (
                        <span className="flex flex-col">
                          <span className="text-ink-3 line-through" aria-label={`${t("clean.preview.before")}: ${row.before[index] ?? t("clean.empty")}`}>
                            {shown(row.before[index] ?? null)}
                          </span>
                          <span className="inline-flex items-center gap-1 text-ink-1">
                            <TriangleAlert size={11} strokeWidth={1.75} className="shrink-0 text-warn" aria-label={t("clean.preview.changed")} />
                            {shown(row.after[index] ?? null)}
                          </span>
                        </span>
                      ) : (
                        <span className="text-ink-2">{shown(row.before[index] ?? null)}</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** A value as it is, its spaces made visible; empty in words. */
function shown(value: string | null) {
  if (value === null) return <em className="text-ink-3">{t("clean.empty")}</em>;
  if (value === "") return <em className="text-ink-3">""</em>;
  return value.replace(/^ +| +$/g, (spaces) => "·".repeat(spaces.length));
}
