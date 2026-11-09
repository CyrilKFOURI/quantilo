import type { ReactNode } from "react";

import type { Changes, SchemaChange } from "../../api/client";
import { formatCount } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";

const KINDS: Record<SchemaChange["kind"], MessageKey> = {
  added: "changes.kind.added",
  removed: "changes.kind.removed",
  renamed: "changes.kind.renamed",
  retyped: "changes.kind.retyped",
  constant: "changes.kind.constant",
  empty: "changes.kind.empty",
};

export const pct = (value: number | null | undefined, digits = 1) =>
  value === null || value === undefined ? "—" : `${value.toFixed(digits)}%`;

export function Empty({ children }: { children: ReactNode }) {
  return <p className="rounded-lg border border-dashed border-line px-4 py-8 text-center text-ink-2">{children}</p>;
}

const head = "px-3 py-1.5 font-normal";
const cell = "px-3 py-1.5";

export function SchemaTab({ changes }: { changes: Changes }) {
  if (changes.schema.length === 0) return <Empty>{t("changes.schema.none")}</Empty>;
  return (
    <div className="overflow-x-auto rounded-md border border-line">
      <table className="w-full border-collapse text-[13px]">
        <thead className="bg-surface-2 text-left text-xs text-ink-3">
          <tr>
            <th scope="col" className={head}>{t("changes.schema.change")}</th>
            <th scope="col" className={head}>{t("changes.col.column")}</th>
            <th scope="col" className={`${head} w-full`}>{t("changes.schema.detail")}</th>
            <th scope="col" className={head}>{t("changes.schema.usedBy")}</th>
          </tr>
        </thead>
        <tbody>
          {changes.schema.map((change) => (
            <tr key={`${change.kind}|${change.column}`} className="border-t border-line align-top">
              <td className={`${cell} whitespace-nowrap`}>
                <span className={change.breaking ? "font-medium text-bad" : "text-ink-1"}>{t(KINDS[change.kind])}</span>
                {change.breaking && <span className="ml-1.5 text-xs text-bad">· {t("changes.level.breaking")}</span>}
              </td>
              <td className={`${cell} font-mono text-xs whitespace-nowrap text-ink-1`}>
                {change.kind === "renamed" ? (
                  <>
                    <span className="text-ink-3">{change.before}</span> → {change.after}
                  </>
                ) : (
                  change.column
                )}
              </td>
              <td className={`${cell} text-ink-2`}>
                {change.detail || (change.kind === "added" && change.after ? t("changes.schema.addedAs", { type: change.after }) : "")}
              </td>
              <td className={`${cell} text-xs text-ink-2`}>{change.used_by.join(", ") || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function QualityTab({ changes }: { changes: Changes }) {
  const rows = [...changes.quality].sort(
    (a, b) => Number(b.notable) - Number(a.notable) || b.missing_after - b.missing_before - (a.missing_after - a.missing_before),
  );
  const duplicatesBefore = changes.before.rows ? (100 * changes.before.duplicates) / changes.before.rows : 0;
  const duplicatesAfter = changes.after.rows ? (100 * changes.after.duplicates) / changes.after.rows : 0;
  const dates = changes.dates.filter((item) => item.future > 0 || item.gaps.length > 0);
  return (
    <div className="flex flex-col gap-5">
      <dl className="grid max-w-2xl grid-cols-2 gap-x-6 gap-y-1 text-[13px] sm:grid-cols-3">
        <div>
          <dt className="text-xs text-ink-3">{t("changes.quality.rows")}</dt>
          <dd className="text-ink-1 tabular-nums">
            {formatCount(changes.before.rows)} → {formatCount(changes.after.rows)}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-ink-3">{t("changes.quality.duplicates")}</dt>
          <dd className="text-ink-1 tabular-nums">
            {formatCount(changes.before.duplicates)} ({pct(duplicatesBefore)}) → {formatCount(changes.after.duplicates)} ({pct(duplicatesAfter)})
          </dd>
        </div>
      </dl>

      {dates.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold">{t("changes.quality.dates")}</h2>
          <ul className="space-y-1 text-[13px] text-ink-2">
            {dates.map((item) => (
              <li key={item.column}>
                <span className="font-mono text-xs text-ink-1">{item.column}</span>
                {item.future > 0 && <> · {t("changes.quality.future", { count: formatCount(item.future) })}</>}
                {item.gaps.length > 0 && <> · {t("changes.quality.gaps", { periods: item.gaps.join(", ") })}</>}
              </li>
            ))}
          </ul>
        </section>
      )}

      <section>
        <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
          <h2 className="text-sm font-semibold">{t("changes.quality.columns")}</h2>
          <span className="text-xs text-ink-3">{t("changes.quality.hint")}</span>
        </div>
        <div className="overflow-x-auto rounded-md border border-line">
          <table className="w-full border-collapse text-[13px]">
            <thead className="bg-surface-2 text-left text-xs text-ink-3">
              <tr>
                <th scope="col" className={`${head} w-full`}>{t("changes.col.column")}</th>
                <th scope="col" className={`${head} text-right`}>{t("changes.quality.missing")}</th>
                <th scope="col" className={`${head} text-right`}>{t("changes.quality.outliers")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((item) => (
                <tr key={item.column} className="border-t border-line">
                  <td className={`${cell} font-mono text-xs text-ink-1`}>
                    {item.column}
                    {item.notable && <span className="ml-2 font-sans text-xs text-warn">{t("changes.quality.flag")}</span>}
                  </td>
                  <td className={`${cell} text-right whitespace-nowrap text-ink-2 tabular-nums`}>
                    {pct(item.missing_before)} → <span className="text-ink-1">{pct(item.missing_after)}</span>
                  </td>
                  <td className={`${cell} text-right whitespace-nowrap text-ink-2 tabular-nums`}>
                    {item.outliers_before === null ? "—" : (
                      <>
                        {pct(item.outliers_before)} → <span className="text-ink-1">{pct(item.outliers_after)}</span>
                      </>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
