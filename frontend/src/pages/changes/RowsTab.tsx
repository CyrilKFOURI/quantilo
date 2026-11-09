import { Link } from "@tanstack/react-router";
import { Download } from "lucide-react";
import { useState } from "react";

import { changeRowsUrl, type Dataset } from "../../api/client";
import { useChangeRows } from "../../api/hooks";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

const head = "px-3 py-1.5 font-normal";
const cell = "px-3 py-1.5";

/** Shown when a tab needs both versions' rows and the previous one was not kept. */
export function NotKept({ reason }: { reason: string }) {
  return (
    <div className="rounded-lg border border-dashed border-line px-6 py-8 text-center">
      <p className="mx-auto max-w-prose text-ink-2">{reason}</p>
      <Link to="/settings" className="mt-2 inline-block text-[13px] text-accent hover:underline">
        {t("changes.openSettings")}
      </Link>
    </div>
  );
}

function Count({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <dt className="text-xs text-ink-3">{label}</dt>
      <dd className="text-lg font-semibold text-ink-1 tabular-nums">{formatCount(value)}</dd>
    </div>
  );
}

export function RowsTab({ dataset, vs }: { dataset: Dataset; vs: string | null }) {
  const [key, setKey] = useState<string | null>(null);
  const rows = useChangeRows(dataset, vs, key, true);
  if (rows.isError) return <ErrorNote>{rows.error.message}</ErrorNote>;
  if (!rows.data) return <Loading label={t("loading.changedRows")} className="h-40 rounded-lg" />;
  const answer = rows.data;
  const result = answer.result;
  if (!answer.available || !result) return <NotKept reason={answer.reason ?? ""} />;
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <label className="flex items-center gap-2 text-[13px] text-ink-2">
          {t("changes.rows.key")}
          <Select value={result.key} onChange={(event) => setKey(event.target.value)}>
            {answer.candidates.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </Select>
        </label>
        <a
          href={changeRowsUrl(dataset.id, vs, result.key)}
          download
          className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2"
        >
          <Download size={15} strokeWidth={1.5} aria-hidden />
          {t("changes.rows.export")}
        </a>
      </div>

      <dl className="grid max-w-2xl grid-cols-2 gap-4 sm:grid-cols-4">
        <Count label={t("changes.rows.added")} value={result.added} />
        <Count label={t("changes.rows.removed")} value={result.removed} />
        <Count label={t("changes.rows.modified")} value={result.modified} />
        <Count label={t("changes.rows.unchanged")} value={result.unchanged} />
      </dl>

      {result.columns.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold">{t("changes.rows.perColumn")}</h2>
          <div className="max-w-md overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th scope="col" className={`${head} w-full`}>{t("changes.col.column")}</th>
                  <th scope="col" className={`${head} text-right`}>{t("changes.rows.modified")}</th>
                </tr>
              </thead>
              <tbody>
                {result.columns.map((edit) => (
                  <tr key={edit.column} className="border-t border-line">
                    <td className={`${cell} font-mono text-xs text-ink-1`}>{edit.column}</td>
                    <td className={`${cell} text-right text-ink-1 tabular-nums`}>{formatCount(edit.modified)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {result.examples.length > 0 && (
        <section>
          <div className="mb-2 flex flex-wrap items-baseline gap-x-3">
            <h2 className="text-sm font-semibold">{t("changes.rows.examples")}</h2>
            <span className="text-xs text-ink-3">{t("changes.rows.examplesHint")}</span>
          </div>
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full border-collapse text-[13px]">
              <thead className="bg-surface-2 text-left text-xs text-ink-3">
                <tr>
                  <th scope="col" className={head}>{result.key}</th>
                  <th scope="col" className={head}>{t("changes.col.column")}</th>
                  <th scope="col" className={head}>{t("changes.before")}</th>
                  <th scope="col" className={head}>{t("changes.after")}</th>
                </tr>
              </thead>
              <tbody>
                {result.examples.map((edit) => (
                  <tr key={`${edit.key}|${edit.column}`} className="border-t border-line">
                    <td className={`${cell} font-mono text-xs text-ink-2`}>{edit.key}</td>
                    <td className={`${cell} font-mono text-xs text-ink-1`}>{edit.column}</td>
                    <td className={`${cell} text-ink-2 tabular-nums`}>{edit.before ?? "(missing)"}</td>
                    <td className={`${cell} text-ink-1 tabular-nums`}>{edit.after ?? "(missing)"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {(result.added_keys.length > 0 || result.removed_keys.length > 0) && (
        <div className="flex flex-wrap gap-x-10 gap-y-2 text-[13px]">
          {result.added_keys.length > 0 && (
            <p className="text-ink-2">
              {t("changes.rows.addedKeys")} <span className="font-mono text-xs text-ink-1">{result.added_keys.join(", ")}</span>
            </p>
          )}
          {result.removed_keys.length > 0 && (
            <p className="text-ink-2">
              {t("changes.rows.removedKeys")} <span className="font-mono text-xs text-ink-1">{result.removed_keys.join(", ")}</span>
            </p>
          )}
        </div>
      )}
    </div>
  );
}
