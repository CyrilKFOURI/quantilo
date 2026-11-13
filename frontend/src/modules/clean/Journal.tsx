import { FileDown } from "lucide-react";

import { formatCount, formatRelativeTime } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { buttonClass } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { cleanUrls, useDraftJournal, useJournal, type CleanVersion, type JournalEntry } from "./api";
import { visible } from "./parts";

/** N46: one journal entry in a sentence of the page's language: who, what, which step (its rule,
 *  op and columns), its suggestion's title, cells or rows. */
export function entrySentence(entry: JournalEntry): string {
  const who = t(`clean.journal.actor.${entry.actor}` as MessageKey);
  const what = t(`clean.journal.action.${entry.action}` as MessageKey);
  const step = entry.step_id ? ` ${entry.step_id} (${[entry.rule, entry.op].filter(Boolean).join(", ")}${entry.columns.length ? ` · ${entry.columns.join(", ")}` : ""})` : "";
  const detail = entry.detail ? `: ${entry.detail}` : "";
  const counts = [entry.cells ? t("clean.count.cells", { count: formatCount(entry.cells) }) : "", entry.rows ? t("clean.count.rows", { count: formatCount(entry.rows) }) : ""].filter(Boolean).join(" · ");
  return `${what}${step}${detail}${counts ? ` · ${counts}` : ""} · ${who}`;
}

function Entries({ entries }: { entries: JournalEntry[] }) {
  return (
    <ol className="divide-y divide-line rounded-md border border-line text-xs">
      {entries.map((entry, index) => (
        <li key={index} className="grid grid-cols-[minmax(0,7rem)_minmax(0,1fr)] gap-x-3 px-3 py-1.5">
          <time dateTime={entry.at} className="text-ink-3 tabular-nums" title={entry.at}>
            {formatRelativeTime(entry.at)}
          </time>
          <span className={entry.actor === "auto" ? "text-ink-2" : "text-ink-1"}>{entrySentence(entry)}</span>
        </li>
      ))}
    </ol>
  );
}

/** N46, in Versions: the changes of the draft since the last saved version, folded. Nothing when
 *  there are none (or an older server). */
export function DraftJournal({ datasetId }: { datasetId: string }) {
  const found = useDraftJournal(datasetId);
  const entries = found.data ?? [];
  if (!entries.length) return null;
  return (
    <details className="rounded-lg border border-line bg-surface-1 px-3 py-2">
      <summary className="cursor-pointer text-[13px] text-ink-1">{t("clean.journal.draft", { count: formatCount(entries.length) })}</summary>
      <p className="mt-1 mb-2 text-xs text-ink-3">{t("clean.journal.draftHint")}</p>
      <Entries entries={entries} />
    </details>
  );
}

/** N46: a version's cleaning log: where each step came from (automatic, ticked, by hand), what it
 *  changed, the rows removed and why, and who changed the draft when; Markdown or HTML to keep. */
export function JournalDialog({ version, onClose }: { version: CleanVersion; onClose: () => void }) {
  const found = useJournal(version.dataset_id);
  const journal = found.data;
  const name = t("clean.versions.name", { seq: String(version.seq) });
  return (
    <Dialog
      open
      onOpenChange={(open) => !open && onClose()}
      title={t("clean.journal.title", { name })}
      size="lg"
      footer={
        <>
          <a href={cleanUrls.journalMarkdown(version.dataset_id)} download className={buttonClass("secondary")}>
            <FileDown size={15} strokeWidth={1.5} aria-hidden />
            {t("clean.journal.markdown")}
          </a>
          <a href={cleanUrls.journalHtml(version.dataset_id)} download className={buttonClass("secondary")}>
            <FileDown size={15} strokeWidth={1.5} aria-hidden />
            {t("clean.journal.html")}
          </a>
        </>
      }
    >
      {found.isError ? (
        <ErrorNote>{found.error.message}</ErrorNote>
      ) : !journal ? (
        <Loading label={t("loading.default")} className="h-32 rounded-lg" compact />
      ) : (
        <div className="space-y-4 text-[13px]">
          <p className="text-ink-1 tabular-nums">
            {t("clean.journal.summary", {
              auto: formatCount(journal.applied.auto ?? 0),
              suggested: formatCount(journal.applied.suggested ?? 0),
              manual: formatCount(journal.applied.manual ?? 0),
            })}
            {journal.report && ` · ${t("clean.banner.cells", { count: formatCount(journal.report.cells_changed) })}`}
            {journal.report && journal.report.rows_removed > 0 && ` · ${t("clean.banner.rows", { count: formatCount(journal.report.rows_removed) })}`}
          </p>
          <section aria-label={t("clean.journal.steps")} className="space-y-1">
            <h3 className="text-xs text-ink-3">{t("clean.journal.steps")}</h3>
            <div className="max-h-80 overflow-auto rounded-md border border-line">
              <table className="w-full border-collapse text-xs tabular-nums">
                <thead className="sticky top-0 bg-surface-1 text-left text-[11px] text-ink-3">
                  <tr className="border-b border-line">
                    <th scope="col" className="px-2 py-1.5 font-normal">{t("clean.journal.col.step")}</th>
                    <th scope="col" className="px-2 py-1.5 font-normal">{t("clean.journal.col.origin")}</th>
                    <th scope="col" className="px-2 py-1.5 font-normal">{t("clean.journal.col.what")}</th>
                    <th scope="col" className="px-2 py-1.5 text-right font-normal">{t("clean.journal.col.cells")}</th>
                    <th scope="col" className="px-2 py-1.5 text-right font-normal">{t("clean.journal.col.rows")}</th>
                  </tr>
                </thead>
                <tbody>
                  {journal.steps.map((step) => (
                    <tr key={step.id} className={`border-b border-line last:border-0 ${step.enabled ? "" : "text-ink-3"}`}>
                      <td className="px-2 py-1 font-mono whitespace-nowrap">
                        {step.id}
                        {step.rule && <span className="text-ink-3"> · {step.rule}</span>}
                      </td>
                      <td className="px-2 py-1">{step.enabled ? t(`clean.journal.origin.${step.origin}` as MessageKey) : t("clean.journal.off")}</td>
                      <td className="px-2 py-1">
                        {step.title}
                        {step.example && <span className="block font-mono text-[11px] text-ink-3">{`${visible(step.example[0] ?? null)} → ${visible(step.example[1] ?? null)}`}</span>}
                      </td>
                      <td className="px-2 py-1 text-right">{step.cells === null ? "" : formatCount(step.cells)}</td>
                      <td className="px-2 py-1 text-right">{step.rows_removed ? formatCount(step.rows_removed) : ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
          {journal.report && Object.keys(journal.report.cells_by_column).length > 0 && (
            <p className="text-xs text-ink-2">
              <span className="text-ink-3">{t("clean.journal.byColumn")} </span>
              {Object.entries(journal.report.cells_by_column)
                .map(([column, cells]) => `${column} ${formatCount(cells)}`)
                .join(" · ")}
            </p>
          )}
          {journal.report && journal.report.type_changes.length > 0 && (
            <p className="text-xs text-ink-2">{t("clean.versions.typeChanges", { columns: journal.report.type_changes.map((change) => `${change.column} (${change.before} → ${change.after})`).join(", ") })}</p>
          )}
          <section aria-label={t("clean.journal.log")} className="space-y-1">
            <h3 className="text-xs text-ink-3">{t("clean.journal.log")}</h3>
            {journal.entries.length ? <Entries entries={journal.entries} /> : <p className="text-xs text-ink-3">{t("clean.journal.none")}</p>}
          </section>
        </div>
      )}
    </Dialog>
  );
}
