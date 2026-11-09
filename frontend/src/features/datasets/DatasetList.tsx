import { Link, useNavigate } from "@tanstack/react-router";
import { GitCompareArrows, Pencil, RefreshCw, Trash2, X } from "lucide-react";
import { useState } from "react";

import type { Dataset } from "../../api/client";
import { useDeleteDataset, useReimportDataset, useRenameDataset } from "../../api/hooks";
import { formatBytes, formatCount, formatRelativeTime } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, TextInput } from "../../ui/fields";
import { StatusBadge } from "./StatusBadge";

/** Of two datasets, the one imported last is the new version; the other is the reference. */
export function versionsOrder(first: Dataset, second: Dataset): [Dataset, Dataset] {
  const stamp = (d: Dataset) => d.imported_at ?? d.created_at;
  return stamp(first) >= stamp(second) ? [first, second] : [second, first];
}

export function DatasetList({ datasets }: { datasets: Dataset[] }) {
  const [deleting, setDeleting] = useState<Dataset | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const navigate = useNavigate();
  const chosen = datasets.filter((dataset) => selected.includes(dataset.id) && dataset.status === "ready");
  // Ticking a third dataset replaces the oldest tick: two are compared at most.
  const toggle = (id: string) =>
    setSelected((current) => (current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(-2)));
  const compare = () => {
    const [first, second] = chosen;
    if (!first || !second) return;
    const [after, before] = versionsOrder(first, second);
    void navigate({ to: "/d/$datasetId/changes", params: { datasetId: after.id }, search: { vs: before.id } });
  };
  const remove = useDeleteDataset();
  const reimport = useReimportDataset();
  const error = remove.error ?? reimport.error;
  // Deleting a dataset that is still importing cancels its import.
  const cancelling = deleting?.status === "importing";

  return (
    <section>
      <div className="mb-2 flex min-h-(--control-h) flex-wrap items-center justify-between gap-3">
        <h2 className="text-xs text-ink-3">{t("datasets.recent")}</h2>
        {chosen.length === 2 ? (
          <Button icon={GitCompareArrows} onClick={compare}>
            {t("changes.compareVersions")}
          </Button>
        ) : (
          datasets.length > 1 && (
            <span className="text-xs text-ink-3">{t(chosen.length === 1 ? "changes.pickOneMore" : "changes.pickTwo")}</span>
          )
        )}
      </div>
      {error && (
        <div className="mb-2">
          <ErrorNote>{error.message}</ErrorNote>
        </div>
      )}
      <div className="overflow-x-auto rounded-lg border border-line bg-surface-1">
        {/* 041: a phone keeps name, shape and status (size and last opened are hidden). */}
        <table className="w-full border-collapse text-[13px] max-sm:[&_tr>:nth-child(4)]:hidden max-sm:[&_tr>:nth-child(5)]:hidden">
          <thead>
            <tr className="border-b border-line text-left text-xs text-ink-3">
              <th scope="col" className="w-0 py-2 pl-3"><span className="sr-only">{t("changes.selectToCompare")}</span></th>
              <th scope="col" className="w-full px-3 py-2 font-normal">{t("datasets.col.name")}</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">{t("datasets.col.shape")}</th>
              <th scope="col" className="px-3 py-2 text-right font-normal">{t("datasets.col.size")}</th>
              <th scope="col" className="px-3 py-2 font-normal">{t("datasets.col.opened")}</th>
              <th scope="col" className="px-3 py-2 font-normal">{t("datasets.col.status")}</th>
              <th scope="col" className="w-0 px-2 py-2"><span className="sr-only">{t("datasets.actions")}</span></th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((dataset) => (
              <DatasetRow
                key={dataset.id}
                dataset={dataset}
                checked={selected.includes(dataset.id)}
                onToggle={() => toggle(dataset.id)}
                onDelete={() => setDeleting(dataset)}
                onReimport={() => reimport.mutate(dataset.id)}
              />
            ))}
          </tbody>
        </table>
      </div>

      {deleting && (
        <Dialog
          open
          onOpenChange={(open) => !open && setDeleting(null)}
          title={t(cancelling ? "datasets.cancelImportTitle" : "datasets.deleteTitle", { name: deleting.name })}
          size="sm"
          footer={
            <>
              <Button variant="ghost" onClick={() => setDeleting(null)}>
                {t("import.cancel")}
              </Button>
              <Button
                variant="danger"
                disabled={remove.isPending}
                onClick={() => remove.mutate(deleting.id, { onSuccess: () => setDeleting(null) })}
              >
                {t(cancelling ? "datasets.cancelImport" : "datasets.delete")}
              </Button>
            </>
          }
        >
          <p className="text-ink-2">{t(cancelling ? "datasets.cancelImportBody" : "datasets.deleteBody")}</p>
        </Dialog>
      )}
    </section>
  );
}

function DatasetRow({
  dataset,
  checked,
  onToggle,
  onDelete,
  onReimport,
}: {
  dataset: Dataset;
  checked: boolean;
  onToggle: () => void;
  onDelete: () => void;
  onReimport: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(dataset.name);
  const rename = useRenameDataset();

  const save = () => {
    const name = draft.trim();
    if (name && name !== dataset.name) rename.mutate({ id: dataset.id, name });
    setEditing(false);
  };

  return (
    <tr className="group border-b border-line last:border-b-0 hover:bg-surface-0">
      <td className="py-2 pl-3">
        <input
          type="checkbox"
          className="size-3.5 accent-(--accent)"
          checked={checked}
          disabled={dataset.status !== "ready"}
          onChange={onToggle}
          aria-label={t("changes.selectDataset", { name: dataset.name })}
        />
      </td>
      <td className="max-w-0 px-3 py-2">
        {editing ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              save();
            }}
          >
            <TextInput
              autoFocus
              aria-label={t("datasets.rename")}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onBlur={save}
              onKeyDown={(event) => event.key === "Escape" && setEditing(false)}
              className="w-full"
            />
          </form>
        ) : (
          <div className="min-w-0">
            {dataset.status === "ready" ? (
              <Link
                to="/d/$datasetId"
                params={{ datasetId: dataset.id }}
                className="block truncate font-medium text-ink-1 hover:text-accent"
              >
                {dataset.name}
              </Link>
            ) : (
              <span className="block truncate font-medium text-ink-2">{dataset.name}</span>
            )}
            <span className="block truncate font-mono text-[11px] text-ink-3" title={dataset.source_path}>
              {dataset.source_path}
            </span>
          </div>
        )}
      </td>
      <td className="px-3 py-2 text-right whitespace-nowrap text-ink-2 tabular-nums">
        {formatCount(dataset.row_count)} × {formatCount(dataset.col_count)}
      </td>
      <td className="px-3 py-2 text-right whitespace-nowrap text-ink-2 tabular-nums">
        {formatBytes(dataset.size_bytes)}
      </td>
      <td className="px-3 py-2 whitespace-nowrap text-ink-2">
        {formatRelativeTime(dataset.last_opened_at ?? dataset.created_at)}
      </td>
      <td className="px-3 py-2 whitespace-nowrap">
        <StatusBadge dataset={dataset} />
      </td>
      <td className="px-2 py-2">
        <div className="flex justify-end gap-0.5 opacity-60 transition-opacity group-focus-within:opacity-100 group-hover:opacity-100">
          <Button
            variant="ghost"
            icon={Pencil}
            iconOnly
            aria-label={t("datasets.rename")}
            title={t("datasets.rename")}
            onClick={() => {
              setDraft(dataset.name);
              setEditing(true);
            }}
          />
          <Button
            variant="ghost"
            icon={RefreshCw}
            iconOnly
            aria-label={t("datasets.reimport")}
            title={t("datasets.reimport")}
            disabled={dataset.status === "importing" || dataset.source_state === "missing"}
            onClick={onReimport}
          />
          <Button
            variant="ghost"
            icon={dataset.status === "importing" ? X : Trash2}
            iconOnly
            aria-label={t(dataset.status === "importing" ? "datasets.cancelImport" : "datasets.delete")}
            title={t(dataset.status === "importing" ? "datasets.cancelImport" : "datasets.delete")}
            onClick={onDelete}
          />
        </div>
      </td>
    </tr>
  );
}
