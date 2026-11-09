import { ArrowUp, FileSpreadsheet, Folder } from "lucide-react";
import { useState } from "react";

import { useBrowse } from "../../api/hooks";
import { formatBytes } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, TextInput } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

type BrowseDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSelect: (path: string) => void;
};

export function BrowseDialog({ open, onOpenChange, onSelect }: BrowseDialogProps) {
  const [path, setPath] = useState<string | null>(null);
  const [typed, setTyped] = useState("");
  const listing = useBrowse(path, open);

  const choose = (selected: string) => {
    onSelect(selected);
    onOpenChange(false);
  };

  const data = listing.data;

  return (
    <Dialog
      open={open}
      onOpenChange={onOpenChange}
      title={t("browse.title")}
      description={t("browse.description")}
      size="md"
      footer={
        <>
          <Button variant="ghost" onClick={() => onOpenChange(false)}>
            {t("import.cancel")}
          </Button>
          <Button variant="primary" disabled={!data} onClick={() => data && choose(data.path)}>
            {t("browse.useFolder")}
          </Button>
        </>
      }
    >
      <form
        className="mb-3 flex gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          if (typed.trim()) setPath(typed.trim());
        }}
      >
        <TextInput
          aria-label={t("browse.pathPlaceholder")}
          placeholder={t("browse.pathPlaceholder")}
          value={typed}
          onChange={(event) => setTyped(event.target.value)}
          className="flex-1 font-mono text-xs"
        />
        <Button type="submit">{t("browse.go")}</Button>
      </form>

      {listing.isError && <ErrorNote>{listing.error.message}</ErrorNote>}

      {data && (
        <>
          <div className="mb-2 flex items-center gap-2">
            <Button
              variant="ghost"
              icon={ArrowUp}
              iconOnly
              aria-label={t("browse.up")}
              title={t("browse.up")}
              disabled={!data.parent}
              onClick={() => setPath(data.parent ?? null)}
            />
            <span className="truncate font-mono text-xs text-ink-2" title={data.path}>
              {data.path}
            </span>
            {data.roots.length > 1 && (
              <select
                aria-label={t("browse.location")}
                className="ml-auto h-7 rounded-md border border-line bg-surface-1 px-1.5 text-xs"
                value=""
                onChange={(event) => event.target.value && setPath(event.target.value)}
              >
                <option value="">{t("browse.goTo")}</option>
                {data.roots.map((root) => (
                  <option key={root} value={root}>
                    {root}
                  </option>
                ))}
              </select>
            )}
          </div>

          <ul className="max-h-[50dvh] divide-y divide-line overflow-y-auto rounded-md border border-line">
            {data.entries.length === 0 && (
              <li className="px-3 py-6 text-center text-ink-3">{t("browse.empty")}</li>
            )}
            {data.entries.map((entry) => (
              <li key={entry.path}>
                <button
                  type="button"
                  onClick={() => (entry.kind === "dir" ? setPath(entry.path) : choose(entry.path))}
                  className="flex w-full items-center gap-2.5 px-3 py-1.5 text-left hover:bg-surface-2"
                >
                  {entry.kind === "dir" ? (
                    <Folder size={15} strokeWidth={1.5} className="text-ink-3" aria-hidden />
                  ) : (
                    <FileSpreadsheet size={15} strokeWidth={1.5} className="text-accent" aria-hidden />
                  )}
                  <span className="min-w-0 flex-1 truncate">{entry.name}</span>
                  {entry.kind === "file" && (
                    <span className="text-xs text-ink-3 tabular-nums">
                      {entry.format} · {formatBytes(entry.size)}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
      {listing.isPending && <Loading label={t("browse.loading")} className="h-24 rounded-md" compact />}
    </Dialog>
  );
}
