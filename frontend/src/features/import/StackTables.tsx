import { useQuery } from "@tanstack/react-query";
import { Layers } from "lucide-react";
import { useState } from "react";

import { api } from "../../api/client";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";

/** S1 (C1), in Advanced: the tables of the workbook that share their columns, looked for when
 *  asked (every sheet is read: never when the step opens), each group with its reason and
 *  "Import stacked": one dataset with a `source_sheet` column. */
export function StackTables({ path, busy, onImport }: { path: string; busy: boolean; onImport: (tables: string[]) => void }) {
  const [asked, setAsked] = useState(false);
  const stacks = useQuery({ queryKey: ["sources", "stacks", path], queryFn: () => api.stacks(path), enabled: asked, staleTime: 60_000 });
  return (
    <section className="mt-3 space-y-2 border-t border-line pt-2" aria-label={t("preview.stack.title")}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-ink-2">{t("preview.stack.hint")}</p>
        {!asked && (
          <Button onClick={() => setAsked(true)}>
            <Layers size={14} strokeWidth={1.5} aria-hidden />
            {t("preview.stack.find")}
          </Button>
        )}
      </div>
      {asked && stacks.isPending && <Loading label={t("preview.stack.loading")} className="h-16 rounded-md" compact />}
      {stacks.isError && <ErrorNote>{stacks.error.message}</ErrorNote>}
      {stacks.data && stacks.data.length === 0 && <p className="text-xs text-ink-3">{t("preview.stack.none")}</p>}
      {stacks.data?.map((stack) => (
        <div key={stack.tables.join("|")} className="flex flex-wrap items-start justify-between gap-2 rounded-md border border-line px-3 py-2">
          <p className="min-w-0 flex-1 basis-64 text-xs text-ink-2">
            <span className="font-medium text-ink-1">{t("preview.stack.group", { count: String(stack.tables.length), rows: formatCount(stack.rows) })}</span> {stack.reason}
          </p>
          <Button variant="primary" disabled={busy} onClick={() => onImport(stack.tables)}>
            {t("preview.stack.import")}
          </Button>
        </div>
      ))}
    </section>
  );
}
