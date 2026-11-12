import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { Column, DistinctiveWords } from "../../api/client";
import { t } from "../../lib/i18n";
import { ErrorNote, Field, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { Folded } from "./Folded";
import { WordsPanel } from "./WordsPanel";

/** A text whose words can be read (R-WRD-06): not a list of choices. */
export const wordColumns = (columns: Column[] | undefined) =>
  (columns ?? []).filter((c) => !c.excluded && c.type === "text" && c.role !== "choices");

/** 059 (R-WRD-07): "Advanced · Distinctive words" of a page that already has its first plane
 *  (Segments, Changes): folded, and asked for only once opened. `fetch` reads one text column. */
export function TextWords({
  columns,
  queryKey,
  fetch,
  hint,
}: {
  columns: Column[];
  queryKey: readonly unknown[];
  fetch: (column: string) => Promise<DistinctiveWords | null>;
  hint: string;
}) {
  if (!columns.length) return null;
  return (
    <Folded title={t("words.advanced")} hint={hint}>
      <WordsBody columns={columns} queryKey={queryKey} fetch={fetch} />
    </Folded>
  );
}

function WordsBody({
  columns,
  queryKey,
  fetch,
}: {
  columns: Column[];
  queryKey: readonly unknown[];
  fetch: (column: string) => Promise<DistinctiveWords | null>;
}) {
  const [column, setColumn] = useState(columns[0]?.name ?? "");
  const words = useQuery({
    queryKey: [...queryKey, column],
    queryFn: () => fetch(column),
    enabled: Boolean(column),
    staleTime: Infinity,
    retry: false,
  });
  return (
    <div className="space-y-3">
      {columns.length > 1 && (
        <Field label={t("words.column")}>
          <Select value={column} onChange={(event) => setColumn(event.target.value)} className="w-56">
            {columns.map((c) => (
              <option key={c.name} value={c.name}>
                {c.name}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {words.isError ? (
        <ErrorNote>{words.error.message}</ErrorNote>
      ) : words.isPending ? (
        <Loading label={t("words.loading")} className="h-32 rounded-lg" />
      ) : words.data ? (
        <WordsPanel words={words.data} />
      ) : null}
    </div>
  );
}
