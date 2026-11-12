import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import type { MlPreparation } from "../../api/client";
import { request } from "../../api/client";
import type { components } from "../../api/schema";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { Dialog } from "../../ui/Dialog";
import { ErrorNote, Select } from "../../ui/fields";

type Status = components["schemas"]["EmbeddingsStatus"];
type OnColumn = (column: string, key: "encoding", value: string | undefined) => void;

const statusKey = ["ml", "text-embeddings"] as const;
const PATH = "/modules/ml/text-embeddings";

/** 091: "Text representation: TF-IDF · Sentence embeddings", in Advanced. TF-IDF stays the default;
 *  embeddings are greyed with their reason until the `embed` extra and the model are on this machine,
 *  and the model is downloaded only after the user confirms (size shown). */
export function TextRepresentation({ preparation, onColumn }: { preparation: MlPreparation; onColumn: OnColumn }) {
  const texts = preparation.decisions.filter((d) => d.use && d.type === "text");
  // no free text: nothing to show, and no request
  return texts.length > 0 ? <TextColumns texts={texts} onColumn={onColumn} /> : null;
}

function TextColumns({ texts, onColumn }: { texts: MlPreparation["decisions"]; onColumn: OnColumn }) {
  const client = useQueryClient();
  const status = useQuery({
    queryKey: statusKey,
    queryFn: () => request<Status>("GET", PATH),
    refetchInterval: (query) => (query.state.data?.download === "running" ? 2_000 : false),
  });
  const download = useMutation({
    mutationFn: () => request<Status>("POST", `${PATH}/download`),
    onSuccess: (data) => client.setQueryData(statusKey, data),
  });
  const [asking, setAsking] = useState(false);
  const found = status.data;
  return (
    <section className="mt-3 space-y-2" aria-label={t("ml.prep.textRep.title")}>
      <h4 className="text-[13px] font-medium text-ink-1">{t("ml.prep.textRep.title")}</h4>
      <p className="max-w-prose text-ink-2">{t("ml.prep.textRep.intro")}</p>
      {found && !found.available && <p className="max-w-prose text-ink-3">{found.reason}</p>}
      {found && found.installed && !found.downloaded && (
        <Button variant="secondary" disabled={found.download === "running"} onClick={() => setAsking(true)}>
          {found.download === "running" ? t("ml.prep.textRep.downloading") : t("ml.prep.textRep.download", { size: found.size })}
        </Button>
      )}
      {found?.error && <ErrorNote>{found.error}</ErrorNote>}
      <ul className="grid grid-cols-[repeat(auto-fill,minmax(min(100%,16rem),1fr))] gap-2">
        {texts.map((d) => {
          const reason = d.options.encoding?.find((o) => o.value === "embeddings")?.reason ?? null;
          const value = d.encoding === "embeddings" ? "embeddings" : "tfidf";
          return (
            <li key={d.column} className="flex min-w-0 items-center gap-2">
              <span className="min-w-0 truncate font-mono text-ink-1">{d.column}</span>
              <Select
                aria-label={t("ml.prep.textRep.for", { column: d.column })}
                value={value}
                onChange={(event) => onColumn(d.column, "encoding", event.target.value === "embeddings" ? "embeddings" : undefined)}
                className="ml-auto h-7 max-w-48 text-xs"
              >
                <option value="tfidf">{t("ml.prep.textRep.tfidf")}</option>
                <option value="embeddings" disabled={Boolean(reason)}>
                  {reason ? `${t("ml.prep.encoding.embeddings")} — ${reason}` : t("ml.prep.encoding.embeddings")}
                </option>
              </Select>
            </li>
          );
        })}
      </ul>
      {found && (
        <Dialog
          open={asking}
          onOpenChange={setAsking}
          size="sm"
          title={t("ml.prep.textRep.confirmTitle")}
          description={t("ml.prep.textRep.confirmBody", { model: found.model, size: found.size, license: found.license })}
          footer={
            <>
              <Button variant="ghost" onClick={() => setAsking(false)}>
                {t("ml.prep.textRep.cancel")}
              </Button>
              <Button
                variant="primary"
                onClick={() => {
                  setAsking(false);
                  download.mutate();
                }}
              >
                {t("ml.prep.textRep.confirm", { size: found.size })}
              </Button>
            </>
          }
        >
          <p className="text-[13px] text-ink-2">{t("ml.prep.textRep.confirmNote")}</p>
        </Dialog>
      )}
    </section>
  );
}
