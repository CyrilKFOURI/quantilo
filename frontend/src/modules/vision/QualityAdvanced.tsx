import { Ban, ChevronDown, RotateCcw, Search } from "lucide-react";
import { useState } from "react";

import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { useFindSemantic, useMislabelled, useSemantic, type Mislabelled, type Semantic } from "./api";
import type { Decision, Draft, ImageRow } from "./model";
import { Thumbs } from "./QualityThumbs";

type AdvancedProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  draft: Draft;
  onDecide: (keys: string[], decision: Decision | null) => void;
  onOpen: (row: ImageRow) => void;
};

const GROUPS = 12;

/**
 * b2, Quality › Advanced: the two rules that need more than the import. Nothing is read before
 * the fold opens, and the semantic near-copies (R-VQ-17) are computed only when the user asks
 * (onion rule): R-VQ-16 reads the latest classification run, R-VQ-17 the DINOv2 vectors.
 */
export function QualityAdvanced(props: AdvancedProps) {
  const [open, setOpen] = useState(false);
  return (
    <details className="group rounded-md border border-line bg-surface-1" onToggle={(event) => setOpen(event.currentTarget.open)}>
      <summary className="flex min-h-9 cursor-pointer list-none items-center gap-1.5 px-3 text-[13px] text-ink-1 pointer-coarse:min-h-11">
        <ChevronDown size={15} strokeWidth={1.5} aria-hidden className="text-ink-3 transition-transform duration-150 group-not-open:-rotate-90" />
        {t("vision.quality.advanced")}
      </summary>
      {open && (
        <div className="space-y-6 border-t border-line px-3 py-4">
          <MislabelledSection {...props} />
          <SemanticSection {...props} />
        </div>
      )}
    </details>
  );
}

function keysOf(rows: readonly ImageRow[], positions: readonly number[]): string[] {
  return positions.map((position) => rows[position]?.key).filter((key): key is string => key !== undefined);
}

function DecideButtons({ keys, draft, onDecide }: { keys: string[]; draft: Draft; onDecide: AdvancedProps["onDecide"] }) {
  const decided = keys.filter((key) => draft[key] !== undefined);
  if (keys.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1">
      <Button variant="ghost" icon={Ban} onClick={() => onDecide(keys, "exclude")}>
        {t("vision.quality.excludeThese", { count: formatCount(keys.length) })}
      </Button>
      {decided.length > 0 && (
        <Button variant="ghost" icon={RotateCcw} onClick={() => onDecide(decided, null)}>
          {t("vision.decision.reset")}
        </Button>
      )}
    </div>
  );
}

function MislabelledSection({ datasetId, rows, draft, onDecide, onOpen }: AdvancedProps) {
  const found = useMislabelled(datasetId, true);
  return (
    <section aria-labelledby="vision-mislabelled" className="space-y-2">
      <h3 id="vision-mislabelled" className="flex flex-wrap items-baseline gap-x-2 text-[14px] font-semibold text-ink-1">
        <span className="font-mono text-xs font-normal text-ink-3">R-VQ-16</span>
        {t("vision.quality.mislabelled")}
        {found.data && found.data.count > 0 && (
          <span className="text-[13px] font-normal text-ink-2 tabular-nums">{t("vision.quality.count", { count: formatCount(found.data.count) })}</span>
        )}
      </h3>
      {found.isPending && <Loading label={t("vision.quality.mislabelledLoading")} compact className="h-16 rounded-md" />}
      {found.isError && <ErrorNote>{found.error.message}</ErrorNote>}
      {found.data && <MislabelledBody data={found.data} datasetId={datasetId} rows={rows} draft={draft} onDecide={onDecide} onOpen={onOpen} />}
    </section>
  );
}

function MislabelledBody({ data, datasetId, rows, draft, onDecide, onOpen }: AdvancedProps & { data: Mislabelled }) {
  const captions = new Map(data.images.map((image) => [image.row, `${image.label} → ${image.suggested} ${Math.round(image.suggested_proba * 100)} %`]));
  return (
    <>
      <p className="max-w-prose text-[13px] text-ink-1">{data.message}</p>
      <p className="max-w-prose text-[13px] text-ink-2">{data.reason}</p>
      {data.run_id && (
        <p className="text-xs text-ink-3">
          {t("vision.quality.mislabelledFrom", { model: data.model ?? "", checked: formatCount(data.checked) })} {data.notes.join(" ")}
        </p>
      )}
      {data.rows.length > 0 && (
        <>
          <Thumbs datasetId={datasetId} rows={rows} positions={data.rows} draft={draft} onOpen={onOpen} captions={captions} />
          <DecideButtons keys={keysOf(rows, data.rows)} draft={draft} onDecide={onDecide} />
        </>
      )}
    </>
  );
}

function SemanticSection({ datasetId, rows, draft, onDecide, onOpen }: AdvancedProps) {
  const state = useSemantic(datasetId, true);
  const find = useFindSemantic(datasetId);
  const [shown, setShown] = useState(GROUPS);
  const data = state.data;
  return (
    <section aria-labelledby="vision-semantic" className="space-y-2">
      <h3 id="vision-semantic" className="flex flex-wrap items-baseline gap-x-2 text-[14px] font-semibold text-ink-1">
        <span className="font-mono text-xs font-normal text-ink-3">R-VQ-17</span>
        {t("vision.quality.semantic")}
      </h3>
      <p className="max-w-prose text-[13px] text-ink-2">{t("vision.quality.semanticHelp", { cosine: String(data?.cosine ?? 0.95) })}</p>
      {state.isPending && <Loading label={t("vision.quality.semanticLoading")} compact className="h-16 rounded-md" />}
      {state.isError && <ErrorNote>{state.error.message}</ErrorNote>}
      {find.isError && <ErrorNote>{find.error.message}</ErrorNote>}
      {data && <SemanticBody data={data} onFind={(confirm) => find.mutate(confirm)} busy={find.isPending} />}
      {data?.state === "ready" && (
        <>
          <p className="text-[13px] text-ink-1">{data.style ?? data.sentence}</p>
          {(data.leaks ?? 0) > 0 && <p className="text-xs text-bad">{t("vision.quality.semanticLeaks", { count: formatCount(data.leaks ?? 0) })}</p>}
          <ul className="space-y-3">
            {data.groups.slice(0, shown).map((group) => {
              const keys = keysOf(rows, group.rows);
              return (
                <li key={group.group} className="space-y-1.5 rounded-md border border-line px-3 py-2">
                  <p className="flex flex-wrap gap-x-3 text-xs text-ink-2 tabular-nums">
                    <span>{t("vision.quality.copies", { count: formatCount(group.rows.length) })}</span>
                    <span>{t("vision.quality.semanticCosine", { cosine: group.cosine.toFixed(3) })}</span>
                    {group.two_labels && <span className="text-bad">{t("vision.quality.twoLabels")}</span>}
                    {group.leak && <span className="text-bad">{t("vision.quality.twoSplits")}</span>}
                  </p>
                  <Thumbs datasetId={datasetId} rows={rows} positions={group.rows} draft={draft} onOpen={onOpen} />
                  <DecideButtons keys={keys.slice(1)} draft={draft} onDecide={onDecide} />
                </li>
              );
            })}
          </ul>
          {data.groups.length > shown && (
            <Button onClick={() => setShown(shown + GROUPS)}>
              {t("vision.quality.moreGroups", { count: formatCount(Math.min(GROUPS, data.groups.length - shown)) })}
            </Button>
          )}
        </>
      )}
    </section>
  );
}

function SemanticBody({ data, onFind, busy }: { data: Semantic; onFind: (confirmLong: boolean) => void; busy: boolean }) {
  if (data.state === "running") {
    return (
      <p className="text-[13px] text-ink-2 tabular-nums">
        {data.message ?? t("vision.quality.semanticRunning")}
        {data.total ? ` · ${formatCount(data.done ?? 0)} / ${formatCount(data.total)}` : ""}
      </p>
    );
  }
  if (!data.available) {
    return <p className="max-w-prose text-[13px] text-ink-2">{t("vision.quality.semanticGreyed", { reason: data.reason ?? "" })}</p>;
  }
  const cached = t("vision.quality.semanticCached", { cached: formatCount(data.cached), images: formatCount(data.images), network: data.network });
  if (data.state === "failed" && data.long_run) {
    return (
      <div className="space-y-1.5">
        <p className="text-[13px] text-ink-1">{data.error}</p>
        <Button icon={Search} disabled={busy} onClick={() => onFind(true)}>
          {t("vision.quality.semanticGoOn")}
        </Button>
      </div>
    );
  }
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <Button icon={Search} disabled={busy} onClick={() => onFind(false)}>
        {data.state === "ready" ? t("vision.quality.semanticAgain") : t("vision.quality.semanticFind")}
      </Button>
      <span className="text-xs text-ink-3 tabular-nums">{cached}</span>
      {data.state === "failed" && <ErrorNote>{data.error}</ErrorNote>}
    </div>
  );
}
