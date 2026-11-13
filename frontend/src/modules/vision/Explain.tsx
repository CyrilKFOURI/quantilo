import { HelpCircle } from "lucide-react";
import { useMemo } from "react";

import { imageUrl } from "../../features/import/imageSetApi";
import { t } from "../../lib/i18n";
import { ErrorNote, Select } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { Loading } from "../../ui/Loading";
import { useExplainOverview, useVisionRuns, type ExplainOverview, type WhyImage } from "./explainApi";
import type { ImageRow } from "./model";
import { ExplainMore } from "./MoreExplain";

/**
 * 117 (study §§ 8, 9.1): the Explain tab, what the model reads across the whole set. First plane:
 * for each class, the prototypes (k-medoids in the model's own space, with the share of the class
 * each stands for) and the criticisms (the images the prototypes represent worst, MMD-critic);
 * with a classic logistic model, the descriptor families it leans on (permutation importance).
 * Every image opens its own "Why?". Computed once per run and model, then cached with the run.
 */

type ExplainProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  run: string | null;
  model: string | null;
  onRun: (run: string | undefined) => void;
  onModel: (model: string | undefined) => void;
  onWhy: (row: ImageRow) => void;
};

export function Explain({ datasetId, rows, run, model, onRun, onModel, onWhy }: ExplainProps) {
  const runs = useVisionRuns(datasetId);
  const explained = (runs.data ?? []).filter((summary) => summary.task !== "clusters");
  const chosen = explained.find((summary) => summary.id === run) ?? explained[0] ?? null;
  const overview = useExplainOverview(datasetId, chosen?.id ?? null, model);
  const byId = useMemo(() => new Map(rows.map((r) => [r.id, r])), [rows]);

  if (runs.isPending) return <Loading label={t("vision.why.loadingRuns")} className="h-48 rounded-lg" />;
  if (runs.isError) return <ErrorNote>{(runs.error as Error).message}</ErrorNote>;
  if (!chosen) {
    return (
      <div className="rounded-lg border border-dashed border-line px-5 py-10 text-center text-[13px]">
        <p className="font-medium text-ink-1">{t("vision.why.noRun")}</p>
        <p className="mx-auto mt-1 max-w-[56ch] text-ink-2">{t("vision.explain.noRunHint")}</p>
      </div>
    );
  }

  return (
    <div className="min-w-0 space-y-6 text-[13px]">
      <div className="flex min-w-0 flex-wrap items-end gap-x-4 gap-y-2">
        {explained.length > 1 && (
          <label className="flex min-w-0 flex-col gap-1">
            <span className="text-xs text-ink-2">{t("vision.why.run")}</span>
            <Select value={chosen.id} onChange={(event) => onRun(event.target.value === explained[0]?.id ? undefined : event.target.value)} className="max-w-full">
              {explained.map((summary) => (
                <option key={summary.id} value={summary.id}>
                  {new Date(summary.created_at).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })} · {summary.default_title ?? summary.task}
                </option>
              ))}
            </Select>
          </label>
        )}
        {overview.data && overview.data.models.length > 1 && (
          <label className="flex min-w-0 flex-col gap-1">
            <span className="text-xs text-ink-2">{t("vision.why.model")}</span>
            <Select
              value={overview.data.model.key}
              onChange={(event) => onModel(overview.data?.models.find((m) => m.key === event.target.value)?.default ? undefined : event.target.value)}
              className="max-w-full"
            >
              {overview.data.models.map((option) => (
                <option key={option.key} value={option.key}>
                  {option.title}
                  {option.default ? ` (${t("vision.why.runDefault")})` : ""}
                </option>
              ))}
            </Select>
          </label>
        )}
      </div>

      <p className="flex max-w-[75ch] gap-2 text-ink-2">
        <HelpCircle size={15} strokeWidth={1.5} className="mt-0.5 shrink-0 text-ink-3" aria-hidden />
        {t("vision.explain.intro")}
      </p>

      {overview.isPending ? (
        <Loading label={t("vision.explain.loading")} className="h-72 rounded-lg" />
      ) : overview.isError ? (
        <ErrorNote>{(overview.error as Error).message}</ErrorNote>
      ) : (
        <Overview datasetId={datasetId} overview={overview.data} byId={byId} onWhy={onWhy} />
      )}

      {/* 118: Advanced › More, nothing asked of the server before it is opened */}
      {chosen.task === "classification" && <ExplainMore datasetId={datasetId} runId={chosen.id} byId={byId} onWhy={onWhy} />}
    </div>
  );
}

function Overview({
  datasetId,
  overview,
  byId,
  onWhy,
}: {
  datasetId: string;
  overview: ExplainOverview;
  byId: ReadonlyMap<string, ImageRow>;
  onWhy: (row: ImageRow) => void;
}) {
  return (
    <div className="min-w-0 space-y-8">
      <section className="min-w-0 space-y-3" aria-labelledby="explain-prototypes">
        <div className="space-y-1">
          <h3 id="explain-prototypes" className="flex items-center gap-1.5 text-sm font-semibold text-ink-1">
            {t("vision.explain.prototypes")}
            <InfoTip label={t("vision.explain.prototypes")} text={t("vision.explain.prototypesTip")} />
          </h3>
          <p className="max-w-[75ch] text-ink-1">{overview.sentence}</p>
          <p className="max-w-[75ch] text-xs text-ink-3">{t("vision.why.space", { space: overview.space })}</p>
        </div>
        <div className="divide-y divide-line rounded-lg border border-line bg-surface-1">
          {overview.classes.map((group) => (
            <div key={group.label} className="grid min-w-0 gap-4 px-4 py-4 lg:grid-cols-[180px_minmax(0,1fr)_minmax(0,1fr)]">
              <div className="min-w-0">
                <p className="truncate font-mono font-semibold text-ink-1">{group.label}</p>
                <p className="text-xs text-ink-3 tabular-nums">{t("vision.explain.images", { count: group.size.toLocaleString("en-US") })}</p>
                {group.notes.map((note) => (
                  <p key={note} className="mt-1 text-xs text-ink-3">
                    {note}
                  </p>
                ))}
              </div>
              <Thumbs
                title={t("vision.explain.typical")}
                datasetId={datasetId}
                refs={group.prototypes}
                byId={byId}
                onWhy={onWhy}
                detail={(ref) => t("vision.explain.share", { share: `${Math.round((ref.share ?? 0) * 100)} %` })}
              />
              <Thumbs
                title={t("vision.explain.critics")}
                datasetId={datasetId}
                refs={group.criticisms}
                byId={byId}
                onWhy={onWhy}
                empty={t("vision.explain.noCritics")}
                detail={(ref) => t("vision.explain.witness", { value: (ref.witness ?? 0).toFixed(3) })}
              />
            </div>
          ))}
        </div>
      </section>

      {overview.families && overview.families.length > 0 && <Families overview={overview} />}

      {overview.notes.map((note) => (
        <p key={note} className="text-xs text-ink-3">
          {note}
        </p>
      ))}
    </div>
  );
}

function Thumbs({
  title,
  datasetId,
  refs,
  byId,
  onWhy,
  detail,
  empty,
}: {
  title: string;
  datasetId: string;
  refs: readonly WhyImage[];
  byId: ReadonlyMap<string, ImageRow>;
  onWhy: (row: ImageRow) => void;
  detail: (ref: WhyImage) => string;
  empty?: string;
}) {
  return (
    <div className="min-w-0 space-y-1.5">
      <p className="text-xs text-ink-2">{title}</p>
      {refs.length === 0 ? (
        <p className="text-xs text-ink-3">{empty}</p>
      ) : (
        <ul className="grid grid-cols-3 gap-2">
          {refs.map((ref) => {
            const row = byId.get(ref.image_id);
            return (
              <li key={ref.row} className="min-w-0">
                <button
                  type="button"
                  disabled={!row}
                  onClick={() => row && onWhy(row)}
                  title={row ? t("vision.explain.openWhy", { name: row.name }) : ref.image_id}
                  className="group/thumb block w-full text-left"
                >
                  <span className="relative block aspect-square overflow-hidden rounded-md bg-surface-2">
                    <img src={imageUrl(datasetId, ref.image_id, "thumb")} alt={row?.name ?? ref.image_id} loading="lazy" decoding="async" className="size-full object-cover" />
                    <span aria-hidden className="pointer-events-none absolute inset-0 rounded-md ring-1 ring-ink-1/8 ring-inset group-hover/thumb:ring-2 group-hover/thumb:ring-accent" />
                  </span>
                  <span className="mt-1 block truncate text-[11px] text-ink-3 tabular-nums">{detail(ref)}</span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}

/** R-VEX-13: one series, so one hue and no legend box; the title names it, values sit at the bars. */
function Families({ overview }: { overview: ExplainOverview }) {
  const families = overview.families ?? [];
  const top = Math.max(0.001, ...families.map((family) => family.drop + family.spread));
  return (
    <section className="min-w-0 space-y-3" aria-labelledby="explain-families">
      <div className="space-y-1">
        <h3 id="explain-families" className="flex items-center gap-1.5 text-sm font-semibold text-ink-1">
          {t("vision.explain.families")}
          <InfoTip label={t("vision.explain.families")} text={t("vision.explain.familiesTip")} />
        </h3>
        {overview.families_sentence && <p className="max-w-[75ch] text-ink-1">{overview.families_sentence}</p>}
      </div>
      <table className="w-full max-w-2xl text-xs">
        <caption className="sr-only">{t("vision.explain.families")}</caption>
        <thead>
          <tr className="text-left text-ink-3">
            <th scope="col" className="py-1 pr-3 font-normal">
              {t("vision.explain.family")}
            </th>
            <th scope="col" className="w-1/2 py-1 pr-3 font-normal">
              {t("vision.explain.drop")}
            </th>
            <th scope="col" className="py-1 text-right font-normal">
              {t("vision.explain.columns")}
            </th>
          </tr>
        </thead>
        <tbody>
          {families.map((family) => (
            <tr key={family.key} className="border-t border-line">
              <th scope="row" className="py-2 pr-3 text-left font-normal text-ink-1">
                {family.title}
              </th>
              <td className="py-2 pr-3">
                <span className="flex items-center gap-2">
                  <span className="relative h-2 min-w-0 flex-1" aria-hidden>
                    <span className="absolute inset-y-0 left-0 rounded-r-[2px] bg-(--chart-1)" style={{ width: `${Math.max(0.5, (Math.max(0, family.drop) / top) * 100)}%` }} />
                  </span>
                  <span className="w-24 shrink-0 text-right text-ink-1 tabular-nums">
                    {family.drop >= 0 ? "" : "−"}
                    {Math.abs(family.drop).toFixed(3)} <span className="text-ink-3">± {family.spread.toFixed(3)}</span>
                  </span>
                </span>
              </td>
              <td className="py-2 text-right text-ink-2 tabular-nums">{family.columns.toLocaleString("en-US")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
