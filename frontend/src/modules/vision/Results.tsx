import { ArrowRight, Star } from "lucide-react";

import { Plot } from "../../charts/Plot";
import { formatCount, formatRelativeTime } from "../../lib/format";
import { t, type MessageKey } from "../../lib/i18n";
import { Button } from "../../ui/Button";
import { ErrorNote, Select } from "../../ui/fields";
import { Loading } from "../../ui/Loading";
import { AnomalyResults } from "./AnomalyResults";
import { ClassificationResults } from "./ClassificationResults";
import type { ImageRow } from "./model";
import { MoreResults } from "./MoreResults";
import { MultilabelResults, RegressionResults } from "./TargetResults";
import { usePredictions, useRun, useRuns, type Prediction, type VisionModel, type VisionResult } from "./modelApi";
import { barsFigure } from "./resultFigures";
import { mainScore, metric, rankModels, text } from "./scores";
import { Block, Gallery, Tiles } from "./resultsUi";

type ResultsProps = {
  datasetId: string;
  rows: readonly ImageRow[];
  runId: string | undefined;
  modelKey: string | undefined;
  threshold: number | undefined;
  set: (patch: { run?: string; model?: string; threshold?: number }) => void;
  onOpenImage: (row: ImageRow) => void;
  onModelTab: () => void;
  /** 117: open an image's "Why?" (classification images reach it through their sheet). */
  onWhy?: (row: ImageRow) => void;
};

const METRIC: Record<string, MessageKey> = {
  balanced_accuracy: "vision.results.m.balanced_accuracy",
  accuracy: "vision.results.m.accuracy",
  auroc: "vision.results.m.auroc",
  flagged: "vision.results.m.flagged",
  r2: "vision.target.m.r2",
  macro_f1: "vision.target.m.macroF1",
};

/**
 * 113: the Results tab. A kept run (the latest by default), the sentence that says what it found,
 * every model ranked by its main metric (the default first), then every metric of the chosen
 * model. Run, model and a moved threshold are in the URL (R-113-11).
 */
export function Results({ datasetId, rows, runId, modelKey, threshold, set, onOpenImage, onModelTab, onWhy }: ResultsProps) {
  const runs = useRuns(datasetId, true);
  const chosen = runId ?? runs.data?.[0]?.id;
  const run = useRun(datasetId, chosen);
  const predictions = usePredictions(datasetId, chosen);

  if (runs.isPending) return <Loading label={t("vision.results.loadingRuns")} className="h-72 rounded-lg" />;
  if (runs.isError) return <ErrorNote>{(runs.error as Error).message}</ErrorNote>;
  if (!chosen)
    return (
      <div className="flex flex-col items-start gap-3 rounded-lg border border-dashed border-line px-5 py-8">
        <p className="text-[15px] font-medium text-ink-1">{t("vision.results.emptyTitle")}</p>
        <p className="max-w-prose text-[13px] text-ink-2">{t("vision.results.emptyBody")}</p>
        <Button variant="primary" onClick={onModelTab}>
          {t("vision.results.toModel")} <ArrowRight size={14} strokeWidth={1.5} aria-hidden />
        </Button>
      </div>
    );
  if (run.isError) return <ErrorNote>{(run.error as Error).message}</ErrorNote>;
  if (run.isPending || predictions.isPending) return <Loading label={t("vision.results.loadingRun")} className="h-96 rounded-lg" />;
  if (predictions.isError) return <ErrorNote>{(predictions.error as Error).message}</ErrorNote>;

  const result = run.data.result;
  const ranked = rankModels(result.models);
  const model = ranked.find((m) => m.key === modelKey) ?? ranked[0];
  const openRow = (row: number) => {
    const image = rows[row];
    if (image) onOpenImage(image);
  };

  return (
    <div className="flex flex-col gap-7">
      <header className="flex flex-col gap-3 border-b border-line pb-5">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <label className="flex min-w-0 flex-col gap-1">
            <span className="text-xs text-ink-2">{t("vision.results.run")}</span>
            <Select value={chosen} onChange={(event) => set({ run: event.target.value, model: undefined, threshold: undefined })} className="max-w-full min-w-0 sm:min-w-80">
              {(runs.data ?? []).map((summary) => (
                <option key={summary.id} value={summary.id}>
                  {`${formatRelativeTime(summary.created_at)}, ${summary.default_title ?? summary.task}${summary.score !== null && summary.score !== undefined ? `, ${metric(summary.score)}` : ""}`}
                </option>
              ))}
            </Select>
          </label>
          <p className="text-xs text-ink-3 tabular-nums">
            {t("vision.results.runFacts", { scored: formatCount(new Set(result.split.folds.flatMap((f) => f.test)).size), seconds: result.seconds.toFixed(0), split: result.split.code })}
          </p>
        </div>
        <p className="max-w-[80ch] text-[15px] text-pretty text-ink-1">{result.sentence}</p>
        {result.notes.map((note) => (
          <p key={note} className="text-xs text-ink-3">
            {note}
          </p>
        ))}
      </header>

      {ranked.length > 0 && model && (
        <ModelTable models={ranked} current={model.key} onPick={(key) => set({ model: key, threshold: undefined })} />
      )}

      {model && <ModelHead model={model} />}
      {model?.level === "more" && <MoreResults model={model} />}

      {result.task.task === "clusters" ? (
        <ClusterResults datasetId={datasetId} rows={rows} result={result} predictions={predictions.data} onOpen={openRow} />
      ) : model?.regression ? (
        <RegressionResults key={`${chosen}-${model.key}`} datasetId={datasetId} rows={rows} model={model} predictions={predictions.data} target={result.task.target} onOpen={(row) => openRow(row)} />
      ) : model?.multilabel ? (
        <MultilabelResults key={`${chosen}-${model.key}`} datasetId={datasetId} rows={rows} model={model} predictions={predictions.data} onOpen={(row) => openRow(row)} />
      ) : model?.classification ? (
        <ClassificationResults
          key={`${chosen}-${model.key}`}
          datasetId={datasetId}
          rows={rows}
          model={model}
          report={model.classification}
          predictions={predictions.data}
          onOpen={(row) => openRow(row)}
        />
      ) : model ? (
        <AnomalyResults
          key={`${chosen}-${model.key}`}
          datasetId={datasetId}
          runId={chosen}
          rows={rows}
          model={model}
          predictions={predictions.data}
          threshold={threshold}
          onThreshold={(value) => set({ threshold: value })}
          onWhy={
            onWhy &&
            ((row) => {
              const image = rows[row];
              if (image) onWhy(image);
            })
          }
        />
      ) : null}
    </div>
  );
}

/** Every model of the run on one line each, the default first (★), then the best; a click shows
 *  its metrics. The main metric with its interval, and its time. */
function ModelTable({ models, current, onPick }: { models: VisionModel[]; current: string; onPick: (key: string) => void }) {
  const best = Math.max(...models.map((m) => mainScore(m).value ?? -Infinity));
  return (
    <section aria-labelledby="vision-models">
      <h3 id="vision-models" className="text-[13px] font-medium text-ink-1">
        {t("vision.results.models", { count: formatCount(models.length) })}
      </h3>
      <div className="mt-2 max-w-full overflow-x-auto">
        <table className="w-full border-collapse text-[13px] tabular-nums">
          <thead>
            <tr className="text-xs text-ink-3">
              <th scope="col" className="py-1.5 pr-3 text-left font-normal">{t("vision.results.model")}</th>
              <th scope="col" className="px-2 py-1.5 text-right font-normal">{t("vision.results.mainCol")}</th>
              <th scope="col" className="px-2 py-1.5 text-right font-normal max-sm:hidden">{t("vision.results.intervalCol")}</th>
              <th scope="col" className="py-1.5 pl-2 text-right font-normal max-sm:hidden">{t("vision.results.timeCol")}</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-line border-y border-line">
            {models.map((m) => {
              const score = mainScore(m);
              const selected = m.key === current;
              return (
                <tr key={m.key} className={selected ? "bg-surface-2" : "hover:bg-surface-2/60"}>
                  <th scope="row" className="p-0 text-left font-normal">
                    <button
                      type="button"
                      aria-pressed={selected}
                      onClick={() => onPick(m.key)}
                      className="flex w-full min-w-0 items-center gap-2 py-2 pr-3 pl-1 text-left focus-visible:outline-2 focus-visible:outline-accent"
                    >
                      {m.default ? <Star size={13} strokeWidth={1.5} className="shrink-0 text-accent" aria-label={t("vision.model.default")} /> : <span className="w-[13px] shrink-0" aria-hidden />}
                      <span className={`min-w-0 ${selected ? "font-medium text-ink-1" : "text-ink-2"}`}>{m.title}</span>
                      <span className="shrink-0 text-xs text-ink-3">{m.family === "deep" ? t("vision.results.deep") : t("vision.results.classic")}</span>
                      {m.level === "more" && <span className="shrink-0 text-xs text-ink-3">· {t("vision.more.level")}</span>}
                    </button>
                  </th>
                  <td className={`px-2 py-2 text-right whitespace-nowrap ${score.value === best ? "font-semibold text-ink-1" : "text-ink-1"}`}>
                    {score.metric === "flagged" ? formatCount(score.value ?? 0) : metric(score.value)}{" "}
                    <span className="text-xs font-normal text-ink-3">{METRIC[score.metric] ? t(METRIC[score.metric] as MessageKey) : score.metric}</span>
                  </td>
                  <td className="px-2 py-2 text-right whitespace-nowrap text-ink-3 max-sm:hidden">{score.low !== null && score.high !== null ? `[${metric(score.low)}; ${metric(score.high)}]` : "—"}</td>
                  <td className="py-2 pl-2 text-right whitespace-nowrap text-ink-3 max-sm:hidden">{t("vision.model.seconds", { seconds: m.seconds.toFixed(1) })}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/** The chosen model: its title, setting and notes (the rules that shaped it). */
function ModelHead({ model }: { model: VisionModel }) {
  return (
    <section aria-label={t("vision.results.modelNotes")} className="flex flex-col gap-1">
      <h3 className="text-base font-semibold text-balance text-ink-1">{model.title}</h3>
      {model.setting && <p className="text-xs text-ink-3">{model.setting}</p>}
      {model.notes.length > 0 && (
        <details className="mt-0.5 text-xs">
          <summary className="w-fit cursor-pointer text-ink-2 select-none hover:text-ink-1">{t("vision.results.howTrained", { count: formatCount(model.notes.length) })}</summary>
          <ul className="mt-1.5 flex max-w-[80ch] flex-col gap-1 text-pretty text-ink-2">
            {model.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

function ClusterResults({ datasetId, rows, result, predictions, onOpen }: { datasetId: string; rows: readonly ImageRow[]; result: VisionResult; predictions: readonly Prediction[]; onOpen: (row: number) => void }) {
  const clusters = result.clusters;
  if (!clusters) return null;
  const sizes = Object.entries(clusters.sizes).sort((a, b) => Number(a[0]) - Number(b[0]));
  const byK = Object.entries(clusters.by_k).map(([k, value]) => ({ name: `k = ${k}`, value, highlight: Number(k) === clusters.k }));
  const medoids = Object.entries(clusters.medoids).map(([cluster, row]) => ({
    row,
    caption: t("vision.results.cluster", { cluster }),
    detail: t("vision.results.clusterSize", { count: formatCount(clusters.sizes[Number(cluster)] ?? 0) }),
  }));
  const members = (cluster: string) =>
    predictions
      .filter((p) => String(p.cluster) === cluster)
      .slice(0, 24)
      .map((p) => ({ row: p.row, caption: text(p.truth) || t("vision.results.cluster", { cluster }) }));
  return (
    <div className="flex flex-col gap-8">
      <Tiles
        label={t("vision.results.mainMetrics")}
        tiles={[
          { label: t("vision.results.m.k"), value: String(clusters.k), main: true },
          { label: t("vision.results.m.silhouette"), value: metric(clusters.silhouette), tip: t("vision.results.tip.silhouette") },
          { label: t("vision.results.m.ari"), value: metric(clusters.adjusted_rand ?? null), detail: clusters.adjusted_rand === null || clusters.adjusted_rand === undefined ? t("vision.results.noLabels") : undefined, tip: t("vision.results.tip.ari") },
        ]}
      />
      <div className="grid gap-x-10 gap-y-8 lg:grid-cols-2">
        <Block id="vision-silhouette" title={t("vision.results.silhouetteTitle")} conclusion={clusters.sentence}>
          <Plot figure={barsFigure(byK, { range: [Math.min(0, ...byK.map((b) => b.value)), Math.max(...byK.map((b) => b.value)) * 1.15] })} height={Math.max(160, 26 * byK.length + 30)} label={t("vision.results.silhouetteTitle")} />
        </Block>
        <Block id="vision-medoids" title={t("vision.results.medoids")} conclusion={t("vision.results.medoidsSentence")}>
          <Gallery datasetId={datasetId} rows={rows} items={medoids} empty={t("vision.results.noImage")} onOpen={(row) => onOpen(row)} label={t("vision.results.medoids")} />
        </Block>
      </div>
      {sizes.map(([cluster, size]) => (
        <Block key={cluster} id={`vision-cluster-${cluster}`} title={t("vision.results.clusterTitle", { cluster, count: formatCount(size) })}>
          <Gallery datasetId={datasetId} rows={rows} items={members(cluster)} empty={t("vision.results.noImage")} onOpen={(row) => onOpen(row)} label={t("vision.results.clusterTitle", { cluster, count: formatCount(size) })} />
        </Block>
      ))}
    </div>
  );
}
