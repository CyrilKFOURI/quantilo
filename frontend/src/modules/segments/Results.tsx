import { Link } from "@tanstack/react-router";
import { Database, Download, Lock } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { formatNumber } from "../../charts/figures";
import { Plot } from "../../charts/Plot";
import { formatCount } from "../../lib/format";
import { t } from "../../lib/i18n";
import { LITE } from "../../lite/flag";
import { Button } from "../../ui/Button";
import { ErrorNote, TextInput } from "../../ui/fields";
import { InfoTip } from "../../ui/InfoTip";
import { SegmentedControl } from "../../ui/SegmentedControl";
import {
  segmentsApi,
  useCancelSegmentJob,
  useSegmentActions,
  useSegmentCompare,
  useSegmentJob,
  type SegmentPortrait,
  type SegmentProfileRow,
  type SegmentRun,
  type Segmentation,
} from "./api";
import { dendrogramFigure, mapFigure, namesOf, scoreFigure, shownScores, stabilityFigure } from "./figures";
import { JobStatus } from "./SegmentsPage";
import { Loading } from "../../ui/Loading";

const pct = (share: number | null | undefined) => (share === null || share === undefined ? "—" : `${Math.round(share * 100)} %`);

function silhouetteVerdict(value: number | null): string {
  if (value === null) return t("segments.verdict.none");
  if (value >= 0.5) return t("segments.verdict.clear");
  if (value >= 0.25) return t("segments.verdict.moderate");
  return t("segments.verdict.weak");
}

export function Results({ datasetId, run }: { datasetId: string; run: SegmentRun }) {
  const result = run.result;
  const names = useMemo(() => namesOf(result), [result]);
  return (
    <div className="space-y-6">
      <header className="space-y-1">
        <h2 className="text-lg font-semibold tracking-tight">{result.choice.why}</h2>
        <p className="text-[13px] text-ink-2">
          {result.method_why}{" "}
          {t("segments.rowsSegmented", { rows: formatCount(result.rows), inputs: String(result.inputs) })}
          {result.sampled ? ` ${t("segments.sampledNote", { rows: formatCount(result.fit_rows) })}` : ""}
        </p>
        {result.warnings.map((warning) => (
          <p key={warning} role="alert" className="rounded-md border border-warn/40 bg-warn/8 px-3 py-2 text-[13px] text-ink-1">
            {warning}
          </p>
        ))}
      </header>
      <Headline result={result} />
      {result.scores.length > 1 && (
        <section aria-labelledby="segments-scores" className="space-y-1">
          <h3 id="segments-scores" className="flex items-center gap-1 text-sm font-medium">
            {t("segments.scores")}
            <InfoTip label={t("segments.scores")} text={t("segments.scoresHelp")} />
          </h3>
          <div className="grid gap-x-4 gap-y-2 sm:grid-cols-2 2xl:grid-cols-4">
            {shownScores(result.scores).map((s) => (
              <figure key={s.key} className="min-w-0">
                <figcaption className="text-xs text-ink-2">{s.label}</figcaption>
                <Plot figure={scoreFigure(result.scores, s.key, result.k)} height={160} label={s.label} />
              </figure>
            ))}
          </div>
        </section>
      )}
      <Portraits datasetId={datasetId} run={run} />
      <MapSection datasetId={datasetId} run={run} names={names} />
      <ProfileTable result={result} profiles={result.profiles} />
      <div className="grid gap-6 xl:grid-cols-2">
        {result.stability && (
          <section aria-labelledby="segments-stability" className="space-y-1">
            <h3 id="segments-stability" className="flex items-center gap-1 text-sm font-medium">
              {t("segments.stabilityTitle")}
              <InfoTip label={t("segments.stabilityTitle")} text={t("segments.stabilityHelp")} />
            </h3>
            <Plot figure={stabilityFigure(result.stability.runs)} height={220} label={t("segments.stabilityTitle")} />
          </section>
        )}
        {result.dendrogram && (
          <section aria-labelledby="segments-tree" className="space-y-1">
            <h3 id="segments-tree" className="text-sm font-medium">
              {t("segments.dendrogram")}
            </h3>
            <p className="text-xs text-ink-3">{t("segments.dendrogramNote", { rows: formatCount(result.dendrogram.rows) })}</p>
            <Plot figure={dendrogramFigure(result.dendrogram)} height={260} label={t("segments.dendrogram")} />
          </section>
        )}
      </div>
      <SaveSegments datasetId={datasetId} run={run} />
    </div>
  );
}

function Headline({ result }: { result: Segmentation }) {
  const stability = result.stability;
  const figures = [
    { key: "k", label: t("segments.headline.k"), value: String(result.k), detail: result.method_label },
    {
      key: "silhouette",
      label: t("segments.silhouette"),
      value: formatNumber(result.chosen.silhouette),
      detail: silhouetteVerdict(result.chosen.silhouette),
    },
    {
      key: "stability",
      label: t("segments.stability"),
      value: stability ? formatNumber(stability.ari) : "—",
      detail: stability ? (stability.stable ? t("segments.stable") : t("segments.unstable")) : t("segments.verdict.none"),
      tone: stability && !stability.stable ? "text-warn" : "text-ink-1",
    },
    { key: "rows", label: t("segments.headline.rows"), value: formatCount(result.rows), detail: t("segments.headline.columns", { count: String(result.preparation.decisions.filter((d) => d.use).length) }) },
  ];
  return (
    <dl className="grid grid-cols-2 border-y border-line lg:grid-cols-4">
      {figures.map((f, i) => (
        <div key={f.key} className={`px-4 py-3 ${i % 2 === 1 ? "border-l border-line" : ""} ${i > 0 ? "lg:border-l lg:border-line" : ""}`}>
          <dt className="text-xs text-ink-3">{f.label}</dt>
          <dd className={`text-[28px] leading-tight font-semibold tabular-nums ${f.tone ?? "text-ink-1"}`}>{f.value}</dd>
          <dd className="text-[13px] text-ink-2">{f.detail}</dd>
        </div>
      ))}
    </dl>
  );
}

function Portraits({ datasetId, run }: { datasetId: string; run: SegmentRun }) {
  const actions = useSegmentActions(datasetId);
  return (
    <section aria-labelledby="segments-portraits" className="space-y-2">
      <h3 id="segments-portraits" className="text-sm font-medium">
        {t("segments.portraits")}
      </h3>
      <div className="grid gap-px overflow-hidden rounded-lg border border-line bg-line md:grid-cols-2 xl:grid-cols-3">
        {run.result.segments.map((segment) => (
          <Portrait
            key={`${run.id}-${segment.id}`}
            segment={segment}
            color={segment.id < 0 ? "var(--ink-3)" : `var(--chart-${(segment.id % 8) + 1})`}
            onRename={(name) => actions.rename.mutate({ runId: run.id, names: { [String(segment.id)]: name } })}
          />
        ))}
      </div>
      {actions.rename.isError && <ErrorNote>{actions.rename.error.message}</ErrorNote>}
    </section>
  );
}

function Portrait({ segment, color, onRename }: { segment: SegmentPortrait; color: string; onRename: (name: string) => void }) {
  const [name, setName] = useState(segment.name);
  const commit = () => {
    if (name.trim() && name.trim() !== segment.name) onRename(name.trim());
  };
  return (
    <article className="space-y-2 bg-surface-1 p-4" aria-label={segment.name}>
      <header className="space-y-0.5">
        {segment.grouped ? (
          <TextInput
            aria-label={t("segments.rename", { id: String(segment.id + 1) })}
            value={name}
            onChange={(e) => setName(e.target.value)}
            onBlur={commit}
            onKeyDown={(e) => {
              if (e.key === "Enter") e.currentTarget.blur();
            }}
            className="-ml-1 h-7 w-full border-transparent bg-transparent px-1 font-medium hover:border-line"
          />
        ) : (
          <span className="block font-medium">{segment.name}</span>
        )}
        <p className="flex items-center gap-1.5 text-xs text-ink-2 tabular-nums">
          <span className="size-2.5 shrink-0 rounded-full" style={{ background: color }} aria-hidden />
          {t("segments.size", { rows: formatCount(segment.size), share: pct(segment.share) })}
        </p>
      </header>
      {segment.grouped ? (
        <>
          <ul className="space-y-0.5 text-[13px] text-ink-1">
            {segment.traits.map((trait) => (
              <li key={trait.column} className="flex gap-1.5">
                <span aria-hidden className="text-ink-3">
                  {trait.kind === "number" ? (trait.effect > 0 ? "▲" : "▼") : "●"}
                </span>
                {trait.text}
              </li>
            ))}
          </ul>
          {segment.rules && segment.rules.rules.length > 0 && (
            <div className="space-y-0.5 border-t border-line pt-2 text-xs">
              <p className="text-ink-3">{t("segments.rules")}</p>
              {segment.rules.rules.map((rule) => (
                <p key={rule.text} className="font-mono text-[11px] text-ink-1">
                  {rule.text}
                </p>
              ))}
              <p className="text-ink-2">{segment.rules.text}</p>
            </div>
          )}
          {segment.membership !== null && <p className="text-xs text-ink-3">{t("segments.membership", { share: pct(segment.membership) })}</p>}
        </>
      ) : (
        <p className="text-[13px] text-ink-2">{t("segments.notGrouped")}</p>
      )}
    </article>
  );
}

function MapSection({ datasetId, run, names }: { datasetId: string; run: SegmentRun; names: Record<number, string> }) {
  const [projection, setProjection] = useState<"pca" | "tsne">("pca");
  const actions = useSegmentActions(datasetId);
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useSegmentJob(jobId);
  const cancel = useCancelSegmentJob();
  const tsne = run.result.tsne;
  // When t-SNE ends, reload the run to show it (once per job).
  const finished = job.data?.status === "done" ? job.data.run_id : null;
  const refreshed = useRef<string | null>(null);
  useEffect(() => {
    if (finished && refreshed.current !== jobId) {
      refreshed.current = jobId;
      actions.refresh(finished);
    }
  }, [finished, jobId, actions]);
  const shown = projection === "tsne" && tsne ? tsne : run.result.map;
  return (
    <section aria-labelledby="segments-map" className="space-y-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 id="segments-map" className="flex items-center gap-1 text-sm font-medium">
          {t("segments.map")}
          <InfoTip label={t("segments.map")} text={t(projection === "pca" ? "segments.mapPcaHelp" : "segments.mapTsneHelp")} />
        </h3>
        <span className="flex items-center gap-2">
          {LITE ? (
            <span className="inline-flex items-center gap-1 text-xs text-ink-3" title={t("segments.tsneLocked")}>
              <Lock size={13} strokeWidth={1.5} aria-hidden /> {t("segments.tsneLocked")}
            </span>
          ) : tsne ? (
            <SegmentedControl
              label={t("segments.projection")}
              value={projection}
              options={[
                { value: "pca", label: "PCA" },
                { value: "tsne", label: "t-SNE" },
              ]}
              onChange={setProjection}
            />
          ) : (
            <Button
              variant="ghost"
              disabled={actions.tsne.isPending || job.data?.status === "running"}
              onClick={() =>
                actions.tsne.mutate(run.id, {
                  onSuccess: (created) => {
                    setJobId(created.id);
                    setProjection("tsne");
                    if (created.status === "done") actions.refresh(run.id);
                  },
                })
              }
            >
              {t("segments.tsne")}
            </Button>
          )}
        </span>
      </div>
      {job.data?.status === "running" && jobId && <JobStatus job={job.data} onCancel={() => cancel.mutate(jobId)} />}
      {job.data?.status === "failed" && <ErrorNote>{job.data.error}</ErrorNote>}
      <p className="text-xs text-ink-3">
        {shown.points < shown.of ? t("segments.mapSample", { points: formatCount(shown.points), of: formatCount(shown.of) }) : t("segments.mapAll", { points: formatCount(shown.points) })}
      </p>
      <Plot figure={mapFigure(shown, names)} height={420} label={t("segments.map")} />
    </section>
  );
}

function ProfileTable({ result, profiles }: { result: Segmentation; profiles: SegmentProfileRow[] }) {
  const cell = (value: number | string | null, share: number | null) =>
    value === null ? "—" : typeof value === "number" ? formatNumber(value) : share !== null ? `${value} (${pct(share)})` : value;
  return (
    <section aria-labelledby="segments-profiles" className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <h3 id="segments-profiles" className="flex items-center gap-1 text-sm font-medium">
          {t("segments.profiles")}
          <InfoTip label={t("segments.profiles")} text={t("segments.profilesHelp")} />
        </h3>
      </div>
      <div className="overflow-x-auto border-y border-line">
        <table className="w-full text-[13px]">
          <caption className="sr-only">{t("segments.profiles")}</caption>
          <thead>
            <tr className="border-b border-line text-xs text-ink-3">
              <th className="py-1.5 pr-3 text-left font-normal">{t("segments.column")}</th>
              <th className="py-1.5 pr-3 text-right font-normal">{t("segments.overall")}</th>
              {result.segments.map((s) => (
                <th key={s.id} className="py-1.5 pr-3 text-right font-normal">
                  {s.name}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {profiles.map((row) => (
              <tr key={row.column} className="border-b border-line last:border-0">
                <td className="py-1.5 pr-3 font-mono text-xs text-ink-1">{row.column}</td>
                <td className="py-1.5 pr-3 text-right text-ink-2 tabular-nums">{cell(row.overall.value, row.overall.share)}</td>
                {row.cells.map((c, i) => (
                  <td
                    key={i}
                    className={`py-1.5 pr-3 text-right tabular-nums ${c.strong ? "bg-accent-soft font-semibold text-ink-1" : "text-ink-1"}`}
                    title={c.effect !== null ? t("segments.effect", { effect: c.effect.toFixed(2) }) : undefined}
                  >
                    {c.strong && row.kind === "number" ? <span aria-hidden className="mr-1 text-ink-3">{(c.effect ?? 0) > 0 ? "▲" : "▼"}</span> : null}
                    {cell(c.value, c.share)}
                    {c.strong && <span className="sr-only"> ({t("segments.strong")})</span>}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">{t("segments.profilesNote")}</p>
    </section>
  );
}

function SaveSegments({ datasetId, run }: { datasetId: string; run: SegmentRun }) {
  const actions = useSegmentActions(datasetId);
  const created = actions.save.data?.dataset_id ?? run.created_dataset_id;
  return (
    <section className="space-y-2 border-t border-line pt-4">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" icon={Database} onClick={() => actions.save.mutate(run.id)} disabled={actions.save.isPending}>
          {t("segments.save")}
        </Button>
        <a
          href={segmentsApi.profilesUrl(datasetId, run.id)}
          download
          className="inline-flex h-(--control-h) items-center gap-1.5 rounded-md border border-line bg-surface-1 px-3 text-[13px] text-ink-1 hover:bg-surface-2"
        >
          <Download size={15} strokeWidth={1.5} aria-hidden />
          {t("segments.exportProfiles")}
        </a>
        {created && (
          <p className="text-[13px] text-ink-2" role="status">
            {t("segments.savedAs")}{" "}
            <Link to="/d/$datasetId" params={{ datasetId: created }} className="text-accent hover:underline">
              {t("segments.openDataset")}
            </Link>
          </p>
        )}
      </div>
      <p className="text-xs text-ink-3">{t("segments.saveHint")}</p>
      {actions.save.error && <ErrorNote>{actions.save.error.message}</ErrorNote>}
    </section>
  );
}

/** Runs side by side, and how much their segments agree (adjusted Rand index on shared rows). */
export function RunCompare({ datasetId, runIds }: { datasetId: string; runIds: string[] }) {
  const compared = useSegmentCompare(datasetId, runIds);
  if (compared.isError) return <ErrorNote>{compared.error.message}</ErrorNote>;
  if (!compared.data) return <Loading label={t("loading.compareRuns")} className="h-40 rounded-md" />;
  const { runs, agreement } = compared.data;
  const label = (i: number) => `#${i + 1} ${runs[i]?.method_label ?? ""}`;
  return (
    <section aria-labelledby="segments-compare" className="space-y-3 border-t border-line pt-4">
      <h3 id="segments-compare" className="text-sm font-medium">
        {t("segments.compare")}
      </h3>
      <div className="overflow-x-auto">
        <table className="w-full text-[13px]">
          <caption className="sr-only">{t("segments.compare")}</caption>
          <thead>
            <tr className="border-b border-line text-xs text-ink-3">
              <th className="py-1.5 pr-3 text-left font-normal">{t("segments.runHeader")}</th>
              <th className="py-1.5 pr-3 text-right font-normal">k</th>
              <th className="py-1.5 pr-3 text-right font-normal">{t("segments.silhouette")}</th>
              <th className="py-1.5 pr-3 text-right font-normal">{t("segments.stability")}</th>
              <th className="py-1.5 pr-3 text-left font-normal">{t("segments.columns")}</th>
              {runs.map((_, i) => (
                <th key={i} className="py-1.5 pr-3 text-right font-normal">
                  {t("segments.agreementWith", { run: label(i) })}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {runs.map((r, i) => (
              <tr key={r.id} className="border-b border-line last:border-0">
                <td className="py-1.5 pr-3">{label(i)}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{r.k}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{formatNumber(r.silhouette)}</td>
                <td className="py-1.5 pr-3 text-right tabular-nums">{formatNumber(r.stability)}</td>
                <td className="py-1.5 pr-3 font-mono text-xs text-ink-2">{r.columns.join(", ")}</td>
                {agreement[i]?.map((value, j) => (
                  <td key={j} className="py-1.5 pr-3 text-right tabular-nums">
                    {formatNumber(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-3">{t("segments.agreementHelp")}</p>
    </section>
  );
}
